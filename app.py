import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
import os
import io

# Page config
st.set_page_config(
    page_title="Diffusion Face Generator",
    page_icon="🎨",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
.generated-image {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Model Architecture
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return self.mlp(embeddings)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        num_groups = 8
        while out_channels % num_groups != 0:
            num_groups //= 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels), nn.GELU()
        )
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, out_channels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels), nn.GELU(), nn.Dropout(dropout)
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
    def forward(self, x, time_emb):
        residual = self.residual_conv(x)
        h = self.conv1(x) + self.time_mlp(time_emb)[:, :, None, None]
        return self.conv2(h) + residual

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64,
                 time_emb_dim=128, channel_mults=(1, 2, 4)):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        channels = [base_channels * m for m in channel_mults]

        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        prev = base_channels
        for ch in channels:
            self.encoder_blocks.append(nn.ModuleList([
                ResidualBlock(prev, ch, time_emb_dim),
                ResidualBlock(ch, ch, time_emb_dim)
            ]))
            self.downsample_blocks.append(Downsample(ch) if ch != channels[-1] else nn.Identity())
            prev = ch

        self.bottleneck = nn.ModuleList([
            ResidualBlock(channels[-1], channels[-1], time_emb_dim),
            ResidualBlock(channels[-1], channels[-1], time_emb_dim)
        ])

        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            self.upsample_blocks.append(Upsample(prev) if i != 0 else nn.Identity())
            self.decoder_blocks.append(nn.ModuleList([
                ResidualBlock(prev + ch, ch, time_emb_dim),
                ResidualBlock(ch, ch, time_emb_dim)
            ]))
            prev = ch

        ng = 8
        while base_channels % ng != 0: ng //= 2
        self.final_conv = nn.Sequential(
            nn.GroupNorm(ng, base_channels), nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        te = self.time_embedding(t)
        x = self.init_conv(x)
        skips = []
        for blocks, down in zip(self.encoder_blocks, self.downsample_blocks):
            for b in blocks: x = b(x, te)
            skips.append(x)
            x = down(x)
        for b in self.bottleneck: x = b(x, te)
        for up, blocks, skip in zip(self.upsample_blocks, self.decoder_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            for b in blocks: x = b(x, te)
        return self.final_conv(x)

# Noise schedule
TIMESTEPS = 500
IMAGE_SIZE = 128
betas = torch.linspace(0.0001, 0.02, TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# Model loading - UPDATED WITH YOUR MODEL PATH
MODEL_URL = "https://github.com/Abdulbaset1/Diffusion-Models-for-High-Resolution-Image-Generation/releases/download/v1/best_model.pt"
MODEL_PATH = "best_model.pt"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    # Load model
    model = UNet(in_channels=3, out_channels=3, base_channels=64,
                 time_emb_dim=128, channel_mults=(1, 2, 4)).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, device

def tensor_to_pil(tensor):
    img = torch.clamp((tensor + 1) / 2, 0, 1)
    arr = (img[0].permute(1, 2, 0).cpu().detach().numpy() * 255)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')

@torch.no_grad()
def generate_image(model, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, t in enumerate(range(TIMESTEPS - 1, -1, -1)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        
        alpha = alphas[t].to(device)
        alpha_cumprod_t = alphas_cumprod[t].to(device)
        alpha_cumprod_prev_t = alphas_cumprod_prev[t].to(device)
        
        x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        
        mean = (torch.sqrt(alpha_cumprod_prev_t) * (1 - alpha) / (1 - alpha_cumprod_t)) * x0_pred + \
               (torch.sqrt(alpha) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * x
        variance = (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha)
        
        x = mean + torch.sqrt(variance) * noise
        
        progress_bar.progress((idx + 1) / TIMESTEPS)
        if t % 100 == 0:
            status_text.info(f"Denoising step: {t}/{TIMESTEPS}")
    
    status_text.empty()
    return tensor_to_pil(x)

# UI
st.title("🎨 Diffusion Face Generator")
st.markdown("Generate realistic faces using Denoising Diffusion Probabilistic Models")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    seed_option = st.radio("Seed", ["Random", "Fixed"], horizontal=True)
    seed_value = None
    if seed_option == "Fixed":
        seed_value = st.number_input("Seed Value", min_value=0, max_value=99999, value=42)
    
    st.markdown("---")
    
    if st.button("🚀 Generate Image", type="primary"):
        st.session_state.generate = True

# Main content
if 'generate' not in st.session_state:
    st.session_state.generate = False

if st.session_state.generate:
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    st.info(f"Model loaded on {device.type.upper()}")
    
    # Generate image
    with st.spinner("Generating image..."):
        image = generate_image(model, device, seed=seed_value)
    
    # Display result
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Generated Face", use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label="💾 Download Image",
            data=buf.getvalue(),
            file_name=f"generated_face_{seed_value or 'random'}.png",
            mime="image/png"
        )
    
    # Reset
    st.session_state.generate = False

else:
    # Placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px;">
            <h3>✨ Ready to Generate</h3>
            <p>Click the Generate button in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with PyTorch | DDPM Architecture | 128×128 Resolution")
