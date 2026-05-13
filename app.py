import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
import io
import os
import requests
from tqdm import tqdm

# Model Definitions
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = ResidualBlock(64, 64, time_emb_dim)
        self.down1_2 = ResidualBlock(64, 64, time_emb_dim)
        
        self.downsample1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2 = ResidualBlock(128, 128, time_emb_dim)
        self.down2_2 = ResidualBlock(128, 128, time_emb_dim)
        
        self.downsample2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.down3 = ResidualBlock(256, 256, time_emb_dim)
        self.down3_2 = ResidualBlock(256, 256, time_emb_dim)
        
        self.bottleneck1 = ResidualBlock(256, 256, time_emb_dim)
        self.bottleneck2 = ResidualBlock(256, 256, time_emb_dim)
        
        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up1 = ResidualBlock(128, 128, time_emb_dim)
        self.up1_2 = ResidualBlock(128, 128, time_emb_dim)
        
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2 = ResidualBlock(64, 64, time_emb_dim)
        self.up2_2 = ResidualBlock(64, 64, time_emb_dim)
        
        self.up3 = ResidualBlock(64, 64, time_emb_dim)
        self.up3_2 = ResidualBlock(64, 64, time_emb_dim)
        
        self.output = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x1 = self.conv1(x)
        d1 = self.down1(x1, t_emb)
        d1 = self.down1_2(d1, t_emb)
        
        d2 = self.downsample1(d1)
        d2 = self.down2(d2, t_emb)
        d2 = self.down2_2(d2, t_emb)
        
        d3 = self.downsample2(d2)
        d3 = self.down3(d3, t_emb)
        d3 = self.down3_2(d3, t_emb)
        
        b = self.bottleneck1(d3, t_emb)
        b = self.bottleneck2(b, t_emb)
        
        u1 = self.upsample1(b)
        u1 = u1 + d2
        u1 = self.up1(u1, t_emb)
        u1 = self.up1_2(u1, t_emb)
        
        u2 = self.upsample2(u1)
        u2 = u2 + d1
        u2 = self.up2(u2, t_emb)
        u2 = self.up2_2(u2, t_emb)
        
        u3 = u2 + x1
        u3 = self.up3(u3, t_emb)
        u3 = self.up3_2(u3, t_emb)
        
        return self.output(u3)

class DiffusionModel:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=1):
        model.eval()
        x = torch.randn((batch_size, 3, image_size, image_size)).to(self.device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
        
        return x

def denormalize(tensor):
    img = tensor.cpu().detach().permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img

def tensor_to_pil(tensor):
    img = denormalize(tensor)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=256).to(device)
    
    model_path = 'diffusion_model_final.pth'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from GitHub...'):
            url = "https://github.com/Abdulbaset1/Diffusion-Models-for-High-Resolution-Image-Generation/releases/download/v1/diffusion_model_final.pth"
            response = requests.get(url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, device

# Streamlit UI
st.set_page_config(page_title="Diffusion Model Image Generator", layout="wide")
st.title("🎨 Diffusion Model for High-Resolution Image Generation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    num_images = st.slider("Number of Images", 1, 8, 4)
    image_size = st.selectbox("Image Size", [64, 128], index=1)
    timesteps = st.slider("Diffusion Steps", 100, 300, 300, step=50)
    seed = st.number_input("Random Seed", value=42, step=1)
    
    if st.button("🚀 Generate Images", type="primary", use_container_width=True):
        st.session_state['generate'] = True

# Main content
if 'generate' not in st.session_state:
    st.session_state['generate'] = False

if st.session_state['generate']:
    # Load model
    model, device = load_model()
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Initialize diffusion
    diffusion = DiffusionModel(timesteps=timesteps, device=device)
    
    # Generate images
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    images = []
    for i in range(num_images):
        status_text.text(f"Generating image {i+1}/{num_images}")
        img_tensor = diffusion.sample(model, image_size, batch_size=1)
        images.append(tensor_to_pil(img_tensor[0]))
        progress_bar.progress((i + 1) / num_images)
    
    progress_bar.empty()
    status_text.text("Generation complete!")
    
    # Display images
    cols = st.columns(min(3, num_images))
    for idx, img in enumerate(images):
        with cols[idx % len(cols)]:
            st.image(img, caption=f"Image {idx+1}", use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label=f"Download {idx+1}",
                data=buf.getvalue(),
                file_name=f"generated_image_{idx+1}.png",
                mime="image/png",
                key=f"download_{idx}"
            )
    
    # Reset generation flag
    st.session_state['generate'] = False

# Info section
with st.expander("About"):
    st.markdown("""
    This app uses a diffusion model trained on face images to generate new images from pure noise.
    
    **Parameters:**
    - **Number of Images**: How many images to generate
    - **Image Size**: Resolution (64 or 128 pixels)
    - **Diffusion Steps**: More steps = better quality but slower
    - **Random Seed**: For reproducible results
    
    The model will be downloaded automatically on first run.
    """)
