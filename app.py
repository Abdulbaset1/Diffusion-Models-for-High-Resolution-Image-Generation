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
import zipfile
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
    
    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from GitHub...'):
            url = "https://github.com/Abdulbaset1/Diffusion-Models-for-High-Resolution-Image-Generation/releases/download/v1/diffusion_model_final.pth"
            
            # Download with progress
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with st.progress(0) as pbar:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pbar.progress(min(1.0, downloaded / total_size))
    
    # Load model with weights_only=False (trusted source)
    try:
        # First try with weights_only=False (required for PyTorch 2.6+)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure the model file is not corrupted.")
        raise e
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # Load state dict with strict=False to allow missing keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            st.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5 missing keys
        if unexpected_keys:
            st.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
    else:
        # Checkpoint is the model itself
        model = checkpoint.to(device)
    
    model.eval()
    return model, device

# Streamlit UI
st.set_page_config(
    page_title="Diffusion Model Image Generator", 
    layout="wide",
    page_icon="🎨"
)

st.title("🎨 Diffusion Model for High-Resolution Image Generation")
st.markdown("*Generate high-quality face images using a trained diffusion model*")

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generating' not in st.session_state:
    st.session_state.generating = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    num_images = st.slider("Number of Images", 1, 8, 4, 
                           help="Number of images to generate at once")
    image_size = st.selectbox("Image Size", [64, 128], index=1,
                              help="Larger size = better quality but slower")
    timesteps = st.slider("Diffusion Steps", 100, 300, 300, step=50,
                          help="More steps = better quality but slower generation")
    seed = st.number_input("Random Seed", value=42, step=1,
                          help="Set for reproducible results")
    
    st.markdown("---")
    
    if st.button("🚀 Generate Images", type="primary", use_container_width=True):
        st.session_state.generating = True
        st.session_state.num_images = num_images
        st.session_state.image_size = image_size
        st.session_state.timesteps = timesteps
        st.session_state.seed = seed
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Adjust generation parameters
    2. Click 'Generate Images'
    3. Wait for model to load (first time only)
    4. Download individual images or all as ZIP
    """)

# Main content area
if st.session_state.generating:
    # Create placeholder for progress
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Load model
        with status_placeholder.container():
            st.info("🔄 Loading model and generating images...")
        
        model, device = load_model()
        
        with status_placeholder.container():
            st.success("✅ Model loaded successfully!")
        
        # Set seed
        torch.manual_seed(st.session_state.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(st.session_state.seed)
        
        # Initialize diffusion
        diffusion = DiffusionModel(
            timesteps=st.session_state.timesteps, 
            device=device
        )
        
        # Generate images
        progress_bar = progress_placeholder.progress(0)
        status_text = status_placeholder.empty()
        
        generated_images = []
        for i in range(st.session_state.num_images):
            status_text.info(f"🖼️ Generating image {i+1}/{st.session_state.num_images}")
            img_tensor = diffusion.sample(
                model, 
                st.session_state.image_size, 
                batch_size=1
            )
            generated_images.append(tensor_to_pil(img_tensor[0]))
            progress_bar.progress((i + 1) / st.session_state.num_images)
        
        # Clear progress indicators
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Store in session state
        st.session_state.generated_images = generated_images
        st.session_state.generating = False
        
        st.success(f"✅ Successfully generated {len(generated_images)} images!")
        st.rerun()
        
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Error during generation: {str(e)}")
        st.session_state.generating = False
        st.stop()

# Display generated images
if st.session_state.generated_images:
    st.markdown("---")
    st.subheader(f"✨ Generated Images")
    
    # Display in grid
    num_images = len(st.session_state.generated_images)
    cols_per_row = min(4, num_images)
    cols = st.columns(cols_per_row)
    
    for idx, img in enumerate(st.session_state.generated_images):
        with cols[idx % cols_per_row]:
            st.image(img, caption=f"Image {idx+1}", use_container_width=True)
            
            # Download button for individual image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label=f"📥 Download",
                data=buf.getvalue(),
                file_name=f"generated_image_{idx+1}.png",
                mime="image/png",
                key=f"download_{idx}"
            )
    
    # Option to download all as zip
    if num_images > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zip_file:
                for idx, img in enumerate(st.session_state.generated_images):
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    zip_file.writestr(f"image_{idx+1}.png", img_buf.getvalue())
            
            st.download_button(
                label="📦 Download All Images (ZIP)",
                data=zip_buf.getvalue(),
                file_name="generated_images.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear All Images", use_container_width=True):
        st.session_state.generated_images = []
        st.rerun()

# Info section
with st.expander("ℹ️ About & Technical Details"):
    st.markdown("""
    ### 🧠 How Diffusion Models Work
    
    Diffusion models generate images by:
    1. Starting with pure random noise
    2. Gradually denoising the image over many steps
    3. Learning to reverse the noise addition process
    
    ### 🎯 Model Architecture
    - **Type**: U-Net with residual blocks
    - **Parameters**: ~20 million
    - **Training Data**: FFHQ (Flickr-Faces-HQ) dataset
    - **Input/Output**: 3-channel RGB images
    
    ### ⚡ Performance Tips
    - **Faster generation**: Use 64px size and 100-200 steps
    - **Better quality**: Use 128px size and 300 steps
    - **First run**: Model download takes 1-2 minutes
    - **Subsequent runs**: Model loads from cache (~10 seconds)
    
    ### 🔧 Technical Notes
    - Built with PyTorch and Streamlit
    - Uses CUDA acceleration when available
    - Model downloaded from GitHub releases
    - Supports reproducible results with random seeds
    
    ### 📚 References
    - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
    - [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Built with ❤️ using PyTorch & Streamlit | "
    "<a href='https://github.com/Abdulbaset1/Diffusion-Models-for-High-Resolution-Image-Generation'>GitHub Repository</a>"
    "</div>",
    unsafe_allow_html=True
)
