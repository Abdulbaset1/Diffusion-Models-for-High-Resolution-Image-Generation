import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os


# =========================
# MODEL DEFINITIONS (SAME AS TRAINING)
# =========================

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
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
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
    def __init__(self):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, 256),
            nn.SiLU()
        )

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)

        self.down1 = ResidualBlock(64, 64, 256)
        self.down1_2 = ResidualBlock(64, 64, 256)

        self.downsample1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2 = ResidualBlock(128, 128, 256)
        self.down2_2 = ResidualBlock(128, 128, 256)

        self.downsample2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.down3 = ResidualBlock(256, 256, 256)
        self.down3_2 = ResidualBlock(256, 256, 256)

        self.bottleneck1 = ResidualBlock(256, 256, 256)
        self.bottleneck2 = ResidualBlock(256, 256, 256)

        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up1 = ResidualBlock(128, 128, 256)
        self.up1_2 = ResidualBlock(128, 128, 256)

        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2 = ResidualBlock(64, 64, 256)
        self.up2_2 = ResidualBlock(64, 64, 256)

        self.up3 = ResidualBlock(64, 64, 256)
        self.up3_2 = ResidualBlock(64, 64, 256)

        self.output = nn.Conv2d(64, 3, 1)

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

        u3 = self.up3(u2, t_emb)
        u3 = self.up3_2(u3, t_emb)

        return self.output(u3)


class Diffusion:
    def __init__(self, timesteps=300, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


    @torch.no_grad()
    def sample(self, model, image_size=128):
        model.eval()
        x = torch.randn((1, 3, image_size, image_size)).to(self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
            noise_pred = model(x, t_batch)

            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            )

            if t > 0:
                x = x + torch.randn_like(x) * torch.sqrt(self.betas[t])

        return x


def denormalize(img):
    img = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img


# =========================
# STREAMLIT APP
# =========================

st.title("🧠 Diffusion Model Image Generator")
st.write("Generate high-quality images using your trained diffusion model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load("models/FinalModel.pt", map_location=device))
    model.eval()
    return model

model = load_model()
diffusion = Diffusion(device=device)

st.sidebar.header("Controls")
num_images = st.sidebar.slider("Number of Images", 1, 5, 1)
image_size = st.sidebar.selectbox("Image Size", [128])

if st.button("Generate Images"):
    st.write("Generating... please wait")

    images = []

    for _ in range(num_images):
        img = diffusion.sample(model, image_size=image_size)
        images.append(denormalize(img[0]))

    cols = st.columns(num_images)

    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Generated {i+1}", use_column_width=True)

st.markdown("---")
st.caption("Powered by PyTorch + DDPM")
