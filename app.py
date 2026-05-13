import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
import os
import io

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DDPM Face Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --accent: #7c6aff;
    --accent2: #ff6a9d;
    --accent3: #6affda;
    --text: #e8e8f0;
    --text-muted: #6b6b8a;
    --border: #2a2a3d;
}
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(124,106,255,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(255,106,157,0.06) 0%, transparent 50%);
}
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.5rem;
    font-weight: 300;
}
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
}
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent3);
    margin-top: 4px;
}
.steps-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 4px;
    letter-spacing: 1px;
}
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.75rem !important;
}
.status-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent3);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent3);
    margin: 1rem 0;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────
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

        self.encoder_blocks    = nn.ModuleList()
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
        self.decoder_blocks  = nn.ModuleList()
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
        x  = self.init_conv(x)
        skips = []
        for blocks, down in zip(self.encoder_blocks, self.downsample_blocks):
            for b in blocks: x = b(x, te)
            skips.append(x); x = down(x)
        for b in self.bottleneck: x = b(x, te)
        for up, blocks, skip in zip(self.upsample_blocks, self.decoder_blocks, reversed(skips)):
            x = up(x); x = torch.cat([x, skip], dim=1)
            for b in blocks: x = b(x, te)
        return self.final_conv(x)


# ─────────────────────────────────────────
# NOISE SCHEDULE
# ─────────────────────────────────────────
TIMESTEPS  = 500
IMAGE_SIZE = 128
betas               = torch.linspace(0.0001, 0.02, TIMESTEPS)
alphas              = 1.0 - betas
alphas_cumprod      = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
MODEL_URL  = "https://github.com/Mustehsan-Nisar-Rao/DDPM/releases/download/v1/best_model.pt"
MODEL_PATH = "best_model.pt"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100_000:
        r = requests.get(MODEL_URL, headers={"Accept": "application/octet-stream"}, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet(in_channels=3, out_channels=3, base_channels=64,
                  time_emb_dim=128, channel_mults=(1, 2, 4)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def tensor_to_pil(t):
    img = torch.clamp((t + 1) / 2, 0, 1)
    arr = (img[0].permute(1, 2, 0).cpu().detach().numpy() * 255)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


@torch.no_grad()
def generate(model, device, num_steps_to_show=8, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    capture_at   = set(np.linspace(TIMESTEPS - 1, 0, num_steps_to_show, dtype=int))
    snapshots    = []
    progress_bar = st.progress(0, text="Starting denoising...")
    status_text  = st.empty()

    for idx, t_val in enumerate(range(TIMESTEPS - 1, -1, -1)):
        t_b  = torch.full((1,), t_val, device=device, dtype=torch.long)
        pred = model(x, t_b)

        a     = alphas[t_val].to(device)
        ahat  = alphas_cumprod[t_val].to(device)
        ahatm = alphas_cumprod_prev[t_val].to(device)

        x0    = torch.clamp((x - torch.sqrt(1 - ahat) * pred) / torch.sqrt(ahat), -1, 1)
        noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)
        mean  = (torch.sqrt(ahatm) * (1 - a) / (1 - ahat)) * x0 \
              + (torch.sqrt(a) * (1 - ahatm) / (1 - ahat)) * x
        var   = (1 - ahatm) / (1 - ahat) * (1 - a)
        x     = mean + torch.sqrt(var) * noise

        if t_val in capture_at:
            snapshots.append((t_val, tensor_to_pil(x)))

        progress_bar.progress((idx + 1) / TIMESTEPS,
                               text=f"Denoising... {TIMESTEPS - t_val}/{TIMESTEPS}")
        if t_val % 100 == 0:
            status_text.markdown(
                f'<div class="status-box">⟳ Timestep {t_val} — removing noise</div>',
                unsafe_allow_html=True
            )

    progress_bar.progress(1.0, text="Done!")
    status_text.empty()
    return tensor_to_pil(x), snapshots


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>DDPM · FACE SYNTHESIS</h1>
    <p>Denoising Diffusion Probabilistic Model · CelebA-HQ · 128×128</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙ Controls")
    seed_mode = st.radio("Seed", ["Random", "Fixed"], horizontal=True)
    seed_val  = None
    if seed_mode == "Fixed":
        seed_val = st.number_input("Seed value", min_value=0, max_value=99999, value=42)

    st.markdown("---")
    n_steps = st.slider("Steps to show", min_value=4, max_value=16, value=8, step=2)

    st.markdown("---")
    st.markdown("### ℹ Model Info")
    st.markdown("""
    <div style='font-size:0.78rem; color:#6b6b8a; line-height:1.8'>
    <b style='color:#e8e8f0'>Architecture</b> — UNet DDPM<br>
    <b style='color:#e8e8f0'>Dataset</b> — CelebA-HQ<br>
    <b style='color:#e8e8f0'>Resolution</b> — 128 × 128<br>
    <b style='color:#e8e8f0'>Timesteps</b> — 500<br>
    <b style='color:#e8e8f0'>Epochs</b> — 50<br>
    <b style='color:#e8e8f0'>Best Loss</b> — 0.01636
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    generate_btn = st.button("🎲  GENERATE")

# ── Main layout ──
col_main, col_final = st.columns([2, 1])
with col_main:
    st.markdown('<div class="step-badge">DENOISING PROCESS</div>', unsafe_allow_html=True)
    steps_placeholder = st.empty()
with col_final:
    st.markdown('<div class="step-badge">FINAL OUTPUT</div>', unsafe_allow_html=True)
    final_placeholder    = st.empty()
    download_placeholder = st.empty()

# ── Load model ──
with st.spinner("Loading model weights..."):
    model, device = load_model()

st.markdown(
    f'<div class="status-box">✓ Model ready · Running on {"GPU" if device.type=="cuda" else "CPU"}</div>',
    unsafe_allow_html=True
)

# ── Generate ──
if generate_btn:
    final_img, snapshots = generate(model, device, num_steps_to_show=n_steps, seed=seed_val)

    with steps_placeholder.container():
        cols_per_row = 4
        for row_start in range(0, len(snapshots), cols_per_row):
            row_snaps = snapshots[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_snaps))
            for col, (t_val, img) in zip(cols, row_snaps):
                with col:
                    st.image(img)
                    noise_pct = round(t_val / TIMESTEPS * 100)
                    st.markdown(
                        f'<div class="steps-label">T={t_val} · {noise_pct}% noise</div>',
                        unsafe_allow_html=True
                    )

    with final_placeholder.container():
        st.image(final_img)
        st.markdown('<div class="steps-label">GENERATED FACE</div>', unsafe_allow_html=True)

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    download_placeholder.download_button(
        label="⬇ Download PNG",
        data=buf.getvalue(),
        file_name=f"ddpm_face_seed{seed_val or 'rand'}.png",
        mime="image/png"
    )

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-label">Timesteps</div>
            <div class="metric-value">500</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Best Loss</div>
            <div class="metric-value">0.016</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Resolution</div>
            <div class="metric-value">128²</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Device</div>
            <div class="metric-value" style="font-size:1rem">{device.type.upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    with steps_placeholder.container():
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem; border-style:dashed;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎲</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.8rem;
                        color:#6b6b8a; letter-spacing:2px;">PRESS GENERATE TO START</div>
            <div style="color:#3a3a55; font-size:0.75rem; margin-top:0.5rem;">
                Intermediate denoising steps will appear here</div>
        </div>
        """, unsafe_allow_html=True)
    with final_placeholder.container():
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem;
                                  border-style:dashed; min-height:200px;">
            <div style="font-size:2rem;">✦</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                        color:#3a3a55; letter-spacing:1px; margin-top:0.5rem;">AWAITING</div>
        </div>
        """, unsafe_allow_html=True)
