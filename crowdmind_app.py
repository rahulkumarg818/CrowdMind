import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
from crowdmind_model import RGB512Autoencoder

# Glassmorphism + Background Styling
st.markdown("""
    <style>
    body {
        background-image: url("https://images.unsplash.com/photo-1549924231-f129b911e442");
        background-size: cover;
        background-attachment: fixed;
    }
    .glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #00c6ff;
        text-shadow: 1px 1px 2px #000;
    }
    .subtitle {
        font-size: 20px;
        color: #ffffff;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

    

# Branding Header
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="title">CrowdMind</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Crowd Behavior Analysis</div>', unsafe_allow_html=True)

# 🔧 Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RGB512Autoencoder().to(device)
if os.path.exists("crowdmind_model.pth"):
    try:
        model.load_state_dict(torch.load("crowdmind_model.pth", map_location=device))
        model.eval()
    except RuntimeError:
        st.error("Model loading failed. Check if the .pth file matches the model architecture.")
        st.stop()
else:
    st.error("Model file not found. Please upload 'crowdmind_model.pth' to your repo.")
    st.stop()

# Upload Video
uploaded_video = st.file_uploader("🎥 Upload a crowd video", type=["mp4", "avi"])
FRAME_SIZE = (512, 512)
FPS = 20

def frame_to_time(idx):
    seconds = idx / FPS
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def preprocess_frame(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    normalized = resized / 255.0
    tensor = torch.tensor(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)

def compute_error(original, reconstructed):
    return ((original - reconstructed) ** 2).mean().item()

if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    cap = cv2.VideoCapture(temp_video.name)

    errors = []
    frames = []

    with st.spinner("Scanning crowd behavior..."):
        with torch.no_grad():
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                input_tensor = preprocess_frame(frame)
                recon = model(input_tensor).cpu()
                error = compute_error(input_tensor.cpu(), recon)
                errors.append(error)
                idx += 1
            cap.release()

    threshold = np.percentile(errors, 95)
    anomalies = np.where(np.array(errors) > threshold)[0]

    # Dashboard Layout
    
    # Top Anomalous Frames
    st.subheader("Top Anomalous Frames")
    top_idxs = np.argsort(errors)[-5:][::-1]
    cols = st.columns(len(top_idxs))
    for i, idx in enumerate(top_idxs):
        frame = cv2.resize(frames[idx], FRAME_SIZE)
        timestamp = frame_to_time(idx)
        error_score = errors[idx]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cols[i].image(image, caption=f"Frame {idx}\nTime: {timestamp}\nError: {error_score:.4f}", use_column_width=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Anomaly Score Timeline")
        fig, ax = plt.subplots()
        ax.plot(errors, label="Reconstruction Error")
        ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Error")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.metric("Frames Processed", len(errors))
        st.metric("Anomalies Detected", len(anomalies))

    # Generate Anomaly Video
    st.subheader("Downloadable Anomaly Video")
    out_path = "CrowdMind_AnomalyVideo.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, FRAME_SIZE)
    cap = cv2.VideoCapture(temp_video.name)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(errors):
            break
        resized = cv2.resize(frame, FRAME_SIZE)
        timestamp = frame_to_time(frame_idx)
        error_score = errors[frame_idx]
        label = f"CrowdMind | Frame {frame_idx} | Time: {timestamp} | Error: {error_score:.4f}"
        cv2.putText(resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if frame_idx in anomalies:
            cv2.rectangle(resized, (0, 0), (511, 511), (0, 0, 255), 2)
        out.write(resized)
        frame_idx += 1
    cap.release()
    out.release()

    with open(out_path, "rb") as f:
        st.download_button("Download Anomaly Video", f.read(), file_name="CrowdMind_AnomalyVideo.mp4")

    # Save Dashboard Image
    st.subheader("Download Dashboard Image")
    fig, axs = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle("CrowdMind – Top Anomalous Frames", fontsize=16)
    for i, idx in enumerate(top_idxs):
        frame = cv2.resize(frames[idx], FRAME_SIZE)
        timestamp = frame_to_time(idx)
        error_score = errors[idx]
        axs[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Frame {idx}\nTime: {timestamp}\nError: {error_score:.4f}")
        axs[i].axis('off')
    fig.tight_layout()
    fig.savefig("CrowdMind_AnomalyDashboard.png")

    with open("CrowdMind_AnomalyDashboard.png", "rb") as f:
        st.download_button("Download Dashboard Image", f.read(), file_name="CrowdMind_AnomalyDashboard.png")


