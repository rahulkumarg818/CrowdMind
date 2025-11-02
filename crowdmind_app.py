import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
from crowdmind_model import RGB512Autoencoder

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
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

# Streamlit UI
st.set_page_config(page_title="CrowdMind", layout="wide")
st.title("CrowdMind: Crowd Behavior Analysis")

uploaded_video = st.file_uploader("Upload a crowd video", type=["mp4", "avi"])
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

    with st.spinner("Processing video for anomalies..."):
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

    # Error timeline
    st.subheader("Anomaly Score Timeline")
    fig, ax = plt.subplots()
    ax.plot(errors, label="Reconstruction Error")
    ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Error")
    ax.legend()
    st.pyplot(fig)

    # Top anomaly dashboard
    st.subheader("Top Anomalous Frames")
    top_idxs = np.argsort(errors)[-5:][::-1]
    cols = st.columns(len(top_idxs))
    for i, idx in enumerate(top_idxs):
        frame = cv2.resize(frames[idx], FRAME_SIZE)
        timestamp = frame_to_time(idx)
        error_score = errors[idx]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cols[i].image(image, caption=f"Frame {idx}\nTime: {timestamp}\nError: {error_score:.4f}", use_column_width=True)

    # Generate anomaly video
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

    # Save dashboard image
    st.subheader("📸 Download Dashboard Image")
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
