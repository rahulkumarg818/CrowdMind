import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from crowdmind_model import RGB512Autoencoder

# Folder with sample crowd images
FRAME_DIR = "crowd_frames"

class CrowdDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.resize(img, (512, 512))
        img = img / 255.0
        tensor = torch.tensor(img.transpose(2, 0, 1)).float()
        return tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CrowdDataset(FRAME_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = RGB512Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(5):  # Keep it short for dummy training
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "crowdmind_model.pth")
print("✅ Model saved as crowdmind_model.pth")
