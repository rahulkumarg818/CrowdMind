import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from crowdmind_model import RGB512Autoencoder

# Replace with your own dataset loader
class CrowdDataset(Dataset):
    def __init__(self, frame_list):
        self.frames = frame_list  # List of NumPy arrays (512x512x3)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx] / 255.0
        tensor = torch.tensor(frame.transpose(2, 0, 1)).float()
        return tensor

# Load your training frames
train_frames = [...]  # Load normal frames here
dataset = CrowdDataset(train_frames)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = RGB512Autoencoder().to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "crowdmind_model.pth")
