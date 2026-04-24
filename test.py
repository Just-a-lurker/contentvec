import torch
data = torch.load("checkpoint_best.pt")
print("Độ lệch chuẩn:", data.std().item())