import torch
ckpt = torch.load("checkpoint_best_legacy_500.pt", map_location="cpu")
print(ckpt["cfg"]["model"])

