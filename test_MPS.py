import torch
if torch.backends.mps.is_available():
    print("MPS is available!")
else:
    print("MPS not available.")