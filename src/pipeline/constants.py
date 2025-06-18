import torch 


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'