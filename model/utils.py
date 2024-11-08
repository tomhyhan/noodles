import os
import random
import numpy as np
import torch

def test_colab():
    print("Hello Colab! from Noodles ha")

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def save_model(model, optimizer, loss_scaler, epoch, out_file):
    
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'epoch': epoch
    }
    torch.save(to_save, out_file)

def load_model(model, optimizer, loss_scaler, out_file):
    checkpoint = torch.load(out_file, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_scaler.load_state_dict(checkpoint['scaler'])
    return int(checkpoint['epoch']) + 1

