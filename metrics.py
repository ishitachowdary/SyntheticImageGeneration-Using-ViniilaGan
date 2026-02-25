import torch

def discriminator_accuracy(real_pred, fake_pred):
    real_acc = (real_pred > 0.5).float().mean()
    fake_acc = (fake_pred < 0.5).float().mean()
    return ((real_acc + fake_acc) / 2).item()