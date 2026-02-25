# ================= IMPORTS =================
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torch import nn, optim
from models import Generator, Discriminator
from metrics import discriminator_accuracy
import os


# ================= SETTINGS =================
device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
batch_size = 128
lr = 0.0002

os.makedirs("static/generated", exist_ok=True)


# ================= TRAIN FUNCTION =================
def train(epochs):

    acc_list = []
    g_list = []
    d_list = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(
        root="data",
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)

    for epoch in range(epochs):

        g_loss_epoch = 0
        d_loss_epoch = 0
        acc_epoch = 0

        for real, _ in loader:

            real = real.to(device)
            bs = real.size(0)

            real_label = torch.ones(bs, 1).to(device)
            fake_label = torch.zeros(bs, 1).to(device)

            # -------- Train D --------
            noise = torch.randn(bs, latent_dim).to(device)
            fake = G(noise)

            real_pred = D(real)
            fake_pred = D(fake.detach())

            d_loss = criterion(real_pred, real_label) + \
                     criterion(fake_pred, fake_label)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # -------- Train G --------
            fake_pred = D(fake)
            g_loss = criterion(fake_pred, real_label)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            acc = discriminator_accuracy(real_pred, fake_pred)

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            acc_epoch += acc

        acc_list.append(acc_epoch/len(loader))
        g_list.append(g_loss_epoch)
        d_list.append(d_loss_epoch)

        print(f"Epoch {epoch+1}/{epochs} | Acc={acc_list[-1]:.3f}")

        # save generated image
        noise = torch.randn(64, latent_dim).to(device)
        fake = G(noise)
        vutils.save_image(
            fake,
            f"static/generated/epoch_{epoch+1}.png",
            normalize=True
        )

    # -------- Save metrics graph --------
    plt.figure()
    plt.plot(acc_list, label="Accuracy")
    plt.plot(g_list, label="G Loss")
    plt.plot(d_list, label="D Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.savefig("static/generated/metrics.png")
    plt.close()

    return acc_list