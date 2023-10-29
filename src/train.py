# from network.model import Generator, Discriminator
from dataloader.dataloader import ImageLoader

# from torch import nn
# from torch.optim import Adam
# import torch
from utils.utils import read_config  # , get_device

# import mlflow

# config = read_config("./config/config.yml")
# seed = config["seed"]


class TrainerCycleGan:
    def __init__(self, config_path: str) -> None:
        config = read_config(config_path)
        self.lr = config["train"]["lr"]
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.num_workers = config["train"]["num_workers"]
        self.weight_decay = config["train"]["weight_decay"]
        self.experiment_name = config["train"]["experiment_name"]
        self.image_size = config["train"]["image_size"]
        train_dataset_path = config["train"]["data"]
        train_label_path = config["train"]["label"]
        val_dataset_path = config["val"]["data"]
        val_label_path = config["val"]["label"]
        self.train_loader = ImageLoader(train_dataset_path, train_label_path, self.image_size)
        self.val_loader = ImageLoader(val_dataset_path, val_label_path, self.image_size)

    def training_loop(self):
        pass

    def validation_loop(self):
        pass

    def training(self):
        pass


# G_T1c_to_T2 = Generator()
# G_T2_to_T1c = Generator()
# D_T1c = Discriminator()
# D_T2 = Discriminator()

# criterion_GAN = nn.MSELoss()
# criterion_cycle = nn.L1Loss()

# optimizer_G = Adam(
#     list(G_T1c_to_T2.parameters()) + list(G_T2_to_T1c.parameters()), lr=0.0002, betas=(0.5, 0.999)
# )
# optimizer_D_T1c = Adam(D_T1c.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D_T2 = Adam(D_T2.parameters(), lr=0.0002, betas=(0.5, 0.999))

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         real_T1c = batch["T1c_image"].to(get_device())
#         real_T2 = batch["T2_image"].to(get_device())

#         # Generator updates
#         optimizer_G.zero_grad()

#         # Forward pass through generators
#         fake_T2 = G_T1c_to_T2(real_T1c)
#         recon_T1c = G_T2_to_T1c(fake_T2)

#         # GAN loss
#         loss_GAN_T2 = criterion_GAN(D_T2(fake_T2), torch.ones_like(D_T2(fake_T2)))
#         loss_GAN_T1c = criterion_GAN(D_T1c(recon_T1c), torch.ones_like(D_T1c(recon_T1c)))

#         # Cycle loss
#         loss_cycle_T1c = criterion_cycle(recon_T1c, real_T1c)
#         loss_cycle_T2 = criterion_cycle(fake_T2, real_T2)

#         # Total generator loss
#         loss_G = loss_GAN_T2 + loss_GAN_T1c + loss_cycle_T1c + loss_cycle_T2
#         loss_G.backward()
#         optimizer_G.step()

#         # Discriminator updates
#         optimizer_D_T1c.zero_grad()
#         optimizer_D_T2.zero_grad()

#         loss_D_T1c_real = criterion_GAN(D_T1c(real_T1c), torch.ones_like(D_T1c(real_T1c)))
#         loss_D_T1c_fake = criterion_GAN(
#             D_T1c(recon_T1c.detach()), torch.zeros_like(D_T1c(recon_T1c.detach()))
#         )
#         loss_D_T2_real = criterion_GAN(D_T2(real_T2), torch.ones_like(D_T2(real_T2)))
#         loss_D_T2_fake = criterion_GAN(
#             D_T2(fake_T2.detach()), torch.zeros_like(D_T2(fake_T2.detach()))
#         )

#         loss_D_T1c = 0.5 * (loss_D_T1c_real + loss_D_T1c_fake)
#         loss_D_T2 = 0.5 * (loss_D_T2_real + loss_D_T2_fake)

#         loss_D_T1c.backward()
#         loss_D_T2.backward()
#         optimizer_D_T1c.step()
#         optimizer_D_T2.step()
