import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from numpy import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class facemapdataset(Dataset):
    def __init__(self, data_file="data/facemap_224.pt", transform=None):
        super().__init__()

        self.transform = transform
        self.data, self.targets = torch.load(data_file)
        self.targets = torch.Tensor(self.targets)
        self.targets = torch.nan_to_num(self.targets, nan=1.0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if index % 2 == 1 and self.transform == 'flip':
            image = image.flip([2])  # Flip the image horizontally
            label[::2] = 224 - label[::2]  # Adjust x-coordinates for flipped label positions

        elif index % 2 == 1 and self.transform == 'rotate':
        #elif self.transform == 'rotate':
            # Set a random rotation angle
            angle_image = 15  # Clockwise rotation for the image
            angle_label = -15  # Counterclockwise rotation for the labels

            # Rotate the image
            image = TF.rotate(image, angle_image)

            # Convert angles to radians
            angle_label_rad = torch.tensor(angle_label * 3.14159265 / 180)

            # Center point of the image (assuming 224x224)
            cx, cy = 112, 112

            # Apply opposite rotation transformation to each (x, y) coordinate for the labels
            label = label.clone()
            x_coords, y_coords = label[::2] - cx, label[1::2] - cy  # Shift to origin
            new_x_coords = x_coords * torch.cos(angle_label_rad) - y_coords * torch.sin(angle_label_rad)
            new_y_coords = x_coords * torch.sin(angle_label_rad) + y_coords * torch.cos(angle_label_rad)
            label[::2] = new_x_coords + cx  # Shift back
            label[1::2] = new_y_coords + cy

        # Plotting the image and labels
        #plt.imshow(image.permute(1, 2, 0).numpy())  # Convert to (H, W, C) for plotting
        #plt.scatter(label[::2].numpy(), label[1::2].numpy(), c='red', marker='x')  # Plot labels as red 'x'
        #plt.title("Image with Rotated Labels")
        #plt.axis('off')  # Hide axes for a cleaner plot
        #plt.show()

        return image, label

# Plot all images and corresponding labels in the dataset
def plot_all_images_with_labels(dataset):
    for i in range(len(dataset)):
        # Retrieve the image and label pair
        image, label = dataset[i]

        # Plot the image
        plt.figure(figsize=(4, 4))
        plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap="gray")  # (H, W, C) for plotting
        plt.scatter(label[::2].cpu().numpy(), label[1::2].cpu().numpy(), c="red", marker="x")
        
        # Title and formatting
        plt.title(f"Image {i+1} with Labels")
        plt.axis("off")  # Hide axes for a cleaner plot

        plt.show()

### Make dataset
dataset = facemapdataset(transform='rotate')
#dataset = facemapdataset(transform='flip')  # (transform='flip')


x = dataset[0][0]
dim = x.shape[-1]
print("Using %d size of images" % dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * N), int(0.8 * N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * N), N))
batch_size = 4
# Initialize loss and metrics
loss_fun = torch.nn.MSELoss(reduction="sum")

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print(
    "Num. train = %d, Num. val = %d, Num. test = %d" % (num_train, num_valid, num_test)
)

# Initialize dataloaders
loader_train = DataLoader(
    dataset=dataset,
    drop_last=False,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=train_sampler,
)
loader_valid = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=valid_sampler,
)
loader_test = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=1,
    pin_memory=True,
    sampler=test_sampler,
)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

### hyperparam
lr = 5e-4

num_epochs = 1000

num_input_channels = 1  # Change this to the desired number of input channels
num_output_classes = 24  # Change this to the desired number of output classes


model = timm.create_model(
    "vit_base_patch8_224",
    pretrained=True,
    in_chans=1,
    num_classes=num_output_classes,
    patch_size=224,
)

model = model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%d M" % (nParam / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
minLoss = 1e6
convIter = 0
patience = 1000
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    tr_loss = 0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores = F.softplus(model(inputs))
        loss = loss_fun(torch.log(scores[labels!=0]), torch.log(F.softplus(labels[labels!=0])))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, nTrain, loss.item()
            )
        )
        tr_loss += loss.item()
    train_loss.append(tr_loss / (i + 1))

    with torch.no_grad():
        val_loss = 0
        for i, (inputs, labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = F.softplus(model(inputs))
            loss = loss_fun(torch.log(scores[labels!=0]), torch.log(F.softplus(labels[labels!=0])))
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

        valid_loss.append(val_loss)

        print("Val. loss :%.4f" % val_loss)

        img = inputs.squeeze().detach().cpu().numpy()
        pred = scores.squeeze().detach().cpu().numpy()
        labels = labels.cpu().numpy()
        plt.clf()
        plt.figure(figsize=(16, 6))
        for i in range(batch_size):
            plt.subplot(1, batch_size, i + 1)
            plt.imshow(img[i], cmap="gray")
            plt.plot(pred[i, ::2], pred[i, 1::2], "x", c="tab:red", label="pred.")
            plt.plot(labels[i, ::2], labels[i, 1::2], "o", c="tab:green", label="label")
        plt.tight_layout()
        plt.savefig("logs/epoch_%03d.jpg" % epoch)
        plt.close()


        if minLoss > val_loss:
            convEpoch = epoch
            minLoss = val_loss
            convIter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
        else:
            convIter += 1

        if convIter == patience:
            print(
                "Converged at epoch %d with val. loss %.4f" % (convEpoch + 1, minLoss)
            )
            break
plt.clf()
plt.plot(train_loss, label="Training")
plt.plot(valid_loss, label="Valid")
plt.plot(convEpoch, valid_loss[convEpoch], "x", label="Final Model")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.pdf")

### Load best model for inference
with torch.no_grad():
    val_loss = 0

    for i, (inputs, labels) in enumerate(loader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores = F.softplus(model(inputs))
        loss = loss_fun(torch.log(scores[labels!=0]), torch.log(F.softplus(labels[labels!=0])))
        val_loss += loss.item()

        img = inputs.squeeze().detach().cpu().numpy()
        pred = scores.squeeze().detach().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        plt.clf()
        plt.imshow(img, cmap="gray")
        plt.plot(pred[::2], pred[1::2], "x", c="tab:red")
        plt.plot(labels[::2], labels[1::2], "o", c="tab:green")
        plt.tight_layout()
        plt.savefig("preds/test_%03d.jpg" % i)

    val_loss = val_loss / (i + 1)

    print("Test. loss :%.4f" % val_loss)
