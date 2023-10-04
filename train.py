import torch
import glob
import os
from dataset import ChestXrayDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from net import CNNnet
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

folder_path = "./chest_xray/"

train_filenames = glob.glob(os.path.join(folder_path + "train/", "*/*"))
val_filenames = glob.glob(os.path.join(folder_path + "test/", "*/*"))

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def target_transform(label):
    if label == "NORMAL":
        return 0
    elif label == "PNEUMONIA":
        return 1
    else:
        raise ValueError("Unknown label")


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    model.train()

    for i, data in enumerate(training_loader):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.view(-1, 1).float().to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = bce(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / len(training_loader)
        tb_writer.add_scalar("Loss/train", avg_loss, epoch_index + 1)

    return last_loss


BATCH_SIZE = 512

train_data = ChestXrayDataset(
    train_filenames, transform=transform, target_transform=target_transform
)
val_data = ChestXrayDataset(
    val_filenames, transform=transform, target_transform=target_transform
)
training_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


model = CNNnet().to(device)
bce = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
epoch_number = 0
best_vloss = 1_000_000.0
no_improvement_count = 0

EPOCHS = 100
PATIENCE = 10

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.view(-1, 1).float().to(device)
            voutputs = model(vinputs)
            vloss = bce(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1,
    )
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}.pth".format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= PATIENCE:
        print(f"Early stopping after {PATIENCE} epochs without improvement.")
        break

    epoch_number += 1
