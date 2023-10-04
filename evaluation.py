import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ChestXrayDataset
from net import CNNnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "./chest_xray/"
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


validation_dataset = ChestXrayDataset(
    val_filenames, transform=transform, target_transform=target_transform
)

validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=True)

model = CNNnet().to(device)

model.load_state_dict(torch.load("./weight/model_20231004_055715_12.pth"))
model.eval()

criterion = nn.BCELoss()

total_correct = 0
total_samples = 0
total_loss = 0.0

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        labels = labels.view(-1, 1).float().to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
average_loss = total_loss / len(validation_loader)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Loss: {average_loss:.4f}")
