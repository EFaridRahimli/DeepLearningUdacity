import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os

import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_data_loaders(path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    train_data = datasets.ImageFolder(os.path.join(path, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(path, "valid"), transform=transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=False),
    )


def net():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, 133)
    return model


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"ValidationAccuracy:{accuracy:.2f}")
    return accuracy


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for im, labels in train_loader:
        im, labels = im.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(im)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_data_loaders(args.data_dir, args.batch_size)
    model = net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    for i in range(args.epochs):
        model = train(model, train_loader, criterion, optimizer, device)
        accuracy = test(model, val_loader, device)
        print(f"Epoch {i+1}: ValidationAccuracy={accuracy}")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    main(args)
