import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
from PIL import ImageFile

import smdebug.pytorch as smd
from smdebug import modes

ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    )

def net():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    return model

def train(model, train_loader, criterion, optimizer, device, hook):
    hook.set_mode(modes.TRAIN)
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        hook.record_tensor_value(tensor_name="loss", tensor_value=loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"TrainLoss: {running_loss / len(train_loader):.4f}")
    return running_loss / len(train_loader)

def test(model, test_loader, device, criterion, hook):
    hook.set_mode(modes.EVAL)
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    print(f"TestLoss: {running_loss / len(test_loader):.4f}")
    print(f"TestAccuracy: {accuracy:.2f}%")
    return accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_data_loaders(args.data_dir, args.batch_size)
    model = net().to(device)

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    criterion = nn.CrossEntropyLoss()
    hook.register_loss(criterion)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, hook)
        val_accuracy = test(model, val_loader, device, criterion, hook)

        if args.model_dir:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    main(args)
