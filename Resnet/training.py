import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, trainloader, vaildloader, device, criterion, optimizer):
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="Training")

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch{epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    #vaild data
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
    for inputs, labels in vaildloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct = (predicted == labels).sum().item()
    print("Finished training")