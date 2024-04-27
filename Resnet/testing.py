import torch

def test(model, testloader, device, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Accuracy of the network on CIFAR100: {100 * correct / total}%")
        print()

        class_correct = list(0. for i in range(100))
        class_total = list(0. for i in range(100))

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    print("-------------------------------------------")
    for i in range(100):
        print(f"Accuracy of {classes[i]} : {100 * class_correct[i]/class_total[i]}%")