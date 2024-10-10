import torch


def validate_model(model, data_loader, criterion):

    model.eval()
    running_loss, correct_val = 0.0, 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.clone())

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss / len(data_loader)
    val_acc = correct_val / len(data_loader.dataset)

    return val_loss, val_acc
