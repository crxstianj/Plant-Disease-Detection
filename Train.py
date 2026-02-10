import torch
import torch.nn.functional as F
from Model.CapNet import CapsuleNet, CapsuleLoss, NUM_CLASSES
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
NUM_EPOCHS = 200

# ====================
# Funciones de entrenamiento/evaluación
# ====================
def train(model, train_loader, test_loader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels_onehot = F.one_hot(labels, NUM_CLASSES).float()

            classes, reconstructions = model(images, labels_onehot)
            loss = criterion(images, labels_onehot, classes, reconstructions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = classes.max(1)[1]
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        test_acc, test_loss = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        torch.save(model.state_dict(), f"epochs/capsnet_epoch{epoch+1}.pth")


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels_onehot = F.one_hot(labels, NUM_CLASSES).float()

            classes, reconstructions = model(images, labels_onehot)
            loss = criterion(images, labels_onehot, classes, reconstructions)

            total_loss += loss.item()
            preds = classes.max(1)[1]
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total, total_loss / len(test_loader)


# ====================
# Main
# ====================
if __name__ == "__main__":
    #EJECUTAR SOLO LA PRIMERA VEZ PARA OBTENER VALORES
    # ====================
    # Transform inicial solo para Tensor (sin normalizar)
    # ====================
    # base_transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor()
    # ])
    #
    # # Cargar datasets
    # train_dataset = datasets.ImageFolder("PlantVillageS/train", transform=base_transform)
    # test_dataset = datasets.ImageFolder("PlantVillageS/test", transform=base_transform)
    #
    # # DataLoader para calcular mean y std
    # loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    #
    # mean = 0.0
    # std = 0.0
    # nb_samples = 0
    #
    # for data, _ in loader:
    #     batch_samples = data.size(0)
    #     data = data.view(batch_samples, data.size(1), -1)  # (batch, channels, H*W)
    #     mean += data.mean(2).sum(0)
    #     std += data.std(2).sum(0)
    #     nb_samples += batch_samples
    #
    # mean /= nb_samples
    # std /= nb_samples
    #
    # print("Media por canal:", mean)
    # print("Desviación estándar por canal:", std)

    # ====================
    # Ahora sí aplicamos normalización con los valores calculados
    # ====================
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4668, 0.4891, 0.4120], std=[0.1568, 0.1289, 0.1753])
    ])

    # Volvemos a cargar datasets con normalización
    train_dataset = datasets.ImageFolder("PlantVillageS/train", transform=transform)
    test_dataset = datasets.ImageFolder("PlantVillageS/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo, optimizador y criterio
    model = CapsuleNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = CapsuleLoss()

    print("Entrenando CapsNet en PlantVillage Completo...")
    train(model, train_loader, test_loader, optimizer, criterion, num_epochs=NUM_EPOCHS)
