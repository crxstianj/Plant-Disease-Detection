import os
import re
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Model.CapNet import CapsuleNet, NUM_CLASSES
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CapsuleNet().to(DEVICE)
model.load_state_dict(torch.load("epochs/capsnet_epoch182.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4668, 0.4891, 0.4120], std=[0.1568, 0.1289, 0.1753])
])

val_dataset = datasets.ImageFolder("Prueba", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

technique_correct = defaultdict(int)
technique_total = defaultdict(int)

all_labels = []
all_preds = []
all_probs = []  # Para curvas ROC

technique_class_correct = defaultdict(lambda: defaultdict(int))
technique_class_total = defaultdict(lambda: defaultdict(int))

start_time = time.time()

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        batch_start = time.time()  # tiempo inicio batch

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        classes, _ = model(images)
        probs = F.softmax(classes, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        start_idx = batch_idx * val_loader.batch_size
        end_idx = start_idx + images.size(0)
        paths = [val_dataset.samples[i][0] for i in range(start_idx, end_idx)]

        for path, pred, label in zip(paths, preds, labels):
            filename = os.path.basename(path)
            match = re.match(r"(.*)_image", filename)
            technique = match.group(1) if match else "unknown"

            # Conteo global
            technique_total[technique] += 1
            if pred == label:
                technique_correct[technique] += 1

            # Conteo por clase
            class_idx = label.item()
            technique_class_total[technique][class_idx] += 1
            if pred == label:
                technique_class_correct[technique][class_idx] += 1

        batch_end = time.time()  # tiempo fin batch
        print(f"Batch {batch_idx+1}/{len(val_loader)} procesado en {batch_end - batch_start:.2f} s")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTiempo total de inferencia sobre {len(val_dataset)} imágenes: {total_time:.2f} s")
print(f"Tiempo promedio por imagen: {total_time/len(val_dataset):.4f} s")

# ---------------------------
# Resultados por técnica
# ---------------------------
print("\nResultados por técnica de aumento:")
for technique in sorted(technique_total.keys()):
    acc = 100 * technique_correct[technique] / technique_total[technique]
    print(f"{technique:15s} -> {acc:.2f}% ({technique_correct[technique]}/{technique_total[technique]})")

total_correct = sum(technique_correct.values())
total_images = sum(technique_total.values())
print(f"\nAccuracy total: {100*total_correct/total_images:.2f}%")

# ---------------------------
# Resultados por técnica y por clase
# ---------------------------
print("\nResultados por técnica y por clase (CapsNet):")
for technique in sorted(technique_class_total.keys()):
    print(f"\nTécnica: {technique}")
    for class_idx in range(NUM_CLASSES):
        total = technique_class_total[technique][class_idx]
        correct = technique_class_correct[technique][class_idx]
        acc = 100 * correct / total if total > 0 else 0
        print(f"  Clase {val_dataset.classes[class_idx]:30s} -> {acc:.2f}% ({correct}/{total})")

# ---------------------------
# Métricas globales
# ---------------------------
print("\nReporte de clasificación (precision, recall, f1 por clase):")
print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

# ---------------------------
# Matriz de confusión con números y etiquetas horizontales
# ---------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusión", fontsize=18)
plt.colorbar()


tick_marks = np.arange(len(val_dataset.classes))
short_labels = [f"C{i}" for i in range(len(val_dataset.classes))]
plt.xticks(tick_marks, short_labels, rotation=0, fontsize=15)
plt.yticks(tick_marks, short_labels, fontsize=15)

# Números dentro de cada celda
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

plt.ylabel("Etiqueta real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.show()

# ---------------------------
# Curvas ROC
# ---------------------------
y_true = np.array(all_labels)
y_score = np.array(all_probs)
n_classes = NUM_CLASSES

# One-hot encoding de etiquetas
y_true_bin = np.zeros((y_true.size, n_classes))
y_true_bin[np.arange(y_true.size), y_true] = 1

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Clase {val_dataset.classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("Curvas ROC por clase")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.legend(loc="lower right")
plt.show()
