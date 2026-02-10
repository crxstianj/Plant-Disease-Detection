import os
import shutil
import random

# Configuración
dataset_dir = "PlantVillage"
output_dir = "PlantVillageS"
train_ratio = 0.7

# Crear carpetas de salida
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Obtener clases
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Crear carpetas por clase en train/test
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Mover archivos
    for img in train_images:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in test_images:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))

print("Dataset separado")
