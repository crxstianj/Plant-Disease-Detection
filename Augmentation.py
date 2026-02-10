import os
from PIL import Image
import numpy as np
import albumentations as A

# Transformaciones individuales
individual_transforms = {
    "horizontal_flip": A.HorizontalFlip(p=1.0),
    "gamma": A.RandomGamma(gamma_limit=(80, 120), p=1.0),
    "gauss_noise": A.GaussNoise(p=1.0),
    "rgb_shift": A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
    "rotate": A.Rotate(limit=(-30,30), p=1.0),
    "scale": A.Affine(scale=(0.9, 1.1), p=1.0)
}

# Transformación combinada
combined_transform = A.Compose(list(individual_transforms.values()), p=1.0)

# Directorios
input_dir = "PlantVillage"
output_dir = "PlantVillagea"
os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Imágenes por técnica individual
        for tech_name, transform in individual_transforms.items():
            augmented = transform(image=image_np)
            aug_image = Image.fromarray(augmented["image"])
            save_path = os.path.join(output_class_path, f"{tech_name}_{img_name}")
            aug_image.save(save_path)

        # Imagen con todas combinadas
        augmented_combined = combined_transform(image=image_np)
        aug_combined_image = Image.fromarray(augmented_combined["image"])
        save_path_combined = os.path.join(output_class_path, f"combined_{img_name}")
        aug_combined_image.save(save_path_combined)
