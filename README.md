# Plant Disease Detection with CapsNet

## ¿Cómo funciona?

La CapsNet recibe imágenes de hojas de 32×32 px y aprende a clasificarlas en 39 categorías (enfermedades y plantas sanas). A diferencia de una CNN tradicional, las cápsulas codifican relaciones espaciales entre características, lo que las hace más robustas ante variaciones de orientación y escala. El modelo también incluye un **decodificador** que reconstruye la imagen de entrada como señal de regularización durante el entrenamiento.

## Arquitectura
```
Conv2d(3→256) → ReLU
Conv2d(256→512) → ReLU
PrimaryCapsules (4 cápsulas, kernel 9x9, stride 2)
DigitCapsules (39 cápsulas de salida, routing dinámico x3)
↓
Decoder: Linear(16×39 → 256 → 1024 → 32×32×3)

La función de pérdida combina **margin loss** (clasificación) y **reconstruction loss** (MSE sobre la imagen reconstruida).
```

## Estructura del proyecto
```
├── Train.py           # Entrenamiento del modelo
├── Test.py            # Evaluación e inferencia
├── Augmentation.py    # Data augmentation con Albumentations
├── split.py           # División train/test del dataset
└── Model/
└── CapNet.py      # CapsuleLayer, CapsuleNet y CapsuleLoss
```

## Preparación del dataset

El proyecto usa **PlantVillage** (39 clases, ~54,000 imágenes).
```bash
# 1. Generar imágenes aumentadas
python Augmentation.py

# 2. Dividir en train/test (70/30)
python split.py
```
> Dataset NO aumentado: https://data.mendeley.com/datasets/tywbtsjrjv/1  
Las augmentaciones aplicadas son: flip horizontal, ajuste de gamma, ruido gaussiano, desplazamiento RGB, rotación (±30°) y escala. También se genera una versión con todas combinadas.

## Entrenamiento
```bash
python Train.py
```

El modelo se guarda por época en `epochs/capsnet_epoch{N}.pth`. Configuración por defecto:

| Parámetro | Valor |
|-----------|-------|
| Épocas | 200 |
| Batch size | 100 |
| Optimizer | Adam (lr=0.002) |
| Routing iterations | 3 |
| Clases | 39 |

## Dependencias
```bash
pip install torch torchvision albumentations pillow numpy
```
