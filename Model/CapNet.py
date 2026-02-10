import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 39
NUM_ROUTING_ITERATIONS = 3

# ====================
# Funciones auxiliares
# ====================
def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)


# ====================
# Definición de cápsulas
# ====================
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels,
                 kernel_size=None, stride=None, num_iterations=NUM_ROUTING_ITERATIONS):
        super().__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
            )
        else:
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
                for _ in range(num_capsules)
            ])

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size(), device=x.device)
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    logits = logits + (priors * outputs).sum(dim=-1, keepdim=True)
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = squash(outputs)
        return outputs


# ====================
# Modelo CapsNet
# ====================
class CapsuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=5, stride=1)
        # self.batch1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=5, stride=1)
        # self.batch2 = nn.BatchNorm2d(512)
        self.primary_capsules = CapsuleLayer(
            num_capsules=4, num_route_nodes=-1, in_channels=512, out_channels=32,
            kernel_size=9, stride=2
        )
        self.digit_capsules = CapsuleLayer(
            num_capsules=NUM_CLASSES, num_route_nodes=32*8*8, in_channels=4, out_channels=16
        )

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32*32*3),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        batch_size = x.size(0)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        x = x.squeeze(dim=3).squeeze(dim=2).permute(1, 0, 2)

        classes = (x ** 2).sum(dim=-1).sqrt()
        classes = F.softmax(classes, dim=-1)

        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = torch.zeros(batch_size, NUM_CLASSES, device=x.device)
            y.scatter_(1, max_length_indices.unsqueeze(1), 1.0)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))
        return classes, reconstructions


# ====================
# Pérdida CapsNet
# ====================
class CapsuleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="sum")

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        images = images.view(reconstructions.size(0), -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
