import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import models



class ResNetPatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.out_channels = backbone.layer4[-1].conv2.out_channels
        self.proj = nn.Conv2d(self.out_channels, 768, 1, 1, 0)
        self.fc = nn.Linear(768, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.proj(x)
        #print(x.shape)
        x = x.flatten(1)
        x = self.fc(x)

        return x

model = ResNetPatchEmbedding()

for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
model.proj = nn.Conv2d(512, 768, 1, 1)
model.fc = nn.Linear(model.fc.in_features, 100)

model.load_state_dict(torch.load('model_resnet1.pth'))

from torchvision.transforms.functional import pad

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
model.train()
summary(model,(3,32,32))




num_epochs = 10
train_losses = []
train_correct = 0
train_total = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda')).to('cpu')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader.dataset))
    train_accuracy = train_correct / train_total
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

print(f'Finished fine-tuning with {train_accuracy} accuracy')

torch.save(model.state_dict(), 'model_resnet1.pth')


def plot_training_loss(trainloss, title="Training Loss", save_path=None):

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(trainloss) + 1)

    plt.plot(epochs, trainloss, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    min_loss = min(trainloss)
    min_epoch = trainloss.index(min_loss) + 1
    plt.plot(min_epoch, min_loss, 'ro', markersize=5, label=f'Min: {min_loss:.4f}')
    plt.legend(fontsize=11)

    plt.tight_layout()

    plt.savefig('./resnet.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved to {save_path}")
    plt.close()

plot_training_loss(train_losses)
with open('resnet.txt', 'w', encoding='utf-8') as file:
    for loss in train_losses:
        file.write(str(loss) + '\n')