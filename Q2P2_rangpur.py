import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Running on CPU.")

# Hyper-parameters
num_epochs = 40
learning_rate = 0.1
num_classes = 10

# Normalise data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect')
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Batch size
batch_size_train = 128
batch_size_test = 100

# Create train and test data
train_data = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
test_data = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, num_workers=2)

# Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 1st Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Pytorch has downsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # For every one of two blocks in a layer, first will downsample by a factor of 2,
        # the second one will compute the convolutional layer
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])    

model = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()

# SGD does not change learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Piecewise learning rate scheduler
total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=learning_rate, steps_per_epoch=total_step, epochs=num_epochs
)

# Construct scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Train the model
model.train()
print("> Training")
start = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scale loss. Calls backward on scaled loss to create scaled gradients
        scaler.scale(loss).backward()

        # First unscales the gradients of the optimiser's assigned parameters. If these gradients do not contain infs or NaNs,
        # optimizer.step() is then called, otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        # Backwards and optimise
        optimizer.zero_grad()

        if (i+1) % 300 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch+1, num_epochs, i+1, total_step, loss.item()
            ))

        scheduler.step()
end = time.time()
elapsed = end - start
print("Training time: {:.2f} sec or {:.2f} min".format(elapsed, elapsed / 60))

# Test the model
model.eval()
print("> Testing")
start = time.time()
with torch.no_grad():
    correct= 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        # Count number of correct predictions
        correct += (predicted == labels).sum().item()
    print("Test Accuracy: {} %".format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing time: {:.2f} sec or {:.2f} min".format(elapsed, elapsed / 60))