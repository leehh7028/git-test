import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# CIFAR10 데이터셋 다운로드
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# VGG-16 모델 생성
model = torchvision.models.vgg16(num_classes=10)
print ("Create model")

# Loss function 및 Optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print ("Loss & Optimizer")

# 학습
for epoch in range(2):  # 데이터셋을 2번 훑음
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 입력 데이터와 레이블
        inputs, labels = data

        # 경사 초기화
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')




