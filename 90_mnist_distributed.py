import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 初始化分布式训练环境
def init_distributed(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)

# 平均同步网络参数
def average_gradients(model, world_size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 分布式训练函数
def train(rank, world_size, model, train_loader):
    # 设置优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据和目标移到相应的设备上
            data = data.to(rank)
            target = target.to(rank)

            # 在每个设备上进行正向传播和计算损失
            output = model(data)
            loss = criterion(output, target)

            # 在每个设备上进行反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()

            # 同步梯度
            average_gradients(model, world_size)

            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx, len(train_loader), loss.item()))

# 主函数
def main():
    # 设置分布式训练参数
    rank = 0  # 当前进程的rank
    world_size = 1  # 总进程数

    # 初始化分布式训练环境
    init_distributed(rank, world_size)

    # 创建模型和数据加载器
    model = Net()
    model = model.to(torch.device("cuda"))
    model = model.train()

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)

    # 在每个进程上运行训练函数
    train(rank, world_size, model, train_loader)

# 运行主函数
if __name__ == '__main__':
    main()