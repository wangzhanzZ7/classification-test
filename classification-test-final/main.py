import torch
import os
from model.resnet50 import ResNet50
from data.fashion_mnist import build_loader
from torch import optim
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from itertools import product


def main():
    # 参数
    epochs = 100
    parameters = dict(
        lr = [.01, .001],
        batch_size = [32, 64, 128]
    )
    param_values = [v for v in parameters.values()]
    loss_fn = torch.nn.CrossEntropyLoss()

    for lr, batch_size in product(*param_values):
        comment = f' batch_size={batch_size} lr={lr}'
        
        # model
        model = ResNet50()
        model.cuda()

        # data
        train_loader, test_loader = build_loader(batch_size)

        # train
        loss_fn.cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        max_accuracy = 0.0
        total_train_step = 0

        writer = SummaryWriter(comment=comment)

        # 模型权重文件夹
        folder_name = f"{lr}_{batch_size}"
        folder_path = os.path.join('/media/ps/data/wz/save_models', folder_name)
        os.makedirs(folder_path)

        for epoch in range(epochs):
            # train
            total_train_step = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, epochs, writer, total_train_step)

            # val
            acc1 = validate(model, test_loader, epoch, writer)

            # save model
            if acc1 > max_accuracy:
                max_accuracy = acc1
                save_path = os.path.join(folder_path, f'ckpt_epoch_{epoch}.pth')
                print(f"{save_path} saving...")
                torch.save(model.state_dict(), save_path)
                print(f"{save_path} saved")

            print(f'current Max accuracy: {max_accuracy:.2f}%')

        writer.close()



def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, epochs, writer, total_train_step):
    model.train()

    num_steps = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        acc1 = accuracy(outputs, labels, topk=(1,))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 50 == 0:
            print(f'Train: [{epoch}/{epochs}][{i}/{num_steps}]\t')
            print(f'Train loss: {loss.item():.3f}\t')  # loss.item()表示获取当前批次的平均损失值
            print(f'Train accuracy: {acc1[0].item():.3f}\t')
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            writer.add_scalar("train_accuracy", acc1[0].item(), total_train_step)
        
    return total_train_step


@torch.no_grad()
def validate(model, test_loader, epoch, writer):
    model.eval()

    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        acc1 = accuracy(outputs, labels, topk=(1,))

    print(f'Accuracy1 on test images: {acc1[0].item():.3f}')
    writer.add_scalar("test_accuracy", acc1[0].item(), epoch)

    return acc1[0].item()


if __name__ == '__main__':
    main()
