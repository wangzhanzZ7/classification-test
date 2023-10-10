from torch.utils.data import Dataset
from PIL import Image
import os
import gzip
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

class FashionMnist(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.labels = self.load_image()

    def load_image(self):
        kind = f"{'train' if self.train else 't10k'}"
        labels_path = os.path.join(self.root,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(self.root,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        # 未解压
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels


    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]

        img = Image.fromarray(img.reshape(28,28), mode="L")
        label = int(label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.images)


def build_loader(batch_size):
    # 定义transform和target_transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = FashionMnist('/media/ps/data/wz/dataset/fashion_mnist',
                             train=True, transform=transform)
    test_data = FashionMnist('/media/ps/data/wz/dataset/fashion_mnist',
                             train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == '__main__':
    # 训练集
    # dataset = FashionMnist(r"D:\PaperLearning\Code\classification-test\data\fashion-mnist\data\fashion")
    # img,label = dataset[0]
    #     # img.show()
    #     # print(type(label))


    # train_data, test_data = build_loader()
    # print(train_data.__len__())
    # print(test_data.__len__())

    train_data, test_data, train_loader, test_loader = build_loader()
    img, label = train_data[0]
    print(img.shape)
    print(label)



