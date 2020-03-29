from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from models import *
import numpy as np
import torch.utils.data as data


def test(model, criterion, device, test_loader, tag='Test', save_result=False, save_file_path='./result.txt'):
    model.eval()
    test_loss = 0
    correct = 0
    idx = 0
    results = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(data.size())
            # print(target.size())
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            batch_loss = criterion(output, target).item()
            test_loss += batch_loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if save_result:
                results.append(output.cpu().numpy())

            if idx % 100 == 0:
                # print(target)
                total_test = (idx + 1) * data.size()[0]
                print(idx, ' loss ', test_loss, ' ', correct, '/', total_test, ' acc ', correct / total_test)
            idx += 1

    if save_result:
        results = np.array(results)
        results = results.reshape(-1, results.shape[-1])
        np.save(save_file_path, results)
        print('file saved ', save_file_path, ' shape ', results.shape)

    test_loss /= len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        tag, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class RandomDataset(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        np.random.seed(0)
        # gaussian
        # self.data = np.random.randn(10000, 3, 32, 32).astype(np.float32)
        # uniform
        self.data = np.random.rand(10000, 3, 32, 32).astype(np.float32)
        # print('self.test_data.shape', self.test_data.shape)
        self.labels = np.zeros((10000,)).astype(np.int64)

    def __getitem__(self, index):
        img,target = self.data[index],self.labels[index]
        # 此处应该是(H x W x C)，因此调整
        # img = img.transpose(1, 2, 0)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    batch_size = 200

    # densenet 自训练 cifar10
    # model_path = '../model/densenet121_cifar.pth'
    # net = DenseNet121().to(device)
    # Test set: Average loss: 0.0010, Accuracy: 9516 / 10000(95 %)

    # vgg16 自训练 cifar10
    model_path = '../model/vgg16_cifar10.pth'
    net = VGG('VGG16').to(device)
    # Test set: Average loss: 0.0015, Accuracy: 9337 / 10000(93 %)

    if use_cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(model_path, ' epoch ', start_epoch, ' acc ', best_acc)

    # cifar10 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    # 若不Normalize Test set: Average loss: 0.0063, Accuracy: 7169 / 10000(72 %)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    # 处理过的TinyImagenet 数据，借用cifar10的transform
    # out_dataset_path = "../data/Imagenet"
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(out_dataset_path, transform=transform),
    #     batch_size=batch_size, shuffle=False, num_workers=4)

    # 随机数据
    # test_loader = torch.utils.data.DataLoader(
    #     RandomDataset(),
    #     batch_size=batch_size, shuffle=False, num_workers=4)

    # test(model,criterion, device, train_loader,'Train')
    # test(net, criterion, device, test_loader)
    # test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_densenet_imagenet.npy')
    # test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_densenet_gaussian.npy')
    # test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_densenet_uniform.npy')
    # test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_vgg_imagenet.npy')
    # test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_vgg_gaussian.npy')
    # test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_vgg_uniform.npy')
    test(net, criterion, device, test_loader, save_result=True, save_file_path='result/result_vgg_cifar.npy')
