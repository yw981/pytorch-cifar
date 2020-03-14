from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from models import *


def test(model, criterion, device, test_loader, tag='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    idx = 0
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
            if idx % 100 == 0:
                total_test = (idx + 1) * data.size()[0]
                print(idx, ' loss ', test_loss, ' ', correct, '/', total_test, ' acc ', correct / total_test)
            idx += 1

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        tag, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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

    # test(model,criterion, device, train_loader,'Train')
    test(net, criterion, device, test_loader)
