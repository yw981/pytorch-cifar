from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from models import *
import numpy as np
from torch.autograd import Variable


def test(model, criterion, device, test_loader, tag='Test'):
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    temper = 1000
    noiseMagnitude1 = 0.0014

    model.eval()
    test_loss = 0
    correct = 0
    idx = 0

    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device).requires_grad_(), target.to(device)
        # print(data.size())
        # print(target.size())
        outputs = model(data)
        # print(output)
        # print(torch.max(output[0]))
        # exit(0)

        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()

        # print('nnOutputs size', nnOutputs.shape)

        # 这里就是在softmax ？减去最大值，让所有数字都变负数，最大得分0
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, axis=1)
        # print(maxIndexTemp)
        # print(target)
        # labels = torch.LongTensor(np.array([maxIndexTemp])).to(device)
        # labels = Variable(torch.LongTensor(np.array([maxIndexTemp])).cuda())
        labels = torch.LongTensor(maxIndexTemp).to(device)
        # print(labels)

        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
        # print(inputs.grad.data)
        # print(gradient.size())
        print(torch.mean(gradient))
        # Normalizing the gradient to the same space of image
        gradient[:, 0, :, :] = gradient[:, 0, :, :] / (63.0 / 255.0)
        gradient[:, 1, :, :] = gradient[:, 1, :, :] / (62.1 / 255.0)
        gradient[:, 2, :, :] = gradient[:, 2, :, :] / (66.7 / 255.0)
        print(torch.mean(gradient))
        # Adding small perturbations to images
        # add函数的规则 torch.add(input, value=1, other, out=None) out=input+value×other
        tempInputs = torch.add(data, -noiseMagnitude1, gradient)
        outputs = model(tempInputs)
        print('output')
        # outputs = net1(Variable(inputs.data))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # batch_loss = criterion(outputs, target).item()
        # test_loss += batch_loss
        # pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # correct += pred.eq(target.view_as(pred)).sum().item()
        # if idx % 10 == 0:
        #     total_test = (idx + 1) * data.size()[0]
        #     print(idx, ' loss ', test_loss, ' ', correct, '/', total_test, ' acc ', correct / total_test)
        # idx += 1

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        tag, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    batch_size = 4

    # densenet 自训练 cifar10
    model_path = '../model/densenet121_cifar.pth'
    net = DenseNet121().to(device)
    # Test set: Average loss: 0.0010, Accuracy: 9516 / 10000(95 %)

    # vgg16 自训练 cifar10
    # model_path = '../model/vgg16_cifar10.pth'
    # net = VGG('VGG16').to(device)
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
