from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
import numpy as np
import torchvision
import time
from models import DenseNet121, VGG
import torch.backends.cudnn as cudnn


def cal_scores_save(base_path, new_path, model, criterion, device, test_loader, tag='Test'):
    use_temperature = True
    use_gradient = True
    f1 = open(base_path, 'w')
    g1 = open(new_path, 'w')
    temper = 200
    noiseMagnitude1 = 0.000064

    print('Process ', tag, ' temperature = ', temper, ' epsilon = ', noiseMagnitude1)

    model.eval()
    t0 = time.time()
    idx = 0

    for data, target in test_loader:
        data = Variable(data.cuda(0), requires_grad=True)
        # data = data.to(device).requires_grad_()

        outputs = model(data)
        # print(outputs)
        # outputs = torch.tensor(
        #     [[-1.2211, -0.9080, 1.4694, 17.9587, -2.6866, 0.1167, -0.6995, -4.0835, -3.9281, -6.0281]]).to(device)
        # outputs.requires_grad_()

        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        # print(np.argmax(nnOutputs, axis=1))
        # print(target)

        # 这里就是在softmax 减去最大值，让所有数字都变负数，最大得分0，用处不大，去掉
        # nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        for i in range(nnOutputs.shape[0]):
            # print("{}, {}, {}".format(temper, noiseMagnitude1, np.max(nnOutputs[i])))
            f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[i])))

        if use_temperature:
            # Using temperature scaling
            outputs = outputs / temper

        if use_gradient:
            # the sign of gradient of cross entropy loss w.r.t. input
            maxIndexTemp = np.argmax(nnOutputs, axis=1)
            # print(maxIndexTemp)
            # print(target)
            # labels = torch.LongTensor(maxIndexTemp).to(device)
            labels = Variable(torch.LongTensor(maxIndexTemp)).cuda(0)
            # labels = Variable(torch.LongTensor(np.array([maxIndexTemp])).cuda(0))
            # print(labels)

            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(data.grad, 0)
            gradient = (gradient.float() - 0.5) * 2
            # print(inputs.grad.data)
            # print(torch.mean(gradient))
            # Normalizing the gradient to the same space of image
            gradient[:, 0, :, :] = gradient[:, 0, :, :] / (63.0 / 255.0)
            gradient[:, 1, :, :] = gradient[:, 1, :, :] / (62.1 / 255.0)
            gradient[:, 2, :, :] = gradient[:, 2, :, :] / (66.7 / 255.0)
            # print(gradient)
            # Adding small perturbations to images
            # add函数的规则 torch.add(input, value=1, other, out=None) out=input+value×other
            tempInputs = torch.add(data, -noiseMagnitude1, gradient)
            outputs = model(tempInputs)
            # print('output')
            # outputs = net1(Variable(inputs.data))
            if use_temperature:
                outputs = outputs / temper

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        # print(np.max(nnOutputs, axis=1))
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        for i in range(nnOutputs.shape[0]):
            # print(nnOutputs[i])
            # print("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[i])))
            g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[i])))

        if idx % 10 == 0 or (idx + 1) == len(test_loader):
            print("{:4}/{:4} batch processed, {:.1f} seconds used.".format(idx, len(test_loader), time.time() - t0))
            t0 = time.time()
        idx += 1


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device = 'cuda'
    criterion = torch.nn.CrossEntropyLoss()

    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    batch_size = 50
    num_worker = 2

    # 模型！

    # densenet 自训练 cifar10
    model_path = '../model/densenet121_cifar.pth'
    net = DenseNet121().to(device)
    # Test set: Average loss: 0.0010, Accuracy: 9516 / 10000(95 %)

    # Python 官方 densenet 自训练 cifar10
    # model_path = '../model/of_densenet_cifar10_ep200.pth'
    # net = torchvision.models.densenet.densenet121(drop_rate=0,
    #                                               # num_classes=10
    #                                               ).to(device)

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
        batch_size=batch_size, shuffle=False, num_workers=num_worker
    )

    result_save_path = './softmax_scores/'
    base_in_path = result_save_path + "confidence_Base_In.txt"
    base_out_path = result_save_path + "confidence_Base_Out.txt"
    new_in_path = result_save_path + "confidence_Our_In.txt"
    new_out_path = result_save_path + "confidence_Our_Out.txt"
    out_dataset_path = "../data/Imagenet"

    out_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(out_dataset_path, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=num_worker)

    cal_scores_save(base_in_path, new_in_path, net, criterion, device, test_loader, 'In ')
    cal_scores_save(base_out_path, new_out_path, net, criterion, device, out_loader, 'Out ')
