from __future__ import print_function
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import argparse
from torch.autograd import Variable
from func import RESULT_DIR
from models import DenseNet3

CUDA_DEVICE = 0


def test_save(model, data, label, temperature=-1, softmax_flag=False):
    # print(data.size())
    # if type(data) is torch.FloatTensor:
    #     data = Variable(data.cuda(CUDA_DEVICE))
    #
    # data = Variable(data.data.cpu().cuda(CUDA_DEVICE))
    data = Variable(data.cuda(0), requires_grad=True)
    # data = data.to(device).requires_grad_()
    outputs = model.forward(data)

    use_temperature = True
    use_gradient = True
    temper = 1000
    noiseMagnitude1 = 0.0014

    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    # print(np.argmax(nnOutputs, axis=1))
    # print(target)

    # 这里就是在softmax 减去最大值，让所有数字都变负数，最大得分0，用处不大，去掉
    # nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

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

        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
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

    np.save(RESULT_DIR + '/%s.npy' % label, nnOutputs)
    print(label, ' saved')


def test_origin_and_affines(affine_params, model, data, tag, temperature=-1):
    # data = data.to(device)
    test_save(model, data, tag, temperature)
    for i in range(10):
        label = '%s_%d' % (tag, i)
        affine_param = torch.from_numpy(affine_params[i]).float()
        grid = F.affine_grid(affine_param.repeat(data.size()[0], 1, 1), data.size())
        trans_data = F.grid_sample(data, grid)
        test_save(model, trans_data, label, temperature)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--affine', type=str, default='affine_params_random.npy', metavar='N',
                        help='affine array file name')

    args = parser.parse_args()
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1234)

    # Robust OOD提供的densenet 自训练 cifar10
    normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    model = DenseNet3(100, 10, normalizer=normalizer).to(device)
    model_path = '../model/rood_densenet_cifar10_ep100.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # Test set: Average loss: 0.0012, Accuracy: 9448/10000 (94%)
    affine_params = np.load(RESULT_DIR + '/' + args.affine)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    data, _ = next(iter(test_loader))
    temperature = -1

    tag = 'densenet_in'
    test_origin_and_affines(affine_params, model, data, tag, temperature)

    testsetout = torchvision.datasets.ImageFolder("../data/Imagenet", transform=transform)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=100, shuffle=False, num_workers=2)
    data, _ = next(iter(test_loader))
    tag = 'densenet_imagenet'
    test_origin_and_affines(affine_params, model, data, tag, temperature)

    data = torch.from_numpy(np.random.randn(100, 3, 32, 32)).float()
    tag = 'densenet_gaussian'
    test_origin_and_affines(affine_params, model, data, tag, temperature)

    data = torch.from_numpy(np.random.uniform(size=(100, 3, 32, 32))).float()
    tag = 'densenet_uniform'
    test_origin_and_affines(affine_params, model, data, tag, temperature)


if __name__ == '__main__':
    main()
