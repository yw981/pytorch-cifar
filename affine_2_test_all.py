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
    if type(data) is torch.FloatTensor:
        data = Variable(data.cuda(CUDA_DEVICE))

    data = Variable(data.data.cpu().cuda(CUDA_DEVICE))
    output = model.forward(data)
    if temperature > 0:
        output = output / temperature
    if softmax_flag:
        output = F.softmax(output, dim=1)
    np.save(RESULT_DIR + '/%s.npy' % label, output.data.cpu().numpy())
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
    torch.manual_seed(1234)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
