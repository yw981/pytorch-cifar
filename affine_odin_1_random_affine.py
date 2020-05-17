from __future__ import print_function
import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import os
from func import arr_stat, RESULT_DIR
from models import DenseNet3

if __name__ == '__main__':
    CUDA_DEVICE = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device = 'cuda'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    data, labels = next(iter(test_loader))
    labels = labels.numpy()
    np.random.seed(1234)
    # use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    epochs = 2

    # 模型！
    # Robust OOD提供的densenet 自训练 cifar10
    normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
    net = DenseNet3(100, 10, normalizer=normalizer).to(device)
    model_path = '../model/rood_densenet_cifar10_ep100.pth'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    # Test set: Average loss: 0.0012, Accuracy: 9448/10000 (94%)

    v_data = Variable(data.cuda(CUDA_DEVICE), requires_grad=True)

    # in
    output = net.forward(v_data)
    output = F.softmax(output, dim=1)
    lg = output.data.cpu().numpy()
    amr = np.argmax(lg, axis=1)
    aml = labels
    wrong_indices = (amr != aml)
    right_indices = ~wrong_indices
    base_acc = (1 - np.sum(wrong_indices + 0) / aml.shape[0])

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    np.save(RESULT_DIR + '/densenet_in.npy', lg)
    print('in saved base acc = ', base_acc)
    # arr_stat('in ',lg)
    # print5('in ',lg)

    # 老版本？因为densenet引用了老版本，必须用回老板
    # device = torch.device("cuda" if use_cuda else "cpu")
    # data = data.to(device)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # model = Net().to(device)
    # model.load_state_dict(torch.load('model/lenet_mnist_model.pth'))
    k = 0
    params = []
    for i in range(9999):
        if i % 100 == 0:
            print('it ' + str(i))

        # result
        # tfparam = np.array([[1.1, 0., 0.], [0., 1.1, 0.]])
        # tfseed = (np.random.rand(2, 3) - 0.5) * np.array([[0.2, 0.2, 6], [0.2, 0.2, 6]])

        # result_min
        # # 缩小变化 初值[[1.1, 0., 0.], [0., 1.1, 0.]] -> [[1.0, 0., 0.], [0., 1.0, 0.]]
        tfparam = np.array([[1.05, 0., 0.], [0., 1.05, 0.]])
        # # 缩小变化 最后 * 0.1
        tfseed = (np.random.rand(2, 3) - 0.5) * np.array([[0.2, 0.2, 6], [0.2, 0.2, 6]]) * 0.4

        # result_large
        # tfparam = np.array([[1.2, 0., 0.], [0., 1.2, 0.]])
        # tfseed = (np.random.rand(2, 3) - 0.5) * np.array([[0.2, 0.2, 6], [0.2, 0.2, 6]]) * 1.3

        tfparam += tfseed

        affine_param = torch.from_numpy(tfparam).float()
        # print(affine_param.size())
        p2 = data.size()
        # print(p2)
        # print(p2[0])
        p1 = affine_param.repeat(data.size()[0], 1, 1)

        # print(p1)

        grid = F.affine_grid(p1, p2)
        trans_data = F.grid_sample(data, grid)
        v_trans_data = Variable(trans_data.data.cpu().cuda(CUDA_DEVICE), requires_grad=True)
        output = net.forward(v_trans_data)
        output = F.softmax(output, dim=1)

        lg = output.data.cpu().numpy()

        # print(lg.shape)
        # print(mnist.test.labels.shape)
        amr = np.argmax(lg, axis=1)
        # print(amr)
        # aml = np.reshape(mnist.test.labels, (10000,1))
        aml = labels
        # print(aml)
        wrong_indices = (amr != aml)
        # print(wrong_indices)
        right_indices = ~wrong_indices
        acc = (1 - np.sum(wrong_indices + 0) / aml.shape[0])
        # print("acc = %f" % acc)
        if acc > base_acc * 0.95:
            print("acc = %f" % acc)
            print('!!!!!!!! save #%d' % i)
            print(tfparam)
            params.append(tfparam)
            # np.save('result/exp_affine_in_%d.npy' % k, lg_softmax)
            print(np.linalg.norm(tfparam))
            k += 1
            if k >= 10:
                break

    params = np.array(params)
    print(params.shape)
    np.save(RESULT_DIR + '/affine_params_random.npy', params)
