import numpy as np
from func import arr_stat, RESULT_DIR


def tpr95(in_data, out_data, is_diff=False):
    return 0
    # calculate the falsepositive error when tpr is 95%
    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))
    gap = (end - start) / 10000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr / total

    return fprBase


def auroc(in_data, out_data, is_diff=False):
    # calculate the AUROC

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 10000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr
    return aurocBase


def auprIn(in_data, out_data, is_diff=False):
    # calculate the AUPR

    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 10000
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def detection(in_data, out_data, is_diff=False):
    # calculate the minimum detection error
    start = np.min(np.array([in_data, out_data]))
    end = np.max(np.array([in_data, out_data]))

    gap = (end - start) / 10000
    # print(out_data.shape)
    # arr_stat('out data ', out_data)
    # 原著只有最高分进去比了，此处已修改
    Y1 = out_data if is_diff else np.max(out_data, axis=1)
    X1 = in_data if is_diff else np.max(in_data, axis=1)
    errorBase = 1.0

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    return errorBase


def calculate_softmax_score_diff(tag, softmax_result, origin):
    softmax_result = softmax_result.squeeze()

    # 先算origin的，原始未经变换的结果
    origin_max_scores = np.max(origin, axis=1)
    origin_max_indices = np.argmax(origin, axis=1)
    # 对应的得分
    cor_scores = softmax_result[tuple(np.arange(softmax_result.shape[0])), tuple(origin_max_indices)]

    # 原计划，按逻辑diff_score = origin_max_scores - cor_scores，分数越低（可为负的）越好（越in），为计算AUROC取反了
    # 统一取绝对值的相反数，最大0最好（越in），越小越out
    diff_score = -np.abs(cor_scores - origin_max_scores)

    return diff_score


def calculate_out_diff_scores(key, origin_out_result):
    out_diff_scores = []

    for i in range(10):
        result = np.load(RESULT_DIR + '/densenet_%s_%d.npy' % (key, i))
        out_diff_score = calculate_softmax_score_diff('%s %d' % (key, i), result, origin_out_result)
        out_diff_scores.append(out_diff_score)

    return np.array(out_diff_scores)


def evaluate_score(tag, in_data, out_data, is_diff=False):
    # print(tag, ' fpr at tpr95 ', tpr95(in_data, out_data))
    # print(tag, ' error ', detection(in_data, out_data))
    # print(tag, ' AUROC ', auroc(in_data, out_data))
    # print(tag, ' AUPR in ', auprIn(in_data, out_data))
    # str = "{} error : {:8.2f}% FPR at TPR95 : {:8.2f}% AUROC : {:>8.2f}% AUPR in : {:>8.2f}% "
    str = "{} , {}, {}, {}, {} "
    print(np.max(in_data),np.min(in_data))
    print(np.max(out_data),np.min(out_data))
    print(str.format(tag,
                     detection(in_data,
                               out_data,
                               is_diff) * 100,
                     tpr95(in_data,
                           out_data,
                           is_diff) * 100,
                     auroc(in_data,
                           out_data,
                           is_diff) * 100,
                     auprIn(in_data,
                            out_data,
                            is_diff) * 100))


# 测试
if __name__ == "__main__":
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    # ])
    # testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    # _, labels = next(iter(test_loader))
    # labels = labels.numpy()

    origin_in_result = np.load(RESULT_DIR + '/densenet_in.npy')
    # calculate_softmax_score_diff('in', origin_in_result, origin_in_result)
    in_diff_scores = []
    for i in range(10):
        result = np.load(RESULT_DIR + '/densenet_in_%d.npy' % i)
        in_diff_score = calculate_softmax_score_diff('cifar10 %d' % i, result, origin_in_result)
        in_diff_scores.append(in_diff_score)

    in_diff_scores = np.array(in_diff_scores)

    for key in ['imagenet', 'gaussian', 'uniform']:
        origin_out_result = np.load(RESULT_DIR + '/densenet_%s.npy' % key)

        # arr_stat('in', origin_in_result)
        # arr_stat('out', origin_out_result)

        evaluate_score('Baseline ', origin_in_result, origin_out_result)

        out_diff_scores = calculate_out_diff_scores(key, origin_out_result)

        for i in range(10):
            # print('----------')
            evaluate_score('%s %d' % (key, i), in_diff_scores[i], out_diff_scores[i], True)
            # arr_stat('in', in_diff_scores[i])
            # arr_stat('out', out_diff_scores[i])
