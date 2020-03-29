import numpy as np

# model_name = 'densenet'
model_name = 'vgg'

# out_name = 'uniform'
out_name = 'gaussian'
# out_name = 'imagenet'

d1 = np.load('result/result_{}_cifar.npy'.format(model_name))
d2 = np.load('result/result_{}_{}.npy'.format(model_name, out_name))

data = np.array(np.vstack((d1, d2)))

l1 = np.zeros([10000, 1])
l2 = np.ones([10000, 1])

label = np.vstack((l1, l2)).astype(np.int64).flatten()

np.save('../data/{}_cifar_{}_feature_data.npy'.format(model_name, out_name), data)
np.save('../data/{}_cifar_{}_feature_label.npy'.format(model_name, out_name), label)

print(model_name,out_name,data.shape)
print(label.shape)
