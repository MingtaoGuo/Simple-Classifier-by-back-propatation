import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def update(W, b, alpha, dw, db):
    W = W - alpha * dw
    b = b - alpha * db
    return W, b

def linear(x, W, b):
    #x: batch_size x 1024, W: 1024 x class_num
    score = np.matmul(x, W) + b
    h = np.size(W, 0)
    w = np.size(W, 1)
    dw = np.zeros([h, w])
    batch_size = np.size(x, 0)
    sum_dw = 0
    for b_i in range(batch_size):
        v = x[b_i, :]
        for c_i in range(w):
            dw[:, c_i] = v
        sum_dw += dw
    dw = sum_dw
    db = np.ones([1, w]) * batch_size
    return score, dw, db

def hinge_loss(score, label):
    #score: batch_size x class_num, label: batch_size x 1
    batch_size = np.size(score, 0)
    s = np.zeros([batch_size])
    for b_i in range(batch_size):
        s[b_i] = score[b_i, label[b_i]]
    s = np.reshape(s, [-1, 1])
    temp = np.maximum(score - np.ones_like(score) * s + 1, np.zeros_like(score))
    loss = np.sum(temp, axis=1)
    temp[np.where(temp >= 1)] = 1
    for b_i in range(batch_size):
        temp[b_i, label[b_i]] = -1
    temp[np.where(temp < 1) and np.where(temp > 0)] = 1
    temp[np.where(temp == 0)] = 0
    ds = temp
    return loss, ds

def total_loss(hinge_l):
    batch_size = np.size(hinge_l, 0)
    t_loss = np.mean(hinge_l)
    dl = 1 / batch_size
    return t_loss, dl

def back_propagation(batch_data, label, W, b, alpha=1e-3):
    score, dw, db = linear(batch_data, W, b)
    hinge_l, ds = hinge_loss(score, label)
    t_loss, dl = total_loss(hinge_l)
    ds_sum = 0
    db_sum = 0
    for j in range(np.size(ds, 0)):
        ds_sum += ds[j, :] * dw
        db_sum += ds[j, :] * db
    W, b = update(W, b, alpha, dl * ds_sum, dl * db_sum)
    return W, b, t_loss

def validation(data, label, W, b):
    score = np.matmul(data, W) + b
    acc = np.mean(np.int32(np.argmax(score, axis=1) == label))
    return acc

def train(data, labels, class_num, iter_num, alpha):
    dim = np.size(data, 1)
    W = np.random.normal(0, 0.02, [dim, class_num])
    b = np.zeros([1, class_num])
    data_num = np.size(data, 0)
    plt_acc = []
    plt_loss = []
    for i in range(iter_num):
        random_sample = np.random.randint(0, data_num, [50])
        batch = data[random_sample, :]
        label = labels[random_sample]
        W, b, t_loss = back_propagation(batch, label, W, b, alpha)
        acc = validation(batch, label, W, b)
        print("iteration: %d, loss: %f, acc: %f" % (i, t_loss, acc))
        plt_acc.append(acc)
        plt_loss.append(t_loss)
    plt.plot(np.arange(0, iter_num), plt_loss)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.figure()
    plt.plot(np.arange(0, iter_num), plt_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    return W, b

if __name__ == "__main__":
    data = sio.loadmat("./cxlzk_pca.mat")
    W, b = train(np.reshape(data["traindata"], [25000, -1]), np.argmax(data["trainlabel"], axis=1), 5, 10000, alpha=1e-3)
    test_acc = validation(data["testdata"], np.argmax(data["testlabel"], axis=1), W, b)
    print("Test Accuracy: %f"%(test_acc))
    pass

