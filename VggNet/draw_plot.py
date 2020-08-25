import matplotlib.pylab as plt
import numpy as np

train_epoch_correct_rate = []
test_epoch_correct_rate = []
train_epoch_loss_ave = []
test_epoch_loss_ave = []
epoch = 5
# correct_rate的图
with open('./train_epoch_correct_rate', "r") as f:
    while len(train_epoch_correct_rate) < epoch:
        data = float(f.readline())
        train_epoch_correct_rate.append(data)

with open('./test_epoch_correct_rate', "r") as f:
    while len(test_epoch_correct_rate) < epoch:
        data = float(f.readline())
        test_epoch_correct_rate.append(data)

plt.title("train+test_epoch_correct_rate")
plt.plot(np.arange(1, epoch + 1), train_epoch_correct_rate, label="train_epoch_correct_rate")
plt.plot(np.arange(1, epoch + 1), test_epoch_correct_rate, label="test_epoch_correct_rate")
plt.legend()  # 显示图例
plt.show()
# loss的图
with open('./train_epoch_loss_ave', "r") as f:
    while len(train_epoch_loss_ave) < epoch:
        data = float(f.readline())
        train_epoch_loss_ave.append(data)

with open('./test_epoch_loss_ave', "r") as f:
    while len(test_epoch_loss_ave) < epoch:
        data = float(f.readline())
        test_epoch_loss_ave.append(data)

plt.title("train+test_epoch_loss_ave")
plt.plot(np.arange(1, epoch + 1), train_epoch_loss_ave, label="train_epoch_loss_ave")
plt.plot(np.arange(1, epoch + 1), test_epoch_loss_ave, label="test_epoch_loss_ave")
plt.legend()  # 显示图例
plt.show()
