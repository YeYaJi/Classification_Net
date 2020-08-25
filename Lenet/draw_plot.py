import matplotlib.pylab as plt
import numpy as np

epoch_correct_rate = []
epoch_loss_rate = []
epoch = 200
with open('./train_epoch_correct_rate', "r") as f:
    while len(epoch_correct_rate) < epoch:
        data = float(f.readline())
        epoch_correct_rate.append(data)

plt.title("train_epoch_correct_rate")
plt.plot(np.arange(1, epoch + 1), epoch_correct_rate)
plt.show()

with open('./train_epoch_loss_rate', "r") as f:
    while len(epoch_loss_rate) < epoch:
        data = float(f.readline())
        epoch_loss_rate.append(data)

plt.title("train_epoch_loss_rate")
plt.plot(np.arange(1, epoch + 1), epoch_loss_rate)
plt.show()
