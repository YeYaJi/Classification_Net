# input = (N, C, H, W)

def conv2d(H, W, stride, kernel_size, dilation, padding):
    H_out = int((H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = int((W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    # print("H_out", H_out, "\n", "W_out", W_out)
    return H_out, W_out


def maxpool2d(H, W, kernel_size, stride, dilation, padding):
    H_out = int((H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = int((W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return H_out, W_out


if __name__ == "__main__":
    H, W = conv2d(H=224, W=224,  kernel_size=(7, 7), stride=(2, 2),dilation=(1, 1), padding=(3, 3))
    print(H, W)
    H, W = maxpool2d(H, W, kernel_size=(3, 3),  stride=(2, 2),dilation=(1, 1), padding=(1, 1))
    print(H, W)
    H, W = conv2d(H, W,  kernel_size=(3, 3), stride=(1, 1),dilation=(1, 1), padding=(0, 0))
    print(H, W)
    H, W =maxpool2d(H, W, kernel_size=(3, 3), stride=(2, 2),dilation=(1, 1), padding=(2, 2))
    print(H, W)
    H, W = maxpool2d(H, W, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), padding=(1,1))
    print("c",H, W)
    H, W = maxpool2d(H, W, kernel_size=(5, 5), stride=(3, 3), dilation=(1, 1), padding=(1, 1))
    print("c", H, W)
    H, W = conv2d(H, W, stride=(1, 1), kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))
    print(H, W)
    H, W = conv2d(H, W, stride=(1, 1), kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))
    print(H, W)
    H, W = conv2d(H, W, stride=(1, 1), kernel_size=(3, 3), dilation=(1, 1), padding=(1, 1))
    print(H, W)
    H, W = maxpool2d(H, W, stride=(2, 2), kernel_size=(3, 3), dilation=(1, 1), padding=(0, 0))
    print(H, W)