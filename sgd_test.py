"""手写一个两层 MLP(784-48-10), 全程不用 autograd, 纯手推反向传播 + SGD。

数据: 把 MNIST 的 idx 文件解压到 ./mnist_data/ 下即可, 本脚本只读不下载。
"""

import copy
import struct

import numpy as np
import torch

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081   # 官方全量训练集的像素均值/标准差


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_mnist(data_dir: str = "mnist_data", n_max: int = 20000):
    """读本地 idx 文件, 返回 (x, y): x 是 float32 [n, 784] 已标准化, y 是 int64 [n]。"""
    try:
        img = open(f"{data_dir}/train-images-idx3-ubyte", "rb").read()
        lbl = open(f"{data_dir}/train-labels-idx1-ubyte", "rb").read()
    except FileNotFoundError:
        raise SystemExit(f"{data_dir}/ 下缺少 idx 文件, 请先下载 MNIST 并解压到该目录。")

    n, rows, cols = struct.unpack(">III", img[4:16])
    x = np.frombuffer(img, np.uint8, offset=16).reshape(n, rows * cols)[:n_max]
    y = np.frombuffer(lbl, np.uint8, offset=8)[:n_max]
    # 像素非负且集中在 0 附近, 不标准化的话 z1 标准差只有 ~0.03, sigmoid 全是 0.5 学不动
    x = (x.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


# --------------------------------------------------------------------------- #
# 网络
# --------------------------------------------------------------------------- #
class MyNN:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # randn 的标准差是 1, 必须乘上 weight_init_std, 否则初值太大 sigmoid 会饱和
        self.W1 = torch.randn(input_size, hidden_size) * weight_init_std
        self.B1 = torch.randn(hidden_size) * weight_init_std
        self.W2 = torch.randn(hidden_size, output_size) * weight_init_std
        self.B2 = torch.zeros(output_size)
        self._cache = None

    def params(self):
        """让 params() 返回真实的张量对象, 这样 p -= lr * g 才是原地更新权重。"""
        return {"W1": self.W1, "B1": self.B1, "W2": self.W2, "B2": self.B2}

    def to(self, dtype) -> "MyNN":
        for p in self.params().values():
            p.data = p.data.to(dtype)
        self._cache = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = x @ self.W1 + self.B1          # [N, H]
        a1 = torch.sigmoid(z1)              # [N, H]
        z2 = a1 @ self.W2 + self.B2         # [N, O]
        a2 = torch.softmax(z2, dim=1)       # [N, O] 概率
        self._cache = (x, a1, a2, z2)
        return a2

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        """交叉熵。用 logsumexp(z2) 而不是 log(softmax(z2)), 数值上稳很多。"""
        _, _, _, z2 = self._cache
        idx = torch.arange(z2.shape[0])
        log_z = torch.logsumexp(z2, dim=1, keepdim=True)
        return (log_z.squeeze(1) - z2[idx, y]).mean()

    def backward(self, y: torch.Tensor):
        """链式法则从输出层往回推到输入层, 返回每个参数的梯度。"""
        x, a1, a2, _ = self._cache
        idx = torch.arange(x.shape[0])

        # dL/dz2: softmax + 交叉熵的经典结论 —— (预测概率 - onehot) / N
        d2 = a2.clone()
        d2[idx, y] -= 1.0
        d2 /= x.shape[0]

        gW2 = a1.T @ d2                    # [H, O] = [H, N] @ [N, O]
        gb2 = d2.sum(dim=0)                # [O]     偏置要对 batch 求和

        d1 = d2 @ self.W2.T                # [N, H]
        d1 = d1 * a1 * (1 - a1)            # 穿过 sigmoid: da1/dz1 = a1(1-a1)

        gW1 = x.T @ d1                     # [I, H] = [I, N] @ [N, H]
        gb1 = d1.sum(dim=0)                # [H]
        return {"W1": gW1, "B1": gb1, "W2": gW2, "B2": gb2}

    def sgd_step(self, grads, lr: float) -> None:
        with torch.no_grad():
            for name, p in self.params().items():
                p -= lr * grads[name]


# --------------------------------------------------------------------------- #
# 梯度校验: 数值微分 vs 解析梯度
# --------------------------------------------------------------------------- #
def grad_check(net, x, y, eps: float = 1e-5, n_probe: int = 20, seed: int = 0) -> float:
    """扰动每个参数一点点看 loss 变化, 验证 backward 的公式没写错。返回最大相对误差。

    必须在 float64 下做: float32 中 loss 本身的分辨率(~2.4e-7)远大于单个参数的
    二阶差分, lp 和 lm 会被舍成同一个数, 数值梯度直接变成 0。
    """
    net = copy.deepcopy(net).to(torch.float64)
    x = x.double()

    net.forward(x)
    grads = net.backward(y)
    g = torch.Generator().manual_seed(seed)
    worst = 0.0

    for name, p in net.params().items():
        flat = p.view(-1)                    # view 共享存储, 原地改 flat 就是改 p
        probe = torch.randperm(flat.numel(), generator=g)[:n_probe]
        for i in probe.tolist():
            old = flat[i].item()
            # 每次扰动后都要重新前向, 否则 net.loss 读的还是 _cache 里的旧值
            flat[i] = old + eps
            net.forward(x)
            lp = net.loss(y).item()
            flat[i] = old - eps
            net.forward(x)
            lm = net.loss(y).item()
            flat[i] = old
            numeric = (lp - lm) / (2 * eps)
            analytic = grads[name].view(-1)[i].item()
            worst = max(worst, abs(numeric - analytic) / max(1e-8, abs(numeric) + abs(analytic)))
    return worst


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def evaluate(net: MyNN, x: torch.Tensor, y: torch.Tensor) -> float:
    """准确率: 前向取每行概率最大的类别, 和 y 比较后求均值。x 是 [N, 784], y 是 [N]。"""
    return (net.forward(x).argmax(dim=1) == y).float().mean().item()


def train(net, x, y, epochs=5, batch_size=64, lr=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)   # 每个 epoch 打乱顺序
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            net.forward(x[idx])
            net.sgd_step(net.backward(y[idx]), lr)
        print(f"epoch {ep}  train_acc {evaluate(net, x, y):.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    x, y = load_mnist()

    print(f"x shape: {x.shape}") # torch.Size([20000, 784])
    print(f"y shape: {y.shape}") # torch.Size([20000])

    x_val, y_val = x[:1000], y[:1000]
    x_train, y_train = x[1000:], y[1000:]

    net = MyNN(784, 48, 10)

    # 1) 先验梯度对不对
    err = grad_check(net, x_train[:64], y_train[:64])
    print(f"梯度校验最大相对误差: {err:.2e}  ->  {'通过' if err < 1e-4 else '不通过, backward 写错了'}")

    # 2) 再看训练前的初始 loss, 健康值大约是 ln(10) ≈ 2.30
    net.forward(x_train)
    print(f"初始 loss {net.loss(y_train).item():.4f}  初始 acc {evaluate(net, x_train, y_train):.4f}")

    # 3) 训练
    train(net, x_train, y_train, epochs=10, batch_size=64, lr=0.1)

    # 4) 验证集
    net.forward(x_val)
    print(f"最终 loss {net.loss(y_val).item():.4f}  验证 acc {evaluate(net, x_val, y_val):.4f}")