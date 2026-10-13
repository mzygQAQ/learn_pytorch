# learn_pytorch

手写一个两层 MLP(784-48-10) 识别 MNIST 手写数字，**全程不用 autograd**——前向传播、反向传播、SGD 更新全部自己推、自己写。

`torch.nn` 把反向传播藏起来了。想搞懂梯度究竟怎么从 loss 流回每一个权重，手写一遍是最直接的办法。

## 快速开始

### 1. 准备数据

脚本只读本地文件，不下载。需要 MNIST 的两个训练集 idx 文件解压到 `./mnist_data/`。

**macOS / Linux / WSL / Git Bash：**

```bash
mkdir -p mnist_data
curl -L -o MNIST.tar.gz \
  "https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/mnist/repo?Revision=master&FilePath=raw/MNIST.tar.gz"
tar -xzf MNIST.tar.gz
gunzip MNIST/*.gz
mv MNIST/* mnist_data/
```

**Windows（无 `gunzip` 时）：**

```powershell
python -c "import gzip,tarfile,urllib.request,os; os.makedirs('mnist_data',exist_ok=True); urllib.request.urlretrieve('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/mnist/repo?Revision=master&FilePath=raw/MNIST.tar.gz','MNIST.tar.gz'); t=tarfile.open('MNIST.tar.gz'); [(open('mnist_data/'+n,'wb').write(gzip.GzipFile(fileobj=t.extractfile('MNIST/'+n+'.gz')).read())) for n in ('train-images-idx3-ubyte','train-labels-idx1-ubyte')]; t.close(); os.remove('MNIST.tar.gz')"
```

> 魔搭的 CDN 域名偶发 DNS 解析失败，报 `getaddrinfo failed` 时重跑一次即可。
> 国内访问 PyTorch 官方源 `ossci-datasets.s3.amazonaws.com` 只有 ~10 KB/s，上面的魔搭源实测 7.6 MB/s。
> 海外网络可以直接用 `https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz` 等四个文件。

### 2. 装依赖

```bash
uv sync            # 或 pip install torch numpy
```

### 3. 跑

```bash
python sgd_test.py
```

参考输出：

```
梯度校验最大相对误差: 3.01e-05  ->  通过
初始 loss 2.3024  初始 acc 0.0901
epoch 1  train_acc 0.8642
...
epoch 10 train_acc 0.9545
最终 loss 0.2327  验证 acc 0.9400
```

初始 loss 约等于 `ln(10) = 2.3026`，说明初值正常；最终 94% 准确率对应这个规模的网路（48 个隐藏单元、纯 sigmoid、19000 条训练样本）。

## 实现要点

反向传播的核心就四行，`dL/dz2 = (预测概率 - onehot) / N` 是 softmax + 交叉熵的经典结论：

```python
d2 = a2.clone(); d2[idx, y] -= 1.0; d2 /= x.shape[0]   # [N, O]
gW2 = a1.T @ d2;      gb2 = d2.sum(dim=0)               # [H,N]@[N,O]   偏置对 batch 求和
d1 = d2 @ self.W2.T;  d1 = d1 * a1 * (1 - a1)           # 穿过 sigmoid
gW1 = x.T @ d1;       gb1 = d1.sum(dim=0)               # [I,N]@[N,H]
```

几个容易踩的坑，都在代码里标了注释：

**`weight_init_std` 必须真的用上。** `torch.randn()` 的标准差是 1，不乘初始化标准差的话初值太大，sigmoid 直接饱和、梯度趋近 0。

**梯度校验必须在 float64 下做。** float32 中 loss 在 2.3 附近的分辨率约 `2.4e-7`，而单个参数造成的 loss 二阶差分只有 `1e-8`，`loss(w+eps)` 和 `loss(w-eps)` 会被舍成同一个数，数值梯度恒为 0。

**输入必须标准化。** 只除以 255 不够：像素非负且集中在 0 附近，`z1` 的标准差只有约 0.03，48 个隐藏单元算出的 sigmoid 全是 0.5，区分不开任何输入。按官方均值/标准差 `(0.1307, 0.3081)` 标准化后恢复到 0.28，训练才正常收敛。

**交叉熵用 `logsumexp(z2)` 而不是 `log(softmax(z2))`**，避免数值下溢。

## 为什么训练前先跑梯度校验

`backward` 写错了**不会报错**。维度对得上、矩阵乘法合法，loss 照样单调下降，曲线看起来完全正常——只是收敛到更差的地方。比如漏乘 sigmoid 的导数 `(1-a1)`：

```
grad_check  正确 -> 3.01e-05        错误 -> 1.00e+00     差 4 个数量级
正确  loss: 0.6142  0.3767  0.3089  0.2676  0.2400    acc=0.9342
错误  loss: 1.2241  0.9485  0.8220  0.7415  0.6964    acc=0.7805
```

错误版本的 loss 曲线形状跟正确版本几乎一个模子。光看曲线你会得出"在学了，只是慢"的结论，然后去调 lr、加 epoch——而真正的原因是公式错了。数值微分是唯一能证伪它的手段，必须在权重还随机的时候做才有基准可比。

## 目录

```
sgd_test.py     全部代码：数据加载 + MyNN + 梯度校验 + 训练循环
mnist_data/     MNIST idx 文件（不入库，需自行下载）
```