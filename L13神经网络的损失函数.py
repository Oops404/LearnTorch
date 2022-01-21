import torch
from torch.nn import MSELoss

# 回顾↓：
# 激活函数 sign sigmoid relu tanh 等。
# 多分类表现层 softmax。
#
# 损失函数和机器学习中的类似，用来衡量真实值与预测值之间的差异，例如SSE。
# 通常用L(w)表示。
# 将损失函数L(w)转变为凸函数的数学方法，常见的有拉格朗日变换等。
# 在凸函数上求解L(w)的最小值对应的w方法，也就是以梯度下降为代表的优化算法。


# 模型训练
# 定义基本模型 -> 定义损失函数 -> 定义优化算法 -> 以最小化损失函数为目标，求解权重

# SSE = Σ(i=1~m) (z[i]-zhat[i])^2
# MSE = (1/m) (Σ(i=1~m) (z[i]-zhat[i])^2)

yhat = torch.randn(size=(50,), dtype=torch.float32)
y = torch.randn(size=(50,), dtype=torch.float32)

# 评估指标，reduction默认为mean，  mean 为MSE。 sum 输出sse
criterion = MSELoss()  # reduction = 'mean' / 'sum'
loss = criterion(yhat, y)
print(loss)
