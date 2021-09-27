import torch

t1 = torch.arange(4)
# 加法中的广播
print(t1 + t1)
print(t1 + 1)

# ---------相同维度，不同形状的张量之间的计算

t2 = torch.tensor([1, 2, 3, 4])
t3 = torch.full([3, 4], fill_value=0, dtype=torch.int16)
# t2 发生了广播，复制了3次和t3形状相同后相加
print(t2 + t3)

# 列广播 相加
# 3行1列全1矩阵
t4 = torch.ones(3, 1)
print(t3 + t4)

# 三维张量的广播

t5 = torch.zeros(3, 4, 5)
t6 = torch.ones(3, 4, 1)
print(t5 + t6)

# 广播过程就是复制那个为1尺寸的对象。


# 不同维度张量相加，先转换成相同维度
# 举例升维
t7 = torch.arange(4).reshape(2, 2)
t7 = t7.reshape(1, 2, 2)
