import torch


# softmax = e^(z[k]) / Σ(0~k){e^z}

# 案例 有 苹果、梨、百香果，样本i被分类为百香果的概率 σ(百香果)=：
# σ(百香果) = e^(z[百香果]) / e^(z[苹果]) + e^(z[梨]) + e^(z[百香果])

# 可以写为
def softmax(z1):
    # 存在一个广播，细品
    return torch.exp(z1) / torch.sum(torch.exp(z1))


# e^1000 指数级别，必然直接崩
Z = torch.tensor([1010, 300, 990], dtype=torch.float32)
print(softmax(Z))

# 因此常用调整后的torch.softmax(z,0)，原理以后说明
print(torch.softmax(Z, 0))

Z = torch.tensor([10, 9, 5], dtype=torch.float32)
print(softmax(Z))
print(torch.softmax(Z, 0))
