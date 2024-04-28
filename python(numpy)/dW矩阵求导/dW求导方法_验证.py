import torch

#将Calculate_differentiation中numpy版本的函数用pytorch复现一遍，并且用pytorch自带的自动微分验证
class dW_differentiation_torch:
    def __init__(self):
        pass

    def zhuanzhi_hl_sum(self,tensor1,tensor2,hl):
        if hl=='yh':
            sums = torch.sum(tensor1, dim=0).unsqueeze(0).T
            new_matrix = sums.expand_as(tensor2)

        if hl=='zl':
            sums = torch.sum(tensor2, dim=1)
            new_matrix = sums.expand_as(tensor1)

        return new_matrix

    def divide_by_2_until_odd(self,tensor):
        result = tensor.clone()  # 复制输入张量，避免修改原始张量
        while torch.all(result % 2 == 0):  # 循环直到整个张量中的任一元素都无法再被2整除
            result = torch.div(result, 2)  # 将整个张量中的所有元素都除以2
            if torch.any(result == 0):
                break
        return result

def test_one():
    dtype = torch.float64
    # 创建张量，并设置requires_grad=True
    w1 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=dtype).view(2, 3).requires_grad_(True)
    b1 = torch.tensor([1, 2, 3, 4], dtype=dtype).view(2, 2).requires_grad_(True)
    w2 = torch.tensor([1, 3, 4, 9], dtype=dtype).view(2, 2).requires_grad_(True)
    b2 = torch.tensor([4, 3, 6, 7], dtype=dtype).view(2, 2).requires_grad_(True)

    def ReLU(Z):
        return torch.relu(Z)

    def ReLU_deriv(Z):
        return torch.where(Z > 0, torch.tensor(1.0), torch.tensor(0.0))

    A0 = torch.tensor([1, 1, 1, 2, 1, 3], dtype=dtype).view(3, 2).requires_grad_(True)
    z1 = w1 @ A0 + b1
    z1.retain_grad()  # 保留 z1 的梯度
    A1 = ReLU(z1)
    A1.retain_grad()  # 保留 A1 的梯度
    z2 = w2 @ A1 + b2

    # 计算梯度
    z2.backward(torch.ones_like(z2))

    # 输出梯度
    print("w1的梯度:", w1.grad)
    print("b1的梯度:", b1.grad)
    print("w2的梯度:", w2.grad)
    print("b2的梯度:", b2.grad)
    print("z1的梯度:", z1.grad)
    print("A1的梯度:", A1.grad,'\n')

    dd=dW_differentiation_torch()
    A1_grad = dd.zhuanzhi_hl_sum(w2,A1, 'yh')
    z1_grad = A1_grad * ReLU_deriv(z1)
    w1_grad=z1_grad@dd.zhuanzhi_hl_sum(w1,A0,'zl')
    b1_grad=z1_grad
    w2_grad=dd.zhuanzhi_hl_sum(w2,A1,'zl')
    b2_grad=torch.tensor([[1., 1.],[1., 1.]]).to(torch.float64)


    print("w1的梯度:", dd.divide_by_2_until_odd(w1_grad))
    print("b1的梯度:", dd.divide_by_2_until_odd(b1_grad))
    print("w2的梯度:", dd.divide_by_2_until_odd(w2_grad))
    print("b2的梯度:", dd.divide_by_2_until_odd(b2_grad))
    print("z1的梯度:", dd.divide_by_2_until_odd(z1_grad))
    print("A1的梯度:", dd.divide_by_2_until_odd(A1_grad))

def test_multiple_examples(a):#看看功能zhuanzhi_hl_sum(tensor1,tensor2,hl)对否
    a=str(a)
    k=int(a[2])
    j=int(a[1])
    i=int(a[0])
    tensor1 = torch.randn(i, j).requires_grad_(True)
    tensor2 = torch.randn(j, k).requires_grad_(True)

    dd = dW_differentiation_torch()
    yh=dd.zhuanzhi_hl_sum(tensor1,tensor2,'yh')

    z=tensor1@tensor2
    z.backward(torch.ones_like(z))
    t2=tensor2.grad

    zl=dd.zhuanzhi_hl_sum(tensor1,tensor2,'zl')
    t1=tensor1.grad

    # 检查是否近似相等
    are_equal1 = torch.allclose(yh, t2, atol=1e-4)
    are_equal2 = torch.allclose(zl, t1, atol=1e-4)
    equal=are_equal1 and are_equal2
    if equal:
        return True
    else:
        return False

def begin_test_multiple_examples():
    for i in range(111,999):
        i=str(i)
        k=0
        for j in range(len(i)):
            if i[j]!=0:
                k+=1
        if k==3:
            i=int(i)
            a=i
            if test_multiple_examples(a)==False:#有错误才会打印出来
                print(a)
    print(a,'done')#只打印一个done说明没错



if __name__=='__main__':
    print('test_one:')
    test_one()
    print('\n'*5)
    print('test_multiple_examples:')
    begin_test_multiple_examples()

