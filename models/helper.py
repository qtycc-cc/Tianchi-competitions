import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple


def grouped_value(
        value: torch.Tensor,
        gidx: torch.Tensor,
        sgidx: List[torch.Tensor],
        equalsize: bool
) -> torch.Tensor:

    if equalsize:
        num_groups = gidx.max().item() + 1
        g_value = torch.zeros(num_groups, dtype=value.dtype, device=value.device)
        g_value.scatter_add_(0, gidx, value)
    else:
        num_groups = len(sgidx)
        g_value = torch.zeros(num_groups, dtype=value.dtype, device=value.device)
        for i, idx in enumerate(sgidx):
            g_value[i] = value[idx].sum()
    return g_value


def get_Rx(
        x: torch.Tensor,
        w: torch.Tensor,
        num_groups: int,
        p: int,
        x_norm: int,
        gidx: torch.Tensor,
        sgidx: List[torch.Tensor],
        equalsize: bool
) -> torch.Tensor:
    """
    <w, ||x_g||p, q>
    :param x:
    :param w:
    :param num_groups:
    :param p: number of feature
    :param x_norm: alias to p from ||.||p,q
    :param gidx:
    :param sgidx:
    :param equalsize:
    :return: (tensor with a single value)
    """
    if p != num_groups:
        if x_norm == 1:
            Rx_value = (w * x).abs().sum()
        elif x_norm == 2:
            tmp = w * x
            x_square = tmp ** 2
            grouped_x_square = grouped_value(x_square, gidx, sgidx, equalsize)
            Rx_value = torch.sqrt(grouped_x_square).sum()
        else:
            raise ValueError("Illegal forms of norm")
    else:
        Rx_value = (w * x).abs().sum()
    return Rx_value

def my_logistic(
        x: torch.Tensor,
        A: torch.Tensor,
        y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    逻辑回归
    :param x: 权重
    :param A: 特征矩阵
    :param y: label向量，确保shape为n
    :return: 损失函数，梯度
    """
    u = (A.T @ x)
    f = torch.mean(torch.log(1 + torch.exp(-torch.abs(u))) +
                   torch.clamp(u, min=0) - y * u)
    df = A @ (torch.sigmoid(u) - y) / A.shape[1]
    return f, df

def draw_loss_history(loss_history, lambda_param):
    title = f"Loss vs Iterations with lambda = {lambda_param} and loss = {loss_history[-1]}"
    plt.figure(figsize=(10, 6))

    iterations = list(range(len(loss_history)))

    plt.plot(iterations, loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.ticklabel_format(axis='y', useOffset=False)

    plt.tight_layout()
    plt.show()

    print(f"Final loss: {loss_history[-1]}")
    print(f"Total iterations: {len(loss_history) - 1}")


def draw_loss_history_agg(loss_history, lambda_param, save_path=None):
    """
    绘制损失历史
    loss_history: 损失历史列表
    lambda_param: lambda参数
    save_path: 保存路径，如果为None则自动生成
    show: 是否显示图像（需要合适的后端）
    """
    title = f"Loss vs Iterations with lambda = {lambda_param} and loss = {loss_history[-1]}"
    plt.figure(figsize=(10, 6))

    iterations = list(range(len(loss_history)))

    plt.plot(iterations, loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.ticklabel_format(axis='y', useOffset=False)

    plt.tight_layout()

    # 保存图像
    if save_path is None:
        save_path = f'loss_history_lambda_{lambda_param}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {save_path}")

    print(f"Final loss: {loss_history[-1]}")
    print(f"Total iterations: {len(loss_history) - 1}")