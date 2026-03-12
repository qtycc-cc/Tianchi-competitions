# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/yandex-research/tabm/blob/main/example.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>
# 
# # TabM + WeightedSAM

# %%
# !pip install rtdl_num_embeddings
# !pip install tabm

# %%
import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import rtdl_num_embeddings
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor

# %% [markdown]
# # WeightedSAM 优化器（修复 BatchNorm 处理）

# %%
def disable_running_stats(model):
    """禁用 BatchNorm 的运行统计（支持 1d 和 2d）"""
    def _disable(module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.backup_momentum = module.momentum
            module.momentum = 0
            # 同时设置为 eval 模式以禁用统计更新
            # module.eval()

    model.apply(_disable)


def enable_running_stats(model):
    """启用 BatchNorm 的运行统计"""
    def _enable(module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
            # 恢复为 train 模式
            # module.train()

    model.apply(_enable)


class WeightedSAM(torch.optim.Optimizer):
    r"""WeightedSAM for TabM"""
    
    def __init__(
        self,
        model,
        base_optimizer,
        rho=0.05,
        gamma=0.9,
        sam_eps=1e-12,
        adaptive=False,
        decouple=True,
        max_norm=None,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.model = model
        self.base_optimizer = base_optimizer
        self.decouple = decouple
        self.max_norm = max_norm
        alpha = gamma / (1 - gamma)
        defaults = dict(rho=rho, alpha=alpha, sam_eps=sam_eps, adaptive=adaptive, **kwargs)
        defaults.update(self.base_optimizer.defaults)
        super(WeightedSAM, self).__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["sam_eps"])
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 计算扰动方向
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w, alpha=1.0)  # w -> w + e(w)
                self.state[p]["e_w"] = e_w
        
        # 梯度裁剪
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        
        # 保存第一次的梯度
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["grad"] = p.grad.detach().clone()
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # 恢复参数到原始位置
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(self.state[p]["e_w"], alpha=-1.0)  # w + e(w) -> w
        
        # 梯度裁剪
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        
        # 处理梯度：解耦或耦合模式
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self.decouple:
                    # 耦合模式：混合两次梯度
                    p.grad.mul_(group["alpha"]).add_(self.state[p]["grad"], alpha=1.0 - group["alpha"])
                else:
                    # 解耦模式：分离锐度梯度
                    self.state[p]["sharpness"] = p.grad.detach().clone() - self.state[p]["grad"]
                    p.grad.mul_(0.0).add_(self.state[p]["grad"], alpha=1.0)  # 只用第一次梯度
        
        # 基础优化器更新（使用第一次梯度）
        self.base_optimizer.step()
        
        # 解耦模式：额外用锐度梯度更新
        if self.decouple:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.add_(self.state[p]["sharpness"], alpha=-group["lr"] * group["alpha"])
        
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        
        # 第一次前向-后向（启用 BN 统计）
        enable_running_stats(self.model)
        loss = closure()
        self.first_step(zero_grad=True)
        
        # 第二次前向-后向（禁用 BN 统计，避免污染）
        disable_running_stats(self.model)
        closure()
        self.second_step()
        
        # 恢复 BN 状态
        enable_running_stats(self.model)
        
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    # def zero_grad(self):
    #     self.base_optimizer.zero_grad()


# %% [markdown]
# # 数据集准备（与原版相同）

# %%
seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
pass

# %%
TaskType = Literal['regression', 'binclass', 'multiclass']

# Regression.
task_type: TaskType = 'regression'
n_classes = None
dataset = sklearn.datasets.fetch_california_housing()
X_num: np.ndarray = dataset['data']
Y: np.ndarray = dataset['target']

task_is_regression = task_type == 'regression'

# Numerical features.
X_num: np.ndarray = X_num.astype(np.float32)
n_num_features = X_num.shape[1]

# Categorical features.
cat_cardinalities = []
X_cat = None

# Labels.
if task_type == 'regression':
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)

# Split.
all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8
)
data_numpy = {
    'train': {'x_num': X_num[train_idx], 'y': Y[train_idx]},
    'val': {'x_num': X_num[val_idx], 'y': Y[val_idx]},
    'test': {'x_num': X_num[test_idx], 'y': Y[test_idx]},
}

for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(f'{part:<5}    {key:<5}    {value.shape!r:<10}    {value.dtype}')

# %% [markdown]
# # 数据预处理

# %%
# Preprocessing.
x_num_train_numpy = data_numpy['train']['x_num']
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, x_num_train_numpy.shape)
    .astype(x_num_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train_numpy + noise)

for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])

# Label preprocessing.
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float

Y_train = data_numpy['train']['y'].copy()
if task_type == 'regression':
    regression_label_stats = RegressionLabelStats(
        Y_train.mean().item(), Y_train.std().item()
    )
    Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
else:
    regression_label_stats = None

# %% [markdown]
# # PyTorch 设置

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(Y_train, device=device)

# 注意：SAM 与 AMP 的兼容性较差，建议禁用 AMP
amp_enabled = False
amp_dtype = None
grad_scaler = None

compile_model = False

print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}')
print(f'torch.compile: {compile_model}')

# %% [markdown]
# # 模型和优化器（使用 WeightedSAM）

# %%
# 使用 Piecewise-linear embeddings（效果最好）
num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=48),
    d_embedding=16,
    activation=False,
    version='B',
)

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=1 if n_classes is None else n_classes,
    num_embeddings=num_embeddings,
).to(device)

# 基础优化器
base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=2e-3)

# 包装为 WeightedSAM
optimizer = WeightedSAM(
    model=model,
    base_optimizer=base_optimizer,
    rho=0.05,       # 扰动半径，可根据需要调整
    gamma=0.9,      # 锐度权重
    adaptive=True,
    decouple=True,  # 解耦模式通常更稳定
    max_norm=1.0,   # 梯度裁剪
)

# 注意：SAM 内部会处理梯度裁剪，这里设为 None 避免重复
gradient_clipping_norm: Optional[float] = None

if compile_model:
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

# %% [markdown]
# # 训练（适配 SAM 的 closure 模式）

# %%
share_training_batches = True

@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(
            data[part]['x_num'][idx],
            data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
        )
        .squeeze(-1)
        .float()
    )


base_loss_fn = (
    nn.functional.mse_loss if task_is_regression else nn.functional.cross_entropy
)


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    if share_training_batches:
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        y_true = y_true.flatten(0, 1)
    return base_loss_fn(y_pred, y_true)


@evaluation_mode()
def evaluate(part: str) -> float:
    model.eval()
    eval_batch_size = 8096
    y_pred: np.ndarray = (
        torch.cat(
            [
                apply_model(part, idx)
                for idx in torch.arange(len(data[part]['y']), device=device).split(
                    eval_batch_size
                )
            ]
        )
        .cpu()
        .numpy()
    )
    if task_type == 'regression':
        assert regression_label_stats is not None
        y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

    if not task_is_regression:
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)

    y_true = data[part]['y'].cpu().numpy()
    score = (
        -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        if task_type == 'regression'
        else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    )
    return float(score)


print(f'Test score before training: {evaluate("test"):.4f}')

# %% [markdown]
# # 训练循环（关键修改：适配 SAM）

# %%
n_epochs = 1000  # 减少 epoch 数，SAM 收敛通常更快
train_size = len(train_idx)
batch_size = 256
epoch_size = math.ceil(train_size / batch_size)

epoch = -1
metrics = {'val': -math.inf, 'test': -math.inf}


def make_checkpoint() -> dict[str, Any]:
    return deepcopy({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    })


best_checkpoint = make_checkpoint()
patience = 16
remaining_patience = patience

for epoch in range(n_epochs):
    batches = (
        torch.randperm(train_size, device=device).split(batch_size)
        if share_training_batches
        else (
            torch.rand((train_size, model.backbone.k), device=device)
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )
    )
    
    for batch_idx in batches:
        model.train()
        
        # SAM 需要 closure
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            loss.backward()
            return loss
        
        # 使用 SAM 的 step（自动执行两次前向-后向）
        optimizer.step(closure)

    # 评估
    metrics = {part: evaluate(part) for part in ['val', 'test']}
    val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

    print(
        f'{"*" if val_score_improved else " "}'
        f' [epoch] {epoch:<3}'
        f' [val] {metrics["val"]:.3f}'
        f' [test] {metrics["test"]:.3f}'
    )

    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
    else:
        remaining_patience -= 1

    if remaining_patience < 0:
        break

# 加载最佳模型
model.load_state_dict(best_checkpoint['model'])

print('\n[Summary]')
print(f'best epoch:  {best_checkpoint["epoch"]}')
print(f'val score:  {best_checkpoint["metrics"]["val"]}')
print(f'test score: {best_checkpoint["metrics"]["test"]}')