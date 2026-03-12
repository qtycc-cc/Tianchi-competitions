# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/yandex-research/tabm/blob/main/example.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>
# 
# # TabM with MomoAdam

# %%
# !pip install rtdl_num_embeddings
# !pip install tabm
# !pip install momo-optim

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

# 导入 MomoAdam
from momo import MomoAdam  # 官方实现

# %%
seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
pass

# %% [markdown]
# # Dataset

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
    assert set(Y.tolist()) == set(range(n_classes))

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
if X_cat is not None:
    data_numpy['train']['x_cat'] = X_cat[train_idx]
    data_numpy['val']['x_cat'] = X_cat[val_idx]
    data_numpy['test']['x_cat'] = X_cat[test_idx]

for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(f'{part:<5}    {key:<5}    {value.shape!r:<10}    {value.dtype}')

# %% [markdown]
# # Data preprocessing

# %%
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
del x_num_train_numpy

for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])

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
# # PyTorch settings

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(Y_train, device=device)
if task_type == 'regression':
    for part in data:
        data[part]['y'] = data[part]['y'].float()
    Y_train = Y_train.float()

amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
amp_enabled = False and amp_dtype is not None
grad_scaler = None#torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None

compile_model = False

print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')
print(f'torch.compile: {compile_model}')

# %% [markdown]
# # Model and optimizer (MomoAdam)

# %%
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

# MomoAdam 优化器
optimizer = MomoAdam(
    model.parameters(), 
    lr=1e-2,           # 比 AdamW 大 5-10 倍
    weight_decay=3e-4,
    use_fstar=True,
)

# MomoAdam 不需要梯度裁剪
gradient_clipping_norm: Optional[float] = None

if compile_model:
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

# %% [markdown]
# # Training

# %%
share_training_batches = True

@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
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

# %%
n_epochs = 1_000_000_000
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
        optimizer.zero_grad()
        
        # 前向传播
        output = apply_model('train', batch_idx)
        loss = loss_fn(output, Y_train[batch_idx])
        
        # 反向传播
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
        
        # ========== MomoAdam 关键：step 需要传入 loss ==========
        if grad_scaler is None:
            optimizer.step(loss=loss)  # MomoAdam 需要 loss 参数！
        else:
            # AMP 模式下，需要先 unscale
            grad_scaler.unscale_(optimizer)
            optimizer.step(loss=loss)  # MomoAdam 需要 loss 参数！
            # grad_scaler.update()  # Momo 自己处理，不需要
            
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

model.load_state_dict(best_checkpoint['model'])

print('\n[Summary]')
print(f'best epoch:  {best_checkpoint["epoch"]}')
print(f'val score:  {best_checkpoint["metrics"]["val"]}')
print(f'test score: {best_checkpoint["metrics"]["test"]}')

# [Summary]
# best epoch:  56
# val score:  -0.4482693466568687
# test score: -0.4508472942980571