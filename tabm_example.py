# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/yandex-research/tabm/blob/main/example.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>
# 
# # TabM
# 
# This notebook provides a usage example of the `tabm` package from the
# [TabM](https://github.com/yandex-research/tabm) project.

# %%
# !pip install rtdl_num_embeddings
# !pip install tabm

# %%
import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
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

# %%
seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
pass

# %% [markdown]
# # Dataset

# %%
# >>> Dataset.
TaskType = Literal['regression', 'binclass', 'multiclass']

# Regression.
task_type: TaskType = 'regression'
n_classes = None
dataset = sklearn.datasets.fetch_california_housing()
X_num: np.ndarray = dataset['data']
Y: np.ndarray = dataset['target']

# Classification.
# n_classes = 2
# assert n_classes >= 2
# task_type: TaskType = 'binclass' if n_classes == 2 else 'multiclass'
# X_num, Y = sklearn.datasets.make_classification(
#     n_samples=20000,
#     n_features=8,
#     n_classes=n_classes,
#     n_informative=3,
#     n_redundant=2,
# )

task_is_regression = task_type == 'regression'

# >>> Numerical (continuous) features.
X_num: np.ndarray = X_num.astype(np.float32)
n_num_features = X_num.shape[1]

# >>> Categorical features.
# NOTE: the above datasets do not have categorical features, however,
# for the demonstration purposes, it is possible to generate them.
cat_cardinalities = [
    # NOTE: uncomment the two lines below to add two categorical features.
    # 4,  # Allowed values: [0, 1, 2, 3].
    # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
]
X_cat = (
    np.column_stack([np.random.randint(0, c, (len(X_num),)) for c in cat_cardinalities])
    if cat_cardinalities
    else None
)

# >>> Labels.
if task_type == 'regression':
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(range(n_classes)), (
        'Classification labels must form the range [0, 1, ..., n_classes - 1]'
    )

# >>> Split the dataset.
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
        del key, value
    del part, part_data

# %% [markdown]
# # Data preprocessing

# %%
# Feature preprocessing.
# NOTE
# The choice between preprocessing strategies depends on a task and a model.

# Simple preprocessing strategy.
# preprocessing = sklearn.preprocessing.StandardScaler().fit(
#     data_numpy['train']['x_num']
# )

# Advanced preprocessing strategy.
# The noise is added to improve the output of QuantileTransformer in some cases.
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

# Apply the preprocessing.
for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])


# Label preprocessing.
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


Y_train = data_numpy['train']['y'].copy()
if task_type == 'regression':
    # For regression tasks, it is highly recommended to standardize the training labels.
    regression_label_stats = RegressionLabelStats(
        Y_train.mean().item(), Y_train.std().item()
    )
    Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
else:
    regression_label_stats = None

# %% [markdown]
# #  PyTorch settings

# %%
# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convert data to tensors
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(Y_train, device=device)
if task_type == 'regression':
    for part in data:
        data[part]['y'] = data[part]['y'].float()
    Y_train = Y_train.float()

# Automatic mixed precision (AMP)
# torch.float16 is implemented for completeness,
# but it was not tested in the project,
# so torch.bfloat16 is used by default.
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
# Changing False to True can speed up training
# of large enough models on compatible hardware.
amp_enabled = False and amp_dtype is not None
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

# torch.compile
compile_model = False

# fmt: off
print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')
print(f'torch.compile: {compile_model}')
# fmt: on

# %% [markdown]
# # Model and optimizer
# 
# The best performance is usually achieved with `num_embeddings`
# from the `rtdl_num_embeddings` package. Typically, `PiecewiseLinearEmbeddings`
# and `PeriodicEmbeddings` perform best.

# %%
# No embeddings.
num_embeddings = None

# Simple embeddings.
num_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_num_features)

# Periodic embeddings.
num_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)

# Piecewise-linear embeddings.
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=3e-4)
gradient_clipping_norm: Optional[float] = 1.0

if compile_model:
    # NOTE
    # `torch.compile(model, mode="reduce-overhead")` caused issues during training,
    # so the `mode` argument is not used.
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

# %% [markdown]
# # Training

# %%
# A quick reminder: TabM represents an ensemble of k MLPs.
#
# The option below determines if the MLPs are trained
# on the same batches (share_training_batches=True) or
# on different batches. Technically, this option determines:
# - How the loss function is implemented.
# - How the training batches are constructed.
#
# `True` is recommended by default because of better training efficiency.
# On some tasks, `False` may provide better performance.
share_training_batches = True

# %%
@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(
            data[part]['x_num'][idx],
            data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression tasks.
        .float()
    )


base_loss_fn = (
    nn.functional.mse_loss if task_is_regression else nn.functional.cross_entropy
)


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # TabM produces k predictions. Each of them must be trained separately.

    # Regression:     (batch_size, k)            -> (batch_size * k,)
    # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
    y_pred = y_pred.flatten(0, 1)

    if share_training_batches:
        # (batch_size,) -> (batch_size * k,)
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        # (batch_size, k) -> (batch_size * k,)
        y_true = y_true.flatten(0, 1)

    return base_loss_fn(y_pred, y_true)


@evaluation_mode()
def evaluate(part: str) -> float:
    model.eval()

    # When using torch.compile, you may need to reduce the evaluation batch size.
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
        # Transform the predictions back to the original label space.
        assert regression_label_stats is not None
        y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

    # Compute the mean of the k predictions.
    if not task_is_regression:
        # For classification, the mean must be computed in the probability space.
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)

    y_true = data[part]['y'].cpu().numpy()
    score = (
        -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        if task_type == 'regression'
        else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    )
    return float(score)  # The higher -- the better.


print(f'Test score before training: {evaluate("test"):.4f}')

# %%
n_epochs = 1_000_000_000
train_size = len(train_idx)
batch_size = 256
epoch_size = math.ceil(train_size / batch_size)

epoch = -1
metrics = {'val': -math.inf, 'test': -math.inf}


def make_checkpoint() -> dict[str, Any]:
    return deepcopy(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }
    )


best_checkpoint = make_checkpoint()

# Early stopping: the training stops if the validation score
# does not improve for more than `patience` consecutive epochs.
patience = 16
remaining_patience = patience

for epoch in range(n_epochs):
    batches = (
        # Create one standard batch sequence.
        torch.randperm(train_size, device=device).split(batch_size)
        if share_training_batches
        # Create k independent batch sequences.
        else (
            torch.rand((train_size, model.backbone.k), device=device)
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )
    )
    for batch_idx in batches:
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()  # type: ignore

        if gradient_clipping_norm is not None:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), gradient_clipping_norm
            )

        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()

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

# To make final predictions, load the best checkpoint.
model.load_state_dict(best_checkpoint['model'])

print('\n[Summary]')
print(f'best epoch:  {best_checkpoint["epoch"]}')
print(f'val score:  {best_checkpoint["metrics"]["val"]}')
print(f'test score: {best_checkpoint["metrics"]["test"]}')


# [Summary] AdamW 2e-3
# best epoch:  37
# val score:  -0.4514310678517364
# test score: -0.45359759514072767

# [Summary] AdamW 1e-2
# best epoch:  42
# val score:  -0.449956056648022
# test score: -0.45241574981026467
