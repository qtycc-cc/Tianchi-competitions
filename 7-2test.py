import numpy as np
import pandas as pd
import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import rtdl_num_embeddings
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor

# 读取数据
attr = pd.read_csv('data/7/Antai_AE_round1_item_attr_20190626.csv')
train = pd.read_csv('data/7/Antai_AE_round1_train_20190626.csv')
test = pd.read_csv('data/7/Antai_AE_round1_test_20190626.csv')

def transform_date(df: pd.DataFrame) -> pd.DataFrame:
    df['create_order_time'] = pd.to_datetime(df['create_order_time'])
    df['month'] = df['create_order_time'].dt.month
    df['day'] = df['create_order_time'].dt.day
    df['hour'] = df['create_order_time'].dt.hour
    df['dayofweek'] = df['create_order_time'].dt.dayofweek
    df['dayofyear'] = df['create_order_time'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    return df

train = transform_date(train)
test = transform_date(test)

train = train.merge(attr, on='item_id', how='left')
test = test.merge(attr, on='item_id', how='left')

# 用户统计特征
temp = train.groupby('buyer_admin_id').size().reset_index().rename(columns={0: 'count_u'})
train = train.merge(temp, on='buyer_admin_id', how='left')
temp = test.groupby('buyer_admin_id').size().reset_index().rename(columns={0: 'count_u'})
test = test.merge(temp, on='buyer_admin_id', how='left')

temp = train.groupby('buyer_admin_id').create_order_time.nunique().reset_index().rename(columns={'create_order_time': 'days_u'})
train = train.merge(temp, on='buyer_admin_id', how='left')
temp = test.groupby('buyer_admin_id').create_order_time.nunique().reset_index().rename(columns={'create_order_time': 'days_u'})
test = test.merge(temp, on='buyer_admin_id', how='left')

temp = train.groupby('buyer_admin_id')[['item_id', 'cate_id', 'store_id']].nunique().reset_index().rename(columns={'item_id': 'item_count_u', 'cate_id': 'cate_count_u', 'store_id': 'store_count_u'})
train = train.merge(temp, on='buyer_admin_id', how='left')
temp = test.groupby('buyer_admin_id')[['item_id', 'cate_id', 'store_id']].nunique().reset_index().rename(columns={'item_id': 'item_count_u', 'cate_id': 'cate_count_u', 'store_id': 'store_count_u'})
test = test.merge(temp, on='buyer_admin_id', how='left')

def flatten_multiindex_user(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    fix_mapping = {
        'buyer_admin_id_': 'buyer_admin_id',
    }
    df = df.rename(columns=lambda x: fix_mapping.get(x, x))
    return df

agg_dict = {
    'item_price':['first','mean','min','max'],
}
temp = train.groupby(['buyer_admin_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex_user(temp)
train = train.merge(temp, on=['buyer_admin_id'], how='left', suffixes=('', '_user'))

temp = test.groupby(['buyer_admin_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex_user(temp)
test = test.merge(temp, on=['buyer_admin_id'], how='left', suffixes=('', '_user'))

def flatten_multiindex_store(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    fix_mapping = {
        'store_id_': 'store_id',
    }
    df = df.rename(columns=lambda x: fix_mapping.get(x, x))
    return df

temp = train.groupby(['store_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex_store(temp)
train = train.merge(temp, on=['store_id'], how='left', suffixes=('', '_store'))

temp = test.groupby(['store_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex_store(temp)
test = test.merge(temp, on=['store_id'], how='left', suffixes=('', '_store'))

# 频次编码
def frequency_encoding(df: pd.DataFrame, column: str, normalize=False):
    freq = df[column].value_counts(normalize=normalize)
    encoded_col = df[column].map(freq)
    return encoded_col, freq.to_dict()

train['item_id_freq'], freq_dict = frequency_encoding(train, 'item_id', normalize=True)
test['item_id_freq'] = test['item_id'].map(freq_dict).fillna(0)

train['cate_id_freq'], freq_dict = frequency_encoding(train, 'cate_id', normalize=True)
test['cate_id_freq'] = test['cate_id'].map(freq_dict).fillna(0)

train['store_id_freq'], freq_dict = frequency_encoding(train, 'store_id', normalize=True)
test['store_id_freq'] = test['store_id'].map(freq_dict).fillna(0)

# 创建标签
def create_irank1_flag(df):
    irank1_items = df[df['irank'] == 1][['buyer_admin_id', 'item_id']].drop_duplicates()
    irank1_items = irank1_items.rename(columns={'item_id': 'irank1_item_id'})
    df_with_irank1 = df.merge(irank1_items, on='buyer_admin_id', how='left')
    df_with_irank1['is_irank1_item'] = (df_with_irank1['item_id'] == df_with_irank1['irank1_item_id']).astype(int)
    df_with_irank1 = df_with_irank1.drop('irank1_item_id', axis=1)
    return df_with_irank1

train = create_irank1_flag(train)

# 处理测试集
train_item_ids = set(train['item_id'].unique())
for irank in [7]:
    mask = test['irank'] == irank
    test.loc[mask, 'item_id'] = np.where(
        test.loc[mask, 'item_id'].isin(train_item_ids),
        test.loc[mask, 'item_id'],
        5595070
    )
test = test[test['item_id'].isin(train_item_ids)].copy()
test = test.sort_values(
    by=['buyer_admin_id', 'irank'],
    ascending=[True, True]
).reset_index(drop=True)
test['irank'] = test.groupby(['buyer_admin_id']).cumcount() + 1

train = train[train['irank']<=30]
test = test[test['irank']<=30]

# 准备特征和目标
drop_columns = ['buyer_admin_id', 'buyer_country_id', 'item_id', 'cate_id', 'store_id', 'create_order_time']
X = train.drop(columns=['is_irank1_item', *drop_columns])
y = train['is_irank1_item']
X_test = test.drop(columns=drop_columns)

X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# ============== TabM Training Start ==============
# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# 准备数据
X_num = X.values.astype(np.float32)
y = y.values.astype(np.int64)

# 数据集划分
all_idx = np.arange(len(y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8, random_state=seed
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8, random_state=seed
)

data_numpy = {
    'train': {'x_num': X_num[train_idx], 'y': y[train_idx]},
    'val': {'x_num': X_num[val_idx], 'y': y[val_idx]},
    'test': {'x_num': X_num[test_idx], 'y': y[test_idx]},
}

# 数据预处理
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

# PyTorch设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

# AMP设置
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
amp_enabled = False
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None

# 创建TabM模型
n_num_features = X_num.shape[1]
n_classes = 2  # 二分类

# 使用PiecewiseLinearEmbeddings
# num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
#     rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=48),
#     d_embedding=16,
#     activation=False,
#     version='B',
# )

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=None,
    d_out=n_classes,
    num_embeddings=None,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
gradient_clipping_norm = 1.0

# 训练设置
share_training_batches = True
evaluation_mode = torch.inference_mode

# 定义训练函数
@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
def apply_model(part: str, idx: Tensor) -> Tensor:
    return model(data[part]['x_num'][idx], None).float()

base_loss_fn = nn.functional.cross_entropy

def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)  # (batch_size * k, n_classes)
    if share_training_batches:
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        y_true = y_true.flatten(0, 1)
    return base_loss_fn(y_pred, y_true)

@evaluation_mode()
def evaluate(part: str) -> float:
    model.eval()
    eval_batch_size = 1024
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
    
    y_true = data[part]['y'].cpu()
    
    # 将numpy转回tensor进行计算
    y_pred_tensor = torch.from_numpy(y_pred)  # (total_samples, k, n_classes)
    
    if torch.is_floating_point(y_true):
        # 回归任务
        res = (-torch.nn.functional.log_softmax(y_pred_tensor, dim=-1) * y_true.unsqueeze(1).unsqueeze(-1)).sum(dim=-1)
    else:
        # 分类任务
        y_true_expanded = y_true.unsqueeze(1).unsqueeze(-1).expand(-1, y_pred_tensor.shape[1], -1)
        res = -torch.nn.functional.log_softmax(y_pred_tensor, dim=-1).gather(-1, y_true_expanded).squeeze(-1)
    
    score = -res.mean().item()
    
    return float(score)

# 训练循环
n_epochs = 1000
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
patience = 16
remaining_patience = patience

print("Starting TabM training...")
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
        loss = loss_fn(apply_model('train', batch_idx), data['train']['y'][batch_idx])

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()

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

    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
    else:
        remaining_patience -= 1

    # if epoch % 10 == 0 or val_score_improved:
    print(f'Epoch {epoch:3d} | Val: {metrics["val"]:.4f} | Test: {metrics["test"]:.4f}')

    if remaining_patience < 0:
        print(f"Early stopping at epoch {epoch}")
        break

# 加载最佳模型
model.load_state_dict(best_checkpoint['model'])
print(f'Best epoch: {best_checkpoint["epoch"]}')
print(f'Best val score: {best_checkpoint["metrics"]["val"]:.4f}')
print(f'Best test score: {best_checkpoint["metrics"]["test"]:.4f}')

# ============== 预测测试集 ==============
# 准备测试集数据
X_test_num = X_test.values.astype(np.float32)
X_test_num = preprocessing.transform(X_test_num)
X_test_tensor = torch.as_tensor(X_test_num, device=device)

# 预测
model.eval()
eval_batch_size = 1024
with torch.no_grad():
    y_pred = torch.cat([
        model(X_test_tensor[idx], None).float()
        for idx in torch.arange(len(X_test_tensor), device=device).split(eval_batch_size)
    ])  # (n_samples, k, n_classes)

# 在概率空间取平均
y_pred_proba = torch.softmax(y_pred, dim=-1).mean(dim=1)  # (n_samples, n_classes)
test['label_prob'] = y_pred_proba[:, 1].cpu().numpy()  # 取正类概率

# ============== 后续处理（保持不变） ==============
test = test.sort_values(
    by=['buyer_admin_id','label_prob'],
    ascending=[True, False]
).reset_index(drop=True)

def Repeat_Purchase_Reranking(df):
    df = df.drop_duplicates(subset=['buyer_admin_id', 'item_id'])
    df['irank'] = df.groupby(['buyer_admin_id']).cumcount() + 1
    return df

train = Repeat_Purchase_Reranking(train)
test = Repeat_Purchase_Reranking(test)
test = test[test['irank']<=20]

# 获取热门商品信息
train_original = pd.read_csv('data/7/Antai_AE_round1_train_20190626.csv')
train_original = pd.merge(train_original, attr, on='item_id', how='left')
train_original = train_original[train_original['buyer_country_id']=='yy']
train_original.drop(columns=['create_order_time','buyer_country_id'],inplace=True)

popular_items = train_original['item_id'].value_counts().index.tolist()
top_popular_items = popular_items[:30]

cate_top_items = {}
for cate_id in test['cate_id'].unique():
    cate_items = train_original[train_original['cate_id'] == cate_id]['item_id'].value_counts()
    if len(cate_items) > 0:
        cate_top_items[cate_id] = cate_items.head(15).index.tolist()

# 生成最终提交
last_second_purchases = test[test['irank'] == 1][['buyer_admin_id', 'cate_id']].drop_duplicates()
last_third_purchases = test[test['irank'] == 2][['buyer_admin_id', 'cate_id']].drop_duplicates()

last_second_purchases = last_second_purchases.rename(columns={'cate_id': 'cate1'})
last_third_purchases = last_third_purchases.rename(columns={'cate_id': 'cate2'})

final_submission = pd.DataFrame({
    'buyer_admin_id': test['buyer_admin_id'].unique()
})

final_submission = final_submission.merge(last_second_purchases, on='buyer_admin_id', how='left')
final_submission = final_submission.merge(last_third_purchases, on='buyer_admin_id', how='left')

for i in range(1, 31):
    print(i, 'running')
    train_label = test[test['irank'] == int(i)][['buyer_admin_id', 'item_id']]
    train_label = train_label.rename(columns={'item_id': f'predict {i}'})
    final_submission = final_submission.merge(train_label, on='buyer_admin_id', how='left')
    
    if final_submission[f'predict {i}'].isna().any():
        na_mask = final_submission[f'predict {i}'].isna()
        for idx in final_submission[na_mask].index:
            existing_items = set()
            for j in range(1, i):
                existing_items.add(final_submission.at[idx, f'predict {j}'])
            fill_value = None

            if fill_value is None:
                cate1 = final_submission.at[idx, 'cate1']
                if pd.notna(cate1) and cate1 in cate_top_items and len(cate_top_items[cate1]) > 0:
                    for candidate in cate_top_items[cate1]:
                        if candidate not in existing_items:
                            fill_value = candidate
                            break
            
            if fill_value is None:
                cate2 = final_submission.at[idx, 'cate2']
                if pd.notna(cate2) and cate2 in cate_top_items and len(cate_top_items[cate2]) > 0:
                    for candidate in cate_top_items[cate2]:
                        if candidate not in existing_items:
                            fill_value = candidate
                            break
            
            if fill_value is None:
                for candidate in top_popular_items:
                    if candidate not in existing_items:
                        fill_value = candidate
                        break
            
            if fill_value is None:
                fill_value = top_popular_items[(i-1) % len(top_popular_items)]
            
            final_submission.at[idx, f'predict {i}'] = fill_value

final_submission = final_submission.drop(['cate1', 'cate2'], axis=1)
final_submission.to_csv('data/7/submission_tabm.csv', index=False, header=False)
print('Submission csv saved!')