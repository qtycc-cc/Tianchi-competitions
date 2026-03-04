# 7test.py - 基于原文件，增加用户类型区分和关联推荐（使用原item-cf逻辑）

import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models.iwt_classifier import IWT_Classifier
warnings.filterwarnings('ignore')

print("="*60)
print("安泰杯跨境电商智能算法大赛 - 法国南部团队方案")
print("="*60)

# 读取数据
print("\n[1] 读取数据...")
attr = pd.read_csv('data/7/Antai_AE_round1_item_attr_20190626.csv')
train = pd.read_csv('data/7/Antai_AE_round1_train_20190626.csv')
test = pd.read_csv('data/7/Antai_AE_round1_test_20190626.csv')

print(f"训练集大小: {train.shape}, 训练集用户数: {train['buyer_admin_id'].nunique()}")
print(f"测试集大小: {test.shape}, 测试集用户数: {test['buyer_admin_id'].nunique()}")
print(f"商品属性表大小: {attr.shape}")

# 合并商品属性
train = train.merge(attr, on='item_id', how='left')
test = test.merge(attr, on='item_id', how='left')

## 1. 数据预处理与特征工程

def transform_date(df: pd.DataFrame) -> pd.DataFrame:
    """日期特征工程"""
    df['create_order_time'] = pd.to_datetime(df['create_order_time'])
    df['month'] = df['create_order_time'].dt.month
    df['day'] = df['create_order_time'].dt.day
    df['hour'] = df['create_order_time'].dt.hour
    df['dayofweek'] = df['create_order_time'].dt.dayofweek
    df['dayofyear'] = df['create_order_time'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    return df

print("\n[2] 特征工程...")
train = transform_date(train)
test = transform_date(test)

## 2. 用户特征提取（原代码）

# 用户购买次数
temp = train.groupby('buyer_admin_id').size().reset_index().rename(columns={0: 'count_u'})
train = train.merge(temp, on='buyer_admin_id', how='left')
temp = test.groupby('buyer_admin_id').size().reset_index().rename(columns={0: 'count_u'})
test = test.merge(temp, on='buyer_admin_id', how='left')

# 用户活跃天数
temp = train.groupby('buyer_admin_id')['create_order_time'].nunique().reset_index().rename(columns={'create_order_time': 'days_u'})
train = train.merge(temp, on='buyer_admin_id', how='left')
temp = test.groupby('buyer_admin_id')['create_order_time'].nunique().reset_index().rename(columns={'create_order_time': 'days_u'})
test = test.merge(temp, on='buyer_admin_id', how='left')

# 用户购买品类/店铺/商品多样性
temp = train.groupby('buyer_admin_id')[['item_id', 'cate_id', 'store_id']].nunique().reset_index()\
    .rename(columns={'item_id': 'item_count_u', 'cate_id': 'cate_count_u', 'store_id': 'store_count_u'})
train = train.merge(temp, on='buyer_admin_id', how='left')
temp = test.groupby('buyer_admin_id')[['item_id', 'cate_id', 'store_id']].nunique().reset_index()\
    .rename(columns={'item_id': 'item_count_u', 'cate_id': 'cate_count_u', 'store_id': 'store_count_u'})
test = test.merge(temp, on='buyer_admin_id', how='left')

# 用户价格偏好
def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    fix_mapping = {'buyer_admin_id_': 'buyer_admin_id'}
    df = df.rename(columns=lambda x: fix_mapping.get(x, x))
    return df

agg_dict = {'item_price': ['first', 'mean', 'min', 'max']}
temp = train.groupby(['buyer_admin_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex(temp)
train = train.merge(temp, on=['buyer_admin_id'], how='left', suffixes=('', '_user'))

temp = test.groupby(['buyer_admin_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex(temp)
test = test.merge(temp, on=['buyer_admin_id'], how='left', suffixes=('', '_user'))

## 3. 店铺特征（原代码）

def flatten_multiindex_store(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    fix_mapping = {'store_id_': 'store_id'}
    df = df.rename(columns=lambda x: fix_mapping.get(x, x))
    return df

temp = train.groupby(['store_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex_store(temp)
train = train.merge(temp, on=['store_id'], how='left', suffixes=('', '_store'))

temp = test.groupby(['store_id'], as_index=False).agg(agg_dict)
temp = flatten_multiindex_store(temp)
test = test.merge(temp, on=['store_id'], how='left', suffixes=('', '_store'))

## 4. 频率编码（原代码）

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

## 5. 创建irank=1标志（原代码）

def create_irank1_flag(df):
    irank1_items = df[df['irank'] == 1][['buyer_admin_id', 'item_id']].drop_duplicates()
    irank1_items = irank1_items.rename(columns={'item_id': 'irank1_item_id'})
    df_with_irank1 = df.merge(irank1_items, on='buyer_admin_id', how='left')
    df_with_irank1['is_irank1_item'] = (df_with_irank1['item_id'] == df_with_irank1['irank1_item_id']).astype(int)
    df_with_irank1 = df_with_irank1.drop('irank1_item_id', axis=1)
    return df_with_irank1

train = create_irank1_flag(train)

## 6. 计算商品相似度矩阵（用于关联商品模型）- 使用原item-cf逻辑

def calculate_time_decay_co_occurrence(train: pd.DataFrame):
    """原代码中的时间衰减共现矩阵计算"""
    train['create_order_time'] = pd.to_datetime(train['create_order_time'], errors='coerce')
    # 1. 计算每个用户的购买时间（取最后一次购买该商品的时间）
    user_item_last_time = train.groupby(['buyer_admin_id', 'item_id'])['create_order_time'].max().reset_index()
    # 2. 计算时间衰减因子（距离数据最大时间越近，权重越高）
    max_time = train['create_order_time'].max()
    user_item_last_time['time_decay'] = 1 / (1 + (max_time - user_item_last_time['create_order_time']).dt.days / 30)

    # 3. 按用户分组，带时间衰减的物品列表
    user_item_groups = user_item_last_time.groupby('buyer_admin_id').apply(
        lambda x: list(zip(x['item_id'], x['time_decay']))
    ).reset_index()

    # 4. 统计带时间衰减的共现矩阵
    C = dict()
    N = dict()
    for _, row in user_item_groups.iterrows():
        items_with_decay = row[0]
        items = [i for i, d in items_with_decay]
        decays = [d for i, d in items_with_decay]

        # 更新N（用户覆盖度，带衰减）
        for i, d in zip(items, decays):
            if i not in N:
                N[i] = 0
            N[i] += d
        # 更新C（共现矩阵，带衰减）
        for i_idx, (i, d_i) in enumerate(items_with_decay):
            if i not in C:
                C[i] = dict()
            for j_idx, (j, d_j) in enumerate(items_with_decay):
                if i == j:
                    continue
                decay = d_i * d_j
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += decay
    return C, N

def precompute_co_occurrence_recall(C, top_n=30):
    """原代码中的共现召回"""
    co_recall_dict = {}
    for i in C.keys():
        sorted_co_items = sorted(C[i].items(), key=lambda x: x[1], reverse=True)[:top_n]
        co_recall_dict[i] = [j for j, _ in sorted_co_items]
    return co_recall_dict

print("\n[3] 计算商品相似度矩阵（原item-cf逻辑）...")
C, N = calculate_time_decay_co_occurrence(train[train['buyer_country_id']=='yy'])
co_recall_dict = precompute_co_occurrence_recall(C, top_n=50)

# 全局热门商品（用于冷启动）
popular_items = train['item_id'].value_counts().index.tolist()
top_popular_items = popular_items[:100]

## 7. 构建用户类型判别模型

print("\n[4] 构建用户类型判别模型...")

# 找出每个用户的irank==1的商品
user_last_items = train[train['irank'] == 1][['buyer_admin_id', 'item_id']].drop_duplicates()
user_last_items = user_last_items.rename(columns={'item_id': 'last_item'})

# 找出每个用户之前购买过的所有商品（irank>1）
user_history_items = train[train['irank'] > 1].groupby('buyer_admin_id')['item_id'].apply(list).reset_index()
user_history_items.columns = ['buyer_admin_id', 'history_items']

# 合并
user_purchase_info = user_last_items.merge(user_history_items, on='buyer_admin_id', how='left')
user_purchase_info['history_items'] = user_purchase_info['history_items'].fillna('').apply(
    lambda x: [] if x == '' else x
)

# 判断是否为历史用户（last_item在history_items中出现过）
user_purchase_info['is_history_user'] = user_purchase_info.apply(
    lambda x: 1 if x['last_item'] in x['history_items'] else 0, axis=1
)

print(f"历史用户比例: {user_purchase_info['is_history_user'].mean():.4f}")

# 用户特征用于分类
user_features = ['count_u', 'days_u', 'item_count_u', 'cate_count_u', 'store_count_u']

# 获取每个用户的特征
user_features_df = train.groupby('buyer_admin_id')[user_features].mean().reset_index()
user_purchase_info = user_purchase_info.merge(user_features_df, on='buyer_admin_id', how='left').fillna(0)

# 训练用户分类模型
from sklearn.model_selection import train_test_split
from pytabkit import LGBM_TD_Classifier

X_user = user_purchase_info[user_features]
y_user = user_purchase_info['is_history_user']

X_train_u, X_val_u, y_train_u, y_val_u = train_test_split(X_user, y_user, test_size=0.2, random_state=42)

user_clf = LGBM_TD_Classifier(
    verbosity=1,
    val_metric_name='cross_entropy',
)
user_clf.fit(
    X_train_u, y_train_u,
)

print(f"用户分类模型验证集准确率: {user_clf.score(X_val_u, y_val_u):.4f}")

# 预测每个测试用户的类型概率
test_user_features = test.groupby('buyer_admin_id')[user_features].mean().reset_index().fillna(0)
test_user_probs = user_clf.predict_proba(test_user_features[user_features])[:, 1]

# 为每个用户打上类型标签
test_user_types = pd.DataFrame({
    'buyer_admin_id': test_user_features['buyer_admin_id'],
    'history_user_prob': test_user_probs,
    'user_type': ['history' if p > 0.7 else 'cold' if p < 0.3 else 'mixed' for p in test_user_probs]
})

print(f"用户类型分布: \n{test_user_types['user_type'].value_counts()}")

## 8. 处理测试集中的缺失商品（原代码）

train_item_ids = set(train['item_id'].unique())

for irank in [7]:
    mask = test['irank'] == irank
    test.loc[mask, 'item_id'] = np.where(
        test.loc[mask, 'item_id'].isin(train_item_ids),
        test.loc[mask, 'item_id'],
        5595070
    )

test = test[test['item_id'].isin(train_item_ids)].copy()
test = test.sort_values(['buyer_admin_id', 'irank']).reset_index(drop=True)
test['irank'] = test.groupby(['buyer_admin_id']).cumcount() + 1

train = train[train['irank'] <= 30]
test = test[test['irank'] <= 30]

## 9. 准备模型输入（原代码）

drop_columns = ['buyer_admin_id', 'buyer_country_id', 'item_id', 'cate_id', 'store_id', 'create_order_time']
X = train.drop(columns=['is_irank1_item', *drop_columns], errors='ignore')
y = train['is_irank1_item']
X_test = test.drop(columns=drop_columns, errors='ignore')

X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print(f"\n[5] 模型特征维度: {X.shape}")
print(f"特征列: {X.columns.tolist()}")

## 10. 训练历史交互商品模型

print("\n[6] 训练历史交互商品模型...")

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_groups = X.shape[1]
num_features = X.shape[1]
avg_size = round(num_features // num_groups)
gidx_list = []
for k in range(num_groups):
    gidx_list.extend([k] * avg_size)
gidx_list.extend([num_groups - 1] * (num_features - avg_size * num_groups))
gidx = torch.tensor(gidx_list, dtype=torch.long, device=device)
sgidx = []
for kki in range(num_groups):
    idx = torch.where(gidx == kki)[0]
    sgidx.append(idx)

iwt_clf = IWT_Classifier(
    num_groups=len(sgidx),
    s=num_features,
    gidx=gidx,
    sgidx=sgidx,
    strategy='B',
    equalsize=True,
    verbose=False,
    draw_loss=True
)

pipeline = make_pipeline(
    StandardScaler(), iwt_clf
)

# 训练模型
history_model = pipeline
history_model.fit(
    X_train, y_train,
)

# 预测概率
train_pred = history_model.predict_proba(X_train)[:, 1]
val_pred = history_model.predict_proba(X_val)[:, 1]
test_pred = history_model.predict_proba(X_test)[:, 1]

train_loss = log_loss(y_train, train_pred)
val_loss = log_loss(y_val, val_pred)
print(f"历史模型训练集Log Loss: {train_loss:.6f}, 验证集Log Loss: {val_loss:.6f}")
print(f"历史模型训练集AUC: {roc_auc_score(y_train, train_pred):.4f}, 验证集AUC: {roc_auc_score(y_val, val_pred):.4f}")

# 将预测结果添加到test
test['history_score'] = test_pred

# 合并用户类型信息
test = test.merge(test_user_types[['buyer_admin_id', 'user_type']], on='buyer_admin_id', how='left')
test['user_type'].fillna('mixed', inplace=True)

## 11. 对不同类型用户进行差异化处理 - 使用原item-cf逻辑

print("\n[7] 对不同类型用户进行差异化处理...")

# 为冷启动用户准备关联推荐
cold_users = test[test['user_type'] == 'cold']['buyer_admin_id'].unique()
print(f"找到 {len(cold_users)} 个冷启动用户")

# 创建一个字典存储冷启动用户的推荐
cold_user_recs = {}

# 为每个冷启动用户生成推荐
for user in cold_users:
    # 获取该用户的历史商品
    user_history = test[test['buyer_admin_id'] == user].sort_values('irank')
    history_items = user_history['item_id'].tolist()
    
    # 基于item-cf生成推荐
    candidate_items = []
    candidate_scores = {}
    
    # 遍历历史商品，找相似商品
    for hist_item in history_items[:5]:  # 只考虑最近5个
        if hist_item in co_recall_dict:
            for sim_item in co_recall_dict[hist_item][:10]:
                if sim_item not in candidate_scores:
                    candidate_scores[sim_item] = 0
                candidate_scores[sim_item] += 1
    
    # 按得分排序
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_items = [item for item, score in sorted_candidates[:30]]
    
    # 如果不足30条，用热门商品补全
    if len(candidate_items) < 30:
        for item in top_popular_items:
            if item not in candidate_items and item not in history_items:
                candidate_items.append(item)
                if len(candidate_items) >= 30:
                    break
    
    cold_user_recs[user] = candidate_items

# 为混合用户也生成推荐（备用）
mixed_users = test[test['user_type'] == 'mixed']['buyer_admin_id'].unique()
mixed_user_recs = {}

for user in mixed_users:
    user_history = test[test['buyer_admin_id'] == user].sort_values('irank')
    history_items = user_history['item_id'].tolist()
    
    candidate_items = []
    candidate_scores = {}
    
    for hist_item in history_items[:3]:
        if hist_item in co_recall_dict:
            for sim_item in co_recall_dict[hist_item][:5]:
                if sim_item not in candidate_scores:
                    candidate_scores[sim_item] = 0
                candidate_scores[sim_item] += 1
    
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_items = [item for item, score in sorted_candidates[:15]]
    
    mixed_user_recs[user] = candidate_items

# 替换冷启动用户的预测分数
test.loc[test['user_type'] == 'cold', 'history_score'] = 0.1

## 12. 生成最终预测结果（完全保留原代码的填充逻辑）

print("\n[8] 生成最终预测结果...")

# 按用户和最终得分排序
test_sorted = test.sort_values(['buyer_admin_id', 'history_score'], ascending=[True, False]).reset_index(drop=True)

# 去除重复商品
def Repeat_Purchase_Reranking(df):
    df = df.drop_duplicates(subset=['buyer_admin_id', 'item_id'])
    df['irank'] = df.groupby(['buyer_admin_id']).cumcount() + 1
    return df

train = Repeat_Purchase_Reranking(train)
test_sorted = Repeat_Purchase_Reranking(test_sorted)
test_sorted = test_sorted[test_sorted['irank'] <= 20]

# 重新读取训练数据用于热门商品统计
train_orig = pd.read_csv('data/7/Antai_AE_round1_train_20190626.csv')
train_orig = pd.merge(train_orig, attr, on='item_id', how='left')
train_orig = train_orig[train_orig['buyer_country_id']=='yy']
train_orig.drop(columns=['create_order_time','buyer_country_id'], inplace=True, errors='ignore')

# 获取全局热门商品
popular_items = train_orig['item_id'].value_counts().index.tolist()
top_popular_items = popular_items[:30]
print(f"Top popular items: {top_popular_items[:5]}...")

# 获取每个cate_id的热门商品
cate_top_items = {}
for cate_id in test_sorted['cate_id'].unique():
    if pd.notna(cate_id):
        cate_items = train_orig[train_orig['cate_id'] == cate_id]['item_id'].value_counts()
        if len(cate_items) > 0:
            cate_top_items[cate_id] = cate_items.head(15).index.tolist()

# 获取每个客户的倒数第二次和倒数第三次购买记录
last_second_purchases = test_sorted[test_sorted['irank'] == 1][['buyer_admin_id', 'cate_id']].drop_duplicates()
last_third_purchases = test_sorted[test_sorted['irank'] == 2][['buyer_admin_id', 'cate_id']].drop_duplicates()

last_second_purchases = last_second_purchases.rename(columns={'cate_id': 'cate1'})
last_third_purchases = last_third_purchases.rename(columns={'cate_id': 'cate2'})

# 创建最终提交DataFrame
final_submission = pd.DataFrame({
    'buyer_admin_id': test_sorted['buyer_admin_id'].unique()
})

# 合并倒数第二次和第三次购买的cate_id信息
final_submission = final_submission.merge(last_second_purchases, on='buyer_admin_id', how='left')
final_submission = final_submission.merge(last_third_purchases, on='buyer_admin_id', how='left')

# 生成预测
for i in range(1, 31):
    print(f"  生成第{i}列预测...")
    
    # 获取当前rank的预测
    current_predictions = test_sorted[test_sorted['irank'] == i][['buyer_admin_id', 'item_id']]
    current_predictions = current_predictions.rename(columns={'item_id': f'predict_{i}'})
    final_submission = final_submission.merge(current_predictions, on='buyer_admin_id', how='left')
    
    # 填充缺失值
    if final_submission[f'predict_{i}'].isna().any():
        na_mask = final_submission[f'predict_{i}'].isna()
        for idx in final_submission[na_mask].index:
            user_id = final_submission.at[idx, 'buyer_admin_id']
            
            # 根据用户类型选择不同的填充策略
            user_type_row = test_user_types[test_user_types['buyer_admin_id'] == user_id]
            user_type = user_type_row['user_type'].iloc[0] if len(user_type_row) > 0 else 'mixed'
            
            # 如果是冷启动用户，优先用item-cf推荐
            if user_type == 'cold' and user_id in cold_user_recs:
                rec_items = cold_user_recs[user_id]
                if i-1 < len(rec_items):
                    final_submission.at[idx, f'predict_{i}'] = rec_items[i-1]
                    continue
            
            # 如果是混合用户，也可以尝试用item-cf推荐
            if user_type == 'mixed' and user_id in mixed_user_recs and i-1 < len(mixed_user_recs[user_id]):
                rec_items = mixed_user_recs[user_id]
                if i-1 < len(rec_items):
                    final_submission.at[idx, f'predict_{i}'] = rec_items[i-1]
                    continue
            
            # 否则用原填充逻辑
            existing_items = set()
            for j in range(1, i):
                if f'predict_{j}' in final_submission.columns:
                    existing_items.add(final_submission.at[idx, f'predict_{j}'])
            
            fill_value = None
            
            # 尝试使用cate1的热门商品
            cate1 = final_submission.at[idx, 'cate1']
            if pd.notna(cate1) and cate1 in cate_top_items and len(cate_top_items[cate1]) > 0:
                for candidate in cate_top_items[cate1]:
                    if candidate not in existing_items:
                        fill_value = candidate
                        break
            
            # 再尝试使用cate2的热门商品
            if fill_value is None:
                cate2 = final_submission.at[idx, 'cate2']
                if pd.notna(cate2) and cate2 in cate_top_items and len(cate_top_items[cate2]) > 0:
                    for candidate in cate_top_items[cate2]:
                        if candidate not in existing_items:
                            fill_value = candidate
                            break
            
            # 如果没找到合适的cate_id商品，使用全局热门商品
            if fill_value is None:
                for candidate in top_popular_items:
                    if candidate not in existing_items:
                        fill_value = candidate
                        break
            
            # 如果还是没找到，使用默认热门商品
            if fill_value is None:
                fill_value = top_popular_items[(i-1) % len(top_popular_items)]
            
            # 填充缺失值
            final_submission.at[idx, f'predict_{i}'] = fill_value

# 删除辅助列
final_submission = final_submission.drop(['cate1', 'cate2'], axis=1, errors='ignore')

# 保存结果
output_file = 'data/7/submission.csv'
final_submission.to_csv(output_file, index=False, header=False)
print(f"\n[9] 提交文件已保存至: {output_file}")
print(f"提交文件大小: {final_submission.shape}")
print("\n" + "="*60)
print("完成!")