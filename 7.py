import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from util import result_beep

def calculate_time_decay_co_occurrence(train: pd.DataFrame):
    train['create_order_time'] = pd.to_datetime(train['create_order_time'], errors='coerce')  # 转换为datetime
    # 1. 计算每个用户的购买时间（取最后一次购买该商品的时间）
    user_item_last_time = train.groupby(['buyer_admin_id', 'item_id'])['create_order_time'].max().reset_index()
    # 2. 计算时间衰减因子（距离数据最大时间越近，权重越高）
    max_time = train['create_order_time'].max()
    user_item_last_time['time_decay'] = 1 / (1 + (max_time - user_item_last_time['create_order_time']).dt.days / 30)  # 30天半衰期

    # 3. 按用户分组，带时间衰减的物品列表
    user_item_groups = user_item_last_time.groupby('buyer_admin_id').apply(
        lambda x: list(zip(x['item_id'], x['time_decay']))
    ).reset_index()

    # 4. 统计带时间衰减的共现矩阵
    C = dict()
    N = dict()
    for _, row in user_item_groups.iterrows():
        items_with_decay = row[0]  # (item_id, time_decay)列表
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
                decay = d_i * d_j  # 共现衰减=两个商品的衰减乘积
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += decay
    return C, N

def precompute_co_occurrence_recall(C, top_n=20):
    co_recall_dict = {}
    for i in C.keys():
        sorted_co_items = sorted(C[i].items(), key=lambda x: x[1], reverse=True)[:top_n]
        co_recall_dict[i] = [j for j, _ in sorted_co_items]
    return co_recall_dict

def transform_date(df: pd.DataFrame) -> pd.DataFrame:
    df['create_order_time'] = pd.to_datetime(df['create_order_time'])
    df['month'] = df['create_order_time'].dt.month
    df['day'] = df['create_order_time'].dt.day
    df['hour'] = df['create_order_time'].dt.hour
    df['dayofweek'] = df['create_order_time'].dt.dayofweek
    df['dayofyear'] = df['create_order_time'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    # df.drop(columns='create_order_time',inplace=True)
    return df

def flatten_multiindex(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    fix_mapping = {
        f'{key}_': key,
    }
    df = df.rename(columns=lambda x: fix_mapping.get(x, x))
    return df

def frequency_encoding(df: pd.DataFrame, column: str, normalize=False):
    freq = df[column].value_counts(normalize=normalize)
    encoded_col = df[column].map(freq)
    return encoded_col, freq.to_dict()

def create_irank1_flag(df: pd.DataFrame):
    irank1_items = df[df['irank'] == 1][['buyer_admin_id', 'item_id']].drop_duplicates()
    irank1_items = irank1_items.rename(columns={'item_id': 'irank1_item_id'})
    df_with_irank1 = df.merge(irank1_items, on='buyer_admin_id', how='left')
    df_with_irank1['is_irank1_item'] = (df_with_irank1['item_id'] == df_with_irank1['irank1_item_id']).astype(int)
    df_with_irank1 = df_with_irank1.drop('irank1_item_id', axis=1)
    return df_with_irank1

def Repeat_Purchase_Reranking(df):
    df = df.drop_duplicates(subset=['buyer_admin_id', 'item_id'])
    df['irank'] = df.groupby(['buyer_admin_id']).cumcount() + 1
    return df

def train_model(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth':6,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'seed': 42,
        'verbose': -1,
        'device': 'cpu',
    }

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    ens_test = []

    for fold, (train_ix, test_ix) in enumerate(skf.split(X, y)):
        print(f'Fold {fold + 1}')

        X_train, X_val = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_val = y.iloc[train_ix], y.iloc[test_ix]

        lgb_clf = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
        lgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=200)
            ],
        )
        # lgb_val_proba = lgb_clf.predict_proba(X_val)[:, 1]
        lgb_test_proba = lgb_clf.predict_proba(X_test)[:, 1]
        # lgb_val_proba_ = np.where(lgb_val_proba > 0.5, 1, 0)
        ens_test.append(lgb_test_proba)

    mean_preds = np.mean(ens_test,axis=0)
    # 0.16582 24
    # 0.16556 iwt->16
    return mean_preds

@result_beep
def main():
    print("start")
    attr = pd.read_csv('data/7/Antai_AE_round1_item_attr_20190626.csv')
    train = pd.read_csv('data/7/Antai_AE_round1_train_20190626.csv')
    test = pd.read_csv('data/7/Antai_AE_round1_test_20190626.csv')

    # C, N = calculate_time_decay_co_occurrence(train)
    # co_recall_dict = precompute_co_occurrence_recall(C, top_n=10)

    train = transform_date(train)
    test = transform_date(test)

    train = train.merge(attr, on='item_id', how='left')
    test = test.merge(attr, on='item_id', how='left')

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

    agg_dict = {
        'item_price':['first','mean','min','max'],
    }
    temp = train.groupby(['buyer_admin_id'], as_index=False).agg(agg_dict)
    temp = flatten_multiindex(temp, key='buyer_admin_id')
    train = train.merge(temp, on=['buyer_admin_id'], how='left', suffixes=('', '_user'))

    temp = test.groupby(['buyer_admin_id'], as_index=False).agg(agg_dict)
    temp = flatten_multiindex(temp, key="buyer_admin_id")
    test = test.merge(temp, on=['buyer_admin_id'], how='left', suffixes=('', '_user'))

    temp = train.groupby(['store_id'], as_index=False).agg(agg_dict)
    temp = flatten_multiindex(temp, key="store_id")
    train = train.merge(temp, on=['store_id'], how='left', suffixes=('', '_store'))

    temp = test.groupby(['store_id'], as_index=False).agg(agg_dict)
    temp = flatten_multiindex(temp, key="store_id")
    test = test.merge(temp, on=['store_id'], how='left', suffixes=('', '_store'))

    train['item_id_freq'], freq_dict = frequency_encoding(train, 'item_id', normalize=True)
    test['item_id_freq'] = test['item_id'].map(freq_dict).fillna(0)

    train['cate_id_freq'], freq_dict = frequency_encoding(train, 'cate_id', normalize=True)
    test['cate_id_freq'] = test['cate_id'].map(freq_dict).fillna(0)

    train['store_id_freq'], freq_dict = frequency_encoding(train, 'store_id', normalize=True)
    test['store_id_freq'] = test['store_id'].map(freq_dict).fillna(0)

    train = create_irank1_flag(train)

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

    train = train[train['irank']<=30] # !important
    test = test[test['irank']<=30]

    drop_columns = ['buyer_admin_id', 'buyer_country_id', 'item_id', 'cate_id', 'store_id', 'create_order_time']
    X = train.drop(columns=['is_irank1_item', *drop_columns])
    y = train['is_irank1_item']
    X_test = test.drop(columns=drop_columns)

    X.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    test['label_prob'] = train_model(X, y, X_test)

    test = test.sort_values(
        by=['buyer_admin_id','label_prob'],  # 排序的列名列表
        ascending=[True, False]           # 排序方向：用户ID升序，irank升序
    ).reset_index(drop=True)

    train = Repeat_Purchase_Reranking(train)
    test = Repeat_Purchase_Reranking(test)
    test = test[test['irank']<=20]

    train = pd.read_csv('data/7/Antai_AE_round1_train_20190626.csv')
    train = pd.merge(train, attr, on='item_id', how='left')
    train = train[train['buyer_country_id']=='yy']
    train.drop(columns=['create_order_time','buyer_country_id'],inplace=True)

    # 获取全局热门商品
    popular_items = train['item_id'].value_counts().index.tolist()
    top_popular_items = popular_items[:30]
    print(f"Top popular items: {top_popular_items}")

    # 获取每个cate_id的热门商品（基于训练集）
    cate_top_items = {}
    for cate_id in test['cate_id'].unique():
        cate_items = train[train['cate_id'] == cate_id]['item_id'].value_counts()
        if len(cate_items) > 0:
            cate_top_items[cate_id] = cate_items.head(15).index.tolist()

    # 获取每个客户的倒数第二次和倒数第三次购买记录
    last_second_purchases = test[test['irank'] == 1][['buyer_admin_id', 'cate_id']].drop_duplicates()
    last_third_purchases = test[test['irank'] == 2][['buyer_admin_id', 'cate_id']].drop_duplicates()

    last_second_purchases = last_second_purchases.rename(columns={'cate_id': 'cate1'})
    last_third_purchases = last_third_purchases.rename(columns={'cate_id': 'cate2'})

    # 创建最终提交DataFrame
    final_submission = pd.DataFrame({
        'buyer_admin_id': test['buyer_admin_id'].unique()
    })

    # 合并倒数第二次和倒数第三次购买的cate_id信息
    final_submission = final_submission.merge(last_second_purchases, on='buyer_admin_id', how='left')
    final_submission = final_submission.merge(last_third_purchases, on='buyer_admin_id', how='left')

    # 生成预测
    for i in range(1, 31):
        print(i, 'running')
        train_label = test[test['irank'] == int(i)][['buyer_admin_id', 'item_id']]
        train_label = train_label.rename(columns={'item_id': f'predict {i}'})
        final_submission = final_submission.merge(train_label, on='buyer_admin_id', how='left')
        # 填充缺失值
        if final_submission[f'predict {i}'].isna().any():
            na_mask = final_submission[f'predict {i}'].isna()
            for idx in final_submission[na_mask].index:
                # 获取该用户已有的推荐商品
                existing_items = set()
                hist_item = final_submission.at[idx, 'predict 1']
                for j in range(1, i):
                    existing_items.add(final_submission.at[idx, f'predict {j}'])
                fill_value = None

                # if fill_value is None:
                #     if hist_item in co_recall_dict:
                #         # 从共现列表中找未推荐过的商品
                #         for co_item in co_recall_dict[hist_item]:
                #             if co_item not in existing_items:
                #                 fill_value = co_item
                #                 break
                # 先尝试使用cate1的热门商品
                if fill_value is None:
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
                final_submission.at[idx, f'predict {i}'] = fill_value

    final_submission = final_submission.drop(['cate1', 'cate2'], axis=1)
    # 保存结果
    final_submission.to_csv('data/7/submission.csv', index=False, header=False)
    print('Submission csv saved!')

if __name__ == '__main__':
    main()