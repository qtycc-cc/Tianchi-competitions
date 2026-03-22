import gc
import os
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from util import result_beep

# --------------------------
# 全局配置
# --------------------------
# 缓存路径
CACHE_USER_ALL = "data/8/cache_user_all.pkl"
CACHE_ITEM_P = "data/8/cache_item_p.pkl"
CACHE_TRAIN_SET = "data/8/cache_train_set.pkl"
CACHE_TEST_SET = "data/8/cache_test_set.pkl"
USE_CACHE = True

# 时间窗口（答辩文档：滑动窗口策略）
FEATURE_EXTRACTION_SLOT = 7  # 特征提取窗口：7天
TRAIN_START_DATE = datetime.datetime(2014, 11, 18)
LAST_BEHAVIOR_DATE = datetime.datetime(2014, 12, 18)
PREDICT_DATE = datetime.datetime(2014, 12, 19)

# 数据分片
CHUNK_SIZE = 20_000_000

# 数据路径
USER_PART_A_PATH = "data/8/tianchi_fresh_comp_train_user_online_partA.txt"
USER_PART_B_PATH = "data/8/tianchi_fresh_comp_train_user_online_partB.txt"
ITEM_PATH = "data/8/tianchi_fresh_comp_train_item_online.txt"
OUTPUT_PATH = "data/8/tianchi_prediction_1219.txt"

# --------------------------
# 工具函数
# --------------------------
def optimize_data_types(df):
    """优化数据类型，减少内存占用"""
    for col in df.select_dtypes(include=['int64', 'int32']).columns:
        col_max = df[col].max()
        if col_max <= 255:
            df[col] = df[col].astype(np.uint8)
        elif col_max <= 65535:
            df[col] = df[col].astype(np.uint16)
        else:
            df[col] = df[col].astype(np.uint32)
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    
    return df

def _process_user_chunk(chunk, valid_item_ids):
    """处理单块用户行为"""
    chunk['item_id'] = chunk['item_id'].astype(str)
    chunk = chunk[chunk['item_id'].isin(valid_item_ids)]
    if chunk.empty:
        return chunk
    
    # 解析时间
    chunk['daystime'] = pd.to_datetime(chunk['time'].str.split(' ').str[0])
    chunk['hours'] = chunk['time'].str.split(' ').str[1].astype(np.uint8)
    chunk = chunk.drop(columns=['time'], errors='ignore')
    
    # 筛选时间范围
    chunk = chunk[chunk['daystime'] >= TRAIN_START_DATE]
    
    return chunk

# --------------------------
# 数据加载（参考8.ipynb）
# --------------------------
def load_and_merge_data_with_cache():
    print("=== 开始加载原始数据 ===")
    
    # 1. 加载商品子集P
    if USE_CACHE and os.path.exists(CACHE_ITEM_P):
        print(f"加载商品缓存：{CACHE_ITEM_P}")
        item_p = pd.read_pickle(CACHE_ITEM_P)
    else:
        print("生成商品数据缓存...")
        item_cols = ['item_id', 'item_geohash', 'item_category']
        item_p = pd.read_csv(ITEM_PATH, sep='\t', names=item_cols)
        item_p = optimize_data_types(item_p)
        item_p.to_pickle(CACHE_ITEM_P)
        print(f"商品缓存已保存：{CACHE_ITEM_P}")
    
    valid_item_ids = set(item_p['item_id'].astype(str))
    print(f"商品子集P规模：{item_p.shape}（{len(valid_item_ids)}个有效商品）")
    
    # 2. 加载用户行为数据
    if USE_CACHE and os.path.exists(CACHE_USER_ALL):
        print(f"加载用户行为缓存：{CACHE_USER_ALL}")
        user_all = pd.read_pickle(CACHE_USER_ALL)
    else:
        print("生成用户行为数据缓存...")
        user_cols = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        user_chunks = []
        
        # 加载PartA
        for i, chunk in enumerate(pd.read_csv(USER_PART_A_PATH, sep='\t', names=user_cols, chunksize=CHUNK_SIZE)):
            chunk = _process_user_chunk(chunk, valid_item_ids)
            if not chunk.empty:
                user_chunks.append(chunk)
            print(f"PartA加载：{i+1}块，累计{sum(len(c) for c in user_chunks):,}行")
        
        # 加载PartB
        for i, chunk in enumerate(pd.read_csv(USER_PART_B_PATH, sep='\t', names=user_cols, chunksize=CHUNK_SIZE)):
            chunk = _process_user_chunk(chunk, valid_item_ids)
            if not chunk.empty:
                user_chunks.append(chunk)
            print(f"PartB加载：{i+1}块，累计{sum(len(c) for c in user_chunks):,}行")
        
        user_all = pd.concat(user_chunks, ignore_index=True)
        user_all = optimize_data_types(user_all)
        user_all.to_pickle(CACHE_USER_ALL)
        print(f"用户缓存已保存：{CACHE_USER_ALL}")
    
    # 数据校验
    behavior_dist = user_all['behavior_type'].value_counts().sort_index()
    print(f"\n行为分布：1=浏览,2=收藏,3=加购,4=购买")
    print(f"行为统计：{behavior_dist.to_dict()}")
    print(f"购买占比：{behavior_dist.get(4,0)/len(user_all):.4%}")
    print(f"用户数据规模：{user_all.shape}")
    print(f"时间范围：{user_all['daystime'].min()} ~ {user_all['daystime'].max()}")
    
    return user_all, item_p

# --------------------------
# 用户清洗（答辩文档第11页）
# --------------------------
def clean_users(user_all, item_p):
    """
    答辩文档用户清洗策略：
    1. 无收藏、购物车、购买行为的用户
    2. 浏览数过多但从没购买的用户
    3. 对商品子集无收藏、购物车、购买行为的用户
    4. 浏览数/购买数比例过大的用户
    """
    print("\n=== 开始用户清洗 ===")
    
    # 获取商品子集ID
    subset_items = set(item_p['item_id'].astype(str))
    
    # 用户行为统计
    user_stats = user_all.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
    for bt in [1,2,3,4]:
        if bt not in user_stats.columns:
            user_stats[bt] = 0
    user_stats.columns = ['user_id', 'view_count', 'fav_count', 'cart_count', 'buy_count']
    
    # 对商品子集的行为统计
    subset_data = user_all[user_all['item_id'].isin(subset_items)]
    if not subset_data.empty:
        user_subset_stats = subset_data.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
        for bt in [1,2,3,4]:
            if bt not in user_subset_stats.columns:
                user_subset_stats[bt] = 0
        user_subset_stats.columns = ['user_id', 'subset_view', 'subset_fav', 'subset_cart', 'subset_buy']
        user_stats = user_stats.merge(user_subset_stats, on='user_id', how='left').fillna(0)
    else:
        for col in ['subset_view', 'subset_fav', 'subset_cart', 'subset_buy']:
            user_stats[col] = 0
    
    # 清洗规则
    initial_count = len(user_stats)
    
    # 规则1: 无收藏、购物车、购买行为的用户
    mask1 = (user_stats['fav_count'] + user_stats['cart_count'] + user_stats['buy_count'] == 0)
    
    # 规则2: 浏览数过多但从没购买（浏览>100且购买=0）
    mask2 = (user_stats['view_count'] > 100) & (user_stats['buy_count'] == 0)
    
    # 规则3: 对商品子集无收藏、购物车、购买
    mask3 = (user_stats['subset_fav'] + user_stats['subset_cart'] + user_stats['subset_buy'] == 0)
    
    # 规则4: 浏览数/购买数比例过大（>1000且购买>0）
    mask4 = (user_stats['view_count'] / (user_stats['buy_count'] + 1) > 1000)
    
    # 合并所有清洗条件（保留正常用户）
    clean_mask = ~(mask1 | mask2 | mask3 | mask4)
    valid_users = user_stats[clean_mask]['user_id'].values
    
    print(f"清洗前用户数: {initial_count}")
    print(f"清洗后用户数: {len(valid_users)}")
    print(f"移除用户数: {initial_count - len(valid_users)} ({((initial_count - len(valid_users))/initial_count):.2%})")
    
    # 过滤用户行为数据
    user_all_cleaned = user_all[user_all['user_id'].isin(valid_users)]
    
    return user_all_cleaned

# --------------------------
# 特征工程（答辩文档第16-24页）
# --------------------------
def extract_features(data_window, base_pairs):
    """
    答辩文档特征工程：
    - 八种角度特征
    - User-Item / User-Category 交叉特征
    - 排序特征
    - 比例特征
    """
    print(f"特征提取窗口大小: {len(data_window)} 行")
    
    # 1. 用户特征（用户行为计数）
    user_feat = data_window.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
    for bt in [1,2,3,4]:
        if bt not in user_feat.columns:
            user_feat[bt] = 0
    user_feat.columns = ['user_id'] + [f'user_view', f'user_fav', f'user_cart', f'user_buy']
    
    # 用户活跃天数
    user_active = data_window.groupby('user_id')['daystime'].nunique().reset_index(name='user_active_days')
    user_feat = user_feat.merge(user_active, on='user_id', how='left')
    user_feat['user_view_to_buy'] = user_feat['user_buy'] / (user_feat['user_view'] + 1)
    user_feat['user_cart_to_buy'] = user_feat['user_buy'] / (user_feat['user_cart'] + 1)
    user_feat['user_fav_to_buy'] = user_feat['user_buy'] / (user_feat['user_fav'] + 1)
    
    # 2. 商品特征
    item_feat = data_window.groupby('item_id')['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
    for bt in [1,2,3,4]:
        if bt not in item_feat.columns:
            item_feat[bt] = 0
    item_feat.columns = ['item_id'] + [f'item_view', f'item_fav', f'item_cart', f'item_buy']

    temp = data_window.groupby('item_id').size().reset_index().rename(columns={0: 'item_count'})
    item_feat = item_feat.merge(temp, on='item_id', how='left')

    temp = data_window.groupby('item_id')[['user_id']].nunique().reset_index().rename(columns={'user_id': 'user_count_i'})
    item_feat = item_feat.merge(temp, on='item_id', how='left')
    
    # 商品转化率
    item_feat['item_view_to_buy'] = item_feat['item_buy'] / (item_feat['item_view'] + 1)
    item_feat['item_cart_to_buy'] = item_feat['item_buy'] / (item_feat['item_cart'] + 1)
    item_feat['item_fav_to_buy'] = item_feat['item_buy'] / (item_feat['item_fav'] + 1)
    item_feat['item_buy_freq'] = item_feat['item_buy'] / (item_feat['item_count'] + 1)
    
    # 3. 品类特征
    cate_feat = data_window.groupby('item_category')['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
    for bt in [1,2,3,4]:
        if bt not in cate_feat.columns:
            cate_feat[bt] = 0
    cate_feat.columns = ['item_category'] + [f'cate_view', f'cate_fav', f'cate_cart', f'cate_buy']

    temp = data_window.groupby('item_category').size().reset_index().rename(columns={0: 'cate_count'})
    cate_feat = cate_feat.merge(temp, on='item_category', how='left')

    temp = data_window.groupby('item_category')[['user_id']].nunique().reset_index().rename(columns={'user_id': 'user_count_c'})
    cate_feat = cate_feat.merge(temp, on='item_category', how='left')

    cate_feat['cate_view_to_buy'] = cate_feat['cate_buy'] / (cate_feat['cate_view'] + 1)
    cate_feat['cate_cart_to_buy'] = cate_feat['cate_buy'] / (cate_feat['cate_cart'] + 1)
    cate_feat['cate_fav_to_buy'] = cate_feat['cate_buy'] / (cate_feat['cate_fav'] + 1)
    
    # 4. User-Item 交叉特征（答辩文档第22页）
    ui_feat = data_window.groupby(['user_id', 'item_id'])['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
    for bt in [1,2,3,4]:
        if bt not in ui_feat.columns:
            ui_feat[bt] = 0
    ui_feat.columns = ['user_id', 'item_id'] + [f'ui_view', f'ui_fav', f'ui_cart', f'ui_buy']
    
    # UI交叉比例特征
    ui_feat['ui_view_to_buy'] = ui_feat['ui_buy'] / (ui_feat['ui_view'] + 1)
    ui_feat['ui_cart_to_buy'] = ui_feat['ui_buy'] / (ui_feat['ui_cart'] + 1)
    ui_feat['ui_fav_to_buy'] = ui_feat['ui_buy'] / (ui_feat['ui_fav'] + 1)
    
    # 5. User-Category 交叉特征
    uc_feat = data_window.groupby(['user_id', 'item_category'])['behavior_type'].value_counts().unstack(fill_value=0).reset_index()
    for bt in [1,2,3,4]:
        if bt not in uc_feat.columns:
            uc_feat[bt] = 0
    uc_feat.columns = ['user_id', 'item_category'] + [f'uc_view', f'uc_fav', f'uc_cart', f'uc_buy']
    
    uc_feat['uc_buy_ratio'] = uc_feat['uc_buy'] / (uc_feat['uc_view'] + uc_feat['uc_fav'] + uc_feat['uc_cart'] + 1)
    
    # 6. 排序特征（答辩文档第23-24页）
    # 用户在品类内对该商品的浏览次数排名
    ui_with_cate = data_window.groupby(['user_id', 'item_category', 'item_id']).size().reset_index(name='count')
    ui_with_cate['view_rank_in_uc'] = ui_with_cate.groupby(['user_id', 'item_category'])['count'].rank(ascending=False, method='dense')
    rank_feat = ui_with_cate[['user_id', 'item_id', 'view_rank_in_uc']]
    
    # 合并所有特征到基础样本
    base_pairs = base_pairs.merge(user_feat, on='user_id', how='left')
    base_pairs = base_pairs.merge(item_feat, on='item_id', how='left')
    base_pairs = base_pairs.merge(cate_feat, on='item_category', how='left')
    base_pairs = base_pairs.merge(ui_feat, on=['user_id', 'item_id'], how='left')
    base_pairs = base_pairs.merge(uc_feat, on=['user_id', 'item_category'], how='left')
    base_pairs = base_pairs.merge(rank_feat, on=['user_id', 'item_id'], how='left')
    
    # 填充缺失值
    base_pairs = base_pairs.fillna(0)
    
    return base_pairs

# --------------------------
# 训练集构建（答辩文档第13-14页，滑动窗口策略）
# --------------------------
def build_train_set_with_cache(user_all, item_p):
    print("\n=== 开始处理训练集 ===")
    
    if USE_CACHE and os.path.exists(CACHE_TRAIN_SET):
        print(f"加载训练集缓存：{CACHE_TRAIN_SET}")
        train_set = pd.read_pickle(CACHE_TRAIN_SET)
        return train_set
    
    print("生成训练集（滑动窗口策略）...")
    
    # 获取所有日期
    all_dates = sorted(user_all['daystime'].unique())
    train_set_list = []
    
    # 滑动窗口：使用多个日期作为标签日
    for days_offset in range(5, 0, -1):  # 使用最近5天作为标签日
        label_date = LAST_BEHAVIOR_DATE - datetime.timedelta(days=days_offset)
        if label_date < TRAIN_START_DATE + datetime.timedelta(days=FEATURE_EXTRACTION_SLOT):
            continue
            
        feat_start = label_date - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT)
        
        print(f"\n处理窗口: 特征 [{feat_start.date()} ~ {label_date.date()}]")
        
        # 特征窗口数据
        feat_data = user_all[(user_all['daystime'] >= feat_start) & 
                             (user_all['daystime'] < label_date)]
        
        # 基础样本：标签日前一天有交互的用户-商品对（答辩文档第13页）
        prev_day = label_date - datetime.timedelta(days=1)
        base_samples = user_all[user_all['daystime'] == prev_day].copy()
        base_samples = base_samples.drop_duplicates(['user_id', 'item_id'])
        
        if base_samples.empty:
            continue
            
        # 正样本：标签日当天的购买行为
        pos_samples = user_all[(user_all['daystime'] == label_date) & 
                               (user_all['behavior_type'] == 4)]
        pos_pairs = set(zip(pos_samples['user_id'], pos_samples['item_id']))
        
        # 打标
        base_samples['label'] = base_samples.apply(
            lambda x: 1 if (x['user_id'], x['item_id']) in pos_pairs else 0, axis=1
        )
        
        print(f"  基础样本: {len(base_samples)} 对")
        print(f"  正样本: {len(pos_samples)} 对 ({base_samples['label'].sum()} 匹配)")
        
        # 提取特征
        base_samples = extract_features(feat_data, base_samples)
        train_set_list.append(base_samples)
        
        # 内存管理
        del feat_data, pos_samples
        gc.collect()
    
    if not train_set_list:
        raise ValueError("没有生成任何训练样本！")
    
    train_set = pd.concat(train_set_list, ignore_index=True)
    
    # 保存缓存
    train_set.to_pickle(CACHE_TRAIN_SET)
    print(f"\n训练集缓存已保存：{CACHE_TRAIN_SET}")
    print(f"训练集规模：{train_set.shape}")
    print(f"正样本数：{train_set['label'].sum()} ({train_set['label'].mean():.4%})")
    
    return train_set

# --------------------------
# 测试集构建（答辩文档第13页）
# --------------------------
def build_test_set_with_cache(user_all, item_p):
    print("\n=== 开始处理测试集 ===")
    
    if USE_CACHE and os.path.exists(CACHE_TEST_SET):
        print(f"加载测试集缓存：{CACHE_TEST_SET}")
        test_set = pd.read_pickle(CACHE_TEST_SET)
        return test_set
    
    print("生成测试集...")
    
    # 特征窗口：12.11-12.18
    feat_start = LAST_BEHAVIOR_DATE - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT)
    feat_data = user_all[(user_all['daystime'] >= feat_start) & 
                         (user_all['daystime'] <= LAST_BEHAVIOR_DATE)]
    
    # 基础样本：12.18有交互的用户-商品对
    base_test = user_all[user_all['daystime'] == LAST_BEHAVIOR_DATE].copy()
    base_test = base_test.drop_duplicates(['user_id', 'item_id'])
    
    # 确保只预测商品子集P中的商品
    subset_items = set(item_p['item_id'].astype(str))
    base_test = base_test[base_test['item_id'].isin(subset_items)]
    
    print(f"测试集基础样本: {len(base_test)} 对")
    
    # 提取特征
    test_set = extract_features(feat_data, base_test)
    
    # 保存缓存
    test_set.to_pickle(CACHE_TEST_SET)
    print(f"测试集缓存已保存：{CACHE_TEST_SET}")
    print(f"测试集规模：{test_set.shape}")
    
    return test_set

# --------------------------
# 样本平衡（答辩文档第31页）
# --------------------------
def balance_train_set(train_set, pos_neg_ratio=30):
    """负样本抽样，控制正负比"""
    print("\n=== 开始样本平衡 ===")
    
    train_pos = train_set[train_set['label'] == 1].copy()
    train_neg = train_set[train_set['label'] == 0].copy()
    
    pos_count = len(train_pos)
    neg_count = len(train_neg)
    
    print(f"平衡前：正样本 {pos_count:,}，负样本 {neg_count:,}，比例 1:{neg_count/pos_count:.1f}")
    
    # 负样本抽样
    sample_neg_size = min(neg_count, pos_count * pos_neg_ratio)
    train_neg_sampled = train_neg.sample(n=sample_neg_size, random_state=42)
    
    balanced_train = pd.concat([train_pos, train_neg_sampled], ignore_index=True)
    balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"平衡后：正样本 {len(train_pos):,}，负样本 {len(train_neg_sampled):,}，比例 1:{len(train_neg_sampled)/pos_count:.1f}")
    
    return balanced_train

# --------------------------
# 模型训练与预测（答辩文档第28-30页）
# --------------------------
def train_and_predict(balanced_train, test_set):
    print("\n=== 开始模型训练与预测 ===")
    
    # 准备特征
    feature_cols = [col for col in balanced_train.columns 
                    if col not in ['user_id', 'daystime', 'label', 'user_geohash', 'item_category', 'item_id']]
    
    print(f"使用特征数: {len(feature_cols)}")
    print(f"使用特征: {feature_cols}")
    
    X = balanced_train[feature_cols]
    y = balanced_train['label']
    X_test = test_set[feature_cols]
    categorical_features = ['behavior_type', 'hours']
    for col in categorical_features:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    # lgb_params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': 'AUC',
    #     'max_depth': 4,
    #     'learning_rate': 0.01,
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.8,
    #     'seed': 42,
    #     'verbose': -1,
    #     'n_estimators': 8000
    # }

    # xgb_params = {
    #     'max_depth': 4,
    #     'learning_rate': 0.01,
    #     'n_estimators': 8000,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'random_state': 42,
    #     'eval_metric': 'auc',
    #     'enable_categorical': True,
    #     'use_label_encoder': False
    # }

    # cat_params = {
    #     'iterations': 8000,
    #     'learning_rate': 0.01,
    #     'depth': 5,
    #     'l2_leaf_reg': 3,
    #     'random_seed': 42,
    #     'loss_function': 'Logloss',
    #     'eval_metric': 'AUC',
    #     'od_type': 'Iter',
    #     'od_wait': 200,
    #     'verbose': 200
    # }

    # skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    # fold_auc_lgb = []
    # fold_auc_xgb = []
    # fold_auc_cat = []
    # fold_auc_ensemble = []

    # ens_test_lgb = []
    # ens_test_xgb = []
    # ens_test_cat = []

    # print("\nStarting training...")

    # for fold, (train_ix, val_ix) in enumerate(skf.split(X, y)):
    #     print(f"\n========== Fold {fold+1} ==========")
    #     X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    #     y_train, y_val = y.iloc[train_ix], y.iloc[val_ix]
        
    #     # ----- LightGBM -----
    #     print("--- Training LightGBM ---")
    #     lgb_clf = lgb.LGBMClassifier(**lgb_params)
    #     lgb_clf.fit(
    #         X_train, y_train,
    #         eval_set=[(X_val, y_val)],
    #         callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=200)],
    #         categorical_feature=categorical_features
    #     )
    #     lgb_val_proba = lgb_clf.predict_proba(X_val)[:, 1]
    #     lgb_test_proba = lgb_clf.predict_proba(X_test)[:, 1]
    #     lgb_auc = roc_auc_score(y_val, lgb_val_proba)
    #     print(f'LGB AUC: {lgb_auc:.6f}')
    #     fold_auc_lgb.append(lgb_auc)
    #     ens_test_lgb.append(lgb_test_proba)
        
    #     # ----- XGBoost -----
    #     print("--- Training XGBoost ---")
    #     X_train_xgb = X_train.copy()
    #     X_val_xgb = X_val.copy()
    #     X_test_xgb = X_test.copy()
    #     for col in categorical_features:
    #         X_train_xgb[col] = X_train_xgb[col].cat.codes
    #         X_val_xgb[col] = X_val_xgb[col].cat.codes
    #         X_test_xgb[col] = X_test_xgb[col].cat.codes
        
    #     xgb_clf = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=200)
    #     xgb_clf.fit(
    #         X_train_xgb, y_train,
    #         eval_set=[(X_val_xgb, y_val)],
    #         # early_stopping_rounds=200,
    #         # verbose=200
    #     )
    #     xgb_val_proba = xgb_clf.predict_proba(X_val_xgb)[:, 1]
    #     xgb_test_proba = xgb_clf.predict_proba(X_test_xgb)[:, 1]
    #     xgb_auc = roc_auc_score(y_val, xgb_val_proba)
    #     print(f'XGB AUC: {xgb_auc:.6f}')
    #     fold_auc_xgb.append(xgb_auc)
    #     ens_test_xgb.append(xgb_test_proba)
        
    #     # ----- CatBoost -----
    #     print("--- Training CatBoost ---")
    #     cat_features_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    #     cat_clf = cb.CatBoostClassifier(**cat_params)
    #     cat_clf.fit(
    #         X_train, y_train,
    #         eval_set=(X_val, y_val),
    #         early_stopping_rounds=200,
    #         cat_features=cat_features_indices,
    #         verbose=200,
    #         use_best_model=True
    #     )
    #     cat_val_proba = cat_clf.predict_proba(X_val)[:, 1]
    #     cat_test_proba = cat_clf.predict_proba(X_test)[:, 1]
    #     cat_auc = roc_auc_score(y_val, cat_val_proba)
    #     print(f'Cat AUC: {cat_auc:.6f}')
    #     fold_auc_cat.append(cat_auc)
    #     ens_test_cat.append(cat_test_proba)
        
    #     # ----- 融合 -----
    #     ensemble_val_proba = lgb_val_proba * 0.4 + xgb_val_proba * 0.2 + cat_val_proba * 0.4
    #     ensemble_auc = roc_auc_score(y_val, ensemble_val_proba)
    #     print(f'Ensemble AUC: {ensemble_auc:.6f}')
    #     fold_auc_ensemble.append(ensemble_auc)

    # print("\n" + "="*50)
    # print("Cross-Validation Results:")
    # print(f"LightGBM  : {np.mean(fold_auc_lgb):.6f} ± {np.std(fold_auc_lgb):.6f}")
    # print(f"XGBoost   : {np.mean(fold_auc_xgb):.6f} ± {np.std(fold_auc_xgb):.6f}")
    # print(f"CatBoost  : {np.mean(fold_auc_cat):.6f} ± {np.std(fold_auc_cat):.6f}")
    # print(f"Ensemble  : {np.mean(fold_auc_ensemble):.6f} ± {np.std(fold_auc_ensemble):.6f}")

    # mean_preds_lgb = np.mean(ens_test_lgb, axis=0)
    # mean_preds_xgb = np.mean(ens_test_xgb, axis=0)
    # mean_preds_cat = np.mean(ens_test_cat, axis=0)
    # test_pred = mean_preds_lgb * 0.4 + mean_preds_xgb * 0.2 + mean_preds_cat * 0.4

    from pytabkit import LGBM_TD_Classifier

    model = LGBM_TD_Classifier(
        verbosity=1,
        random_state=42,
        val_metric_name='1-auc_ovr_alt',
        n_cv=10,
    )
    model.fit(X, y)
    test_pred = model.predict_proba(X_test)[:, 1]

    # 输出Top N推荐
    test_set['score'] = test_pred
    result = test_set[['user_id', 'item_id', 'score']].copy()
    result['user_id'] = result['user_id'].astype(str)
    result['item_id'] = result['item_id'].astype(str)
    
    # 按得分排序，取前10000条（控制输出规模）
    print(f"The len of result is: {len(result)}")
    result = result.sort_values('score', ascending=False).head(50000)
    
    # 保存结果（赛题要求格式）
    result[['user_id', 'item_id']].to_csv(OUTPUT_PATH, sep='\t', index=False, header=False)
    print(f"预测结果已保存：{OUTPUT_PATH}")
    print(f"输出购买对数量：{len(result)}")
    
    return result

# --------------------------
# 主流程
# --------------------------
@result_beep
def main():
    print("=" * 50)
    print("阿里移动推荐算法大赛")
    print("=" * 50)
    
    # 1. 加载数据
    user_all, item_p = load_and_merge_data_with_cache()
    
    # 2. 用户清洗 不需要
    # user_all_cleaned = clean_users(user_all, item_p)
    user_all_cleaned = user_all
    
    # 3. 构建训练集
    train_set = build_train_set_with_cache(user_all_cleaned, item_p)
    
    # 4. 构建测试集
    test_set = build_test_set_with_cache(user_all_cleaned, item_p)
    
    # 5. 样本平衡
    balanced_train = balance_train_set(train_set)
    
    # 6. 模型训练与预测
    final_result = train_and_predict(balanced_train, test_set)
    
    print("\n=== 完成 ===")

if __name__ == '__main__':
    main()