import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import xgboost as xgb
import catboost as cb
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# from models.iwt_classifier import IWT_Classifier

# ==================== 1. 数据加载 ====================
data_path = 'data/2/'

train = pd.read_csv(data_path + 'train_format1.csv')
test = pd.read_csv(data_path + 'test_format1.csv')
user_info = pd.read_csv(data_path + 'user_info_format1.csv')
user_log = pd.read_csv(data_path + 'user_log_format1.csv')

user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
test = test.drop(columns=['prob'])

print(f"Train: {train.shape}, Test: {test.shape}")

# ==================== 2. 时间编码 ====================
def mmdd_to_dayofyear(mmdd_str, year=2025):
    s = str(mmdd_str).strip()
    length = len(s)
    
    try:
        if length == 3:
            month = int(s[0])
            day = int(s[1:3])
        elif length == 4:
            first_two = int(s[:2])
            if first_two <= 12:
                month = first_two
                day = int(s[2:4])
            else:
                month = int(s[0])
                day = int(s[1:4])
                if day > 31:
                    raise ValueError("无效日期")
        else:
            raise ValueError("长度必须是3或4")
        
        if month < 1 or month > 12 or day < 1 or day > 31:
            raise ValueError("无效月/日")
        
        date_obj = datetime(year, month, day)
        return date_obj.timetuple().tm_yday
    
    except (ValueError, IndexError):
        return None

user_log['time_stamp'] = user_log['time_stamp'].apply(mmdd_to_dayofyear)

# ==================== 3. 特征工程 ====================
user_log['time11'] = np.where(user_log['time_stamp'] == 315, 1, 0)

action_dummies = pd.get_dummies(user_log['action_type'], prefix='action')
user_log = pd.concat([user_log, action_dummies], axis=1)

user_log = user_log.sort_values(
    by=['user_id', 'merchant_id', 'time_stamp'],
    ascending=[True, True, False]
).reset_index(drop=True)

def extract_features(user_log, train_data, test_data, user_info):
    """提取特征"""
    
    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    
    train_labels = train_data['label'].copy()
    
    common_cols = ['user_id', 'merchant_id']
    data = pd.concat([
        train_data[common_cols + ['is_train']], 
        test_data[common_cols + ['is_train']]
    ], ignore_index=True)
    
    data = data.merge(user_info, on='user_id', how='left')
    data['age_range'].fillna(0, inplace=True)
    data['gender'].fillna(2, inplace=True)
    
    um_last = user_log.drop_duplicates(subset=['user_id', 'merchant_id'], keep='first')
    data = data.merge(
        um_last[['user_id', 'merchant_id', 'item_id', 'cat_id', 'brand_id']], 
        on=['user_id', 'merchant_id'], 
        how='left'
    )
    
    # ==================== 各级别特征 ====================
    um_group = user_log.groupby(['user_id', 'merchant_id'])
    
    um_time11 = um_group['time11'].agg(['mean', 'min', 'max', 'sum']).reset_index()
    um_time11.columns = ['user_id', 'merchant_id', 'time11_mean', 'time11_min', 'time11_max', 'time11_sum']
    data = data.merge(um_time11, on=['user_id', 'merchant_id'], how='left')
    
    um_time = um_group['time_stamp'].agg(['mean', 'min', 'max', 'nunique']).reset_index()
    um_time.columns = ['user_id', 'merchant_id', 'time_stamp_mean', 'time_stamp_min', 'time_stamp_max', 'time_stamp_nunique']
    data = data.merge(um_time, on=['user_id', 'merchant_id'], how='left')
    
    for action in [0, 1, 2, 3]:
        um_action = um_group[f'action_{action}'].agg(['mean', 'nunique', 'sum']).reset_index()
        um_action.columns = ['user_id', 'merchant_id', f'action_{action}_mean', f'action_{action}_nunique', f'action_{action}_sum']
        data = data.merge(um_action, on=['user_id', 'merchant_id'], how='left')
    
    um_item = um_group['item_id'].agg(['nunique', 'count']).reset_index()
    um_item.columns = ['user_id', 'merchant_id', 'item_id_nunique', 'item_id_count']
    data = data.merge(um_item, on=['user_id', 'merchant_id'], how='left')
    
    um_cat = um_group['cat_id'].agg(['nunique', 'count']).reset_index()
    um_cat.columns = ['user_id', 'merchant_id', 'cat_id_nunique', 'cat_id_count']
    data = data.merge(um_cat, on=['user_id', 'merchant_id'], how='left')
    
    # User级别
    u_group = user_log.groupby('user_id')
    
    u_time11 = u_group['time11'].agg(['mean', 'min', 'max', 'sum']).reset_index()
    u_time11.columns = ['user_id', 'time11_mean_log', 'time11_min_log', 'time11_max_log', 'time11_sum_log']
    data = data.merge(u_time11, on='user_id', how='left')
    
    u_time = u_group['time_stamp'].agg(['mean', 'min', 'max', 'nunique']).reset_index()
    u_time.columns = ['user_id', 'time_stamp_mean_log', 'time_stamp_min_log', 'time_stamp_max_log', 'time_stamp_nunique_log']
    data = data.merge(u_time, on='user_id', how='left')
    
    for action in [0, 1, 2, 3]:
        u_action = u_group[f'action_{action}'].agg(['mean', 'nunique', 'sum']).reset_index()
        u_action.columns = ['user_id', f'action_{action}_mean_log', f'action_{action}_nunique_log', f'action_{action}_sum_log']
        data = data.merge(u_action, on='user_id', how='left')
    
    u_item = u_group['item_id'].agg(['nunique', 'count']).reset_index()
    u_item.columns = ['user_id', 'item_id_nunique_log', 'item_id_count_log']
    data = data.merge(u_item, on='user_id', how='left')
    
    u_cat = u_group['cat_id'].agg(['nunique', 'count']).reset_index()
    u_cat.columns = ['user_id', 'cat_id_nunique_log', 'cat_id_count_log']
    data = data.merge(u_cat, on='user_id', how='left')
    
    # Merchant级别
    m_group = user_log.groupby('merchant_id')
    
    m_user = m_group['user_id'].agg(['nunique', 'count']).reset_index()
    m_user.columns = ['merchant_id', 'user_id_nunique', 'user_id_count']
    data = data.merge(m_user, on='merchant_id', how='left')
    
    for action in [0, 1, 2, 3]:
        m_action = m_group[f'action_{action}'].agg(['mean', 'nunique', 'sum']).reset_index()
        m_action.columns = ['merchant_id', f'action_{action}_mean_merchant', f'action_{action}_nunique_merchant', f'action_{action}_sum_merchant']
        data = data.merge(m_action, on='merchant_id', how='left')
    
    # Item级别
    item_group = user_log.groupby('item_id')
    for action in [0, 1, 2, 3]:
        item_action = item_group[f'action_{action}'].agg(['mean', 'nunique', 'sum']).reset_index()
        item_action.columns = ['item_id', f'action_{action}_mean_item', f'action_{action}_nunique_item', f'action_{action}_sum_item']
        data = data.merge(item_action, on='item_id', how='left')
    
    # Cat级别
    cat_group = user_log.groupby('cat_id')
    for action in [0, 1, 2, 3]:
        cat_action = cat_group[f'action_{action}'].agg(['mean', 'nunique', 'sum']).reset_index()
        cat_action.columns = ['cat_id', f'action_{action}_mean_cat', f'action_{action}_nunique_cat', f'action_{action}_sum_cat']
        data = data.merge(cat_action, on='cat_id', how='left')
    
    # Brand级别
    if 'brand_id' in data.columns:
        brand_group = user_log.groupby('brand_id')
        for action in [0, 1, 2, 3]:
            brand_action = brand_group[f'action_{action}'].agg(['mean', 'nunique', 'sum']).reset_index()
            brand_action.columns = ['brand_id', f'action_{action}_mean_brand', f'action_{action}_nunique_brand', f'action_{action}_sum_brand']
            data = data.merge(brand_action, on='brand_id', how='left')
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = data[numerical_cols].fillna(0)
    
    return data, train_labels

# 提取特征
print("Extracting features...")
feature_data, y = extract_features(user_log, train, test, user_info)

X = feature_data[feature_data['is_train'] == 1].drop(columns=['is_train'])
X_test = feature_data[feature_data['is_train'] == 0].drop(columns=['is_train'])

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# ==================== 4. 类别特征处理 ====================
cate_counts = X['cat_id'].value_counts()
top_cates = cate_counts.head(180).index
X['cat_id'] = X['cat_id'].apply(lambda x: x if x in top_cates else -1)
X_test['cat_id'] = X_test['cat_id'].apply(lambda x: x if x in top_cates else -1)

item_counts = X['item_id'].value_counts()
top_items = item_counts.head(300).index
X['item_id'] = X['item_id'].apply(lambda x: x if x in top_items else -1)
X_test['item_id'] = X_test['item_id'].apply(lambda x: x if x in top_items else -1)

categorical_features = ['user_id', 'merchant_id', 'cat_id', 'item_id']
for col in categorical_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

X_test = X_test[X.columns]
# TabM need to process
# X = X.drop(columns=['user_id', 'merchant_id'])
# X_test = X_test.drop(columns=['user_id', 'merchant_id'])
print(f"Final X shape: {X.shape}, X_test shape: {X_test.shape}")

print(f"cols = {X.columns.to_list()}")

# ==================== 5. 三模型融合训练 ====================
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'AUC',
    'max_depth': 4,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'seed': 42,
    'verbose': -1,
    'n_estimators': 8000
}

xgb_params = {
    'max_depth': 4,
    'learning_rate': 0.01,
    'n_estimators': 8000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'auc',
    'enable_categorical': True,
    'use_label_encoder': False
}

cat_params = {
    'iterations': 8000,
    'learning_rate': 0.01,
    'depth': 5,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'od_wait': 200,
    'verbose': 200
}

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

fold_auc_lgb = []
fold_auc_xgb = []
fold_auc_cat = []
fold_auc_ensemble = []

ens_test_lgb = []
ens_test_xgb = []
ens_test_cat = []

print("\nStarting training...")

for fold, (train_ix, val_ix) in enumerate(skf.split(X, y)):
    print(f"\n========== Fold {fold+1} ==========")
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y.iloc[train_ix], y.iloc[val_ix]
    
    # ----- LightGBM -----
    print("--- Training LightGBM ---")
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=200)],
        categorical_feature=categorical_features
    )
    lgb_val_proba = lgb_clf.predict_proba(X_val)[:, 1]
    lgb_test_proba = lgb_clf.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_val_proba)
    print(f'LGB AUC: {lgb_auc:.6f}')
    fold_auc_lgb.append(lgb_auc)
    ens_test_lgb.append(lgb_test_proba)
    
    # ----- XGBoost -----
    print("--- Training XGBoost ---")
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()
    for col in categorical_features:
        X_train_xgb[col] = X_train_xgb[col].cat.codes
        X_val_xgb[col] = X_val_xgb[col].cat.codes
        X_test_xgb[col] = X_test_xgb[col].cat.codes
    
    xgb_clf = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=200)
    xgb_clf.fit(
        X_train_xgb, y_train,
        eval_set=[(X_val_xgb, y_val)],
        # early_stopping_rounds=200,
        # verbose=200
    )
    xgb_val_proba = xgb_clf.predict_proba(X_val_xgb)[:, 1]
    xgb_test_proba = xgb_clf.predict_proba(X_test_xgb)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_val_proba)
    print(f'XGB AUC: {xgb_auc:.6f}')
    fold_auc_xgb.append(xgb_auc)
    ens_test_xgb.append(xgb_test_proba)
    
    # ----- CatBoost -----
    print("--- Training CatBoost ---")
    cat_features_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    
    cat_clf = cb.CatBoostClassifier(**cat_params)
    cat_clf.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features_indices,
        early_stopping_rounds=200,
        verbose=200,
        use_best_model=True
    )
    cat_val_proba = cat_clf.predict_proba(X_val)[:, 1]
    cat_test_proba = cat_clf.predict_proba(X_test)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_val_proba)
    print(f'Cat AUC: {cat_auc:.6f}')
    fold_auc_cat.append(cat_auc)
    ens_test_cat.append(cat_test_proba)
    
    # ----- 融合 -----
    ensemble_val_proba = lgb_val_proba * 0.4 + xgb_val_proba * 0.2 + cat_val_proba * 0.4
    ensemble_auc = roc_auc_score(y_val, ensemble_val_proba)
    print(f'Ensemble AUC: {ensemble_auc:.6f}')
    fold_auc_ensemble.append(ensemble_auc)

# ==================== 6. 结果汇总 ====================
print("\n" + "="*50)
print("Cross-Validation Results:")
print(f"LightGBM  : {np.mean(fold_auc_lgb):.6f} ± {np.std(fold_auc_lgb):.6f}")
print(f"XGBoost   : {np.mean(fold_auc_xgb):.6f} ± {np.std(fold_auc_xgb):.6f}")
print(f"CatBoost  : {np.mean(fold_auc_cat):.6f} ± {np.std(fold_auc_cat):.6f}")
print(f"Ensemble  : {np.mean(fold_auc_ensemble):.6f} ± {np.std(fold_auc_ensemble):.6f}")

# ==================== 7. 生成提交文件 ====================
mean_preds_lgb = np.mean(ens_test_lgb, axis=0)
mean_preds_xgb = np.mean(ens_test_xgb, axis=0)
mean_preds_cat = np.mean(ens_test_cat, axis=0)
final_preds = mean_preds_lgb * 0.4 + mean_preds_xgb * 0.2 + mean_preds_cat * 0.4

submission = pd.DataFrame({
    'user_id': test['user_id'],
    'merchant_id': test['merchant_id'],
    'prob': final_preds
})
submission.to_csv('data/2/submission.csv', index=False)
print("\nSubmission saved to submission.csv")