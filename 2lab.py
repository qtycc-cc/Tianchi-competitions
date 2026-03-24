import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from pytabkit import (
    TabM_D_Classifier as TTabM,
)

from models.iwt_classifier import IWT_Classifier
from util import result_beep
# ===============tabm=================
# ['merchant_label_ratio', 'item_label_ratio', 'cat_label_ratio', 'brand_label_ratio', 'days_u', 'cat_count_u', 'merchant_count_u', 'brand_count_u', 'view_count_u', 'cart_count_u', 'buy_count_u', 'fav_count_u', 'buy_fav_ratio_u', 'buy_freq_u', 'repurchase_rate_u', 'merchant_buy_count', 'loyal_u', 'count_m', 'days_m', 'item_count_m', 'user_count_m', 'brand_count_m', 'view_count_m', 'cart_count_m', 'buy_count_m', 'fav_count_m', 'action_weight_m', 'buy_view_ratio_m', 'buy_fav_ratio_m', 'buy_cart_ratio_m', 'buy_freq_m', 'repurchase_rate_m', 'loyal_m', 'count_um', 'days_um', 'item_count_um', 'cat_count_um', 'brand_count_um', 'view_count_um', 'buy_count_um', 'fav_count_um', 'buy_view_ratio_um', 'buy_fav_ratio_um', 'buy_cart_ratio_um', 'buy_freq_um', 'view_ratio', 'buy_ratio', 'age_range', 'gender']
# 0.6946730
def iwt_tabm():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])
    print(f"X shape: {X.shape}") #(130432, 60)
    print(f"X columns: {X.columns.to_list()}")

    categorical_features = ['age_range', 'gender']
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_groups = X_num.shape[1]
    num_features = X_num.shape[1]
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

    print(f"总组数：{len(sgidx)}")

    iwt = IWT_Classifier(
        num_groups=len(sgidx),
        s=47,
        gidx=gidx,
        tau=100.,
        lambda_param=1.0,
        sgidx=sgidx,
        strategy='B',
        equalsize=True,
        verbose=False,
        draw_loss=True,
        need_normalize=True,
    )
    iwt.fit(X_num.values, y.values)

    selected_numerical = X_num.columns[iwt.w_ == 0].tolist()

    final_features = selected_numerical + categorical_features

    print(f"最终特征数量: {len(final_features)}")
    print(f"最终特征: {final_features}")

    X = X[final_features]
    X_test = X_test[final_features]
    for col in categorical_features:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    model = TTabM(
        verbosity=2,
        random_state=42,
        val_metric_name='1-auc_ovr',
        n_cv=5,
    )
    model.fit(X, y)
    test_pred = model.predict_proba(X_test)[:, 1]
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv('data/2/submission_iwt_tabm.csv', index=False)
    print('Submission csv saved!')

# 0.6936018
def lr_tabm():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])
    print(f"X shape: {X.shape}") #(130432, 60)
    print(f"X columns: {X.columns.to_list()}")

    categorical_features = ['age_range', 'gender']
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    lr_selector = SelectFromModel(
        estimator=LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
        ),
        max_features=47,
        threshold=-np.inf
    )
    
    lr_selector.fit(X_num, y)
    
    mask = lr_selector.get_support()
    selected_numerical = X_num.columns[mask].tolist()
    
    print(f"LR选择的数值特征数量: {len(selected_numerical)}")
    print(f"LR选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")

    final_features = selected_numerical + categorical_features

    print(f"\n最终特征总数: {len(final_features)}")
    print(f"最终特征列表: {final_features}")

    X = X[final_features]
    X_test = X_test[final_features]

    for col in categorical_features:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    model = TTabM(
        verbosity=2,
        random_state=42,
        val_metric_name='1-auc_ovr',
        n_cv=5,
    )
    model.fit(X, y)

    test_pred = model.predict_proba(X_test)[:, 1]
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv('data/2/submission_lr_tabm.csv', index=False)
    print('Submission csv saved!')

# 0.6928687
def lasso_tabm():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])
    print(f"X shape: {X.shape}") #(130432, 60)
    print(f"X columns: {X.columns.to_list()}")

    categorical_features = ['age_range', 'gender']
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    lr_selector = SelectFromModel(
        estimator=Lasso(),
        max_features=47,
        threshold=-np.inf
    )
    
    lr_selector.fit(X_num, y)
    
    mask = lr_selector.get_support()
    selected_numerical = X_num.columns[mask].tolist()
    
    print(f"Lasso选择的数值特征数量: {len(selected_numerical)}")
    print(f"Lasso选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")

    final_features = selected_numerical + categorical_features

    print(f"\n最终特征总数: {len(final_features)}")
    print(f"最终特征列表: {final_features}")

    X = X[final_features]
    X_test = X_test[final_features]

    for col in categorical_features:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    model = TTabM(
        verbosity=2,
        random_state=42,
        val_metric_name='1-auc_ovr',
        n_cv=5,
    )
    model.fit(X, y)

    test_pred = model.predict_proba(X_test)[:, 1]
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv('data/2/submission_lasso_tabm.csv', index=False)
    print('Submission csv saved!')

# ===============skl_mlp=================
# 0.6881
def iwt_mlp_skl():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])
    
    print(f"X shape: {X.shape}")  
    print(f"X columns: {X.columns.to_list()}")

    categorical_features = ['age_range', 'gender']
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    # ========== IWT 特征选择部分（保持不变） ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_groups = X_num.shape[1]
    num_features = X_num.shape[1]
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

    print(f"总组数：{len(sgidx)}")

    iwt = IWT_Classifier(
        num_groups=len(sgidx),
        s=47,
        gidx=gidx,
        tau=100.,
        lambda_param=1.0,
        sgidx=sgidx,
        strategy='B',
        equalsize=True,
        verbose=False,
        draw_loss=True,
        need_normalize=True,
    )
    iwt.fit(X_num.values, y.values)

    # 获取选择的特征
    selected_numerical = X_num.columns[iwt.w_ == 0].tolist()
    final_features = selected_numerical + categorical_features

    print(f"最终特征数量: {len(final_features)}")
    print(f"最终特征: {final_features}")

    # 筛选特征
    X = X[final_features].copy()
    X_test = X_test[final_features].copy()
    
    # 更新数值特征列表
    numerical_features = [col for col in final_features if col not in categorical_features]
    print(f"数值特征数: {len(numerical_features)}, 分类特征数: {len(categorical_features)}")

    # ========== 5 折交叉验证（类似 LightGBM 风格） ==========
    print("\n========== 开始 5 折交叉验证 ==========")
    
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    ens_test = []  # 存储每折对测试集的预测
    fold_aucs = []  # 存储每折验证集 AUC

    for fold, (train_ix, val_ix) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold + 1}/5')
        
        # 划分数据
        X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
        y_train, y_val = y.iloc[train_ix], y.iloc[val_ix]
        
        # 构建预处理（每折单独 fit，防止数据泄漏）
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # 构建 Pipeline
        mlp_params = {
            'hidden_layer_sizes': (512, 512, 512),
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
        }
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('mlp', MLPClassifier(**mlp_params))
        ])
        
        # 训练
        model.fit(X_train, y_train)
        
        # 验证集评估
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        fold_aucs.append(val_auc)
        print(f'  Val AUC: {val_auc:.4f}')
        
        # 测试集预测
        test_proba = model.predict_proba(X_test)[:, 1]
        ens_test.append(test_proba)
        print(f'  Test pred shape: {test_proba.shape}')

    # ========== 汇总结果 ==========
    print(f"\n========== 交叉验证汇总 ==========")
    print(f'每折 Val AUC: {[f"{auc:.4f}" for auc in fold_aucs]}')
    print(f'平均 Val AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs) * 2:.4f})')

    # 集成预测：5 折平均
    mean_preds = np.mean(ens_test, axis=0)
    print(f'集成预测 shape: {mean_preds.shape}')

    # ========== 生成提交文件 ==========
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = mean_preds
    submission.to_csv('data/2/submission_iwt_mlp_skl.csv', index=False)
    print(f'\nSubmission csv saved! 共 {len(submission)} 条预测结果')
    print(f'预测概率范围: [{mean_preds.min():.4f}, {mean_preds.max():.4f}]')

# 0.6868
def lr_mlp_skl():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])
    print(f"X shape: {X.shape}") #(130432, 60)
    print(f"X columns: {X.columns.to_list()}")

    categorical_features = ['age_range', 'gender']
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    lr_selector = SelectFromModel(
        estimator=LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
        ),
        max_features=47,
        threshold=-np.inf
    )
    
    lr_selector.fit(X_num, y)
    
    mask = lr_selector.get_support()
    selected_numerical = X_num.columns[mask].tolist()
    
    print(f"LR选择的数值特征数量: {len(selected_numerical)}")
    print(f"LR选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")

    final_features = selected_numerical + categorical_features

    print(f"\n最终特征总数: {len(final_features)}")
    print(f"最终特征列表: {final_features}")

    # 筛选特征
    X = X[final_features].copy()
    X_test = X_test[final_features].copy()
    
    # 更新数值特征列表
    numerical_features = [col for col in final_features if col not in categorical_features]
    print(f"数值特征数: {len(numerical_features)}, 分类特征数: {len(categorical_features)}")

    # ========== 5 折交叉验证（类似 LightGBM 风格） ==========
    print("\n========== 开始 5 折交叉验证 ==========")
    
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    ens_test = []  # 存储每折对测试集的预测
    fold_aucs = []  # 存储每折验证集 AUC

    for fold, (train_ix, val_ix) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold + 1}/5')
        
        # 划分数据
        X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
        y_train, y_val = y.iloc[train_ix], y.iloc[val_ix]
        
        # 构建预处理（每折单独 fit，防止数据泄漏）
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # 构建 Pipeline
        mlp_params = {
            'hidden_layer_sizes': (512, 512, 512),
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
        }
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('mlp', MLPClassifier(**mlp_params))
        ])
        
        # 训练
        model.fit(X_train, y_train)
        
        # 验证集评估
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        fold_aucs.append(val_auc)
        print(f'  Val AUC: {val_auc:.4f}')
        
        # 测试集预测
        test_proba = model.predict_proba(X_test)[:, 1]
        ens_test.append(test_proba)
        print(f'  Test pred shape: {test_proba.shape}')

    # ========== 汇总结果 ==========
    print(f"\n========== 交叉验证汇总 ==========")
    print(f'每折 Val AUC: {[f"{auc:.4f}" for auc in fold_aucs]}')
    print(f'平均 Val AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs) * 2:.4f})')

    # 集成预测：5 折平均
    mean_preds = np.mean(ens_test, axis=0)
    print(f'集成预测 shape: {mean_preds.shape}')

    # ========== 生成提交文件 ==========
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = mean_preds
    submission.to_csv('data/2/submission_lr_mlp_skl.csv', index=False)
    print(f'\nSubmission csv saved! 共 {len(submission)} 条预测结果')
    print(f'预测概率范围: [{mean_preds.min():.4f}, {mean_preds.max():.4f}]')

# 0.6865
def lasso_mlp_skl():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])
    print(f"X shape: {X.shape}") #(130432, 60)
    print(f"X columns: {X.columns.to_list()}")

    categorical_features = ['age_range', 'gender']
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    lr_selector = SelectFromModel(
        estimator=Lasso(random_state=42),
        max_features=47,
        threshold=-np.inf
    )
    
    lr_selector.fit(X_num, y)
    
    mask = lr_selector.get_support()
    selected_numerical = X_num.columns[mask].tolist()
    
    print(f"Lasso选择的数值特征数量: {len(selected_numerical)}")
    print(f"Lasso选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")

    final_features = selected_numerical + categorical_features

    print(f"\n最终特征总数: {len(final_features)}")
    print(f"最终特征列表: {final_features}")

    # 筛选特征
    X = X[final_features].copy()
    X_test = X_test[final_features].copy()
    
    # 更新数值特征列表
    numerical_features = [col for col in final_features if col not in categorical_features]
    print(f"数值特征数: {len(numerical_features)}, 分类特征数: {len(categorical_features)}")

    # ========== 5 折交叉验证（类似 LightGBM 风格） ==========
    print("\n========== 开始 5 折交叉验证 ==========")
    
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    ens_test = []  # 存储每折对测试集的预测
    fold_aucs = []  # 存储每折验证集 AUC

    for fold, (train_ix, val_ix) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold + 1}/5')
        
        # 划分数据
        X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
        y_train, y_val = y.iloc[train_ix], y.iloc[val_ix]
        
        # 构建预处理（每折单独 fit，防止数据泄漏）
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # 构建 Pipeline
        mlp_params = {
            'hidden_layer_sizes': (512, 512, 512),
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
        }
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('mlp', MLPClassifier(**mlp_params))
        ])
        
        # 训练
        model.fit(X_train, y_train)
        
        # 验证集评估
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        fold_aucs.append(val_auc)
        print(f'  Val AUC: {val_auc:.4f}')
        
        # 测试集预测
        test_proba = model.predict_proba(X_test)[:, 1]
        ens_test.append(test_proba)
        print(f'  Test pred shape: {test_proba.shape}')

    # ========== 汇总结果 ==========
    print(f"\n========== 交叉验证汇总 ==========")
    print(f'每折 Val AUC: {[f"{auc:.4f}" for auc in fold_aucs]}')
    print(f'平均 Val AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs) * 2:.4f})')

    # 集成预测：5 折平均
    mean_preds = np.mean(ens_test, axis=0)
    print(f'集成预测 shape: {mean_preds.shape}')

    # ========== 生成提交文件 ==========
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = mean_preds
    submission.to_csv('data/2/submission_lasso_mlp_skl.csv', index=False)
    print(f'\nSubmission csv saved! 共 {len(submission)} 条预测结果')
    print(f'预测概率范围: [{mean_preds.min():.4f}, {mean_preds.max():.4f}]')

@result_beep
def main():
    iwt_tabm()

if __name__ == '__main__':
    main()