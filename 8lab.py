import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
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

CACHE_TRAIN_SET = "data/8/cache_train_set.pkl"
CACHE_TEST_SET = "data/8/cache_test_set.pkl"

def balance_train_set(train_set: pd.DataFrame, pos_neg_ratio=30):
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

# 0.0719 0.0909 0.0595(f1 recall precision)
def iwt_tabm():
    train_set = pd.read_pickle(CACHE_TRAIN_SET)
    test_set = pd.read_pickle(CACHE_TEST_SET)


    balanced_train = balance_train_set(train_set)

    feature_cols = [col for col in balanced_train.columns 
                        if col not in ['user_id', 'daystime', 'label', 'user_geohash', 'item_category', 'item_id']]
        
    print(f"使用特征数: {len(feature_cols)}")
    print(f"使用特征: {feature_cols}")

    X = balanced_train[feature_cols]
    y = balanced_train['label']
    X_test = test_set[feature_cols]
    print(f"X shape: {X.shape}") #(920607, 42)
    categorical_features = ['behavior_type', 'hours']

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
        s=28,
        gidx=gidx,
        lambda_param=0.1,
        tau=500.,
        sgidx=sgidx,
        strategy='B',
        equalsize=True,
        verbose=False,
        draw_loss=True,
        need_normalize=True,
    )
    iwt.fit(X_num.values, y.values)
    # 获取IWT筛选后保留的数值特征
    selected_numerical = X_num.columns[iwt.w_ == 0].tolist()

    # 最终特征 = 保留的数值特征 + 所有类别特征
    final_features = selected_numerical + categorical_features

    print(f"最终特征数量: {len(final_features)}")
    print(f"最终特征: {final_features}")

    X = balanced_train[final_features]
    X_test = test_set[final_features]
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
    test_set['score'] = test_pred
    result = test_set[['user_id', 'item_id', 'score']].copy()
    result['user_id'] = result['user_id'].astype(str)
    result['item_id'] = result['item_id'].astype(str)

    print(f"The len of result is: {len(result)}")
    result = result.sort_values('score', ascending=False).head(50000)

    result[['user_id', 'item_id']].to_csv("data/8/tianchi_prediction_1219.txt", sep='\t', index=False, header=False)
    print(f"预测结果已保存：data/8/tianchi_prediction_1219.txt")
    print(f"输出购买对数量：{len(result)}")

# 0.0716 0.0905 0.0593(f1 recall precision)
def lr_tabm():
    train_set = pd.read_pickle(CACHE_TRAIN_SET)
    test_set = pd.read_pickle(CACHE_TEST_SET)


    balanced_train = balance_train_set(train_set)

    feature_cols = [col for col in balanced_train.columns 
                        if col not in ['user_id', 'daystime', 'label', 'user_geohash', 'item_category', 'item_id']]
        
    print(f"使用特征数: {len(feature_cols)}")
    print(f"使用特征: {feature_cols}")

    X = balanced_train[feature_cols]
    y = balanced_train['label']
    X_test = test_set[feature_cols]
    print(f"X shape: {X.shape}") #(920607, 42)
    categorical_features = ['behavior_type', 'hours']

    X_num = X[[col for col in X.columns if col not in categorical_features]]

    lr_selector = SelectFromModel(
        estimator=LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
        ),
        max_features=28,
        threshold=-np.inf
    )
    
    lr_selector.fit(X_num, y)
    
    mask = lr_selector.get_support()
    selected_numerical = X_num.columns[mask].tolist()
    
    print(f"LR选择的数值特征数量: {len(selected_numerical)}")
    print(f"LR选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")
    # 最终特征 = 保留的数值特征 + 所有类别特征
    final_features = selected_numerical + categorical_features

    print(f"最终特征数量: {len(final_features)}")
    print(f"最终特征: {final_features}")

    X = balanced_train[final_features]
    X_test = test_set[final_features]
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
    test_set['score'] = test_pred
    result = test_set[['user_id', 'item_id', 'score']].copy()
    result['user_id'] = result['user_id'].astype(str)
    result['item_id'] = result['item_id'].astype(str)

    print(f"The len of result is: {len(result)}")
    result = result.sort_values('score', ascending=False).head(50000)

    result[['user_id', 'item_id']].to_csv("data/8/tianchi_prediction_1219_lr_tabm.txt", sep='\t', index=False, header=False)
    print(f"预测结果已保存：data/8/tianchi_prediction_1219_lr_tabm.txt")
    print(f"输出购买对数量：{len(result)}")

# 0.0705 0.0891 0.0583
def iwt_mlp_skl():
    train_set = pd.read_pickle(CACHE_TRAIN_SET)
    test_set = pd.read_pickle(CACHE_TEST_SET)


    balanced_train = balance_train_set(train_set)

    feature_cols = [col for col in balanced_train.columns 
                        if col not in ['user_id', 'daystime', 'label', 'user_geohash', 'item_category', 'item_id']]
        
    print(f"使用特征数: {len(feature_cols)}")
    print(f"使用特征: {feature_cols}")

    X = balanced_train[feature_cols]
    y = balanced_train['label']
    X_test = test_set[feature_cols]
    print(f"X shape: {X.shape}") #(920607, 42)
    categorical_features = ['behavior_type', 'hours']

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
        s=28,
        gidx=gidx,
        lambda_param=0.1,
        tau=500.,
        sgidx=sgidx,
        strategy='B',
        equalsize=True,
        verbose=False,
        draw_loss=True,
        need_normalize=True,
    )
    iwt.fit(X_num.values, y.values)
    # 获取IWT筛选后保留的数值特征
    selected_numerical = X_num.columns[iwt.w_ == 0].tolist()

    # 最终特征 = 保留的数值特征 + 所有类别特征
    final_features = selected_numerical + categorical_features

    print(f"最终特征数量: {len(final_features)}")
    print(f"最终特征: {final_features}")

    X = balanced_train[final_features]
    X_test = test_set[final_features]

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

    test_set['score'] = mean_preds
    result = test_set[['user_id', 'item_id', 'score']].copy()
    result['user_id'] = result['user_id'].astype(str)
    result['item_id'] = result['item_id'].astype(str)

    print(f"The len of result is: {len(result)}")
    result = result.sort_values('score', ascending=False).head(50000)

    result[['user_id', 'item_id']].to_csv("data/8/tianchi_prediction_1219_iwt_mlp_skl.txt", sep='\t', index=False, header=False)
    print(f"预测结果已保存：data/8/tianchi_prediction_1219_iwt_mlp_skl.txt")
    print(f"输出购买对数量：{len(result)}")

# 0.0705 0.0890 0.0583
def lr_mlp_skl():
    train_set = pd.read_pickle(CACHE_TRAIN_SET)
    test_set = pd.read_pickle(CACHE_TEST_SET)


    balanced_train = balance_train_set(train_set)

    feature_cols = [col for col in balanced_train.columns 
                        if col not in ['user_id', 'daystime', 'label', 'user_geohash', 'item_category', 'item_id']]
        
    print(f"使用特征数: {len(feature_cols)}")
    print(f"使用特征: {feature_cols}")

    X = balanced_train[feature_cols]
    y = balanced_train['label']
    X_test = test_set[feature_cols]
    print(f"X shape: {X.shape}") #(920607, 42)
    categorical_features = ['behavior_type', 'hours']

    X_num = X[[col for col in X.columns if col not in categorical_features]]

    lr_selector = SelectFromModel(
        estimator=LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
        ),
        max_features=28,
        threshold=-np.inf
    )
    
    lr_selector.fit(X_num, y)
    
    mask = lr_selector.get_support()
    selected_numerical = X_num.columns[mask].tolist()
    
    print(f"LR选择的数值特征数量: {len(selected_numerical)}")
    print(f"LR选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")

    # 最终特征 = 保留的数值特征 + 所有类别特征
    final_features = selected_numerical + categorical_features

    print(f"最终特征数量: {len(final_features)}")
    print(f"最终特征: {final_features}")

    X = balanced_train[final_features]
    X_test = test_set[final_features]

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

    test_set['score'] = mean_preds
    result = test_set[['user_id', 'item_id', 'score']].copy()
    result['user_id'] = result['user_id'].astype(str)
    result['item_id'] = result['item_id'].astype(str)

    print(f"The len of result is: {len(result)}")
    result = result.sort_values('score', ascending=False).head(50000)

    result[['user_id', 'item_id']].to_csv("data/8/tianchi_prediction_1219_lr_mlp_skl.txt", sep='\t', index=False, header=False)
    print(f"预测结果已保存：data/8/tianchi_prediction_1219_lr_mlp_skl.txt")
    print(f"输出购买对数量：{len(result)}")

@result_beep
def main():
    iwt_tabm()

if __name__ == '__main__':
    main()