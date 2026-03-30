import torch
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from pytabkit import (
    TabM_D_Classifier as TTabM,
)

from models.iwt_classifier import IWT_Classifier
from util import result_beep, log

NUM_FEATURES = 28
categorical_features = ['behavior_type', 'hours']

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

################################################################################

def _iwt_select(
        strategy: Literal['B', 'T', 'M'],
        mu: float,
        X: pd.DataFrame,
        y: pd.Series,
    ):
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

    print(f"总组数：{len(sgidx)}")

    gmi = torch.as_tensor([0.0010, 0.0004, 0.0046, 0.0043, 0.0290, 0.0029, 0.0049, 0.0024, 0.0031,
        0.0014, 0.0056, 0.0023, 0.0034, 0.0016, 0.0038, 0.0035, 0.0041, 0.0038,
        0.0057, 0.0044, 0.0052, 0.0043, 0.0057, 0.0055, 0.0049, 0.0044, 0.0047,
        0.0505, 0.0049, 0.0236, 0.0014, 0.0004, 0.0015, 0.0011, 0.0019, 0.0014,
        0.0086, 0.0067, 0.0009, 0.0277]).to(device)
    gmi = 0.1 * gmi

    iwt = IWT_Classifier(
        num_groups=len(sgidx),
        s=NUM_FEATURES,
        gidx=gidx,
        tau=500.,
        lambda_param=0.1,
        sgidx=sgidx,
        strategy=strategy,
        gmi=gmi,
        mu=mu,
        equalsize=True,
        verbose=False,
        draw_loss=True,
        need_normalize=True,
    )
    iwt.fit(X, y)

    return iwt.w_ == 0

def _lr_elastic_net_select(
        tfms,
        X: pd.DataFrame,
        y: pd.Series,
    ):
    est = make_pipeline(
        tfms,
        LogisticRegression(
            C=0.1,
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            random_state=42,
            max_iter=1000,
        )
    )

    print(f'est={est}')

    return SelectFromModel(
        estimator=est,
        max_features=NUM_FEATURES,
        threshold=-np.inf,
        importance_getter="named_steps.logisticregression.coef_"
    ).fit(X, y).get_support()

def _lr_l1_select(
        tfms,
        X: pd.DataFrame,
        y: pd.Series,
    ):
    est = make_pipeline(
        tfms,
        LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
        )
    )

    print(f'est={est}')

    return SelectFromModel(
        estimator=est,
        max_features=NUM_FEATURES,
        threshold=-np.inf,
        importance_getter="named_steps.logisticregression.coef_"
    ).fit(X, y).get_support()

def _rf_select(
        tfms,
        X: pd.DataFrame,
        y: pd.Series,
    ):
    est = make_pipeline(
        tfms,
        RandomForestClassifier(
            random_state=42,
            n_estimators=1000,
        )
    )
    print(f'est={est}')
    return SelectFromModel(
        estimator=est,
        max_features=NUM_FEATURES,
        threshold=-np.inf,
        importance_getter="named_steps.randomforestclassifier.feature_importances_"
    ).fit(X, y).get_support()

def _mi_select(
        tfms,
        X: pd.DataFrame,
        y: pd.Series,
    ):
    return SelectKBest(
        score_func=mutual_info_classif,
        k=NUM_FEATURES
    ).fit(X, y).get_support()

def selector_factory(
    X: pd.DataFrame,
    y: pd.Series,
    tfms,
    selector: Literal['iwt', 'lr_l1', 'lr_elastic_net', 'rf', 'mi'],
    strategy: Literal['B', 'T', 'M'] | None,
    mu: float | None,
):
    X_num = X[[col for col in X.columns if col not in categorical_features]]

    selector_func_dict = {
        'iwt': _iwt_select,
        'lr_l1': _lr_l1_select,
        'lr_elastic_net': _lr_elastic_net_select,
        'rf': _rf_select,
        'mi': _mi_select,
    }

    if selector not in selector_func_dict:
        raise ValueError(f"Unsupported selector: {selector}")
    selector_func = selector_func_dict[selector]
    if selector == 'iwt':
        if strategy != 'M':
            mu = 0.5 if mu is None else mu
        mask = selector_func(strategy, mu, X_num, y)
    else:
        mask = selector_func(tfms, X_num, y)
    selected_numerical = X_num.columns[mask].tolist()

    print(f"{selector}选择的数值特征数量: {len(selected_numerical)}")
    print(f"{selector}选择的数值特征: {selected_numerical}")
    print(f"被剔除的数值特征: {[col for col in X_num.columns if col not in selected_numerical]}")

    final_features = selected_numerical + categorical_features

    print(f"\n最终特征总数: {len(final_features)}")
    print(f"最终特征列表: {final_features}")

    return final_features

################################################################################

def train_factory(
    features: list[str],
    model_name: Literal['mlp', 'tabm'],
    use_larger_hidden_layer: bool,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
):
    X = X[features]
    X_test = X_test[features]

    if model_name == 'tabm':
        clf = TTabM(
            verbosity=2,
            random_state=42,
            val_metric_name='1-auc_ovr',
            n_cv=5,
        )
        clf.fit(X, y, cat_col_names=categorical_features)
        return clf.predict_proba(X_test)[:, 1]

    elif model_name == 'mlp':
        numerical_features = [col for col in features if col not in categorical_features]
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
                'hidden_layer_sizes': (512, 512, 512) if use_larger_hidden_layer else (100,),
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True,
                'verbose': True,
            }

            clf = Pipeline([
                ('preprocessor', preprocessor),
                ('mlp', MLPClassifier(**mlp_params))
            ])

            # 训练
            clf.fit(X_train, y_train)

            # 验证集评估
            val_proba = clf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            fold_aucs.append(val_auc)
            print(f'  Val AUC: {val_auc:.4f}')

            # 测试集预测
            test_proba = clf.predict_proba(X_test)[:, 1]
            ens_test.append(test_proba)
            print(f'  Test pred shape: {test_proba.shape}')

        # ========== 汇总结果 ==========
        print(f"\n========== 交叉验证汇总 ==========")
        print(f'每折 Val AUC: {[f"{auc:.4f}" for auc in fold_aucs]}')
        print(f'平均 Val AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs) * 2:.4f})')

        # 集成预测：5 折平均
        mean_preds = np.mean(ens_test, axis=0)
        print(f'集成预测 shape: {mean_preds.shape}')
        return mean_preds

    else:
        raise ValueError(f"Unsupported model: {model_name}")

@log(enable_file=True, file_path="8diff.log")
@result_beep
def main():
    CACHE_TRAIN_SET = "data/8/cache_train_set.pkl"
    CACHE_TEST_SET = "data/8/cache_test_set.pkl"
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

    selector = 'lr_elastic_net'
    strategy = None
    mu = None
    model_name = 'tabm'
    use_larger_hidden_layer = False
    tfms_dict = {
        'std': StandardScaler(),
        'quantile': QuantileTransformer(
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
        ),
        'none': None,
    }
    tfms = tfms_dict['std']

    features = selector_factory(
        X=X.copy(),
        y=y.copy(),
        tfms=tfms,
        selector=selector,
        strategy=strategy,
        mu=mu,
    )

    test_pred = train_factory(
        features=features,
        model_name=model_name,
        use_larger_hidden_layer=use_larger_hidden_layer,
        X=X.copy(),
        y=y.copy(),
        X_test=X_test.copy(),
    )
    test_set['score'] = test_pred
    result = test_set[['user_id', 'item_id', 'score']].copy()
    result['user_id'] = result['user_id'].astype(str)
    result['item_id'] = result['item_id'].astype(str)
    result = result.sort_values('score', ascending=False).head(50000)
    result[['user_id', 'item_id']].to_csv(f'data/8/submission_{selector}_{strategy}_{mu}_{tfms.__class__.__name__ if tfms is not None else 'None'}_{model_name}_{use_larger_hidden_layer}.txt', sep='\t', index=False, header=False)
    print("submission saved")

if __name__ == "__main__":
    main()
