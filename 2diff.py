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

NUM_FEATURES = 47
categorical_features = ['age_range', 'gender']

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

    gmi = torch.as_tensor([9.3414e-03, 1.6661e-02, 5.6912e-03, 8.8767e-03, 1.7678e-03, 2.3599e-03,
        8.5237e-05, 6.0632e-04, 6.4988e-04, 2.7677e-04, 1.1144e-03, 2.5794e-04,
        2.5120e-03, 1.8366e-03, 1.9747e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        5.0511e-04, 9.9226e-04, 2.0158e-03, 4.7884e-03, 9.6173e-03, 7.8607e-03,
        6.7476e-03, 1.4207e-03, 1.0243e-02, 2.9496e-03, 1.0073e-02, 4.2554e-03,
        8.6148e-03, 9.0128e-03, 1.0113e-02, 1.0821e-02, 1.2121e-02, 1.1187e-02,
        1.1216e-02, 1.0863e-02, 8.0682e-03, 9.9338e-03, 3.8083e-03, 1.0412e-02,
        5.5326e-03, 1.3963e-02, 2.8088e-02, 3.4663e-03, 0.0000e+00, 1.8292e-02,
        1.3881e-03, 1.5205e-04, 4.1286e-03, 1.1561e-03, 1.0588e-03, 5.1074e-03,
        1.0501e-03, 0.0000e+00, 0.0000e+00, 4.2081e-03]).to(device)

    iwt = IWT_Classifier(
        num_groups=len(sgidx),
        s=NUM_FEATURES,
        gidx=gidx,
        tau=1000.,
        lambda_param=1.0,
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

@log(enable_file=True, file_path="2diff.log")
@result_beep
def main():
    data = pd.read_csv(f'data/2/features_cat.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])

    selector = 'lr_elastic_net'
    strategy = None
    mu = None
    model_name = 'mlp'
    use_larger_hidden_layer = True
    tfms_dict = {
        'std': StandardScaler(),
        'quantile': QuantileTransformer(
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            output_distribution='normal',
            subsample=10**9,
        ),
        'none': None,
    }
    tfms = tfms_dict['quantile']

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
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv(f'data/2/submission_{selector}_{strategy}_{mu}_{tfms.__class__.__name__ if tfms is not None else 'None'}_{model_name}_{use_larger_hidden_layer}.csv', index=False)
    print("submission saved")

if __name__ == "__main__":
    main()
