import gc
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pytabkit import TabM_D_Classifier

from util import result_beep

def process_data(expose_size: float = 1/2):
    user_log = pd.read_csv('data/2/user_log_format1.csv')
    user_info = pd.read_csv('data/2/user_info_format1.csv')
    train = pd.read_csv('data/2/train_format1.csv')
    test = pd.read_csv('data/2/test_format1.csv')

    user_log.rename(columns={"seller_id": "merchant_id"}, inplace=True)

    user_info["age_range"].fillna(0, inplace=True)
    user_info["gender"].fillna(2, inplace=True)
    user_log["brand_id"].fillna(0, inplace=True)

    train_main, train_ratio = train_test_split(
        train,
        test_size=expose_size,
        random_state=42,
        stratify=train['label']
    )

    del train
    gc.collect()

    train_main["origin"] = "train"
    test["origin"] = "test"
    data = pd.concat([train_main, test], sort=False)
    data = data.drop(["prob"], axis=1)

    del train_main, test
    gc.collect()

    merchant_label_ratio = train_ratio.groupby('merchant_id')['label'].mean().reset_index().rename(
        columns={'label': 'merchant_label_ratio'})
    data = data.merge(merchant_label_ratio, on='merchant_id', how='left')

    item_label_ratio_max = pd.merge(user_log[(user_log['action_type'] == 2)], train_ratio,
                                    how='inner').drop_duplicates()
    item_label_ratio_mean = item_label_ratio_max.groupby(['item_id'])['label'].mean().reset_index().rename(
        columns={'label': 'item_label_ratio'})
    item_label_ratio_max = pd.merge(item_label_ratio_max, item_label_ratio_mean, how='left')
    del item_label_ratio_mean
    gc.collect()
    item_label_ratio_max = item_label_ratio_max.groupby(['merchant_id'])['item_label_ratio'].max().reset_index()

    cat_label_ratio_max = pd.merge(user_log[(user_log['action_type'] == 2)], train_ratio, how='inner').drop_duplicates()
    cat_label_ratio_mean = cat_label_ratio_max.groupby(['cat_id'])['label'].mean().reset_index().rename(
        columns={'label': 'cat_label_ratio'})
    cat_label_ratio_max = pd.merge(cat_label_ratio_max, cat_label_ratio_mean, how='inner')
    del cat_label_ratio_mean
    gc.collect()
    cat_label_ratio_max = cat_label_ratio_max.groupby(['merchant_id'])['cat_label_ratio'].max().reset_index()

    brand_label_ratio_max = pd.merge(user_log[(user_log['action_type'] == 2)], train_ratio,
                                     how='inner').drop_duplicates().dropna()
    brand_label_ratio_mean = brand_label_ratio_max.groupby(['brand_id'])['label'].mean().reset_index().rename(
        columns={'label': 'brand_label_ratio'})
    brand_label_ratio_max = pd.merge(brand_label_ratio_max, brand_label_ratio_mean, how='left')
    del brand_label_ratio_mean
    gc.collect()
    brand_label_ratio_max = brand_label_ratio_max.groupby(['merchant_id'])['brand_label_ratio'].max().reset_index()

    data = data.merge(item_label_ratio_max, on='merchant_id', how='left')
    data = data.merge(cat_label_ratio_max, on='merchant_id', how='left')
    data = data.merge(brand_label_ratio_max, on='merchant_id', how='left')

    del train_ratio, merchant_label_ratio, item_label_ratio_max, cat_label_ratio_max, brand_label_ratio_max
    gc.collect()

    # 性别、年龄独热编码处理
    data = data.merge(user_info, on="user_id", how="left")

    temp = pd.get_dummies(data["age_range"], prefix="age", dtype='int32')
    temp2 = pd.get_dummies(data["gender"], prefix="gender", dtype='int32')

    data = pd.concat([data, temp, temp2], axis=1)
    data.drop(columns=["age_range", "gender"], inplace=True)

    del temp, temp2
    gc.collect()

    # 按user_id,merchant_id分组，购买天数>1则复购标记为1，反之为0
    groups_rb = user_log[user_log["action_type"] == 2].groupby(["user_id", "merchant_id"])
    temp_rb = groups_rb.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "n_days"})
    temp_rb["label_um"] = [(1 if x > 1 else 0) for x in temp_rb["n_days"]]

    groups = user_log.groupby(["user_id"])
    # 日志数
    temp = groups.size().reset_index().rename(columns={0: "count_u"})
    data = pd.merge(data, temp, on="user_id", how="left")

    # 交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "days_u"})
    data = data.merge(temp, on="user_id", how="left")

    # 访问商品，品类，品牌，商家数
    temp = groups[['item_id', 'cat_id', 'merchant_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'item_count_u', 'cat_id': 'cat_count_u', 'merchant_id': 'merchant_count_u',
        'brand_id': 'brand_count_u'})
    data = data.merge(temp, on="user_id", how="left")

    # 各行为类型次数
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
        0: 'view_count_u', 1: 'cart_count_u', 2: 'buy_count_u', 3: 'fav_count_u'})
    data = data.merge(temp, on="user_id", how="left")

    # 行为权重
    data['action_weight_u'] = (
                data['view_count_u'] * 0.1 + data['cart_count_u'] * 0.2 + data['fav_count_u'] * 0.2 + data[
            'buy_count_u'] * 0.5)

    # 统计购买点击比
    data["buy_view_ratio_u"] = data["buy_count_u"] / data["view_count_u"]
    # 统计购买收藏比
    data['buy_fav_ratio_u'] = data['buy_count_u'] / data['fav_count_u']
    # 统计购买加购比
    data['buy_cart_ratio_u'] = data['buy_count_u'] / data['cart_count_u']
    # 购买频率
    data['buy_freq_u'] = data['buy_count_u'] / data['count_u']

    # 复购率 = 复购过的商家数/购买过的总商家数
    temp = temp_rb.groupby(["user_id", "label_um"]).size().unstack(fill_value=0).reset_index()
    temp["repurchase_rate_u"] = temp[1] / (temp[0] + temp[1])
    data = data.merge(temp[["user_id", "repurchase_rate_u"]], on="user_id", how="left")

    # 购买量/购买商家数
    temp = user_log[user_log['action_type'] == 2].groupby(['user_id']).merchant_id.nunique().reset_index().rename(
        columns={'merchant_id': 'merchant_buy_count'})
    data = data.merge(temp, on='user_id', how='left')
    data['loyal_u'] = data['buy_count_u'] / data['merchant_buy_count']

    groups = user_log.groupby(["merchant_id"])

    # 日志数
    temp = groups.size().reset_index().rename(columns={0: "count_m"})
    data = data.merge(temp, on="merchant_id", how="left")

    # 交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "days_m"})
    data = data.merge(temp, on="merchant_id", how="left")

    # 访问商品，品类，品牌，用户数
    temp = groups[['item_id', 'cat_id', 'user_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'item_count_m', 'cat_id': 'cat_count_m', 'user_id': 'user_count_m', 'brand_id': 'brand_count_m'})
    data = data.merge(temp, on="merchant_id", how="left")

    # 各行为类型次数
    temp = groups.action_type.value_counts().unstack().reset_index().rename(columns={
        0: 'view_count_m', 1: 'cart_count_m', 2: 'buy_count_m', 3: 'fav_count_m'})
    data = data.merge(temp, on="merchant_id", how="left")

    # 行为权重
    data['action_weight_m'] = (
                data['view_count_m'] * 0.1 + data['cart_count_m'] * 0.2 + data['fav_count_m'] * 0.2 + data[
            'buy_count_m'] * 0.5)

    # 统计购买点击比
    data["buy_view_ratio_m"] = data["buy_count_m"] / data["view_count_m"]
    # 统计购买收藏比
    data['buy_fav_ratio_m'] = data['buy_count_m'] / data['fav_count_m']
    # 统计购买加购比
    data['buy_cart_ratio_m'] = data['buy_count_m'] / data['cart_count_m']
    # 购买频率
    data['buy_freq_m'] = data['buy_count_m'] / data['count_m']

    # 复购率
    temp = temp_rb.groupby(["merchant_id", "label_um"]).size().unstack(fill_value=0).reset_index()
    temp["repurchase_rate_m"] = temp[1] / (temp[0] + temp[1])
    data = data.merge(temp[["merchant_id", "repurchase_rate_m"]], on="merchant_id", how="left")

    # 购买量/购买用户数
    temp = user_log[user_log['action_type'] == 2].groupby(['merchant_id']).user_id.nunique().reset_index().rename(
        columns={'user_id': 'user_buy_count'})
    data = data.merge(temp, on='merchant_id', how='left')
    data['loyal_m'] = data['buy_count_m'] / data['user_buy_count']

    groups = user_log.groupby(['user_id', 'merchant_id'])

    # 日志数
    temp = groups.size().reset_index().rename(columns={0: "count_um"})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "days_um"})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 访问商品，品类，品牌数
    temp = groups[['item_id', 'cat_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'item_count_um', 'cat_id': 'cat_count_um', 'brand_id': 'brand_count_um'})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 各行为类型次数
    temp = groups.action_type.value_counts().unstack().reset_index().rename(columns={
        0: 'view_count_um', 1: 'cart_count_um', 2: 'buy_count_um', 3: 'fav_count_um'})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 行为权重
    data['action_weight_um'] = (
                data['view_count_um'] * 0.1 + data['cart_count_um'] * 0.2 + data['fav_count_um'] * 0.2 + data[
            'buy_count_um'] * 0.5)
    # 统计购买点击比
    data["buy_view_ratio_um"] = data["buy_count_um"] / data["view_count_um"]
    # 统计购买收藏比
    data['buy_fav_ratio_um'] = data['buy_count_um'] / data['fav_count_um']
    # 统计购买加购比
    data['buy_cart_ratio_um'] = data['buy_count_um'] / data['cart_count_um']
    # 购买频率
    data['buy_freq_um'] = data['buy_count_um'] / data['count_um']

    # 交互点击比
    data['view_ratio'] = data['view_count_um'] / data['view_count_u']
    # 交互加购比
    data['cart_ratio'] = data['cart_count_um'] / data['cart_count_u']
    # 交互收藏比
    data['fav_ratio'] = data['fav_count_um'] / data['fav_count_u']
    # 交互购买比
    data['buy_ratio'] = data['buy_count_um'] / data['buy_count_u']

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = data[numerical_cols].fillna(0)

    gc.collect()
    data.to_csv("data/2/features.csv", index=False)

def train_model():
    if not os.path.exists('data/2/features.csv'):
        process_data()

    data = pd.read_csv(f'data/2/features.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, Y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])

    model = TabM_D_Classifier(
        n_cv=10,
        verbosity=2,
        val_metric_name='1-auc_ovr',
        arch_type='tabm-mini',
        dropout=0.0,
        lr=0.0001334456483981059,
        weight_decay=0.012340787659576354,
        n_blocks=5,
        d_block=1024,
        num_emb_type='pwl',
        d_embedding=32,
        num_emb_n_bins=2,
    )
    # auc=0.6911001967503175 pytabkit-momoAdamW
    # auc=0.7020404973897176 pytabkit-adamW
    # auc=0.6940894141366357 myTabM-adamW
    # auc=0.6919860979491992 myTabM-momoAdamW
    model.fit(X, Y)
    # , cat_col_names=['age_0.0','age_1.0','age_2.0','age_3.0','age_4.0','age_5.0','age_6.0','age_7.0','age_8.0','gender_0.0','gender_1.0','gender_2.0']

    test_pred = model.predict_proba(X_test)[:, 1]

    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv('data/2/submission.csv', index=False)
    print('Submission csv saved!')

@result_beep
def main():
    train_model()

if __name__ == '__main__':
    main()

"""
tab-mini weight[16  0  8  3  4  0  1  2  0  0] 0.6959399114348 5k 10steps

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0001334456483981059), 'weight_decay': np.float64(0.012340787659576354), 'n_blocks': np.int64(5), 'd_block': np.int64(912), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(32), 'num_emb_n_bins': np.int64(2)} 10k 0.6956

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.00021816266197247693), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(3), 'd_block': np.int64(640), 'dropout': np.float64(0.33055586392588004), 'num_emb_type': 'pwl', 'd_embedding': np.int64(32), 'num_emb_n_bins': np.int64(26)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.00022029015158645592), 'weight_decay': np.float64(0.0030179664067750372), 'n_blocks': np.int64(3), 'd_block': np.int64(432), 'dropout': np.float64(0.3544382638112154), 'num_emb_type': 'pwl', 'd_embedding': np.int64(28), 'num_emb_n_bins': np.int64(127)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.00013795505877650676), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(3), 'd_block': np.int64(864), 'dropout': np.float64(0.08366780807852209), 'num_emb_type': 'pwl', 'd_embedding': np.int64(12), 'num_emb_n_bins': np.int64(5)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.001048334934585524), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(5), 'd_block': np.int64(432), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(12), 'num_emb_n_bins': np.int64(114)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0006668459082109882), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(3), 'd_block': np.int64(192), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(28), 'num_emb_n_bins': np.int64(121)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0003276095002985988), 'weight_decay': np.float64(0.00431990661862382), 'n_blocks': np.int64(2), 'd_block': np.int64(928), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(32), 'num_emb_n_bins': np.int64(57)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.001492706280794318), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(4), 'd_block': np.int64(816), 'dropout': np.float64(0.4826120323148692), 'num_emb_type': 'pwl', 'd_embedding': np.int64(8), 'num_emb_n_bins': np.int64(82)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.002515274885156232), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(2), 'd_block': np.int64(864), 'dropout': np.float64(0.10326996792916132), 'num_emb_type': 'pwl', 'd_embedding': np.int64(8), 'num_emb_n_bins': np.int64(52)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm-mini', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0014449145634601243), 'weight_decay': np.float64(0.0009636806325505294), 'n_blocks': np.int64(4), 'd_block': np.int64(448), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(16), 'num_emb_n_bins': np.int64(78)}

=======================================================================================================================================

tabm weight[1 2 0 3 0 0 8 8 0 0] 0.6941 10steps

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.00013754583614053662), 'weight_decay': np.float64(0.0015053977006493477), 'n_blocks': np.int64(2), 'd_block': np.int64(512), 'dropout': np.float64(0.21324884955756607), 'num_emb_type': 'pwl', 'd_embedding': np.int64(8), 'num_emb_n_bins': np.int64(58)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0003419510923533141), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(5), 'd_block': np.int64(928), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(32), 'num_emb_n_bins': np.int64(10)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0015543461230552596), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(3), 'd_block': np.int64(272), 'dropout': np.float64(0.291544993424212), 'num_emb_type': 'pwl', 'd_embedding': np.int64(8), 'num_emb_n_bins': np.int64(34)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0018594110485356079), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(5), 'd_block': np.int64(432), 'dropout': np.float64(0.09514423012628914), 'num_emb_type': 'pwl', 'd_embedding': np.int64(28), 'num_emb_n_bins': np.int64(125)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0001249171880339904), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(4), 'd_block': np.int64(704), 'dropout': np.float64(0.2808460055312869), 'num_emb_type': 'pwl', 'd_embedding': np.int64(32), 'num_emb_n_bins': np.int64(32)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0024293963603361076), 'weight_decay': np.float64(0.013370890566153053), 'n_blocks': np.int64(3), 'd_block': np.int64(928), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(20), 'num_emb_n_bins': np.int64(97)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.001427730613920002), 'weight_decay': np.float64(0.00039996334669144577), 'n_blocks': np.int64(3), 'd_block': np.int64(1024), 'dropout': np.float64(0.1759765320133561), 'num_emb_type': 'pwl', 'd_embedding': np.int64(8), 'num_emb_n_bins': np.int64(23)}

*self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.00017803074484857816), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(4), 'd_block': np.int64(896), 'dropout': np.float64(0.07276177983723575), 'num_emb_type': 'pwl', 'd_embedding': np.int64(28), 'num_emb_n_bins': np.int64(14)}

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.0008480921904553091), 'weight_decay': np.float64(0.0), 'n_blocks': np.int64(3), 'd_block': np.int64(176), 'dropout': np.float64(0.14206798398187792), 'num_emb_type': 'pwl', 'd_embedding': np.int64(32), 'num_emb_n_bins': np.int64(20)} 10k 0.6950

self.fit_params[0]={'batch_size': 'auto', 'patience': 16, 'allow_amp': False, 'arch_type': 'tabm', 'tabm_k': 32, 'share_training_batches': False, 'lr': np.float64(0.002099242883105876), 'weight_decay': np.float64(0.002385878732091997), 'n_blocks': np.int64(3), 'd_block': np.int64(848), 'dropout': np.float64(0.0), 'num_emb_type': 'pwl', 'd_embedding': np.int64(20), 'num_emb_n_bins': np.int64(109)}
"""