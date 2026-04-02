# import os

# os.system('cls')

"""
game7
cat=['is_weekend'] robust_scale
n_cv=1 0.16583
n_cv=5 0.16573
"""
l = ['irank', 'month', 'day', 'hour', 'dayofweek', 'dayofyear', 'item_price', 'days_u', 'item_count_u', 'cate_count_u', 'store_count_u', 'item_price_first', 'item_price_mean', 'item_price_min', 'item_price_max', 'item_price_first_store', 'item_price_min_store', 'item_price_max_store', 'item_id_freq', 'store_id_freq', 'is_weekend']

"""
game7
cat=['is_weekend', 'month', 'day', 'hour', 'dayofyear', 'dayofweek'] kdi dropout=0.5 d_block=256
n_cv=1 0.16548
"""
l1 = ['irank', 'item_price', 'days_u', 'item_count_u', 'cate_count_u', 'store_count_u', 'item_price_first', 'item_price_mean', 'item_price_min', 'item_price_first_store', 'item_price_mean_store', 'item_price_min_store', 'item_price_max_store', 'item_id_freq', 'store_id_freq', 'is_weekend', 'month', 'day', 'hour', 'dayofyear', 'dayofweek']

"""
game7
cat=['is_weekend'] robust_scale
n_cv=1 0.16557
n_cv=5 0.16566
================================
cat=['is_weekend', 'dayofweek'] robust_scale
n_cv=1 0.16574
n_cv=5 0.16546
================================
cat=['is_weekend', 'month', 'day', 'hour', 'dayofyear', 'dayofweek'] kdi dropout=0.5 d_block=256
n_cv=5 0.16563
"""
all = ['irank', 'month', 'day', 'hour', 'dayofweek', 'dayofyear', 'is_weekend', 'item_price', 'count_u', 'days_u', 'item_count_u', 'cate_count_u', 'store_count_u', 'item_price_first', 'item_price_mean', 'item_price_min', 'item_price_max', 'item_price_first_store', 'item_price_mean_store', 'item_price_min_store', 'item_price_max_store', 'item_id_freq', 'cate_id_freq', 'store_id_freq']


l1 = ['irank', 'item_price', 'days_u', 'item_count_u', 'cate_count_u', 'store_count_u', 'item_price_mean', 'item_price_min', 'item_price_first_store', 'item_price_mean_store', 'item_price_min_store', 'item_price_max_store', 'item_id_freq', 'store_id_freq', 'is_weekend', 'month', 'day', 'hour', 'dayofyear', 'dayofweek']

def diff(l1, l2):
    print(len(l1))
    print(len(l2))

    print(f"l1 unique={set(l1)-set(l2)}")
    print(f"l2 unique={set(l2)-set(l1)}")

diff(l, l1)

feat_mi = [0.30160282, 0., 0.12042699, 0.08945839, 0.18780828, 0.11091473,
        0., 0.00633881, 0.01688161, 0.05682857, 0.09940738, 0.12088164,
        0.10091264, 0.00552836, 0.01033121, 0.00752983, 0.00773936, 0.00707915,
        0.01158789, 0.00636465, 0.00817519, 0.00746154, 0.02924712, 0.00782708]

# 拼接成字典
mi_dict = dict(zip(all, feat_mi))

# 按分数从大到小排序
sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

# 输出结果
print("特征互信息分数排序（从高到低）：")
print("-" * 50)
for feature, score in sorted_mi:
    print(f"{feature:25s}: {score:.6f}")

# # 也可以只输出特征名称（如果只需要特征列表）
# print("\n按分数排序的特征列表：")
# feature_list = [f for f, _ in sorted_mi]
# print(feature_list)

# import torch
# Gmi = torch.tensor([0.3017, 0.0000, 0.1202, 0.0891, 0.1880, 0.1107, 0.0066, 0.0172, 0.0566,
#         0.0995, 0.1210, 0.1009, 0.0057, 0.0105, 0.0076, 0.0077, 0.0070, 0.0117,
#         0.0064, 0.0081, 0.0076, 0.0293, 0.0079], device='cuda:0')
# X_num_list = ['irank', 'month', 'day', 'hour', 'dayofweek', 'dayofyear', 'item_price', 'count_u', 'days_u', 'item_count_u', 'cate_count_u', 'store_count_u', 'item_price_first', 'item_price_mean', 'item_price_min', 'item_price_max', 'item_price_first_store', 'item_price_mean_store', 'item_price_min_store', 'item_price_max_store', 'item_id_freq', 'cate_id_freq', 'store_id_freq']
# C0000008SIH00S7SJSFS7SIS7SI00
