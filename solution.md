# 天池大赛三赛道方案总结

## 赛道二：重复购买者预测

**任务**：预测双十一新买家是否会在未来6个月内成为重复购买者
**评估**：AUC

### 核心方案

**特征工程**（多维度统计特征 + 标签比率特征）：

#### 1. 用户维度特征 (User)

| 特征名 | 描述 |
|--------|------|
| `count_u` | 用户日志总数 |
| `days_u` | 用户交互天数 |
| `item_count_u` | 用户访问的商品数 |
| `cat_count_u` | 用户访问的品类数 |
| `merchant_count_u` | 用户访问的商家数 |
| `brand_count_u` | 用户访问的品牌数 |
| `view_count_u` | 用户浏览次数 (action_type=0) |
| `cart_count_u` | 用户加购次数 (action_type=1) |
| `buy_count_u` | 用户购买次数 (action_type=2) |
| `fav_count_u` | 用户收藏次数 (action_type=3) |
| `action_weight_u` | 行为权重 (0.1\*浏览 + 0.2\*加购 + 0.2\*收藏 + 0.5\*购买) |
| `buy_view_ratio_u` | 购买/浏览比 |
| `buy_fav_ratio_u` | 购买/收藏比 |
| `buy_cart_ratio_u` | 购买/加购比 |
| `buy_freq_u` | 购买频率 (购买数/日志数) |
| `repurchase_rate_u` | 用户复购率 (复购商家数/购买总商家数) |
| `loyal_u` | 用户忠诚度 (购买数/购买商家数) |

#### 2. 商家维度特征 (Merchant)

| 特征名 | 描述 |
|--------|------|
| `count_m` | 商家日志总数 |
| `days_m` | 商家交互天数 |
| `item_count_m` | 商家商品数 |
| `cat_count_m` | 商家品类数 |
| `user_count_m` | 商家用户数 |
| `brand_count_m` | 商家品牌数 |
| `view_count_m` | 商家浏览次数 |
| `cart_count_m` | 商家加购次数 |
| `buy_count_m` | 商家购买次数 |
| `fav_count_m` | 商家收藏次数 |
| `action_weight_m` | 商家行为权重 |
| `buy_view_ratio_m` | 商家购买/浏览比 |
| `buy_fav_ratio_m` | 商家购买/收藏比 |
| `buy_cart_ratio_m` | 商家购买/加购比 |
| `buy_freq_m` | 商家购买频率 |
| `repurchase_rate_m` | 商家复购率 |
| `loyal_m` | 商家忠诚度 (购买数/购买用户数) |

#### 3. 用户-商家交叉特征 (User-Merchant)

| 特征名 | 描述 |
|--------|------|
| `count_um` | 交叉日志数 |
| `days_um` | 交叉交互天数 |
| `item_count_um` | 交叉商品数 |
| `cat_count_um` | 交叉品类数 |
| `brand_count_um` | 交叉品牌数 |
| `view_count_um` | 交叉浏览次数 |
| `cart_count_um` | 交叉加购次数 |
| `buy_count_um` | 交叉购买次数 |
| `fav_count_um` | 交叉收藏次数 |
| `action_weight_um` | 交叉行为权重 |
| `buy_view_ratio_um` | 交叉购买/浏览比 |
| `buy_fav_ratio_um` | 交叉购买/收藏比 |
| `buy_cart_ratio_um` | 交叉购买/加购比 |
| `buy_freq_um` | 交叉购买频率 |
| `view_ratio` | 交叉浏览/用户浏览比 |
| `cart_ratio` | 交叉加购/用户加购比 |
| `fav_ratio` | 交叉收藏/用户收藏比 |
| `buy_ratio` | 交叉购买/用户购买比 |

#### 4. 用户画像特征

| 特征名 | 描述 |
|--------|------|
| `age` | 年龄 (0-8) |
| `gender` | 性别(0-2) |

#### 5. 标签比率特征（利用训练集label信息泄漏）

| 特征名 | 描述 |
|--------|------|
| `merchant_label_ratio` | 商家维度label均值 |
| `item_label_ratio` | 商品维度label均值（取merchant维度最大值） |
| `cat_label_ratio` | 品类维度label均值（取merchant维度最大值） |
| `brand_label_ratio` | 品牌维度label均值（取merchant维度最大值） |

### 模型：`TabM_D_Classifier`（深度学习表格模型）

### 测试

TabM默认参数, IWT lambda=1 tau=100

| 本地测试 (-(1-auc))    | 线上测试              | 选择特征    |
|----------|-------------------|-------|
| -0.30346590280532837(cv=1) | 0.6938424(cv=1) 0.6945190(cv=5) | 58 + 2(all) |
| -0.30225688219070435(cv=1) | 0.6938(cv=1) 0.6946730(cv=5) | 47 + 2(all) |

| 模型    | 选择方法(47 + 2)              | 线上评分(AUC)    |
|----------|-------------------|-------|
| TabM | IWT | 0.6946730 |
| TabM | L1正则 | 0.6936018 |
| MLP | IWT | 0.6881 |
| MLP | L1正则 | 0.6868 |
---

## 赛道七：跨国家用户购买预测

**任务**：预测待成熟国家(YY)用户最后一条购买商品（Top30）
**评估**：MRR（Mean Reciprocal Rank）

### 核心方案

**特征工程**：

#### 1. 时间特征

| 特征名 | 描述 |
|--------|------|
| `month` | 月份 |
| `day` | 日期 |
| `hour` | 小时 |
| `dayofweek` | 星期几 (0-6) |
| `dayofyear` | 一年中的第几天 |
| `is_weekend` | 是否周末 (1/0) |

#### 2. 用户统计特征

| 特征名 | 描述 |
|--------|------|
| `count_u` | 用户购买记录数 |
| `days_u` | 用户交互天数 |
| `item_count_u` | 用户购买的商品数 |
| `cate_count_u` | 用户购买的品类数 |
| `store_count_u` | 用户购买的店铺数 |

#### 3. 价格统计特征

| 特征名 | 描述 |
|--------|------|
| `item_price_first` | 用户购买商品的价格（首值） |
| `item_price_mean` | 用户购买商品的平均价格 |
| `item_price_min` | 用户购买商品的最低价格 |
| `item_price_max` | 用户购买商品的最高价格 |
| `item_price_first_store` | 店铺商品价格（首值） |
| `item_price_mean_store` | 店铺商品平均价格 |
| `item_price_min_store` | 店铺商品最低价格 |
| `item_price_max_store` | 店铺商品最高价格 |

#### 4. 频率编码特征

| 特征名 | 描述 |
|--------|------|
| `item_id_freq` | 商品出现频率（归一化） |
| `cate_id_freq` | 品类出现频率（归一化） |
| `store_id_freq` | 店铺出现频率（归一化） |

#### 5. 类别特征(待定)

| 特征名 | 描述 |
|--------|------|
| `cate_id` | Top350 |
| `store_id` | Top25 |

#### 6. irank标记特征

| 特征名 | 描述 |
|--------|------|
| `is_irank1_item` | 该商品是否为用户倒数第1条购买记录的商品（目标标签） |

### 预测流程：
1. 用TabM二分类预测每个用户-商品对的irank=1概率
2. 按概率排序生成Top30预测
3. 补全策略：共现商品 → 最后一次购买的品类的热门商品 → 倒数第二次购买的品类的热门商品 → 全局热门商品

### 模型：`TabM_D_Classifier`

---

## 赛道八：移动端商品推荐

**任务**：预测用户在接下来一天对商品子集P的购买行为
**评估**：F1-score

### 核心方案

**滑动窗口策略**：
- 特征窗口：7天
- 标签日：最近5天（每日构建一个训练窗口）

**特征体系**（八种角度）：

#### 1. 用户特征 (User)

| 特征名 | 描述 |
|--------|------|
| `user_view` | 用户浏览次数 (behavior_type=1) |
| `user_fav` | 用户收藏次数 (behavior_type=2) |
| `user_cart` | 用户加购次数 (behavior_type=3) |
| `user_buy` | 用户购买次数 (behavior_type=4) |
| `user_active_days` | 用户活跃天数 |
| `user_view_to_buy` | 用户浏览转购买率 |
| `user_cart_to_buy` | 用户加购转购买率 |
| `user_fav_to_buy` | 用户收藏转购买率 |

#### 2. 商品特征 (Item)

| 特征名 | 描述 |
|--------|------|
| `item_view` | 商品浏览次数 |
| `item_fav` | 商品收藏次数 |
| `item_cart` | 商品加购次数 |
| `item_buy` | 商品购买次数 |
| `item_count` | 商品被交互总次数 |
| `user_count_i` | 对商品有交互的用户数 |
| `item_view_to_buy` | 商品浏览转购买率 |
| `item_cart_to_buy` | 商品加购转购买率 |
| `item_fav_to_buy` | 商品收藏转购买率 |
| `item_buy_freq` | 商品购买频率 |

#### 3. 品类特征 (Category)

| 特征名 | 描述 |
|--------|------|
| `cate_view` | 品类浏览次数 |
| `cate_fav` | 品类收藏次数 |
| `cate_cart` | 品类加购次数 |
| `cate_buy` | 品类购买次数 |
| `cate_count` | 品类被交互总次数 |
| `user_count_c` | 对品类有交互的用户数 |
| `cate_view_to_buy` | 品类浏览转购买率 |
| `cate_cart_to_buy` | 品类加购转购买率 |
| `cate_fav_to_buy` | 品类收藏转购买率 |

#### 4. User-Item交叉特征

| 特征名 | 描述 |
|--------|------|
| `ui_view` | 用户对商品浏览次数 |
| `ui_fav` | 用户对商品收藏次数 |
| `ui_cart` | 用户对商品加购次数 |
| `ui_buy` | 用户对商品购买次数 |
| `ui_view_to_buy` | 用户-商品浏览转购买率 |
| `ui_cart_to_buy` | 用户-商品加购转购买率 |
| `ui_fav_to_buy` | 用户-商品收藏转购买率 |

#### 5. User-Category交叉特征

| 特征名 | 描述 |
|--------|------|
| `uc_view` | 用户对品类浏览次数 |
| `uc_fav` | 用户对品类收藏次数 |
| `uc_cart` | 用户对品类加购次数 |
| `uc_buy` | 用户对品类购买次数 |
| `uc_buy_ratio` | 用户对品类购买占总行为比 |

#### 6. 排序特征

| 特征名 | 描述 |
|--------|------|
| `view_rank_in_uc` | 用户在品类内对该商品的浏览次数排名 |

#### 7. 类别特征

| 特征名 | 描述 |
|--------|------|
| `behavior_type` | 浏览、收藏、加购物车、购买，对应取值分别是1、2、3、4 |
| `hour` | 0-24 |

### 样本平衡：负样本抽样（正负比1:30）

### 模型：`TabM_D_Classifier`

### 测试

TabM默认参数, IWT lambda=0.1 tau=500

| 本地测试 cross_entropy    | 线上测试 f1 recall precision  | 选择特征    |
|----------|-------------------|-------|
| -0.09849593934104052(cv=1) | 0.0709 0.0896 0.0587(cv=1) 0.0716 0.0905 0.0593(cv=5) | 40 + 2(all) |
| -0.09851281741648511(cv=1) | 0.0712 0.0899 0.0589(cv=1) 0.0713 0.0901 0.0590(cv=5) | 35 + 2(all) |

| 模型    | 选择方法(28 + 2)              | 线上评分(f1 recall precision)    |
|----------|-------------------|-------|
| TabM | IWT | 0.0719 0.0909 0.0595 |
| TabM | L1正则 | 0.0716 0.0905 0.0593 |
| MLP | IWT | 0.0705 0.0891 0.0583 |
| MLP | L1正则 | 0.0705 0.0890 0.0583 |
