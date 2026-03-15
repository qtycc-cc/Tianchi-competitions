import torch
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from models.iwt_classifier import IWT_Classifier
from util import calculate_group_mi

X, y = make_classification(
    n_samples=2000,
    n_features=100,
    random_state=42
)

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

# gmi = calculate_group_mi(
#     torch.tensor(X.to_numpy(), dtype=torch.float32, device=device),
#     torch.tensor(y.to_numpy(), dtype=torch.float32, device=device),
#     gidx,
#     sgidx,
#     True
# )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
# DataFrame convert to ndarray
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, X_train.shape)
    .astype(X_train.dtype)
)

preprocessing = QuantileTransformer(
    n_quantiles=max(min(len(X) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
    random_state=42,
).fit(X_train.copy())

X_train = preprocessing.transform(X_train)
X_test = preprocessing.transform(X_test)

model = IWT_Classifier(
    num_groups=len(sgidx),
    s=num_groups,
    gidx=gidx,
    sgidx=sgidx,
    strategy='B',
    equalsize=True,
    verbose=True,
    draw_loss=True
)

model.fit(X_train, y_train)
print(f"x={model.X_} | x_len={len(model.X_)}")
print(f"w={model.w_} | w_len={len(model.w_)}")
print(f"t={model.T_} | t_len={len(model.T_)}")
cross_entropy = log_loss(y_test, model.predict_proba(X_test))
f1 = f1_score(y_test, model.predict(X_test))
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Cross_entropy={cross_entropy} | f1={f1} | auc={auc}")
# Cross_entropy=0.2543304364255411 | f1=0.9019607843137255 | auc=0.96115
# Cross_entropy=0.30437657579364347 | f1=0.8774509803921569 | auc=0.9510500000000001