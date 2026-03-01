import torch
from typing import List
from sklearn.feature_selection import mutual_info_classif

def calculate_group_mi(
        X: torch.Tensor,
        y: torch.Tensor,
        gidx: torch.Tensor,
        sgidx: List[torch.Tensor],
        equalsize: bool = True
) -> torch.Tensor :
    """
    每组MI

    :param X: n * p
    :param y: n * 1
    :param gidx:
    :param sgidx:
    :param equalsize:
    :return: num_groups * 1
    """
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    feat_mi = mutual_info_classif(
        X_np, y_np, random_state=42, discrete_features=False
    )
    device = X.device
    feat_mi = torch.tensor(feat_mi, dtype=X.dtype, device=device)

    if equalsize:
        num_groups = gidx.max().item() + 1

        group_mi_sum = torch.zeros(num_groups, dtype=X.dtype, device=device)
        group_mi_sum.scatter_add_(0, gidx, feat_mi)

        group_size = torch.zeros(num_groups, dtype=torch.long, device=device)
        group_size.scatter_add_(0, gidx, torch.ones_like(gidx, dtype=torch.long))

        group_size = group_size.clamp(min=1)
        group_mi = group_mi_sum / group_size.float()

    else:
        num_groups = len(sgidx)
        group_mi = torch.zeros(num_groups, dtype=X.dtype, device=device)

        for i, idx in enumerate(sgidx):
            group_feat_mi = feat_mi[idx]
            group_mi[i] = group_feat_mi.mean()

    return group_mi
