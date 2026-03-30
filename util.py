import os
import sys
import functools
import datetime
import torch
import winsound
import traceback
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

def result_beep(func):
    """
    decorator for beep
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            winsound.Beep(500, 3000)
            return res
        except BaseException as ex:
            traceback.print_exc()
            for i in range(3):
                freq = 800 + i * 200
                winsound.Beep(freq, 1000)
            return None
    return wrapper

def log(enable_file=False, file_path="app.log"):
    """
    日志注解 - 通过重定向sys.stdout实现
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout  # 保存原标准输出
            # 如果需要文件输出，重定向stdout
            if enable_file:
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
                file_handle = open(file_path, 'w+', encoding='utf-8')
                sys.stdout = TeeOutput(original_stdout, file_handle)  # 同时输出到控制台和文件
            try:
                # 打印调用信息
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] >>> 进入函数: {func.__name__}")
                if args or kwargs:
                    print(f"           参数: args={args}, kwargs={kwargs}")
                # 执行函数（里面的print都会被捕获）
                result = func(*args, **kwargs)
                # 打印返回信息
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] <<< 退出函数: {func.__name__}, 返回: {result}")
                return result
            finally:
                # 恢复stdout，关闭文件
                if enable_file:
                    sys.stdout = original_stdout
                    file_handle.close()
        return wrapper
    return decorator


class TeeOutput:
    """同时输出到多个目标（控制台 + 文件）"""
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            output.write(message)
            output.flush()  # 立即刷新，避免缓冲

    def flush(self):
        for output in self.outputs:
            output.flush()
