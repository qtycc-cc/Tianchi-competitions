import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import tabm
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, Dict, Any, Literal
import sklearn
import sklearn.preprocessing
import sklearn.base
import sklearn.utils.validation
from math import ceil
import rtdl_num_embeddings
from copy import deepcopy

class TabMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """
    Scikit-learn compatible classifier for TabM (Tabular MLP with Ensembling).
    支持内部交叉验证（n_cv参数）。
    
    Parameters
    ----------
    n_epochs : int, default=1_000_000_000
        Maximum number of training epochs.
        
    batch_size : int, default=256
        Batch size for training.
        
    lr : float, default=2e-3
        Learning rate for AdamW optimizer.
        
    weight_decay : float, default=3e-4
        Weight decay (L2 regularization) coefficient.
        
    patience : int, default=16
        Early stopping patience (number of epochs without improvement).
    
    n_blocks : int or 'auto', default='auto'
        Number of blocks in TabM. If 'auto', set to 3 if no num_embeddings, else 2.

    d_block : int, default=512
        Block dimension (number of neurons in each block).
    
    dropout : float, default=0.1
        Dropout rate for TabM blocks.
        
    gradient_clipping_norm : float, default=1.0
        Gradient clipping norm. Set to None to disable.
        
    device : str or torch.device, default='auto'
        Device to use for training ('cuda', 'cpu', or 'auto').
        
    random_state : int, optional, default=42
        Random seed for reproducibility.
        
    quantile_transform : bool, default=True
        Whether to apply QuantileTransformer preprocessing.
        
    n_quantiles : int, default=None
        Number of quantiles for QuantileTransformer. If None, computed as
        max(min(n_samples // 30, 1000), 10).
        
    eval_batch_size : int, default=1024
        Batch size for evaluation.
        
    share_training_batches : bool, default=True
        Whether to share batches across ensemble members during training.
        
    verbose : int, default=1
        Verbosity level (0: silent, 1: epoch info, 2: detailed).
        
    num_embeddings : str or None, default=None
        数值特征嵌入方式。可选：
        - None: 不使用特殊嵌入（默认TabM行为）
        - 'linear': LinearReLUEmbeddings
        - 'periodic': PeriodicEmbeddings (lite=False)
        - 'piecewise_linear': PiecewiseLinearEmbeddings (version='B', d_embedding=16, n_bins=48)
        
    piecewise_linear_bins : int, default=48
        PiecewiseLinearEmbeddings的bin数量，仅在num_embeddings='piecewise_linear'时使用。
        
    piecewise_linear_d_embedding : int, default=16
        PiecewiseLinearEmbeddings的嵌入维度。
        
    piecewise_linear_version : str, default='B'
        PiecewiseLinearEmbeddings的版本 ('A' 或 'B')。
        
    use_amp : bool, default=False
        Whether to use Automatic Mixed Precision training.
        
    class_weight : str or dict, default=None
        If 'balanced', automatically adjust weights inversely proportional
        to class frequencies. Can also be a manual dict {class: weight}.
        
    n_cv : int, default=1
        内部交叉验证折数。如果>1，则自动进行n_cv折交叉验证训练，
        并返回所有fold模型的平均预测结果。
        
    cv_agg_method : str, default='mean'
        交叉验证集成方法：'mean'（平均概率）或 'vote'（投票）。
    """
    
    def __init__(
        self,
        n_epochs: int = 1_000_000_000,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 3e-4,
        patience: int = 16,
        n_blocks: int | str = 'auto',
        d_block: int = 512,
        dropout: float = 0.1,
        gradient_clipping_norm: float = 1.0,
        device: Union[str, torch.device] = 'auto',
        random_state: Optional[int] = 42,
        quantile_transform: bool = True,
        n_quantiles: Optional[int] = None,
        eval_batch_size: int = 1024,
        share_training_batches: bool = True,
        verbose: int = 1,
        num_embeddings: Optional[Literal['linear', 'periodic', 'piecewise_linear']] = None,
        piecewise_linear_bins: int = 48,
        piecewise_linear_d_embedding: int = 16,
        piecewise_linear_version: str = 'B',
        use_amp: bool = False,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        n_cv: int = 1,
        cv_agg_method: str = 'mean',
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.d_block = d_block
        self.dropout = dropout
        self.gradient_clipping_norm = gradient_clipping_norm
        self.device = device
        self.random_state = random_state
        self.quantile_transform = quantile_transform
        self.n_quantiles = n_quantiles
        self.eval_batch_size = eval_batch_size
        self.share_training_batches = share_training_batches
        self.verbose = verbose
        self.num_embeddings = num_embeddings
        self.piecewise_linear_bins = piecewise_linear_bins
        self.piecewise_linear_d_embedding = piecewise_linear_d_embedding
        self.piecewise_linear_version = piecewise_linear_version
        self.use_amp = use_amp
        self.class_weight = class_weight
        self.n_cv = n_cv
        self.cv_agg_method = cv_agg_method
        
        if n_blocks == 'auto':
            self.n_blocks = 3 if self.num_embeddings is None else 2
        else:
            self.n_blocks = n_blocks
            
        # Attributes set during fit
        self.model_ = None  # 单模型模式或CV中的最后一个模型
        self.cv_models_ = []  # CV模式：存储所有fold的模型
        self.cv_preprocessings_ = []  # CV模式：存储所有fold的预处理
        self.preprocessing_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.best_epoch_ = None
        self.best_val_score_ = None
        self.history_ = {'train_loss': [], 'val_score': [], 'test_score': []}
        self.num_embeddings_module_ = None
        self.is_fitted_ = False
        
    def _get_device(self) -> torch.device:
        """Determine the device to use."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)
    
    def _set_random_seeds(self, seed_offset=0):
        """Set random seeds for reproducibility."""
        if self.random_state is not None:
            seed = self.random_state + seed_offset
            import random
            random.seed(seed)
            np.random.seed(seed + 1)
            torch.manual_seed(seed + 2)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 3)
    
    def _create_num_embeddings(self, n_num_features: int, x_train: np.ndarray) -> Optional[nn.Module]:
        """创建数值特征嵌入模块。"""
        if self.num_embeddings is None:
            return None
        
        if self.num_embeddings == 'linear':
            return rtdl_num_embeddings.LinearReLUEmbeddings(n_num_features)
        
        elif self.num_embeddings == 'periodic':
            return rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)
        
        elif self.num_embeddings == 'piecewise_linear':
            x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
            bins = rtdl_num_embeddings.compute_bins(x_train_tensor, n_bins=self.piecewise_linear_bins)
            
            return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins=bins,
                d_embedding=self.piecewise_linear_d_embedding,
                activation=False,
                version=self.piecewise_linear_version,
            )
        
        else:
            raise ValueError(f"Unknown num_embeddings: {self.num_embeddings}")
    
    def _prepare_data_single(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        val_size_from_train: float = 0.2
    ):
        """为单模型准备数据（带内部验证集划分）。"""
        self.n_features_in_ = X.shape[1]
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        
        X_num = X.astype(np.float32)
        y_arr = y_encoded.astype(np.int64)
        
        all_idx = np.arange(len(y_arr))
        
        trainval_idx, test_idx = train_test_split(
            all_idx, 
            train_size=1-validation_split, 
            random_state=self.random_state
        )
        train_idx, val_idx = train_test_split(
            trainval_idx, 
            train_size=1-val_size_from_train, 
            random_state=self.random_state
        )
        
        data_numpy = {
            'train': {'x_num': X_num[train_idx], 'y': y_arr[train_idx]},
            'val': {'x_num': X_num[val_idx], 'y': y_arr[val_idx]},
            'test': {'x_num': X_num[test_idx], 'y': y_arr[test_idx]},
        }
        
        # PiecewiseLinearEmbeddings
        if self.num_embeddings == 'piecewise_linear':
            self.num_embeddings_module_ = self._create_num_embeddings(
                self.n_features_in_, 
                data_numpy['train']['x_num']
            )
            if self.verbose:
                print(f"Created PiecewiseLinearEmbeddings with {self.piecewise_linear_bins} bins")
        
        # Quantile transformation
        preprocessing = None
        if self.quantile_transform:
            x_num_train = data_numpy['train']['x_num']
            noise = (
                np.random.default_rng(0)
                .normal(0.0, 1e-5, x_num_train.shape)
                .astype(x_num_train.dtype)
            )
            
            n_quantiles = self.n_quantiles
            if n_quantiles is None:
                n_quantiles = max(min(len(train_idx) // 30, 1000), 10)
            
            preprocessing = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution='normal',
                subsample=10**9,
            ).fit(x_num_train + noise)
            
            for part in data_numpy:
                data_numpy[part]['x_num'] = preprocessing.transform(
                    data_numpy[part]['x_num']
                )
        
        # Convert to tensors
        device = self._get_device()
        data = {
            part: {
                k: torch.as_tensor(v, device=device) 
                for k, v in data_numpy[part].items()
            }
            for part in data_numpy
        }
        
        return data, len(train_idx), preprocessing
    
    def _prepare_data_cv(self, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray):
        """为CV的某一fold准备数据。"""
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.classes_, _ = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        
        X_num = X.astype(np.float32)
        y_arr = y.astype(np.int64)
        
        data_numpy = {
            'train': {'x_num': X_num[train_idx], 'y': y_arr[train_idx]},
            'val': {'x_num': X_num[val_idx], 'y': y_arr[val_idx]},
        }
        
        # PiecewiseLinearEmbeddings - 每个fold独立计算bins
        num_embeddings_module = None
        if self.num_embeddings == 'piecewise_linear':
            num_embeddings_module = self._create_num_embeddings(
                self.n_features_in_, 
                data_numpy['train']['x_num']
            )
        
        # Quantile transformation - 每个fold独立拟合
        preprocessing = None
        if self.quantile_transform:
            x_num_train = data_numpy['train']['x_num']
            noise = (
                np.random.default_rng(0)
                .normal(0.0, 1e-5, x_num_train.shape)
                .astype(x_num_train.dtype)
            )
            
            n_quantiles = self.n_quantiles
            if n_quantiles is None:
                n_quantiles = max(min(len(train_idx) // 30, 1000), 10)
            
            preprocessing = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution='normal',
                subsample=10**9,
            ).fit(x_num_train + noise)
            
            for part in data_numpy:
                data_numpy[part]['x_num'] = preprocessing.transform(
                    data_numpy[part]['x_num']
                )
        
        # Convert to tensors
        device = self._get_device()
        data = {
            part: {
                k: torch.as_tensor(v, device=device) 
                for k, v in data_numpy[part].items()
            }
            for part in data_numpy
        }
        
        return data, len(train_idx), preprocessing, num_embeddings_module
    
    def _create_model(self, num_embeddings_module=None) -> nn.Module:
        """Create TabM model."""
        device = self._get_device()
        
        if self.use_amp and device.type == 'cuda':
            self.amp_dtype_ = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() 
                else torch.float16
            )
            self.amp_enabled_ = True
            self.grad_scaler_ = torch.cuda.amp.GradScaler() if self.amp_dtype_ == torch.float16 else None
        else:
            self.amp_dtype_ = torch.float32
            self.amp_enabled_ = False
            self.grad_scaler_ = None
        
        model = tabm.TabM.make(
            n_num_features=self.n_features_in_,
            cat_cardinalities=None,
            d_out=self.n_classes_,
            num_embeddings=num_embeddings_module,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
        ).to(device)
        
        return model
    
    def _compute_class_weights(self, y: Tensor) -> Optional[Tensor]:
        """Compute class weights for imbalanced datasets."""
        if self.class_weight is None:
            return None
        
        if isinstance(self.class_weight, str) and self.class_weight == 'balanced':
            classes, counts = torch.unique(y, return_counts=True)
            total = len(y)
            weights = total / (len(classes) * counts.float())
            weight_tensor = torch.ones(self.n_classes_, device=y.device)
            for cls, w in zip(classes.tolist(), weights.tolist()):
                weight_tensor[cls] = w
            return weight_tensor
        elif isinstance(self.class_weight, dict):
            weight_tensor = torch.ones(self.n_classes_, device=y.device)
            for cls, w in self.class_weight.items():
                cls_idx = np.where(self.classes_ == cls)[0]
                if len(cls_idx) > 0:
                    weight_tensor[cls_idx[0]] = w
            return weight_tensor
        
        return None
    
    def _train_single_model(self, data: Dict, train_size: int, seed_offset: int = 0):
        """训练单个模型，返回训练好的模型和最佳验证分数。"""
        self._set_random_seeds(seed_offset)
        
        model = self._create_model(self.num_embeddings_module_)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        class_weights = self._compute_class_weights(data['train']['y'])
        
        base_loss_fn = nn.functional.cross_entropy
        
        def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
            y_pred = y_pred.flatten(0, 1)
            if self.share_training_batches:
                y_true = y_true.repeat_interleave(model.backbone.k)
            else:
                y_true = y_true.flatten(0, 1)
            
            if class_weights is not None:
                return base_loss_fn(y_pred, y_true, weight=class_weights)
            return base_loss_fn(y_pred, y_true)
        
        device = self._get_device()
        best_val_score = -float('inf')
        best_state = None
        remaining_patience = self.patience
        
        for epoch in range(self.n_epochs):
            if self.share_training_batches:
                batches = torch.randperm(train_size, device=device).split(self.batch_size)
            else:
                batches = (
                    torch.rand((train_size, model.backbone.k), device=device)
                    .argsort(dim=0)
                    .split(self.batch_size, dim=0)
                )
            
            epoch_losses = []
            model.train()
            
            for batch_idx in batches:
                optimizer.zero_grad()
                
                with torch.autocast(
                    device.type, 
                    enabled=self.amp_enabled_, 
                    dtype=self.amp_dtype_ if self.amp_enabled_ else torch.float32
                ):
                    x_batch = data['train']['x_num'][batch_idx]
                    y_batch = data['train']['y'][batch_idx]
                    y_pred = model(x_batch, None).float()
                    loss = loss_fn(y_pred, y_batch)
                
                epoch_losses.append(loss.item())
                
                if self.grad_scaler_ is None:
                    loss.backward()
                else:
                    self.grad_scaler_.scale(loss).backward()
                
                if self.gradient_clipping_norm is not None:
                    if self.grad_scaler_ is not None:
                        self.grad_scaler_.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.gradient_clipping_norm
                    )
                
                if self.grad_scaler_ is None:
                    optimizer.step()
                else:
                    self.grad_scaler_.step(optimizer)
                    self.grad_scaler_.update()
            
            # Evaluation
            val_score = self._evaluate_model(model, data['val'])
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {
                    'model': deepcopy(model.state_dict()),
                    'epoch': epoch,
                }
                remaining_patience = self.patience
                improved = True
            else:
                remaining_patience -= 1
                improved = False
            
            if self.verbose >= 2 or (self.verbose == 1 and (epoch % 10 == 0 or improved)):
                print(f'Epoch {epoch:3d} | Loss: {np.mean(epoch_losses):.4f} | Val: {val_score:.4f}')
            
            if remaining_patience < 0:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state['model'])
        
        return model, best_val_score
    
    def _evaluate_model(self, model: nn.Module, data_val: Dict) -> float:
        """评估单个模型在验证集上的性能。"""
        model.eval()
        device = self._get_device()
        
        with torch.no_grad():
            y_pred_list = []
            for idx in torch.arange(len(data_val['y']), device=device).split(
                self.eval_batch_size
            ):
                x_batch = data_val['x_num'][idx]
                with torch.autocast(
                    device.type, 
                    enabled=self.amp_enabled_, 
                    dtype=self.amp_dtype_ if self.amp_enabled_ else torch.float32
                ):
                    pred = model(x_batch, None).float()
                y_pred_list.append(pred)
            
            y_pred = torch.cat(y_pred_list)
            y_true = data_val['y']
            
            y_true_expanded = y_true.unsqueeze(1).unsqueeze(-1).expand(
                -1, y_pred.shape[1], -1
            )
            res = -nn.functional.log_softmax(y_pred, dim=-1).gather(
                -1, y_true_expanded
            ).squeeze(-1)
            
            score = -res.mean().item()
        
        return float(score)
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        eval_set: Optional[tuple] = None
    ) -> 'TabMClassifier':
        """
        Fit the TabM classifier.
        如果n_cv > 1，自动进行交叉验证训练。
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target labels.
            
        eval_set : tuple (X_val, y_val), optional
            External validation set. 仅在n_cv=1时有效。
            
        Returns
        -------
        self : TabMClassifier
            Fitted estimator.
        """
        X = sklearn.utils.validation.check_array(X, dtype=np.float32, accept_sparse=False)
        y = sklearn.utils.validation.check_array(y, ensure_2d=False, dtype=None)
        
        # 编码标签
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y_arr = y_encoded.astype(np.int64)
        
        if self.n_cv > 1:
            # 交叉验证模式
            if self.verbose:
                print(f"Starting {self.n_cv}-fold cross validation training...")
            
            skf = StratifiedKFold(n_splits=self.n_cv, shuffle=True, random_state=self.random_state)
            
            self.cv_models_ = []
            self.cv_preprocessings_ = []
            fold_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_arr)):
                if self.verbose:
                    print(f"\n{'='*50}")
                    print(f"Fold {fold_idx + 1}/{self.n_cv}")
                    print(f"{'='*50}")
                
                # 准备该fold的数据
                data, train_size, preprocessing, num_emb_module = self._prepare_data_cv(
                    X, y_arr, train_idx, val_idx
                )
                
                # 临时设置num_embeddings_module_用于该fold
                original_num_emb = self.num_embeddings_module_
                self.num_embeddings_module_ = num_emb_module
                
                # 训练模型
                model, val_score = self._train_single_model(data, train_size, seed_offset=fold_idx * 100)
                
                # 恢复
                self.num_embeddings_module_ = original_num_emb
                
                self.cv_models_.append(model)
                self.cv_preprocessings_.append(preprocessing)
                fold_scores.append(val_score)
                
                if self.verbose:
                    print(f"Fold {fold_idx + 1} best val score: {val_score:.4f}")
            
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"CV completed. Mean val score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
                print(f"{'='*50}")
            
            self.is_fitted_ = True
            self.best_val_score_ = np.mean(fold_scores)
            
        else:
            # 单模型模式
            if eval_set is not None:
                X_val, y_val = eval_set
                y_val_encoded = np.searchsorted(self.classes_, y_val)
                X_combined = np.vstack([X, X_val])
                y_combined = np.concatenate([y_arr, y_val_encoded])
                
                data, train_size, preprocessing = self._prepare_data_single(
                    X_combined, y_combined, validation_split=0
                )
                
                n_train = len(X)
                device = self._get_device()
                data['train'] = {
                    'x_num': data['train']['x_num'][:n_train],
                    'y': data['train']['y'][:n_train]
                }
                data['val'] = {
                    'x_num': data['train']['x_num'][n_train:],
                    'y': data['train']['y'][n_train:]
                }
                self.preprocessing_ = preprocessing
            else:
                data, train_size, self.preprocessing_ = self._prepare_data_single(X, y_arr)
            
            self.model_, self.best_val_score_ = self._train_single_model(data, train_size)
            self.is_fitted_ = True
        
        return self
    
    def _predict_single(self, X: np.ndarray, model: nn.Module, preprocessing=None) -> np.ndarray:
        """使用单个模型进行预测。"""
        X_num = X.astype(np.float32)
        if self.quantile_transform and preprocessing is not None:
            X_num = preprocessing.transform(X_num)
        
        device = self._get_device()
        X_tensor = torch.as_tensor(X_num, device=device)
        
        model.eval()
        y_pred_list = []
        
        with torch.no_grad():
            for idx in torch.arange(len(X_tensor), device=device).split(
                self.eval_batch_size
            ):
                x_batch = X_tensor[idx]
                with torch.autocast(
                    device.type, 
                    enabled=self.amp_enabled_, 
                    dtype=self.amp_dtype_ if self.amp_enabled_ else torch.float32
                ):
                    pred = model(x_batch, None).float()
                y_pred_list.append(pred)
        
        y_pred = torch.cat(y_pred_list)
        y_pred_proba = torch.softmax(y_pred, dim=-1).mean(dim=1)
        return y_pred_proba.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        如果n_cv > 1，返回所有fold模型的平均预测。
        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X, dtype=np.float32, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        if self.n_cv > 1:
            # CV模式：平均所有fold的预测
            all_probas = []
            for model, preprocessing in zip(self.cv_models_, self.cv_preprocessings_):
                proba = self._predict_single(X, model, preprocessing)
                all_probas.append(proba)
            
            all_probas = np.array(all_probas)  # (n_cv, n_samples, n_classes)
            
            if self.cv_agg_method == 'mean':
                return np.mean(all_probas, axis=0)
            elif self.cv_agg_method == 'vote':
                # 投票：先取每个模型的argmax，然后投票
                votes = np.argmax(all_probas, axis=2)  # (n_cv, n_samples)
                final_votes = np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=self.n_classes_).argmax(),
                    axis=0,
                    arr=votes
                )
                # 转换为one-hot概率
                result = np.zeros((len(X), self.n_classes_))
                result[np.arange(len(X)), final_votes] = 1.0
                return result
            else:
                raise ValueError(f"Unknown cv_agg_method: {self.cv_agg_method}")
        else:
            # 单模型模式
            return self._predict_single(X, self.model_, self.preprocessing_)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        获取所有ensemble成员的预测（包括CV fold和TabM内部ensemble）。
        返回形状: (n_samples, n_cv * k, n_classes) 如果n_cv > 1
                 (n_samples, k, n_classes) 如果n_cv == 1
        """
        sklearn.utils.validation.check_is_fitted(self)
        X = sklearn.utils.validation.check_array(X, dtype=np.float32, accept_sparse=False)
        
        X_num = X.astype(np.float32)
        
        if self.n_cv > 1:
            # 收集所有fold的所有ensemble成员预测
            all_preds = []
            for model, preprocessing in zip(self.cv_models_, self.cv_preprocessings_):
                if self.quantile_transform and preprocessing is not None:
                    X_processed = preprocessing.transform(X_num)
                else:
                    X_processed = X_num
                
                device = self._get_device()
                X_tensor = torch.as_tensor(X_processed, device=device)
                
                model.eval()
                fold_preds = []
                
                with torch.no_grad():
                    for idx in torch.arange(len(X_tensor), device=device).split(
                        self.eval_batch_size
                    ):
                        x_batch = X_tensor[idx]
                        with torch.autocast(
                            device.type, 
                            enabled=self.amp_enabled_, 
                            dtype=self.amp_dtype_ if self.amp_enabled_ else torch.float32
                        ):
                            pred = model(x_batch, None).float()
                        fold_preds.append(pred)
                
                fold_pred = torch.cat(fold_preds)  # (n_samples, k, n_classes)
                all_preds.append(fold_pred)
            
            # 合并所有fold: (n_samples, n_cv * k, n_classes)
            final_pred = torch.cat(all_preds, dim=1)
            return torch.softmax(final_pred, dim=-1).cpu().numpy()
        else:
            # 单模型模式
            if self.quantile_transform and self.preprocessing_ is not None:
                X_processed = self.preprocessing_.transform(X_num)
            else:
                X_processed = X_num
            
            device = self._get_device()
            X_tensor = torch.as_tensor(X_processed, device=device)
            
            self.model_.eval()
            y_pred_list = []
            
            with torch.no_grad():
                for idx in torch.arange(len(X_tensor), device=device).split(
                    self.eval_batch_size
                ):
                    x_batch = X_tensor[idx]
                    with torch.autocast(
                        device.type, 
                        enabled=self.amp_enabled_, 
                        dtype=self.amp_dtype_ if self.amp_enabled_ else torch.float32
                    ):
                        pred = self.model_(x_batch, None).float()
                    y_pred_list.append(pred)
            
            y_pred = torch.cat(y_pred_list)
            return torch.softmax(y_pred, dim=-1).cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    def get_params(self, deep=True):
        """Get parameters."""
        return {
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'n_blocks': self.n_blocks,
            'd_block': self.d_block,
            'dropout': self.dropout,
            'gradient_clipping_norm': self.gradient_clipping_norm,
            'device': self.device,
            'random_state': self.random_state,
            'quantile_transform': self.quantile_transform,
            'n_quantiles': self.n_quantiles,
            'eval_batch_size': self.eval_batch_size,
            'share_training_batches': self.share_training_batches,
            'verbose': self.verbose,
            'num_embeddings': self.num_embeddings,
            'piecewise_linear_bins': self.piecewise_linear_bins,
            'piecewise_linear_d_embedding': self.piecewise_linear_d_embedding,
            'piecewise_linear_version': self.piecewise_linear_version,
            'use_amp': self.use_amp,
            'class_weight': self.class_weight,
            'n_cv': self.n_cv,
            'cv_agg_method': self.cv_agg_method,
        }
    
    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    
    # 生成示例数据
    X, y = make_classification(
        n_samples=5000, 
        n_features=20, 
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=" * 60)
    print("1. Single Model Mode (n_cv=1)")
    print("=" * 60)
    
    clf_single = TabMClassifier(
        n_epochs=50,
        batch_size=256,
        patience=10,
        random_state=42,
        verbose=1,
        n_cv=1,  # 单模型
    )
    clf_single.fit(X_train, y_train)
    proba_single = clf_single.predict_proba(X_test)[:, 1]
    print(f"Single model ROC AUC: {roc_auc_score(y_test, proba_single):.4f}")
    
    print("\n" + "=" * 60)
    print("2. Cross Validation Mode (n_cv=5)")
    print("=" * 60)
    
    clf_cv = TabMClassifier(
        n_epochs=30,  # 每fold训练轮数减少，因为有多fold
        batch_size=256,
        patience=5,
        random_state=42,
        verbose=1,
        n_cv=5,  # 5折交叉验证
        cv_agg_method='mean',  # 平均概率
    )
    clf_cv.fit(X_train, y_train)
    proba_cv = clf_cv.predict_proba(X_test)[:, 1]
    print(f"CV ensemble ROC AUC: {roc_auc_score(y_test, proba_cv):.4f}")
    
    # 获取详细的ensemble预测（所有fold的所有成员）
    ensemble_preds = clf_cv.predict_ensemble(X_test[:5])
    print(f"\nEnsemble predictions shape: {ensemble_preds.shape}")
    print(f"(n_samples={ensemble_preds.shape[0]}, n_folds * k_ensemble={ensemble_preds.shape[1]}, n_classes={ensemble_preds.shape[2]})")