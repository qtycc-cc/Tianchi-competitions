import math
import torch
import numpy as np
from typing import Callable, Tuple, Literal, List, Optional
from sklearn.base import ClassifierMixin, BaseEstimator, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from torch.types import Number

from models.helper import get_Rx, grouped_value, draw_loss_history, draw_loss_history_agg, my_logistic

class IWTLogger:
    def __init__(self):
        self.x: Optional[torch.Tensor] = None
        self.w: Optional[torch.Tensor] = None
        self.T: List[int] = []
        self.g: Optional[torch.Tensor] = None
        self.tau: float = -math.inf
        self.loss_history: List[Number] = []

class HOMOLogger:
    def __init__(self):
        self.x: Optional[torch.Tensor] = None
        self.T: List[int] = []
        self.w: Optional[torch.Tensor] = None

def IWT_GSC(
        fun_obj: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
        w: torch.Tensor,
        p: int,
        num_groups: int,
        s: int,
        gidx: torch.Tensor,
        sgidx: List[torch.Tensor],
        *,
        verbose: bool = False,
        app: Literal['LS', 'LR'] = 'LS',
        lambda_param: float = 1e-1,
        tau: float = 1.0,
        taumax: float = 1e8,
        taumin: float = 1e-8,
        stepsizeShrink: float = 0.5,
        tol_x: float = 1e-3,
        maxIter: int = 5000,
        equalsize: bool = True,
        x_norm: int = 2,
        strategy: Literal['B', 'T', 'H', 'M'] = 'B',
        gamma: float = 0.7,
        mu: float = 0.5,
        gmi: torch.Tensor | None = None,
) -> IWTLogger:

    device = x.device
    dtype = x.dtype

    gidx = gidx.to(device)
    sgidx = [idx.to(device) for idx in sgidx]

    if equalsize:
        gs = p // num_groups

    T = []
    fx, g = fun_obj(x)
    Rx = get_Rx(x, w, num_groups, p, x_norm, gidx, sgidx, equalsize)
    Fx = fx + lambda_param * Rx # real loss(1 * 1)
    ck = Fx.item()
    thetak = 1.0
    sigma = 5e-10

    iwt_logger = IWTLogger()
    iwt_logger.loss_history.append(Fx.item())

    for j in range(1, maxIter + 1):
        Fx_old = Fx.item()
        x_old = x.clone()
        g_old = g.clone()
        tau_old = tau
        T_old = T.copy()

        backtrackCount = 0
        ls_pass = False
        # grouped_z = torch.zeros(num_groups, dtype=dtype, device=device)
        while tau > taumin and backtrackCount < 20:
            backtrackCount += 1

            z = x_old - tau * g_old #(p * 1)
            z_abs = torch.abs(z)
            nu = lambda_param * tau

            if x_norm == 2:
                z_square = z ** 2
                grouped_z = grouped_value(z_square, gidx, sgidx, equalsize)
                zg_norm = torch.sqrt(grouped_z)
                gc = torch.clamp(zg_norm - nu, min=0) / torch.clamp(zg_norm, min=1e-8)

                # recover from num_groups to p
                if equalsize:
                    gca = gc.repeat_interleave(gs)
                else:
                    gca = torch.zeros(p, dtype=dtype, device=device)
                    start_idx = 0
                    for kki, idx in enumerate(sgidx):
                        end_idx = start_idx + len(idx)
                        gca[start_idx:end_idx] = gc[kki]
                        start_idx = end_idx

                soft_z = gca * z #(p * 1)
            else:  # x_norm == 1
                soft_z = torch.sign(z) * torch.clamp(z_abs - nu, min=0)

            if strategy == 'B':
                if num_groups != p:
                    if x_norm == 2:
                        lx = 0.5 * ((soft_z - z) ** 2) / tau #(p * 1)
                        soft_z_square = soft_z ** 2
                        grouped_lx = grouped_value(lx, gidx, sgidx, equalsize)
                        grouped_rx = grouped_value(soft_z_square, gidx, sgidx, equalsize)
                        grouped_rx = torch.sqrt(grouped_rx)
                        grouped_loss = grouped_lx + lambda_param * grouped_rx
                    else:
                        loss_function_value = 0.5 * ((soft_z - z) ** 2) / tau + lambda_param * z_abs
                        grouped_loss = grouped_value(loss_function_value, gidx, sgidx, equalsize)
                else:
                    grouped_loss = 0.5 * ((soft_z - z) ** 2) / tau + lambda_param * torch.abs(soft_z)
                _, sorted_loss_idx = torch.sort(grouped_loss, descending=True)
                n_nonzero = (grouped_loss > 0).sum().item()
                T = sorted_loss_idx[:min(n_nonzero, s)].cpu().tolist()

            elif strategy == 'T':
                if num_groups != p:
                    if x_norm == 2:
                        grouped_z = zg_norm
                    else:
                        grouped_z = grouped_value(z_abs, gidx, sgidx, equalsize)
                else:
                    grouped_z = z_abs

                _, sorted_z_idx = torch.sort(grouped_z, descending=True)
                n_nonzero = (grouped_z > 0).sum().item()
                T = sorted_z_idx[:min(n_nonzero, s)].cpu().tolist()

            elif strategy == 'H':
                if num_groups != p:
                    if x_norm == 2:
                        lx = 0.5 * ((soft_z - z) ** 2) / tau
                        soft_z_square = soft_z ** 2
                        grouped_lx = grouped_value(lx, gidx, sgidx, equalsize)
                        grouped_rx = grouped_value(soft_z_square, gidx, sgidx, equalsize)
                        grouped_rx = torch.sqrt(grouped_rx)
                        grouped_loss = grouped_lx + lambda_param * grouped_rx
                    else:
                        loss_function_value = 0.5 * ((soft_z - z) ** 2) / tau + lambda_param * z_abs
                        grouped_loss = grouped_value(loss_function_value, gidx, sgidx, equalsize)
                else:
                    grouped_loss = 0.5 * ((soft_z - z) ** 2) / tau + lambda_param * torch.abs(soft_z)

                _, sorted_loss_idx = torch.sort(grouped_loss, descending=True)
                _, sorted_z_idx = torch.sort(z_abs, descending=True)

                n_loss_nonzero = (grouped_loss > 0).sum().item()
                n_z_nonzero = (z_abs > 0).sum().item()
                T1 = sorted_loss_idx[:min(n_loss_nonzero, s)].cpu().tolist()

                T2_tmp = sorted_z_idx[:min(n_z_nonzero, s)].cpu().tolist()
                T2 = gidx[T2_tmp].cpu().tolist()

                T = sorted(list(set(T1 + T2)))

            elif strategy == 'M':
                if num_groups != p:
                    if x_norm == 2:
                        lx = 0.5 * ((soft_z - z) ** 2) / tau
                        soft_z_square = soft_z ** 2
                        grouped_lx = grouped_value(lx, gidx, sgidx, equalsize)
                        grouped_rx = grouped_value(soft_z_square, gidx, sgidx, equalsize)
                        grouped_rx = torch.sqrt(grouped_rx)
                        grouped_loss = grouped_lx + lambda_param * grouped_rx
                    else:
                        loss_function_value = 0.5 * ((soft_z - z) ** 2) / tau + lambda_param * z_abs
                        grouped_loss = grouped_value(loss_function_value, gidx, sgidx, equalsize)
                else:
                    grouped_loss = 0.5 * ((soft_z - z) ** 2) / tau + lambda_param * torch.abs(soft_z)

                if num_groups != p:
                    if x_norm == 2:
                        grouped_z = zg_norm
                    else:
                        grouped_z = grouped_value(z_abs, gidx, sgidx, equalsize)
                else:
                    grouped_z = z_abs

                loss_min = torch.min(grouped_loss)
                loss_max = torch.max(grouped_loss)
                grouped_loss_norm = (grouped_loss - loss_min) / (loss_max - loss_min + 1e-8)

                z_min = torch.min(grouped_z)
                z_max = torch.max(grouped_z)
                grouped_z_norm = (grouped_z - z_min) / (z_max - z_min + 1e-8)

                alpha_best = 0.75
                alpha_top = 0.25
                S_origin = alpha_best * grouped_loss_norm + alpha_top * grouped_z_norm

                hybrid_scores = mu * gmi + (1 - mu) * S_origin

                _, sorted_hybrid_idx = torch.sort(hybrid_scores, descending=True)
                n_hybrid_nonzero = (hybrid_scores > 0).sum().item()
                select_num = min(n_hybrid_nonzero, s)
                T = sorted_hybrid_idx[:select_num].cpu().tolist()

            else:
                raise NotImplementedError

            w = torch.ones(p, dtype=dtype, device=device) #(p * 1)
            if s != 0:
                if num_groups != p:
                    ind = torch.cat([sgidx[i] for i in T]) #(about = len(T) * num_groups, 1)
                else:
                    ind = torch.tensor(T, dtype=torch.int64, device=device)
                x = soft_z.clone()
                w[ind] = 0
                x[ind] = z[ind]
            else:
                x = torch.zeros(p, dtype=dtype, device=device)

            fx, g = fun_obj(x)
            Rx = get_Rx(x, w, num_groups, p, x_norm, gidx, sgidx, equalsize)
            Fx = fx + lambda_param * Rx
            Fx_threshold = ck - (sigma * (torch.norm(x - x_old) ** 2) / tau)

            if Fx <= Fx_threshold:
                ls_pass = True
                break
            else:
                tau = max(taumin, tau * stepsizeShrink)
                if verbose:
                    print(f"IWT backtracking... inner iteration = {j}, backtrackCount = {backtrackCount}, fx = {fx.item():.6e}, stepsize = {tau:.2e}")

        if not ls_pass:
            if verbose:
                print(f"IWT backtrack failed! Current iter is {j}")
            x = x_old
            g = g_old
            Fx = torch.tensor(Fx_old, dtype=dtype, device=device)
            T = T_old
            tau = tau_old

        iwt_logger.loss_history.append(Fx.item())

        if app == 'LR':
            residue = abs(Fx_old - Fx.item())
            HaltCond = residue < tol_x
        else:
            relerr = torch.norm(x - x_old) / (torch.norm(x_old) + 1e-8)
            HaltCond = relerr < tol_x

        if HaltCond:
            if verbose:
                print(f"Iter {j} reach stop condition in IWT Loop")
            break

        thetak = 1 + gamma * thetak
        ck = ((thetak - 1) * ck + Fx.item()) / thetak

        if verbose:
            print(f"find stepsize, iteration = {j}, obj = {fx.item():.6e}, stepsize = {tau:.2e}")

        dx = x - x_old
        dg = g - g_old
        dotprod = torch.dot(dx, dg).real

        tau = (torch.norm(dx) ** 2) / (dotprod + 1e-8)
        if tau <= 0 or torch.isinf(tau) or torch.isnan(tau):
            tau = tau_old * 1.5

        tau = max(taumin, min(taumax, tau.item() if torch.is_tensor(tau) else tau))

    iwt_logger.x = x
    iwt_logger.w = w
    iwt_logger.T = T
    iwt_logger.g = g
    iwt_logger.tau = tau

    return iwt_logger

def HIWT_GSC(
        fun_obj: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        A: torch.Tensor,
        x0: torch.Tensor,
        num_groups: int,
        s: int,
        gidx: torch.Tensor,
        *,
        equalsize: bool = True,
        sgidx: List[torch.Tensor] | None = None,
        verbose: bool = False,
        draw_loss: bool = False,
        num_stages: int = 5000,
        tol_x: float = 1e-3,
        tau: float = 1.0,
        de_tolx: float = 0.2,
        eta: float = 1.1,
        app: Literal['LS', 'LR'] = 'LS',
        min_tolx: float = 1e-10,
        lambda_param: float = 0.01,
        strategy: Literal['B', 'T', 'H', 'M'] = 'B',
        mu: float = 0.5,
        gmi: torch.Tensor | None = None,
) -> HOMOLogger:

    device = A.device
    dtype = A.dtype

    gidx = gidx.to(device)

    if sgidx is None:
        sgidx = []
        for kki in range(num_groups):
            idx = torch.where(gidx == kki)[0]
            sgidx.append(idx)


    delta_s = math.ceil(s / 4)
    s1 = delta_s
    p = A.shape[1]
    x = x0.clone()
    w = torch.ones(p, dtype=dtype, device=device)

    loss_history = []
    for k in range(num_stages):
        iwt_logger: IWTLogger = IWT_GSC(fun_obj=fun_obj,
                                        x=x,
                                        w=w,
                                        p=p,
                                        num_groups=num_groups,
                                        s=s1,
                                        lambda_param=lambda_param,
                                        gidx=gidx,
                                        sgidx=sgidx,
                                        verbose=verbose,
                                        app=app,
                                        tau=tau,
                                        tol_x=tol_x,
                                        strategy=strategy,
                                        mu=mu,
                                        gmi=gmi,
                                        equalsize=equalsize)
        loss_history.extend(iwt_logger.loss_history)
        x = iwt_logger.x
        w = iwt_logger.w
        g = iwt_logger.g
        tau = iwt_logger.tau

        halt_cond = s1 == s
        halt_cond1 = (w * x).abs().sum() == 0
        residue = torch.norm(g, p=float('inf')) / lambda_param
        halt_cond2 = residue < 1e-3

        if halt_cond and halt_cond1 and halt_cond2:
            if verbose:
                print(f"Iter {k} reach stop condition in HOMO Loop")
            break

        lambda_param = eta * lambda_param
        tol_x = max(min_tolx, tol_x * de_tolx)
        s1 = min(s, math.ceil(s1 * 2))

    if draw_loss:
        draw_loss_history_agg(loss_history, lambda_param)

    homo_logger = HOMOLogger()
    homo_logger.x = x
    homo_logger.T = sorted(iwt_logger.T)
    homo_logger.w = iwt_logger.w

    return homo_logger

class IWT_Classifier(ClassifierMixin, BaseEstimator):
    _parameter_constraints = {
        "num_groups": [int],
        "s": [int],
        "gidx": [torch.Tensor],
        "tau": [float],
        "tol_x": [float],
        "lambda_param": [float],
        "strategy": [str],
        "equalsize": [bool],
        "sgidx": [list, type(None)],
        "mu": [float],
        "gmi": [torch.Tensor, type(None)],
        "verbose": [bool],
        "draw_loss": [bool],
    }

    def __init__(
            self,
            num_groups: int,
            s: int,
            gidx: torch.Tensor,
            strategy: Literal['B', 'T', 'H', 'M'],
            *,
            tau: float = 1.0,
            tol_x: float = 1e-3,
            lambda_param: float = 1.0,
            equalsize: bool = True,
            sgidx: List[torch.Tensor] | None = None,
            mu: float = 0.5,
            gmi: torch.Tensor | None = None,
            verbose: bool = False,
            draw_loss: bool = False,
            need_normalize: bool = False,
    ):
        if strategy == 'M' and gmi is None:
            raise ValueError('Gmi is required in strategy M')
        self.num_groups = num_groups
        self.s = s
        self.gidx = gidx
        self.tau = tau
        self.tol_x = tol_x
        self.lambda_param = lambda_param
        self.strategy = strategy
        self.equalsize = equalsize
        self.sgidx = sgidx
        self.mu = mu
        self.gmi = gmi
        self.verbose = verbose
        self.draw_loss = draw_loss
        self.need_normalize = need_normalize

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        device = self.gidx.device
        if self.need_normalize:
            X = self._normalize_columns(X)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        n, p = X.shape
        x0 = torch.zeros(p, device=device)

        result = HIWT_GSC(
            lambda x: my_logistic(x, X.T, y),
            X,
            x0,
            self.num_groups,
            self.s,
            self.gidx,
            tau=self.tau,
            tol_x=self.tol_x,
            lambda_param=self.lambda_param,
            equalsize=self.equalsize,
            sgidx=self.sgidx,
            app='LR',
            strategy=self.strategy,
            mu=self.mu,
            gmi=self.gmi,
            verbose=self.verbose,
            draw_loss=self.draw_loss
        )

        self.X_ = result.x.cpu().numpy()
        self.w_ = result.w.cpu().numpy()
        self.T_ = result.T

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        device = self.gidx.device
        if self.need_normalize:
            X = self._normalize_columns(X)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        logits = X @ torch.tensor(self.X_, device=device)
        probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def _normalize_columns(self, X):
        """每列除以其 L2 范数"""
        norms = np.linalg.norm(X, axis=0)
        # norms[norms == 0] = 1  # 避免除零
        return X / norms