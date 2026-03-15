import torch
import functools
import math
import random
import numpy as np
from pathlib import Path
from torch import nn
from sklearn.metrics import log_loss
from typing import Optional, List, Dict, Any, Union, Tuple, Literal

from pytabkit.models import utils
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.sklearn.sklearn_base import AlgInterfaceClassifier
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface
from pytabkit.models.alg_interfaces.ensemble_interfaces import AlgorithmSelectionAlgInterface, \
    CaruanaEnsembleAlgInterface

from pytabkit.models.training.metrics import Metrics

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.alg_interfaces import SingleSplitAlgInterface, RandomParamsAlgInterface

from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.nn_models import rtdl_num_embeddings
from pytabkit.models.nn_models.base import Fitter
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.nn_models.tabm import Model, make_parameter_groups
from pytabkit.models.training.logging import Logger

class RealMLPHPOConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, Path]] = None, verbosity: int = 0,
                 n_hyperopt_steps: Optional[int] = None, val_metric_name: Optional[str] = None,
                 calibration_method: Optional[str] = None, hpo_space_name: Optional[str] = None,
                 n_caruana_steps: Optional[int] = None, n_epochs: Optional[int] = None,
                 use_caruana_ensembling: Optional[bool] = None, train_metric_name: Optional[str] = None,
                 time_limit_s: Optional[float] = None,
                 ):
        """

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices or an explicit validation set are given in fit(),
            `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification).
            For n_cv=1, a single train-validation split will be used,
            where `val_fraction` controls the fraction of validation samples.
            If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Folder in which models can be stored.
            Setting this allows reducing RAM/VRAM usage by not having all models in RAM at the same time.
            In this case, the folder needs to be preserved as long as the model exists
            (including when the model is pickled to disk).
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param n_hyperopt_steps: Number of random hyperparameter configs
            that should be used to train models (default=50).
        :param val_metric_name: Name of the validation metric (used for selecting the best epoch).
            Not used for all models but at least for RealMLP and probably TabM.
            Defaults are 'class_error' for classification and 'rmse' for regression.
            Main available classification metrics (all to be minimized): 'class_error', 'cross_entropy', '1-auc_ovo',
            '1-auc_ovr', '1-auc_mu', 'brier', '1-balanced_accuracy', '1-mcc', 'ece'.
            Main available regression metrics: 'rmse', 'mae', 'max_error',
            'pinball(0.95)' (also works with other quantiles specified directly in the string).
            For more metrics, we refer to `models.training.metrics.Metrics.apply()`.
        :param calibration_method: Post-hoc calibration method (only for classification) (default=None).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param hpo_space_name: Name of the HPO space (default='default').
            The search space used in the paper for RealMLP is 'default'. However, we recommend using 'tabarena' for the best results.
        :param n_caruana_steps: Number of weight update iterations for Caruana et al. weighted ensembling (default=40).
            This parameter is only used when use_caruana_ensembling=True.
        :param n_epochs: Number of epochs to train for each NN (default=None).
            If set, it will override the values from the search space. (Might be ignored for non-RealMLP methods.)
        :param use_caruana_ensembling: Whether to use the algorithm by Caruana et al. (2004)
            to select a weighted ensemble of models instead of only selecting the best model (default=False).
        :param train_metric_name: Name of the training metric
            (default is cross_entropy for classification and mse for regression).
            For regression, pinball/multi_pinball can be used instead. (Might be ignored for non-RealMLP methods.)
        :param time_limit_s: Time limit in seconds (default=None).
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_hyperopt_steps = n_hyperopt_steps
        self.val_metric_name = val_metric_name
        self.calibration_method = calibration_method
        self.hpo_space_name = hpo_space_name
        self.n_caruana_steps = n_caruana_steps
        self.n_epochs = n_epochs
        self.use_caruana_ensembling = use_caruana_ensembling
        self.train_metric_name = train_metric_name
        self.time_limit_s = time_limit_s

class TabMConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, Path]] = None, verbosity: int = 0,
                 arch_type: Optional[str] = None,
                 tabm_k: Optional[int] = None,
                 num_emb_type: Optional[str] = None,
                 num_emb_n_bins: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 lr: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 n_epochs: Optional[int] = None,
                 patience: Optional[int] = None,
                 d_embedding: Optional[int] = None,
                 d_block: Optional[int] = None,
                 n_blocks: Optional[Union[str, int]] = None,
                 dropout: Optional[float] = None,
                 compile_model: Optional[bool] = None,
                 allow_amp: Optional[bool] = None,
                 tfms: Optional[List[str]] = None,
                 gradient_clipping_norm: Optional[Union[float, Literal['none']]] = None,
                 calibration_method: Optional[str] = None,
                 share_training_batches: Optional[bool] = None,
                 val_metric_name: Optional[str] = None,
                 train_metric_name: Optional[str] = None,
                 ):
        """

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices are given in fit(), `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification). If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Temporary folder in which data can be stored during fit().
            (Currently unused for TabM and variants.) If None, methods generally try to not store intermediate data.
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param arch_type: Architecture type for TabM, one of ['tabm', 'tabm-mini', 'tabm-normal', 'tabm-mini-normal', 'plain'].
        :param tabm_k: Value of $k$ (number of memory-efficient ensemble members). Default is 32.
        :param num_emb_type: Type of numerical embedding, one of ['none', 'pwl']. Default is 'none'.
            'pwl' stands for piecewise linear embeddings.
        :param num_emb_n_bins: Number of bins for piecewise linear embeddings (default=48).
        Only used when piecewise linear numerical embeddings are used.
        Must be at most the number of training samples, but >1.
        :param batch_size: Batch size, default is 256.
        :param lr: Learning rate, default is 2e-3.
        :param weight_decay: Weight decay, default is 0.
        :param n_epochs: Maximum number of epochs (if early stopping doesn't apply). Default is 1 billion.
        :param patience: Patience for early stopping. Default is 16
        :param d_embedding: Embedding dimension for numerical embeddings.
        :param d_block: Hidden layer size.
        :param n_blocks: Number of linear layers, or 'auto'. Default is 'auto', which will use
            3 when num_emb_type=='none' and 2 otherwise.
        :param dropout: Dropout probability. Default is 0.1.
        :param compile_model: Whether torch.compile should be applied to the model (default=False).
        :param allow_amp: Whether automatic mixed precision should be used if the device is a GPU (default=False).
        :param tfms: Preprocessing transformations, see models.nn_models.models.PreprocessingFactory.
            Default is ['quantile_tabr']. Categorical values will be one-hot encoded by the model.
            Note that in the original experiments, it seems that when cat_policy='ordinal',
            the ordinal-encoded categorical values will later be one-hot encoded by the model.
        :param gradient_clipping_norm: Norm for gradient clipping.
            Default is None from the example code (no gradient clipping), but the experiments from the paper use 1.0.
        :param calibration_method: Post-hoc calibration method (only for classification).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param share_training_batches: New in v1.4.1: Whether TabM should use the same training samples
            for each model in the batch (default=False).
            We adopt the default value False from the newer version of TabM,
            while the old code (prior to 1.4.1) was equivalent to share_training_batches=True,
            except that the new code also excludes certain parameters from weight decay.
        :param val_metric_name: Name of the validation metric used for early stopping.
            For classification, the default is 'class_error' but could be 'cross_entropy', 'brier', '1-auc_ovr' etc.
            For regression, the default is 'rmse' but could be 'mae'.
        :param train_metric_name: Name of the metric (loss) used for training.
            For classification, the default is 'cross_entropy'.
            For regression, it is 'mse' but could be set to something like 'multi_pinball(0.05,0.95)'.
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity

        self.arch_type = arch_type
        self.num_emb_type = num_emb_type
        self.num_emb_n_bins = num_emb_n_bins
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_embedding = d_embedding
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.tabm_k = tabm_k
        self.allow_amp = allow_amp
        self.tfms = tfms
        self.gradient_clipping_norm = gradient_clipping_norm
        self.calibration_method = calibration_method
        self.share_training_batches = share_training_batches
        self.val_metric_name = val_metric_name
        self.train_metric_name = train_metric_name

class TabM_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    """
    HPO spaces ('default', 'tabarena') use TabM-mini with numerical embeddings
    """
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        # from pytabkit.models.alg_interfaces.tabm_interface import RandomParamsTabMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsTabMAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

class TabM_D_Classifier(TabMConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.TABM_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        # from pytabkit.models.alg_interfaces.tabm_interface import TabMSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

def get_tabm_auto_batch_size(n_train: int) -> int:
    # by Yury Gorishniy, inferred from the choices in the TabM paper.
    if n_train < 2_800:
        return 32
    if n_train < 4_500:
        return 64
    if n_train < 6_400:
        return 128
    if n_train < 32_000:
        return 256
    if n_train < 108_000:
        return 512
    return 1024

class TabMSubSplitInterface(SingleSplitAlgInterface):
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        raise NotImplementedError()

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1
        assert idxs_list[0].n_trainval_splits == 1

        seed = idxs_list[0].sub_split_seeds[0]
        # print(f'Setting seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # hyperparams
        arch_type = self.config.get('arch_type', 'tabm')
        num_emb_type = self.config.get('num_emb_type', 'none')
        n_epochs = self.config.get('n_epochs', 1_000_000_000)
        patience = self.config.get('patience', 16)
        batch_size = self.config.get('batch_size', 256)
        compile_model = self.config.get('compile_model', False)
        lr = self.config.get('lr', 2e-3)
        d_embedding = self.config.get('d_embedding', 16)
        d_block = self.config.get('d_block', 512)
        dropout = self.config.get('dropout', 0.1)
        tabm_k = self.config.get('tabm_k', 32)
        allow_amp = self.config.get('allow_amp', False)
        n_blocks = self.config.get('n_blocks', 'auto')
        num_emb_n_bins = self.config.get('num_emb_n_bins', 48)
        # set default to True for backward compatibility
        share_training_batches = self.config.get("share_training_batches", False)
        val_metric_name = self.config.get('val_metric_name', None)
        train_metric_name = self.config.get('train_metric_name', None)

        weight_decay = self.config.get('weight_decay', 0.0)
        gradient_clipping_norm = self.config.get('gradient_clipping_norm', None)

        TaskType = Literal['regression', 'binclass', 'multiclass']

        n_train = idxs_list[0].n_train
        n_classes = ds.get_n_classes()
        cat_cardinalities = ds.tensor_infos['x_cat'].get_cat_sizes().numpy().tolist()
        task_type: TaskType = 'regression' if n_classes == 0 else ('binclass' if n_classes == 2 else 'multiclass')
        device = interface_resources.gpu_devices[0] if len(interface_resources.gpu_devices) >= 1 else 'cpu'
        device = torch.device(device)

        if num_emb_n_bins >= n_train:
            print(f'Reducing num_emb_n_bins to be smaller than n_train')
            num_emb_n_bins = n_train-1

        if val_metric_name is None:
            val_metric_name = 'rmse' if task_type == 'regression' else 'class_error'

        if batch_size == "auto":
            batch_size = get_tabm_auto_batch_size(n_train=n_train)

        self.n_classes_ = n_classes
        self.task_type_ = task_type
        self.device_ = device

        # create preprocessing factory
        factory = self.config.get('factory', None)
        if 'tfms' not in self.config:
            self.config['tfms'] = ['quantile_tabr']
        if factory is None:
            factory = PreprocessingFactory(**self.config)

        if idxs_list[0].val_idxs is None:
            raise ValueError(f'Training without validation set is currently not implemented')

        ds_parts = {'train': ds.get_sub_dataset(idxs_list[0].train_idxs[0]),
                    'val': ds.get_sub_dataset(idxs_list[0].val_idxs[0]),
                    # 'test': ds.get_sub_dataset(idxs_list[0].test_idxs)
                    }

        part_names = ['train', 'val']  # no test
        non_train_part_names = ['val']

        # transform according to factory
        fitter: Fitter = factory.create(ds.tensor_infos)
        self.tfm_, ds_parts['train'] = fitter.fit_transform(ds_parts['train'])
        for part in non_train_part_names:
            ds_parts[part] = self.tfm_(ds_parts[part])

        # filter out numerical columns with only a single value
        x_cont_train = ds_parts['train'].tensors['x_cont']

        for part in part_names:
            ds_parts[part] = ds_parts[part].to(device)

        # mask of which columns are not constant
        self.num_col_mask_ = ~torch.all(x_cont_train == x_cont_train[0:1, :], dim=0)

        for part in part_names:
            ds_parts[part].tensors['x_cont'] = ds_parts[part].tensors['x_cont'][:, self.num_col_mask_]
            # tensor infos are not correct anymore, but might not be used either

        # update
        n_cont_features = ds_parts['train'].tensors['x_cont'].shape[1]

        Y_train = ds_parts['train'].tensors['y'].clone()
        if task_type == 'regression':
            assert Y_train.shape[-1] == 1
            self.y_mean_ = ds_parts['train'].tensors['y'].mean(dim=0, keepdim=True).item()
            self.y_std_ = ds_parts['train'].tensors['y'].std(dim=0, keepdim=True, correction=0).item()
            self.y_max_ = ds_parts['train'].tensors['y'].max().item()
            self.y_min_ = ds_parts['train'].tensors['y'].min().item()

            Y_train = (Y_train - self.y_mean_) / (self.y_std_ + 1e-30)

        data = {part: utils.join_dicts(
            dict(x_cont=ds_parts[part].tensors['x_cont'], y=ds_parts[part].tensors['y']),
            dict(x_cat=ds_parts[part].tensors['x_cat']) if ds.tensor_infos['x_cat'].get_n_features() > 0 else dict())
                for part in part_names}

        # adapted from https://github.com/yandex-research/tabm/blob/main/example.ipynb

        # Automatic mixed precision (AMP)
        # torch.float16 is implemented for completeness,
        # but it was not tested in the project,
        # so torch.bfloat16 is used by default.
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        # Changing False to True will result in faster training on compatible hardware.
        amp_enabled = allow_amp and amp_dtype is not None and device.type == 'cuda'
        grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

        # fmt: off
        logger.log(1,
            f'Device:        {device.type.upper()}'
            f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
            f'\ntorch.compile: {compile_model}'
        )
        # fmt: on
        pass

        # Choose one of the two configurations below.

        # TabM
        bins = None if num_emb_type != 'pwl' or n_cont_features == 0 else rtdl_num_embeddings.compute_bins(data['train']['x_cont'], n_bins=num_emb_n_bins)
        d_out = n_classes if n_classes > 0 else 1
        if train_metric_name is not None and train_metric_name.startswith('multi_pinball'):
            d_out = train_metric_name.count(',')+1

        model = Model(
            n_num_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=d_out,
            backbone={
                'type': 'MLP',
                'n_blocks': n_blocks if n_blocks != 'auto' else (3 if bins is None else 2),
                'd_block': d_block,
                'dropout': dropout,
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': d_embedding,
                    'activation': False,
                    'version': 'B',
                }
            ),
            arch_type=arch_type,
            k=tabm_k,
            share_training_batches=share_training_batches,
        ).to(device)

        # import tabm
        # num_embeddings = None if bins is None else rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        #     bins=bins,
        #     d_embedding=d_embedding,
        #     activation=False,
        #     version='B',
        # )
        # model = tabm.TabM(
        #     n_num_features=n_cont_features,
        #     cat_cardinalities=cat_cardinalities,
        #     d_out = n_classes if n_classes > 0 else 1,
        #     num_embeddings = num_embeddings,
        #     n_blocks=n_blocks if n_blocks != 'auto' else (3 if bins is None else 2),
        #     d_block=d_block,
        #     dropout=dropout,
        #     arch_type=arch_type,
        #     k=tabm_k,
        #     # todo: can introduce activation
        #     share_training_batches=share_training_batches,  # todo: disappeared?
        # )
        optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=lr, weight_decay=weight_decay)


        if compile_model:
            # NOTE
            # `torch.compile` is intentionally called without the `mode` argument
            # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
            model = torch.compile(model)
            evaluation_mode = torch.no_grad
        else:
            evaluation_mode = torch.inference_mode

        @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
        def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
            return (
                model(
                    data[part]['x_cont'][idx],
                    data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
                )
                .float()
            )

        if train_metric_name is None:
            train_metric_name = 'mse' if self.n_classes_ == 0 else 'cross_entropy'

        if train_metric_name == 'mse':
            base_loss_fn = torch.nn.functional.mse_loss
        elif train_metric_name == 'cross_entropy':
            base_loss_fn = lambda a, b: nn.functional.cross_entropy(a, b.squeeze(-1))
        else:
            base_loss_fn = functools.partial(Metrics.apply, metric_name=train_metric_name)

        def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # TabM produces k predictions per object. Each of them must be trained separately.
            # (regression)     y_pred.shape == (batch_size, k)
            # (classification) y_pred.shape == (batch_size, k, n_classes)
            k = y_pred.shape[1]
            # print(f'{y_pred.flatten(0, 1).shape=}, {y_true.shape=}')
            return base_loss_fn(
                y_pred.flatten(0, 1),
                y_true.repeat_interleave(k) if model.share_training_batches else y_true,
            )

        @evaluation_mode()
        def evaluate(part: str) -> float:
            model.eval()

            # When using torch.compile, you may need to reduce the evaluation batch size.
            eval_batch_size = 1024
            y_pred: torch.Tensor = (
                torch.cat(
                    [
                        apply_model(part, idx)
                        for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                    ]
                )
            )
            if task_type == 'regression':
                # Transform the predictions back to the original label space.
                y_pred = y_pred * self.y_std_ + self.y_mean_

            # Compute the mean of the k predictions.
            average_logits = self.config.get('average_logits', False)
            if average_logits:
                y_pred = y_pred.mean(dim=1)
            if task_type != 'regression':
                # For classification, the mean must be computed in the probability space.
                y_pred = y_pred.softmax(dim=-1)
            if not average_logits:
                y_pred = y_pred.mean(dim=1)

            y_true = data[part]['y'].cpu()
            y_pred = y_pred.cpu()

            if task_type == 'regression' and len(y_true.shape) == 1:
                y_true = y_true.unsqueeze(-1)
            if task_type == 'regression' and len(y_pred.shape) == 1:
                y_pred = y_pred.unsqueeze(-1)
            # use minus so higher=better
            if val_metric_name == 'cross_entropy':
                score = -log_loss(y_true.numpy(), y_pred.numpy())
            else:
                score = -Metrics.apply(y_pred, y_true, val_metric_name).item()
            return float(score)  # The higher -- the better.

        # print(f'Test score before training: {evaluate("test"):.4f}')

        epoch_size = math.ceil(n_train / batch_size)
        best = {
            'val': -math.inf,
            # 'test': -math.inf,
            'epoch': -1,
        }
        best_params = [p.clone() for p in model.parameters()]
        # Early stopping: the training stops when
        # there are more than `patience` consecutive bad updates.
        remaining_patience = patience

        try:
            if self.config.get('verbosity', 0) >= 1:
                from tqdm.std import tqdm
            else:
                tqdm = lambda arr, desc: arr
        except ImportError:
            tqdm = lambda arr, desc: arr

        logger.log(1, '-' * 88 + '\n')
        for epoch in range(n_epochs):
            batches = (
                torch.randperm(n_train, device=device).split(batch_size)
                if model.share_training_batches
                else [
                    x.transpose(0, 1).flatten()
                    for x in torch.rand((model.k, n_train), device=device).argsort(dim=1).split(batch_size, dim=1)
                ]
            )

            model.train()
            for batch_idx in tqdm(batches, desc=f"Epoch {epoch}"):
                optimizer.zero_grad(set_to_none=True)

                preds = apply_model('train', batch_idx)
                loss = loss_fn(preds, Y_train[batch_idx])

                if grad_scaler is None:
                    loss.backward()
                    if gradient_clipping_norm not in (None, 'none'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)  # type: ignore
                    optimizer.step()
                else:
                    grad_scaler.scale(loss).backward()
                    if gradient_clipping_norm not in (None, 'none'):
                        # unscale before clipping so the grads are in FP32
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)  # type: ignore
                    grad_scaler.step(optimizer)
                    grad_scaler.update()


            val_score = evaluate('val')
            # test_score = evaluate('test')
            # logger.log(1, f'(val) {val_score:.4f} (test) {test_score:.4f}')
            logger.log(1, f'(val) {val_score:.4f}')

            if val_score > best['val']:
                logger.log(1, '🌸 New best epoch! 🌸')
                # best = {'val': val_score, 'test': test_score, 'epoch': epoch}
                best = {'val': val_score, 'epoch': epoch}
                remaining_patience = patience
                with torch.no_grad():
                    for bp, p in zip(best_params, model.parameters()):
                        bp.copy_(p)
            else:
                remaining_patience -= 1

            if remaining_patience < 0:
                break

            logger.log(1, '')

        logger.log(1, '\n\nResult:')
        logger.log(1, str(best))

        logger.log(1, f'Restoring best model')
        with torch.no_grad():
            for bp, p in zip(best_params, model.parameters()):
                p.copy_(bp)

        self.model_ = model

        return None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        self.model_.eval()

        ds = self.tfm_(ds).to(self.device_)

        ds.tensors['x_cont'] = ds.tensors['x_cont'][:, self.num_col_mask_]

        eval_batch_size = 1024
        with torch.no_grad():
            y_pred: torch.Tensor = (
                torch.cat(
                    [
                        self.model_(
                            ds.tensors['x_cont'][idx],
                            ds.tensors['x_cat'][idx] if not ds.tensor_infos['x_cat'].is_empty() else None,
                        )
                        .float()
                        for idx in torch.arange(ds.n_samples, device=self.device_).split(
                        eval_batch_size
                    )
                    ]
                )
            )
        if self.task_type_ == 'regression':
            # Transform the predictions back to the original label space.
            y_pred = y_pred.mean(1)
            y_pred = y_pred * self.y_std_ + self.y_mean_
            if self.config.get('clamp_output', False):
                y_pred = torch.clamp(y_pred, self.y_min_, self.y_max_)
        else:
            average_logits = self.config.get('average_logits', False)
            if average_logits:
                y_pred = y_pred.mean(1)
            else:
                # For classification, the mean must be computed in the probability space.
                y_pred = torch.log(torch.softmax(y_pred, dim=-1).mean(1) + 1e-30)

        return y_pred[None].cpu()  # add n_models dimension

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=2), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_train': 8e-5, 'n_samples*n_features': 8e-8}
        ram_params = {'': 0.15, 'ds_onehot_size_gb': 2.0}
        # gpu_ram_params = {'': 0.3, 'ds_onehot_size_gb': 1.0, 'n_train': 1e-6, 'n_features': 3e-4,
        #                   'cat_size_sum': 2e-3}
        gpu_ram_params = {'': 0.5, 'ds_onehot_size_gb': 5.0, 'n_train': 6e-6, 'n_features': 1.5e-3,  # reduced from 2e-3
                          'cat_size_sum': 1e-4}  # reduced from 1e-3
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=0.02)  # , gpu_ram_params)
        return rc.get_required_resources(ds, n_train=n_train)

class RandomParamsTabMAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        # adapted from Grinsztajn et al. (2022)
        hpo_space_name = self.config.get('hpo_space_name', 'default')
        if hpo_space_name == 'default':
            params = {
                "batch_size": "auto",
                "patience": 16,
                "allow_amp": True,
                "arch_type": "tabm-mini",
                "tabm_k": 32,
                # "gradient_clipping_norm": 1.0, # wasn't correctly implemented so we remove it in v1.7.0
                # this makes it probably slower with numerical embeddings, and also more RAM intensive
                # according to the paper it's not very important but should be a bit better (?)
                "share_training_batches": False,
                "lr": np.exp(rng.uniform(np.log(1e-4), np.log(3e-3))),
                "weight_decay": rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))]),
                "n_blocks": rng.choice([1, 2, 3, 4]),
                "d_block": rng.choice([i for i in range(64, 1024 + 1) if i % 16 == 0]),
                "dropout": rng.choice([0.0, rng.uniform(0.0, 0.5)]),
                # numerical embeddings
                "num_emb_type": "pwl",
                "d_embedding": rng.choice([i for i in range(8, 32 + 1) if i % 4 == 0]),
                "num_emb_n_bins": rng.integers(2, 128, endpoint=True),
            }
        elif hpo_space_name == 'tabarena':
            params = {
                "batch_size": "auto",
                "patience": 16,
                "allow_amp": False,  # only for GPU, maybe we should change it to True?
                "arch_type": "tabm-mini",
                "tabm_k": 32,
                # "gradient_clipping_norm": 1.0, # wasn't correctly implemented so we remove it in v1.7.0
                # this makes it probably slower with numerical embeddings, and also more RAM intensive
                # according to the paper it's not very important but should be a bit better (?)
                "share_training_batches": False,
                "lr": np.exp(rng.uniform(np.log(1e-4), np.log(3e-3))),
                "weight_decay": rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))]),
                # removed n_blocks=1 according to Yury Gurishniy's advice
                "n_blocks": rng.choice([2, 3, 4, 5]),
                # increased lower limit from 64 to 128 according to Yury Gorishniy's advice
                "d_block": rng.choice([i for i in range(128, 1024 + 1) if i % 16 == 0]),
                "dropout": rng.choice([0.0, rng.uniform(0.0, 0.5)]),
                # numerical embeddings
                "num_emb_type": "pwl",
                "d_embedding": rng.choice([i for i in range(8, 32 + 1) if i % 4 == 0]),
                "num_emb_n_bins": rng.integers(2, 128, endpoint=True),
            }
        else:
            raise ValueError(f'Unknown {hpo_space_name=}')
        return params

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([TabMSubSplitInterface(**config) for i in range(n_tv_splits)])

    def get_available_predict_params(self) -> Dict[str, Dict[str, Any]]:
        return TabMSubSplitInterface(**self.config).get_available_predict_params()

    def set_current_predict_params(self, name: str) -> None:
        super().set_current_predict_params(name)