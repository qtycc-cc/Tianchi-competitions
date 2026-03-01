import numpy as np
from typing import List, Dict, Any
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, RandomParamsAlgInterface
from pytabkit.models.alg_interfaces.ensemble_interfaces import CaruanaEnsembleAlgInterface, AlgorithmSelectionAlgInterface
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.tabm_interface import TabMSubSplitInterface
from pytabkit.models.sklearn.sklearn_base import AlgInterfaceClassifier
from pytabkit.models.sklearn.sklearn_interfaces import RealMLPHPOConstructorMixin

class MyRandomParamsTabMAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        # adapted from Grinsztajn et al. (2022)
        hpo_space_name = self.config.get('hpo_space_name', 'default')
        if hpo_space_name == 'default':
            params = {
                "batch_size": "auto",
                "patience": 16,
                "allow_amp": True,
                "arch_type": "tabm",
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
                "arch_type": "tabm",
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

class My_TabM_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    """
    HPO spaces ('default', 'tabarena') use TabM-mini with numerical embeddings
    """
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([MyRandomParamsTabMAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']
