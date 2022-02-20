from pytorch_tabular_main.pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
import torch
import numpy as np
from torch.functional import norm
from sklearn.datasets import fetch_covtype


class Tabtransformer:

    def __init__(
            self,
            num_classes,
            shared_embedding_fraction=0.25,
            share_embedding=True,
            share_embedding_strategy="add",
            task="classification"):

        self.data_config = DataConfig(
            target=target_name,
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
            continuous_feature_transform=None,  # "quantile_normal",
            normalize_continuous_features=False,
        )

        self.model_config = TabTransformerConfig(
            task="classification",
            metrics=["f1", "accuracy"],
            share_embedding=True,
            share_embedding_strategy="add",
            shared_embedding_fraction=0.25,
            metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
        )
