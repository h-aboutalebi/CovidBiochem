import torch
import numpy as np
from pytorch_tabular_main.pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular_main.pytorch_tabular.config import ModelConfig, ExperimentConfig, OptimizerConfig, TrainerConfig, DataConfig
from pytorch_tabular_main.pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular_main.pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy
from torch.functional import norm
from sklearn.datasets import fetch_covtype


class Tabtransformer:

    def __init__(
            self,
            num_classes,
            target_name,
            num_col_names,
            cat_col_names,
            shared_embedding_fraction=0.25,
            share_embedding=True,
            share_embedding_strategy="add",
            task="classification"):
        self.target_name = target_name
        self.data_config = DataConfig(
            target=target_name,
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
            continuous_feature_transform=None,  # "quantile_normal",
            normalize_continuous_features=False,
        )

        self.model_config = TabTransformerConfig(
            task=task,
            metrics=["f1", "accuracy"],
            share_embedding=share_embedding,
            share_embedding_strategy=share_embedding_strategy,
            shared_embedding_fraction=shared_embedding_fraction,
            metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
        )

        # self.experiment_config = ExperimentConfig(project_name="PyTorch Tabular Example",
        #                                           run_name="node_forest_cov",
        #                                           exp_watch="gradients",
        #                                           log_target="wandb",
        #                                           log_logits=True)
        self.optimizer_config = OptimizerConfig()

    def train(self, train, epochs, batch_size):
        trainer_config = TrainerConfig(
            gpus=-1,
            auto_select_gpus=True,
            fast_dev_run=True,
            max_epochs=epochs,
            batch_size=batch_size)
        tabular_model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=trainer_config,
            # experiment_config=experiment_config,
        )
        sampler = get_balanced_sampler(train[self.target_name].values.ravel())
        tabular_model.fit(
            train=train,
            # validation=val,
            # loss=cust_loss,
            train_sampler=sampler)
