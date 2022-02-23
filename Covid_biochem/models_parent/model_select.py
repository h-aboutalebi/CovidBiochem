import lightgbm as lgb

from models_parent.tabtransformer import Tabtransformer
from utility.utils import print_metrics

class Model_select():

    def __init__(self, model_name, num_col_names, categorical_feature, target_col, num_classes, lr_scheduler, seed):
        self.model_name = model_name
        self.categorical_feature = categorical_feature
        self.num_col_names = num_col_names
        self.target_col = target_col
        self.seed = seed
        self.num_classes = num_classes
        self.model = None
        self.lr_scheduler = lr_scheduler

    def create_model(self, **kwargs):
        if(self.model_name == "lightgbm"):
            self.model = lgb.LGBMClassifier(random_state=self.seed)
        elif(self.model_name == "tabtransformer"):
            self.model = Tabtransformer(num_classes=self.num_classes,
                                        target_name=self.target_col,
                                        num_col_names=self.num_col_names,
                                        cat_col_names=self.categorical_feature,
                                        lr_scheduler=self.lr_scheduler,
                                        )
        else:
            raise Exception("Model not supported!")

    def train_model(self, train_set, **kwargs):
        if(self.model_name == "lightgbm"):
            self.model.fit(
                train_set.drop(columns=self.target_col),
                train_set[self.target_col],
                categorical_feature=self.categorical_feature)
        elif(self.model_name == "tabtransformer"):
            self.model.train(
                train_set,
                gradient_clip_val=kwargs["gradient_clip_val"],
                epochs=kwargs["epochs"], 
                batch_size=kwargs["batch_size"],
                early_stopping_patience=kwargs["early_stopping_patience"],
                checkpoints_save_top_k=kwargs["checkpoints_save_top_k"],
                auto_lr_find=kwargs["auto_lr_find"],
                cuda_n=kwargs["cuda_n"], 
                seed=kwargs["seed"],
                )
        else:
            raise Exception("Model not supported!")

    def test_model(self, test_set):
        if(self.model_name == "lightgbm"):
            test_pred = self.model.predict(test_set.drop(columns=self.target_col))
            print_metrics(test_set[self.target_col], test_pred, "Holdout")
        elif(self.model_name == "tabtransformer"):
            test_pred = self.model.predict(test_set)
        else:
            raise Exception("Model not supported!")
        return test_pred
