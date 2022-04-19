import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from models_parent.swin import Swin
from models_parent.tabtransformer import Tabtransformer
from utility.utils import print_metrics


class Model_select():

    def __init__(self, model_name, num_col_names, categorical_feature, target_col,
                 num_classes, lr_scheduler, init_lr, seed):
        self.model_name = model_name
        self.categorical_feature = categorical_feature
        self.num_col_names = num_col_names
        self.target_col = target_col
        self.seed = seed
        self.num_classes = num_classes
        self.model = None
        self.lr_scheduler = lr_scheduler
        self.init_lr = init_lr

    def create_model(self, **kwargs):
        if (self.model_name == "lightgbm"):
            self.model = lgb.LGBMClassifier(random_state=self.seed)
        elif (self.model_name == "XGBoost"):
            self.model = XGBClassifier()
            self.model.set_params(seed=self.seed, tree_method='gpu_hist')
        elif (self.model_name == "catboost"):
            self.model = CatBoostClassifier(iterations=100,
                                            learning_rate=0.3,
                                            task_type="GPU",
                                            devices=kwargs["cuda_n"])
        elif (self.model_name == "tabtransformer" or self.model_name == "FTTransformer" or
              self.model_name == "tabnet"):
            self.model = Tabtransformer(
                model_name=self.model_name,
                num_classes=self.num_classes,
                target_name=self.target_col,
                num_col_names=self.num_col_names,
                cat_col_names=self.categorical_feature,
                lr_scheduler=self.lr_scheduler,
                init_lr=self.init_lr,
            )
        elif (self.model_name == "swintransformer"):
            self.model = Swin(num_classes=self.num_classes, device=kwargs["device"])
        else:
            raise Exception("Model not supported!")

    def train_model(self, train_set, **kwargs):
        if (self.model_name == "lightgbm"):
            self.model.fit(train_set.drop(columns=self.target_col),
                           train_set[self.target_col],
                           categorical_feature=self.categorical_feature,
                           eval_set=(kwargs["val_set"].drop(columns=self.target_col),
                                     kwargs["val_set"][self.target_col]))
        elif (self.model_name == "XGBoost"):
            self.model.fit(train_set.drop(columns=self.target_col),
                           train_set[self.target_col],
                           eval_set=[(train_set.drop(columns=self.target_col),
                                      train_set[self.target_col]),
                                     (kwargs["val_set"].drop(columns=self.target_col),
                                      kwargs["val_set"][self.target_col])])
        elif (self.model_name == "catboost"):
            self.model.fit(train_set.drop(columns=self.target_col),
                           train_set[self.target_col],
                           cat_features=self.categorical_feature)
        elif (self.model_name == "tabtransformer" or self.model_name == "FTTransformer" or
              self.model_name == "tabnet"):
            self.model.train(train_set,
                             gradient_clip_val=kwargs["gradient_clip_val"],
                             epochs=kwargs["epochs"],
                             batch_size=kwargs["batch_size"],
                             early_stopping_patience=kwargs["early_stopping_patience"],
                             checkpoints_save_top_k=kwargs["checkpoints_save_top_k"],
                             auto_lr_find=kwargs["auto_lr_find"],
                             cuda_n=kwargs["cuda_n"],
                             seed=kwargs["seed"],
                             val_set=kwargs["val_set"])
        elif (self.model_name == "swintransformer"):
            self.model.train(trainloader=train_set,
                             testloader=kwargs["testset"],
                             epochs=kwargs["epochs"],
                             lr=kwargs["lr"],
                             milestones=kwargs["milestones"],
                             gamma=kwargs["gamma"],
                             momentum=kwargs["momentum"],
                             weight_decay=kwargs["weight_decay"])
        else:
            raise Exception("Model not supported!")

    def test_model(self, test_set):
        if (self.model_name == "lightgbm"):
            test_pred = self.model.predict(test_set.drop(columns=self.target_col))
            print_metrics(test_set[self.target_col], test_pred)
        elif (self.model_name == "XGBoost"):
            test_pred = self.model.predict(test_set.drop(columns=self.target_col))
            print_metrics(test_set[self.target_col], test_pred)
        elif (self.model_name == "catboost"):
            test_pred = self.model.predict(test_set.drop(columns=self.target_col))
            print_metrics(test_set[self.target_col], test_pred)
        elif (self.model_name == "tabtransformer" or self.model_name == "FTTransformer" or
              self.model_name == "tabnet"):
            test_pred = self.model.predict(test_set)
            print_metrics(test_set[self.target_col], test_pred)
        elif (self.model_name == "swintransformer"):
            test_pred = self.model.predict(test_set)
        else:
            raise Exception("Model not supported!")
        return test_pred
