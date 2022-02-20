import lightgbm as lgb


class Model_select():

    def __init__(self, model_name, categorical_feature, target_col, num_classes, seed):
        self.model_name = model_name
        self.categorical_feature = categorical_feature
        self.target_col = target_col
        self.seed = seed
        self.num_classes=num_classes
        self.model = None

    def create_model(self, **args):
        if(self.model_name == "lightgbm"):
            self.model = lgb.LGBMClassifier(random_state=self.seed)
        elif(self.model_name == "tabtransformer"):
            pass    
        else:
            raise Exception("Model not supported!")

    def train_model(self, train_set):
        if(self.model_name == "lightgbm"):
            self.model.fit(
                train_set.drop(columns=self.target_col),
                train_set[self.target_col],
                categorical_feature=self.categorical_feature)
        else:
            raise Exception("Model not supported!")

    def test_model(self, test_set):
        if(self.model_name == "lightgbm"):
            test_pred = self.model.predict(test_set.drop(columns=self.target_col))
        else:
            raise Exception("Model not supported!")
        return test_pred
