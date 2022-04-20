# Covid_biochem

This code is for the paper **COVID-Net Biochem: An Explainability-driven Framework
to Building Machine Learning Models for Predicting
Survival and Kidney Injury of COVID-19 Patients from
Clinical and Biochemistry Data**

This code is for doing survival prediction and Acute Kidney Injury prediction of COVID-19 patients. For survival prediction, use ```main```  branch. For Acute Kidney Injury predictio, use ```hossein/kidney```  branch.

The main file for doing prediction is ```main.py```.
```main.py``` has argparser where you can set the model for prediction, change learning rate, select test size and other configs.

Currently we support the following models:

1- [TabTransformer](https://arxiv.org/abs/2012.06678) by selecting: ```--model tabtransformer```

2- [FTTransformer](https://arxiv.org/pdf/2106.11959.pdf) by selecting: ```--model FTTransformer```

3- [TabNet](https://arxiv.org/abs/1908.07442) by selecting: ```--model tabnet```

4- [CatBoost](https://arxiv.org/abs/1706.09516) by selecting: ```--model catboost```

5- [LightGBM](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) by selecting: ```--model lightgbm```

6- [XGBoost](https://arxiv.org/abs/1603.02754) by selecting: ```--model XGBoost```

For a simple run of selecting TabTransformer model with learning rate 0.00015 and batch size 256 and validation size 0.05 for 200 epochs:

```main.py --model tabtransformer --lr 0.00015 --batch_size 256 --epochs 200 --val_size 0.05```


