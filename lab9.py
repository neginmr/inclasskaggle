import xgboost as xgb
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('cs-training.csv', index_col= 0, na_values='?')
df = df.fillna(df.mean(axis = 0))

df_test = pd.read_csv('cs-test.csv', index_col= 0, na_values='?')
df_test = df_test.fillna(df_test.mean(axis=0))

X_train = df.drop('SeriousDlqin2yrs',1)
y_train = df.SeriousDlqin2yrs

X_test = df_test.drop('SeriousDlqin2yrs',1)

#X_train = VarianceThreshold(1.3).fit_transform(X_train)
#X_test = VarianceThreshold(1.3).fit_transform(X_test)

xgbmodel = xgb.XGBClassifier(max_depth=4, n_estimators=1000, learning_rate=0.01, min_child_weight=44, gamma=.8, subsample=.4, reg_alpha=.5, colsample_bytree=.4,reg_lambda=.93)
gbdt = xgbmodel.fit(X_train, y_train)
print("Making prediction and saving results...")
preds = xgbmodel.predict_proba(X_test)

print(xgbmodel.feature_importances_)

solution = pd.DataFrame({'id':df_test.index, "Probability":preds[:,1]})
solution.to_csv("vcsubmission.csv", index = False)

df1 = df[df['age'] >40 ]
X_train1 = df1.drop('SeriousDlqin2yrs',1)
y_train1 = df1.SeriousDlqin2yrs

gbdt1 = xgbmodel.fit(X_train1, y_train1)
print("older", xgbmodel.feature_importances_)

df2 = df[df['age'] <= 40 ]
X_train2 = df2.drop('SeriousDlqin2yrs',1)
y_train2 = df2.SeriousDlqin2yrs

gbdt2 = xgbmodel.fit(X_train2, y_train2)
print("younger", xgbmodel.feature_importances_)

