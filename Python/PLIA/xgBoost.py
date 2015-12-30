import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
import os
os.chdir("/Users/swapnil.jamthe/work/Kaggle/out/PLIA")
print "Hello"
data = pd.read_csv("trans_train.csv")
test = pd.read_csv("trans_test.csv",na_values="NA")
d = DV(sparse = True)
print test["Medical_History_15"]
data = data.fillna(-9999)
test = test.fillna(-9999)
print test["Medical_History_15"]

ftTrain = d.fit_transform(data.T.to_dict().values())
