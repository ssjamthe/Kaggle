import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
import os
os.chdir("/Users/swapnil/work/Kaggle/out/PLIA")
print "Hello"
data = pd.read_csv("trans_train.csv")
test = pd.read_csv("trans_test.csv",na_values="NA")
d = DV(sparse = True)

data = data.fillna(-9999)
test = test.fillna(-9999)

xData = data.drop("Response",axis=1)
print xData.columns

trainData, cvData, yTrain, yCv = train_test_split(xData,data["Response"], test_size=0.8, random_state=42)
ftTrain = d.fit_transform(trainData.T.to_dict().values())
