import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from skll import kappa
import os


#outputDf = pd.DataFrame({"first" : [],"second" : []})
#outputDf = outputDf.append(pd.DataFrame({"first" : [2],"second" : [3]}))
os.chdir("/Users/swapnil/work/Kaggle/out/PLIA")
print "hello"

#print outputDf

print int(round(3))



data = pd.read_csv("trans_train.csv")
test = pd.read_csv("trans_test.csv",na_values="NA")
d = DV(sparse = True)

data = data.fillna(-9999)
test = test.fillna(-9999)

trainData, cvData, yTrain, yCv = train_test_split(data,data["Response"], test_size=0.2, random_state=42)

trainData = trainData.drop("Response",axis=1)
trainData = trainData.drop("Id",axis=1)

ftTrain = d.fit_transform(trainData.T.to_dict().values())
ftCv = d.transform(cvData.T.to_dict().values())
ftTest = d.transform(test.T.to_dict().values())

print ftTrain.shape
print ftCv.shape
print ftTest.shape