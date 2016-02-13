import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from skll import kappa
from sklearn.externals import joblib
import numpy as np


import os

def adjustResponse(resp):
	if resp < 1:
		return 1
	elif resp > 8:
		return 8
	else:
		return int(round(resp))


os.chdir("/Users/swapnil/work/Kaggle/out/PLIA")

print "Hello"
data = pd.read_csv("trans_train.csv")
test = pd.read_csv("trans_test.csv",na_values="NA")
d = DV(sparse = False)

data = data.fillna(-9999)
test = test.fillna(-9999)

dataPred = data["Response"]

data = data.drop("Response",axis=1)
data = data.drop("Id",axis=1)


ftData = d.fit_transform(data.T.to_dict().values())
ftTest = d.transform(test.T.to_dict().values())

msk = np.random.rand(ftData.shape[0]) < 0.8

ftTrain =  ftData[msk,0:ftData.shape[1]]
ftCv = ftData[~msk,0:ftData.shape[1]]

yTrain = dataPred[msk]
yCv = dataPred[~msk]

clf = RandomForestRegressor(n_estimators=5000)
clf.fit(ftTrain,yTrain)

joblib.dump(clf,"RFModel5000_sameCvMat/RFModel5000_sameCvMat")


trainPred = clf.predict(ftTrain)
trainPred = [adjustResponse(resp) for resp in trainPred]
kTrain = kappa(yTrain,trainPred,weights="quadratic")
print "Ktrain : " + str(kTrain)

cvPred = clf.predict(ftCv)
cvPred = [adjustResponse(resp) for resp in cvPred]
kCv = kappa(yCv,cvPred,weights="quadratic")
print "cvPred : " + str(kCv)