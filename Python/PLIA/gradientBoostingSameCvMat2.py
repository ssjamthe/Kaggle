import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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


clf = GradientBoostingRegressor(n_estimators=3000,max_depth=8,min_samples_split=30,min_samples_leaf=8,max_features="auto",verbose=1,random_state=1988)
clf.fit(ftTrain,yTrain)

#Ktrain : 0.969452696665
#cvPred : 0.593926668587


trainPred = clf.predict(ftTrain)
trainPred = [adjustResponse(resp) for resp in trainPred]
kTrain = kappa(yTrain,trainPred,weights="quadratic")
print "Ktrain : " + str(kTrain)

cvPred = clf.predict(ftCv)
cvPred = [adjustResponse(resp) for resp in cvPred]
kCv = kappa(yCv,cvPred,weights="quadratic")
print "cvPred : " + str(kCv)

joblib.dump(clf,"GBMModel3000_2_sameCvMat/GBMModel3000_2_sameCvMat")


clf = GradientBoostingRegressor(n_estimators=4000,max_depth=8,min_samples_split=30,min_samples_leaf=8,max_features="auto",verbose=1,random_state=1988)
clf.fit(ftTrain,yTrain)

#Ktrain : 0.983890005566
#cvPred : 0.592046743359


trainPred = clf.predict(ftTrain)
trainPred = [adjustResponse(resp) for resp in trainPred]
kTrain = kappa(yTrain,trainPred,weights="quadratic")
print "Ktrain : " + str(kTrain)

cvPred = clf.predict(ftCv)
cvPred = [adjustResponse(resp) for resp in cvPred]
kCv = kappa(yCv,cvPred,weights="quadratic")
print "cvPred : " + str(kCv)

joblib.dump(clf,"GBMModel4000_2_sameCvMat/GBMModel4000_2_sameCvMat")