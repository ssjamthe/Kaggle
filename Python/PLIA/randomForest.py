import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skll import kappa
from sklearn.externals import joblib


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
data = pd.read_csv("imp_train_10iter_50percent")
test = pd.read_csv("trans_test.csv",na_values="NA")
d = DV(sparse = True)

test = test.fillna(-9999)


trainData, cvData, yTrain, yCv = train_test_split(data,data["Response"], test_size=0.2, random_state=42)

trainData = trainData.drop("Response",axis=1)
trainData = trainData.drop("Id",axis=1)

ftTrain = d.fit_transform(trainData.T.to_dict().values())
ftCv = d.transform(cvData.T.to_dict().values())
ftTest = d.transform(test.T.to_dict().values())

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(ftTrain,yTrain)

joblib.dump(clf,"rfmodels/RFModel1000")


trainPred = clf.predict(ftTrain)
trainPred = [adjustResponse(resp) for resp in trainPred]
kTrain = kappa(yTrain,trainPred,weights="quadratic")
print "Ktrain : " + str(kTrain)

cvPred = clf.predict(ftCv)
cvPred = [adjustResponse(resp) for resp in cvPred]
kCv = kappa(yCv,cvPred,weights="quadratic")
print "cvPred : " + str(kCv)





