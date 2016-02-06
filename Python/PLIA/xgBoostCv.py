import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from skll import kappa


import os


def adjustResponse(resp):
	if resp < 1:
		return 1
	elif resp > 8:
		return 8
	else:
		return int(round(resp))

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    k = kappa(labels,preds,weights="quadratic")
    return 'error',k


os.chdir("/Users/swapnil.jamthe/work/Kaggle/out/PLIA")
print "Hello"
data = pd.read_csv("trans_train_sumMK.csv")
test = pd.read_csv("trans_train_sumMK.csv",na_values="NA")
d = DV(sparse = True)

data = data.fillna(-9999)
test = test.fillna(-9999)

trainData, cvData, yTrain, yCv = train_test_split(data,data["Response"], test_size=0.2, random_state=42)

trainData = trainData.drop("Response",axis=1)
trainData = trainData.drop("Id",axis=1)

ftTrain = d.fit_transform(trainData.T.to_dict().values())
ftCv = d.transform(cvData.T.to_dict().values())
ftTest = d.transform(test.T.to_dict().values())

#print ftCv.columns

dtrain=xgb.DMatrix(ftTrain,label=yTrain)
dCv = xgb.DMatrix(ftCv)
dTest = xgb.DMatrix(ftTest)

param = {'max_depth':2, 'eta':1, 'silent':1,'eval_metric': 'rmse'}

bst = xgb.cv(param, dtrain, 2, nfold = 5, seed = 1988, feval=evalerror)

trainPred = bst.predict(dtrain)
trainPred = [adjustResponse(resp) for resp in trainPred]
kTrain = kappa(yTrain,trainPred,weights="quadratic")
cvPred = bst.predict(dCv)
cvPred = [adjustResponse(resp) for resp in cvPred]
k = kappa(yCv,cvPred,weights="quadratic")

print "k=" + str(k) + ",kTrain=" + str(kTrain)



