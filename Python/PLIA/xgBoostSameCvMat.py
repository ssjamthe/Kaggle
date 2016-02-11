import xgboost as xgb
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

dtrain=xgb.DMatrix(ftTrain,label=yTrain)
dCv = xgb.DMatrix(ftCv)
dTest = xgb.DMatrix(ftTest)




ntrees = [5000,3000,2000,100,300,500,700,150,200,10,20,30,40,50,60,70,80]
depths = [5,6,7,8,10,12,14,18,24]
etas = [0.2,0.1,0.11,0.13,0.17,0.23,0.3]

trials = []

for i,ntree in enumerate(ntrees):
    for j,depth in enumerate(depths):
        for k,eta in enumerate(etas):
            trials.append((ntree,depth,eta))

obs = pd.DataFrame({"kappa" : [], "kappaTrain" : [], "ntree" : [], "depth" : [], "eta" : []})

currIter = 0
bestKappa = -2
bestIter = 0
for i,trial in enumerate(trials):
    
    currNtree = trial[0]
    currDepth = trial[1]
    currEta = trial[2]

    param = {'max_depth':currDepth, 'eta':currEta, 'silent':1,'eval_metric': 'rmse','objective':'reg:linear'}
    
    watchlist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, currNtree, watchlist)
    trainPred = bst.predict(dtrain)
    trainPred = [adjustResponse(resp) for resp in trainPred]
    kTrain = kappa(yTrain,trainPred,weights="quadratic")
    cvPred = bst.predict(dCv)
    cvPred = [adjustResponse(resp) for resp in cvPred]
    k = kappa(yCv,cvPred,weights="quadratic")
    trainPred = bst.predict(dTest)
    trainPred = [adjustResponse(resp) for resp in trainPred]

    currIterLog = "iter=" + str(currIter) + ",k=" + str(k) + ",ktrain=" + str(kTrain) + ",ntree=" + str(currNtree) + ", depth=" + str(currDepth) + ",eta=" + str(currEta)+ ",best was " + str(bestKappa) + " best iter was " + str(bestIter)
    print currIterLog

    obs = obs.append(pd.DataFrame({"kappa" : [k], "kappaTrain" : [kTrain], "ntree" : [currNtree], "depth" : [currDepth], "eta" : [currEta]}))

    

    if bestKappa < k:
    	bestKappa = k
    	bestIter = currIter
    	print("Best till now " + currIterLog)
    	d = {"Id":test["Id"],"Response":trainPred}
    	outputDf = pd.DataFrame(data = d)
    	fileName = "predictions/python/xgBoostSameCvMat/predBoost" + "_" + str(currNtree) + "_" + str(currDepth) + "_" + str(currEta) + "_" + str(currIter)
    	outputDf.to_csv(fileName,index_label=False,index=False)
    	
    	cvPredData = {"Response" : yCv,"predResponse" : cvPred}
        cvOutputDf = pd.DataFrame(data = cvPredData)
        fileName = "cvPredictions/python/xgBoostSameCvMat/predBoost" + "_" + str(currNtree) + "_" + str(currDepth) + "_" + str(currEta) + "_" + str(currIter)
        cvOutputDf.to_csv(fileName,index_label=False,index=False)

    currIter = currIter + 1









