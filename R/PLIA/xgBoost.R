set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
library(Metrics)
require(xgboost)

roundResponse<-function (resp)
{
  resp<-round(resp)
  grt8<-resp>8
  resp[grt8]<-8
  less1<-resp<1
  resp[less1]<-1
  
  resp
}

setwd("/Users/swapnil/work/Kaggle/out/PLIA")

data<-read.csv("trans_train.csv")

nasFrac<-sapply(names(data),function(x){sum(is.na(data[,x]))/nrow(data)})
data<-data[,nasFrac==0]

data<-select(data,-(Id))
trainingIndex<-createDataPartition(y=data$Response,p=0.8,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

modelMatTrain<-stats::model.matrix(Response~.-1,data = trainingData)
modelMatCv<-model.matrix(Response~.-1,data = cvData)

testFinal<-read.csv("trans_test.csv")
testFinal<-testFinal[,nasFrac[1:length(nasFrac)-1]==0]
modelMatTestFinal<-model.matrix(~.-1,data = testFinal)

obs<-data.frame(kappa=numeric(0),kappaTrain=numeric(0),ntree=integer(0),depth=integer(0),eta=integer(0))

trials<-data.frame(ntree=integer(0),depth=integer(0),eta=integer(0))
trials<-rbind(trials,data.frame(ntree=1,depth=1,eta=0.15))

ntrees<-c(10,20,30,40,50,60,70,80)
depths<-c(3,4,5,6,7,8,10,12,14)
etas<-c(0.05,0.07,0.09,0.11,0.13,0.17,0.23,0.3)

for(nt in  ntrees)
{
  for(dp in depths)
  {
    for(et in etas)
    {
      trials<-rbind(trials,data.frame(ntree=nt,depth=dp,eta=et))
      
    }
  }
}

iter<-0
bestKappa <- -2
bestIter<-0

for(it in 1:nrow(trials))
{
  currEta<-trials[it,"eta"]
  currNtree<-trials[it,"ntree"]
  currDepth<-trials[it,"depth"]
  
  currTrainMat<-modelMatTrain
  currCvMat<-modelMatCv
  currTestMat<-modelMatTestFinal
  
  modelBoost<-xgboost(data = currTrainMat, booster="gbtree",label = trainingData$Response, max.depth = currDepth, eta = currEta, nthread = 2, nround = currNtree, objective = "reg:linear",verbose=0)
  predBoost<-predict(modelBoost,currTestMat)
  predBoost<-roundResponse(predBoost)
  predBoostFrame<-data.frame(Id=testFinal$Id,Response=predBoost)
  predCv<-predict(modelBoost,currCvMat)
  predCv<-roundResponse(predCv)
  k<-ScoreQuadraticWeightedKappa(cvData$Response,predCv,1,8)
  predTrain<-predict(modelBoost,currTrainMat)
  predTrain<-roundResponse(predTrain)
  ktrain<-ScoreQuadraticWeightedKappa(trainingData$Response,predTrain,1,8)
  
  currIterLog<-paste0("iter=",iter, ",k=",k,",ktrain=",ktrain,",ntree=",currNtree,", depth=",currDepth,",eta=",currEta,",best was ",bestKappa," best iter was ",bestIter)
  print(currIterLog)
  obs<-rbind(obs,data.frame(kappa=k,kappaTrain=ktrain,ntree=currNtree,depth=currDepth,eta=currEta))
  
  if(bestKappa < k)
  {
    bestKappa<-k
    bestIter<-iter
    
    write.table(predBoostFrame,file = paste0("predictions/xgBoost/predBoost","_",currNtree,"_",currDepth,"_",currEta,"_",iter),quote = FALSE,sep = ",",row.names = FALSE)
    print(paste0("Best till now ",currIterLog))
  }
  
  iter<-iter + 1;
  
}




