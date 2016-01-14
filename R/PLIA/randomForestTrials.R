set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
library(Metrics)
require(xgboost)
library(gbm)
library(mice)
library(VIM)
library(randomForest)

roundResponse<-function (resp)
{
  resp<-round(resp)
  grt8<-resp>8
  resp[grt8]<-8
  less1<-resp<1
  resp[less1]<-1
  
  resp
}

setwd("/Users/swapnil.jamthe/work/Kaggle/out/PLIA")

data<-read.csv("imp_train_10iter_50percent")

#nasFrac<-sapply(names(data),function(x){sum(is.na(data[,x]))/nrow(data)})
#data<-data[,nasFrac==0]
#data[is.na(data)]<- -9999

data<-select(data,-(Id))

for(colName in names(data))
{
  
  if(class(data[,colName]) == "factor")
  {
    u<-length(unique(data[colName,]))
    print(paste0("NUmber of unique values are ",u," for col ",colName))
    if( u > 53)
    {
      print(paste0("Removing column as length is more than 53 ",colName))
      #data<-select(data,-(colName))
    }
  }
} 

trainingIndex<-createDataPartition(y=data$Response,p=0.8,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

testFinal<-read.csv("trans_test.csv")
#testFinal<-testFinal[,nasFrac[1:length(nasFrac)-1]==0]
testFinal[is.na(testFinal)]<- -9999


obs<-data.frame(kappa=numeric(0),kappaTrain=numeric(0),ntree=integer(0))

trials<-data.frame(ntree=integer(0))
trials<-rbind(trials,data.frame(ntree=1))

ntrees<-c(25000,20000,15000,10000,5000,3000,1000,100,120,140,160,10,20,30,40,50,60,70,80)


for(nt in  ntrees)
{
  
      trials<-rbind(trials,data.frame(ntree=nt))
      
    
}

iter<-0
bestKappa <- -2
bestIter<-0

for(it in 1:nrow(trials))
{
  
  currNtree<-trials[it,"ntree"]
  
  
  modelRf<-randomForest(Response~.,data = trainingData,ntree=nt,importance=FALSE)
  predRf<-predict(modelRf,testFinal,n.trees = currNtree)
  predRf<-roundResponse(predRf)
  predRfFrame<-data.frame(Id=testFinal$Id,Response=predRf)
  predCv<-predict(modelRf,cvData,n.trees = currNtree)
  predCv<-roundResponse(predCv)
  k<-ScoreQuadraticWeightedKappa(cvData$Response,predCv,1,8)
  predTrain<-predict(modelRf,trainingData,n.trees = currNtree)
  predTrain<-roundResponse(predTrain)
  ktrain<-ScoreQuadraticWeightedKappa(trainingData$Response,predTrain,1,8)
  
  currIterLog<-paste0("iter=",iter, ",k=",k,",ktrain=",ktrain,",ntree=",currNtree,",best was ",bestKappa," best iter was ",bestIter)
  print(currIterLog)
  obs<-rbind(obs,data.frame(kappa=k,kappaTrain=ktrain,ntree=currNtree))
  
  if(bestKappa < k)
  {
    bestKappa<-k
    bestIter<-iter
    
    write.table(predBoostFrame,file = paste0("predictions/randomForestTrials/predRf","_",currNtree,"_",iter),quote = FALSE,sep = ",",row.names = FALSE)
    print(paste0("Best till now ",currIterLog))
  }
  
  iter<-iter + 1;
  
}