set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
library(Metrics)
require(xgboost)
library(gbm)
library(mice)
library(VIM)
library(caret)

roundResponse<-function (resp)
{
  resp<-round(resp)
  grt8<-resp>8
  resp[grt8]<-8
  less1<-resp<1
  resp[less1]<-1
  
  resp
}

scoreFunc<-function(scoreData,lev,model)
{
  actual<-roundResponse(scoreData$obs)
  pred<-roundResponse(scoreData$pred)
  score<-ScoreQuadraticWeightedKappa(actual,pred,1,8)
  r<-c(score)
  names(r)<-"squaredKappa"
}

setwd("/Users/swapnil.jamthe/work/Kaggle/out/PLIA")


data<-read.csv("imp_train_10iter_50percent")

#nasFrac<-sapply(names(data),function(x){sum(is.na(data[,x]))/nrow(data)})
#data<-data[,nasFrac==0]
#data[is.na(data)]<- -9999

data<-select(data,-(Id))
trainingIndex<-createDataPartition(y=data$Response,p=0.9,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

testFinal<-read.csv("trans_test.csv")
#testFinal<-testFinal[,nasFrac[1:length(nasFrac)-1]==0]
testFinal[is.na(testFinal)]<- -9999

fitControl<-trainControl(method = "cv",number=10,summaryFunction=scoreFunc,savePredictions = TRUE)

m<-train(Response~.,data=trainingData,method="rf",metric="squaredKappa")
