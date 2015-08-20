set.seed(1988)

#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  #df
  df$random = (1:nrow(df))/nrow(df)
  #df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  #print(df)
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}


set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
setwd("/Users/swapnil.jamthe/work/Kaggle/LMG")
data<-read.csv("train.csv")
data<-select(data,-(Id))

trainingIndex<-createDataPartition(y=data$Hazard,p=0.8,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

trainHazard<-as.numeric(trainingData$Hazard)

modelMatTrain<-model.matrix(Hazard~.-1,data = trainingData)
modelMatCv<-model.matrix(Hazard~.-1,data = cvData)

testFinal<-read.csv("test.csv")
modelMatTestFinal<-model.matrix(~.-1,data = testFinal)

imp<-read.csv("predictions/attrImpBoosted/imp")
imp<-imp[order(imp$X.IncMSE,decreasing=TRUE),]

obs<-data.frame(numAttr=integer(0),gini=numeric(0),giniTrain=numeric(0),ntree=integer(0),depth=integer(0),eta=integer(0))

trials<-data.frame(ntree=integer(0),depth=integer(0),eta=integer(0))
ntrees<-c(25,30,35,40,45,50,60,80,100,120)
depths<-c(2,3,4,5,6,7,8,9)
etas<-c(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8)

for(ntree in ntrees)
{
  for(depth in depths)
  {
    for(eta in etas)
    {
      trials<-rbind(trials,data.frame(ntree=ntree,depth=depth,eta=eta))
    }
  }
}

topStartAttr = 2
require(xgboost)
library("gdata")

currMaxAttr<-1

selectedCols = function(col){
  match = FALSE
  for(colS in imp[1:currMaxAttr,"attr"])
  {
    if(startsWith(col,colS) == TRUE)
    { 
      match =  TRUE
    }
  } 
  return(match)
}
iter<-0
bestGini = 0
bestNumAttr = 0
iter = 0;
bestIter<-0
for(it in 1:nrow(trials))
{
currEta<-trials[it,"eta"]
currNtree<-trials[it,"ntree"]
currDepth<-trials[it,"depth"]

for(numAttr in seq(32,topStartAttr,-1))
{
  iter<-iter + 1
  print(paste0("Ver 1 attrImpOOBXgBoost: Training for numAttr=",numAttr))
  currMaxAttr<-numAttr
  consideredCols<-sapply(colnames(modelMatTrain),selectedCols)
  
  currTrainMat<-modelMatTrain[,consideredCols]
  currCvMat<-modelMatCv[,consideredCols]
  currTestMat<-modelMatTestFinal[,consideredCols]
  
  modelBoost<-xgboost(data = currTrainMat, label = trainHazard, max.depth = currDepth, eta = currEta, nthread = 2, nround = currNtree, objective = "reg:linear",verbose=0)
  predBoost<-predict(modelBoost,currTestMat)
  predBoostFrame<-data.frame(Id=testFinal$Id,Hazard=predBoost)
  predCv<-predict(modelBoost,currCvMat)
  g<-NormalizedGini(cvData$Hazard,predCv)
  gtrain<-NormalizedGini(trainingData$Hazard,predict(modelBoost,currTrainMat))
  obs<-rbind(obs,data.frame(attrs=numAttr,gini=g,giniTrain=gtrain,ntree=currNtree,depth=currDepth,eta=currEta))
  
  currIterLog<-paste0("iter=",iter, ",g=",g,",numAttr=",numAttr," ,gtrain=",gtrain,",ntree=",currNtree,", depth=",currDepth,",eta=",currEta,",numAttr=",numAttr,",ncol=",ncol(currTrainMat),",best was ",bestGini," best iter was ",bestIter)
  print(currIterLog)
  
  if(bestGini < g)
  {
    bestGini = g
    bestNumAttr = numAttr
    bestIter<-iter
    
    write.table(predBoostFrame,file = paste0("predictions/attrImpOOBXgBoost/predBoost","_",currNtree,"_",currDepth,"_",currEta,"_",numAttr,"_",iter),quote = FALSE,sep = ",",row.names = FALSE)
    
    
    print(paste0("Best till now ",currIterLog))
    
  }
  
}
}

