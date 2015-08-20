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
imp<-imp[order(imp$IncNodePurity,decreasing=TRUE),]
#imp<-imp[order(imp$X.IncMSE,decreasing=TRUE),]

obs<-data.frame(numAttr=integer(0),gini=numeric(0),giniTrain=numeric(0),ntree=integer(0),depth=integer(0),eta=integer(0))



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

topStartAttr <- 26
topEndAttr<-26
trials<-data.frame(ntree=integer(0),depth=integer(0),eta=numeric(0),nlinear=integer(0),etalinear=numeric(0),ntree1=integer(0),depth1=integer(0),eta1=numeric(0))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.23001,nlinear=200,etalinear=0.5,ntree1=14,depth1=6,eta1=0.5))
tryExpHyBrid()

iter<-0

iter <- 0



tryExpHyBrid = function()
  {
  for(it in 1:nrow(trials))
  {
    currEta<-trials[it,"eta"]
    currNtree<-trials[it,"ntree"]
    currDepth<-trials[it,"depth"]
    
    currEta1<-trials[it,"eta1"]
    currNtree1<-trials[it,"ntree1"]
    currDepth1<-trials[it,"depth1"]
    
    currNlinear<-trials[it,"nlinear"]
    currEtalinear<-trials[it,"etalinear"]
    
    for(numAttr in seq(topEndAttr,topStartAttr,-1))
    {
      iter<-iter + 1
      print(paste0("Ver 1 manualHybridXgBoost: Training for numAttr=",numAttr))
      currMaxAttr<<-numAttr
      consideredCols<-sapply(colnames(modelMatTrain),selectedCols)
      
      currTrainMat<-modelMatTrain[,consideredCols]
      currCvMat<-modelMatCv[,consideredCols]
      currTestMat<-modelMatTestFinal[,consideredCols]
      
      modelBoostTree<-xgboost(data = currTrainMat, booster="gbtree",label = trainHazard, max.depth = currDepth, eta = currEta, nthread = 2, nround = currNtree, objective = "reg:linear",verbose=0)
      predBoostTree<-predict(modelBoostTree,currTestMat)
      
      modelBoostTree1<-xgboost(data = currTrainMat, booster="gbtree",label = trainHazard, max.depth = currDepth1, eta = currEta1, nthread = 2, nround = currNtree1, objective = "reg:linear",verbose=0)
      predBoostTree1<-predict(modelBoostTree1,currTestMat)
      
      modelBoostLinear<-xgboost(data = currTrainMat, booster="gblinear",label = trainHazard, eta = currEtalinear, nthread = 2, nround = currNlinear, objective = "reg:linear",verbose=0)
      predBoostLinear<-predict(modelBoostLinear,currTestMat)
      
      predBoost<-(predBoostTree + predBoostTree1)/3
      predBoostFrame<-data.frame(Id=testFinal$Id,Hazard=predBoost)
      
      predCvTree<-predict(modelBoostTree,currCvMat)
      predCvTree1<-predict(modelBoostTree1,currCvMat)
      predCvLinear<-predict(modelBoostLinear,currCvMat)
      predCv<-(predCvTree + predCvTree1)/2
      
      predTrainTree<-predict(modelBoostTree,currTrainMat)
      predTrainTree1<-predict(modelBoostTree1,currTrainMat)
      predTrainLinear<-predict(modelBoostLinear,currTrainMat)
      predTrain<-(predTrainTree + predTrainTree1)/2
      
      g<-NormalizedGini(cvData$Hazard,predCv)
      gTree<-NormalizedGini(cvData$Hazard,predCvTree)
      gTree1<-NormalizedGini(cvData$Hazard,predCvTree1)
      gLinear<-NormalizedGini(cvData$Hazard,predCvLinear)
      
      gtrain<-NormalizedGini(trainingData$Hazard,predTrain)
      gtrainTree<-NormalizedGini(trainingData$Hazard,predTrainTree)
      gtrainTree1<-NormalizedGini(trainingData$Hazard,predTrainTree1)
      gtrainLinear<-NormalizedGini(trainingData$Hazard,predTrainLinear)
      
      obs<-rbind(obs,data.frame(attrs=numAttr,gini=g,giniTrain=gtrain,ntree=currNtree,depth=currDepth,eta=currEta))
      
      currIterLog<-paste0("iter=",iter,",numAttr=",numAttr, ",g=",g," ,gTree=",gTree,",gTree1=",gTree1," ,gLinear=",gLinear," ,gtrain=",gtrain," ,gtrainTree=",gtrainTree," ,gtrainTree1=",gtrainTree1," ,gtrainLinear=",gtrainLinear,",ntree=",currNtree,", depth=",currDepth,",eta=",currEta,",numAttr=",numAttr,",ncol=",ncol(currTrainMat))
      print(currIterLog)
      
      write.table(predBoostFrame,file = paste0("predictions/manualHybridXgBoost/predBoost","_",currNtree,"_",currDepth,"_",currEta,"_",numAttr,"_",iter),quote = FALSE,sep = ",",row.names = FALSE)
      
    }
  }
}
