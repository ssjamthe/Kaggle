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

library(leaps)
library(dplyr)
library(caret)
setwd("/Users/swapnil/work/Kaggle/out/LMG/")
data<-read.csv("train.csv")
data<-select(data,-(Id))

trainingIndex<-createDataPartition(y=data$Hazard,p=0.8,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

testFinal<-read.csv("test.csv")

obs<-data.frame(attrs=integer(0),gini=numeric(0))

bestGini = 0
bestNtrees = 0
iter = 0;


#modelRf<-randomForest(Hazard~.,data=trainingData,ntree=500,importance=TRUE)
#imp<-as.data.frame(importance(modelRf))
#imp$attr<-rownames(imp)
#imp<-imp[order(imp$`%IncMSE`,decreasing = TRUE),]
#write.table(imp,file = "predictions/attrImp/imp",quote = FALSE,sep = ",",row.names = FALSE)

imp<-read.csv("attrImpBoosted/imp")

imp<-imp[order(imp$IncNodePurity,decreasing=TRUE),]

library(caretEnsemble)

numAttr<-32

obs<-data.frame(numAttr=integer(0),gini=numeric(0))

topStartAttr = 29
ntree = 15000
depth=25

formulaStr = paste0("Hazard~",imp[1,"attr"])
for(i in 2:numAttr)
{
  formulaStr<-paste0(formulaStr,"+",imp[i,"attr"])
}

fitControl <- trainControl(
  number = 1000)

modelBoost <- train(as.formula(formulaStr), data = trainingData,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = TRUE)

predBoost<-predict(modelBoost,testFinal)
predBoostFrame<-data.frame(Id=testFinal$Id,Hazard=predBoost)
write.table(predBoostFrame,file = paste0("predictions/caretTrial/predBoost",depth,"_",ntree,"_",numAttr),quote = FALSE,sep = ",",row.names = FALSE)


predCv<-predict(modelBoost,cvData)
g<-NormalizedGini(cvData$Hazard,predCv)

predTrain<-predict(modelBoost,trainingData)
gTrain<-NormalizedGini(trainingData$Hazard,predTrain)

