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
library(caretEnsemble)
setwd("/Users/swapnil/work/Kaggle/out/LMG")
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

obs<-data.frame(numAttr=integer(0),gini=numeric(0))
bestGini = 0
bestNumAttr = 0
iter = 0;

topStartAttr = 29
endAttr = 29
ntree = 15000
depth=25
library(gbm)
for(numAttr in seq(endAttr,topStartAttr,-2))
{
  formulaStr = paste0("Hazard~",imp[1,"attr"])
  for(i in 2:numAttr)
  {
    formulaStr<-paste0(formulaStr,"+",imp[i,"attr"])
  }
  print(formulaStr)
  print(paste0("Ver 1 : Training for numAttr=",numAttr))
  iter<-iter + 1
  
  modelList<-caretList(
    as.formula(formulaStr), data=trainingData,
    tuneList=list(
      b=caretModelSpec(method='glmboost', tuneGrid=data.frame(mstop = 15000,prune=TRUE)),
      r=caretModelSpec(method='rf', tuneGrid=data.frame(mtry=10))
      
    )
  )
  
}

