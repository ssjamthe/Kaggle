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
setwd("/Users/swapnil.jamthe/work/Kaggle/LMG")
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

library(randomForest)
#modelRf<-randomForest(Hazard~.,data=trainingData,ntree=500,importance=TRUE)
#imp<-as.data.frame(importance(modelRf))
#imp$attr<-rownames(imp)
#imp<-imp[order(imp$`%IncMSE`,decreasing = TRUE),]
#write.table(imp,file = "predictions/attrImp/imp",quote = FALSE,sep = ",",row.names = FALSE)

imp<-read.csv("predictions/attrImp/imp")

imp<-imp[order(imp$IncNodePurity,decreasing=TRUE),]

obs<-data.frame(numAttr=integer(0),gini=numeric(0))
bestGini = 0
bestNumAttr = 0
iter = 0;

topStartAttr = 15
numTrees = 500
for(numAttr in seq(topStartAttr,32,2))
{
formulaStr = paste0("Hazard~",imp[1,"attr"])
for(i in 2:numAttr)
  {
  formulaStr<-paste0(formulaStr,"+",imp[i,"attr"])
}
print(formulaStr)
print(paste0("Ver 1 : Training for numAttr=",numAttr))
iter<-iter + 1
modelRf<-randomForest(as.formula(formulaStr),data=trainingData,ntree=numTrees,importance=FALSE)
predRf<-predict(modelRf,data.frame(testFinal))
predRfFrame<-data.frame(Id=testFinal$Id,Hazard=predRf)
write.table(predRfFrame,file = paste0("predictions/attrImp/predRf","_",numAttr,"_",numTrees),quote = FALSE,sep = ",",row.names = FALSE)
predCv<-predict(modelRf,cvData)
g<-NormalizedGini(cvData$Hazard,predCv)
obs<-rbind(obs,data.frame(attrs=numAttr,gini=g))

print(paste0("iter=",iter, ",g=",g,",numAttr=",numAttr))

if(bestGini < g)
{
  bestGini = g
  bestNumAttr = numAttr
  
  print(paste0("Best g=",g,",bestNumAttr=",numAttr))
  
}
}






