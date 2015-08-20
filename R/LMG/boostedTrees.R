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

trainingIndex<-createDataPartition(y=data$Hazard,p=0.85,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

modelMatTrain<-model.matrix(Hazard~.,data=trainingData)
modelFrameTrain<-data.frame(modelMatTrain)
modelFrameTrain$Hazard<-trainingData$Hazard

modelMatCv<-model.matrix(~.,data = cvData)

testFinal<-read.csv("test.csv")
modelMatTestFinal<-model.matrix(~.,data = testFinal)

depths<-c(6,10,14,18,22,16,30)
ntrees<-c(9000,13000,18000,25000)
obs<-data.frame(dep=integer(0),nt=integer(0),gini=numeric(0))

bestGini = 0
bestDepth = 0
bestNtrees = 0
iter = 0;

library(gbm)
for(depth in depths)
{
  for(ntree in ntrees)
  {
    iter = iter + 1;
    modelBoost<-gbm(Hazard~.,data = modelFrameTrain,distribution = "gaussian",n.trees = ntree,interaction.depth = depth)
    predBoost<-predict(modelBoost,data.frame(modelMatTestFinal),n.trees = ntree)
    predBoostFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predBoost)
    write.table(predBoostFrame,file = paste0("predictions/boostedTrees/predBoost",depth,"_",ntree),quote = FALSE,sep = ",",row.names = FALSE)
    predCv<-predict(modelBoost,data.frame(modelMatCv),n.trees = ntree)
    g<-NormalizedGini(cvData$Hazard,predCv)
    obs<-rbind(obs,data.frame(dep=depth,nt=ntree,gini=g))
    
    print(paste0("iter=",iter, ",g=",g,",bestDepth=",depth,",bestNtrees=",ntree))
    
    if(bestGini < g)
    {
      bestGini = g
      bestDepth = depth
      bestNtrees = ntree
      
      print(paste0("Best g=",g,",bestDepth=",depth,",bestNtrees=",ntree))
      
    }
  }
}


