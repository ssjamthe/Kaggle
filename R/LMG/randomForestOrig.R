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

obs<-data.frame(nt=integer(0),gini=numeric(0))

bestGini = 0
bestNtrees = 0
iter = 0;

ntrees = c(1,1500,10000,15000,20000)

library(randomForest)
for(nt in ntrees)
{
  print(paste0("Training for ntree=",nt))
  iter<-iter + 1
  modelRf<-randomForest(Hazard~.,data=trainingData,ntree=nt,importance=FALSE)
  predRf<-predict(modelRf,data.frame(testFinal))
  predRfFrame<-data.frame(Id=testFinal$Id,Hazard=predRf)
  write.table(predRfFrame,file = "predictions/randomForest/predRf",quote = FALSE,sep = ",",row.names = FALSE)
  predCv<-predict(modelRf,cvData)
  g<-NormalizedGini(cvData$Hazard,predCv)
  obs<-rbind(obs,data.frame(nt=nt,gini=g))
  
  print(paste0("iter=",iter, ",g=",g,",bestNtrees=",nt))
  
  if(bestGini < g)
  {
    bestGini = g
    bestNtrees = nt
    
    print(paste0("Best g=",g,",bestNtrees=",nt))
    
  }
}





