set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
setwd("/Users/swapnil/work/Kaggle/LMG")
data<-read.csv("train.csv")
data<-select(data,-(Id))

trainingIndex<-createDataPartition(y=data$Hazard,p=0.85,list=FALSE)
trainingData<-data[trainingIndex,]
cvData<-data[-trainingIndex,]

subsetVars<-regsubsets(Hazard~.,trainingData,nvmax = 95,really.big = T,method="forward")
subsetSummary<-summary(subsetVars)

modelMatCv<-model.matrix(~.,data = cvData)
minErrorInd <- -1;
minError <- 1000000;
errors<-rep(NA,94)

for(i in 1:94)
{
  coefi<-coef(subsetVars,id=i)
  pred<-modelMatCv[,names(coefi)]%*%coefi
  errors[i]<-mean((cvData$Hazard - pred)^2)
  if(minError > errors[i])
  {
    minErrorInd <- i
    minError <- errors[i]
  }
}


testFinal<-read.csv("test.csv")
modelMatTestFinal<-model.matrix(~.,data = testFinal)

coefiCv<-coef(subsetVars,id=minErrorInd)
predCv<-modelMatTestFinal[,names(coefiCv)]%*%coefiCv
predCvFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predCv)
write.table(predCvFrame,file = "predictions/predCv",quote = FALSE,sep = ",",row.names = FALSE)

#library(doParallel)
#cl <- makeCluster(4)
#registerDoParallel(cl)
#stopCluster(cl)

modelMat<-model.matrix(Hazard~.,data=data)
modelMatBestCv<-modelMat[,names(coefiCv)]
modelFrameBestCv<-data.frame(modelMatBestCv)
modelFrameBestCv$Hazard<-data$Hazard
#modelFrameBestCv<-modelFrame[,c("Hazard",names(coefiCv))]
#tc5 <- trainControl("cv",5)
#modFitRpart<-train(Hazard~.,data = modelFrameBestCv,method="rf")
#predRPart<-predict(modFitRpart5,newdata=modelMatTestFinal)

library(tree)
modelTree<-tree(Hazard~.,data=modelFrameBestCv)
predTree<-predict(modelTree,data.frame(modelMatTestFinal))
predTreeFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predTree)
write.table(predTreeFrame,file = "predictions/predTree",quote = FALSE,sep = ",",row.names = FALSE)

library(randomForest)
modelBag<-randomForest(Hazard~.,data=modelFrameBestCv,mtry=94,importance=TRUE)
predBag<-predict(modelBag,data.frame(modelMatTestFinal))
predBagFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predBag)
write.table(predBagFrame,file = "predictions/predBag",quote = FALSE,sep = ",",row.names = FALSE)


library(randomForest)
modelRf<-randomForest(Hazard~.,data=modelFrameBestCv,importance=TRUE)
predRf<-predict(modelRf,data.frame(modelMatTestFinal))
predRfFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predRf)
write.table(predRfFrame,file = "predictions/predRf",quote = FALSE,sep = ",",row.names = FALSE)

library(gbm)
modelBoost<-gbm(Hazard~.,data = modelFrameBestCv,distribution = "gaussian",n.trees = 5000,interaction.depth = 3)
predBoost<-predict(modelBoost,data.frame(modelMatTestFinal),n.trees = 5000)
predBoostFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predBoost)
write.table(predBoostFrame,file = "predictions/predBoost",quote = FALSE,sep = ",",row.names = FALSE)



#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df
  df$random = (1:nrow(df))/nrow(df)
  df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  print(df)
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}




