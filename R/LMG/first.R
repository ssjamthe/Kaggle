set.seed(1988)
library(leaps)
library(dplyr)
setwd("/Users/swapnil/work/Kaggle/LMG")
train<-read.csv("train.csv")
train1<-select(train,-(Id))
subsetVars<-regsubsets(Hazard~.,train1,nvmax = 95,really.big = T,method="forward")
subsetSummary<-summary(subsetVars)
par(mfrow=c(2,2))
plot(subsetSummary$rss,xlab="Number of Variables",ylab = "RSS",type="l")
plot(subsetSummary$adjr2,xlab="Number of Variables",ylab = "Adj RSq",type="l")
adjr2ModNum<-which.max(subsetSummary$adjr2)
plot(subsetSummary$cp,xlab="Number of Variables",ylab = "Cp",type="l")
cpModNum<-which.min(subsetSummary$cp)
plot(subsetSummary$bic,xlab="Number of Variables",ylab = "BIC",type="l")
bicModNum<-which.min(subsetSummary$bic)

modelMat<-model.matrix(Hazard~.,data=train1)

testFinal<-read.csv("test.csv")
modelMatTestFinal<-model.matrix(~.,data = testFinal)

coefiAdjr2<-coef(subsetVars,id=adjr2ModNum)
predAdjr2<-modelMatTestFinal[,names(coefiAdjr2)]%*%coefiAdjr2
predAdjr2Frame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predAdjr2)
write.table(predAdjr2Frame,file = "predictions/predAdjr2",quote = FALSE,sep = ",",row.names = FALSE)

modelMatAdjr2<-modelMat[,names(coefiAdjr2)]
modelFrameAdjr2<-data.frame(modelMatAdjr2)
modelFrameAdjr2$Hazard<-train1$Hazard
library(randomForest)
modelBagAdjr2<-randomForest(Hazard~.,data=modelFrameAdjr2,mtry=adjr2ModNum,importance=TRUE)
predBagAdjr2<-predict(modelBagAdjr2,data.frame(modelMatTestFinal))
predBagAdjr2Frame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predBagAdjr2)
write.table(predBagAdjr2Frame,file = "predictions/predBagAdjr2",quote = FALSE,sep = ",",row.names = FALSE)


coefiCp<-coef(subsetVars,id=cpModNum)
predCp<-modelMatTestFinal[,names(coefiCp)]%*%coefiCp
predCpFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predCp)
write.table(predCpFrame,file = "predictions/predCp",quote = FALSE,sep = ",",row.names = FALSE)

coefiBic<-coef(subsetVars,id=bicModNum)
predBic<-modelMatTestFinal[,names(coefiBic)]%*%coefiBic
predBicFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predBic)
write.table(predBicFrame,file = "predictions/predBic",quote = FALSE,sep = ",",row.names = FALSE)

modelMatBic<-modelMat[,names(coefiBic)]
modelFrameBic<-data.frame(modelMatBic)
modelFrameBic$Hazard<-train1$Hazard
library(randomForest)
modelBagBic<-randomForest(Hazard~.,data=modelFrameBic,mtry=bicModNum,importance=TRUE)
predBagBic<-predict(modelBagBic,data.frame(modelMatTestFinal))
predBagBicFrame<-data.frame(Id=modelMatTestFinal[,2],Hazard=predBagBic)
write.table(predBagBicFrame,file = "predictions/predBagBic",quote = FALSE,sep = ",",row.names = FALSE)




