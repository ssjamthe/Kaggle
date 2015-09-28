set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
library(randomForest)
library(imputeR)
library(AUC)
setwd("/Users/swapnil/work/Kaggle/out/SMR")
cat("loading data")
train<-read.csv("train.csv",stringsAsFactors=FALSE)
test<-read.csv("test.csv",stringsAsFactors=FALSE)
test$target<-integer(nrow(test))

train<-select(train,-(ID))

col_ct = sapply(train, function(x) length(unique(x[!is.na(x)])))
cat("Constant feature count:", length(col_ct[col_ct==1]))

train = train[, !names(train) %in% names(col_ct[col_ct==1])]
test = test[, !names(test) %in% names(col_ct[col_ct==1])]

train_numr = train[, sapply(train, is.numeric)]
train_char = train[, sapply(train, is.character)]
cat("Numerical column count : ", dim(train_numr)[2], 
    "; Character column count : ", dim(train_char)[2])

test_numr = test[,names(train_numr)]
test_char = test[,names(train_char)]

str(lapply(train_char, unique), vec.len = 4)

train_char[train_char==-1] = NA
train_char[train_char==""] = NA
train_char[train_char=="[]"] = NA

test_char[test_char==-1] = NA
test_char[test_char==""] = NA
test_char[test_char=="[]"] = NA

train_date = train_char[,grep("JAN1|FEB1|MAR1", train_char),]
test_date = test_char[,names(train_date)]

train_char = train_char[, !colnames(train_char) %in% colnames(train_date)]
train_date = sapply(train_date, function(x) strptime(x, "%d%B%y:%H:%M:%S"))
train_date = do.call(cbind.data.frame, train_date)

test_char = test_char[, names(train_char)]
test_date = sapply(test_date, function(x) strptime(x, "%d%B%y:%H:%M:%S"))
test_date = do.call(cbind.data.frame, test_date)

train_time = train_date[,colnames(train_date) %in% c("VAR_0204","VAR_0217")]
train_time = data.frame(sapply(train_time, function(x) strftime(x, "%H:%M:%S")))
train_hour = as.data.frame(sapply(train_time, function(x) as.numeric(as.character(substr( x ,1, 2)))))

test_time = test_date[,colnames(test_date) %in% c("VAR_0204","VAR_0217")]
test_time = data.frame(sapply(test_time, function(x) strftime(x, "%H:%M:%S")))
test_hour = as.data.frame(sapply(test_time, function(x) as.numeric(as.character(substr( x ,1, 2)))))


names(train_hour)<- lapply(names(train_hour),function(x){paste0(x,"_hour")})
names(test_hour)<- lapply(names(test_hour),function(x){paste0(x,"_hour")})

baseDate <- as.Date("01/01/05", format="%m/%d/%y")
train_date_int = data.frame(sapply(train_date, function(x) {as.Date(x) - baseDate}))
train_date_int = do.call(cbind.data.frame, train_date_int)

test_date_int = data.frame(sapply(test_date, function(x) {as.Date(x) - baseDate}))
test_date_int = do.call(cbind.data.frame, test_date_int)

names(train_date_int) <- lapply(names(train_date_int),function(x){paste0(x,"_days")})
names(test_date_int) <- lapply(names(test_date_int),function(x){paste0(x,"_days")})

nas_in_cols_fraction<-sapply(names(train_numr),function(x){sum(is.na(train_numr[,x]))/nrow(train_numr)})
cols_more_nas<-names(train_numr)[nas_in_cols_fraction>0.1]
train_numr<-train_numr[,!names(train_numr) %in% cols_more_nas]
test_numr<-test_numr[,!names(test_numr) %in% cols_more_nas]
for(n in names(train_numr))
{
  notNasTrain<-!is.na(train_numr[,n])
  avgTrain<-sum(as.numeric(train_numr[notNasTrain,n]))/sum(notNasTrain)
  notNasTest<-!is.na(test_numr[,n])
  avgTest<-sum(as.numeric(test_numr[notNasTest,n]))/sum(notNasTest)
  
  if(class(train_numr[,n]) == "integer")
  {
    avgTrain<-as.integer(round(avgTrain))
    avgTest<-as.integer(round(avgTest))
  }
  
  train_numr[!notNasTrain,n]<-avgTrain
  test_numr[!notNasTest,n]<-avgTest
  nasTrain<-is.na(train_numr[,n])
}
#train_numr_imp<-impute(test_numr,lmFun = "lassoR",cFun = "lassoC",conv = FALSE)


nas_in_cols_fraction<-sapply(names(train_char),function(x){sum(is.na(train_char[,x]))/nrow(train_char)})
cols_more_nas<-names(train_char)[nas_in_cols_fraction>0.3]
train_char<-train_char[,!names(train_char) %in% cols_more_nas]
test_char<-test_char[,!names(test_char) %in% cols_more_nas]
for(n in names(train_char))
{
  cat("doing for ",n)
  train_char[is.na(train_char[,n]),n] = 'UNKNOWN_IMPUTED'
  test_char[is.na(test_char[,n]),n] = 'UNKNOWN_IMPUTED'
  uniqueTrainVals<-unique(train_char[,n])
  newTestValsInd<-sapply(test_char[,n],function(x){x %in% uniqueTrainVals})
  test_char[!newTestValsInd,n] = 'UNKNOWN_IMPUTED'
  train_char[,n] = factor(train_char[,n])
  test_char[,n] = factor(test_char[,n])
  levels(test_char[,n])<-levels(train_char[,n])
}

moreUnique<-sapply(names(train_char),function(x){length(unique(train_char[,x]))>53})
colsMoreUnique<-names(train_char)[moreUnique]
train_char<-train_char[,!names(train_char) %in% colsMoreUnique]
test_char<-test_char[,!names(test_char) %in% colsMoreUnique]


#nas_in_cols_fraction<-sapply(names(train_hour),function(x){sum(is.na(train_hour[,x]))/nrow(train_hour)})
#cols_more_nas<-names(train_hour)[nas_in_cols_fraction>0.3]
#train_hour<-train_hour[,!names(train_hour) %in% cols_more_nas]
#test_hour<-test_hour[,!names(test_hour) %in% cols_more_nas]
for(n in names(train_hour))
{
  notNasTrain<-!is.na(train_hour[,n])
  avgTrain<-sum(as.numeric(train_hour[notNasTrain,n]))/sum(notNasTrain)
  notNasTest<-!is.na(test_hour[,n])
  avgTest<-sum(as.numeric(test_hour[notNasTest,n]))/sum(notNasTest)
  avgTrain<-as.integer(round(avgTrain))
  avgTest<-as.integer(round(avgTest))
  
  train_hour[!notNasTrain,n]<-avgTrain
  test_hour[!notNasTest,n]<-avgTest
}

nas_in_cols_fraction<-sapply(names(train_date_int),function(x){sum(is.na(train_date_int[,x]))/nrow(train_date_int)})
cols_more_nas<-names(train_date_int)[nas_in_cols_fraction>0.5]
train_date_int<-train_date_int[,!names(train_date_int) %in% cols_more_nas]
test_date_int<-test_date_int[,!names(test_date_int) %in% cols_more_nas]
for(n in names(train_date_int))
{
  notNasTrain<-!is.na(train_date_int[,n])
  avgTrain<-sum(as.numeric(train_date_int[notNasTrain,n]))/sum(notNasTrain)
  notNasTest<-!is.na(test_date_int[,n])
  avgTest<-sum(as.numeric(test_date_int[notNasTest,n]))/sum(notNasTest)
  avgTrain<-as.integer(round(avgTrain))
  avgTest<-as.integer(round(avgTest))
  
  train_date_int[!notNasTrain,n]<-avgTrain
  test_date_int[!notNasTest,n]<-avgTest
}


train_proc<-cbind(train_numr,train_char,train_hour,train_date_int)
test_proc<-cbind(test_numr,test_char,test_hour,test_date_int)
test_proc<-select(test_proc,-(target))

remove(col_ct)
remove(train_numr)
remove(train_char)
remove(test_numr)
remove(test_char)
remove(train_date)
remove(test_date)
remove(train_time)
remove(test_time)
remove(train_hour)
remove(test_hour)
remove(train_date_int)
remove(test_date_int)

trainingIndex<-createDataPartition(y=train_proc$target,p=0.8,list=FALSE)
trainingData<-train_proc[trainingIndex,]
cvData<-train_proc[-trainingIndex,]

trainingData$target = as.numeric(trainingData$target)
cvData$target = as.numeric(cvData$target) 

fullModelMat<-model.matrix(~.,data = train_proc)
preProc<-preProcess(fullModelMat)
processedFullModelMat<-predict(preProc,fullModelMat)
modelMatTrain<-model.matrix(target~.,data = trainingData)
modelMatCv<-model.matrix(~.,data = cvData)
modelMatTestFinal<-model.matrix(~.,data = test_proc)


subsetVars<-regsubsets(target~.,data.frame(processedFullModelMat),nvmax = 1991,really.big = T,method="forward")
subsetSummary<-summary(subsetVars)
par(mfrow=c(2,2))
plot(subsetSummary$rss,xlab="Number of Variables",ylab = "RSS",type="l")
plot(subsetSummary$adjr2,xlab="Number of Variables",ylab = "Adj RSq",type="l")
#1192
adjr2ModNum<-which.max(subsetSummary$adjr2)
plot(subsetSummary$cp,xlab="Number of Variables",ylab = "Cp",type="l")
#672
cpModNum<-which.min(subsetSummary$cp)
plot(subsetSummary$bic,xlab="Number of Variables",ylab = "BIC",type="l")
#162
bicModNum<-which.min(subsetSummary$bic)

# minErrorInd <- -1;
# minError <- 1000000;
# errors<-rep(NA,1990)
# 
# for(i in 1:1990)
# {
#   coefi<-coef(subsetVars,id=i)
#   newNamesInd<-sapply(names(coefi),function(x){!(x %in% colnames(modelMatCvNew))})
#   if(sum(newNamesInd) > 0 )
#   {
#     print(names(coefi)[newNamesInd])
#   }
#   pred<-modelMatCv[,names(coefi)]%*%coefi
#   errors[i]<-mean((cvData$target - pred)^2)
#   if(minError > errors[i])
#   {
#     minErrorInd <- i
#     minError <- errors[i]
#   }
# }


metric="adjr2"
coefiAdjr2<-coef(subsetVars,id=adjr2ModNum)
modelMatTrain<-modelMatTrain[,names(coefiAdjr2)]
modelMatCv<-modelMatCv[,names(coefiAdjr2)]

#modelFrameTrainAdjr2<-data.frame(modelMatTrainAdjr2)
#modelFrameTrainAdjr2$target<-trainingData$target

obs<-data.frame(iter=integer(0),rocscore=numeric(0),rocscoreTrain=numeric(0),ntree=integer(0),depth=integer(0),eta=integer(0))

trials<-data.frame(ntree=integer(0),depth=integer(0),eta=integer(0))
trials<-rbind(trials,data.frame(ntree=2,depth=2,eta=0.06))

trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.08))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.1))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.15))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.2))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.25))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.3))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.35))
trials<-rbind(trials,data.frame(ntree=50,depth=5,eta=0.4))

trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.08))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.1))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.15))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.2))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.25))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.3))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.35))
trials<-rbind(trials,data.frame(ntree=50,depth=6,eta=0.4))


trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.08))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.1))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.15))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.2))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.25))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.3))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.35))
trials<-rbind(trials,data.frame(ntree=45,depth=5,eta=0.4))

trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.08))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.1))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.15))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.2))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.25))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.3))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.35))
trials<-rbind(trials,data.frame(ntree=45,depth=6,eta=0.4))

trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.08))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.1))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.15))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.2))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.25))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.3))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.35))
trials<-rbind(trials,data.frame(ntree=40,depth=5,eta=0.4))

trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.08))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.1))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.15))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.2))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.25))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.3))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.35))
trials<-rbind(trials,data.frame(ntree=40,depth=6,eta=0.4))


cat("iter,rocscore,rocscoreTrain,ntree,depth,eta,metric\n",file="obs/forwardSelection/obs.txt",append=TRUE)

library(xgboost)
iter = 0
bestR<--1;
bestIter<--1;
for(it in 1:nrow(trials))
{
  iter = iter + 1;
  currEta<-trials[it,"eta"]
  currNtree<-trials[it,"ntree"]
  currDepth<-trials[it,"depth"]
  
  currTrainMat<-modelMatTrain
  currCvMat<-modelMatCv
  currTestMat<-modelMatTestFinal
  
  modelBoost<-xgboost(data = currTrainMat, label = trainingData$target, max.depth = currDepth, eta = currEta, nthread = 2, nround = currNtree, objective = "binary:logistic",verbose=0)
  predBoost<-predict(modelBoost,currTestMat)
  predBoostFrame<-data.frame(Id=test$ID,target=predBoost)
  predCv<-predict(modelBoost,currCvMat)
  r<-auc(roc(predictions = predCv,labels = factor(cvData$target)))
  predTrain<-predict(modelBoost,currTrainMat)
  rtrain<-auc(roc(predictions = predTrain,labels = factor(trainingData$target)))
  
  obs<-rbind(obs,data.frame(iter=iter,rocscore=r,rocscoreTrain=rtrain,ntree=currNtree,depth=currDepth,eta=currEta))
  currIterLog<-paste0("iter=",iter,",rocscore=",r,",rocscoreTrain=",rtrain,",ntree=",currNtree,",depth=",currDepth,",eta=",currEta)
  print(currIterLog)
  
  csvLog<-paste0(iter,r,rtrain,currNtree,currDepth,currEta,metric,"\n")
  cat(csvLog,file="obs/forwardSelection/obs.txt",append=TRUE)
  
  
  if(bestR < r)
  {
    
    print(paste0("Best till now ",currIterLog,":: lastBest=",bestR,",lastBestIter=",bestIter))
    
    bestR<-r
    bestIter<-iter
    
    print(paste0("Best till now ",currIterLog))
    
    write.table(predBoostFrame,file = paste0("predictions/forwardSelection/predBoost","_",currNtree,"_",currDepth,"_",currEta,"_",iter),quote = FALSE,sep = ",",row.names = FALSE)
    
  }
  
}






