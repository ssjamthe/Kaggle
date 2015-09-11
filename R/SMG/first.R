set.seed(1988)
library(leaps)
library(dplyr)
library(caret)
library(randomForest)
library(imputeR)
setwd("/Users/swapnil/work/Kaggle/out/SMR")
train<-read.csv("train.csv",stringsAsFactors=FALSE)
test<-read.csv("test.csv",stringsAsFactors=FALSE)
test$target<-integer(nrow(test))

train<-select(train,-(ID))
test<-select(test,-(ID))

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

train_numr_imp<-impute(test_numr,lmFun = "lassoR",cFun = "lassoC",conv = FALSE)


nas_in_cols_fraction<-sapply(names(train_char),function(x){sum(is.na(train_char[,x]))/nrow(train_char)})
cols_more_nas<-names(train_char)[nas_in_cols_fraction>0.3]
train_char<-train_char[,!names(train_char) %in% cols_more_nas]
test_char<-train_char[,!names(test_char) %in% cols_more_nas]
for(n in names(train_char))
{
  train_char[is.na(train_char[,n]),n] = 'UNKNOWN_IMPUTED'
  test_char[is.na(test_char[,n]),n] = 'UNKNOWN_IMPUTED'
}


#nas_in_cols_fraction<-sapply(names(train_hour),function(x){sum(is.na(train_hour[,x]))/nrow(train_hour)})
#cols_more_nas<-names(train_hour)[nas_in_cols_fraction>0.3]
#train_hour<-train_hour[,!names(train_hour) %in% cols_more_nas]
#train_hour<-train_hour[,!names(train_hour) %in% cols_more_nas]

nas_in_cols_fraction<-sapply(names(train_date_int),function(x){sum(is.na(train_date_int[,x]))/nrow(train_date_int)})
cols_more_nas<-names(train_date_int)[nas_in_cols_fraction>0.5]
train_date_int<-train_date_int[,!names(train_date_int) %in% cols_more_nas]
train_date_int<-train_date_int[,!names(train_date_int) %in% cols_more_nas]


train_proc<-cbind(train_numr,train_char,train_hour,train_date_int)
test_proc<-cbind(test_numr,test_char,test_hour,test_date_int)

nas_in_cols_fraction<-sapply(names(train_proc),function(x){sum(is.na(train_proc[,x]))/nrow(train_proc)})
cols_more_nas<-names(train_proc)[nas_in_cols_fraction>0.5]

train_proc<-train_proc[,!names(train_proc) %in% cols_more_nas]
test_proc<-test_proc[,!names(test_proc) %in% cols_more_nas]

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

train_proc_imp<-impute(train_proc,lmFun = "lassoR",cFun = "lassoC")
test_proc_imp<-impute(test_proc,lmFun = "lassoR",cFun = "lassoC")

trainingIndex<-createDataPartition(y=train_proc$target,p=0.8,list=FALSE)
trainingData<-train_proc[trainingIndex,]
cvData<-train_proc[-trainingIndex,]
trainingData<-select(trainingData,-(ID))
trainingData$target = factor(trainingData$target)
cvData$target = factor(cvData$target)

modelRf<-randomForest(target~.,data=trainingData,ntree=100,importance=TRUE)





