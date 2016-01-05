set.seed(1988)
library(mice)
library(VIM)

setwd("/Users/swapnil/work/Kaggle/out/PLIA")
data<-read.csv("trans_train.csv")
p<-md.pattern(data)

#aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


nas_in_cols_fraction<-sapply(names(data),function(x){sum(is.na(data[,x]))/nrow(data)})
cols_more_nas<-names(data)[nas_in_cols_fraction>0.5]

test<-read.csv("trans_test.csv")

dataSel<-data[,!names(train_numr) %in% cols_more_nas]
testSel<-testFinal[,!names(test_numr) %in% cols_more_nas]

imp<-mice(data,m=1,maxit=10,meth='pmm',seed=1988)