set.seed(1988)
library(mice)
library(VIM)
library(dplyr)

setwd("/Users/swapnil/work/Kaggle/out/PLIA")
data<-read.csv("trans_train.csv")
p<-md.pattern(data)

#aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


nas_in_cols_fraction<-sapply(names(data),function(x){sum(is.na(data[,x]))/nrow(data)})
cols_more_nas<-names(data)[nas_in_cols_fraction>0.5]

test<-read.csv("trans_test.csv")


dataSel<-data[,!names(data) %in% cols_more_nas]
testSel<-test[,!names(test) %in% cols_more_nas]

imp<-mice(dataSel,m=1,maxit=100,meth='pmm',seed=1988)

write.table(complete(imp),file = "imp_train_100iter_50percent",quote = FALSE,sep = ",",row.names = FALSE)

impTest<-mice(testSel,m=1,maxit=100,meth='pmm',seed=1988)

write.table(complete(impTest),file = "imp_test_100iter_50percent",quote = FALSE,sep = ",",row.names = FALSE)


