set.seed(1988)
library(leaps)
library(dplyr)
library(caret)

setwd("/Users/swapnil/work/Kaggle/out/PLIA")
data<-read.csv("train.csv")

nonCategorical<-c("Id","Product_Info_4", 
"Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1"
,"Employment_Info_4","Employment_Info_6","Insurance_History_5"
,"Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"
,"Medical_History_1","Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32","Response")

binaryValCols = character(0)

for(col in names(data))
{
  uniqueVals = length(unique(data[,col]));
  print(paste0("uniqueVals for col ",col, " are ",uniqueVals))
  
  if(uniqueVals == 2)
  {
    
    binaryValCols <- c(binaryValCols,col)
  }
}

transData<-data.frame(dummyCol=integer(nrow(data)))
for(col in names(data))
{
  
  if(col %in% nonCategorical | grepl("Medical_Keyword",col) | col %in% binaryValCols)
  {
    transData[,col] = data[,col]
  }
  else
  {
    transData[,col] = paste0("t_",data[,col])
  }
}

transData<-select(transData,-(dummyCol))

write.table(transData,file = "trans_train.csv",quote = FALSE,sep = ",",row.names = FALSE)


testData<-read.csv("test.csv")

transTestData<-data.frame(dummyCol=integer(nrow(testData)))
for(col in names(testData))
{
  if(col %in% nonCategorical | grepl("Medical_Keyword",col) | col %in% binaryValCols)
  {
    transTestData[,col] = testData[,col]
  }
  else
  {
    transTestData[,col] = paste0("t_",testData[,col])
  }
}

transTestData<-select(transTestData,-(dummyCol))

write.table(transTestData,file = "trans_test.csv",quote = FALSE,sep = ",",row.names = FALSE)










