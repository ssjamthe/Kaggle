set.seed(1988)

library(dplyr)

transDataByMkSum<-function(data)
{
  #print(names(data))
  data$mkSum = rep(0,nrow(data))
  for(i in 1:nrow(data))
  {
    mkSum = 0
    for(colName in names(data))
    {
      if((grepl("Medical_Keyword",colName))[1] == TRUE)
      {
        #print(paste0(class(data[i,colName]),colName,data[i,colName]))
        if(!is.na(data[i,colName]))
        {
          mkSum = mkSum + data[i,colName]
        }
      }
    }
    
    data[i,"mkSum"] = mkSum
  }
  
  
  data<-select(data, -contains("Medical_Keyword"))
  
  return(data)
}

setwd("/Users/swapnil.jamthe/work/Kaggle/out/PLIA")
data<-read.csv("trans_train.csv")
data<-transDataByMkSum(data)


write.table(data,file = "trans_train_sumMK.csv",quote = FALSE,sep = ",",row.names = FALSE)

dataTest<-read.csv("trans_test.csv")
dataTest<-transDataByMkSum(dataTest)


write.table(dataTest,file = "trans_test_sumMK.csv",quote = FALSE,sep = ",",row.names = FALSE)

