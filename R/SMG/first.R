setwd("/Users/swapnil/work/Kaggle/out/SMR")
train<-read.csv("train.csv",stringsAsFactors=FALSE)
test<-read.csv("test.csv",stringsAsFactors=FALSE)
test$target<-integer(nrow(test))

col_ct = sapply(train, function(x) length(unique(x)))
cat("Constant feature count:", length(col_ct[col_ct==1]))

train = train[, !names(train) %in% names(col_ct[col_ct==1])]

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


