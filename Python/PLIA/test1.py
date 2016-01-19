import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from skll import kappa
import os


#outputDf = pd.DataFrame({"first" : [],"second" : []})
#outputDf = outputDf.append(pd.DataFrame({"first" : [2],"second" : [3]}))
os.chdir("/Users/swapnil/work/Kaggle/out/PLIA")
print "hello"


df1Data = {"col1" : [1,2,3],"col2" : ["swap","kals","bang"]}
df1 = pd.DataFrame(data = df1Data)

d = DV(sparse = True)
d1 = d.fit_transform(df1.T.to_dict().values())
d1

df2Data = {"col1" : [1,20,30,40],"col2" : ["swap","kals1","bang","nag"],"col3":[22,33,44,55]}
df2 = pd.DataFrame(data = df2Data)
d1 = d.transform(df2.T.to_dict().values())
