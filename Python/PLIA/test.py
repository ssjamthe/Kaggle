import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split
from skll import kappa


outputDf = pd.DataFrame({"first" : [],"second" : []})
outputDf = outputDf.append(pd.DataFrame({"first" : [2],"second" : [3]}))

print "hello"

print outputDf