import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.feature_extraction.text import TfidfTransformer

print("read training data")
train = pd.read_csv("../Data/train.csv")
target = train['target']
NAME=train.columns[0:]
id=train['id']
del train['id'] 
del train['target']

transformer = TfidfTransformer()
train = transformer.fit_transform(train)

train=pd.DataFrame(train.toarray())
train.columns=NAME[1:-1]
train=pd.concat([id,train,target],axis=1)
train.to_csv("../Data/train_tfidf.csv",index=False)

test  = pd.read_csv("../Data/test.csv")
id = test['id']
del test['id']
test = transformer.transform(test)
test=pd.DataFrame(test.toarray())
test.columns=NAME[1:-1]
test=pd.concat([id,test],axis=1)
test.to_csv("C:/Users/kawa/Desktop/kaggle/otto/test_tfidf.csv",index=False)



