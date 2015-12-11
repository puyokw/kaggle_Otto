import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
np.random.seed(131)
path = '../Data/'
NumK = 11 

print("read training data")
train = pd.read_csv(path+"train.csv")
label = train['target']
trainID = train['id']
del train['id'] 
del train['target']

clf = KNeighborsClassifier(n_neighbors=2**NumK,n_jobs=-1)
clf.fit(train.values, label)

print("read test data")
test  = pd.read_csv(path+"test.csv")
ID = test['id']
del test['id']

preds = clf.predict_proba(test)

sample = pd.read_csv(path+'sampleSubmission.csv')
print("writing submission data")
submission = pd.DataFrame(preds, index=ID, columns=sample.columns[1:])
submission.to_csv(path+"knn"+str(NumK)+".csv",index_label='id')



sample = pd.read_csv(path+'sampleSubmission.csv')
# retrain

submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
nfold=5 
score = np.zeros(nfold)
i=0
skf = StratifiedKFold(label, nfold, random_state=131)
for tr, te in skf:
	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
	clf = KNeighborsClassifier(n_neighbors=2**NumK, n_jobs=-1)
	clf.fit(X_train, y_train)
	pred = clf.predict_proba(X_test)
	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
	submission.iloc[te] = pred
	score[i]= log_loss(y_test,pred,eps=1e-15, normalize=True)
	print(score[i])
	i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))


print(log_loss(label,submission.values,eps=1e-15, normalize=True))
submission.to_csv(path+"knn"+str(NumK)+"_retrain.csv",index_label='id')

# N=1, nfold 3, 5.066327 + 0.11068598
# N=1, nfold 5, 4.994438 + 0.14557539 
# N=2, nfold 3, 2.996444 + 0.0730516
# N=2, nfold 5, 2.928231 + 0.0877265
# N=3, nfold 3, 1.781440 + 0.017608977
# N=3, nfold 5, 1.7617141 + 0.0572461
# N=4, nfold 3, 1.189100 + 0.00468322
# N=4, nfold 5, 1.156454 + 0.03361558
# N=5, nfold 3, 0.8854114 + 0.007309473
# N=5, nfold 5, 0.8763264 + 0.0344725
# N=6, nfold 3, 0.768170944 + 0.004770267
# N=6, nfold 5, 0.7533085 + 0.0200288
# N=7, nfold 3, 0.72544669 + 0.0008944366
# N=7, nfold 5, 0.7137548 + 0.0201850
# N=8, nfold 3, 0.7400074 + 0.0035640
# N=8, nfold 5, 0.7263900 + 0.0132361
# N=9, nfold 3, 0.787988 + 0.002485020
# N=9, nfold 5, 0.769268 + 0.01120435
# N=10, nfold 3, 0.8754144 + 0.00259936
# N=10, nfold 5, 0.848127 + 0.00570724
# N=11, nfold 3, 1.0070159 + 0.0020668
# N=11, nfold 5, 0.9660492 + 0.00472689

