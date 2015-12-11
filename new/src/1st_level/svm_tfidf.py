import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

print("read training data")
path = '../Data/'
train = pd.read_csv(path+'train_tfidf.csv')
label = train['target']
trainID = train['id']
del train['id'] 
del train['target']

np.random.seed(131)
svc = svm.SVC(kernel='rbf',C=10,probability=True,verbose=True) 
svc.fit(train.values, label)
#calibrated_svc = CalibratedClassifierCV(OneVsRestClassifier(svc,n_jobs=-1), method='isotonic', cv=5)
#calibrated_svc.fit(train.values, label)

print("read test data")
test  = pd.read_csv(path+'test_tfidf.csv')
ID = test['id']
del test['id']
clf_probs = svc.predict_proba(test.values)
#clf_probs = calibrated_svc.predict_proba(test.values)

sample = pd.read_csv(path+'sampleSubmission.csv')
print("writing submission data")
submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
submission.to_csv(path+'svm_tfidf.csv',index_label='id')


sample = pd.read_csv(path+'sampleSubmission.csv')
# retrain

submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
nfold=5
skf = StratifiedKFold(label, nfold, random_state=131)
for tr, te in skf:
	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
	np.random.seed(131)
	svc = svm.SVC(kernel='rbf',C=10,probability=True,verbose=True) 
	svc.fit(X_train, y_train)
	pred = svc.predict_proba(X_test)
	print(log_loss(y_test,pred,eps=1e-15, normalize=True))
	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
	submission.iloc[te] = pred

print(submission)
print(log_loss(label,submission.values,eps=1e-15, normalize=True))

submission.to_csv(path+'svm_tfidf_retrain.csv',index_label='id')
	
# 0.61888