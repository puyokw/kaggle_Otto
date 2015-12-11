import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
path = '../Data/'
np.random.seed(131)
NumK = 11 

print("read training data")
train = pd.read_csv(path+"train_tfidf.csv")
label = train['target']
trainID = train['id']
del train['id'] 
del train['target']

clf = KNeighborsClassifier(n_neighbors=2**NumK,n_jobs=-1)
clf.fit(train.values, label)

print("read test data")
test  = pd.read_csv(path+"test_tfidf.csv")
ID = test['id']
del test['id']

preds = clf.predict_proba(test)

sample = pd.read_csv(path+'sampleSubmission.csv')
print("writing submission data")
submission = pd.DataFrame(preds, index=ID, columns=sample.columns[1:])
submission.to_csv(path+"knn"+str(NumK)+"_tfidf.csv",index_label='id')



sample = pd.read_csv(path+'sampleSubmission.csv')
# retrain

submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
nfold=5 
score = np.zeros(nfold)
i=0
skf = StratifiedKFold(label, nfold, random_state=131)
for tr, te in skf:
	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
	clf = KNeighborsClassifier(n_neighbors=2**NumK,n_jobs=-1)
	clf.fit(X_train, y_train)
	pred = clf.predict_proba(X_test)
	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
	submission.iloc[te] = pred
	score[i]= log_loss(y_test,pred,eps=1e-15, normalize=True)
	print(score[i])
	i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))


print(log_loss(label,submission.values,eps=1e-15, normalize=True))
submission.to_csv(path+"knn"+str(NumK)+"_tfidf_retrain.csv",index_label='id')

# N=1, nfold 3, 4.8548235 + 0.034561311
# N=1, nfold 5, 4.7705264 + 0.0880885
# N=2, nfold 3, 2.950239 + 0.0361604
# N=2, nfold 5, 2.8713026 + 0.0550144
# N=3, nfold 3, 1.8448740 + 0.04320835
# N=3, nfold 5, 1.7967596 + 0.059249
# N=4, nfold 3, 1.28595 + 0.03731698
# N=4, nfold 5, 1.258785 + 0.065777
# N=5, nfold 3, 0.9882000 + 0.0242380
# N=5, nfold 5, 0.9700529 + 0.0479838
# N=6, nfold 3, 0.841384 + 0.0110819
# N=6, nfold 5, 0.8229223 + 0.04416355
# N=7, nfold 3, 0.758282 + 0.0067465
# N=7, nfold 5, 0.748291 + 0.03160732
# N=8, nfold 3, 0.7228624 + 0.00348746
# N=8, nfold 5, 0.71268157 + 0.0246045
# N=9, nfold 3, 0.723000 + 0.0063083
# N=9, nfold 5, 0.711495 + 0.0185380
# N=10, nfold 3, 0.7569855 + 0.00438494
# N=10, nfold 5, 0.7423447 + 0.0110112
# N=11, nfold 3, 0.8170302 + 0.0048478
# N=11, nfold 5, 0.7977566 + 0.0088295

