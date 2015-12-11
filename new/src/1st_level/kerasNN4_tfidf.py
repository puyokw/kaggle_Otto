from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
SEED = 71 
np.random.seed(SEED)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD, Optimizer
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import BaggingClassifier 
from sklearn.cross_validation import StratifiedKFold, KFold
path = '../Data/'

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

print("Loading data...")
X, labels = load_data(path+'train_tfidf.csv', train=True)
#X=np.log(X+1)
#X=np.sqrt(X+(3/8))

X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data(path+'test_tfidf.csv', train=False)
#X_test=np.log(X_test+1)
#X_test=np.sqrt(X_test+(3/8))

X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')


sample = pd.read_csv(path+'sampleSubmission.csv')
N = X.shape[0]
trainId = np.array(range(N))
submissionTr = pd.DataFrame(index=trainId,columns=sample.columns[1:])

nfold=5
RND = np.random.randint(0,10000,nfold)
pred = np.zeros((X_test.shape[0],9))
score = np.zeros(nfold)
i=0
skf = StratifiedKFold(labels, nfold, random_state=SEED)
for tr, te in skf:
	X_train, X_valid, y_train, y_valid = X[tr], X[te], y[tr], y[te]
	predTr = np.zeros((X_valid.shape[0],9))
	n_bag=10
	for j in range(n_bag):
		print('nfold: ',i,'/',nfold, ' n_bag: ',j,' /',n_bag)
		print("Building model...")
		model = Sequential()
		model.add(Dense(512, input_shape=(dims,)))
		model.add(PReLU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(512))
		model.add(PReLU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(512))
		model.add(PReLU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(512))
		model.add(PReLU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))
		ADAM=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
		sgd=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer="adam")
		print("Training model...")
		earlystopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1)
		checkpointer = ModelCheckpoint(filepath=path+"tmp/weights.hdf5", verbose=0, save_best_only=True)
		model.fit(X_train, y_train, nb_epoch=1000, batch_size=128, verbose=2, 
		validation_data=(X_valid,y_valid), callbacks=[earlystopping,checkpointer])
		model.load_weights(path+"tmp/weights.hdf5")
		print("Generating submission...")
		pred += model.predict_proba(X_test)
		predTr += model.predict_proba(X_valid)
	predTr /= n_bag
	submissionTr.iloc[te] = predTr
	score[i]= log_loss(y_valid,predTr,eps=1e-15, normalize=True)
	print(score[i])
	i+=1

pred /= (nfold*n_bag)
print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))

make_submission(pred, ids, encoder, fname=path+'kerasNN4_tfidf.csv')
print(log_loss(labels,submissionTr.values,eps=1e-15, normalize=True))
submissionTr.to_csv(path+"kerasNN4_tfidf_retrain.csv",index_label='id')

# seed 1337
# nfold 2: 0.545350 + 0.0022795
# nfold 3: 0.523434 + 0.00791283

# nfold 3, bagging  5: 0.502407 + 0.0055760
# nfold 5, bagging  5: 0.490772 + 0.0127235
# nfold 5, bagging 10: 0.486696 + 0.0112351
