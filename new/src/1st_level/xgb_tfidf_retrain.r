library(readr)
library(data.table)
library(xgboost)
variance <- function(x) var(x)*(length(x)-1)/length(x) 
stddev <- function(x) sqrt(variance(score))
mlogloss <- function(actual,pred){
	actual <- as.integer(as.factor(actual))
	probs <- rep(0,length(actual))
	for(i in 1:length(actual)) { probs[i] <- pred[i,actual[i]]}
	probs <- as.numeric(probs)
	probs[which(probs>1-1e-15)] <- 1-1e-15
	probs[which(probs<1e-15)] <- 1e-15
	return(-(1/length(actual))*sum(log(probs)))
}

path <- "../Data/"
train <- fread(paste0(path,"train_tfidf.csv"))
train <- as.data.frame(train)
train$id <- NULL
tSNE <- fread(paste0(path,"tfidf_train_tsne.csv"))
tSNE <- as.data.frame(tSNE)
train <- cbind(train,tSNE)

y <- train$target
y <- gsub('Class_','',y)
y <- as.integer(y)-1 #xgboost take features in [0,numOfClass)
train$target <- NULL
train$target <- y

for(i in 1:(ncol(train)-1)) { train[,i] <- as.numeric(train[,i])}

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
			  "max_depth" = 12,
			  "eta" = 0.01, # 0.002
			  "min_child_weight" = 10, # default 1; 4 ,gbm: 10 
			  "subsample" =  0.9, # default 1, 0.9
			  "colsample_bytree" = 0.8 # default 1 0.8
			  )
cv.nround <- 10000
gc(TRUE)
set.seed(131)
bst <- xgb.cv(param=param, data=data.matrix(train[,c(-ncol(train))]), label=train$target,
nrounds=cv.nround, nfold=5, early.stop.round=10)
print(tmp <- min(bst$test.mlogloss.mean))
(minNum <- which.min(bst$test.mlogloss.mean))

# eta 0.01: 2327, 0.464444 
# eta 0.025: 979, 0.465486
# eta 0.1: 225, 0.471174

set.seed(131)
bst <- xgboost(param=param, data=data.matrix(train[,c(-ncol(train))]), 
label=train$target, nrounds=minNum)

test  <- fread(paste0(path,"test_tfidf.csv"))
test <- as.data.frame(test)
tSNE <- fread(paste0(path,"tfidf_test_tsne.csv"))
tSNE <- as.data.frame(tSNE)
test <- cbind(test,tSNE)

id <- test$id
test$id <- NULL

for(i in 1:ncol(test)) { test[,i] <- as.numeric(test[,i])}
pred <- predict(bst,data.matrix(test))
pred <- matrix(pred,9,length(pred)/9)
pred <- t(pred)
submission <- data.frame(id,pred)
names(submission) <- c('id', paste0('Class_',1:9))
write_csv(submission,paste0(path,"xgb_tfidf.csv"))

# retrain
rm(pred)
library(cvTools)
NAME <- paste0('Class_',1:9)
tmp <- matrix(rep(NA,9),nrow(train),9)
tmp <- as.data.frame(tmp)
colnames(tmp) <- NAME
submission <- cbind(id=1:nrow(train),tmp)
nfold <- 5
cv.folds <- cvFolds(nrow(train),nfold)
cvs <- cv.folds$which
id <- 1:nrow(train)
score <- rep(0,nfold)

for(NUMCV in 1:nfold){
test.x <- train[which(cvs==NUMCV),-ncol(train)]
test.y <- train[which(cvs==NUMCV),ncol(train)]
train.x <- train[which(cvs!=NUMCV),-ncol(train)]
train.y <- train[which(cvs!=NUMCV),ncol(train)]

set.seed(131)
bst <- xgboost(param=param, data=data.matrix(train.x), label=train.y, nrounds=minNum)
pred <- predict(bst,data.matrix(test.x))
pred <- matrix(pred,9,length(pred)/9)
submission[which(cvs==NUMCV),2:10] <- t(pred)
print(score[NUMCV] <- mlogloss(test.y,as.data.frame(t(pred))))
}
sprintf("mean: %s, stddev: %s", mean(score), stddev(score))

for(i in 1:ncol(submission)) { submission[,i] <- as.character(submission[,i])}
write_csv(submission,paste0(path,"xgb_tfidf_retrain.csv"))

# eta 0.01, nfold 5: "mean: 0.463530922400871, stddev: 0.00154293768677079"
# eta 0.025, nfold 10: "mean: 0.45832260734072, stddev: 0.00886914421281806"
# eta 0.025, nfold 5: "mean: 0.464373214332066, stddev: 0.00147590469750059"
# nfold 5: "mean: 0.470238490824427, stddev: 0.00150969970537236"


imp<-xgb.importance(names(train[,-ncol(train)]),model=bst)
print(imp)
xgb.plot.importance(imp)

#xgb.plot.tree(feature_names=names(train[,-ncol(train)]),model=bst, n_first_tree=2)
