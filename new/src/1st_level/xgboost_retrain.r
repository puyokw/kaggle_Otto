library(readr)
library(data.table)
library(mice)
library(xgboost)
variance <- function(x) var(x)*(length(x)-1)/length(x) 
stddev <- function(x) sqrt(variance(x))
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
train <- fread(paste0(path,"train.csv"))
train <- as.data.frame(train)
train$id <- NULL

tSNE <- fread(paste0(path,"train_tsne.csv"))
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
stddev <- bst$test.mlogloss.std[minNum]
(sprintf("%s: %s+%s",minNum, tmp, stddev))

# eta 0.01: 2392, 0.452167 + 0.007736 
# eta 0.025 : 974, 0.452789+0.007242
# eta 0.05  : 462, 0.455121+0.007011
# eta 0.1   : 235, 0.458957+0.007107


set.seed(131)
bst <- xgboost(param=param, data=data.matrix(train[,c(-ncol(train))]), 
label=train$target, nrounds=minNum)

test  <- fread(paste0(path,"test.csv"))
test <- as.data.frame(test)
id <- test$id
test$id <- NULL
tSNE <- fread(paste0(path,"test_tsne.csv"))
tSNE <- as.data.frame(tSNE)
test <- cbind(test,tSNE)


for(i in 1:ncol(test)) { test[,i] <- as.numeric(test[,i])}
pred <- predict(bst,data.matrix(test))
pred <- matrix(pred,9,length(pred)/9)
pred <- t(pred)
submission <- data.frame(id,pred)
names(submission) <- c('id', paste0('Class_',1:9))
for(i in 1:ncol(submission)) { submission[,i] <- as.character(submission[,i])}
write_csv(submission,paste0(path,"xgb.csv"))

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
write_csv(submission,paste0(path,"xgb_retrain.csv"))

# nfold 10, eta 0.025:  "mean: 0.446157339370614, stddev: 0.00933124503154594"
# eta 0.01: "mean: 0.452150431771192, stddev: 0.00525177443564338"
# eta 0.025: "mean: 0.453862183666749, stddev: 0.00515462846734756"
# tsne cv score : "235: 0.458957+0.007107"
# score: 0.44172, 0.44407
# nfold 10: "mean: 0.451309402874783, stddev: 0.00871565654067067"
# nfold 8, "mean: 0.453524368582917, stddev: 0.01272269826227"
# nfold 6, "mean: 0.454186404467429, stddev: 0.0086829232150271"
# nfold 5, "mean: 0.458860357190608, stddev: 0.00455707136495085"
# nfold 3, "mean: 0.468769785104636, stddev: 0.0079971558899458"
# nfold 2, "mean: 0.48750612552813, stddev: 0.00130107011402603"

imp<-xgb.importance(names(train[,-ncol(train)]),model=bst)
print(imp)
xgb.plot.importance(imp)

#xgb.plot.tree(feature_names=names(train[,-ncol(train)]),model=bst, n_first_tree=2)
