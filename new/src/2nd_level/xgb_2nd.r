library(readr)
library(data.table)
# @param 
# actual is vector, pred is data.frame
# mlogloss(actual,pred)
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
train <- fread(paste0(path,"train_2nd.csv"))
train <- as.data.frame(train)
train$id <- NULL

tsne <- fread(paste0(path,"train_2nd_tsne.csv"))
tsne <- as.data.frame(tsne)
train <- cbind(train,tsne)

y <- train$target
y <- gsub('Class_','',y)
y <- as.integer(y)-1 #xgboost take features in [0,numOfClass)
train$target <- NULL 
train$target <- y

for(i in 1:(ncol(train)-1)) { train[,i] <- as.numeric(train[,i])}

# make test data
test <- fread(paste0(path,"test_2nd.csv"))
test <- as.data.frame(test)
test$id <- NULL

tsne <- fread(paste0(path,"test_2nd_tsne.csv"))
tsne <- as.data.frame(tsne)
test <- cbind(test,tsne)

for(i in 1:ncol(test)) { test[,i] <- as.numeric(test[,i])}

predAll <- rep(0,9)
nfold <- 1 
set.seed(131)
rnd <- sample(100000,nfold)
rnd <- 131 
for(i in 1:nfold){
library(xgboost)
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
			  "max_depth" = 4,
			  "eta" = 0.02, # 0.002
			  "min_child_weight" = 1, # 10 
			  "subsample" =  0.9, # 0.9
			  "colsample_bytree" = 0.8 # 0.8
			  )
cv.nround <- 10000
gc(TRUE)
set.seed(rnd[i])
bst <- xgb.cv(param=param, data=data.matrix(train[,c(-ncol(train))]), label=train$target,
nrounds=cv.nround, nfold=10, early.stop.round=10)
print(tmp <- min(bst$test.mlogloss.mean))
(minNum <- which.min(bst$test.mlogloss.mean))

# tsne
# eta 0.025: 602, 0.410142  

#minNum <- 518
set.seed(rnd[i])
bst <- xgboost(param=param, data=data.matrix(train[,c(-ncol(train))]), 
label=train$target, nrounds=minNum)

pred <- predict(bst,data.matrix(test))
pred <- matrix(pred,9,length(pred)/9)
predAll <- predAll + t(pred)
}
predAll <- predAll / nfold
submission <- data.frame(id=1:nrow(test),predAll)
names(submission) <- c('id', paste0('Class_',1:9))
head(submission,10)
for(i in 1:ncol(submission)) { submission[,i] <- as.character(submission[,i])}
write_csv(submission,paste0(path,"xgb_2nd.csv"))

imp<-xgb.importance(names(train[,-ncol(train)]),model=bst)
print(imp)
xgb.plot.importance(imp)

#xgb.plot.tree(feature_names=names(train[,-ncol(train)]),model=bst, n_first_tree=2)


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
}

for(i in 1:ncol(submission)) { submission[,i] <- as.character(submission[,i])}
write_csv(submission,paste0(path,"xgb_2nd_retrain.csv"))

mlogloss(train$target,submission[,2:10])
