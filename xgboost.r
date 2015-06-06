library(xgboost)
library(methods)
library(car)
 
train <- read.csv("train.csv",header=TRUE,stringsAsFactors = F)
test <- read.csv("test.csv",header=TRUE,stringsAsFactors = F)
train <- train[,-1]
test <- test[,-1]

# normalization
train[,-ncol(train)] <- log(1+train[,-ncol(train)])
test <- log(1+test)

y <- train[,ncol(train)]
y <- gsub('Class_','',y)
y <- as.integer(y)-1
 
x <- rbind(train[,-ncol(train)],test)
x <- as.matrix(x)
x <- matrix(as.numeric(x),nrow(x),ncol(x))
 
trind <- 1:length(y)
teind <- (nrow(train)+1):nrow(x)

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "max_depth" = 12,
              "eta" = 0.002,
              "min_child_weight" = 8,
              "max_delta_step" = 0,
              "subsample" = 0.9,
              "colsample_bytree" = 0.8
              )
set.seed(131)
nround <- 14275
bst <- xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
 
pred <- predict(bst,x[teind,])
pred <- matrix(pred,9,length(pred)/9)
pred <- t(pred)

pred <- format(pred, digits=2,scientific=F) # shrink file size
pred <- data.frame(1:nrow(pred),pred)
names(pred) <- c('id', paste0('Class_',1:9))
write.csv(pred,file="submission.csv", quote=FALSE,row.names=FALSE)
