library(readr)
library(data.table)
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
xgb <- fread(paste0(path,"xgb_2nd.csv"))
xgb <- as.data.frame(xgb)
NN <- fread(paste0(path,"kerasNN2_2nd.csv"),header=T)
NN <- as.data.frame(NN)
ET <- fread(paste0(path,"extraTree_2nd.csv"),header=T)
ET <- as.data.frame(ET)
id <- xgb$id
xgb$id <- NULL
NN$id <- NULL
ET$id <- NULL 
for(i in 1:ncol(xgb)){xgb[,i] <- as.numeric(xgb[,i])}
for(i in 1:ncol(NN)){NN[,i] <- as.numeric(NN[,i])}
for(i in 1:ncol(ET)){ET[,i] <- as.numeric(ET[,i])}
pred <- (NN * 0.4 + xgb * 0.6)
submission <- data.frame(id,pred)
for(i in 1:ncol(submission)){ submission[,i] <- as.character(submission[,i])}
write_csv(submission,paste0(path,"xgbNN_ensemble.csv"))
