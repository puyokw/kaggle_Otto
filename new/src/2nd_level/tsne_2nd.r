library(readr)
library(data.table)
library(Rtsne)
library(Metrics)
path <- "../Data/"
train <- fread(paste0(path,"train_2nd.csv"))
train <- as.data.frame(train)
train$id <-NULL
for(i in 1:(ncol(train)-1)) { train[,i] <- as.numeric(train[,i])}
test <- fread(paste0(path,"test_2nd.csv"))
test <- as.data.frame(test)
test$id <- NULL
for(i in 1:ncol(test)) { test[,i] <- as.numeric(test[,i])}
TsneDim <- 3 # 2 or 3
tmp <- as.factor(train$target)
train$target <- NULL
x <- rbind(train,test)
dim(x)
N <- nrow(train)
rm(train)
rm(test)
gc()

print("constructing t-sne")
set.seed(131)
tsne <- Rtsne(x, dims = TsneDim, perplexity = 50, initial_dims = 50,
theta = 0.5, check_duplicates = F, pca = T, verbose=TRUE, max_iter = 500)

colors = rainbow(length(unique(tmp)))
names(colors) = unique(tmp)

if(TsneDim==2){
  plot(tsne$Y[1:N,], t='n', main="tsne")
  text(tsne$Y[1:N,], labels=tmp, col=colors[tmp])
}
if(TsneDim==3){
  library(rgl)
  plot3d(tsne$Y[1:N,1],tsne$Y[1:N,2],tsne$Y[1:N,3],col=colors[tmp])
}

train <- tsne$Y[1:N,]
test <- tsne$Y[(N+1):nrow(tsne$Y),]
if(TsneDim==2){
colnames(train) <- c("dim1","dim2")
colnames(test) <- c("dim1","dim2")
}
if(TsneDim==3){
colnames(train) <- c("dim1","dim2","dim3")
colnames(test) <- c("dim1","dim2","dim3")
}

train <- as.data.frame(train)
test <- as.data.frame(test)
for(i in 1:3){
train[,i] <- as.character(train[,i])
test[,i] <- as.character(test[,i])
}
gc()

write_csv(train,paste0(path,"train_2nd_tsne.csv"),col_names=T)
write_csv(test,paste0(path,"test_2nd_tsne.csv"),col_names=T)
