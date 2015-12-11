library(readr)
library(data.table)
library(Rtsne)
library(Metrics)
path <- "../Data/"
train <- read_csv(paste0(path,"train_tfidf.csv"))
train <- as.data.frame(train)
dim(train)
test <- read_csv(paste0(path,"test_tfidf.csv"))
test <- as.data.frame(test)
dim(test)
TsneDim <- 3 # 2 or 3
tmp <- as.factor(train$target)
train$id <-NULL
test$id <- NULL
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

write_csv(train,paste0(path,"tfidf_train_tsne.csv"),col_names=T)
write_csv(test,paste0(path,"tfidf_test_tsne.csv"),col_names=T)
