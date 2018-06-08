library(ggplot2)
library(keras)

# regression, predicting a continuous value
# predict the median price of Boston suburb in mid-1970s
# data set of crime rate, local property tax rate ...
# few data points (only 506), each feature has a different scale
housing <- dataset_boston_housing()

c(c(train_data, train_targets), c(test_data, test_targets)) %<-% housing
str(train_data)
str(test_data)
str(train_targets)

# data preparation
# do not feed neural networks with too heterogeneous data
# do feature-wise normalization: for each feature in the input data
#   - subtract the mean of the feature
#   - devide by the standard derivation
# -> feature becomes centered around 0 and has a unit standard derivation
# ! never use a quantity (e.g. like the mean, standard derviation) from the test data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center=mean, scale=std)
test_data <- scale(test_data, center=mean, scale=std)

# the less training data you have, the worse overfitting will be
# use small networks to mitigate overfitting
create_model <- function() {
    model <- keras_model_sequential() %>%
        layer_dense(units=64, activation="relu", input_shape=dim(train_data)[[2]]) %>%
        layer_dense(units=64, activation="relu") %>%
        # singel unit + no activation function
        # typical for scalar regression (e.g. predicting a single continuous value)
        # an activation fucntion would constrain the range the output can take
        # this last layer is linear, network can learn to predict values in any range
        layer_dense(units=1)

    model %>% compile(
        optimizer="rmsprop",
        loss="mse", # mean squared error, square of the difference between the prediction and the target
        metrics=c("mae") # eman absolute error, absolute value of the difference between the predictions and the targets
    )
}

# create validation set
# use K-fold cross validation by small data set
# split available data into K paritions (typically K=4/5)
# instantiating K identical models and train each one one K-1 paritions while evaluating on the remaining partition
# validation score == average of the K validation scores obtained
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks=k, labels=FALSE)
num_epochs <- 30
#num_epochs <- 100
all_scores <- c()
all_mae_histories <- NULL
for (i in 1:k) {
    cat("processing fold #", i, "\n")
    val_indices <- which(folds==i, arr.ind=TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]

    model <- create_model()
    history <- model %>%
        fit(partial_train_data, partial_train_targets,
            validation_data=list(val_data, val_targets),
            epochs=num_epochs, batch_size=1, vebose=0)
    results <- model %>% evaluate(val_data, val_targets, verbose=0)
    all_scores <- c(all_scores, results$mean_absolute_error)
    mae_history <- history$metrics$val_mean_absolute_error
    all_mae_histories <- rbind(all_mae_histories, mae_history)
}

all_scores
mean(all_scores)

average_mae_history <- data.frame(epoch=seq(1:ncol(all_mae_histories)), validation_mae=apply(all_mae_histories, 2, mean))
#ggplot(average_mae_history, aes(x=epoch, y=validation_mae)) + geom_smooth()

# train finaly on all training data
model <- create_model()
model %>% fit(train_data, train_targets, epochs=30, batch_size=16, verbose=0)
result <- model %>% evaluate(test_data, test_targets)
result
