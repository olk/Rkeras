library(keras)

# two-class classification (== binary classification)
# classify IMDB reviews as postive or negative based on the text content of the review
# load dataset with the top occuring 10000 words
imdb <- dataset_imdb(num_words=10000)

# `train_data` and `test_data` are lists of reviews
# each review is a list of word indices (encondign a sequence of words)
# `train_label` and `test_label` are lists of 0s and 1s, where 0==negative, 1==postive
c(c(train_data, train_label), c(test_data, test_label)) %<-% imdb
#str(train_data)
#str(train_label)
train_data[[1]]

# data preparation
vectorize_sequences <- function(sequences, dimension=10000) {
    results <- matrix(0, nrow=length(sequences), ncol=dimension) # creates an all-zero matrix
    for (i in 1:length(sequences)) {
        results[i, sequences[[i]]] <- 1 # set specific indices of result[i] to `1`
    }
    results
}
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
#str(x_train[1,])
y_train <- as.numeric(train_label)
y_test <- as.numeric(test_label)

# build network
model <- keras_model_sequential() %>%
    # activation function `relu`: rectified linear unit function
    layer_dense(units=16, activation="relu", input_shape=c(10000)) %>%
    layer_dense(units=16, activation="relu") %>%
    # activation function `sigmoid`: sigmoid function
    layer_dense(units=1, activation="sigmoid")
model %>% compile(
    optimizer=optimizer_rmsprop(lr=0.001),
    loss=loss_binary_crossentropy,
    metrics=metric_binary_accuracy
)

# create validation set
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# training
history <- model %>% fit(partial_x_train,
                         partial_y_train,
                         epochs=4,
                         batch_size=512,
                         validation_data=list(x_val, y_val))
str(history)
#plot(history)

# evaluate test data
results <- model %>% evaluate(x_test, y_test)
results

# compute likelihood of review being postive by using `predict` method
x_test[1,] %>% head(10) # [1] 1 1 0 1 1 1 1 1 1 1
model %>% predict(x_test[1:10,])
