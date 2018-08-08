# MIT License
# 
# Copyright (c) 2017 François Chollet
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

library(keras)

# single label, mutliclassification
# classify Reuters news into 46 mutually exclusive topics based on the text content of the news
# load dataset with the top occuring 10000 words
reuters <- dataset_reuters(num_words=10000)

c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters
length(train_data)
length(test_data)
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

to_one_hot <- function(labels, dimension=46) {
    results <- matrix(0, nrow=length(labels), ncol=dimension)
    for (i in 1:length(labels)) {
        results[i, labels[[i]] + 1] <- 1
    }
    results
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

# build network
model <- keras_model_sequential() %>%
    # activation function `relu`: rectified linear unit function
    layer_dense(units=64, activation="relu", input_shape=c(10000)) %>%
    layer_dense(units=64, activation="relu") %>%
    # activation function `sigmoid`: sigmoid function
    layer_dense(units=46, activation="softmax")
model %>% compile(
    optimizer=optimizer_rmsprop(lr=0.001),
    loss=loss_binary_crossentropy,
    metrics=metric_binary_accuracy
)

# create validation set
val_indices <- 1:1000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- one_hot_train_labels[val_indices,]
partial_y_train <- one_hot_train_labels[-val_indices,]

# training
history <- model %>% fit(partial_x_train,
                         partial_y_train,
                         epochs=9,
                         batch_size=512,
                         validation_data=list(x_val, y_val))
#str(history)
plot(history)

# evaluate test data
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# prediction
predictions <- model %>% predict(x_test)
dim(predictions)
sum(predictions[1,])
which.max(predictions[1,])
