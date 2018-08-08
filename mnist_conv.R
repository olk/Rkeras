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

model <- keras_model_sequential() %>%
    layer_conv_2d(filters=32, kernel_size=c(3, 3), activation="relu", input_shape=c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size=c(2, 2)) %>%
    layer_conv_2d(filters=64, kernel_size=c(3, 3), activation="relu") %>%
    layer_max_pooling_2d(pool_size=c(2, 2)) %>%
    layer_conv_2d(filters=64, kernel_size=c(3, 3), activation="relu") %>%
    layer_flatten() %>%
    layer_dense(units=64, activation="relu") %>%
    layer_dense(units=10, activation="softmax")

model

model %>%
    compile(
            optimizer="rmsprop", # mechanism to update the model based on data and loss function
            loss="categorical_crossentropy", # loss function
            metrics=c("accuracy"))

# MNIST dataset provided by keras library
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28, 28, 1)) # 4D tensor of shape (60000, 28, 28, 1), 60000 samples
train_images <- train_images/255 # scale grey value to range [0,1]

test_images <- array_reshape(test_images, c(10000, 28, 28, 1)) # 4D tensor of shape (10000, 28, 28, 1), 10000 samples
test_images <- test_images/255 # scale grey value to range [0,1]

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% fit(train_images, train_labels, epochs=5, batch_size=64) # interate on training in mini-batches of 128 samples, 5 iterations over colple training set

result <- model %>% evaluate(test_images, test_labels)
result
