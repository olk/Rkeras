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

# MNIST dataset provided by keras library
mnist <- dataset_mnist()
train_images <- mnist$train$x # array of shape (60000, 28, 28)
train_labels <- mnist$train$y
test_images <- mnist$test$x # array of shape (10000, 28, 28)
test_labels <- mnist$test$y

str(train_images)
str(train_labels)

network <- keras_model_sequential() %>% # 
    layer_dense(units=512, activation="relu", input_shape=c(28*28)) %>% # chain of two dense layers
    layer_dense(units=10, activation="softmax")

network %>%
    compile(
            optimizer="rmsprop", # mechanism to update the network based on data and loss function
            loss="categorical_crossentropy", # loss function
            metrics=c("accuracy"))

train_images <- array_reshape(train_images, c(60000, 28*28)) # 2D tensor of shape (60000, 784), 60000 samples, 785-dim feature vector
train_images <- train_images/255 # scale grey value to range [0,1]

test_images <- array_reshape(test_images, c(10000, 28*28)) # 2D tensor of shape (10000, 784), 10000 samples, 785-dim feature vector
test_images <- test_images/255 # scale grey value to range [0,1]

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

network %>% fit(train_images, train_labels, epochs=5, batch_size=128) # interate on training in mini-batches of 128 samples, 5 iterations over colple training set

metrics <- network %>% evaluate(test_images, test_labels)
metrics

network %>% predict_classes(test_images[1:10, ])
