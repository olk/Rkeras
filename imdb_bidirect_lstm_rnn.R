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

max_features <- 1000 # number of words to consider as feature
maxlen <- 500 # cut off texts after this number of words

imdb <- dataset_imdb(num_words=max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

x_train <- lapply(x_train, rev)
x_test <- lapply(x_test, rev)

x_train <- pad_sequences(x_train, maxlen=maxlen)
x_test <- pad_sequences(x_test, maxlen=maxlen)

model <- keras_model_sequential() %>%
    layer_embedding(input_dim=max_features, output_dim=128) %>%
    bidirectional(layer_lstm(units=32)) %>%
    layer_dense(units=1, activation="sigmoid")

model %>% compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=c("acc"))

history <- model %>% fit(x_train,
                         y_train,
                         epochs=10,
                         batch_size=128,
                         vailidation_split=0.2)
plot(history)
