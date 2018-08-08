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

max_features <- 10000 # number of words considered as features
maxlen <- 500 # cur off texts after thisnumber of words
batch_size <- 32

cat("Loading data ...\n")
imdb <- dataset_imdb(num_words=max_features)
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences\n")

cat("Pad sequences (samples x time)\n")
input_train <- pad_sequences(input_train, maxlen=maxlen)
input_test <- pad_sequences(input_test, maxlen=maxlen)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

model <- keras_model_sequential() %>%
    layer_embedding(input_dim=max_features, output_dim=32) %>%
    layer_simple_rnn(units=32) %>%
    layer_dense(units=1, activation="sigmoid")

model %>% compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=c("acc")
)

history <- model %>% fit(input_train,
                         y_train,
                         epochs=10,
                         batch_size=128,
                         validation_split=0.2)

plot(history)
