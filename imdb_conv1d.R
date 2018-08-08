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

cat("Loading data ...\n")
imdb <- dataset_imdb(num_words=max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
cat(length(x_train), "train sequences\n")
cat(length(x_test), "test sequences\n")

cat("Pad sequences (samples x time)\n")
x_train <- pad_sequences(x_train, maxlen=maxlen)
x_test <- pad_sequences(x_test, maxlen=maxlen)
cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")

model <- keras_model_sequential() %>%
    layer_embedding(input_dim=max_features,
                    output_dim=32,
                    input_length=maxlen) %>%
    layer_conv_1d(filters=32,
                  kernel_size=7,
                  activation="relu") %>%
    layer_max_pooling_1d(pool_size=5) %>%
    layer_conv_1d(filters=32,
                  kernel_size=7,
                  activation="relu") %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(units=1)

summary(model)

model %>% compile(optimizer=optimizer_rmsprop(lr=1e-4),
                  loss="binary_crossentropy",
                  metrics=c("acc"))

history <- model %>% fit(x_train,
                         y_train,
                         epochs=10,
                         batch_size=128,
                         validation_split=0.2)

plot(history)
