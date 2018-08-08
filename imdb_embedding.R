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

max_features <- 10000 # number of words to consider
maxlen <- 20 # cut of the text after this number of words
imdb <- dataset_imdb(num_words=max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb # loads the data as list of integers
x_train <- pad_sequences(x_train, maxlen=maxlen) # tunrs the lists of integers into 2D integer tensor of shae (samples, maxlen)
x_test <- pad_sequences(x_test, maxlen=maxlen) # tunrs the lists of integers into 2D integer tensor of shae (samples, maxlen)
model <- keras_model_sequential() %>%
    layer_embedding(input_dim=10000, output_dim=8, input_length=maxlen) %>% #max input length to the embedding layer
    layer_flatten() %>% # flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, maxlen)
    layer_dense(units=1, activation="sigmoid") # classifier on top
model %>% compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=c("acc")
)
summary(model)
history <- model %>% fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
plot(history)
