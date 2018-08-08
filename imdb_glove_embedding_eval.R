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

imdb_dir <- "~/Downloads/aclImdb"
test_dir <- file.path(imdb_dir, "test")

maxlen <- 100 # cut of the text after 100 words
max_words <- 10000 # consider only the to 10000 words in the data set

embedding_dim <- 100

labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg=0, pos=1)
    dir_name <- file.path(test_dir, label_type)
    for (fname in list.files(dir_name, pattern=glob2rx("*.txt"), full.names=TRUE)) {
        texts <- c(texts, readChar(fname, file.info(fname)$size))
        labels <- c(labels, label)
    }
}

tokenizer <- text_tokenizer(num_words=max_words) %>%
    fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen=maxlen)
y_test <- as.array(labels)

# model
model <- keras_model_sequential() %>%
    layer_embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen) %>% #max input length to the embedding layer
    layer_flatten() %>% # flatten the 3D tensor of embeddings into a 2D tensor of shape (samples, maxlen)
    layer_dense(units=32, activation="relu") %>%
    layer_dense(units=1, activation="sigmoid") # classifier on top

# training and evaluation
model %>% compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=c("accuracy")
)

# evaluate
results <- model %>%
    load_model_weights_hdf5("data/pre_trained_glove_model.h5") %>%
    evaluate(x_test, y_test)
results
