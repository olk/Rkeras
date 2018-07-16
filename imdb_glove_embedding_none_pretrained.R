library(keras)

# prepare raw IMDB data
imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg=0, pos=1)
    dir_name <- file.path(train_dir, label_type)
    for (fname in list.files(dir_name, pattern=glob2rx("*.txt"), full.names=TRUE)) {
        texts <- c(texts, readChar(fname, file.info(fname)$size))
        labels <- c(labels, label)
    }
}

# tokenizing the text
maxlen <- 100 # cut of the text after 100 words
training_samples <- 200 # trains 200 samples
validation_samples <- 10000 # validates 10000 samples
max_words <- 10000 # consider only the to 10000 words in the data set

tokenizer <- text_tokenizer(num_words=max_words) %>% fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)

word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen=maxlen)

labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")
cat("Shape of label tensor:", dim(labels), "\n")

indices <- sample(1:nrow(data)) # splits the data into a training and a validation set, ordered: negatives first, positive second
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):(training_samples + validation_samples)]

x_train <- data[training_indices,]
y_train <- labels[training_indices]

x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

# prepare GloVe word embeddings
glove_dir <- "~/Downloads/glove.6B"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))

embeddings_index <- new.env(has=TRUE, parent=emptyenv())
for (i in 1:length(lines)) {
    line <- lines[[i]]
    values <- strsplit(line, " ")[[1]]
    word <- values[[1]]
    embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")

# model
embedding_dim <- 100
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
summary(model)
history <- model %>%
    fit(x_train, y_train, epochs=20, batch_size=32, validation_data=list(x_val, y_val))
plot(history)
