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
    bidirectional(layer_gru(units=32),
                  input_shape=list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units=1)

model %>% compile(optimizer=optimizer_rmsprop(),
                  loss="mae")

history <- model %>% fit(x_train,
                         y_train,
                         epochs=10,
                         batch_size=128,
                         vailidation_split=0.2)
plot(history)
