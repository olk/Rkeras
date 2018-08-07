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

dir.create("/tmp/log") # create directory for TensorBoard log files
tensorboard("/tmp/log") # launch TensorBoard
callbacks = list( callback_tensorboard( log_dir = "/tmp/log",
                                        histogram_freq = 1, # records activation histograms every 1 epoche
                                        embeddings_freq = 1) ) # records embedding data every 1 epoche


history <- model %>% fit(x_train,
                         y_train,
                         epochs=20,
                         batch_size=128,
                         validation_split=0.2,
                         callbacks = callbacks)

plot(history)
