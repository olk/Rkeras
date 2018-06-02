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
