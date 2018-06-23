library(keras)

# use pretrained VGG16 model
conv_base <- application_vgg16(weights="imagenet", # from which to initialize the weights
                               include_top=FALSE, # do not include the dense connected classifier on top of the network
                               input_shape=c(150, 150, 3)) # shape of the image tensor the will be feed to the network
conv_base


base_dir <- "~/Downloads/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

# `imange_data_generator()` converts image files on disk into batches of preprocessed tensors
datagen <- image_data_generator(rescale=1/255) # rescale all images (1/255 -> [0,1]
batch_size <- 20

# extracting features using the pretrained VGG16 network in `conv_base`
extract_features <- function(directory, sample_count) {
    features <- array(0, dim=c(sample_count, 4, 4, 512)) # feature map has shape (4, 4, 512)
    labels <- array(0, dim=c(sample_count))

    generator <- flow_images_from_directory(directory=directory,
                                            generator=datagen,
                                            target_size=c(150, 150),
                                            batch_size=batch_size,
                                            class_mode="binary") # binary labels, because binary_crossentropy loss function used

    i <- 0
    while (TRUE) {
        batch <- generator_next(generator)
        inputs_batch <- batch[[1]]
        labels_batch <- batch[[2]]
        # predict features from input
        features_batch <- conv_base %>% predict(inputs_batch)

        index_range <- ((i * batch_size) + 1) : ((i + 1) * batch_size)
        features[index_range,,,] <- features_batch
        labels[index_range] <- labels_batch

        i <- i + 1
        if (i * batch_size >= sample_count) {
            break
        }
    }

    # return value
    list(features=features, labels=labels)
}


train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

# features are of shape (samples, 4, 4, 512) after etraction
# reshape them to (samples, 4 * 4 * 512)
reshape_features <- function(features) {
    array_reshape(features, dim=c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)


# build model; densly connected classifier
model <- keras_model_sequential() %>%
    layer_dense(units=256, activation="relu", input_shape=4 * 4 * 512) %>%
    layer_dropout(rate=0.5) %>% # use droput for regularization
    layer_dense(units=1, activation="sigmoid")

model %>% compile(optimizer=optimizer_rmsprop(lr=2e-5),
                  loss="binary_crossentropy",
                  metrics=c("accuracy"))

history <- model %>% fit(train$features, train$labels,
                         epochs=30,
                         batch_size=20,
                         validation_data=list(validation$features, validation$labels))

plot(history)
