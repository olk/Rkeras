library(keras)

model <- keras_model_sequential() %>%
    layer_conv_2d(filters=32, kernel_size=c(3, 3), activation="relu", input_shape=c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size=c(2, 2)) %>%
    layer_conv_2d(filters=64, kernel_size=c(3, 3), activation="relu") %>%
    layer_max_pooling_2d(pool_size=c(2, 2)) %>%
    layer_conv_2d(filters=128, kernel_size=c(3, 3), activation="relu") %>%
    layer_max_pooling_2d(pool_size=c(2, 2)) %>%
    layer_conv_2d(filters=128, kernel_size=c(3, 3), activation="relu") %>%
    layer_max_pooling_2d(pool_size=c(2, 2)) %>%
    layer_flatten() %>%
    layer_dropout(rate=0.5) %>%
    layer_dense(units=512, activation="relu") %>%
    layer_dense(units=1, activation="sigmoid")

model

model %>%
    compile(
            optimizer=optimizer_rmsprop(lr=1e-4),
            loss="binary_crossentropy",
            metrics=c("acc"))

base_dir <- "~/Downloads/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")

# data augmentation generates more training data from existing training samples
# by augmenting samples via a number of random transformations
# -> the model does NOT see the exact sampe picture twice
# `imange_data_generator()` converts image files on disk into batches of preprocessed tensors
datagen <- image_data_generator(rescale=1/255, # rescale images to range [0,1]
                                rotation_range=40, # andomly rotate image; range [[0°,180°]
                                width_shift_range=0.2, # randomly translate image horizontal
                                height_shift_range=0.2, # randomly translate image vertically
                                shear_range=0.2, # randomly apply hear transformations
                                zoom_range=0.2, # randomly toom into image
                                horizontal_flip=TRUE) # strategy for filling in newly created pixels
test_datagen <- image_data_generator(rescale=1/255)

train_generator <- flow_images_from_directory(train_dir, # target directory
                                              datagen, # data generator
                                              target_size=c(150, 150), # resize all images to 150x150
                                              batch_size=32,
                                              class_mode="binary") # binary_crossentropy loss function requires binary labels
validation_generator <- flow_images_from_directory(validation_dir,
                                                   test_datagen,
                                                   target_size=c(150, 150),
                                                   batch_size=32,
                                                   class_mode="binary")

history <- model %>% fit_generator(train_generator,
                                   steps_per_epoch=100,
                                   epochs=100,
                                   validation_data=validation_generator,
                                   validation_steps=50)

model %>% save_model_hdf5("data/cats__and__dogs__small_2.h5")

plot(history)
