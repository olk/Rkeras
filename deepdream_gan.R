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

latent_dim <- 32
height <- 32
width <- 32
channels <- 3

# GAN generator network
generator_input <- layer_input(shape=c(latent_dim))

generator_output <- generator_input %>%
    # transform input into 16x16, 128 channel featue map
    layer_dense(units=128 * 16 * 16) %>%
    layer_activation_leaky_relu() %>%
    layer_reshape(target_shape=c(16, 16, 128)) %>%

    layer_conv_2d(filters=256,
                  kernel_size=5,
                  padding="same") %>%
    layer_activation_leaky_relu() %>%

    # upsampling to 32x32
    layer_conv_2d_transpose(filters=256, kernel_size=4, strides=2, padding="same") %>%
    layer_activation_leaky_relu() %>%

    layer_conv_2d(filters=256,
                  kernel_size=5,
                  padding="same") %>%
    layer_activation_leaky_relu() %>%
    layer_conv_2d(filters=256,
                  kernel_size=5,
                  padding="same") %>%
    layer_activation_leaky_relu() %>%

    # create 32x32 3-channel feature map
    layer_conv_2d(filters=channels, kernel_size=7, activation="tanh", padding="same")

# generator model maps input of shape (latent_dim) into an image of shape (32, 32, 3)
generator <- keras_model(generator_input, generator_output)

# GAN discriminator network
discriminator_input <- layer_input(shape=c(height, width, channels))

discriminator_output <- discriminator_input %>%
    layer_conv_2d(filters=128, kernel_size=3) %>%
    layer_activation_leaky_relu() %>%
    layer_conv_2d(filters=128,
                  kernel_size=4,
                  strides=2) %>%
    layer_activation_leaky_relu() %>%
    layer_conv_2d(filters=128,
                  kernel_size=4,
                  strides=2) %>%
    layer_activation_leaky_relu() %>%
    layer_conv_2d(filters=128,
                  kernel_size=4,
                  strides=2) %>%
    layer_activation_leaky_relu() %>%
    layer_flatten() %>%
    layer_dropout(rate=0.4) %>% # important
    layer_dense(units=1, activation="sigmoid") # classification layer

# turns (32, 32, 3) into a binary classification decision (fake/real)
discriminator <- keras_model(discriminator_input, discriminator_output)

discriminator_optimizer <- optimizer_rmsprop(lr=0.0008,
                                             clipvalue=1.0, # gradient clipping
                                             decay=1e-8) # to stabilize training

discriminator %>% compile(optimizer=discriminator_optimizer,
                         loss="binary_crossentropy")

# adversarial network
freeze_weights(discriminator) # discriminator weights are not trainable

gan_input <- layer_input(shape=c(latent_dim))
gan_output <- discriminator(generator(gan_input))
gan <- keras_model(gan_input, gan_output)

gan_optimizer <- optimizer_rmsprop(lr=0.0004,
                                   clipvalue=1.0,
                                   decay=1e-8)

gan %>% compile(optimizer=gan_optimizer,
                loss="binary_crossentropy")

# GAN training
cifar10 <- dataset_cifar10() # loads CIFAR10 data
c(c(x_train, y_train), c(x_test, y_test)) %<-% cifar10

x_train <- x_train[6 == as.integer(y_train),,,] # select from images (class 6)
x_train <- x_train / 255 # normalize data


iterations <- 20000
batch_size <- 20
save_dir <- "/tmp/deepdream"
dir.create(save_dir)

start <- 1

for (step in 1:iterations) {
    # sample random points in the latent space
    random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), nrow=batch_size, ncol=latent_dim)
    # decode sampled random points to fake images
    generated_images <- generator %>%
        predict(random_latent_vectors)

    # combines generated witrh real images
    stop <- start + batch_size - 1
    real_images <- x_train[start:stop,,,]
    rows <- nrow(real_images)
    combined_images <- array(0, dim=c(2 * rows, dim(real_images)[-1]))
    combined_images[1:rows,,,] <- generated_images
    combined_images[(rows + 1):(2 * rows),,,] <- real_images

    # assembles labels, disciminating real from fake images
    labels <- rbind(matrix(1, nrow=batch_size, ncol=1),
                    matrix(0, nrow=batch_size, ncol=1))

    # add random noise to the labels -> important
    labels <- labels + (0.5 * array(runif(prod(dim(labels))), dim=dim(labels)))

    # train the discriminator
    d_loss <- discriminator %>%
        train_on_batch(combined_images, labels)

    # sample random points in the latent space
    random_latent_vectors <- matrix(rnorm(batch_size * latent_dim),
                                    nrow=batch_size,
                                    ncol=latent_dim)

    misleading_targets <- array(0, dim=c(batch_size, 1)) # fake postive labels -> real images (not true)

    # train the generator via GAN model, where discriminator weigths are frozen
    a_loss <- gan %>% train_on_batch(random_latent_vectors,
                                     misleading_targets)

    start <- start + batch_size
    if (start > (nrow(x_train) - batch_size)) {
        start <- 1
    }

    if (0 == step %% 100) {
        save_model_weights_hdf5(gan, "data/gan.h5") # save model weights
        cat("disciminator loss:", d_loss, "\n")
        cat("adversarial loss:", a_loss, "\n")

        image_array_save(generated_images[1,,,] * 255,
                         path=file.path(save_dir, paste0("generated_frog", step, ".png")))

        image_array_save(real_images[1,,,] * 255,
                         path=file.path(save_dir, paste0("real_frog", step, ".png")))
    }
}
