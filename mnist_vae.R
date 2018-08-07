library(keras)
library(R6)

img_shape <- c(28, 28, 1)
batch_size <- 16
latent_dim <- 2L # 2D plane

input_img <- layer_input(shape=img_shape)

# VAE encoder network
x <- input_img %>%
    layer_conv_2d(filters=32, kernel_size=3, padding="same", activation="relu") %>%
    layer_conv_2d(filters=64, kernel_size=3, padding="same", activation="relu", strides=c(2, 2)) %>%
    layer_conv_2d(filters=64, kernel_size=3, padding="same", activation="relu") %>%
    layer_conv_2d(filters=64, kernel_size=3, padding="same", activation="relu")

shape_before_flattening <- k_int_shape(x)

x <- x %>%
    layer_flatten() %>%
    layer_dense(units=32, activation="relu")

# input image will be encoded in these two parameters
z_mean <- x %>%
    layer_dense(units=latent_dim)
z_log_var <- x %>%
    layer_dense(units=latent_dim)

# latent-space-sampling
sampling <- function(args) {
    c(z_mean, z_log_var) %<-% args
    epsilon <- k_random_normal(shape=list(k_shape(z_mean)[1], latent_dim), mean=0, stddev=1)
    z_mean + k_exp(z_log_var) * epsilon
}
z <- list(z_mean, z_log_var) %>%
    layer_lambda(sampling)

# VAE decoder network
decoder_input <- layer_input(k_int_shape(z)[-1]) # were z feeded into
x <- decoder_input %>%
    layer_dense(units=prod(as.integer(shape_before_flattening[-1])), activation="relu") %>% # up-sampling input
    layer_reshape(target_shape=shape_before_flattening[-1]) %>% # reshape z into a featue map with the same shape as before the last layer_flatten() in the encoder
    # use layer_conv_2d_transpose() and layer_conv_2d() to decode z into a featue map of the same size as the original input
    layer_conv_2d_transpose(filters=32, kernel_size=3, padding="same", activation="relu", strides=c(2, 2)) %>%
    layer_conv_2d(filters=1, kernel_size=3, padding="same", activation="sigmoid")
decoder <- keras_model(decoder_input, x) # creates decode model that turns decoder_input into the decoded image
z_decoded <- decoder(z) # applies z to recover the decoded z

# compute VAE loss
CustomVariationalLayer <- R6Class("CustomeVariationalLayer",
                                  inherit=KerasLayer,
                                  public=list(vae_loss=function(x, z_decoded) {
                                                x <- k_flatten(x)
                                                z_decoded <- k_flatten(z_decoded)
                                                xent_loss <- metric_binary_crossentropy(x, z_decoded)
                                                kl_loss <- -5e-4 * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis=-1L)
                                                k_mean(xent_loss + kl_loss)
                                              },
                                              call=function(inputs, mask=NULL) { # custom layer
                                                    x <- inputs[[1]]
                                                    z_decoded <- inputs[[2]]
                                                    loss <- self$vae_loss(x, z_decoded)
                                                    self$add_loss(loss, inputs=inputs)
                                                    x # output isn't used but layer has to returne something
                                              }))

# wrap the R6 class in a Keras layer
layer_variational <- function(object) {
    create_layer(CustomVariationalLayer, object, list())
}

# call the custom layer on the input
# decode output
y <- list(input_img, z_decoded) %>%
    layer_variational()

# VAE training
vae <- keras_model(input_img, y)

vae %>% compile(optimizer="rmsprop", loss=NULL)

mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

x_train <- x_train / 255
x_train <- array_reshape(x_train, dim=c(dim(x_train), 1))

x_test <- x_test / 255
x_test <- array_reshape(x_test, dim=c(dim(x_test), 1))

vae %>% fit(x=x_train,
            y=NULL,
            epochs=10,
            batch_size=batch_size,
            validation_data=list(x_test, NULL))

# sampling grid of points from 2D latent space
n <- 15 # grid of 15x15 digits
digit_size <- 28
# transform linearly spaced coordinates using the qnorm
# produces values of the latent variable z
# prior of the latent space is Gaussian
grid_x <- qnorm(seq(0.05, 0.95, length.out=n))
grid_y <- qnorm(seq(0.05, 0.95, length.out=n))

op <- par(mfrow=c(n, n), mar=c(0, 0, 0, 0), bg="black")
for (i in 1:length(grid_x)) {
    yi <- grid_x[[i]]
    for (j in 1:length(grid_y)) {
        xi <- grid_y[[i]]
        z_sample <- matrix(c(xi, yi), nrow=1, ncol=2)
        # repeatss z times to form a complete batch
        z_sample <- t(replicate(batch_size, z_sample, simplify="matrix"))
        x_decoded <- decoder %>%
            predict(z_sample, batch_size=batch_size)
        # reshape from 28x28x1 to 28x28
        digit <- array_reshape(x_decoded[1,,,], dim=c(digit_size, digit_size))
        plot(as.raster(digit))
    }
}
par(op)
