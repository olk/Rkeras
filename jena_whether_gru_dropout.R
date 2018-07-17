library(keras)
library(readr)
library(tibble)

data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

# data preparation
# convert data to floating-point matrix
data <- data.matrix(data[,-1])
# normalizing
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center=mean, scale=std)

# generator yielding timeseries samples
#  yields batches of data from the recent past, alogn with the target temperature in the future
generator <- function(data, lookback, delay, min_index, max_index, shuffle=FALSE, batch_size=128, step=6) {
    if (is.null(max_index)) {
        max_index <- nrow(data) - delay - 1
    }
    i <- min_index + lookback
    return(function() {
        if (shuffle) {
            rows <- sample(c((min_index+lookback):max_index), size=batch_size)
        } else {
            if (i + batch_size >= max_index) {
                i <<- min_index + lookback
            }
            rows <- c(i:min(i+batch_size, max_index))
            i <<- i + length(rows)
        }
        samples <- array(0, dim=c(length(rows), lookback/step, dim(data)[[-1]]))
        targets <- array(0, dim=c(length(rows)))
        for (j in 1:length(rows)) {
            indices <- seq(rows[[j]] - lookback, rows[[j]], length.out=dim(samples)[[2]])
            samples[j,,] <- data[indices,] # input data
            targets[[j]] <- data[rows[[j]] + delay, 2] # target temperatures
        }
        return(list(samples, targets))
    })
}

# preparing training, validation and test generators
lookback <- 14400 # how many timestamps back the input data should go; observations, go 10 days back; a timestep is 10min
step <- 6 # observations sampled at one data point per 1h
delay <- 144 # how many timesteps in the future the target should be; will be 24h in the future
batch_size <- 128 # number of samples per batch

train_gen <- generator(data,
                       lookback=lookback,
                       delay=delay,
                       min_index=1,
                       max_index=200000,
                       shuffle=TRUE,
                       step=step,
                       batch_size=batch_size)

val_gen <- generator(data,
                      lookback=lookback,
                      delay=delay,
                      min_index=200001,
                      max_index=300000,
                      step=step,
                      batch_size=batch_size)

test_gen <- generator(data,
                      lookback=lookback,
                      delay=delay,
                      min_index=300001,
                      max_index=NULL,
                      step=step,
                      batch_size=batch_size)

val_steps <- (300000 - 200001 - lookback) / batch_size # how many steps to draw from val_gen in order to see the entire validation set
test_steps <- (nrow(data) - 300001 - lookback) / batch_size # how many steps to draw from test_gen in order to see the entire test set

 #evaluate_naive_method <- function() {
 #    batch_maes <- c()
 #    for (step in 1:val_steps) {
 #        c(camples, targets) %<-% val_gen()
 #        preds <- samples[, dim(samples)[[2]], 2]
 #        mae <- mean(abs(preds - targets))
 #        batch_maes <- c(batch_maes, mae)
 #    }
 #    print(mean(batch_maes))
 #}
 #evaluate_naive_method()

model <- keras_model_sequential() %>%
    layer_gru(units=32,
              dropout=0.2,
              recurrent_dropout=0.2,
              input_shape=list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units=1)

model %>% compile(
    optimizer=optimizer_rmsprop(),
    loss="mae"
)

history <- model %>% fit_generator(train_gen,
                                   steps_per_epoch=500,
                                   epochs=40,
                                   validation_data=val_gen,
                                   validation_steps=val_steps)

plot(history)
