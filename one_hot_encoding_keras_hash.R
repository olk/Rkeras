library(hashFunction)

samples <- c("ABC NOP HIJ", "EFG KLM NOP QRST")
print("samples:")
samples

dimensionality <- 10
max_length <- 10
results <- array(0, dim=c(length(samples), max_length, dimensionality))
for (i in 1:length(samples)) {
    sample <- samples[[i]]
    words <- head(strsplit(sample, " ")[[1]], n=max_length)
    for (j in 1:length(words)) {
        index <- abs(spooky.32(words[[j]])) %% dimensionality
        results[[i, j, index]] <- 1
    }
}
print("results:")
results
