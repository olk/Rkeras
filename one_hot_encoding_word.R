samples <- c("ABC NOP HIJ", "EFG KLM NOP QRST")
print("samples:")
samples

token_index <- list()
for (sample in samples) {
    for (word in strsplit(sample, " ")[[1]]) {
        if (!word %in% names(token_index)) {
            token_index[[word]] <- length(token_index) + 2
        }
    }
}
print("token_index:")
token_index

max_length <- 10
results <- array(0, dim=c(length(samples), max_length, max(as.integer(token_index))))
for (i in 1:length(samples)) {
    sample <- samples[[i]]
    words <- head(strsplit(sample, " ")[[1]], n=max_length)
    for (j in 1:length(words)) {
        index <- token_index[[words[[j]]]]
        results[[i, j, index]] <- 1
    }
}
print("results:")
results
