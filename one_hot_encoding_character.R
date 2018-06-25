samples <- c("ABC NOP HIJ", "EFG KLM NOP QRST")
print("samples:")
samples

ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar))
print("ascii_tokens:")
ascii_tokens

token_index <- c(1:length(ascii_tokens))
names(token_index) <- ascii_tokens
print("token_index:")
token_index

max_length <- 50
results <- array(0, dim=c(length(samples), max_length, length(token_index)))
for (i in 1:length(samples)) {
    sample <- samples[[i]]
    characters <- strsplit(sample, "")[[1]]
    for (j in 1:length(characters)) {
        character <- characters[[j]]
        results[i, j, token_index[[character]]] <- 1
    }
}
print("results:")
results
