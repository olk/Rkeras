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
