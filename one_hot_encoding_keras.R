library(keras)

print("samples:")
samples

tokenizer <- text_tokenizer(num_words=10) %>% # create tokenizer; takes up to 1000 words
    fit_text_tokenizer(samples) # build the word index

sequences <- texts_to_sequences(tokenizer, samples) # transofrm strings into list of integer inidices
print("sequences:")
sequences

one_hot_results <- texts_to_matrix(tokenizer, samples, mode="binary")
print("one_hot_results:")
one_hot_results

word_index <- tokenizer$word_index
print("word_index:")
word_index
