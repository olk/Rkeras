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

 original_dataset_dir <- "~/Downloads/kaggle_original_data"

 base_dir <- "~/Downloads/cats_and_dogs_small"
 dir.create(base_dir)

 train_dir <- file.path(base_dir, "train")
 dir.create(train_dir)
 validation_dir <- file.path(base_dir, "validation")
 dir.create(validation_dir)
 test_dir <- file.path(base_dir, "test")
 dir.create(test_dir)

 train_cats_dir <- file.path(train_dir, "cats")
 dir.create(train_cats_dir)

 train_dogs_dir <- file.path(train_dir, "dogs")
 dir.create(train_dogs_dir)

 validation_cats_dir <- file.path(validation_dir, "cats")
 dir.create(validation_cats_dir)

 validation_dogs_dir <- file.path(validation_dir, "dogs")
 dir.create(validation_dogs_dir)

 test_cats_dir <- file.path(test_dir, "cats")
 dir.create(test_cats_dir)

 test_dogs_dir <- file.path(test_dir, "dogs")
 dir.create(test_dogs_dir)

 fnames <- paste0("cat.",  1:1000, ".jpg")
 file.copy(file.path(original_dataset_dir, fnames),
           file.path(train_cats_dir))

 fnames <- paste0("cat.",  1001:1500, ".jpg")
 file.copy(file.path(original_dataset_dir, fnames),
           file.path(validation_cats_dir))

 fnames <- paste0("cat.",  1501:2000, ".jpg")
 file.copy(file.path(original_dataset_dir, fnames),
           file.path(test_cats_dir))

 fnames <- paste0("dog.",  1:1000, ".jpg")
 file.copy(file.path(original_dataset_dir, fnames),
           file.path(train_dogs_dir))

 fnames <- paste0("dog.",  1001:1500, ".jpg")
 file.copy(file.path(original_dataset_dir, fnames),
           file.path(validation_dogs_dir))

 fnames <- paste0("dog.",  1501:2000, ".jpg")
 file.copy(file.path(original_dataset_dir, fnames),
           file.path(test_dogs_dir))

 # sanity check

cat("total training cat images: ", length(list.files(train_cats_dir)), "\n")
cat("total training dogs images: ", length(list.files(train_dogs_dir)), "\n")
cat("total validation cats images: ", length(list.files(validation_cats_dir)), "\n")
cat("total validation dogs images: ", length(list.files(validation_dogs_dir)), "\n")
cat("total test cats images: ", length(list.files(test_cats_dir)), "\n")
cat("total test dogs images: ", length(list.files(test_dogs_dir)), "\n")
