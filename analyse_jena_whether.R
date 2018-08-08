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

library(ggplot2)
library(readr)
library(tibble)

dir.create("~/Downloads/jena_climate", recursive=TRUE)
download.file("https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
              "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip")
unzip("~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip",
      exdir="~/Downloads/jena_climate")

data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

#glimpse(data)

#ggplot(data, aes(x=1:nrow(data), y=`T (degC)`)) + geom_line() # plot the temperature timeseries
ggplot(data[1:1440,], aes(x=1:1440, y=`T (degC)`)) + geom_line() # plot the first 10 days
