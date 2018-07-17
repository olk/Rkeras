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
