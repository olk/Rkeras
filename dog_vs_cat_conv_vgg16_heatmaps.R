library(keras)
library(magick)
library(viridis)

# use pretrained VGG16 model
model <- application_vgg16(weights="imagenet") # keep densley conencted layers

# preprocess input images
img_path <- "./data/creative_commons_elephants.jpg"
img <- image_load(img_path, target_size=c(224, 224)) %>%
        image_to_array() %>% # array of shape (224, 224, 3)
        array_reshape(dim=c(1, 224, 224, 3)) %>% # add sample dimension in order to transform it to a batch
        imagenet_preprocess_input() # preprocess the batch; does channel-wise color normalization

preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top=3)[[1]] # top three predicted classes for the image

pos <- which.max(preds[1,]) # position of the class with the highest prediction

# Grad-CAM
african_elephant_output <- model$output[, pos] # `african elephant` in preditionvector
last_conv_layer <- model %>% get_layer("block5_conv3") # output feature map; last convolution layer
grads <- k_gradients(african_elephant_output, last_conv_layer$output)[[1]] # gradient of the `african elephant` class
pooled_grads <- k_mean(grads, axis=c(1, 2, 3)) # vector of shape (512); each entry is the mean intensity of the gradient over the specific feature-map channel
iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,])) # iterator access the values of pooled_grads and output-feature map
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
for (i in 1:512) {
    # multiply each channel in the featue-map array by its impackt at the prediction for the `elephant` class
    conv_layer_output_value[,,i] <- conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1, 2), mean) # channel-wise mean of the resulting feature-map == heatmap

# heatmap post-processing
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)

write_heatmap <- function(heatmap, filename, width=224, height=224, bg="white", col=terrain.colors(12)) {
    png(filename, width=width, height=height, bg=bg)
    op=par(mar=c(0,0,0,0))
    on.exit({par(op); dev.off()}, add=TRUE)
    rotate <- function(x) {
        t(apply(x, 2, rev))
    }
    image(rotate(heatmap), axes=FALSE, asp=1, col=col)
}
write_heatmap(heatmap, "./data/elephant_heatmap.png")


# superimposing the heatmap
image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)

pal <- col2rgb(viridis(20), alpha=TRUE)
alpha <- floor(seq(0, 255, length=ncol(pal)))
pal_col <- rgb(t(pal), alpha=alpha, maxColorValue=255)
write_heatmap(heatmap, "./data/elephant_overlay.png", width=14, height=14, bg=NA, col=pal_col)
image_read("./data/elephant_overlay.png") %>%
    image_resize(geometry, filter="quadratic") %>%
    image_composite(image, operator="blend", compose_args="20") %>%
    plot()
