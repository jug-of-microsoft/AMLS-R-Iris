options(repos="https://ftp.osuosl.org/pub/cran/")

install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
install.packages("ellipse")
install.packages(pkgs="kernlab")
install.packages("azuremlsdk")

library(caret)
library(optparse)
library(lattice)
library(ggplot2)
library(azuremlsdk)

# Setup Data Folder
options <- list(
  make_option(c("-d", "--data_folder"))
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste("Data Folder: ", opt$data_folder)

# Setup Output directory
output_dir = "outputs"
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}

#Load the data

# Use only when testing on local machine
#iris_df <- read.csv(file="IrisDataset.csv", header=TRUE, stringsAsFactors=TRUE)

# Use only when deploying to AzureML
iris_df <- read.csv(file=file.path(opt$data_folder, "IrisDataset.csv"), header=TRUE, stringsAsFactors=TRUE)

paste("Total Data Size: ", length(iris_df$Id))

# Split data set
#   set random seed
set.seed(12)

#   create a list of 70% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(iris_df$Species, p=0.70, list=FALSE)
paste("Validation Index: ", length(validation_index))

#   select 20% of the data for validation
test_df <- iris_df[-validation_index,]

#   use the remaining 70% of data to training and testing the models
train_df <- iris_df[validation_index,]

# Inspect Training Data
#   Dimesion
dim(train_df)

#   Attributes
sapply(train_df, class)

#   Peek
head(train_df)

#   Levels
levels(train_df$Species)

#   Distribution
percentage <- prop.table(table(train_df$Species)) * 100
cbind(freq=table(train_df$Species), percentage=percentage)

#   summarize attribute distributions
summary(train_df)


# Visualize
#   Univariate Plot
# split input and output
x <- train_df[,2:5]
y <- train_df[,6]

# boxplot for each attribute on one image
graphic_file <- "Attributes.jpg"
#png(filename = file.path(output_dir,"Attributes.png"))
jpeg(filename = file.path(output_dir,graphic_file),
     width = 480, 
     height = 480, 
     units = "px", 
     pointsize = 12,
     quality = 75,
     bg = "white")

par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

dev.off()

# Plot Species
graphic_file <- "Species.jpg"
#png(filename = file.path(output_dir,graphic_file))
jpeg(filename = file.path(output_dir,graphic_file),
     width = 480, 
     height = 480, 
     units = "px", 
     pointsize = 12,
     quality = 75,
     bg = "white")

plot(y)

dev.off()

# Multivariate Plots
#   scatterplot matrix
graphic_file <- "Features-ellipse.jpg"
#png(filename = file.path(output_dir,graphic_file))
jpeg(filename = file.path(output_dir,graphic_file),
     width = 480, 
     height = 480, 
     units = "px", 
     pointsize = 12,
     quality = 75,
     bg = "white")

featurePlot(x=x, y=y, plot="ellipse")

dev.off()

#   Boxplot per Specie
#   box and whisker plots for each attribute
graphic_file <- "Features-box.jpg"
#png(filename = file.path(output_dir, graphic_file))
jpeg(filename = file.path(output_dir,graphic_file),
     width = 480, 
     height = 480, 
     units = "px", 
     pointsize = 12,
     quality = 75,
     bg = "white")

featurePlot(x=x, y=y, plot="box")

dev.off()

# Gaussian-like Distribution of each feature per specie
# density plots for each attribute by class value
graphic_file <- "Features-Density.jpg"
#png(filename = file.path(output_dir,graphic_file))
jpeg(filename = file.path(output_dir,graphic_file),
     width = 480, 
     height = 480, 
     units = "px", 
     pointsize = 12,
     quality = 75,
     bg = "white")

scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

dev.off()

# Build Model
#   Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Models: Linear Algorithm, Non-Linear Algorithm, kNN, SVM, Random Forest
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=train_df, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=train_df, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=train_df, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=train_df, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=train_df, method="rf", metric=metric, trControl=control)

# Summarize Models
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# Visualize performance
graphic_file <- "ModelPerformance.jpg"
#png(filename = file.path(output_dir,graphic_file))
jpeg(filename = file.path(output_dir,graphic_file),
     width = 480, 
     height = 480, 
     units = "px", 
     pointsize = 12,
     quality = 75,
     bg = "white")

dotplot(results)

dev.off()

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, test_df)
confusionMatrix(predictions, test_df$Species)

# Log metric
log_metric_to_run("Accuracy", fit.lda$results$Accuracy)

saveRDS(fit.lda, file = "./outputs/model.rds")
message("Model saved")