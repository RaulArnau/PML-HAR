
# ------------------------
# Course Project
# ------------------------
library(caret)
library(ggplot2)

# Clean workspace
rm(list=ls())
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
SAVE_RESULTS = FALSE
APPLY_PCA = FALSE



# read data
dataSet <- read.csv('pml-training.csv', na.strings=c("NA", ""))
validationSet <- read.csv('pml-testing.csv', na.strings=c("NA", ""))
dim(dataSet)
dim(validationSet)

# too many columns for head -> show a few
dataSet[1:5, c('user_name', 'classe', 'num_window', 'roll_belt', 'pitch_belt', 'yaw_belt')]
# a lot of NA
mean(is.na(dataSet))

# two options to get rid of NA values:
# 1- remove features with majority of NA's
# 2- replace with zeros and let PCA get rid of them

# convert interesting features to numeric values, including user name
dataSetNum <- dataSet[, c(2, 7:159)]
validationSetNum <- validationSet[, c(2, 7:159)]
y <- dataSet$classe

dataSetNum <- data.frame(sapply(dataSetNum, as.numeric))
validationSetNum <- data.frame(sapply(validationSetNum, as.numeric))

# replace NA with Zeros
# dataSetNum[is.na(dataSetNum)] <- 0
# validationSetNum[is.na(validationSetNum)] <- 0

# remove NA
completeFeatures <- complete.cases(t(dataSetNum))
dataSetNum <- dataSetNum[, completeFeatures]
validationSetNum <- validationSetNum[, completeFeatures]


# Split training set into training and testing to build our predictor
set.seed(3232)
inTrain <- createDataPartition(y=y, p=0.75, list=FALSE)
training <- dataSetNum[inTrain, ]
testing <- dataSetNum[-inTrain, ]
dim(training)
dim(testing)


# clean-up

# remove near-zero variates
# nzv <- nearZeroVar(training)
# training <- training[,-nzv]
# testing <- testing[, -nzv]
# validation <- validationSetNum[, -nzv]

# remove highly correlated features
corIdx <- findCorrelation(cor(training))
training <- training[, -corIdx]
testing <- testing[, -corIdx]
validation <- validation[, -corIdx]


# use PCA to get rid of some features
# keep only those that explain the 90% of the variability
if (APPLY_PCA){
    modPca <- preProcess(training, method = 'pca', threshold = 0.95)
    trainingPca <- predict(modPca, training)
    testingPca <- predict(modPca, testing)
    validationPca <- predict(modPca, validation)
} else {
    trainingPca <- training
    testingPca <- testing
    validationPca <- validation
}
# add the class to the features datasets
trainingPca$classe <- y[inTrain]
testingPca$classe <- y[-inTrain]



set.seed(32335)
fitControl <- trainControl(# k-fold CV
                            method = "repeatedcv", 
                            number = 5, # folds
                            # repeated 5 times
                            repeats = 5)
modRf <- train(classe ~., 
               method='rf',
               trControl = fitControl, 
               data = trainingPca)


# in sample error
trainingPrediction <- predict(modRf, trainingPca)
table(trainingPca$classe, trainingPrediction)
confusionMatrix(trainingPca$classe, trainingPrediction)

# out of sample error
testingPrediction <- predict(modRf, testingPca)
table(testingPca$classe, testingPrediction)

confusionMatrix(testingPca$classe, testingPrediction)

# plotting the resampling profile
trellis.par.set(caretTheme())
plot(modRf)

# Prediction
prediction <- predict(modRf, validationPca)

if (SAVE_RESULTS) 
{
    # save results in the submission format
    pml_write_files = function(x) {
        n = length(x)
        for (i in 1:n) {
            filename = paste0("problem_id_", i, ".txt")
            write.table(x[i], file = filename, 
                        quote = FALSE, 
                        row.names = FALSE, 
                        col.names = FALSE)
        }
    }
    pml_write_files(prediction)
}

answers <- c("B", "A", "B", "A", "A", "E", "D", "B", "A", "A", "B", "C", "B", "A", "E", "E", "A", "B", "B", "B")
