# Step 1: Install and load required libraries
# Run these install commands only once (comment them out after installation)

#install.packages("data.table")  # For fast reading of large datasets
#install.packages("caret")       # For machine learning
#install.packages("randomForest") # For Random Forest model
#install.packages("e1071")       # For SVM and other models
#install.packages("dplyr")       # For data manipulation
#install.packages("lightgbm")    # For LightGBM model

# Load the libraries
library(data.table)
library(caret)
library(randomForest)
library(e1071)
library(dplyr)

# Set working directory to your project folder
setwd("c:/Environment/Purely Claude")  #your working directory

print("Libraries loaded successfully!")

# Step 2: Load the preprocessed data
print("Loading the dataset... This may take a few minutes for 2.8GB file")

# Use fread for fast reading of large CSV files
data <- fread("C:/Environment/College/Parkinson/park_encoded.csv") #your data csv file

# Check if data loaded successfully
print(paste("Dataset loaded successfully!"))
print(paste("Number of rows:", nrow(data)))
print(paste("Number of columns:", ncol(data)))

# Display first few rows
print("First few rows of the dataset:")
head(data)

# Display column names
print("Column names:")
print(colnames(data))

# Display structure of the data
print("Data structure:")
str(data)

# Step 3: Data exploration and target variable analysis
print("STEP 3: DATA EXPLORATION")

# Check for missing values
print("Checking for missing values:")
missing_values <- colSums(is.na(data))
print(missing_values[missing_values > 0])

if(sum(missing_values) == 0) {
  print("No missing values found!")
}

# Check the distribution of target variable (Parkinsons_encoded)
print("\nTarget Variable Distribution:")
print(table(data$Parkinsons_encoded))

# Calculate percentage distribution
target_dist <- prop.table(table(data$Parkinsons_encoded)) * 100
print("\nPercentage Distribution:")
print(target_dist)

# Visualize class distribution
print("\nClass Balance Check:")
class_0 <- sum(data$Parkinsons_encoded == 0)
class_1 <- sum(data$Parkinsons_encoded == 1)

print(paste("Class 0 (No Parkinson):", class_0, "samples"))
print(paste("Class 1 (Parkinson):", class_1, "samples"))
print(paste("Ratio:", round(class_1/class_0, 2)))

# Check data types
print("\nData Types of all columns:")
print(sapply(data, class))

# Step 4: Separate features and target variable
print("==================================================")
print("STEP 4: SEPARATING FEATURES AND TARGET")
print("==================================================")

# Convert target to factor (required for classification in caret)
data$Parkinsons_encoded <- as.factor(data$Parkinsons_encoded)

# Separate target variable (y) and features (X)
target_col <- "Parkinsons_encoded"
feature_cols <- setdiff(colnames(data), target_col)

print(paste("Target variable:", target_col))
print(paste("Number of features:", length(feature_cols)))
print("\nFeature columns:")
print(feature_cols)

# Create X (features) and y (target)
X <- data[, ..feature_cols]
y <- data[[target_col]]

print(paste("\nFeatures (X) shape: ", nrow(X), "rows x", ncol(X), "columns"))
print(paste("Target (y) length:", length(y)))
print(paste("Target levels:", paste(levels(y), collapse = ", ")))

# Verify the separation worked
print("\nFirst few rows of features:")
print(head(X, 3))
print("\nFirst few values of target:")
print(head(y, 10))

# Step 5: Train-Test Split
print("==================================================")
print("STEP 5: TRAIN-TEST SPLIT")
print("==================================================")

# Set seed for reproducibility
set.seed(42)

# Create train-test split (80% train, 20% test)
# Using createDataPartition from caret (maintains class distribution)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)

# Split features
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]

# Split target
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Display split information
print(paste("Total samples:", nrow(data)))
print(paste("\nTraining set size:", nrow(X_train), "samples"))
print(paste("Testing set size:", nrow(X_test), "samples"))

# Check class distribution in training set
print("\nTraining set class distribution:")
print(table(y_train))
print(prop.table(table(y_train)) * 100)

# Check class distribution in testing set
print("\nTesting set class distribution:")
print(table(y_test))
print(prop.table(table(y_test)) * 100)

print("\nTrain-Test split completed successfully!")

# Step 6: Install and load LightGBM
print("==================================================")
print("STEP 6: LIGHTGBM MODEL SETUP")
print("==================================================")

# Install lightgbm (run only once, then comment out)

# Load lightgbm library
library(lightgbm)

print("LightGBM library loaded successfully!")

# Prepare data for LightGBM
# LightGBM requires matrix format and numeric labels (0, 1)
print("\nPreparing data for LightGBM...")

# Convert features to matrix
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

# Convert factor labels to numeric (0, 1)
y_train_numeric <- as.numeric(as.character(y_train))
y_test_numeric <- as.numeric(as.character(y_test))

print(paste("Training features shape:", nrow(X_train_matrix), "x", ncol(X_train_matrix)))
print(paste("Training labels range:", min(y_train_numeric), "to", max(y_train_numeric)))

print("\nData preparation completed!")

# Step 7: Create LightGBM datasets and set parameters
print("==================================================")
print("STEP 7: CREATE LIGHTGBM DATASETS AND PARAMETERS")
print("==================================================")

# Create LightGBM datasets
print("Creating LightGBM dataset objects...")

# Training dataset
dtrain <- lgb.Dataset(
  data = X_train_matrix,
  label = y_train_numeric
)

# Testing dataset (for validation during training)
dtest <- lgb.Dataset.create.valid(
  dtrain,
  data = X_test_matrix,
  label = y_test_numeric
)

print("LightGBM datasets created successfully!")

# Set LightGBM parameters
params <- list(
  objective = "binary",           # Binary classification
  metric = "binary_logloss",      # Evaluation metric
  boosting = "gbdt",              # Gradient boosting decision tree
  num_leaves = 31,                # Number of leaves in one tree
  learning_rate = 0.05,           # Learning rate
  feature_fraction = 0.9,         # Randomly select 90% of features
  bagging_fraction = 0.8,         # Randomly select 80% of data
  bagging_freq = 5,               # Perform bagging every 5 iterations
  verbose = 1                     # Print training progress
)

print("\nModel parameters set:")
print(params)

print("\nReady for model training!")

# Step 8: Train the LightGBM model
print("==================================================")
print("STEP 8: TRAIN LIGHTGBM MODEL")
print("==================================================")

print("Starting model training...")
print("This may take several minutes due to large dataset size...")
print("")

# Record start time
start_time <- Sys.time()

# Train the model
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,                  # Number of boosting iterations
  valids = list(test = dtest),    # Validation data
  early_stopping_rounds = 10,     # Stop if no improvement for 10 rounds
  verbose = 1                     # Print progress every iteration
)

# Record end time
end_time <- Sys.time()
training_time <- end_time - start_time

print("")
print("==================================================")
print("MODEL TRAINING COMPLETED!")
print("==================================================")
print(paste("Training time:", round(training_time, 2), units(training_time)))
print(paste("Best iteration:", model$best_iter))
print(paste("Best score:", model$best_score))

# Step 9: Make predictions on test set
print("==================================================")
print("STEP 9: MAKE PREDICTIONS")
print("==================================================")

print("Making predictions on test set...")

# Predict probabilities
y_pred_prob <- predict(model, X_test_matrix)

# Convert probabilities to class labels (0 or 1)
# Using 0.5 as threshold
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)

print("Predictions completed!")
print(paste("Total predictions made:", length(y_pred)))

# Show first few predictions
print("\nFirst 10 predictions:")
print(data.frame(
  Actual = y_test_numeric[1:10],
  Predicted = y_pred[1:10],
  Probability = round(y_pred_prob[1:10], 4)
))

print("\nPrediction summary:")
print(table(y_pred))

# Step 10: Evaluate model performance
print("==================================================")
print("STEP 10: MODEL EVALUATION")
print("==================================================")

# Create confusion matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_test_numeric)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate performance metrics
TP <- conf_matrix[2, 2]  # True Positives
TN <- conf_matrix[1, 1]  # True Negatives
FP <- conf_matrix[2, 1]  # False Positives
FN <- conf_matrix[1, 2]  # False Negatives

# Accuracy
accuracy <- (TP + TN) / sum(conf_matrix)

# Precision
precision <- TP / (TP + FP)

# Recall (Sensitivity)
recall <- TP / (TP + FN)

# F1 Score
f1_score <- 2 * (precision * recall) / (precision + recall)

# Specificity
specificity <- TN / (TN + FP)

print("\n==================================================")
print("PERFORMANCE METRICS:")
print("==================================================")
print(paste("Accuracy:   ", round(accuracy * 100, 2), "%"))
print(paste("Precision:  ", round(precision * 100, 2), "%"))
print(paste("Recall:     ", round(recall * 100, 2), "%"))
print(paste("F1-Score:   ", round(f1_score * 100, 2), "%"))
print(paste("Specificity:", round(specificity * 100, 2), "%"))

# Step 10b: Check for overfitting - Compare training vs testing accuracy
print("==================================================")
print("CHECKING FOR OVERFITTING")
print("==================================================")

print("Calculating training set accuracy...")

# Predict on training set
y_train_pred_prob <- predict(model, X_train_matrix)
y_train_pred <- ifelse(y_train_pred_prob > 0.5, 1, 0)

# Training confusion matrix
train_conf_matrix <- table(Predicted = y_train_pred, Actual = y_train_numeric)
print("\nTraining Confusion Matrix:")
print(train_conf_matrix)

# Training accuracy
train_accuracy <- sum(diag(train_conf_matrix)) / sum(train_conf_matrix)

print("\n==================================================")
print("TRAINING vs TESTING COMPARISON:")
print("==================================================")
print(paste("Training Accuracy:  ", round(train_accuracy * 100, 2), "%"))
print(paste("Testing Accuracy:   ", round(accuracy * 100, 2), "%"))
print(paste("Difference:         ", round((train_accuracy - accuracy) * 100, 2), "%"))

if (train_accuracy - accuracy < 0.02) {
  print("\n✓ Model is generalizing well! No significant overfitting.")
} else {
  print("\n⚠ Warning: Possible overfitting detected (>2% difference)")
}

#Optional step to save the model

# Step 11: Save the trained model
print("==================================================")
print("STEP 11: SAVE THE MODEL")
print("==================================================")

# Save the LightGBM model
model_filename <- "parkinson_lightgbm_model.txt"
lgb.save(model, model_filename)

print(paste("Model saved successfully as:", model_filename))
print(paste("File location:", file.path(getwd(), model_filename)))

# Save model performance metrics for reference
metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "Specificity"),
  Training = c(train_accuracy, NA, NA, NA, NA),
  Testing = c(accuracy, precision, recall, f1_score, specificity)
)

write.csv(metrics, "model_metrics.csv", row.names = FALSE)
print("\nModel metrics saved as: model_metrics.csv")

print("\n==================================================")
print("MODEL TRAINING COMPLETE!")
print("==================================================")
