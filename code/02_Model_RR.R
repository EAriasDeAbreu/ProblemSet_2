# ******************************************************************************
# ******************************************************************************
# *Authors: Edmundo + Lu + Juandi
# *Coder: Edmundo Arias De Abreu
# *Project: Machine Learning -- P_set 2
# *Data: provided by pset
# *Stage: Model Development -- Random Forest 
# 
# *Last checked: 14.10.2024
# 
# /*
# ******************************************************************************
# *                                 Contents                                   *
# ******************************************************************************
#   
# This script aims to...
#
#
#     Inputs:
#       -
# 
#     Output:
#       -
# 
# ******************************************************************************
# Clear the Environment
# ---------------------------------------------------------------------------- #

rm(list = ls())

# ---------------------------------------------------------------------------- #
# Load Necessary Libraries
# ---------------------------------------------------------------------------- #

if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

pacman::p_load(
  tidyverse,  # for data manipulation and visualization
  caret,      # for machine learning
  glmnet,     # for regularized generalized linear models
  randomForest, # for random forest models
  xgboost,    # for gradient boosting
  data.table, # for efficient data manipulation
  lubridate,  # for date/time operations
  ggplot2,    # for advanced plotting
  scales,     # for scale functions for visualization
  gridExtra,  # for arranging multiple plots
  pROC,       # for ROC curves
  ranger,     # for random forests
  dplyr,      # for data manipulation
  DT,         # for interactive tables
  smotefamily       # for smote
)

# ---------------------------------------------------------------------------- #
# Data Import 
# ---------------------------------------------------------------------------- #

train_data <- read.csv("/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/scripts/train_hogares.csv")
test_data <- read.csv("/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/scripts/test_hogares.csv")

# extra: person-level
train_person <- read.csv("/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/scripts/train_personas.csv")
test_person <- read.csv("/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/scripts/test_personas.csv")



colnames(train_data)
colnames(test_data)
# ---------------------------------------------------------------------------- #
# Adversarial Validation
# ---------------------------------------------------------------------------- #

# Explanation: ...




# Function to perform adversarial validation
adversarial_validation <- function(data, test_size = 0.2) {
  # handle missing values                                                       # TODO: make more robust
  data <- data %>%
    mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
    mutate_if(is.factor, ~ifelse(is.na(.), mode(.), .))
  
  # check if 'target' column exists and has more than one unique value
  if ("target" %in% names(data) && length(unique(data$target)) > 1) {
    set.seed(42)
    train_index <- createDataPartition(data$target, p = 1 - test_size, list = FALSE)
  } else {
    set.seed(42)
    train_index <- sample(1:nrow(data), size = floor((1 - test_size) * nrow(data)))
  }
  
  # split data into train and pseudo-test sets
  train_data <- data[train_index, ]
  pseudo_test_data <- data[-train_index, ]
  
  # add is_test column and convert to factor
  train_data$is_test <- factor(0)
  pseudo_test_data$is_test <- factor(1)
  combined_df <- bind_rows(train_data, pseudo_test_data)
  
  # feature selection for adversarial training, excluding identifiers
  features <- setdiff(names(combined_df), c("is_test", "target", "id", "Fex_c", "Fex_dpto"))
  
  # split combined data for validation
  set.seed(43)
  adv_train_index <- sample(1:nrow(combined_df), size = floor(0.7 * nrow(combined_df)))
  adv_train_data <- combined_df[adv_train_index, ]
  adv_val_data <- combined_df[-adv_train_index, ]
  
  # confimr is_test is a factor
  adv_train_data$is_test <- factor(adv_train_data$is_test)
  adv_val_data$is_test <- factor(adv_val_data$is_test)
  
  # train a Random Forest (RF) model
  rf_model <- ranger(
    factor(is_test) ~ ., 
    data = adv_train_data[, c(features, "is_test")],
    num.trees = 100,
    importance = 'impurity',
    probability = TRUE
  )
  
  # predictions on validation set
  predictions <- predict(rf_model, data = adv_val_data[, features])$predictions[, "1"]
  
  # AUC score
  auc_score <- roc(as.numeric(as.character(adv_val_data$is_test)), predictions)$auc
  
  # F1 score
  actuals <- as.numeric(as.character(adv_val_data$is_test))
  binary_predictions <- ifelse(predictions > 0.5, 1, 0)
  confusion_matrix <- table(Actual = actuals, Predicted = binary_predictions)
  
  # Handle division by zero
  if (sum(confusion_matrix[,2]) == 0 || sum(confusion_matrix[2,]) == 0 || confusion_matrix[2,2] == 0) {
    precision <- 0
    recall <- 0
    f1_score <- 0
  } else {
    precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
    recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
    f1_score <- 2 * (precision * recall) / (precision + recall)
  }
  
  # feature importances
  importances <- rf_model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  return(list(
    auc_score = auc_score, 
    f1_score = f1_score, 
    importances = importances_df,
    train_data = train_data,
    pseudo_test_data = pseudo_test_data,
    confusion_matrix = confusion_matrix
  ))
}

# function to get mode of a vector
mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# run adversarial validation
result <- adversarial_validation(train_data, test_size = 0.2)

# Print results
print(paste("AUC Score:", result$auc_score))
print(paste("F1 Score:", result$f1_score))
print("Top 10 most important features:")
print(head(result$importances, 10))



# confusion matrix 
cm_data <- as.data.frame(result$confusion_matrix)
names(cm_data) <- c("Actual", "Predicted", "Frequency")

ggplot(cm_data, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), color = "white", size = 10) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Confusion Matrix",
       x = "Predicted",
       y = "Actual") +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.5))




# ---------------------------------------------------------------------------- #
# RF Model for Prediction -- v1
# ---------------------------------------------------------------------------- #


# Set-up ----------------------------------------------------------------------#

# show Pobre distribution
ggplot(train_data, aes(x = Pobre)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Pobre",
       x = "Pobre",
       y = "Count") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.5))

# print distribution of 'Pobre'
table(train_data$Pobre)


# Drop extra features not present in 'Test'                                     #TODO: feature engineering, think of more stuff
train_data <- train_data %>%
  select(id, Clase, Dominio, P5000, P5010, P5090, P5100, P5130, P5140, Nper, Npersug, Li, Lp, Fex_c, Depto, Fex_dpto, Pobre)


colnames(train_data)
colnames(test_data)

# Model Development ------------------------------------------------------------#

# Helper function to get mode of a vector
mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Prepare data for poverty prediction
prepare_data <- function(data, train_columns = NULL, is_train = TRUE) {
  # Define leaky features
  leaky_features <- c("Npobres", "Indigente", "Nindigentes")
  
  # Remove any leaky features that exist in the dataset
  existing_leaky_features <- intersect(leaky_features, colnames(data))
  data <- data %>% select(-all_of(existing_leaky_features))
  
  # If it's the training data, ensure 'Pobre' is a factor with correct levels
  if (is_train) {
    if ("Pobre" %in% colnames(data)) {
      data$Pobre <- factor(data$Pobre, levels = c("0", "1"))
      
      # Remove rows where Pobre is NA
      data <- data %>% filter(!is.na(Pobre))
      
      # Print levels and distribution of 'Pobre' for debugging
      print("Levels of Pobre:")
      print(levels(data$Pobre))
      print("Distribution of Pobre:")
      print(table(data$Pobre, useNA = "ifany"))
    }
  }
  
  # Handle missing values (if any)
  data <- data %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(across(where(is.factor), ~ifelse(is.na(.), mode(.), as.character(.))))
  
  # If test data, ensure it has all the columns used in training
  if (!is_train & !is.null(train_columns)) {
    # Find missing columns in test data and add them with NA
    missing_columns <- setdiff(train_columns, colnames(data))
    if (length(missing_columns) > 0) {
      data[missing_columns] <- NA
    }
    
    # Ensure the test data has columns in the same order as training data
    data <- data[train_columns]
  }
  
  return(data)
}

# Split data into train and test sets
split_data <- function(data, test_size = 0.2) {
  set.seed(42)
  train_index <- createDataPartition(data$Pobre, p = 1 - test_size, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  return(list(train = train_data, test = test_data))
}

# Apply undersampling to handle class imbalance
apply_undersampling <- function(train_data) {
  # Separate the majority and minority classes
  majority_class <- train_data %>% filter(Pobre == "0")
  minority_class <- train_data %>% filter(Pobre == "1")
  
  print(paste("Majority class (non-poor) size:", nrow(majority_class)))
  print(paste("Minority class (poor) size:", nrow(minority_class)))
  
  # Randomly undersample the majority class to match the size of the minority class
  set.seed(42)
  majority_class_undersampled <- majority_class %>%
    sample_n(nrow(minority_class))
  
  # Combine the undersampled majority class with the minority class
  train_data_undersampled <- bind_rows(majority_class_undersampled, minority_class)
  
  return(train_data_undersampled)
}

# Train Random Forest model
train_model <- function(train_data) {
  rf_model <- ranger(
    factor(Pobre) ~ ., 
    data = train_data,
    num.trees = 500,
    importance = 'impurity',
    probability = TRUE
  )
  return(rf_model)
}

# Evaluate model
evaluate_model <- function(model, test_data) {
  predictions <- predict(model, data = test_data)$predictions
  
  # Ensure test_data$Pobre is a factor with correct levels
  test_data$Pobre <- factor(test_data$Pobre, levels = c("0", "1"))
  
  # Calculate AUC
  auc_score <- roc(as.numeric(as.character(test_data$Pobre)), predictions[,2])$auc
  
  # Calculate F1 score
  binary_predictions <- ifelse(predictions[,2] > 0.8, "1", "0")
  binary_predictions <- factor(binary_predictions, levels = c("0", "1"))
  
  # Ensure both predictions and actual values have the same levels
  print("Levels of binary_predictions:")
  print(levels(binary_predictions))
  print("Levels of test_data$Pobre:")
  print(levels(test_data$Pobre))
  
  # Create confusion matrix
  cm <- confusionMatrix(binary_predictions, test_data$Pobre)
  f1_score <- cm$byClass['F1']
  
  # Get feature importances
  importances <- model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  return(list(auc = auc_score, f1 = f1_score, importances = importances_df, confusion_matrix = cm$table))
}

# Cross-validation function
cross_validate <- function(data, k = 5) {
  set.seed(42)
  folds <- createFolds(data$Pobre, k = k)
  cv_results <- list()
  
  for (i in 1:k) {
    print(paste("Fold", i))
    
    # Split data into training and validation sets
    train_data <- data[-folds[[i]], ]
    val_data <- data[folds[[i]], ]
    
    # Prepare data
    train_data <- prepare_data(train_data)
    val_data <- prepare_data(val_data, train_columns = colnames(train_data), is_train = FALSE)
    
    # Apply undersampling to training data
    train_data_undersampled <- apply_undersampling(train_data)
    
    # Train model
    model <- train_model(train_data_undersampled)
    
    # Evaluate model
    eval_results <- evaluate_model(model, val_data)
    
    cv_results[[i]] <- eval_results
  }
  
  # Calculate average metrics across folds
  avg_auc <- mean(sapply(cv_results, function(x) x$auc))
  avg_f1 <- mean(sapply(cv_results, function(x) x$f1))
  
  print(paste("Average AUC:", avg_auc))
  print(paste("Average F1 Score:", avg_f1))
  
  return(list(fold_results = cv_results, avg_auc = avg_auc, avg_f1 = avg_f1))
}

# Main function
poverty_prediction <- function(data) {
  # Perform cross-validation
  cv_results <- cross_validate(data, k = 5)
  
  # Prepare data for final model
  prepared_data <- prepare_data(data)
  
  # Apply undersampling to the entire dataset for final model
  final_data_undersampled <- apply_undersampling(prepared_data)
  
  # Train final model on undersampled data
  final_model <- train_model(final_data_undersampled)
  
  # Get feature importances
  importances <- final_model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  print("Top 10 most important features:")
  print(head(importances_df, 10))
  
  return(list(cv_results = cv_results, final_model = final_model, feature_importance = importances_df))
}

# Run the analysis
result <- poverty_prediction(train_data)

# Now, use the trained final model to predict on the test dataset
rf_model <- result$final_model

### Avg. F1 := 0.80569725833551 (with c = 0.5)
### Avg. F1 := 0.893444447169399 (with c = 0.7)
### Avg. F1 := 0.89600595483144 (with c = 0.75)
### Avg. F1 := 0.89504564840167 (with c = 0.8)

# --> Let's go with c* = 0.75
# ---------------------------------------------------------------------------- #
# Kaggle Submission
# ---------------------------------------------------------------------------- #


# Format prep -----------------------------------------------------------#

# Prepare the test data by ensuring it has all the columns used during training
train_columns <- colnames(train_data)  # Columns used in training
test_data_prepared <- prepare_data(test_data, train_columns = train_columns, is_train = FALSE)

# Generate predictions on the test dataset
test_predictions <- predict(rf_model, test_data_prepared)$predictions
binary_predictions <- ifelse(test_predictions[, 2] > 0.5, 1, 0)

# Create the submission dataframe
submission <- data.frame(id = test_data$id, pobre = binary_predictions)
output_dir <- "/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/models/"
write.csv(output_dir, "RF_Trees500_v1.csv", row.names = FALSE)

submission_file <- paste0(output_dir, "RF_Trees500_v1.csv")

# Save the submission file to the specified path
write.csv(submission, submission_file, row.names = FALSE)

# API 2 Kaggle ----------------------------------------------------------------#

# Uncomment this & enter in terminal:
# kaggle competitions submit -c uniandes-bdml-2024-20-ps-2 -f /Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/models/RF_Trees500_v1.csv -m "submission 1"

# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# RF Model for Prediction -- v2 (Optimal Cut-off)
# ---------------------------------------------------------------------------- #


# Helper function to get mode of a vector
mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Prepare data for poverty prediction
prepare_data <- function(data, train_columns = NULL, is_train = TRUE) {
  # Define leaky features
  leaky_features <- c("Npobres", "Indigente", "Nindigentes")
  
  # Remove any leaky features that exist in the dataset
  existing_leaky_features <- intersect(leaky_features, colnames(data))
  data <- data %>% select(-all_of(existing_leaky_features))
  
  # If it's the training data, ensure 'Pobre' is a factor with correct levels
  if (is_train) {
    if ("Pobre" %in% colnames(data)) {
      data$Pobre <- factor(data$Pobre, levels = c("0", "1"))
      
      # Remove rows where Pobre is NA
      data <- data %>% filter(!is.na(Pobre))
      
      # Print levels and distribution of 'Pobre' for debugging
      print("Levels of Pobre:")
      print(levels(data$Pobre))
      print("Distribution of Pobre:")
      print(table(data$Pobre, useNA = "ifany"))
    }
  }
  
  # Handle missing values (if any)
  data <- data %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(across(where(is.factor), ~ifelse(is.na(.), mode(.), as.character(.))))
  
  # If test data, ensure it has all the columns used in training
  if (!is_train & !is.null(train_columns)) {
    # Find missing columns in test data and add them with NA
    missing_columns <- setdiff(train_columns, colnames(data))
    if (length(missing_columns) > 0) {
      data[missing_columns] <- NA
    }
    
    # Ensure the test data has columns in the same order as training data
    data <- data[train_columns]
  }
  
  return(data)
}

# Split data into train and test sets
split_data <- function(data, test_size = 0.2) {
  set.seed(42)
  train_index <- createDataPartition(data$Pobre, p = 1 - test_size, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  return(list(train = train_data, test = test_data))
}

# Apply undersampling to handle class imbalance
apply_undersampling <- function(train_data) {
  # Separate the majority and minority classes
  majority_class <- train_data %>% filter(Pobre == "0")
  minority_class <- train_data %>% filter(Pobre == "1")
  
  print(paste("Majority class (non-poor) size:", nrow(majority_class)))
  print(paste("Minority class (poor) size:", nrow(minority_class)))
  
  # Randomly undersample the majority class to match the size of the minority class
  set.seed(42)
  majority_class_undersampled <- majority_class %>%
    sample_n(nrow(minority_class))
  
  # Combine the undersampled majority class with the minority class
  train_data_undersampled <- bind_rows(majority_class_undersampled, minority_class)
  
  return(train_data_undersampled)
}

# Train Random Forest model
train_model <- function(train_data) {
  rf_model <- ranger(
    factor(Pobre) ~ ., 
    data = train_data,
    num.trees = 500,
    importance = 'impurity',
    probability = TRUE
  )
  return(rf_model)
}

# Evaluate model
evaluate_model <- function(model, test_data) {
  predictions <- predict(model, data = test_data)$predictions
  
  # Ensure test_data$Pobre is a factor with correct levels
  test_data$Pobre <- factor(test_data$Pobre, levels = c("0", "1"))
  
  # Calculate AUC
  auc_score <- roc(as.numeric(as.character(test_data$Pobre)), predictions[,2])$auc
  
  # Calculate F1 score
  binary_predictions <- ifelse(predictions[,2] > 0.75, "1", "0")
  binary_predictions <- factor(binary_predictions, levels = c("0", "1"))
  
  # Ensure both predictions and actual values have the same levels
  print("Levels of binary_predictions:")
  print(levels(binary_predictions))
  print("Levels of test_data$Pobre:")
  print(levels(test_data$Pobre))
  
  # Create confusion matrix
  cm <- confusionMatrix(binary_predictions, test_data$Pobre)
  f1_score <- cm$byClass['F1']
  
  # Get feature importances
  importances <- model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  return(list(auc = auc_score, f1 = f1_score, importances = importances_df, confusion_matrix = cm$table))
}

# Cross-validation function
cross_validate <- function(data, k = 5) {
  set.seed(42)
  folds <- createFolds(data$Pobre, k = k)
  cv_results <- list()
  
  for (i in 1:k) {
    print(paste("Fold", i))
    
    # Split data into training and validation sets
    train_data <- data[-folds[[i]], ]
    val_data <- data[folds[[i]], ]
    
    # Prepare data
    train_data <- prepare_data(train_data)
    val_data <- prepare_data(val_data, train_columns = colnames(train_data), is_train = FALSE)
    
    # Apply undersampling to training data
    train_data_undersampled <- apply_undersampling(train_data)
    
    # Train model
    model <- train_model(train_data_undersampled)
    
    # Evaluate model
    eval_results <- evaluate_model(model, val_data)
    
    cv_results[[i]] <- eval_results
  }
  
  # Calculate average metrics across folds
  avg_auc <- mean(sapply(cv_results, function(x) x$auc))
  avg_f1 <- mean(sapply(cv_results, function(x) x$f1))
  
  print(paste("Average AUC:", avg_auc))
  print(paste("Average F1 Score:", avg_f1))
  
  return(list(fold_results = cv_results, avg_auc = avg_auc, avg_f1 = avg_f1))
}

# Main function
poverty_prediction <- function(data) {
  # Perform cross-validation
  cv_results <- cross_validate(data, k = 5)
  
  # Prepare data for final model
  prepared_data <- prepare_data(data)
  
  # Apply undersampling to the entire dataset for final model
  final_data_undersampled <- apply_undersampling(prepared_data)
  
  # Train final model on undersampled data
  final_model <- train_model(final_data_undersampled)
  
  # Get feature importances
  importances <- final_model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  print("Top 10 most important features:")
  print(head(importances_df, 10))
  
  return(list(cv_results = cv_results, final_model = final_model, feature_importance = importances_df))
}

# Run the analysis
result <- poverty_prediction(train_data)

# Now, use the trained final model to predict on the test dataset
rf_model <- result$final_model


# ---------------------------------------------------------------------------- #
# Kaggle Submission
# ---------------------------------------------------------------------------- #


# Format prep -----------------------------------------------------------#

# Prepare the test data by ensuring it has all the columns used during training
train_columns <- colnames(train_data)  # Columns used in training
test_data_prepared <- prepare_data(test_data, train_columns = train_columns, is_train = FALSE)

# Generate predictions on the test dataset
test_predictions <- predict(rf_model, test_data_prepared)$predictions
binary_predictions <- ifelse(test_predictions[, 2] > 0.75, 1, 0)

# Create the submission dataframe
submission <- data.frame(id = test_data$id, pobre = binary_predictions)
output_dir <- "/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/models/"
write.csv(output_dir, "RF_Trees500_v2.csv", row.names = FALSE)

submission_file <- paste0(output_dir, "RF_Trees500_v2.csv")

# Save the submission file to the specified path
write.csv(submission, submission_file, row.names = FALSE)

# API 2 Kaggle ----------------------------------------------------------------#

# Uncomment this & enter in terminal:
# kaggle competitions submit -c uniandes-bdml-2024-20-ps-2 -f /Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/models/RF_Trees500_v2.csv -m "submission 2 - correct optimal cutoff"

### score TEST: 0.289

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Check train & test set distributions
# ---------------------------------------------------------------------------- #


train <- train_data
test <- test_data


# Function to impute missing values using mean (numeric) or mode (categorical)
impute_na <- function(x) {
  if (is.numeric(x)) {
    return(ifelse(is.na(x), mean(x, na.rm = TRUE), x))  # Impute with mean for numeric
  } else {
    mode_value <- names(sort(table(x), decreasing = TRUE))[1]  # Get mode for categorical
    return(ifelse(is.na(x), mode_value, x))  # Impute with mode for categorical
  }
}

# Ensure that only numeric columns are used
numeric_cols <- colnames(train)[sapply(train, is.numeric)]  # Get numeric columns only

# Impute missing values for both train and test datasets
train_imputed <- train
test_imputed <- test

# Apply imputation to both datasets
for (col in numeric_cols) {
  if (col != "Pobre") {
    train_imputed[[col]] <- impute_na(train[[col]])  # Impute NA in train set
    test_imputed[[col]] <- impute_na(test[[col]])    # Impute NA in test set
  }
}

# Perform KS test on imputed numeric columns
for (col in numeric_cols) {
  if (col != "Pobre") {
    # Extract the imputed numeric columns from both train and test sets
    train_col <- train_imputed[[col]]
    test_col <- test_data[[col]]
    
    # Perform the KS test
    ks_result <- ks.test(train_col, test_col)
    print(paste("Feature:", col, 
                "KS Statistic:", ks_result$statistic, 
                "p-value:", ks_result$p.value))
  }
}


### --> Many features are statistically different between train and test sets !

# ---------------------------------------------------------------------------- #
# Stratified Sampling
# ---------------------------------------------------------------------------- #


library(MatchIt)
library(data.table)
library(dplyr)

train <- train_data
test <- test_data

# 1. Label train and test sets and combine them
train$source <- 0  # Label train as 0
test$source <- 1   # Label test as 1

# Remove 'Pobre' from the train set before combining
train_subset <- train[, !colnames(train) %in% "Pobre"]

# Combine train and test sets
combined_data <- rbind(train_subset, test)

# Convert to data.table for efficiency
combined_data <- as.data.table(combined_data)

# 2. Impute missing values for all columns (numeric: mean, categorical: mode)
impute_na <- function(x) {
  if (is.numeric(x)) {
    return(ifelse(is.na(x), mean(x, na.rm = TRUE), x))  # Impute with mean for numeric
  } else {
    mode_value <- names(sort(table(x), decreasing = TRUE))[1]  # Mode for categorical
    return(ifelse(is.na(x), mode_value, x))  # Impute with mode for categorical
  }
}

# Apply the imputation to all columns in combined_data
combined_data <- combined_data[, lapply(.SD, impute_na)]


# 3. Choose the top 5 most statistically different features from the KS test
key_features <- c("Clase", "P5000", "P5090", "P5100", "P5130")

# Ensure key_features exist in the combined data after imputation
if (!all(key_features %in% colnames(combined_data))) {
  stop("One or more key features are missing from the dataset.")
}

# 4. Calculate proportions for the test set
test_proportions <- test %>%
  group_by(across(all_of(key_features))) %>%
  summarise(count = n()) %>%
  mutate(prop = count / sum(count))

# 5. Perform stratified sampling on the train set based on the test set proportions
new_train_set <- train_subset %>%
  group_by(Clase, P5000, P5090, P5100, P5130) %>%
  sample_frac(min(1, test_proportions$prop[match(paste(Clase, P5000, P5090, P5100, P5130), 
                                                 paste(test_proportions$Clase, test_proportions$P5000, 
                                                       test_proportions$P5090, test_proportions$P5100, 
                                                       test_proportions$P5130))], na.rm = TRUE))

# KS test ---------------------------------------------------------------------#

numeric_cols <- colnames(train)[sapply(train, is.numeric)]  # Get numeric columns only

# Perform KS test on imputed numeric columns
for (col in numeric_cols) {
  if (col != "Pobre") {
    # Extract the imputed numeric columns from both train and test sets
    train_col <- new_train_set[[col]]
    test_col <- test_data[[col]]
    
    # Perform the KS test
    ks_result <- ks.test(train_col, test_col)
    print(paste("Feature:", col, 
                "KS Statistic:", ks_result$statistic, 
                "p-value:", ks_result$p.value))
  }
}

colnames(new_train_set)

# ---------------------------------------------------------------------------- #
# Model Training on Enhanced Training Set
# ---------------------------------------------------------------------------- #

# 1. Add a unique identifier to both datasets
train_with_id <- train %>%
  mutate(row_id = row_number())

new_train_set_with_id <- new_train_set %>%
  mutate(row_id = row_number())

# 2. Perform the join
new_train_set_with_pobre <- new_train_set_with_id %>%
  left_join(train_with_id %>% select(row_id, Pobre), by = "row_id")

# 3. Remove the temporary row_id column
new_train_set_with_pobre <- new_train_set_with_pobre %>%
  select(-row_id)

# 4. Verify the merge
print(head(new_train_set_with_pobre))
print(sum(is.na(new_train_set_with_pobre$Pobre)))

train_data <- new_train_set_with_pobre


# Model Deployment ------------------------------------------------------------#


library(dplyr)
library(caret)
library(ranger)
library(pROC)

# Helper function to get mode of a vector
mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Prepare data for poverty prediction
prepare_data <- function(data, train_columns = NULL, is_train = TRUE) {
  # Define leaky features
  leaky_features <- c("Npobres", "Indigente", "Nindigentes")
  
  # Remove any leaky features that exist in the dataset
  existing_leaky_features <- intersect(leaky_features, colnames(data))
  data <- data %>% select(-all_of(existing_leaky_features))
  
  # If it's the training data, ensure 'Pobre' is a factor with correct levels
  if (is_train) {
    if ("Pobre" %in% colnames(data)) {
      data$Pobre <- factor(data$Pobre, levels = c("0", "1"))
      
      # Remove rows where Pobre is NA
      data <- data %>% filter(!is.na(Pobre))
      
      # Print levels and distribution of 'Pobre' for debugging
      print("Levels of Pobre:")
      print(levels(data$Pobre))
      print("Distribution of Pobre:")
      print(table(data$Pobre, useNA = "ifany"))
    }
  }
  
  # Handle missing values
  data <- data %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(across(where(is.factor), ~as.factor(ifelse(is.na(.), mode(as.character(.)), as.character(.)))))
  
  # Special handling for P5100, P5130, P5140
  problematic_cols <- c("P5100", "P5130", "P5140")
  for (col in problematic_cols) {
    if (col %in% names(data)) {
      if (is.numeric(data[[col]])) {
        data[[col]] <- ifelse(is.na(data[[col]]), median(data[[col]], na.rm = TRUE), data[[col]])
      } else {
        data[[col]] <- as.factor(ifelse(is.na(data[[col]]), mode(as.character(data[[col]])), as.character(data[[col]])))
      }
    }
  }
  
  # Check for any remaining NA values
  na_cols <- colnames(data)[colSums(is.na(data)) > 0]
  if (length(na_cols) > 0) {
    warning(paste("Columns with remaining NA values:", paste(na_cols, collapse = ", ")))
    # Remove columns with all NA values
    data <- data %>% select_if(~!all(is.na(.)))
  }
  
  # If test data, ensure it has all the columns used in training
  if (!is_train & !is.null(train_columns)) {
    # Find missing columns in test data and add them with NA
    missing_columns <- setdiff(train_columns, colnames(data))
    if (length(missing_columns) > 0) {
      data[missing_columns] <- NA
    }
    
    # Ensure the test data has columns in the same order as training data
    data <- data[train_columns]
  }
  
  return(data)
}

# Split data into train and test sets
split_data <- function(data, test_size = 0.2) {
  set.seed(42)
  train_index <- createDataPartition(data$Pobre, p = 1 - test_size, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  return(list(train = train_data, test = test_data))
}

# Apply undersampling to handle class imbalance
apply_undersampling <- function(train_data) {
  # Separate the majority and minority classes
  majority_class <- train_data %>% filter(Pobre == "0")
  minority_class <- train_data %>% filter(Pobre == "1")
  
  print(paste("Majority class (non-poor) size:", nrow(majority_class)))
  print(paste("Minority class (poor) size:", nrow(minority_class)))
  
  # Randomly undersample the majority class to match the size of the minority class
  set.seed(42)
  majority_class_undersampled <- majority_class %>%
    slice_sample(n = nrow(minority_class))
  
  # Combine the undersampled majority class with the minority class
  train_data_undersampled <- bind_rows(majority_class_undersampled, minority_class)
  
  print(paste("Undersampled dataset size:", nrow(train_data_undersampled)))
  
  return(train_data_undersampled)
}

# Train Random Forest model
train_model <- function(train_data) {
  rf_model <- ranger(
    factor(Pobre) ~ ., 
    data = train_data,
    num.trees = 500,
    importance = 'impurity',
    probability = TRUE
  )
  return(rf_model)
}

# Evaluate model
evaluate_model <- function(model, test_data) {
  predictions <- predict(model, data = test_data)$predictions
  
  # Ensure test_data$Pobre is a factor with correct levels
  test_data$Pobre <- factor(test_data$Pobre, levels = c("0", "1"))
  
  # Calculate AUC
  auc_score <- roc(as.numeric(as.character(test_data$Pobre)), predictions[,2])$auc
  
  # Calculate F1 score
  binary_predictions <- ifelse(predictions[,2] > 0.5, "1", "0")
  binary_predictions <- factor(binary_predictions, levels = c("0", "1"))
  
  # Create confusion matrix
  cm <- confusionMatrix(binary_predictions, test_data$Pobre)
  f1_score <- cm$byClass['F1']
  
  # Get feature importances
  importances <- model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  return(list(auc = auc_score, f1 = f1_score, importances = importances_df, confusion_matrix = cm$table))
}

# Cross-validation function
cross_validate <- function(data, k = 5) {
  set.seed(42)
  folds <- createFolds(data$Pobre, k = k)
  cv_results <- list()
  
  for (i in 1:k) {
    print(paste("Fold", i))
    
    # Split data into training and validation sets
    train_data <- data[-folds[[i]], ]
    val_data <- data[folds[[i]], ]
    
    # Prepare data
    train_data <- prepare_data(train_data)
    val_data <- prepare_data(val_data, train_columns = colnames(train_data), is_train = FALSE)
    
    # Apply undersampling to training data
    train_data_undersampled <- apply_undersampling(train_data)
    
    # Train model
    model <- train_model(train_data_undersampled)
    
    # Evaluate model
    eval_results <- evaluate_model(model, val_data)
    
    cv_results[[i]] <- eval_results
  }
  
  # Calculate average metrics across folds
  avg_auc <- mean(sapply(cv_results, function(x) x$auc))
  avg_f1 <- mean(sapply(cv_results, function(x) x$f1))
  
  print(paste("Average AUC:", avg_auc))
  print(paste("Average F1 Score:", avg_f1))
  
  return(list(fold_results = cv_results, avg_auc = avg_auc, avg_f1 = avg_f1))
}

# Main function
poverty_prediction <- function(data) {
  # Perform cross-validation
  cv_results <- cross_validate(data, k = 5)
  
  # Prepare data for final model
  prepared_data <- prepare_data(data)
  
  # Apply undersampling to the entire dataset for final model
  final_data_undersampled <- apply_undersampling(prepared_data)
  
  # Train final model on undersampled data
  final_model <- train_model(final_data_undersampled)
  
  # Get feature importances
  importances <- final_model$variable.importance
  importances_df <- data.frame(
    feature = names(importances),
    importance = as.numeric(importances)
  ) %>%
    arrange(desc(importance))
  
  print("Top 10 most important features:")
  print(head(importances_df, 10))
  
  return(list(cv_results = cv_results, final_model = final_model, feature_importance = importances_df))
}

# Run the analysis
result <- poverty_prediction(train_data)

# Now, use the trained final model to predict on the test dataset
rf_model <- result$final_model



# ---------------------------------------------------------------------------- #
# Kaggle submission 3
# ---------------------------------------------------------------------------- #

# Format prep -----------------------------------------------------------------#


# Prepare the test data by ensuring it has all the columns used during training
train_columns <- colnames(train_data)  # Columns used in training
test_data_prepared <- prepare_data(test, train_columns = train_columns, is_train = FALSE)

# Generate predictions on the test dataset
test_predictions <- predict(rf_model, test_data_prepared)$predictions
binary_predictions <- ifelse(test_predictions[, "1"] <= 0.5, 1, 0)  # Note the change from > to <=

# Create the submission dataframe
submission <- data.frame(id = test_data$id, pobre = binary_predictions)
output_dir <- "/Users/edmundoarias/Documents/Uniandes/2024-2/BigDataML-Group2024/Problem_Set_2/models/"
write.csv(output_dir, "RF_Trees500_v3.csv", row.names = FALSE)

submission_file <- paste0(output_dir, "RF_Trees500_v3.csv")

# Save the submission file to the specified path
write.csv(submission, submission_file, row.names = FALSE)

# how many rows in submission
nrow(submission)




