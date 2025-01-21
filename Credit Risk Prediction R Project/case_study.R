# ---- Install and Load Necessary Packages ----
install.packages(c("tidyverse", "caret", "DataExplorer", "corrplot", "randomForest", "DMwR", "pROC"))
library(tidyverse)     
library(caret)         
library(DataExplorer)   
library(corrplot)      
library(randomForest)   
library(DMwR)           
library(pROC)          

# ---- Load the Dataset ----
data <- read.csv("C:/Users/User/Documents/CaseStudy/credit_data.csv")


# View dataset structure and summary
glimpse(data)
summary(data)

# Check for missing values
plot_missing(data)

# Visualize distribution of all numeric variables
numeric_vars <- data %>% select(where(is.numeric)) %>% drop_na()

numeric_vars %>%
  gather() %>%
  ggplot(aes(value)) + 
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) + 
  facet_wrap(~key, scales = "free") + 
  theme_minimal() +
  ggtitle("Distributions of Numeric Variables")

# Visualize correlation among numeric variables
numeric_vars <- data %>% select(where(is.numeric))
corrplot(cor(numeric_vars, use = "complete.obs"), method = "circle", title = "Correlation Matrix")



# ---- Handle Missing Values ----
data$occupation <- ifelse(data$occupation %in% c("", "NA", "unknown"), NA, data$occupation)
cat("Missing values in 'occupation':", sum(is.na(data$occupation)), "\n")


if (any(is.na(data$occupation))) {
  most_frequent_occupation <- names(sort(table(data$occupation), decreasing = TRUE))[1]
  data$occupation[is.na(data$occupation)] <- most_frequent_occupation
}


data$age[is.na(data$age)] <- median(data$age, na.rm = TRUE)


data <- data %>%
  mutate(across(where(is.character), ~na_if(.x, ""))) %>%
  mutate(across(where(is.character), as.factor))


for (col in names(data)) {
  if (is.factor(data[[col]]) && any(is.na(data[[col]]))) {
    most_frequent_value <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
    data[[col]][is.na(data[[col]])] <- most_frequent_value
  }
}


cat("Missing values per column after handling:\n")
print(colSums(is.na(data)))

# ---- Convert and Encode Categorical Variables ----
data_encoded <- as.data.frame(model.matrix(~ . - 1, data = data))


data_encoded$credit_risk <- data$credit_risk


data_encoded$credit_risk <- ifelse(data_encoded$credit_risk == 1, "good", "bad")
data_encoded$credit_risk <- as.factor(data_encoded$credit_risk)

# ---- Feature Scaling ----
numeric_cols <- sapply(data_encoded, is.numeric)
data_encoded[numeric_cols] <- scale(data_encoded[numeric_cols])

# ---- Split the Dataset ----
set.seed(123)
trainIndex <- createDataPartition(data_encoded$credit_risk, p = 0.8, list = FALSE)
train_data <- data_encoded[trainIndex, ]
test_data <- data_encoded[-trainIndex, ]


train_smote <- SMOTE(credit_risk ~ ., data = train_data, perc.over = 100, perc.under = 200)


cat("Class distribution after SMOTE:\n")
print(table(train_smote$credit_risk))


train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)

rf_model <- train(
  credit_risk ~ ., 
  data = train_smote, 
  method = "rf", 
  trControl = train_control,
  tuneLength = 5 
)

# ---- Evaluate Model ----

rf_predictions <- predict(rf_model, newdata = test_data)
rf_probabilities <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

# Confusion Matrix
confusionMatrix(rf_predictions, as.factor(test_data$credit_risk))

# Feature Importance
importance_rf <- varImp(rf_model, scale = FALSE)
plot(importance_rf, main = "Feature Importance (Random Forest Model)")

# ---- ROC Curve ----

binary_target <- ifelse(test_data$credit_risk == "good", 1, 0)

roc_curve <- roc(response = binary_target, predictor = rf_probabilities, levels = c(0, 1), direction = "<")

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve (Random Forest Model)", col = "blue")

# Calculate and display the AUC
auc_value <- auc(roc_curve)
cat("AUC: ", auc_value, "\n")

# ---- Save Results ----
write.csv(rf_predictions, file = "credit_risk_predictions_rf.csv", row.names = FALSE)

# Start timer
start_time <- Sys.time()

documentation <- c(
  "1. Missing values in 'age' replaced with median.",
  "2. Non-standard missing values in 'occupation' replaced with NA and imputed with mode.",
  "3. Class imbalance addressed using SMOTE to oversample the minority class and undersample the majority.",
  "4. Random Forest model trained using 10-fold cross-validation for robust evaluation.",
  paste("5. Achieved AUC score of", round(auc_value, 3), "on the test set, indicating good predictive performance."),
  paste("6. Execution completed at", Sys.time())
)


cat(paste(documentation, collapse = "\n"), "\n")


writeLines(documentation, "model_workflow_documentation.txt")


end_time <- Sys.time()
cat("\nExecution Time: ", difftime(end_time, start_time, units = "mins"), "minutes.\n")

summary_stats <- c(
  paste("Execution Time: ", difftime(end_time, start_time, units = "mins"), "minutes."),
  paste("AUC Score: ", round(auc_value, 3))
)
writeLines(summary_stats, "model_execution_summary.txt")

