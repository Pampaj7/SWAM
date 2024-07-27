# Set CRAN repository
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install and load necessary packages
if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(e1071)) install.packages("e1071", dependencies = TRUE)
library(caret)
library(e1071)

# Load the dataset
csv_file_path <- "../../datasets/breastcancer/breastcancer.csv"
data <- read.csv(csv_file_path)

# Convert diagnosis column to binary numeric format
data$diagnosis <- ifelse(data$diagnosis == "M", 1, 0)

# Set seed for reproducibility
set.seed(42)

# Split data into training and testing sets
train_index <- createDataPartition(data$diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the logistic regression model using glm directly
model <- glm(diagnosis ~ ., data = train_data, family = binomial)

# Make predictions on the test set
predictions_prob <- predict(model, test_data, type = "response")
predictions <- ifelse(predictions_prob > 0.5, 1, 0)  # Convert probabilities to binary outcomes

# Evaluate the model
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(test_data$diagnosis))

# Extract and print the accuracy
accuracy <- conf_matrix$overall["Accuracy"]
cat(accuracy, "\n")
