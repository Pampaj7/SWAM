# cloud
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(e1071)) install.packages("e1071", dependencies = TRUE)
library(caret) # lib for ml
library(e1071) 

csv_file_path <- "../../datasets/breastcancer/breastcancer.csv"
data <- read.csv(csv_file_path)

data$diagnosis <- ifelse(data$diagnosis == "M", 1, 0)

set.seed(42)
train_index <- createDataPartition(data$diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

model <- train(diagnosis ~ ., data = train_data, method = "glm", family = "binomial")

#train: Funzione del pacchetto caret che addestra un modello.
#diagnosis ~ .: Formula che specifica che diagnosis è la variabile dipendente e tutte le altre colonne sono variabili indipendenti.
#method = "glm": Specifica che il metodo di addestramento è la regressione logistica (Generalized Linear Model).
#family = "binomial": Specifica che si tratta di una regressione logistica binaria.

predictions <- predict(model, test_data)

# basically this conversion as factor is needet to use the data (?)
# need to convert both to be similar
# XXX TODO non concordo con questo codice, i fattori trancianoi valori
predictions <- as.factor(predictions)
test_data$diagnosis <- as.factor(test_data$diagnosis)

# Livelli coerenti per entrambi i fattori
levels(predictions) <- levels(test_data$diagnosis)

# Valutazione del modello
conf_matrix <- confusionMatrix(predictions, test_data$diagnosis)

# Restituire solo l'accuratezza
accuracy <- conf_matrix$overall["Accuracy"]
cat(accuracy)