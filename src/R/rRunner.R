# Load necessary libraries
library(ggplot2)

# Assume the dataset is stored in 'breastcancer.csv' in the current working directory
data <- read.csv("/Users/pippodima/PycharmProjects/SWAM/datasets/breastcancer/breastcancer.csv")

# Inspect the data to confirm it loaded correctly
print(head(data))   # Print the first few rows
print(names(data))  # Print the column names

# Create a simple plot - for example, a scatter plot of 'radius_mean' vs 'texture_mean' colored by 'diagnosis'
ggplot(data, aes(x = radius_mean, y = texture_mean, color = diagnosis)) +
  geom_point() +
  labs(title = "Radius Mean vs. Texture Mean",
       x = "Radius Mean",
       y = "Texture Mean",
       color = "Diagnosis") +
  theme_minimal()
