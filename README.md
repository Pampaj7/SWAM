# Machine Learning Efficiency Project

This project evaluates the efficiency of machine learning algorithms implemented in various programming languages by
tracking energy consumption, CO2 emissions, and performance metrics. The goal is to understand the trade-offs between
speed, accuracy, and environmental impact across languages and algorithms.

---

## Table of Contents

- [Introduction](#introduction)
- [Languages and Algorithms](#languages-and-algorithms)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

---

## Introduction

This project investigates the trade-offs between machine learning performance and sustainability. By comparing the same
algorithms across multiple programming languages (Python, MATLAB, R, Java, C++, etc.), we aim to highlight which
languages and implementations are the most energy-efficient while maintaining model performance.

We track:

- **Execution Time**
- **Energy Consumption**
- **CO2 Emissions**

The emissions are monitored using the `CodeCarbon` library in Python and integrated into other languages through system
calls.

---

## Languages and Algorithms
 
### Supported Languages:

- Python
- MATLAB
- R
- Java
- C++
- (Future) Rust, Go, Julia, Scala, Swift, Fortran

### Algorithms Implemented:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVC)**
- **Naive Bayes**

All algorithms are implemented with a minimal set of hyperparameters to ensure consistency across different languages.

---

## Datasets

The following datasets are used in the experiments:

- **Breast Cancer**
- **Wine Quality**
- **Iris Dataset**

Each dataset has been preprocessed and saved as CSV files located in the `datasets/` directory.

---


---

## Installation

### Prerequisites:
1. [Conda](https://docs.conda.io/en/latest/miniconda.html) (for managing environments)
2. [MATLAB](https://www.mathworks.com/products/matlab.html) (with **Statistics and Machine Learning Toolbox**)
3. [Java SDK](https://www.oracle.com/java/technologies/javase-downloads.html)
4. [R](https://cran.r-project.org/)
   1. [pandoc](https://pandoc.org) (brew install pandoc, to handle conversion from .Rmd to html)
5. [g++](https://gcc.gnu.org/) for C++

### Steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ml-efficiency-project.git
   cd ml-efficiency-project


## Usage

1. Navigate to the src directory of the project using your terminal:
   1. `cd ../SWAM/src`
2. Run the execution script: 
   1. `./allRunner.sh`
3. Ensure that all prerequisites are met. If so, the program will begin executing.
4. Upon completion, a file named `raw_merged_emissions.csv` will be generated in the `src` folder, containing:
   1. 30 rows for each unique combination of dataset, algorithm, programming language, and phase (training or testing).
   

## Result

After the program completes its execution, you will find all the generated plots in the graphics folder.
Additionally, the processedDataset folder will contain all the datasets used to create those plots.
