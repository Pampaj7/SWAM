# Energy, Emissions and Performance: Cross-Language and Cross-Algorithm Analysis in Machine Learning

This repository is a companion page for the following research, submitted for revision at the 9th International Workshop on Green and Sustainable Software (GREENS’25)

> Authors Blinded for Review. 2025. Energy, Emissions and Performance: Cross-Language and Cross-Algorithm Analysis in Machine Learning.Submitted for revision at the 9th International Workshop on Green and Sustainable Software (GREENS’25)

---

## Table of Contents

- [Introduction](#introduction)
- [Languages and Algorithms](#languages-and-algorithms)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Paper](#Paper)

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
<!-- - (Future) Rust, Go, Julia, Scala, Swift, Fortran -->

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

## Quick Start

### Prerequisites:

1. [Conda](https://docs.conda.io/en/latest/miniconda.html) (for managing environments)
2. [MATLAB](https://www.mathworks.com/products/matlab.html) (with **Statistics and Machine Learning Toolbox**)
3. [Java SDK](https://www.oracle.com/java/technologies/javase-downloads.html)
4. [R](https://cran.r-project.org/)
   1. [pandoc](https://pandoc.org) (brew install pandoc, to handle conversion from .Rmd to html)
5. [g++](https://gcc.gnu.org/) for C++ or(gcc or whatever you need to run cpp on your system)

### Steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pampaj7/SWAM.git
   cd SWAM
   ```

## Usage

0. You need to change the python paths on all the languages, since we're calling python from within each language you need to link paths of your python installation
1. Ensure you have created conda environments named like the one in the allRunner.sh
2. Navigate to the src directory of the project using your terminal:
   1. `cd ../SWAM/src`
3. Run the execution script:
   1. `./allRunner.sh`
4. Ensure that all prerequisites are met. If so, the program will begin executing. If not cd into specific language folder and make sure you have the dependencies installed, some languages like cpp require system level libraries
5. Upon completion, a file named `raw_merged_emissions.csv` will be generated in the `data` folder, containing:
   1. 30 rows for each unique combination of dataset, algorithm, programming language, and phase (training or testing).

## Result

After the program completes its execution, you will find all the generated plots in the graphics folder.
Additionally, the data folder will contain all the datasets used to create those plots.

## Repository Structure

This is the root directory of the repository. The directory is structured as follows:

    .
    ├── README.md
    ├── data
    │   └── mean_emissions.csv          Final experimental results data
    ├── plots
    │   ├── qqplot-energy.pdf            QQ-plot of energy measurements distribution
    │   ├── qqplot-f1.pdf                QQ-plot of F1-scores distribution
    │   ├── rq1.pdf                      RQ1 plot
    │   ├── rq2.pdf                      RQ2 plot
    │   └── rq3.pdf                      RQ3 plot
    ├── requirements.txt
    └── src
        ├── cpp                         Folder containing cpp files
        ├── java                        Folder containing java files
        ├── matlab                      Folder containing matlab files
        ├── processedDataset            Folder containing processedDataset files
        ├── python                      Folder containing python files
        └── Utils                       Folder containing a .py used to preprocess the datasets
