#!/bin/bash

echo "Activating the first environment (sw) for all languages except C++..."
conda activate sw

echo "Installing required Python packages from requirementsAll.txt..."
pip install -r requirementsAll.txt

echo "Running MATLAB and Python scripts using allRunner.py..."
python allRunner.py
# add matlab env

echo "Navigating to the R directory and setting up the R environment..."
cd R

echo "Checking if Rmarkdown is installed..."
Rscript -e "if (!require('rmarkdown')) install.packages('rmarkdown', repos='http://cran.rstudio.com/')"

echo "Rendering requirements.Rmd to install required R packages..."
Rscript -e "rmarkdown::render('requirements.Rmd')"

echo "Running rRunner.py..."
python3 rRunner.py

cd ..

echo "Navigating to the Java directory, building with Maven, and running jRunner.py..."
cd java
mvn clean install
python jRunner.py
cd ..

echo "Switching to the C++ environment to avoid conflicts..."
cd cpp
conda activate cpp

echo "Installing required Python packages from requirementsCpp.txt..."
pip install -r requirementsCpp.txt

echo "Running C++ related scripts with cRunner.py..."
python cRunner.py
# sorry this will most likely not run 

cd ../

echo "Creating Plots!"

conda activate sw
python handleCsv.py
python HandlePlot.py
echo "All tasks completed!"

