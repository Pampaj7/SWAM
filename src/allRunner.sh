#this script runs all the project

# this is the first enviroment used to run all the languages execpt c++
conda activate sw
pip install -r requirementsAll.txt
python allRunner.py
cd R
python3 rRunner.py
cd ..
cd java
python jRunner.py
cd ..

#need th secondo enviroment to avoid conflicts with the first one
# the requirements.txt file are provided
cd cpp
conda cativate cpp
pip install -r requirementsCpp.txt
python cRunner.py