
echo "Switching to the C++ environment to avoid conflicts..."
cd cpp
source ~/anaconda3/bin/activate cpp

echo "Installing required Python packages from requirementsCpp.txt..."
cd ..
pip install -r requirementsCpp.txt
cd cpp

echo "Running C++ related scripts with cRunner.py..."
python cRunner.py

cd ..

echo "Creating raw_merged_emission.csv..."
python -c "from utils import retrieve_data; retrieve_data(True)"

echo "Creating Plots!"
source ~/anaconda3/bin/activate sw
python handleCsv.py
python HandlePlot.py
echo "All tasks completed!"
