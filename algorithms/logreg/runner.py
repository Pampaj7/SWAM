import subprocess
import matlab.engine
from python import lr_bc as lr


def run_matlab_script(path):
    # chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run(path, nargout=0)
    eng.quit()


def run_r_script(path):
    result = subprocess.run(["Rscript", path], capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


print("-----------------------")
print("Running R script...")
run_r_script("r/lr_bc.r")
print("-----------------------")
print("-----------------------")

print("Running MATLAB script...")
run_matlab_script("matlab/lr_bc.m")
run_matlab_script("../decisionTree/matlab/dt_bc.m")
print("-----------------------")
print("-----------------------")

print("Running Python script...")
lr.logreg()
print("-----------------------")
