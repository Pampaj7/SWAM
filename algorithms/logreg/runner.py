import subprocess
import matlab.engine
from python import lr_bc as lr


def run_matlab_script():
    # chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run("matlab/lr_bc.m", nargout=0)
    eng.quit()


def run_r_script():
    result = subprocess.run(["Rscript", "r/lr_bc.r"], capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


print("-----------------------")
print("Running R script...")
run_r_script()
print("-----------------------")
print("-----------------------")

print("Running MATLAB script...")
run_matlab_script()
print("-----------------------")
print("-----------------------")

print("Running Python script...")
lr.logreg()
print("-----------------------")
