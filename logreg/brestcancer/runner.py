import subprocess
import matlab.engine
import lr_bc as lr


def run_matlab_script():
    # chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run('lr_bc.m', nargout=0)
    eng.quit()


def run_r_script():
    result = subprocess.run(["Rscript", "lr_bc.r"],
                            capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


print("Running R script...")
run_r_script()
print("Running MATLAB script...")
run_matlab_script()
print("Running Python script...")
lr.logreg()
