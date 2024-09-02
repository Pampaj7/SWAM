import subprocess
from R.rRunner import mainR
from R.rRunner import maintest
# import matlab.engine
from codecarbon import EmissionsTracker


def run_r_script():
    # result = subprocess.run(
    #     ["python", "R/rRunner.py"], capture_output=True, text=True
    # )
    # print("Python script output:")
    # print(result.stdout)
    # print(result.stderr)
    mainR()


def run_python():
    result = subprocess.run(
        ["python", "python/pythonRunner.py"], capture_output=True, text=True
    )
    print("Python script output:")
    print(result.stdout)
    print(result.stderr)


def run_matlab_script():
    result = subprocess.run(
        ["python", "matlab/matlabRunner.py"], capture_output=True, text=True
    )
    print("Python script output:")
    print(result.stdout)
    print(result.stderr)


# WATCHOUT!!!! first compile the java file with javac script.java and then run it with java script. now works with allRunner.py
def run_java_program():
    result = subprocess.run(
        ["python", "java/jRunner.py"], capture_output=True, text=True
    )
    print("Python script output:")
    print(result.stdout)
    print(result.stderr)


def run_cpp_program():
    result = subprocess.run(
        ["python", "cpp/cRunner.py"], capture_output=True, text=True
    )
    print("Python script output:")
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    # print("Esecuzione di tutti i runner:")
    #
    #print("\nEsecuzione R script:")
    run_r_script()

    # print("\nEsecuzione matlabRunner:")
    # run_matlab_script()
    #
    # print("\nEsecuzione programma Java:")
    # #run_java_program()
    #
    # print("\nEsecuzione programma C++:")
    # #run_cpp_program()
    # print("\nEsecuzione script Python:")
    # run_python()
