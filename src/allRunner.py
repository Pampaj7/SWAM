import subprocess
# import matlab.engine
from codecarbon import EmissionsTracker


def run_r_script():
    result = subprocess.run(["Rscript", "R/rRunner.R"], capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


def run_python():
    result = subprocess.run(["python", "python/pythonRunner.py"], capture_output=True, text=True)
    print("Python script output:")
    print(result.stdout)
    print(result.stderr)


def run_matlab_script():
    # chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run('matlab/runAlgorithm.m', nargout=0)
    eng.quit()


# WATCHOUT!!!! first compile the java file with javac script.java and then run it with java script. now works with allRunner.py
def compile_java():
    try:
        # Compila il codice Java
        subprocess.run(["javac", "script.java"], check=True)
        print("Compilazione completata con successo.")
    except subprocess.CalledProcessError as e:
        print("Errore durante la compilazione:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def run_java_program():
    compile_java()
    try:
        # Esegui il programma Java
        result = subprocess.run(["java", "script"], capture_output=True, text=True, check=True)
        print("Output del programma Java:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del programma Java:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def compile_cpp():
    try:
        # Compila il codice C++
        subprocess.run(["g++", "-o", "script", "script.cpp"], check=True)
        print("Compilazione completata con successo.")
    except subprocess.CalledProcessError as e:
        print("Errore durante la compilazione:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def run_cpp_program():
    compile_cpp()
    try:
        # Esegui l'eseguibile compilato
        result = subprocess.run(["./script"], capture_output=True, text=True, check=True)
        print("Output del programma C++:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del programma C++:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


if __name__ == "__main__":
    tracker = EmissionsTracker()

    print("Esecuzione script R:")
    tracker.start()
    run_r_script()
    tracker.stop()

    # print("\nEsecuzione matlabRunner:")
    # tracker.start()
    # run_matlab_script()
    # tracker.stop()
    #
    # print("\nEsecuzione programma Java:")
    # tracker.start()
    # run_java_program()
    # tracker.stop()
    #
    # print("\nEsecuzione programma C++:")
    # tracker.start()
    # run_cpp_program()
    # tracker.stop()
    # print("\nEsecuzione script Python:")
    # tracker.start()
    # run_python()
    # tracker.stop()
