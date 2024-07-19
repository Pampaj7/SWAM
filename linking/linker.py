import subprocess
import matlab.engine


def run_r_script():
    result = subprocess.run(["Rscript", "script.R"], capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


def run_matlab_script():
    #chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run('script.m', nargout=0)
    eng.quit()


def run_java_program():
    try:  # WATCHOUT!!!! first compile the java file with javac script.java and then run it with java script. now works with linker.py
        # Esegui il programma Java
        result = subprocess.run(["java", "script"], capture_output=True, text=True, check=True)
        print("Java program output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore durante l'esecuzione del programma Java:")
        print(e)
        print("Output di errore:")
        print(e.stderr)


def run_cpp_program():
    result = subprocess.run(["./script"], capture_output=True, text=True)
    print("C++ program output:")
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    print("Esecuzione script R:")
    run_r_script()

    print("\nEsecuzione script MATLAB:")
    run_matlab_script()

    print("\nEsecuzione programma Java:")
    run_java_program()

    print("\nEsecuzione programma C++:")
    run_cpp_program()
