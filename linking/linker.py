import subprocess
import matlab.engine


def run_r_script():
    result = subprocess.run(["Rscript", "script.R"], capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


def run_matlab_script():
    # chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run('script.m', nargout=0)
    eng.quit()


# WATCHOUT!!!! first compile the java file with javac script.java and then run it with java script. now works with linker.py
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
    print("Esecuzione script R:")
    run_r_script()

    print("\nEsecuzione script MATLAB:")
    run_matlab_script()

    print("\nEsecuzione programma Java:")
    run_java_program()

    print("\nEsecuzione programma C++:")
    run_cpp_program()
