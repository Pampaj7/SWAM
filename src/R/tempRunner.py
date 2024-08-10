import subprocess
from codecarbon import EmissionsTracker

def run_r_script():
    result = subprocess.run(["Rscript", "rRunner.R"], capture_output=True, text=True)
    print("R script output:")
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    tracker = EmissionsTracker()

    print("Esecuzione script R:")
    tracker.start()
    run_r_script()
    tracker.stop()