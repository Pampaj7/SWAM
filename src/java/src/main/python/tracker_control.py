import os
import sys
from codecarbon import EmissionsTracker
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

tracker = None
output_dir = os.path.abspath("/Users/pampaj/PycharmProjects/SWAM/src/java/output")  # Use absolute path for output directory


def start_tracker(output_file):
    global tracker
    if tracker is None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            tracker = EmissionsTracker(
                output_dir=output_dir, output_file=output_file, log_level="error"
            )
            tracker.start()
            logging.info(
                f"Tracker started. Output will be saved to {os.path.join(output_dir, output_file)}"
            )
        except Exception as e:
            logging.error(f"Error starting tracker: {str(e)}")
    else:
        logging.info("Tracker is already running.")
    sys.stdout.flush()


def stop_tracker():
    global tracker
    if tracker is not None:
        try:
            emissions = tracker.stop()

            if emissions is None:
                logging.warning("Tracker stopped, but emissions data is None.")
            else:
                logging.info(f"Tracker stopped. Emissions: {emissions} kg CO2")
            tracker = None
        except Exception as e:
            logging.error(f"Error stopping tracker: {str(e)}")
    else:
        logging.info("Tracker was not running.")
    sys.stdout.flush()


def command_listener():
    for line in sys.stdin:
        command = line.strip()
        if command.startswith("start"):
            _, output_file = command.split(maxsplit=1)
            start_tracker(output_file)
        elif command == "stop":
            stop_tracker()
        elif command == "exit":
            logging.info("Exiting...")
            sys.stdout.flush()
            break

    # Ensure any remaining output is written
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    command_listener()
