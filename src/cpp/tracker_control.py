from codecarbon import EmissionsTracker

# Initialize a global tracker variable
tracker = None


class Tracker:

    @staticmethod
    def start_tracker(output_dir, output_file):
        global tracker
        tracker = EmissionsTracker(output_dir=output_dir, output_file=output_file)
        tracker.start()
        print("Tracker started.")

    def stop_tracker():
        global tracker
        if tracker is not None:
            emissions = tracker.stop()
            print(f"Tracker stopped. Emissions: {emissions} kg CO2")
            tracker = None
        else:
            print("Tracker was not running.")
