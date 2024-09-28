import os
import pandas as pd

def processCsv(language, folderPath):
    # Initialize an empty list to hold dataframes
    merged_dataframes = []

    # Loop through all files in the folder
    for filename in os.listdir(folderPath):
        if filename.endswith('.csv'):
            # Split the filename into components
            try:
                algorithm, dataset, phase, _ = filename.split('_')
            except ValueError:
                # If filename doesn't match the expected pattern, skip it
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            # Construct the full file path
            file_path = os.path.join(folderPath, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Add new columns for algorithm, dataset, language, and phase
            df['algorithm'] = algorithm
            df['dataset'] = dataset
            df['language'] = language
            df['phase'] = phase

            # Append the DataFrame to the list
            merged_dataframes.append(df)

    # Concatenate all DataFrames in the list
    if merged_dataframes:
        merged_df = pd.concat(merged_dataframes, ignore_index=True)
    else:
        print("No CSV files were found or no files matched the expected naming convention.")
        return None

    # Save the final merged DataFrame to a new CSV file
    output_file = f"emissions_detailed.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")

