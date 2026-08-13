import pandas as pd
import time
# Path to the AI4I 2020 dataset
DATA_PATH = "data/raw/ai4i2020.csv"
def simulate_stream():
    """
    Simulate a live machine sensor stream by reading
    and printing one dataset row at a time.
    """
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    print("Starting predictive maintenance sensor stream...")
    print(f"Total sensor readings: {len(df)}")
    print("Press Ctrl+C to stop the stream.\n")
    try:
        # Read and display one row at a time
        for index, row in df.iterrows():
            print(f"--- Sensor Reading {index + 1} ---")
            print(row.to_dict())
            # Wait 1 second before the next sensor reading
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
# Run the simulation when this file is executed directly
if __name__ == "__main__":
    simulate_stream()