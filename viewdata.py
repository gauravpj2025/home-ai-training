# view_data.py
import pandas as pd

# --- Change this to the CSV file you want to view ---
file_to_view = 'signal_data.csv' 
# Other options: 'junction_data.csv', 'signal_data.csv', 'schedule_timetable_data.csv', etc.

# Set display options to show all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150) # Adjust width for your terminal

try:
    # Read the specified CSV file into a pandas DataFrame
    df = pd.read_csv(file_to_view)

    print(f"🚂 Displaying a preview of '{file_to_view}'...")
    print("=" * 50)
    
    # --- This is the key part: print the first 10 rows ---
    print(df.head(10))
    
    print("=" * 50)
    print(f"📄 Table has {len(df)} total rows and {len(df.columns)} columns.")
    print("💡 Tip: Use df.tail(10) to see the last 10 rows or df.sample(10) to see 10 random rows.")

except FileNotFoundError:
    print(f"Error: The file '{file_to_view}' was not found.")
    print("Please make sure you have run the generator script first and the CSV file is in the same directory.")

except Exception as e:
    print(f"An error occurred: {e}")