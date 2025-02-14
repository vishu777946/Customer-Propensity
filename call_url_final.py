import pandas as pd
import requests
from pipeline_vs import pipeline

# Step 1: Preprocess data using the pipeline
file_path = r"C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\salesforce_production_report.csv"
processed_data = pipeline(file_path)  # DataFrame with 'Site Visit Name' as index

# Step 2: Predict using FastAPI
url = "http://10.11.1.82:443/predict"
predictions = []

for idx, row in processed_data.iterrows():
    input_data = row.to_dict()

    # Skip rows with NaN values in critical columns
    if pd.isna(input_data['Walkin Month']) or pd.isna(input_data['Walkin Day of Week']):
        print(f"Skipping row {idx} due to NaN in 'Walkin Month' or 'Walkin Day of Week'.")
        continue

    # Convert 'Walkin Month' and 'Walkin Day of Week' to string
    input_data['Walkin Month'] = str(int(input_data['Walkin Month']))
    input_data['Walkin Day of Week'] = str(int(input_data['Walkin Day of Week']))

    # Send POST request
    response = requests.post(url, json=input_data)
    if response.status_code == 200:
        result = response.json()
        booking_probability = result.get('probabilities', {}).get('booking', 'N/A')
        predictions.append([idx, booking_probability])
        print(f"Response for {idx}: {booking_probability}")
    else:
        print(f"Failed for {idx}: {response.status_code}, {response.text}")

# Create DataFrame with correct columns
predictions_df = pd.DataFrame(predictions, columns=["Site Visit Name", "Probability_0(Booking)"])
predictions_df.set_index("Site Visit Name", inplace=True)

# Save to CSV
predictions_df.to_csv('final_predictions.csv')

# Print DataFrame
print(predictions_df)
