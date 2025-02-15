import pandas as pd
from simple_salesforce import Salesforce
import requests

# Salesforce credentials (Production)
security_token = "abac223"
domain_url = "lntrealty.my.salesforce.com"
username = "user.com"
password = "password"

# Salesforce Authentication (Production)
sf_instance = f"https://{domain_url}"
try:
    sf = Salesforce(
        username=username,
        password=password,
        security_token=security_token,
        domain="login"  # Specify 'login' for production environment
    )
    print("Successfully authenticated to Salesforce!")
except Exception as e:
    print("Error during Salesforce authentication:", e)
    exit()

# Load Excel file (Ensure the correct path)
file_path = r"C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\predictions_1_month_trail.xlsx"
try:
    df = pd.read_excel(file_path)  # Use read_excel for Excel files
    print("Excel file loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Rename columns for better handling
df.columns = ["Site_Visit_Name", "Booking_Propensity"]

# Clean and convert Booking Propensity values
df["Booking_Propensity"] = df["Booking_Propensity"].str.rstrip('%').astype(float) / 100  # Convert to decimal

# Function to get Salesforce record ID for each Site_Visit_Name
def get_salesforce_id(site_visit_name):
    """Fetch Salesforce record ID based on Site Visit Name."""
    query = f"SELECT Id FROM Site_Visit__c WHERE Name = '{site_visit_name}' LIMIT 1"
    try:
        result = sf.query(query)
        if result["records"]:
            return result["records"][0]["Id"]
    except Exception as e:
        print(f"Error querying Salesforce for {site_visit_name}: {e}")
    return None  # Return None if no record found

# Add Salesforce IDs to DataFrame
df["Salesforce_Id"] = df["Site_Visit_Name"].apply(get_salesforce_id)

# Remove records without valid Salesforce IDs
df = df.dropna(subset=["Salesforce_Id"])

# Update Salesforce records
for index, row in df.iterrows():
    record_id = row["Salesforce_Id"]
    booking_propensity = row["Booking_Propensity"]

    try:
        sf.Site_Visit__c.update(record_id, {
            "Booking_Propensity__c": booking_propensity
        })
        print(f"Updated {row['Site_Visit_Name']} (ID: {record_id}) with Booking Propensity: {booking_propensity}")
    except Exception as e:
        print(f"Error updating {row['Site_Visit_Name']} (ID: {record_id}): {e}")

print("Salesforce update completed!")

# Fetch the report data from Salesforce
report_id = "00OGB00000E6qji2AB"
report_url = f"{sf_instance}/services/data/v57.0/analytics/reports/{report_id}"
headers = {
    "Authorization": f"Bearer {sf.session_id}",
    "Content-Type": "application/json",
}
response = requests.get(report_url, headers=headers)

if response.status_code == 200:
    print("Report fetched successfully!")
    report_data = response.json()

    # Debugging: Inspect detailColumns structure
    print("detailColumns:", report_data['reportMetadata']['detailColumns'])

    # Extract report data
    rows = []
    for row in report_data['factMap']['T!T']['rows']:
        rows.append([cell['label'] for cell in row['dataCells']])

    # Extract column headers correctly
    if isinstance(report_data['reportMetadata']['detailColumns'][0], str):
        # If detailColumns is a list of strings
        columns = report_data['reportMetadata']['detailColumns']
    else:
        # If detailColumns contains dictionaries with 'label' keys
        columns = [field['label'] for field in report_data['reportMetadata']['detailColumns']]

    # Convert to DataFrame
    df_report = pd.DataFrame(rows, columns=columns)

    # Save the report to a CSV file
    output_file = "salesforce_production_report.csv"
    df_report.to_csv(output_file, index=False)
    print(f"Report saved to {output_file}")
else:
    print("Failed to fetch report:", response.status_code, response.text)
