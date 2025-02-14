# import pandas as pd
# from simple_salesforce import Salesforce

# try:
#     sf = Salesforce(
#         username="ltr_bot1@larsentoubro.com",
#         password="ltrbot@123",
#         security_token = "C8A1xS6voj2VUpxyUSjPBhr3k",
#         domain="login"  # Specify 'login' for production environment
#     )
#     print("Successfully authenticated to Salesforce!")
# except Exception as e:
#     print("Error during Salesforce authentication:", e)
#     exit()

# # Load CSV file (Ensure the correct path)
# # file_path = r"C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\app\final_predictions.csv"
# file_path = r"C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\predictions_1_month_trail.xlsx"

# try:
#     df = pd.read_excel(file_path)
#     print("Excel file loaded successfully!")
# except FileNotFoundError:
#     print(f"Error: File not found at {file_path}")
#     exit()

# # Rename columns for better handling
# df.columns = ["Site_Visit_Name", "Booking_Propensity"]

# # Clean and convert Booking Propensity values
# df["Booking_Propensity"] = df["Booking_Propensity"].str.rstrip('%').astype(float)  # Convert to decimal

# # Function to get Salesforce record ID for each Site_Visit_Name
# def get_salesforce_id(site_visit_name):
#     """Fetch Salesforce record ID based on Site Visit Name."""
#     query = f"SELECT Id FROM Site_Visit__c WHERE Name = '{site_visit_name}' LIMIT 1"
#     try:
#         result = sf.query(query)
#         if result["records"]:
#             return result["records"][0]["Id"]
#     except Exception as e:
#         print(f"Error querying Salesforce for {site_visit_name}: {e}")
#     return None  # Return None if no record found

# # Add Salesforce IDs to DataFrame
# df["Salesforce_Id"] = df["Site_Visit_Name"].apply(get_salesforce_id)

# # Remove records without valid Salesforce IDs
# df = df.dropna(subset=["Salesforce_Id"])

# # Update Salesforce records
# for index, row in df.iterrows():
#     record_id = row["Salesforce_Id"]
#     booking_propensity = row["Booking_Propensity"]

#     try:
#         sf.Site_Visit__c.update(record_id, {
#             "Booking_Propensity__c": booking_propensity
#         })
#         print(f"Updated {row['Site_Visit_Name']} (ID: {record_id}) with Booking Propensity: {booking_propensity}")
#     except Exception as e:
#         print(f"Error updating {row['Site_Visit_Name']} (ID: {record_id}): {e}")

# print("Salesforce update completed!")


import pandas as pd
from simple_salesforce import Salesforce
import sys
import logging

# Configure logging
logging.basicConfig(filename="salesforce_update.log", level=logging.ERROR)

# Authenticate with Salesforce
try:
    sf = Salesforce(
        username="ltr_bot1@larsentoubro.com",
        password="ltrbot@123",
        security_token="C8A1xS6voj2VUpxyUSjPBhr3k",
        domain="login"  # Specify 'login' for production
    )
    print("Successfully authenticated to Salesforce!")
except Exception as e:
    print("Error during Salesforce authentication:", e)
    sys.exit(1)

# File path to the Excel data
file_path = r"C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\predictions_1_month_trail.xlsx"

# Ensure openpyxl is installed
try:
    import openpyxl
except ImportError:
    print("Missing optional dependency 'openpyxl'. Install it using 'pip install openpyxl'.")
    sys.exit(1)

# Load the Excel file
try:
    df = pd.read_excel(file_path)
    print("Excel file loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

# Rename columns for easier handling
df.columns = ["Site_Visit_Name", "Prediction", "Record_ID", "Probability_0_Booking", "Probability_1_In_Pipeline"]

# Clean and convert `Probability_0_Booking` to decimals
df["Booking_Propensity"] = df["Probability_0_Booking"].str.rstrip('%').astype(float)

# Update Salesforce records by matching `Site_Visit_Name`
for index, row in df.iterrows():
    site_visit_name = row["Site_Visit_Name"]
    booking_propensity = row["Booking_Propensity"]

    try:
        # Query Salesforce to find the record by Site_Visit_Name
        query = f"SELECT Id, Name FROM Site_Visit__c WHERE Name = '{site_visit_name}'"
        result = sf.query(query)

        if result["totalSize"] == 0:
            print(f"No matching record found in Salesforce for Site Visit Name: {site_visit_name}")
            continue

        # Get the Salesforce record ID
        record_id = result["records"][0]["Id"]

        # Update the record
        sf.Site_Visit__c.update(record_id, {
            "Booking_Propensity__c": booking_propensity
        })
        print(f"Updated {site_visit_name} (ID: {record_id}) with Booking Propensity: {booking_propensity:.2f}")
    except Exception as e:
        error_message = f"Error updating {site_visit_name}: {e}"
        print(error_message)
        logging.error(error_message)

print("Salesforce update completed!")
