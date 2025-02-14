# ----------------------------------------------------------- # imports
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import re
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from sklearn.preprocessing import LabelEncoder
# -------------------------------------------------------  #  

def pipeline(excel_path): 
    # List of new column names
    new_columns = [
        'Project: Project Name', 'Site Visit Name', 'Created Date', 'Walkin Date', 
        'Permanent Zip/Postal Code', 'Are you associated with L&T?', 'Age', 'Gender', 
        'Nature of Purchase', 'Current Residential Type', 'Home Loan Status', 
        'Occupation', 'Company Name', 'Designation', 'Industry', 
        'Possession Required', 'Configuration', 'Budget', 'Desired Carpet Area', 
        'Cultural Background', 'Office location', 'Actual Budget', 'Address details', 
        'Tower/Unit pitched', 'Visited with Family', 'Task Status', 
        'Primary Reason', 'Remarks', 'ICs', 'Associated Account: Account Name', 
        'Referred by Account: Account Name', 'No of revisit', 'Last Modified Date', 
        'Number of Activity', 'Lead Source', 'Office Location', 'Organization', 
        'Account Name', 'Stage', 'Channel Partner: Account Name'
    ]

    # Load the Excel file
    df = pd.read_csv(excel_path)
    print("check 0")

    # Replace columns with the new column names
    df.columns = new_columns
    print(df.columns)
    df.isnull().sum() / df.shape[0] * 100
    missing_percentage = df.isnull().sum() / df.shape[0] * 100
    columns_to_drop = missing_percentage[missing_percentage > 15].index
    print("Columns with more than 15% missing values:", columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    df.transpose()
    vishal = df.transpose()
    df.drop(columns=['Primary Reason'], inplace=True)
    invalid_values = ['NA', 'NaN', 'blank', 'other', 'others', 'Na']  # Add any other invalid placeholder values
    df.replace(invalid_values, np.nan, inplace=True)
    missing_values = df.isnull().sum()
    print(missing_values)
    if 'column1' in df.columns and 'column2' in df.columns and pd.api.types.is_numeric_dtype(df['column1']) and pd.api.types.is_numeric_dtype(df['column2']):
        sns.scatterplot(x='column1', y='column2', data=df)
        plt.title('Scatter Plot: column1 vs column2')
        plt.show()
    numeric_columns = df.select_dtypes(include=np.number).columns
    for i in df.select_dtypes(include="object").columns:
        print(df[i].value_counts())
        print("***"*10)
    import re
    df['Walkin Date'].value_counts()
    max_date = df['Walkin Date'].max()
    min_date = df['Walkin Date'].min()
    df['No of revisit'].value_counts()
    df['Project: Project Name'].apply(lambda x: 'Powai' if (x == 'Elixir Reserve') or (x == 'Emerald Isle, Powai') else x)
    df['Project: Project Name'].value_counts()
    num_features = ['Configuration', 'Actual Budget', 'No of revisit','Number of Activity']
    df[df['Budget'] == 0.00]
    df = df[df['Budget'] != 0]
    df = df[df['Budget'] != 0.0]
    df = df[df['Budget'] != 0.00]
    df['Actual Budget'].value_counts()
    df['Budget'].value_counts()
    df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
    min_value = df['Budget'].min()
    max_value = df['Budget'].max()
    df['Budget'].dtype
    df['Actual Budget'].isnull().sum()
    min_value = df['Actual Budget'].min()
    max_value = df['Actual Budget'].max()
    df['Actual Budget'] = pd.to_numeric(df['Actual Budget'], errors='coerce')
    df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
    df.loc[df['Actual Budget'] == 0, 'Actual Budget'] = df.loc[df['Actual Budget'] == 0, 'Budget']
    min_value = df['Actual Budget'].min()
    max_value = df['Actual Budget'].max()
    grouped_columns = df.columns.to_series().groupby(df.dtypes).groups
    num_features = ['Configuration', 'Actual Budget', 'No of revisit','Number of Activity']
    df['Actual Budget'].isnull().sum()
    df['Possession Required'] = df['Possession Required'].replace(['Immediate', 'Upfront'], '1 months')
    df['Possession Required'].value_counts()
    df['Possession Required'] = df['Possession Required'].str.replace('months', '').str.strip()
    df['Possession Required'] = pd.to_numeric(df['Possession Required'], errors='coerce')
    df[['Possession Required']].value_counts()
    df['Configuration'].value_counts()
    def process_jodi_value(value):
        if not isinstance(value, (str, bytes)):
            value = str(value)
        match = re.search(r'Jodi\((\d+)\+(\d+)\)', value)
        if match:
            n1 = int(match.group(1))  # First number
            n2 = int(match.group(2))  # Second number
            total = n1 + n2           # Calculate the sum
            return f'{total} BHK'     # Return in "sum BHK" format
        return value  # In case the pattern is not matched, return the original value
    df['Configuration'] = df['Configuration'].apply(process_jodi_value)
    df['Configuration'].value_counts()
    df.dropna(subset=['Configuration'], inplace=True)
    df['Desired Carpet Area'].replace({'550 to 650 sq.ft.': 600})
    df['Desired Carpet Area'] = df['Desired Carpet Area'].astype(str)
    df['Desired Carpet Area'] = df['Desired Carpet Area'].str.replace('sq.ft.', '').str.strip()
    df['Desired Carpet Area'] = pd.to_numeric(df['Desired Carpet Area'], errors='coerce')
    df[['Desired Carpet Area']].value_counts()
    plt.rcParams['figure.figsize'] = (36, 24)  # Adjust width and height as needed
    if 'Last Modified Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Last Modified Date']):
            try:
                df['Last Modified Date'] = pd.to_datetime(df['Last Modified Date'])
            except ValueError:
                print("Could not convert 'Last Modified Date' to datetime objects.")
    df_sorted = df.sort_values(by='Last Modified Date')  
    mapping_dict = {
        'Channel Partner': 'Channel Partner',
        'Social Media': 'Digital',
        'Website': 'Digital',
        'Online Campaign': 'Digital',
        'WhatsApp': 'Digital',
        'SMS': 'Digital',
        'Emailer': 'Digital',
        'Chatbot': 'Digital',
        'Property Portals': 'Digital',
        'Radio': 'Digital',
        'Online%Campaign': 'Digital',
        'Virtual Property Expo': 'Digital',
        'Property Portal': 'Digital',
        'Webinar': 'Digital',
        'Google AdWords': 'Digital',
        'Ozonetel CTI': 'Digital',
        'Social+Media': 'Digital',
        'Prgrammatic': 'Digital',
        'Radio Ad': 'Digital',
        'Hoarding': 'Offline',
        'Print Ad': 'Offline',
        'Paper Ads.': 'Offline',
        'Epinet': 'Offline',
        'Exhibition & event': 'Offline',
        'Direct': 'Offline',
        'Newspaper Leaflet': 'Offline',
        'Others': 'Offline',
        'Cross Project': 'Offline',
        'Indirect': 'Offline',
        'Exhibition': 'Offline',
        'Paper Ads': 'Offline',
        'Digital Screens': 'Offline',
        'Purchased List': 'Offline',
        'Trade Show': 'Offline',
        'Partner': 'Offline',
        'Advertisement': 'Offline',
        'Customer Event': 'Offline',
        'Other': 'Offline',
        'Phone': 'Offline',
        'Referral': 'Referral',
        'Employee Referral': 'Referral'
    }
    df['Lead Source'] = df['Lead Source'].map(mapping_dict).fillna(df['Lead Source'])
    df['Stage'] = df['Stage'].replace('APP 2 Approved', 'Booked')
    df = df[df['Stage'] != 'Cancelled']
    df['Stage'] = df['Stage'].where(df['Stage'] == 'Booked', 'In Pipeline')
    sia = SentimentIntensityAnalyzer()
    def get_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    df['Remarks'] = df['Remarks'].fillna('').astype(str)
    df['Sentiment'] = df['Remarks'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['Sentiment_Label'] = df['Sentiment'].apply(get_sentiment)
    invalid_count = df['Permanent Zip/Postal Code'].isnull().sum() + (df['Permanent Zip/Postal Code'] == '000000').sum()
    five_digit_zip_codes = df[df['Permanent Zip/Postal Code'].astype(str).str.match(r'^\d{5}$')]
    count_five_digit_zip_codes = five_digit_zip_codes.shape[0]
    # Check for pin codes that are not exactly 6 digits in the 'Permanent Zip/Postal Code' column
    invalid_zip_codes = df[~df['Permanent Zip/Postal Code'].astype(str).str.match(r'^\d{6}$')]

    # Display the rows with invalid pin codes (not 6 digits)
    print(invalid_zip_codes)

    # Optional: Count how many rows have invalid pin codes
    count_invalid_zip_codes = invalid_zip_codes.shape[0]
    print(f"Number of rows with invalid pin codes (not 6 digits): {count_invalid_zip_codes}")

    # Define a function to fix the postal codes
    def fix_postal_code(code):
        code_str = str(code)  # Convert the code to a string
        if code_str.startswith('400'):
            if len(code_str) == 5:  # If the length is 5, add a '0'
                return code_str[:2] + '0' + code_str[2:]
            elif len(code_str) > 6:  # If the length is more than 6, trim to 6 digits
                return code_str[:6]
        return code_str  # Return the original code if no changes are needed
    df['Permanent Zip/Postal Code'] = df['Permanent Zip/Postal Code'].apply(fix_postal_code)
    print(df['Permanent Zip/Postal Code'])
    invalid_zip_codes = df[~df['Permanent Zip/Postal Code'].astype(str).str.match(r'^\d{6}$')]
    print(invalid_zip_codes)
    count_invalid_zip_codes = invalid_zip_codes.shape[0]
    print(f"Number of rows with invalid pin codes (not 6 digits): {count_invalid_zip_codes}")
    df['Permanent Zip/Postal Code'].value_counts()
    df['Permanent Zip/Postal Code'] = df['Permanent Zip/Postal Code'].apply(lambda x: '400076' if (pd.isnull(x) or not isinstance(x, str) or not x.isdigit() or len(x) != 6) else x)
    geolocator = Nominatim(user_agent="geoapiExercises", timeout=10)  # Set timeout to 10 seconds
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    def get_lat_long(postal_code):
        try:
            location = geolocator.geocode(postal_code)
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except Exception as e:
            print(f"Error for {postal_code}: {e}")
            return None, None
    df.set_index('Site Visit Name', inplace=True)
    df.drop(columns=['Created Date','Occupation','Industry','Cultural Background','Designation','Budget','Address details','Tower/Unit pitched','Remarks','Task Status','Last Modified Date','Organization', 'Account Name'], inplace=True)
    vishal = df.copy()
    def process_jodi_value(value):
        match = re.search(r'Jodi\((\d+)\+(\d+)\)', value)
        if match:
            n1 = int(match.group(1))  # First number
            n2 = int(match.group(2))  # Second number
            total = n1 + n2           # Calculate the sum
            return f'{total} BHK'     # Return in "sum BHK" format
        return value  # In case the pattern is not matched, return the original value
    df['Configuration'] = df['Configuration'].apply(process_jodi_value)
    df.transpose().sample(10)
    df['Configuration'] = df['Configuration'].astype(str)
    df['Configuration'] = df['Configuration'].str.replace(r'\s*BHK', '', regex=True).astype('float64')
    frequency_encoding = df['Permanent Zip/Postal Code'].value_counts(normalize=True)
    df['Zip Code Frequency'] = df['Permanent Zip/Postal Code'].map(frequency_encoding)
    df.transpose()
    df['Walkin Date'] = pd.to_datetime(df['Walkin Date'], errors='coerce')
    df['Walkin Year'] = df['Walkin Date'].dt.year
    df['Walkin Month'] = df['Walkin Date'].dt.month
    df['Walkin Day'] = df['Walkin Date'].dt.day
    df['Walkin Day of Week'] = df['Walkin Date'].dt.dayofweek
    df['Walkin Quarter'] = df['Walkin Date'].dt.quarter
    df['Walkin Date'] = pd.to_datetime(df['Walkin Date'], errors='coerce')
    df['Walkin Ordinal'] = df['Walkin Date'].map(lambda x: x.toordinal() if pd.notna(x) else pd.NaT)
    df['Walkin Month'] = df['Walkin Month'].astype('category')
    df['Walkin Day of Week'] = df['Walkin Day of Week'].astype('category')
    print("check 1")
    print(df.columns)
    df.drop(columns=['Walkin Date','Permanent Zip/Postal Code','No of revisit'], inplace=True)
    columns_to_drop = ['Walkin Year', 'Walkin Day', 'Walkin Quarter', 'Walkin Ordinal']
    df = df.drop(columns=columns_to_drop)
    print("check 2")
    print(df.columns)
    numerical_df = df.select_dtypes(include=['number'])
    correlation_matrix = numerical_df.corr()
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age'] = pd.to_numeric(df['Age'])
    numerical_df = df.select_dtypes(include=['number'])
    correlation_matrix = numerical_df.corr()
    categorical_columns = ['Project: Project Name','Gender', 'Nature of Purchase', 'Current Residential Type', 'Home Loan Status',
                        'Visited with Family', 'Lead Source', 'Stage','Sentiment_Label','Walkin Month',
        'Walkin Day of Week']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df_encoded.head()
    X = df.drop(columns=['Stage'])  # Features
    y = df['Stage']  # Target variable
    vishal = vishal.drop(columns=['Stage'])
    savex = X.copy()
    savey = y.copy()
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X

#file_path = r'C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\Site Visit All Details-2024-12-03-12-36-21.xlsx'
#X = pipeline(file_path)
# # print(X.head())
# print("chewcki 3")
# X = pd.read_excel(file_path)
# print(X.columns)

# file_path = r'C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\salesforce_production_report.csv'
# df = pd.read_csv(file_path)
# X = pipeline(file_path)
# print(X.head())
# print(X.columns)


