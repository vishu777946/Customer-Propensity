from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import sys
from fastapi.staticfiles import StaticFiles
import nest_asyncio

print(sys.executable)
print(sys.path)

# Load the pre-trained model
model = joblib.load(r'C:\Users\lti-10761456\Desktop\Marketing.Ai_New\Model_pred\app\rf_model (1).pkl')  # Replace with your model path

# Initialize the FastAPI app
app = FastAPI()

# # Mount the frontend directory
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Define the input data schema using Pydantic
class PredictionInput(BaseModel):
    Project: int = Field(alias="Project: Project Name")
    Age: int
    Gender: int
    Nature_of_Purchase: int = Field(alias="Nature of Purchase")
    Current_Residential_Type: int = Field(alias="Current Residential Type")
    Home_Loan_Status: int = Field(alias="Home Loan Status")
    Possession_Required: int = Field(alias="Possession Required")
    Configuration: float
    Desired_Carpet_Area: int = Field(alias="Desired Carpet Area")
    Actual_Budget: float = Field(alias="Actual Budget")
    Visited_with_Family: int = Field(alias="Visited with Family")
    Number_of_Activity: int = Field(alias="Number of Activity")
    Lead_Source: int = Field(alias="Lead Source")
    Sentiment: float
    Sentiment_Label: int = Field(alias="Sentiment_Label")
    Zip_Code_Frequency: float = Field(alias="Zip Code Frequency")
    Walkin_Month: str = Field(alias="Walkin Month")
    Walkin_Day_of_Week: str = Field(alias="Walkin Day of Week")


# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict(by_alias=True)])

        # Ensure column order matches model training
        input_df = input_df[[
            "Project: Project Name", "Age", "Gender", "Nature of Purchase",
            "Current Residential Type", "Home Loan Status", "Possession Required",
            "Configuration", "Desired Carpet Area", "Actual Budget",
            "Visited with Family", "Number of Activity", "Lead Source",
            "Sentiment", "Sentiment_Label", "Zip Code Frequency",
            "Walkin Month", "Walkin Day of Week"
        ]]
        # # Perform prediction
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
           # Create response dictionary
        response = {
            # "index": input_df.index[0],
            "prediction": prediction.tolist()[0],  # Convert to list and get first element
            "probabilities": {
                "booking": f"{probabilities[0][0]*100:.2f}%",
                # "in_pipeline": f"{probabilities[0][1]*100:.2f}%"
            }
        }

        return response
        # predictions_df = pd.DataFrame({
        #           'Prediction': Prediction,
        #           'Probability_0(Booking)': [f"{p*100:.2f}%" for p in probabilities[:, 0]],  # Probability of class 0
        #           'Probability_1(In pipeline)': [f"{p*100:.2f}%" for p in probabilities[:, 1]]  # Probability of class 1
        # }, index=input_df.index) 
        # # return {"prediction": prediction.tolist()}, {"prediction": predictions_1}
        # return {Predictions_df}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    # # For batch predictions
# @app.post("/predict_batch")
# async def predict_batch(input_data_list: list[PredictionInput]):
#     try:
#         # Process each input and collect results
#         results = []
#         for input_data in input_data_list:
#             # Convert input data to DataFrame
#             input_dict = input_data.dict(by_alias=True)
#             index_value = input_dict.pop('index')
#             input_df = pd.DataFrame([input_dict], index=[index_value])
            
#             # Ensure column order
#             input_df = input_df[[
#                 "Project: Project Name", "Age", "Gender", "Nature of Purchase",
#                 "Current Residential Type", "Home Loan Status", "Possession Required",
#                 "Configuration", "Desired Carpet Area", "Actual Budget",
#                 "Visited with Family", "Number of Activity", "Lead Source",
#                 "Sentiment", "Sentiment_Label", "Zip Code Frequency",
#                 "Walkin Month", "Walkin Day of Week"
#             ]]

#             # Perform prediction
#             prediction = model.predict(input_df)
#             probabilities = model.predict_proba(input_df)

#             # Add result to list
#             results.append({
#                 "index": index_value,
#                 "prediction": prediction.tolist()[0],
#                 "probabilities": {
#                     "booking": f"{probabilities[0][0]*100:.2f}%",
#                     "in_pipeline": f"{probabilities[0][1]*100:.2f}%"
#                 }
#             })

#         return results

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")


# Setup for running the application
if __name__ == "__main__":
    nest_asyncio.apply()  # Allow nested event loops
    uvicorn.run(app, host="10.11.1.82", port=443)  # Replace with your desired host and port
