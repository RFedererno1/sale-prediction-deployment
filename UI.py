import streamlit as st
import requests
import time
import pandas as pd

# Title of the app
st.title("Sales Prediction App")

BACKEND_API_URL = 'http://localhost:8000'

# Input for item_id
item_id = st.number_input("Enter Item ID:", min_value=0)

# Function to check prediction result
def check_prediction(inference_id):
    try:
        # API call to check the result
        response = requests.get(f"{BACKEND_API_URL}/result/result/{inference_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

# Submit button to start prediction
if st.button("Predict"):
    # Show a spinner while waiting for the prediction
    with st.spinner("Sending request to the server..."):
        try:
            # Step 1: Send POST request to FastAPI for prediction
            response = requests.post(f"{BACKEND_API_URL}/inference/inference?item_id={int(item_id)}")
            response.raise_for_status()  # Raise an error for bad responses
            result = response.json()

            if 'inference_id' in result:
                inference_id = result['inference_id']
                st.write(f"Waiting for result from server.")

                # Step 2: Polling for result
                poll_interval = 0.5  # seconds between polls
                max_polls = 10      # maximum number of polls
                polls = 0
                prediction_result = None
                
                # Loop for polling
                with st.spinner("Waiting for prediction result..."):
                    while polls < max_polls:
                        # Wait before next poll
                        time.sleep(poll_interval)
                        
                        # Check prediction result
                        prediction_result = check_prediction(inference_id)
                        
                        if 'predicted_value' in prediction_result:
                            break  # Stop polling when result is found
                        polls += 1

                # Step 3: Display results if found
                if prediction_result and 'predicted_value' in prediction_result:
                    st.success("Prediction result received!")
                    predicted_value = prediction_result['predicted_value']
                    predicted_value = {int(key): value for key, value in predicted_value.items()}
                    predicted_value = dict(sorted(predicted_value.items()))
                    # Create a DataFrame from the result
                    prediction_df = pd.DataFrame(predicted_value.items(), columns=['Shop ID', 'Sell Predcition'])
                    st.write("### Sale prediction for 11/2015 for item {item_id}:")
                    st.table(prediction_df)
                else:
                    st.error("No prediction result received yet. Please try again later.")
            else:
                st.error("Prediction request failed. No inference ID received.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

