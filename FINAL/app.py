import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import logging
from tensorflow.keras.models import load_model
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Predictive Maintenance App", page_icon="✈️")

# Title and description
st.title("Predictive Maintenance for Aircraft Engines")
st.markdown("""
This application predicts the Remaining Useful Life (RUL) for aircraft engines using an LSTM model.
Upload a test dataset in the same format as `test_FD001.txt` (26 columns, space-separated, no header) to get predictions.
""")

# File uploader
uploaded_file = st.file_uploader("Upload test dataset (CSV or TXT)", type=['csv', 'txt'])

# Function to preprocess data
def preprocess_data(df, sequence_length=50):
    try:
        # Define columns
        columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 
                   's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 
                   's18', 's19', 's20', 's21']
        df.columns = columns
        
        # MinMax normalization
        cols_normalize = df.columns.difference(['id', 'cycle'])
        scaler = MinMaxScaler()
        norm_df = pd.DataFrame(scaler.fit_transform(df[cols_normalize]), 
                               columns=cols_normalize, 
                               index=df.index)
        df = df[['id', 'cycle']].join(norm_df)
        
        # Generate sequences for LSTM
        sequence_cols = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 
                         's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 
                         's19', 's20', 's21']
        sequences = []
        for engine_id in df['id'].unique():
            engine_data = df[df['id'] == engine_id][sequence_cols]
            if len(engine_data) >= sequence_length:
                for i in range(len(engine_data) - sequence_length + 1):
                    seq = engine_data.iloc[i:i + sequence_length].values
                    sequences.append(seq)
            else:
                logger.warning(f"Engine {engine_id} has insufficient data for sequence length {sequence_length}")
        
        sequences = np.array(sequences)
        logger.info(f"Generated {len(sequences)} sequences with shape {sequences.shape}")
        return sequences, scaler
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

# Function to predict RUL
def predict(model, sequences):
    try:
        predictions = model.predict(sequences, verbose=0)
        logger.info(f"Predictions shape: {predictions.shape}")
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        st.error(f"Prediction failed: {str(e)}")
        return None

# Load model
# @st.cache_resource
# def load_lstm_model():
#     try:
#         model = load_model('E:/Code/DEPI_AMIT/final Project/model/turbofan_model.h5')
#         logger.info("Model loaded successfully")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load model: {str(e)}")
#         st.error(f"Failed to load model: {str(e)}. Ensure the model is saved with TensorFlow 2.17.0 in .h5 format.")
#         return None

import os  # Add this line with your other imports
import streamlit as st
import pandas as pd
# ... rest of your existing imports
@st.cache_resource
def load_lstm_model():
    try:
        # Using relative path - model should be in a 'model' subdirectory
        model_path = os.path.join('model', 'E:/Code/DEPI_AMIT/final Project/turbofan_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}. Ensure the model is saved with TensorFlow 2.17.0 in .h5 format.")
        return None
if uploaded_file is not None:
    # Read uploaded file
    try:
        test_df = pd.read_csv(uploaded_file, sep=" ", header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
        logger.info(f"Test data loaded with shape: {test_df.shape}")
        
        # Preprocess data
        with st.spinner('Preprocessing data...'):
            sequence_length = 20  # Same as used in training
            sequences, scaler = preprocess_data(test_df, sequence_length)
        
        # Load model and predict
        with st.spinner('Loading model and making predictions...'):
            model = load_lstm_model()
            if model is None:
                st.error("Model loading failed. Please retrain the model using TensorFlow 2.17.0 and save as 'turbofan_model.h5'.")
            else:
                predictions = predict(model, sequences)
                if predictions is not None:
                    # Display predictions
                    st.subheader("Predictions")
                    st.write("Predicted Remaining Useful Life (RUL) for sequences:")
                    results = pd.DataFrame(predictions, columns=['Predicted RUL'])
                    st.dataframe(results)
                    
                    # Optional: Download predictions
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a test dataset to get predictions.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Model: LSTM for Predictive Maintenance")