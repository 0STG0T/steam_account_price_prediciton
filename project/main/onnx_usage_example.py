#!/usr/bin/env python
# coding: utf-8

# In[1]:


import onnxruntime as ort
import numpy as np
import pandas as pd

def load_onnx_model(onnx_file_path):
    """
    Loads a saved ONNX model from the given file path.

    Parameters:
    - onnx_file_path: str - Path to the ONNX model file.

    Returns:
    - onnxruntime.InferenceSession - Loaded ONNX model.
    """
    try:
        session = ort.InferenceSession(onnx_file_path)
        print(f"ONNX model loaded successfully from {onnx_file_path}")
        return session
    except Exception as e:
        print(f"Failed to load ONNX model from {onnx_file_path}: {e}")
        return None

def predict_with_onnx(session, X):
    """
    Runs predictions using the loaded ONNX model.

    Parameters:
    - session: onnxruntime.InferenceSession - Loaded ONNX model session.
    - X: pd.DataFrame or np.array - Input features for prediction.

    Returns:
    - np.array - Model predictions.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()  # Convert DataFrame to numpy array
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    preds = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    return preds


# In[4]:


from single_cat_model import SingleCategoryModel

model_placeholder = SingleCategoryModel(category_number=1)

# Load the ONNX model
onnx_model_path = "./models/onnx/category_1_model.onnx"
onnx_session = load_onnx_model(onnx_model_path)

# prepare the data
df = pd.read_json('../data/dataset.json').drop(columns=['sold_price'])
df = model_placeholder.preprocess_data(df=df)

df = df.drop(columns=['sold_price'], errors='ignore')

df.head()


# In[5]:


# Predict using the ONNX model
if onnx_session is not None:
    predictions = predict_with_onnx(onnx_session, df)
    print("Predictions:", predictions)


# In[ ]:




