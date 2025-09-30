# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional

# Load your trained model and label encoder
try:
    model = joblib.load("iris_model.pkl")
    le = joblib.load("label_encoder.pkl")
    print("✅ Model and label encoder loaded successfully!")
except FileNotFoundError:
    print("❌ Model files not found. Please run model_training.py first!")
    exit(1)

app = FastAPI(
    title="Iris Flower Classification API",
    description="API for classifying iris flower species using Iris.csv dataset",
    version="1.0.0"
)

# Define the expected structure of the input data using Pydantic
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Define the structure of the response
class PredictionOutput(BaseModel):
    predicted_species: str
    confidence: float
    species_code: int

# For batch predictions
class BatchPredictionInput(BaseModel):
    samples: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

# Health Check Endpoint
@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "message": "Iris Classification API is running",
        "model_loaded": True,
        "timestamp": pd.Timestamp.now().isoformat()
    }

# Main Prediction Endpoint
@app.post("/predict", response_model=PredictionOutput, summary="Predict Iris Species")
def predict(input_data: PredictionInput):
    """
    Predict the species of an iris flower based on its measurements.
    
    - **sepal_length**: Sepal length in cm
    - **sepal_width**: Sepal width in cm  
    - **petal_length**: Petal length in cm
    - **petal_width**: Petal width in cm
    
    Returns the predicted species and confidence score.
    """
    try:
        # Validate input ranges (based on your CSV data)
        if not (4.0 <= input_data.sepal_length <= 8.0):
            raise HTTPException(status_code=400, detail="Sepal length should be between 4.0 and 8.0 cm")
        if not (2.0 <= input_data.sepal_width <= 4.5):
            raise HTTPException(status_code=400, detail="Sepal width should be between 2.0 and 4.5 cm")
        if not (1.0 <= input_data.petal_length <= 7.0):
            raise HTTPException(status_code=400, detail="Petal length should be between 1.0 and 7.0 cm")
        if not (0.1 <= input_data.petal_width <= 2.5):
            raise HTTPException(status_code=400, detail="Petal width should be between 0.1 and 2.5 cm")

        # Convert the input data into a numpy array for the model
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])

        # Make prediction
        prediction_encoded = model.predict(features)[0]  # This gives a number (0, 1, 2)
        predicted_species = le.inverse_transform([prediction_encoded])[0]  # Convert number to species name

        # Get prediction probabilities for confidence score
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max())  # Get the highest probability

        # Return the response
        return PredictionOutput(
            predicted_species=predicted_species,
            confidence=confidence,
            species_code=int(prediction_encoded)
        )

    except HTTPException:
        raise
    except Exception as e:
        # If anything goes wrong, return a 400 error with the message
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Batch Prediction Endpoint (Bonus)
@app.post("/predict-batch", response_model=BatchPredictionOutput, summary="Batch Predict Iris Species")
def predict_batch(input_data: BatchPredictionInput):
    """
    Predict multiple iris species at once.
    
    Provide a list of iris measurements for batch prediction.
    """
    try:
        features_list = []
        for sample in input_data.samples:
            features_list.append([
                sample.sepal_length,
                sample.sepal_width,
                sample.petal_length,
                sample.petal_width
            ])
        
        features_array = np.array(features_list)
        
        # Make batch predictions
        predictions_encoded = model.predict(features_array)
        predicted_species = le.inverse_transform(predictions_encoded)
        
        # Get probabilities for confidence scores
        probabilities = model.predict_proba(features_array)
        confidences = probabilities.max(axis=1)
        
        # Prepare response
        predictions_output = []
        for i in range(len(predictions_encoded)):
            predictions_output.append(
                PredictionOutput(
                    predicted_species=predicted_species[i],
                    confidence=float(confidences[i]),
                    species_code=int(predictions_encoded[i])
                )
            )
        
        return BatchPredictionOutput(predictions=predictions_output)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

# Model Info Endpoint
@app.get("/model-info", summary="Get Model Information")
def model_info():
    """
    Get information about the trained model and dataset.
    """
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "multiclass classification",
        "features_used": [
            "sepal_length (cm)",
            "sepal_width (cm)", 
            "petal_length (cm)",
            "petal_width (cm)"
        ],
        "target_classes": le.classes_.tolist(),
        "class_mapping": {
            0: "Iris-setosa",
            1: "Iris-versicolor", 
            2: "Iris-virginica"
        },
        "data_source": "Iris.csv",
        "feature_ranges": {
            "sepal_length": {"min": 4.3, "max": 7.9},
            "sepal_width": {"min": 2.0, "max": 4.4},
            "petal_length": {"min": 1.0, "max": 6.9},
            "petal_width": {"min": 0.1, "max": 2.5}
        }
    }

# Dataset Info Endpoint
@app.get("/dataset-info", summary="Get Dataset Information")
def dataset_info():
    """
    Get statistics and information about the Iris dataset.
    """
    try:
        df = pd.read_csv('Iris.csv')
        
        dataset_info = {
            "total_samples": len(df),
            "features": df.columns.tolist(),
            "species_distribution": df['Species'].value_counts().to_dict(),
            "feature_statistics": {
                "SepalLengthCm": {
                    "min": float(df['SepalLengthCm'].min()),
                    "max": float(df['SepalLengthCm'].max()),
                    "mean": float(df['SepalLengthCm'].mean()),
                    "std": float(df['SepalLengthCm'].std())
                },
                "SepalWidthCm": {
                    "min": float(df['SepalWidthCm'].min()),
                    "max": float(df['SepalWidthCm'].max()),
                    "mean": float(df['SepalWidthCm'].mean()),
                    "std": float(df['SepalWidthCm'].std())
                },
                "PetalLengthCm": {
                    "min": float(df['PetalLengthCm'].min()),
                    "max": float(df['PetalLengthCm'].max()),
                    "mean": float(df['PetalLengthCm'].mean()),
                    "std": float(df['PetalLengthCm'].std())
                },
                "PetalWidthCm": {
                    "min": float(df['PetalWidthCm'].min()),
                    "max": float(df['PetalWidthCm'].max()),
                    "mean": float(df['PetalWidthCm'].mean()),
                    "std": float(df['PetalWidthCm'].std())
                }
            }
        }
        return dataset_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

# Example data endpoint
@app.get("/examples", summary="Get Example Data")
def get_examples():
    """
    Get example iris measurements from the dataset.
    """
    examples = [
        {
            "description": "Iris-setosa example",
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "expected_species": "Iris-setosa"
        },
        {
            "description": "Iris-versicolor example", 
            "sepal_length": 6.0,
            "sepal_width": 2.7,
            "petal_length": 5.1,
            "petal_width": 1.6,
            "expected_species": "Iris-versicolor"
        },
        {
            "description": "Iris-virginica example",
            "sepal_length": 6.5,
            "sepal_width": 3.0,
            "petal_length": 5.2,
            "petal_width": 2.0,
            "expected_species": "Iris-virginica"
        }
    ]
    return {"examples": examples}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)