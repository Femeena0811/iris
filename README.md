# Iris Flower Classification API

A FastAPI-based machine learning application that classifies iris flower species based on their sepal and petal measurements.

## 📋 Project Overview

This project demonstrates end-to-end machine learning model deployment using FastAPI. The application uses a Logistic Regression model trained on the famous Iris dataset to predict iris species (Iris-setosa, Iris-versicolor, Iris-virginica) from flower measurements.

## 🚀 Features

- **Real-time predictions** with high accuracy
- **RESTful API** with proper HTTP status codes
- **Input validation** using Pydantic models
- **Interactive documentation** with Swagger UI
- **Confidence scores** and probability distributions
- **Error handling** and input validation
- **Model metadata** endpoint

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Machine Learning**: Scikit-learn
- **Model**: Logistic Regression
- **Data Processing**: Pandas, NumPy
- **Serialization**: Joblib
- **Validation**: Pydantic

## 📁 Project Structure

```
iris
│
├── main.py                 # FastAPI application
├── iris_training.ipynb     # Model training notebook
├── requirements.txt        # Dependencies
├── Iris.csv               # Dataset
├── iris_model.pkl         # Trained model
├── label_encoder.pkl      # Label encoder
└── README.md              # This file
```

## ⚡ Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Iris_FastAPI_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already done)
   - Open and run `iris_training.ipynb` in Jupyter Notebook
   - This will generate `iris_model.pkl` and `label_encoder.pkl`

4. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API**
   - API: http://127.0.0.1:8000
   - Interactive Docs: http://127.0.0.1:8000/docs
   - Alternative Docs: http://127.0.0.1:8000/redoc

## 📊 API Endpoints

### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "status": "healthy",
  "message": "Iris Classification API is running",
  "model_loaded": true,
  "model_type": "LogisticRegression"
}
```

### 2. Make Prediction
```http
POST /predict
```
**Request:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
**Response:**
```json
{
  "prediction": "Iris-setosa"
}
```

### 3. Prediction with Confidence
```http
POST /predict-with-confidence
```
**Response:**
```json
{
  "prediction": "Iris-setosa",
  "confidence": 0.98,
  "probabilities": {
    "Iris-setosa": 0.98,
    "Iris-versicolor": 0.01,
    "Iris-virginica": 0.01
  }
}
```

### 4. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "LogisticRegression",
  "problem_type": "multiclass classification",
  "features_used": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "target_classes": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
  "accuracy": "1.0 (from training)",
  "data_source": "Iris.csv"
}
```

### 5. Test Examples
```http
GET /test-examples
```
Returns predefined test cases with expected results.

## 🧪 Testing the API

### Using Interactive Documentation
1. Visit http://127.0.0.1:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Enter your data and click "Execute"

### Using curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Using Python
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(response.json())
```

## 📈 Model Performance

- **Algorithm**: Logistic Regression
- **Accuracy**: 100% on test set
- **Training Time**: < 1 second
- **Inference Speed**: Real-time predictions
- **Dataset**: 150 samples (Iris dataset)

## 🔧 Development

### Dependencies
All required packages are listed in `requirements.txt`:
```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.25.2
joblib==1.3.2
```

### Running in Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📝 Dataset Information

The Iris dataset contains 150 samples of iris flowers with the following features:
- **SepalLengthCm**: Sepal length in cm
- **SepalWidthCm**: Sepal width in cm
- **PetalLengthCm**: Petal length in cm
- **PetalWidthCm**: Petal width in cm

**Target Classes:**
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## ⚠️ Limitations

- Model trained specifically on Iris dataset
- Input features must follow training data distribution
- No advanced preprocessing applied
- Best performance within feature ranges of training data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is for educational purposes as part of a class assignment.

