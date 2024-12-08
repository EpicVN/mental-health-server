from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả domain. Thay bằng ["http://localhost:3000"] nếu cần.
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức (GET, POST, OPTIONS,...)
    allow_headers=["*"],  # Cho phép tất cả headers
)

# Load model và encoders
model = joblib.load('weights/best_model.joblib')
encoders = joblib.load('encoders/label_encoders.joblib')

# Định nghĩa lớp dữ liệu đầu vào
class DepressionInput(BaseModel):
    Gender: Optional[str] = "Unknown"
    Age: int
    City: Optional[str] = "Unknown"
    Working_Professional_or_Student: Optional[str] = "Unknown"
    Profession: Optional[str] = "Unknown"
    Work_Pressure: int
    Job_Satisfaction: int
    Sleep_Duration: Optional[str] = "Unknown"
    Dietary_Habits: Optional[str] = "Unknown"
    Degree: Optional[str] = "Unknown"
    Have_Suicidal_Thoughts: Optional[str] = "Unknown"
    Work_Hours: int
    Financial_Stress: int
    Family_History: Optional[str] = "Unknown"

# Load data
path = 'data/train.csv'
df = pd.read_csv(path)

# Filling null values with the mean (or median) for numerical columns
df["Work Pressure"] = df["Work Pressure"].fillna(df["Work Pressure"].median())
df["Job Satisfaction"] = df["Job Satisfaction"].fillna(df["Job Satisfaction"].median())

# Filling null values with a default category
df["Profession"] = df["Profession"].fillna("Unknown")
df["Degree"] = df["Degree"].fillna(df["Degree"].mode()[0])
df["Dietary Habits"] = df["Dietary Habits"].fillna(df["Dietary Habits"].mode()[0])
df["Financial Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].mode()[0])


# Endpoint để lấy các tùy chọn
@app.get("/options")
def get_options():
    options = {
        "Gender": df["Gender"].unique().tolist(),
        "City": df["City"].unique().tolist(),
        "Working_Professional_or_Student": df["Working Professional or Student"].unique().tolist(),
        "Profession": df["Profession"].unique().tolist(),
        "Sleep_Duration": df["Sleep Duration"].unique().tolist(),
        "Dietary_Habits": df["Dietary Habits"].unique().tolist(),
        "Degree": df["Degree"].unique().tolist(),
        "Have_Suicidal_Thoughts": df["Have you ever had suicidal thoughts ?"].unique().tolist(),
        "Family_History": df["Family History of Mental Illness"].unique().tolist(),
    }
    return options


@app.post("/predict")
def predict(data: DepressionInput):
    # Create Input
    columns = [
        'Gender', 'Age', 'City', 'Working Professional or Student', 'Profession', 'Work Pressure',
        'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
        'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness'
    ]
    
    data_input = [
        data.Gender, data.Age, data.City, data.Working_Professional_or_Student, 
        data.Profession, data.Work_Pressure, data.Job_Satisfaction, data.Sleep_Duration, 
        data.Dietary_Habits, data.Degree, data.Have_Suicidal_Thoughts, data.Work_Hours, 
        data.Financial_Stress, data.Family_History
    ]
    
    input_data = pd.DataFrame([data_input], columns=columns)
    
    # Xử lý dự đoán (giả sử mô hình đã load từ trước)
    # Chuyển đổi dữ liệu nếu cần (sử dụng encoder)
    for name, encoder in encoders.items():
            if name in input_data.columns:
                try:
                    input_data[name] = encoder.transform(input_data[name])
                except ValueError as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Encoding error for column '{name}': {e}"
                    )

    # Dự đoán
    prediction = model.predict(input_data)

    # Kết quả
    id_to_name = {0: "NO DEPRESSION", 1: "DEPRESSION"}
    result = id_to_name[prediction[0]]
    return {"result": result}
