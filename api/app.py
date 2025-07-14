from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import os
import uuid
import joblib
import librosa
import numpy as np

# 🚨 패키지 임포트 경로: src 폴더를 파이썬 모듈로 인식시켜야 함
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.feature_extractor import AudioFeatureExtractor

app = FastAPI()

# 모델 로드
model_obj = joblib.load("best_model_random_forest.pkl")  # 경로 확인
model = model_obj.get("model")
scaler = model_obj.get("scaler")

# ✅ Feature Extractor 인스턴스화
extractor = AudioFeatureExtractor()

# ✅ 오디오에서 51차원 feature 추출
def extract_features(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    features = extractor.extract_all_features(y, sr)
    return features.reshape(1, -1)  # (1, 51)

# ✅ 업로드 폼 (GET)
@app.get("/upload-form", response_class=HTMLResponse)
def upload_form():
    return """
    <html><body>
    <h2>음성 파일 업로드 (.wav, .m4a)</h2>
    <form action="/predict" enctype="multipart/form-data" method="post">
        <input name="file" type="file" accept=".wav,.m4a">
        <input type="submit" value="예측">
    </form>
    </body></html>
    """

# ✅ 예측 API (POST)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".wav", ".m4a", ".mp3"]:
        return JSONResponse(status_code=400, content={"error": "지원 형식: .wav, .m4a, .mp3"})

    temp_path = f"temp_{uuid.uuid4()}{ext}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        features = extract_features(temp_path)
        if scaler:
            features = scaler.transform(features)
        pred = model.predict(features)[0]
        label = "익었다" if pred == 1 else "안 익었다"
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"예측 실패: {str(e)}"})
    finally:
        os.remove(temp_path)

    return {"filename": file.filename, "result": label}
