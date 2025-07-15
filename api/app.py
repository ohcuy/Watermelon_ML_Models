from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import joblib
import librosa
import numpy as np
import uvicorn
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.feature_extractor import AudioFeatureExtractor
from src.data.preprocessor import AudioPreprocessor 

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 배포시에는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_obj = joblib.load("best_model_random_forest.pkl")
model = model_obj.get("model")
scaler = model_obj.get("scaler")

extractor = AudioFeatureExtractor()
preprocessor = AudioPreprocessor()

def extract_features(file_path: str):
    try:
        # 오디오 파일 로드 (m4a 파일 처리 개선)
        try:
            y, sr = librosa.load(file_path, sr=22050)
        except Exception as load_error:
            logger.warning(f"librosa 로드 실패: {load_error}")
            # 대안: ffmpeg 사용 (필요시)
            raise Exception("오디오 파일 로드 실패")
        
        logger.info(f"오디오 로드 성공: {file_path}, 길이: {len(y)/sr:.2f}초")

        # 전처리 실행
        processed_audio, processing_info = preprocessor.preprocess_audio(y, sr)
        
        # processed_audio는 numpy array이므로 직접 사용
        logger.info(f"전처리 완료: {type(processed_audio)}, shape: {processed_audio.shape}")
        
        # 특성 추출
        features = extractor.extract_all_features(processed_audio, sr)
        logger.info(f"특성 추출 완료: {features.shape}")
    
        return features.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"특성 추출 실패: {file_path}, 오류: {e}")
        raise

# 서버 상태 확인 (Swift 앱에서 연결 테스트용)
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "수박 당도 예측 서버가 실행 중입니다"}

# 예측 API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        # 파일 확장자 검사
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".wav", ".m4a", ".mp3"]:
            return JSONResponse(
                status_code=400, 
                content={
                    "success": False,
                    "error": "지원하지 않는 파일 형식입니다. .wav, .m4a, .mp3 파일만 업로드 가능합니다."
                }
            )

        # 임시 파일 저장
        temp_path = f"temp_{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"파일 업로드 완료: {file.filename} -> {temp_path}")

        # 특성 추출 및 예측
        features = extract_features(temp_path)
        
        if scaler:
            features = scaler.transform(features)
            logger.info("특성 스케일링 완료")
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        logger.info(f"예측 완료: {prediction}")
        
        # 결과 반환
        result = {
            "success": True,
            "filename": file.filename,
            "prediction": int(prediction),
            "result": "익었다" if prediction == 1 else "안 익었다",
            "confidence": float(max(probability)) if probability is not None else None
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "success": False,
                "error": f"예측 중 오류가 발생했습니다: {str(e)}"
            }
        )
    finally:
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"임시 파일 삭제: {temp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")

# 지원되는 파일 형식 정보
@app.get("/supported-formats")
def get_supported_formats():
    return {
        "formats": [".wav", ".m4a", ".mp3"],
        "description": "지원되는 오디오 파일 형식"
    }

if __name__ == "__main__":
    import socket
    
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    print(f"🍉 수박 당도 예측 서버 시작")
    print(f"   - 로컬: http://localhost:8000")
    print(f"   - 네트워크: http://{local_ip}:8000")
    print(f"   - 상태 확인: http://{local_ip}:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
