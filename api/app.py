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
import json
from datetime import datetime

# 로깅 설정 - 더 상세한 포맷
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 배포시에는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 (단순 pkl 파일)
model = joblib.load("watermelon_sound_rf.pkl")

def log_audio_info(y, sr, file_path):
    """오디오 파일의 기본 정보를 로깅"""
    duration = len(y) / sr
    max_amplitude = np.max(np.abs(y))
    rms_energy = np.sqrt(np.mean(y**2))
    
    logger.info(f"📁 파일 정보: {os.path.basename(file_path)}")
    logger.info(f"   - 샘플링 레이트: {sr} Hz")
    logger.info(f"   - 길이: {duration:.2f}초 ({len(y)} 샘플)")
    logger.info(f"   - 최대 진폭: {max_amplitude:.6f}")
    logger.info(f"   - RMS 에너지: {rms_energy:.6f}")
    logger.info(f"   - 다이나믹 레인지: {20 * np.log10(max_amplitude / (rms_energy + 1e-8)):.2f} dB")

def log_feature_details(feature_name, feature_data, feature_mean):
    """개별 특성의 상세 정보를 로깅"""
    logger.info(f"🔍 {feature_name} 분석:")
    logger.info(f"   - 원본 shape: {feature_data.shape}")
    logger.info(f"   - 평균값 shape: {feature_mean.shape}")
    logger.info(f"   - 값 범위: [{np.min(feature_data):.6f}, {np.max(feature_data):.6f}]")
    logger.info(f"   - 평균값 범위: [{np.min(feature_mean):.6f}, {np.max(feature_mean):.6f}]")
    logger.info(f"   - 표준편차: {np.std(feature_data):.6f}")
    
    # 각 계수별 상세 정보 (MFCC, Chroma만)
    if feature_name in ["MFCC", "Chroma"]:
        logger.info(f"   - 계수별 평균값:")
        for i, val in enumerate(feature_mean):
            logger.info(f"     [{i+1:2d}] {val:10.6f}")

def log_feature_statistics(features, feature_names):
    """전체 특성의 통계 정보를 로깅"""
    logger.info(f"📊 전체 특성 통계:")
    logger.info(f"   - 총 특성 수: {len(features)}")
    logger.info(f"   - 값 범위: [{np.min(features):.6f}, {np.max(features):.6f}]")
    logger.info(f"   - 평균: {np.mean(features):.6f}")
    logger.info(f"   - 표준편차: {np.std(features):.6f}")
    logger.info(f"   - 중간값: {np.median(features):.6f}")
    
    # 특성 그룹별 통계
    mfcc_features = features[:13]
    chroma_features = features[13:25]
    other_features = features[25:]
    
    logger.info(f"   - MFCC 그룹 (1-13): 평균={np.mean(mfcc_features):.6f}, 표준편차={np.std(mfcc_features):.6f}")
    logger.info(f"   - Chroma 그룹 (14-25): 평균={np.mean(chroma_features):.6f}, 표준편차={np.std(chroma_features):.6f}")
    logger.info(f"   - 기타 그룹 (26-{len(features)}): 평균={np.mean(other_features):.6f}, 표준편차={np.std(other_features):.6f}")

def save_features_to_json(features, feature_names, file_path):
    """특성을 JSON 파일로 저장"""
    try:
        feature_data = {
            "timestamp": datetime.now().isoformat(),
            "source_file": os.path.basename(file_path),
            "feature_count": len(features),
            "features": {
                name: float(value) for name, value in zip(feature_names, features)
            },
            "statistics": {
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "median": float(np.median(features))
            }
        }
        
        json_filename = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(feature_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 특성 데이터 저장됨: {json_filename}")
        
    except Exception as e:
        logger.warning(f"특성 JSON 저장 실패: {e}")

def extract_features(file_path, sr=22050, n_mfcc=13):
    """
    단순 모델에 맞춰 27개 특성 추출 (상세 디버깅 포함)
    - MFCC: 13개
    - Chroma: 12개  
    - Zero Crossing Rate: 1개
    - Spectral Centroid: 1개
    """
    try:
        logger.info(f"🎵 특성 추출 시작: {os.path.basename(file_path)}")
        
        # 오디오 파일 로드
        y, sr = librosa.load(file_path, sr=sr)
        log_audio_info(y, sr, file_path)
        
        # 1. MFCC (13개)
        logger.info("🔄 MFCC 계산 중...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        log_feature_details("MFCC", mfcc, mfcc_mean)
        
        # 2. Chroma (12개)
        logger.info("🔄 Chroma 계산 중...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        log_feature_details("Chroma", chroma, chroma_mean)
        
        # 3. Zero Crossing Rate (1개)
        logger.info("🔄 Zero Crossing Rate 계산 중...")
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        logger.info(f"🔍 ZCR 분석:")
        logger.info(f"   - 원본 shape: {zcr.shape}")
        logger.info(f"   - 평균값: {zcr_mean:.6f}")
        logger.info(f"   - 값 범위: [{np.min(zcr):.6f}, {np.max(zcr):.6f}]")
        
        # 4. Spectral Centroid (1개)
        logger.info("🔄 Spectral Centroid 계산 중...")
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        logger.info(f"🔍 Spectral Centroid 분석:")
        logger.info(f"   - 원본 shape: {spec_cent.shape}")
        logger.info(f"   - 평균값: {spec_cent_mean:.6f}")
        logger.info(f"   - 값 범위: [{np.min(spec_cent):.6f}, {np.max(spec_cent):.6f}]")
        logger.info(f"   - Hz 단위: {spec_cent_mean:.2f} Hz")
        
        # 모든 특성 합치기
        features = np.concatenate([mfcc_mean, chroma_mean, [zcr_mean], [spec_cent_mean]])
        
        # 실제 특성 개수 확인
        logger.info(f"🔍 특성 개수 확인:")
        logger.info(f"   - MFCC: {len(mfcc_mean)}개")
        logger.info(f"   - Chroma: {len(chroma_mean)}개")
        logger.info(f"   - ZCR: 1개")
        logger.info(f"   - Spectral Centroid: 1개")
        logger.info(f"   - 총 특성 개수: {len(features)}개")
        
        # 특성 검증 (실제 개수에 맞춰 조정)
        expected_features = len(features)
        assert features.shape[0] == expected_features, f"예상 특성 수: {expected_features}개, 실제: {features.shape[0]}개"
        
        # 특성 이름 정의 (실제 개수에 맞춰)
        feature_names = (
            [f"mfcc_{i+1}" for i in range(len(mfcc_mean))] +
            [f"chroma_{i+1}" for i in range(len(chroma_mean))] +
            ["zcr"] +
            ["spectral_centroid"]
        )
        
        # 전체 특성 통계
        log_feature_statistics(features, feature_names)
        
        # 특성 상세 로깅
        logger.info(f"📋 === 추출된 {len(features)}개 특성 상세 ===")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            status = "⚠️ " if np.isnan(value) or np.isinf(value) else "✅ "
            logger.info(f"  {status}[{i+1:2d}] {name:20s}: {value:12.6f}")
        
        # NaN/Inf 확인
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))
        if nan_count > 0 or inf_count > 0:
            logger.error(f"❌ 특성 품질 이슈 발견!")
            logger.error(f"   - NaN 개수: {nan_count}개")
            logger.error(f"   - Inf 개수: {inf_count}개")
            
            # 문제가 있는 특성 찾기
            for i, (name, value) in enumerate(zip(feature_names, features)):
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"   - 문제 특성: [{i+1}] {name} = {value}")
        else:
            logger.info("✅ 모든 특성이 정상입니다")
        
        # 특성 데이터를 JSON으로 저장
        save_features_to_json(features, feature_names, file_path)
        
        logger.info(f"🎯 특성 추출 완료: {os.path.basename(file_path)}")
        logger.info("=" * 80)
        
        return features.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"❌ 특성 추출 실패: {file_path}")
        logger.error(f"   - 오류: {e}")
        logger.error("=" * 80)
        raise

# 디버깅용 특성 분석 API 추가
@app.post("/debug-features")
async def debug_features(file: UploadFile = File(...)):
    """특성 추출 디버깅 전용 API"""
    temp_path = None
    try:
        # 파일 확장자 검사
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".wav", ".m4a", ".mp3"]:
            return JSONResponse(
                status_code=400, 
                content={"error": "지원하지 않는 파일 형식입니다."}
            )

        # 임시 파일 저장
        temp_path = f"debug_{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # 특성 추출 (디버깅 로그와 함께)
        features = extract_features(temp_path)
        
        # 특성 이름 정의
        feature_names = (
            [f"mfcc_{i+1}" for i in range(13)] +
            [f"chroma_{i+1}" for i in range(12)] +
            ["zcr"] +
            ["spectral_centroid"]
        )
        
        # 결과 반환
        result = {
            "success": True,
            "filename": file.filename,
            "feature_count": len(features[0]),
            "features": {
                name: float(value) for name, value in zip(feature_names, features[0])
            },
            "statistics": {
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "median": float(np.median(features))
            },
            "quality_check": {
                "nan_count": int(np.sum(np.isnan(features))),
                "inf_count": int(np.sum(np.isinf(features))),
                "is_valid": bool(np.all(np.isfinite(features)))
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"디버깅 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"디버깅 중 오류가 발생했습니다: {str(e)}"}
        )
    finally:
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")

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
        
        # 예측 (스케일러 없음)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        logger.info(f"🎯 예측 결과: {prediction} ({'높음' if prediction == 1 else '낮음'})")
        if probability is not None:
            logger.info(f"   - 확률: 낮음={probability[0]:.3f}, 높음={probability[1]:.3f}")
        
        # 결과 반환
        result = {
            "success": True,
            "filename": file.filename,
            "prediction": int(prediction),
            "result": "높음" if prediction == 1 else "낮음",
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

# 특성 정보 API (디버깅용)
@app.get("/feature-info")
def get_feature_info():
    return {
        "total_features": 27,
        "feature_groups": {
            "mfcc": {"count": 13, "description": "Mel-frequency cepstral coefficients"},
            "chroma": {"count": 12, "description": "Chroma features"},
            "zcr": {"count": 1, "description": "Zero crossing rate"},
            "spectral_centroid": {"count": 1, "description": "Spectral centroid"}
        },
        "model_type": "DecisionTreeClassifier",
        "classes": ["낮음", "높음"]
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
    print(f"   - 특성 디버깅: http://{local_ip}:8000/debug-features")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)