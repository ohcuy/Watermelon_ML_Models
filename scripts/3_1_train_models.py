#!/usr/bin/env python3
"""
🍉 수박 음 높낮이 분류 모델 훈련 스크립트

전통적인 ML 모델(GBT, SVM, Random Forest)을 훈련하고 평가하는 메인 스크립트입니다.

Usage:
    python scripts/train_models.py [--config CONFIG_PATH] [--quick] [--no-viz]

Author: Watermelon ML Team
Date: 2025-01-15
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 프로젝트 루트 디렉토리를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import MLTrainer, create_trainer_from_config
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultVisualizer
from src.training.trainer import MLTrainer, create_trainer_from_config
from src.models.traditional_ml import ModelFactory

# 경고 메시지 억제
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'experiments' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        PROJECT_ROOT / 'experiments',
        PROJECT_ROOT / 'models' / 'saved',
        PROJECT_ROOT / 'experiments' / 'results',
        PROJECT_ROOT / 'experiments' / 'plots'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"디렉토리 설정 완료: {len(directories)}개")


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 파일 로드 완료: {config_path}")
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        raise


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """훈련/검증/테스트 데이터셋을 로드하고 라벨을 인코딩합니다."""
    try:
        base_path = PROJECT_ROOT / 'data' / 'splits' / 'full_dataset'
        
        train_df = pd.read_csv(base_path / 'train.csv')
        val_df = pd.read_csv(base_path / 'val.csv')
        test_df = pd.read_csv(base_path / 'test.csv')
        
        # 라벨 인코딩: 'low' -> 0, 'high' -> 1
        label_mapping = {'low': 0, 'high': 1}
        
        train_df['pitch_label'] = train_df['pitch_label'].map(label_mapping)
        val_df['pitch_label'] = val_df['pitch_label'].map(label_mapping)
        test_df['pitch_label'] = test_df['pitch_label'].map(label_mapping)
        
        logger.info(f"데이터셋 로드 완료:")
        logger.info(f"  - 훈련: {len(train_df)}개")
        logger.info(f"  - 검증: {len(val_df)}개") 
        logger.info(f"  - 테스트: {len(test_df)}개")
        
        # 음 높낮이 분포 확인 (인코딩 후)
        train_pitch_counts = train_df['pitch_label'].value_counts()
        logger.info(f"  - 훈련 세트 음 높낮이 분포: {dict(train_pitch_counts)} (0: 낮음, 1: 높음)")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise


def print_training_summary(config: Dict[str, Any]) -> None:
    """훈련 설정 요약을 출력합니다."""
    print("\n" + "="*80)
    print("🍉 수박 음 높낮이 분류 모델 훈련 시작")
    print("="*80)
    
    print(f"📅 훈련 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 성능 목표:")
    print(f"   - 정확도 > {config['performance']['target_accuracy']:.1%}")
    print(f"   - F1-score > {config['performance']['target_f1_score']:.2f}")
    
    print(f"🤖 훈련 모델:")
    for model_name in config['models'].keys():
        print(f"   - {model_name}")
    
    print(f"⚙️  훈련 설정:")
    print(f"   - 교차 검증: {config['cross_validation']['n_folds']}-fold")
    print(f"   - 스케일러: {config['preprocessing']['scaler_type']}")
    print(f"   - Random State: {config['global']['random_state']}")
    print("="*80 + "\n")


def evaluate_and_visualize(
    trainer: MLTrainer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """모델을 평가하고 시각화합니다."""
    
    logger.info("모델 평가 시작...")
    
    # 특징과 타겟 분리
    feature_cols = [col for col in train_df.columns if col != 'pitch_label']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['pitch_label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['pitch_label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['pitch_label'].values
    
    # 평가자 초기화
    evaluator = ModelEvaluator()
    all_results = {}
    
    # 각 모델 평가
    for model_name, model in trainer.models.items():
        # 모델이 훈련되었는지 확인
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            logger.warning(f"  {model_name} 모델이 훈련되지 않았습니다. 건너뜁니다.")
            continue
            
        logger.info(f"  {model_name} 평가 중...")
        
        try:
            # 예측
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"  {model_name} 예측 실패: {e}")
            continue
        
        # 훈련 세트 평가
        train_results = evaluator.evaluate_model_performance(
            y_train, y_train_pred, model_name, "훈련", "classification"
        )
        
        # 검증 세트 평가
        val_results = evaluator.evaluate_model_performance(
            y_val, y_val_pred, model_name, "검증", "classification"
        )
        
        # 테스트 세트 평가
        test_results = evaluator.evaluate_model_performance(
            y_test, y_test_pred, model_name, "테스트", "classification"
        )
        
        # 결과 저장
        all_results[model_name] = {
            'train': train_results,
            'val': val_results,
            'test': test_results,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred,
                'test': y_test_pred
            }
        }
    
    # 시각화 생성
    if create_visualizations:
        logger.info("시각화 생성 중...")
        visualizer = ResultVisualizer()
        
        # 성능 비교 차트
        performance_data = []
        for model_name, results in all_results.items():
            for dataset_type in ['train', 'val', 'test']:
                result = results[dataset_type]
                metrics = result['metrics'] if isinstance(result, dict) else result
                performance_data.append({
                    'Model': model_name,
                    'Dataset': dataset_type,
                    'Accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else getattr(metrics, 'accuracy', 0),
                    'F1_Score': metrics.get('f1_score', 0) if isinstance(metrics, dict) else getattr(metrics, 'f1_score', 0),
                    'Precision': metrics.get('precision', 0) if isinstance(metrics, dict) else getattr(metrics, 'precision', 0),
                    'Recall': metrics.get('recall', 0) if isinstance(metrics, dict) else getattr(metrics, 'recall', 0)
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        # 성능 데이터를 CSV로 저장 (시각화 대신)
        performance_df.to_csv(PROJECT_ROOT / 'experiments' / 'plots' / 'model_performance_comparison.csv', index=False)
        logger.info("성능 비교 데이터를 CSV로 저장했습니다.")
        
        # 각 모델별 예측 결과 저장
        for model_name, results in all_results.items():
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': results['predictions']['test']
            })
            predictions_df.to_csv(
                PROJECT_ROOT / 'experiments' / 'plots' / f'{model_name}_predictions.csv', 
                index=False
            )
        
        # 특징 중요도 저장 (Random Forest와 GBT만)
        for model_name, model in trainer.models.items():
            if hasattr(model.model, 'feature_importances_') and model.is_fitted:
                try:
                    importance_array = model.get_feature_importance()
                    if importance_array is not None:
                        # numpy 배열을 특징 이름과 함께 DataFrame으로 변환
                        feature_names = feature_cols  # 이미 정의된 특징 이름 사용
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance_array
                        }).sort_values('importance', ascending=False)
                        
                        importance_df.to_csv(
                            PROJECT_ROOT / 'experiments' / 'plots' / f'{model_name}_feature_importance.csv',
                            index=False
                        )
                        logger.info(f"{model_name} 특징 중요도를 CSV로 저장했습니다.")
                except Exception as e:
                    logger.warning(f"{model_name} 특징 중요도 저장 실패: {e}")
        
        logger.info(f"시각화 완료: experiments/plots/ 디렉토리")
    
    return all_results


def print_results_summary(all_results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """결과 요약을 출력하고 최고 성능 모델을 반환합니다."""
    
    print("\n" + "="*80)
    print("📊 모델 성능 요약")
    print("="*80)
    
    # 테스트 성능 요약 테이블
    print(f"{'모델':<20} {'정확도':<8} {'F1':<8} {'정밀도':<8} {'재현율':<8} {'목표달성':<10}")
    print("-" * 80)
    
    best_model = None
    best_f1 = 0.0
    target_accuracy = config['performance']['target_accuracy']
    target_f1 = config['performance']['target_f1_score']
    
    for model_name, results in all_results.items():
        test_result = results['test']
        metrics = test_result['metrics'] if isinstance(test_result, dict) else test_result
        accuracy = metrics.get('accuracy', 0) if isinstance(metrics, dict) else getattr(metrics, 'accuracy', 0)
        f1 = metrics.get('f1_score', 0) if isinstance(metrics, dict) else getattr(metrics, 'f1_score', 0)
        precision = metrics.get('precision', 0) if isinstance(metrics, dict) else getattr(metrics, 'precision', 0)
        recall = metrics.get('recall', 0) if isinstance(metrics, dict) else getattr(metrics, 'recall', 0)
        
        # 목표 달성 여부
        accuracy_ok = "✅" if accuracy > target_accuracy else "❌"
        f1_ok = "✅" if f1 > target_f1 else "❌"
        goal_status = f"{accuracy_ok} {f1_ok}"
        
        print(f"{model_name:<20} {accuracy:<8.3f} {f1:<8.3f} {precision:<8.3f} {recall:<8.3f} {goal_status:<10}")
        
        # 최고 성능 모델 찾기 (F1-score 기준)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_name
    
    print("-" * 80)
    print(f"🏆 최고 성능 모델: {best_model} (F1: {best_f1:.3f})")
    
    # 목표 달성 요약
    print(f"\n🎯 성능 목표 달성 현황:")
    models_meeting_accuracy = sum(1 for results in all_results.values() 
                                if (results['test']['metrics'].get('accuracy', 0) if isinstance(results['test'], dict) 
                                    else getattr(results['test'], 'accuracy', 0)) > target_accuracy)
    models_meeting_f1 = sum(1 for results in all_results.values() 
                          if (results['test']['metrics'].get('f1_score', 0) if isinstance(results['test'], dict) 
                              else getattr(results['test'], 'f1_score', 0)) > target_f1)
    total_models = len(all_results)
    
    print(f"   - 정확도 > {target_accuracy:.1%}: {models_meeting_accuracy}/{total_models} 모델")
    print(f"   - F1-score > {target_f1:.2f}: {models_meeting_f1}/{total_models} 모델")
    
    if best_f1 > target_f1:
        print(f"   ✅ 주요 목표 달성! (F1 > {target_f1:.2f})")
    else:
        print(f"   ❌ 주요 목표 미달성 (F1 <= {target_f1:.2f})")
    
    print("="*80 + "\n")
    
    return best_model or "Unknown"


def save_best_model(trainer: MLTrainer, best_model_name: str) -> None:
    """최고 성능 모델을 저장합니다."""
    
    try:
        best_model = trainer.models[best_model_name]
        
        # 모델 저장
        model_path = PROJECT_ROOT / 'models' / 'saved' / 'best_model.pkl'
        best_model.save_model(str(model_path))
        
        # 스케일러 저장 (있는 경우)
        if hasattr(best_model, 'scaler') and best_model.scaler is not None:
            scaler_path = PROJECT_ROOT / 'models' / 'saved' / 'scaler.pkl'
            import joblib
            joblib.dump(best_model.scaler, scaler_path)
            logger.info(f"스케일러 저장: {scaler_path}")
        
        # 모델 설정 저장
        config_path = PROJECT_ROOT / 'models' / 'saved' / 'model_config.yaml'
        model_config = {
            'best_model': best_model_name,
            'model_type': type(best_model).__name__,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(best_model.feature_names_) if hasattr(best_model, 'feature_names_') else 51,
            'performance': {
                'test_accuracy': float(trainer.latest_results[best_model_name]['test'].accuracy),
                'test_f1_score': float(trainer.latest_results[best_model_name]['test'].f1_score)
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"최고 성능 모델 저장 완료:")
        logger.info(f"  - 모델: {model_path}")
        logger.info(f"  - 설정: {config_path}")
        
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")


def save_training_results(
    all_results: Dict[str, Any],
    training_summary: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """훈련 결과를 파일로 저장합니다."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 성능 요약 CSV
        performance_data = []
        for model_name, results in all_results.items():
            for dataset_type in ['train', 'val', 'test']:
                result = results[dataset_type]
                metrics = result['metrics'] if isinstance(result, dict) else result
                performance_data.append({
                    'model': model_name,
                    'dataset': dataset_type,
                    'accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else getattr(metrics, 'accuracy', 0),
                    'f1_score': metrics.get('f1_score', 0) if isinstance(metrics, dict) else getattr(metrics, 'f1_score', 0),
                    'precision': metrics.get('precision', 0) if isinstance(metrics, dict) else getattr(metrics, 'precision', 0),
                    'recall': metrics.get('recall', 0) if isinstance(metrics, dict) else getattr(metrics, 'recall', 0),
                    'performance_grade': result.get('performance_grade', 'N/A') if isinstance(result, dict) else 'N/A'
                })
        
        performance_df = pd.DataFrame(performance_data)
        performance_path = PROJECT_ROOT / 'experiments' / 'results' / f'performance_summary_{timestamp}.csv'
        performance_df.to_csv(performance_path, index=False)
        
        # 상세 결과 YAML
        detailed_results = {
            'experiment_info': {
                'timestamp': timestamp,
                'config_used': config,
                'training_summary': training_summary
            },
            'model_results': {}
        }
        
        for model_name, results in all_results.items():
            detailed_results['model_results'][model_name] = {
                'train': results['train'] if isinstance(results['train'], dict) else results['train'].__dict__,
                'val': results['val'] if isinstance(results['val'], dict) else results['val'].__dict__,
                'test': results['test'] if isinstance(results['test'], dict) else results['test'].__dict__
            }
        
        detailed_path = PROJECT_ROOT / 'experiments' / 'results' / f'detailed_results_{timestamp}.yaml'
        with open(detailed_path, 'w', encoding='utf-8') as f:
            yaml.dump(detailed_results, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"훈련 결과 저장 완료:")
        logger.info(f"  - 성능 요약: {performance_path}")
        logger.info(f"  - 상세 결과: {detailed_path}")
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='수박 음 높낮이 분류 모델 훈련')
    parser.add_argument('--config', default='configs/models.yaml', help='설정 파일 경로')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 모드 (작은 하이퍼파라미터)')
    parser.add_argument('--no-viz', action='store_true', help='시각화 건너뛰기')
    
    args = parser.parse_args()
    
    try:
        # 초기 설정
        setup_directories()
        
        # 설정 로드
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        
        # 빠른 모드 설정 조정
        if args.quick:
            logger.info("빠른 테스트 모드 활성화")
            if 'gradient_boosting' in config['models']:
                config['models']['gradient_boosting']['n_estimators'] = 50
            if 'random_forest' in config['models']:
                config['models']['random_forest']['n_estimators'] = 50
            config['cross_validation']['n_folds'] = 3
        
        # 데이터 로드
        train_df, val_df, test_df = load_datasets()
        
        # 훈련 요약 출력
        print_training_summary(config)
        
        # 트레이너 생성 및 훈련
        logger.info("모델 훈련 시작...")
        trainer = create_trainer_from_config(str(config_path))
        
        # 특징과 타겟 분리
        feature_cols = [col for col in train_df.columns if col != 'pitch_label']
        X_train = train_df[feature_cols].values
        y_train = train_df['pitch_label'].values
        X_val = val_df[feature_cols].values  
        y_val = val_df['pitch_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['pitch_label'].values
        
        # 데이터 딕셔너리 생성
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # 훈련 실행
        training_results = trainer.train_all_models(data, feature_cols)
        
        logger.info("훈련 완료! 평가 시작...")
        
        # 평가 및 시각화
        all_results = evaluate_and_visualize(
            trainer, train_df, val_df, test_df, config, 
            create_visualizations=not args.no_viz
        )
        
        # 결과 요약 출력
        best_model_name = print_results_summary(all_results, config)
        
        # 최고 성능 모델 저장
        save_best_model(trainer, best_model_name)
        
        # 훈련 결과 저장
        training_summary = training_results.get_summary()
        save_training_results(all_results, training_summary, config)
        
        # === 🔗 호환성 요약 파일 생성 (후속 스크립트 연동용) ===
        logger.info("후속 스크립트 호환성을 위한 요약 파일 생성 중...")
        try:
            # 테스트 성능을 3_simple_train.py 형식으로 변환
            compatible_test_performance = {}
            for model_name, results in all_results.items():
                test_result = results['test']
                metrics = test_result['metrics'] if isinstance(test_result, dict) else test_result
                
                # numpy 객체를 float로 안전하게 변환
                accuracy = float(metrics.get('accuracy', 0)) if hasattr(metrics.get('accuracy', 0), 'item') else float(metrics.get('accuracy', 0))
                f1_score = float(metrics.get('f1_score', 0)) if hasattr(metrics.get('f1_score', 0), 'item') else float(metrics.get('f1_score', 0))
                precision = float(metrics.get('precision', 0)) if hasattr(metrics.get('precision', 0), 'item') else float(metrics.get('precision', 0))
                
                compatible_test_performance[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'precision': precision
                }
            
            # 3_simple_train.py 호환 형식으로 저장
            compatible_summary = {
                'test_performance': compatible_test_performance,
                'timestamp': datetime.now().isoformat(),
                'source': '3_1_train_models.py'  # 출처 명시
            }
            
            # models/saved/training_summary_simple.yaml 생성
            saved_dir = PROJECT_ROOT / "models" / "saved"
            saved_dir.mkdir(parents=True, exist_ok=True)
            
            compatible_file = saved_dir / "training_summary_simple.yaml"
            with open(compatible_file, 'w', encoding='utf-8') as f:
                yaml.dump(compatible_summary, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ 호환성 요약 파일 생성 완료: {compatible_file}")
            
        except Exception as e:
            logger.warning(f"호환성 요약 파일 생성 실패: {e}")
        
        # 완료 메시지
        print("\n🎉 모델 훈련 및 평가 완료!")
        print(f"📁 결과 파일: experiments/results/")
        print(f"📊 시각화: experiments/plots/")
        print(f"🏆 최고 모델: models/saved/")
        
        return 0
        
    except Exception as e:
        logger.error(f"훈련 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 메모리 정리
        import gc
        gc.collect()
        logger.info("메모리 정리 완료")


if __name__ == "__main__":
    sys.exit(main()) 