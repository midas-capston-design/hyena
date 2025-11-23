# Hyena Indoor Positioning System

지자기 센서 데이터를 이용한 실내 측위 시스템 (Hyena Architecture)

## 🎯 프로젝트 개요

**Midas Capstone Design Project**

Hyena 아키텍처 기반 딥러닝 모델을 사용하여 지자기 센서 데이터(MagX, MagY, MagZ)로부터 실내 위치를 추정하는 시스템입니다.

### 핵심 성과 ✅

| 지표 | 목표 | 달성 | 상태 |
|------|------|------|------|
| **P90** | < 2m | **1.801m** | ✅ |
| **MAE** | < 1.4m | **1.008m** | ✅ |
| **Median** | < 1m | **0.678m** | ✅ |
| **RMSE** | < 1.8m | 2.019m | ⚠️ |

- 🎯 **90%의 예측이 1.8m 이내 오차**
- 📊 **평균 오차 1m 이내**
- 🏆 **중앙값 0.7m** (절반이 0.7m 이내)

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.13 권장
bash scripts/setup.sh
source venv/bin/activate
```

### 2. 데이터 전처리

```bash
python src/preprocess_sliding.py \
  --raw-dir data/raw \
  --nodes data/nodes_final.csv \
  --output data/sliding_mag4 \
  --feature-mode mag4
```

### 3. 모델 학습

```bash
python src/train_sliding.py \
  --data-dir data/sliding_mag4 \
  --epochs 200 \
  --batch-size 64 \
  --hidden-dim 384 \
  --depth 10
```

### 4. 테스트

```bash
# 저장된 체크포인트로 테스트
python src/test_only.py \
  --checkpoint checkpoints_sliding_mag4/best.pt \
  --data-dir data/sliding_mag4
```

## 🧠 핵심 기술

### Hyena Architecture
- **O(n log n) 복잡도**: Transformer의 O(n²)보다 효율적
- **FFT 기반 Long Convolution**: 긴 시퀀스 패턴 학습
- **250 타임스텝** 전체 경로 분석

### 최신 학습 기법 (2025)
- ✅ Mixed Precision Training (AMP)
- ✅ 5-Epoch Moving Average Adaptive LR (양방향 조절)
- ✅ Learning Rate Warmup
- ✅ Early Stopping (P90 기준)
- ✅ Gradient Clipping
- ✅ Wavelet Denoising

### 데이터 처리
- **Sliding Window**: 250 steps, stride 50
- **Graph-based Path Finding**: BFS 최단 경로 탐색
- **Turn Node Interpolation**: 회전 노드 기반 세그먼트 보간
- **Adaptive Normalization**: Z-score per file (캘리브레이션 drift 대응)

## 📁 프로젝트 구조

```
lstm/
├── src/                      # 소스 코드
│   ├── model.py             # Hyena 모델
│   ├── preprocess_sliding.py # 전처리
│   ├── train_sliding.py     # 학습
│   └── test_only.py         # 테스트
│
├── data/                     # 데이터 (Git LFS)
│   ├── raw/                 # 원본 CSV (404개)
│   ├── sliding_mag4/        # 전처리 데이터
│   └── nodes_final.csv      # 노드 정보
│
├── checkpoints_*/           # 모델 체크포인트 (Git LFS)
│   └── best.pt
│
├── analysis/                # 분석 도구
├── map/                     # 맵 시각화
├── scripts/                 # 실행 스크립트
│
├── DOCUMENTATION.md         # 📘 상세 기술 문서
├── README.md                # 이 파일
└── requirements.txt         # 의존성
```

## 📊 데이터셋

- **원본 데이터**: 404개 CSV 파일 (87개 경로 × 4-5개 샘플)
- **전처리 후**: 13,611개 샘플 (Sliding Window)
- **분할**: Train 60%, Val 20%, Test 20%
- **센서**: MagX, MagY, MagZ, Magnitude
- **노드**: 30개 (회전 노드 6개: 4, 10, 11, 20, 27, 28)

## 🔧 Git & Git LFS 설정

이 프로젝트는 대용량 파일 관리를 위해 **Git LFS**를 사용합니다.

### LFS로 관리되는 파일
- `*.pt`, `*.pth`: 모델 체크포인트 (~80MB/파일)
- `*.csv`: 센서 데이터 (404개 파일)
- `*.jsonl`: 전처리 데이터 (~286MB)
- `data/**`: 모든 데이터 디렉토리

### 저장소 클론

```bash
# 1. Git LFS 설치 (처음 한 번만)
brew install git-lfs        # macOS
# 또는
apt-get install git-lfs    # Ubuntu

# 2. LFS 초기화
git lfs install

# 3. 저장소 클론 (LFS 파일 자동 다운로드)
git clone git@github.com:midas-capston-design/hyena.git
cd hyena
```

### LFS 파일 확인

```bash
# LFS로 관리되는 파일 확인
git lfs ls-files

# LFS 상태 확인
git lfs status
```

### 주의사항

- ⚠️ **LFS 없이 클론하면**: 대용량 파일이 포인터 파일로만 다운로드됨 (사용 불가)
- ✅ **LFS 설치 후 클론**: 모든 파일이 정상적으로 다운로드됨
- 📦 **저장소 크기**: ~800MB (LFS 파일 포함)

## 📖 상세 문서

더 자세한 내용은 [DOCUMENTATION.md](DOCUMENTATION.md)를 참고하세요:

- 🎯 기술 선택 이유 및 흐름
- 🧠 Hyena 아키텍처 상세 설명
- 🔄 데이터 전처리 파이프라인
- 🎓 학습 기법 (Warmup, Adaptive LR, Early Stopping 등)
- 📊 평가 메트릭 및 목표
- 🐛 트러블슈팅
- 📝 체크포인트 재평가 방법

## 🤝 기여

Midas Capstone Design Team

## 📄 라이선스

MIT License

---

**Last Updated**: 2025-11-24
**Version**: 1.0
**Best Model**: P90=1.801m (checkpoints_sliding_mag4/best.pt)
**Repository**: https://github.com/midas-capston-design/hyena
