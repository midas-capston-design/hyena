# LSTM 프로젝트 기술 문서

## 📋 프로젝트 개요

지자기 센서 데이터를 이용한 실내 측위(Indoor Positioning) 시스템으로, Hyena 아키텍처 기반의 딥러닝 모델을 사용하여 사용자의 위치를 추정합니다.

### 핵심 목표
- **입력**: 지자기 센서 데이터 (MagX, MagY, MagZ) + 선택적으로 추가 가능한 데이터(Geomagnetic field magnitude, yaw, pitch, roll)
- **출력**: 2D 좌표 (x, y)
- **방식**: Sliding Window + Causal Hyena Model
- **성능 지표**: RMSE, MAE, Median Error, P90 Error

---

## 🎯 핵심 기술 선택 이유 및 흐름

### 기술 스택 의존성 흐름도

```
문제: 실내 측위 (가까운 위치를 구분하기 위해선 경로가 긴 시계열 데이터가 필수)
    ↓
[1] Hyena 아키텍처 선택
    ↓ (긴 시퀀스 처리 → 데이터 부족)
[2] Sliding Window 방식
    ↓ (많은 샘플 → 학습 시간 증가)
[3] Mixed Precision (AMP)
    ↓ (큰 모델 → 초기 불안정)
[4] Learning Rate Warmup
    ↓ (과적합 위험)
[5] ReduceLROnPlateau + Early Stopping + Dropout
    ↓ (재현성 필요)
[6] Random Seed 고정
    ↓
완성된 파이프라인
```

---

### 1️⃣ Hyena 아키텍처

#### 🔍 무엇인가?
**Liquid AI의 차세대 시퀀스 모델**
- Transformer의 Self-Attention 대신 **FFT 기반 Long Convolution** 사용
- Implicit Filter로 긴 필터를 작은 MLP로 생성
- O(n log n) 복잡도 (Transformer는 O(n²))

#### 💡 왜 사용했나?
1. **긴 경로 패턴 학습 필요**
   - 실내 측위는 전체 trajectory의 "지문(fingerprint)" 필요
   - 단순 지자기 값이 아닌 **전체 경로 패턴**으로 위치 파악
   - 250 타임스텝 전체를 한번에 봐야 함

2. **Transformer 대비 효율성**
   ```
   Transformer Self-Attention: O(250²) = 62,500 연산
   Hyena FFT Conv: O(250 log 250) = 1,993 연산
   → 약 30배 효율적
   ```

3. **장거리 의존성 (Long-range Dependency)**
   - 시작점의 지자기 → 끝점 위치 예측에 영향
   - Hyena의 Long Conv가 전체 시퀀스를 하나의 필터로 처리

#### 🔗 연결
**Hyena 사용 → 긴 시퀀스 처리 가능 → 하지만 데이터 부족 문제 발생**

---

### 2️⃣ Sliding Window 방식

#### 🔍 무엇인가?
고정 크기 윈도우(250)를 일정 간격(50)으로 슬라이딩하며 샘플 생성
```python
원본 데이터: 1000 타임스텝
Sliding Window (size=250, stride=50):
  [0:250]    → Sample 1
  [50:300]   → Sample 2
  [100:350]  → Sample 3
  ...
  → 16개 샘플 생성
```

#### 💡 왜 사용했나?
1. **샘플 생성 (Sample Generation)**
   - 원본: 474개 파일
   - Sliding 후: **13,611개 샘플**
   - 약 28배 샘플 생성 → 딥러닝 학습에 충분한 데이터

2. **다양한 시작점 학습**
   - 같은 경로라도 시작 위치가 다르면 다른 샘플
   - 모델이 **어떤 위치에서 시작해도** 대응 가능

3. **Overlap으로 부드러운 전환**
   - Stride=50 → 80% 겹침
   - 급격한 변화 없이 연속적인 패턴 학습

#### 🔗 연결
**샘플 생성 → 샘플 많아짐 → 학습 시간 증가 → 속도 최적화 필요**

---

### 3️⃣ Mixed Precision Training (AMP)

#### 🔍 무엇인가?
**FP16(반정밀도)와 FP32(단정밀도)를 혼합 사용**
```python
# 일반 학습: 모든 연산 FP32
loss = model(x)  # FP32

# Mixed Precision: 대부분 FP16, 필요시만 FP32
with autocast():
    loss = model(x)  # FP16 (빠름)
    # 손실 등 중요한 값만 FP32
```

#### 💡 왜 사용했나?
1. **학습 속도 2-3배 향상**
   - FP16 연산이 FP32보다 빠름
   - GPU/MPS 가속 효율 극대화
   - 배치당 1.99초 → 0.5초 예상

2. **메모리 사용량 50% 절감**
   - FP16은 FP32의 절반 메모리
   - 배치 크기를 2배로 늘릴 수 있음
   - 32 → 64 batch size

3. **정확도 손실 최소**
   - Loss scaling으로 언더플로우 방지
   - 중요한 연산은 FP32 유지

#### 🔗 연결
**빠른 학습 → 큰 모델 사용 가능 → 하지만 초기 학습 불안정**

---

### 4️⃣ Learning Rate Warmup

#### 🔍 무엇인가?
**초기 몇 에포크 동안 학습률을 점진적으로 증가**
```python
Epoch 1: LR = 0.00003  (10%)
Epoch 2: LR = 0.00012  (40%)
Epoch 3: LR = 0.00021  (70%)
Epoch 4: LR = 0.00027  (90%)
Epoch 5: LR = 0.0003   (100%)
Epoch 6+: LR = 0.0003  (유지)
```

#### 💡 왜 사용했나?
1. **큰 모델의 안정적 초기화**
   - Hidden=384, Depth=10 → 5M 파라미터
   - 초기에 큰 LR → Gradient explosion 위험
   - Warmup으로 안정적 시작

2. **Adam Optimizer와 궁합**
   - Adam은 초기 moment 추정 필요
   - Warmup으로 추정 안정화

3. **실험적 검증**
   - BERT, GPT 등 대형 모델에서 필수
   - 2025년 표준 관행

#### 🔗 연결
**안정적 학습 → 하지만 과적합 가능성 → 적응적 LR 조절 필요**

---

### 5️⃣ 5-Epoch Moving Average Adaptive LR (적응적 학습률)

#### 🔍 무엇인가?
**5-epoch 이동평균 기반 양방향 학습률 조절**
```python
# 최근 5 에포크 Val RMSE 이동평균 계산
rmse_history = [2.5, 2.3, 2.2, 2.1, 2.0]  # 최근 5개
current_avg = mean(rmse_history) = 2.22m
previous_avg = 2.30m

# 개선되면 LR 증가, 악화되면 감소
if current_avg < previous_avg:  # 개선
    LR = LR × 1.05  # 5% 증가
else:  # 악화
    LR = LR × 0.95  # 5% 감소

# 최소/최대 제한
LR = clamp(LR, min_lr=1e-6, max_lr=initial_lr)
```

#### 💡 왜 사용했나?
1. **양방향 조절로 과적합/과소적합 동시 대응**
   - 개선 시 LR 증가 → 빠른 수렴
   - 악화 시 LR 감소 → 세밀한 탐색
   - **ReduceLROnPlateau는 감소만 가능** (비효율)

2. **5-epoch 이동평균으로 노이즈 필터링**
   - Validation은 에포크마다 변동 (노이즈)
   - 5개 평균으로 진짜 추세 파악
   - 안정적인 LR 조절 가능

3. **사람 개입 불필요**
   - 자동으로 최적 학습률 탐색
   - 과적합 자동 감지 및 대응
   - 하이퍼파라미터 튜닝 부담 감소

#### 🔗 연결
**적응적 LR → 하지만 완전 중단 기준도 필요 → Early Stopping**

---

### 6️⃣ Early Stopping

#### 🔍 무엇인가?
**일정 기간 개선 없으면 학습 조기 종료**
```python
Patience = 15

Best Val RMSE = 3.0m (Epoch 30)
Epoch 31-45: Val RMSE > 3.0m (15 에포크 개선 없음)
→ 학습 중단 (50 에포크 예정이었지만 45에 종료)
```

#### 💡 왜 사용했나?
1. **과적합 방지 (최종 안전장치)**
   - ReduceLROnPlateau로도 개선 없으면 → 더 학습해도 무의미
   - 시간 낭비 방지

2. **최적 모델 자동 선택**
   - Best checkpoint 자동 저장
   - 과적합 전 최고 성능 모델 확보

3. **계산 자원 절약**
   - 100 에포크 설정해도 보통 30-50에 종료
   - 불필요한 학습 시간 절감

#### 🔗 연결
**Early Stopping → 하지만 실험 재현 필요 → Random Seed 고정**

---

### 7️⃣ Random Seed 고정

#### 🔍 무엇인가?
**모든 난수 생성기를 고정값으로 초기화**
```python
set_seed(42)
# PyTorch, NumPy, Random 모두 고정
# 매번 같은 순서로 데이터 셔플, 가중치 초기화
```

#### 💡 왜 사용했나?
1. **완벽한 재현성 (Reproducibility)**
   ```
   시드 없음:
   실험1 (LR=0.0001): RMSE=3.2m
   실험2 (LR=0.0003): RMSE=3.5m
   → LR 차이? 운? 알 수 없음

   시드 있음:
   실험1 (seed=42, LR=0.0001): RMSE=3.2m
   실험2 (seed=42, LR=0.0003): RMSE=3.1m
   → 확실히 LR=0.0003이 좋음!
   ```

2. **공정한 하이퍼파라미터 비교**
   - Hidden=256 vs 384
   - Depth=8 vs 10
   - 동일 조건에서 비교 가능

3. **디버깅 용이**
   - 에러 발생 → 같은 시드로 재현
   - 원인 파악 쉬움

#### 🔗 연결
**재현성 확보 → 신뢰할 수 있는 실험 → 논문/발표 가능**

---

### 8️⃣ Wavelet Denoising

#### 🔍 무엇인가?
**Wavelet 변환으로 노이즈 제거**
```python
원본 신호 → Wavelet 분해 (3 레벨)
         → 고주파 성분(노이즈) 제거 (Soft thresholding)
         → Wavelet 재구성
         → 깨끗한 신호
```

#### 💡 왜 사용했나?
1. **센서 노이즈 특성**
   - 지자기 센서는 고주파 노이즈 많음
   - 사람의 움직임, 주변 금속 영향

2. **신호 보존하며 노이즈 제거**
   - Low-pass filter는 신호도 뭉개짐
   - Wavelet은 **중요한 특징은 보존**

3. **표준 신호처리 기법**
   - 의료, 금융, 센서 데이터에서 검증됨
   - db4 wavelet이 센서 데이터에 적합

#### 🔗 연결
**깨끗한 신호 → 모델 학습 품질 향상**

---

### 9️⃣ Dropout & Weight Decay

#### 🔍 무엇인가?
**정규화 (Regularization) 기법**
```python
Dropout(0.12): 학습 시 12% 뉴런 랜덤 제거
Weight Decay(0.01): 가중치 크기에 페널티
```

#### 💡 왜 사용했나?
1. **과적합 방지 (근본 대책)**
   - 모델이 훈련 데이터만 외우는 것 방지
   - 일반화 능력 향상

2. **앙상블 효과 (Dropout)**
   - 매 배치마다 다른 서브네트워크 학습
   - 추론 시 모든 뉴런 사용 = 암묵적 앙상블

3. **가중치 폭발 방지 (Weight Decay)**
   - 큰 가중치 → 과적합 신호
   - L2 정규화로 억제

#### 🔗 연결
**정규화 → ReduceLROnPlateau, Early Stopping과 함께 과적합 3중 방어**

---

### 🔟 Gradient Clipping

#### 🔍 무엇인가?
**Gradient 크기 제한**
```python
clip_grad_norm_(model.parameters(), 1.0)
# Gradient norm > 1.0이면 1.0으로 스케일링
```

#### 💡 왜 사용했나?
1. **Gradient Explosion 방지**
   - 깊은 네트워크 (Depth=10)
   - 긴 시퀀스 (250 타임스텝)
   - Gradient가 폭발적으로 커질 수 있음

2. **학습 안정성**
   - 이상한 배치 하나가 모델 망치는 것 방지
   - 꾸준한 수렴

3. **RNN/Sequence 모델 표준**
   - LSTM, Transformer, Hyena 등에서 필수

---

## 🔄 전체 기술 통합 흐름

```
[데이터 수집]
    ↓
[Wavelet Denoising] ← 센서 노이즈 제거
    ↓
[Sliding Window] ← 샘플 생성 (13,611 샘플)
    ↓
[Hyena 모델] ← 긴 시퀀스 패턴 학습 (O(n log n))
    ↓
[Mixed Precision] ← 학습 속도 2-3배 (메모리 절반)
    ↓
[Warmup (5 ep)] ← 안정적 초기화
    ↓
[Training Loop]
  ├─ Gradient Clipping ← 안정성
  ├─ Dropout ← 과적합 방지
  └─ Weight Decay ← 정규화
    ↓
[5-Epoch MA Adaptive LR] ← 양방향 LR 조절 (개선 시 증가, 악화 시 감소)
    ↓
[Early Stopping] ← 최적 모델 저장 (P90 기준)
    ↓
[Best Model] (MAE=0.948m, P90=1.660m, RMSE=2.202m, Median=0.552m)
```

---

## 📊 기술 선택 비교표

| 기술 | 대안 | 선택 이유 |
|------|------|-----------|
| **Hyena** | Transformer | O(n log n) vs O(n²), 긴 시퀀스 효율 |
| **Sliding Window** | Full Sequence | 샘플 생성 (28배), 다양한 시작점 |
| **AMP** | FP32 Only | 속도 2-3배, 메모리 50% 절감 |
| **Warmup** | Fixed LR | 큰 모델 안정화, BERT/GPT 검증됨 |
| **5-Epoch MA LR** | ReduceLROnPlateau | 양방향 조절, 노이즈 필터링, 빠른 수렴 |
| **Early Stopping** | Full Epochs | 시간 절약, 과적합 방지 |
| **AdamW** | Adam/SGD | Weight decay 분리, 2025 표준 |
| **Wavelet** | Gaussian Filter | 신호 보존, 주파수 선택적 제거 |
| **Dropout** | None | 앙상블 효과, 과적합 방지 |

---

## 🏗️ 프로젝트 구조

```
lstm/
├── src/                      # 핵심 소스 코드
│   ├── model.py             # Hyena 모델 정의
│   ├── preprocess_sliding.py # 회전 노드 기반 전처리
│   └── train_sliding.py     # P90 기준 학습 파이프라인
│
├── data/                     # 데이터 저장소
│   ├── raw/                 # 원본 센서 데이터 (404개 CSV)
│   ├── sliding_mag4/        # 전처리된 학습 데이터 (mag4 + adaptive)
│   └── nodes_final.csv      # 노드 위치 + 회전 노드(4,10,11,20,27,28)
│
├── scripts/                  # 실행 스크립트
│   ├── setup.sh             # 환경 설정
│   ├── run_all_sliding.sh   # mag3/mag4/full 비교
│   └── filter_*.py          # 데이터 필터링
│
├── analysis/                 # 데이터 분석 도구
│   ├── outputs/             # 분석 결과물
│   ├── analyze_*.py         # 다양한 분석 스크립트
│   └── visualize_features.py # 센서 데이터 시각화
│
├── map/                      # 맵 시각화
│   ├── draw_node_map.py     # 노드 그래프 시각화
│   └── test_node_graph.py   # 경로 테스트
│
├── checkpoints_sliding_mag3/ # mag3 실험 모델 (백업용)
│   └── best.pt
│
├── checkpoints_sliding_mag4/ # 최고 성능 모델 ⭐
│   └── best.pt              # MAE=0.948m, P90=1.660m 달성
│
└── run_sliding_window.sh     # 메인 실행 스크립트
```

---

## 🧠 모델 아키텍처 (src/model.py)

### Hyena Operator
Liquid AI의 Hyena 아키텍처를 기반으로 한 장거리 의존성 학습 모델

#### 1. **PositionalEncoding**
```python
class PositionalEncoding(nn.Module)
```
- Sinusoidal 방식의 위치 인코딩
- 시퀀스의 각 타임스텝에 위치 정보 부여
- Transformer의 positional encoding과 동일한 방식

#### 2. **ImplicitFilter**
```python
class ImplicitFilter(nn.Module)
```
- **Hyena의 핵심 구성 요소**
- 작은 MLP로 긴 필터를 암묵적으로 생성
- 파라미터 효율적으로 장거리 의존성 처리
- 구조: Linear(1→64) → GELU → Linear(64→64) → GELU → Linear(64→dim)

#### 3. **HyenaOperator**
```python
class HyenaOperator(nn.Module)
```
**핵심 연산:**
1. **Multiple Path Projection**: 입력을 여러 경로로 분리 (v, u, z)
2. **Implicit Long Filter**: 긴 시퀀스 패턴 학습용 필터 생성
3. **Short Convolution**: 지역적 패턴 추출 (kernel_size=3, depthwise)
4. **FFT Long Convolution**:
   - FFT로 주파수 도메인 변환
   - 필터와 곱셈 (O(n log n) 복잡도)
   - Inverse FFT로 복원
5. **Multiple Gating**: v * filtered * sigmoid(z)

**장점:**
- O(n log n) 복잡도로 긴 시퀀스 처리
- Transformer의 O(n²)보다 효율적
- 전체 경로 패턴(trajectory fingerprint) 학습

#### 4. **HyenaBlock**
```python
class HyenaBlock(nn.Module)
```
- Layer Normalization
- HyenaOperator
- Dropout
- Residual Connection

#### 5. **HyenaPositioning** (최종 모델)
```python
class HyenaPositioning(nn.Module)
```

**구조:**
```
Input (batch, 250, 3)
  ↓
Input Projection → (batch, 250, 384)
  ↓
+ Positional Encoding
+ Edge Embedding (경로 방향성)
  ↓
HyenaBlock × 10 layers
  ↓
Layer Norm
  ↓
Output Head (Linear → GELU → Dropout → Linear)
  ↓
Output (batch, 250, 2)  # 각 타임스텝의 (x, y) 좌표
```

**파라미터:**
- `input_dim=3`: MagX, MagY, MagZ
- `hidden_dim=384`: 은닉층 차원
- `output_dim=2`: (x, y) 좌표
- `depth=10`: Hyena 블록 개수
- `order=2`: Gating paths 수
- `dropout=0.12`: 정규화

**현재 설정 (2025 최신):**
- 파라미터 수: ~5M
- Mixed Precision (AMP) 지원
- MPS/CUDA 가속

---

## 🔄 데이터 전처리 (src/preprocess_sliding.py)

### Sliding Window 방식

#### 1. **데이터 정규화**
```python
# 지자기 정규화
BASE_MAG = (-33.0, -15.0, -42.0)
normalize_mag(val, base) = (val - base) / 10.0

# 좌표 정규화
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0
normalize_coord(x, y) = ((x - center) / scale, (y - center) / scale)
```

#### 2. **Wavelet Denoising**
```python
wavelet_denoise(signal, wavelet='db4', level=3)
```
- Daubechies 4 wavelet 사용
- 3 레벨 분해
- Soft thresholding으로 노이즈 제거
- 지자기 신호의 고주파 노이즈 필터링

#### 3. **Sliding Window 생성**
```python
window_size = 250  # 윈도우 크기
stride = 50        # 슬라이딩 간격
```

**프로세스:**
1. CSV 파일 읽기 (파일명: `{from}_{to}_{count}.csv`)
2. 센서 데이터 추출 (MagX, MagY, MagZ)
3. Wavelet denoising 적용
4. **그래프 기반 경로 탐색 및 회전 노드 보간**
   ```python
   # 복도 구조 그래프 (5m 이내, 같은 행/열만 연결)
   graph = build_graph(positions)
   # 차단된 연결: (10,28), (24,25)

   # 최단 경로 찾기 (BFS)
   path = find_shortest_path(graph, start_node, end_node)

   # 회전 노드만 추출 (4, 10, 11, 20, 27, 28)
   waypoints = get_turn_waypoints(path, turn_nodes)
   # 예: [1, 4, 23] (회전 노드 4만 포함)

   # 세그먼트별 선형 보간
   # 세그먼트 1: 1→4 (13.5m)
   # 세그먼트 2: 4→23 (9.0m)
   ```
5. Sliding window로 샘플 생성
   - 각 window: `[250 타임스텝, n features]`
   - Target: 마지막 타임스텝의 정규화된 좌표 `(x_norm, y_norm)`

#### 4. **Feature 모드**
- **mag3**: MagX, MagY, MagZ (기본)
- **mag4**: mag3 + 지자기 크기(magnitude)
- **full**: mag3 + Pitch, Roll, Yaw

#### 5. **데이터 분할**
- Train: 60%
- Validation: 20%
- Test: 20%
- Random shuffle 적용

**출력 형식 (JSONL):**
```json
{
  "features": [[mag_x, mag_y, mag_z], ...],  // 250개
  "target": [x_norm, y_norm]
}
```

---

## 🎓 학습 파이프라인 (src/train_sliding.py)

### 2025년 최신 학습 기법 적용

#### 1. **재현성 (Reproducibility)**
```python
set_seed(seed=42)
```
- Random, NumPy, PyTorch 시드 고정
- Deterministic 모드 활성화
- 완전한 결과 재현 보장

#### 2. **Mixed Precision Training (AMP)**
```python
torch.cuda.amp.autocast()
GradScaler()
```
- **FP16/FP32 혼합 정밀도**
- 학습 속도 2-3배 향상
- 메모리 사용량 50% 절감
- MPS/CUDA 자동 감지

#### 3. **Learning Rate Warmup**
```python
warmup_epochs = 5
warmup_lr = base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
```
- 초기 5 에포크 동안 학습률 점진 증가
- 10% → 100% (기본 LR)
- 큰 모델의 안정적 학습

#### 4. **5-Epoch Moving Average Adaptive LR** ⭐ 새로운 기법
```python
# 최근 5 에포크 RMSE 이동평균 기반 양방향 조절
from collections import deque
import numpy as np

rmse_history = deque(maxlen=5)
base_lr = 3e-4
min_lr = 1e-6

# 매 에포크마다 실행 (warmup 이후)
if epoch > warmup_epochs:
    rmse_history.append(val_rmse)

    if len(rmse_history) >= 5:
        current_avg = np.mean(list(rmse_history))

        # 5 에포크 전 평균과 비교
        if hasattr(self, 'previous_avg'):
            if current_avg < previous_avg:
                lr = lr * 1.05  # 개선 → 5% 증가
            else:
                lr = lr * 0.95  # 악화 → 5% 감소

            # 경계값 제한
            lr = np.clip(lr, min_lr, base_lr)

            # Optimizer에 적용
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        self.previous_avg = current_avg
```

**특징:**
- **양방향 조절**: 개선 시 증가(×1.05), 악화 시 감소(×0.95)
- **노이즈 필터링**: 5-epoch 이동평균으로 validation 노이즈 제거
- **과적합/과소적합 동시 대응**: 상황에 따라 자동 조절
- **ReduceLROnPlateau 대비**: 증가도 가능, 빠른 수렴

**실제 효과 (Epoch 1-110):**
- Warmup: Epoch 1-5 (LR 점진 증가)
- 빠른 수렴: Epoch 6-95 (Best P90=1.660m)
- 안정화: Epoch 96-110 (Early Stopping)

#### 5. **Early Stopping (P90 기준)**
```python
patience = 15
```
- **Validation P90 개선 모니터링** (outlier에 강건)
- 15 에포크 개선 없으면 학습 중단
- Best model: P90 기준으로 저장 (실용적 성능)

#### 6. **정규화 (Regularization)**
- **Dropout**: 0.12
- **Weight Decay**: 0.01 (AdamW)
- **Gradient Clipping**: 1.0

#### 7. **Optimizer**
```python
AdamW(lr=3e-4, weight_decay=0.01)
```
- AdamW optimizer (2025 표준)
- Decoupled weight decay

### 학습 프로세스

```python
for epoch in range(1, epochs + 1):
    # 1. Warmup (처음 5 에포크)
    if epoch <= 5:
        adjust_learning_rate(optimizer, epoch)

    # 2. Training
    for features, targets in train_loader:
        with autocast():  # Mixed Precision
            outputs = model(features, edge_ids)
            loss = criterion(outputs[:, -1, :], targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # 3. Validation
    val_rmse, val_p90 = validate(model, val_loader)

    # 4. Learning Rate Scheduling (RMSE 기준 - 부드러운 학습)
    if epoch > 5:
        scheduler.step(val_rmse)

    # 5. Checkpointing (P90 기준 - 실용적 선택)
    if val_p90 < best_val_p90:
        save_checkpoint(model, optimizer, epoch, val_p90)

    # 6. Early Stopping (P90 기준)
    if no_improvement >= 15:
        break
```

### 평가 지표

#### 역정규화
```python
def denormalize_coord(x_norm, y_norm):
    x = x_norm * 50.0 + (-41.0)
    y = y_norm * 50.0 + 0.0
    return (x, y)
```

#### 거리 오차 계산
```python
distance = sqrt((pred_x - target_x)² + (pred_y - target_y)²)
```

#### 지표 및 사용 전략

**모니터링 지표:**
- **RMSE** (Root Mean Square Error): 평균 제곱근 오차
- **MAE** (Mean Absolute Error): 평균 절대 오차
- **Median Error**: 중앙값 오차
- **P90 Error**: 90번째 백분위수 오차

**Hybrid 전략:**
- **LR Scheduler**: RMSE 기준 (부드러운 지표, 학습 동역학 최적)
- **Best Model 선택**: P90 기준 (outlier에 강건, 실용적 성능)
- **이유**: RMSE로 안정적 학습, P90로 robust한 모델 선택

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
bash scripts/setup.sh
```

### 2. 전체 파이프라인 실행
```bash
./run_sliding_window.sh
```

**실행 단계:**
1. **전처리**: Sliding window 생성
2. **학습**: Hyena 모델 학습
3. **평가**: Test set 평가

### 3. 하이퍼파라미터 설정

`run_sliding_window.sh` 편집:
```bash
EPOCHS=100         # 학습 반복 횟수
BATCH_SIZE=64      # 배치 크기
HIDDEN_DIM=384     # 은닉층 차원
DEPTH=10           # Hyena 레이어 수
PATIENCE=15        # Early stopping
```

### 4. 학습 모니터링

**실시간 출력:**
```
🎲 Random seed: 42
🍎 Apple Silicon GPU (MPS) 사용
⚡ Mixed Precision (AMP) 활성화
🔥 Warmup 시작: 5 에포크 동안 LR 4.50e-05 → 3.00e-04

Epoch 1/100 [Train]: 100%|████| 127/127 [00:15<00:00, loss: 0.0234, dist: 3.45m]
Epoch 1/100 [Val]  : 100%|████| 42/42 [00:02<00:00]

[Epoch 001] TrainLoss=0.0245 TrainRMSE=3.521m | ValRMSE=3.234m MAE=2.891m Median=2.456m P90=5.123m
   💾 Best model saved (RMSE=3.234m)
```

---

## 📊 데이터 분석 도구 (analysis/)

### 주요 스크립트

#### 1. **analyze_file_quality.py**
- 파일별 품질 점수 계산
- 길이, 안정성, 노이즈 체크
- 좋은 파일/나쁜 파일 분류
- 출력: `analysis/outputs/good_bad_files.txt`

**실행:**
```bash
python analysis/analyze_file_quality.py
```

#### 2. **visualize_features.py**
- 센서 데이터 시각화
- 6개 subplot: MagX/Y/Z, Pitch/Roll/Yaw, 지자기 평면, 크기
- 출력: `analysis/outputs/feature_analysis_*.png`

**실행:**
```bash
python analysis/visualize_features.py
```

#### 3. **analyze_calibration_cause.py**
- 캘리브레이션 차이 분석
- Bad vs Raw 데이터 비교

#### 4. **move_good_bad_to_raw.py**
- 품질 좋은 Bad 파일을 Raw로 이동
- 캘리브레이션 보정 적용

**모든 분석 스크립트는 루트 디렉토리에서 실행:**
```bash
python analysis/[스크립트명].py
```

---

## 🔧 기술 스택

### 핵심 라이브러리
```
torch>=2.0.0          # PyTorch (MPS/CUDA 지원)
numpy>=1.24.0         # 수치 계산
tqdm>=4.65.0          # 진행률 표시
PyWavelets>=1.4.0     # Wavelet denoising
matplotlib>=3.7.0     # 시각화
```

### 모델 특징
- **아키텍처**: Hyena (Liquid AI)
- **복잡도**: O(n log n) - FFT 기반
- **장점**: 긴 시퀀스, 전체 경로 패턴 학습
- **파라미터**: ~5M (hidden=384, depth=10)

### 학습 최적화 (2025 표준)
- ✅ Mixed Precision (AMP)
- ✅ Learning Rate Warmup (5 epochs)
- ✅ **5-Epoch Moving Average Adaptive LR** (양방향 조절)
- ✅ Early Stopping (P90 기준)
- ✅ Gradient Clipping
- ✅ Weight Decay (AdamW)
- ✅ Dropout (0.12)
- ✅ Seed 고정 (42)

---

## 📈 성능 목표

### 평가 메트릭

#### 📊 기본 메트릭:
- **MAE** (Mean Absolute Error): 평균 절대 오차 (위치 오차의 평균)
- **RMSE** (Root Mean Squared Error): 제곱 평균 제곱근 (큰 오차에 민감)

#### 📈 분포 메트릭:
- **Median (P50)**: 중앙값, "50%가 이 값 이하"
- **P90**: 90th percentile, "90%가 이 값 이하"
- **P95**: 95th percentile, "95%가 이 값 이하"
- **Min / Max**: 최소 / 최대 오차

#### 📍 CDF (Cumulative Distribution Function):
- **≤ 1m**: 1m 이내 예측 비율
- **≤ 2m**: 2m 이내 예측 비율
- **≤ 3m**: 3m 이내 예측 비율
- **≤ 5m**: 5m 이내 예측 비율

#### 🔊 Noise Robustness:
- Gaussian noise (σ=0.1, 0.2, 0.5) 추가 후 성능 저하율
- 실전 환경의 센서 노이즈 대응력 측정

#### 거리 계산:
- **Manhattan Distance** 사용: `|Δx| + |Δy|`
- 복도 구조 특성상 대각선 이동 불가 → 실제 이동 거리 반영

### 목표 성능 vs 실제 달성 ✅

| 지표 | 목표 | 달성 (sliding_mag4) | 상태 |
|------|------|---------------------|------|
| **MAE** | < 1.4m | **0.948m** | ✅ 달성 |
| **P90** | < 2m | **1.660m** | ✅ 달성 |
| **Median** | < 1m | **0.552m** | ✅ 달성 |
| **RMSE** | < 1.8m | **2.202m** | ⚠️ 근접 |
| **CDF ≤1m** | - | **75.2%** | 📊 |
| **CDF ≤2m** | - | **93.4%** | 📊 |
| **CDF ≤3m** | > 85% | **96.5%** | ✅ 달성 |

**🎯 핵심 성과:**
- P90 < 2m 목표 달성 (90%가 1.66m 이내)
- 평균 오차 0.95m (MAE < 1m)
- 중앙값 0.55m (절반이 0.6m 이내)
- CDF 96.5% (3m 이내 예측)
- Epoch 95에서 최적 성능, Epoch 110에서 Early Stopping

---

## 🐛 트러블슈팅

### 1. Windows 줄바꿈 문제
```bash
# 해결됨: 모든 .sh 파일 LF로 변환 완료
```

### 2. MPS 미작동
```python
# src/train_sliding.py에서 자동 감지
if torch.backends.mps.is_available():
    device = torch.device("mps")
```

### 3. 메모리 부족
```bash
# BATCH_SIZE 줄이기
BATCH_SIZE=32  # 64 → 32
```

### 4. 학습 속도 느림
```bash
# AMP 확인
⚡ Mixed Precision (AMP) 활성화  # 이 메시지 확인
```

---

## 📝 체크포인트 구조 및 재평가

### 체크포인트 구조

```python
checkpoint = {
    "model_state": model.state_dict(),       # 모델 가중치
    "optimizer_state": optimizer.state_dict(), # 옵티마이저 상태
    "epoch": epoch,                          # 에포크 번호
    "val_rmse": val_rmse,                   # Validation RMSE
    "val_p90": val_p90,                     # Validation P90
    "meta": {                                # 메타데이터
        "feature_mode": "mag3",
        "n_features": 3,
        "window_size": 250,
        ...
    }
}
```

### 체크포인트 재평가 방법

#### 방법 1: test_only.py 사용 (추천)

**기본 사용:**
```bash
source venv/bin/activate

python src/test_only.py \
  --checkpoint checkpoints_adaptive/best.pt \
  --data-dir data/sliding_mag4_adaptive
```

**옵션:**
```bash
# CPU 강제 사용
python src/test_only.py \
  --checkpoint checkpoints_adaptive/best.pt \
  --data-dir data/sliding_mag4_adaptive \
  --cpu

# Noise test 건너뛰기 (빠른 평가)
python src/test_only.py \
  --checkpoint checkpoints_adaptive/best.pt \
  --data-dir data/sliding_mag4_adaptive \
  --no-noise-test

# 배치 크기 조정
python src/test_only.py \
  --checkpoint checkpoints_adaptive/best.pt \
  --data-dir data/sliding_mag4_adaptive \
  --batch-size 64
```

**출력 예시:**
```
🧪 Test Only Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Checkpoint: checkpoints_adaptive/best.pt
  Data dir: data/sliding_mag4_adaptive

📊 데이터 정보:
   Features: 4
   Window size: 250
   Test: 2722개 샘플

🔄 Checkpoint 로드 중...
   Epoch: 45
   Val RMSE: 1.850m
   Val P90: 2.120m
✅ 모델 로드 완료

[Test Results]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 기본 메트릭:
  PE (Positioning Error):  1.820m
  MAE (Mean Absolute):     1.820m
  RMSE (Root Mean Sq):     2.150m

📈 분포:
  Median (P50):  1.450m
  P90:           3.200m
  P95:           4.100m
  Min:           0.120m
  Max:           8.500m

📍 CDF (누적 분포):
  ≤ 1m:  42.5%
  ≤ 2m:  68.3%
  ≤ 3m:  85.2%
  ≤ 5m:  96.1%

🔊 Noise Robustness Test:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Noise σ=0.1: MAE=1.935m (degradation: +6.3%)
  Noise σ=0.2: MAE=2.148m (degradation: +18.0%)
  Noise σ=0.5: MAE=3.251m (degradation: +78.6%)

✅ 테스트 완료!
```

#### 방법 2: 여러 체크포인트 비교

```bash
# 모든 체크포인트 평가
for ckpt in checkpoints_adaptive/*.pt; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $ckpt"
    python src/test_only.py \
      --checkpoint "$ckpt" \
      --data-dir data/sliding_mag4_adaptive \
      --no-noise-test
done
```

#### 방법 3: 수동 로드 (Python 코드)

```python
import torch
from model import HyenaPositioning

# 체크포인트 로드
checkpoint = torch.load("checkpoints_adaptive/best.pt", map_location="cpu")

# 모델 생성 및 가중치 로드
model = HyenaPositioning(
    input_dim=checkpoint["meta"]["n_features"],
    hidden_dim=384,
    output_dim=2,
    depth=10,
    dropout=0.1,
    num_edge_types=1,
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 평가 진행
# ... (test loop)
```

### 체크포인트 활용 시나리오

1. **다른 테스트 데이터 평가**
   ```bash
   python src/test_only.py \
     --checkpoint checkpoints_adaptive/best.pt \
     --data-dir data/new_test_dataset
   ```

2. **성능 비교 (이전 vs 현재)**
   ```bash
   # 이전 모델
   python src/test_only.py --checkpoint old/best.pt --data-dir data/test
   # 현재 모델
   python src/test_only.py --checkpoint new/best.pt --data-dir data/test
   ```

3. **앙상블 평가** (여러 체크포인트 결과 평균)
   - 다양한 시드로 학습한 모델들 평가
   - 각 모델의 예측 평균

---

## 🎯 다음 단계

### 개선 가능 항목
1. **WandB Logging**: 실험 추적 (여러 실험 비교 시)
2. **Model EMA**: 추론 안정성 향상 (최종 단계)
3. **Ensemble**: 여러 모델 앙상블

### 현재 수준
- ✅ 2023-2024 표준 학습 파이프라인
- ✅ 견고한 Baseline 구축
- ⚠️ 최신 실험 추적 도구 미적용

---

**작성일**: 2025년 11월 23일
**버전**: 1.0
**Python**: 3.13
**PyTorch**: 2.0+
