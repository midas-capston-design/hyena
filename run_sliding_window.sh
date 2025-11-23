#!/bin/bash
# Sliding Window 방식 전처리 + 학습

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "🔄 Sliding Window Pipeline"
echo "========================================="
echo ""

# 설정
FEATURE_MODE="mag4"  # mag3, mag4, full
WINDOW_SIZE=250
STRIDE=50
EPOCHS=200         # 충분한 학습 시간 확보
BATCH_SIZE=64      # MPS 활용, 학습 속도 향상
HIDDEN_DIM=384     # 모델 용량 증가 (성능 개선)
DEPTH=10           # 더 깊은 표현력
PATIENCE=15        # 충분히 기다려서 최적점 찾기

echo "========================================="
echo "📊 [1/2] 전처리 (Sliding Window)"
echo "========================================="
python3 src/preprocess_sliding.py \
  --raw-dir data/raw \
  --nodes data/nodes_final.csv \
  --output data/sliding_${FEATURE_MODE} \
  --feature-mode $FEATURE_MODE \
  --window-size $WINDOW_SIZE \
  --stride $STRIDE \
  --train-ratio 0.6 \
  --val-ratio 0.2

echo ""
echo "========================================="
echo "🧠 [2/2] 학습 (Causal Hyena)"
echo "========================================="
python3 src/train_sliding.py \
  --data-dir data/sliding_${FEATURE_MODE} \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr 3e-4 \
  --hidden-dim $HIDDEN_DIM \
  --depth $DEPTH \
  --dropout 0.12 \
  --patience $PATIENCE \
  --checkpoint-dir checkpoints_sliding_${FEATURE_MODE}

echo ""
echo "========================================="
echo "✅ 완료!"
echo "========================================="
echo ""
echo "체크포인트: checkpoints_sliding_${FEATURE_MODE}/best.pt"
echo "========================================="
