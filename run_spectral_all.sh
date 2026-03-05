#!/bin/bash
# run_spectral_all.sh
# Run spectral loss training sequentially on all 5 datasets.
# This ensures full GPU for each dataset, giving the cleanest results.

set -e
cd /home/harshney/Desktop/PC-LiquidGAN
source venv/bin/activate

DATASETS=("chilli" "cbsr" "medical" "kaist")
LAMBDA_SPEC=0.5
EPOCHS=100
BATCH=32

echo "============================================================"
echo "  Sequential Spectral Loss Training — Remaining 4 Datasets"
echo "  Started: $(date)"
echo "============================================================"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Starting $DS at $(date)"
    python3 train_spectral.py \
        --dataset "$DS" \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lambda_spec $LAMBDA_SPEC
    echo ">>> Finished $DS at $(date)"
done

echo ""
echo "============================================================"
echo " ALL SPECTRAL TRAININGS COMPLETE at $(date)"
echo "============================================================"
