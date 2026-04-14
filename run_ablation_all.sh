#!/bin/bash
# run_ablation_all.sh — Sequential ablation study across all 4 remaining datasets.
# Runs 3 conditions per dataset (12 total training runs):
#   1. Baseline (no physics, no spectral) — train_ablation.py
#   2. + Physics loss                     — train_physics.py
#   3. + Physics + Spectral loss          — train_spectral.py
#
# Estimated total time: ~12–16 hours (sequential, single GPU)
#
# Usage:
#   chmod +x run_ablation_all.sh
#   nohup ./run_ablation_all.sh > logs/ablation_all.log 2>&1 &

set -e
mkdir -p logs

DATASETS=("kaist" "medical" "cbsr" "chilli")
EPOCHS=100
BATCH=16   # Use 16 to be safe with GPU memory (dopri5 is memory-heavy)

echo "========================================================"
echo "  PC-LiquidGAN — Full Ablation Study"
echo "  Datasets: kaist, medical, cbsr, chilli"
echo "  Conditions: baseline | +physics | +physics+spectral"
echo "  Epochs per run: $EPOCHS  Batch: $BATCH"
echo "  Started: $(date)"
echo "========================================================"

for DS in "${DATASETS[@]}"; do

    echo ""
    echo "========================================================"
    echo "  DATASET: ${DS^^}"
    echo "========================================================"

    # ── Condition 1: Baseline (no physics, no spectral) ──────────────────────
    echo ""
    echo ">>> [${DS^^}] Condition 1/3: Baseline (no physics, no spectral)"
    echo "    Started: $(date)"
    python train_ablation.py \
        --dataset "$DS" \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        2>&1 | tee "logs/ablation_${DS}_baseline.log"
    echo "    Done: $(date)"

    # ── Condition 2: + Physics loss ───────────────────────────────────────────
    echo ""
    echo ">>> [${DS^^}] Condition 2/3: + Physics loss"
    echo "    Started: $(date)"
    python train_physics.py \
        --dataset "$DS" \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        2>&1 | tee "logs/ablation_${DS}_physics.log"
    echo "    Done: $(date)"

    # ── Condition 3: + Physics + Spectral ─────────────────────────────────────
    echo ""
    echo ">>> [${DS^^}] Condition 3/3: + Physics + Spectral"
    echo "    Started: $(date)"
    python train_spectral.py \
        --dataset "$DS" \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lambda_spec 0.5 \
        2>&1 | tee "logs/ablation_${DS}_spectral.log"
    echo "    Done: $(date)"

    echo ""
    echo "  ✓ ${DS^^} ablation complete at $(date)"

done

echo ""
echo "========================================================"
echo "  ALL ABLATION RUNS COMPLETE"
echo "  Finished: $(date)"
echo ""
echo "  Results in:     results_ablation/, results_physics/, results_spectral/"
echo "  Checkpoints in: checkpoints_ablation/, checkpoints_physics/, checkpoints_spectral/"
echo ""
echo "  Per-dataset logs:"
for DS in "${DATASETS[@]}"; do
    echo "    $DS: logs/ablation_${DS}_*.log"
done
echo "========================================================"
