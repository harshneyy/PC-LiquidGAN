"""
config.py — All hyperparameters and configuration for PC-LiquidGAN
"""

class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    IMG_SIZE    = 256       # Input/output image resolution
    BATCH_SIZE  = 8         # Reduce to 4 if GPU OOM
    NUM_WORKERS = 2         # DataLoader workers

    # ── Model ─────────────────────────────────────────────────────────────────
    LATENT_DIM  = 128       # Neural ODE latent space size
    HIDDEN_SIZE = 256       # LNN hidden state size
    ODE_METHOD  = 'dopri5'  # ODE solver: dopri5, euler, rk4
    ODE_RTOL    = 1e-3      # Relative tolerance for ODE solver
    ODE_ATOL    = 1e-4      # Absolute tolerance for ODE solver
    LNN_STEPS   = 6         # LNN integration steps per forward pass

    # ── Training ──────────────────────────────────────────────────────────────
    EPOCHS      = 100
    LR_G        = 2e-4      # Generator learning rate
    LR_D        = 2e-4      # Discriminator learning rate
    BETA1       = 0.5
    BETA2       = 0.999
    GRAD_CLIP   = 1.0       # Gradient clipping norm

    # ── Loss Weights ──────────────────────────────────────────────────────────
    LAMBDA_ADV    = 1.0     # Adversarial loss weight
    LAMBDA_REC    = 10.0    # Reconstruction (L1) loss weight
    LAMBDA_FLUX   = 0.5     # Heat diffusion loss weight
    LAMBDA_ENERGY = 0.25    # Energy conservation loss weight

    # ── Thermal Physics ───────────────────────────────────────────────────────
    THERMAL_ALPHA = 0.001   # Thermal diffusivity constant (α)

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR      = './data'
    SAVE_DIR      = './checkpoints'
    LOG_DIR       = './logs'
    RESULTS_DIR   = './results'

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_FREQ     = 10      # Evaluate every N epochs
    SAVE_FREQ     = 10      # Save checkpoint every N epochs
    LOG_FREQ      = 50      # Log every N steps
