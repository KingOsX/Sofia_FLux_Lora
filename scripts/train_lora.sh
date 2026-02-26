#!/bin/bash
# =============================================================================
# SOFIA BELMONT — Entraînement LoRA Flux.1-dev
# Optimisé pour NVIDIA A40 (48GB VRAM)
# =============================================================================
# Usage : bash /workspace/scripts/train_lora.sh
# =============================================================================

set -e

# ─────────────────────────────────────────────
# CONFIG — Modifier selon vos besoins
# ─────────────────────────────────────────────
WORKSPACE=/workspace
COMFYUI_DIR="$WORKSPACE/ComfyUI"
KOHYA_DIR="$WORKSPACE/kohya_ss"
SCRIPTS_DIR="$KOHYA_DIR/sd-scripts"

# Modèle Flux — prend le premier .safetensors valide (>1GB)
MODEL_PATH=$(find $WORKSPACE/models/flux -name "*.safetensors" -size +1M 2>/dev/null | head -1)

# CLIP — priorité aux fichiers safetensors ComfyUI
CLIP_L_PATH="$COMFYUI_DIR/models/clip/clip_l.safetensors"
T5XXL_PATH="$COMFYUI_DIR/models/clip/t5xxl_fp8_e4m3fn.safetensors"
AE_PATH="$WORKSPACE/models/vae/ae.safetensors"
CONFIG_PATH="$WORKSPACE/sofia_lora_config.toml"
OUTPUT_DIR="$WORKSPACE/loras"
OUTPUT_NAME="sofia_belmont_v1"
LOGS_DIR="$WORKSPACE/logs"

# Paramètres d'entraînement — optimisés A40 48GB
# A40 avantages : 48GB VRAM → network_dim élevé sans OOM
MAX_TRAIN_STEPS=1500
SAVE_EVERY_N_STEPS=250
NETWORK_DIM=64          # 32 sur A100 40GB → 64 possible sur A40 48GB
NETWORK_ALPHA=32        # Toujours la moitié de dim pour stabilité
LEARNING_RATE="5e-4"
LR_WARMUP_STEPS=100
SEED=42

# ─────────────────────────────────────────────
# COULEURS
# ─────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✅]${NC} $1"; }
warn()  { echo -e "${YELLOW}[⚠️ ]${NC} $1"; }
error() { echo -e "${RED}[❌]${NC} $1"; exit 1; }

# ─────────────────────────────────────────────
# VÉRIFICATIONS PRÉ-ENTRAÎNEMENT
# ─────────────────────────────────────────────
echo -e "${CYAN}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   Sofia Belmont — LoRA Training Flux      ║"
echo "  ║   steps: $MAX_TRAIN_STEPS | dim: $NETWORK_DIM | lr: $LEARNING_RATE     ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# Vérifier les fichiers requis
[ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ] || error "Aucun modèle Flux trouvé dans $WORKSPACE/models/flux/"
[ -f "$CLIP_L_PATH" ]   || error "CLIP encoder non trouvé : $CLIP_L_PATH"
[ -f "$T5XXL_PATH" ]    || error "T5-XXL non trouvé : $T5XXL_PATH"
[ -f "$AE_PATH" ]       || error "VAE non trouvé : $AE_PATH"
[ -f "$CONFIG_PATH" ]   || error "Config TOML non trouvée : $CONFIG_PATH"
[ -f "$SCRIPTS_DIR/flux_train_network.py" ] || error "flux_train_network.py non trouvé : $SCRIPTS_DIR"

log "Tous les fichiers requis sont présents"

# Compter les images
IMAGE_COUNT=$(find $WORKSPACE/dataset/sofia/processed -name "*.jpg" -o -name "*.png" | wc -l)
CAPTION_COUNT=$(find $WORKSPACE/dataset/sofia/captions -name "*.txt" | wc -l)

echo ""
echo -e "  📸 Images   : ${YELLOW}$IMAGE_COUNT${NC}"
echo -e "  📝 Captions : ${YELLOW}$CAPTION_COUNT${NC}"
echo ""

[ "$IMAGE_COUNT" -lt 10 ] && error "Minimum 10 images requis (trouvé: $IMAGE_COUNT)"
[ "$IMAGE_COUNT" -ne "$CAPTION_COUNT" ] && warn "Images ($IMAGE_COUNT) ≠ Captions ($CAPTION_COUNT)"

# GPU info
echo "GPU :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

mkdir -p $OUTPUT_DIR $LOGS_DIR

echo ""
log "Démarrage de l'entraînement..."
echo -e "  → Modèle base  : flux1-dev"
echo -e "  → Steps        : $MAX_TRAIN_STEPS"
echo -e "  → Network dim  : $NETWORK_DIM"
echo -e "  → Learning rate: $LEARNING_RATE"
echo -e "  → Output       : $OUTPUT_DIR/${OUTPUT_NAME}.safetensors"
echo ""

START_TIME=$(date +%s)

# ─────────────────────────────────────────────
# LANCEMENT DE L'ENTRAÎNEMENT
# ─────────────────────────────────────────────
export PYTHONPATH=$SCRIPTS_DIR:$KOHYA_DIR:$PYTHONPATH

python $SCRIPTS_DIR/flux_train_network.py \
    --pretrained_model_name_or_path="$MODEL_PATH" \
    --clip_l="$CLIP_L_PATH" \
    --t5xxl="$T5XXL_PATH" \
    --ae="$AE_PATH" \
    --dataset_config="$CONFIG_PATH" \
    \
    --output_dir="$OUTPUT_DIR" \
    --output_name="$OUTPUT_NAME" \
    --save_model_as="safetensors" \
    --save_every_n_steps=$SAVE_EVERY_N_STEPS \
    \
    --network_module="networks.lora_flux" \
    --network_dim=$NETWORK_DIM \
    --network_alpha=$NETWORK_ALPHA \
    --network_train_unet_only \
    \
    --optimizer_type="adamw8bit" \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="cosine_with_restarts" \
    --lr_warmup_steps=$LR_WARMUP_STEPS \
    \
    --max_train_steps=$MAX_TRAIN_STEPS \
    \
    --mixed_precision="bf16" \
    --cache_latents \
    --cache_latents_to_disk \
    --gradient_checkpointing \
    --sdpa \
    \
    --min_bucket_reso=512 \
    --max_bucket_reso=2048 \
    --bucket_reso_steps=64 \
    \
    --timestep_sampling="sigmoid" \
    --model_prediction_type="raw" \
    --discrete_flow_shift=3.1582 \
    --loss_type="l2" \
    \
    --log_with="tensorboard" \
    --logging_dir="$LOGS_DIR" \
    \
    --seed=$SEED

# ─────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Entraînement terminé !${NC}"
echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo ""
echo -e "  Durée       : ${YELLOW}${DURATION} minutes${NC}"
echo -e "  Checkpoints : ${YELLOW}$OUTPUT_DIR/${NC}"
echo ""

ls -lh $OUTPUT_DIR/*.safetensors 2>/dev/null || warn "Aucun checkpoint trouvé dans $OUTPUT_DIR"

echo ""
echo -e "  ${YELLOW}Prochaines étapes :${NC}"
echo -e "  1. Visualiser les courbes : tensorboard --logdir $LOGS_DIR --host 0.0.0.0 --port 6006"
echo -e "  2. Tester les checkpoints : python $WORKSPACE/scripts/test_lora.py"
echo -e "  3. Lancer ComfyUI        : python $WORKSPACE/ComfyUI/main.py --listen 0.0.0.0 --port 8188"
echo ""
