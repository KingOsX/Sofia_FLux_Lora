#!/bin/bash
# =============================================================================
# SOFIA BELMONT — RunPod Start Script
# =============================================================================
# À coller dans RunPod → Template → "Start Script"
# (ou pod settings → "On-Start Script")
#
# Ce script s'exécute automatiquement à CHAQUE démarrage du pod.
# Il monte le volume persistant, clone/met à jour le repo git,
# et prépare l'environnement sans retélécharger les modèles.
# =============================================================================

set -e

# ─────────────────────────────────────────────
# VARIABLES
# ─────────────────────────────────────────────
WORKSPACE=/workspace
REPO_URL="https://github.com/KingOsX/Sofia_FLux_Lora.git"
REPO_DIR=$WORKSPACE/Sofia_FLux_Lora
MODELS_DIR=$WORKSPACE/models
KOHYA_DIR=$WORKSPACE/kohya_ss
COMFYUI_DIR=$WORKSPACE/ComfyUI
LOG_FILE=$WORKSPACE/startup.log

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✅ $(date +%H:%M:%S)]${NC} $1" | tee -a $LOG_FILE; }
info() { echo -e "${CYAN}[ℹ️  $(date +%H:%M:%S)]${NC} $1" | tee -a $LOG_FILE; }
warn() { echo -e "${YELLOW}[⚠️  $(date +%H:%M:%S)]${NC} $1" | tee -a $LOG_FILE; }

echo "" | tee -a $LOG_FILE
echo "╔═══════════════════════════════════════════════╗" | tee -a $LOG_FILE
echo "║   Sofia Belmont — RunPod Start Script         ║" | tee -a $LOG_FILE
echo "║   $(date '+%Y-%m-%d %H:%M:%S')                        ║" | tee -a $LOG_FILE
echo "╚═══════════════════════════════════════════════╝" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ─────────────────────────────────────────────
# 1. GPU INFO
# ─────────────────────────────────────────────
info "GPU détecté :"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a $LOG_FILE
log "CUDA disponible : $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ─────────────────────────────────────────────
# 2. TOKENS & VARIABLES D'ENVIRONNEMENT
# ─────────────────────────────────────────────

# Charger .env depuis le volume persistant (s'il existe)
ENV_FILE=$WORKSPACE/.env
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    log ".env chargé → $ENV_FILE"
fi

# Les tokens peuvent aussi venir des env vars RunPod (pod settings)
# → Pod Settings → Environment Variables → HF_TOKEN / CIVITAI_TOKEN
[ -n "$HF_TOKEN" ]       && log "HF_TOKEN      ✓" || warn "HF_TOKEN manquant (requis pour setup)"
[ -n "$CIVITAI_TOKEN" ]  && log "CIVITAI_TOKEN ✓" || warn "CIVITAI_TOKEN manquant"

export WORKSPACE=$WORKSPACE
export MODELS_DIR=$MODELS_DIR
export LORA_DIR=$WORKSPACE/loras
export DATASET_DIR=$WORKSPACE/dataset/sofia
export PYTHONPATH=$KOHYA_DIR:$PYTHONPATH

# Persistance dans .bashrc (si premier démarrage)
if ! grep -q "Sofia Belmont" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# ── Sofia Belmont LoRA Pipeline ──
export WORKSPACE=/workspace
export MODELS_DIR=/workspace/models
export LORA_DIR=/workspace/loras
export DATASET_DIR=/workspace/dataset/sofia
export KOHYA_DIR=/workspace/kohya_ss
export COMFYUI_DIR=/workspace/ComfyUI
export PYTHONPATH=/workspace/kohya_ss:$PYTHONPATH
[ -f /workspace/.env ] && source /workspace/.env
EOF
    log ".bashrc mis à jour"
fi

# ─────────────────────────────────────────────
# 3. CLONE / MISE À JOUR DU REPO GIT
# ─────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    info "Repo déjà cloné → mise à jour..."
    cd $REPO_DIR
    git pull origin master 2>&1 | tee -a $LOG_FILE
    log "Repo mis à jour : $REPO_DIR"
else
    info "Clonage du repo..."
    git clone $REPO_URL $REPO_DIR 2>&1 | tee -a $LOG_FILE
    log "Repo cloné : $REPO_DIR"
fi

# Rendre les scripts exécutables
chmod +x $REPO_DIR/scripts/*.sh 2>/dev/null || true

# Copier la config TOML si pas déjà dans workspace
if [ ! -f "$WORKSPACE/sofia_lora_config.toml" ]; then
    cp $REPO_DIR/sofia_lora_config.toml $WORKSPACE/sofia_lora_config.toml
    log "sofia_lora_config.toml copié dans $WORKSPACE"
fi

# ─────────────────────────────────────────────
# 4. VÉRIFICATION DES MODÈLES
# ─────────────────────────────────────────────
info "Vérification des modèles..."

check_model() {
    local path=$1
    local name=$2
    if [ -f "$path" ] || [ -d "$path" ]; then
        log "$name ✓"
    else
        warn "$name MANQUANT → $path"
        warn "  Lancer : bash $REPO_DIR/scripts/setup_runpod.sh"
    fi
}

check_model "$MODELS_DIR/vae/ae.safetensors"                   "VAE"
check_model "$COMFYUI_DIR/models/clip/clip_l.safetensors"      "CLIP-L (ComfyUI)"
check_model "$COMFYUI_DIR/models/clip/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL fp8 (ComfyUI)"

# Vérifier qu'au moins un modèle Flux est présent
FLUX_COUNT=$(find $MODELS_DIR/flux -name "*.safetensors" -size +1M 2>/dev/null | wc -l)
if [ "$FLUX_COUNT" -gt 0 ]; then
    log "Modèles Flux : $FLUX_COUNT trouvé(s)"
    find $MODELS_DIR/flux -name "*.safetensors" -size +1M 2>/dev/null | while read f; do
        SIZE=$(du -sh "$f" | cut -f1)
        log "  ✓ $(basename $f) ($SIZE)"
    done
else
    warn "Aucun modèle Flux valide → lancer setup_runpod.sh"
fi

# ─────────────────────────────────────────────
# 5. VÉRIFICATION KOHYA SS
# ─────────────────────────────────────────────
if [ -d "$KOHYA_DIR" ]; then
    log "Kohya SS présent"
else
    warn "Kohya SS non installé → lancer setup_runpod.sh"
fi

# ─────────────────────────────────────────────
# 6. COMFYUI — DÉPENDANCES + SYMLINKS
# (pip packages perdus à chaque restart container)
# ─────────────────────────────────────────────
if [ -d "$COMFYUI_DIR" ]; then
    log "ComfyUI présent → réinstallation des dépendances..."
    pip install -q --root-user-action=ignore -r $COMFYUI_DIR/requirements.txt 2>&1 | tail -1 | tee -a $LOG_FILE
    log "ComfyUI dépendances OK"

    # Dépendances des custom nodes (perdues à chaque restart)
    for node_dir in $COMFYUI_DIR/custom_nodes/*/; do
        req="$node_dir/requirements.txt"
        if [ -f "$req" ]; then
            node_name=$(basename "$node_dir")
            pip install -q --root-user-action=ignore -r "$req" 2>&1 | tail -1 | tee -a $LOG_FILE
            log "Custom node deps : $node_name ✓"
        fi
    done

    # Symlinks modèles → ComfyUI
    mkdir -p $COMFYUI_DIR/models/{checkpoints,unet,vae,clip,loras,controlnet}

    # Tous les modèles Flux → checkpoints ET unet
    for f in $MODELS_DIR/flux/*.safetensors; do
        [ -f "$f" ] || continue
        bn=$(basename "$f")
        [ ! -e "$COMFYUI_DIR/models/checkpoints/$bn" ] && \
            ln -sf "$f" "$COMFYUI_DIR/models/checkpoints/$bn"
        [ ! -e "$COMFYUI_DIR/models/unet/$bn" ] && \
            ln -sf "$f" "$COMFYUI_DIR/models/unet/$bn"
    done

    # VAE
    [ -f "$MODELS_DIR/vae/ae.safetensors" ] && \
        ln -sf "$MODELS_DIR/vae/ae.safetensors" \
               "$COMFYUI_DIR/models/vae/ae.safetensors" 2>/dev/null || true

    # LoRAs
    [ -d "$WORKSPACE/loras" ] && \
        ln -sf "$WORKSPACE/loras" "$COMFYUI_DIR/models/loras" 2>/dev/null || true

    # ControlNet
    [ -d "$MODELS_DIR/controlnet" ] && \
        ln -sf "$MODELS_DIR/controlnet" "$COMFYUI_DIR/models/controlnet" 2>/dev/null || true

    log "Symlinks ComfyUI recréés"
else
    warn "ComfyUI non installé → lancer setup_runpod.sh option [1]"
fi

# ─────────────────────────────────────────────
# 7. DATASET STATUS
# ─────────────────────────────────────────────
info "Status du dataset :"
IMAGE_COUNT=$(find $WORKSPACE/dataset/sofia/processed -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
CAPTION_COUNT=$(find $WORKSPACE/dataset/sofia/captions -name "*.txt" 2>/dev/null | wc -l)

echo "  📸 Images traitées : $IMAGE_COUNT" | tee -a $LOG_FILE
echo "  📝 Captions        : $CAPTION_COUNT" | tee -a $LOG_FILE

# ─────────────────────────────────────────────
# 8. DÉMARRAGE AUTOMATIQUE SERVICES
# ─────────────────────────────────────────────

# --- Jupyter Lab (port 8888) ---
pip install -q --root-user-action=ignore jupyterlab 2>/dev/null || true
nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir=$WORKSPACE \
    > $WORKSPACE/jupyter.log 2>&1 &
log "Jupyter Lab démarré → port 8888 (sans mot de passe)"

# --- TensorBoard (port 6006) ---
# if [ -d "$WORKSPACE/logs" ]; then
#     nohup tensorboard --logdir=$WORKSPACE/logs --host=0.0.0.0 --port=6006 \
#         > $WORKSPACE/tensorboard.log 2>&1 &
#     log "TensorBoard démarré → port 6006"
# fi

# --- ComfyUI (port 8188) ---
if [ -d "$COMFYUI_DIR" ]; then
    nohup python $COMFYUI_DIR/main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-cors-header \
        > $WORKSPACE/comfyui.log 2>&1 &
    log "ComfyUI démarré → port 8188"
fi

# ─────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────
echo "" | tee -a $LOG_FILE
echo "╔═══════════════════════════════════════════════╗" | tee -a $LOG_FILE
echo "║   ✅ Environnement prêt !                     ║" | tee -a $LOG_FILE
echo "╚═══════════════════════════════════════════════╝" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "  Commandes rapides :" | tee -a $LOG_FILE
echo "  → Setup initial     : bash $REPO_DIR/scripts/setup_runpod.sh" | tee -a $LOG_FILE
echo "  → Préparer dataset  : python $REPO_DIR/scripts/prepare_dataset.py" | tee -a $LOG_FILE
echo "  → Générer captions  : python $REPO_DIR/scripts/generate_captions.py" | tee -a $LOG_FILE
echo "  → Entraîner LoRA    : bash $REPO_DIR/scripts/train_lora.sh" | tee -a $LOG_FILE
echo "  → Logs startup      : cat $WORKSPACE/startup.log" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
