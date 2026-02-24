#!/bin/bash
# =============================================================================
# SOFIA BELMONT — Setup RunPod complet
# Flux.1-dev + Kohya SS + ComfyUI
# =============================================================================
# Usage : bash /workspace/scripts/setup_runpod.sh
# =============================================================================

set -e  # Stop on error

# ─────────────────────────────────────────────
# COULEURS
# ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()    { echo -e "${GREEN}[✅]${NC} $1"; }
info()   { echo -e "${BLUE}[ℹ️ ]${NC} $1"; }
warn()   { echo -e "${YELLOW}[⚠️ ]${NC} $1"; }
error()  { echo -e "${RED}[❌]${NC} $1"; exit 1; }
header() { echo -e "\n${CYAN}═══════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════${NC}\n"; }

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WORKSPACE=/workspace
MODELS_DIR=$WORKSPACE/models
DATASET_DIR=$WORKSPACE/dataset/sofia
LORAS_DIR=$WORKSPACE/loras
SCRIPTS_DIR=$WORKSPACE/scripts
LOGS_DIR=$WORKSPACE/logs
OUTPUTS_DIR=$WORKSPACE/outputs

KOHYA_DIR=$WORKSPACE/kohya_ss
COMFYUI_DIR=$WORKSPACE/ComfyUI

HF_TOKEN=""   # ← Rempli par la fonction ask_hf_token

# ─────────────────────────────────────────────
# VÉRIFICATIONS INITIALES
# ─────────────────────────────────────────────
check_gpu() {
    header "Vérification GPU"
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi non trouvé. Assurez-vous d'être sur un pod avec GPU."
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    log "GPU détecté"
}

check_disk() {
    header "Espace disque"
    df -h $WORKSPACE | tail -1
    AVAIL=$(df $WORKSPACE | tail -1 | awk '{print $4}')
    # Minimum 80GB recommandé (en KB)
    if [ "$AVAIL" -lt 83886080 ]; then
        warn "Moins de 80GB disponibles. Recommandé : 150GB+"
    else
        log "Espace disque suffisant"
    fi
}

ask_hf_token() {
    header "HuggingFace Token"
    echo -e "${YELLOW}Flux.1-dev est un modèle à accès restreint.${NC}"
    echo -e "→ Obtenez votre token sur : ${CYAN}https://huggingface.co/settings/tokens${NC}"
    echo -e "→ Et acceptez les conditions : ${CYAN}https://huggingface.co/black-forest-labs/FLUX.1-dev${NC}"
    echo ""
    read -p "Entrez votre HuggingFace token (hf_xxx...) : " HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        error "Token HuggingFace requis pour télécharger Flux.1-dev"
    fi
    export HF_TOKEN=$HF_TOKEN
    log "Token HuggingFace configuré"
}

# ─────────────────────────────────────────────
# STEP 1 : STRUCTURE DE DOSSIERS
# ─────────────────────────────────────────────
create_structure() {
    header "Création de la structure de dossiers"

    mkdir -p $MODELS_DIR/{flux,clip,vae,controlnet}
    mkdir -p $DATASET_DIR/{raw,processed,captions}
    mkdir -p $LORAS_DIR
    mkdir -p $SCRIPTS_DIR
    mkdir -p $LOGS_DIR
    mkdir -p $OUTPUTS_DIR

    log "Structure créée dans $WORKSPACE"
    tree $WORKSPACE -L 2 2>/dev/null || ls -la $WORKSPACE
}

# ─────────────────────────────────────────────
# STEP 2 : DÉPENDANCES SYSTÈME
# ─────────────────────────────────────────────
install_system_deps() {
    header "Installation des dépendances système"

    apt-get update -qq
    apt-get install -y -qq \
        git \
        curl \
        wget \
        aria2 \
        tree \
        htop \
        tmux \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        2>/dev/null

    log "Dépendances système installées"
}

# ─────────────────────────────────────────────
# STEP 3 : PYTHON & PIP
# ─────────────────────────────────────────────
install_python_deps() {
    header "Installation des dépendances Python"

    pip install -q --upgrade pip
    pip install -q \
        huggingface_hub \
        Pillow \
        tqdm \
        tensorboard \
        accelerate \
        diffusers \
        transformers \
        safetensors

    log "Dépendances Python installées"
}

# ─────────────────────────────────────────────
# STEP 4 : TÉLÉCHARGEMENT FLUX.1-DEV
# ─────────────────────────────────────────────
download_flux() {
    header "Téléchargement Flux.1-dev (~23GB)"
    warn "Cela peut prendre 20-40 minutes selon la connexion..."

    if [ -f "$MODELS_DIR/flux/flux1-dev.safetensors" ]; then
        log "flux1-dev.safetensors déjà présent, skip."
        return
    fi

    huggingface-cli download \
        black-forest-labs/FLUX.1-dev \
        --local-dir $MODELS_DIR/flux \
        --include "flux1-dev.safetensors" \
        --token $HF_TOKEN

    log "Flux.1-dev téléchargé → $MODELS_DIR/flux/"
}

# ─────────────────────────────────────────────
# STEP 5 : TÉLÉCHARGEMENT CLIP ENCODERS
# ─────────────────────────────────────────────
download_clip() {
    header "Téléchargement CLIP Text Encoder (~1.7GB)"

    CLIP_DIR=$MODELS_DIR/clip/clip-vit-large-patch14

    if [ -d "$CLIP_DIR" ] && [ "$(ls -A $CLIP_DIR)" ]; then
        log "CLIP déjà présent, skip."
    else
        huggingface-cli download \
            openai/clip-vit-large-patch14 \
            --local-dir $CLIP_DIR \
            --token $HF_TOKEN
        log "CLIP téléchargé"
    fi
}

download_t5() {
    header "Téléchargement T5-XXL Encoder (~9.4GB)"

    T5_DIR=$MODELS_DIR/clip/t5-v1_1-xxl

    if [ -d "$T5_DIR" ] && [ "$(ls -A $T5_DIR)" ]; then
        log "T5-XXL déjà présent, skip."
    else
        huggingface-cli download \
            google/t5-v1_1-xxl \
            --local-dir $T5_DIR \
            --include "*.safetensors" "*.json" "*.txt" \
            --token $HF_TOKEN
        log "T5-XXL téléchargé"
    fi
}

# ─────────────────────────────────────────────
# STEP 6 : TÉLÉCHARGEMENT VAE
# ─────────────────────────────────────────────
download_vae() {
    header "Téléchargement VAE Flux (~335MB)"

    if [ -f "$MODELS_DIR/vae/ae.safetensors" ]; then
        log "VAE déjà présent, skip."
        return
    fi

    huggingface-cli download \
        black-forest-labs/FLUX.1-dev \
        ae.safetensors \
        --local-dir $MODELS_DIR/vae \
        --token $HF_TOKEN

    log "VAE téléchargé → $MODELS_DIR/vae/ae.safetensors"
}

# ─────────────────────────────────────────────
# STEP 7 : INSTALLATION KOHYA SS
# ─────────────────────────────────────────────
install_kohya() {
    header "Installation Kohya SS"

    if [ -d "$KOHYA_DIR" ]; then
        log "Kohya SS déjà installé. Mise à jour..."
        cd $KOHYA_DIR
        git pull
    else
        cd $WORKSPACE
        git clone https://github.com/bmaltais/kohya_ss.git
        cd $KOHYA_DIR
    fi

    # Installation avec support CUDA 12.1
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -q -r requirements.txt

    # Fichier requirements_flux.txt (peut ne pas exister dans toutes les versions)
    if [ -f "requirements_flux.txt" ]; then
        pip install -q -r requirements_flux.txt
    fi

    # Vérification
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponible!'; print(f'✅ CUDA OK — GPU: {torch.cuda.get_device_name(0)}')"

    log "Kohya SS installé → $KOHYA_DIR"
}

# ─────────────────────────────────────────────
# STEP 8 : INSTALLATION COMFYUI (optionnel)
# ─────────────────────────────────────────────
install_comfyui() {
    header "Installation ComfyUI"

    if [ -d "$COMFYUI_DIR" ]; then
        log "ComfyUI déjà installé. Mise à jour..."
        cd $COMFYUI_DIR && git pull
    else
        cd $WORKSPACE
        git clone https://github.com/comfyanonymous/ComfyUI.git
        cd $COMFYUI_DIR
        pip install -q -r requirements.txt
    fi

    # Liens symboliques vers les modèles
    info "Création des liens symboliques..."

    FLUX_LINK=$COMFYUI_DIR/models/unet/flux1-dev.safetensors
    VAE_LINK=$COMFYUI_DIR/models/vae/ae.safetensors

    [ ! -f "$FLUX_LINK" ] && ln -sf $MODELS_DIR/flux/flux1-dev.safetensors $FLUX_LINK && log "Lien flux1-dev créé"
    [ ! -f "$VAE_LINK" ]  && ln -sf $MODELS_DIR/vae/ae.safetensors $VAE_LINK           && log "Lien VAE créé"

    # Lien clip si pas déjà lié
    CLIP_LINK=$COMFYUI_DIR/models/clip
    [ ! -L "$CLIP_LINK" ] && ln -sf $MODELS_DIR/clip $CLIP_LINK && log "Lien CLIP créé"

    # Lien loras
    LORAS_LINK=$COMFYUI_DIR/models/loras
    [ ! -L "$LORAS_LINK" ] && ln -sf $LORAS_DIR $LORAS_LINK && log "Lien loras créé"

    # Custom nodes essentiels
    info "Installation des custom nodes..."
    NODES_DIR=$COMFYUI_DIR/custom_nodes

    for repo in \
        "https://github.com/ltdrdata/ComfyUI-Manager.git" \
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git" \
        "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git"
    do
        name=$(basename $repo .git)
        if [ ! -d "$NODES_DIR/$name" ]; then
            git clone -q $repo $NODES_DIR/$name
            log "Custom node installé : $name"
        else
            log "Custom node déjà présent : $name"
        fi
    done

    log "ComfyUI installé → $COMFYUI_DIR"
}

# ─────────────────────────────────────────────
# STEP 9 : VARIABLES D'ENVIRONNEMENT
# ─────────────────────────────────────────────
setup_env() {
    header "Configuration des variables d'environnement"

    cat >> ~/.bashrc << 'EOF'

# ── Sofia Belmont LoRA Pipeline ──
export WORKSPACE=/workspace
export MODELS_DIR=/workspace/models
export LORA_DIR=/workspace/loras
export DATASET_DIR=/workspace/dataset/sofia
export KOHYA_DIR=/workspace/kohya_ss
export COMFYUI_DIR=/workspace/ComfyUI
export PYTHONPATH=/workspace/kohya_ss:$PYTHONPATH
EOF

    source ~/.bashrc
    log "Variables d'environnement configurées dans ~/.bashrc"
}

# ─────────────────────────────────────────────
# STEP 10 : COPIER LES SCRIPTS DU PROJET
# ─────────────────────────────────────────────
copy_project_scripts() {
    header "Vérification des scripts du projet"

    SCRIPT_LIST=(
        "prepare_dataset.py"
        "generate_captions.py"
        "train_lora.sh"
        "test_lora.py"
    )

    for script in "${SCRIPT_LIST[@]}"; do
        if [ -f "$SCRIPTS_DIR/$script" ]; then
            log "$script présent"
        else
            warn "$script manquant dans $SCRIPTS_DIR/"
        fi
    done

    if [ -f "$WORKSPACE/sofia_lora_config.toml" ]; then
        log "sofia_lora_config.toml présent"
    else
        warn "sofia_lora_config.toml manquant dans $WORKSPACE/"
    fi
}

# ─────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────
final_report() {
    header "Setup terminé — Rapport"

    echo -e "${GREEN}Modèles téléchargés :${NC}"
    ls -lh $MODELS_DIR/flux/ 2>/dev/null    | grep safetensors && true
    ls -lh $MODELS_DIR/vae/ 2>/dev/null     | grep safetensors && true
    ls -lh $MODELS_DIR/clip/ 2>/dev/null    | grep -v total    && true

    echo ""
    echo -e "${GREEN}Espace utilisé :${NC}"
    du -sh $MODELS_DIR 2>/dev/null || true
    df -h $WORKSPACE | tail -1

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${CYAN}  PROCHAINES ÉTAPES${NC}"
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo ""
    echo -e "1. ${YELLOW}Uploader vos photos Sofia${NC} dans :"
    echo -e "   $DATASET_DIR/raw/"
    echo ""
    echo -e "2. ${YELLOW}Préparer le dataset :${NC}"
    echo -e "   python $SCRIPTS_DIR/prepare_dataset.py"
    echo ""
    echo -e "3. ${YELLOW}Générer les captions :${NC}"
    echo -e "   python $SCRIPTS_DIR/generate_captions.py"
    echo ""
    echo -e "4. ${YELLOW}Lancer l'entraînement :${NC}"
    echo -e "   bash $SCRIPTS_DIR/train_lora.sh"
    echo ""
    echo -e "5. ${YELLOW}Lancer ComfyUI :${NC}"
    echo -e "   python $COMFYUI_DIR/main.py --listen 0.0.0.0 --port 8188"
    echo ""
    echo -e "${GREEN}✅ Setup Sofia Belmont LoRA Pipeline terminé !${NC}"
}

# ─────────────────────────────────────────────
# MENU PRINCIPAL
# ─────────────────────────────────────────────
main() {
    echo -e "${CYAN}"
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║   Sofia Belmont — LoRA Training Setup     ║"
    echo "  ║   Flux.1-dev + Kohya SS + ComfyUI         ║"
    echo "  ╚═══════════════════════════════════════════╝"
    echo -e "${NC}"

    check_gpu
    check_disk

    # Mode automatique ou interactif
    if [ "$1" == "--auto" ]; then
        if [ -z "$HF_TOKEN" ]; then
            error "Mode --auto requiert HF_TOKEN dans l'environnement : export HF_TOKEN=hf_xxx..."
        fi
    else
        ask_hf_token
    fi

    echo ""
    echo -e "${YELLOW}Que voulez-vous installer ?${NC}"
    echo "  [1] Setup complet (Flux + Kohya + ComfyUI)"
    echo "  [2] Flux + Kohya seulement (sans ComfyUI)"
    echo "  [3] Uniquement les modèles Flux (download seul)"
    echo "  [4] Uniquement Kohya SS"
    echo ""
    read -p "Votre choix [1-4] : " CHOICE

    create_structure
    install_system_deps
    install_python_deps

    case $CHOICE in
        1)
            download_flux
            download_clip
            download_t5
            download_vae
            install_kohya
            install_comfyui
            ;;
        2)
            download_flux
            download_clip
            download_t5
            download_vae
            install_kohya
            ;;
        3)
            download_flux
            download_clip
            download_t5
            download_vae
            ;;
        4)
            install_kohya
            ;;
        *)
            error "Choix invalide"
            ;;
    esac

    setup_env
    copy_project_scripts
    final_report
}

main "$@"
