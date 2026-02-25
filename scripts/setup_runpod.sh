#!/bin/bash
# =============================================================================
# SOFIA BELMONT — Setup RunPod
# Flux.1-dev + Kohya SS + ComfyUI — NVIDIA A40 48GB
# =============================================================================
# Usage :
#   bash setup_runpod.sh              # interactif
#   bash setup_runpod.sh --auto       # non-interactif (tokens via env vars)
# =============================================================================

set -e

# ─────────────────────────────────────────────
# COULEURS
# ─────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()    { echo -e "${GREEN}[✅]${NC} $1"; }
info()   { echo -e "${BLUE}[ℹ️ ]${NC} $1"; }
warn()   { echo -e "${YELLOW}[⚠️ ]${NC} $1"; }
error()  { echo -e "${RED}[❌]${NC} $1"; exit 1; }
header() { echo -e "\n${CYAN}${BOLD}═══ $1 ═══${NC}\n"; }

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
WORKSPACE=/workspace
MODELS_DIR=$WORKSPACE/models
LORAS_DIR=$WORKSPACE/loras
LOGS_DIR=$WORKSPACE/logs
OUTPUTS_DIR=$WORKSPACE/outputs
SCRIPTS_DIR=$WORKSPACE/Sofia_FLux_Lora/scripts
KOHYA_DIR=$WORKSPACE/kohya_ss
COMFYUI_DIR=$WORKSPACE/ComfyUI
ENV_FILE=$WORKSPACE/.env

# ─────────────────────────────────────────────
# HELPERS HUGGINGFACE (via Python API)
# ─────────────────────────────────────────────
hf_snapshot() {
    # Usage: hf_snapshot <repo_id> <local_dir> [pattern1 pattern2 ...]
    python3 -c "
from huggingface_hub import snapshot_download
import os, sys
repo, local_dir = sys.argv[1], sys.argv[2]
patterns = sys.argv[3:] if len(sys.argv) > 3 else None
snapshot_download(repo, local_dir=local_dir, allow_patterns=patterns,
    token=os.environ.get('HF_TOKEN'),
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*','rust_model*'])
" "$@"
}

hf_file() {
    # Usage: hf_file <repo_id> <filename> <local_dir>
    python3 -c "
from huggingface_hub import hf_hub_download
import os, sys
hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3],
    token=os.environ.get('HF_TOKEN'))
" "$@"
}

# ─────────────────────────────────────────────
# CATALOGUE DES MODÈLES FLUX
# Format : "NOM|SOURCE|ID_OU_URL|DESCRIPTION"
# Sources : HF (HuggingFace) | CIVITAI
# ─────────────────────────────────────────────
declare -A MODELS
MODELS=(
    [1]="flux1-dev|HF|black-forest-labs/FLUX.1-dev|Official Flux.1-dev (28GB) — meilleur pour LoRA"
    [2]="flux1-schnell|HF|black-forest-labs/FLUX.1-schnell|Official Flux.1-schnell (28GB) — plus rapide, 4 steps"
    [3]="flux1-dev-fp8|CIVITAI|361593|Flux.1-dev FP8 optimisé (~16GB) — économise VRAM"
    [4]="flux1-dev-nf4|CIVITAI|363989|Flux.1-dev NF4 quantized (~8GB) — plus léger"
    [5]="fluxed-up-nsfw|CIVITAI_URL|https://civitai.com/api/download/models/2577735?type=Model&format=SafeTensor&size=pruned&fp=fp16|Fluxed Up NSFW — Flux.1-D fp16 pruned — CivitAI 2577735"
    [6]="custom|CUSTOM||Entrer une URL CivitAI ou HF personnalisée"
)

# ─────────────────────────────────────────────
# CHARGER LES TOKENS
# ─────────────────────────────────────────────
load_tokens() {
    # Priorité : env vars existantes > fichier .env
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
        log ".env chargé depuis $ENV_FILE"
    fi

    # Vérifier HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        if [ "$1" == "--auto" ]; then
            error "HF_TOKEN manquant. Définir dans $ENV_FILE ou en variable d'environnement."
        fi
        echo -e "${YELLOW}HuggingFace Token requis pour certains modèles.${NC}"
        echo -e "→ Obtenir sur : ${CYAN}https://huggingface.co/settings/tokens${NC}"
        read -p "HF Token (hf_xxx...) ou Entrée pour passer : " HF_TOKEN
        [ -n "$HF_TOKEN" ] && echo "HF_TOKEN=$HF_TOKEN" >> $ENV_FILE
    else
        log "HF_TOKEN configuré ✓"
    fi

    # Vérifier CIVITAI_TOKEN
    if [ -z "$CIVITAI_TOKEN" ]; then
        if [ "$1" != "--auto" ]; then
            echo -e "${YELLOW}CivitAI Token requis pour les modèles CivitAI.${NC}"
            echo -e "→ Obtenir sur : ${CYAN}https://civitai.com/user/account${NC}"
            read -p "CivitAI Token ou Entrée pour passer : " CIVITAI_TOKEN
            [ -n "$CIVITAI_TOKEN" ] && echo "CIVITAI_TOKEN=$CIVITAI_TOKEN" >> $ENV_FILE
        fi
    else
        log "CIVITAI_TOKEN configuré ✓"
    fi

    export HF_TOKEN CIVITAI_TOKEN
}

# ─────────────────────────────────────────────
# MENU SÉLECTION DU MODÈLE FLUX
# ─────────────────────────────────────────────
select_flux_model() {
    header "Sélection du modèle Flux de base"

    echo -e "  ${BOLD}Modèles disponibles :${NC}\n"
    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r name source model_id desc <<< "${MODELS[$key]}"
        echo -e "  ${CYAN}[$key]${NC} ${BOLD}$name${NC}  (${YELLOW}$source${NC})"
        echo -e "      $desc"
        echo ""
    done

    read -p "Votre choix [1-6] : " MODEL_CHOICE

    IFS='|' read -r FLUX_NAME FLUX_SOURCE FLUX_ID FLUX_DESC <<< "${MODELS[$MODEL_CHOICE]}"

    if [ -z "$FLUX_NAME" ]; then
        error "Choix invalide"
    fi

    # Si custom → demander l'URL
    if [ "$FLUX_SOURCE" == "CUSTOM" ]; then
        echo ""
        echo -e "${YELLOW}Entrez l'URL ou l'ID du modèle :${NC}"
        echo -e "  • HuggingFace : ${CYAN}author/model-name${NC}"
        echo -e "  • CivitAI URL : ${CYAN}https://civitai.com/api/download/models/XXXXXX${NC}"
        echo -e "  • CivitAI ID  : ${CYAN}123456${NC}"
        echo ""
        read -p "URL / ID : " CUSTOM_INPUT

        if [[ "$CUSTOM_INPUT" == https://civitai.com* ]]; then
            FLUX_SOURCE="CIVITAI_URL"
            FLUX_ID="$CUSTOM_INPUT"
            FLUX_NAME="flux-custom"
        elif [[ "$CUSTOM_INPUT" =~ ^[0-9]+$ ]]; then
            FLUX_SOURCE="CIVITAI"
            FLUX_ID="$CUSTOM_INPUT"
            FLUX_NAME="flux-custom-${CUSTOM_INPUT}"
        else
            FLUX_SOURCE="HF"
            FLUX_ID="$CUSTOM_INPUT"
            FLUX_NAME=$(echo $CUSTOM_INPUT | tr '/' '-')
        fi
    fi

    log "Modèle sélectionné : $FLUX_NAME ($FLUX_SOURCE)"
}

# ─────────────────────────────────────────────
# TÉLÉCHARGER MODÈLE FLUX (selon source)
# ─────────────────────────────────────────────
download_flux_model() {
    local out_dir="$MODELS_DIR/flux"
    local out_file="$out_dir/${FLUX_NAME}.safetensors"

    mkdir -p "$out_dir"

    if [ -f "$out_file" ]; then
        log "$FLUX_NAME déjà présent → skip"
        return
    fi

    header "Téléchargement $FLUX_NAME"

    case "$FLUX_SOURCE" in

        HF)
            [ -z "$HF_TOKEN" ] && error "HF_TOKEN requis pour télécharger depuis HuggingFace"
            info "Source : HuggingFace → $FLUX_ID"
            warn "Download ~23-28GB — prendre un café ☕"

            # Chercher le fichier .safetensors principal dans le repo HF
            hf_snapshot "$FLUX_ID" "$out_dir/tmp_$FLUX_NAME" "*.safetensors"

            # Trouver le fichier principal (le plus gros)
            MAIN_FILE=$(find "$out_dir/tmp_$FLUX_NAME" -name "*.safetensors" -not -name "ae.safetensors" | sort -S | tail -1)
            mv "$MAIN_FILE" "$out_file"
            rm -rf "$out_dir/tmp_$FLUX_NAME"
            ;;

        CIVITAI)
            [ -z "$CIVITAI_TOKEN" ] && error "CIVITAI_TOKEN requis pour télécharger depuis CivitAI"
            info "Source : CivitAI model ID → $FLUX_ID"

            CIVITAI_URL="https://civitai.com/api/download/models/${FLUX_ID}?token=${CIVITAI_TOKEN}"

            info "Téléchargement en cours..."
            curl -L \
                --progress-bar \
                -H "Authorization: Bearer $CIVITAI_TOKEN" \
                -o "$out_file" \
                "$CIVITAI_URL"
            ;;

        CIVITAI_URL)
            [ -z "$CIVITAI_TOKEN" ] && error "CIVITAI_TOKEN requis"
            info "Source : CivitAI URL directe"

            # Ajouter le token si pas déjà dans l'URL
            if [[ "$FLUX_ID" != *"token="* ]]; then
                CIVITAI_URL="${FLUX_ID}?token=${CIVITAI_TOKEN}"
            else
                CIVITAI_URL="$FLUX_ID"
            fi

            curl -L \
                --progress-bar \
                -H "Authorization: Bearer $CIVITAI_TOKEN" \
                -o "$out_file" \
                "$CIVITAI_URL"
            ;;
    esac

    # Vérifier le download
    if [ ! -f "$out_file" ] || [ ! -s "$out_file" ]; then
        error "Téléchargement échoué ou fichier vide : $out_file"
    fi

    SIZE=$(du -sh "$out_file" | cut -f1)
    log "$FLUX_NAME téléchargé → $out_file ($SIZE)"

    # Sauvegarder le modèle actif
    echo "ACTIVE_FLUX_MODEL=$out_file" >> $ENV_FILE
    echo "ACTIVE_FLUX_NAME=$FLUX_NAME"  >> $ENV_FILE
    export ACTIVE_FLUX_MODEL="$out_file"
}

# ─────────────────────────────────────────────
# TÉLÉCHARGER LES COMPOSANTS ANNEXES
# ─────────────────────────────────────────────
download_vae() {
    header "VAE Flux (~335MB)"
    VAE_FILE="$MODELS_DIR/vae/ae.safetensors"

    if [ -f "$VAE_FILE" ]; then log "VAE déjà présent → skip"; return; fi

    mkdir -p "$MODELS_DIR/vae"
    [ -z "$HF_TOKEN" ] && error "HF_TOKEN requis pour le VAE"

    hf_file "black-forest-labs/FLUX.1-dev" "ae.safetensors" "$MODELS_DIR/vae"

    log "VAE → $VAE_FILE"
}

download_clip() {
    header "CLIP Text Encoder (~1.7GB)"
    CLIP_DIR="$MODELS_DIR/clip/clip-vit-large-patch14"

    if [ -d "$CLIP_DIR" ] && [ "$(ls -A $CLIP_DIR)" ]; then
        log "CLIP déjà présent → skip"; return
    fi

    mkdir -p "$CLIP_DIR"
    hf_snapshot "openai/clip-vit-large-patch14" "$CLIP_DIR"

    log "CLIP → $CLIP_DIR"
}

download_t5() {
    header "T5-XXL Text Encoder (~9.4GB)"
    T5_DIR="$MODELS_DIR/clip/t5-v1_1-xxl"

    if [ -d "$T5_DIR" ] && [ "$(ls -A $T5_DIR)" ]; then
        log "T5-XXL déjà présent → skip"; return
    fi

    mkdir -p "$T5_DIR"
    hf_snapshot "google/t5-v1_1-xxl" "$T5_DIR" "*.safetensors" "*.json" "*.txt"

    log "T5-XXL → $T5_DIR"
}

# ─────────────────────────────────────────────
# TÉLÉCHARGER LES MODÈLES CONTROLNET FLUX
# ─────────────────────────────────────────────
download_controlnet_flux() {
    header "ControlNet Flux — InstantX (~3.3GB × 2)"

    mkdir -p "$MODELS_DIR/controlnet"

    _dl_cn() {
        local repo="$1"
        local output_name="$2"
        local out="$MODELS_DIR/controlnet/$output_name"

        if [ -f "$out" ]; then
            log "$output_name déjà présent → skip"
            return
        fi

        info "Téléchargement : $repo"
        local tmp="$MODELS_DIR/controlnet/tmp_$(basename $repo)"

        # Continuer même si un modèle est indisponible
        if ! hf_snapshot "$repo" "$tmp" "*.safetensors"; then
            warn "$repo indisponible → skip"
            rm -rf "$tmp"
            return
        fi

        local found
        found=$(find "$tmp" -name "*.safetensors" | head -1)
        if [ -z "$found" ]; then
            warn "Fichier non trouvé pour $repo"
            rm -rf "$tmp"
            return
        fi

        mv "$found" "$out"
        rm -rf "$tmp"
        SIZE=$(du -sh "$out" | cut -f1)
        log "$output_name → $out ($SIZE)"
    }

    _dl_cn "InstantX/FLUX.1-dev-Controlnet-Canny" "flux-controlnet-canny.safetensors"
    _dl_cn "InstantX/FLUX.1-dev-Controlnet-Union" "flux-controlnet-union.safetensors"

    log "ControlNet Flux prêt → $MODELS_DIR/controlnet/"
}

# ─────────────────────────────────────────────
# LISTER LES MODÈLES FLUX DÉJÀ TÉLÉCHARGÉS
# ─────────────────────────────────────────────
list_flux_models() {
    header "Modèles Flux disponibles sur le volume"
    local found=0

    for f in "$MODELS_DIR/flux/"*.safetensors; do
        [ -f "$f" ] || continue
        SIZE=$(du -sh "$f" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $(basename $f)  (${YELLOW}$SIZE${NC})"
        found=1
    done

    [ $found -eq 0 ] && echo -e "  ${YELLOW}Aucun modèle Flux trouvé dans $MODELS_DIR/flux/${NC}"
    echo ""
}

# ─────────────────────────────────────────────
# SÉLECTIONNER UN MODÈLE DÉJÀ TÉLÉCHARGÉ
# ─────────────────────────────────────────────
select_active_model() {
    header "Modèle Flux actif pour l'entraînement"

    mapfile -t EXISTING < <(find "$MODELS_DIR/flux" -name "*.safetensors" 2>/dev/null | sort)

    if [ ${#EXISTING[@]} -eq 0 ]; then
        warn "Aucun modèle Flux présent. Télécharger d'abord."
        return
    fi

    echo -e "  Modèles présents sur le volume :\n"
    for i in "${!EXISTING[@]}"; do
        SIZE=$(du -sh "${EXISTING[$i]}" | cut -f1)
        echo -e "  ${CYAN}[$i]${NC} $(basename ${EXISTING[$i]})  (${YELLOW}$SIZE${NC})"
    done
    echo ""

    read -p "Utiliser quel modèle pour l'entraînement ? [0-$((${#EXISTING[@]}-1))] : " IDX

    SELECTED="${EXISTING[$IDX]}"
    [ -z "$SELECTED" ] && error "Index invalide"

    # Mettre à jour le script d'entraînement
    TRAIN_SCRIPT="$WORKSPACE/Sofia_FLux_Lora/scripts/train_lora.sh"
    if [ -f "$TRAIN_SCRIPT" ]; then
        sed -i "s|MODEL_PATH=.*|MODEL_PATH=\"$SELECTED\"|" "$TRAIN_SCRIPT"
        log "train_lora.sh mis à jour → MODEL_PATH=$SELECTED"
    fi

    # Sauvegarder dans .env
    sed -i '/^ACTIVE_FLUX/d' "$ENV_FILE" 2>/dev/null || true
    echo "ACTIVE_FLUX_MODEL=$SELECTED" >> "$ENV_FILE"
    echo "ACTIVE_FLUX_NAME=$(basename $SELECTED .safetensors)" >> "$ENV_FILE"

    log "Modèle actif : $(basename $SELECTED)"
}

# ─────────────────────────────────────────────
# INSTALLATION KOHYA SS
# ─────────────────────────────────────────────
install_kohya() {
    header "Installation Kohya SS"

    if [ -d "$KOHYA_DIR" ]; then
        info "Kohya SS déjà présent → mise à jour..."
        cd $KOHYA_DIR && git pull -q
        git submodule update --init --recursive -q
    else
        cd $WORKSPACE
        git clone --recurse-submodules https://github.com/bmaltais/kohya_ss.git
    fi

    cd $KOHYA_DIR
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -q --root-user-action=ignore -r requirements.txt
    [ -f requirements_flux.txt ] && pip install -q -r requirements_flux.txt || true

    python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA ✓ — {torch.cuda.get_device_name(0)}')"
    log "Kohya SS prêt → $KOHYA_DIR"
}

# ─────────────────────────────────────────────
# INSTALLATION COMFYUI
# ─────────────────────────────────────────────
install_comfyui() {
    header "Installation ComfyUI"

    if [ -d "$COMFYUI_DIR" ]; then
        info "ComfyUI déjà présent → mise à jour..."
        cd $COMFYUI_DIR && git pull -q
    else
        cd $WORKSPACE
        git clone https://github.com/comfyanonymous/ComfyUI.git
    fi

    # Toujours installer/mettre à jour les dépendances (y compris alembic, sqlalchemy, comfy_aimdo)
    cd $COMFYUI_DIR
    pip install -q --root-user-action=ignore -r requirements.txt

    # Liens symboliques
    mkdir -p $COMFYUI_DIR/models/{unet,vae,clip,loras,controlnet}

    # Lier TOUS les modèles Flux disponibles
    for f in $MODELS_DIR/flux/*.safetensors; do
        [ -f "$f" ] || continue
        LINK="$COMFYUI_DIR/models/unet/$(basename $f)"
        [ ! -f "$LINK" ] && ln -sf "$f" "$LINK" && log "Lien unet → $(basename $f)"
    done

    [ -f "$MODELS_DIR/vae/ae.safetensors" ] && \
        ln -sf "$MODELS_DIR/vae/ae.safetensors" "$COMFYUI_DIR/models/vae/ae.safetensors" 2>/dev/null || true

    [ -d "$MODELS_DIR/clip" ] && \
        ln -sf "$MODELS_DIR/clip" "$COMFYUI_DIR/models/clip" 2>/dev/null || true

    [ -d "$LORAS_DIR" ] && \
        ln -sf "$LORAS_DIR" "$COMFYUI_DIR/models/loras" 2>/dev/null || true

    [ -d "$MODELS_DIR/controlnet" ] && \
        ln -sf "$MODELS_DIR/controlnet" "$COMFYUI_DIR/models/controlnet" 2>/dev/null || true

    # Custom nodes
    NODES_DIR=$COMFYUI_DIR/custom_nodes
    for repo in \
        "https://github.com/ltdrdata/ComfyUI-Manager.git" \
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git" \
        "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git"
    do
        name=$(basename $repo .git)
        [ ! -d "$NODES_DIR/$name" ] && git clone -q $repo $NODES_DIR/$name && log "Node : $name"
    done

    log "ComfyUI prêt → $COMFYUI_DIR"
}

# ─────────────────────────────────────────────
# STRUCTURE DE DOSSIERS
# ─────────────────────────────────────────────
create_structure() {
    mkdir -p $MODELS_DIR/{flux,clip,vae,controlnet}
    mkdir -p $WORKSPACE/dataset/sofia/{raw,processed,captions}
    mkdir -p $LORAS_DIR $LOGS_DIR $OUTPUTS_DIR
    log "Structure créée"
}

# ─────────────────────────────────────────────
# DEPS SYSTÈME
# ─────────────────────────────────────────────
install_deps() {
    apt-get update -qq
    apt-get install -y -qq git curl wget aria2 tree htop tmux 2>/dev/null
    pip install -q --upgrade pip
    pip install -q --upgrade "huggingface_hub[cli]" Pillow tqdm tensorboard accelerate
    # S'assurer que huggingface-cli est dans le PATH
    export PATH="$HOME/.local/bin:$PATH"
    log "Dépendances installées"
}

# ─────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────
final_report() {
    header "Setup terminé"

    list_flux_models

    echo -e "  ${YELLOW}Commandes rapides :${NC}"
    echo -e "  → Entraîner          : bash $WORKSPACE/Sofia_FLux_Lora/scripts/train_lora.sh"
    echo -e "  → Changer de modèle  : bash $0 --select-model"
    echo -e "  → TensorBoard        : tensorboard --logdir $LOGS_DIR --host 0.0.0.0 --port 6006"
    echo -e "  → ComfyUI            : python $COMFYUI_DIR/main.py --listen 0.0.0.0 --port 8188"
    echo ""
}

# ─────────────────────────────────────────────
# MENU PRINCIPAL
# ─────────────────────────────────────────────
main() {
    echo -e "${CYAN}${BOLD}"
    echo "  ╔═══════════════════════════════════════════════╗"
    echo "  ║   Sofia Belmont — LoRA Training Setup        ║"
    echo "  ║   Flux.1-dev + Kohya SS — NVIDIA A40 48GB   ║"
    echo "  ╚═══════════════════════════════════════════════╝"
    echo -e "${NC}"

    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || warn "nvidia-smi non disponible"

    load_tokens "$1"

    # Mode spécial : juste changer le modèle actif
    if [ "$1" == "--select-model" ]; then
        select_active_model
        exit 0
    fi

    # Mode spécial : juste lister les modèles
    if [ "$1" == "--list-models" ]; then
        list_flux_models
        exit 0
    fi

    echo ""
    echo -e "  ${BOLD}Que voulez-vous faire ?${NC}"
    echo ""
    echo -e "  ${CYAN}[1]${NC} Setup complet  (Flux + Kohya + ComfyUI + ControlNet)"
    echo -e "  ${CYAN}[2]${NC} Flux + Kohya   (sans ComfyUI)"
    echo -e "  ${CYAN}[3]${NC} Télécharger un modèle Flux seulement"
    echo -e "  ${CYAN}[4]${NC} Kohya SS seulement"
    echo -e "  ${CYAN}[5]${NC} Changer le modèle Flux actif"
    echo -e "  ${CYAN}[6]${NC} ControlNet Flux seulement (Canny / Depth / Union)"
    echo ""
    read -p "Votre choix [1-6] : " SETUP_CHOICE

    create_structure
    install_deps

    case $SETUP_CHOICE in
        1)
            select_flux_model
            download_flux_model
            download_vae
            download_clip
            download_t5
            install_kohya
            install_comfyui
            download_controlnet_flux
            ;;
        2)
            select_flux_model
            download_flux_model
            download_vae
            download_clip
            download_t5
            install_kohya
            ;;
        3)
            select_flux_model
            download_flux_model
            ;;
        4)
            install_kohya
            ;;
        5)
            select_active_model
            ;;
        6)
            download_controlnet_flux
            ;;
        *)
            error "Choix invalide"
            ;;
    esac

    final_report
}

main "$@"
