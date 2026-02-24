# 🎨 Sofia Belmont — Flux LoRA Training Pipeline

> Setup complet : RunPod + ComfyUI + Flux.1-dev + Kohya LoRA Training

---

## 📋 Table des matières

- [Prérequis](#-prérequis)
- [Setup RunPod](#-setup-runpod)
- [Installation Flux.1-dev](#-installation-flux1-dev)
- [Installation Kohya SS](#-installation-kohya-ss)
- [Préparation Dataset](#-préparation-dataset)
- [Entraînement LoRA](#-entraînement-lora)
- [Installation ComfyUI](#-installation-comfyui)
- [Test du LoRA](#-test-du-lora)
- [Structure du projet](#-structure-du-projet)
- [Troubleshooting](#-troubleshooting)

---

## ✅ Prérequis

| Composant | Requis | Recommandé |
|-----------|--------|------------|
| GPU | RTX 3090 (24GB) | **A100 40GB** |
| VRAM | 24GB min | 40GB |
| Storage | 100GB | **200GB** |
| RunPod plan | Secure Cloud | Secure Cloud |
| Python | 3.10+ | 3.10.x |
| CUDA | 11.8+ | **12.1** |

> 💡 **Conseil coût** : Utiliser A100 pour le training (3-4h, ~8€), puis switcher sur RTX 4090 ($0.69/h) pour la génération.

---

## 🖥️ Setup RunPod

### 1. Créer le Pod

```bash
# Template recommandé sur RunPod
Template  : "RunPod PyTorch 2.1"
GPU       : A100 SXM 40GB  (training)
Disk      : 50GB Container + 150GB Network Volume
Ports     : 8188 (ComfyUI), 7860 (Kohya UI), 22 (SSH)
```

### 2. Monter le Network Volume

```bash
# Toujours monter sur /workspace pour persistance
# Les modèles restent même quand le pod est STOP
ls /workspace
```

### 3. Connexion VS Code (Remote SSH)

```bash
# Dans VS Code → Remote-SSH → Add new host
ssh root@[POD_IP] -p [PORT] -i ~/.ssh/id_rsa

# Ou via le proxy RunPod
ssh [POD_ID]-[PORT]@ssh.runpod.io -i ~/.ssh/id_rsa
```

### 4. Variables d'environnement

```bash
# Ajouter dans ~/.bashrc
export WORKSPACE=/workspace
export MODELS_DIR=/workspace/models
export LORA_DIR=/workspace/loras
export DATASET_DIR=/workspace/dataset/sofia

echo 'export WORKSPACE=/workspace' >> ~/.bashrc
echo 'export MODELS_DIR=/workspace/models' >> ~/.bashrc
source ~/.bashrc
```

---

## 📦 Installation Flux.1-dev

### 1. Créer la structure de dossiers

```bash
mkdir -p /workspace/{models,loras,dataset,outputs,scripts}
mkdir -p /workspace/models/{flux,clip,vae,controlnet}
mkdir -p /workspace/dataset/sofia/{raw,processed,captions}
```

### 2. Télécharger Flux.1-dev

> ⚠️ Requiert un compte Hugging Face avec accès au modèle (gratuit, formulaire à remplir)

```bash
pip install huggingface_hub

# Login HuggingFace
huggingface-cli login
# → Entrer ton token HF (https://huggingface.co/settings/tokens)

# Download Flux.1-dev (28GB — prendre un café ☕)
huggingface-cli download \
  black-forest-labs/FLUX.1-dev \
  --local-dir /workspace/models/flux \
  --include "*.safetensors" "*.json"

# Vérifier le download
ls -lh /workspace/models/flux/
# → flux1-dev.safetensors (~23GB)
```

### 3. Télécharger les composants annexes

```bash
# CLIP Text Encoder (requis pour Flux)
huggingface-cli download \
  openai/clip-vit-large-patch14 \
  --local-dir /workspace/models/clip/clip-vit-large-patch14

# T5 Text Encoder (requis pour Flux)
huggingface-cli download \
  google/t5-v1_1-xxl \
  --local-dir /workspace/models/clip/t5-v1_1-xxl \
  --include "*.safetensors" "*.json" "*.txt"

# VAE Flux
huggingface-cli download \
  black-forest-labs/FLUX.1-dev \
  --local-dir /workspace/models/vae \
  --include "ae.safetensors"
```

---

## 🔧 Installation Kohya SS

> Kohya SS est le framework standard pour entraîner des LoRA Flux.

### 1. Cloner et installer

```bash
cd /workspace

git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# Installer les dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements_flux.txt

# Vérifier l'installation
python -c "import torch; print(torch.cuda.is_available())"
# → True
```

### 2. Lancer l'interface web (optionnel)

```bash
cd /workspace/kohya_ss
python kohya_gui.py --listen 0.0.0.0 --port 7860

# Accessible sur : http://[POD_IP]:7860
```

---

## 🖼️ Préparation Dataset

### Structure requise

```
/workspace/dataset/sofia/
├── raw/              ← Tes photos originales (garder en backup)
├── processed/        ← Photos recadrées 1024x1024
│   ├── sofia_001.jpg
│   ├── sofia_002.jpg
│   └── ...
└── captions/         ← Fichiers .txt (même nom que l'image)
    ├── sofia_001.txt
    ├── sofia_002.txt
    └── ...
```

### Critères de sélection des photos

```
✅ À INCLURE
- 20 à 30 photos minimum
- Résolution : 1024x1024 ou supérieure
- Angles : frontal (40%), 3/4 (40%), profil (20%)
- Éclairages : jour naturel, flash, nuit, studio
- Expressions : neutre, sourire, pout, regard intense
- Tenues : variées (pour ne pas surfit outfit)

❌ À EXCLURE
- Photos floues ou sous-exposées
- Visage partiellement caché
- Lunettes de soleil
- Doublons (angles trop similaires)
- Images avec plusieurs personnes
```

### Script de préparation automatique

```python
# scripts/prepare_dataset.py
from PIL import Image
import os
from pathlib import Path

def crop_center_square(img):
    """Centre-crop vers un carré parfait"""
    width, height = img.size
    min_dim = min(width, height)
    left   = (width - min_dim) // 2
    top    = (height - min_dim) // 2
    right  = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))

def process_dataset(
    input_dir  : str = "/workspace/dataset/sofia/raw",
    output_dir : str = "/workspace/dataset/sofia/processed",
    target_size: int = 1024
):
    Path(output_dir).mkdir(exist_ok=True)
    images = list(Path(input_dir).glob("*.{jpg,jpeg,png,webp}"))
    
    print(f"📸 {len(images)} images trouvées")
    
    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        img = crop_center_square(img)
        img = img.resize((target_size, target_size), Image.LANCZOS)
        
        output_name = f"sofia_{i+1:03d}.jpg"
        img.save(
            Path(output_dir) / output_name,
            "JPEG",
            quality=95
        )
        print(f"  ✅ {output_name}")
    
    print(f"\n✨ Dataset prêt : {len(images)} images dans {output_dir}")

if __name__ == "__main__":
    process_dataset()
```

```bash
python /workspace/scripts/prepare_dataset.py
```

### Générer les captions automatiquement

```python
# scripts/generate_captions.py
# Utilise Florence-2 ou BLIP-2 pour auto-captioning

from pathlib import Path

# Ton trigger word — DOIT être unique et rare
TRIGGER_WORD = "sofia_bel"

# Template de caption
CAPTION_TEMPLATE = (
    "{trigger}, young woman, brunette wavy hair, "
    "tan warm skin, full lips, defined brows, "
    "{description}, photorealistic portrait"
)

# Descriptions manuelles par image (optionnel mais recommandé)
MANUAL_DESCRIPTIONS = {
    "sofia_001": "facing camera, neutral expression, natural light",
    "sofia_002": "three-quarter angle, slight smile, indoor lighting",
    "sofia_003": "profile view, looking left, soft backlight",
    # Ajouter pour chaque image...
}

def generate_captions(
    image_dir  : str = "/workspace/dataset/sofia/processed",
    caption_dir: str = "/workspace/dataset/sofia/captions"
):
    Path(caption_dir).mkdir(exist_ok=True)
    images = list(Path(image_dir).glob("*.jpg"))
    
    for img_path in images:
        stem = img_path.stem  # "sofia_001"
        desc = MANUAL_DESCRIPTIONS.get(stem, "looking at camera")
        
        caption = CAPTION_TEMPLATE.format(
            trigger=TRIGGER_WORD,
            description=desc
        )
        
        caption_file = Path(caption_dir) / f"{stem}.txt"
        caption_file.write_text(caption)
        print(f"  📝 {stem}.txt → {caption[:60]}...")

if __name__ == "__main__":
    generate_captions()
    print("\n✅ Captions générées")
```

```bash
python /workspace/scripts/generate_captions.py
```

---

## 🚀 Entraînement LoRA

### Config d'entraînement recommandée

```bash
# Créer le fichier de config
cat > /workspace/sofia_lora_config.toml << 'EOF'

[general]
enable_bucket         = true
min_bucket_resolution = 512
max_bucket_resolution = 2048

[[datasets]]
resolution         = 1024
batch_size         = 1

  [[datasets.subsets]]
  image_dir       = "/workspace/dataset/sofia/processed"
  caption_extension = ".txt"
  num_repeats     = 10

EOF
```

### Script d'entraînement Flux LoRA

```bash
# scripts/train_lora.sh

#!/bin/bash

export PYTHONPATH=/workspace/kohya_ss:$PYTHONPATH

python /workspace/kohya_ss/flux_train_network.py \
  --pretrained_model_name_or_path="/workspace/models/flux/flux1-dev.safetensors" \
  --clip_l="/workspace/models/clip/clip-vit-large-patch14" \
  --t5xxl="/workspace/models/clip/t5-v1_1-xxl" \
  --ae="/workspace/models/vae/ae.safetensors" \
  --dataset_config="/workspace/sofia_lora_config.toml" \
  \
  --output_dir="/workspace/loras" \
  --output_name="sofia_belmont_v1" \
  \
  --network_module="networks.lora_flux" \
  --network_dim=32 \
  --network_alpha=32 \
  --network_train_unet_only \
  \
  --optimizer_type="adamw8bit" \
  --learning_rate=5e-4 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100 \
  \
  --max_train_steps=1500 \
  --save_every_n_steps=250 \
  --save_model_as="safetensors" \
  \
  --mixed_precision="bf16" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --sdpa \
  \
  --timestep_sampling="sigmoid" \
  --model_prediction_type="raw" \
  --discrete_flow_shift=3.1582 \
  --loss_type="l2" \
  \
  --log_with="tensorboard" \
  --logging_dir="/workspace/logs" \
  \
  --seed=42

echo "✅ Training terminé → /workspace/loras/sofia_belmont_v1.safetensors"
```

```bash
chmod +x /workspace/scripts/train_lora.sh
bash /workspace/scripts/train_lora.sh
```

### Suivi de l'entraînement

```bash
# Dans un terminal séparé — TensorBoard
tensorboard --logdir /workspace/logs --host 0.0.0.0 --port 6006
# Accessible sur : http://[POD_IP]:6006

# Surveiller la VRAM
watch -n 5 nvidia-smi

# Durée estimée sur A100 40GB
# → 1500 steps, batch 1 : ~2.5h
# → Coût : ~5€ (A100 @ $1.99/h)
```

### Checkpoints — Choisir le meilleur

```
/workspace/loras/
├── sofia_belmont_v1-000250.safetensors  ← Underfitting
├── sofia_belmont_v1-000500.safetensors  ← Tester
├── sofia_belmont_v1-000750.safetensors  ← Tester  
├── sofia_belmont_v1-001000.safetensors  ← Souvent optimal ✅
├── sofia_belmont_v1-001250.safetensors  ← Tester
└── sofia_belmont_v1-001500.safetensors  ← Risque overfitting
```

> 💡 **Règle d'or** : tester les checkpoints 750, 1000, 1250 et garder celui qui a la meilleure balance identité/généralisation.

---

## 🎨 Installation ComfyUI

```bash
cd /workspace

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

pip install -r requirements.txt

# Lier les modèles déjà téléchargés
ln -s /workspace/models/flux/flux1-dev.safetensors \
      /workspace/ComfyUI/models/unet/flux1-dev.safetensors

ln -s /workspace/models/vae/ae.safetensors \
      /workspace/ComfyUI/models/vae/ae.safetensors

ln -s /workspace/models/clip \
      /workspace/ComfyUI/models/clip

ln -s /workspace/loras \
      /workspace/ComfyUI/models/loras

# Installer les custom nodes essentiels
cd /workspace/ComfyUI/custom_nodes

git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Lancer ComfyUI
cd /workspace/ComfyUI
python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --enable-cors-header \
  --preview-method auto

# Accessible sur : http://[POD_IP]:8188
```

---

## 🧪 Test du LoRA

### Prompt de test Sofia

```
Positive:
sofia_bel, young woman, brunette wavy hair, tan warm skin, 
full lips, defined brows, sitting on luxury sofa, 
looking at camera, confident expression,
photorealistic, 8K, sharp details, natural skin texture,
film photography aesthetic, no CGI look

Negative:
deformed face, blurry, low quality, cartoon, CGI, 
plastic skin, artificial, extra limbs, bad anatomy,
face alteration, identity change
```

### Paramètres ComfyUI pour test

```
Model      : flux1-dev
LoRA       : sofia_belmont_v1 (strength: 0.85)
Sampler    : euler
Steps      : 28
CFG        : 3.5
Resolution : 768x1152 (portrait 2:3)
Seed       : fixe pour comparaisons (ex: 42)
```

### Grille de validation identité

```python
# scripts/test_lora.py
# Génère une grille de 9 images pour valider la cohérence

TEST_PROMPTS = [
    "sofia_bel, facing camera, neutral expression, natural light",
    "sofia_bel, three-quarter angle, slight smile, warm light",
    "sofia_bel, profile view, looking away, moody shadows",
    "sofia_bel, close-up portrait, intense gaze, studio light",
    "sofia_bel, full body, standing, modern interior",
    "sofia_bel, seated on sofa, legs crossed, evening ambiance",
    "sofia_bel, outdoor, sunlight, casual pose",
    "sofia_bel, nightlife glam, dark background, flash light",
    "sofia_bel, luxury bedroom, morning light, relaxed pose",
]

LORA_STRENGTHS_TO_TEST = [0.7, 0.85, 1.0]

# → Générer et comparer visuellement les 27 variations
# → Choisir la force LoRA optimale pour Sofia
```

---

## 📁 Structure du projet

```
/workspace/
├── 📂 models/
│   ├── flux/
│   │   └── flux1-dev.safetensors        (23GB)
│   ├── clip/
│   │   ├── clip-vit-large-patch14/      (1.7GB)
│   │   └── t5-v1_1-xxl/                (9.4GB)
│   ├── vae/
│   │   └── ae.safetensors               (335MB)
│   └── controlnet/                      (à remplir)
│
├── 📂 loras/
│   ├── sofia_belmont_v1-001000.safetensors  ✅ Best
│   └── sofia_belmont_v1-001500.safetensors
│
├── 📂 dataset/
│   └── sofia/
│       ├── raw/          (photos originales)
│       ├── processed/    (1024x1024 préparées)
│       └── captions/     (fichiers .txt)
│
├── 📂 scripts/
│   ├── prepare_dataset.py
│   ├── generate_captions.py
│   ├── train_lora.sh
│   ├── test_lora.py
│   └── batch_generate.py   (étape suivante)
│
├── 📂 ComfyUI/              (génération images)
├── 📂 kohya_ss/             (training)
├── 📂 outputs/              (images/vidéos générées)
├── 📂 logs/                 (tensorboard)
│
└── sofia_lora_config.toml
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Réduire batch size dans le .toml
batch_size = 1  # Déjà à 1, ne pas augmenter

# Activer gradient checkpointing (déjà dans le script)
--gradient_checkpointing

# Réduire la résolution temporairement
resolution = 768  # au lieu de 1024

# Vider le cache CUDA entre les runs
python -c "import torch; torch.cuda.empty_cache()"
```

### LoRA ne ressemble pas au personnage

```bash
# Augmenter les steps
--max_train_steps=2000   # au lieu de 1500

# Augmenter les repeats dans le .toml
num_repeats = 15         # au lieu de 10

# Vérifier les captions — le trigger word DOIT être en premier
# ✅ "sofia_bel, young woman..."
# ❌ "young woman... sofia_bel"

# Vérifier la qualité du dataset
# → Supprimer les photos floues ou trop similaires
```

### LoRA trop fort (overfitting)

```bash
# Tester un checkpoint antérieur
# sofia_belmont_v1-000750.safetensors

# Réduire le network_dim
--network_dim=16   # au lieu de 32

# Réduire la force dans ComfyUI
lora_strength = 0.7  # au lieu de 0.85
```

### Kohya ne trouve pas Flux

```bash
# Vérifier les chemins
ls -lh /workspace/models/flux/flux1-dev.safetensors

# Le fichier doit être un .safetensors complet (~23GB)
# Si taille incorrecte → re-télécharger
du -sh /workspace/models/flux/flux1-dev.safetensors
```

### Port ComfyUI inaccessible

```bash
# Vérifier que le port 8188 est bien exposé dans RunPod
# Settings du pod → "Expose HTTP Ports" → ajouter 8188

# Ou utiliser le tunnel RunPod
# Dans l'interface RunPod → Connect → "Connect to HTTP Service [8188]"
```

---

## 📊 Estimation coûts & temps

| Étape | GPU | Durée | Coût estimé |
|-------|-----|-------|-------------|
| Download modèles | A100 | 30-45min | ~$1.50 |
| Install Kohya | A100 | 15min | ~$0.50 |
| Préparer dataset | A100 | 10min | ~$0.33 |
| **Training LoRA** | **A100** | **2.5-3h** | **~$6** |
| Test ComfyUI | RTX 4090 | 30min | ~$0.35 |
| **Total one-time** | | **~4h** | **~$9** |

> 💡 **Storage** : Après setup, passer en pod STOP. Seul le network volume ($7/mois) est facturé. Le pod redémarre en ~2min avec tout en place.

---

## ➡️ Prochaines étapes

Une fois le LoRA validé :

```
[ ] Intégrer sofia_lora dans le workflow WAN 2.2 existant
[ ] Installer ControlNet DWPose pour contrôle des poses
[ ] Créer le script batch_generate.py (pose × outfit × décor)
[ ] Connecter le pipeline avec les 20 poses référence
[ ] Automatiser l'export vers les formats Reels/Stories
```

---

*Sofia Belmont Project — LoRA Pipeline v1.0*  
*Stack : RunPod + Flux.1-dev + Kohya SS + ComfyUI*
