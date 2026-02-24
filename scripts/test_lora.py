"""
Sofia Belmont — Test et validation du LoRA
Génère une grille de validation pour choisir le meilleur checkpoint

Usage :
  python test_lora.py                                          # Test interactif
  python test_lora.py --lora /workspace/loras/sofia_v1.safetensors
  python test_lora.py --all-checkpoints                       # Teste tous les checkpoints
"""

# NOTE : Ce script utilise diffusers + peft pour la génération en ligne de commande.
# Pour une utilisation avec ComfyUI, charger le LoRA directement dans l'interface.

import argparse
import sys
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LORAS_DIR   = "/workspace/loras"
OUTPUTS_DIR = "/workspace/outputs"
FLUX_MODEL  = "/workspace/models/flux/flux1-dev.safetensors"

TRIGGER_WORD = "sofia_bel"

# Prompts de validation identité (9 variations)
TEST_PROMPTS = [
    f"{TRIGGER_WORD}, facing camera, neutral expression, natural daylight, photorealistic portrait",
    f"{TRIGGER_WORD}, three-quarter angle, slight smile, warm indoor lighting, photorealistic",
    f"{TRIGGER_WORD}, profile view, looking left, soft backlight, outdoor, photorealistic",
    f"{TRIGGER_WORD}, close-up portrait, intense gaze, studio lighting, sharp details",
    f"{TRIGGER_WORD}, full body, standing, modern interior, confident pose, 8K",
    f"{TRIGGER_WORD}, seated on luxury sofa, legs crossed, evening ambiance, photorealistic",
    f"{TRIGGER_WORD}, outdoor setting, natural sunlight, casual relaxed pose",
    f"{TRIGGER_WORD}, elegant evening look, dark background, cinematic lighting",
    f"{TRIGGER_WORD}, morning light, bedroom, relaxed candid expression, photorealistic",
]

# Forces LoRA à tester
LORA_STRENGTHS = [0.7, 0.85, 1.0]

# Paramètres de génération
GEN_CONFIG = {
    "num_inference_steps": 28,
    "guidance_scale":      3.5,
    "width":               768,
    "height":             1152,   # Portrait 2:3
    "seed":                42,
}


# ─────────────────────────────────────────────
# LISTE DES CHECKPOINTS
# ─────────────────────────────────────────────
def list_checkpoints(loras_dir: str) -> list[Path]:
    path = Path(loras_dir)
    checkpoints = sorted(path.glob("sofia_belmont_v1*.safetensors"))
    return checkpoints


def print_checkpoints(loras_dir: str):
    checkpoints = list_checkpoints(loras_dir)
    if not checkpoints:
        print(f"❌ Aucun checkpoint dans {loras_dir}")
        print("   Lancez l'entraînement d'abord : bash /workspace/scripts/train_lora.sh")
        sys.exit(1)

    print(f"📦 Checkpoints disponibles ({len(checkpoints)}) :")
    for i, cp in enumerate(checkpoints):
        size = cp.stat().st_size / 1e6
        print(f"  [{i}] {cp.name}  ({size:.0f} MB)")

    return checkpoints


# ─────────────────────────────────────────────
# GÉNÉRATION (diffusers)
# ─────────────────────────────────────────────
def generate_with_lora(lora_path: str, output_dir: str):
    """Génère la grille de test pour un checkpoint donné."""
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError:
        print("❌ diffusers non installé : pip install diffusers accelerate")
        sys.exit(1)

    lora_path   = Path(lora_path)
    output_path = Path(output_dir) / lora_path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Chargement Flux.1-dev sur {device}...")

    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL,
        torch_dtype=torch.bfloat16
    ).to(device)

    print(f"🔌 Chargement LoRA : {lora_path.name}")
    pipe.load_lora_weights(str(lora_path))

    generator = torch.Generator(device=device)

    for strength in LORA_STRENGTHS:
        pipe.set_adapters("default", adapter_weights=[strength])

        for i, prompt in enumerate(TEST_PROMPTS):
            generator.manual_seed(GEN_CONFIG["seed"])

            print(f"  🎨 Prompt {i+1}/{len(TEST_PROMPTS)} | strength={strength}")

            image = pipe(
                prompt              = prompt,
                num_inference_steps = GEN_CONFIG["num_inference_steps"],
                guidance_scale      = GEN_CONFIG["guidance_scale"],
                width               = GEN_CONFIG["width"],
                height              = GEN_CONFIG["height"],
                generator           = generator,
            ).images[0]

            filename = f"strength{int(strength*100)}_prompt{i+1:02d}.jpg"
            image.save(output_path / filename, "JPEG", quality=90)

    print(f"\n✅ {len(TEST_PROMPTS) * len(LORA_STRENGTHS)} images générées → {output_path}")


# ─────────────────────────────────────────────
# RAPPORT ComfyUI (sans génération locale)
# ─────────────────────────────────────────────
def print_comfyui_guide(lora_path: str):
    """Affiche les paramètres ComfyUI pour tester le LoRA manuellement."""
    lora_name = Path(lora_path).name

    print(f"""
╔═══════════════════════════════════════════════════════╗
║         Guide Test ComfyUI — Sofia Belmont            ║
╚═══════════════════════════════════════════════════════╝

LoRA : {lora_name}

┌─────────────────────────────────────────────────────┐
│  PARAMÈTRES COMFYUI                                 │
├─────────────────────────────────────────────────────┤
│  Model     : flux1-dev                              │
│  LoRA      : {lora_name:<35}       │
│  Sampler   : euler                                  │
│  Steps     : 28                                     │
│  CFG       : 3.5                                    │
│  Resolution: 768 x 1152 (portrait 2:3)              │
│  Seed      : 42 (fixer pour comparaisons)           │
└─────────────────────────────────────────────────────┘

TESTER CES FORCES LORA :
  • 0.7  → expression naturelle, moins rigide
  • 0.85 → balance identité / généralisation  ← recommandé
  • 1.0  → identité maximale, risque overfitting

PROMPTS DE TEST :
""")
    for i, p in enumerate(TEST_PROMPTS, 1):
        print(f"  [{i}] {p}")

    print(f"""
PROMPTS NÉGATIFS :
  deformed face, blurry, low quality, cartoon, CGI,
  plastic skin, extra limbs, bad anatomy, identity change

ORDRE DE TEST DES CHECKPOINTS :
  1. sofia_belmont_v1-001000 ← souvent optimal
  2. sofia_belmont_v1-000750
  3. sofia_belmont_v1-001250
  4. sofia_belmont_v1-001500 ← si besoin de plus d'identité
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test et validation du LoRA Sofia")
    parser.add_argument("--lora",            default=None,    help="Chemin vers un checkpoint .safetensors")
    parser.add_argument("--all-checkpoints", action="store_true", help="Tester tous les checkpoints")
    parser.add_argument("--list",            action="store_true", help="Lister les checkpoints disponibles")
    parser.add_argument("--output",          default=OUTPUTS_DIR, help="Dossier de sortie")
    parser.add_argument("--guide",           action="store_true", help="Afficher le guide ComfyUI")
    args = parser.parse_args()

    if args.list:
        print_checkpoints(LORAS_DIR)

    elif args.guide:
        lora = args.lora or str(list_checkpoints(LORAS_DIR)[-1]) if list_checkpoints(LORAS_DIR) else "sofia_belmont_v1.safetensors"
        print_comfyui_guide(lora)

    elif args.all_checkpoints:
        checkpoints = print_checkpoints(LORAS_DIR)
        for cp in checkpoints:
            print(f"\n{'='*50}")
            print(f"Testing : {cp.name}")
            generate_with_lora(str(cp), args.output)

    elif args.lora:
        generate_with_lora(args.lora, args.output)

    else:
        # Mode interactif par défaut
        checkpoints = print_checkpoints(LORAS_DIR)
        print()
        print_comfyui_guide(str(checkpoints[-1]))
