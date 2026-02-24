"""
Sofia Belmont — Génération des Captions
Génère les fichiers .txt pour chaque image du dataset

Usage :
  python generate_captions.py                    # Mode manuel (template)
  python generate_captions.py --auto             # Mode auto (Florence-2)
  python generate_captions.py --check            # Vérifier les captions existantes
"""

from pathlib import Path
import argparse
import sys


# ─────────────────────────────────────────────
# CONFIG — À ADAPTER
# ─────────────────────────────────────────────
TRIGGER_WORD   = "sofia_bel"
IMAGE_DIR      = "/workspace/dataset/sofia/processed"
CAPTION_DIR    = "/workspace/dataset/sofia/captions"

# Caractéristiques physiques constantes de Sofia
# → Ces tokens sont toujours présents dans chaque caption
CONSTANT_TRAITS = (
    "young woman, brunette wavy hair, "
    "tan warm skin, full lips, defined brows"
)

# Template de base — {trigger} et {description} sont remplacés
CAPTION_TEMPLATE = (
    "{trigger}, {traits}, {description}, photorealistic portrait"
)

# ─────────────────────────────────────────────
# DESCRIPTIONS MANUELLES PAR IMAGE
# Remplir au fur et à mesure — laisser vide pour utiliser "looking at camera"
# ─────────────────────────────────────────────
MANUAL_DESCRIPTIONS: dict[str, str] = {
    # "sofia_001": "facing camera, neutral expression, natural light, indoor",
    # "sofia_002": "three-quarter angle, slight smile, warm indoor lighting",
    # "sofia_003": "profile view, looking left, soft backlight, outdoor",
    # "sofia_004": "close-up portrait, intense gaze, studio lighting",
    # "sofia_005": "full body, standing, modern interior, confident pose",
    # ...
    # Ajouter les descriptions selon vos photos
}

DEFAULT_DESCRIPTION = "looking at camera, natural expression, soft lighting"


# ─────────────────────────────────────────────
# MODE MANUEL (template)
# ─────────────────────────────────────────────
def generate_manual(image_dir: str, caption_dir: str, overwrite: bool = False):
    image_path   = Path(image_dir)
    caption_path = Path(caption_dir)
    caption_path.mkdir(parents=True, exist_ok=True)

    images = sorted(image_path.glob("*.jpg")) + sorted(image_path.glob("*.png"))

    if not images:
        print(f"❌ Aucune image dans {image_dir}")
        sys.exit(1)

    print(f"📝 Génération des captions pour {len(images)} images")
    print(f"   Trigger word : {TRIGGER_WORD}")
    print()

    ok, skip = 0, 0

    for img_path in images:
        stem         = img_path.stem       # "sofia_001"
        caption_file = caption_path / f"{stem}.txt"

        if caption_file.exists() and not overwrite:
            print(f"  ⏭️  {stem}.txt déjà présent, skip")
            skip += 1
            continue

        description = MANUAL_DESCRIPTIONS.get(stem, DEFAULT_DESCRIPTION)

        caption = CAPTION_TEMPLATE.format(
            trigger     = TRIGGER_WORD,
            traits      = CONSTANT_TRAITS,
            description = description
        )

        caption_file.write_text(caption, encoding="utf-8")
        print(f"  ✅ {stem}.txt")
        print(f"     → {caption[:80]}...")
        ok += 1

    print()
    print(f"✨ Terminé : {ok} créées, {skip} skippées")
    print(f"   Captions dans : {caption_dir}")


# ─────────────────────────────────────────────
# MODE AUTO (Florence-2)
# ─────────────────────────────────────────────
def generate_auto(image_dir: str, caption_dir: str, overwrite: bool = False):
    print("🤖 Mode auto — Chargement de Florence-2...")

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from PIL import Image
    except ImportError:
        print("❌ Dépendances manquantes. Installer : pip install transformers Pillow torch")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device : {device}")

    model_id = "microsoft/Florence-2-large"
    print(f"   Chargement {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    image_path   = Path(image_dir)
    caption_path = Path(caption_dir)
    caption_path.mkdir(parents=True, exist_ok=True)

    images = sorted(image_path.glob("*.jpg")) + sorted(image_path.glob("*.png"))
    print(f"\n📸 {len(images)} images à traiter\n")

    for img_path in images:
        stem         = img_path.stem
        caption_file = caption_path / f"{stem}.txt"

        if caption_file.exists() and not overwrite:
            print(f"  ⏭️  {stem}.txt déjà présent, skip")
            continue

        img    = Image.open(img_path).convert("RGB")
        inputs = processor(text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids          = inputs["input_ids"],
                pixel_values       = inputs["pixel_values"],
                max_new_tokens     = 256,
                do_sample          = False,
                num_beams          = 3,
            )

        auto_description = processor.batch_decode(generated, skip_special_tokens=True)[0]
        # Nettoyer la description générée
        auto_description = auto_description.replace("<MORE_DETAILED_CAPTION>", "").strip()

        # Préfixer avec trigger et traits constants
        caption = f"{TRIGGER_WORD}, {CONSTANT_TRAITS}, {auto_description}"

        caption_file.write_text(caption, encoding="utf-8")
        print(f"  ✅ {stem}.txt → {caption[:80]}...")

    print(f"\n✨ Captions auto générées dans {caption_dir}")


# ─────────────────────────────────────────────
# VÉRIFICATION DES CAPTIONS
# ─────────────────────────────────────────────
def check_captions(image_dir: str, caption_dir: str):
    image_path   = Path(image_dir)
    caption_path = Path(caption_dir)

    images   = set(p.stem for p in image_path.glob("*.jpg")) | set(p.stem for p in image_path.glob("*.png"))
    captions = set(p.stem for p in caption_path.glob("*.txt"))

    print(f"📊 Vérification des captions")
    print(f"   Images   : {len(images)}")
    print(f"   Captions : {len(captions)}")
    print()

    missing_captions = images - captions
    orphan_captions  = captions - images

    if missing_captions:
        print(f"⚠️  Images sans caption ({len(missing_captions)}) :")
        for name in sorted(missing_captions):
            print(f"   → {name}")
    else:
        print("✅ Toutes les images ont une caption")

    if orphan_captions:
        print(f"\n⚠️  Captions sans image ({len(orphan_captions)}) :")
        for name in sorted(orphan_captions):
            print(f"   → {name}")

    # Vérifier le trigger word
    print(f"\n🔍 Vérification du trigger word '{TRIGGER_WORD}' :")
    errors = 0
    for cap_file in sorted(caption_path.glob("*.txt")):
        content = cap_file.read_text(encoding="utf-8")
        if not content.startswith(TRIGGER_WORD):
            print(f"  ❌ {cap_file.name} — ne commence pas par '{TRIGGER_WORD}'")
            errors += 1

    if errors == 0:
        print(f"  ✅ Toutes les captions commencent par '{TRIGGER_WORD}'")
    else:
        print(f"  ⚠️  {errors} captions incorrectes")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère les captions du dataset Sofia")
    parser.add_argument("--image-dir",   default=IMAGE_DIR,   help="Dossier images")
    parser.add_argument("--caption-dir", default=CAPTION_DIR, help="Dossier captions")
    parser.add_argument("--auto",        action="store_true", help="Mode auto (Florence-2)")
    parser.add_argument("--check",       action="store_true", help="Vérifier les captions")
    parser.add_argument("--overwrite",   action="store_true", help="Écraser les captions existantes")
    args = parser.parse_args()

    if args.check:
        check_captions(args.image_dir, args.caption_dir)
    elif args.auto:
        generate_auto(args.image_dir, args.caption_dir, args.overwrite)
    else:
        generate_manual(args.image_dir, args.caption_dir, args.overwrite)
