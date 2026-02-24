"""
Sofia Belmont — Préparation du Dataset
Recadre et redimensionne les images vers 1024x1024
"""

from PIL import Image
from pathlib import Path
import shutil
import sys


INPUT_DIR   = "/workspace/dataset/sofia/raw"
OUTPUT_DIR  = "/workspace/dataset/sofia/processed"
TARGET_SIZE = 1024


def crop_center_square(img: Image.Image) -> Image.Image:
    """Centre-crop vers un carré parfait."""
    width, height = img.size
    min_dim = min(width, height)
    left   = (width  - min_dim) // 2
    top    = (height - min_dim) // 2
    return img.crop((left, top, left + min_dim, top + min_dim))


def process_dataset(
    input_dir:   str = INPUT_DIR,
    output_dir:  str = OUTPUT_DIR,
    target_size: int = TARGET_SIZE
):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Dossier source introuvable : {input_dir}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
    images = [f for f in input_path.iterdir() if f.suffix in extensions]

    if not images:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        sys.exit(1)

    print(f"📸 {len(images)} images trouvées dans {input_dir}")
    print(f"   → Sortie : {output_dir} ({target_size}x{target_size})")
    print()

    ok, skip, fail = 0, 0, 0

    for i, img_path in enumerate(sorted(images), start=1):
        output_name = f"sofia_{i:03d}.jpg"
        output_file = output_path / output_name

        # Skip si déjà traité
        if output_file.exists():
            print(f"  ⏭️  {output_name} déjà présent, skip")
            skip += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            original_size = img.size

            img = crop_center_square(img)
            img = img.resize((target_size, target_size), Image.LANCZOS)
            img.save(output_file, "JPEG", quality=95, optimize=True)

            print(f"  ✅ {output_name}  ({original_size[0]}×{original_size[1]} → {target_size}×{target_size})")
            ok += 1

        except Exception as e:
            print(f"  ❌ Erreur sur {img_path.name} : {e}")
            fail += 1

    print()
    print(f"✨ Terminé : {ok} converties, {skip} skippées, {fail} erreurs")
    print(f"   Dataset prêt dans : {output_dir}")

    if ok + skip == 0:
        print("⚠️  Aucune image traitée — vérifiez le dossier raw/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prépare le dataset Sofia pour LoRA")
    parser.add_argument("--input",  default=INPUT_DIR,   help=f"Dossier source (défaut: {INPUT_DIR})")
    parser.add_argument("--output", default=OUTPUT_DIR,  help=f"Dossier sortie (défaut: {OUTPUT_DIR})")
    parser.add_argument("--size",   default=TARGET_SIZE, type=int, help=f"Taille cible (défaut: {TARGET_SIZE})")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.size)
