"""
Sofia Belmont — Préparation du Dataset
Redimensionne les images en conservant le ratio d'origine (bucket-friendly).
Compatible 9:16, 4:3, 1:1, etc. — Kohya gère les buckets automatiquement.

Usage :
  python prepare_dataset.py                   # 9:16 par défaut (768x1365)
  python prepare_dataset.py --square          # force carré 1024x1024 (crop)
  python prepare_dataset.py --max-side 1024   # côté max personnalisé
"""

from PIL import Image
from pathlib import Path
import sys


INPUT_DIR    = "/workspace/dataset/sofia/raw"
OUTPUT_DIR   = "/workspace/dataset/sofia/processed"
MAX_SIDE     = 1024   # le côté le plus long sera ramené à cette valeur
BUCKET_ALIGN = 64     # arrondir W et H au multiple de 64 (requis par Kohya)


def resize_keep_ratio(img: Image.Image, max_side: int) -> Image.Image:
    """Redimensionne en gardant le ratio — côté le plus long = max_side."""
    w, h  = img.size
    scale = max_side / max(w, h)
    nw    = round(w * scale / BUCKET_ALIGN) * BUCKET_ALIGN
    nh    = round(h * scale / BUCKET_ALIGN) * BUCKET_ALIGN
    return img.resize((nw, nh), Image.LANCZOS)


def crop_center_square(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop vers un carré parfait (mode --square)."""
    w, h    = img.size
    min_dim = min(w, h)
    left    = (w - min_dim) // 2
    top     = (h - min_dim) // 2
    img     = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((size, size), Image.LANCZOS)


def process_dataset(
    input_dir:  str  = INPUT_DIR,
    output_dir: str  = OUTPUT_DIR,
    max_side:   int  = MAX_SIDE,
    square:     bool = False,
):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Dossier source introuvable : {input_dir}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
    images     = sorted(f for f in input_path.iterdir() if f.suffix in extensions)

    if not images:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        sys.exit(1)

    mode = f"carré {max_side}×{max_side} (crop)" if square else f"ratio conservé (côté max {max_side}px, aligné ×{BUCKET_ALIGN})"
    print(f"📸 {len(images)} images trouvées")
    print(f"   Mode     : {mode}")
    print(f"   Entrée   : {input_dir}")
    print(f"   Sortie   : {output_dir}")
    print()

    ok, skip, fail = 0, 0, 0

    for i, img_path in enumerate(images, start=1):
        output_name = f"sofia_{i:03d}.jpg"
        output_file = output_path / output_name

        if output_file.exists():
            print(f"  ⏭️  {output_name} déjà présent, skip")
            skip += 1
            continue

        try:
            img  = Image.open(img_path).convert("RGB")
            orig = img.size

            if square:
                img = crop_center_square(img, max_side)
            else:
                img = resize_keep_ratio(img, max_side)

            img.save(output_file, "JPEG", quality=95, optimize=True)
            print(f"  ✅ {output_name}  ({orig[0]}×{orig[1]} → {img.size[0]}×{img.size[1]})")
            ok += 1

        except Exception as e:
            print(f"  ❌ Erreur sur {img_path.name} : {e}")
            fail += 1

    print()
    print(f"✨ Terminé : {ok} converties, {skip} skippées, {fail} erreurs")
    print(f"   Dataset prêt dans : {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prépare le dataset Sofia pour LoRA")
    parser.add_argument("--input",    default=INPUT_DIR, help="Dossier source")
    parser.add_argument("--output",   default=OUTPUT_DIR, help="Dossier sortie")
    parser.add_argument("--max-side", default=MAX_SIDE, type=int, help="Côté max en pixels (défaut: 1024)")
    parser.add_argument("--square",   action="store_true", help="Force carré 1024×1024 avec crop")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.max_side, args.square)
