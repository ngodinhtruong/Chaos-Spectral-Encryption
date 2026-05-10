import os
from pathlib import Path
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
IMG_SIZE = 224

PREP = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize(int(IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
])

def _process_one(args):
    src_path, dst_path = args
    try:
        dst_path = Path(dst_path)
        if dst_path.exists():
            return True

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as im:
            im = PREP(im)

            # lưu PNG (lossless)
            im.save(dst_path, format="PNG", compress_level=0)

        return True
    except Exception:
        return False

def preprocess_split_mp(src_dir, out_dir, max_workers=None):

    src_dir = str(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ImageFolder(src_dir)
    classes = ds.classes

    tasks = []
    for idx, (path, label) in enumerate(ds.samples):
        cls_name = classes[label]

        # đổi đuôi sang png
        dst = out_dir / cls_name / f"{idx:06d}.png"

        tasks.append((path, str(dst)))

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    ok = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(
            ex.map(_process_one, tasks, chunksize=64),
            total=len(tasks),
            desc=f"Preprocess {Path(src_dir).name} (workers={max_workers})"
        ):
            ok += int(bool(res))

    print(f"Done {Path(src_dir).name}: {ok}/{len(tasks)} files ok. Saved to: {out_dir}")
    return classes