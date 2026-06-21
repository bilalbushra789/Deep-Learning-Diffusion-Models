# =====================================================================
# SECTION 1: DATA LOADER
# =====================================================================
#
# Reads images from a folder structured as:
#     root/
#         ClassA/
#             img1.jpg
#             img2.jpg
#         ClassB/
#             ...
#
# Picks a subset of classes (default 5) and a subset of images per class
# (default 20),
# " select a portion of images for your training (let's say 20
# images) from any 5 classes".
#
# Images are resized and scaled to [-1, 1], the standard input range
# used for DDPM training (matches the Gaussian prior N(0, I) used in
# the forward process).

import os
import random
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_available_classes(root_dir):
    """Return sorted list of class (sub-folder) names found in root_dir."""
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


class AnimalDiffusionDataset(Dataset):
    """
    Picks `num_classes` classes and `images_per_class` images per class
    from `root_dir`, and serves them as normalized tensors in [-1, 1].

    Parameters
    ----------
    root_dir : str
        Path to the folder that contains one sub-folder per animal class.
    image_size : int
        Output (square) resolution fed to the model.
    num_classes : int
        How many classes to sample from (assignment default: 5).
    images_per_class : int
        How many images to take from each chosen class (assignment default: 20).
    selected_classes : list[str] or None
        If given, use exactly these class names instead of randomly picking
        `num_classes` of them. Useful for reproducibility.
    seed : int
        Random seed used when randomly choosing classes/images.
    """

    def __init__(
        self,
        root_dir,
        image_size=64,
        num_classes=5,
        images_per_class=20,
        selected_classes=None,
        seed=42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        rng = random.Random(seed)

        all_classes = list_available_classes(root_dir)
        if len(all_classes) == 0:
            raise RuntimeError(f"No class folders found inside {root_dir}")

        if selected_classes is not None:
            self.classes = list(selected_classes)
        else:
            self.classes = sorted(rng.sample(all_classes, k=min(num_classes, len(all_classes))))

        self.image_paths = []
        self.labels = []
        for class_idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            files = [
                f for f in sorted(glob(os.path.join(class_dir, "*")))
                if f.lower().endswith(IMG_EXTENSIONS)
            ]
            rng.shuffle(files)
            chosen = files[:images_per_class]
            if len(chosen) < images_per_class:
                print(
                    f"[WARN] class '{cls}' only has {len(chosen)} images "
                    f"(< requested {images_per_class})."
                )
            self.image_paths.extend(chosen)
            self.labels.extend([class_idx] * len(chosen))

        if len(self.image_paths) == 0:
            raise RuntimeError("No images collected — check root_dir / class names.")

        # Necessary transformations before feeding data to the diffusion process:
        #   1. Resize to a fixed square size
        #   2. Convert to tensor in [0, 1]
        #   3. Normalize to [-1, 1] (mean=0.5, std=0.5 per channel)
        #   4. Light augmentation (random horizontal flip) since training set is tiny
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def class_name(self, label_idx):
        return self.classes[label_idx]
