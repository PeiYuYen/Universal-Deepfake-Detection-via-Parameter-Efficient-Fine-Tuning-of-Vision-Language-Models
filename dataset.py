import os
import shutil
import random
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, json_path, root_dir):
        with open(json_path, "r") as f:
            self.samples = json.load(f)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                                 std=[0.2686, 0.2613, 0.2758])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample["path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = sample["label"]

        filename = os.path.basename(sample["path"])    
        video_id = filename.split("_")[0]
        return {
        "image": image,
        "label": label,
        "video_id": video_id
        }
    
random.seed(42)

CLASS_TO_LABEL = {"real": 0, "fake": 1}

def collect_video_folders(src_dir):
    return [p for p in Path(src_dir).iterdir() if p.is_dir()]

def split_and_copy(video_dirs, split_ratios, src_root, dst_root, class_name, split_records):
    random.shuffle(video_dirs)
    n_total = len(video_dirs)
    n_train = int(split_ratios["train"] * n_total)
    n_val = int(split_ratios["val"] * n_total)
    n_test = n_total - n_train - n_val

    splits = (
        ("train", video_dirs[:n_train]),
        ("val", video_dirs[n_train:n_train + n_val]),
        ("test", video_dirs[n_train + n_val:])
    )

    for split_name, split_videos in splits:
        print(f"{split_name.upper()} ({class_name}) - {len(split_videos)} videos")
        for video_dir in tqdm(split_videos, desc=f"Processing {split_name}/{class_name}"):
            image_paths = list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg"))
            out_dir = Path(dst_root) / "images" / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(image_paths, desc=f"  Copying {video_dir.name}", leave=False):
                dst_name = f"{video_dir.name}_{img_path.name}"
                dst_path = out_dir / dst_name
                shutil.copy(img_path, dst_path)

                split_records[split_name].append({
                    "path": str(dst_path.relative_to(dst_root)),
                    "label": CLASS_TO_LABEL[class_name]
                })

def copy_all_video_frames(src_dir, dst_root, split_name, class_name, split_records):
    video_dirs = collect_video_folders(src_dir)
    print(f"{split_name.upper()} ({class_name}) - {len(video_dirs)} videos")
    for video_dir in tqdm(video_dirs, desc=f"Processing {split_name}/{class_name}"):
        image_paths = list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg"))
        out_dir = Path(dst_root) / "images" / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(image_paths, desc=f"  Copying {video_dir.name}", leave=False):
            dst_name = f"{video_dir.name}_{img_path.name}"
            dst_path = out_dir / dst_name
            shutil.copy(img_path, dst_path)

            split_records[split_name].append({
                "path": str(dst_path.relative_to(dst_root)),
                "label": CLASS_TO_LABEL[class_name]
            })

if __name__ == "__main__":
    src_root = "data"
    dst_root = "data/data_split"

    split_records = {"train": [], "val": [], "test": []}

    # 切分 Real_youtube: 80% train, 10% val, 10% test
    real_videos = collect_video_folders(f"{src_root}/Real_youtube")
    split_and_copy(real_videos, {"train": 0.8, "val": 0.1, "test": 0.1}, src_root, dst_root, "real", split_records)

    # 切分 FaceSwap: 90% train, 10% val
    fake_videos = collect_video_folders(f"{src_root}/FaceSwap")
    split_and_copy(fake_videos, {"train": 0.9, "val": 0.1, "test": 0.0}, src_root, dst_root, "fake", split_records)

    # NeuralTextures 全部作為 test/fake
    copy_all_video_frames(f"{src_root}/NeuralTextures", dst_root, "test", "fake", split_records)

    # 儲存 json list
    for split in ["train", "val", "test"]:
        with open(Path(dst_root) / f"{split}_list.json", "w") as f:
            json.dump(split_records[split], f, indent=2)

    # 儲存 classnames.txt
    with open(Path(dst_root) / "classnames.txt", "w") as f:
        f.write("real\nfake\n")

    # 印出分布 summary
    print("\n=== Split Summary ===")
    for split in ["train", "val", "test"]:
        count_real = sum(1 for item in split_records[split] if item["label"] == CLASS_TO_LABEL["real"])
        count_fake = sum(1 for item in split_records[split] if item["label"] == CLASS_TO_LABEL["fake"])
        print(f"{split.upper()}: real={count_real}, fake={count_fake}, total={len(split_records[split])}")

    print("\n完成資料切分與標註輸出")
