import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from collections import defaultdict

from model import CLIPLoRADetector, CLIPLinearProbe 
from dataset import DeepfakeDataset 

def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    frame_results = []
    video_map = defaultdict(lambda: {"scores": [], "labels": []})

    for batch in tqdm(loader, desc="Testing"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()
        video_ids = batch["video_id"]

        # labels = labels.view(-1, 1)  # 如果用原始模型CLIPLinearProbe

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze()

        for vid, prob, label in zip(video_ids, probs.cpu().numpy(), labels.cpu().numpy()):
            frame_results.append({
                "video_id": vid,
                "score": float(prob),
                "label": int(label)
            })
            video_map[vid]["scores"].append(prob)
            video_map[vid]["labels"].append(label)

    # --- Frame-level metrics ---
    all_labels = [f["label"] for f in frame_results]
    all_scores = [f["score"] for f in frame_results]
    all_preds  = [1 if s > 0.5 else 0 for s in all_scores]

    auc_frame = roc_auc_score(all_labels, all_scores)
    f1_frame  = f1_score(all_labels, all_preds)
    acc_frame = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    eer_frame  = compute_eer(fpr, tpr)

    # --- Video-level metrics (mean/max) ---
    video_avg_scores, video_labels = [], []
    for vid, v in video_map.items():
        mean_score = np.mean(v["scores"])
        max_score  = np.max(v["scores"])
        # Assume all labels in same video are same
        video_labels.append(int(v["labels"][0]))
        video_avg_scores.append(float(mean_score))  # use mean here; can switch to max

    video_preds = [1 if s > 0.5 else 0 for s in video_avg_scores]
    auc_video = roc_auc_score(video_labels, video_avg_scores)
    f1_video  = f1_score(video_labels, video_preds)
    acc_video = accuracy_score(video_labels, video_preds)
    fpr_v, tpr_v, _ = roc_curve(video_labels, video_avg_scores)
    eer_video = compute_eer(fpr_v, tpr_v)

    # --- Save results ---
    os.makedirs("results", exist_ok=True)

    with open("results/frame_results.json", "w") as f:
        json.dump({
            "metrics": {
                "auc": auc_frame, "f1": f1_frame, "acc": acc_frame, "eer": eer_frame
            },
            "frames": frame_results
        }, f, indent=2)

    with open("results/video_results.json", "w") as f:
        json.dump({
            "metrics": {
                "auc": auc_video, "f1": f1_video, "acc": acc_video, "eer": eer_video
            },
            "videos": [
                {"video_id": vid, "mean_score": float(np.mean(v["scores"])), "label": int(v["labels"][0])}
                for vid, v in video_map.items()
            ]
        }, f, indent=2)

    # ROC Curve for video-level
    plt.figure()
    plt.plot(fpr_v, tpr_v, label=f"Video ROC (AUC={auc_video:.4f}, EER={eer_video:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Video-level ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png")
    print("ROC curve saved to roc_curve.png")

    return auc_frame, f1_frame, acc_frame, eer_frame, auc_video, f1_video, acc_video, eer_video

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "data/data_split"
    test_json = os.path.join(root_dir, "test_list.json")
    val_json = os.path.join(root_dir, "val_list.json")

    test_dataset = DeepfakeDataset(test_json, root_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CLIPLoRADetector().to(device)
    model.load_state_dict(torch.load("model_weight/best_model.pt", map_location=device, weights_only=True))

    auc_f, f1_f, acc_f, eer_f, auc_v, f1_v, acc_v, eer_v = evaluate(model, test_loader, device)

    print("== Frame-level ==")
    print(f"AUC: {auc_f:.4f}, F1: {f1_f:.4f}, ACC: {acc_f:.4f}, EER: {eer_f:.4f}")
    print("== Video-level ==")
    print(f"AUC: {auc_v:.4f}, F1: {f1_v:.4f}, ACC: {acc_v:.4f}, EER: {eer_v:.4f}")
