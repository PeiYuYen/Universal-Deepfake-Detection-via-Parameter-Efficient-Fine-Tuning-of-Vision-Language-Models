import os
import json
from unittest import loader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from model import CLIPLoRADetector, CLIPLinearProbe
from dataset import DeepfakeDataset 
from utils import set_seed  # 設定 random seed

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()

        # labels = labels.view(-1, 1)  # 如果用原始模型CLIPLinearProbe

        optimizer.zero_grad()
        outputs = model(images)  # (B,)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = (torch.sigmoid(outputs) > 0.5).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return sum(losses) / len(losses), acc, auc, f1

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()

            # labels = labels.view(-1, 1)  ## 如果用原始模型CLIPLinearProbe
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()

            # all_preds += probs.cpu().tolist()
            # all_labels += labels.cpu().tolist()

            all_preds += probs.squeeze(1).cpu().tolist()# #####原始模型
            all_labels += labels.squeeze(1).cpu().tolist()

    acc = correct / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return total_loss / len(loader.dataset), acc, auc, f1



def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路徑
    root_dir = "data/data_split"
    train_json = os.path.join(root_dir, "train_list.json")
    val_json = os.path.join(root_dir, "val_list.json")

    # Dataset
    train_set = DeepfakeDataset(train_json, root_dir)
    val_set = DeepfakeDataset(val_json, root_dir)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    # 模型與 PEFT 建立
    model = CLIPLoRADetector().to(device)
    # model = CLIPLinearProbe().to(device)  # 如果用原始模型CLIPLinearProbe
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    best_auc = 0
    log_file = open("origin_train_log.txt", "w")

    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/10")
        train_loss, train_acc, train_auc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} auc={train_auc:.4f} f1={train_f1:.4f}")
        print(f"Val:   loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} f1={val_f1:.4f}")
        log_file.write(json.dumps({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_f1": val_f1
        }) + "\n")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "model_weight/original_best_model.pt")

    log_file.close()

if __name__ == "__main__":
    main()
