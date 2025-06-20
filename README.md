# Universal-Deepfake-Detection-via-Parameter-Efficient-Fine-Tuning-of-Vision-Language-Models

## Dataset
請先下載經過預處理的資料集： [FaceForensics++ C40](https://www.dropbox.com/t/2Amyu4D5TulaIofv)，將資料集下載解壓縮後創建"data"資料集並放置。

資料結構應該長這樣:
```plaintext
# Data Folder Structure
Universal-Deepfake-Detection-via-Parameter-Efficient-Fine-Tuning-of-Vision-Language-Models/
└── data/
    ├── FaceSwap
    ├── NeuralTextures
    └── Real_youtube

`````
請先切分資料集(確保資料結構已長得像上述那樣)，請用以下指令(5min):
```
python dataset.py
```


## Setup
使用 Conda 環境：
```bash
conda create -n dfvlm python=3.8
conda activate dfvlm
git clone https://github.com/PeiYuYen/Universal-Deepfake-Detection-via-Parameter-Efficient-Fine-Tuning-of-Vision-Language-Models.git
cd Universal-Deepfake-Detection-via-Parameter-Efficient-Fine-Tuning-of-Vision-Language-Models/
pip install -r requirements.txt
```

## Quick Start
直接使用以下指令快速開始：

### Inference Command
載入訓練後的權重檔後可進行推理(約2min)：
```
python test.py
```

### training
若要重新訓練模型(約5min)，則：
```
python train.py
```

## Pretrained Weights
[預訓練權重檔](https://huggingface.co/pui8838/dfvlm_rora/tree/main)
請下載到本repo下新建"model_weights"的子資料夾下。

## Results
於results/ 裡
* 包含每個影片的分數 (per-video scores)
* 包含每一禎的分數 (per-frame scores)
* 整體指標 (overall metrics)

ROC curve -> roc_curve.png
