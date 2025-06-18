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
conda create -n dfvlm python= 3.8
conda activate dfvlm
pip install -r requirements.txt
```

## Quick Start
直接使用以下指令快速開始：

### Inference Command
載入訓練後的權重檔：
```
python test.py  #約1min
```

### training
```
python train.py #約5min
```

## Pretrained Weights
[預訓練權重檔](https://huggingface.co/pui8838/dfvlm_rora/tree/main)

## Results
於results/ 裡
* 包含每個影片的分數 (per-video scores)
* 包含每一禎的分數 (per-frame scores)
* 整體指標 (overall metrics)
