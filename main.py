import os
import torch
import argparse
from data_process import data_process
from train import train_model
from test import valid_and_test

"""
請檢查roboflow匯出的目錄格式應為
└──train/
|   └── images
|   └── labels
└──valid/
|   └── images
|   └── labels
└──test/
    └── images
    └── labels
請直接新建一個目錄，命名為dataset並把載下來的zip放入解壓縮
"""

# 來源資料夾(從roboflow下載後解壓縮的檔案請放這)
SRC_BASE_DIR = "./dataset"
# 目的資料夾(訓練好後產生的目錄)
DST_BASE_DIR = "./runs/detect"

# 訓練參數
EPOCHS = 50
IMG_SIZE = 640
SEED = 42

if __name__ == '__main__':
    # 檢查來源目錄是否存在
    if not os.path.isdir(SRC_BASE_DIR):
        print(f"-----Please check if the directory[{SRC_BASE_DIR}] exists")
        exit()

    # 當前是使用CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("-----Current Device is GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("-----Current Device is  CPU")

    train_model(EPOCHS,IMG_SIZE,SEED,device)
    valid_and_test(DST_BASE_DIR)