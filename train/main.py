import os
import torch
import argparse
from data_process import data_process
from train import train_model
from test import valid_and_test

"""
決定是否執行資料前處理
1.1. 若roboflow匯出的目錄格式為

└──train/
|   └── images
|   └── labels
└──valid/
|   └── images
|   └── labels
└──test/
    └── images
    └── labels
請直接新建一個目錄，命名為dataset並把載下來的zip放入解壓縮即可，***不需進行資料前處理***

1.2. 若roboflow匯出的目錄格式為

└──train/
    └── images
    └── labels
才需進行資料前處理
處理步驟如下
(1) roboflow匯出成zip時，請選擇yolov8-obb格式
(2) 把zip放入roboflow_export解壓縮
(3) 請把train中的兩個目錄放到與train同一層，移動後刪除train目錄

"""

# 來源資料夾(從roboflow下載後解壓縮的檔案請放這)
SRC_BASE_DIR = "./roboflow_export"
# 目標資料夾(資料處理後檔案放置路徑)
DEST_BASE_DIR = "./dataset"

# 訓練參數
EPOCHS = 50
IMG_SIZE = 640
SEED = 42

if __name__ == '__main__':

    # 判斷訓練是使用CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("-----Current Device is GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("-----Current Device is  CPU")

    # 判斷是否做資料前處理
    parser = argparse.ArgumentParser(description="決定是否做資料前處理")
    parser.add_argument("--doDP", action="store_true", help="顯示是否執行資料前處理")
    args = parser.parse_args()
    if args.doDP:
        print("-----Execute data process")
        data_process(SRC_BASE_DIR,DEST_BASE_DIR)
    else:
        print("-----Do not execute data process")
        if not os.path.isdir(DEST_BASE_DIR):
            print(f"-----Please check if the directory[{DEST_BASE_DIR}] exists")
            exit()

    train_model(EPOCHS,IMG_SIZE,SEED,device)
    valid_and_test(DEST_BASE_DIR)