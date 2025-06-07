import os
import random
import shutil

def data_process(source_base_dir,dest_base_dir):

    source_img_dir = source_base_dir+"/images"
    source_lbl_dir = source_base_dir+"/labels"

    random.seed(42)  # 固定隨機種子，方便重現結果

    # 設定切分比例
    train_ratio = 0.7   # 訓練集比例
    valid_ratio = 0.2   # 驗證集比例
    test_ratio  = 0.1   # 測試集比例

    # (YOLOv8 的目錄結構)
    dest_dirs = {
        "train": {
            "images": os.path.join(dest_base_dir, "train", "images"),
            "labels": os.path.join(dest_base_dir, "train", "labels")
        },
        "valid": {
            "images": os.path.join(dest_base_dir, "valid", "images"),
            "labels": os.path.join(dest_base_dir, "valid", "labels")
        },
        "test": {
            "images": os.path.join(dest_base_dir, "test", "images"),
            "labels": os.path.join(dest_base_dir, "test", "labels")
        }
    }

    # 建立目的資料夾，若不存在則建立
    for split in dest_dirs:
        for folder in dest_dirs[split].values():
            os.makedirs(folder, exist_ok=True)

    # 讀取來源圖片清單 
    image_extensions = [".jpg", ".jpeg", ".png"]
    all_images = [f for f in os.listdir(source_img_dir)
                if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"-----Found {len(all_images)} images")

    # 隨機打亂圖片順序
    random.shuffle(all_images)

    # 計算劃分每個資料集的圖片數量
    num_images = len(all_images)
    num_train = int(num_images * train_ratio)
    num_valid = int(num_images * valid_ratio)
    num_test  = num_images - num_train - num_valid  # 保證總數一致

    train_images = all_images[:num_train]
    valid_images = all_images[num_train:num_train+num_valid]
    test_images  = all_images[num_train+num_valid:]

    print(f"-----Train: {len(train_images)} | Valid: {len(valid_images)} | Test: {len(test_images)}")

    # 開始複製圖片及對應標籤檔案到各自的資料夾
    for split, image_list in zip(["train", "valid", "test"], [train_images, valid_images, test_images]):
        for img_file in image_list:
            # 複製圖片
            src_img = os.path.join(source_img_dir, img_file)
            dst_img = os.path.join(dest_dirs[split]["images"], img_file)
            copy_file(src_img, dst_img)

            # 對應的標籤檔案，假設與圖片同名但副檔名為 .txt
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"
            src_label = os.path.join(source_lbl_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(dest_dirs[split]["labels"], label_file)
                copy_file(src_label, dst_label)
            else:
                print(f"-----[Warning] Image {img_file} has no corresponding label file:{label_file}")
    
    # 複製data.yaml至dataset中
    try:
        shutil.copy(source_base_dir+"/data.yaml",dest_base_dir+"/data.yaml")
    except Exception as e:
        print(f"-----An error occurred while copying data.yaml. Please check if there is such a file in the directory {source_base_dir}")
        exit()
    print("-----Data segmentation and directory creation completed!")

# 定義檔案複製功能
def copy_file(src_path, dst_path):
    """複製檔案至目標路徑，若資料夾不存在則建立"""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)
