# DL2025_Team3_Convenience_Store_Shelf_Product_Recognition(便利商店貨架貨物辨識)
## 📌 專案簡介與目標
本專案旨在訓練一個能夠偵測與分類貨架商品的模型，以飲料作為辨識對象，並預期能夠辨識商品兩層的結構

### 🎯 辨識層級：
- 第一層：容器類型 6 種（如 鋁箔包、鐵鋁罐、玻璃瓶、寶特瓶等）
- 第二層：商品類別 11 種（如 茶、酒、水、乳製品等）
> [!NOTE]
> 詳細標籤名稱請至下方 Roboflow 連結中的 overview 瀏覽

## 🗂️ 專案檔案說明

| 檔案/資料夾         | 說明 |
|---------------------|------|
| `README.md`         | 本說明文件 |
| `requirements.txt`  | 套件需求，建議安裝於虛擬環境 |
| `main.py`           | 主程式 |
| `train.py`         | 訓練模型 |
| `test.py`          | 預測測試集圖片並產生結果 |
| `detect.py`        | 實作辨識 |

## 🔍 資料集來源與說明
- 來源：便利商店及超市實地拍攝，相機長寬比設定為 16:9 ，直拍與橫拍皆有
- 說明：標註好的資料集用資料擴增增加圖片數量，擴增方法選擇 rotation±7°、saturation±20%、brightness±13%
- 原始拍攝張數為 573 張，以 70% / 20% / 10% 的比例做切分(訓練集 401 ，驗證集 115 ，測試集 57 )，後把訓練集擴增至 1206 張
- Roboflow 連結（訓練用公開集）：
🔗 [https://universe.roboflow.com/holelabel-pan10/merged-dl-v2](https://universe.roboflow.com/holelabel-pan10/merged-dl-v2/dataset/12)

## ⚙️ 使用方法
1. 下載本程式碼
2. 下載上方 roboflow 的資料集 (匯出時請選擇 yolov8-obb 格式) ，建立 `./dataset` ，將 zip 放入後解壓縮，目錄內容格式應為
    ```
      └──train/
        |   └── images
        |   └── labels
        └──valid/
        |   └── images
        |   └── labels
        └──test/
            └── images
            └── labels
      ```
3. 執行 `python main.py` 即開始訓練模型
4. 訓練完成後即可執行 `python detect.py` 測試模型辨識效果
5. 您也可以下載已訓練好的模型，連結 🔗 https://huggingface.co/wen15/DL2025_Team3_YOLOv8x

## 💡 補充說明
- 模型標籤數量實際為 48 種，因有些包裝的材質並不會裝某類型的飲料，例如塑膠瓶裝酒、鋁箔包裝汽水等
- 資料集的圖片標註與資料擴增皆於 Roboflow 完成
- `requirements.txt` 中的 `torch` 版本請依照自身環境下載對應版本
