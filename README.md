# DL2024_Team3_Convenience_Store_Shelf_Product_Recognition
# Convenience Store Shelf Item Detection(便利商店貨架商品偵測系統)
## 📌 專案目標

本專案旨在開發一套基於 YOLOv8 模型的電腦視覺系統，能夠準確辨識便利商店貨架上的商品，提升店家在商品上架管理、自動補貨及銷售分析上的效率。

### 🎯 預期辨識層級：
- 第一層：容器類型（如 鋁箔包、鐵鋁罐、玻璃瓶、寶特瓶等）
- 第二層：商品類別（如 茶、酒、水、乳製品等）
- 第三層：品牌（未來計畫擴充）
### 容器 (7 個)
AlminumFoilPack  Can  Glass  PET  PlasticBottle  TetraPak  YakultBottle
### 種類 (11 個)
Alcohol  Coffee  DairyProducts  EnergyDrink  FruitJuice  LacticAcid  Soda  SoyaMilk  SportsDrink  Tea  Water

## 🗂️ 專案檔案說明

| 檔案/資料夾         | 說明 |
|---------------------|------|
| `README.md`         | 本說明文件 |
| `requirements.txt`  | 套件需求，建議安裝於虛擬環境 |
| `data/`             | 資料集由各個組員拍照收集而成 |
| `model/`            | 訓練後模型或下載連結 |
| `src/`              | 程式碼檔案（如推論腳本） |

## 📁 資料集來源與說明
標註資料集：由組員至便利商店與超市實地拍攝，共拍攝近 XXX 張圖片，經 Roboflow 進行標記與轉換。

Roboflow 連結（訓練用公開集）：
🔗 https://universe.roboflow.com/holelabel-pan10/merged-dl-v2

## 📦 使用方法
1. 下載資料集 https://
使用 Roboflow 進行圖片上標註，匯出 YOLOv8 或是 YOLO11 格式資料集。
訓練模型範例說明:model.train(data="/content/Merged-DL-v2-2/data.yaml", epochs=50, imgsz=640)