import cv2
from ultralytics import YOLO

# 載入訓練好的模型 (請確認best.pt路徑正確)
model = YOLO("runs/detect/train/weights/best.pt")

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推論
    results = model(frame)

    # 畫出預測結果
    annotated_frame = results[0].plot()

    # 顯示畫面
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 按 'q' 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
