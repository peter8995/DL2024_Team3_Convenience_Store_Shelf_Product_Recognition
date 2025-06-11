from ultralytics import YOLO

def valid_and_test(dest_base_dir):
    # 載入模型
    model = YOLO(dest_base_dir+"/train/weights/best.pt")
 
    # 驗證模型表現(驗證集)
    print("-----Running evaluation on valid set...")
    metrics = model.val()
    print("-----Valid performance: ",metrics.results_dict,"\n")

    # 驗證模型表現(測試集)
    print("-----Running evaluation on test set...")
    metrics = model.val(split='test')
    print("-----Test performance: ",metrics.results_dict,"\n")

    print("---------------------------Start inferring the image---------------------------")
    # 推論所有測試集圖片並儲存結果
    test_img_dir = dest_base_dir+"/test/images"
    results = model.predict(source=test_img_dir, save=True, conf=0.5)
    print("-----Image inference completed")