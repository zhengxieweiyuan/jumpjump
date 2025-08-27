import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw # 增加了 ImageDraw
import os
import xml.etree.ElementTree as ET
import math

 

# --- 4. 推理、可视化和距离计算函数 ---
def get_jump_distance(model_path, image_path, output_image_path="result.png"):
    """
    加载模型进行推理，计算距离，并将结果可视化。
    - model_path: 模型文件路径。
    - image_path: 输入的测试图片路径。
    - output_image_path: 可视化结果的保存路径。
    """
    # 加载训练好的模型
    num_classes = 3
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 图像预处理
    img = Image.open(image_path).convert("RGB")
    # 创建一个绘图对象
    draw = ImageDraw.Draw(img)
    
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    
    with torch.no_grad():
        prediction = model([img_tensor])
        
    # 解析结果
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    piece_center = None
    target_center = None
    best_piece_box = None
    best_target_box = None
    
    # 找到置信度最高的棋子和目标
    best_piece_score = 0
    best_target_score = 0

    for i in range(len(boxes)):
        score = scores[i].item()
        label = labels[i].item()
        box = boxes[i].tolist()

        # 置信度阈值可以根据模型实际表现调整
        if score > 0.7:
            if label == 1 and score > best_piece_score: # 棋子
                best_piece_score = score
                best_piece_box = box
                # 计算棋子的底部中心点
                piece_center = ((box[0] + box[2]) / 2, box[3]) 
            elif label == 2 and score > best_target_score: # 目标
                best_target_score = score
                best_target_box = box
                # 目标方块我们依然用几何中心
                target_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    # --- 可视化部分 ---
    # 绘制棋子的框和中心点
    if piece_center:
        draw.rectangle(best_piece_box, outline="red", width=3)
        draw.text((best_piece_box[0], best_piece_box[1] - 15), "piece", fill="red")
        # 在中心点画一个半径为5的圆
        cx, cy = piece_center
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="red")
        print(f"找到棋子, 中心: {piece_center}, 置信度: {best_piece_score:.2f}")

    # 绘制目标的框和中心点
    if target_center:
        draw.rectangle(best_target_box, outline="lime", width=3)
        draw.text((best_target_box[0], best_target_box[1] - 15), "target", fill="lime")
        # 在中心点画一个半径为5的圆
        cx, cy = target_center
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="lime")
        print(f"找到目标, 中心: {target_center}, 置信度: {best_target_score:.2f}")

    # 计算并绘制距离连线
    if piece_center and target_center:
        distance = math.sqrt((piece_center[0] - target_center[0])**2 + (piece_center[1] - target_center[1])**2)
        # 在图上画出连接线
        draw.line([piece_center, target_center], fill="yellow", width=2)
        print(f"中心点像素距离: {distance:.2f}")
    else:
        print("未能同时找到棋子和目标方块。")
        distance = None

    # 保存并显示图片
    img.save(output_image_path)
    print(f"可视化结果已保存至: {output_image_path}")
    img.show() # 直接打开图片预览

    return distance

# --- 主程序入口 ---
if __name__ == '__main__':
    DATA_PATH = 'data'
    MODEL_SAVE_PATH = 'jump_jump_model.pth'
    
    # **步骤一：训练模型**
    # 准备好数据后，取消下面的注释来开始训练
    # train_model(DATA_PATH, MODEL_SAVE_PATH, num_epochs=20)
    
    # **步骤二：使用模型进行推理和可视化**
    # 训练完成后，用下面的代码来测试一张新图片
    if os.path.exists(MODEL_SAVE_PATH):
        TEST_IMAGE_PATH = 'test.png' # 换成你的测试图片
        get_jump_distance(MODEL_SAVE_PATH, TEST_IMAGE_PATH)
    else:
        print("模型文件未找到。请先训练模型。")