import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image ,ImageDraw
import os
import xml.etree.ElementTree as ET
import math
import time
FAILED_IMAGE_SAVE_DIR =".\data\images"

# --- 1. 自定义数据集类 ---
class JumpDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        # 类别映射：'piece' -> 1, 'target' -> 2。背景是0。
        self.class_map = {"piece": 1, "target": 2}

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for member in root.findall('object'):
            class_name = member.find('name').text
            if class_name not in self.class_map:
                continue
            
            labels.append(self.class_map[class_name])
            
            bndbox = member.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.imgs)

# --- 2. 模型修改 ---
def get_model(num_classes):
    # 加载一个在COCO上预训练的模型
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    # 新模型: 使用 MobileNetV3-Large FPN 作为骨干网络，速度更快
    # 获取分类器的输入特征数
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替换预训练的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# --- 3. 训练函数 ---
def train_model(data_path, model_save_path, num_epochs=10):
    # 类别数量 = 你的类别数 + 背景
    # 我们有 'piece' 和 'target'，所以是 2 + 1 = 3
    num_classes = 3
    
        # 定义数据变换（加入数据增强）
    transforms = T.Compose([
        T.ToImage(),  # 替换为新的方法
        T.ToDtype(torch.float32, scale=True),  # 明确指定数据类型和缩放

        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.RandomPhotometricDistort(), # 随机调整亮度、对比度、色调等
        T.RandomHorizontalFlip(p=0.5), # 50%的概率水平翻转
        
    ])
    
    # 创建数据集和数据加载器
    dataset = JumpDataset(root=data_path, transforms=transforms)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model = get_model(num_classes)
    model.to(device)
    
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    print("--- 开始训练 ---")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}")
        
    torch.save(model.state_dict(), model_save_path)
    print(f"--- 训练完成，模型已保存至 {model_save_path} ---")

# --- 4. 推理和距离计算函数 ---
def get_jump_distance(model_path, image_path, visualize=False):
    """
    加载模型进行推理，计算距离，并根据需要将结果可视化。
    - model_path: 模型文件路径。
    - image_path: 输入的测试图片路径。
    - visualize: 是否绘制检测框和中心点。
    Returns:
    - distance: 计算出的像素距离，如果失败则为 None。
    - img: 绘制了或未绘制检测框和中心点的 PIL.Image 对象。
    """
    # 加载训练好的模型
    num_classes = 3
    device = torch.device('cpu')
    model = get_model(num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"错误：模型文件未找到 at {model_path}")
        return None, Image.open(image_path).convert("RGB")
    model.to(device)
    model.eval()

    # 图像预处理
    img = Image.open(image_path).convert("RGB")
    # transform = T.Compose([T.ToTensor()])
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])

    # 解析结果
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    piece_center, target_center = None, None
    best_piece_box, best_target_box = None, None
    best_piece_score, best_target_score = 0, 0

    for i in range(len(boxes)):
        score = scores[i].item()
        label = labels[i].item()
        box = boxes[i].tolist()

        if score > 0.6: # 置信度阈值
            if label == 1 and score > best_piece_score: # 棋子
                best_piece_score = score
                best_piece_box = box
                piece_center = ((box[0] + box[2]) / 2, box[3])
            elif label == 2 and score > best_target_score: # 目标
                best_target_score = score
                best_target_box = box
                target_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
     
    timestamp_suffix = int(time.time())
    fail_filename = f"fail_picture5_{timestamp_suffix}.png"
    fail_save_path = os.path.join(FAILED_IMAGE_SAVE_DIR, fail_filename)

    distance = None 
    if piece_center and target_center: 
        distance = math.sqrt((piece_center[0] - target_center[0]) ** 2 + (piece_center[1] - target_center[1]) ** 2)
        if piece_center[1] < target_center[1]:
            print("检测到异常情况：目标方块位于棋子下方，判定为失败。")
            img.save(fail_save_path) 
            distance = -2
    elif piece_center is None or target_center is None:
        img.save(fail_save_path) 
        distance = -2
        print("未检测到棋子和或标方块，无法计算距离。")
    else:
        img.save(fail_save_path) 
        distance = -2
        print("没有检测到棋子和目标方块，无法计算距离。")
    # --- 可视化部分 ---
    if visualize:
        draw = ImageDraw.Draw(img)
        # 绘制棋子
        if piece_center:
            draw.rectangle(best_piece_box, outline="red", width=3)
            draw.text((best_piece_box[0], best_piece_box[1] - 15), f"piece ({best_piece_score:.2f})", fill="red")
            cx, cy = piece_center
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="red")
        # 绘制目标
        if target_center:
            draw.rectangle(best_target_box, outline="lime", width=3)
            draw.text((best_target_box[0], best_target_box[1] - 15), f"target ({best_target_score:.2f})", fill="lime")
            cx, cy = target_center
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="lime")
        # 如果成功计算了距离，绘制连线
        if distance:
            draw.line([piece_center, target_center], fill="yellow", width=2)
        
            
    return distance, img

# --- 主程序入口 ---
if __name__ == '__main__':
    DATA_PATH = 'data'
    MODEL_SAVE_PATH = 'jump_jump_model.pth'
    
    # **步骤一：训练模型**
    # 准备好数据后，取消下面的注释来开始训练
    train_model(DATA_PATH, MODEL_SAVE_PATH, num_epochs=30)
    
   # **步骤二：使用模型进行推理**
   # 训练完成后，用下面的代码来测试一张新图片
    if os.path.exists(MODEL_SAVE_PATH):
        TEST_IMAGE_PATH = "D:\\jumpjump\\data\\images\\fail_picture_1756219413.png" # 换成你的测试图片
        distance, result_img = get_jump_distance(MODEL_SAVE_PATH, TEST_IMAGE_PATH)
        if distance is not None:
            output_path = "result.png"
            result_img.save(output_path)
            print(f"可视化结果已保存至: {output_path}")
            result_img.show()
    else:
        print("模型文件未找到。请先训练模型。")