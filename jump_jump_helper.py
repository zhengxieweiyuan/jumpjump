import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import v2 as T
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image ,ImageDraw
import os
import xml.etree.ElementTree as ET
import math
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
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
 
    # 加载一个在COCO上预训练的模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # 获取分类器的输入特征数
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
    T.ToTensor(), # 将图片转换为Tensor
    # 增强数据多样性
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.RandomPhotometricDistort(), # 随机调整亮度、对比度、色调等
    T.RandomHorizontalFlip(p=0.5), # 50%的概率水平翻转
    T.ToDtype(torch.float, scale=True), # 转换数据类型并归一化
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
     # 添加学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10, # 每10个epoch
                                                   gamma=0.1)   # 学习率乘以0.1
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
        # 更新学习率
        lr_scheduler.step()
        
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
    # 确保模型在CPU上加载，以防运行环境没有GPU
    device = torch.device('cpu')
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 图像预处理
    img = Image.open(image_path).convert("RGB")
    
    # 仅在需要可视化时创建绘图对象
    if visualize:
        draw = ImageDraw.Draw(img)
    
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)
    
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
        if score > 0.6:
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

    # --- 可视化部分 (如果开启) ---
    if visualize:
        # 绘制棋子的框和中心点
        if piece_center:
            draw.rectangle(best_piece_box, outline="red", width=3)
            draw.text((best_piece_box[0], best_piece_box[1] - 15), "piece", fill="red")
            # 在中心点画一个半径为5的圆
            cx, cy = piece_center
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="red")

        # 绘制目标的框和中心点
        if target_center:
            draw.rectangle(best_target_box, outline="lime", width=3)
            draw.text((best_target_box[0], best_target_box[1] - 15), "target", fill="lime")
            # 在中心点画一个半径为5的圆
            cx, cy = target_center
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="lime")

    # 计算距离
    if piece_center and target_center: 
        distance = math.sqrt((piece_center[0] - target_center[0])**2 + (piece_center[1] - target_center[1])**2)
        # 如果可视化开启，则绘制连接线
        if visualize:
            draw.line([piece_center, target_center], fill="yellow", width=2)
        print(f"找到棋子, 中心: {piece_center}, 置信度: {best_piece_score:.2f}")
        print(f"找到目标, 中心: {target_center}, 置信度: {best_target_score:.2f}")
        print(f"中心点像素距离: {distance:.2f}")
    else:
        print("未能同时找到棋子和目标方块。")
        
        distance = None

    # 不再在此处保存或显示图片，而是返回图片对象
    return distance, img
# --- 主程序入口 ---
if __name__ == '__main__':
    DATA_PATH = 'data'
    MODEL_SAVE_PATH = 'jump_jump_model.pth'
    
    # **步骤一：训练模型**
    # 准备好数据后，取消下面的注释来开始训练
    train_model(DATA_PATH, MODEL_SAVE_PATH, num_epochs=50)
    
   # **步骤二：使用模型进行推理**
   # 训练完成后，用下面的代码来测试一张新图片
    if os.path.exists(MODEL_SAVE_PATH):
        TEST_IMAGE_PATH = "D:\\jumpjump\\data\\images\\fail_picture4_1756748844.png" # 换成你的测试图片
        distance, result_img = get_jump_distance(MODEL_SAVE_PATH, TEST_IMAGE_PATH)
        if distance is not None:
            output_path = "result.png"
            result_img.save(output_path)
            print(f"可视化结果已保存至: {output_path}")
            result_img.show()
    else:
        print("模型文件未找到。请先训练模型。")