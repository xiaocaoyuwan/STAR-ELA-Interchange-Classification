import os

# 设置环境变量以忽略警告
os.environ['PYTHONWARNINGS'] = 'ignore'
# 设置特定的 PyTorch 环境变量 (如果有)
os.environ['TORCH_WARN_ONCE'] = '1'  # 只显示每种警告一次
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ultralytics import YOLO
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from tqdm import tqdm

if __name__ == '__main__':
    # === 配置 ===
    modules_use = "r18-attention-ELA"  # READY
    yamlname = {
        "cls11": "11/yolo11n-cls.yaml",
        "cls11-s": "11/yolo11s-cls.yaml",
        "cls12": "12/yolo12n-cls.yaml",
        "cls12-s": "12/yolo12s-cls.yaml",
        "cls8": "v8/yolov8n-cls.yaml",
        "cls8-s": "v8/yolov8s-cls.yaml",
        "clsresnet18": "class/yolo11-cls-resnet18.yaml",
        "origin-res18": "class/cls-originres18.yaml",
        "yolo11-resnet18": "11/yolo11-cls-resnet18.yaml",
        "convnextv2": "class/cls-convnextv2.yaml",
        "convnextv2-f-auto": "class/cls-convnextv2-fusion.yaml",
        "convnextv2-femto": "class/cls-convnextv2-femto.yaml",
        "fasternet": "class/cls-fasternet.yaml",
        "fasternet-f-auto": "class/cls-fasternet-fusion.yaml",
        "fasternet-t1-f-auto": "class/cls-fasternet-t1-fusion.yaml",
        "mobilenet": "class/cls-mobilenet.yaml",
        "mobilenet-f-auto": "class/cls-mobilenet-fusion.yaml",
        "mobilenet-m": "class/cls-mobilenet-m.yaml",
        "mobilenet-m-f": "class/cls-mobilenet-m-fusion.yaml",
        "efficientVit": "class/cls-efficientvit.yaml",
        "efficientVit-f": "class/cls-efficientvit-fusion.yaml",
        "efficientVit-f-auto": "class/cls-efficientvit-fusion.yaml",
        "efficientVit-m1-f-auto": "class/cls-efficientvit-m1-fusion.yaml",

        "r18-star": "improve/r18-star.yaml",
        "r18-star-RepC3-3": "improve/r18-star.yaml",
        "r18-star-ELA": "improve/r18-star-ELA.yaml",
        "r18-DySample": "improve/r18-DySample.yaml",
        "r18-DBBC3": "improve/r18-DBBC3.yaml",
        "r18-LFEC3": "improve/r18-LFEC3.yaml",
        "r18-DTAB": "improve/r18-DTAB.yaml",
        "r18-ETB": "improve/r18-ETB.yaml",
        "r18-FDT": "improve/r18-FDT.yaml",
        "r18-fsa": "improve/r18-fsa.yaml",
        "r18-FreqFFPN": "improve/r18-FreqFFPN.yaml",
        "r18-DRBC3": "improve/r18-DRBC3.yaml",
        "r18-DGCST": "improve/r18-DGCST.yaml",
        "r18-DGCST2": "improve/r18-DGCST2.yaml",
        "r18-KANC3": "improve/r18-KANC3.yaml",
        "r18-affregateAtt": "improve/r18-affregateAtt.yaml",
        "r18-star-affregateAtt": "improve/r18-star-affregateAtt.yaml",
        "r18-attention-AFGCAttention": "improve/r18-attention-AFGCAttention.yaml",
        "r18-attention-SimAM": "improve/r18-attention-SimAM.yaml",
        "r18-attention-ELA": "improve/r18-attention-ELA.yaml",
        "r18-attention-SEAtt": "improve/r18-attention-ELA.yaml",
        "r18-attention-CoordAtt": "improve/r18-attention-CoordAtt.yaml",
        "r18-attention-BiLevelRoutingAttention_nchw": "improve/r18-attention-BiLevelRoutingAttention_nchw.yaml",

    }
    yamlname_use = yamlname[modules_use]
    data_dir = 'D:/Code/DeepLearning/datasets/CYH_interchange_cls_dataAu'
    val_dir = os.path.join(data_dir, 'val')

    exp_name = f'cls-{modules_use}-300-'

    project_root = 'runs/yolo_class'

    # === Step 1: 训练 ===
    model = YOLO(f'./ultralytics/cfg/models/{yamlname_use}', task='classify')
    model.train(
        data=data_dir,
        cache="disk",
        imgsz=640,
        epochs=300,
        batch=4,
        workers=1,
        project=project_root,
        name=exp_name,
        pretrained=False,
        amp=False,
        half=False
    )

    # === Step 2: 加载训练后的模型 ===
    model_path = f"{project_root}/{exp_name}/weights/best.pt"
    model = YOLO(model_path)
    class_names = list(model.names.values())
    class2idx = {name: i for i, name in enumerate(class_names)}

    # === Step 3: 对验证集推理 ===
    y_true = []
    y_pred = []

    print("\n🔍 正在推理验证集...")
    for class_name in os.listdir(val_dir):
        class_folder = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for image_file in tqdm(os.listdir(class_folder), desc=f'Class: {class_name}'):
            img_path = os.path.join(class_folder, image_file)
            if not img_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            # 推理
            result = model.predict(img_path, verbose=False)[0]
            pred_idx = int(result.probs.top1)

            # 记录标签
            y_pred.append(pred_idx)
            y_true.append(class2idx[class_name])

    # === Step 4: 保存指标 ===
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = f"{project_root}/{exp_name}/classification_report-{modules_use}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n")
        f.write(report + "\n")
    print(f"\n✅ 分类报告保存至: {report_path}")

    # === Step 5: 混淆矩阵图像 ===
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    cm_path = f"{project_root}/{exp_name}/confusion_matrix-{modules_use}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ 混淆矩阵图像保存至: {cm_path}")
