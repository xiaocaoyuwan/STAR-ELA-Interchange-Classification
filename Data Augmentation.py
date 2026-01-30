import os
import cv2
import albumentations as A
import random
import shutil

# 设置随机种子（确保可复现）
random.seed(42)

# 图像增强策略
augmentation_pipeline = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.3)
])

def augment_images_in_folder(folder_path, save_dir, target_num=100, save_suffix='aug'):
    # 类别名（用于构建保存目录）
    class_name = os.path.basename(folder_path.rstrip('/\\'))
    save_class_dir = os.path.join(save_dir, class_name)

    os.makedirs(save_class_dir, exist_ok=True)

    # 复制原图到保存目录
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img_name in images:
        src = os.path.join(folder_path, img_name)
        dst = os.path.join(save_class_dir, img_name)
        shutil.copy(src, dst)

    current_num = len(images)
    print(f"📂 类别 [{class_name}]：原始 {current_num} 张，目标 {target_num} 张，开始增强...")

    img_idx = 0
    while len(os.listdir(save_class_dir)) < target_num:
        img_name = images[img_idx % current_num]
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 读取失败：{img_path}")
            continue

        augmented = augmentation_pipeline(image=img)
        aug_img = augmented['image']

        base_name = os.path.splitext(img_name)[0]
        save_name = f"{base_name}_{save_suffix}_{img_idx}.jpg"
        save_path = os.path.join(save_class_dir, save_name)

        cv2.imwrite(save_path, aug_img)
        img_idx += 1

    print(f"✅ 类别 [{class_name}] 增强完成，最终图片数：{len(os.listdir(save_class_dir))}")

# ========== 用法示例 ==========
# 原始分类数据文件夹
original_data_root = 'D:/Code/DeepLearning/datasets/CYH_interchange_cls_dataAu/val'   # 例如：data/T形, data/Y形, ...
# 增强数据保存到这个目录中
augmented_save_root = 'D:/Code/DeepLearning/datasets/CYH_interchange_cls_dataAu'

# 需要增强的类别路径
categories_to_augment = [
    os.path.join(original_data_root, 'T-interchange'),
    os.path.join(original_data_root, 'Diamond interchange'),
    os.path.join(original_data_root, 'Turbine interchange'),
    os.path.join(original_data_root, 'Roundabout interchange')
]

# 每类增强到多少张
target_num_per_class = 50

# 遍历增强
for class_folder in categories_to_augment:
    augment_images_in_folder(class_folder, save_dir=augmented_save_root, target_num=target_num_per_class)
