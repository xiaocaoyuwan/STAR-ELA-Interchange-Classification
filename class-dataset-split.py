import os
import shutil
import random

# 原始数据路径
src_root = 'D:/Code/DeepLearning/datasets/ChuanYuHu interchange_classification'

# 目标输出路径
dst_root = 'D:/Code/DeepLearning/datasets/CYH_interchange_cls'

# 训练验证测试集比例
train_ratio = 0.75
val_ratio = 0.25
test_ratio = 0  # 可选

# 是否需要test集
include_test = False

# 获取所有类别
classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

for split in ['train', 'val'] + (['test'] if include_test else []):
    for cls in classes:
        os.makedirs(os.path.join(dst_root, split, cls), exist_ok=True)

# 分配文件
for cls in classes:
    src_cls_path = os.path.join(src_root, cls)
    images = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:] if include_test else []

    for img_name in train_imgs:
        shutil.copy(os.path.join(src_cls_path, img_name), os.path.join(dst_root, 'train', cls, img_name))
    for img_name in val_imgs:
        shutil.copy(os.path.join(src_cls_path, img_name), os.path.join(dst_root, 'val', cls, img_name))
    for img_name in test_imgs:
        shutil.copy(os.path.join(src_cls_path, img_name), os.path.join(dst_root, 'test', cls, img_name))