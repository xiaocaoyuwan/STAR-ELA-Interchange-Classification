# STAR-ELA-Interchange-Classification
STAR-ELA-Interchange-Classification implements a structure-enhanced deep learning framework for multi-class highway interchange classification from high-resolution Earth Observation imagery, based on ResNet18 with StarInterchange and ELA modules.
English Version

1. Overview
This repository provides the official implementation of the STAR–ELA–enhanced interchange classification framework proposed in our manuscript:
High-Resolution Interchange Classification Using Structural Feature–Enhanced Deep Learning
(Manuscript under review)
The proposed framework focuses on multi-class classification of complex highway interchange structures from high-resolution Earth Observation (EO) imagery, emphasizing explicit structural feature modeling.
Highway interchanges are characterized by multi-level topology, intertwined road geometries, and diverse structural layouts, which pose significant challenges for conventional remote sensing scene classification. To address these challenges, this work introduces a structure-aware image-based classification framework that does not rely on vector data, point clouds, or handcrafted geometric rules.

2. Method Summary
The proposed framework is built upon ResNet18 and enhanced with:
1）StarInterchange module
Introduces multiplicative feature interactions derived from the STAR operation to strengthen geometric and topological representation.
2）Enhanced Local Attention (ELA)
Emphasizes fine-grained local structural cues critical for distinguishing visually similar interchange layouts.
3）Multi-scale feature aggregation
Improves robustness to scale variation in high-resolution EO imagery.

3. HRIC Dataset
All experiments in this repository are conducted on the HRIC (High-Resolution Interchange Classification) Dataset, which is publicly available on Zenodo.
📌 Dataset DOI:
👉 https://doi.org/10.5281/zenodo.17972106

Dataset Summary
Total images: 542
Sensors: Gaofen-2, Jilin-1
Spatial resolution: 0.5–0.75 m
Image size: 1075 × 924 pixels
Image format: RGB (JPG)
Task type: Image-level classification

Interchange Categories (6 classes)
Cloverleaf Interchange
Diamond Interchange
Roundabout Interchange
T Interchange
Trumpet Interchange
Turbine Interchange

The dataset reflects real-world class imbalance commonly observed in urban road networks.

Note:
The HRIC dataset is not included in this repository. Please download it from Zenodo using the DOI above.

4. Citation
If you use this code or dataset in your research, please cite:
Yu, W.; Liu, G.; He, J.; Luo, Z.
High-Resolution Interchange Classification Using Structural Feature–Enhanced Deep Learning.
Manuscript under review.
(The citation will be updated upon acceptance.)

5. Author & Contact
Authors:
Yu, Wan’er; Liu, Gang*; He, Jing; Luo, Zhiyong
Affiliation:
School of Geography and Planning,
Chengdu University of Technology
Contact:
📧 xiaocaoyuwan@163.com

中文版本（Chinese Version）
1. 项目简介
本代码仓库为论文 《High-Resolution Interchange Classification Using Structural Feature–Enhanced Deep Learning》（投稿中）的官方实现。
本研究面向 高分辨率地球观测（EO）影像中的复杂城市立交结构分类任务，重点关注 显式结构特征建模，用于解决多层拓扑、道路交织等复杂结构带来的分类挑战。
与依赖矢量数据或点云数据的方法不同，本文提出的框架 直接基于光学遥感影像进行立交类型分类，无需任何先验道路网络或人工几何规则。

2. 方法概述
所提出的 STAR–ELA 框架以 ResNet18 为基础，并引入：
1）StarInterchange 模块
基于 STAR Operation 的乘性特征交互机制，用于增强几何与拓扑结构表达能力。
2）增强局部注意力（ELA）机制
强化对立交关键局部结构（如匝道连接关系）的建模能力。
3）多尺度特征融合策略
提高对不同尺度立交结构的鲁棒性。

3. HRIC 数据集
本仓库实验基于 HRIC（High-Resolution Interchange Classification）数据集。
📌 Zenodo DOI：
👉 https://doi.org/10.5281/zenodo.17972106

数据集信息
图像数量：542
数据来源：高分二号（GF-2）、吉林一号（Jilin-1）
空间分辨率：0.5–0.75 m
图像尺寸：1075 × 924
图像格式：RGB（JPG）

立交类型（6 类）
苜蓿叶立交
菱形立交
环岛立交
T 形立交
喇叭形立交
涡轮立交

数据集呈现真实城市路网中常见的类别不均衡特性。
⚠️ 本仓库 不包含原始 HRIC 图像数据，请通过 Zenodo DOI 下载。

4. 引用方式
（同英文部分）

5. 作者信息
作者：余莞尔；刘刚*；何敬；罗智勇
单位：成都理工大学 地理与规划学院
联系方式：xiaocaoyuwan@163.com
