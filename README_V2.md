# 阿坝州降雨诱发滑坡易发性评价 — Pipeline V2

基于 **阿坝州地质灾害隐患点数据（7315条）**，采用**空间约束负样本生成 + 二阶段易发性-暴露度分离**的机器学习建模框架，实现滑坡易发性（Susceptibility）的二分类评价。

> 📌 **与 V1 的区别**：V1 使用原始灾害点内部分类（降雨滑坡 vs 其他），V2 采用**外部生成非灾害点**作为负样本，更符合滑坡易发性评价的国际学术规范。

---

## 目录

- [方法论概述](#方法论概述)
- [数据说明](#数据说明)
- [整体流程](#整体流程)
- [模块详解](#模块详解)
  - [1. 数据清洗](#1-数据清洗)
  - [2. 缺失值插补](#2-缺失值插补)
  - [3. 特征工程](#3-特征工程)
  - [4. 负样本生成](#4-负样本生成)
  - [5. 二阶段建模策略](#5-二阶段建模策略)
- [模型列表](#模型列表)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [评估指标](#评估指标)
- [引用说明](#引用说明)

---

## 方法论概述

### 核心创新

| 创新点 | 说明 |
|--------|------|
| **易发性·暴露度二阶段分离** | 分类模型仅用地形+降雨+植被特征，暴露特征（人口/财产）不参与分类，用于风险分析 |
| **空间约束负样本生成** | BallTree 空间索引 + 500m 缓冲区约束，生成非灾害点负样本 |
| **多尺度降雨特征** | rain_3d（短期）、rain_7d（中期）、rain_30d（长期）+ 降雨强度比 |
| **综合暴露指数** | 加权融合威胁人口、户数、财产的归一化暴露指标 |

### 研究范式

```text
┌──────────────────────────────────────────┐
│           阶段1: 滑坡易发性                │
│  特征: 地形 + 降雨 + 植被 + 土地利用       │
│  模型: RF / XGB / LGB / CAT 等           │
│  输出: P(灾害|地形,降雨,植被)             │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│           阶段2: 滑坡风险                  │
│  输入: 易发性概率 × 暴露度(人口/财产)      │
│  输出: Risk = Susceptibility × Exposure   │
└──────────────────────────────────────────┘
```

---

## 数据说明

### 原始数据

| 项目 | 内容 |
|------|------|
| 数据文件 | `dataset/aba_disaster_distribution.csv` |
| 样本量 | **7,315 条** |
| 维度 | **31 列** |
| 数据来源 | Landsat 7/8/5 遥感影像 + 气象站 + 实地普查 |
| 研究区域 | 四川省阿坝藏族羌族自治州 |
| 经度范围 | 100.02°E – 104.40°E |
| 纬度范围 | 30.67°N – 34.26°N |
| 高程范围 | 845 m – 4,514 m |
| 时间跨度 | 2008 – 2020 年 |

### 灾害类型分布

| 类型 | 样本数 | 占比 | 是否保留 |
|------|:------:|:----:|:--------:|
| **滑坡** | 2,204 | 30.1% | ✅ 保留 |
| **不稳定斜坡** | 1,113 | 15.2% | ✅ 保留 |
| **泥石流** | 2,305 | 31.5% | ✅ 保留 |
| **崩塌** | 1,689 | 23.1% | ❌ 排除 |
| **地面塌陷** | 4 | 0.1% | ❌ 排除 |

> 保留滑坡 + 不稳定斜坡 + 泥石流共 **5,622 条**作为正样本（均为降雨诱发的坡体灾害）。

### 危险等级分布

| 等级 | 样本数 | 占比 |
|:----:|:------:|:----:|
| 小 | 6,311 | 86.3% |
| 中 | 944 | 12.9% |
| 大 | 42 | 0.57% |
| 特大 | 18 | 0.25% |

> 高危险样本（大+特大）仅 60 条(0.8%)，因此不以危险等级作为二分类标签。

---

## 整体流程

```text
dataset/aba_disaster_distribution.csv (7315×31)
                         │
    ┌────────────────────▼────────────────────┐
    │          data_cleaning.py               │
    │  • 灾害类型筛选 → 滑坡+不稳定斜坡+泥石流   │
    │  • 删除10个冗余字段（id, data_source等）  │
    │  • 空值行删除 + 异常值IQR过滤 + 去重      │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │           imputation.py                 │
    │  • NDVI (43%缺失) → 随机森林回归插补     │
    │  • NDWI (43%缺失) → NDVI线性回归 + RF   │
    │  • 降雨 (19.6%缺失) → MICE多变量插补     │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │         pipeline_v2.py 特征工程          │
    │  • 地形因子：elevation, aspect, tpi...   │
    │  • 降雨因子：rain_3d/7d/30d, api         │
    │  • 植被因子：ndvi, ndwi                  │
    │  • 衍生特征：rain_ratio, valley_risk等   │
    │  • landuse_type One-hot编码              │
    │  • 暴露特征分离（不参与分类）              │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │      negative_sampling.py               │
    │  • BallTree空间索引构建                  │
    │  • 经纬度范围随机采样                    │
    │  • 距灾害点 > 500m 过滤                 │
    │  • 最近邻加权特征继承 + 噪声             │
    │  • 暴露特征全部置0                      │
    │  正:负 = 1:1 或 1:0.5                  │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │         train_models_v2.py              │
    │  • 9个基模型 + 4个集成模型               │
    │  • 阈值优化 + 概率校准                  │
    │  • SHAP特征重要性分析                   │
    │  • ROC/PR评估曲线                      │
    │  • 结果排名CSV输出                      │
    └─────────────────────────────────────────┘
```

---

## 模块详解

### 1. 数据清洗

**文件**: `pre_process/data_cleaning.py`

#### 1a. 灾害类型筛选
只保留与滑坡同源的坡体灾害：

| 保留 | 原因 |
|------|------|
| 滑坡 | 核心研究对象 |
| 不稳定斜坡 | 滑坡前期状态 |
| 泥石流 | 与滑坡同降雨驱动机制，源自有滑坡堆积物 |

#### 1b. 删除字段

```
删除: id, discovery_date, location, data_source,
      development_trend, industry, industry_transferred,
      disaster_type, disaster_scale, scale_grade
原因: 纯元数据 / 缺失 > 80% / 与hazard_type冗余
```

#### 1c. 空值处理

| 字段 | 缺失率 | 处理方式 |
|------|:------:|----------|
| `triggering_factor` | 0.9% | 行删除 |
| `threat_target` | 1.0% | 行删除 |

#### 1d. 异常值过滤

```
• elevation ∈ [500, 5500] m
• aspect ∈ [0, 360]°
• rain_* ≥ 0
• tpi: 1%~99% IQR
• distance_to_river: 1%~99% IQR
```

---

### 2. 缺失值插补

**文件**: `pre_process/imputation.py`

#### 2a. NDVI 插补（43% 缺失）

```python
# 用随机森林回归预测NDVI
特征: elevation + tpi + distance_to_river + lon + lat + landuse
模型: RandomForestRegressor(n_estimators=200, max_depth=10)
验证: 训练R² ≈ 0.6-0.8
```

#### 2b. NDWI 插补（43% 缺失）

```python
# 两步插补
Step 1: NDWI ~ NDVI 线性回归 (NDVI-NDWI高度负相关)
Step 2: 剩余缺失用RF预测
```

#### 2c. 降雨数据插补（19.6% 缺失）

```python
# MICE 多重链式方程插补
模型: IterativeImputer(RF, max_iter=30)
联合插补: rain_3d + rain_7d + rain_30d + api + elevation + lon + lat
```

---

### 3. 特征工程

**文件**: `pre_process/pipeline_v2.py`

#### 易发性特征（27个，用于分类）

| 类别 | 特征 | 数量 |
|:----|:-----|:----:|
| **地形** | `elevation`, `aspect`, `tpi`, `distance_to_river` | 4 |
| **降雨** | `rain_3d`, `rain_7d`, `rain_30d`, `api` | 4 |
| **植被** | `ndvi`, `ndwi` | 2 |
| **坐标** | `lon`, `lat` | 2 |
| **土地利用** | `landuse` + One-hot(6个) | 7 |
| **衍生特征** | `rain_ratio_3d_30d`, `rain_ratio_7d_30d`, `rain_intensity` | |
| | `valley_risk`, `elev_rain`, `aspect_rain`, `ndvi_elev` | 7 |
| | `log_dist_river` | 1 |

#### 暴露特征（不参与分类，用于风险分析）

| 特征 | 说明 |
|:-----|:-----|
| `threatened_people` | 威胁人口数 |
| `threatened_households` | 威胁户数 |
| `threatened_property` | 威胁财产（万元） |
| `exposure_index` | 综合暴露指数 = 0.4×People + 0.3×HH + 0.3×Property |

> **关键设计**：暴露特征在正样本中有正值，负样本中全为0。若加入分类特征，模型会"作弊"（仅靠暴露度就能区分），无法学到真实的滑坡规律。因此将其分离到风险分析阶段。

#### 衍生特征详解

| 特征 | 公式 | 物理意义 |
|:-----|:-----|----------|
| `rain_ratio_3d_30d` | rain_3d / rain_30d | 短期降雨集中度 |
| `rain_ratio_7d_30d` | rain_7d / rain_30d | 中期降雨集中度 |
| `rain_intensity` | rain_3d / rain_7d | 降雨强度变化率 |
| `valley_risk` | max(0, -tpi) / (dist_river + 1) | 沟谷汇水风险 |
| `elev_rain` | elevation × log(1 + rain_3d) | 高海拔+强降雨交互 |
| `aspect_rain` | cos(aspect) × rain_3d | 迎风坡降雨增强 |
| `ndvi_elev` | ndvi × elevation / 1000 | 植被覆盖与高程耦合 |

---

### 4. 负样本生成

**文件**: `pre_process/negative_sampling.py`

#### 算法原理

```python
1. 确定采样范围: 正样本经纬度范围扩展5%
2. 构建空间索引: BallTree(haversine距离)
3. 循环采样:
   a. 在范围内随机生成候选点(lat, lon)
   b. 查询最近灾害点距离
   c. 距离 > 500m → 接受，否则重试
4. 特征继承:
   a. 找到最近3个灾害点
   b. 加权平均(按反距离) + 高斯噪声
   c. 暴露度置0
5. 标签: 0（非灾害点）
```

#### 参数配置

| 参数 | 默认值 | 说明 |
|:-----|:------:|------|
| `ratio` | 1.0 | 负:正样本比例 |
| `min_dist_km` | 0.5 | 距灾害点最小距离(km) |
| `n_neighbors` | 3 | 特征继承的最近邻数 |

#### 质量检查

```
• 特征分布对比（均值漂移 < 1σ）
• 暴露度全为0校验
• 空间范围覆盖校验
```

---

### 5. 二阶段建模策略

这是本项目的核心方法论设计。

#### 第一阶段：滑坡易发性（Susceptibility）

```python
输入特征: 地形 + 降雨 + 植被 + 土地利用（27维）
标签: 1 = 灾害点, 0 = 非灾害点
模型: 9个分类器 + 4个集成方法
输出: P(灾害 | 地形,降雨,植被)
```

#### 第二阶段：滑坡风险（Risk）

```python
输入: 易发性概率 P_sus
      暴露度指数 E = 0.4·People + 0.3·HH + 0.3·Property
风险: R = P_sus × E
分级: 极低 / 低 / 中 / 高 / 极高
```

这样设计的原因：

| 常见错误做法 | 本项目的正确做法 |
|:-------------|:----------------|
| 暴露特征直接加入分类模型 | ❌ 模型学会"有人的地方就是灾害" | 暴露特征分离到风险阶段 | ✅ 模型学习真实滑坡规律 |
| 忽略暴露度，仅预测灾害概率 | ❌ 无法评估实际危害程度 | 二阶段级联：易发性 × 暴露度 | ✅ 同时考虑自然和人文因素 |

---

## 模型列表

### 基模型（9个）

| 模型 | 参数配置 |
|:----|:---------|
| 逻辑回归 | `class_weight='balanced', C=1.0, solver='saga'` |
| 决策树 | `max_depth=8, class_weight='balanced'` |
| SVM | `kernel='rbf', probability=True, class_weight='balanced'` |
| 随机森林 | `n_estimators=200, class_weight='balanced_subsample'` |
| KNN | `n_neighbors=7, weights='distance'` |
| GBDT | `n_estimators=200, max_depth=4, lr=0.1` |
| XGBoost | `scale_pos_weight=正负比` |
| CatBoost | `auto_class_weights='Balanced'` |
| LightGBM | `class_weight='balanced'` |

### 集成模型（4个）

| 模型 | 策略 |
|:----|:------|
| **平均值集成** | 9个模型概率的加权平均 |
| **Voting** | 软投票（概率均值） |
| **Stacking** | 5折交叉验证 + 逻辑回归元学习器 |
| **Blending** | hold-out验证集 + 逻辑回归元学习器 |

### 后处理

```python
• 概率校准: CalibratedClassifierCV(sigmoid, cv=3)
• 阈值优化: 多目标搜索 (0.6×F1 + 0.4×特异性), 步长0.01
```

---

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行全部训练

```bash
# 默认 1:1 正负样本比例
python train_models_v2.py

# 自定义负样本比例（快速测试用0.5）
python train_models_v2.py --neg-ratio 0.5

# 开启Optuna超参优化（对RF/XGB/LGB/CAT各跑30次）
python train_models_v2.py --optuna-trials 30

# 完整命令：负样本1:1 + Optuna调参 + SHAP分析
python train_models_v2.py --neg-ratio 1.0 --optuna-trials 30

# 跳过SHAP分析
python train_models_v2.py --no-shap
```

### 危险性分级与风险制图

先训练模型，然后运行：

```bash
# 自动选择最佳模型
python hazard_mapping.py

# 指定模型
python hazard_mapping.py --model-path models_v2/xgboost.pkl

# 用等距分级法（默认五分位数）
python hazard_mapping.py --method equal
```

### 输出文件

```text
results_v2/
├── model_performance.csv         # 13+个模型按F1排名
├── roc_curves.png                # ROC曲线对比
├── pr_curves.png                 # PR曲线对比
├── shap_summary.png              # SHAP特征重要性
├── shap_bar.png                  # SHAP柱状图
├── shap_importance.csv           # SHAP值排序
│
├── susceptibility_distribution.png   # 易发性概率分布
├── hazard_classification.png         # 五级危险性空间分布
├── risk_map.png                      # 综合风险(易发性×暴露度)空间分布
├── hazard_analysis_dashboard.png     # 综合分析仪表盘
├── hazard_breakdown.csv              # 各等级统计汇总
└── hazard_prediction_full.csv        # 全量预测数据

models_v2/
├── xgboost.pkl
├── lightgbm.pkl
├── random_forest.pkl
├── catboost.pkl
├── xgboost_tuned.pkl          # Optuna调优版
├── random_forest_tuned.pkl
├── voting.pkl
├── stacking.pkl
├── blending.pkl
└── ...
```

### 自定义调用

```python
from pre_process.pipeline_v2 import preprocess_data_v2

# 获取训练数据
X_train, X_test, y_train, y_test, extra = preprocess_data_v2(
    neg_ratio=1.0, verbose=True
)

# 易发性特征
feature_names = extra['feature_names']  # 27个特征名

# 暴露特征（风险分析用）
exposure_train = extra['exposure_train']  # 训练集暴露度
exposure_test = extra['exposure_test']    # 测试集暴露度

# 用你训练的模型预测易发性
y_prob = model.predict_proba(X_test)[:, 1]

# 计算综合风险
risk = y_prob * exposure_test['exposure_index'].values
```

---

## 项目结构

```
landslide_prediction/
├── dataset/
│   └── aba_disaster_distribution.csv       # 原始数据（7315×31）
│
├── pre_process/                            # 预处理模块（新增）
│   ├── __init__.py                         # 导出全部新模块
│   ├── data_cleaning.py                    # 数据清洗（NEW）
│   ├── imputation.py                       # 缺失值插补（NEW）
│   ├── negative_sampling.py                # 负样本生成（NEW）
│   ├── pipeline_v2.py                      # 核心编排（NEW）
│   ├── preprocess.py                       # 原有（V1保留）
│   ├── data_augmentation.py                # 原有（V1保留）
│   └── feature_analysis.py                 # 原有（V1保留）
│
├── ML_algorithms/                          # 9个基模型（原有）
│   ├── train_logistic_regression.py
│   ├── train_decision_tree.py
│   └── ...                                 # 全部保留
│
├── improve_algorithms/                     # 4个集成模型（原有）
│   ├── train_blending.py
│   ├── train_ensemble_avg.py
│   ├── train_voting.py
│   └── train_stacking.py
│
├── train_models_v2.py                      # V2训练入口（NEW）
├── train_models.py                         # V1训练入口（原有保留）
│
├── models_v2/                              # V2模型输出（自动创建）
├── results_v2/                             # V2结果输出（自动创建）
├── models/                                 # V1模型输出（原有）
├── train_logs/                             # 训练日志（原有）
│
├── visualisation.py                        # 可视化模块（原有）
├── requirements.txt                        # 依赖（已更新）
├── README.md                               # V1文档
└── README_V2.md                            # 本文档（NEW）
```

---

## 评估指标

| 指标 | 公式 | 说明 |
|:-----|:-----|:-----|
| **准确率** | (TP+TN) / (TP+TN+FP+FN) | 总体正确率 |
| **精确率** | TP / (TP+FP) | 预测为正的样本中实际为正的比例 |
| **召回率** | TP / (TP+FN) | 实际为正的样本中被正确识别的比例 |
| **F1值** | 2×P×R / (P+R) | 精确率和召回率的调和平均 |
| **特异性** | TN / (TN+FP) | 负样本被正确识别的比例 |
| **AUC-ROC** | ROC曲线下面积 | 整体排序质量 |
| **AUC-PR** | PR曲线下面积 | 适合不平衡数据 |

> **排序指标**：所有模型按 **F1值** 排序（兼顾精确率和召回率）。

---

## Optuna 超参优化

`train_models_v2.py` 集成了 Optuna 贝叶斯超参优化，支持对4个主要模型进行自动调参。

### 优化目标

| 模型 | 调优参数 | 搜索空间 |
|:----|:---------|:---------|
| **随机森林** | n_estimators, max_depth, min_samples_split, max_features | 100-600棵树, 5-25层 |
| **XGBoost** | n_estimators, max_depth, lr, subsample, colsample, gamma, reg | 含正则化参数 |
| **LightGBM** | n_estimators, max_depth, num_leaves, lr, subsample, reg | 含叶子节点数 |
| **CatBoost** | iterations, depth, lr, subsample, l2_leaf_reg | 4-10层深度 |

### 运行方式

```bash
# 每个模型30次调参（约10-15分钟）
python train_models_v2.py --optuna-trials 30

# 每个模型50次（更充分，约20-30分钟）
python train_models_v2.py --optuna-trials 50
```

### 输出

```text
Optuna 调参 random_forest (30 次)... 最佳F1=0.9852
Optuna 调参 xgboost (30 次)...      最佳F1=0.9921
Optuna 调参 lightgbm (30 次)...     最佳F1=0.9915
Optuna 调参 catboost (30 次)...     最佳F1=0.9918
```

调优后的模型自动保存为 `models_v2/{name}_tuned.pkl`，并参与最终排名。

---

## 危险性分级与风险分析

`hazard_mapping.py` 在模型训练完成后，对全量数据进行易发性分级和综合风险评估。

### 二阶段风险框架

```text
第一阶段: 滑坡易发性 Susceptibility
  输入: 地形 + 降雨 + 植被 + 土地利用
  输出: P(灾害) ∈ [0, 1]
  分级: 极低 / 低 / 中 / 高 / 极高

第二阶段: 综合风险 Risk
  风险 = 易发性 × 暴露度
  暴露度 = 0.4·(People) + 0.3·(HH) + 0.3·(Property)
  分级: 极低 / 低 / 中 / 高 / 极高
```

### 分级方法

| 方法 | 说明 |
|:-----|:-----|
| `quantile`（默认） | 五分位数等样本量分级，每级恰好20% |
| `equal` | 等距分级 [0,0.2,0.4,0.6,0.8,1] |

### 输出图表

| 图表 | 内容 |
|:-----|:------|
| `susceptibility_distribution.png` | 易发性概率分布直方图 + 五级标注 |
| `hazard_classification.png` | 五级危险性空间散点图 |
| `risk_map.png` | 综合风险(易发性×暴露度)空间分布 |
| `hazard_analysis_dashboard.png` | 综合分析仪表盘（分布+空间+统计） |
| `hazard_breakdown.csv` | 各等级统计汇总 |
| `hazard_prediction_full.csv` | 全量预测数据下载 |

---

## 引用说明

若使用本项目的代码或方法，请引用：

```bibtex
@software{landslide_prediction_v2,
  title = {阿坝州降雨诱发滑坡易发性评价 - Pipeline V2},
  author = {Deng Shuanglin},
  year = {2026},
  description = {基于空间约束负样本生成和二阶段易发性-暴露度分离的滑坡易发性机器学习评价}
}
```

---

## 注意事项

1. **负样本生成**：负样本的合理性直接影响模型质量。建议根据研究区实际情况调整 `min_dist_km` 参数。
2. **NDVI插补**：43%的NDVI缺失率较高，若有条件建议从GEE重新提取以替代插补。
3. **暴露度分析**：当前风险分析为简化模型（线性加权），实际应用中可考虑更复杂的暴露度评估方法。
4. **模型选择**：若追求可解释性优先选择随机森林或XGBoost；若追求性能可尝试Stacking集成。
