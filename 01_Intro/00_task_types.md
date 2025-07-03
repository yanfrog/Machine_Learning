# 機器學習任務分類

機器學習可依照資料是否具備標籤（Label）分為三大任務類型：

---

## 監督式學習（Supervised Learning）

### 特徵：
- 有輸入與對應的標籤（X → y）
- 學習目標：找到從輸入到標籤的映射函數

### 常見任務：
- 回歸（Regression）：預測連續值（如房價）
- 分類（Classification）：預測類別（如圖片分類）

### 常見演算法：
- 線性/邏輯回歸、KNN、SVM、決策樹、隨機森林、神經網路等

---

## 非監督式學習（Unsupervised Learning）

### 特徵：
- 資料只有輸入（X），沒有標籤（y）
- 目標是從資料中找出結構或模式

### 常見任務：
- 分群（Clustering）：如顧客分群
- 降維（Dimensionality Reduction）：如 PCA

### 常見演算法：
- K-Means、DBSCAN、層次分群、Autoencoder、主成分分析（PCA）

---

## 強化學習（Reinforcement Learning）

### 特徵：
- 智能體（Agent）透過與環境互動來學習策略
- 根據獎勵（Reward）最大化學習行為

### 常見應用：
- 遊戲、機器人控制、自駕車、金融交易

### 常見演算法：
- Q-Learning、Deep Q Network（DQN）、Policy Gradient、Actor-Critic

---

## 機器學習三大任務比較：回歸、分類、聚類

| 項目              | Regression（回歸）                  | Classification（分類）               | Clustering（聚類）                   |
|-------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| **問題性質**      | 預測連續值                          | 預測離散類別（標籤）                 | 找出資料中的內部群組                 |
| **輸出類型**      | 實數（ℝ）                           | 類別標籤（如 0/1、A/B/C）            | 類別編號（無標籤，通常為群組編號）   |
| **是否有標籤**    | 監督式學習（Supervised）         | 監督式學習                        | 非監督式學習（Unsupervised）     |
| **典型演算法**    | Linear Regression、SVR、XGBoost     | Logistic Regression、SVM、Random Forest | KMeans、DBSCAN、Gaussian Mixture    |
| **評估指標**      | MSE、RMSE、MAE、R²                   | Accuracy、Precision、Recall、F1       | 輪廓係數（Silhouette Score）、SSE   |
| **範例應用**      | 房價預測、股價預測、溫度預測        | 郵件分類、圖片辨識、癌症檢測         | 顧客分群、圖像分群、新聞主題分群     |

---

### 差異總結

- **監督 vs 非監督**：回歸與分類都屬於監督式學習，聚類則是非監督式。
- **輸出形式**：
  - 回歸 → 連續數值（例：預測溫度 30.5℃）
  - 分類 → 類別（例：圖像屬於「貓」或「狗」）
  - 聚類 → 群組 ID（例：顧客屬於第 3 群）
- **使用場景不同**：
  - 回歸：數量型預測
  - 分類：類別判斷任務
  - 聚類：資料探索與自動分群

---

## 深度學習

> **深度學習（Deep Learning）是一種方法，不是任務類型。**

它可以應用在所有上面三種類型中：

| 任務分類     | 可否使用深度學習 | 常見模型示例                          |
|--------------|------------------|---------------------------------------|
| 監督式學習   | ✅ 是            | CNN 圖像分類、RNN 序列標註            |
| 非監督式學習 | ✅ 是            | Autoencoder、GAN                      |
| 強化學習     | ✅ 是            | Deep Q Network（DQN）、Policy Gradient |

