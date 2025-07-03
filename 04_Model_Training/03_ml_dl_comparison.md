# 機器學習與深度學習中常見要素比較（ML vs DL）

---

## 分類總覽表

| 項目         | 用途                                 | 機器學習（ML） | 深度學習（DL） | 說明與補充                                  |
|--------------|--------------------------------------|----------------|----------------|---------------------------------------------|
| **損失函數** | 衡量模型預測與真實值的誤差           | ✅              | ✅              | ML 與 DL 訓練皆需損失函數作為優化目標       |
| **優化器**   | 最小化損失函數，更新模型參數         | ❌（內建演算法）| ✅              | 傳統 ML 通常內含封裝解法；DL 需手動指定    |
| **評估指標** | 測量模型在驗證集 / 測試集的表現       | ✅              | ✅              | 模型評估皆需，與訓練過程無直接關係         |
| **激活函數** | 為神經網路引入非線性，使模型能學複雜關係 | ❌              | ✅              | 僅用於神經網路，增加非線性表達能力         |


---

## 補充說明

### 1. 損失函數（Loss Function）

- **ML 範例：**
  - 線性回歸 → MSE
  - Logistic Regression → BCE（Binary Cross Entropy）
- **DL 範例：**
  - CNN / RNN → CrossEntropyLoss
  - GAN → 對抗損失

### 2. 優化器（Optimizer）

- **ML：**
  - 模型內含訓練方法，如 SVM 的 SMO、XGBoost 的 boosting tree 更新
- **DL：**
  - 手動指定：`SGD`, `Adam`, `RMSprop` 等
  - PyTorch 使用範例：
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ```

### 3. 評估指標（Evaluation Metrics）

- **ML & DL 共用：**
  - `accuracy_score`, `precision`, `recall`, `f1_score`, `r2_score`
- **DL 常見：**
  - 可搭配 `torchmetrics`、或自訂寫法搭配 `torch.no_grad()`

### 4. 激活函數（Activation Function）

- 僅用於深度學習中，用來引入**非線性**，否則多層神經網路會退化成線性模型

| 函數     | 特性與應用         |
|----------|--------------------|
| ReLU     | 區塊狀線性，收斂快  |
| Sigmoid  | 將輸出壓縮至 0~1    |
| Tanh     | 輸出壓縮至 -1~1     |
| Softmax  | 分類時轉為機率輸出  |

---

## 結論整理

- 損失函數與評估指標是通用元件，ML/DL 都會使用
- 優化器與激活函數是深度學習模型專屬的訓練關鍵
- 兩者可在流程設計上互補：以 ML 做 baseline，再進階用 DL 增強表現
