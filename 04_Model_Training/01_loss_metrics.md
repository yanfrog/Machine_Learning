# 損失函數與模型評估指標（Loss Functions & Evaluation Metrics）

---

## 一、損失函數（Loss Functions）

損失函數（Loss Function）

- 用途：衡量模型預測與真實值的差距
    - 每一次訓練都會計算損失值（loss）來知道模型表現好不好
    - 衡量模型預測值與實際值的差異，是模型訓練的優化目標
- 使用時機：
    - 定義問題類型後立即決定，必須在模型訓練前先選好
    - 不同任務使用不同損失函數

### 1. 回歸任務常用

| 名稱           | 說明                     | PyTorch 對應函數     |
|----------------|--------------------------|----------------------|
| MSE（均方誤差） | 對誤差平方後取平均       | `nn.MSELoss()`       |
| MAE（平均絕對誤差）| 對誤差取絕對值後平均    | `nn.L1Loss()`        |
| Huber Loss     | 兼具 MAE 與 MSE 特性     | `nn.SmoothL1Loss()`  |

**MSE 公式：**
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

---

### 2. 分類任務常用

| 類型         | 說明                                            | PyTorch 損失函數           |
|--------------|--------------------------------------------------|----------------------------|
| 二元分類     | 適合輸出機率為單一值（搭配 Sigmoid）             | `nn.BCELoss()`             |
| 二元分類     | 搭配未加 Sigmoid 的輸出（數值穩定）              | `nn.BCEWithLogitsLoss()`   |
| 多類別分類   | 適合 Softmax 多分類（輸出 logits）               | `nn.CrossEntropyLoss()`    |

**Binary Cross Entropy：**
\[
\mathcal{L}(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
\]

**Cross Entropy（多分類）：**
\[
\mathcal{L}(y, \hat{y}) = - \sum_{c=1}^C y_c \log(\hat{y}_c)
\]

---

## 二、優化器（Optimizers）

優化器負責根據損失函數的結果，調整模型參數以降低誤差。

| 名稱       | 特性說明                                     | PyTorch 實作                 |
|------------|----------------------------------------------|------------------------------|
| SGD        | 最基本隨機梯度下降，需手動調參學習率         | `torch.optim.SGD()`          |
| SGD + Momentum | 引入慣性避免震盪                          | `torch.optim.SGD(momentum=0.9)` |
| Adam       | 自適應學習率，收斂快、穩定性好               | `torch.optim.Adam()`         |
| AdamW      | 改進版 Adam，分離權重衰減                    | `torch.optim.AdamW()`        |
| RMSProp    | 適用於非平穩目標、序列資料                    | `torch.optim.RMSprop()`      |

```python
# 使用 Adam 優化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# SGD + Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

---

## 三、學習率調整器（Scheduler，進階）

可根據 epoch 或驗證集表現自動調整學習率。

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# gamma=0.1 表示每 step_size 個 epoch 將 lr 乘以 0.1
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

---

## 四、模型評估指標（Evaluation Metrics）

### 1. 分類任務

| 指標         | 定義                                       | 適用類型       |
|--------------|--------------------------------------------|----------------|
| Accuracy     | 所有預測中正確的比例                        | 二元、多類別   |
| Precision    | 預測為正例中，實際為正例的比例              | 偏好低誤報     |
| Recall       | 實際為正例中，被預測為正例的比例            | 偏好低漏報     |
| F1-score     | Precision 與 Recall 的調和平均              | 綜合評估       |
| ROC-AUC      | 分類閾值改變時，TPR 對 FPR 的面積            | 二元分類       |

**Precision / Recall 公式**
\[
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}
\]

**F1-score 公式：**
\[
F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

---

### 2. 回歸任務

| 指標         | 說明                             | 備註                         |
|--------------|----------------------------------|------------------------------|
| MSE          | 均方誤差                         | 同上                         |
| MAE          | 平均絕對誤差                     | 對異常值較穩定               |
| RMSE         | MSE 開根號                       | 與資料單位一致               |
| \( R^2 \)    | 解釋變異比例，越接近 1 越好      | 負值表示模型表現劣於平均值  |

**R² Score：**
\[
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
\]

### 3.範例

```python
from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

---

## 五、總結對照表

| 任務類別     | 選擇條件                         | 損失函數（Loss）               | 優化器（Optimizer）     | 評估指標（Metrics）                     |
|--------------|----------------------------------|-------------------------------|--------------------------|------------------------------------------|
| **回歸**     | 連續數值預測                     | `MSELoss`, `L1Loss`           | `Adam`, `SGD`            | MAE, MSE, RMSE, R² Score                 |
| **二元分類** | 預測 0/1，輸出機率               | `BCELoss`, `BCEWithLogitsLoss`| `Adam`, `AdamW`, `SGD`   | Accuracy, Precision, Recall, F1, ROC-AUC|
| **多分類**   | 多類標籤分類，輸出 logits        | `CrossEntropyLoss`            | `Adam`, `SGD`, `AdamW`   | Accuracy, Macro-F1, Confusion Matrix    |


---

## 六、PyTorch 中使用方式簡例

```python
# 損失函數
loss_fn = nn.CrossEntropyLoss()  # 多分類
loss = loss_fn(pred_logits, target_labels)

# 評估（以 accuracy 為例）
with torch.no_grad():
    pred_labels = torch.argmax(pred_logits, dim=1)
    acc = (pred_labels == target_labels).float().mean()
```