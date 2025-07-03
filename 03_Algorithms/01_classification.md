# 分類模型（Classification）

分類是預測離散類別的監督式學習任務。

## 常見演算法

| 模型        | 特性與應用                         |
|-------------|------------------------------------|
| 邏輯回歸     | 線性可分問題，適合二元分類         |
| KNN         | 基於距離的鄰近分類器               |
| SVM         | 尋找最大間隔的分隔超平面           |
| 決策樹       | 規則性強，解釋性佳                 |
| 隨機森林     | 多棵決策樹組合，抗過擬合           |

## 評估指標

- Accuracy、Precision、Recall、F1-score
- Confusion Matrix、ROC Curve

## PyTorch 提示

通常使用 `nn.CrossEntropyLoss` 作為分類任務損失函數。


# Logistic Regression（邏輯式回歸）

## 一、概念說明

- 適用於 **二元分類問題**，例如：是/否、正/負、0/1。
- 是一種 **線性分類器**，但輸出經過 Sigmoid 函數轉為機率。
- 屬於 **監督式學習**。

---

## 二、數學模型與推導

### 模型假設：

\[
\hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
\]

- \( x \)：輸入特徵向量
- \( w \)：權重向量
- \( b \)：偏差項
- \( \sigma(\cdot) \)：Sigmoid 函數（將線性結果壓縮為 [0, 1] 機率）

---

### 決策邊界：

將預測機率與 0.5 比較，決定預測類別：

\[
\hat{y} \geq 0.5 \Rightarrow y = 1，\quad \hat{y} < 0.5 \Rightarrow y = 0
\]

---

## 三、損失函數

使用 **Binary Cross Entropy（BCE）**：

\[
\mathcal{L}(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
\]

- 適合機率輸出
- 對數損失可放大錯誤預測的懲罰

---

## 四、PyTorch 實作教學

```python
import torch
import torch.nn as nn

# 假設特徵數量為 10
class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 初始化模型
model = LogisticModel(input_dim=10)

# 損失函數使用 BCE（適用 Sigmoid 輸出）
loss_fn = nn.BCELoss()

# 優化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 假設資料 X: (batch_size, 10)，y: (batch_size, 1)
X = torch.randn(16, 10)
y = torch.randint(0, 2, (16, 1)).float()

# 訓練步驟
for epoch in range(100):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```