# 專案實作案例（Practical ML Projects）

以下為三個常見任務的完整實作概覽，包括任務定義、資料來源、模型選擇與 PyTorch 實作概念。

---

## 1️⃣ 房價預測（回歸任務）

- **任務類型**：回歸（Regression）
- **資料來源**：[Kaggle Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **目標**：預測房屋的最終售價

### 🔧 技術點
- 特徵工程（面積、房齡、區域等）
- 類別特徵編碼（One-hot）
- log 轉換（對於偏態的房價）
- 數值標準化（StandardScaler）

### PyTorch 建模範例
```python
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
```

---

## 2️⃣ 圖片分類（多類別分類）

- **任務類型**：多類別分類
- **資料來源**：CIFAR-10 / 自定義資料集
- **目標**：將圖片分成 10 個類別（貓、狗、車、飛機等）

### 🔧 技術點
- 使用 CNN 架構進行影像分類
- 資料增強（旋轉、剪裁、翻轉）
- Dropout、BatchNorm 提升泛化能力

### PyTorch 模型（簡化 CNN）
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)  # CIFAR-10 = 32x32x3
        )

    def forward(self, x):
        return self.net(x)
```

---

## 3️⃣ 文本分類（情緒分析）

- **任務類型**：二元分類（Positive / Negative）
- **資料來源**：IMDb / Yelp Reviews
- **目標**：預測評論為正面或負面

### 🔧 技術點
- 文本前處理（清理、Tokenize、Lowercase）
- Padding、Embedding 向量化
- 使用 RNN / LSTM 進行序列建模

### PyTorch 模型（簡化版 LSTM）
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return torch.sigmoid(self.fc(hn.squeeze(0)))
```

---