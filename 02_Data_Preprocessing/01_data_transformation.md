# 資料轉換（Data Transformation）

資料轉換是將原始資料轉為適合模型處理的形式，常見轉換包括標準化、正規化與編碼。

---

## 標準化（Standardization）

將特徵轉換為均值為 0、標準差為 1 的分佈。

公式：
```
x' = (x - μ) / σ
```

工具：
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 正規化（Normalization）

將數值壓縮到特定範圍（如 [0,1]）。

公式：
```
x' = (x - min) / (max - min)
```

工具：
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 類別資料編碼（Encoding）

將文字標籤轉換為數值

| 方法         | 說明                           |
|--------------|--------------------------------|
| Label Encoding | 將每個類別轉為數值            |
| One-Hot Encoding | 為每個類別建立獨立欄位      |

工具：
```python
pd.get_dummies(df['category'])
LabelEncoder().fit_transform(df['label'])
```

---

## 資料增強（Data Augmentation）

資料增強常用於圖像任務，透過隨機變換提高模型泛化能力。

| 技術         | 說明                           |
|--------------|--------------------------------|
| 隨機旋轉     | 在一定角度內隨機旋轉圖像        |
| 隨機裁剪     | 剪裁圖像部分區域                |
| 水平翻轉     | 左右翻轉圖像                    |
| 色彩調整     | 改變亮度、對比、飽和度等        |

工具範例：
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
```

---

## Dropout（隨機捨棄）

Dropout 是一種防止過擬合的技術，在訓練過程中隨機將部分神經元設為 0。

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 10)
)
```

---

## 批次正規化（Batch Normalization）

對每層輸出做標準化（均值為 0、方差為 1），能加速訓練並提升穩定性。

```python
nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3),
    nn.BatchNorm2d(16),
    nn.ReLU()
)
```
