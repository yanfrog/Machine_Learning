# Python 工具（Machine Learning 常用套件）

以下為在機器學習流程中常用的 Python 套件，依用途分類與實際舉例。

---

| 套件         | 用途                        | 舉例說明                                               |
|--------------|-----------------------------|--------------------------------------------------------|
| **NumPy**     | 陣列運算、線性代數          | 建立矩陣、內積、轉置等操作：`np.dot(A, B)`             |
| **Pandas**    | 資料處理與表格操作          | 讀取 CSV、缺值填補、欄位轉換：`df.fillna(0)`          |
| **Matplotlib**| 資料視覺化                  | 繪製折線圖、直方圖、熱圖等：`plt.plot(x, y)`          |
| **Scikit-learn** | 演算法與前處理工具       | 包含 SVM、KNN、標準化、切分資料：`train_test_split()` |
| **PyTorch**   | 建立深度學習模型            | 使用 `nn.Module` 建構 CNN、LSTM、Transformer 等模型   |

---

## 常見使用範例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
# NumPy - 建立隨機陣列
a = np.random.rand(3, 3)

# Pandas - 讀資料、簡單處理
df = pd.read_csv("data.csv")
df["age"].fillna(df["age"].mean(), inplace=True)

# Matplotlib - 畫圖
plt.hist(df["age"])
plt.title("Age Distribution")
plt.show()

# Scikit-learn - 資料切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# PyTorch - 模型定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

## 延伸工具

| 套件          | 補充用途                 |
| ----------- | -------------------- |
| Seaborn     | 更美觀的統計圖形             |
| tqdm        | 顯示訓練進度條              |
| Optuna      | 自動化超參數搜尋             |
| TensorBoard | 訓練過程視覺化（可搭配 PyTorch） |