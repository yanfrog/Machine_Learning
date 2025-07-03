# 回歸模型（Regression）

預測連續數值的「監督式學習」任務。

## 線性回歸（Linear Regression）

目標：學習一條最佳直線來「預測 y」

模型：
```
y = w^T x + b
```

損失函數：最小化 MSE（均方誤差）

## 多項式回歸（Polynomial Regression）

在線性回歸中加入高次項：
```
y = w0 + w1x + w2x^2 + ...
```

## Lasso / Ridge / ElasticNet

- Lasso：L1 正則化（可做特徵選擇）
- Ridge：L2 正則化（避免過擬合）
- ElasticNet：混合 L1 + L2

## PyTorch 實作簡例

```python
import torch
from torch import nn

model = nn.Linear(in_features=1, out_features=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
