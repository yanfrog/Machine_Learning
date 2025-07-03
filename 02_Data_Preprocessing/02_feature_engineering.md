# 特徵工程（Feature Engineering）

特徵工程是在資料中找出對模型有幫助的特徵，是影響預測效能的關鍵因素之一。

---

## 特徵選取（Feature Selection）

挑選與目標最有關的特徵

| 方法           | 說明                          |
|----------------|-------------------------------|
| Filter 方法     | 根據統計量（如卡方、相關係數）選取 |
| Wrapper 方法    | 利用模型評估（如遞迴特徵消除）    |
| Embedded 方法   | 模型自帶選擇（如 Lasso）         |

---

## 特徵建立（Feature Construction）

- 新增統計值欄位（如平均、總和、標準差）
- 建立交互作用特徵（如 A × B）
- 文本特徵：TF-IDF、N-gram

---

## 特徵降維（Dimensionality Reduction）

| 方法     | 說明                      |
|----------|---------------------------|
| PCA      | 主成分分析，保留主要資訊  |
| t-SNE    | 資料視覺化                |
| LDA      | 為分類問題做降維          |

工具：
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)
```
