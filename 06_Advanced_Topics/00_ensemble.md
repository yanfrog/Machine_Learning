# 集成學習（Ensemble Learning）

集成學習是一種將多個基礎模型組合以提升預測效能的方法，能有效減少過擬合、提升準確率與穩定性。

---

## 一、主要概念

- 結合多個「弱學習器」變成「強學習器」
- 模型之間可以是相同架構（如多棵決策樹）或不同架構（Stacking）

---

## 二、常見類型

| 方法        | 說明                                               | 常見演算法        |
|-------------|----------------------------------------------------|-------------------|
| **Bagging** | 並行訓練多個模型並取平均或投票結果                  | Random Forest     |
| **Boosting**| 序列式訓練模型，每次聚焦於前一次錯誤樣本            | XGBoost, AdaBoost |
| **Stacking**| 將多模型輸出作為特徵餵給另一個「總管模型」          | 任意組合皆可       |

---

## 三、常見集成模型

### Random Forest（隨機森林）

- Bagging 的代表性模型
- 訓練多棵決策樹，結果取投票（分類）或平均（回歸）

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

---

### Gradient Boosting（梯度提升機）

- Boosting 代表模型之一
- 每一個新模型都是為了修正前一個模型的誤差

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
```

---

### AdaBoost（自適應提升）

- 根據前一輪錯誤樣本給予更高權重

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(X_train, y_train)
```

---

### XGBoost（極端梯度提升）

- 高效能 Boosting 演算法，支援 missing value、自動剪枝、並行計算

```python
import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
```

---

## 四、使用時機

- 資料不大但模型表現不穩定時（適合 Bagging）
- 想壓榨出最佳效能時（Boosting 表現更好）
- 不同模型預測結果互補性高時（可考慮 Stacking）

---

## 五、優缺點比較

| 方法     | 優點                            | 缺點                          |
|----------|---------------------------------|-------------------------------|
| Bagging  | 穩定、減少過擬合                | 訓練資源較高，解釋性略低       |
| Boosting | 高準確率、效能佳                | 訓練時間長、易過擬合（需正規化）|
| Stacking | 可融合多模型特性、效果彈性大    | 建構與調參較複雜               |
