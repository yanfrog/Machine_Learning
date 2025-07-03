# 資料視覺化（Data Visualization）

---

## 一、常用套件

| 套件      | 特性                             | 常見用途                           |
|-----------|----------------------------------|------------------------------------|
| Matplotlib| 最基本、低階控制多                | 折線圖、長條圖、散佈圖、直方圖等   |
| Seaborn   | 建構於 Matplotlib 上的美化套件     | 熱力圖、盒鬚圖、類別分群視覺化     |
| Plotly    | 支援互動式視覺化、支援網頁嵌入     | 動態圖表、儀表板、3D 圖             |

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

---

## 二、常見圖表類型與用途

| 圖表類型    | 用途                         | 建議套件          |
|-------------|------------------------------|-------------------|
| 散佈圖      | 顯示兩特徵關係（分群、趨勢）   | Matplotlib, Seaborn |
| 熱力圖      | 顯示變數之間的相關性矩陣       | Seaborn           |
| 盒鬚圖      | 查看離群值、資料分布狀況       | Seaborn           |
| ROC 曲線    | 分類模型性能評估（TPR vs FPR）| Matplotlib        |
| 長條圖      | 類別資料數量或平均值比較       | Matplotlib        |
| 折線圖      | 時序數據視覺化                 | Matplotlib        |

---

## 三、簡易範例

### 散佈圖（scatter plot）

```python
plt.scatter(df["feature1"], df["feature2"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Feature1 vs Feature2")
plt.show()
```

### 熱力圖（heatmap）

```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

### 盒鬚圖（boxplot）

```python
sns.boxplot(x="label", y="value", data=df)
plt.title("Boxplot by Label")
plt.show()
```

### ROC 曲線（for binary classification）

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
```

---

## 四、建議

- 資料探索初期使用 **Seaborn 熱力圖 / 盒鬚圖**
- 模型評估後期使用 **ROC 曲線、混淆矩陣圖**
- 發表與 demo 可用 **Plotly** 提升互動性與美觀
