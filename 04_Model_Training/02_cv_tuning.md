# 調參與交叉驗證

## 交叉驗證（Cross Validation）

- K-Fold：將資料分成 K 份，輪流做驗證集
- Stratified K-Fold：針對分類任務分層抽樣

## 調參技巧

| 方法             | 說明                           |
|------------------|--------------------------------|
| Grid Search      | 枚舉所有參數組合，耗時高        |
| Random Search    | 隨機抽樣參數組合，較快          |
| Bayesian Opt     | 根據歷史調參結果導向搜索        |

## 工具

- Scikit-learn: `GridSearchCV`
- PyTorch: 自定義搜尋與 EarlyStopping
