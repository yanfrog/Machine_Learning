
# 神經網路（NN）重點總整理

## 一、人工神經網路 ANN（Artificial Neural Network）

### 結構組成

```
Input Layer → Hidden Layer(s) → Output Layer
```

- 每層由多個「神經元（Neuron）」組成
- 每個神經元計算：  
  z = w^T x + b，a = σ(z)

### 激活函數（Activation Functions）

| 函數      | 公式                     | 特性                            |
|-----------|--------------------------|---------------------------------|
| Sigmoid   | σ(x) = 1 / (1 + e^(-x))  | 非線性，會造成梯度消失         |
| ReLU      | f(x) = max(0, x)         | 快速收斂，簡單有效              |
| Tanh      | (e^x - e^-x)/(e^x + e^-x)| 零中心，仍可能梯度消失         |


## 二、神經網路訓練：梯度下降與反向傳播

### 損失函數（Loss Function）
- 常用：Mean Squared Error (MSE)、Categorical Cross-Entropy

### 🔁 反向傳播（Backpropagation, BP）
1. Forward Pass：計算每層輸出  
2. Loss 計算：比較預測與實際  
3. Backward Pass：反向傳遞誤差，更新權重

### 梯度下降法（Gradient Descent）

| 變種           | 說明                                 |
|----------------|--------------------------------------|
| Batch GD       | 全資料一次訓練，穩定但慢              |
| Stochastic GD  | 每筆資料更新，收斂不穩定但快          |
| Mini-Batch GD  | 實用平衡方式，多用此法                |
| Adam Optimizer | 自動調整學習率，廣泛應用              |


## 三、CNN：卷積神經網路（Convolutional Neural Network）

### 架構組成：

```
Input → [Conv → ReLU → Pooling]×N → FC Layer → Output
```

- **Convolution Layer**：提取區域特徵  
- **Pooling Layer**：降維（Max / Average）  
- **Fully Connected Layer**：進行分類

### 卷積（Convolution）
- 核心概念：滑動 filter 擷取特徵圖
- 優點：權重共享、區域感知


## 四、ResNet：深度殘差神經網路

### 問題：深層網路容易退化

### 解法：殘差連接（Residual Connection）

y = F(x) + x

- 避免梯度消失
- 可建立非常深的網路結構


## 五、RNN：循環神經網路（Recurrent Neural Network）

### 結構：

```
x₁ → h₁ → y₁
      ↑
x₂ → h₂ → y₂
      ↑
x₃ → h₃ → y₃
```

- 處理「序列資料」：語音、文字等

### 缺點：
- 容易梯度消失，訓練不穩定

### 發展版本：

| 類型   | 說明                       |
|--------|----------------------------|
| LSTM   | 長期記憶，保留長期依賴     |
| GRU    | 簡化版 LSTM，效率較佳      |


## 各網路比較總表

| 網路類型 | 用途         | 優點                         | 缺點                          |
|----------|--------------|------------------------------|-------------------------------|
| ANN      | 通用模型     | 結構簡單                     | 不善處理空間/時間資料         |
| CNN      | 圖像處理     | 擅長局部特徵提取、參數少     | 不適用序列資料                |
| RNN      | 序列資料     | 記憶前一時刻資訊             | 易梯度消失，訓練不穩定        |
| ResNet   | 深層模型     | 可訓練非常深的網路           | 結構複雜，需要殘差理解         |
| Autoencoder | 資料壓縮/降維 | 無監督、自動特徵萃取     | 可能過擬合，重建品質限制       |
