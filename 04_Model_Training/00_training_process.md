# 模型訓練流程

## 流程步驟

1. 資料分割（訓練集、驗證集、測試集）
2. 建立模型
3. 損失函數與優化器設定
4. 訓練迴圈：forward → loss → backward → update
5. 驗證與調參
6. 最終測試與部署

## PyTorch 典型訓練迴圈

```python
for epoch in range(epochs):
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
