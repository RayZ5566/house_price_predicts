# House_Price_Predicts_Advanced
![hp1](/readme/housesbanner.png)

### 運用及比較進階回歸模型預測房價
**STEP.1** 運用**heatmap**分析各個factor與saleprice的相關性

![hp2](/readme/heatmap.PNG)  

**STEP.2** **判斷**可能導致模型過度擬合(overfitting)的因子=>減少參數以及outliers  
(下圖僅為參考)
      
![hp3](/readme/example.PNG)  

**STEP.3** 將資料過濾整理後，以**RandomForestRegressor**分析，並比較多個n_estimators  
  
![hp4](/readme/rf.PNG)  
  
**STEP.4** 以**XGBRegressor**分析，並將Feature Importance視覺化，觀察各feature對房價預測的影響程度  
  
![hp5](/readme/feature_importance.PNG)  


## 成果  
  
於**KAGGLE** - "House Prices: Advanced Regression Techniques"競賽中取得房價預測 **RMSE = 0.14881**  
![hp6](/readme/outcome.PNG)  
