# Short Term Load Forecast (one day ahead peak load forecast)
Implemented load forecast algorithms to predict one day ahead hourly peak load.

### Architecture
- Multivariate Linear Regression - (1)
- Two layer neural nets - (2)
- Moving average mean in combination with (1) -(3)

### Error rate (mae, rmse, mape) between one day ahead prediction and the actual daily peak load
 - (1) : 2.1 mw, 3.02 mw, 2.75 %
 - (2) : 1.89 mw, 2.76 mw, 2.47 %
 - (3) : 1.60 mw, 1.77 mw, 2.03 %

### Future work 
- Using Recurrent Neural Networks, Long Short Term Memory
- Applying Reinforcement Learning to efficiently operate cooling load
