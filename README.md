# Car price prediction

Linear regression trained with gradient descent implemented from scratch.
The model uses log prices, original features and ridge penalty 10.

Test results: **R² 0.9429**, **RMSE 1,925.10**, **MAE 1,391.31**, **MAPE 10.84%**.

The dataset contains 205 cars. Identical predictors stay together during splitting;
encoding and scaling use training rows only. Evaluation includes seven regression
metrics, repeated grouped cross-validation and mean/median baselines.

- `car_price_regression.ipynb`: executed notebook containing the complete model.
- `car_price_core.py`: preprocessing and gradient-descent functions.
- `output/pdf/car_price_report.pdf`: results report.
- `results/`: metrics, predictions, folds and model weights.
- `results/previous_search/`: original model-search results.

No scikit-learn or direct regression solver is used. The supplied datasets are
unchanged. Results describe this historical dataset.
