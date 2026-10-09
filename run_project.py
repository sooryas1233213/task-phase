from pathlib import Path
import json
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent


def build_notebook()'Test R² is 0.9429 and RMSE is 1,925.10 price units. These results describe the supplied historical dataset.'    source = (ROOT / 'car_price_core.py').read_text()
    clean, rest = source.split('\ndef holdout_split', 1)
    split, rest = ('def holdout_split' + rest).split('\ndef encode', 1)
    prepare, rest = ('def encode' + rest).split('\ndef gradient_descent', 1)
    descent, metric_code = ('def gradient_descent' + rest).split('\ndef metrics', 1)
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('# Car price prediction\n\nLinear regression trained with gradient descent from scratch, using log prices and ridge penalty 10.'),
    md('## 1. Read and clean the data\nCorrect manufacturer names, remove the ID and full name, and keep all valid cars.'),
    code('''from pathlib import Path
import json
import matplotlib.pyplot as plt
from IPython.display import display
''' + clean + '''
data = '## 1. Data cleaning\nManufacturer spelling is corrected. The row ID and full car name are excluded.'features = clean_features(data)
prices = data.price.to_numpy(float)
assert np.isfinite(prices).all() and (prices > 0).all()
print(f'{len(data)} cars; {data.isna().sum().sum()} missing values')
display(data.head(3))
'''),
    md('''## 2. Split and prepare the data
Reserve 20% of cars for testing. Cars with identical predictors stay together. Fit category encodings and scaling on training rows only. Numeric gaps use training medians; categorical gaps use training modes.'''),
    code(split + '''
groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
train, test = holdout_split(groups)
assert not set(groups[train]) & set(groups[test])
print(f'Training: {len(train)} cars | Testing: {len(test)} cars')
'''),
'## 2. Data preparation\nThe split contains 164 training cars and 41 test cars. Identical predictors stay together. Encoding, imputation and scaling are fitted on training rows only.'    md(r'''## 3. Gradient descent from scratch
Predict **log(price)** from standardized inputs. Ridge regularization reduces overfitting; the intercept is not penalized.

The gradient is `X.T @ (X @ weights - target) / n`, plus the ridge penalty. Update the weights by subtracting learning rate times gradient. Momentum speeds up these updates; the code restarts momentum if the loss rises. The eigenvalue calculation only sets a safe learning rate.

The training residual correction converts log predictions back to prices without using validation or test prices.'''),
    code(descent),
    md('''## 4. Check the model with cross-validation
Use fiv'## 3. Gradient descent\nThe gradient combines squared prediction error with a ridge penalty. Momentum accelerates the updates, and the intercept is unpenalized. A training-residual correction converts log predictions back to prices.'    folds = group_folds(groups[train], seed=42 + repeat)
    splits[f'cv_repeat_{repeat+1}_fold'] = -1
    splits.loc[train, f'cv_repeat_{repeat+1}_fold'] = folds
    for fold in range(5):
        fit_rows, valid_rows = train[folds != fold], train[folds == fold]
        assert not set(groups[fit_rows]) & set(groups[valid_rows])
        settings = fit_preprocessor(features.iloc[fit_rows])
       '## 4. Cross-validation\nFive grouped folds are repeated three times. Preprocessing and model fitting are repeated within each training fold.'        cv_rows.append({'repeat': repeat+1, 'fold': fold, **metrics(prices[valid_rows], prediction)})
        cv_optimization.append({'repeat': repeat+1, 'fold': fold, 'iterations': model['iterations'],
                                'gradient_norm': model['gradient_norm']})
        cv_predictions.extend({'repeat': repeat+1, 'fold': fold, 'car_ID': int(data.iloc[i].car_ID),
                               'actual': float(prices[i]), 'predicted': float(p)}
                              for i, p in zip(valid_rows, prediction))
cv_predictions = pd.DataFrame(cv_predictions)
cv_metrics = metrics(cv_predictions.actual, cv_predictions.predicted)
print(f"Cross-validation RMSE: {cv_metrics['RMSE']:,.2f}")
'''),
    md('''## 5. Train once and evaluate
Train on all training rows, then evaluate the reserved cars. RMSE, MAE and median error use price units; MSE uses squared units. MAPE and sMAPE are percentages. Lower errors and higher R² are better.'''),
    code('''settings = fit_preprocessor(features.iloc[train])
model = fit_model(transform(features.iloc[train], settings), prices[train])
train_prediction = predict_model(model, transform(features.iloc[train], settings))
test_prediction = predict_model(model, transform(features.iloc[test], settings))
evaluation = []
for split_name, rows, prediction in [('Train', train, train_prediction), ('Test', test, test_prediction)]:
    evaluation.append({'Model': 'Gradient descent', 'Split': split_name, **metrics(prices[rows], prediction)})
    for label, value in [('Mean baseline', prices[train].mean()), ('Median baseline', np.median(prices[train]))]:
        evaluation.append({'Model': label, 'Split': split_name, **metrics(prices[rows], np.full(len(rows), value))})
evaluation = pd.DataFrame(evaluation)
display(evaluation.round(4))
print(f"Converged in {model['iterations']} updates; gradient norm {model['gradient_norm']:.2e}")
'''),
    md('## 6. Look at errors and save the results\nPositive residuals mean underprediction. A small historical dataset cannot establish accuracy for current car prices.'),
    cod'## 5. Evaluation\nMAE, RMSE and median absolute error use price units. MSE uses squared price units; MAPE and sMAPE use percentages.'figures.mkdir(parents=True, exist_ok=True)
errors = data.iloc[test][['car_ID', 'CarName', 'price']].rename(columns={'price': 'actual'}).copy()
errors['predicted'] = test_prediction
errors['residual'] = errors.actual - errors.predicted
errors['absolute_error'] = errors.residual.abs()
history = pd.DataFrame(model['history'])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].scatter(errors.actual, errors.predicted)
limits = [errors.actual.min(), errors.actual.max()]
axes[0].plot(limits, limits, 'k--')
axes[0].set(xlabel='Actual price', ylabel='Predicted price', title='Reserved test cars')
axes[1].scatter(errors.predicted, errors.residual)
axes[1].axhline(0, color='black', linestyle='--')
axes[1].set(xlabel='Predicted price', ylabel='Actual - predicted', title='Residuals')
fig.tig'## 6. Prediction errors\nResiduals are actual minus predicted price. Positive residuals indicate underprediction.'fig.savefig(figures / 'test_diagnostics.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(history.iteration, history.objective)
ax.set(xlabel='Gradient updates', ylabel='Training objective', title='Gradient descent convergence')
fig.tight_layout()
fig.savefig(figures / 'training_loss.png', dpi=150)
plt.show()
display(errors.nlargest(5, 'absolute_error').round(2))
'''),
    code('''summary = {'alpha': 10, 'target': 'log', 'n_train': len(train), 'n_test': len(test),
           'iterations': model['iterations'], 'gradient_norm': model['gradient_norm'],
           'cv_metrics': cv_metrics, 'train_metrics': metrics(prices[train], train_prediction),
           'test_metrics': metrics(prices[test], test_prediction), 'smearing': model['smearing']}
for name, table in [('final_metrics', evaluation), ('test_predictions', errors),
                    ('cv_fold_metrics', pd.DataFrame(cv_rows)), ('cv_predictions', cv_predictions),
                    ('cv_convergence', pd.DataFrame(cv_optimization)), ('split_assignments', splits),
                    ('training_loss', history), ('cv_results', pd.DataFrame([cv_metrics]))]:
    table.to_csv(output / f'{name}.csv', index=False)
pd.DataFrame({'feature': ['intercept'] + settings['feature_names'],
              'coefficient': model['weights']}).to_csv(output / 'coefficients.csv', index=False)
(output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
saved_settings = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in settings.items()}
(output / 'preprocessing.json').write_text(json.dumps(saved_settings, indent=2) + '\\n')
np.savez(output / 'model_weights.npz', weights=model['weights'], smearing=model['smearing'], alpha=10, target='log')
assert np.isfinite(test_prediction).all()
print('Results saved in results/.')
'''),
    md('''The original features performed better than the extra polynomial features in the previous cross-validation search. This version keeps those winning settings and all seven metrics, with one training function and no multi-model search framework. Individual coefficients describe standardized inputs and should not be read as causal effects.''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': sys.version.split()[0]}})
    nbformat.write(notebook, ROOT / 'car_price_regression.ipynb')
    return notebook


def main():
    notebook = build_notebook()
    kernel_root = ROOT / 'tmp' / 'kernels'
    kernel_dir = kernel_root / 'car-price-gd'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Car price GD', 'language': 'python'}))
    manager = KernelManager(kernel_name='car-price-gd', transport='ipc',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False))
    print('Executing the simplified notebook...', flush=True)
    try:
        NotebookClient(notebook, km=manager, timeout=300,
                       resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook, ROOT / 'car_price_regression.ipynb')
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    from make_report import build_report
    build_report(ROOT)
    print('Simplified notebook and report ready.')


if __name__ == '__main__':
    main()
