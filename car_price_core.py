import numpy as np
import pandas as pd

CATEGORIES = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody',
              'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber',
              'fuelsystem', 'manufacturer']
BRANDS = {'alfa-romero': 'alfa-romeo', 'maxda': 'mazda', 'porcshce': 'porsche',
          'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'}


def clean_features(data):
    features = data.drop(columns=['car_ID', 'CarName', 'price'], errors='ignore').copy()
    names = data.CarName.astype('string').str.strip().str.lower()
    features['manufacturer'] = names.str.split().str[0].replace(BRANDS)
    for column in CATEGORIES:
        features[column] = features[column].astype('string').str.strip().str.lower()
    return features


def holdout_split(groups, seed=42):
    selected, count = [], 0
    for group in np.random.default_rng(seed).permutation(np.unique(groups)):
        selected.append(group)
        count += np.sum(groups == group)
        if count >= np.ceil(0.2 * len(groups)):
            break
    is_test = np.isin(groups, selected)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def group_folds(groups, seed=42):
    unique, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind='stable')]
    sizes = np.zeros(5, dtype=int)
    folds = np.empty(len(groups), dtype=int)
    for i in order:
        fold = int(rng.choice(np.flatnonzero(sizes == sizes.min())))
        folds[groups == unique[i]] = fold
        sizes[fold] += counts[i]
    return folds


def encode(features, settings):
    table = features.copy()
    for column, median in settings['medians'].items():
        table[column] = pd.to_numeric(table[column], errors='coerce').replace(
            [np.inf, -np.inf], np.nan).fillna(median)
    for column in CATEGORIES:
        values = table[column].fillna(settings['modes'][column])
        table[column] = pd.Categorical(values, categories=settings['levels'][column])
    table = pd.get_dummies(table, columns=CATEGORIES, drop_first=True,
                           prefix_sep='=', dtype=float)
    return np.ascontiguousarray(table.to_numpy(float)), list(table.columns)


def fit_preprocessor(features):
    numeric = [column for column in features if column not in CATEGORIES]
    settings = {'medians': {}, 'modes': {}, 'levels': {}}
    for column in numeric:
        values = pd.to_numeric(features[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        settings['medians'][column] = float(values.median()) if values.notna().any() else 0.0
    for column in CATEGORIES:
        values = features[column].dropna()
        mode = str(values.mode().iloc[0]) if len(values) else '__missing__'
        settings['modes'][column] = mode
        settings['levels'][column] = sorted(features[column].fillna(mode).unique())
    matrix, settings['feature_names'] = encode(features, settings)
    settings['means'] = matrix.mean(axis=0)
    settings['scales'] = matrix.std(axis=0)
    settings['scales'][settings['scales'] < 1e-12] = 1.0
    return settings


def transform(features, settings):
    matrix, names = encode(features, settings)
    assert names == settings['feature_names']
    return (matrix - settings['means']) / settings['scales']


def gradient_descent(design, target, alpha=10, max_iter=25000, tolerance=1e-7,
                     accelerated=True):
    n = len(target)
    target_mean, target_scale = np.mean(target), np.std(target)
    target_scale = max(float(target_scale), 1e-12)
    z = (target - target_mean) / target_scale
    gram, rhs = design.T @ design / n, design.T @ z / n
    penalty = np.full(design.shape[1], alpha / n)
    penalty[0] = 0
    rate = 1 / (max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12) + alpha / n)
    weights = np.zeros(design.shape[1])
    lookahead, momentum_time = weights.copy(), 1.0

    def objective(w):
        return float(0.5*np.mean(z*z) - np.sum(rhs*w)
                     + 0.5*np.sum(w*(gram@w + penalty*w)))

    loss = objective(weights)
    history = [{'iteration': 0, 'objective': loss}]
    for iteration in range(1, max_iter + 1):
        gradient = gram @ lookahead - rhs + penalty * lookahead
        updated = lookahead - rate * gradient
        if objective(updated) > loss + 1e-12:
            updated = weights - rate * (gram @ weights - rhs + penalty * weights)
            momentum_time = 1.0
        next_time = (1 + np.sqrt(1 + 4*momentum_time**2)) / 2
        momentum = (momentum_time - 1) / next_time if accelerated else 0
        lookahead = updated + momentum * (updated - weights)
        weights, momentum_time = updated, next_time
        loss = objective(weights)
        if iteration % 25 == 0 or iteration == max_iter:
            history.append({'iteration': iteration, 'objective': loss})
            norm = float(np.max(np.abs(gram @ weights - rhs + penalty * weights)))
            if norm <= tolerance:
                break
    if norm > tolerance:
        raise RuntimeError('Gradient descent did not converge')
    weights *= target_scale
    weights[0] += target_mean
    return {'weights': weights, 'iterations': iteration, 'gradient_norm': norm,
            'learning_rate': rate, 'history': history}


def fit_model(matrix, prices):
    design = np.column_stack([np.ones(len(matrix)), matrix])
    log_prices = np.log(prices)
    model = gradient_descent(design, log_prices)
    model['smearing'] = float(np.exp(log_prices - design @ model['weights']).mean())
    return model


def predict_model(model, matrix):
    design = np.column_stack([np.ones(len(matrix)), matrix])
    return np.exp(design @ model['weights']) * model['smearing']


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = actual - predicted
    absolute = np.abs(error)
    variance = np.sum((actual - actual.mean())**2)
    nonzero = np.abs(actual) > 1e-12
    total = np.abs(actual) + np.abs(predicted)
    return {'MAE': float(absolute.mean()), 'MSE': float(np.mean(error**2)),
            'RMSE': float(np.sqrt(np.mean(error**2))),
            'R2': float(1 - np.sum(error**2)/variance) if variance > 0 else None,
            'MAPE_pct': float(100*np.mean(absolute[nonzero]/np.abs(actual[nonzero]))) if nonzero.any() else None,
            'sMAPE_pct': float(200*np.mean(np.divide(absolute, total, out=np.zeros_like(absolute), where=total > 1e-12))),
            'MedianAE': float(np.median(absolute))}
