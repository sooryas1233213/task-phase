import unittest
import numpy as np
import pandas as pd
from car_price_core import (gradient_descent, metrics, clean_features, holdout_split,
                            group_folds, fit_preprocessor, transform, fit_model, predict_model)


class ModelTests(unittest.TestCase):
    def test_gradient_descent_and_ridge(self):
        x = np.linspace(-2, 2, 21)
        design = np.column_stack([np.ones(len(x)), x])
        for accelerated in [False, True]:
            model = gradient_descent(design, 7+3*x, alpha=0, accelerated=accelerated)
            np.testing.assert_allclose(model['weights'], [7, 3], atol=1e-6)
        model = gradient_descent(design, 7+3*x, alpha=5)
        expected = [7, 3*np.sum(x*x)/(np.sum(x*x)+5)]
        np.testing.assert_allclose(model['weights'], expected, atol=1e-6)

    def test_metrics(self):
        result = metrics(np.array([1., 2., 3.]), np.array([1., 3., 2.]))
        self.assertAlmostEqual(result['MAE'], 2/3)
        self.assertAlmostEqual(result['MSE'], 2/3)
        self.assertAlmostEqual(result['RMSE'], np.sqrt(2/3))
        self.assertAlmostEqual(result['R2'], 0.)
        self.assertAlmostEqual(result['MAPE_pct'], 100*(.5+1/3)/3)
        self.assertAlmostEqual(result['sMAPE_pct'], 200*(.2+.2)/3)
        self.assertAlmostEqual(result['MedianAE'], 1.)

    def test_training_statistics_and_unknown_categories(self):
        features = clean_features(pd.read_csv('CarPrice_Assignment.csv'))
        training = features.iloc[:12].copy()
        training.loc[0, 'horsepower'] = np.nan
        settings = fit_preprocessor(training)
        self.assertEqual(settings['medians']['horsepower'], training.horsepower.median())
        means = settings['means'].copy()
        new = features.iloc[[20]].copy()
        new['manufacturer'], new['horsepower'] = 'new-brand', np.nan
        self.assertTrue(np.isfinite(transform(new, settings)).all())
        np.testing.assert_array_equal(settings['means'], means)

    def test_disjoint_and_repeatable_splits(self):
        features = clean_features(pd.read_csv('CarPrice_Assignment.csv'))
        groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
        train, test = holdout_split(groups)
        self.assertFalse(set(groups[train]) & set(groups[test]))
        np.testing.assert_array_equal(holdout_split(groups)[0], train)
        for repeat in range(3):
            folds = group_folds(groups[train], seed=42+repeat)
            for fold in range(5):
                self.assertFalse(set(groups[train][folds == fold]) & set(groups[train][folds != fold]))

    def test_original_model_performance_is_preserved(self):
        data = pd.read_csv('CarPrice_Assignment.csv')
        features = clean_features(data)
        groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
        train, test = holdout_split(groups)
        settings = fit_preprocessor(features.iloc[train])
        model = fit_model(transform(features.iloc[train], settings), data.price.to_numpy()[train])
        predicted = predict_model(model, transform(features.iloc[test], settings))
        reference = pd.read_csv('results/previous_search/test_predictions.csv')
        np.testing.assert_array_equal(data.iloc[test].car_ID, reference.car_ID)
        np.testing.assert_allclose(predicted, reference.predicted, rtol=0, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
