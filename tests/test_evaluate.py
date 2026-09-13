import numpy as np

from disaster_tweets.evaluate import best_f1_threshold, compute_metrics, expected_calibration_error


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert m.f1 == 1.0 and m.roc_auc == 1.0 and m.tp == 2 and m.tn == 2


def test_ece_is_zero_when_calibrated_and_large_when_not():
    y = np.array([0, 1] * 500)
    assert expected_calibration_error(y, np.full(1000, 0.5)) < 1e-9
    assert expected_calibration_error(y, np.where(y == 1, 0.05, 0.95)) > 0.8


def test_best_threshold_prefers_the_separating_point():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0, 0.3, 50), np.linspace(0.31, 1, 50)])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.31
