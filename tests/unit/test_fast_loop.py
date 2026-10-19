from time import perf_counter

import numpy as np

from src.ml.fast_loop import FastLoopNodeEngine, OnlineDriftDetector, SplitFailureClassifier


def test_analytical_vs_numerical_gradients():
    model = SplitFailureClassifier(seed=7, l2_reg=1e-3)
    x = np.random.default_rng(8).normal(size=10)
    _, analytical = model._loss_and_gradients(x, label=1, confidence_weight=0.7)
    epsilon = 1e-6
    for name in ("W1", "b1", "W2", "b2", "Wh", "bh"):
        parameter = getattr(model, name)
        for flat_index in np.linspace(0, parameter.size - 1, 5, dtype=int):
            index = np.unravel_index(flat_index, parameter.shape)
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model._loss_and_gradients(x, 1, 0.7)[0]
            parameter[index] = original - epsilon
            minus = model._loss_and_gradients(x, 1, 0.7)[0]
            parameter[index] = original
            numerical = (plus - minus) / (2 * epsilon)
            relative = abs(numerical - analytical[name][index]) / max(
                1e-8, abs(numerical) + abs(analytical[name][index]))
            assert relative < 1e-5, (name, index, numerical, analytical[name][index])


def test_encoder_sync_preserves_personal_head():
    source = SplitFailureClassifier(seed=1)
    target = SplitFailureClassifier(seed=2)
    head_before = (target.Wh.copy(), target.bh.copy())
    old_encoder = target.get_encoder_weights()
    target.set_encoder_weights(source.get_encoder_weights())
    assert any(not np.array_equal(old_encoder[key], getattr(target, key))
               for key in old_encoder)
    assert np.array_equal(target.Wh, head_before[0])
    assert np.array_equal(target.bh, head_before[1])


def sample(rng, label, regime):
    x = rng.normal(0, 0.03, 10)
    if regime == "A":
        x[0] = 0.15 if label == 0 else 0.85
        x[9] = 0.05
    else:
        x[0] = 0.85 if label == 0 else 0.15
        x[9] = 1.0
    return x


def test_drift_detection_and_fast_recovery():
    rng = np.random.default_rng(11)
    engine = FastLoopNodeEngine(
        SplitFailureClassifier(seed=12, default_lr=0.08),
        OnlineDriftDetector(window_size=20, threshold=2.0), base_lr=0.08)
    for index in range(160):
        label = index % 2
        engine.train(sample(rng, label, "A"), label)
    triggered = False
    for index in range(30):
        label = index % 2
        engine.train(sample(rng, label, "B"), label)
        triggered |= engine.drift_detected
    predictions = []
    for index in range(100):
        label = index % 2
        prediction = int(np.argmax(engine.classifier.predict_proba(sample(rng, label, "B"))))
        predictions.append(prediction == label)
    assert triggered
    assert np.mean(predictions) > 0.85
    assert engine.is_ready


def test_micro_overhead_benchmark():
    model = SplitFailureClassifier(seed=21)
    rng = np.random.default_rng(22)
    samples = rng.random((1_000, 10))
    started = perf_counter()
    for index, x in enumerate(samples):
        model.predict_proba(x)
        model.partial_fit(x, index % 3, 1.0)
    elapsed = perf_counter() - started
    assert elapsed < 0.100, f"1,000 online steps took {elapsed:.3f}s"
