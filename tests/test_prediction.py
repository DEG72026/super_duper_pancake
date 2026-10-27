"""Tests for the training pipeline and the model it produces.

They run the same step functions as the Vertex AI pipeline, but directly in
Python on data/iris.csv, so no Google Cloud is needed.

    pip install -r tests/requirements.txt
    pytest tests/

Optional: test a real model file, e.g. one downloaded from the model bucket:

    MODEL_PATH=model.joblib pytest tests/
"""

import json
import os
import sys
from pathlib import Path

import joblib
import pytest
from kfp import compiler
from kfp.dsl import Dataset, Metrics, Model

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline"))

import iris_pipeline as p  # noqa: E402

DATA = REPO / "data" / "iris.csv"

# same feature order as the API, one typical flower per species
EXAMPLES = [
    ([5.1, 3.5, 1.4, 0.2], 0),  # setosa
    ([6.0, 2.9, 4.5, 1.5], 1),  # versicolor
    ([6.9, 3.1, 5.4, 2.1], 2),  # virginica
]


def run_training(tmp_path):
    """Run load -> split -> train/evaluate (all depths) -> select, like the pipeline."""
    dataset = Dataset(uri=str(tmp_path / "dataset.csv"))
    train_set = Dataset(uri=str(tmp_path / "train.csv"))
    test_set = Dataset(uri=str(tmp_path / "test.csv"))

    p.load_data.python_func(data_path=str(DATA), dataset=dataset)
    p.split_data.python_func(dataset=dataset, test_size=0.2, random_state=42,
                             train_set=train_set, test_set=test_set)

    models, accs = {}, {}
    for depth in p.MAX_DEPTHS:
        model = Model(uri=str(tmp_path / f"model_{depth}.joblib"))
        p.train_model.python_func(train_set=train_set, max_depth=depth, model=model)
        accs[depth] = p.evaluate_model.python_func(
            model=model, test_set=test_set, max_depth=depth,
            metrics=Metrics(uri=str(tmp_path / f"metrics_{depth}")))
        models[depth] = model

    best = Model(uri=str(tmp_path / "best.joblib"))
    result = p.select_best_model.python_func(
        model_d2=models[2], model_d3=models[3], model_d4=models[4], model_dnone=models[0],
        acc_d2=accs[2], acc_d3=accs[3], acc_d4=accs[4], acc_dnone=accs[0],
        best_model=best)
    return best, result


def gate(best, accuracy, model_dir, tmp_path, min_accuracy=0.9):
    return p.check_model.python_func(
        best_model=best, accuracy=accuracy, model_dir=str(model_dir),
        min_accuracy=min_accuracy, gate_metrics=Metrics(uri=str(tmp_path / "gate")))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("run")
    best, result = run_training(tmp_path)
    return tmp_path, best, result


def test_pipeline_compiles(tmp_path):
    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(pipeline_func=p.iris_pipeline, package_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_best_model_is_accurate(trained):
    _, _, result = trained
    assert result.accuracy >= 0.9
    # depth 3 and no limit both reach 96.7%; on a tie the simpler tree wins
    assert result.max_depth == 3


def test_model_predicts_known_flowers(trained):
    _, best, _ = trained
    tree = joblib.load(best.path)
    for features, label in EXAMPLES:
        # the API sends a list in this order and does int(prediction)
        assert int(tree.predict([features])[0]) == label


def test_first_model_gets_deployed(trained, tmp_path):
    _, best, result = trained
    model_dir = tmp_path / "models"  # empty: no champion yet
    assert gate(best, result.accuracy, model_dir, tmp_path) == "deploy"

    p.save_model.python_func(best_model=best, model_dir=str(model_dir),
                             accuracy=result.accuracy, max_depth=result.max_depth)
    assert (model_dir / "model.joblib").exists()
    assert json.loads((model_dir / "metrics.json").read_text())["accuracy"] == result.accuracy
    assert len(list((model_dir / "versions").glob("model-*.joblib"))) == 1


def test_worse_model_does_not_replace_champion(trained, tmp_path):
    _, best, _ = trained
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.95}))
    assert gate(best, 0.93, model_dir, tmp_path) == "keep"


def test_model_below_minimum_is_rejected(trained, tmp_path):
    _, best, _ = trained
    assert gate(best, 0.8, tmp_path / "empty", tmp_path) == "keep"


def test_equally_good_model_is_deployed(trained, tmp_path):
    _, best, result = trained
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text(json.dumps({"accuracy": result.accuracy}))
    assert gate(best, result.accuracy, model_dir, tmp_path) == "deploy"


@pytest.mark.skipif(not os.getenv("MODEL_PATH"), reason="set MODEL_PATH to test a real model file")
def test_real_model_file():
    tree = joblib.load(os.environ["MODEL_PATH"])
    for features, label in EXAMPLES:
        assert int(tree.predict([features])[0]) == label
