"""Vertex AI training pipeline for the Iris Decision Tree (version 3).

Based on Renske's local script (DecisionTreeModel.py), split into steps:

    load_data -> split_data -> train_model x4 -> evaluate_model x4
              -> select_best_model -> check_model -> save_model (only if it passes)

Every step runs in its own container, so steps can't share variables.
Data goes from one step to the next as files (artifacts).

On Vertex AI our buckets are mounted under /gcs/, so
gs://group7-iris-models/model can be used as the folder /gcs/group7-iris-models/model.
"""
#DEMO-1
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

# Same versions as the API (api/requirements.txt), otherwise the API
# might not be able to load the model we save.
PACKAGES = [
    "pandas==2.2.2",
    "numpy==1.26.4",
    "scikit-learn==1.4.1.post1",
    "joblib==1.3.2",
]

# Which container image the steps run in:
# - PIPELINE_IMAGE set (Cloud Build does this): our own image from
#   pipeline/Dockerfile with everything pre-installed, so steps start
#   right away without installing anything.
# - not set (laptop, or before the image exists): a plain Python image,
#   and each step installs PACKAGES when it starts.
PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE")
if PIPELINE_IMAGE:
    STEP_SETTINGS = {"base_image": PIPELINE_IMAGE, "install_kfp_package": False}
else:
    STEP_SETTINGS = {"base_image": "python:3.10-slim", "packages_to_install": PACKAGES}

# Tree depths we compare. 0 means no depth limit (max_depth=None),
# because a pipeline parameter can't be None.
MAX_DEPTHS = [2, 3, 4, 0]


# Step 1: read the csv from the data bucket and check it looks right
@dsl.component(**STEP_SETTINGS)
def load_data(data_path: str, dataset: Output[Dataset]):
    import pandas as pd

    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    df = pd.read_csv(data_path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    # the API expects 0 = setosa, 1 = versicolor, 2 = virginica
    if not set(df["species"].unique()) <= {0, 1, 2}:
        raise ValueError("species must be 0, 1 or 2")

    df = df[columns].dropna()
    df.to_csv(dataset.path, index=False)
    print(f"Loaded {len(df)} rows from {data_path}")


# Step 2: 80/20 split, stratified, same settings as Renske's script
@dsl.component(**STEP_SETTINGS)
def split_data(
    dataset: Input[Dataset],
    test_size: float,
    random_state: int,
    train_set: Output[Dataset],
    test_set: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset.path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["species"]
    )
    train_df.to_csv(train_set.path, index=False)
    test_df.to_csv(test_set.path, index=False)
    print(f"Train: {len(train_df)} rows, test: {len(test_df)} rows")


# Step 3: train one tree (this step runs once for every depth)
@dsl.component(**STEP_SETTINGS)
def train_model(train_set: Input[Dataset], max_depth: int, model: Output[Model]):
    import joblib
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # same order the API uses when it builds the input for predict()
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.read_csv(train_set.path)

    depth = None if max_depth == 0 else max_depth
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    # .values = plain numbers without column names, like the API sends them
    tree.fit(df[features].values, df["species"].values)

    joblib.dump(tree, model.path)
    model.metadata["max_depth"] = max_depth
    print(f"Trained tree with max_depth={depth}")


# Step 4: accuracy on the test set (also runs once per depth)
@dsl.component(**STEP_SETTINGS)
def evaluate_model(
    model: Input[Model],
    test_set: Input[Dataset],
    max_depth: int,
    metrics: Output[Metrics],
) -> float:
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.read_csv(test_set.path)

    tree = joblib.load(model.path)
    accuracy = float(accuracy_score(df["species"].values, tree.predict(df[features].values)))

    # logged metrics show up in the Vertex AI pipeline UI
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("max_depth", max_depth)
    print(f"max_depth={max_depth}: accuracy={accuracy:.3f}")
    return accuracy


# Step 5: keep the most accurate tree. On a tie we keep the simpler
# (less deep) tree because it's less likely to overfit.
@dsl.component(**STEP_SETTINGS)
def select_best_model(
    model_d2: Input[Model],
    model_d3: Input[Model],
    model_d4: Input[Model],
    model_dnone: Input[Model],
    acc_d2: float,
    acc_d3: float,
    acc_d4: float,
    acc_dnone: float,
    best_model: Output[Model],
) -> NamedTuple("BestModel", [("accuracy", float), ("max_depth", int)]):
    import shutil
    from collections import namedtuple

    # simplest first, so on a tie the first (simplest) one stays the best
    candidates = [
        (2, acc_d2, model_d2),
        (3, acc_d3, model_d3),
        (4, acc_d4, model_d4),
        (0, acc_dnone, model_dnone),
    ]
    best_depth, best_acc, best = candidates[0]
    for depth, acc, candidate in candidates[1:]:
        if acc > best_acc:
            best_depth, best_acc, best = depth, acc, candidate

    shutil.copy(best.path, best_model.path)
    best_model.metadata["max_depth"] = best_depth
    best_model.metadata["accuracy"] = best_acc
    print(f"Best: max_depth={best_depth or 'None'}, accuracy={best_acc:.3f}")

    BestModel = namedtuple("BestModel", ["accuracy", "max_depth"])
    return BestModel(best_acc, best_depth)


# Step 6: quality gate (champion vs challenger, like in Lab 4).
# The new model (challenger) only replaces the current one (champion) if:
#   - its accuracy is at least min_accuracy
#   - its accuracy is at least as good as the champion's
#   - it gets three known example flowers right
# Returns "deploy" or "keep". save_model only runs on "deploy".
@dsl.component(**STEP_SETTINGS)
def check_model(
    best_model: Input[Model],
    accuracy: float,
    model_dir: str,
    min_accuracy: float,
    gate_metrics: Output[Metrics],
) -> str:
    import json
    import os

    import joblib

    tree = joblib.load(best_model.path)

    # one typical flower per species (taken from the Iris data set)
    examples = [
        ([5.1, 3.5, 1.4, 0.2], 0),  # setosa
        ([6.0, 2.9, 4.5, 1.5], 1),  # versicolor
        ([6.9, 3.1, 5.4, 2.1], 2),  # virginica
    ]
    examples_ok = all(int(tree.predict([x])[0]) == y for x, y in examples)

    # the champion's accuracy is saved next to the model by save_model
    champion_file = os.path.join(model_dir, "metrics.json")
    champion_acc = -1.0  # no model deployed yet -> any good model wins
    if os.path.exists(champion_file):
        with open(champion_file) as f:
            champion_acc = float(json.load(f)["accuracy"])

    decision = "deploy"
    if accuracy < min_accuracy:
        decision = "keep"
        print(f"Accuracy {accuracy:.3f} is below the minimum {min_accuracy}")
    elif accuracy < champion_acc:
        decision = "keep"
        print(f"Accuracy {accuracy:.3f} is worse than the current model ({champion_acc:.3f})")
    elif not examples_ok:
        decision = "keep"
        print("Model gets one of the example flowers wrong")

    gate_metrics.log_metric("challenger_accuracy", accuracy)
    gate_metrics.log_metric("champion_accuracy", champion_acc)
    gate_metrics.log_metric("deploy", 1 if decision == "deploy" else 0)
    print(f"Decision: {decision}")
    return decision


# Step 7: save the model where the API picks it up
#   model_dir/model.joblib            -> always the current model (API reads this)
#   model_dir/metrics.json            -> its accuracy, used by the next check_model
#   model_dir/versions/model-<time>   -> a copy of every model we deployed
@dsl.component(**STEP_SETTINGS)
def save_model(best_model: Input[Model], model_dir: str, accuracy: float, max_depth: int) -> str:
    import json
    import os
    import shutil
    from datetime import datetime, timezone

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(model_dir, "versions"), exist_ok=True)

    version_path = os.path.join(model_dir, "versions", f"model-{version}.joblib")
    shutil.copy(best_model.path, version_path)

    info = {"accuracy": accuracy, "max_depth": max_depth, "version": version}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(info, f)

    # written last on purpose: this is the file that triggers the API redeploy
    shutil.copy(best_model.path, os.path.join(model_dir, "model.joblib"))

    print(f"Saved model {version} to {model_dir}")
    return version_path


# The pipeline itself: which step gets which input
@dsl.pipeline(
    name="iris-decision-tree-training",
    description="Trains Decision Trees on Iris, checks the best one and saves it for the API.",
)
def iris_pipeline(
    data_path: str = "/gcs/group7-iris-data/iris.csv",
    model_dir: str = "/gcs/group7-iris-models/model",
    test_size: float = 0.2,
    random_state: int = 42,
    min_accuracy: float = 0.9,
):
    load_task = load_data(data_path=data_path)
    split_task = split_data(
        dataset=load_task.outputs["dataset"],
        test_size=test_size,
        random_state=random_state,
    )

    # one train + evaluate per depth; they don't depend on each other,
    # so Vertex AI runs the four of them in parallel
    models, accuracies = {}, {}
    for depth in MAX_DEPTHS:
        name = "none" if depth == 0 else str(depth)
        train_task = train_model(
            train_set=split_task.outputs["train_set"], max_depth=depth
        ).set_display_name(f"train-depth-{name}")
        eval_task = evaluate_model(
            model=train_task.outputs["model"],
            test_set=split_task.outputs["test_set"],
            max_depth=depth,
        ).set_display_name(f"evaluate-depth-{name}")
        models[name] = train_task.outputs["model"]
        accuracies[name] = eval_task.outputs["Output"]

    select_task = select_best_model(
        model_d2=models["2"],
        model_d3=models["3"],
        model_d4=models["4"],
        model_dnone=models["none"],
        acc_d2=accuracies["2"],
        acc_d3=accuracies["3"],
        acc_d4=accuracies["4"],
        acc_dnone=accuracies["none"],
    )

    check_task = check_model(
        best_model=select_task.outputs["best_model"],
        accuracy=select_task.outputs["accuracy"],
        model_dir=model_dir,
        min_accuracy=min_accuracy,
    )
    # the champion's metrics.json changes on every deploy, so never reuse
    # an old (cached) result of this step
    check_task.set_caching_options(False)

    with dsl.If(check_task.outputs["Output"] == "deploy", name="model-is-good-enough"):
        save_model(
            best_model=select_task.outputs["best_model"],
            model_dir=model_dir,
            accuracy=select_task.outputs["accuracy"],
            max_depth=select_task.outputs["max_depth"],
        ).set_caching_options(False)
