"""Vertex AI training pipeline for the Iris Decision Tree classifier.

The training logic is based on Renske's local script
(pipeline/DecisionTreeModel.py), split into separate pipeline components:

    load_data -> split_data -> train_model (x4 depths, in parallel)
              -> evaluate_model (x4) -> select_best_model -> save_model

Each component runs in its own container on Vertex AI. Components cannot
share Python variables, so data moves between them as artifacts (files).

On Vertex AI, Cloud Storage buckets are mounted under /gcs/, so
gs://group7-iris-data/iris.csv can be read as /gcs/group7-iris-data/iris.csv.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

# Same versions as the prediction API (api/requirements.txt), so the saved
# model loads in the API without version problems.
BASE_IMAGE = "python:3.10-slim"
PACKAGES = [
    "pandas==2.2.2",
    "numpy==1.26.4",
    "scikit-learn==1.4.1.post1",
    "joblib==1.3.2",
]

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL = "species"  # numeric: 0 = setosa, 1 = versicolor, 2 = virginica

# Tree depths to compare. 0 means "no limit" (max_depth=None in scikit-learn),
# because pipeline parameters cannot be None.
MAX_DEPTHS = [2, 3, 4, 0]


# --- Step 1: load the dataset from the data bucket ---------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def load_data(data_path: str, dataset: Output[Dataset]):
    """Read the Iris CSV and check it has the expected columns and labels."""
    import pandas as pd

    df = pd.read_csv(data_path)

    expected = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if not set(df["species"].unique()) <= {0, 1, 2}:
        raise ValueError("Labels must be numeric: 0, 1 or 2")

    df = df[expected].dropna()
    df.to_csv(dataset.path, index=False)
    print(f"Loaded {len(df)} rows from {data_path}")


# --- Step 2: split into train and test sets ----------------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def split_data(
    dataset: Input[Dataset],
    test_size: float,
    random_state: int,
    train_set: Output[Dataset],
    test_set: Output[Dataset],
):
    """Stratified train/test split, same settings as the local script."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset.path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["species"],
    )
    train_df.to_csv(train_set.path, index=False)
    test_df.to_csv(test_set.path, index=False)
    print(f"Train rows: {len(train_df)}, test rows: {len(test_df)}")


# --- Step 3: train one Decision Tree (runs once per depth) -------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def train_model(train_set: Input[Dataset], max_depth: int, model: Output[Model]):
    """Train a Decision Tree with the given depth (0 = no limit)."""
    import joblib
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.read_csv(train_set.path)

    depth = None if max_depth == 0 else max_depth
    classifier = DecisionTreeClassifier(max_depth=depth, random_state=0)
    # .values: train on a plain array, like the API sends at prediction time
    classifier.fit(df[features].values, df["species"].values)

    joblib.dump(classifier, model.path)
    model.metadata["max_depth"] = max_depth
    print(f"Trained Decision Tree with max_depth={depth}")


# --- Step 4: evaluate one model on the test set (runs once per depth) --------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def evaluate_model(
    model: Input[Model],
    test_set: Input[Dataset],
    max_depth: int,
    metrics: Output[Metrics],
) -> float:
    """Return the model's accuracy and log it so it shows in the Vertex UI."""
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.read_csv(test_set.path)

    classifier = joblib.load(model.path)
    predictions = classifier.predict(df[features].values)
    accuracy = float(accuracy_score(df["species"].values, predictions))

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("max_depth", max_depth)
    print(f"max_depth={max_depth}: accuracy={accuracy:.3f}")
    return accuracy


# --- Step 5: pick the model with the highest accuracy -------------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
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
    best_metrics: Output[Metrics],
):
    """Copy the most accurate model to the output. Ties go to the simpler
    (shallower) tree, because it is less likely to overfit."""
    import shutil

    # Ordered from simplest to most complex, so ties pick the simpler tree.
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
    best_metrics.log_metric("best_accuracy", best_acc)
    best_metrics.log_metric("best_max_depth", best_depth)
    print(f"Best model: max_depth={best_depth or 'None'}, accuracy={best_acc:.3f}")


# --- Step 6: save the best model where the API can find it -------------------
@dsl.component(base_image=BASE_IMAGE, packages_to_install=PACKAGES)
def save_model(best_model: Input[Model], model_dir: str) -> str:
    """Write the model to the model bucket twice:
    - model_dir/model.joblib            fixed name, loaded by the API
    - model_dir/versions/<time>.joblib  a kept copy of every training run
    """
    import os
    import shutil
    from datetime import datetime, timezone

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(model_dir, "versions"), exist_ok=True)

    latest_path = os.path.join(model_dir, "model.joblib")
    version_path = os.path.join(model_dir, "versions", f"model-{version}.joblib")
    shutil.copy(best_model.path, version_path)
    shutil.copy(best_model.path, latest_path)

    print(f"Saved model to {latest_path} (version copy: {version_path})")
    return version_path


# --- The pipeline: connects the steps ----------------------------------------
@dsl.pipeline(
    name="iris-decision-tree-training",
    description="Trains Decision Trees on Iris, picks the best and saves it to Cloud Storage.",
)
def iris_pipeline(
    data_path: str = "/gcs/group7-iris-data/iris.csv",
    model_dir: str = "/gcs/group7-iris-models/model",
    test_size: float = 0.2,
    random_state: int = 42,
):
    load_task = load_data(data_path=data_path)

    split_task = split_data(
        dataset=load_task.outputs["dataset"],
        test_size=test_size,
        random_state=random_state,
    )

    # Train and evaluate one model per depth. These four branches do not
    # depend on each other, so Vertex AI runs them in parallel.
    models = {}
    accuracies = {}
    for depth in MAX_DEPTHS:
        name = "none" if depth == 0 else str(depth)

        train_task = train_model(
            train_set=split_task.outputs["train_set"],
            max_depth=depth,
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

    save_model(best_model=select_task.outputs["best_model"], model_dir=model_dir)
