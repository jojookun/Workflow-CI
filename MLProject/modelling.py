"""Retraining baseline terpilih untuk MLflow Project dan image serving."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, recall_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("bank_marketing_preprocessing"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Folder output harus baru atau kosong agar artefak antar-run tidak tercampur.")
    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    y_train, y_val = train.pop("y"), validation.pop("y")
    for name, frame, target in [("train", train, y_train), ("validation", validation, y_val)]:
        if set(target.unique()) != {0, 1} or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Data {name} harus numerik, finite, dan memiliki kedua kelas 0/1.")
    if list(train.columns) != list(validation.columns):
        raise ValueError("Urutan fitur train dan validation berbeda.")
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("Bank-Marketing-CI")
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(train, y_train)
        mlflow.sklearn.log_model(
            model, "model", input_example=train.head(5),
            signature=mlflow.models.infer_signature(train, model.predict(train)),
            pip_requirements=str(Path(__file__).parent / "requirements.txt"),
        )
        probability = model.predict_proba(validation)[:, 1]
        metrics = {
            "validation_average_precision": average_precision_score(y_val, probability),
            "validation_roc_auc": roc_auc_score(y_val, probability),
            "validation_log_loss": log_loss(y_val, probability),
            "validation_positive_recall": recall_score(y_val, model.predict(validation), zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"author": "Jonathan Christian Herutomo", "selected_from": "Kriteria 2 validation comparison",
                         "git_commit": os.environ.get("GITHUB_SHA", "local"),
                         "evaluation_limitation": "full-dataset EDA preceded chronological split; test excluded from CI"})
        hashes = {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                  for name in ["train.csv", "validation.csv", "metadata.json"]}
        mlflow.log_dict(hashes, "data_hashes.json")
        for name in ["modelling.py", "requirements.txt", "conda.yaml", "MLproject"]:
            mlflow.log_artifact(str(Path(__file__).parent / name), "source")
        output.mkdir(parents=True, exist_ok=True)
        model_path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="model", dst_path=str(output))
        restored = mlflow.pyfunc.load_model(model_path)
        example = validation.head(10)
        expected = model.predict(example).tolist()
        np.testing.assert_array_equal(restored.predict(example), expected)
        request = {"dataframe_split": example.to_dict(orient="split")}
        (output / "request.json").write_text(json.dumps(request), encoding="utf-8")
        (output / "expected_predictions.json").write_text(json.dumps(expected), encoding="utf-8")
        result = {"run_id": run.info.run_id, "model_uri": f"runs:/{run.info.run_id}/model",
                  "git_commit": os.environ.get("GITHUB_SHA", "local"), "data_hashes": hashes, "metrics": metrics}
        (output / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
