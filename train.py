"""
Assignment 1 — training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file — main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py train`, this file must, start to finish, with no
manual steps in between:

- Load your committed dataset.
- Apply the feature engineering, preprocessing, and train/val/test split
  consistent with the business framing you wrote in REPORT.md.
- Train the same model — linear regression for a regression target,
  logistic regression for a binary one — three ways, on the same split:
  scikit-learn, a from-scratch PyTorch loop (Session 5's manual approach:
  raw tensors, autograd, a manual gradient-step update), and the standard
  torch.nn.Module + torch.optim workflow.
- Evaluate all three against each other and against a naive baseline.
- Save whatever app.py needs to build its dashboard without retraining
  anything.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from data_prep import clean, load_raw
from features import add_features, get_X_y
from preprocessing import build_preprocessor, time_split

MODELS_DIR = Path("models")
RANDOM_STATE = 42

# Regularization strengths to try for sklearn (C = 1 / strength):
# small C = strong regularization (simpler model), big C = weak regularization
C_VALUES = [0.01, 0.1, 1.0, 10.0]


def evaluate(name, y_true, prob, threshold=0.5):
    """Print and return the main metrics for one model."""
    pred = (prob >= threshold).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(prob)) > 1 else 0.5,
        "log_loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7)),
    }
    print(f"  {name:28s} accuracy {metrics['accuracy']:.3f} | "
          f"ROC-AUC {metrics['roc_auc']:.3f} | log loss {metrics['log_loss']:.3f}")
    return metrics


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # ---------- 1. Data: clean -> features -> time split ----------
    data = add_features(clean(load_raw(), verbose=False))
    train, val, test = time_split(data)
    X_train, y_train = get_X_y(train)
    X_val, y_val = get_X_y(val)
    X_test, y_test = get_X_y(test)

    # ---------- 2. Preprocessing: learn on train only ----------
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    print(f"Train {Xtr.shape} | Val {Xva.shape} | Test {Xte.shape}")

    # ---------- 3. Naive baseline ----------
    # "Always predict the training cancellation rate" -> predicts 0 (no cancel)
    base_rate = y_train.mean()
    print(f"\nNaive baseline: always predict P(cancel) = {base_rate:.3f}")

    # ---------- 4. sklearn: choose C on VALIDATION ----------
    print("\nChoosing regularization strength C on validation:")
    best_C, best_loss = None, np.inf
    for C in C_VALUES:
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(Xtr, y_train)
        loss = log_loss(y_val, model.predict_proba(Xva)[:, 1])
        print(f"  C = {C:<6} validation log loss = {loss:.4f}")
        if loss < best_loss:
            best_C, best_loss = C, loss
    print(f"  -> best C = {best_C}")

    sk_model = LogisticRegression(C=best_C, max_iter=2000)
    sk_model.fit(Xtr, y_train)
    joblib.dump(sk_model, MODELS_DIR / "sklearn_logreg.joblib")

    # ---------- 5. Results on VALIDATION (test is kept for the end) ----------
    print("\nValidation results (threshold 0.5):")
    evaluate("Naive baseline", y_val, np.full(len(y_val), base_rate))
    evaluate("sklearn LogisticRegression", y_val, sk_model.predict_proba(Xva)[:, 1])

    # ---------- 6. Which features matter most? ----------
    names = pre.get_feature_names_out()
    coefs = sk_model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:10]
    print("\nTop 10 features by size of coefficient (+ = more likely to cancel):")
    for i in order:
        print(f"  {names[i]:40s} {coefs[i]:+.3f}")


if __name__ == "__main__":
    main()