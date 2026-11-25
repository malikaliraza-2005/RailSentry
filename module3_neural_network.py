# Student name: [Your Name]
# Roll number: [Your Roll Number]
# Course code: [Your Course Code]
# Module description: Railway station neural networks and model evaluation

"""
Module 3:
1. Train two simple Keras neural networks.
2. Compare and evaluate the models.
3. Compare KNN and ANN.
4. Save the best model.
5. Predict a new railway incident entered by the user.

Data split (same cleaned data as Module 2):
    - 20% test set: used ONLY for the final comparison table
    - the remaining 80% is split again into training (80%) and
      validation (20%). The validation set is used for early stopping,
      for tuning KNN, and for choosing the best ANN.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import module2_data_ml


# All paths are relative to this file, so the program works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

RANDOM_SEED = 42
RISK_CLASSES = ["Low", "Medium", "High"]
RISK_TO_NUMBER = {name: number for number, name in enumerate(RISK_CLASSES)}

NUMERIC_COLUMNS = [
    "grid_x",
    "grid_y",
    "crowd_density_pct",
    "hazard_severity_1_5",
    "equipment_health_pct",
    "obstruction_level_0_3",
    "hour_of_day",
]

CATEGORICAL_COLUMNS = ["incident_type"]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

VALID_INCIDENT_TYPES = [
    "Structural",
    "Medical",
    "Crowd_Control",
    "Electrical",
    "Equipment_Failure",
    "Fire_Smoke",
]

K_VALUES = [3, 5, 7, 9, 11]
DISTANCE_METRICS = ["euclidean", "manhattan"]


def load_data() -> pd.DataFrame:
    """Load the railway dataset using exactly the same cleaning as Module 2."""
    data = module2_data_ml.load_prepared_data(verbose=True)

    print("Dataset shape:", data.shape)
    print("Risk counts:")
    print(data["risk_level"].value_counts().reindex(RISK_CLASSES).to_string())

    return data


def make_preprocessor():
    """
    Impute, scale numbers and one-hot encode incident type.

    The imputer learns medians only from the data it is fitted on
    (the training split), so no test information leaks into training.
    """
    return module2_data_ml.make_preprocessing_pipeline(
        NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
    )


def to_dense_float(array) -> np.ndarray:
    """Convert a (possibly sparse) matrix to a float32 NumPy array for Keras."""
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype="float32")


def prepare_training_data(data: pd.DataFrame) -> dict:
    """Split data into train / validation / test and create Keras arrays."""
    X = data[FEATURE_COLUMNS]
    y = data["risk_level"].map(RISK_TO_NUMBER).to_numpy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    preprocessor = make_preprocessor()
    X_train_array = to_dense_float(preprocessor.fit_transform(X_train))
    X_validation_array = to_dense_float(preprocessor.transform(X_validation))
    X_test_array = to_dense_float(preprocessor.transform(X_test))

    joblib.dump(preprocessor, MODEL_DIR / "module3_preprocessor.joblib")
    joblib.dump(RISK_TO_NUMBER, MODEL_DIR / "module3_risk_classes.joblib")

    print(
        f"Split sizes -> training: {len(X_train)}, "
        f"validation: {len(X_validation)}, test: {len(X_test)}"
    )

    return {
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "X_train_array": X_train_array,
        "X_validation_array": X_validation_array,
        "X_test_array": X_test_array,
        "preprocessor": preprocessor,
    }


def build_model_architecture_1(input_size: int):
    """Build ANN Architecture 1: 64 neurons, then 32 neurons."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,), name="incident_features"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_architecture_2(input_size: int):
    """Build ANN Architecture 2: 128 neurons, dropout, 64, then 32."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,), name="incident_features"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class EpochReporter(tf.keras.callbacks.Callback):
    """Send each finished epoch's metrics to a function (used by the web UI)."""

    def __init__(self, model_name: str, report_function) -> None:
        super().__init__()
        self.model_name = model_name
        self.report_function = report_function

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.report_function({
            "model": self.model_name,
            "epoch": epoch + 1,
            **{key: float(value) for key, value in logs.items()},
        })


def train_ann_models(split: dict, epoch_callback=None) -> tuple[dict, dict]:
    """
    Train both neural network architectures (up to 50 epochs each).

    - Early stopping watches the validation set (not the test set).
    - Class weights make mistakes on the rare High / Medium classes cost
      more, because only 10% of incidents are High risk.
    """
    X_train_array = split["X_train_array"]
    y_train = split["y_train"]
    validation_data = (split["X_validation_array"], split["y_validation"])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(RISK_CLASSES)),
        y=y_train,
    )
    class_weight = {number: float(weight) for number, weight in enumerate(weights)}
    print("Class weights:", {RISK_CLASSES[k]: round(v, 3) for k, v in class_weight.items()})

    trained = []
    builders = [
        ("ANN Architecture 1", build_model_architecture_1, "ann_architecture_1.keras"),
        ("ANN Architecture 2", build_model_architecture_2, "ann_architecture_2.keras"),
    ]
    for name, builder, filename in builders:
        # Same starting seed for both models, so the comparison is fair.
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        model = builder(X_train_array.shape[1])

        # A new EarlyStopping object for each model.
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        )

        callbacks = [early_stop]
        if epoch_callback is not None:
            callbacks.append(EpochReporter(name, epoch_callback))

        print(f"\nTraining {name}...")
        history = model.fit(
            X_train_array,
            y_train,
            epochs=50,
            batch_size=16,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )
        model.save(MODEL_DIR / filename)
        trained.append({"name": name, "model": model, "history": history})

    return trained[0], trained[1]


def plot_training_history(history_1: dict, history_2: dict) -> None:
    """Plot accuracy and loss for both neural-network architectures."""
    histories = [history_1, history_2]

    plt.figure(figsize=(10, 5))
    for history in histories:
        history_object = history["history"].history
        plt.plot(history_object["accuracy"], label=history["name"] + " training")
        plt.plot(history_object["val_accuracy"], linestyle="--",
                 label=history["name"] + " validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ANN Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ann_accuracy_curves.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    for history in histories:
        history_object = history["history"].history
        plt.plot(history_object["loss"], label=history["name"] + " training")
        plt.plot(history_object["val_loss"], linestyle="--",
                 label=history["name"] + " validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ann_loss_curves.png")
    plt.show()
    plt.close()


def calculate_classifier_metrics(y_true, y_predicted) -> dict:
    """Calculate required classification metrics (macro averages)."""
    return {
        "accuracy": accuracy_score(y_true, y_predicted),
        "precision": precision_score(
            y_true, y_predicted, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_predicted, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_predicted, average="macro", zero_division=0
        ),
    }


def plot_confusion_matrix(y_true, y_predicted, title: str) -> None:
    """Draw a confusion matrix with all three risk classes."""
    matrix = confusion_matrix(y_true, y_predicted, labels=[0, 1, 2])

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted risk")
    plt.ylabel("Actual risk")
    plt.xticks([0, 1, 2], RISK_CLASSES)
    plt.yticks([0, 1, 2], RISK_CLASSES)

    for row in range(3):
        for column in range(3):
            plt.text(column, row, matrix[row, column],
                     ha="center", va="center", color="black")

    plt.tight_layout()
    safe_title = title.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{safe_title}_confusion_matrix.png")
    plt.show()
    plt.close()


def train_and_evaluate_knn(split: dict, data: pd.DataFrame) -> dict:
    """
    Tune and train KNN on the same features and split as the ANNs.

    k and the distance metric are chosen on the validation set, then the
    chosen model is refitted on the full training data and tested once.
    """
    best = None
    for k_value in K_VALUES:
        for metric in DISTANCE_METRICS:
            model = Pipeline([
                ("preprocessor", make_preprocessor()),
                ("classifier", KNeighborsClassifier(n_neighbors=k_value, metric=metric)),
            ])
            model.fit(split["X_train"], split["y_train"])
            predictions = model.predict(split["X_validation"])
            score = (
                f1_score(split["y_validation"], predictions, average="macro"),
                accuracy_score(split["y_validation"], predictions),
            )
            if best is None or score > best[0]:
                best = (score, k_value, metric)

    _, best_k, best_metric = best
    print(f"\nBest KNN on validation set: k={best_k}, metric={best_metric}")

    knn_classifier = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)),
    ])
    knn_classifier.fit(split["X_train_full"], split["y_train_full"])
    knn_predictions = knn_classifier.predict(split["X_test"])
    knn_probabilities = knn_classifier.predict_proba(split["X_test"])
    knn_metrics = calculate_classifier_metrics(split["y_test"], knn_predictions)

    # Response-time regressor (used by Module 4 to estimate response time).
    data_with_target = data.dropna(subset=["response_time_min"])
    X = data_with_target[FEATURE_COLUMNS]
    target = data_with_target["response_time_min"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X,
        target,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )
    X_fit_reg, X_val_reg, y_fit_reg, y_val_reg = train_test_split(
        X_train_reg,
        y_train_reg,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    best_reg = None
    for k_value in K_VALUES:
        for metric in DISTANCE_METRICS:
            model = Pipeline([
                ("preprocessor", make_preprocessor()),
                ("regressor", KNeighborsRegressor(n_neighbors=k_value, metric=metric)),
            ])
            model.fit(X_fit_reg, y_fit_reg)
            validation_mae = mean_absolute_error(y_val_reg, model.predict(X_val_reg))
            if best_reg is None or validation_mae < best_reg[0]:
                best_reg = (validation_mae, k_value, metric)

    _, reg_k, reg_metric = best_reg
    knn_regressor = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("regressor", KNeighborsRegressor(n_neighbors=reg_k, metric=reg_metric)),
    ])
    knn_regressor.fit(X_train_reg, y_train_reg)
    reg_predictions = knn_regressor.predict(X_test_reg)

    mae = mean_absolute_error(y_test_reg, reg_predictions)
    mse = mean_squared_error(y_test_reg, reg_predictions)

    joblib.dump(knn_classifier, MODEL_DIR / "module3_knn_classifier.joblib")
    joblib.dump(knn_regressor, MODEL_DIR / "module3_knn_regressor.joblib")

    return {
        "classifier": knn_classifier,
        "predictions": knn_predictions,
        "probabilities": knn_probabilities,
        "metrics": knn_metrics,
        "k": best_k,
        "metric": best_metric,
        "regressor_k": reg_k,
        "regressor_metric": reg_metric,
        "mae": mae,
        "mse": mse,
    }


def macro_auc(y_true, probabilities) -> float:
    """One-vs-rest macro ROC-AUC for the three risk classes."""
    return float(roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="macro",
        labels=[0, 1, 2],
    ))


def plot_roc_curves(
    y_true,
    knn_probabilities,
    ann_probabilities,
    ann_name: str,
) -> None:
    """Plot KNN and best-ANN ROC curves for each risk class."""
    y_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=3)

    plt.figure(figsize=(8, 6))
    for class_number, class_name in enumerate(RISK_CLASSES):
        knn_false, knn_true, _ = roc_curve(
            y_one_hot[:, class_number],
            knn_probabilities[:, class_number],
        )
        ann_false, ann_true, _ = roc_curve(
            y_one_hot[:, class_number],
            ann_probabilities[:, class_number],
        )

        plt.plot(knn_false, knn_true, "--",
                 label=f"KNN {class_name}")
        plt.plot(ann_false, ann_true,
                 label=f"{ann_name} {class_name}")

    plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - KNN and Best ANN (test set)")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_ann_roc_curves.png")
    plt.show()
    plt.close()


def analyse_wrong_predictions(
    y_test,
    predictions,
    X_test_original,
) -> None:
    """Print at least three incorrect prediction examples and return all of them."""
    wrong_positions = np.where(np.asarray(y_test) != np.asarray(predictions))[0]

    print("\nWRONG PREDICTION ANALYSIS")
    print(f"Wrong predictions on the test set: {len(wrong_positions)} of {len(y_test)}")
    if len(wrong_positions) == 0:
        print("No incorrect predictions were found.")
        return []

    wrong_examples = []
    for index in wrong_positions:
        actual = RISK_CLASSES[int(y_test[index])]
        predicted = RISK_CLASSES[int(predictions[index])]
        record = X_test_original.iloc[index]
        wrong_examples.append({
            "actual": actual,
            "predicted": predicted,
            "under_estimated": RISK_TO_NUMBER[predicted] < RISK_TO_NUMBER[actual],
            **{column: (None if pd.isna(record[column]) else record[column])
               for column in FEATURE_COLUMNS},
        })

    number_to_show = min(3, len(wrong_positions))

    for index in wrong_positions[:number_to_show]:
        actual = RISK_CLASSES[int(y_test[index])]
        predicted = RISK_CLASSES[int(predictions[index])]
        record = X_test_original.iloc[index]

        print(
            f"- Actual: {actual}, Predicted: {predicted}, "
            f"Incident: {record['incident_type']}, "
            f"Severity: {record['hazard_severity_1_5']}, "
            f"Crowd: {record['crowd_density_pct']}%"
        )
        if RISK_TO_NUMBER[predicted] < RISK_TO_NUMBER[actual]:
            print(
                "  Routing effect: risk was under-estimated, so the vehicle may "
                "be routed through a dangerous area."
            )
        else:
            print(
                "  Routing effect: risk was over-estimated, so the vehicle may "
                "take an unnecessary detour and arrive later."
            )

    return wrong_examples


def validate_new_incident(values: dict) -> pd.DataFrame:
    """Validate values entered for a new railway incident."""
    try:
        grid_x = float(values["grid_x"])
        grid_y = float(values["grid_y"])
        crowd = float(values["crowd_density_pct"])
        severity = float(values["hazard_severity_1_5"])
        equipment = float(values["equipment_health_pct"])
        obstruction = float(values["obstruction_level_0_3"])
        hour = int(values["hour_of_day"])
    except (TypeError, ValueError) as error:
        raise ValueError("All numeric fields must contain valid numbers.") from error

    if not 0 <= grid_x <= 14:
        raise ValueError("grid_x must be between 0 and 14.")
    if not 0 <= grid_y <= 14:
        raise ValueError("grid_y must be between 0 and 14.")
    if not 0 <= crowd <= 100:
        raise ValueError("crowd_density_pct must be between 0 and 100.")
    if not 1 <= severity <= 5:
        raise ValueError("hazard_severity_1_5 must be between 1 and 5.")
    if not 0 <= equipment <= 100:
        raise ValueError("equipment_health_pct must be between 0 and 100.")
    if not 0 <= obstruction <= 3:
        raise ValueError("obstruction_level_0_3 must be between 0 and 3.")
    if not 0 <= hour <= 23:
        raise ValueError("hour_of_day must be between 0 and 23.")

    incident_type = str(values["incident_type"]).strip()
    if incident_type not in VALID_INCIDENT_TYPES:
        raise ValueError(
            "incident_type must be one of: " + ", ".join(VALID_INCIDENT_TYPES)
        )

    new_record = pd.DataFrame([{
        "grid_x": grid_x,
        "grid_y": grid_y,
        "incident_type": incident_type,
        "crowd_density_pct": crowd,
        "hazard_severity_1_5": severity,
        "equipment_health_pct": equipment,
        "obstruction_level_0_3": obstruction,
        "hour_of_day": hour,
    }])
    return new_record[FEATURE_COLUMNS]


def predict_risk_probabilities(records: pd.DataFrame) -> np.ndarray:
    """Load the saved best ANN and return risk probabilities for each record."""
    preprocessor = joblib.load(MODEL_DIR / "module3_preprocessor.joblib")
    model = tf.keras.models.load_model(MODEL_DIR / "best_risk_model.keras")
    array = to_dense_float(preprocessor.transform(records[FEATURE_COLUMNS]))
    return model.predict(array, verbose=0)


def predict_incident(values: dict) -> dict:
    """
    Validate one incident and return the ANN risk prediction and the KNN
    response-time prediction (used by the terminal menu and the web UI).
    Raises ValueError for invalid input and FileNotFoundError if models are missing.
    """
    new_record = validate_new_incident(values)
    if not (MODEL_DIR / "best_risk_model.keras").exists():
        raise FileNotFoundError("The saved ANN model is missing. Train the models first.")

    probabilities = predict_risk_probabilities(new_record)[0]
    predicted_risk = RISK_CLASSES[int(np.argmax(probabilities))]

    response_time = None
    regressor_path = MODEL_DIR / "module3_knn_regressor.joblib"
    if regressor_path.exists():
        regressor = joblib.load(regressor_path)
        response_time = float(regressor.predict(new_record)[0])

    return {
        "predicted_risk": predicted_risk,
        "probabilities": {name: float(p) for name, p in zip(RISK_CLASSES, probabilities)},
        "response_time_min": response_time,
    }


def predict_new_incident() -> None:
    """Reload the best saved ANN and predict a new user-entered incident."""
    print("\nEnter details for a new incident")

    values = {}
    try:
        values["grid_x"] = input("grid_x (0-14): ")
        values["grid_y"] = input("grid_y (0-14): ")
        values["incident_type"] = input(
            "incident_type (" + ", ".join(VALID_INCIDENT_TYPES) + "): "
        )
        values["crowd_density_pct"] = input("crowd_density_pct (0-100): ")
        values["hazard_severity_1_5"] = input(
            "hazard_severity_1_5 (1-5): "
        )
        values["equipment_health_pct"] = input(
            "equipment_health_pct (0-100): "
        )
        values["obstruction_level_0_3"] = input(
            "obstruction_level_0_3 (0-3): "
        )
        values["hour_of_day"] = input("hour_of_day (0-23): ")

        result = predict_incident(values)

        print("\nPredicted risk:", result["predicted_risk"])
        print("Class probabilities:")
        for class_name, probability in result["probabilities"].items():
            print(f"  {class_name}: {probability:.4f}")

        if result["response_time_min"] is not None:
            print(f"Predicted response time: {result['response_time_min']:.2f} minutes")

    except (ValueError, FileNotFoundError, OSError) as error:
        print("Prediction error:", error)
        print("Run option 1 (train models) first if the saved model is missing.")
    except (EOFError, KeyboardInterrupt):
        print("Prediction cancelled.")


def roc_points(y_true, probabilities) -> dict:
    """ROC curve points for each class (for the web UI charts)."""
    y_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=3)
    curves = {}
    for number, name in enumerate(RISK_CLASSES):
        false_rate, true_rate, _ = roc_curve(y_one_hot[:, number], probabilities[:, number])
        curves[name] = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(false_rate, true_rate)]
    return curves


def run_module3(ask_to_predict: bool = True, epoch_callback=None) -> dict:
    """
    Run all Module 3 requirements.

    ask_to_predict: ask in the terminal whether to predict a new incident.
    epoch_callback: optional function called after every training epoch.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    data = load_data()
    split = prepare_training_data(data)
    y_test = split["y_test"]

    ann_1, ann_2 = train_ann_models(split, epoch_callback)
    plot_training_history(ann_1, ann_2)

    # Choose the best ANN on the VALIDATION set (the test set stays unseen).
    for ann in (ann_1, ann_2):
        validation_probabilities = ann["model"].predict(split["X_validation_array"], verbose=0)
        ann["validation_metrics"] = calculate_classifier_metrics(
            split["y_validation"],
            np.argmax(validation_probabilities, axis=1),
        )
        ann["test_probabilities"] = ann["model"].predict(split["X_test_array"], verbose=0)
        ann["test_predictions"] = np.argmax(ann["test_probabilities"], axis=1)
        ann["test_metrics"] = calculate_classifier_metrics(y_test, ann["test_predictions"])

    print("\nValidation results used to choose the best ANN:")
    for ann in (ann_1, ann_2):
        metrics = ann["validation_metrics"]
        print(f"  {ann['name']}: macro F1 = {metrics['macro_f1']:.4f}, "
              f"accuracy = {metrics['accuracy']:.4f}")

    best_ann = max(
        (ann_1, ann_2),
        key=lambda ann: (ann["validation_metrics"]["macro_f1"],
                         ann["validation_metrics"]["accuracy"]),
    )

    knn = train_and_evaluate_knn(split, data)

    plot_roc_curves(
        y_test,
        knn["probabilities"],
        best_ann["test_probabilities"],
        best_ann["name"],
    )
    plot_confusion_matrix(y_test, knn["predictions"], "KNN Confusion Matrix")
    plot_confusion_matrix(y_test, best_ann["test_predictions"], "ANN Confusion Matrix")

    comparison = pd.DataFrame([
        {"Model": f"KNN (k={knn['k']}, {knn['metric']})", **knn["metrics"],
         "ROC_AUC": macro_auc(y_test, knn["probabilities"])},
        {"Model": ann_1["name"], **ann_1["test_metrics"],
         "ROC_AUC": macro_auc(y_test, ann_1["test_probabilities"])},
        {"Model": ann_2["name"], **ann_2["test_metrics"],
         "ROC_AUC": macro_auc(y_test, ann_2["test_probabilities"])},
    ])
    comparison = comparison.rename(columns={
        "macro_f1": "macro_F1",
        "ROC_AUC": "ROC-AUC",
    })

    print("\nMODEL COMPARISON TABLE (test set)")
    print(comparison.to_string(index=False))
    print(f"\nKNN Regressor (k={knn['regressor_k']}, {knn['regressor_metric']})")
    print("MAE:", round(knn["mae"], 4), "minutes")
    print("MSE:", round(knn["mse"], 4))

    wrong_examples = analyse_wrong_predictions(
        y_test, best_ann["test_predictions"], split["X_test"]
    )

    best_ann["model"].save(MODEL_DIR / "best_risk_model.keras")
    print("\nBest ANN (chosen on validation macro F1):", best_ann["name"])
    print("Best saved model:", MODEL_DIR / "best_risk_model.keras")

    if ask_to_predict:
        try:
            predict_choice = input("\nPredict a new incident now? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            predict_choice = "no"
        if predict_choice in ["yes", "y"]:
            predict_new_incident()

    print("\nModule 3 completed.")
    print("Graphs saved in:", OUTPUT_DIR)
    print("Models saved in:", MODEL_DIR)

    return {
        "knn_classifier": knn["classifier"],
        "ann_architecture_1": ann_1,
        "ann_architecture_2": ann_2,
        "best_ann_name": best_ann["name"],
        "comparison": comparison,
        "knn_mae": knn["mae"],
        "knn_mse": knn["mse"],
        "knn_k": knn["k"],
        "knn_metric": knn["metric"],
        "validation_metrics": {ann["name"]: ann["validation_metrics"] for ann in (ann_1, ann_2)},
        "histories": {ann["name"]: {key: [float(v) for v in values]
                                    for key, values in ann["history"].history.items()}
                      for ann in (ann_1, ann_2)},
        "confusion_matrices": {
            "KNN": confusion_matrix(y_test, knn["predictions"], labels=[0, 1, 2]).tolist(),
            best_ann["name"]: confusion_matrix(
                y_test, best_ann["test_predictions"], labels=[0, 1, 2]).tolist(),
        },
        "roc": {
            "KNN": roc_points(y_test, knn["probabilities"]),
            best_ann["name"]: roc_points(y_test, best_ann["test_probabilities"]),
        },
        "wrong_predictions": wrong_examples,
        "split_sizes": {"train": len(split["X_train"]),
                        "validation": len(split["X_validation"]),
                        "test": len(split["X_test"])},
    }


def module3_menu() -> dict:
    """Simple menu for Module 3."""
    print("\nMODULE 3 - Neural Networks and Evaluation")
    print("1. Train and evaluate models")
    print("2. Predict a new incident")
    print("3. Exit")

    try:
        choice = input("Enter choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("Exiting Module 3.")
        return {}

    if choice == "1":
        return run_module3()
    if choice == "2":
        predict_new_incident()
        return {}
    if choice == "3":
        print("Exiting Module 3.")
        return {}

    print("Invalid choice.")
    return {}


if __name__ == "__main__":
    module3_menu()
