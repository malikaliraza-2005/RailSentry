# Student name: [Your Name]
# Roll number: [Your Roll Number]
# Course code: [Your Course Code]
# Module description: Railway station dataset analysis and machine learning

"""
Module 2:
Loads and cleans the railway incident dataset.
Prepares data, draws graphs, trains KNN models,
and creates K-Means location clusters.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_FILE = "railway_station_incident_response_dataset.csv"
# All paths are relative to this file, so the program works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

RISK_ORDER = ["Low", "Medium", "High"]

# Valid physical range of each numeric column. A value outside its range is
# a data-entry error (for example 190% crowd density), so it is treated as
# missing and filled later by the imputer.
VALID_RANGES = {
    "grid_x": (0, 14),
    "grid_y": (0, 14),
    "crowd_density_pct": (0, 100),
    "hazard_severity_1_5": (1, 5),
    "equipment_health_pct": (0, 100),
    "obstruction_level_0_3": (0, 3),
    "hour_of_day": (0, 23),
}

NUMERIC_COLUMNS = [
    "grid_x",
    "grid_y",
    "crowd_density_pct",
    "hazard_severity_1_5",
    "equipment_health_pct",
    "obstruction_level_0_3",
    "hour_of_day",
]

REQUIRED_COLUMNS = {
    "incident_id",
    "grid_x",
    "grid_y",
    "incident_type",
    "crowd_density_pct",
    "hazard_severity_1_5",
    "equipment_health_pct",
    "obstruction_level_0_3",
    "hour_of_day",
    "risk_level",
    "response_time_min",
}


def find_dataset() -> Path:
    """Find the railway CSV in the project folder."""
    local_file = PROJECT_DIR / DATASET_FILE
    if local_file.exists():
        return local_file

    raise FileNotFoundError(
        f"{DATASET_FILE} was not found. Copy it beside module2_data_ml.py."
    )


def load_and_check_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load the CSV and perform basic data checks."""
    path = csv_path if csv_path is not None else find_dataset()

    try:
        data = pd.read_csv(path)
    except Exception as error:
        raise ValueError(f"Could not load the CSV: {error}") from error

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset schema is missing columns: {sorted(missing_columns)}"
        )

    print("\nFirst five records:")
    print(data.head().to_string(index=False))
    print("\nDataset shape:", data.shape)
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values:")
    print(data.isna().sum().to_string())
    print("\nStatistical summary:")
    print(data.describe(include="all").to_string())

    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare missing values without leaking test-set information.

    Missing feature values are NOT filled here. They are filled inside each
    model pipeline by SimpleImputer, which learns the median / most frequent
    value from the training split only. Here we only:
        - convert numeric columns to numbers
        - remove duplicate records
        - remove records with no risk label (they cannot be used to train
          or test the classifier)
    Records with a missing response time are kept for classification and
    skipped only by the response-time regressor.
    """
    cleaned = data.copy()

    for column in NUMERIC_COLUMNS + ["response_time_min"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates(subset="incident_id")
    cleaned = cleaned.dropna(subset=["risk_level"])
    cleaned = cleaned[cleaned["risk_level"].isin(RISK_ORDER)]

    return cleaned.reset_index(drop=True)


def handle_outliers(data: pd.DataFrame, report: dict = None) -> pd.DataFrame:
    """
    Handle impossible and extreme values.

    Features: values outside the column's valid range are data-entry errors,
    so they are set to missing (and imputed later). Normal IQR clipping is
    NOT used on features, because it would damage valid values - for example
    obstruction level 3 would be clipped to 2.5.

    Target: response times above Q3 + 3 * IQR (an "extreme outlier" fence)
    are treated as recording errors and set to missing, so the regressor
    ignores them. Values just above the normal 1.5 * IQR fence are kept
    because they are realistic slow responses.
    """
    cleaned = data.copy()

    for column, (lowest, highest) in VALID_RANGES.items():
        invalid = cleaned[column].lt(lowest) | cleaned[column].gt(highest)
        if invalid.any():
            print(f"{column}: {int(invalid.sum())} impossible value(s) set to missing")
            if report is not None:
                report[column] = {
                    "reason": f"outside valid range {lowest}-{highest}",
                    "values": cleaned.loc[invalid, column].tolist(),
                    "incident_ids": cleaned.loc[invalid, "incident_id"].tolist(),
                }
        cleaned.loc[invalid, column] = np.nan

    response = cleaned["response_time_min"]
    q1 = response.quantile(0.25)
    q3 = response.quantile(0.75)
    upper_limit = q3 + 3 * (q3 - q1)
    extreme = response.gt(upper_limit) | response.le(0)
    if extreme.any() and report is not None:
        report["response_time_min"] = {
            "reason": f"extreme outlier above {upper_limit:.2f} min (Q3 + 3 x IQR)",
            "values": response[extreme].tolist(),
            "incident_ids": cleaned.loc[extreme, "incident_id"].tolist(),
        }
    if extreme.any():
        print(
            f"response_time_min: {int(extreme.sum())} extreme value(s) above "
            f"{upper_limit:.2f} min set to missing: "
            f"{sorted(response[extreme].tolist())}"
        )
    cleaned.loc[extreme, "response_time_min"] = np.nan

    return cleaned


def create_location_clusters(
    data: pd.DataFrame,
    number_of_clusters: int = 3,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Cluster station locations using grid_x and grid_y."""
    location_data = data[["grid_x", "grid_y"]].copy()

    location_scaler = StandardScaler()
    scaled_locations = location_scaler.fit_transform(location_data)

    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(scaled_locations)

    clustered = data.copy()
    clustered["location_cluster"] = cluster_labels
    return clustered, kmeans, location_scaler


def plot_elbow_method(data: pd.DataFrame) -> list:
    """Draw elbow graph for K-Means cluster selection."""
    location_data = data[["grid_x", "grid_y"]]
    location_scaler = StandardScaler()
    scaled_locations = location_scaler.fit_transform(location_data)

    inertia_values = []
    cluster_numbers = range(2, 7)

    for cluster_count in cluster_numbers:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )
        model.fit(scaled_locations)
        inertia_values.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_numbers, inertia_values, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Railway Location Clusters")
    plt.xticks(cluster_numbers)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_elbow.png")
    plt.show()
    plt.close()

    return [{"clusters": int(k), "inertia": float(v)}
            for k, v in zip(cluster_numbers, inertia_values)]


def plot_location_clusters(
    data: pd.DataFrame,
    kmeans: KMeans,
    location_scaler: StandardScaler,
) -> None:
    """Display location clusters and their centroids."""
    plt.figure(figsize=(8, 6))
    colours = data["location_cluster"].map({
        0: "green",
        1: "orange",
        2: "red",
        3: "purple",
        4: "blue",
        5: "cyan",
    })

    plt.scatter(
        data["grid_x"],
        data["grid_y"],
        c=colours,
        alpha=0.65,
        s=35,
    )

    centroids = location_scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="X",
        s=220,
        label="Centroid",
    )

    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.title("Railway Station Location Clusters")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "location_clusters.png")
    plt.show()
    plt.close()


def draw_required_graphs(data: pd.DataFrame, raw_data: pd.DataFrame = None) -> None:
    """
    Draw the four graphs required for Module 2.

    The boxplot uses the raw data (when given) so the outliers that were
    handled during cleaning are visible.
    """
    if raw_data is None:
        raw_data = data
    # 1. Risk class distribution
    plt.figure(figsize=(8, 5))
    # Fix the bar order so each colour always matches its risk level.
    data["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0).plot(
        kind="bar", color=["green", "orange", "red"]
    )
    plt.xlabel("Risk level")
    plt.ylabel("Number of incidents")
    plt.title("Railway Incident Risk Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_distribution.png")
    plt.show()
    plt.close()

    # 2. Response-time histogram
    plt.figure(figsize=(8, 5))
    plt.hist(data["response_time_min"].dropna(), bins=20,
             color="steelblue", edgecolor="black")
    plt.xlabel("Response time (minutes)")
    plt.ylabel("Number of incidents")
    plt.title("Response-Time Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "response_time_histogram.png")
    plt.show()
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 7))
    numeric_for_correlation = data[
        NUMERIC_COLUMNS + ["response_time_min"]
    ].corr(numeric_only=True)
    sns.heatmap(numeric_for_correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.show()
    plt.close()

    # 4. Outlier boxplot
    plt.figure(figsize=(12, 6))
    raw_numeric = raw_data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    raw_numeric.boxplot(rot=45)
    plt.title("Numeric Feature Boxplots (raw data, before cleaning)")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_boxplot.png")
    plt.show()
    plt.close()


def make_preprocessing_pipeline(
    numeric_columns: list,
    categorical_columns: list,
) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline([
        ("missing", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("missing", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


FEATURE_COLUMNS = NUMERIC_COLUMNS + ["incident_type", "location_cluster"]
MODEL_NUMERIC_COLUMNS = NUMERIC_COLUMNS
# location_cluster is a group label (0, 1, 2), not a measurement, so it is
# one-hot encoded instead of being scaled like a number.
MODEL_CATEGORICAL_COLUMNS = ["incident_type", "location_cluster"]
K_VALUES = [3, 5, 7, 9, 11]
DISTANCE_METRICS = ["euclidean", "manhattan"]


def build_knn_pipeline(estimator) -> Pipeline:
    """Create preprocessing + KNN in one pipeline (imputer fitted on training data)."""
    return Pipeline([
        ("preprocessing", make_preprocessing_pipeline(
            MODEL_NUMERIC_COLUMNS,
            MODEL_CATEGORICAL_COLUMNS,
        )),
        ("model", estimator),
    ])


def train_knn_classifier(data: pd.DataFrame) -> dict:
    """Tune KNN classifier using a validation set, not the test set."""
    X = data[FEATURE_COLUMNS]
    y = data["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train_part, X_validation, y_train_part, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    results = []
    for k_value in K_VALUES:
        for metric in DISTANCE_METRICS:
            model = build_knn_pipeline(
                KNeighborsClassifier(n_neighbors=k_value, metric=metric)
            )
            model.fit(X_train_part, y_train_part)
            predictions = model.predict(X_validation)

            results.append({
                "k": k_value,
                "metric": metric,
                "accuracy": accuracy_score(y_validation, predictions),
                "macro_f1": f1_score(y_validation, predictions, average="macro"),
            })

    results_table = pd.DataFrame(results)
    print("\nKNN Classification Tuning Table (validation set):")
    print(results_table.to_string(index=False))

    plt.figure(figsize=(9, 5))
    for metric in DISTANCE_METRICS:
        metric_rows = results_table[results_table["metric"] == metric]
        plt.plot(metric_rows["k"], metric_rows["accuracy"],
                 marker="o", label=f"{metric} accuracy")
        plt.plot(metric_rows["k"], metric_rows["macro_f1"],
                 marker="s", linestyle="--", label=f"{metric} macro F1")
    plt.xlabel("k value")
    plt.ylabel("Validation score")
    plt.title("KNN k versus Validation Accuracy and Macro F1")
    plt.xticks(K_VALUES)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_k_accuracy.png")
    plt.show()
    plt.close()

    # The classes are imbalanced (most incidents are Low risk), so accuracy
    # alone favours predicting "Low". Macro F1 treats all three classes
    # equally, so it is used to choose k; accuracy breaks ties.
    best_row = results_table.sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).iloc[0]
    best_k = int(best_row["k"])
    best_metric = str(best_row["metric"])

    best_model = build_knn_pipeline(
        KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
    )
    best_model.fit(X_train, y_train)

    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_macro_f1 = f1_score(y_test, test_predictions, average="macro")

    print("\nBest KNN classifier (chosen by validation macro F1):")
    print("k =", best_k)
    print("Distance metric =", best_metric)
    print("Test accuracy =", round(test_accuracy, 4))
    print("Test macro F1 =", round(test_macro_f1, 4))

    joblib.dump(best_model, MODEL_DIR / "knn_risk_classifier.joblib")

    return {
        "model": best_model,
        "best_k": best_k,
        "best_metric": best_metric,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "tuning_table": results_table,
    }


def train_knn_regressor(data: pd.DataFrame) -> dict:
    """Tune and train KNN to predict response time in minutes."""
    # Rows whose response time is missing (or was an extreme recording
    # error) cannot be used for regression.
    data_with_target = data.dropna(subset=["response_time_min"])

    X = data_with_target[FEATURE_COLUMNS]
    y = data_with_target["response_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
    X_train_part, X_validation, y_train_part, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
    )

    results = []
    for k_value in K_VALUES:
        for metric in DISTANCE_METRICS:
            model = build_knn_pipeline(
                KNeighborsRegressor(n_neighbors=k_value, metric=metric)
            )
            model.fit(X_train_part, y_train_part)
            predictions = model.predict(X_validation)
            results.append({
                "k": k_value,
                "metric": metric,
                "validation_MAE": mean_absolute_error(y_validation, predictions),
            })

    results_table = pd.DataFrame(results)
    print("\nKNN Regression Tuning Table (validation set):")
    print(results_table.to_string(index=False))

    best_row = results_table.sort_values("validation_MAE").iloc[0]
    best_k = int(best_row["k"])
    best_metric = str(best_row["metric"])

    model = build_knn_pipeline(
        KNeighborsRegressor(n_neighbors=best_k, metric=best_metric)
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))

    print("\nKNN Regression Results (test set):")
    print("k =", best_k, "| Distance metric =", best_metric)
    print("MAE =", round(mae, 4), "minutes")
    print("MSE =", round(mse, 4))
    print("RMSE =", round(rmse, 4), "minutes")

    joblib.dump(model, MODEL_DIR / "knn_response_time_regressor.joblib")

    return {
        "model": model,
        "best_k": best_k,
        "best_metric": best_metric,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def predict_new_cluster(
    grid_x: float,
    grid_y: float,
    kmeans: KMeans,
    location_scaler: StandardScaler,
) -> int:
    """Return the cluster for a new railway location."""
    new_location = pd.DataFrame([[grid_x, grid_y]], columns=["grid_x", "grid_y"])
    scaled_location = location_scaler.transform(new_location)
    cluster = int(kmeans.predict(scaled_location)[0])
    return cluster


def load_prepared_data(verbose: bool = False) -> pd.DataFrame:
    """
    Load and clean the dataset with the same steps as Module 2.

    Modules 3 and 4 call this so every module trains and tests on exactly
    the same cleaned data.
    """
    path = find_dataset()
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset schema is missing columns: {sorted(missing_columns)}")

    if verbose:
        data = handle_outliers(handle_missing_values(data))
    else:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            data = handle_outliers(handle_missing_values(data))
    return data


def run_module2(new_location: tuple = None) -> dict:
    """
    Run all Module 2 steps.

    new_location: optional (grid_x, grid_y) to assign to a cluster. When it
    is not given, the user is asked to type it (terminal menu behaviour).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    raw_data = load_and_check_dataset()
    print("\nCleaning the data:")
    data = handle_missing_values(raw_data)
    cleaning_report = {}
    data = handle_outliers(data, cleaning_report)
    print("Records after cleaning:", len(data))
    print("Missing values after cleaning (filled later by the model imputer):")
    print(data.isna().sum().to_string())

    np_average_hazard = np.nanmean(data["hazard_severity_1_5"].to_numpy(dtype=float))
    print("\nNumPy average hazard severity:",
          round(float(np_average_hazard), 3))

    data, kmeans, location_scaler = create_location_clusters(
        data,
        number_of_clusters=3,
    )

    joblib.dump(kmeans, MODEL_DIR / "location_kmeans.joblib")
    joblib.dump(location_scaler, MODEL_DIR / "location_scaler.joblib")

    elbow = plot_elbow_method(data)
    plot_location_clusters(data, kmeans, location_scaler)
    draw_required_graphs(data, raw_data)

    cluster_number = None
    try:
        if new_location is None:
            new_grid_x = float(input("\nEnter a new location grid_x (0-14): "))
            new_grid_y = float(input("Enter a new location grid_y (0-14): "))
        else:
            new_grid_x, new_grid_y = float(new_location[0]), float(new_location[1])

        if not (0 <= new_grid_x <= 14 and 0 <= new_grid_y <= 14):
            raise ValueError("Grid coordinates must be between 0 and 14.")

        cluster_number = predict_new_cluster(
            new_grid_x,
            new_grid_y,
            kmeans,
            location_scaler,
        )
        print("New location assigned to cluster:", cluster_number)
        print(
            "Cluster interpretation:",
            "Station location group based on grid X and grid Y similarity.",
        )
    except (ValueError, EOFError):
        print("Invalid cluster input. No new-record cluster was calculated.")

    classifier_result = train_knn_classifier(data)
    regressor_result = train_knn_regressor(data)

    print("\nModule 2 completed successfully.")
    print("Graphs saved in:", OUTPUT_DIR.resolve())
    print("Models saved in:", MODEL_DIR.resolve())

    return {
        "data": data,
        "raw_data": raw_data,
        "cleaning_report": cleaning_report,
        "elbow": elbow,
        "new_location_cluster": cluster_number,
        "kmeans": kmeans,
        "location_scaler": location_scaler,
        "classifier": classifier_result,
        "regressor": regressor_result,
    }


def module2_menu() -> dict:
    """Simple menu for running Module 2."""
    print("\nMODULE 2 - Data and Machine Learning")
    print("1. Run complete Module 2")
    print("2. Exit")

    try:
        choice = input("Enter your choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("Exiting Module 2.")
        return {}

    if choice == "1":
        return run_module2()
    if choice == "2":
        print("Exiting Module 2.")
        return {}

    print("Invalid choice. Enter 1 or 2.")
    return {}


if __name__ == "__main__":
    module2_menu()
