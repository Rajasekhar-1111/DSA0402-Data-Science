"""Bank Marketing subscription prediction and customer segmentation workflow."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
RESULTS = OUTPUTS / "results"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def make_output_dirs():
    for directory in (FIGURES, TABLES, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load a local bank CSV, or retrieve the official UCI Bank Marketing data."""
    candidates = [ROOT / "bank.csv", ROOT / "bank-full.csv", ROOT / "data" / "bank.csv"]
    source = next((path for path in candidates if path.exists()), None)
    if source is not None:
        try:
            frame = pd.read_csv(source, sep=None, engine="python")
        except Exception:
            frame = pd.read_csv(source, sep=";")
        print(f"Data source: {source}")
        return frame

    print("No local bank.csv found. Downloading the official UCI Bank Marketing dataset...")
    with urlopen(DATA_URL, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))
        nested_name = "bank.zip"
        nested_archive = ZipFile(BytesIO(archive.read(nested_name)))
        member = next(name for name in nested_archive.namelist() if name.endswith("bank-full.csv"))
        with nested_archive.open(member) as csv_file:
            frame = pd.read_csv(csv_file, sep=";")
    frame.to_csv(ROOT / "bank.csv", index=False)
    print(f"Data source: UCI archive ({nested_name}/{member}); saved as bank.csv")
    return frame


def inspect_data(frame):
    print("\n=== DATA INSPECTION ===")
    print(f"Rows: {frame.shape[0]}")
    print(f"Columns: {frame.shape[1]}")
    print(f"Column names: {list(frame.columns)}")
    print("Data types:\n", frame.dtypes.to_string())
    print("First 5 rows:\n", frame.head().to_string(index=False))
    print("Missing values:\n", frame.isna().sum().to_string())
    print(f"Duplicate rows: {frame.duplicated().sum()}")
    categorical = frame.select_dtypes(include="object")
    unique_values = {column: sorted(categorical[column].dropna().unique().tolist())
                     for column in categorical.columns}
    print("Unique categorical values:")
    print(json.dumps(unique_values, indent=2))
    print("Target distribution:\n", frame["y"].value_counts(dropna=False).to_string())
    pd.DataFrame({"missing_count": frame.isna().sum(),
                  "missing_percent": frame.isna().mean().mul(100)}).to_csv(
        TABLES / "missing_values.csv")
    pd.DataFrame({"column": list(unique_values.keys()),
                  "unique_values": [", ".join(map(str, values)) for values in unique_values.values()]}) \
        .to_csv(TABLES / "categorical_unique_values.csv", index=False)


def clean_data(frame):
    cleaned = frame.copy()
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    unknown_summary = pd.DataFrame({
        "unknown_count": [(cleaned[column].astype(str).str.lower() == "unknown").sum()
                           for column in cleaned.columns],
    }, index=cleaned.columns)
    unknown_summary["unknown_percent"] = unknown_summary["unknown_count"].div(len(cleaned)).mul(100)
    unknown_summary.to_csv(TABLES / "unknown_values.csv")
    numeric_columns = cleaned.select_dtypes(include=np.number).columns.tolist()
    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in cleaned.select_dtypes(include="object").columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna("unknown")
    print("\n=== CLEANING ===")
    print(f"Duplicates removed: {duplicate_count}")
    print("Unknown-value counts:\n", unknown_summary.to_string())
    return cleaned


def _cap_outliers(frame, columns):
    summary = []
    capped = frame.copy()
    for column in columns:
        values = capped[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((capped[column] < lower) | (capped[column] > upper)).sum())
        capped[column] = capped[column].clip(lower, upper)
        summary.append({"feature": column, "q1": q1, "q3": q3, "lower_bound": lower,
                        "upper_bound": upper, "outlier_count": count, "treatment": "IQR capping"})
    return capped, pd.DataFrame(summary)


def preprocess_data(frame):
    """Prepare model data, retaining all useful records and excluding duration leakage."""
    data = frame.copy()
    target = data.pop("y").map({"no": 0, "yes": 1}).astype(int)
    data = data.drop(columns=["duration"], errors="ignore")
    numeric = data.select_dtypes(include=np.number).columns.tolist()
    # Cap extreme continuous/count values for stable modelling without deleting records.
    data, outliers = _cap_outliers(data, numeric)
    outliers.to_csv(TABLES / "outlier_analysis.csv", index=False)
    binary_columns = [column for column in ["default", "housing", "loan"] if column in data]
    for column in binary_columns:
        data[column] = data[column].map({"no": 0, "yes": 1}).fillna(0).astype(int)
    categorical = data.select_dtypes(include="object").columns.tolist()
    categorical = [column for column in categorical if column not in binary_columns]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), data.select_dtypes(include=np.number).columns.tolist()),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop")
    print("\n=== PREPROCESSING ===")
    print("Removed target: y")
    print("Removed leakage feature: duration")
    print(f"Numeric features: {data.select_dtypes(include=np.number).columns.tolist()}")
    print(f"One-hot categorical features: {categorical}")
    print(f"Rows retained after duplicate removal: {len(data)}")
    return data, target, preprocessor, outliers


def _savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150)
    plt.close()


def perform_eda(frame):
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=frame, x="y")
    plt.title("Term Deposit Subscription Distribution"); plt.xlabel("Subscribed (y)"); plt.ylabel("Customers"); _savefig("target_distribution.png")
    numeric = [column for column in ["age", "balance", "campaign", "pdays", "previous"] if column in frame]
    frame[numeric].hist(figsize=(11, 7), bins=30)
    plt.suptitle("Numerical Feature Distributions"); _savefig("numeric_distributions.png")
    important = [column for column in ["age", "balance", "campaign"] if column in frame]
    plt.figure(figsize=(10, 4)); frame[important].plot(kind="box"); plt.title("Important Numerical Feature Box Plots"); plt.xlabel("Feature"); plt.ylabel("Value"); _savefig("numeric_boxplots.png")
    for column in ["job", "education", "marital", "housing", "loan"]:
        if column in frame:
            plt.figure(figsize=(10, 4)); order = frame[column].value_counts().index
            sns.countplot(data=frame, y=column, order=order); plt.title(f"Customer Distribution by {column.title()}"); plt.xlabel("Customers"); plt.ylabel(column.title()); _savefig(f"distribution_{column}.png")
    for column in ["job", "education", "marital", "housing", "loan"]:
        if column in frame:
            rates = frame.groupby(column, dropna=False)["y"].apply(lambda values: (values == "yes").mean()).sort_values(ascending=False)
            plt.figure(figsize=(10, 4)); rates.plot(kind="bar", color="#0f766e"); plt.title(f"Subscription Rate by {column.title()}"); plt.xlabel(column.title()); plt.ylabel("Subscription rate"); plt.xticks(rotation=35, ha="right"); _savefig(f"subscription_by_{column}.png")
    balance_bins = pd.cut(frame["balance"], bins=[-np.inf, 0, 500, 1500, 5000, np.inf],
                          labels=["negative", "0-500", "501-1500", "1501-5000", "5000+"])
    balance_rates = frame.assign(balance_band=balance_bins).groupby("balance_band", observed=False)["y"].apply(lambda values: (values == "yes").mean())
    plt.figure(figsize=(9, 4)); balance_rates.plot(kind="bar", color="#0f766e"); plt.title("Subscription Rate by Balance Band"); plt.xlabel("Balance band"); plt.ylabel("Subscription rate"); plt.xticks(rotation=25, ha="right"); _savefig("subscription_by_balance.png")
    for column in ["campaign", "contact", "poutcome"]:
        if column in frame:
            plt.figure(figsize=(10, 4)); sns.barplot(data=frame, x=column, y="y", estimator=lambda values: (values == "yes").mean(), errorbar=None); plt.title(f"Subscription Rate by {column.title()}"); plt.xlabel(column.title()); plt.ylabel("Subscription rate"); plt.xticks(rotation=35, ha="right"); _savefig(f"subscription_by_{column}.png")
    correlation = frame.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(10, 7)); sns.heatmap(correlation, annot=True, cmap="vlag", center=0, fmt=".2f"); plt.title("Numerical Feature Correlation Heatmap"); plt.xlabel("Feature"); plt.ylabel("Feature"); _savefig("correlation_heatmap.png")


def perform_statistics(frame):
    numeric = frame.select_dtypes(include=np.number)
    descriptive = numeric.agg(["mean", "median", "var", "std"]).T
    descriptive.columns = ["mean", "median", "variance", "std_dev"]
    descriptive.to_csv(TABLES / "descriptive_statistics.csv")
    numeric.cov().to_csv(TABLES / "covariance_matrix.csv")
    numeric.corr(method="pearson").to_csv(TABLES / "pearson_correlation_matrix.csv")
    print("\n=== DESCRIPTIVE STATISTICS ===\n", descriptive.to_string())
    return descriptive


def perform_statistical_inference(frame):
    ages = frame["age"].dropna().astype(float)
    sample_size, sample_mean = len(ages), ages.mean()
    sample_std = ages.std(ddof=1)
    margin = stats.t.ppf(0.975, sample_size - 1) * sample_std / np.sqrt(sample_size)
    result = pd.DataFrame([{"sample_size": sample_size, "sample_mean": sample_mean,
                            "sample_std_dev": sample_std, "confidence_level": 0.95,
                            "ci_lower": sample_mean - margin, "ci_upper": sample_mean + margin}])
    result.to_csv(TABLES / "age_mean_confidence_interval.csv", index=False)
    print("\n=== 95% CI FOR POPULATION MEAN AGE ===\n", result.to_string(index=False))
    return result


def train_classification_models(data, target, preprocessor, knn_neighbors=5, tree_max_depth=5, logistic_max_iter=1000):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2,
                                                        stratify=target, random_state=RANDOM_STATE)
    models = {
        "kNN": KNeighborsClassifier(n_neighbors=knn_neighbors),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=tree_max_depth, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=logistic_max_iter, random_state=RANDOM_STATE),
    }
    fitted, predictions = {}, {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        predictions[name] = pipeline.predict(x_test)
    return fitted, predictions, y_test, data, target


def evaluate_models(fitted, predictions, y_test, full_data, full_target):
    rows = []
    print("\n=== MODEL EVALUATION ===")
    for name, predicted in predictions.items():
        metrics = {"Model": name, "Accuracy": accuracy_score(y_test, predicted),
                   "Precision": precision_score(y_test, predicted, zero_division=0),
                   "Recall": recall_score(y_test, predicted, zero_division=0),
                   "F1-score": f1_score(y_test, predicted, zero_division=0)}
        rows.append(metrics)
        report = classification_report(y_test, predicted, target_names=["Not subscribed", "Subscribed"], output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(RESULTS / f"{name.lower().replace(' ', '_')}_classification_report.csv")
        matrix = confusion_matrix(y_test, predicted)
        pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(TABLES / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv")
        plt.figure(figsize=(5, 4)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False); plt.title(f"{name} Confusion Matrix"); plt.xlabel("Predicted label"); plt.ylabel("Actual label"); _savefig(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        print(f"{name}: accuracy={metrics['Accuracy']:.4f}, precision={metrics['Precision']:.4f}, recall={metrics['Recall']:.4f}, f1={metrics['F1-score']:.4f}")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    best_name = comparison.sort_values(["F1-score", "Recall", "Precision", "Accuracy"], ascending=False).iloc[0]["Model"]
    best = comparison[comparison["Model"] == best_name].iloc[0].to_dict()
    print(f"Best Model: {best_name}\nReason for selection: highest positive-class F1-score, with recall as tie-breaker.")
    print("Best metrics:", {key: best[key] for key in ["Accuracy", "Precision", "Recall", "F1-score"]})
    full_predictions = fitted[best_name].predict(full_data)
    return comparison, best_name, full_predictions


def perform_kmeans(data, target):
    education_order = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 1}
    cluster_frame = pd.DataFrame({
        "age": data["age"], "balance": data["balance"], "campaign": data["campaign"],
        "previous": data["previous"], "education": data["education"].map(education_order).fillna(1),
    })
    scaled = StandardScaler().fit_transform(cluster_frame)
    inertias = []
    for k in range(1, 11):
        inertias.append(KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(scaled).inertia_)
    elbow = pd.DataFrame({"k": range(1, 11), "wcss_inertia": inertias})
    elbow.to_csv(TABLES / "wcss_values.csv", index=False)
    plt.figure(figsize=(7, 4)); plt.plot(elbow["k"], elbow["wcss_inertia"], marker="o"); plt.title("Elbow Method for K-means"); plt.xlabel("Number of clusters (K)"); plt.ylabel("WCSS / inertia"); _savefig("elbow_curve.png")
    # Select the point with maximum perpendicular distance from the endpoint line.
    points = elbow[["k", "wcss_inertia"]].to_numpy(float)
    normalized = (points - points.min(axis=0)) / np.where(points.ptp(axis=0) == 0, 1, points.ptp(axis=0))
    start, end = normalized[0], normalized[-1]
    distances = np.abs(np.cross(end - start, normalized - start)) / np.linalg.norm(end - start)
    selected_k = int(elbow.iloc[int(np.argmax(distances))]["k"])
    if selected_k < 2:
        selected_k = 2
    final = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    labels = final.fit_predict(scaled)
    clustered = cluster_frame.copy(); clustered["cluster"] = labels; clustered["y"] = target.to_numpy()
    distribution = clustered["cluster"].value_counts().sort_index().rename("customer_count").to_frame()
    distribution["percentage"] = distribution["customer_count"].div(len(clustered)).mul(100)
    distribution.to_csv(RESULTS / "cluster_distribution.csv")
    pca = PCA(n_components=2, random_state=RANDOM_STATE); components = pca.fit_transform(scaled)
    plt.figure(figsize=(8, 5)); sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="tab10", s=22, alpha=0.7); plt.title(f"Customer Clusters PCA (K={selected_k})"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(title="Cluster"); _savefig("cluster_pca.png")
    print("\n=== K-MEANS ===")
    print("WCSS values:\n", elbow.to_string(index=False)); print(f"Optimal K selected from elbow geometry: {selected_k}"); print("Cluster distribution:\n", distribution.to_string())
    return clustered, elbow, selected_k


def analyze_clusters(clustered):
    summary = clustered.groupby("cluster").agg(customer_count=("cluster", "size"), mean_age=("age", "mean"), mean_balance=("balance", "mean"), mean_campaign=("campaign", "mean"), mean_previous=("previous", "mean"), mean_education=("education", "mean"), actual_subscription_rate=("y", lambda values: (values == 1).mean())).reset_index()
    summary.to_csv(RESULTS / "cluster_summary.csv", index=False)
    print("\n=== CLUSTER SUMMARY ===\n", summary.to_string(index=False))
    return summary


def compare_predictions_and_clusters(clustered, full_predictions, summary):
    clustered = clustered.copy(); clustered["predicted"] = full_predictions
    comparison = clustered.groupby("cluster").agg(actual_subscription_rate=("y", "mean"), predicted_subscription_rate=("predicted", "mean"), customer_count=("cluster", "size")).reset_index()
    comparison["actual_subscription_rate"] = comparison["actual_subscription_rate"].mul(100)
    comparison["predicted_subscription_rate"] = comparison["predicted_subscription_rate"].mul(100)
    actual = comparison["actual_subscription_rate"]
    comparison["segment"] = pd.cut(actual, bins=[-np.inf, actual.quantile(1/3), actual.quantile(2/3), np.inf], labels=["Low-response", "Moderate-potential", "High-potential"], duplicates="drop").astype(str)
    comparison.to_csv(RESULTS / "cluster_prediction_comparison.csv", index=False)
    print("\n=== CLUSTER VS PREDICTION ===\n", comparison.to_string(index=False))
    return clustered, comparison


def generate_recommendations(clustered, comparison, summary):
    rows = []
    for _, row in comparison.iterrows():
        cluster = int(row["cluster"]); profile = summary.loc[summary["cluster"] == cluster].iloc[0]
        segment = row["segment"]
        if segment == "High-potential":
            strategy, priority = "Prioritize personalized follow-up and focused term-deposit offers.", "High"
        elif segment == "Moderate-potential":
            strategy, priority = "Use targeted, informative outreach and test timing or channel before increasing contact volume.", "Medium"
        else:
            strategy, priority = "Use low-cost, selective outreach and review contact frequency before further targeting.", "Low"
        characteristics = (f"mean age {profile['mean_age']:.2f}, mean balance {profile['mean_balance']:.2f}, "
                           f"mean campaign contacts {profile['mean_campaign']:.2f}, mean previous contacts {profile['mean_previous']:.2f}, "
                           f"mean education code {profile['mean_education']:.2f}")
        rows.append({"cluster": cluster, "segment": segment, "main_characteristics": characteristics,
                     "actual_subscription_rate_percent": row["actual_subscription_rate"],
                     "predicted_subscription_rate_percent": row["predicted_subscription_rate"],
                     "recommended_marketing_strategy": strategy, "priority": priority,
                     "reason": "Priority is based on the observed cluster subscription rate and the selected model's cluster prediction rate."})
    recommendations = pd.DataFrame(rows)
    recommendations.to_csv(RESULTS / "marketing_recommendations.csv", index=False)
    print("\n=== MARKETING RECOMMENDATIONS ===\n", recommendations.to_string(index=False))
    return recommendations


def main():
    make_output_dirs()
    raw = load_data()
    inspect_data(raw)
    cleaned = clean_data(raw)
    perform_eda(cleaned)
    model_data, target, preprocessor, _ = preprocess_data(cleaned)
    statistics_frame = cleaned.copy()
    statistics_frame["y"] = statistics_frame["y"].map({"no": 0, "yes": 1})
    perform_statistics(statistics_frame)
    perform_statistical_inference(cleaned)
    fitted, predictions, y_test, model_data, target = train_classification_models(model_data, target, preprocessor)
    _, best_name, full_predictions = evaluate_models(fitted, predictions, y_test, model_data, target)
    clustered, _, _ = perform_kmeans(cleaned.drop(columns=["y", "duration"], errors="ignore"), target)
    summary = analyze_clusters(clustered)
    clustered, comparison = compare_predictions_and_clusters(clustered, full_predictions, summary)
    generate_recommendations(clustered, comparison, summary)
    print(f"\nWorkflow complete. Best model: {best_name}")
    print(f"Generated outputs under: {OUTPUTS}")


if __name__ == "__main__":
    main()
