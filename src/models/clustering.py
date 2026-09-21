
from pathlib import Path
import json
import joblib

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "raw" / "students.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DATA_PATH = OUTPUT_DIR / "students_clustered.csv"

OUTPUT_CENTROID_PATH = OUTPUT_DIR / "cluster_centroids.csv"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

KMEANS_PATH = MODEL_DIR / "kmeans.pkl"

METADATA_PATH = MODEL_DIR / "cluster_metadata.json"

FEATURES = [
    "diagnostic_score",
    "assignment_avg",
    "quiz_avg",
    "uts_score",
    "learning_speed",
]

N_CLUSTERS = 3

RANDOM_STATE = 42


# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ==========================================
# PREPROCESSING
# ==========================================

def prepare_features(data):
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Fitur tidak ditemukan: {missing_features}"
        )

    if data[FEATURES].isnull().sum().sum() > 0:
        raise ValueError(
            "Terdapat missing value pada fitur clustering."
        )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        data[FEATURES]
    )

    return scaled_features, scaler


# ==========================================
# CLUSTERING
# ==========================================

def perform_clustering(scaled_features):
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    cluster_labels = model.fit_predict(
        scaled_features
    )

    return model, cluster_labels


# ==========================================
# PENAMAAN CLUSTER
# ==========================================

def assign_cluster_labels(data, model):
    centroids = model.cluster_centers_

    centroid_df = pd.DataFrame(
        centroids,
        columns=FEATURES,
    )

    centroid_df["overall_score"] = (
        centroid_df[FEATURES].mean(axis=1)
    )

    sorted_clusters = (
        centroid_df["overall_score"]
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    cluster_name_mapping = {
        int(sorted_clusters[0]): "Fast Learner",
        int(sorted_clusters[1]): "Steady Learner",
        int(sorted_clusters[2]): "Needs Guidance",
    }

    data["cluster_name"] = data["cluster"].map(
        cluster_name_mapping
    )

    centroid_df["cluster_name"] = (
        centroid_df.index
        .map(cluster_name_mapping)
    )

    return (
        data,
        centroid_df,
        cluster_name_mapping,
    )


# ==========================================
# SAVE MODEL
# ==========================================

def save_models(scaler, model, cluster_mapping):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    joblib.dump(
        model,
        KMEANS_PATH,
    )

    metadata = {
        "features": FEATURES,
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "cluster_mapping": {
            str(cluster_id): cluster_name
            for cluster_id, cluster_name
            in cluster_mapping.items()
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nModel berhasil disimpan:")
    print(f"- Scaler: {SCALER_PATH}")
    print(f"- K-Means: {KMEANS_PATH}")
    print(f"- Metadata: {METADATA_PATH}")


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("EDUADAPT AI - K-MEANS CLUSTERING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    print(f"\nJumlah data: {len(data)}")
    print(f"Fitur yang digunakan: {FEATURES}")

    scaled_features, scaler = prepare_features(
        data
    )

    model, cluster_labels = perform_clustering(
        scaled_features
    )

    data["cluster"] = cluster_labels

    silhouette = silhouette_score(
        scaled_features,
        cluster_labels,
    )

    (
        data,
        centroid_df,
        mapping,
    ) = assign_cluster_labels(
        data,
        model,
    )

    data.to_csv(
        OUTPUT_DATA_PATH,
        index=False,
    )

    centroid_df.to_csv(
        OUTPUT_CENTROID_PATH,
        index=True,
    )

    save_models(
        scaler,
        model,
        mapping,
    )

    print("\nDistribusi cluster:")
    print(data["cluster_name"].value_counts())

    print("\nPemetaan cluster:")

    for cluster_id, cluster_name in mapping.items():
        print(
            f"Cluster {cluster_id}: {cluster_name}"
        )

    print("\nNilai centroid:")
    print(centroid_df)

    print(
        f"\nSilhouette Score: {silhouette:.4f}"
    )

    print(
        "\nDataset hasil clustering disimpan di:"
    )
    print(OUTPUT_DATA_PATH)

    print(
        "\nCentroid disimpan di:"
    )
    print(OUTPUT_CENTROID_PATH)


if __name__ == "__main__":
    main()