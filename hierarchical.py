# ============================================================
# UNIVERSITY ANALYSIS - HIERARCHICAL CLUSTERING
# ============================================================


# ============================================================
# CELL 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# CELL 2: MOUNT GOOGLE DRIVE
# ============================================================

drive.mount('/content/drive')


# ============================================================
# CELL 3: LOAD DATASET
# ============================================================

# Change this path if your dataset is stored somewhere else

file_path = "/content/drive/MyDrive/University Analysis/student_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

display(df.head())


# ============================================================
# CELL 4: DATASET INFORMATION
# ============================================================

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# ============================================================
# CELL 5: CHECK DATA QUALITY
# ============================================================

print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# CELL 6: SELECT FEATURES
# ============================================================

features = ["Attendance", "CGPA"]

print("Features selected for clustering:")
print(features)

print("\nFeature statistics:")
display(df[features].describe())


# ============================================================
# CELL 7: SELECT 500 RANDOM STUDENTS
# ============================================================

# Same sample size and random_state as the other methods

sample = df.sample(
n=500,
random_state=42
).copy()

print("Sample size:", sample.shape)

display(sample[features].head())


# ============================================================
# CELL 8: SAMPLE STATISTICS
# ============================================================

print("Average Attendance:",
round(sample["Attendance"].mean(), 2))

print("Average CGPA:",
round(sample["CGPA"].mean(), 2))

print("\nMinimum Attendance:",
round(sample["Attendance"].min(), 2))

print("Maximum Attendance:",
round(sample["Attendance"].max(), 2))

print("\nMinimum CGPA:",
round(sample["CGPA"].min(), 2))

print("Maximum CGPA:",
round(sample["CGPA"].max(), 2))


# ============================================================
# CELL 9: STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
sample[features]
)

print("Features standardized successfully.")

print("\nFirst 5 standardized values:")
print(X_scaled[:5])


# ============================================================
# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================
# ============================================================


# ============================================================
# CELL 10: CREATE HIERARCHICAL LINKAGE
# ============================================================

# Ward linkage builds the hierarchy by minimizing
# within-cluster variance.

linkage_matrix = linkage(
X_scaled,
method="ward"
)

print("Hierarchical linkage matrix created successfully!")

print("\nLinkage matrix shape:")
print(linkage_matrix.shape)


# ============================================================
# CELL 11: DISPLAY DENDROGRAM
# ============================================================

plt.figure(figsize=(14, 7))

dendrogram(
linkage_matrix,
truncate_mode="lastp",
p=30
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster / Sample")
plt.ylabel("Distance")

plt.grid(True)

plt.show()


# ============================================================
# CELL 12: APPLY AGGLOMERATIVE HIERARCHICAL CLUSTERING
# ============================================================

hierarchical = AgglomerativeClustering(
n_clusters=2,
linkage="ward"
)

cluster_labels = hierarchical.fit_predict(
X_scaled
)

# Convert cluster labels from 0,1 to 1,2
sample["Cluster"] = cluster_labels + 1

print("Hierarchical clustering completed successfully!")


# ============================================================
# CELL 13: VIEW CLUSTER ASSIGNMENTS
# ============================================================

print("Student Cluster Assignments:")

display(
sample[
[
"Student ID",
"Attendance",
"CGPA",
"Cluster"
]
].head(20)
)


# ============================================================
# CELL 14: CLUSTER SIZES
# ============================================================

cluster_counts = (
sample["Cluster"]
.value_counts()
.sort_index()
)

print("Cluster Sizes:")

for cluster, count in cluster_counts.items():
print(f"Cluster {cluster}: {count} students")


# ============================================================
# CELL 15: CLUSTER SUMMARY
# ============================================================

cluster_summary = (
sample
.groupby("Cluster")[features]
.mean()
.round(2)
)

print("Average values for each cluster:")

display(cluster_summary)


# ============================================================
# CELL 16: FIND CLUSTER CENTERS FOR VISUALIZATION
# ============================================================

# Hierarchical clustering does not produce centroids.
# These are only the average feature values of each
# cluster, used here to show the cluster locations.

cluster_centers = (
sample
.groupby("Cluster")[features]
.mean()
)

print("Average position of each cluster:")

display(cluster_centers)


# ============================================================
# CELL 17: VISUALIZE HIERARCHICAL CLUSTERS
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
sample["Attendance"],
sample["CGPA"],
c=cluster_labels,
alpha=0.6
)

# Mark average position of each cluster
plt.scatter(
cluster_centers["Attendance"],
cluster_centers["CGPA"],
marker="X",
s=250,
label="Cluster Average"
)

plt.xlabel("Attendance")
plt.ylabel("CGPA")

plt.title(
"Hierarchical Clustering of Students"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# CELL 18: FINAL RESULTS
# ============================================================

print("=" * 50)
print("FINAL HIERARCHICAL CLUSTERING RESULTS")
print("=" * 50)

print("\nNumber of students used:", len(sample))

print("Features used:", features)

print("Number of clusters:", 2)

print("\nCluster Sizes:")
print(cluster_counts)

print("\nCluster Averages:")

display(cluster_summary)


# ============================================================
# CELL 19: SAVE RESULTS
# ============================================================

output_path = (
"/content/drive/MyDrive/"
"University Analysis/"
"University_Analysis_Hierarchical_500_Results.csv"
)

sample.to_csv(
output_path,
index=False
)

print("Results saved successfully!")

print(output_path)