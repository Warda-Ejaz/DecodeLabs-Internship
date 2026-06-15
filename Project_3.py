import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("customer_data.csv")

print("Dataset:")
print(df.head())

X = df[["Age", "AnnualIncome", "SpendingScore"]]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:")
print(X_pca.shape)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_pca)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)

df["Cluster"] = clusters


score = silhouette_score(X_pca, clusters)

print("\nSilhouette Score:", score)


plt.figure(figsize=(7,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

plt.title("Customer Segmentation")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()


print("\nCustomer Personas")

for cluster in sorted(df["Cluster"].unique()):

    group = df[df["Cluster"] == cluster]

    print("\nCluster", cluster)

    print(group[
        ["Age",
         "AnnualIncome",
         "SpendingScore"]
    ].mean())

print("\nProject Completed Successfully!")
