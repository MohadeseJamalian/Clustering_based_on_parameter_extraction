import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np


data = pd.read_csv('output.csv')


labels = data['Country']
features = data.drop(['Country'], axis=1)


pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features)

scaler=MinMaxScaler()
features_reduced=scaler.fit_transform(features_reduced)


sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features_reduced)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=4)


kmeans.fit(features_reduced)


cluster_labels = kmeans.predict(features_reduced)


cluster_centers = kmeans.cluster_centers_

n_clusters = kmeans.n_clusters


colors = ['r', 'g', 'b', 'y']
markers = ['o', 'o', 'o', 'o']
center_markers = ['o','o','o','o']
for i in range(n_clusters):
    plt.scatter(features_reduced[cluster_labels==i, 0], features_reduced[cluster_labels==i, 1], c=colors[i], marker=markers[i], label=f'Cluster {i}')
    plt.scatter(cluster_centers[i, 0], cluster_centers[i, 1], c='black', marker='o', s=300 )
    
    

plt.legend(title='Clusters', loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Clustered Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


for j in range(len(labels)):
    print(labels[j], cluster_labels[j])
    
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette_score = silhouette_score(features_reduced, cluster_labels, metric='euclidean')
davies_bouldin_score = davies_bouldin_score(features_reduced, cluster_labels)

print("Silhouette Score: ", silhouette_score)
print("Davies-Bouldin Index: ", davies_bouldin_score)





