import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
# 28 countries list
countries = ['usa', 'australia', 'brazil', 'canada', 'china', 'france', 'germany', 'india','chile',
             'iran', 'italy', 'japan', 'mexico', 'pakistan','uae','singapore','switzerland', 'belarus',
             'russia', 'turkey', 'spain', 'peru', 'qatar','ireland','portugal','sweden','ecuador','belgium']

# Read tweets from files and store them in a list
data = []
for country in countries:
    with open(f'{country}.txt', 'r', encoding='utf-8') as f:
        tweets = f.readlines()
        data.extend(tweets)

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform data with vectorizer object
X = vectorizer.fit_transform(data)

# Apply LSA to reduce dimensionality
lsa = TruncatedSVD(n_components=4)
X = lsa.fit_transform(X)

# Find optimal K with elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plot elbow graph to visualize optimal K
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with optimal K
k = 4  # change this to optimal K obtained from elbow method
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)



# Apply LSA to reduce dimensionality
lsa = TruncatedSVD(n_components=2)
features_reduced = lsa.fit_transform(X)

# Calculate clustering performance metrics
silhouette_score = silhouette_score(features_reduced, y_kmeans, metric='euclidean')
davies_bouldin_score = davies_bouldin_score(features_reduced, y_kmeans)

# Print clustering performance metrics
print("Silhouette Score: ", silhouette_score)
print("Davies-Bouldin Index: ", davies_bouldin_score)




colors = ['r', 'g', 'b', 'y'] 
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = colors[i], label = f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of Tweets')
plt.xlabel('LSA Component 1')
plt.ylabel('LSA Component 2')
plt.legend()
plt.show()


# Print results
for i in range(k):
    print(f'Cluster {i+1}:')
    for j in np.where(y_kmeans==i)[0]:
        if j < len(countries):
            print(f'  {countries[j]}')
            