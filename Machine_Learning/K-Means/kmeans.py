from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 아이리스 데이터 로드
iris = load_iris()
X = iris.data  # 특성 데이터

# k-means 모델 생성 및 학습
k = 3  # 클러스터 개수
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# 클러스터링 결과
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 결과 출력
print("Centroids:")
print(centroids)
print("\nLabels:")
print(labels)
