import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 아이리스 데이터 로드
iris = load_iris()
X = iris.data  # 특성 데이터

# k 값 범위 설정
k_range = range(1, 11)

# 각 k 값에 대한 클러스터 내 SSE 계산
sse = []
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# 엘보우 그래프 그리기
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.show()
