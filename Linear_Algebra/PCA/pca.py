import numpy as np

def pca(X, k):
    # 데이터의 중심화
    X_meaned = X - np.mean(X, axis=0)
    
    # 공분산 행렬 계산
    cov_matrix = np.cov(X_meaned, rowvar=False)
    
    # 공분산 행렬의 고유값과 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 고유값을 내림차순으로 정렬하고, 해당하는 고유벡터들을 선택
    sorted_indices = np.argsort(eigenvalues)[::-1]
    topk_indices = sorted_indices[:k]
    topk_eigenvectors = eigenvectors[:, topk_indices]
    
    # 데이터를 주성분으로 변환
    X_pca = np.dot(X_meaned, topk_eigenvectors)
    
    return X_pca

# 예제 데이터 생성
data = np.random.rand(100, 3)  # 100개의 샘플과 3개의 특성을 가진 데이터

# 주성분 분석 수행 (3차원 데이터를 2차원으로 축소)
k = 2
data_pca = pca(data, k)

# 결과 출력
print("PCA 결과 ({}차원 → {}차원):".format(data.shape[1], k))
print(data_pca)
