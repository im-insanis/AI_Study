import numpy as np

def svd(X, k):
    # SVD 수행
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # 상위 k개의 특잇값과 대응하는 왼쪽 특이벡터 선택
    Uk = U[:, :k]
    sk = np.diag(s[:k])
    Vk = Vt[:k, :]
    
    # 데이터를 주성분으로 변환
    X_svd = np.dot(Uk, np.dot(sk, Vk))
    
    return X_svd

# 예제 데이터 생성
data = np.random.rand(100, 3)  # 100개의 샘플과 3개의 특성을 가진 데이터

# 특잇값 분해 수행 (3차원 데이터를 2차원으로 축소)
k = 2
data_svd = svd(data, k)

# 결과 출력
print("SVD 결과 ({}차원 → {}차원):".format(data.shape[1], k))
print(data_svd)
