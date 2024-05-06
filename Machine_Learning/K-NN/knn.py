from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 아이리스 데이터 로드
iris = load_iris()
X = iris.data  # 특성 데이터
y = iris.target  # 타겟 데이터

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# k-NN 모델 생성 및 학습
k = 3  # 이웃의 개수
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 테스트 세트에 대한 예측
y_pred = knn.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
