from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 보스턴 주택 가격 데이터셋 로드
boston = load_boston()
X = boston.data  # 특성 데이터
y = boston.target  # 타겟 데이터

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
lr = LinearRegression()
lr.fit(X_train, y_train)

# 테스트 세트에 대한 예측
y_pred = lr.predict(X_test)

# 평균 제곱 오차 계산
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
