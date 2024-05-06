import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 푸아송 분포의 파라미터 설정
mu = 3.0  # 평균 및 분산
num_samples = 1000

# 푸아송 분포에서 샘플 생성
poisson_data = np.random.poisson(mu, num_samples)

# 푸아송 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(poisson_data, bins=30, density=True, alpha=0.6, color='b')

# 푸아송 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = poisson.pmf(x, mu)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Poisson Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
