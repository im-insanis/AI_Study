import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 이항 분포의 모수 설정
n = 10  # 시행 횟수
p = 0.5  # 성공 확률
num_samples = 1000

# 이항 분포에서 샘플 생성
binomial_data = np.random.binomial(n, p, num_samples)

# 이항 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(binomial_data, bins=30, density=True, alpha=0.6, color='b')

# 이항 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = binom.pmf(x, n, p)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Binomial Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
