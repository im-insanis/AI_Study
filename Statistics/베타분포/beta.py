import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 베타 분포의 모수 설정
a = 2.0
b = 5.0
num_samples = 1000

# 베타 분포에서 샘플 생성
beta_data = beta.rvs(a, b, size=num_samples)

# 베타 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(beta_data, bins=30, density=True, alpha=0.6, color='b')

# 베타 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = beta.pdf(x, a, b)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Beta Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
