import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 감마 분포의 모수 설정
shape = 2.0
scale = 2.0
num_samples = 1000

# 감마 분포에서 샘플 생성
gamma_data = gamma.rvs(a=shape, scale=scale, size=num_samples)

# 감마 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(gamma_data, bins=30, density=True, alpha=0.6, color='b')

# 감마 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = gamma.pdf(x, a=shape, scale=scale)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Gamma Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
