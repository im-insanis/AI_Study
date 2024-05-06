import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# 지수 분포의 모수 설정
scale = 1.0  # 스케일 파라미터 (역수로서 람다)
num_samples = 1000

# 지수 분포에서 샘플 생성
exponential_data = np.random.exponential(scale, num_samples)

# 지수 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(exponential_data, bins=30, density=True, alpha=0.6, color='b')

# 지수 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = expon.pdf(x, scale=scale)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
