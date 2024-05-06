import numpy as np
import matplotlib.pyplot as plt

# 정규 분포의 평균과 표준편차 설정
mean = 0
std_dev = 1
num_samples = 1000

# 정규 분포 데이터 생성
normal_data = np.random.normal(mean, std_dev, num_samples)

# 정규 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='b')

# 정규 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = ((1 / (np.sqrt(2 * np.pi) * std_dev)) *
     np.exp(-0.5 * ((x - mean) / std_dev) ** 2))
plt.plot(x, p, 'k', linewidth=2)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
