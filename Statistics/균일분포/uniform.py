import numpy as np
import matplotlib.pyplot as plt

# 균일 분포의 최솟값과 최댓값 설정
low = 0
high = 1
num_samples = 1000

# 균일 분포 데이터 생성
uniform_data = np.random.uniform(low, high, num_samples)

# 균일 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(uniform_data, bins=30, density=True, alpha=0.6, color='b')
plt.axhline(y=1/(high-low), color='r', linestyle='-', linewidth=2)  # 이론적 밀도 함수
plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
