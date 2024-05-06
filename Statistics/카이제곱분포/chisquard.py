import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 카이 제곱 분포의 자유도 설정
df = 5  # 자유도 (degrees of freedom)
num_samples = 1000

# 카이 제곱 분포에서 샘플 생성
chi2_data = np.random.chisquare(df, num_samples)

# 카이 제곱 분포 데이터 시각화
plt.figure(figsize=(8, 6))
plt.hist(chi2_data, bins=30, density=True, alpha=0.6, color='b')

# 카이 제곱 분포의 이론적 밀도 함수 그리기
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = chi2.pdf(x, df)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Chi-squared Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
