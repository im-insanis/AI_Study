import numpy as np
import cv2

def save_transformed_image(image, transformation_name, transformed_image):

    filename = f"{transformation_name}.jpg"
    cv2.imwrite(filename, transformed_image)
    print(f"{filename} saved successfully!")

# 이미지 파일 읽어오기
image = cv2.imread('picture.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 선형 변환 행렬 정의
transformation_matrices = {
    'translation': np.float32([[1, 0, 50], [0, 1, 50]]),  # 이동
    'scaling': np.float32([[0.5, 0, 0], [0, 0.5, 0]]),     # 확대/축소
    'rotation': cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 45, 1),  # 회전
    'shearing': np.float32([[1, 0.5, 0], [0.5, 1, 0]]),   # 기울임
    'mirror': np.float32([[-1, 0, image.shape[1]], [0, 1, 0]]),  # 거울 이미지
    'coordinate_transform': np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # 좌표축 변환
}

# 선형 변환을 적용하고 이미지 저장
for transformation_name, transformation_matrix in transformation_matrices.items():
    transformed_image = cv2.warpAffine(gray_image, transformation_matrix, (gray_image.shape[1], gray_image.shape[0]))
    save_transformed_image(gray_image, transformation_name, transformed_image)
