"""
Subject : 모델 학습 (수정)
Version : 2.2
Started : 2023-10-15
Updated : 2025-08-18
Language: Python
Supervised: Jihoon, Park
"""

import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import modules

# 커맨드라인 인자 처리
if len(sys.argv) < 2:
    print(
        """-------------------------------------------------
선수명 코드를 입력하여 주십시오. python3 model_learning_2.py {code}
-------------------------------------------------"""
    )
    sys.exit(1)

player_code = sys.argv[1]

# 데이터 경로 설정
Data_path = os.path.join(os.getcwd(), "Data")
Npy_path = os.path.join(Data_path, "npy")

# numpy 데이터 로드
x_train, y_train = modules.load_npy(Npy_path, player_code)
print(f"Counter({player_code}):", Counter(y_train))

# array 변환
x_train = np.array(x_train)
y_train = np.array(y_train)

# 10점 스코어 -> 3개 클래스 변환
def score_to_class(y_scores):
    y_classes = np.zeros_like(y_scores)
    y_classes[y_scores <= 6] = 0   # 낮은 점수
    y_classes[(y_scores >= 7) & (y_scores <= 8)] = 1  # 중간 점수
    y_classes[y_scores >= 9] = 2   # 높은 점수
    return y_classes

y_train_class = score_to_class(y_train)
num_classes = 3
y_train_onehot = to_categorical(y_train_class, num_classes=num_classes)

# 학습 & 테스트 자료 분리
x_train, x_test, y_train_onehot, y_test_onehot = train_test_split(
    x_train, y_train_onehot, test_size=0.3, random_state=42, shuffle=True
)

# 학습 가능 여부 확인
if y_train_onehot.shape[0] > 8:
    # 모델 학습
    model, history, test_loss, test_acc = modules.train_or_finetune_archery_model(
        player_code=player_code,
        x_train=x_train,
        y_train=y_train_onehot,
        x_test=x_test,
        y_test=y_test_onehot,
    )
    
    # 모델 저장
    os.makedirs("models", exist_ok=True)
    model_save_path = f"models/{player_code}_model.keras"
    model.save(model_save_path)
    
    print(f"[{player_code}] 선수 슈팅 최적동작 학습완료. 모델 저장: {model_save_path}")
else:
    print(f"[{player_code}] 선수의 슈팅자료가 부족합니다. {y_train_onehot.shape[0]}/1,000건")

