#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder #문자를 숫자화

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv('iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
#sepal꽃받침  petal 꽃잎 species 종
# 그래프로 확인
#colormap = plt.summer()
#sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
#sns.pairplot(df, hue='species')
sns.pairplot(df, hue='species',palette="husl",vars=['sepal_length', 'petal_length']) #관계그래프 https://steadiness-193.tistory.com/198
# 빅데이터 시각화  hue : 범례구분기준  / palette:색감 / palette=https://lovelydiary.tistory.com/423
#hue='species' :다른 항목은 행렬을 만들수 없다
plt.show()

# 데이터 분류
#print(df.info())
dataset = df.values
#print(dataset)

X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

#print(df)
e = LabelEncoder()# 문자열을 숫자로 변환 객체
e.fit(Y_obj) # 문자열 피처는 일반적으로 카테고리형과 텍스트형 분류
print(Y_obj)
Y = e.transform(Y_obj) #수치데이터로 전환
print(Y)


Y_encoded = tf.keras.utils.to_categorical(Y)
print(Y_encoded)
# 모델의 설정
model = Sequential()
model.add(Dense(16,  input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))
