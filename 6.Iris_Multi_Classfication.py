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
dataset = df.values #numpy 타입으로 변환
#print(dataset)
print(dataset.shape)
print(df.dtypes)  #판다스 파일 데이터타입
print(dataset.dtype)  #object는 문자열

X = dataset[:,0:4].astype(float)
print(X.dtype)
print(X)
Y_obj = dataset[:,4]

#print(df)
e = LabelEncoder()# 문자열을 숫자로 변환 객체
e.fit(Y_obj) # 문자열 피처는 일반적으로 카테고리형과 텍스트형 분류
print(Y_obj)
Y = e.transform(Y_obj) #수치데이터로 전환
print(Y)


Y_encoded = tf.keras.utils.to_categorical(Y)
#Y = e.fit_transform(Y_obj)  # fit + transform
print(Y_encoded)
# 모델의 설정
model = Sequential()
model.add(Dense(16,  input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', #데이터 결과 값이 0 또는 1 인 경우에는  binary_crossentropy
                                              # one hot encoding 인 경우: categorical_crossentropy
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))

eval = model.evaluate(X, Y_encoded, verbose=0)
print('정답률 = ', eval[1],'loss=', eval[0])

eval = model.evaluate(X, Y_encoded, verbose=0)
print('정답률 = ', eval[1],'loss=', eval[0])

xx=[ [6.1,3,4.6,1.4]]
print(model.predict(xx))
y=np.argmax(model.predict(xx)) #argmax:집합 X 안에서 최대값의 위치
print(y)
Y_col=np.unique(Y_obj) #열 이름 중복 제거
print(Y_col)
print(Y_col[y])