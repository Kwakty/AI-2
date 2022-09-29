#!/usr/bin/env python

# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# pandas 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 불러올 때 각 컬럼에 해당하는 이름을 지정합니다. 지정하지 않으면 첫줄이 이름이다.
df = pd.read_csv('pima-indians-diabetes.csv',
               names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])
# 처음 5줄을 봅니다.
#print(df.head(5))

# 데이터의 전반적인 정보를 확인해 봅니다.
#print(df.info())

#pd.set_option('display.max_row', None)
#pd.set_option('display.max_columns', None)  # 판다스 열을 무제한 확장
#pd.set_option('display.width', 1000) #디스플레이 폭 확장
# 각 정보별 특징을 좀더 자세히 출력합니다.
#print(df.describe())

# 데이터 중 임신 정보와 클래스 만을 출력해 봅니다.
#print(df[['pregnant', 'class']].groupby(['pregnant'])) # groupby 함수를 사용하면 DataFrameGroupBy  라는 객체가 생성됩니다.
pima=df.groupby(['pregnant'])
#pima=df[['pregnant', 'class']].groupby(['pregnant'])
#print(pima)
#print(pima.size()) #그룹별 갯수
#print(pima.count()) #각 항목별 갯수
#print(pima.mean())
#mean, median, min, max sum, prod, std, var(평균, 중앙값, 최소값,합,곱,표준편자,분산)
#first, las 첫번째, 마지막데이터
# 데이터 간의 상관관계를 그래프로 표현해 봅니다.

#colormap = plt.cm.gist_heat   #그래프의 색상 구성을 정합니다.
colormap = plt.summer() #spring,summer,autumn, winter   #그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12,12))   #그래프의 크기를 정합니다.

# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True) #.corr() 상관계수
plt.show()

grid = sns.FacetGrid(df, col='class') #FacetGrid에 데이터프레임과 구분할 row, col, hue 등을 전달해 객체 생성
# hue 인수에 카테고리 변수 이름을 지정하여 카테고리 값에 따라 색상을 다르게 할 수 있다.
grid.map(plt.hist, 'plasma',  bins=10) #plasma가 0괴 1에 미치는 영향
plt.show()

# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터를 불러 옵니다.
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset=df.values
'''
X = dataset[:,0:8]
Y = dataset[:,8]
'''
X = dataset[0:500,0:8]
Y = dataset[0:500,8]
X_eval = dataset[500:,0:8]
Y_eval = dataset[500:,8]
print(eval.shape)
# 모델을 설정합니다.
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', #binary_crossentropy 클래스가 두 개인 이진 분류 문제
             optimizer='adam',
             metrics=['accuracy']) #정확도, 1.XXX값은 1로 분류해버리는 분류모델 같은 경우에 사용하는 지표가 'accuracy'입니다.
                                   #소수점을 사용하는 회귀 모델 같은 경우는 accuracy를 사용할 수 없습니다.


# 모델을 실행합니다.
model.fit(X, Y, epochs=20, batch_size=10)

# 결과를 출력합니다.
print (model.evaluate(X_eval, Y_eval)) #과적합 확인  오차값, 정확도
