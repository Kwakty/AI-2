from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv('sonar.csv', header=None)

# 데이터 개괄 보기
#print(df.info())

# 데이터의 일부분 미리 보기
#print(df.head())


pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#print(df)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]
Y_col=numpy.unique(Y_obj) #열 이름 중복 제거
print(Y_col)

# 문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
#print("Y===>",Y)

'''
X_train = X[0:146,:]
Y_train = Y[0:146]
X_test = X[146:,:]
Y_test = Y[146:,]
#print("X_test[145]==>",X_train)
#print("X_train[0]==>",X_test)
'''
'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
#순차적으로(shuffle=False) 분할,default는 True
#시계열 데이터와 같이 순서를 유지하는 것이 필요한 경우 shuffle=False
#random_state 데이터셋을 나눌 때 난수 이용, 동일한 결과를 얻기 위함
'''

# 모델 설정
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=5)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
#print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))  # 불러온 모델로 테스트 실행

print(X[114,0:60])
xx=X[114,0:60].reshape(1,60) #2차원으로 변환
print("xx==>",xx)
xx=xx.reshape(60) #1차원으로 변환
print("xx==>",xx)

print(model.predict(xx))
y=numpy.around(model.predict(xx)).astype(int)
print (y)
print(Y_col[y])