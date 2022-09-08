import numpy as np
import pandas as pd
import keras as k
import matplotlib.pyplot as plt

#from numpy.random import normal,rand
#한글사용
from matplotlib import font_manager, rc #폰트 사용 라이브러리

font_path = "C:/Windows/Fonts/NGULIM.TTF"  #폰트는 C:\Windows\Fonts에 위치합니다.
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#공부시간 X와 성적 Y의 리스트를 만듭니다.

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

plt.figure(figsize=(10,6))
plt.scatter(x,y,color='r')  #X,Y열만 선택
colors = np.array(["red","green","blue","yellow"])
#plt.scatter(x,y,color=colors)
#plt.xlabel('study time')
#plt.ylabel('score')
plt.xlabel('공부시간')
plt.ylabel('점수', rotation='horizontal') # ‘vertical’ , ‘horizontal’
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.axis([0, 12, 70,100])  #X축 범위 :0부터 12  Y축 범위 :70부터 100 범위

plt.scatter(x,y,color=colors,label='price') #라벨 표시
plt.legend() #선에 라벨정보를 제공 해 주었다면 범례 표시
plt.title('X Y Graph')
plt.savefig('./plot.png') #사진 저장, 사진 저장할 때는 절대로 plt.show()를 해서는 안된다

plt.show()

x_data = np.array(x)
y_data = np.array(y)

np.random.seed(3)
model = k.models.Sequential()
model.add(k.layers.Dense(1, input_dim=1))

sgd=k.optimizers.SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd) #SGD 확률적 경사하강법
#metrics=['accuracy']  선형회귀모델에서는 사용 불가
# 훈련
model.fit(x_data, y_data, epochs=2000)  # 100번 반복 훈련


# 테스트
plt.axis([0, 12, 70,100])  #x축 y축 범위
Y1=model.predict([[0]]).reshape(1)
Y2=model.predict([[2]]).reshape(1)
print(Y1,Y2)
plt.plot([0,12],[Y1,Y2])
plt.show()