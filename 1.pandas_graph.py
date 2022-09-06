
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal,rand #normal : 정규(가우스) 분포에서 무작위 샘플

#matplotlib 구성
#matplotlib에서 그래프는 Figure 객체 내에 존재합니다. 따라서 그래프를 그리려면 다음 코드에서처럼 figure 함수를 사용해 Figure
# 객체를 생성해야 합니다.
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)  #빈 Figure 객체에 Axes 객체(또는 subplot)를 생성하려면 add_subplot 메서드를 사용하면 됩니다.
#type(fig)
#plt.show() #Figure 객체를 생성한 후 plt.show 함수를 호출하면   Figure 객체가 화면에 출력됩니다.
           #Axes 객체가 아직 포함되지 않으면  그래프를 그릴 수는 없습니다. 비어있는 객체 출력

'''
# =====================================
grp1 = pd.DataFrame(np.random.randn(20,7), columns=['X','Y','C','D','E','F','G'])
grp1.plot.scatter(x='X', y='Y')  #X,Y열만 선택
print(grp1)
plt.show()
'''
'''
# 히스토리그램
x = normal(size=200) #size : 갯수
print(x)
plt.hist(x, bins=10)     #bins : X축 막대 갯수
plt.show()

'''
'''
grp2 = pd.Series(rand(10))
print(grp2)
grp2.plot()
plt.show()

'''
'''
grp3 = pd.Series(np.random.randn(10)) #randn : 가우시안 난수
print(grp3)
grp3 = grp3.cumsum() # 각 시점 이전 데이터들의 합
print('====================')
print(grp3)
grp3.plot()
plt.show()

'''
'''
#분산 그래프
grp4 = pd.Series(np.random.randn(10))
grp4.plot.kde()  #
print(grp4)
plt.show()

# ======3. 데이터를 시각화하기 대소니
'''
'''
plt.plot([1,3,2,4]) # y축 : 1,3,2,4  x축 : 1,2,3,4(순서대로 자동부여)
plt.plot([1,3,2,7],[10,20,30,40])
#위 두 라인을 같이 표현 가능
# plt.grid(True)
plt.show() #그래프 보임
# ============================

'''

# 한글 폰트 지정
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.plot([10,20,30,40],[2,6,8,12], label='price')
plt.xlabel('X value')
plt.ylabel('Y 값')
plt.legend() #선에 라벨정보를 제공 해 주었다면 범례 표시
plt.title('X Y Graph')
plt.show()

'''
#===========================================
plt.plot([10,20,30,40],[2,6,8,12], label='price')
plt.axis([15, 35, 3, 10])  #X축 범위 :15부터 35  Y축 범위 :3부터 10 범위
plt.legend() #범례 표시
plt.title('X Y Graph')
plt.show()
#=========================================
'''
'''
import numpy as np
d = np.arange(0., 10., 0.4)  #0부터 10까지 0.4 간격으로 배열 생성
plt.plot(d,d*2,'r-', d,d*3,'y--')  #'-'는 실선, --는 점선 (x축, y축, 선스타일,x출,y축,선스타일)

#black 의 k
#red 의 r
#green 의 g
#blue 의 b
#yellow 의 y

plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('X Y Graph')
plt.show()
'''
'''
d = np.arange(0., 10., 0.4) #0부터 10 미만의 0.4 간격
print(d)
plt.figure('test') #'test' : figure 이름, 문자 또는 숫자

plt.subplot(211) #2 : 가로방향 2, 세로방향 1, 그리고 마지막 숫자는 몇번 째 위치??
plt.plot(d,d*2,'r-')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('Double Graph')

plt.subplot(212)  # 마지막 숫자를 상호 교환 해 볼것
plt.plot(d,d*-2,'b--')
plt.xlabel('X value')
plt.ylabel('Y value')

plt.show()

# =============================
'''
'''
plt.scatter(np.random.normal(5, 3, 1000), np.random.normal(3, 5, 1000)) #(x축,y축)
plt.show()
'''
'''

#막대그래프
'''
'''
s = pd.DataFrame([[2, 3], [3, 4]], columns=['a', 'b'],index=['F','M'])
s.plot(kind='bar')
plt.show()
print(s)
'''
'''
s = pd.DataFrame([[1789522, 2655864], [2852440, 4467147]], columns=['a', 'b'],index=['F','M'])
#s.plot(kind='bar')
s.plot(kind='bar', title="Year", rot=0, color='blue') #색깔 조견표 https://codetorial.net/matplotlib/set_color.html
#rot=0 X축 눈금값의 기울기
plt.show()
'''
'''
x = np.arange(3)
years = ['2017', '2018', '2019']
values = [100, 400, 900]

plt.bar(x, values, width=0.2, align='edge', color="springgreen",
        edgecolor="gray", linewidth=3, tick_label=years, log=True)
plt.show()
'''
#width: 막대의 너비입니다. 디폴트 값은 0.8이며, 0.6으로 설정
#align : 틱 (tick(X축))과 막대의 위치를 조절합니다. 디폴트 값은 ‘center’인데, ‘edge’로 설정하면 막대의 왼쪽 끝에 x_tick이 표시
#color : 막대의 색 지정.
#edgecolor : 막대의 테두리 색 지정.
#linewidth : 테두리의 두께 지정
#tick_label :  어레이 형태로 지정하면, 틱(X축)에 어레이의 문자열을 순서대로 나타낼 수 있다.
#log=True로 설정하면, y축이 로그 스케일로 표시
#plt.show()
#===================================================
#subplot
#plt.subplot(2,1,1)
#plt.subplot(2,1,2)
#plt.show()

# subplots의 예
'''
'''
fig, axes = plt.subplots(nrows=2, ncols=1) #plt.subplots(2,1)와 같음
print (fig,axes)
plt.show()
