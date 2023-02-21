import matplotlib.pyplot as plt
import numpy as np


# 1. 환경설정

# 마이너스 폰트 깨지는 문제 대처
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 전역 설정
plt.rcParams["font.family"] = 'NanumGothicCoding'#'NanumGothic'#'UnDotum'#'NanumBarunGothic' #UnDotum

#이미지 크기 조정
# 그림의 글자크기 조정 11pt
plt.rcParams["font.size"] = 12 
# 그림의 크기를 가로 7 세로 5
plt.rcParams["figure.figsize"] = (8,6) 

# -2pi과 2pi 사이를 100개의 점으로 나타냄
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x,y)



# 2. 기본 플롯

"""
타이틀 / X, Y축 이름

colors

b: blue; - g: green; - c: cyan; - m: magenta; - y: yellow; - k: black; - w: white
Line style -: solid; --: dashed; -.: dash-dot; : dotted

Symbol markers

.: point; ,: pixcel; o: circle; v: down trangle; ^ up-triangle; s: square
*: star; +: +, D: diamond
레전드 표시
그리드 표시
"""


plt.plot(x,y,'r:^', label="legend 표시")
plt.plot(x,0.5*y, color ='blue', marker='o', linestyle=':', label='legned2')

plt.title('sine graph')
plt.xlabel(r'$x$')
plt.ylabel(r'$sin(4 \pi x)$')

plt.grid(True) # 그리도 표시
plt.legend(loc='upper left')






# 3. Subplot
x = np.linspace(0,1,50)
y1 = np.cos(4*np.pi*x)
y2 = np.cos(4*np.pi*x)*np.exp(-2*x)

plt.subplot(2,1,1) # 2행1 열 idnum?
plt.plot(x,y1,'r-*', iw=1)
plt.grid(True)
plt.ylabel(r'$sin(4 \pi x)$')

plt.subplot(2,1,2)
plt.plot(x,y2,'bo',lw=1)
plt.grid(True)
plt.xlabel('x')
plt.ylabel(r'$ e^{-2x} sin(4\pi x) $')

plt.show()
plt.axis([0, 1,-1.5,1.5])



# 4. 다양한 plot
"""
plt.plot()
plt.boxplot()
plt.hist()
plt.scatter([x축 데이터], [y축 데이터])
plt.bar([x축 데이터], [y축 데이터])
plt.pie([비율 데이터])


"""
data = np.random.normal(0, 4, 100)
a = plt.boxplot(data, vert=False) # vert: 가로/세로 상자그림
plt.title('Basic Plot')
plt.show()


import seaborn as sns
iris = sns.load_dataset("iris")    # 붓꽃 데이터
#iris.columns
s1 = iris['sepal_length'].to_list()
s2 = iris['sepal_width'].to_list()
plt.boxplot([s1,s2]) # vert: 가로/세로 상자그림

plt.title('Iris data')
plt.xticks([1,2], ['sepal_length', 'sepal_width'])

plt.show()


plt.hist([s1,s2], label=['sepal_length', 'sepal_width'])



plt.scatter(s1, s2, color='r')
#plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.xlabel("sepal_width")


groups = iris.groupby('species')

fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.petal_length, 
            group.petal_width, 
            marker='o', 
            label=name)
ax.legend(fontsize=12, loc='upper left') # legend position
plt.title('Scatter Plot by matplotlib', fontsize=15)
plt.xlabel('Petal Length', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)


labels = ['G1', 'G2', 'G3', 'G4', 'G5']
means = [20, 34, 30, 35, 27]

x = np.arange(len(labels))  # the label locations
rec = plt.bar(x, means, 0.5) # 막대의 두께: 0.5
plt.xticks(x, labels)

labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

x = plt.pie(sizes, explode=explode, labels=labels, autopct='%d',
        shadow=True, startangle=90)
plt.show()        