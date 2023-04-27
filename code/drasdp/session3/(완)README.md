## 세션 3 과제

### 필수 과제
- (사전과제) EDA 마크다운으로 정리
- FIFA / WINE 과제파일 中 하나 골라서 전처리

### 선택 사항
- 실습 파일 클론코딩

***

### (세션4 사전과제) EDA 마크다운으로 정리

# Feature 분석 및 시각화
- 데이터 유형에 따라 다양하게 시각화
- 명목형자료, 자료개형, 시계열, 변수 간 관계 등에 따라 적합한 시각화 도구가 다름.

    ## Feature의 종류
    1. Categorical Feature(범주형):
        - 종류를 표시하는. 순서가 없는.
        - 예: 성별

    2. Ordinal Feature(순서형):
        - 순서를 가지는 범주형.
        - 예: 키(Height)에 대한 작음(short), 중간(medium), 큼(tall)

    3. Continuous Feature(연속형):
        - 연속적인 실수의 값을 가지는.
        - 예: 나이, 키의 실측값

    4. Discrete Feature(이산형):
        - 정수로 셀 수 있는.
        - 예: 책의 페이지

    ## 시각화 기본 표현
    - Figure: 그래프가 그려지는 바탕, 도화지
    - Figure를 그리고 plt.subplots로 바탕을 분할해 각 부분에 그래프를 그림.
    - plt.figure을 명시하는 것이 좋지만, 코드를 적지 않아도 자동으로 생성됨
    - figure: plt.gcf()
    - Size 조절: fig.set_size_inches(18.5, 10.5) 또는 plt.figure(figsize=(10,5)) 또는 plt.rcParams['figure.figsize'] = (10,7)
    - Axes: plot이 그려지는 공간, Axis : plot의 축

    ## 어떤 시각화를 사용해야 할까
    1. 데이터가 Numeric인지 Categorical인지 확인한다
    2. 빠진 데이터가 있는지 확인한다
    3. 속성들을 파악하고 그 성질에 따라 적절한 plot을 고민한다
        * plot : 둘 이상의 변수 간의 관계를 보여주는 그래프로 데이터 세트를 나타내는 그래픽 기술
        - 시간에 따른 : line, area, bar
        - 랭킹 : bar
        - 연관성 : scatter(산점도)
        - 분포 : box plot, histogram
        - 비율 : pie, bar

    ## Matplotlib
    - EDA의 대표적인 라이브러리
    - 파이썬과 Numpy를 활용한 시각화라이브러리

    ## Seaborn
    - matplotlib을 기반으로 만들어진, 통계데이터에 최적화된 시각화라이브러리

    ## Matplotlib vs Seaborn
    - Seaborn = Matplotlib + 알파


    ## 타이타닉 
    ```
    import seaborn as sns

    data = sns.load_dataset("titanic") # 데이터셋을 로드하는 함수. DataFrame 형태로 저장됨
    display(data.head())
    ```
    - 수치형 데이터: fare(연속형), age(이산형)
    - 범주형 데이터: sex, embarked, class 등
    

# 수치형 데이터 시각화   
   ## 히스토그램(Histplot)
   - 구간별 빈도수. 
   - ```sns.histplot(x='age',data = data) ```
    - age 열에 대한 도수분포를 그림
    - x축은 age의 범위를. y축은 해당 범위의 빈도수를.

   - ```sns.histplot(x='age', hue = 'alive', data=data)```
    - hue : 데이터를 추가적인 기준으로 구분하는 매개변수
    - alive 여부에 따라 색이 구분된다.
    - hue에는 일반적으로 카테고리 변수가 들어간다

    ## 커널밀도추정함수(kdeplot)
    - 히스토그램을 곡선으로 연결한
    - ```sns.kdeplot(x='age', data=data)```
        - AxesSubplot:xlabel='age', ylabel='Density'

    ## 막대그래프(barplot)
    - 범주형데이터에 따라 수치형데이터가 어떻게 달라지는지 파악할 때
    - 타이타닉 탑승자 등급별 운임을 barlot()으로 표현하면..
        - 범주형데이터(class)에 대한 수치형데이터(fare)의 변화
        - `sns.barplot(x='class',y='fare', data=data)`

    ## 포인트플롯(pointplot)
    - 막대그래프와 동일한 정보를 제공함. 모양이 다름.
    - `sns.pointplot(x='class', y='fare', data=data)`

    ## 박스플롯(boxplot)
    - 막대그래프 + Q1, Q2, Q3
    - `sns.boxplot(x='class', y='age', data=data)`

    ## 바이올린 플롯(violin plot)
    - 박스플롯 + 커널밀도추정함수
    - `sns.violinplot(x='class', y='age', hue='sex', data=data, split=True)`
        - hue : 성별에 대한 추가적인 구분.
        - split : 데이터를 합침.

    ## 카운트플롯(countplot)
    - `sns.countplot(x='class', data=data)`
    - 범주형 데이터에 대한 빈도

    ## 파이그래프(pie)
    - 코드
    ```python
    import matplotlib.pyplot as plt
    x=[10,60,30]
    labels=['A','B','C']
    plt.pie(x, labels=labels, autopct='%.1f%%')
    ``` 
    - autopct = '%.1f%%' : 소수점 첫째자리까지 비율 출력

***

# 데이터분석의 이해

- 데이터분석의 종류
    1. Optimization : 문제, 분석법 앎
    2. Solution : 문제 알지만, 분석법 모름
    3. Insight : 기존의 분석법은 알지만, 대상을 명확하게 알지 못 함
    4. Discovery : 분석대상과 분석법 모두 모름

- '분석의 대상'을 공부한다
- 통찰(Insight) -> 최적화(Optimization).
- 데이터 수집-> 시각화탐색 -> 패턴도출 -> 인사이트 발견

## EDA
- 탐색적데이터분석. 데이터의 특징과 구조적 관계를 파악함.
- 수집한 데이터를 다양한 각도에서 관찰, 이해.
- '통계' + '데이터시각화'를 통해서 

- binary classification 문제인 'Titanic'으로 공부한다.
    - EDA 순서
        1. 데이터 및 결측치 확인
        2. Feature 분석 및 시각화
            - 결측치 처리, 유형에 따른 시각화.
        3. Feature Engineering, Data Cleaning
            - insight 바탕을 새로운 feature 만들기
            - 머신러닝을 위한 클리닝
    
# 데이터 및 결측치 확인(step 1)
- 코드 : 
```py
# 필요한 라이브러리 불러오기
# 데이터 핸들링을 위한 라이브러리
import numpy as np
import pandas as pd

# 데이터 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')  # matplotlib style 설정

# 사진 파일 불러오기
!pip install IPython
from IPython.display import Image

# 문법 변경에 의한 경고를 무시
import warnings

warnings.filterwarnings('ignore')

# matplotlib 시각화 결과를 jupyter notebook에서 바로 볼 수 있게 해주는 command
%matplotlib inline

# 데이터 불러오기
data = pd.read_csv('./train.csv')

data.head()
data.info()

data.isnull().sum() #결측치 확인
```
- 데이터를 분석하기 전, 데이터에 결측치(null)가 있는지 확인하고 처리해야
- Age, Cabin, Embarked에 결측치가 존재하는데, feature를 살피면서 처리해보자
- 시각화를 통해 많은 데이터를 효과적으로 확인할 수 있음

```py
import matplotlib.pyplot as plt
import seaborn as sns
import warnins

warnings.filterwarnings('ignore')

x=[1,2,3] #리스트로 표현
y=[2,4,6]

fig, ax = plt.subplots(2,3,figsize=(15,10))
# 2x3의 subplot을 생성
# figsize는 전체 그림의 크기
# ax[0,0]부터 ax[1,2]까지의 6개 subplot을 생성

fig.suptitle('graphs')
# 서브플롯 전체에 대한 제목을 추가하는 메소드

#첫번째 그래프
ax[0,0].bar(x, y, label='label')
# x, y 데이터를 막대 그래프로 그린다. 범례에 label
ax[0,0].set_title('number1')
# 타이틀을 'number1'로 설정
ax[0,0].legend() #범례설정
ax[0,0].set(xlabel='x_data',ylabel='y_data')
# x축 라벨을 'x_data', y축 라벨을 'y_data'로 설정


ax[0,1].set_title('number2')
# ...
ax[1,2].set_title('number6')
```

```py
fig = plt.figure()
fig.suptitle('figure sample plots')
fig, ax_lst = plt.subplots(2,2,figsize=(8,5))

# 시각화 스타일
ax_lst[0][0].plot([1,2,3,4], 'ro-')
# 첫번째 subplot에 대해 [1,2,3,4]를 x축으로, ro- 스타일의 빨간색 선 그래프를 그립니다
ax_lst[0][1].plot(np.random.randn(4, 10), np.random.randn(4,10), 'bo--')
# 두번째 subplot에 대해 랜덤으로 생성한 4x10 크기의 배열을 x축과 y축으로 하여 bo-- 스타일의 파란색 점선 그래프를 그립니다
ax_lst[1][0].plot(np.linspace(0.0, 5.0), np.cos(2 * np.pi * np.linspace(0.0, 5.0)))
# 세번째 subplot에 대해 np.linspace(0.0, 5.0)를 x축으로, np.cos(2 * np.pi * np.linspace(0.0, 5.0))를 y축으로 하는 그래프를 그립니다
ax_lst[1][1].plot([3,5], [3,5], 'bo:')
# 네번째 subplot에 대해 [3,5]를 x축으로, [3,5]를 y축으로 하는 파란색 점 그래프를 그립니다. 'bo:' 스타일은 파란색 점과 점선을 의미합니다
ax_lst[1][1].plot([3,7], [5,4], 'kx')
# 네번째 subplot에 대해 [3,7]를 x축으로, [5,4]를 y축으로 하는 검은색 X 마크 그래프를 그립니다
plt.show()

#그래프 서식 설정하기
parameters = {
    'axes.titlesize': 25, #제목크기
    'axes.labelsize': 20, #라벨크기
    'ytick.labelsize': 20 #y축 눈금라벨크기
}
plt.rcParams.update(parameters)
```

# 피처별 시각화 및 결측값 채우기

## 생존자 수 (PieChart/CountPlot)
- 여러 그래프를 그리기 위한 matplotlib.pyplot.subplot(). `plt.subplot(row,column,index)`
- Pie Chart
    - explode : 부채꼴이 중심에서 벗어나는 정도
    - autopct : 부채꼴 안에 표시될 숫자형식
    - shadow : 그림자

- Count Plot 
    - 항목별 개수
    - 특정 column을 구성하는 value들을 구분하여 보여줌

```py
f, ax = plt.subplots(1, 2, figsize=(18, 8))
#1행 2열의 subplot을 생성하고, 이를 변수 f와 ax에 할당
# Pie Chart
data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%',ax=ax[0], shadow=True, fontsize=20)
# data['Survived'].value_counts() : 생존여부에 대한 값(0 또는 1)들의 개수를 계산.
# .plot.pie() 함수를 통해 파이차트를 생성함
    ## 'explode' : Survived 0은 중심에서 0%만큼, 1은 중심에서 10%만큼 벗어나게
    ## 'autopct' : 백분율을 소수점 앞 한자리 소수점 아래 한자리까지 %로 표시
    ## 'ax' : subplot의 위치
ax[0].set_title('Survived') #제목설정
ax[0].set_ylabel('')

#카운트플랏
sns.countplot('Survived', data=data, ax=ax[1])
#생존여부에 대한 값(0 또는 1)들의 개수를 countplot으로 시각화
ax[1].set_title('Survived')
plt.show()
```
## 생존자 성별(Bar/Countplot)
`data.groupby(['Sex', 'Survived'])['Survived'].count() #성별과 생존 여부에 따라 생존한 사람 수를 계산`
1. Bar Plot (가로형)
    - bar (세로형)
    - barh (가로형) ('h'orizontal, 수평의)
    <br>
2. Count Plot
    - x, y: 데이터의 변수명을 갖는 파라미터
    - hue: (optional) 색깔 인코딩을 위해 컬럼명을 갖는 파라미터
    - data: (optional) 그래프로 나타내기 위한 데이터셋

```py
f, ax = plt.subplots(1, 2, figsize=(18, 8))

# Bar Plot
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.barh(ax=ax[0])
## sex별 탑승객 수를 수평 막대그래프로
## .groupby(['sex']).mean() : 성별에 따른 생존비율
ax[0].set_title('Survived vs Sex')
## 그래프 위치설정, 0번째 row, 0번쨰 col

# Countplot
sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
## sex내에서 Survived 여부를 나눠서 plot
## sns.countplot('Sex') : 'Sex'카테고리의 빈도수를 막대 그래프로 표시
## 'hue': 추가적인 범주형 변수를 지정하여 그룹화 후 색상으로 구분하여 시각화
ax[1].set_title('Sex: Survived vs Dead')

plt.show()
```
## 상관관계와 인과관계
- 상관관계 : X Y 두 사건에 연관이 존재
- 인과관계 : A이면 B하다. 특정 사건이 다른 사건에 직접적으로 영향을 줌.
- 상관관계가 인과관계를 의미하지 않는다.

## Pclass: Ordinal Feature (Box/Bar/Count/FactorPlot)
1. Cross-tabulation
    - 범주형변수의 수치 확인에 편리
2. Box Plot
    - 분포 확인에 유용

1. 코드(크로스태블릿)
```py
#크로스태블릿
pd.crosstab(data.Pclass,
            data.Survived, margins=True).style.background_gradient(
                cmap='summer_r')
        # 'data.Pclass'와 'data.Survived' 간의 교차테이블
        # margins = True : All 항목도 함께 출력
```
2. 코드(Box Plot)
```py
plot = sns.catplot(x="Fare", y="Survived", row="Pclass",
                kind="box", orient="h", height=1.5, aspect=4,## box plot을 수평적으로
                data=data.query("Fare > 0")) ## 요금이 0초과인 데이터만
plot.set(xscale="log"); # x축의 scale을 로그 스케일로 설정
# sns.catplot() : 범주형 데이터를 시각화하는 데에 사용
    # x, y : x축, y축을 지정함.
    # row = "Pclass" : Pclass를 기준으로 각 row별로 데이터를 분할하여 보여줌
```
```py
f, ax = plt.subplots(1, 2, figsize=(18, 8))

# 가로형 막대 그래프: barh
data['Pclass'].value_counts().plot.barh(
    color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0]) ## Pclass별 탑승객 수, color 지정
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')

# 카운트 플랏
sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax[1])## countplot 함수를 이용하여 Pclass별 생존 여부(Survived)를 나눠 plot함
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()

# Sex와 Pclass 간의 교차테이블
pd.crosstab([data.Sex, data.Survived], data.Pclass,
            margins=True)
# 성별과 생존 여부에 따른 각 좌석 등급의 승객 수를 교차표(crosstab) 형태로
# crosstab 함수의 첫 번째 인자는 인덱스로 사용할 열을, 두 번째 인자는 컬럼으로 사용할 열을 지정
```
### pd.crosstab()의 매개변수
- index: 생성하려는 교차표의 행 인덱스를 지정합니다.
- columns: 생성하려는 교차표의 열 인덱스를 지정합니다.
- values: 데이터프레임에서 사용할 값(데이터)을 지정합니다.
- aggfunc: index, columns로 그룹화 된 values에 적용할 집계 함수를 지정합니다.
- rownames: 생성되는 교차표의 행 이름을 지정합니다.
- colnames: 생성되는 교차표의 열 이름을 지정합니다.
- margins: True로 설정하면 행과 열의 합계를 구한 결과를 마지막 행과 열에 추가합니다



# Factor Plot
```py
sns.factorplot('Pclass', 'Survived', hue='Sex', data=data) ## 'Pclass'에 따른 'Survived'의 분포를 그리되 hue = 'Sex'를 추가함으로써 또 하나의 구분을 만든다.
plt.show()
```

# Continuous Featue(Violin/FactorPlot/Histogram)
```py
print('Oldest Passenger was of:', data['Age'].max(), 'Years')
print('Youngest Passenger was of:', data['Age'].min(), 'Years')
print('Average Age on the ship:', data['Age'].mean(), 'Years')
## data['age']의 반환형 Series에 .max/min/mean() 연산을 취함
```

1. Violin Plot
- 카테고리별 분포를 확인하기 좋다
```py
sns.violinplot('Pclass', ## 행에 표시할
               'Age', ## 열에 표시할 데이터
               hue='Survived',## 새로운 범주구분을 색상으로써 추가
               data=data,
               split=True, ## hue에 맞춰서, 그래프가 겹치지 않게 분리해줌
               ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived') ## 제목
ax[0].set_yticks(range(0, 110, 10)) ## y축 눈금 값 설정
```

- (세션자료)
```
Age feature에는 177개의 결측치가 존재합니다.

이 결측치를 데이터 셋의 평균 나이로 대체할 수도 있지만, 평균 나이로 대체하기엔 사람들의 나이대가 매우 다양하다는 문제가 있습니다. 결측치를 처리할 다른 방법은 없을까요?

Name feature를 활용하는 방법이 있습니다!

이름에는 Mr 또는 Mrs와 같은 salutation이 존재합니다. 따라서 Mr와 Mrs 그룹 각각의 평균값을 활용해볼 수 있을 것 같습니다.

"What is in a Name?" -> new Feature
```

### Age 결측치 처리

- 'Inital' 추출하기
```py
data['Initial'] = 0 ## 열 초기화
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')  ## 정규표현식에 일치하는 부분 추출
    ## 이 정규표현식은 대문자나 소문자로 이루어진 연속된 문자열 다음에 마침표가 오는 부분.
```

- crosstab으로 'Inital'과 'Sex의 관계 파악
```py
pd.crosstab(data.Initial, data.Sex).T.style.background_gradient(
    cmap='summer_r') 
## 'Inital'과 'Sex'를 크로스탭하여 호칭과 성별 간 빈도수를 계산
## .T : 전치(Transpose)
```

- misspelled initials
```py
data['Initial'].replace([
    'Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col',
    'Rev', 'Capt', 'Sir', 'Don'
], [
    'Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other',
    'Other', 'Mr', 'Mr', 'Mr'
],
                        inplace=True)
```

- 'Initial'별 'Age'의 평균
```py
data.groupby('Initial')['Age'].mean()  ## 이니셜별 나이의 평균
```

- 'Age' 결측값 채우기
```py
data.loc[(data.Age.isnull()) & (data.Initial == 'Mr'), 'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Initial == 'Mrs'), 'Age'] = 36
data.loc[(data.Age.isnull()) & (data.Initial == 'Master'), 'Age'] = 5
data.loc[(data.Age.isnull()) & (data.Initial == 'Miss'), 'Age'] = 22
data.loc[(data.Age.isnull()) & (data.Initial == 'Other'), 'Age'] = 46
# 'Age'열이 null인 레코드에 접근하여 'Initial'열의 값을 가지고 'Age'에 평균값을 대입한다.
data.Age.isnull().any()
```
# Histogram
- (세션자료)
```
bins: 가로축 구간의 개수
edgecolor: 막대 테두리 색
color: 막대 색
```
- 코드
```py
f, ax = plt.subplots(1, 2, figsize=(20, 10))

# 히스토그램
data[data['Survived'] == 0].Age.plot.hist(ax=ax[0],
                                          bins=20, # 가로축 구간의 개수
                                          edgecolor='black', # 막대 테두리
                                          color='red') # 막대
## data['조건']인 DataFrame의 'Age'열에 접근하여 hist를 plot한다.
ax[0].set_title('Survived= 0')
x1 = list(range(0, 85, 5))
ax[0].set_xticks(x1) # 눈금 값을 0부터 85까지 5씩 증가하도록

# 히스토그램
data[data['Survived'] == 1].Age.plot.hist(ax=ax[1],
                                          color='green',
                                          bins=20,
                                          edgecolor='black')
ax[1].set_title('Survived= 1')
x2 = list(range(0, 85, 5))
ax[1].set_xticks(x2)
plt.show()
```

# Factor Plot
- 'Pclass'에 대한 'Survived'를 나타내되, 'Inital'을 기준으로 열을 분할함.
- 'Inital'로 구분되는 subplot이 만들어짐.
- 'Initail'별 'Pclass'별 'Survived'
```py
sns.factorplot('Pclass', 'Survived', col='Initial', data=data)## Initial별 pclass, survived plot
plt.show()
```

# Embarked(승선위치) : Categorical Value (Factor/CountPlot)
- cross tab : 범주형데이터의 개수를 열과 행으로 만듦.
    ```py
    pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived],
            margins=True).style.background_gradient(cmap='summer_r')
    # 'Embarked', 'Pclass'를 가지고 행을 구분하고, 'Sex', 'Survived'를 가지고 열을 구분한다
    # 위 네 기준을 가지고 만들어진 cross tab 표에, 해당하는 범주에 속하는 레코드의 개수가 저장됨.

- Factor Plot : 탑승위치(x축)에 따른 생존비율(y축)을 확인하기 위해 축을 지정하여 범주형데이터 표현
    ```py
    sns.factorplot('Embarked', 'Survived', data=data) ## 'Embarked'에 대한 'Survived'의 변화
    fig = plt.gcf() ## 현재 figure에 대해서
    fig.set_size_inches(5, 3) ## figure 사이즈 설정
    plt.show()
    ```

- Count Plot : 
    ```py
    #카운트플랏
    f, ax = plt.subplots(2, 2, figsize=(20, 15))

    sns.countplot('Embarked', data=data, ax=ax[0, 0])
    ax[0, 0].set_title('No. Of Passengers Boarded')
    # 'Embarked' 열에 대한 count를 표현

    sns.countplot('Embarked', hue='Sex', data=data, ax=ax[0, 1])
    ax[0, 1].set_title('Male-Female Split for Embarked')
    # 'Embarked' 열에 대한 count를 표현하되, 'Sex'에 따른 구분을 hue에 추가함
    
    sns.countplot('Embarked', hue='Survived', data=data, ax=ax[1, 0])
    ax[1, 0].set_title('Embarked vs Survived')
    # 'Embarked' 열에 대한 count를 표현하되, 'Survived'에 따른 구분을 hue에 추가함
    
    sns.countplot('Embarked', hue='Pclass', data=data, ax=ax[1, 1])
    ax[1, 1].set_title('Embarked vs Pclass')
    # 'Embarked' 열에 대한 count를 표현하되, 'Pclass'에 따른 구분을 hue에 추가함
    
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()
    ```

- Factor Plot : 'Pclass'에 대한 'Survived' 분포. hue와 col 구분을 추가한.
    ```py
    sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=data)
    plt.show()
    # 'Pclass'에 따른 'Survived'의 분포를 보여주되, 색상으로써 'Sex'를 구분하고, 'Embarked'에 따라 Subplot을 만듦으로써 구분한다. hue와 col 매개변수를 사용하여 추가적인 기준을 지정할 수 있다.
    ```

- Embarked(승선위치) 
    ```py
    data['Embarked'].fillna('S', inplace=True)
    data.Embarked.isnull().any()
    ```

# SibSp(동승한 형제 및 배우자 수): Discrete Feature (Bar/FactorPlot)
- 정의 
    - (세션자료)탑승한 형제 또는 배우자 수에 따른 생존 인원을 확인하고자 하였습니다.
    crosstab으로 정확한 수치를 확인하고, barblot으로 직관적으로 수치를 비교하고자 하였습니다.
    - SibSp = 함께 탑승한 형제 또는 배우자의 수
    Sibling = brother, sister, stepbrother, stepsister
    Spouse = husband, wife
    
- Crosstab '형제,배우자동승자수에 따른 생존여부 집계'
    ```py
    pd.crosstab([data.SibSp], data.Survived).style.background_gradient(cmap='summer_r')
    ```
- barplot '형제,배우자 동승자수에 따른 생존여부'
    ```py
    sns.barplot('SibSp', 'Survived', data=data)
    plt.title('SibSp vs Survived')
    ```
- Factor Plot : 형제, 배우자 동승자수에 대한 생존율
    ```py
    sns.factorplot('SibSp', 'Survived', data=data)
    plt.title('SibSp vs Survived')
    plt.show()
    ```
- Crosstab : 형제, 배우자 동승자수에 따른 Pclass를 Count
    ```py
    pd.crosstab(data.SibSp, data.Pclass).style.background_gradient(cmap='summer_r')
    ```

# Parch : 부모, 자식 동승자 수
- Crosstab : 부모, 자식 동승자 수에 따른 Pclass의 집계
    ```py
    pd.crosstab(data.Parch, data.Pclass).style.background_gradient(cmap='summer_r')
    # 범주형 변수 두 개(Parch, Pclass)를 주고 각각의 범주를 집계
    ```

- Barplot : 부모, 자식 동승자 수에 따른 생존율 분포
    ```py
    sns.barplot('Parch', 'Survived', data=data)
    plt.title('Parch vs Survived')
    plt.show()
    ```

- Factor plot : 부모, 자식 동승자 수에 따른 생존율 분포
    ```py
    sns.factorplot('Parch', 'Survived', data=data, ax=ax[1])
    plt.title('Parch vs Survived')
    plt.show()
    ```

# Fare(비용) : Continuous Feature (Cat/DistPlot)
- 연속형 변수인 비용에 대하여 Cat plot, Distplot를 적용할 수 있다.
    1. Cat Plot : '숫자형 + 하나 이상의 범주형'의 관계
    2. Dist Plot : hist + kdeplot. 분포밀도를 확인하기에 용이

- 사전 작업
    ```py 
    data['Fare'].max/min/mean()
    ```

- catPlot : 'Pclass'(범주형)에 대한 'Fare'(연속형)의 분포
    ```py
    sns.catplot(x='Pclass', y='Fare', hue='Sex', data=data)
    # 범주형 데이터에 대한 연속형 데이터의 분포 + 색깔로써 성별 구분.
    ```

- distplot : 'Fare'에 대한 분포
    ```py
    f, ax = plt.subplots(1, 3, figsize=(20, 8))

    # dist plot
    sns.distplot(data[data['Pclass'] == 1].Fare, ax=ax[0], color='r')
    # 'data['Plass'] == 1'을 만족하는 데이터프레임의 Fare 속성의 분포를 그림
    ax[0].set_title('Fares in Pclass 1')

    sns.distplot(data[data['Pclass'] == 2].Fare, ax=ax[1], color='y')
    ax[1].set_title('Fares in Pclass 2')

    sns.distplot(data[data['Pclass'] == 3].Fare, ax=ax[2], color='g')
    ax[2].set_title('Fares in Pclass 3')

    plt.show()
    ```
# Observations in a Nutshell for all features:
- Sex: 여자의 생존율이 남자보다 높습니다.
- Pclass: 돈이 많을수록 생존율이 높습니다.
- Age: 5-10살 아래의 아이들의 생존율이 높습니다. 15-35살의 탑승객들이 많이 죽었습니다.
- Embarked: Pclass 1의 탑승객이 대부분 Port S에 탑승하였음에도, Port C에 탑승한 사람들의 생존율이 더 높습니다. Port Q의 탑승객은 모두 Pclass 3이다.
- Parch+SibSp: 1-2명의 형제, 배우자 또는 1-3명의 부모ㆍ자녀와 탑승한 사람들의 생존율이 혼자 타거나 가족이 많은 사람들보다 생존율이 높습니다.

# Correlation Between The Features(heatmap)
- heatmap
    - 구체적인 수치 없이도 많은 데이터가 시사하는 바를 패턴으로 나타내는 데 매우 효과적인 시각화 차트로, 열분포 형태의 비쥬얼한 그래픽으로 출력합니다. annot:셀에 수치 표시 True/False
    - 변수들간 상관관계를 heatmap으로 나타내면,
    - 
        ```py 
        sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
        ## data.corr() : 열들 간 상관관계 계산
        ## annot = True : 셀에 수치를 표시
        ## cmap : heatmap 색상 스케일
        ## linewidths : 셀들 사이 경계선 두께

        fig = plt.gcf() ## figure 객체를 반환함
        fig.set_size_inches(10, 8)
        ## 그림 크기 조절

        plt.show()
        ```
    - Interpreting the Heatmap(세션자료)

        - string이 아닌 numeric feature에 대해서만 correlation 분석이 가능합니다.
        - 두 변수의 상관관계가 매우 높다는 것은 두 변수가 거의 동일한 정보를 갖고 있다는 것을 뜻합니다. 이를 다중공선성(Multicollinearity)이라고 합니다.
        - 모델을 학습시킬 때, 다중공선성을 띠는 변수들 중 불필요한 변수들은 제거해야 합니다.
        - 위 Heatmap에서는 그리 높은 상관관계가 존재하지 않는 것 같습니다. 가장 높은 상관계수는 SibSp과 Parch(0.41) 입니다. 따라서 모든 feature들을 계속 들고 가도 될 것 같습니다.

# Feature Engineering & Data Cleaning
- Feature Engineering : (세션자료)초기 데이터로부터 특징을 가공하고 생산하여 모델의 입력 데이터를 생성하는 과정을 말합니다. 예시로 위에서 Name feature을 활용하여 Initials feature을 얻은 것을 생각하시면 됩니다. 모델의 성능에 Feature Engineering이 미치는 영향은 매우 크기 때문에 EDA에서 굉장히 중요한 부분입니다.
    - 문자보다는 숫자, 개별값보다는 범위로 구분하도록 데이터를 가공하는 것이 좋습니다

## Age_band
- 30명의 사람에 대한 30개 개별값이 존재하는 등의 문제
- Bining 혹은 Normalization하여 범주형 값으로 바꾸는 것이 좋다

- Bining의 예

    1. 'Age_band' 생성 및 값 할당
    ```py
    data['Age_band'] = 0 ## 'Age_band'열을 추가하고 0으로 초기화
    data.loc[data['Age'] <= 16, 'Age_band'] = 0 ## 16보다 작거나 같은 경우, Age_band에 0을 대입
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_band'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_band'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age_band'] = 3
    data.loc[data['Age'] > 64, 'Age_band'] = 4
    # data.loc[(조건),'열 이름'] : 조건을 만족하는 레코드들을 선택하고 해당 레코드들의 'Age_band'값을 출력한다
    data.head(2)
    ```

    2. 'Age_band'에 대한 value_counts()
    ```py
    data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer') 
    # 'Age_band' 열에 접근하여 각 value(값)의 개수를 세고, to_frame()함.
    ```

    3. Factor Plot : 'Age_band'에 대한 'Survived'
    ```py
    sns.factorplot('Age_band', 'Survived', data=data, col='Pclass')
    # 'Age_band'에 따른 'Survived'의 분포를 그리되, Pclass로써의 열 구분을 추가하라
    plt.show()
    ```

## Family_size and Alone
- 'Parch'와 'SibSp' 속성을 활용하여
- "Family_Size"와 "Alone"이라는 새로운 feature을 만들어보고, 분석해보자
- 코드
    ```py
    data['Family_Size'] = 0 ## Family_Size 열을 생성하고 초기화
    data['Family_Size'] = data['Parch'] + data['SibSp']  #family size 계산
    data['Alone'] = 0 ## 혼자인지의 여부
    data.loc[data.Family_Size == 0, 'Alone'] = 1  #Alone

    sns.factorplot('Family_Size', 'Survived', data=data, ax=ax[0])
    # 가족 크기에 따른 생존율의 양상
    ax[0].set_title('Family_Size vs Survived')

    sns.factorplot('Alone', 'Survived', data=data, ax=ax[1])
    # 혼자인지의 여부에 따른 생존율의 양상
    ax[1].set_title('Alone vs Survived')
    
    sns.factorplot('Alone', 'Survived', data=data, hue='Sex', col='Pclass')
    # 혼자인지의 여부에 따른 생존율의 양상을 보여주되,
    # hue = 'sex' : 성별로써의 색 구분을 추가하고
    # col = 'Pclass' : 좌석등급으로써의 subplot 구분을 추가한다.

    plt.show()
    ```
    - alone 일 때 생존율이 매우 낮으며, family_size가 4 이상일 때도, 생존율은 감소합니다.
    - alone인 것은 Sex와 Pclass 관계없이 매우 치명적입니다.
    - 다만, 예외적으로 Pclass 3의 여자인 경우, 홀로 탑승할 때 더 높은 생존율은 보입니다.

## Fare_Range
    - 'Fare'은 연속형변수로, 순서형으로 변환하는 것이 좋다
    - pandas.qcut()

    ```py
    data['Fare_Range'] = pd.qcut(data['Fare'], 4)
    # data['Fare'] 열을 4등분 연산하여 레코드가 해당하는 등급을 data['Fare_Range']에 대입
    data.groupby(['Fare_Range'])['Survived'] # ['Fare_Range'] 그룹화, Survived에 접근하여 
    .mean().to_frame() # 평균을 구하고, DataFrame으로 변환함
    # 각각의 요금등급(Fare_Range)에 따른 생존율(평균연산으로 구한)을 DataFrame으로 표현함
    .style.background_gradient(cmap='summer_r') # 스타일
    ```
- Fare_Range가 증가할수록, 생존율이 높아집니다.
- Fare_Range 값을 Age_Band처럼 singleton 값으로 바꾸도록 하겠습니다.

    ```py
    data['Fare_cat'] = 0 # 'Fare_cat' 속성 생성
    data.loc[data['Fare'] <= 7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat'] = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_cat'] = 3
    # 등급 부여

    sns.factorplot('Fare_cat', 'Survived', data=data, hue='Sex')
    # 'Fare_cat'에 따른 'Survived'의 양상, 'Sex'로 추가 색 구분
    plt.show()
    ```
    - Fare_cat이 증가할수록, 생존율 역시 증가합니다.

# Converting String Values into Numeric
```py
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'],
                        [0, 1, 2, 3, 4],
                        inplace=True)
```
- 왜?
- -> string을 머신러닝 모델에 바로 사용할 수 없으므로 이를 numeric 값으로 변환해야 합니다.

# Dropping UnNeeded Features
```py
data.columns
data.drop(
    ['Name', # 범주형으로 변환 불가하므로 불필요한 속성
    'Age', # Age_band 속성이 있으므로 불필요
    'Ticket', # random string이므로 사용할 수 없음
    'Fare', # Fare_cat Feature가 있으므로 불필요
    'Cabin', # 결측치가 많고, 다수 탑승객이 다수 cabin값을 가지므로 불필요
    'Fare_Range', # Fare_cat feature 있으므로 불필요
    'PassengerId'], # 분류가 불가하므로 사용할 수 없음
    axis=1, # 세로방향으로 drop 수행
    inplace=True) # 원본에 적용
sns.heatmap(data.corr(), ## data의 feature에서 correlation heatmap
            annot=True, ## 수치를 보여라
            cmap='RdYlGn',
            linewidths=0.2,
            annot_kws={'size': 20})
fig = plt.gcf() # Get Current Figure. 현재 활성화된 Figure 객체를 반환함
fig.set_size_inches(18, 15) # 그래프의 크기 변경
plt.xticks(fontsize=14) # x축의 눈금 레이블(fontsize)을 14 크기로 설정
plt.yticks(fontsize=14) # y축의 눈금 레이블을 변경
plt.show()
```
- Positively related features: SibSp와 Family_Size, Parch와 Family_Size
- Negatively related features: Alone과 Family_Size

# 시각화 기본 보충

## Line Plot
- 연속적으로 변화하는 값을 순서대로 점으로 나타내고 선으로 연결
- 시간/순서 변화에 적합하여 추세를 알기에 용이
```py
df=pd.DataFrame(np.random.randn(10,4).cumsum(axis=0), columns=['A','B','C','D'], index=np.arange(0,100,10))
# 무작위로 생성된 정규분포 난수
# 0번 축을 기준으로 누적합 계산(.cumsum())
# 열 : A B C D
# 행 : 0부터 90까지 10씩 증가하는 인덱스
df.plot(marker='o', color=['r','b','g','y'])
# marker = 'o' : 동그라미
# color = 각 열의 색상
```

## IRIS dataset
```py
iris = sns.load_dataset("iris") #iris 데이터
iris.head()

plt.plot('petal_length',  # x축
         'petal_width',  # y축
         data=iris,
         linestyle='none', 
         marker='o', 
         markersize=6,
         color='purple', 
         alpha=0.5)
plt.title('Scatter Plot of Petal size', fontsize=20) # 산점도
plt.xlabel('petal_length', fontsize=14)
plt.ylabel('petal_width', fontsize=14)
plt.show()

sns.pairplot(iris) # 비교하고자 하는 변수가 2개 이상일 때
```
## 데이터 전처리 Scaling 시각화
```py
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


import numpy as np

fish_data= np.column_stack((fish_length, fish_weight))
fish_target= np.concatenate((np.ones(35), np.zeros(14)))
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

distances, indexes = kn.kneighbors([[25,150]])

import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')

# ...
```


