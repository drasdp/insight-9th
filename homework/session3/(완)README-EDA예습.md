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
    