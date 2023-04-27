# 통계세션 필기

# 라이브러리
```py
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, linspace, mean
from scipy.stats import expon, zscore, norm, uniform
import math
%matplotlib inline
# 세션에서 소개된 통계 관련 라이브러리들들.
```

# 통계
- 표본(가지고 있는 데이터)을 분석하여 모집단(가지고 있지 않은 데이터)을 이해하는 활동
    1. 기술통계 : 가지고 있는 데이터를 설명, 요약함
    2. 추측통계 : 알고 있는 데이터 분석을 통해 모르는 데이터에 대한 예측과 추측

# 기술통계
1. 수치기술통계
    - 숫자로 데이터를 요약함
    - (① 중심 위치 척도, ② 변동성 척도, ③ 연관성 척도)
2. 표와 그래프 기술통계
    - 데이터 시각화
    - 도수분포표, 파이, 막대, hist, box, ...

# 수치기술통계
- 자료의 기본특성 : ① 중심위치 척도, ② 변동성 척도, ③ 연관성 척도

- Five Number Summary
    1. 최솟값(min)
    2. 1사분위수(Q1)
    3. 중앙값(median, Q2)
    4. 3사분위수(Q3)
    5. 최댓값(max)

1. 중심위치 척도
    - 평균(mean)
        - 모든 자료 더해 개수로 나눔
        - 이상치(outlier)와 측정오차(measurement error)에 민감함.
    - 중앙값(median)
        - 크기 순으로 놓았을 때 가운데의 값
            1. n이 홀수 -> (n+1)/2 번째의 값
            2. n이 짝수 -> 가운데 있는 두 개 값의 가운데 값
        - 평균에 비해 이상치(outlier)와 측정오차(measurement error)에 '덜 민감함'
    - 최빈값(mode)
        - 빈도수가 가장 많은 값

    ```py
    from statistics import mean, median ,mode 

    np.random.seed(0) # 난수생성 seed를 0으로
    x = (np.random.randint(20, size = 1000))
    # 0부터 19까지의 숫자를 무작위로 1000개 추출하여 x(배열)에 저장
    print("mean(x):", mean(x))
    print("median(x):", median(x))
    print("mode(x):", mode(x))
    ```
    - 해석
        - 평균, 중앙값, 최빈값을 통해 자료의 개형 (skewness 등) 확인 가능
        - 자료의 개형은 이상치 여부 판단, 변수 변환 판단 등에 활용됨
        - 자료의 꼬리가 긴 경우 자료의 개형이 두 개 이상의 봉우리를 가질 가능성이 커지게 됨

2. 변동성 척도
    - 사분위수(Quartile)
        - 표본을 4개의 동일한 부분으로 나눈 값
            - Q1 = 1+(n-1)x0.25
            - Q2 = 중앙값
            - Q3 = 1+(n-1)x0.75
            - (n은 총 도수)

        ```py
        from statistics import quantiles
        # statistics 모듈에서 quantiles()를 import함
        x=[1,2,3,4,5,6,7,8,9,10] #list
        print('method를 inclusive로 했을 때:', quantiles(x, method = 'inclusive'))
        print('method를 exclusive로 했을 때:', quantiles(x, method = 'exclusive'))

        x=np.array([1,2,3,4,5,6,7,8,9,10]) # numpy 배열
        print('numpy로 계산했을 때:', np.quantile(x, [0.25, 0.5, 0.75]))
        # numpy 내 quantile() 함수로 4분위수를 구하는 연산 구현

        #five number summary 
        q=quantiles(x, n=4, method='inclusive')
        print('다섯 범위를 확인할 때', min(x), q[0], q[1], q[2], max(x) )
        ```
        - 사분위수는 데이터를 4등분하는 기준값으로, 25%, 50%, 75%의 값을 의미합니다. inclusive 방법은 전체 데이터 개수가 홀수일 경우 중앙값을 포함하고, even일 경우 중앙값을 포함하지 않는 방법이며, exclusive 방법은 반대로 동작합니다.

        - numpy의 quantile 함수는 배열에서 지정된 분위수를 계산하여 반환합니다.

        - 다섯 번째 요약 통계는 데이터의 최솟값, 1사분위수, 중앙값, 3사분위수, 최댓값으로 구성된 통계입니다. 이를 계산하기 위해 quantiles 함수를 사용할 수 있습니다.

    - 사분위 간 범위(IQR)
        - Interquartile range. Q3 - Q1.
        - 범위가 같더라도 양끝의 변동성이 다를 수 있다.

        ```py
        x=np.array([1,2,3,4,5,6,7,8,9,10]) # numpy 배열 생성

        #x의 IQR
        q = np.quantile(x, [0.25, 0.5, 0.75]) # 사분위수를 구함
        q[2]-q[0] # IQR
        ```

    - 범위(range)
        - max - min.
        - 범위가 같더라도 변동성은 상이할 수 있다.

        ```py
        #x의 range 
        x=[1,2,3,4,5,6,7,8,9,10]
        max(x)-min(x) 
        ```
        
    - 분산(Variance)
        - 각각의 자료가 평균으로부터 떨어져 있는 정도.
        - (편차)^2의 기대값. 여기서 편차란, X - 표본평균.

    - 표준편차(standard deviation)
        - 분산의 제곱근

    - 변동 계수 (coefficient of variation)
        - 표준편차를 평균에 대한 상대적인 값으로 표현
        - 변동 계수(%) = 표준편차/평균 * 100
        - 평균에 대한 표준편차의 비율

    ```py
    #분산, 표준편차, 변동 계수
    from statistics import variance, stdev, mean

    x=[1,2,3,4,5,6,7,8,9,10]

    print('분산 :',variance(x)) # 분산
    print('표준편차 :',stdev(x)) # 표준편차
    print('변동 계수 :', (stdev(x) / mean(x))*100) #변동계수
    ```

3. 연관성 척도
    - 공분산(covariance)
    - 상관계수(correlation)

    - 기타 상관계수
        - 파이계수(phi coefficient)
        - 스피어만 상관계수(Spearman correlation)
        - 점-양분 상관계수(point-biserial correlation)

# 추측통계
- 모집단분포 추정
- 확률분포
    - 이산확률분포
        - 베르누이분포(Bernoulli Distribution)와 이항분포(Binomial Distribution)
        - 기하분포(Geometric Distribution)와 음이항분포(Negative Binomial Distribution)
        - 초기하분포(Hypergeometric Distribution)
        - 포아송분포(Poisson Distribution)
    - 연속확률분포
       - 정규분포(Normal Distribution)
       - T분포/F분포
       - 감마분포(Gamma Distribution)
       - 지수분포(Exponential Distribution)
       - 카이제곱분포(Chi-squared Distribution)
       - 베타분포(Beta Distribution)
       - 균일분포(Uniform Distribution)

    - 다음으로 볼 분포는 매우 중요하다
    - 정규분포(Normal Distribution)
        - 모든 정규분포는 종 모양을 나타내고 좌우 대칭이다.
        - 정규곡선의 '위치는 평균 μ(모집단평균) 에 의해서' 정해지고, 그 '모양은 표준편차 σ(모집단표준편차)에 의해서' 결정된다.
        - 일상생활에서 볼 수 있음(키, 공장에서 생산된 물건의 크기 등)
        
        ```py
        import scipy.stats as stats # norm 함수를 사용하기 위해 import

        x = np.linspace(-5, 5, 101) # -5부터 5까지 101개의 값을 가진 배열 x
        y1 = stats.norm(0, 1).pdf(x) # scipy.stats 라이브러리의 norm 함수를 사용하여 평균이 0이고 표준편차가 1인 정규분포 객체를 생성       
        # pdf() : probability density function을 계산하는 메소드.
        plt.figure(figsize=(6, 4))          # fig 사이즈 지정
        plt.plot(x, y1, color="red")         # 선을 빨강색으로 지정하여 plot 작성          
        plt.xlabel("x")                      # x축 레이블 지정
        plt.ylabel("y")                      # y축 레이블 지정
        plt.grid()                           # 플롯에 격자 보이기
        plt.title("Normal Distribution with scipy.stats") # 타이틀 표시
        plt.legend(["N(0, 1)"])              # 범례 표시
        plt.show()                           # 플롯 보이기
        ```

    - T분포: 표본평균의 분포
        - T분포는 표준정규분포와 같이, 0을 중심으로 종형의 모습을 가진 대칭 분포이다.
        - 꼬리가 표준정규분포보다 두껍다(fat tail).
        - 자유도 n에 따라 모습이 변하는데, 자유도 n이 커짐에 따라 표준정규분포 N(0,1)에 수렴한다.
        - T분포를 쉽게 이해하려면, 이 분포는 순전히 '평균 검정을 하기 위해 고안되었다는 점을 알고있어야' 한다
        - 추후 회귀분석에서 T-test를 이해하는 데 도움이 됨(독립변수 하나의 계수가 통계적으로 유의한지 판단)

        - 표준정규분포보다 T분포가 중심을 기준으로 더 퍼져있음(fat tail)을 볼 수 있습니다. 표준정규분포는 중심과 2정도만 차이나도 다르다는 결과를 주겠지만, T분포는 중심과 3이 차이난다고 하여도 같다는 결과를 줄 것입니다. 이와 같은 결과를 '보수적이다' 혹은 '보수적인 검정이다' 라고 표현합니다(outlier를 조금 더 허용)
        - 이러한 특성 때문에 주로 소표본에서 검정을 위해 정규분포를 대신하여 쓰입니다

        ```py
        fig = plt.figure(figsize=(4, 4)) # fig 정의

        t = np.linspace(-5, 5, 101)   # t 정의
        # np.linespace() : -5부터 5까지 총 101개의 점을 균등하게 찍어서 t라는 numpy 배열에 저장. t는 -5부터 5까지의 범위를 균등하게 나눈 101개의 숫자를 갖는 1차원 배열.

        y1 = stats.t(1).pdf(t)
        # stats.t(1) : 자유도가 1인 t-분포에서
        # .pdf(t) : 변수 t에 해당하는 확률밀도함수를 구하라.
        # y1은 확률밀도를 저장하는 'numpy 배열'.

        plt.plot(t, y1, color="skyblue", label = "t(1)")         
        plt.xlabel("t")                     
        plt.ylabel("y")                     
        plt.grid()                           
        plt.title("t Distribution with scipy.stats")     
            
        y1 = stats.t(3).pdf(t) # 자유도 3으로 변경             
        plt.plot(t, y1, color="blue", label = "t(3)")         
        plt.xlabel("t")                     
        plt.ylabel("y")                     
        plt.grid()                           
        plt.title("t Distribution with scipy.stats")     
        plt.legend(["t(3)"])                      

        y1 = stats.t(100).pdf(t) # 자유도 100으로 변경               
        plt.plot(t, y1, color="navy", label = "t(100)")         
        plt.xlabel("t")                     
        plt.ylabel("y")                     
        plt.grid()                           
        plt.title("t Distribution with scipy.stats")     
        plt.legend()      

        plt.show()
        ```

    - 카이스퀘어 분포 : 표본분산의 분포. 제곱합(sum of squares)의 분포
        - k개의 서로 독립적인 '표준정규 확률변수'를 각각 '제곱한 다음 합'해서 얻어지는 분포이다
        - 정규분포의 제곱은 카이스퀘어 분포이다
        - 음수에 분포가 없다
        - 비대칭(오른쪽으로 긴 꼬리)적인 분포모양을 가집니다. 모수(parameter, 매개변수)인 자유도에 따라 분포의 모양이 변하는데, 자유도가 커질수록 정규분포에 가까워집니다
        ```py
        fig = plt.figure(figsize=(12, 4))

        chi2 = np.linspace(0.5, 50, 100) # 0.5부터 50까지 100개의 구간으로 나누어 배열을 생성
        y1 = stats.chi2(1).pdf(chi2) # 자유도가 1인, 카이제곱분포의 확률밀도함수(pdf)를 구하라       

        ax1 = fig.add_subplot(1, 3, 1)        
        ax1 = plt.plot(chi2, y1, color="red", label=r'$\chi^2$(1)')          
        ax1 = plt.xlabel(r'$\chi^2$')            
        ax1 = plt.ylabel("y")                    
        ax1 = plt.grid()                        
        ax1 = plt.title(r'$\chi^2$ Distribution with scipy.stats')    
        ax1 = plt.legend()                       

        y1 = stats.chi2(10).pdf(chi2)        
        ax2 = fig.add_subplot(1, 3, 2)
        ax2 = plt.plot(chi2, y1, color="orange", label=r'$\chi^2$(10)')        
        ax2 = plt.xlabel(r'$\chi^2$')             
        ax2 = plt.ylabel("y")                     
        ax2 = plt.grid()                        
        ax2 = plt.title(r'$\chi^2$ Distribution with scipy.stats')    
        ax2 = plt.legend()                     

        y1 = stats.chi2(20).pdf(chi2)        
        ax3 = fig.add_subplot(1, 3, 3)
        ax3 = plt.plot(chi2, y1, color="brown", label=r'$\chi^2$(20)')        
        ax3 = plt.xlabel(r'$\chi^2$')             
        ax3 = plt.ylabel("y")                     
        ax3 = plt.grid()                        
        ax3 = plt.title(r'$\chi^2$ Distribution with scipy.stats')    
        ax3 = plt.legend()  

        plt.show()
        ```

    - F 분포 : 분산 비의 분포
        - 정규분포를 이루는 모집단에서 독립적으로 추출한 표본들의 분산비율이 나타내는 연속 확률 분포입니다.
        - 카이제곱분포와 마찬가지로 “분산”을 다룰 때 사용하는 분포인데, 카이제곱분포가 한 집단의 분산을 다뤘다면, F분포는 '두 집단의 분산'을 다룬다
        - 회귀분석의 F-test를 이해하는 데 필요한 분포(독립변수들의 변동과 종속변수의 변동을 확인해서 전체 회귀식이 유의한지 결정)
        - 분산분석을 이해하기 위해 필요한 분포(그룹 간의 평균 차이와 그룹 내의 변동성(분산)을 비교하여, 그룹 간 차이가 우연한 것인지 아니면 유의미한 것인지를 결정)
        ```py
        fig = plt.figure(figsize=(12, 4))

        x = np.linspace(0, 5, 201) # 0부터 5까지 201개의 구간으로 나눈 배열.
        nu1 = 4 ; nu2 = 50 #자유도가 2개 : 집단이 2개이므로.
        # 각 집단에서 구한 표본분산들의 비율 : F분포
        y1 = stats.f(nu1, nu2).pdf(x) # 자유도를 높이면 정규분포의 모양을 띔
        # 자유도가 nu1과 nu2인 F 분포의 확률밀도함수(PDF)를 계산
        # stats.f(nu1,nu2) : 자유도가 nu1, nu2인 F분포 생성
        # .pdf(x) : x배열에 대응하는 확률밀도함수 값을 계산함.

        ax1 = fig.add_subplot(1, 3, 1)   
        ax1 = plt.plot(x, y1, color="red", label='F(4, 50)')   
        ax1 = plt.xlabel('F')                    
        ax1 = plt.ylabel("y")                     
        ax1 = plt.grid()                          
        ax1 = plt.title('F Distribution without scipy')           
        ax1 = plt.legend()  

        nu1 = 50 ; nu2 = 50                 
        y1 = stats.f(nu1, nu2).pdf(x)       

        ax2 = fig.add_subplot(1, 3, 2)   
        ax2 = plt.plot(x, y1, color="orange", label='F(50, 50)')       
        ax2 = plt.xlabel('F')                  
        ax2 = plt.ylabel("y")                   
        ax2 = plt.grid()                          
        ax2 = plt.title('F Distribution without scipy')                         
        ax2 = plt.legend()  

        nu1 = 100 ; nu2 = 100                  
        y1 = stats.f(nu1, nu2).pdf(x)       

        ax3 = fig.add_subplot(1, 3, 3)   
        ax3 = plt.plot(x, y1, color="brown", label='F(100, 100)')         
        ax3 = plt.xlabel('F')                    
        ax3 = plt.ylabel("y")                     
        ax3 = plt.grid()                         
        ax3 = plt.title('F Distribution without scipy')          
        ax3 = plt.legend()  

        plt.show()
        ```

    - 이항분포(Binomial Distribution) + 베르누이분포(Bernoulli Distribution)
        - 이항 분포 (Binomial distribution)는 연속된 n번의 독립시행에서 각 시행이 성공할 (또는 일어날) 확률 p를 가질 때 만들어지는 이산 확률 분포입니다.
        - 예시 : 동전 던지기
        - 베르누이분포 : 시행이 1인 이항분포

        ```py
        from scipy.stats import binom
        p = .5
        plt.figure(figsize=(8, 4))

        for n in arange(1, 41, 4): # N = 1 일때 베르누이 분포
            x = arange(n + 1) # 0부터 n까지의 정수를 x에 저장
            plt.plot(x, binom(n, p).pmf(x), 'o--', label='(n=' + str(n) + ')')
            # binom(시도횟수, 성공확률) : 이하분포함수
            # pmf(x) : x에 대응하는 확률질량함수를 그린다(probability mass function)
            # n번 시도에서 성공할 확률이 p인 베르누이 분포에서 x번 성공할 확률

            
        plt.xlabel('X')
        plt.ylabel('P(X)')
        plt.title('Binomial Distribution(p = .5)')
        plt.grid()
        plt.legend()
        plt.show()
        ```

    - 포아송분포(Poisson Distribution)
        - 단위 시간 안에 사건이 몇 번 일어날 것인지
        ```py
        from scipy.stats import poisson

        plt.figure(figsize=(8, 4))

        for l in arange(4, 30, 4): # 4부터 30까지 4씩 증가하는 배열
            plt.plot(x, poisson(l).pmf(x), 'o--', label=r'$(\lambda =$' + str(l) + ')')
            # poisson(l).pmf(x) : 평균이 l인 포아송 분포에서 x의 확률을 계산
            
        plt.xlabel('X')
        plt.ylabel('P(X)')
        plt.title('Poisson Distribution')
        plt.grid()
        plt.legend()
        plt.show()
        ```

    
- 자유도
    - 데이터에서 독립적인 정보의 수
    - 일반적으로, 자유도 = 관측치 수(n) - 추정된 매개변수의 수(k)
    - 자유도가 높아질수록 정규분포의 모습을 띔

- 모집단과 표본집단

    ```py
    random.seed(10) # 난수 시드
    Height = list() # list 생성
    for i in range(100000) : #100000개의 데이터를 랜덤하게 생성
        Height.append(random.randrange(140, 200))
    mean = sum(Height) / len(Height)
    mean
    ```

    - 항상 모집단을 전부 조사하는 것은 불가하므로, N 개의 표본을 추출한다
        - 이렇게 추출된 N 명 -> 표본집단
        - 표본집단의 평균 -> 표본평균(표본통계랑)
    
    ```py
    random.seed(10)
    
    sample10_Height = random.sample(Height, 10) #표본 10개 추출
    # random.sample() : 'Height' 리스트에서 무작위로 10 개의 샘플을 추출하여
    # sample10_Height에 저장함
    # random.sample() : 리스트에서 중복되지 않는 무작위 샘플을 추출하는 함수
        # 첫 번째 인자는 샘플링할 리스트를, 두 번째 인자는 샘플링할 크기
    sample10_mean = sum(sample10_Height) / len(sample10_Height) # 표본평균
    print("표본 10개 :",sample10_mean)
    
    sample100_Height = random.sample(Height, 100) #표본 10개 추출
    sample100_mean = sum(sample100_Height) / len(sample100_Height)
    print("표본 100개 :",sample100_mean)
    
    sample1000_Height = random.sample(Height, 1000) #표본 10개 추출
    sample1000_mean = sum(sample1000_Height) / len(sample1000_Height)
    print("표본 1000개 :",sample1000_mean)
    
    sample10000_Height = random.sample(Height, 10000) #표본 10개 추출
    sample10000_mean = sum(sample10000_Height) / len(sample10000_Height)
    print("표본 10000개 :",sample10000_mean)
    
    print("모집단 :",mean)
    ```

- 대수의 법칙(라플라스정리, LLN : Law of Large Numbers)
    - 표본의 크기가 커질수록 모수와 비슷해짐.
    - 대수의 법칙 : 표본집단의 크기가 커지면 그 표본평균이 모평균에 가까워짐
        - 정확도가 올라간다 -> 데이터가 많을수록 좋다

- 중심극한정리(CLT : Central Limit Theorem)
    - 표본의 크기가 커질수록, 모집단의 분포 모양과 관계없이 표본평균의 분포가 정규분포에 가까워진다는 정리
    - 한번 추출 할 때 많은 표본을 추출할 수록 그 표본들의 평균의 분포가 정규분포에 가까워진다

    - 활용 : 복수의 검사 항목의 결과값을 합산하여, 이 값(표본평균들로 구성된 항)이 정규분포를 따른다는 것을 가정하고 품질 검사를 수행할 수 있다
        - (세션자료) 중심극한정리는 또한 데이터 분석과 머신 러닝에서도 중요합니다. 예를 들어, 회귀 분석에서 모델의 오차 분포를 정규 분포로 가정하는 것이 일반적이며, 이러한 가정은 중심극한정리의 성질을 활용합니다. 또한, 신경망과 같은 딥러닝 모델에서는 중심극한정리를 바탕으로 가중치 초기화 방법 등을 결정하며, 데이터의 정규화 등에도 중심극한정리의 개념이 적용됩니다.

- 가설검정
    - 가설검정의 절차
        1. 가설 설정
        2. 유의수준 설정
        3. 검정통계량 설정
        4. 기각/채택 판단

    1. 가설설정
        - 귀무가설 H0, 대립가설 H1을 설정한다
            - 귀무가설(H0) : 확인하기 용이하거나 기각하고자 하는 명제
                - 예: 'A반에서 새로운 교육법을 도입했더니 학생 성적이 올랐다'에 대하여, H0 : "Mean(기존) - Mean(새로운) = 0"을 세운다. H0이 참일 경우 새로운 교육법이 의미없음이 입증됨
            - 대립가설(H1) : "Mean(기존) - Mean(새로운) != 0"
            - 채택과 기각을 위해서는 판단 기준이 필요한데 그것을 '유의수준'이라고 한다.

    2. 유의수준 설정
        - 귀무가설의 오류를 판단하는 기준
        - 기호는 α를 쓰며, 1 - α 가 바로 신뢰 수준
        - α가 커지면 '대립가설이 채택될' 가능성이 높아짐.

    3. 검정통계량 산출
    - 검정통계량은 어떤 확률분포(정규분포, t분포, 이항분포 등)에서 가설 검정 목적으로 그 확률분포의 통계량을 산출하는 것을 말합니다. 이 검정통계량을 통해 기각 및 채택 여부를 결정하게 됩니다.
    - 데이터가 분포나 특징에 따라 Z통계량, T통계량 등 어떤 검정통계량을 사용할지 정해야 함
    
    
    4. 가설 채택/기각
    - 검정통계량이 신뢰구간에 위치해 있을 때는 대립가설 기각, 벗어날 시에는 대립가설을 채택하게 됨
    - 가설 검정에 쓰일 수 있는 또 다른 척도인 유의확률(p-value)이 있습니다. 이 유의확률은 검정통계량에서 유도할 수 있으며 유의 확률 p-value가 유의 수준(α)보다 클 시에는 귀무가설을 채택, 작을 시에는 귀무 가설을 기각함.

    ```py T검정
    # 예시 1 : T검정
    # sample A와 B의 평균이 차이가 있는지 검정하기
    from scipy.stats import ttest_ind

    sample_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample_B = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # ttest_ind는 두 집단의 평균에 통계적으로 유의미한 차이가 있는지 T검정하는 함수이며,
    # 귀무가설은 H0 :'두 집단의 평균이 같다'이다 
    t_statistic, p_value = ttest_ind(sample_A, sample_B, equal_var=True)

    # 유의수준 = 0.05로 정하기
    alpha = 0.05

    print("t-statistic:", t_statistic) # 2 thumb rule
    print("p-value:", p_value)

    print(f"유의수준 {alpha}로 가설검정한 결과 :", end = " ")
    if p_value < alpha : print("reject H0")
    elif p_value >= alpha : print("do not reject H0")
    ```

    ```py F검정
    # 예시2 : F검정
    # 각 그룹의 데이터의 평균이 차이가 있는지 검정하기
    from scipy import stats

    group1 = [23, 45, 56, 34, 23, 67, 34, 45]
    group2 = [56, 34, 78, 23, 12, 89, 45, 67]
    group3 = [12, 67, 34, 89, 23, 45, 56, 78]

    # F-검정 수행
    f_statistic, p_value = stats.f_oneway(group1, group2, group3)

    # f_oneway는 여러 집단의 평균이 통계적으로 유의미한 차이가 있는지 F검정하는 함수이며,
    # 귀무가설은 H0 :'집단1,2,3의 평균이 같다'이다 

    # 결과 출력
    print("F-statistic: {:.3f}".format(f_statistic))
    print("p-value: {:.3f}".format(p_value))

    # 유의수준 = 0.01로 정하기
    alpha = 0.01

    print(f"유의수준 {alpha}로 가설검정한 결과 :", end = " ")
    if p_value < alpha : print("reject H0")
    elif p_value >= alpha : print("do not reject H0")
    ```

# 1종, 2종 오류
- 유의수준을 정한다 : 1종오류, 2종오류를 어떻게 허용할 것인가
    1. 1종오류(α) : H0이 참인데 기각할 확률
    2. 2종오류(β) : H0이 거짓인데 채택할 확률     

- alpha : 채택 결정 시 실수할 확률
- p-value : 기각 결정 시 실수할 확률   