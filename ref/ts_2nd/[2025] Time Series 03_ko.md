<a role="toc_link" id="chapter3"></a>
41\. 

# 3. 시계열 회귀 및 탐색적 데이터 분석(Time Series Regression and EDA)

이 장에서는 시계열 예제를 사용하여 회귀 기법을 검토합니다. 최소제곱 추정(least squares estimation)과 정보 기준(information criteria)을 통한 모델 선택에 대해 논의합니다. 그런 다음 시계열 데이터를 위한 탐색적 데이터 분석(exploratory data analysis) 및 평활화(smoothing)를 제시합니다.

## 3.1 시계열을 위한 일반 최소제곱(Ordinary Least Squares for Time Series)

먼저 t\=1,…,n에 대한 시계열 _xt_가 일련의 고정된 입력 계열 zt1,zt2,…,ztq에 의해 영향을 받을 수 있는 문제를 고려합니다. q\=3인 외생 변수(exogenous variables)를 가진 데이터 배치는 다음과 같습니다:

| 시간(Time) | 종속 변수(Dependent Variable) | 독립 변수(Independent Variables) |     |     |
| ---- | ------------------ | --------------------- | --- | --- |
| 1    | _x_1               | z11                   | z12 | z13 |
| 2    | _x_2               | z21                   | z22 | z23 |
| ⋮    | ⋮                  | ⋮                     | ⋮   | ⋮   |
| _n_  | _xn_               | zn1                   | zn2 | zn3 |

선형 회귀 모델을 통해 일반적인 관계를 다음과 같이 표현합니다.

xt\=β0+β1zt1+β2zt2+⋯+βqztq+wt,(3.1)

여기서 β0,β1,…,βq는 알려지지 않은 고정 회귀 계수(regression coefficients)이며, {wt}는 분산이 σw2인 백색 정규 잡음(white normal noise)입니다. 백색 잡음 가정은 일반적으로 위반되지만 완화될 수 있습니다. [섹션 5.4](#chapter5#sec5_4)를 참조하십시오.

예제 3.1 상품의 선형 추세 추정(Estimating the Linear Trend of a Commodity) [본문으로 돌아가기.⏎](chapter3)

상품(Commodities)은 실물 자산이며, 주식 및 유사 자산과는 다른 방식으로 경제 상황 변화에 반응하는 경향이 있습니다. 예를 들어, [그림 3.1](#chapter3#fig3_1)에 표시된 2003년 9월부터 2017년 6월까지의 노르웨이산 연어 1킬로그램당 월별 수출 가격을 고려해 보십시오. 이 계열에는 명백한 상승 추세가 있으며, 단순 선형 회귀를 사용하여 다음과 같은 모델을 피팅(fitting)함으로써 그 추세를 추정할 수 있습니다. 42\.

![Export price of Norwegian salmon and price of chicken with linear trend superimposed](./images/fig3_1.jpg)

그림 3.1: 2003년 9월부터 2017년 6월까지 노르웨이산 연어의 킬로그램당 월별 수출 가격(상단). 2001-2016년 파운드당 미국 센트 단위의 닭고기 가격(하단). 각 플롯은 피팅된 선형 추세와 함께 데이터를 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

xt\=β0+β1zt+wt,zt\=2003812,2003912,…,2017512.

이것은 q\=1인 회귀 모델 (3.1)의 형태입니다. 데이터 _xt_는 **salmon** 파일에 있고, _zt_는 time(**salmon**) 값들을 가지는 연-월(year-month)입니다. 오차 _wt_가 백색 잡음이라는 가정은 선 주위로 진동하는 움직임이 있기 때문에 아마도 사실이 아니겠지만, 지금은 사실이라고 가정하겠습니다. 자기 상관된 오차(autocorrelated errors)의 문제는 [섹션 5.4](#chapter5#sec5_4)에서 자세히 논의될 것입니다.

일반 최소제곱(OLS)에서는 오차 제곱합(error sum of squares)을 최소화합니다.

S\=∑t\=1nwt2\=∑t\=1n(xt−\[β0+β1zt\])2

이는 i\=0,1에 대한 _βi_와 관련이 있습니다. 이 경우, 간단한 미적분학을 사용하여 i\=0,1에 대해 ∂S/∂βi\=0을 계산하고, _β_ 값을 풀기 위한 두 개의 방정식을 얻을 수 있습니다. 계수의 OLS 추정값은 명시적이며 다음과 같이 주어집니다.

β^1\=∑t\=1n(xt−x¯)(zt−z¯)∑t\=1n(zt−z¯)2andβ^0\=x¯−β^1 z¯,

여기서 x¯\=∑txt/n과 z¯\=∑tzt/n은 각각 표본 평균입니다.

이 데이터에 대해 우리는 추정된 기울기 계수 β^1\=.25(표준 오차 .02)를 얻었으며, 이는 _연간_(여기서 시간 단위는 연(year)입니다) 약 25센트의 통계적으로 유의미한 증가 추정을 산출합니다. 43\. [그림 3.1](#chapter3#fig3_1)은 추정된 추세선이 겹쳐진 데이터를 보여줍니다. 이 그림은 또한 2001년 중반부터 2016년 중반까지(180개월) 미국 닭 한 마리의 가격(파운드당)인 **chicken** 데이터 세트에 대한 유사한 분석을 표시합니다. 두 상품 가격 간의 유사성에 주목해 주십시오. 이 예제의 코드는 다음과 같습니다:


`par(mfrow=2:1)`
`**trend**(**salmon**, lwd=2, results=TRUE, ci=FALSE)   _# graphic and results_`
`              Estimate      SE t.value p.value`
`  (Intercept) -503.09    34.44   -14.61        0`
`  time            0.25    0.02    14.76        0`
`  Noise SE estimated as: 0.88 on 164 df`
`**trend**(**chicken**, lwd=2, ci=FALSE)   _# graphic only_`

단순 선형 회귀는 상당히 직관적인 방식으로 다중 선형 회귀로 확장됩니다. 이전 예제와 마찬가지로 OLS 추정은 오차 제곱합을 최소화합니다.

S\=∑t\=1nwt2\=∑t\=1n(xt−\[β0+β1zt1+β2zt2+⋯+βqztq\])2,

이는 β0,β1,…,βq에 대한 것입니다. 이 최소화는 i\=0,1,…,q에 대해 ∂S/∂βi\=0을 푸는 것으로 달성할 수 있으며, 이는 q+1개의 미지수가 있는 q+1개의 방정식을 생성합니다. 이 방정식들은 일반적으로 다음과 같이 주어지는 _정규 방정식(normal equations)_이라고 불립니다.

∑t\=1n(xt−\[β^0+β^1zt1+β^2zt2+⋯+β^qztq\]) ztj\=0,j\=0,1,…,q,

여기서 zt0\=1이고, β^i는 _βi_의 추정치를 나타냅니다. _SSE_로 표시되는 최소화된 오차 제곱합은 다음과 같습니다.

SSE\=∑t\=1n(xt−x^t)2,(3.2)

여기서

x^t\=β^0+β^1zt1+β^2zt2+⋯+β^qztq.

(조정된) 총 제곱합(total sum of squares)은 다음과 같이 정의됩니다.

SST\=∑t\=1n(xt−x¯)2,

그리고 _결정 계수(coefficient of determination)_ 또는 _다중 상관 계수 제곱(squared multiple correlation)_을 사용하여 독립 변수가 설명하는 변동 비율을 측정할 수 있습니다.

R2\=SST−SSESST\=SSRSST\=∑t\=1n(x^t−x¯)2∑t\=1n(xt−x¯)2.(3.3)

(3.3)의 분자에 있는 SSR 항은 _회귀 제곱합(regression sum of squares)_이라고 불립니다.

44\. SST≥SSE이므로 R2∈\[0, 1\]은 단순히 표본 평균을 사용(또는 예측 변수를 사용하지 않음)하는 것과 비교하여, 데이터의 평균적인 동작을 설명하기 위해 독립 예측 변수(independent predictor variables)를 사용하는 것이 얼마나 더 나은지(오차를 줄이는 측면에서) 측정합니다. 불행하게도 종속 변수와 독립 변수 간의 선형 관계와는 아무런 관련이 없는 σw2에 반비례하기 때문에 _R_2의 값을 적합도 측정으로 사용하기는 어렵습니다. 예를 들어, xt\=t+wt 또는 xt\=t+3wt인 경우 _xt_와 _t_ 사이의 선형 관계는 두 경우 모두 동일하지만, 첫 번째 경우에 _R_2 값이 훨씬 더 클 것입니다:


`set.seed(1984)`
`t = 1:10; w = rnorm(10)`
`x = t + w`
`summary( lm(x~ t) )$r.sq   _# cor(x,t)^2 also works in this case_`
`  [1] 0.9073`
`x = t + 3*w`
`summary( lm(x~ t) )$r.sq`
`  [1] 0.5633`

_β_들의 일반 최소제곱 추정량(ordinary least squares estimators)은 편향되지 않으며(unbiased) 선형 비편향 추정량 클래스 내에서 분산이 가장 작습니다. 분산 σw2에 대한 비편향 추정량은 다음과 같습니다.

sw2\=MSE\=SSEn−(q+1),

여기서 _MSE_는 평균 제곱 오차(mean squared error)를 나타냅니다. _i_번째 계수 추정치의 추정된 표준 오차(standard error)는 다음과 같습니다.

se(β^i)\=VIFisw∑t(zti−z¯i)2,

여기서 z¯i\=∑tzti/n이고, 입력 zti에 대한 _분산 팽창 지수(variance inflation factor)_(VIF)는 다음과 같습니다.

VIFi\=(1−Ri2)−1,

여기서 Ri2는 zti와 다른 모든 독립 변수들의 다중 상관 계수 제곱입니다 [독립 변수가 하나만 있는 경우 se(β^1) 공식에서 그 VIF를 1로 설정할 수 있습니다]. VIF는 무한대까지 가능하므로, 독립 변수들 사이에 강한 다중공선성(collinearity)이 있는 경우 Ri2 중 일부는 1에 가까워지고 해당 VIFi, 그리고 결과적으로 그 표준 오차들도 매우 커지게 됩니다.

오차가 정규 분포를 따르기 때문에, se(β^i)가 i\=1,…,q에 대한 _βi_ 추정치의 추정된 표준 오차를 나타낸다면,

T\=(β^i−βi)se(β^i)

이것은 n−(q+1)의 자유도를 갖는 _t_\-분포를 가집니다. 이 결과는 흔히 개별 신뢰 구간 및 귀무가설 H0:βi\=0의 테스트에 사용됩니다.

예제 3.2 45\. 부엌 싱크대 경제학(Kitchen Sink Economics)\*

[Young and Pedregal (1999)](#bibref1#refbib_56)는 제2차 세계 대전 이후 미국의 실업률에 다양한 경제적 요인이 어떤 영향을 미쳤는지에 관심을 가졌습니다. 시계열 데이터는 econ5에 있으며 1948-III부터 1988-II까지 분기별 실업률, GNP, 소비(consumption), 정부 및 민간 투자(government and private investment)로 구성됩니다. 데이터는 해당 성장률과 함께 [그림 3.2](#chapter3#fig3_2)에 표시되어 있습니다.


`gecon5 = diff(log(**econ5**))`
`**tsplot**(cbind(**econ5**, gecon5), byrow=FALSE, ylab=colnames(**econ5**), ncol=2,`
`   col=2:6, lwd=2, title=c("Actual", rep(NA,4),"Growth Rate"))`

![Quarterly unemployment, GNP, consumption, and government and private investment from  in the USA from 1948-III to 1988-II](./images/fig3_2.jpg)

그림 3.2: 제2차 세계 대전 이후 40년 동안의 실업률 및 다양한 경제적 요인(실제 수치 및 성장률). [본문으로 돌아가기.⏎](chapter3)

처음에는 모든 요소를 무턱대고 회귀 분석에 넣고("부엌 싱크대 제외하고 모두 다(everything but the kitchen sink)"), 실업률이 상승하는 추세로 보이기 때문에 선형 추세 항을 포함해 보겠습니다(결과 일부 표시). lm 호출 46\. 에서의 점(dot)은 응답 변수를 제외한 데이터 프레임의 다른 모든 열을 포함한다는 것을 의미합니다.


`**ttable**( lm(**unemp**~ time(**unemp**) +   . , data=econ5), vif=TRUE)`
`  Coefficients:`
`              Estimate      SE    t.value   p.value       VIF`
`  (Intercept) 13.6122 0.7196      18.9166     0e+00`
`  time(unemp)    0.1427 0.0114    12.4618     0e+00   101.8556`
`  gnp           -0.0154 0.0017    -8.9391     0e+00   706.1012`
`  **consum         0.0169 0.0017     9.7853     0e+00   329.0045**`
`  govinv        -0.0096 0.0015    -6.3550     0e+00    16.9628`
`  prinv         -0.0078 0.0021    -3.7193     3e-04    34.4736`

"부엌 싱크대" 회귀에서 VIF는 엄청나게 크므로, 이 문제에 대해 조금 더 길게 생각하는 것이 좋겠습니다. 첫째, 예측 변수들이 거의 선형으로 증가하기 때문에 추세 성분인 time(**unemp**)과 비슷하게 보인다는 점에 유의해 주십시오. 소비의 계수는 양수이지만, 더 높은 소비는 더 낮은 실업률과 관련이 있기 때문에 음수여야 합니다. 계수 부호의 반전은 강한 다중공선성의 많은 해로운 영향 중 하나입니다. 추세 성분을 제거해도 문제는 해결되지 않습니다.

모든 계열이 시간이 지남에 따라 증가한다는 문제를 피하기 위해 성장률에 대해 회귀 분석을 실행할 수 있습니다(결과 일부 표시).


`**ttable**( lm(**unemp**~ . , data=gecon5), vif=TRUE)`
`  Coefficients:`
`              Estimate      SE t.value p.value       VIF`
`  (Intercept)    0.0398 0.0090   4.4164   0.0000`
`  gnp           -5.7132 0.9666 -5.9106    0.0000   3.5000`
`  **consum         0.5690 0.8550   0.6655   0.5067   1.4576**`
`  **govinv         0.3225 0.2505   1.2871   0.2000   1.0343**`
`  **prinv          0.0770 0.1654   0.4654   0.6423   2.8104**`

GNP에 대한 VIF는 여전히 다소 높지만, 흔히 VIF가 4 또는 5 미만이면 큰 문제가 되지 않는다고 조언합니다. 그러나, 그리고 이것은 큰 문제인데, 다른 모든 요인의 계수는 유의미하지 않으며 음수 대신 양수(투자와 소비가 높을수록 실업률이 낮아져야 함)입니다.

이에 대해 조금 더 생각해 보겠습니다. GNP는 여러 구성 요소로 이루어져 있으며 소비, 투자 및 정부 지출은 그 구성 요소 중 세 가지입니다(나머지는 순수출과 해외 소득입니다). 결과적으로 GNP는 이 연구의 다른 구성 요소와 높은 상관관계가 있어야 합니다. 우리가 할 수 있는 한 가지는 GNP에서 이러한 구성 요소를 부분화(partial out)한 다음 회귀 분석을 실행하는 것입니다.


`gnpp = resid( lm(**gnp**~ **consum** + **govinv** + **prinv**, data=gecon5) )`
`**ttable**(lm(**unemp**~ gnpp + **consum** + **govinv** + **prinv**, data=gecon5), vif=TRUE)`
`  Coefficients:`
`              Estimate      SE t.value p.value        VIF`
`  (Intercept)    0.0212 0.0084   2.5124   0.0130`
`  gnpp          -5.7132 0.9666 -5.9106    0.0000   1.0000`
`  consum        -2.0540 0.7308 -2.8108    0.0056   1.0649`
`  govinv         0.3532 0.2505   1.4102   0.1605   1.0338`
`  prinv         -0.6884 0.1029 -6.6894    0.0000   1.0879`
` Residual standard error: 0.07216 on 155 degrees of freedom`
` Multiple R-squared: 0.4062,     Adjusted R-squared: 0.3908`
` F-statistic: 26.5 on 4 and 155 DF, p-value: < 2.2e-16`

47\. 마침내 VIF가 양호해졌고 소비와 민간 투자가 유의미하며 올바른 방향을 가리킵니다. 하지만 정부 투자는 계수가 양수이고 이 기간 동안 유의미하지 않습니다. 정부 투자가 실업률을 낮출 수 있지만 실업률이 높으면 정부 지출이 증가할 수 있다는 피드백이 존재합니다.

다음 단계는 잔차 진단(residual diagnostics)을 수행하는 것입니다. 결과를 표시하지는 않지만 잔차는 백색 잡음으로 보입니다.


`res = resid( lm(**unemp**~ gnpp + **consum** + **govinv** + **prinv**, data=gecon5) )`
`**tsplot**(time(gecon5), res); **acf1**(res)`

경쟁하는 다양한 모델들은 흔히 최상의 독립 변수 하위 집합(best subset of independent variables)을 분리하거나 선택하는 데 관심이 있습니다. 제안된 모델이 r<q인 독립 변수의 하위 집합, 즉 zt,1:r\={zt1,zt2,…,ztr}만이 종속 변수 _xt_에 영향을 미친다고 지정한다고 가정해 보십시오. 이 경우 축소된 모델(reduced model)은 다음과 같습니다.

xt\=β0+β1zt1+⋯+βrztr+wt(3.4)

여기서 β1,β2,…,βr은 원래 _q_개 변수들의 계수 하위 집합입니다. 이 경우 귀무가설은 H0:βr+1\=⋯\=βq\=0입니다. _F_\-통계량을 사용하여 두 모델 하에서의 오차 제곱합을 비교함으로써 전체 모델 (3.1)에 대해 축소된 모델 (3.4)을 테스트할 수 있습니다.

F\=(SSEr−SSE)/(q−r)SSE/(n−q−1)\=MSRMSE,(3.5)

여기서 _SSEr_은 축소된 모델 (3.4) 하에서의 오차 제곱합입니다. (3.5)에서 분자는 평균 제곱 회귀(mean squared regression, MSR)라고 불리고 분모는 평균 제곱 오차(mean squared error, MSE)라고 불립니다. 축소된 모델이 더 적은 매개변수를 가지기 때문에 SSEr≥SSE임에 유의해 주십시오. 만약 H0:βr+1\=⋯\=βq\=0이 참이라면, 해당 _β_들의 추정치가 0에 가까울 것이므로 SSEr≈SSE가 됩니다. 따라서, _MSR_이 MSE에 비해 상대적으로 큰 경우 H0를 신뢰하지 않습니다. 귀무가설 하에서 (3.5)는 분자 자유도가 q−r이고 분모 자유도가 n−q−1인 중앙 _F_\-분포를 따릅니다. 이러한 결과는 종종 [표 3.1](#chapter3#tbl3_1)에 제시된 바와 같이 분산 분석(ANOVA) 표로 요약됩니다. 분자의 차이는 흔히 회귀 제곱합(_SSR_)이라고 불립니다. 만약 F\>Fn−q−1q−r(α), 즉 분자 q−r 및 분모 n−q−1 자유도를 갖는 _F_ 분포의 1−α 백분위수보다 크다면 수준 _α_에서 귀무가설이 기각됩니다.

__표 3.1: 회귀 분석에 대한 분산 분석(Analysis of Variance for Regression) [본문으로 돌아가기.⏎](chapter3)__
| 출처(Source) | df      | 제곱합(Sum of Squares) | 평균 제곱(Mean Square)      | _F_       |
| -------- | ------- | -------------- | ---------------- | --------- |
| zt,r+1:q | q−r     | SSR\=SSEr−SSE  | MSR\=SSR/(q−r)   | F\=MSRMSE |
| 오차(Error)    | n−(q+1) | _SSE_          | MSE\=SSE/(n−q−1) |           |

48\. 

### 모델 선택(Model Selection)

이전 단락에서 논의된 기술은 단계적(stepwise) 또는 모든 하위 집합(all subsets) 회귀를 통한 모델 선택에 사용할 수 있지만, 다른 접근 방식은 _파시모니(parsimony)_ (또는 _오컴의 면도날(Occam's razor)_이라고도 함)에 기반을 두어 최소한의 _복잡성(complexity)_으로 가장 _정확한(accurate)_ 모델을 찾으려고 노력하는 것입니다. 아마도 회귀 분석 과정에서 Mallows _Cp_를 통한 파시모니와 모델 선택에 대해 소개받은 적이 있을 것입니다.

_정확성(accuracy)_을 위해, 피팅된 값(x^t)이 실제 데이터(_xt_)에 얼마나 가까운지 측정하기 때문에 오차 제곱합 SSE\=∑t\=1n(xt−x^t)2을 사용할 수 있습니다. 특히 _k_개의 계수가 있는 정규 회귀 모델에 대해 분산의 최대 우도 추정량(maximum likelihood estimator)을 고려해 보십시오.

σ^k2\=SSE(k)n,(3.6)

여기서 SSE(k)는 _k_개의 회귀 계수가 있는 모델 하에서의 잔차 제곱합을 의미합니다. 모델의 _복잡성_은 모델의 매개변수 개수인 _k_로 특징지을 수 있습니다. [Akaike (1974)](#bibref1#refbib_1)는 피팅의 정확성과 모델의 매개변수 개수 간의 균형을 맞출 것을 제안했습니다.[1](#chapter3#fn3_1)

정의 3.3 _아카이케 정보 기준(Akaike's Information Criterion, AIC)_ [본문으로 돌아가기.⏎](chapter3)

AIC\=log σ^k2+n+2kn,(3.7)

_여기서_ σ^k2_는 (3.6)으로 주어지고 k는 모델의 매개변수 개수입니다._

따라서 파시모니 원칙에 부합하는 모델은 과도하게 복잡하지 않으면서도(작은 _k_) 정확한 모델(작은 오차 σ^k2)이 될 것입니다. 그러므로 고려된 모델 중에서 가장 작은 AIC를 산출하는 모델이 가장 좋은 모델로 지정됩니다.

(3.7)에 주어진 페널티 항에 대한 선택만이 유일한 것은 아니며, 다른 페널티 항을 옹호하는 상당한 양의 문헌이 있습니다. [Sugiura (1978)](#bibref1#refbib_47)가 제안하고 [Hurvich and Tsai (1989)](#bibref1#refbib_26)가 확장한 수정된 형태는 선형 회귀 모델에 대한 소표본(small-sample) 분포 결과에 기반할 수 있습니다. 수정된 형태는 다음과 같이 정의됩니다.

정의 3.4 _AIC, 편향 수정됨(AIC, Bias Corrected, AICc)_

AICc\=log σ^k2+n+kn−k−2,(3.8)

_[정의 3.3](#chapter3#defi3_3)과 동일한 표기법을 사용합니다._

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 1공식적으로 AIC는 −2logLk+2k로 정의되며, 여기서 _Lk_는 우도(likelihood)의 최댓값이고 _k_는 모델의 매개변수 개수입니다. 비교를 위해 BIC는 −2logLk+klogn으로 정의되므로 복잡성에 훨씬 더 큰 페널티가 부여됩니다. 정규 회귀 문제의 경우, AIC는 (3.7)로 주어진 형태로 축소될 수 있습니다. 최대 우도 추정은 [섹션 A.7](#appA#secA_7)에서 논의됩니다. [본문으로 돌아가기.⏎](#chapter3#fn3_13b)

49\. [Schwarz (1978)](#bibref1#refbib_40)와 같이 베이지안 인수를 기반으로 하는 페널티 항을 도출할 수도 있으며, 이는 다음으로 이어집니다.

정의 3.5 _베이지안 정보 기준(Bayesian Information Criterion, BIC)_

BIC\=log σ^k2+klognn,(3.9)

_[정의 3.3](#chapter3#defi3_3)과 동일한 표기법을 사용합니다._

BIC는 슈바르츠 정보 기준(Schwarz Information Criterion, SIC)이라고도 불립니다. 다양한 시뮬레이션 연구에서 대규모 표본에서는 BIC가 올바른 순서를 얻는 데 우수하게 작동하는 경향이 있는 반면, 매개변수의 상대적인 수가 많은 소규모 표본에서는 AICc가 우수한 경향이 있음을 확인하는 편입니다. 자세한 비교는 [McQuarrie and Tsai (1998)](#bibref1#refbib_34)를 참조하십시오.

예제 3.6 오염, 온도 및 사망률(Pollution, Temperature, and Mortality) [본문으로 돌아가기.⏎](chapter3)

[그림 3.3](#chapter3#fig3_3)에 표시된 데이터는 로스앤젤레스 카운티의 주간 사망률에 온도와 오염이 미치는 가능한 영향을 연구한 [Shumway et al. (1988)](#bibref1#refbib_43)에서 추출한 계열입니다. 겨울-여름 변동에 해당하는 모든 계열의 강한 계절적 요소(seasonal components)와 10년 동안 심혈관 질환 사망률의 하락 추세에 유의해 주십시오.

![Weekly cardiovascular mortality, temperature, and particulate pollution in LA county over a decade plotted separately](./images/fig3_3.jpg)

그림 3.3: 로스앤젤레스 카운티의 평균 주간 심혈관 질환 사망률(상단), 온도(중간) 및 미세먼지 오염(하단). 1970~1979년 10년 동안 매일의 값을 필터링하여 얻은 508개의 6일 평활 평균(smoothed averages)입니다. [본문으로 돌아가기.⏎](chapter3)

사망률과 온도 사이의 반비례 관계에 유의해 주십시오. 온도가 낮을수록 사망률이 더 높습니다. 또한 미세먼지 오염 수치가 높을 때 사망률이 증가하는 것으로 보입니다. 이러한 관계는 데이터가 함께 표시된 [그림 3.4](#chapter3#fig3_4)에서 더 잘 볼 수 있습니다. 시계열 플롯은 다음과 같이 생성되었습니다.


`_##-- Figure 3.3 --##_`
`par(mfrow=c(3,1), cex=.8)`

`**tsplot**(**cmort**, main="Cardiovascular Mortality", col=6, type="o", pch=19, ylab=NA)`
`**tsplot**(**tempr**, main="Temperature", col=4, type="o", pch=19, ylab=NA)`
`**tsplot**(**part**, main="Particulates", col=2, type="o", pch=19, ylab=NA)`
`_##-- Figure 3.4 --##_`
`**tsplot**(cbind(**cmort**, **tempr**, **part**), col=2:4, **spaghetti**=TRUE, **addLegend**=TRUE,`
`   legend=c("Mortality", "Temperature", "Particulates"))`

![Weekly cardiovascular mortality, temperature, and particulate pollution in LA county over a decade plotted together](./images/fig3_4.jpg)

그림 3.4: 동일한 플롯에 표시된 사망률 데이터. [본문으로 돌아가기.⏎](chapter3)

이러한 관계를 더 자세히 조사하기 위해 산점도 행렬(scatterplot matrix)이 [그림 3.5](#chapter3#fig3_5)에 표시되어 있으며, 심혈관 질환 사망률이 오염 미세먼지(pollutant particulates)와는 선형적으로 관련되어 있지만 온도와는 비선형적으로 관련되어 있음을 나타냅니다. 온도-사망률 곡선의 곡선 형태는 더 높은 온도뿐만 아니라 더 낮은 온도도 심혈관 사망률 증가와 연관되어 있음을 나타냅니다. [그림 3.5](#chapter3#fig3_5)에 표시된 산점도 행렬은 다음과 같이 생성되었습니다.


`**tspairs**(cbind(Mortality=**cmort**, Temperature=**tempr**, Particulates=**part**),`
`   hist=FALSE, col.diag=6)`

![Scatterplot matrix showing  relations between mortality, temperature, and particulate pollution in LA county](./images/fig3_5.jpg)

그림 3.5: 사망률, 온도 및 오염 간의 관계를 보여주는 산점도 행렬(Scatterplot matrix). 상관계수(correlations)는 오른쪽 상단 모서리에 표시되고 빨간색 선은 국소 가중 회귀(lowess) 피팅선입니다. [본문으로 돌아가기.⏎](chapter3)

50\. 온도와 미세먼지 오염이 거의 상관관계가 없다는 점이 중요합니다. 만약 이 두 독립 변수가 높은 상관관계를 가졌다면(즉, 다중공선성이 있다면), 사망률에 미치는 각각의 영향을 구별하기 어려웠을 것입니다. 그림의 대각선 외의 부분에서 표본 상관계수(sample correlations)가 우측 상단에 표시되며, 겹쳐진 선은 국소 가중 산점도 평활기(locally weighted scatterplot smoothers, lowess)로서 비선형성을 발견하는 데 도움을 줄 수 있습니다. 평활화(smoothing)에 대해서는 [섹션 3.3](#chapter3#sec3_3)에서 논의하겠지만, 지금은 lowess를 국소 회귀 피팅 방법으로 생각해 주십시오.

편의상 심혈관 사망률을 _Mt_, 온도를 _Tt_, 미세먼지 수치를 _Pt_로 표시하겠습니다. 산점도 행렬에 기반하여 _Tt_와 _Pt_ 모두 모델에 포함되어야 하는 것이 분명해 보이지만, 데모 목적으로 51\. 다음과 같은 4가지 모델을 고려해 보겠습니다.

Mt\=β0+β1t+wt(3.10)

Mt\=β0+β1t+β2Tt+wt(3.11)

Mt\=β0+β1t+β2Tt+β3Tt2+wt(3.12)

Mt\=β0+β1t+β2Tt+β3Tt2+β4Pt+wt(3.13)

(3.10)은 추세 전용 모델, (3.11)은 선형 온도 항 추가, (3.12)는 곡선형 온도 항 추가, (3.13)은 오염 항 추가 모델입니다. 다양한 모델 피팅에 대한 몇 가지 통계 요약이 [표 3.2](#chapter3#tbl3_2)에 제시되어 있습니다.

__표 3.2: 사망률 모델에 대한 요약 통계 [본문으로 돌아가기.⏎](chapter3)__
| 모델(Model)  | _k_ | SSE    | df  | MSE  | _R_2 | AIC  | BIC  |
| ------ | --- | ------ | --- | ---- | ---- | ---- | ---- |
| (3.10) | 2   | 40,020 | 506 | 79.0 | .21  | 5.38 | 5.40 |
| (3.11) | 3   | 31,413 | 505 | 62.2 | .38  | 5.14 | 5.17 |
| (3.12) | 4   | 27,985 | 504 | 55.5 | .45  | 5.03 | 5.07 |
| (3.13) | 5   | 20,508 | 503 | 40.8 | .60  | 4.72 | 4.77 |

각 모델은 바로 이전 모델보다 상당히 우수한 성과를 보이며, 온도, 온도 제곱 및 미세먼지를 포함하는 모델이 가장 좋은 결과를 나타내어, 변동성의 약 60%를 설명하고 AIC 및 BIC에서 가장 좋은 값을 가집니다(표본 크기가 크기 때문에 AIC와 AICc는 거의 동일합니다). 오차 제곱합과 (3.5)를 사용하여 임의의 두 모델을 비교할 수 있음에 유의하십시오. 52\. 따라서 추세만 있는 모델은 q\=4,r\=1,n\=508을 사용하여 전체 모델과 비교될 수 있으며 다음과 같습니다.

F3,503\=(40,020−20,508)/320,508/503\=160,

이는 F3,503(.001)\=5.51을 초과합니다.

가장 적합한 모델 (3.13)에 대한 출력 결과는 아래 코드에 나와 있습니다. 예상대로 시간이 지남에 따라 음의 추세가 존재하며 상당한 2차 온도 효과(극단적인 온도는 사망률 증가와 관련이 있음)도 존재합니다. 오염은 양의 가중치를 가지며 단위 미세먼지 오염 당 주간 사망자 수에 미치는 점진적인 기여도로 해석할 수 있습니다. 여전히 잔차 w^t\=Mt−M^t에서 자기상관(실제로 상당한 양의 자기상관이 존재함)이 있는지 확인하는 것이 필수적이지만, 상관 오차가 있는 회귀(regression with correlated errors)를 논의하는 [섹션 5.4](#chapter5#sec5_4)로 이 문제를 미루겠습니다.


`Z = cbind(trnd=time(**cmort**), **tempr**, **tempr**^2, **part**)`
`**ttable**( lm(**cmort**~ Z, na.action=NULL), vif=TRUE )`


`  Coefficients:`
`                   Estimate       SE t.value    p.value       VIF`
`  (Intercept)     2991.1402 199.4043 15.0004          0`
`  Ztrnd             -1.3959   0.1010 -13.8195         0     1.0110`
`  Ztempr            -3.8273   0.4236 -9.0357          0   181.2543`
`  Ztempr^2           0.0226   0.0028   7.9903         0   181.1552`
`  Zpart              0.2553   0.0189 13.5411          0     1.0133`
` `
`  Residual standard error: 6.385 on 503 degrees of freedom`
`  Multiple R-squared: 0.5954,      Adjusted R-squared: 0.5922`
`  F-statistic:    185 on 4 and 503 DF, p-value: < 2.2e-16`
`  AIC = 4.7217      AICc = 4.722     BIC = 4.7717`
`summary( aov(**cmort**~ Z) ) _# Table 3.1_`
`                Df Sum Sq Mean Sq F value Pr(>F)`
`  Z              4 30178     7545     185 <2e-16`
`  Residuals    503 20508       41`

53\. 큰 VIF 값에 대해서는 [예제 3.7](#chapter3#exam3_7)에서 다루겠습니다. [그림 3.4](#chapter3#fig3_4)에서 보면 사망률은 오염이 정점에 도달하고 몇 주 후에 최고치에 달하는 것처럼 보입니다. 이 경우 모델에 오염의 지연값(lagged value)을 포함하고 싶을 수 있습니다. 이 개념은 [문제 3.2](#chapter3#question3_2)에서 더 자세히 탐구됩니다.

예제 3.7 사망률과 환경(Mortality and the Environment) (계속) [본문으로 돌아가기.⏎](chapter3)

[Pozzer et al. (2023)](#bibref1#refbib_36)에 따르면 "2019년 비감염성 질병(non‐communicable diseases)으로 인한 전체 사망 중 약 20%가 환경적 위험 요인(주변 대기 오염, 실내 대기 오염, 납 및 라돈 노출, 기온 극값, 안전하지 않은 물, 위생 시설 및 손 씻기 부족 등 포함) 때문일 수 있다"고 합니다. [예제 3.6](#chapter3#exam3_6)에서는 [그림 3.5](#chapter3#fig3_5)의 산점도 행렬을 통해 기온과 미세먼지 오염이 사망률을 지표할 수 있으며 기온 극값이 부정적인 영향을 미친다는 것이 비교적 명백했습니다. 결과적으로, 모델 (3.13)이 고려된 4가지 모델 중에서 최고였다는 것은 놀라운 일이 아니었습니다. 하지만 해당 연구에는 다른 계열도 포함되어 있으며, lap(LA Pollution Study) 데이터 세트에 있습니다.

예를 들어, 일산화탄소(CO) 농도를 회귀에 포함하여 사망률을 예측하는 데 대한 기여도를 평가하고자 한다고 가정해 보겠습니다.


`**ttable**( lm(**cmort**~ Z + **co**, data=**lap**), vif=TRUE ) _# Z from previous example_`
`  Coefficients:`
`                Estimate      SE t.value p.value         VIF`
`  (Intercept) 2589.3679 232.2554 11.1488    0.0000`
`  Ztrnd          -1.1909  0.1179 -10.1027   0.0000    1.4039`
`  Ztempr         -3.8930  0.4200 -9.2696    0.0000  181.6664`
`  Ztempr^2        0.0231  0.0028   8.2499   0.0000  181.8050`
`  Zpart           0.1318  0.0420   3.1404   0.0018    5.1177`
`  co              0.5869  0.1786   3.2870   0.0011    5.7200`
` `
` Residual standard error: 6.324 on 502 degrees of freedom`
` Multiple R-squared: 0.6039,     Adjusted R-squared:    0.6`
` F-statistic: 153.1 on 5 and 502 DF, p-value: < 2.2e-16`
` AIC = 4.7044     AICc = 4.7047     BIC = 4.7627`

54\. 일산화탄소(CO)는 유의미하며 이것이 포함된 모델이 모든 정보 기준(information criteria)에서 없는 모델보다 선호되는 것을 볼 수 있습니다.


`AIC = 4.7044   AICc = 4.7047   BIC = 4.7627   _# with co_`
`AIC = 4.7217   AICc = 4.7220   BIC = 4.7717   _# without co_`

**tempr**과 **tempr**^2에 대한 VIF가 높은 이유는 이 온도 범위에서 둘이 서로 강하게 상관되어 있기 때문입니다. 온도를 먼저 중심화(centering)함으로써 이러한 큰 VIF 값을 제거하는 것이 가능하지만, 그렇게 하더라도 필수적인 2차 변수인 **tempr**^2의 결과는 변경되지 않습니다. 다른 큰 VIF들은 **part**와 **co**가 모두 여러 연료의 불완전 연소의 결과이기 때문입니다. 한 가지 옵션은 오염 요소들을 하나의 지표로 결합하는 것입니다. 미세먼지 물질은 증가된 일산화탄소 수치에 기여할 수 있으므로, 또 다른 옵션은 분석 전에 CO에서 미세먼지를 부분화(partial out)하는 것입니다:


`cop = resid(lm(**co**~ **part**, data=**lap**)) _# partial out particulates from co_`
`temp = **tempr** - mean(**tempr**)            _# center temperature_`
`Z = cbind(trnd=time(**cmort**), temp, temp^2, **part**, cop)`
`**ttable**( lm(**cmort**~ Z), vif=TRUE )`
`  Coefficients:`
`                Estimate      SE t.value p.value        VIF`
`  (Intercept) 2426.5270 232.8960 10.4189    0.0000`
`  Ztrnd          -1.1909  0.1179 -10.1027   0.0000   1.4039`
`  Ztemp          -0.4564  0.0317 -14.3968   0.0000   1.0349`
`  Ztemp^2         0.0231  0.0028   8.2499   0.0000   1.0190`
`  Zpart           0.2581  0.0187 13.8070    0.0000   1.0154`
`  Zcop            0.5869  0.1786   3.2870   0.0011   1.4291`
` `
` Residual standard error: 6.324 on 502 degrees of freedom`
` Multiple R-squared: 0.6039,     Adjusted R-squared:    0.6`
` F-statistic: 153.1 on 5 and 502 DF, p-value: < 2.2e-16`
` AIC = 4.7044     AICc = 4.7047     BIC = 4.7627`

출력의 하단에 있는 요약 통계는 이전과 동일하다는 것에 주목해 주십시오.

시계열 회귀 모델에 지연 변수(lagged variables)를 포함시키는 것은 약간의 주의를 요하며 가능합니다. 본 교재 전반에 걸쳐 이러한 유형의 문제를 계속해서 논의할 것이며, 다음 예제로 시작하겠습니다.

예제 3.8 지연 변수가 있는 회귀: 포식자-피식자(Regression with Lagged Variables: Predator–Prey) [본문으로 돌아가기.⏎](#chapter1#b1exam3_8)

[예제 1.5](#chapter1#exam1_5)에서는 스라소니(lynx)와 눈신토끼(snowshoe hare) 개체군 간의 포식자-피식자 관계에 대해 논의했습니다. 해당 예제에서 언급했듯이, 피식자(이 경우 토끼, _Ht_)와 포식자(이 경우 스라소니, _Lt_) 사이의 관계는 흔히 로트카-볼테라 방정식(Lotka–Volterra equations)으로 모델링되며 다음과 같습니다.

Ht+1\=αHt−βLtHtLt+1\=δLt+γLtHt,(3.14)

여기서 α\>1은 포식자가 없을 때 피식자의 성장률, 0<δ<1은 먹이원이 없을 때 포식자의 생존율, β\>0은 포식자의 소비율, γ\>0은 55\. 피식자 소비에 기인한 포식자 개체군의 성장률입니다. 해당 모델에서 생성된 데이터는 [그림 3.6](#chapter3#fig3_6)에 나와 있으며, [그림 1.6](#chapter1#fig1_6)에 표시된 실제 데이터와 유사함을 알 수 있습니다.

![Demonstration of the Lotka--Volterra equations describing the interaction between predator and prey](./images/fig3_6.jpg)

그림 3.6: (3.14)에 주어진 로트카-볼테라 방정식을 기반으로 한 포식자-피식자 행동의 예. [그림 1.6](#chapter1#fig1_6)과 비교해 보십시오. [본문으로 돌아가기.⏎](chapter3)

이제 회귀를 통해 로트카-볼테라 모델 (3.14)을 스라소니(Lynx) 데이터에 피팅하려고 한다고 가정해 보겠습니다. 불행히도 기본 R(base R)에서 지연 회귀(lagged regression)를 수행하는 것은 약간 어렵습니다. 회귀 분석을 실행하기 전에 계열(series)이 정렬되어야 하기 때문입니다. 그렇지 않으면 분석 결과가 정확하지 않게 됩니다. 데이터를 사전 처리(pre-process)하는 방법은 ts.intersect를 사용하여 지연 계열을 정렬하고 이를 데이터 프레임(data frame)으로 만드는 것입니다:


`prdpry = ts.intersect(L=**Lynx**, L1=lag(**Lynx**,-1), H1=lag(**Hare**,-1), dframe=TRUE)`
`fit    = lm(L~ L1 + L1:H1, data=prdpry, na.action=NULL)`
`**ttable**(fit)`
`  Coefficients:`
`              Estimate      SE t.value p.value`
`  (Intercept)   7.8498 2.1927    3.5799    6e-04`
`  L1            0.5563 0.0884    6.2932    0e+00`
`  L1:H1         0.0031 0.0009    3.5513    6e-04`
` `
`  Residual standard error: 11.35 on 87 degrees of freedom`
`  Multiple R-squared: 0.6502,      Adjusted R-squared: 0.6421`
`  F-statistic: 80.84 on 2 and 87 DF, p-value: < 2.2e-16`
`  AIC = 5.91306     AICc = 5.91616      BIC = 6.02416`
`_# residuals_`
`par(mfrow=1:2)`
`**tsplot**(resid(fit), col=4, main=NA)`
`**acf1**(resid(fit),   col=4, main=NA)`
`mtext("**Lynx** Residuals", outer=TRUE, line=-1.4, font=2)`

마지막으로 [그림 3.7](#chapter3#fig3_7)은 피팅 결과로부터의 잔차 및 해당 표본 자기상관함수(sample ACF)를 보여주며 잔차가 백색 잡음이 아님이 분명합니다. 실제로 잔차는 강하게 상관되어 있으며 분명한 10년 주기를 보여줍니다. 이 예제에서 명백하듯, 그리고 다른 예제에서도 볼 수 있듯이 고전적인(classical) 56\. 회귀는 종종 시계열의 흥미로운 역학을 모두 설명하기에 불충분합니다. 시계열을 분석할 때 종국적으로는 lm을 거의 사용하지 않기 때문에 이것은 사실 우리에게 좋은 소식입니다. 이 예제는 상관 오차가 있는 회귀에 대해 논의하는 [예제 5.16](#chapter5#exam5_16)에서 계속됩니다.

![Residual analysis of the fitted  predator   Lotka--Volterra equation for the lynx-hare data](./images/fig3_7.jpg)

그림 3.7: 스라소니(lynx) 데이터에 대해 피팅된 포식자 로트카-볼테라 방정식의 잔차 분석. [본문으로 돌아가기.⏎](chapter3)

## 3.2 탐색적 데이터 분석(Exploratory Data Analysis)

시계열의 경우 중요한 것은 계열 값들 간의 의존성을 측정하는 것이며, 적어도 자기상관을 정밀하게 추정할 수 있어야 합니다. 모든 관측치 쌍에 대해 자기상관이 다르다면 이를 측정하기 어려울 것입니다. 그러므로 시계열이 최소한 합리적인 시간 동안 [정의 2.13](#chapter2#defi2_13)에 명시된 정상성(stationarity) 조건을 만족하는 것이 중요합니다. 하지만 많은 경우 그렇지 않으며, 이 섹션에서는 비정상 데이터를 정상화(stationarity)로 변환하거나 평활화(smoothing)하는 몇 가지 방법에 대해 논의하겠습니다.

많은 예제들이 명백히 비정상적인 계열(nonstationary series)에서 비롯되었습니다. [그림 1.1](#chapter1#fig1_1)의 Johnson & Johnson 계열은 시간에 따라 기하급수적으로 증가하는 평균 함수(mean function)를 가지며 이 추세 주위의 변동 크기가 증가하면서 공분산(covariance function) 변화를 초래합니다. 예를 들어, 이 과정의 분산은 계열이 진행됨에 따라 분명히 증가합니다. 또한 [그림 1.4](#chapter1#fig1_4)에 표시된 전 지구 온도 계열은 시간에 따라 증가하지만 비선형적인 추세에 대한 명백한 증거를 포함하고 있습니다.

아마도 다루기 가장 쉬운 비정상성의 형태는 프로세스가 어떤 추세 주위에서 정상적인 동작을 갖는 _추세 정상(trend stationary)_ 모델일 것입니다. 우리는 이러한 유형의 모델을 다음과 같이 작성할 수 있습니다.

xt\=μt+yt(3.15)

여기서 _xt_는 관측치, _μt_는 추세, _yt_는 정상 과정(stationary process)입니다. 우리가 수많은 예제에서 볼 수 있듯이 흔히 강한 추세는 정상 과정 _yt_의 동작을 가리게 됩니다. 57\. 따라서 이러한 시계열의 탐색적 분석의 첫 번째 단계로 추세를 제거하는 것이 몇 가지 이점이 있습니다. 관련된 단계들은 추세 구성요소인 μ^t에 대한 합리적인 추정치를 구한 다음, 다음과 같은 이노베이션(innovations) 또는 잔차(residuals)를 다루는 것입니다.

y^t\=xt−μ^t.

예제 3.9 상품의 추세 제거(Detrending a Commodity) [본문으로 돌아가기.⏎](chapter3)

_xt_가 [예제 3.1](#chapter3#exam3_1)에 제시된 연어 가격 데이터를 나타낸다고 합시다. 여기서 우리는 모델이 (3.15)의 형태라고 가정합니다.

xt\=μt+yt,

[예제 3.1](#chapter3#exam3_1)에서 제안했듯이, 데이터의 추세를 제거하기 위해 직선 모델이 유용할 수 있습니다.

μt\=β0+β1 t,

여기서 _t_는 time(salmon>)의 시간 인덱스입니다. 그 예제에서 일반 최소제곱[2](#chapter3#fn3_2)을 사용하여 추세를 추정했고 다음을 얻었습니다.

μ^t\=−503+.25 t.

[그림 3.1](#chapter3#fig3_1)(상단)은 추정된 추세선이 겹쳐진 데이터를 보여줍니다. 추세가 제거된 계열(detrended series)을 구하려면 단순히 관측치 _xt_에서 μ^t를 빼서 다음과 같이 추세가 제거된 계열을 구합니다.

y^t\=xt+503−.25 t.

[그림 3.8](#chapter3#fig3_8)의 위쪽 그래프는 추세가 제거된 계열을 보여줍니다. [그림 3.9](#chapter3#fig3_9)는 추세가 제거된 데이터의 자기상관함수(ACF)를 보여줍니다(상단 패널).

![Detrended and differenced farm bred Norwegian salmon, export price, US Dollars per kilogram](./images/fig3_8.jpg)

그림 3.8: 추세가 제거된(detrended, 상단) 및 차분된(differenced, 하단) 연어 가격 계열. 원본 데이터는 [그림 3.1](#chapter3#fig3_1)에 나와 있습니다. [본문으로 돌아가기.⏎](chapter3)

![Sample ACFs of the detrended   and  the  differenced  salmon price series](./images/fig3_9.jpg)

그림 3.9: 추세 제거(상단) 및 차분된(하단) 연어 가격 계열의 표본 ACF. [본문으로 돌아가기.⏎](chapter3)

[예제 1.10](#chapter1#exam1_10)에서 무작위 보행(random walk)도 추세에 대한 좋은 모델이 될 수 있음을 확인했습니다. 즉, ([예제 3.9](#chapter3#exam3_9)에서처럼) 고정된 것으로 추세를 모델링하는 대신에, 다음과 같은 표류가 있는 무작위 보행(random walk with drift) 모델을 사용하여 확률적(stochastic) 구성요소로 추세를 모델링할 수도 있습니다.

μt\=δ+μt−1+wt

여기서 _wt_는 백색 잡음(white noise)이고 _yt_와는 독립적입니다. 만약 적절한 모델이 (3.15)라면 데이터 _xt_를 _차분(differencing)_하면 정상 과정을 얻습니다. 즉,

xt−xt−1\=(μt+yt)−(μt−1+yt−1)\=δ+wt+yt−yt−1.(3.16)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 2 _yt_가 백색 잡음이 아니기 때문에, 독자는 일반화 최소제곱(generalized least squares)을 사용해야 한다고 느낄 수 있습니다. 하지만 우리는 _yt_의 동작을 알지 못하며, 이것이 현 단계에서 우리가 평가하려고 시도하는 것입니다. [Grenander and Rosenblatt (2008, ch. 7)](#bibref1#refbib_21)의 주목할 만한 결과는 다항식 회귀나 주기적 회귀의 경우, _yt_에 대한 온건한 조건 하에 표본이 클 때 일반 최소제곱이 효율성 측면에서 일반화 최소제곱과 동등하다는 것입니다. [본문으로 돌아가기.⏎](#chapter3#fn3_23b)

58\. 특성 2.7(Property 2.7)을 사용하면 zt\=yt−yt−1이 정상(stationary)이라는 것을 보여주기 쉽습니다. _yt_가 정상이므로,

γz(h)\=cov(zt+h,zt)\=cov(yt+h−yt+h−1,yt−yt−1)\=2γy(h)−γy(h+1)−γy(h−1)(3.17)

시간에 구애받지 않습니다. (3.16)의 xt−xt−1이 정상이라는 것을 보이는 것은 연습 문제([문제 3.5](#chapter3#question3_5))로 남겨둡니다.

추세를 제거하기 위해 추세 제거(detrending)보다 차분(differencing)을 사용하는 한 가지 이점은 차분 연산에서는 매개변수(parameters)가 추정되지 않는다는 것입니다. 그러나 한 가지 단점은 (3.16)에서 볼 수 있듯이 차분은 정상 과정 _yt_에 대한 추정치를 제공하지 않는다는 것입니다. 만약 _yt_에 대한 추정치가 필수적이라면 추세 제거가 더 적절할 수 있습니다. 예를 들어, 우리가 상품의 경기 순환(business cycle)에 관심이 있다면 그러할 것입니다. 연어 가격은 약 3~4년의 경기 순환을 가지고 있는 것으로 보이며, 이는 Kitchin 순환[(Kitchin, 1923)](#bibref1#refbib_32)으로 알려져 있고 많은 상품 계열에서 나타납니다.

데이터를 강제로 정상화하는 것이 목표라면 차분이 더 적절할 수 있습니다. 차분은 또한 [예제 3.9](#chapter3#exam3_9)와 같이 추세가 고정되어 있는 경우 실현 가능한 도구입니다. 예를 들어, 모델 (3.15)에서 μt\=β0+β1 t라면, 데이터를 차분하여 정상성을 도출할 수 있습니다([문제 3.4](#chapter3#question3_4) 참조).

xt−xt−1\=(μt+yt)−(μt−1+yt−1)\=β1+yt−yt−1.

차분은 시계열 분석에서 중심적인 역할을 수행하기 때문에, 이를 위한 고유한 표기법이 부여됩니다. 첫 번째 차분(first difference)은 다음과 같이 표기합니다.

∇xt\=xt−xt−1.(3.18)

59\. 살펴본 바와 같이 1차 차분은 선형 추세를 제거합니다. 2차 차분(second difference), 즉 (3.18)의 차분은 2차(quadratic) 추세를 제거할 수 있으며, 등등. 더 높은 차분의 정의를 위해 흔히 사용할 표기법의 변형이 필요합니다.

정의 3.10. [본문으로 돌아가기.⏎](#chapter4#b4defi3_10) _**후위 이동 연산자(backshift operator)**는 다음과 같이 정의되며_

Bxt\=xt−1

_거듭제곱으로 확장됩니다_ B2xt\=B(Bxt)\=Bxt−1\=xt−2, _등등. 따라서,_

Bkxt\=xt−k.

이제 (3.18)을 다음과 같이 다시 쓸 수 있습니다.

∇xt\=(1−B)xt,


개념을 더 확장할 수도 있습니다. 예를 들어 2차 차분은 다음과 같습니다.

∇2xt\=(1−B)2xt\=(1−2B+B2)xt\=xt−2xt−1+xt−2

이는 연산자의 선형성(linearity)에 의한 것입니다.

정의 3.11. [본문으로 돌아가기.⏎](#chapter5#b5defi3_11) _**d차 차분(Differences of order d)**은 다음과 같이 정의되며_

∇d\=(1−B)d,

_더 높은 정수 d값에 대해 평가하기 위해 연산자_ (1−B)d_를 대수적으로 전개할 수 있습니다. 만약_ d\=1_이면, 표기법에서 이를 생략합니다._

첫 번째 차분 (3.18)은 추세를 제거하기 위해 적용되는 _선형 필터(linear filter)_의 한 예입니다. _xt_ 근처의 값들을 평균화하여 형성되는 다른 필터들은 [6장](#chapter6)에서처럼 불필요한 다른 종류의 변동을 제거하는 조정된 계열을 생성할 수 있습니다. 차분 기술은 [5장](#chapter5)에서 논의될 ARIMA 모델의 중요한 구성 요소입니다.

예제 3.12 상품의 차분(Differencing a Commodity) [본문으로 돌아가기.⏎](#chapter7#b7exam3_12)

[그림 3.8](#chapter3#fig3_8)의 하단 패널에 표시된 연어 가격 계열의 1차 차분은 회귀(regression)를 통한 추세 제거와는 다른 결과를 생성합니다. 예를 들어, 추세가 제거된 계열에서 관찰했던 키친(Kitchin) 경기 순환은 차분된 계열에서는 명확하지 않습니다(여전히 존재하지만, 이는 [7장](#chapter7)의 기법을 사용하여 확인할 수 있습니다).

차분된 계열의 ACF는 [그림 3.9](#chapter3#fig3_9)의 하단 패널에 나와 있습니다. 이 경우 차분된 계열은 원본 데이터나 추세가 제거된 데이터에서는 명확하지 않았던 강한 연간 주기성을 나타냅니다. [그림 3.8](#chapter3#fig3_8) 및 [그림 3.9](#chapter3#fig3_9)를 재현하는 코드는 다음과 같습니다.


`par(mfrow=2:1)`
`**tsplot**(**detrend**(**salmon**), col=4, main="detrended **salmon** price")`
`**tsplot**(diff(**salmon**), col=4, main="differenced **salmon** price")`
`par(mfrow=2:1)`
`**acf1**(**detrend**(**salmon**), 48, col=4, main="detrended **salmon** price")`
`**acf1**(diff(**salmon**), 48, col=4, main="differenced **salmon** price")`

예제 3.13 60\. 전 지구 온도의 차분(Differencing Global Temperature) [본문으로 돌아가기.⏎](#chapter8#b8exam3_13)

[그림 1.4](#chapter1#fig1_4)에 표시된 전 지구 온도 계열은 추세 정상 계열(trend stationary series)이라기보다는 무작위 보행(random walk)에 더 가깝게 움직이는 것처럼 보입니다. 따라서 데이터의 추세를 제거하는 것보다는 데이터를 강제로 정상화하기 위해 차분을 사용하는 것이 더 적절할 것입니다. 차분된 데이터는 해당 표본 ACF와 함께 [그림 3.10](#chapter3#fig3_10)에 나와 있습니다. 이 경우 차분된 프로세스는 지연 1을 제외하고는 자기상관이 거의 없음을 보여주며, 이는 전 지구 온도 계열이 표류(drift)가 있는 무작위 보행과 거의 같음을 암시할 수 있습니다.

![Differenced annual global temperature deviation time series and its sample autocorrelation function](./images/fig3_10.jpg)

그림 3.10: 차분된 전 지구 온도 계열 및 그 표본 ACF. [본문으로 돌아가기.⏎](chapter3)

만약 그 계열이 표류가 있는 무작위 보행이라면, 차분된 계열의 평균이 표류의 추정치가 된다는 점은 흥미롭습니다. 전 지구 온도 상승이 뚜렷하게 나타난 1980년을 기준으로 전후 계열에만 집중해 보면[(Hansen and Lebedeff, 1987 참조)](#bibref1#refbib_22), 표류는 10배 이상 증가합니다.


`par(mfrow=c(2,1))`
`**tsplot**(diff(**gtemp_land**), col=4, main="differenced global temperature")`
`**acf1**(diff(**gtemp_land**), col=4, nxm=0)`
`mean(window(diff(**gtemp_land**), end=1979))   _# drift before 1980_`
`  [1] 0.00465`
`mean(window(diff(**gtemp_land**), start=1980)) _# drift after 1980_`
`  [1] 0.04909`

61\. 때때로 시계열 데이터에서 이분산성(heteroscedasticity)이 나타납니다. 이 경우에 특히 유용한 변환은 다음과 같습니다.

yt\=logxt,

이는 기본 값(underlying values)이 큰 계열의 부분에서 발생하는 더 큰 변동성을 억제하는 경향이 있습니다. [예제 1.1](#chapter1#exam1_1) 및 [예제 1.2](#chapter1#exam1_2)에서 보았듯이 로그 변환은 작은 퍼센트 변화로 진화하는 시계열에서 자연스럽게 발생합니다. 다른 가능성으로는 다음과 같은 형태의 Box-Cox 계열의 거듭제곱 변환(power transformations)이 있습니다.

yt\={(xtλ−1)/λλ≠0,logxtλ\=0.

거듭제곱 _λ_를 선택하는 방법들이 제공되지만[(Johnson and Wichern, 2002, §4.7 참조)](#bibref1#refbib_29), 여기서는 다루지 않겠습니다. 종종 변환은 정규성(normality)에 대한 근사를 개선하거나 한 계열에서 다른 계열의 값을 예측할 때 선형성(linearity)을 향상시키는 데 사용됩니다.

예제 3.14 고기후 빙하 점토호(Paleoclimatic Glacial Varves) [본문으로 돌아가기.⏎](#chapter4#b4exam3_14)

녹아내리는 빙하는 봄철 해빙기 동안 매년 모래와 미사 층을 퇴적시키는데, 이는 뉴잉글랜드에서 빙하가 퇴각하기 시작한 때(약 12,600년 전)부터 끝난 때(약 6,000년 전)까지 이르는 기간 동안 매년 재구성할 수 있습니다. _점토호(varves)_라고 불리는 이러한 퇴적물은 온도와 같은 고기후 매개변수의 대용물(proxies)로 사용될 수 있습니다. 따뜻한 해에는 후퇴하는 빙하에서 더 많은 모래와 미사가 퇴적되기 때문입니다. [그림 3.11](#chapter3#fig3_11)의 윗부분은 11,834년 전에 시작하여 매사추세츠의 한 위치에서 634년 동안 수집된 62\. 연간 점토호의 두께를 보여줍니다. 더 자세한 정보는 [Shumway and Verosub (1992)](#bibref1#refbib_45)를 참조하십시오.

![Glacial varve thicknesses, the log transformed series, and QQ plots of each series](./images/fig3_11.jpg)

그림 3.11: 로그(log) 변환된 두께(하단)와 비교한 매사추세츠의 n\=634년에 대한 빙하 점토호 두께(상단). 오른쪽의 플롯들은 해당 정규 Q-Q 플롯들입니다. [본문으로 돌아가기.⏎](chapter3)

두께의 변동은 퇴적된 양에 비례하여 증가하므로, 로그 변환은 시간의 함수로서 분산에서 관찰되는 비정상성(nonstationarity)을 제거할 수 있습니다. [그림 3.11](#chapter3#fig3_11)은 원본 및 로그 변환된 점토호를 보여주며 이러한 개선이 일어났음이 분명합니다. 또한, 해당하는 정규 Q-Q 플롯(normal Q-Q plots)도 표시되어 있습니다. 이 플롯들에 대해서는 [섹션 A.1](#appA#secA_1)에서 논의하겠지만, 간단히 말해 이는 정규 분포의 이론적 분위수(theoretical quantiles)에 대한 데이터 분위수의 그래프입니다. 정규 데이터(normal data)는 표시된 일치선(line of equality)에 대략적으로 위치해야 합니다. 이 경우, 로그 변환으로 정규성에 대한 근사가 향상되었다고 주장할 수 있습니다. [그림 3.11](#chapter3#fig3_11)은 다음과 같이 생성되었습니다:


`layout(matrix(1:4,2), widths=c(2.5,1))`
`**tsplot**(**varve**, main=NA, ylab=NA, col=4, **margins**=0)`
` mtext("**varve**", side=3, line=.25, cex=1.1, font=2, adj=0)`
`**tsplot**(log(**varve**), main=NA, ylab=NA, col=4, **margins**=0)`
` mtext("log(**varve**)", side=3, line=.25, cex=1.1, font=2, adj=0)`
`**QQnorm**(**varve**, main=NA, **nxm**=0)`
`**QQnorm**(log(**varve**), main=NA, **nxm**=0)`

다음으로 서로 다른 지연(lags)에서 계열 간의 관계를 시각화할 목적으로 사용되는 또 다른 예비 데이터 처리 기법인 _지연 플롯(lagplot)_에 대해 고려해 보겠습니다. 63\. ACF 및 CCF를 사용할 때 우리는 시계열의 지연된 값(lagged values) 사이의 선형 관계를 측정합니다. 하지만 예측 가능성을 선형성으로 국한하는 ACF 및 CCF는 값 _xt_와 그 과거 값 xt−h, 또는 다른 계열 yt−h 간의 가능한 비선형 관계(nonlinear relationships)를 가릴 수 있습니다.

예제 3.15 지연 플롯: SOI와 신규가입(Lagplots: SOI and Recruitment) [본문으로 돌아가기.⏎](chapter3)

[그림 3.12](#chapter3#fig3_12)는 수평축에 St−h를 두고 수직축에 SOI, _St_를 표시한 지연 플롯을 보여줍니다. 표본 자기상관은 우측 상단 모서리에 표시되며 비선형성을 발견하는 데 도움이 될 수 있는 국소 가중 산점도 평활화(lowess) 선이 지연 플롯에 겹쳐져 있습니다. 다음 섹션에서 평활화에 대해 논의하겠지만, 지금은 lowess를 국소 회귀를 피팅하는 방법, 즉 수평축의 작은 구간에 대한 회귀로 생각해 주십시오.

![Lagplot of the Southern Oscillation  series up to lag 12](./images/fig3_12.jpg)

그림 3.12: 1에서 12까지의 지연(lags)에서 현재와 과거의 SOI 값과 관련된 지연 플롯. 우측 하단 모서리의 값은 표본 자기상관이고 선은 lowess 피팅선입니다. [본문으로 돌아가기.⏎](chapter3)

64\. [그림 3.12](#chapter3#fig3_12)에서 국소적 피팅들이 대략 선형적이라는 점에 주목해 주십시오. 즉 표본 자기상관이 유의미합니다. 또한 지연 h\=1,2,11,12에서 _St_와 St−h 사이의 강한 양의 선형 관계를 볼 수 있으며, 지연 h\=6,7에서 음의 선형 관계를 볼 수 있습니다.

마찬가지로, 두 계열 사이에 가능한 비선형 관계를 찾기 위해 다양한 지연에서 SOI, _St_에 대한 신규가입(Recruitment), _Rt_의 값을 보고 싶을 수 있습니다. 예를 들어 현재 또는 과거의 SOI 계열인 h\=0,1,2,...에 대한 St−h로부터 신규가입 계열 _Rt_를 예측하고자 할 수 있으므로 산점도 행렬을 살펴보는 것은 가치가 있을 것입니다. [그림 3.13](#chapter3#fig3_13)은 수평축에 St−h가 플롯된 것에 대해 수직축에 _Rt_를 플롯한 지연 플롯을 보여줍니다. 추가로, 이 그림은 lowess 피팅선과 표본 교차상관을 보여줍니다.

![Lagplot between the Southern Oscillation Index series and the Recruitment series](./images/fig3_13.jpg)

그림 3.13: h\=0,1,…,8인 지연에서 수평축의 SOI 계열 St−h에 대한 수직축의 신규가입 계열 _Rt_의 지연 플롯. 우측 상단 모서리의 값은 표본 교차상관이고 선은 lowess 피팅선입니다. [본문으로 돌아가기.⏎](chapter3)

[그림 3.13](#chapter3#fig3_13)은 지연 h\=5,6,7,8에 대한 신규가입과 SOI 사이의 상당히 강한 비선형 관계를 보여주며, 이는 SOI 계열이 65\. 이들 지연에서 신규가입 계열을 선행하는 경향이 있음을 나타냅니다. 관계는 음이며, 이는 SOI의 증가가 신규가입의 감소로 이어짐을 뜻합니다. 지연 플롯에서 관찰된 비선형성(겹쳐진 lowess 피팅의 도움을 받음)은 신규가입과 SOI 사이의 동작이 SOI의 양수 값과 음수 값에 대해 다름을 나타냅니다.


`**lag1.plot**(**soi**, 12, col=4, **location**="topleft", lwl=2)   _#   Figure 3.12_`
`**lag2.plot**(**soi**, **rec**, 8, col=4, lwl=2)                   _#   Figure 3.13_`

예제 3.16 회귀를 사용하여 잡음 속 신호 찾기(Using Regression to Discover a Signal in Noise) [본문으로 돌아가기.⏎](chapter3)

[예제 1.11](#chapter1#exam1_11)에서는 다음 모델에서 n\=500개의 관측치를 생성했습니다.

xt\=Acos(2πωt+ϕ)+wt,(3.19)

여기서 ω\=1/50, A\=2, ϕ\=.6π, 및 σw\=5입니다. 데이터는 [그림 3.14](#chapter3#fig3_14)의 상단 패널에 표시되어 있습니다. 여기서 우리는 진동 빈도(frequency of oscillation) ω\=1/50은 알려져 있지만 _A_와 _ϕ_는 알려지지 않은 매개변수라고 가정합니다. 이 경우 매개변수가 모델 (3.19)에서 비선형적으로 나타나므로, 우리는 삼각함수 항등식(trigonometric identity, [섹션 B.5](#appB#secB_5) 참조)을 사용하여 다음과 같이 적을 수 있습니다.

Acos(2πωt+ϕ)\=β1cos(2πωt)+β2sin(2πωt),

![Cosine signal plus large normal noise, and the same time series with the fitted cosine regression line](./images/fig3_14.jpg)

그림 3.14: (3.19)에 의해 생성된 데이터\[상단\] 및 데이터 위에 겹쳐진 피팅된 선\[하단\]. [본문으로 돌아가기.⏎](chapter3)

여기서 β1\=Acos(ϕ)이고 β2\=−Asin(ϕ)입니다.

이제 모델 (3.19)는 (여기서는 절편 항이 필요하지 않음) 다음과 같이 주어진 일반적인 선형 회귀 형태로 쓸 수 있습니다.

xt\=β1cos(2πt/50)+β2sin(2πt/50)+wt.

66\. 선형 회귀를 사용하여, 우리는 σ^w\=5.18로 β^1\=−.74(.33), β^2\=−1.99(.33)을 구했습니다. 괄호 안의 값은 표준 오차입니다. 이 예제에서 계수의 실제 값은 β1\=2cos(.6π)\=−.62이고, β2\=−2sin(.6π)\=−1.90임에 유의해 주십시오. 신호 대 잡음비(signal-to-noise ratio)가 작더라도 회귀 분석을 사용하면 잡음 속에서 신호를 감지할 수 있음이 분명합니다. [그림 3.14](#chapter3#fig3_14)의 윗부분은 (3.19)로 생성된 데이터를 보여줍니다. 신호를 식별하기 어렵고 데이터가 잡음처럼 보입니다. 그러나 그림의 아랫부분은 겹쳐진 피팅선과 함께 동일한 데이터를 보여줍니다. 이제 잡음 속에서 신호를 쉽게 볼 수 있습니다.


`set.seed(90210)                _# so you can reproduce these results_`
`x = 2*cos(2*pi*1:500/50 + .6*pi) + rnorm(500,0,5)`
`z1 = cos(2*pi*1:500/50); z2 = sin(2*pi*1:500/50)`
`**ttable**(fit <- lm(x~ 0 + z1 + z2)) _# zero to exclude the intercept_`
`      Estimate      SE t.value p.value`
`  z1   -0.7442 0.3274 -2.2729     0.0235`
`  z2   -1.9949 0.3274 -6.0926     0.0000`
` Residual standard error: 5.177 on 498 degrees of freedom`
`par(mfrow=c(2,1))`
`**tsplot**(x, col=4, **gg**=TRUE)`
`**tsplot**(x, ylab=bquote(hat(x)), col=**astsa.col**(4,.7), **gg**=TRUE)`
`lines(fitted(fit), col=6, lwd=2)`

예제 3.17 비선형 회귀를 사용하여 잡음 속 신호 찾기(Using Nonlinear Regression to Discover a Signal in Noise)\*

비선형 회귀(nonlinear regression)를 사용하여 미지의 진폭(amplitude), 위상(phase) 및 빈도(frequency)를 가진 [예제 3.16](#chapter3#exam3_16)의 모델 (3.19) 피팅 문제를 다룰 수 있습니다. 여기서는 세부 내용으로 들어가지 않고 stats 패키지의 비선형 최소제곱(nls)을 사용하는 방법을 시연하지만, 가우스-뉴턴(Gauss-Newton)을 통한 비선형 최소제곱은 [예제 4.27](#chapter4#exam4_27)에서 논의됩니다. 또한 중요한 빈도를 발견하는 방법에 대해서는 [6장](#chapter6)과 [7장](#chapter7)에서 다룹니다.

[예제 3.16](#chapter3#exam3_16)에서와 같이 동일한 시드(seed)를 사용하여 다음 모델에서 500개의 관측치를 생성했습니다.

xt\=2cos(2π(t+15)/50)+wt,

여기서 σw\=5입니다.

nls 스크립트에는 괜찮은 시작 값이 필요합니다. [그림 3.14](#chapter3#fig3_14)의 상단을 보면 데이터에 잡음이 많지만 대부분 값들이 ±10 사이에 있으므로 진폭을 A\=10에서 시작합니다. 데이터에서 위상 이동을 감지하기가 쉽지 않으므로 ϕ\=0에서 시작합니다. 빈도의 경우 [6장](#chapter6) 및 [7장](#chapter7) 기법으로 좋은 시작 값을 쉽게 찾을 수 있지만, ACF(표시되지 않음)는 데이터가 약 50포인트마다 한 주기를 만듦을 시사합니다. 하지만 재미를 더하기 위해 우리는 55포인트마다 하나의 주기로 초기화할 것입니다.


`set.seed(90210)`
`t = 1:500`
`x = 2*cos(2*pi*(t+15)/50) + rnorm(500,0,5)`
`**acf1**(x, 200)    _# not displayed_`
`_# run the nonlinear regression_`
`initial.values = list(A=10, omega=1/55, phi=0)`
`summary(fit <- nls(x~ A*cos(2*pi*omega*t + phi), start=initial.values))`
`  Parameters:`
`           Estimate Std. Error t value Pr(>|t|)`
`  A       2.1531217   0.3284401   6.556 1.39e-10`
`  omega 0.0201519     0.0001664 121.100    < 2e-16`
`  phi   -4.6289548    0.3048891 -15.182    < 2e-16`
`  ---`
`  Residual standard error: 5.179 on 497 degrees of freedom`
`  Number of iterations to convergence: 11`
`**tsplot**(x, ylab=bquote(hat(x)), col=4, **gg**=TRUE) _# not shown but looks like_`
` lines(fitted(fit), col=2, lwd=2)                _# the bottom of Figure 3.14_`

피팅된 값은 위상(phi)에 대해 cos(2π(t+15)/50)\=cos(2π(t−35)/50) 및 2π(−35/50)\=−4.4라는 점을 감안할 때 실제 값에 매우 가깝습니다.

## 3.3 67\. 시계열 평활화(Smoothing Time Series)

[예제 1.8](#chapter1#exam1_8)에서는 이동 평균(moving average)을 사용하여 시계열을 평활화하는 개념을 소개했습니다.[3](#chapter3#fn3_3) 이 방법은 장기 추세 및 계절 성분(자세한 내용은 [섹션 6.3](#chapter6#sec6_3) 참조)과 같은 시계열의 특정 특징을 발견하는 데 유용합니다. 특히 _xt_가 관측치를 나타내는 경우 다음을 의미합니다.

mt\=∑j\=−kkajxt−j,(3.20)

여기서 aj\=a−j≥0 및 ∑j\=−kkaj\=1는 대칭 이동 평균입니다.

예제 3.18 이동 평균 평활기(Moving Average Smoother) [본문으로 돌아가기.⏎](chapter3)

[그림 3.15](#chapter3#fig3_15)는 [예제 1.4](#chapter1#exam1_4)에서 논의된 월별 SOI 계열을 (3.20)에서 k\=6 및 가중치 a0\=a±1\=⋯\=a±5\=1/12 및 a±6\=1/24를 사용하여 평활화한 것을 보여줍니다. 이 특정 방법은 명백한 연간 온도 주기를 제거(필터링)하고 엘니뇨 주기를 강조하는 데 도움이 됩니다. 양 끝에 절반 가중치(half-weights)를 사용하는 이유는 동일한 달이 평균에 두 번 포함되지 않게 하기 위해서입니다. 예를 들어, 7월(j\=0)을 중심으로 두면 그 해의 1월(j\=−6)과 내년의 1월(j\=6)이 평활기에 포함됩니다. 결과적으로 각 1월은 절반 가중치를 받게 됩니다. [그림 3.15](#chapter3#fig3_15)를 재현하려면:


`w = c(.5, rep(1,11), .5)/12`
`soif = filter(**soi**, sides=2, filter=w)`
`**tsplot**(**soi**, col=4)`
`lines(soif, lwd=2, col=6)`
`_# insert_`
`par(fig = c(0,.25,0,.25), new = TRUE, col=8)`
`w1 = c(rep(0,20), w, rep(0,20))`
`plot(w1, type="l", ylim = c(-.02,.1), xaxt="n", yaxt="n", ann=FALSE, col=4)`

![The Southern Oscillation Index series smoothed using a moving average filter that attenuates the annual cycle and highlights the El Nino cycle](./images/fig3_15.jpg)

그림 3.15: 계절 이동 평균 평활기를 사용하여 평활화된 SOI 계열. 삽입된 그림은 [예제 3.18](#chapter3#exam3_18)에 설명된 이동 평균 커널\[비율대로 그려지지 않음\]의 형태를 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 3우리는 스크립트 filter를 사용할 것입니다. 만약 dplyr도 로드되었다면 missing_Exercise 힌트의 경고를 확인하십시오. [본문으로 돌아가기.⏎](#chapter3#fn3_33b)

68\. 이동 평균이 엘니뇨 효과를 강조하는 데 좋은 역할을 하지만, 너무 끊어지는(choppy) 느낌이 들 수 있습니다. 대신 가중치에 정규 분포를 사용하면 더 부드러운 피팅을 얻을 수 있습니다.

예제 3.19 커널 평활화(Kernel Smoothing) [본문으로 돌아가기.⏎](chapter3)

커널 평활화(Kernel smoothing)는 관측치를 평균화하기 위해 일반적인 가중치 함수(weight functions) 또는 커널을 사용하는 이동 평균 평활기입니다. [그림 3.16](#chapter3#fig3_16)은 SOI 계열의 커널 평활화를 보여주며, 여기서 _mt_는 다음과 같습니다.

mt\=∑i\=1nwi(t)xti,

![The Southern Oscillation Index series smoothed using a kernel smoother that attenuates the annual cycle and highlights the El Nino cycle](./images/fig3_16.jpg)

그림 3.16: SOI의 커널 평활기. 삽입된 그림은 정규 커널의 형태\[비율대로 그려지지 않음\]를 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

여기서

wi(t)\=K(t−tib)/∑k\=1nK(t−tkb)

가중치이며, K(⋅)는 커널 함수(kernel function)입니다. 이 예제에서는 정규 커널(normal kernel)인 K(z)\=exp(−z2/2)가 사용됩니다.

R에서 이를 구현하기 위해 우리는 대역폭(bandwidth)을 선택할 수 있는 ksmooth 함수를 사용합니다. _b_를 표준 편차로 생각할 수 있으며 대역폭이 클수록 더 부드러운 결과가 나옵니다. 이 경우, 우리는 시간 경과에 따라 평활화를 수행하고 있으며 이는 soi에 대해 t/12의 형태를 취합니다. [그림 3.16](#chapter3#fig3_16)에서는 대략 1년 동안 평활화하는 것에 해당하도록 b\=1의 값을 사용했습니다. 이 예제의 코드는 다음과 같습니다. 69\.


`**tsplot**(**soi**, col=4)`
`lines(ksmooth(time(**soi**), **soi**, "normal", bandwidth=1), lwd=2, col=6)`
`_# insert_`
`par(fig = c(0,.25,0,.25), new = TRUE, col=8)`
`curve(dnorm(x), -3, 3, xaxt="n", yaxt="n", ann=FALSE, col=4)`

SOI의 시간 단위가 달(months)인 경우 동등한 평활기는 12의 대역폭을 사용한다는 점에 유의해 주십시오.


`SOI = ts(**soi**, freq=1) _# make the unit of time a month_`
`**tsplot**(SOI, col=4)    _# not shown_`
`lines(ksmooth(time(SOI), SOI, "normal", bandwidth=12), lwd=2, col=6)`

예제 3.20 Lowess [본문으로 돌아가기.⏎](chapter3)

평활화에 대한 또 다른 접근 방식은 k<n에 대해 데이터를 {xt−k/2,…,xt,…,xt+k/2}만 사용하여 시간에 대한 회귀를 통해 _xt_를 예측하고 그런 다음 mt\=x^t로 설정하는 _k_\-최근접 이웃 회귀(k-nearest neighbor regression)를 기반으로 합니다.

Lowess [(Cleveland, 1979)](#bibref1#refbib_11)는 다소 복잡한 평활화 방법이지만 기본적인 아이디어는 최근접 이웃 회귀와 유사합니다. 첫째, _xt_에 대한 일정 비율의 가장 가까운 이웃들이 가중치 체계(weighting scheme)에 포함됩니다. 시간 상 _xt_에 더 가까운 값이 더 많은 가중치를 받습니다. 그런 다음, 강건 가중 회귀(robust weighted regression)를 사용하여 _xt_를 예측하고 평활화된 값 _mt_를 얻습니다. 포함된 최근접 이웃의 비율이 클수록 피팅이 더 평활해집니다. [예제 3.15](#chapter3#exam3_15)에서 지연 플롯(lag plots)을 위한 lowess 평활화를 소개했습니다. [그림 3.12](#chapter3#fig3_12)와 [그림 3.13](#chapter3#fig3_13)을 기억해 보십시오. 지연 플롯에서 lowess는 두 변수(같은 과정의 지연된 값이든 다른 과정의 지연된 값이든) 간의 비선형 관계를 조사하는 데 사용됩니다.

[그림 3.17](#chapter3#fig3_17)에서 평활기 중 하나는 데이터의 엘니뇨 주기에 대한 추정치를 얻기 위해 데이터의 5%를 사용합니다. 게다가 SOI의 (음의) 추세는 태평양의 장기적인 온난화를 나타냅니다. 이를 조사하기 위해, 우리는 70\. 기본 평활기 스팬(smoother span)과 함께 astsa의 추세(trend)를 사용했습니다. [그림 3.17](#chapter3#fig3_17)은 다음과 같이 재현할 수 있습니다.


`**trend**(**soi**, lowess=TRUE)                   _# trend (with default span)_`
`lines(lowess(soi, f=.05), lwd=2, col=6)   _# El Nio cycle_`

![The Southern Oscillation Index series smoothed using a locally weighted scatterplot smoother (lowess) that attenuates the annual cycle and highlights the El Nino cycle](./images/fig3_17.jpg)

그림 3.17: 추세와 엘니뇨 주기를 강조하기 위한 SOI 계열의 국소 가중 산점도 평활기(lowess). [본문으로 돌아가기.⏎](chapter3)

예제 3.21 다른 계열의 함수로서의 한 계열의 평활화(Smoothing One Series as a Function of Another)

시간 플롯(time plots) 평활화 외에도, [예제 3.15](#chapter3#exam3_15)에서 다양한 지연에서 신규가입과 SOI 사이의 비선형 관계를 시각화하기 위해 lowess를 사용했던 것처럼 평활화 기술을 다른 시계열의 함수로서 한 시계열을 평활화하는 데 적용할 수 있습니다.



[예제 3.6](#chapter3#exam3_6)에서 우리는 사망률과 기온 사이의 비선형 관계를 발견했습니다. [그림 3.18](#chapter3#fig3_18)은 사망률 _Mt_와 기온 _Tt_의 산점도와 함께 lowess를 사용하여 _Tt_의 함수로 평활화된 _Mt_를 보여줍니다. 극단적인 온도에서 사망률이 증가하지만 비대칭적인 방식으로 증가한다는 점에 유의해 주십시오. 더운 온도보다 추운 온도에서 사망률이 더 높습니다. 최소 사망률은 83.4∘ F에서 발생합니다.


`**tsplot**(**tempr**, **cmort**, type="p", col=4, xlab="Temperature", ylab="Mortality")`
`lines(lowess(**tempr**, **cmort**), col=6, lwd=2)`

![Smooth of mortality as a function of temperature using lowess from the LA pollution study](./images/fig3_18.jpg)

그림 3.18: lowess를 사용한 기온의 함수로서의 사망률 평활화. [본문으로 돌아가기.⏎](chapter3)

예제 3.22 고전적 구조 모델링(Classical Structural Modeling) [본문으로 돌아가기.⏎](chapter3)

시계열 분석에 대한 고전적인 접근 방식은 데이터를 추세(_Tt_), 계절성(_St_), 불규칙 또는 잡음(_Nt_)으로 레이블이 지정된 구성 요소로 분해하는 것입니다. 데이터를 _xt_로 나타내면 때때로 다음과 같이 쓸 수 있습니다.

xt\=Tt+St+Nt.

71\. 물론 모든 시계열 데이터가 이러한 패러다임에 맞는 것은 아니며 분해가 고유하지 않을 수도 있습니다. 때때로 경기 순환과 같은 추가적인 순환 구성 요소, 가령 _Ct_가 모델에 추가되기도 합니다.

[그림 3.19](#chapter3#fig3_19)는 stats 패키지의 stl을 사용하여 2002년부터 2016년까지 하와이 호텔의 분기별 객실 점유율에 대해 분해를 피팅한 결과를 보여줍니다. R은 분해를 피팅하기 위한 다른 스크립트도 제공합니다. 예를 들어, 스크립트 decompose는 [예제 3.18](#chapter3#exam3_18)에서와 같이 이동 평균을 사용합니다. 스크립트 stl은 각 구성 요소를 얻기 위해 loess(lowess와 관련됨)를 사용하며 [예제 3.20](#chapter3#exam3_20)에서 사용된 접근 방식과 유사합니다. stl을 사용하려면 계절 평활화 방법(seasonal smoothing method)을 지정해야 합니다. 즉, 문자열 periodic을 지정하거나 계절 추출을 위한 loess 창(window)의 스팬(span)을 지정해야 합니다. 스팬은 홀수여야 하며 최소한 7이어야 합니다(기본값은 없습니다). 계절 창을 사용함으로써 우리는 주기적인(periodic) 계절 성분을 지정하여 강제되는 St\=St−4보다는 St≈St−4를 허용합니다.

![Structural model of the Hawaiian quarterly  occupancy rate displaying the seasonal, trend, and noise components](./images/fig3_19.jpg)

그림 3.19: 하와이 호텔 분기별 점유율의 구조 모델. [본문으로 돌아가기.⏎](chapter3)

[그림 3.19](#chapter3#fig3_19)에서 계절 성분은 매우 규칙적이어서 1분기와 3분기에는 2%~4% 증가를 보이는 반면 2분기와 4분기에는 2%~4% 감소를 보입니다. 추세 구성요소는 추세라고 간주될 수 있는 것보다는 경기 순환에 더 가까울 것입니다. 이전에 암시했듯이 성분은 잘 정의되어 있지 않으며 분해는 고유하지 않습니다. 어떤 사람의 추세는 다른 사람의 경기 순환일 수 있습니다. 이 예제에 대한 기본 R 코드는 다음과 같습니다:


`x = window(**hor**, start=2002)`
`plot(decompose(x))           _# not shown_`
`plot(stl(x, s.window="per")) _# seasons are periodic - not shown_`
`plot(stl(x, s.window=15))`

72\. [그림 3.19](#chapter3#fig3_19)와 유사한 그림은 다음과 같이 생성할 수 있습니다.


`par(mfrow = c(4,1))`
`x = window(**hor**, start=2002)`
`out = stl(x, s.window=15)$time.series`
`**tsplot**(x, main="Hawaiian Occupancy Rate", ylab="% rooms", col=8, type="c")`
` text(x, labels=1:4, col=c(3,4,2,6), cex=1.25)`
`**tsplot**(out[,1], main="Seasonal", ylab="% rooms",col=8, type="c")`
` text(out[,1], labels=1:4, col=c(3,4,2,6), cex=1.25)`
`**tsplot**(out[,2], main="Trend", ylab="% rooms", col=8, type="c")`
` text(out[,2], labels=1:4, col=c(3,4,2,6), cex=1.25)`
`**tsplot**(out[,3], main="Noise", ylab="% rooms", col=8, type="c")`
` text(out[,3], labels=1:4, col=c(3,4,2,6), cex=1.25)`

## 문제(Problems)

* 3.1 **(구조 회귀 모델, Structural Regression Model).** [그림 1.1](#chapter1#fig1_1)에 표시된 Johnson & Johnson 데이터 _yt_에 대해 xt\=log(yt)라고 합시다. 이 문제에서 우리는 특별한 73\. 유형의 구조 모델 xt\=Tt+St+Nt를 피팅할 것인데, 여기서 _Tt_는 추세 구성요소, _St_는 계절 구성요소, _Nt_는 잡음입니다. 우리의 경우 시간 _t_는 분기(1960.00, 1960.25, …)이므로 한 단위의 시간은 1년입니다.  
   1. 다음과 같은 회귀 모델을 피팅하십시오.  
   xt\=βt⏟trend+α1Q1(t)+α2Q2(t)+α3Q3(t)+α4Q4(t)⏟seasonal+wt⏟noise  
   여기서 시간 _t_가 분기 i\=1,2,3,4에 해당하면 Qi(t)\=1이고, 그렇지 않으면 0입니다. Qi(t)는 지시 변수(indicator variables)라고 불립니다. 당분간 _wt_를 가우스 백색 잡음 시퀀스(Gaussian white noise sequence)라고 가정할 것입니다.  
   2. 모델이 정확하다면 주당 로그 수익(logged earnings per share)의 추정 평균 연간 증가율은 얼마입니까?  
   3. 모델이 정확하다면 3분기에서 4분기로 평균 로그 수익률이 증가합니까 아니면 감소합니까? 그리고, 몇 퍼센트나 증가하거나 감소합니까?  
   4. (a)의 모델에 절편 항(intercept term)을 포함하면 어떻게 됩니까? 문제가 발생한 이유를 설명하십시오.  
   5. 데이터 _xt_를 그래프로 그리고 그 위에 피팅된 값 x^t를 겹쳐서 그리십시오. 잔차(residuals) xt−x^t를 살펴보고 결론을 서술하십시오. 모델이 데이터에 잘 피팅되는 것으로 보입니까(잔차가 백색 잡음으로 보입니까)?
* 3.2. [예제 3.6](#chapter3#exam3_6)에서 조사한 사망률 데이터에 대하여:  
   1. 4주 전의 미세먼지 수치를 설명하는 또 다른 구성요소를 (3.13)의 회귀에 추가하십시오. 즉 (3.13)의 회귀에 Pt−4를 추가하십시오. 결론을 서술하십시오.  
   2. AIC 및 BIC를 사용하여, (a)의 모델이 [예제 3.6](#chapter3#exam3_6)의 최종 모델보다 개선된 것입니까?
* 3.3. 이 문제에서는 무작위 보행과 추세 정상 과정 간의 차이를 탐구합니다.  
   1. δ\=.01이고 σw\=1이며 길이가 n\=500인 표류가 있는 무작위 보행 (1.3) 계열 _네 개_를 생성하십시오. t\=1,…,500에 대한 데이터를 _xt_라고 부릅니다. 최소제곱을 사용하여 회귀 xt\=βt+wt를 피팅하십시오. 동일한 그래프에 데이터, 실제 평균 함수(즉, μt\=.01 t), 그리고 피팅된 선 x^t\=β^ t를 플롯하십시오.  
   2. 선형 추세와 잡음 yt\=.01 t+wt로 구성된 길이가 n\=500인 계열 _네 개_를 생성하십시오. 여기서 _t_와 _wt_는 (a) 부분과 동일합니다. 최소제곱을 사용하여 회귀 yt\=βt+wt를 피팅하십시오. 동일한 그래프에 데이터, 실제 평균 함수(즉, μt\=.01 t), 그리고 피팅된 선 y^t\=β^ t를 플롯하십시오. 74\.  
   3. (a) 부분과 (b) 부분의 결과 차이에 대해 의견을 제시하십시오.
* 3.4. 0의 평균과 분산 σw2를 갖는 독립적인 무작위 변수 _wt_로 구성된 가산적 잡음 항(additive noise term)을 가진 선형 추세로 구성된 과정을 고려하십시오. 즉,  
xt\=β0+β1t+wt,  
여기서 β0,β1은 고정된 상수입니다.  
   1. _xt_가 비정상(nonstationary)임을 증명하십시오.  
   2. 평균과 자기공분산 함수를 찾음으로써 1차 차분 계열 ∇xt\=xt−xt−1이 정상(stationary)임을 증명하십시오.  
   3. 만약 _wt_가 평균 함수 _μy_ 및 자기공분산 함수 γy(h)를 갖는 일반적인 정상 과정 _yt_로 대체된다면 (b) 부분을 반복하십시오.
* 3.5. (3.16)에 정의된 xt−xt−1이 정상임을 보여주십시오.
* 3.6. [그림 3.11](#chapter3#fig3_11)에 플롯된 빙하 점토호(glacial varve) 기록은 로그로 변환하여 개선할 수 있는 몇 가지 비정상성(nonstationarity)과 로그를 차분함으로써 교정될 수 있는 몇 가지 추가적인 비정상성을 보여줍니다.  
   1. 데이터의 전반부와 후반부에 대한 표본 분산을 계산하여 빙하 점토호 계열, 가령 _xt_가 이분산성(heteroscedasticity)을 나타낸다고 주장하십시오. 변환 yt\=logxt가 계열에 대한 분산을 안정화한다고 주장하십시오. 데이터 변환으로 정규성에 대한 근사가 향상되었는지 확인하기 위해 _xt_와 _yt_의 QQ 플롯을 제시하십시오.  
   2. 계열 _yt_를 플롯하십시오. [그림 1.4](#chapter1#fig1_4)의 전 지구 온도 기록에서 관찰된 것과 유사한 행동을 관찰할 수 있는 시간 간격(약 100년 단위)이 존재합니까?  
   3. _yt_의 표본 ACF를 검토하고 의견을 제시하십시오.  
   4. 차분 ut\=yt−yt−1을 계산하고 시간 플롯과 표본 ACF를 검토한 후, 로그로 변환된 점토호 데이터를 차분하면 합리적으로 정상인 계열(reasonably stationary series)이 생성된다고 주장하십시오. _ut_에 대한 실용적인 해석을 생각할 수 있습니까?
* 3.7. 전 지구 온도 계열 gtemp\_land의 추세를 추정하기 위해 [예제 3.18](#chapter3#exam3_18), [예제 3.19](#chapter3#exam3_19) 및 [예제 3.20](#chapter3#exam3_20)에서 설명한 세 가지 다른 평활화 기술을 사용하십시오. 의견을 제시하십시오.
* 3.8. [섹션 3.3](#chapter3#sec3_3)에서 우리는 엘니뇨/라니냐 주기가 대략 4년임을 알았습니다. 강한 4년 주기가 있는지 조사하기 위해, 남방 진동 지수(Southern Oscillation Index)에 대한 정현파 피팅(sinusoidal fit, 4년에 한 주기)을 lowess 피팅([예제 3.20](#chapter3#exam3_20)과 같이)과 비교하십시오. 정현파 피팅의 경우 추세 항을 포함시키십시오. 결과에 대해 논의하십시오. 75\.
* 3.9. [문제 3.1](#chapter3#question3_1)에서와 같이 _yt_를 [그림 1.1](#chapter1#fig1_1)에 표시된 원시 Johnson & Johnson 계열이라고 하고, xt\=log(yt)라고 합시다. 로그 변환된 데이터를 xt\=Tt+St+Nt로 분해하기 위해 [예제 3.22](#chapter3#exam3_22)에서 언급된 각 기법을 사용하고 그 결과를 기술하십시오. [문제 3.1](#chapter3#question3_1)을 풀었다면 그 문제의 결과와 이 문제에서 찾은 결과를 비교하십시오. 76은 비어 있습니다.

---


