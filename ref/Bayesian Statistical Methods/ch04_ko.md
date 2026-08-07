121\. 

# 4 선형 모델 (Linear models)

## 목차 (Contents)

* [4.1 정규 평균 분석 (Analysis of normal means)](./12-chapter4.md#sec4_1)  
   * [4.1.1 단일 표본/대응 표본 분석 (One-sample/paired analysis)](./12-chapter4.md#sec4_1_1)  
   * [4.1.2 두 정규 평균의 비교 (Comparison of two normal means)](./12-chapter4.md#sec4_1_2)
* [4.2 선형 회귀 (Linear regression)](./12-chapter4.md#sec4_2)  
   * [4.2.1 고전적 접근법 (Classical approaches)](./12-chapter4.md#sec4_2_1)  
   * [4.2.2 베이지안 선형 회귀를 위한 사전 분포 (Prior distributions for Bayesian linear regression)](./12-chapter4.md#sec4_2_2)  
         * [4.2.2.1 제프리스 사전 분포 (Jeffreys prior)](./12-chapter4.md#sec4_2_2_1)  
         * [4.2.2.2 가우스 사전 분포 (Gaussian prior)](./12-chapter4.md#sec4_2_2_2)  
         * [4.2.2.3 이중 지수 사전 분포 (Double exponential prior)](./12-chapter4.md#sec4_2_2_3)  
   * [4.2.3 행렬 표기법을 사용한 선형 회귀 (Linear regression in matrix notation)](./12-chapter4.md#sec4_2_3)  
   * [4.2.4 선형 회귀 예측 (Linear regression prediction)](./12-chapter4.md#sec4_2_4)  
   * [4.2.5 예제: 집의 마이크로바이옴에 영향을 미치는 요인 (Example: Factors that affect a home's microbiome)](./12-chapter4.md#sec4_2_5)
* [4.3 일반화 선형 모델 (Generalized linear models)](./12-chapter4.md#sec4_3)  
   * [4.3.1 이진 결과 (Binary outcome)](./12-chapter4.md#sec4_3_1)  
   * [4.3.2 범주형 결과 (Categorical outcomes)](./12-chapter4.md#sec4_3_2)  
   * [4.3.3 가산 결과 (Count outcomes)](./12-chapter4.md#sec4_3_3)  
   * [4.3.4 예제: NBA 클러치 자유투에 대한 로지스틱 회귀 (Example: Logistic regression for NBA clutch free throws)](./12-chapter4.md#sec4_3_4)  
   * [4.3.5 예제: 마이크로바이옴 데이터에 대한 베타 회귀 (Example: Beta regression for microbiome data)](./12-chapter4.md#sec4_3_5)
* [4.4 임의 효과 (Random effects)](./12-chapter4.md#sec4_4)
* [4.5 일반화 선형 혼합 모델 (Generalized linear mixed models)](./12-chapter4.md#sec4_5)
* [4.6 상관관계 데이터가 있는 선형 모델 (Linear models with correlated data)](./12-chapter4.md#sec4_6)
* [4.7 연습문제 (Exercises)](./12-chapter4.md#sec4_7)

선형 모델(Linear models)은 많은 통계 모델링과 직관의 기초를 형성합니다. 이 장에서는 일반적으로 사용되는 여러 통계 모델을 소개하고 이를 베이지안 프레임워크(Bayesian framework)에서 구현합니다. 주로 사전 분포(priors) 선택, 계산, 그리고 고전적 방법론(classical methods)과의 비교를 포함하여 이러한 분석의 베이지안 측면에 초점을 맞춥니다. 이 장은 정규 모집단(normal population)의 평균 분석([4.1.1절](./12-chapter4.md#sec4_1_1))과 두 정규 모집단 평균의 비교([4.1.2절](./12-chapter4.md#sec4_1_2))로 시작하며, 이는 고전적인 일표본(one-sample) 및 이표본(two-sample) t-검정과 유사합니다. [4.2절](./12-chapter4.md#sec4_2)에서는 고차원 문제에 적합한 사전 분포를 포함하여 더 일반화된 베이지안 다중 선형 회귀(multiple linear regression) 모델을 소개합니다. [4.3절](./12-chapter4.md#sec4_3)에서는 일반화 선형 모델(generalized linear models)을 통해 비가우스(non-Gaussian) 데이터로, [4.4절](./12-chapter4.md#sec4_4)과 [4.5절](./12-chapter4.md#sec4_5)에서는 선형 혼합 모델(linear mixed models)을 통해 상관관계가 있는 데이터(correlated data)로 다중 회귀를 확장합니다.

## 4.1 122\. 정규 평균 분석 (Analysis of normal means)

### 4.1.1 단일 표본/대응 표본 분석 (One-sample/paired analysis)

단일 표본 연구에서 _n_ 개의 관측치는 Yi|μ,σ2∼iidNormal(μ,σ2)로 모델링되며, 목적은 μ\=0인지 여부를 결정하는 것입니다. 이 모델은 각 단위 _i_가 서로 다른 조건에서 측정한 두 개의 값을 갖고, _Y_ _i_가 그 측정값들의 차이인 실험에 자주 적용됩니다. 예를 들어, 튜토리얼 세션 전후의 학생 _i_의 수학 점수를 각각 Z0i와 Z1i라고 할 때, 모집단 평균인 Yi\=Z1i−Z0i가 0인지를 검정하는 것은 튜토리얼 세션의 효과를 평가하는 한 가지 방법이 됩니다.

베이지안 분석에서는 _μ_와 _σ_2에 대한 사전 분포를 명시하고, _σ_2의 불확실성을 반영하여 _μ_의 주변 사후 분포(marginal posterior)를 요약합니다. 이 장에서는 제프리스 사전 분포(Jeffreys prior)를 사용하지만, 켤레 정규/역감마 사전 분포(conjugate normal/inverse gamma priors)를 사용할 수도 있습니다. 대부분의 경우 사후 밀도(posterior density) p(μ|Y)를 플롯하고 사후 평균(posterior mean) 및 95% 구간(95% interval)을 보고하는 것으로 충분합니다. 또한 공식적인 가설 검정(hypothesis test)을 사용하여 이 문제를 다룰 것입니다. 이 장에서는 단측 가설(one-sided hypotheses) H1:μ≤0 및 H2:μ\>0의 사후 확률(posterior probabilities)을 계산할 것입니다. [6장](./14-chapter6.md)에서는 점 가설(point hypothesis) H1:μ\=0 대 H2:μ≠0를 검정합니다.

**알려진 분산 (Known variance)**: _σ_2이 주어졌을 때, _μ_는 제프리스 사전 분포 π(μ)∝1을 갖습니다. 사후 분포는 μ|Y∼Normal(Y¯,σ2n)이며, 따라서 _μ_에 대한 100(1−α)% 사후 신용 구간(posterior credible interval)은 다음과 같습니다.

Y¯±zα/2σn,(4.1)

여기서 _z_ _τ_는 표준 정규 분포(즉, 평균이 0이고 분산이 1인 정규 분포)의 _τ_ 분위수(quantile)입니다. 이는 고전적인 100(1−α)% 신뢰 구간(confidence interval)과 정확히 일치합니다. 비록 이 경우 신용 구간과 신뢰 구간이 수치적으로 동일하지만, 다르게 해석됩니다. 베이지안 신용 구간은 주어진 데이터세트 **Y**를 바탕으로 _μ_에 대한 불확실성을 수치화하는 반면, 신뢰 구간은 실험을 반복할 경우 기대되는 Y¯의 변동을 수치화합니다.

귀무가설 H1:μ≤0 대 대립가설 H2:μ\>0의 검정을 위한 귀무가설의 사후 확률은 다음과 같습니다.

Prob(H1|Y)\=Prob(μ<0|Y)\=Φ(−Z)(4.2)

여기서 Φ는 표준 정규 누적 분포 함수(standard normal cumulative distribution function)이고 Z\=nY¯/σ이므로 빈도주의(frequentist) p-값(p-value)과 정확히 일치합니다. 정의에 의해 Φ(zτ)\=τ이므로, H1의 사후 확률이 _α_ 미만일 경우 H1을 기각하고 H2를 채택하는 결정 규칙(decision rule)은 −Z<zα, 또는 대칭성으로 인해 동일하게 Z\>z1−α일 때 H1을 기각하는 것과 동등합니다(표준 정규 확률 밀도 함수의 대칭성으로 인해 −zα\=z1−α). 따라서 유의수준(significance level) _α_에서 Z\>z1−α일 때 H1을 기각하고 H2를 지지하는 결정 규칙은 고전적인 단측 z-검정(one-sided z-test)과 동일합니다. 그러나 고전적인 검정과 달리, Prob(H1|Y)를 계산했으므로 가설 H1(또는 H2)이 참일 사후 확률을 사용하여 불확실성을 수량화할 수 있습니다.

**모르는 분산 (Unknown variance)**: [2.4절](./10-chapter2.md#sec2_4)에서 살펴본 바와 같이, (μ,σ2)에 대한 제프리스 사전 분포는 다음과 같습니다.

π(μ,σ2)∝(1σ2)3/2.(4.3)

[부록 A.3](./18-appA.md#secA_3)은 _σ_2을 적분하여 구한 _μ_의 주변 사후 분포가 다음과 같음을 보여줍니다.

μ|Y∼tn(Y¯,σ^2/n),(4.4)

123\. 여기서 σ^2\=∑i\=1n(Yi−Y¯)2/n입니다. 즉, 사후 분포는 위치(location)가 Y¯이고, 척도 모수(scale parameter)가 σ^2/n이며 자유도가 _n_인 스튜던트 t-분포(Student's t-distribution)입니다. 신용 집합(credible sets)이나 _μ_가 양수일 사후 확률 같은 사후 추론은 스튜던트 t-분포의 분위수를 따릅니다. 신용 집합은 빈도주의 신뢰 구간과 약간 다른데, 고전적인 t-검정의 자유도가 n−1인 반면 사후 분포의 자유도는 _n_이기 때문입니다. 이는 _σ_2에 대한 사전 분포의 효과입니다.

고전 통계학에서는 _σ_2을 모를 때 정규 분포 기반의 Z-검정을 스튜던트 t-분포 기반의 t-검정으로 대체합니다. 마찬가지로, 분산에 대한 불확실성을 고려할 때 평균의 사후 분포는 _σ_2이 주어졌을 때의 정규 분포에서 스튜던트 t-분포로 변경됩니다. [그림 4.1](./12-chapter4.md#fig4_1)은 가우스 및 t 밀도 함수를 비교합니다. 밀도 함수는 n\=25일 때 거의 동일하지만, n\=5일 때는 t-분포가 가우스 분포보다 더 두꺼운 꼬리(heavier tails)를 가집니다. 이는 _σ_2의 불확실성을 설명한 결과입니다.

![Two panels show posterior densities for mu under Gaussian and Student t models. For n equal to 5 the curves peak near mu around 10, with the t curve slightly heavier tailed. For n equal to 25 both curves nearly coincide in a narrow peak near 10, showing reduced difference between the models as sample size increases.](./images/fig4_1.jpg)

그림 4.1 **가우스 분포와 스튜던트 t-분포의 비교.** 아래는 평균이 Y¯이고 표준편차가 σ/n인 가우스 확률 밀도 함수(PDF)와 위치가 Y¯, 척도가 σ^/n이고 자유도가 _n_인 스튜던트 t-분포의 PDF를 비교한 것입니다. 플롯은 Y¯\=10, σ\=σ^\=2, n∈{5,25}라고 가정합니다. [본문으로 돌아가기.⏎](chapter4)

2.4절에서 논의된 바와 같이, 대안적인 제프리스 사전 분포는 단변량 사전 분포의 곱인 π(μ,σ2)∝1/σ2입니다. 이것은 다음 주변 사후 분포로 이어집니다.

μ|Y∼tn−1(Y¯,σ^2/n).(4.5)

이 사전 분포를 사용하면 사후 분포가 빈도주의 표집 분포(frequentist sampling distribution)와 일치하며, 따라서 신용 구간이 신뢰 구간 등과 일치하게 됩니다.

### 4.1.2 124\. 두 정규 평균의 비교 (Comparison of two normal means)

이표본 검정(two-sample test)은 두 그룹 간의 평균을 비교합니다. 예를 들어, 도보로 경로를 걷는 우편 배달부 _n_1명과 차를 타고 이동하는 우편 배달부 _n_2명의 무작위 표본의 혈압을 측정하여 이들 그룹 간에 평균 혈압 차이가 있는지 결정하는 실험을 생각해 볼 수 있습니다. i\=1,...,n1에 대해 Yi∼iidNormal(μ,σ2)로 두고 i\=n1+1,...,n1+n2\=n에 대해 Yi∼iidNormal(μ+δ,σ2)로 두어 _δ_가 평균 차이이자 관심 모수가 되도록 합니다. j\=1,2 그룹의 관측치 _n_ _j_ 개의 표본 평균을 Y¯j로 나타내고, 그룹별 분산 추정량을 s^12\=∑i\=1n1(Yi−Y¯1)2/n1 및 s^22\=∑i\=n1+1n1+n2(Yi−Y¯2)2/n2로 둡니다.

**분산이 고정된 조건부 분포 (Conditional distribution with the variance fixed)**: 분산이 주어지고 균등 사전 분포(flat prior) π(μ,δ)∝1이 주어졌을 때, 평균 차이의 사후 분포는 다음과 같이 나타낼 수 있습니다.

δ|Y∼Normal(Y¯2−Y¯1,σ2\[1n1+1n2\]).(4.6)

일표본의 경우와 마찬가지로, 정규 분포의 분위수를 사용하여 가설의 사후 구간과 확률을 계산할 수 있습니다. 일표본의 경우처럼 신용 집합 및 기각 규칙은 수치적으로는 고전적 신뢰 구간 및 단측 z-검정과 일치하지만 해석은 다릅니다.

**모르는 분산 (Unknown variance)**: (μ,δ,σ2)에 대한 제프리스 사전 분포는 다음과 같습니다([2.4절](./10-chapter2.md#sec2_4)).

π(μ,δ,σ2)∝1(σ2)2.(4.7)

[부록 A.3](./18-appA.md#secA_3)은 _μ_와 _σ_2 모두에 대해 적분한 _δ_의 주변 사후 분포가 다음과 같음을 보여줍니다(다중 선형 회귀의 특수한 경우로 [4.2절](./12-chapter4.md#sec4_2) 참조).

δ|Y∼tn\[Y¯2−Y¯1,σ^2(1n1+1n2)\],(4.8)

여기서 σ^2\=(n1s^12+n2s^22)/n은 합동 분산 추정량(pooled variance estimator)입니다. 일표본 모델과 마찬가지로 알려진 분산 대 모르는 분산 경우의 사후 분포 간 차이는 _σ_2의 추정치가 사후 분포에 삽입되고 가우스 분포가 자유도 _n_의 스튜던트 t-분포로 대체된다는 것입니다. 일표본 경우와 같이, 대안적 제프리스 사전 분포 π(μ,σ2)∝1/σ2를 사용하면 빈도주의 표집 분포와 일치하는 사후 분포를 제공합니다.

δ|Y∼tn−2\[Y¯2−Y¯1,σ^2(1n1+1n2)\].(4.9)

베이지안 분석에서는 _σ_2의 추정치를 단순히 “플러그인(plug in)”하지 않았습니다. 오히려 그 불확실성을 설명하고 _μ_ 및 _σ_2에 대해 주변화함으로써 _δ_의 사후 분포가 _δ_ 척도에 맞춰진 _σ_2의 자연스러운 추정량을 갖게 되었습니다. [목록 4.1](./12-chapter4.md#list4_1)은 이 방법을 구현합니다.

목록 4.1 제프리스 사전 분포를 사용하여 두 정규 평균을 비교하는 R 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Y1 is the vector of length n1 with data for group 1
2 # Y2 is the vector of length n2 with data for group 2
3 
4 # Statistics from group 1
5  Ybar1 <- mean(Y1)
6  s21 <- mean((Y1=Ybar1)ˆ2)
7  n1 <- length(Y1)
8 
9 # Statistics from group 2
10  Ybar2 <- mean(Y2)
11  s22    <- mean((Y2=Ybar2)ˆ2)
12  n2     <- length(Y2)
13 
14 # Posterior of the difference assuming equal variance
15  delta hat <- Ybar2=Ybar1
16  s2 <- (n1*s21 + n2*s22)/(n1+n2)
17  scale <- sqrt(s2)*sqrt(1/n1+1/n2)
18  df <- n1+n2
19  cred int <- delta hat + scale*qt(c(0.025,0.975),df=df)
20 
21 # Posterior of delta assuming unequal variance using MC sampling
22  mu1 <- Ybar1 + sqrt(s21/n1)*rt(1000000,df=n1)
23  mu2 <- Ybar2 + sqrt(s22/n2)*rt(1000000,df=n2)
24  delta <- mu2=mu1
25 
26  hist(delta,main=”Posterior distribution of the difference in means”)
27  quantile(delta,c(0.025,0.975)) # 95% credible set
```

**비등분산 (Unequal variance)**: 두 그룹의 분산이 동일하다는 가정이 위반되는 경우, 이표본 모델을 i\=1,...,n1에 대해 Yi∼iidNormal(μ1,σ12)로, i\=n1+1,...,n1+n2에 대해 Yi∼iidNormal(μ2,σ22)로 확장할 수 있습니다. 두 그룹 간에 공유되는 모수가 없으므로, 각 그룹에 대해 별도로 일표본 모델을 적용하여 다음을 얻을 수 있습니다.

μj|Y∼indeptnj(Y¯j,sj2/nj)(4.10)

(j\=1,2에 대하여), 그리고 평균 차이 δ\=μ2−μ1의 사후 분포가 이어집니다. _δ_의 사후 분포는 두 개의 스튜던트 t-분포를 따르는 확률변수의 차이이며, 일반적으로 간단한 형태를 가지지 않습니다. 그러나 [목록 4.1](./12-chapter4.md#list4_1)에서와 같이 몬테카를로 표집(Monte Carlo sampling)을 사용하여 임의의 정밀도로 사후 분포를 근사화할 수 있습니다. 125\. 

## 4.2 126\. 선형 회귀 (Linear regression)

응답(종속 변수, 결과 또는 레이블이라고도 함) _Y_ _i_ 및 공변량(독립 변수, 예측 변수, 특징 또는 입력이라고도 함) Xi1,...,Xip를 갖는 다중 선형 회귀 모델(multiple linear regression model)은 다음과 같습니다.

Yi\=∑j\=1pXijβj+εi,(4.11)

여기서 β\=(β1,...,βp)T는 회귀 계수(regression coefficients)이며 오차(잔차(residuals)라고도 함)는 εi∼iid Normal(0,σ2)입니다. 모든 _i_에 대해 Xi1\=1이라고 가정하므로 _β_1은 절편(intercept)(즉, 다른 모든 공변량이 0일 때의 평균)입니다.

회귀 모델은 응답 _Y_ _i_가 연속형이라고 가정하지만, 공변량으로는 연속형 또는 범주형 변수를 모두 수용할 수 있습니다. 범주형 예측 변수는 일반적으로 이진 더미 변수(binary dummy variables)로 포함되며, 평균 비교 및 분산 분석(ANOVA, Analysis of Variance) 모델을 특수한 경우로 포함할 수 있습니다. 절편 항(p\=1)만 있고 _β_1을 _μ_로 다시 레이블링하면, 선형 회귀 모델은 [4.1.1절](./12-chapter4.md#sec4_1_1)의 단일 표본 평균 모델로 단순화됩니다. [4.1.2절](./12-chapter4.md#sec4_1_2)의 이표본 평균 모델은 p\=2인 특수한 경우로, 관측치 _i_가 두 번째 그룹에 속하면 1이고 그렇지 않으면 0인 이진 더미 변수로 Xi2를 설정하고, 모수를 β1\=μ, β2\=δ로 다시 레이블링한 것입니다. 유사하게, _K_ 레벨을 가진 범주형 변수는 Xi1\=1로 설정하고 관측치 _i_가 그룹 _k_에 속하면 1, 그렇지 않으면 0과 같은 K−1개의 더미 변수 Xi2,...,XiK를 포함하여 각 레벨에 대해 서로 다른 평균을 갖는 ANOVA 프레임워크를 사용하여 포함될 수 있습니다.

계수 _β_ _j_는 j번째 공변량과 연관된 기울기(slope)입니다. 이 하위 섹션의 나머지 부분에서는 (절편 항을 제외한) 모든 p−1 개의 공변량이 평균이 0이고 분산이 1이 되도록 표준화되었다고 가정할 것입니다. 이렇게 하면 공변량의 척도(scales)를 고려하지 않고도 사전 분포를 지정할 수 있습니다. 즉, 원래의 공변량 _j_인 X\~ij가 표본 평균 X¯j 및 표준편차 s^j를 가질 경우, Xij\=(X\~ij−X¯j)/s^j로 설정합니다. 표준화한 후, 기울기 _β_ _j_는 원래 공변량의 1표준편차 단위(s^j) 증가에 해당하는 평균 응답의 변화로 해석됩니다. 마찬가지로, βj/s^j는 원래 공변량이 1만큼 증가할 때 연관된 평균 응답의 예상 증가량입니다. 모델은 실제로 p+1 개의 모수(_p_ 개의 회귀 계수 및 분산 _σ_2)를 가지지만, 여기서는 모델의 총 모수 개수가 아니라 회귀 모수의 개수로 _p_를 일시적으로 사용합니다.

회귀 모델을 선택하고 결과를 해석할 때 핵심 개념은 다중공선성(collinearity)입니다. 다중공선성은 일반적으로 공변량 Xij 간의 상관관계를 의미합니다. _β_ _j_의 해석은 다른 모든 공변량이 고정된 상태에서 Xij를 증가시킬 때의 효과임을 상기해 보십시오. 하지만 Xij가 다른 공변량과 강하게 상관되어 있다면 이를 상상하기 어렵습니다. 기온과 열지수를 공변량으로 삼는 극단적인 경우를 생각해 보십시오. 기온에 대한 기울기를 해석하려면 열지수가 고정된 상태에서 기온이 증가하는 것을 상상해야 하는데, 이는 직관에 어긋납니다.

어떤 공변량의 _n_ 개의 관측치가 다른 공변량들의 _n_ 개 값들의 완벽한 선형 결합(linear combinations)이라면, 해당 공변량들은 선형 종속(linearly dependent)이라고 합니다. 공변량들이 선형 종속이 아니면 선형 독립(linearly independent)이라고 합니다. 예측 변수의 수가 관측치의 수보다 많은 경우(p\>n), 공변량들은 자동으로 선형 종속이 됩니다. 아래에서 논의하듯이 적용할 수 있는 방법론은 공변량이 선형 독립인지 여부에 따라 다릅니다.

### 4.2.1 127\. 고전적 접근법 (Classical approaches)

β에 대한 비-베이지안 추정량(estimators)은 단순한 문제를 위한 고전적인 최소 제곱법(least squares)에서부터 더 복잡한 설정을 위한 현대적인 벌점화 회귀(penalized regression) 방법에 이르기까지 다양합니다. 다음 섹션에서 설명하는 베이지안 접근법은 흔히 벌점 항(penalty term)에 상응하는 사전 분포를 가진 벌점화 회귀 추정량과 동일한 점 추정량(point estimators)을 제공합니다. 물론, 동일한 점 추정량을 갖는 경우라도 베이지안과 고전적 접근법 사이의 불확실성에 대한 해석은 다릅니다.

β의 고전적 추정량은 다음과 같은 최소 제곱 추정량(least squares estimator)입니다.

β^LS\=argminβ∑i\=1n(Yi−∑j\=1pXijβj)2.(4.12)

이 추정량은 공변량이 선형 독립일 때만 유효합니다. β^LS의 표집 분포는 _σ_2을 알 때는 정규 분포를 따르고, 일반적인 경우처럼 _σ_2을 추정해야 할 때는 스튜던트 t-분포를 따릅니다.

_n_ 개의 관측치가 있는 선형 모델에 대한 가능도 함수(likelihood function)는 평균이 ∑j\=1pXijβj이고 분산이 _σ_2인 _n_ 개의 가우스 PDF의 곱입니다.

f(Y|β,σ2)\=∏i\=1n12πσexp\[−12σ2(Yi−∑j\=1pXijβj)2\].(4.13)

가능도를 최대화하는 것은 음의 로그 가능도를 최소화하는 것과 동등하므로, 최대 가능도 추정량(maximum likelihood estimator)은 최소 제곱 추정량과 동일합니다. 즉, β^MLE\=β^LS입니다.

벌점화 회귀는 강한 다중공선성이나 다수의 공변량이 있는 까다로운 문제에 자주 사용됩니다. 능선 회귀 추정량(ridge regression estimator) \[[79](./19-ref01.md#refbib79)\]은 다음과 같습니다.

β^R\=argminβ∑i\=1n(Yi−∑j\=1pXijβj)2+λ∑j\=2pβj2.(4.14)

벌점 항 λ∑j\=2pβj2의 추가는 0에 가까운 추정치가 나오도록 유도하고 신뢰하기 힘들 정도로 큰 추정치를 억제/처벌함으로써 안정성을 더합니다. 사실, 이 해결책(및 아래의 LASSO)은 p\>n이거나 공변량이 선형 종속인 경우에도 유효합니다. 튜닝 파라미터(tuning parameter) _λ_는 추정량이 0을 향해 얼마나 강하게 축소(shrunk)될지를 결정하며 교차 검증(cross validation)을 통해 선택할 수 있습니다.

능선 회귀는 안정성을 추가하지만 β^R의 요소는 결코 정확히 0이 되지 않으므로, 효과 추정치가 작은 공변량조차도 모두 예측 모델에 포함됩니다. 유명한 LASSO \[[155](./19-ref01.md#refbib155)\] 벌점화 회귀 추정량은 추정치를 0으로 축소함과 동시에 일부 추정치 β^j를 정확히 0으로 만들어, 추정과 변수 선택을 동시에 수행합니다. 그 추정량은 다음과 같습니다.

β^LASSO\=argminβ∑i\=1n(Yi−∑j\=1pXijβj)2+λ∑j\=2p|βj|,(4.15)

여기서도 튜닝 파라미터 _λ_는 교차 검증을 사용하여 선택할 수 있습니다. LASSO 추정량은 능선 추정량과 비교해 제곱 벌점이 절대값 벌점으로 대체된 점만 다릅니다. 그러나 원점에서 절대값의 불연속성은 정확히 0인 해를 제공하며, 따라서 LASSO 회귀의 특성은 능선 회귀의 특성과 크게 다릅니다.

### 4.2.2 128\. 베이지안 선형 회귀를 위한 사전 분포 (Prior distributions for Bayesian linear regression)

베이지안 선형 회귀의 모델 가정과 가능도는 고전적 선형 회귀와 동일합니다. 즉, Yi\=∑j\=1pXijβj+εi이고 여기서 εi∼iid Normal(0,σ2)입니다. 베이지안 분석에는 회귀 계수 _β_ _j_와 분산 _σ_2에 대한 사전 분포가 필요합니다. 이 섹션에서는 여러 가지 옵션을 제공하고 각 옵션을 선호하는 시나리오를 논의합니다. 먼저 고전적 통계의 최소 제곱과 유사하게 소수의 공변량이 있는 회귀에 대한 기본 선택인 부적절한(improper) 제프리스 사전 분포를 소개합니다. 이는 다중공선성이 존재하거나 사전 정보를 통합하는 데 유용한 가우스 사전 분포로 일반화됩니다. 마지막으로 공변량이 많고 상당수가 효과가 없어 회귀 계수가 0에 가까울 것이라는 사전 믿음이 있을 때 유용한 연속 축소(continuous shrinkage) 사전 분포에 대해 논의합니다.

#### 4.2.2.1 제프리스 사전 분포 (Jeffreys prior)

_σ_2이 조건부로 주어졌을 때, 제프리스 사전 분포는 π(β)∝1입니다. 이 부적절한 사전 분포는 최소 제곱 추정량이 유효하기 위한 조건, 즉 공변량이 선형 독립이어야 한다는 조건과 동일한 조건 하에서 적절한 사후 분포를 도출합니다. β의 사후 분포는 평균이 β^LS이고 공분산이 _σ_2 및 공변량 Xij에 의존하는 다변량 정규 분포(multivariate normal)입니다. 사실, β의 사후 분포는 최소 제곱 추정량의 표집 분포와 동일합니다. 따라서 이 모델의 사후 신용 구간은 오차 분산이 알려진 최소 제곱 분석의 신뢰 구간과 수치적으로 일치합니다.

_σ_2을 모를 때, 제프리스 사전 분포는 π(β,σ2)∝(σ2)−p/2−1입니다([2.4절](./10-chapter2.md#sec2_4)). n\>p이고 공변량이 선형 독립일 때 이 사전 분포 하에서, _σ_2에 대해 평균을 낸 β의 주변 사후 분포는 위치가 β^LS이고 공분산 행렬이 σ^2 및 공변량에 의존하는 _p_ 차원 스튜던트 t-분포를 따릅니다. [4.2.3절](./12-chapter4.md#sec4_2_3)에 설명된 바와 같이, _σ_2의 추정량과 자유도는 최소 제곱 과정과 약간 다르지만, _n_이 _p_보다 훨씬 큰 경우 이러한 차이는 미미하며, 따라서 신용 구간은 대부분의 경우 최소 제곱 신뢰 구간과 근사하게 일치합니다. 대안적으로 독립적인 제프리스 사전 분포의 곱 π(β,σ2)∝1/σ2는 다음을 제공합니다.

β|Y∼tn−p(β^LS,Σ^)(4.16)

여기서 Σ^는 최소 제곱 분석에서 사용된 β^LS의 표집 분포의 추정된 공분산입니다(세부 사항은 (4.23) 참조). 즉, 이 사전 분포 하에서 β의 사후 분포는 β^LS의 표집 분포와 일치합니다.

#### 4.2.2.2 가우스 사전 분포 (Gaussian prior)

관측치보다 예측 변수가 더 많은 고차원 사례의 경우, 부적절한 제프리스 사전 분포는 유효한 사후 분포를 도출하지 못합니다. 부적절성은 항상 적절한 사후 분포로 이어지는 적절한 사전 분포를 사용하여 해결할 수 있습니다. 적당한 수의 _p_가 있는 덜 극단적인 상황에서도 적절한 사전 분포는 사후 분포를 안정화시키고 부적절한 사전 분포보다 더 나은 결과를 제공할 수 있습니다.

β에 대한 켤레 사전 분포(_σ_2이 조건부로 주어짐)는 다변량 정규 분포입니다. 가장 일반적인 접근 방식은 절편에 대해 β1|σ2∼Normal(μ1,σ2ω2)이고 j\>1인 기울기에 대해 βj|σ2∼iidNormal(μj,σ2τ2)로 하여 β의 요소들에 대한 사전 독립을 가정하는 것입니다. 사전 분산에 _σ_2을 포함시켜 응답의 척도를 반영하지만, 필수적인 것은 아닙니다. _ω_와 _τ_가 작고 _μ_ _j_가 사전 지식을 바탕으로 선택된 경우 이 사전 분포는 유익(informative)할 수 있습니다. 그러나 일반적으로 사전 분포는 모든 _j_에 대해 μj\=0이고 _ω_와 _τ_가 큰 값으로 설정되거나 무정보적(uninformative) 사전 분포가 주어지면서 무정보적입니다. μj\=0이 일반적이므로, 다르게 명시되지 않는 한 사전 평균은 0으로 가정합니다.

129\. 1/ω2\=0이고 λ\=1/τ2인 이 사전 분포 하에서, 사후 평균(그리고 MAP) 추정량은 능선 회귀 추정량(ridge regression estimator) \[[79](./19-ref01.md#refbib79)\]입니다.

E(β|Y)\=βR\=argminβ∑i\=1n(Yi−∑j\=1pXijβj)2+λ∑j\=2pβj2.(4.17)

능선 회귀는 예측 변수의 수가 많거나 공변량에 다중공선성이 있을 때 최소 제곱 문제를 안정화시키기 위해 자주 사용됩니다. 능선 회귀에서는 교차 검증을 기반으로 튜닝 파라미터 _λ_를 선택할 수 있습니다. 완전한 베이지안 분석에서, _τ_2은 무정보적 사전 분포를 제공하기 위해 큰 값으로 고정되거나, [목록 4.2](./12-chapter4.md#list4_2)에서와 같이 조건부 켤레 역감마 사전 분포가 주어져 데이터가 계수를 0으로 얼마나 수축할지를 결정하게 할 수 있습니다. _τ_2(그리고 따라서 λ\=1/τ2)에 사전 분포가 주어진 경우, 절편 항 _β_1은 다른 회귀 계수와 다른 역할을 하므로 다른 분산이 주어져야 합니다.

목록 4.2 가우스 사전 분포를 사용하는 다중 선형 회귀에 대한 JAGS 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood, note that
2 # inprod(X[i,],beta[]) = X[i,1]*beta[1]+...+X[i,p]*beta[p]
3  for(i in 1:n){
4     Y[i] ˜ dnorm(inprod(X[i,],beta[]),taue)
5  }
6 # Priors
7  beta[1] ˜ dnorm(0,0.001) #X[i,1]=1 for the intercept
8  for(j in 2:p){
9     beta[j] ˜ dnorm(0,taub*taue)
10  }
11  taue ˜ dgamma(0.1, 0.1)
12  taub ˜ dgamma(0.1, 0.1)
```

켤레 분포는 아니지만 해석이 더 용이한 _τ_2에 대한 사전 분포는 \[[172](./19-ref01.md#refbib172)\] 및 \[[168](./19-ref01.md#refbib168)\]의 _R_2 사전 분포입니다. 이들은 평균 ηi\=∑j\=1pXijβj에 의해 설명되는 분산의 비율을 다음과 같이 정의합니다.

R2\=Var(ηi)Var(Yi)\=Var(ηi)Var(ηi)+σ2.(4.18)

응답이 표준화되어 β1\=0이고, 공변량이 평균이 0이고 분산이 1이 되도록 표준화되었다고 가정하면([부록](./18-appA.md#appA) 참조), _β_ _j_ 및 공변량의 분포에 대한 _μ_ _i_의 분산은 (p−1)σ2τ2이므로, 다음과 같습니다.

R2\=(p−1)τ2(p−1)τ2+1 즉, τ2\=R2(p−1)(1−R2).

결정계수(coefficient of determination) R2∈(0,1)은 모델의 복잡도를 측정하는 지표이며, R2\=0과 R2\=1은 각각 단순한 모델과 복잡한 모델에 대응됩니다. 그들은 _τ_2에 직접 사전 분포를 두기보다는 더 해석하기 쉬운 사전 분포 R2∼Beta(a,b)를 가정합니다. _R_2는 _τ_2의 일대일 변환이므로, 이는 _τ_2에 사전 분포를 유도합니다. _τ_2의 유도된 사전 분포는 일반화 베타 프라임 분포(generalized beta prime distribution)임이 알려져 있습니다([부록](./18-appA.md#appA) 정의). 기본적으로 적당한 선택은 a\=b\=1이며, 이는 [목록 4.3](./12-chapter4.md#list4_3)과 같이 _R_2에 균등 사전 분포를 제공합니다. 대안으로는 a\=b\=0.5가 있으며, 이는 \[[54](./19-ref01.md#refbib54)\]에서와 같이 표준편차 _τ_에 절반-코시(half-Cauchy) 사전 분포([부록](./18-appA.md#appA) 참조)를 유도합니다. 130\. 

목록 4.3 R2 사전 분포를 사용하는 다중 선형 회귀에 대한 JAGS 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){
3     Y[i] ˜ dnorm(inprod(X[i,],beta[]),taue)
4  }
5 # Priors
6  beta[1] ˜ dnorm(0,0.001) #X[i,1]=1 for the intercept
7  for(j in 2:p){
8     beta[j] ˜ dnorm(0,taub*taue)
9  }
10  taue ˜ dgamma(0.1, 0.1)
11  taub <- (p=1)*(1=R2)/R2
12  R2 ˜ dbeta(1,1)
```

#### 4.2.2.3 이중 지수 사전 분포 (Double exponential prior)

수많은 예측 변수를 가진 회귀에서 종종 _p_ 개의 예측 변수 대부분이 응답에 거의 영향을 미치지 않는다고 가정합니다. 가우스 사전 분포는 사전 분산을 작게 하여 이러한 사전 믿음을 반영할 수 있지만, 작은 사전 분산은 부정적인 부작용을 갖습니다. 즉, 중요한 회귀 계수의 사후 평균을 0으로 축소시키고 편향(bias)을 유발합니다. 이에 대한 대안 \[[120](./19-ref01.md#refbib120)\]은 _β_ _j_에 이중 지수(double exponential) 사전 분포를 사용하는 것입니다. 이 분포는 대부분의 예측 변수가 응답에 영향을 미치지 않는다는 사전 믿음을 반영하기 위해 0에서 정점을 가지지만([그림 4.2](./12-chapter4.md#fig4_2)), 소수의 예측 변수가 강한 영향을 미친다는 사전 믿음을 반영하기 위해 꼬리가 두껍습니다. [목록 4.4](./12-chapter4.md#list4_4)는 이 모델을 구현하기 위한 JAGS 코드를 제공합니다(다만 R 패키지 BLR \[[41](./19-ref01.md#refbib41)\]을 사용하는 것이 아마 더 빠를 것입니다). 영점(zero)에 집중된 이러한 사전 분포의 개념은 8.1절에서 다른 분포들로 확장됩니다.

![A posterior density for beta is shown under Gaussian and B L A S S O priors. Both curves peak at beta equal to 0, but the B L A S S O curve is sharper with heavier tails, while the Gaussian curve is smoother and more rounded, tapering more quickly toward zero away from the centre.](./images/fig4_2.jpg)

그림 4.2 **가우스 사전 분포와 이중 지수 사전 분포의 비교.** 동일한 사분위수 범위(interquartile range)를 갖도록 설정된 표준 정규 분포와 이중 지수(double exponential)/BLASSO PDF가 플롯되어 있습니다. [본문으로 돌아가기.⏎](chapter4)

모델이 Yi|β,σ2∼Normal(∑j\=1pXijβj,σ2)이고, 절편에 대해서는 부적절한 사전 분포를 가지며 이중 지수 사전 분포 βj∼iidDE(λ/σ2) (PDF π(βj)∝exp(−λ2σ2|βj|))를 따른다고 가정하면, 사후 확률 최대화(MAP) 추정량(사후 확률을 최대화하는 것은 음의 로그 사후 확률의 2배를 최소화하는 것과 같으므로)은 정확히 식 (4.15)의 LASSO 추정량이 되며, 따라서 이중 지수 사전 분포는 흔히 베이지안 LASSO(Bayesian LASSO) 사전 분포라고 불립니다. 이 추정량의 매력적인 특징은 일부 추정치 β^j가 정확히 0으로 설정될 수 있으며, 이를 통해 추정과 동시에 변수 선택을 수행한다는 것입니다. 사후 최빈값(posterior mode)은 0이 될 수 있지만 사후 평균은 절대 0이 되지 않습니다. 또한 사전 분포가 연속형 PDF이므로 계수가 정확히 0이 될 사후 확률은 0이며, 따라서 이 모델의 모든 MCMC 표본은 예외 없이 0이 아닌 값을 가지게 됩니다.

목록 4.4 베이지안 LASSO에 대한 JAGS 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){
3     Y[i] ˜ dnorm(inprod(X[i,],beta[]),taue)
4  }
5 # Priors
6  beta[1] ˜ dnorm(0,0.001)
7  for(j in 2:p){
8     beta[j] ˜ ddexp(0,taub*taue)
9  }
10  taue ˜ dgamma(0.1, 0.1)
11  taub ˜ dgamma(0.1, 0.1)
```

### 4.2.3 행렬 표기법을 사용한 선형 회귀 (Linear regression in matrix notation)

선형 회귀 모델은 행렬 표기법으로 편리하게 작성할 수 있습니다. 베이지안 선형 회귀를 구현하는 데 행렬 표기법이 반드시 필요한 것은 아니지만, 가우스 사전 분포에 대해 이 간결한 표기법은 사후 분포의 닫힌 형태(closed-form) 수식으로 이어져 MCMC의 필요성을 없앱니다. 길이가 _n_인 응답 벡터를 Y\=(Y1,...,Yn)T라 하고, 계획 행렬(design matrix) **X**를 첫 번째 열이 절편을 위한 1 벡터이고 _j_ 열의 원소가 X1j,...,Xnj인 n×p 행렬이라고 합시다. 그러면 선형 회귀 모델은 다음과 같습니다.

Y∼Normal(Xβ,σ2In)(4.19)

131\. 132\. 여기서 In은 주대각선(diagonal)이 1이고(모든 응답의 분산이 _σ_2이 됨), 비대각선(off the diagonal)이 0인(응답들이 상관관계를 가지지 않음) n×n 항등 행렬(identity matrix)입니다.

**고전적 접근법 (Classical approaches)**: 행렬 표기법으로 나타낸 일반적인 최소 제곱 추정량은 다음과 같습니다.

β^LS\=(XTX)−1XTY.(4.20)

최소 제곱 추정량은 XTX가 완전 계수(full rank)일 때만 유일하며(즉, p<n이고 **X**의 열 중 어느 것도 중복되지 않으며 (XTX)−1이 존재할 때), 따라서 XTX가 완전 계수가 아니면 추정량이 정의되지 않는다는 점에 유의하십시오. **X**가 완전 계수이고 _σ_2을 안다고 가정할 때, 표집 분포는 β^LS∼Normal(β0,σ2(XTX)−1)이며, 여기서 β0은 참값(true value)입니다. 일반적으로 _σ_2은 잔차 분산(residual variance)을 기반으로 추정되며, 이는 빈도주의 신뢰 구간 및 p-값을 구성하는 데 사용되는 β^LS의 표집 분포에 스튜던트 t-분포를 이끕니다.

능선 회귀 추정량 또한 닫힌 형태로 작성할 수 있습니다.

β^R\=(XTX+λD)−1XTY(4.21)

여기서 **D**는 첫 번째 주대각선 원소인 절편은 0이고 나머지 주대각선 원소는 1인 p×p 대각 행렬입니다. _λ_가 주어질 때, β^R의 표집 분포 역시 불확실성 수량화에 사용될 수 있는 스튜던트 t-분포입니다. 불행히도 LASSO 추정량 β^LASSO는 닫힌 형태를 갖지 않으므로 계산을 위해 정교한 수치 최적화 루틴(numerical optimization routine)이 필요하며 표집 분포도 복잡합니다.

목록 4.5 완전한 제프리스 사전 분포 하에서 베이지안 선형 회귀에 대한 R 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 #   This code assumes:
2 #   Y is the vector of length n with the observations
3 #   X is the n x p matrix of covariates
4 #   The first column of X is all ones for the intercept
5 
6 # Compute posterior mean and 95% interval
7  beta_mean    <- solve(t(X)%*%X)%*%t(X)%*%Y
8  sig2_hat     <- mean((Y=X%*%beta mean)ˆ2)
9  beta_cov     <- sig2 hat*solve(t(X)%*%X)
10  beta_scale  <- sqrt(diag(beta cov))
11  df <- length(Y)
12  beta_025  <- beta_mean + beta_scale*qt(0.025,df=df)
13  beta_975  <- beta_mean + beta_scale*qt(0.975,df=df)
14 
15 # Package the output
16  out           <- cbind(beta mean,beta_025,beta_975)
17  rownames(out) <- colnames(X)
18  colnames(out) <- c(”Mean”,”Q 0.025”,”Q 0.975”)
```

**제프리스 사전 분포 (Jeffreys prior)**: _σ_2이 조건부로 주어졌을 때, 제프리스 사전 분포는 π(β)∝1입니다. 이 부적절한 사전 분포는 XTX가 완전 계수일 때만 적절한 사후 분포를 이끌어내는데, 이는 최소 제곱 추정량이 유일하기 위한 조건과 동일합니다. 이 조건이 충족된다고 가정하면 사후 분포는 다음과 같습니다.

β|Y,σ2∼Normal\[β^LS,σ2(XTX )−1\].(4.22)

사후 평균은 최소 제곱 해이며 사후 공분산 행렬은 최소 제곱 추정량의 표집 분포 공분산 행렬입니다. 따라서 이 모델의 사후 신용 구간은 오차 분산이 알려진 최소 제곱 분석의 신뢰 구간과 수치적으로 일치합니다.

133\. _σ_2을 모를 때 제프리스 사전 분포는 π(β,σ2)∝(σ2)−p/2−1입니다([2.4절](./10-chapter2.md#sec2_4)). XTX가 완전 계수라고 가정할 때, [부록 A.3](./18-appA.md#secA_3)은 β|Y∼tn\[β^LS,σ^2(XTX)−1\]임을 보여줍니다(여기서 σ^2\=(Y−Xβ^LS)T(Y−Xβ^LS)/n). 즉, β의 주변 사후 분포는 위치가 β^LS, 척도 행렬이 σ^2(XTX)−1, 자유도가 _n_인 _p_ 차원 스튜던트 t-분포를 따릅니다. 중규모 이상의 _p_에 대해 종종 선호되는 대안적 제프리스 사전 분포 π(β,σ2)∝1/σ2 하에서 사후 분포는 다음과 같습니다.

β|Y∼tn−p\[β^LS,σ^2(XTX)−1\].(4.23)

다변량 t 분포의 속성 중 하나는 각 요소의 주변 분포가 단변량 t 분포라는 것이며, 따라서 대안적 제프리스 사전 분포 하에서는 다음과 같습니다.

βj|Y∼tn−p(β^j,sj2)(4.24)

여기서 β^j는 β^LS의 j번째 원소이고 sj2는 σ^2(XTX)−1의 j번째 대각 원소입니다(완전한 제프리스 사전 분포 하에서 자유도는 _n_입니다). 따라서 모든 회귀 계수의 결합 분포(joint distribution)와 각 회귀 계수의 주변 분포 모두 알려진 분포족에 속하므로 MCMC 표집 없이 계산할 수 있습니다. [목록 4.5](./12-chapter4.md#list4_5)는 사후 평균과 95% 구간을 계산하는 R 코드를 제공합니다.

**가우스 사전 분포 (Gaussian priors)**: 관측치보다 예측 변수가 많은 고차원 사례의 경우 부적절한 제프리스 사전 분포는 유효한 사후 분포를 이끌어내지 못하므로 적절한 사전 분포가 필요합니다. β에 대한 켤레 사전 분포(_σ_2이 조건부로 주어짐)는 β|σ2∼Normal(μ,σ2Ω)입니다. 앞서 보았듯이, 우리는 응답의 척도를 설명하기 위해 사전 분산에 _σ_2을 포함시킵니다. 일반적으로 사전 분포는 0을 중심으로 집중되므로 지금부터는 μ\=0으로 설정합니다. 이 사전 분포는 가능도 Y|β,σ2∼Normal(X β,σ2In)와 결합되어 다음과 같은 사후 분포를 제공합니다.

β|Y,σ2∼Normal\[(XTX+Ω−1)−1X TY,σ2(XTX+Ω−1)−1\],(4.25)

이는 사전 분포가 적절한 한(Ω가 양정치(positive definite) 행렬임) 적절한 분포입니다. 제프리스 사전 분포와 마찬가지로 _σ_2을 모르고 조건부 켤레 역감마 사전 분포를 가질 경우, β의 주변 사후 분포는 스튜던트 t-분포를 따릅니다.

사전 공분산 행렬 Ω에 대해서는 몇 가지 선택지가 있습니다. [4.2.2.2절](./12-chapter4.md#sec4_2_2_2)의 독립 사전 분포는 대각 원소가 {ω2,τ2,...,τ2}인 대각 행렬로 Ω를 취하여 얻어집니다. 분산 성분 _τ_2은 큰 값으로 고정되거나 \[[172](./19-ref01.md#refbib172)\] 및 \[[168](./19-ref01.md#refbib168)\]의 _R_2 사전 분포 같은 사전 분포가 주어질 수 있습니다. 대안적으로, g\>0인 젤너의 g-사전 분포(Zellner's g-prior) \[[171](./19-ref01.md#refbib171)\] β|σ2∼Normal\[0,σ2g(XTX)−1\] 하에서 사후 분포 형태가 단순화됩니다. 그러면 조건부 사후 분포는 다음과 같습니다.

β|Y,σ2∼Normal\[cβ^LS,cσ2(XTX )−1\],(4.26)

여기서 c\=1/(g+1)∈(0,1)은 축소 인자(shrinkage factor)입니다. β^LS와 (XTX)−1는 MCMC 표집기 외부에서 한 번만 계산할 수 있으므로 계산 속도가 향상됩니다. 축소 인자 _c_는 사후 평균과 공분산이 0을 향해 얼마나 강하게 수축되는지 결정합니다. 일반적인 선택은 g\=1/n이며 따라서 c\=n/(n+1)이 됩니다. 가우스 분포에 대한 피셔 정보 행렬(Fisher's information matrix)은 역공분산 행렬이므로 사전 분포는 가능도의 1/n에 해당하는 정보를 제공하며, 따라서 이 사전 분포는 단위 정보 사전 분포(unit information prior)라고 불립니다 \[[88](./19-ref01.md#refbib88)\].

### 4.2.4 선형 회귀 예측 (Linear regression prediction)

선형 회귀의 한 가지 용도는 공변량의 새로운 세트인 Xpred\=(X1pred,...,Xppred)에 대한 예측을 수행하는 것입니다. 모델 파라미터가 주어졌을 때, 새로운 응답의 분포는 134\. Ypred|β,σ2∼Normal(∑j\=1pXjpredβj,σ2)입니다. 모수의 불확실성을 적절히 설명하려면 β와 _σ_2의 불확실성에 대해 평균을 구하는 사후 예측 분포(PPD, posterior predictive distribution)([1.5절](./09-chapter1.md#sec1_5))를 사용해야 합니다. MCMC는 각 모수의 s\=1,...,S MCMC 표본에 대한 예측 분포에서 표본을 추출하고(Y(s)|β(s),σ2(s)∼Normal(∑j\=1pXjpredβj(s),σ2(s))), _S_ 개의 추출값 Y(1),...,Y(S)를 사용하여 PPD를 근사화함으로써 PPD에서 표집하는 수단을 제공합니다. 그런 다음 사후 평균과 95% 구간 등 다른 사후 분포와 동일한 방식으로 PPD를 요약합니다. 유사한 접근 방식을 사용하여 [7.1.4절](./15-chapter7.md#sec7_1_4)에서와 같이 결측 데이터(missing data)를 분석할 수 있습니다.

[목록 4.6](./12-chapter4.md#list4_6)은 선형 회귀 예측을 생성하는 JAGS 코드를 제공합니다. 이 예제는 독립적인 가우스 사전 분포를 사용하지만, 다른 사전 분포에 대해서도 예측 단계는 동일할 것입니다. 예측 변수 행렬 Xpred를 JAGS에 전달해야 하며 JAGS는 예측값 Ypred를 반환할 것입니다. JAGS로 예측을 수행하면 표집기 속도가 느려지고 메모리가 소모될 수 있으므로, [목록 4.7](./12-chapter4.md#list4_7)과 같이 먼저 JAGS를 사용해 모수들에 대한 MCMC 표집을 수행한 다음 R에서 예측을 수행하는 것이 종종 더 효율적입니다.

목록 4.6 선형 회귀 예측을 위한 JAGS 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){
3     Y[i] ˜ dnorm(inprod(X[i,],beta[]),taue)
4  }
5 # Priors
6  for(j in 1:p){
7     beta[j] ˜ dnorm(0,0.001)
8  }
9  taue ˜ dgamma(0.1, 0.1)
10 
11 # Predictions
12  for(i in 1:n_pred){
13     Y_pred[i] ˜ dnorm(inprod(X_pred[i,],beta[]),taue)
14  }
15 # User must pass JAGS the covariates Xpred and integer npred
16 # JAGS returns PPD samples of Ypred
```

목록 4.7 선형 회귀 예측을 위해 JAGS MCMC 표본을 사용하는 R 코드. [본문으로 돌아가기.⏎](chapter4)

```R
1 #   INPUTS
2 #   beta samples := S x p matrix of MCMC samples (from JAGS)
3 #   taue samples := S x 1 matrix of MCMC samples (from JAGS)
4 #   Xpred := npred x p matrix of prediction covariates
5 
6 S       <- nrow(beta samples)
7 n_pred  <- nrow(X_pred)
8 Y_pred  <- matrix(NA,S,n_pred)
9 sigma   <- 1/sqrt(taue_samples)
10 
11 for(s in 1:S){
12   Y_pred[s,] <- X_pred%*%beta samples[s,]+rnorm(n_pred,0,sigma[s])
13 }
14 
15 # OUTPUT
16 # Ypred := S x npred matrix of PPD samples
```

### 4.2.5 예제: 집의 마이크로바이옴에 영향을 미치는 요인 (Example: Factors that affect a home's microbiome)

우리는 <http://figshare.com/articles/1000homes/1270900>에서 다운로드한 \[[12](./19-ref01.md#refbib12)\]의 데이터를 사용합니다. 이 데이터는 미 대륙 내 n\=1,059개 주택(결측 데이터가 있는 샘플 제거 후; 결측 데이터 방법은 [7.1.4절](./15-chapter7.md#sec7_1_4) 참조)의 출입구 위 선반에서 채취한 먼지 샘플입니다. 생물정보학 처리(Bioinformatics processing)를 통해 763종(기술적으로는 조작적 분류 단위(operational taxonomic units)) 곰팡이의 존재 여부를 감지합니다. 응답 변수는 샘플에 존재하는 곰팡이 종 수의 로그값이며, 이는 종 풍부도(species richness)의 측정치입니다. 목적은 가정의 종 풍부도에 영향을 미치는 요인을 결정하는 것입니다. 각 주택에 대해 경도, 위도, 연평균 기온, 연평균 강수량, 순 일차 생산력(NPP, net primary productivity), 고도, 단독주택 여부 이진 지표, 주택 내 침실 수 등 8가지 공변량이 이 예제에 포함되어 있습니다. 이 공변량들은 모두 평균이 0이고 분산이 1이 되도록 중심을 맞추고(centered) 척도가 조정되었습니다(scaled).

우리는 가우스 모델인 [목록 4.2](./12-chapter4.md#list4_2)를 먼저 βj∼iidNormal(0,1002) ("균등 사전 분포(Flat prior)")와 적용한 후, βj|σ2,τ2∼iidNormal (0,σ2τ2) 및 τ2∼BP(1,1)를 사용하여 모델 _R_2에 대해 균등 분포를 부여하는 형태("가우스 축소 사전 분포(Gaussian shrinkage prior)")와 적용하고, 135\. [목록 4.4](./12-chapter4.md#list4_4)의 베이지안 LASSO 사전 분포를 적용합니다. 세 가지 모델 각각에 대해 번인(burn-in) 10,000개 표본 및 번인 후 20,000개 표본으로 2개의 MCMC 체인을 실행했습니다. 트레이스 플롯(trace plots, 표시되지 않음)은 훌륭한 수렴(convergence)을 보였으며, 모든 모수와 모든 모델에 대해 유효 표본 크기(effective sample size)가 1,000개를 초과했습니다.

결과는 세 가지 사전 분포에 대해 상당히 유사합니다([그림 4.3](./12-chapter4.md#fig4_3)). 세 가지 모델 모두에서 기온, NPP, 고도 및 단독주택 여부가 가장 중요한 예측 변수였으며, 기온과 NPP가 낮고 고도가 높은 단독주택에서 풍부도가 가장 높을 것으로 추정되었습니다. 세 모델 모두 적합값(fitted value)(즉, **Xβ**의 사후 평균)이 가장 큰 샘플은 미국 버몬트주 몬트필리어(Montpelier, VT)에 위치한 침실 3개짜리 단독주택이었고, 가장 적합값이 작은 샘플은 애리조나주 템피(Tempe, AZ)에 위치한 침실 2개짜리 다세대 주택이었습니다.

![A map shows sample locations across the United States. Surrounding panels display posterior density curves for beta coefficients of multiple predictors including longitude latitude temperature precipitation N P P elevation single family home and number of bedrooms. Each panel compares two models with curves often shifted relative to the vertical zero line, highlighting differences in estimated effect size and direction across variables.](./images/fig4_3.jpg)

그림 4.3 **가정의 마이크로바이옴 풍부도에 대한 회귀 분석.** 첫 번째 패널은 샘플 위치를 보여주고 나머지 패널들은 회귀 계수 _β_ _j_의 사후 분포를 나타냅니다. 세 모델은 _β_ _j_에 대한 사전 분포에 의해 구분됩니다. 균등 사전 분포는 βj∼Normal(0,1002)(실선)이고, 가우스 축소 사전 분포는 모델 _R_2에 균등 분포를 부여하기 위해 τ2∼BP(1,1)과 함께 βj∼iidNormal(0,σ2τ2)을 사용하며(점선), 베이지안 LASSO는 τ2∼InvGamma(0.1,0.1)과 함께 βj∼iidDE(0,σ2τ2)을 사용합니다(짧은 점선). [본문으로 돌아가기.⏎](chapter4)

분석 결과가 사전 분포에 크게 민감하지는 않지만, 몇 가지 주목할 만한 차이가 있습니다. 예를 들어, 균등 사전 분포 하의 사후 분포와 비교할 때 위도(latitude) 기울기의 사후 분포([그림 4.3](./12-chapter4.md#fig4_3)의 우측 상단 패널)는 가우스 축소 모델에 의해 0을 향해 축소되었으며 베이지안 LASSO 사전 분포의 경우 원점(origin) 부근에 더 강하게 집중됩니다. 그러나 이 플롯만으로는 이들 3개 적합 중 어떤 것이 더 선호되는지 불분명합니다. 모델 비교는 [6장](./14-chapter6.md)에서 논의합니다.

## 4.3 일반화 선형 모델 (Generalized linear models)

다중 선형 회귀는 응답 변수가 가우스 분포를 따른다고 가정하므로 평균 응답은 임의의 실수가 될 수 있습니다. 하지만 많은 분석이 이러한 가정에 부합하지 않습니다. 예를 들어, [4.3.1절](./12-chapter4.md#sec4_3_1)에서는 {0,1}의 지지(support)를 갖는 이진 응답을 분석하고, [4.3.3절](./12-chapter4.md#sec4_3_3)에서는 {0,1,2,...} 지지를 갖는 가산(count) 데이터를 분석합니다. 분명히 이러한 데이터는 가우스 분포를 따르지 않으며, 이진 데이터의 평균은 0과 1 사이어야 하고 가산 데이터의 경우 양수여야 하므로 평균이 임의의 실수가 될 수도 없습니다. 일반화 선형 모델(GLM, generalized linear model)은 이러한 비가우스(non-Gaussian) 결과를 처리하기 위해 선형 회귀 개념을 확장합니다. GLM에 대한 깊이 있는 이론(지수족(exponential families), 정규 연결(canonical links) 등; \[[103](./19-ref01.md#refbib103)\] 참조)이 있지만 여기서는 몇 가지 예제를 통해 GLM을 베이지안 프레임워크 내에 투영하는 데 초점을 맞춥니다.

136\. (이 페이지의 그림) 137\. GLM을 선택하는 기본 단계는 (1) 응답 변수의 지지 공간을 결정하고 적절한 모수적 분포족(parametric family)을 선택하는 것, (2) 공변량을 이 분포족을 정의하는 모수와 연결하는 것입니다. 예로서, [4.2절](./12-chapter4.md#sec4_2)의 가우스 선형 회귀 모델을 고려해 보십시오. 응답의 지지가 (−∞,∞)라면 자연스러운 모수적 분포족은 가우스 분포입니다. 물론 이러한 지지를 가진 다른 족들도 존재하며, 데이터에 대한 가우스 분포족의 적합성은 경험적으로 검증해야 합니다. 가우스 족이 일단 선택되면, 공변량은 모수 중 하나 또는 둘 모두, 즉 평균이나 분산에 연결되어야 합니다. 공변량들의 선형 결합을 다음과 같다고 둡시다.

ηi\=∑j\=1pXijβj.(4.27)

선형 예측자(linear predictor) _η_ _i_는 Xij에 따라 (−∞,∞) 내의 임의의 값을 가질 수 있습니다. 따라서 공변량을 평균과 연결하려면 표준 선형 회귀와 마찬가지로 단순히 E(Yi)\=ηi로 설정하면 됩니다. 표준 모델을 완성하기 위해 공변량을 분산과 연결하지 않기로 선택하고 단순히 모든 _i_에 대해 Var(Yi)\=σ2로 설정합니다.

목록 4.8 JAGS 내 여러 GLM에 대한 모델 선언문(Model statements). [본문으로 돌아가기.⏎](chapter4)

```R
1
2 # (a) Logistic regression
3  for(i in 1:n){
4    Y[i]               ˜ dbern(q[i])
5    logit(q[i]) <- inprod(X[i,],beta[])
6  }
7  for(j in 1:p){beta[j] ˜ dnorm(0,0.01)}
8 
9 # (b) Probit regression
10  for(i in 1:n){
11    Y[i]              ˜ dbern(q[i])
12    probit(q[i]) <- inprod(X[i,],beta[])
13  }
14  for(j in 1:p){beta[j] ˜ dnorm(0,0.01)}
15 
16 # (c) Poisson regression
17  for(i in 1:n){
18     Y[i]           ˜ dpois(lambda[i])
19     log(lambda[i]) <- inprod(X[i,],beta[])
20  }
21  for(j in 1:p){beta[j] ˜ dnorm(0,0.01)}
22 
23 # (d) Negative binomial regression
24  for(i in 1:n){
25    Y[i]          ˜ dnegbin(q[i],m)
26    q[i]          <- m/(m + lambda[i])
27    log(lambda[i]) <- inprod(X[i,],beta[])
28  }
29  for(j in 1:p){beta[j] ˜ dnorm(0,0.01)}
30  m ˜ dgamma(0.1,0.1)
31 
32 # (e) Zero=inflated Poisson
33  for(i in 1:n){
34     Y[i]         ˜ dpois(q[i])
35     q[i]         <- Z[i]*lambda[i]
36     Z[i]         ˜ dbern(p[i])
37     log(lambda[i]) <- inprod(X[i,],beta[])
38     logit(p[i])  <- inprod(X[i,],alpha[])
39  }
40  for(j in 1:p){beta[j] ˜ dnorm(0,0.01)}
41  for(j in 1:p){alpha[j] ˜ dnorm(0,0.01)}
42 
43 # (f) Beta regression
44  for(i in 1:n){
45     Y[i]               ˜ dbeta(r*q[i],r*(1=q[i]))
46     logit(q[i])   <- inprod(X[i,],beta[])
47  }
48  for(j in 1:p){beta[j] ˜ dnorm(0,0.01)}
49  r ˜ dgamma(0.1,0.1)
```

선형 예측자와 모수를 연결하는 함수를 연결 함수(link function)라고 부릅니다. 응답에 대한 가능도의 모수를 _θ_ _i_(예: E(Yi)\=θi 또는 Var(Yi)\=θi)라고 합시다. 그러면 연결 함수 _g_는 다음과 같습니다.

g(θi)\=ηi.(4.28)

연결 함수는 모수의 허용 가능한 모든 값에 대해 잘 정의되는(well-defined) 역함수(invertible function)여야 합니다. 예를 들어, 가우스 사례에서 평균에 대한 연결 함수는 g(x)\=x인 항등 함수(identity function)이므로 평균은 모든 실수가 될 수 있습니다. 공변량을 분산과 연결하려면 분산이 양수인지 확인해야 하므로 자연 로그 함수 g(x)\=log(x)가 더 적합합니다. 연결 함수는 유일하지 않으며 사용자가 선택해야 합니다. 예를 들어, 평균의 연결 함수를 g(x)\=x3으로 대체하거나 분산의 연결 함수를 g(x)\=log10(x)으로 대체할 수도 있습니다.

GLM을 베이지안으로 적합하려면 사전 분포를 선택하고 사후 분포를 계산해야 합니다. [4.2절](./12-chapter4.md#sec4_2)의 가우스 데이터에 대해 논의된 회귀 계수의 사전 분포를 GLM에 적용할 수 있습니다. 관측치 수가 공변량 수보다 훨씬 큰 경우, _β_ _j_에 무정보적 정규 사전 분포를 선택할 수 있습니다. 축소 사전 분포인 βj|τ2∼Normal(0,τ2)에 _τ_2에 대한 사전 분포를 부여하면 많은 수의 공변량에도 더 안정적입니다. _τ_2에 대한 사전 분포는 조건부 켤레 역감마 분포이거나 [4.2.2.2절](./12-chapter4.md#sec4_2_2_2)에서처럼 모델 _R_2에 대해 베타 분포를 이끌어내는 더 유연한 \[[168](./19-ref01.md#refbib168)\]의 일반화 베타 프라임 사전 분포(R 패키지 r2d2glmm을 통함)일 수 있습니다. GLM의 사후 분포는 일반적으로 복잡하여 닫힌 형태로 사후 분포를 유도하고 특정 사전 분포가 해석 가능한 평균과 공분산을 지닌 스튜던트 t-사후 분포를 이끌어낸다는 것을 증명하기 어렵습니다. 그러나 가우스 선형 모델에서 발전된 통찰력의 많은 부분은 GLM에도 적용됩니다.

비가우스 응답의 경우, _β_ _j_의 완전 조건부 분포(full conditional distributions)는 보통 켤레 형태를 갖지 않으므로 메트로폴리스 표집(Metropolis sampling)을 사용해야 합니다. 최대 가능도 추정치를 초기값으로 사용할 수 있으며, 그에 해당하는 표준오차가 적절한 제안 분포(candidate distributions)를 제시할 수 있습니다. 이 장의 예제들에서는 MCMC 표집을 수행하기 위해 JAGS를 사용합니다. R은 로지스틱 회귀를 위한 MCMClogit 패키지([4.3.1절](./12-chapter4.md#sec4_3_1))와 같이 베이지안 GLM을 위한 전용 패키지도 가지고 있으며, 이들은 필시 JAGS보다 빠를 것입니다. 표본 크기가 크고 균등 사전 분포를 사용할 때에는, 베이지안 중심 극한 정리(Bayesian central limit theorem)([3.1.3절](./11-chapter3.md#sec3_1_3))를 적용하여 사후 분포를 가우스 분포로 근사화하고(예: R의 glm 함수 사용) MCMC 전체를 피하는 것이 훨씬 더 효율적입니다. 138\. 

### 4.3.1 139\. 이진 결과 (Binary outcome)

실험의 결과가 오직 성공 또는 실패의 지표로 기록될 때 이진 결과(Binary outcomes) Yi∈{0,1}가 자주 나타납니다. 예를 들어, [1.2절](./09-chapter1.md#sec1_2)에서는 응답이 HIV 검사가 양성임을 나타내는 이진 지표였습니다. 이진 변수는 베르누이 분포(Bernoulli distribution)를 따라야 합니다. 베르누이 분포는 성공 확률(success probability)인 단일 모수 Prob(Yi\=1)\=qi∈\[0,1\]를 갖습니다. 모수가 확률이므로, 연결 함수는 입력 범위를 \[0,1\]로, 출력 범위를 \[−∞,∞\]로 가져야 합니다. 아래에서는 이러한 두 가지 연결 함수인 로지스틱(logistic)과 프로빗(probit)을 논의합니다.

**로지스틱 회귀 (Logistic regression)**: 로지스틱 연결 함수는 다음과 같습니다.

g(q)\=logit(q)\=log(q1−q).(4.29)

이 연결 함수는 사건 확률 _q_를 먼저 사건의 오즈(odds)인 q/(1−q)\>0으로 변환하고, 그 다음 오즈에 로그를 취해 임의의 실수가 될 수 있는 로그 오즈(log odds)로 변환합니다. 로지스틱 회귀 모델은 다음과 같이 작성됩니다.

Yi∼indepBernoulli(qi) 그리고 logit(qi)\=ηi\=∑j\=1JXijβj.(4.30)

역 로지스틱 함수(역로짓, expit function)는 g−1(x)\=exp(x)/\[1+exp(x)\]이며, 따라서 모델은 Yi∼indepBernoulli(exp(ηi)/\[1+exp(ηi)\])로 표현될 수도 있습니다. 이 모델에 대한 JAGS 코드는 [목록 4.8a](./12-chapter4.md#list4_8)에 있습니다.

Yi\=1일 사건의 로그 오즈가 공변량에 대해 선형적이기 때문에, _β_ _j_는 다른 모든 공변량이 고정된 상태에서 _X_ _j_가 1만큼 증가할 때 그에 따른 로그 오즈의 증가로 해석됩니다. 유사하게, 다른 모든 공변량이 고정되어 있을 때 _X_ _j_를 1 늘리는 것은 오즈에 exp(βj)를 곱하는 것과 같습니다. 따라서 만약 βj\=2.3이라면, _X_ _j_를 1 늘리는 것은 오즈에 10을 곱하는 것과 같고, βj\=−2.3이라면 _X_ _j_를 1 늘리는 것은 오즈를 10으로 나누는 것과 같습니다. 이 해석은 결과를 전달하고 사전 분포를 지정하는 데 편리합니다. 만약 공변량에서 1의 변화가 큰 변화로 여겨진다면, 표준 정규 사전 분포는 무정보적 사전 분포를 표현하기에 충분한 산포를 가질 수 있습니다. 회귀 계수에 대한 사전 분포는 유도된 성공 확률 _q_ _i_에 대한 사전 분포로 평가할 수 있습니다. [그림 4.4](./12-chapter4.md#fig4_4)는 사전 표준편차 10이 _q_ _i_에 대한 사전 분포를 제공하며, 그 질량의 대부분이 0과 1에 위치함을 보여주는데, 이는 많은 문제에서 비현실적일 수 있습니다. 이것은 사전 예측 점검(prior predictive checks)의 한 예이며, [6.5절](./14-chapter6.md#sec6_5)에서 논의할 것입니다.

![A set of prior density curves is plotted against success probability from 0 to 1. One curve is nearly flat across the interior with slight rises near the boundaries. Another rises more sharply toward both 0 and 1, and a third becomes extremely large at both ends, forming steep spikes. All curves illustrate different prior beliefs about probabilities near the extremes versus the centre.](./images/fig4_4.jpg)

그림 4.4 **로지스틱 회귀 사전 분포**. 성공 확률 _q_ _i_에 대한 사전 분포. logit(qi)\=β1+Xiβ2, Xi∼Normal(0,1), 그리고 β1,β2∼Normal(0,τ2)로 가정합니다. 사전 분포는 τ\=1(실선), τ\=2(점선) 및 τ\=10(짧은 점선)에 대해 플롯되었습니다. [본문으로 돌아가기.⏎](chapter4)

**프로빗 회귀 (Probit regression)**: \[0,1\]에서 \[−∞,∞\]로의 다양한 연결 함수가 가능합니다. 사실, 지지가 \[−∞,∞\]인 임의의 연속 확률 변수의 분위수 함수(역 CDF)라면 충분합니다. 로지스틱 회귀의 연결 함수는 로지스틱 분포의 분위수 함수입니다. 프로빗 회귀에서 연결 함수는 다음과 같은 가우스 분위수 함수입니다.

Yi∼indepBernoulli(qi) 그리고 qi\=Φ(ηi),(4.31)

여기서 Φ는 표준 정규 CDF입니다([목록 4.8b](./12-chapter4.md#list4_8)).


불행히도 프로빗 회귀에서의 회귀 계수 _β_ _j_는 로지스틱 회귀에서의 로그 오즈(log-odds) 해석과 같이 깔끔한 해석을 가지지 않습니다. 그러나 프로빗 회귀는 깁스 표집(Gibbs sampling)을 가능하게 하고 이진 변수 간의 종속성을 모델링하는 데 사용될 수 있으므로 유용합니다. 프로빗 회귀는 잠재 변수(latent variables) Zi∼indepNormal(ηi,σ2)를 지정하고, 관측치 Yi\=1이 _Z_ _i_가 임곗값(threshold) _z_를 초과함을 나타낸다고 가정하는 것과 같습니다. 예를 들어, _Z_ _i_는 환자의 혈압을 나타낼 수 있고, _Y_ _i_는 환자가 고혈압(140을 초과하는 것으로 정의됨)인지 여부를 나타내는 140\. 해당 이진 지표일 수 있습니다. 이 예에서 환자가 고혈압일 확률은 다음과 같습니다.

Prob(Yi\=1)\=Prob(Zi\>z)\=Prob\[(Zi−ηi)/σ\>(z−ηi)/σ\]\=1−Φ\[(z−ηi)/σ\]\=Φ\[(ηi−z)/σ\]\=Φ\[(∑j\=1JXijβj−z)/σ\].

우리는 잠재 변수 _Z_ _i_를 절대 관측할 수 없으므로 임곗값 _z_나 분산 _σ_2을 추정할 수 없습니다. 임곗값에 어떤 상수를 더하고 절편에서 동일한 상수를 빼더라도 사건 확률 _q_ _i_에 영향을 미치지 않기 때문에 임곗값 _z_는 절편과 결합되어 있습니다. 따라서 임곗값은 일반적으로 z\=0으로 설정됩니다. 마찬가지로 동일한 상수로 곱하거나 나누더라도 사건 확률에 영향을 미치지 않으므로, 분산도 일반적으로 σ2\=1로 설정됩니다. 이렇게 하면 일반적인 프로빗 회귀 모델 qi\=Φ(ηi)이 얻어집니다.

이 공식에서 회귀 계수 β는 잠재 변수 _Z_ _i_에 조건부로 켤레 완전 조건부 분포(conjugate full conditional distributions)를 갖습니다. 이는 각 MCMC 반복 시 _Z_ _i_가 대체(imputed)될 경우 깁스 표집을 이끕니다 \[[5](./19-ref01.md#refbib5)\]. 또한, 이진 결과 _Y_ _i_ 간의 종속성은 잠재 변수 _Z_ _i_에 대한 다변량 정규 모델(multivariate normal model)에 의해 유도될 수 있습니다.

### 4.3.2 범주형 결과 (Categorical outcomes)

로지스틱 및 프로빗 회귀는 결과가 두 개 이상인 범주형 데이터를 모델링하도록 일반화될 수 있습니다. 범주형 데이터는 설문조사에서 고객에게 R\=4개의 제품 중 어느 것을 선호하는지 물었을 때 그 응답이 Yi∈{1,...,R}로 코딩되거나, 141\. (예: "매우 동의함", "동의함" 등) 어떤 진술에 동의하는 정도를 묻고 그 응답을 역시 Yi∈{1,...,R}로 코딩하는 경우에 발생할 수 있습니다. 응답 레벨의 순서가 있는지(동의 정도) 또는 없는지(제품 선택)에 따라 모델은 다르지만, 두 경우 모두 주요 목적은 공변량 Xi가 각 응답의 가능성에 어떻게 영향을 미치는지 확인하고 예측을 하는 것입니다.

순서가 없는 응답에 대한 자연스러운 모델은 다항 로지스틱 회귀 모델(multinomial logistic regression model)입니다.

Prob(Yi\=r|β)\=exp(Xiβr)∑k\=1Rexp(Xiβk)(4.32)

β\=(β1,...,βR)에 대해. 이 모델에서 _R_개의 각 레벨은 공변량이 응답 _r_의 확률에 미치는 영향을 결정하는 각기 다른 회귀 파라미터 세트 βr을 갖습니다. 선형 예측자 Xiβr은 확률이 양수가 되도록 지수화되며(exponentiated), 분모의 합은 ∑r\=1RProb(Yi\=r|β)\=1을 보장하여 모든 Xi 및 β에 대해 유효한 PMF를 제공합니다.

이 공식은 다음과 같은 로그 오즈 해석을 유지합니다.

log{Prob(Yi\=r|β)Prob(Yi\=s|β)}\=Xi(βr−βs).

위의 로그 오즈에서 알 수 있듯이 회귀 계수 간의 차이만 추정할 수 있습니다. 과적합(overparameterization)을 방지하기 위해 _R_개의 응답 중 하나(예: r\=1)를 β1\=0을 갖는 기준(baseline)으로 설정합니다. 이 제한 하에 기준과 비교한 카테고리 _r_의 로그 오즈는 log{Prob(Yi\=r|β)Prob(Yi\=1|β)}\=Xiβr이 되며, 따라서 βr은 로지스틱 회귀와 동일하게 해석되지만 모든 비교가 기준 응답(보통 가장 흔한 응답으로 선택됨)과 관련된다는 점에 주의해야 합니다. 사실 R\=2 카테고리만 있고 Yi\=1이 실패, Yi\=2가 성공인 경우, 다항 로지스틱 회귀 모델은 이진 데이터에 대한 표준 로지스틱 회귀 모델로 단순화됩니다.

Yi\=r(동의함)이 어떤 의미에서 Yi\=r−1(동의하지 않음)과 Yi\=r+1(매우 동의함) 사이에 위치하는 순서형 데이터(Ordinal data)는 일련의 로지스틱 회귀로 모델링될 수 있습니다. 이 로지스틱 회귀 시퀀스는 응답자가 먼저 첫 번째 응답을 할지 결정하고, 그렇지 않다면 두 번째 응답을 할지 결정하는 식의 과정을 모방합니다. 수학적으로, 첫 번째 로지스틱 회귀는 첫 번째 카테고리를 선택할 확률 대 나머지 카테고리 모두를 선택할 확률 log{Prob(Yi\=1|β)/Prob(Yi\>1|β)}\=Xiβ1입니다. 두 번째 로지스틱 회귀는 Yi\>1인 관측치 데이터만 사용하여 적합되며 log{Prob(Yi\=2|β)/Prob(Yi\>2|β)}\=Xiβ2입니다. 이는 모든 R−1개의 결정이 내려져 모수 β1,...,βR−1을 얻을 때까지 반복됩니다. 순서형 및 비순서형 로지스틱 회귀 모델 모두 4.3.1절에서 논의된 표준 로지스틱 회귀에 대한 모든 사전 분포를 각 βr에 걸쳐 독립적으로 적용할 수 있습니다. _R_이 큰 경우 4.4절의 방법에 따라 _β_ _r_을 임의 효과(random effects)로 취급하는 것이 유익할 수 있습니다.

프로빗 회귀는 범주형 데이터에도 적용할 수 있습니다. 순서형 데이터에 대한 한 가지 접근법은 각 관측치마다 잠재 연속 변수(latent continuous variable) Zi|β∼Normal(Xiβ,1)가 존재하고 응답이 이 잠재값에 의해 결정된다고 가정하는 것입니다. 이 모델에서는 _R_개의 범주가 있는 경우 L0<L1<...<LR을 만족하는 R+1개의 중단점(breakpoints)이 있으며(여기서 L0\=−∞ 및 LR\=∞), Lk−1<Zi<Lk일 때 Yi\=k입니다. 이는 R\=4이고 빗금 친 확률이 Prob(Yi\=3|β)\=Prob(L2<Zi<L3|β)인 [그림 4.5](./12-chapter4.md#fig4_5)에 시각화되어 있습니다.

![A Gaussian density curve is plotted over Z from about minus 5 to 5. Three vertical lines labelled L one L two and L three mark fixed Z values. A shaded region lies between L two and L three under the right side of the curve, covering the upper tail of the distribution. The curve peaks near Z about 2 and tapers symmetrically toward zero on both sides.](./images/fig4_5.jpg)

그림 4.5 **범주형 데이터를 위한 프로빗 회귀.** 곡선은 Xβ\=2일 때 Z|β∼Normal(Xβ,1)의 PDF입니다. 세로선은 컷포인트(cutpoints) _L_1, _L_2, _L_3에 있으며 곡선 아래의 면적은 Y\=3일 확률입니다. [본문으로 돌아가기.⏎](chapter4)

이 모델의 매력은 범주형 데이터 분석을 우리가 잘 이해하고 있는 가우스 선형 회귀와 연결한다는 점입니다. 예를 들어 _Z_ _i_가 어떤 주제에 대한 응답자의 견해를 측정한 연속형 척도라고 한다면, β는 이 연속형 척도에 대한 공변량의 영향을 요약합니다. 그러나 _Z_ _i_는 관측되지 않으며, 대신 응답자에게 주제를 매우 비호의적, 비호의적, 호의적 또는 매우 호의적으로 평가하도록 요청하고 이 응답을 Yi∈{1,...,4}로 코딩합니다.

142\. 범주형 데이터를 위한 프로빗 회귀 모델은 MCMC 계산에 적합합니다. 이진 데이터를 위한 프로빗 회귀 모델과 마찬가지로, 각 반복마다 _Z_ _i_를 대체(impute)할 수 있고 그런 다음 깁스 표집을 사용하여 β를 업데이트할 수 있습니다. 한 가지 까다로운 점은 컷포인트 _L_ _k_를 선택하는 것입니다. 이들은 모든 _k_에 대해 Lk<Lk+1이 되도록 사전 분포가 주어져야 하며 컷포인트 중 하나는 고정되어야 합니다(예: L1\=0). β의 절편을 증가시키고 모든 컷포인트를 같은 양만큼 증가시키면 동일한 구간 확률을 얻기 때문입니다(즉, [그림 4.5](./12-chapter4.md#fig4_5)에서 x축은 이동하겠지만 빗금 친 부분의 넓이는 영향을 받지 않습니다). 한 예로, L1\=0으로 두고 k\>1에 대해 Lk\=Lk−1+δk로 설정한 후 _δ_ _k_에 감마 사전 분포를 부여하여 컷포인트가 적절한 순서를 유지하도록 할 수 있습니다.

### 4.3.3 가산 결과 (Count outcomes)

Yi∈{0,1,2,...} 지지를 갖는 확률 변수는 시간 구간이나 공간 영역에서 발생하는 사건의 횟수로 종종 나타납니다. 예를 들어, [2.1절](./10-chapter2.md#sec2_1)에서는 시즌별 NFL 뇌진탕 횟수를 분석합니다. 이 장에서는 평균을 공변량의 함수로 모델링하는 데 초점을 맞출 것입니다. 선형 예측자와 평균 사이의 연결은 E(Yi)\=λi≥0을 보장해야 합니다. 자연스러운 연결 함수는 로그 연결(log link)입니다.

log(λi)\=∑j\=1pXijβj.(4.33)

이 모델에서는 다른 모든 공변량이 고정된 상태에서 _X_ _j_가 1만큼 증가할 경우 평균에 exp(βj)가 곱해집니다. 이진 데이터와 달리 가산 데이터의 경우 평균을 지정한다고 해서 가능도가 완전히 결정되지는 않습니다. 아래에서 가능도 함수를 위한 두 가지 분포족, 즉 푸아송(Poisson)과 음이항(negative binomial) 분포에 대해 논의합니다.

143\. **푸아송 회귀 (Poisson regression)**: 푸아송 회귀 모델은 다음과 같습니다.

Yi|λi∼indepPoisson(λi) 여기서 λi\=exp(∑j\=1pXijβj).(4.34)

로지스틱 회귀와 마찬가지로 회귀 계수 β1,...,βp에 [4.2절](./12-chapter4.md#sec4_2)에서 논의된 사전 분포를 부여할 수 있으며, 사후 분포를 탐색하기 위해 메트로폴리스 표집을 사용할 수 있습니다([목록 4.8c](./12-chapter4.md#list4_8)). 푸아송 모델의 중요한 가설은 평균과 분산이 같다는 것입니다. 즉,

E(Yi)\=Var(Yi)\=λi.(4.35)

어떤 횟수의 분포는 그것의 분산이 평균보다 크면 과대산포(over-dispersed) 상태이고(작으면 과소산포(under-dispersed) 상태입니다). 과대산포가 존재한다면 푸아송 모델은 부적절하며, 사후 분포가 모든 변동성의 원천을 정확하게 반영할 수 있도록 다른 모델을 고려해야 합니다.

**음이항 회귀 (Negative binomial regression)**: 과대산포를 수용하는 한 가지 접근 방식은 평균이 1이고 분산이 1/m인 감마 확률 변수 ei∼iidGamma(m,m)를 포함시키는 것입니다. 그러면 다음과 같습니다.

Yi|ei,λi,m∼indepPoisson(λiei).(4.36)

_e_ _i_에 대해 주변화하면, _Y_ _i_는 다음 음이항 분포(negative binomial distribution)를 따릅니다.

Yi|λi,m∼NegBinomial(qi,m)(4.37)

여기서 확률은 qi\=m/(λi+m)이고 크기(size)는 _m_입니다. 크기 _m_은 정수일 필요는 없지만 정수라고 가정하고 성공 확률이 _q_ _i_인 일련의 독립적인 베르누이 시행을 상상한다면, _Y_ _i_는 m번째 성공 전에 발생하는 실패 횟수로 해석될 수 있습니다. 과대산포 처리를 위해 더 중요한 것은 E(Yi)\=λi 및 Var(Yi)\=λi+λi2/m\>λi라는 점입니다. 모수 _m_은 과대산포를 제어합니다. _m_이 크면 ei≈1이 되어 모델은 Var(Yi)≈E(Yi)인 푸아송 회귀 모델로 환원됩니다. 반면 _m_이 0에 가까우면 _e_ _i_의 분산이 크므로 Var(Yi)\>E(Yi)가 됩니다. 과대산포 모수에는 [목록 4.8d](./12-chapter4.md#list4_8)와 같이 감마 사전 분포를 부여할 수 있습니다.

**영-과잉 푸아송 (Zero-inflated Poisson)**: 푸아송 분포에서 또 다른 일반적인 편차는 과도한 0의 수입니다. 예를 들어 _Y_ _i_를 방문자 _i_가 어떤 주립공원에서 잡은 물고기 수라고 가정합시다. 대부분의 _Y_ _i_는 0일 수 있는데, 방문객 중 적은 비율(예: _p_)만이 낚시를 했기 때문이며, 반면 낚시를 한 사람들의 _Y_ _i_ 분포는 평균 _λ_인 푸아송 분포를 따를 수 있습니다. 이 경우 관측치가 0일 확률은 비낚시꾼의 확률인 1−p에, 낚시꾼의 푸아송 확률 0일 확률을 곱한 _p_를 더한 값입니다. 이 시나리오에 해당하는 PMF는 다음과 같습니다.

f(y|p,λ)\={(1−p)+pfP(0|λ)y\=0인 경우pfP(y|λ)y\>0인 경우(4.38)

여기서 _f_ _P_는 푸아송 PMF입니다. 이것은 다음의 2단계 모델과 동등합니다.

Yi|Zi,λ∼Poisson(Ziλ) 그리고 Zi∼ Bernoulli(p)

여기서 _Z_ _i_는 방문자 _i_가 낚시를 했음을 나타내는 잠재 지표이며, 따라서 Zi\=0인 비낚시꾼의 경우 _Y_ _i_의 평균은 0이 됩니다. 이 시나리오에서는 _Z_ _i_를 관측할 수 없지만, 이 2단계 모델은 (4.38)과 동등하면서도 표준 분포들만 사용합니다. 결과적으로 [목록 4.8e](./12-chapter4.md#list4_8)와 같이 0의 질량(mass at zero)과 푸아송 비율 모두에 공변량을 포함하여 JAGS로 모델을 코딩할 수 있습니다. 144\. 

### 4.3.4 예제: NBA 클러치 자유투에 대한 로지스틱 회귀 (Example: Logistic regression for NBA clutch free throws)

[1.6절](./09-chapter1.md#sec1_6) 연습문제 17의 표는 선수 i\=1,...,10에 대한 전체 자유투 성공 비율(qi∈\[0,1\]), 클러치 상황(clutch shots)에서 시도하여 성공한 수(_Y_ _i_), 그리고 클러치 상황에서 시도한 총 횟수(_n_ _i_)를 제공합니다. [그림 4.6](./12-chapter4.md#fig4_6)은 대부분의 선수들이 클러치 상황과 비클러치 상황에서 유사한 성공률을 보이지만, 일부 선수는 압박을 받는 상황에서 성공률이 다소 떨어지는 것을 보여줍니다. 우리는 이 관계를 공식적으로 탐구하기 위해 로지스틱 회귀 모델을 적합(fit)합니다.

![A scatterplot compares clutch percentage with overall percentage for labelled individuals, alongside a one to one reference line and a fitted regression line. Points mostly lie near the reference, with some above or below. A panel of posterior densities for beta shows three curves: a narrow peak for Model two intercept, a broader peak for Model one intercept left of zero, and a wide peak for Model one slope right of zero, illustrating differing parameter estimates across models.](./images/fig4_6.jpg)

그림 4.6 **NBA 자유투에 대한 로지스틱 회귀 분석.** 왼쪽 패널은 각 선수(선수의 이니셜로 표시)의 클러치 상황 슛 성공 비율 대비 전체 자유투 성공 비율을 보여줍니다. 실선은 x\=y 선이고 점선은 모델 2에서 적합된 값(fitted value)입니다. 오른쪽 플롯은 모델 1의 기울기와 절편, 그리고 모델 2의 절편에 대한 사후 밀도입니다. [본문으로 돌아가기.⏎](chapter4)

응답의 지지(support)는 Yi∈{0,1,...,ni}이므로, 이항 가능도(binomial likelihood)인 Yi|pi∼Binomial(ni,pi)를 선택합니다. 여기서 _p_ _i_는 선수 _i_가 클러치 슛을 성공할 확률입니다. 슛 횟수 _n_ _i_를 알 수 있으므로 공변량을 성공 확률 _p_ _i_에 연결합니다. 두 모델은 다음과 같습니다.

1. logit(pi)\=β1+β2Xi
2. logit(pi)\=β1+Xi

여기서 Xi\=logit(qi)는 일반 자유투를 성공할 로그 오즈입니다. 모델 1에서 β1\=0 및 β2\=1이거나 모델 2에서 β1\=0이라면 클러치 성과는 전체 성과와 같습니다. 이것이 사실인지 판단하기 위해 번인 10,000 및 추가 20,000 표본이 있는 두 개의 체인과 모든 모수에 대해 무정보적인 정규(0, 100) 사전 분포를 사용하여 JAGS로 모델을 적합합니다. 모델 1의 모델 사양은 다음과 같습니다.

```R
for(i in 1:10){
  Y[i]               ~ dbinom(p[i],n[i])
  logit(p[i]) <- beta[1] + beta[2]*X[i]
}
beta[1] ~ dnorm(0,0.01)
beta[2] ~ dnorm(0,0.01)
```

145\. 그 결과가 [그림 4.6](./12-chapter4.md#fig4_6)에 플롯되어 있습니다. 모델 1의 경우 기울기가 중심인 1에 확실히 위치하고 절편은 0보다 약간 아래에 중심을 두고 있지만, 두 모수 모두 상당한 불확실성을 가지고 있으므로 pi\=qi인 모델은 여전히 그럴듯합니다. 그러나 모델 2의 절편은 사후 확률 0.96으로 음수이므로, NBA 최고의 선수들조차 압박 상황에서는 평소보다 다소 못한 성과를 보인다는 약간의 증거가 있습니다. [그림 4.6](./12-chapter4.md#fig4_6)(왼쪽 패널)의 적합 곡선(점선)은 1/\[1+exp(−β¯1−Xi)\](여기서 β¯1은 사후 평균)이며, 클러치 성과가 전체 비율보다 몇 퍼센트 포인트 더 낮을 수 있음을 보여줍니다.

### 4.3.5 예제: 마이크로바이옴 데이터에 대한 베타 회귀 (Example: Beta regression for microbiome data)

[4.2.5절](./12-chapter4.md#sec4_2_5)에서 우리는 표본의 마이크로바이옴 다양성을 주택의 특징에 회귀시켰습니다. 다양성은 샘플에 존재하는 L\=763종 수의 로그값으로 측정되었습니다. 그러나 이 측정법은 종의 상대적 풍부도(relative abundance)를 반영하지 못합니다. Ail≥0을 샘플 _i_에서 종 _l_의 풍부도라고 합시다. 다양성을 측정하는 또 다른 방법은 샘플 _i_에서 가장 풍부한 종이 차지하는 총 풍부도의 비율입니다.

Yi\=max{Ai1,...,AiL}∑l\=1LAil∈\[0,1\].(4.39)

이 측정치는 [그림 4.7](./12-chapter4.md#fig4_7)(왼쪽 상단)에 플롯되어 있습니다.

![A histogram shows proportions of the most abundant O T U with many small values and few large ones. A scatterplot compares these proportions with longitude, forming vertical clusters. A long trace plot for r over 30000 iterations shows dense stable fluctuations. Density curves for y from three cities display distinct peaks, with Summerville highest near small y, Greensburg peaking around 0.3, and Junction City highest near 0.4 before all taper toward zero.](./images/fig4_7.jpg)

그림 4.7 **마이크로바이옴 데이터를 위한 베타 회귀.** 왼쪽 위 패널은 가장 풍부한 OTU에 할당된 풍부도 관측 비율의 히스토그램을 보여주고, 오른쪽 위 패널은 이 변수를 표본의 경도(longitude)와 비교하여 플롯한 것입니다. 두 번째 행은 모든 모수에 대한 사후 평균으로 평가된 세 샘플에 대한 집중 모수(concentration parameter) _r_과 적합된 Beta\[r^q^,r^(1−q^)\] 밀도의 트레이스 플롯(두 체인은 서로 다른 회색 음영)을 보여줍니다. [본문으로 돌아가기.⏎](chapter4)

_Y_ _i_는 0과 1 사이이므로 가우스 선형 회귀는 부적절합니다. 한 가지 옵션은 지지가 \[0,1\]인 응답을 (−∞,∞)로 변환하고(예: Yi∗\=logit(Yi)), 가우스 선형 모델을 사용하여 변환된 데이터 Yi∗를 모델링하는 것입니다. 다른 옵션은 비가우스 모델을 사용하여 데이터를 직접 모델링하는 것입니다. 지지가 \[0,1\]인 연속 변수에 대한 자연스러운 모델은 베타 분포입니다. 평균 역시 \[0,1\] 내에 있어야 하므로, 로지스틱 연결을 사용하여 공변량을 평균 응답과 연결할 수 있습니다. 따라서 우리는 다음과 같은 베타 회귀 모델을 적합합니다.

Yi|β,r∼Beta\[rqi,r(1−qi)\] 그리고 logit(qi)\=XiTβ.(4.40)

Yi|β,r는 평균 _q_ _i_와 분산 qi(1−qi)/(r+1)을 가지며, 따라서 r\>0은 평균 _q_ _i_ 주변의 베타 분포 집중도를 결정합니다.

146\. 사전 분포 βj∼Normal(0,100) 및 r∼Gamma(0.1,0.1)를 사용하여, [목록 4.8e](./12-chapter4.md#list4_8)의 코드로 JAGS에서 이 모델을 적합합니다(R의 betareg 패키지를 사용하는 편이 표집이 필시 더 빠를 것입니다). 모델을 적합하기 전에 공변량들은 평균이 0이고 분산이 1이 되도록 표준화되었습니다. 모든 모수에 대한 수렴성이 우수했습니다([그림 4.7](./12-chapter4.md#fig4_7)의 왼쪽 아래 패널은 _r_의 트레이스 플롯을 보여줍니다). [표 4.1](./12-chapter4.md#tbl4_1)의 사후 분포는 동부의 서늘한 지역에 위치한 주택과 다침실 다세대 주택에서 평균적으로 더 많은 다양성(더 작은 _Y_ _i_)이 있음을 나타냅니다.

__표 4.1 **마이크로바이옴 데이터의 베타 회귀**. 회귀 계수 _β_ _j_ 및 집중 모수 _r_에 대한 사후 중앙값 및 95% 구간. [본문으로 돌아가기.⏎](chapter4)__
| 중앙값 (Median) | 95% 구간 (95% Interval) | |
| ------------------ | ------------ | -------------- |
| 절편 (Intercept) | −1.01 | (−1.05, −0.96) |
| 경도 (Longitude) | −0.21 | (−0.28, −0.14) |
| 위도 (Latitude) | −0.08 | (−0.22, 0.05) |
| 기온 (Temperature) | −0.15 | (−0.30, −0.01) |
| 강수량 (Precipitation)| 0.03 | (−0.04, 0.11) |
| NPP | −0.04 | (−0.10, 0.02) |
| 고도 (Elevation) | −0.02 | (−0.09, 0.05) |
| 단독주택 (Single-family home) | 0.07 | ( 0.02, 0.13) |
| 침실 수 (Number of bedrooms) | −0.08 | (−0.13, −0.03) |
| _r_ | 7.97 | ( 7.34, 8.66) |

## 4.4 임의 효과 (Random effects)

표준 선형 회귀 모델은 모든 관측치가 독립적이라고 가정합니다. 데이터가 그룹 단위로 수집된 경우 이 가정은 빈약해집니다. 예를 들어 [그림 4.8](./12-chapter4.md#fig4_8)은 m\=4회 방문 동안 측정된 n\=20명 아이의 턱뼈 밀도 측정값을 보여줍니다. 이러한 반복 측정 설계(repeated measures design)에서 얻은 데이터를 분석하려면 동일한 아동에 대한 후속 측정 간의 유력한 상관관계를 설명해야 합니다. 이 섹션에서는 잠재적 종속성이 있는 관측치를 가지는 이러한 사례와 기타 설정들을 설명하기 위해 임의 효과(random effect)의 개념을 소개합니다. 여기에 소개된 임의 효과 방법론은 베이지안 프레임워크의 가장 강력한 응용 중 하나인 계층 모델링(hierarchical modeling, [7.1절](./15-chapter7.md#sec7_1))의 미리 보기(preview)입니다.

![Bone density trajectories across ages show mostly increasing lines for individual patients. Boxplots by patient display wide variability with many outliers. A histogram of the proportion of variance explained by the random effect peaks around 0.7. Posterior densities for mu compare a random effects model and an I I D model, with the I I D curve narrower and more peaked near 50, while the random effects curve is broader with heavier tails.](./images/fig4_8.jpg)

그림 4.8 **턱뼈 데이터에 대한 일원배치 임의 효과 분석.** 왼쪽 위 패널의 점들은 각 환자에 대한 4번의 방문에서의 뼈 밀도를 보여주고(선으로 연결됨), 오른쪽 위 패널은 관측치(점)와 대상 임의 효과 _α_ _i_의 사후 분포(박스플롯)를 비교하며, 왼쪽 아래 패널은 분산 비율 τ2/(τ2+σ2)의 사후 분포를 나타내고, 마지막 패널은 임의 효과 모델에서의 평균 _μ_의 사후 밀도와 독립 모델(Yij∼iidNormal(μ,σ2))에서의 사후 밀도를 비교합니다. [본문으로 돌아가기.⏎](chapter4)

이 20명의 어린이는 더 큰 모집단의 무작위 표본일 뿐이지만, 이 표본을 사용하여 더 큰 모집단에 대해 추론할 수 있습니다. Yij를 환자 _i_에 대한 j번째 측정값이라고 한다면([그림 4.8](./12-chapter4.md#fig4_8)의 오른쪽 상단 패널처럼 연령은 무시), 일원배치 임의 효과 모델(one-way random effects model)은 다음과 같습니다.

Yij|αi∼indepNormal(αi,σ2) 여기서 αi∼iidNormal(μ,τ2).(4.41)

임의 효과 _α_ _i_는 환자 _i_의 참 평균(true mean)이고, 환자 _i_에 대한 관측치들은 _α_ _i_ 주변에서 분산 _σ_2을 가지고 변동합니다. 만약 20명의 다른 어린이를 새 표본으로 실험을 반복한다면 _α_ _i_가 변경될 것이기 때문에 _α_ _i_를 임의 효과(random effects)라고 부릅니다. 이 모델에서 환자별 평균 모집단은 평균 _μ_와 분산 _τ_2을 갖는 정규 분포를 따른다고 가정됩니다.

일원배치 임의 효과 모델의 경우 임의 효과에 대해 주변화(marginally over)하면 평균은 모든 관측치에 대해 E(Yij)\=μ이고, 응답의 분산은 임의 효과와 오차 분산의 합인 Var(Yij)\=τ2+σ2입니다. 임의 효과로 설명되는 분산의 비율은 다음과 같습니다.

R2\=Var(αi)Var(Yij)\=τ2τ2+σ2.

임의 효과에 대한 주변화(Marginalizing)는 같은 그룹의 관측치들 간에 상관관계를 유발합니다. 주변 공분산은 다음과 같습니다.

Cov(Yij,Yuv)\={τ2+σ2i\=u 및 j\=v인 경우τ2i\=u 및 j≠v인 경우0i≠u인 경우.(4.42)

동일 환자의 두 관측치는 공통의 임의 효과를 공유하므로 이들은 공분산 _τ_2을 가지며 따라서 상관관계는 τ2/(τ2+σ2)가 됩니다. 서로 다른 환자의 관측치는 공통의 변동성 원천을 공유하지 않으므로 상관관계가 없습니다.

목록 4.9 JAGS에서의 일원배치 임의 효과 모델. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){for(j in 1:m){
3    Y[i,j] ˜ dnorm(alpha[i],sig2_inv)
4  }}
5 
6 # Random effects
7  for(i in 1:n){alpha[i] ˜ dnorm(mu,tau2_inv)}
8 
9 # Priors
10        mu ˜ dnorm(0,0.0001)
11  sig2_inv ˜ dgamma(0.1,0.1)
12  tau2_inv ˜ dgamma(0.1,0.1)
```

임의 효과 모델에 대한 베이지안 분석에는 모집단 모수에 대한 사전 분포가 필요합니다. 평균 및 147\. (이 페이지 그림) 148\. (이 페이지 그림) 149\. 새로운 관측치 예측에 초점을 맞추는 선형 회귀 같은 분석과 달리, 임의 효과 모델에서는 분산 성분(예: _σ_2 및 _τ_2)이 분석의 주요 초점인 경우가 많습니다. 따라서 이러한 모수들에 사용되는 사전 분포를 철저히 검토하는 것이 중요합니다. 역감마 사전 분포([목록 4.9](./12-chapter4.md#list4_9))는 분산 모수에 대한 켤레 분포이며 종종 단순한 깁스 업데이트로 이어집니다. 그러나 [그림 4.9](./12-chapter4.md#fig4_9)에 표시된 것처럼, 모양 모수(shape parameter)가 작은 분산에 대한 역감마 사전 분포는 원점에서 사전 PDF가 0과 같은, 표준편차에 대한 사전 분포를 유도합니다. 이것은 임의 효과 분산이 존재하지 않을 가능성을 배제하는 것이며, 따라서 여러 문제에 적합하지 않을 수 있습니다.

![Multiple panels compare prior densities for sigma under different families. The top row contrasts Half Cauchy and several Inverse Gamma priors with varying shape a, showing sharp peaks near zero for small a and flatter tails for larger a. The bottom row shows Beta Prime priors with differing parameter pairs. Curves vary from steeply decreasing shapes to gently increasing ones, illustrating how each prior assigns mass across small and moderate sigma values over short and extended ranges.](./images/fig4_9.jpg)

그림 4.9 **표준편차에 대한 사전 분포.** _σ_에 대한 절반-코시(half-Cauchy) 사전 분포와, 모양 모수 _a_(그리고 중앙값을 1로 맞추기 위한 척도 모수 _b_)가 다른 _σ_2의 역감마 사전 분포, 그리고 여러 모양 모수를 갖는 베타 프라임(BP)에 의해 _σ_에 유도된 사전 분포. 왼쪽 및 오른쪽 패널은 표시되는 _σ_의 범위가 다릅니다. [본문으로 돌아가기.⏎](chapter4)

역감마 분포가 유일한 켤레 사전 분포이긴 하지만, 다음을 포함한 많은 가능성이 있습니다.

1. 분산에 대한 역감마 사전 분포
2. 표준편차에 대한 균등 사전 분포
3. 표준편차에 대한 절반-코시 사전 분포
4. 표준편차/분산에 대한 일반화 베타 프라임 사전 분포

2.4.1절에서 논의했듯이 파라미터화(parameterization)를 선택하는 것은 사전 분포를 결정하는 핵심 요소입니다. 여기에서의 주요 선택은 분산 _τ_2이거나 표준편차 _τ_입니다. 표준편차는 관측치 척도와 동일하기 때문에 사실상 표준편차를 위해 사전 정보를 도출해 내는 것이 더 수월합니다. 표준편차가 τmax를 넘을 수 없다는 사전 정보는 균등 사전 분포 τ∼Uniform(0,τmax)로 암호화할 수 있습니다. 위쪽 한계가 정당화된다면 이는 해당 제약을 준수하는 합리적인 무정보적 사전 분포가 됩니다. 그러나 τmax 이상의 사전 확률이 0이기 때문에 결과적으로 데이터에 상관없이 τmax 이상의 사후 확률도 0이 될 것이며, 따라서 위쪽 한계를 신중하게 정하는 것이 대단히 중요합니다.

대안으로, \[[54](./19-ref01.md#refbib54)\]는 표준편차에 대해 절반-코시(HC, half-Cauchy) 사전 분포를 지지합니다. HC 분포([부록](./18-appA.md#appA) 참조)는 양수로 제한된 1자유도 스튜던트 t-분포이며 원점에서 균등한(flat) PDF를 갖는데([그림 4.9](./12-chapter4.md#fig4_9)), 이는 일반적으로 역감마 분포보다 사전 믿음을 더 정확하게 표현합니다. [목록 4.10](./12-chapter4.md#list4_10)은 이 사전 분포에 대한 JAGS 코드를 제공합니다. 이 코드에서 HC 분포는 표준편차에 직접 할당됩니다(즉, τ∼HC(1)). 이는 분산 성분에 대한 켤레 관계를 깨뜨리지만(JAGS에서는 쉽게 처리됨), 2단계 모델(부록 9.4)을 사용하면 켤레 관계를 복원할 수 있습니다. 정밀도를 ω\=1/σ2로 표시한다면 다음 규모 혼합 감마(scale-mixture of gammas) 분포는

ω|b∼Gamma(1/2,b) 그리고 b∼Gamma(1/2,1)(4.43)

150\. (이 페이지 그림) 151\. 표준편차 _σ_에 HC(1) 사전 분포를 유도합니다. 이 모델은 [목록 4.11](./12-chapter4.md#list4_11)에 있으며 이는 152\. [목록 4.10](./12-chapter4.md#list4_10)과 동등합니다. 이 코드는 HC 척도 모수가 1로 고정되어 있다고 가정합니다. 코시 사전 분포는 꼬리가 매우 두껍기 때문에 0.99 사전 분위수는 63이 됩니다. 이 넓은 사전 범위에도 불구하고 HC 사전 분포의 척도를 데이터의 척도에 맞게 조정할 수 있습니다.

목록 4.10 절반-코시 사전 분포가 있는 일원배치 임의 효과 모델. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){for(j in 1:m){
3    Y[i,j] ˜ dnorm(alpha[i],sig2 inv)
4  }}
5 
6 # Random effects
7  for(i in 1:n){alpha[i] ˜ dnorm(mu,tau2 inv)}
8 
9 # Priors
10  mu         ˜ dnorm(0,0.0001)
11  sig2 inv   <- pow(sigma,=2)
12  tau2 inv   <- pow(tau,=2)
13  sigma      ˜ dt(0, 1, 1)T(0,) # Half=Cauchy priors with
14  tau        ˜ dt(0, 1, 1)T(0,) # location 0 and scale 1
```

목록 4.11 감마 혼합 표현을 사용한, 절반-코시 사전 분포가 있는 일원배치 임의 효과 모델. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){for(j in 1:m){
3    Y[i,j] ˜ dnorm(alpha[i],sig2 inv)
4  }}
5 
6 # Random effects
7  for(i in 1:n){alpha[i] ˜ dnorm(mu,tau2 inv)}
8 
9 # Priors
10  mu ˜ dnorm(0,0.0001)
11  sig2_inv ˜ dgamma(0.5,b1); b1 ˜ dgamma(0.5,1) # sigma ˜ HC(1)
12  tau2_inv ˜ dgamma(0.5,b2); b2 ˜ dgamma(0.5,1) # tau ˜ HC(1)
```

일반화 베타 프라임(GBP, generalized beta-prime) 분포는 절반-코시 사전 분포의 일반화로 제안되었습니다 \[[10](./19-ref01.md#refbib10), [91](./19-ref01.md#refbib91), [168](./19-ref01.md#refbib168), [172](./19-ref01.md#refbib172)\]. 4.2절에서 베타 프라임(BP) 사전 분포 τ2∼BP(a,b)는 선형 회귀에 사용되어 모델 _R_2에 Beta(a,b) 사전 분포를 유도했습니다. _a_ 및 _b_가 작으면 전체 모델 적합에 대해 무정보적인 사전 분포를 제공합니다. 아래에 논의된 바와 같이 임의 효과 분산에 대한 BP 사전 분포는 비슷하게 매력적인 해석을 갖습니다.

GBP 분포는 멱 모수(power parameter) _c_와 척도 모수 _d_를 포함하여 BP 분포를 확장합니다. X∼BP(a,b)일 때,

X∗\=dX1/c∼GBP(a,b,c,d)

이며, c\=d\=1인 경우에는 베타 프라임 분포 X∗∼BP(a,b)로 환원됩니다. 멱 모수의 유연성 덕분에 분산과 표준편차 파라미터화가 동일한 족 안에 머무를 수 있습니다. 즉, τ2∼GBP(a,b,c,d)는 τ∼GPD(a,b,2c,d1/2)와 동등하기 때문입니다. 절반-코시 사전 분포는 a\=b\=1/2이고 c\=1인 특수한 경우입니다. 즉, τ∼HC(σ)는 τ2∼GBP(1/2,1/2,1,σ)와 동등합니다. 그러나 [그림 4.9](./12-chapter4.md#fig4_9)에 표시된 것처럼 BP 사전 분포, 나아가 GBP 사전 분포는 더 넓은 범위의 형태를 포함합니다. 예를 들어 원점에서의 PDF는 역감마와 같이 0이 되거나, 절반-코시와 같이 유한(finite)하거나, 무한(infinite)일 수 있습니다.

우리가 GBP 사전 분포 τ2|σ2∼GBP(a,b,1,σ2)를 선택한다면, 일원배치 임의 효과 모델에서 임의 효과에 의해 설명되는 분산의 비율인 R2\=τ2/(τ2+σ2)는 Beta(a,b) 사전 분포를 따릅니다. [목록 4.12](./12-chapter4.md#list4_12)처럼 a\=b\=1을 설정하면 임의 효과에 의해 설명되는 분산 비율에 무정보적인 균등 사전 분포를 부여하게 되고, a\=b\=1/2인 절반-코시 사전 분포를 선택하면 상당히 확산되기는 하지만 0과 1에 가까운 _R_2에 사전 질량을 더 부여하게 됩니다. 반면 _ϵ_가 작을 때의 사전 분포 σ2,τ2∼InvGamma(ϵ,ϵ)는 그 질량의 절반을 R2≈0에, 나머지 절반을 R2≈1에 두기 때문에 _R_2에 대해 비현실적인 사전 분포를 유도합니다.

목록 4.12 R2에 대해 균등 사전 분포가 있는 일원배치 임의 효과 모델. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){for(j in 1:m){
3    Y[i,j] ˜ dnorm(alpha[i],sig2_inv)
4  }}
5 
6 # Random effects
7  for(i in 1:n){alpha[i] ˜ dnorm(mu,tau2_inv)}
8 
9 # Priors
```


```R
10  mu       ˜ dnorm(0,0.0001)
11  sig2_inv ˜ dgamma(0.1,0.1)
12  R2       ˜ dunif(0,1)
13  tau2_inv <- sig2_inv*(1=R2)/R2
```

다시 뼈 밀도 데이터로 돌아가서, [그림 4.8](./12-chapter4.md#fig4_8)은 [목록 4.9](./12-chapter4.md#list4_9)의 모델을 가정하여 각각 30,000개의 표본을 추출하고 처음 10,000개를 번인(burn-in)으로 폐기한 두 개의 체인으로 근사된 사후 분포를 보여줍니다. 임의 효과 τ2/(τ2+σ2)에 기인한 분산 비율의 사후 분포는 0.1에서 0.5 범위입니다(왼쪽 아래 패널). 동일한 환자의 관측치 간의 이러한 상관관계를 설명하는 것은 고정 효과(fixed effect) _μ_의 사후 분포에 영향을 미칩니다(오른쪽 아래). _μ_의 사후 분산은 독립 모델 Yij∼iidNormal(μ,σ2)의 _μ_ 사후 분포와 비교할 때 임의 효과 모델에서 더 큽니다. 이는 80명의 서로 다른 환자에서 얻은 80번의 측정값보다 20명의 환자에 대한 4번의 반복 측정값에 평균에 대한 정보가 더 적기 때문에 예상되는 결과입니다.

사전 분포 민감도를 테스트하기 위해, 우리는 [목록 4.10](./12-chapter4.md#list4_10)의 절반-코시 사전 분포와 [목록 4.12](./12-chapter4.md#list4_12)의 균등 _R_2 사전 분포를 사용하여 모델을 재적합했습니다. 이 경우 분산 성분에 대한 사전 분포는 결과에 거의 영향을 미치지 않았습니다. 역감마 사전 분포를 사용한 _σ_와 _τ_의 95% 사후 신용 집합(posterior credible sets)은 각각 (1.24, 1.78) 및 (1.75, 3.51)인 반면, 절반-코시 사전 분포의 경우 (1.24, 1.77) 및 (1.72, 3.45), _R_2 사전 분포의 경우 (1.25, 1.80) 및 (1.69, 3.35)였습니다.

**주변 모델 (Marginal models)**: 임의 효과에 조건을 지정하여 상관관계를 유도하는 것은, 임의 효과를 포함하지 않고 동일한 그룹의 관측치 간의 상관관계를 직접 지정하는 주변 모델([4.6절](./12-chapter4.md#sec4_6))과 동일합니다. 예를 들어 다음과 같은 일원배치 임의 효과 모델은

Yij|αi∼indepNormal(αi,σ2) 여기서 αi∼iidNormal(μ,τ2)(4.44)

다음 주변 모델과 동일합니다.

Yi∼Normal(μ,Σ),(4.45)

여기서 Yi\=(Yi1,...,Yim)T는 그룹 _i_에 대한 데이터 벡터이고, μ\=(μ,...,μ)T는 평균 벡터이며, Σ는 대각선이 τ2+σ2이고 그 외에는 _τ_2인 공분산 행렬입니다. 주변화 접근 방식의 장점 153\. 은 우리가 더 이상 임의 효과 _α_ _i_를 추정할 필요가 없다는 것입니다. 반면 데이터 세트가 크고 상관 구조가 복잡할 경우 평균 벡터, 특히 공분산 행렬이 커질 수 있어 계산이 느려진다는 것은 단점입니다.

계층적 표현(hierarchical representation)에서 Yi의 원소들은 _α_ _i_가 주어졌을 때 독립적이고 동일하게 분포합니다. 주변 모델에서 그룹 _i_의 _m_개의 관측치는 더 이상 독립적이지 않지만 여전히 교환 가능(exchangeable)합니다. 즉, 이들의 분포는 순서를 섞어도(permuting) 변하지 않습니다. 교환 가능성의 개념은 계층 모델을 구성하는 데 근본적인 역할을 합니다. 브루노 드 피네티(Bruno de Finetti)의 표현 정리(representation theorem)에 따르면 교환 가능한 변수의 무한 수열은 일부 잠재 분포(latent distribution)에 조건부로 독립적이고 동일하게 분포하는 것으로 작성될 수 있습니다. 따라서 이러한 중요한 유형의 종속 데이터는 더 단순한 계층 모델을 사용하여 모델링할 수 있습니다.

베이지안 임의 효과 모델은 또한 4.6절에서와 같이 구조화된 (교환 불가능한) 상관관계도 설명할 수 있습니다. 예를 들어, 공간과 시간에 걸쳐 이루어진 관측치는 공간적 거리나 시간적 시차(lag)에 따라 감소하는 상관관계를 보일 수 있습니다.

## 4.5 일반화 선형 혼합 모델 (Generalized linear mixed models)

다중 선형 회귀([4.2절](./12-chapter4.md#sec4_2))는 관측치 간의 독립성 가정 하에서 변수들 간의 관계를 추정합니다. 회귀 계수(β)는 관심 모수(parameters of interest)이며 분석의 모든 관측치에 적용되고 동일한 모집단의 다른 관측치를 예측하는 데에도 적용됩니다. 이러한 의미에서 이 모수들은 _고정 효과(fixed effects)_입니다. 임의 효과([4.4절](./12-chapter4.md#sec4_4))는 서로 다른 하위 집단에 대해 서로 다른 모수(_α_ _i_)를 허용하며 종속성을 포착하는 데 사용될 수 있습니다. 고정 효과와 임의 효과는 물론 공통 모델에 결합할 수 있으며, 이를 선형 혼합 모델(linear mixed model)이라고 부릅니다. 이 장에서는 가우스 데이터를 위한 선형 혼합 모델과 비가우스 데이터를 위한 일반화 선형 혼합 모델(generalized linear mixed model)을 소개합니다.

나중에 살펴보겠지만, 우리가 고려한 다른 모델과 마찬가지로 혼합 모델에도 동일한 알고리즘(예: 깁스 표집)을 사용할 수 있습니다. 사실 계산상으로는 고정 효과와 임의 효과를 구별할 필요가 없습니다(오히려 혼란을 야기할 수 있음, \[[76](./19-ref01.md#refbib76)\]). 그러나 개념적으로 고정 효과와 임의 효과는 구별되는데, 고정 효과는 모집단을 설명하고 임의 효과는 모집단 내의 개별 항목을 설명하기 때문입니다. 흔한 혼란의 원인 중 하나는 베이지안 분석에서 고정 효과가 확률 변수로 취급된다는 점입니다. 그러나 모든 모수와 마찬가지로 고정 효과에 대한 사전 및 사후 분포는 이들이 더 큰 모집단의 무작위 표본이라는 것이 아니라, 고정되어 있으나 알려지지 않은 모수의 참값에 대한 주관적인 불확실성을 반영하는 것입니다.

**선형 혼합 모델 (Linear mixed model)**: 선형 혼합 모델은 관측치의 군집화(clustering)를 설명하기 위해 임의 효과를 추가함으로써 다중 회귀 모델([4.2절](./12-chapter4.md#sec4_2))을 확장합니다. 선형 혼합 모델은 가우스 응답을 가지며 고정 효과와 임의 효과를 모두 포함합니다. [그림 4.8](./12-chapter4.md#fig4_8)의 뼈 밀도 예제로 돌아가서, 응답 Yij는 방문 j∈{1,...,4}에서 어린이 i∈{1,...,20}의 뼈 밀도입니다. 연령에 따른 증가 추세가 있으므로 방문 _j_에서의 아동 연령 Xij에 대한 고정 효과를 추가할 수 있습니다. 또한 각 어린이에 대한 4번의 관측치 사이에는 분명히 군집화/종속성이 존재하므로 어린이에 대한 임의 효과를 포함하여 다음과 같은 일원배치 임의 효과를 갖는 선형 혼합 모델을 얻을 수 있습니다.

Yij|αi∼indepNormal(μ+Xijβ+αi,σ2) 여기서 αi∼iidNormal(0,τ2),(4.46)

여기서 _μ_는 절편이고 _β_는 고정된 연령 추세입니다. 이 모델은 임의 효과 분포의 평균을 0으로 설정하는데, 이는 Yij의 평균에서 _μ_를 제거하고 _α_ _k_의 평균을 _μ_로 가정하는 것과 동일합니다.

154\. 이 모델에서 연령 추세는 이 데이터셋의 모든 어린이는 물론 표본에 포함되지 않은 모집단의 모든 어린이에게도 적용된다는 점에서 고정(fixed)되어 있습니다. 임의 효과 _α_ _i_는 각각 오직 한 명의 어린이에게만 적용됩니다. 연구의 목적이 모집단에 대해 학습하는 것이라면 주된 관심사는 무작위 표본에 있는 아이들에 대한 임의 효과가 아니라 _τ_2으로 파라 파라미터화되는 모집단 전체의 분포로 일반화하는 것입니다.

한 가지 유형의 임의 효과를 갖는 선형 혼합 모델의 보다 일반화된 공식은 다음과 같습니다.

Yi|β,α∼indepNormal(∑l\=1pXilβl+∑k\=1KZikαk,σ2)(4.47)

여기서 αk∼iidNormal(0,τ2)입니다. 고정 성분 ∑l\=1pXilβl은 다중 선형 회귀와 동일합니다. 이제 임의 성분은 Zik에 의해 결정되며, 이는 주로 이진 군집 지표(binary cluster indicators)입니다. 뼈 밀도 모델은 이 형식에 맞는데, Y1,...,Y80은 하나의 벡터 안에 있는 20명 어린이와 4번의 방문에 대한 관측치이며, 관측치 _i_가 어린이 _k_에서 나온 경우 Zik\=1, 그렇지 않으면 Zik\=0이 됩니다. 이와 같이 모델을 작성하면 행렬 표기법으로 나타낼 수 있으며, (4.19)를 확장하여 n×K 행렬 **Z**를 포함해 Y|β,α∼Normal(Xβ+Zα,σ2In)로 쓸 수 있습니다.

베이지안 선형 혼합 모델은 β, _τ_2, _σ_2에 대한 사전 분포를 설정함으로써 완성됩니다. 이전 절의 사전 분포는 선형 혼합 모델에 맞게 조정할 수 있습니다. 예를 들어 크기가 큰 _c_와 작은 _a_, _b_에 대해 사전 분포 βj∼iidNormal(0,c2)와 σ2,τ2∼iid InvGamma(a,b)는 켤레 분포이며 무정보적입니다. 역감마 분포는 절반-코시 또는 베타 프라임 분포(4.4절)로 대체할 수 있습니다. \[[168](./19-ref01.md#refbib168)\]에서 논의된 바와 같이 베타 프라임 사전 분포는 전체 모델 적합도에 대해 베타 분포를 유도할 수 있습니다.

R2\=Var(∑j\=1pXijβj+∑k\=1KZikαk)Var(Yi).

공변량이 표준화되어 있고 βj∼Normal{0,q1σ2W/(p−1)} 및 τ2\=q2σ2W(단, q1+q2\=1)인 경우, 사전 분포 W∼BP(a,b)는 사전 분포 R2∼Beta(a,b)를 줍니다. 비율(q1,q2)은 고정 효과와 임의 효과의 상대적 기여도를 결정하며 고정되거나 디리클레 사전 분포가 주어질 수 있습니다.

**이원배치 임의 효과 (Two-way random effects)**: 위의 모델은 선형적 시간 추세를 가정하는데, 이는 지나치게 제한적일 수 있습니다. 더 유연한 모델은 4개의 시점 각각의 평균 응답에 대해 별도의 모수 α21,...,α24를 포함합니다.

Yij|αi∼indepNormal(μ+Xijβ+α1i+α2j,σ2)

여기서 이전과 같이 어린이의 임의 효과는 α1i∼iidNormal(0,τ12)로 모델링됩니다. 연령을 임의 효과로 취급하는 것은 의문의 여지가 있는데 고려 중인 4개의 연령은 더 큰 연령 모집단의 무작위 표본이 아니기 때문입니다. 한 가지 접근법은 무정보적 사전 분포를 사용하여 α2j를 고정 효과로 취급하는 것으로, 효과적으로 방문 횟수에 대한 4개의 더미 변수를 공변량으로 포함합니다. 이 모델은 4개 연령의 모집단 평균을 설명하는 6개의 모수(_μ, β, α_21, …, αj4)가 있으므로 과적합 상태이며 최소 2개의 모수를 제거해야 합니다.

대안으로 연령 지표를 α2j∼iidNormal(0,τ22)로 모델링하는 방법이 있습니다. 이 모델에서 α2j는 선형 추세로부터의 편차(deviations from the linear trend)를 나타내며 가우스 사전 분포는 과적합을 방지하기 위해 안정성을 추가합니다. 즉, 작은 _τ_2은 α2j≈0으로 수축(shrinks)시키며 시간 추세는 기울기가 _β_인 대략적인 선형이 됩니다. 이 예제는 임의 효과 모델에 대한 고전적인 해석이 없는 베이지안 모델에서 임의 효과가 편리한 모델링 도구로 어떻게 사용될 수 있는지 보여줍니다 \[[77](./19-ref01.md#refbib77)\].

155\. 임의 효과는 관측치 간의 복잡한 종속 구조를 캡처하는 데에도 사용할 수 있습니다. α2j∼iidNormal(0,τ22)인 이원배치 임의 효과 모델의 공분산은 다음과 같습니다.

Cov(Yij,Yuv)\={σ2+τ12+τ22i\=u 및 j\=v인 경우τ12i\=u 및 j≠v인 경우τ22i≠u 및 j\=v인 경우0i≠u 및 j≠v인 경우.(4.48)

따라서 임의 효과는 동일한 어린이의 서로 다른 방문 관측치 간에 τ12/(σ2+τ12+τ22) 상관관계를 유도하고, 동일한 방문 시간에 다른 어린이의 관측치 간에 τ22/(σ2+τ12+τ22) 상관관계를 유도합니다.

**임의 기울기 모델 (Random slopes model)**: 뼈 성장 데이터에 대한 이전 모델들은 각 어린이마다 서로 다른 절편을 허용하지만, 시간에 따른 궤적은 모든 어린이에 대해 동일하다고 가정합니다. 그러나 [그림 4.8](./12-chapter4.md#fig4_8)은 시간에 따른 증가율이 어린이마다 다르다는 것을 나타냅니다. 따라서 어린이 고유의 기울기들을 임의 효과로 모델링할 수 있습니다.

Yij|αi∼indepNormal(αi1+αi2Xj,σ2) 여기서 αi∼iidNormal(β,Ω),(4.49)

환자 _i_에 대한 임의 절편과 기울기는 αi\=(αi1,αi2)T입니다. 평균 벡터 β는 모집단 평균 절편 및 기울기를 포함하며 따라서 고정 효과입니다. 2×2 모집단 공분산 행렬 Ω는 모집단에 대한 임의 효과의 분산을 결정합니다. 베이지안 모델을 완성하기 위해 사전 분포 σ2∼InvGamma(0.1,0.1), β∼Normal(0,1002I2) 및 Ω∼InvWishart(3.1,I2/3.1)을 지정합니다. 공분산 행렬에 대한 역 위샤트 사전 분포(inverse Wishart prior)는 사전 평균(ν\=3.1\>2+1을 취하기 때문에 존재함) I2(2×2 단위 행렬)을 갖습니다([부록 A.1](./18-appA.md#secA_1) 참조). 이 임의 기울기 모델을 위한 JAGS 코드는 [목록 4.13](./12-chapter4.md#list4_13)에 있습니다.

목록 4.13 JAGS에서의 임의 기울기 모델. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){for(j in 1:m){
3    Y[i,j] ˜ dnorm(alpha[i,1]+alpha[i,2]*age[j],tau)
4  }}
5 
6 # Random effects
7  for(i in 1:n){
8     alpha[i,1:2] ˜ dmnorm(beta[1:2],OmegaInv[1:2,1:2])
9  }
10 
11 # Priors
12  tau ˜ dgamma(0.1,0.1)
13  for(j in 1:2){beta[j] ˜ dnorm(0,0.0001)}
14 
15   nvar <- 2 # Set the Wishart parameters so E(Omega) = I nvar
16   nu <- (nvar+1) + 0.1
17   R[1,1] <- nu = (nvar+1)
18   R[1,2] <- 0
19   R[2,1] <- 0
20   R[2,2] <- nu = (nvar+1)
21 
22   OmegaInv[1:2,1:2] ˜ dwish(R[,],nu)
```

156\. 모집단 공분산 행렬의 사후 평균은 다음과 같습니다.

E(Ω|Y)\=\[84.83−9.36−9.361.11\](./4.50)

그리고 상관관계 Cor(αi1,αi2)\=Ω12/Ω11Ω22의 95% 사후 구간은 (−0.986,−0.904)입니다. 따라서 절편과 기울기 사이에는 강한 음의 종속성이 있으며, 이는 8세 때 뼈 밀도가 낮았던 어린이일수록 뼈 밀도가 빠르게 증가함을 나타냅니다(반대의 경우도 마찬가지).

[그림 4.10](./12-chapter4.md#fig4_10)은 세 명의 환자에 대해 _X_가 8세에서 10세 사이일 때의 적합값 αi1+αi2X에 대한 사후 분포를 플롯한 것입니다. 각 환자와 각 연령에 대해 _S_ 사후 표본 αi1(s)+αi2(s)X의 분위수를 사용하여 95% 구간을 계산합니다. 또한 10세 측정 뼈 밀도에 대한 사후 예측 분포(PPD, Posterior Predictive Distribution)도 플롯합니다. PPD는 각 반복에서 Yi∗(s)∼Normal(αi1(s)+αi2(s)10,σ2(s))를 샘플링한 다음 _S_개 예측의 분위수를 계산하여 근사시킵니다. PPD는 환자의 임의 효과 αi의 불확실성과 분산이 _σ_2인 측정 오차를 모두 고려합니다. [그림 4.10](./12-chapter4.md#fig4_10)의 구간들은 임의 효과의 불확실성이 변동의 주요 원천임을 시사합니다.

![A plot shows bone density versus age for three patients marked by circles triangles and plus signs. Each patient has an estimated regression line with surrounding dashed credible intervals. All three lines slope upward, indicating increasing bone density with age. A vertical bracket at age 10 highlights the spread of predicted densities across patients, showing notable between patient variation despite similar upward trends.](./images/fig4_10.jpg)

그림 4.10 **턱뼈 데이터의 혼합 효과 분석.** 8~10년 범위의 _X_에 대해 세 피험자의 관측 뼈 밀도(점) 대 적합값 αi1+αi2X의 사후 중앙값(실선) 및 95% 구간(점선), 그리고 연령 X\=10에서의 측정된 응답에 대한 사후 예측 분포의 95% 신용 구간(연령=10에서의 수직선). [본문으로 돌아가기.⏎](chapter4)

**일반화 선형 혼합 모델 (Generalized linear mixed models)**: 일반화 선형 혼합 모델(GLMM)은 자연스러운 방식으로 4.3절의 GLM에 임의 효과를 포함합니다. GLM의 경우 평균 응답(또는 분포의 다른 특징) E(Yi)이 연결 함수(link function) _g_를 통해 선형 예측자 157\. _η_ _i_와 g{E(Yi)}\=ηi로 연결됨을 상기하십시오. 예를 들어, 로지스틱 회귀에서 이진 응답 변수 _Y_ _i_는 로지스틱 연결 함수 logit{E(Yi)}\=ηi\=∑l\=1pXilβl을 사용하여 공변량 Xi에 회귀되었습니다. 이 연결 함수는 _η_ _i_에 대해 평균 응답이 0과 1 사이가 되도록 보장합니다.

GLMM은 선형 예측자에 임의 효과를 추가합니다. 예를 들어 Yij가 그룹 _j_의 i번째 관측치인 일원배치 GLMM을 고려해 보겠습니다. 선형 예측자 g{E(Yij)}\=ηij는 고정 효과 공변량 Xij와 임의 효과 _α_ _j_를 모두 포함합니다.

ηij\=μ+Xijβ+αj

임의 효과 분포는 αj∼iidNormal(0,τ2)입니다. GLM과 마찬가지로 다양한 응답 유형에 따라 적절한 연결 함수가 다릅니다. 예를 들어 Yij가 이진 데이터인 경우 로지스틱 연결 함수를 사용하고, Yij가 가산 데이터(count)인 경우 로그 연결 함수가 포함된 푸아송 회귀를 사용할 수 있습니다. 또한 선형 혼합 모델은 항등 연결 함수와 응답 분포에 대한 가우스 가정을 가지는 GLMM의 특수한 경우입니다.

임의 효과에 대한 동기 부여와 해석은 가우스 선형 혼합 모델이나 GLMM이나 같습니다. 즉, 임의 효과의 분포는 표집 단위들의 모집단에 걸친 변동을 나타내는 것으로 해석됩니다. 통계적 관심사는 모집단 모수 _τ_2이거나 혹은 때로는 표본에서 크거나 작은 _α_ _j_를 갖는 극단적인 단위를 식별하는 것에 있습니다.

임의 효과는 또한 같은 군집 내의 관측치 사이의 종속성을 설명하는 데에도 사용됩니다. 가우스의 경우와 달리 상관관계를 분석적으로 계산할 수는 없지만, 공유되는 임의 효과가 종속성을 이끈다는 통찰력은 그대로 유지됩니다. 이를 확인하기 위해, 고정 효과 없이 logit{Prob(Yij\=1|αj)}\=αj인 이진 데이터에 대한 로지스틱 모델을 고려하십시오. 평균은 모든 _τ_에 대해 임의 효과를 주변화했을 때 Prob(Yij\=1)\=0.5입니다. 아래 표는 동일한 군집에서 얻은 두 관측치에 대한 결합 PMF를 제공합니다. 이러한 확률은 몬테카를로 표집을 사용하여 근사된 것으로, 먼저 αj∼Normal(0,τ2)를 추출하고 그런 다음 _α_ _j_ 추출값을 감안하여 Y1j와 Y2j를 독립적으로 생성합니다. τ\=0인 경우 관측치는 독립적이며 결합 확률은 주변 확률의 곱이 됩니다. 큰 _τ_에 대해 두 관측치는 동일할 확률이 1로 수렴하는 종속성을 갖습니다. 왜냐하면 큰 _α_ _j_가 추출된 경우 두 관측치 모두 1이 될 가능성이 높고, 작은 _α_ _j_가 추출된 경우 두 관측치 모두 0이 될 가능성이 높기 때문입니다. 따라서 임의 효과는 이러한 이진 응답 간의 양의 종속성을 유발합니다.

| _τ_                 | 0    | 5    | 10   | 15   | 20   |
| ------------------- | ---- | ---- | ---- | ---- | ---- |
| Prob(Y1j\=0,Y2j\=0) | 0.25 | 0.42 | 0.46 | 0.47 | 0.48 |
| Prob(Y1j\=0,Y2j\=1) | 0.25 | 0.08 | 0.04 | 0.03 | 0.02 |
| Prob(Y1j\=1,Y2j\=0) | 0.25 | 0.08 | 0.04 | 0.03 | 0.02 |
| Prob(Y1j\=1,Y2j\=1) | 0.25 | 0.42 | 0.46 | 0.47 | 0.48 |

아래의 예시에서, 우리는 이진 결과와 임의 효과가 있는 GLMM을 적합하기 위해 JAGS를 사용합니다. JAGS는 이와 같이 작거나 중간 규모의 분석에 충분히 빠릅니다. 그러나 보다 큰 분석의 경우 brms \[[26](./19-ref01.md#refbib26)\]와 같이 GLMM 전용 패키지를 선호할 수 있습니다.

**아동 말라리아 예제 (Childhood malaria example)**: Diggle 외 \[[44](./19-ref01.md#refbib44)\]는 아동 말라리아와 관련된 요인을 연구합니다. 이 데이터는 geoR 패키지에서 사용할 수 있으며, 잠비아의 L\=65개 마을([그림 4.11](./12-chapter4.md#fig4_11)의 점들)의 아동 n\=1,332명으로 구성되어 있습니다. 응답 변수 _Yi_는 어린이 _i_가 말라리아 양성 판정을 받았으면 Yi\=1이고, 음성 판정을 받았으면 Yi\=0입니다. si∈{1,...,L}을 어린이 _i_의 마을 번호라고 합시다. 마을 이외에도 관측치 _i_에는 7개의 공변량(절편을 포함하여 p\=8)이 있습니다: 마을의 x 좌표, 마을의 y 좌표, 아동의 나이, 모기장 사용 여부를 나타내는 지표, 모기장 처리(treated) 여부를 나타내는 지표, 마을의 녹지도(greenness), 그리고 마을 내 보건소 유무를 나타내는 지표. 모든 공변량은 평균이 0이고 분산이 1이 되도록 표준화되었습니다.

![Three maps show fitted spatial summaries using shaded circles. The first map displays average fitted probability, with darker circles indicating higher values from about 0.59 to 0.81. The second map shows random effect means, ranging from strongly negative to strongly positive, again shaded by magnitude. The third map shows the probability the random effect is positive, with dark circles representing values near one. All maps share the same underlying region outline with points distributed along its length.](./images/fig4_11.jpg)

그림 4.11 **잠비아 분석의 적합값.** (a) 각 마을에 대한 적합된 확률 _q_ _i_ 사후 평균들의 평균, (b) 마을 임의 효과 _α_ _k_의 사후 평균, (c) αk\>0일 사후 확률. [본문으로 돌아가기.⏎](chapter4)

이 분석의 목적은 공변량 효과를 추정하고 말라리아 초과 위험(excess risk)이 있는 마을, 즉 공변량으로는 설명할 수 없는 높은 말라리아 발생률을 가진 마을을 식별하는 것입니다. 공변량에는 고정 효과를 158\. 마을에는 임의 효과를 사용하는 일반화 로지스틱 회귀 모델을 사용합니다.

qi\=Prob(Yi\=1|β,α) 그리고 logit(qi)\=∑j\=1pXijβj+αsi(4.51)

임의 효과 분포는 αk∼iidNormal(0,τ2)입니다. 임의 효과는 마을 지표 _s_ _i_를 사용하여 코딩되므로, 어린이 _i_가 마을 _l_에 있는 경우 si\=k이고 αsi\=αk입니다. 임의 효과는 동일한 마을의 두 어린이 간의 상관관계를 모두 포착하고, 말라리아 초과 위험이 있고 αk\>0인 마을을 식별하는 데 사용할 수 있습니다. [목록 4.14](./12-chapter4.md#list4_14)는 무정보적 사전 분포 βj∼Normal(0,10) 및 τ2∼BP(0.5,0.5), 즉 τ∼HC(1)를 갖는 이 모델을 적합하는 JAGS 코드를 제공합니다.

목록 4.14 잠비아 데이터에 대한 임의 효과 로지스틱 회귀. [본문으로 돌아가기.⏎](chapter4)

```R
1 for(i in 1:n){
2   Y[i] ˜ dbern(q[i])
3   logit(q[i]) <- X[i,1]*beta[1] + X[i,2]*beta[2] +
4                      X[i,3]*beta[3] + X[i,4]*beta[4] +
5                      X[i,5]*beta[5] + X[i,5]*beta[6] +
6                      alpha[s[i]]
7 }
8 for(j in 1:6){ # Priors for the fixed effects
9   beta[j] <- dnorm(0,0.1)
10 }
11 for(k in 1:65){ # Village random effects
12   alpha[k] <- dnorm(0,tau2 inv)
13 }
14 tau2 inv <- (1=R2)/R2 # Beta=prime/Half=Cauchy prior
15 R2 ˜ beta(0.5,0.5)
```

[표 4.2](./12-chapter4.md#tbl4_2)는 공변량 효과의 사후 분포를 요약합니다. 예상대로 모기장 사용과 보건소와의 인접성은 말라리아 유병률을 줄입니다. 마을 효과의 표준편차 _τ_의 사후 평균은 0.88입니다. [그림 4.2b](./12-chapter4.md#fig4_2)는 마을 임의 효과 _α_ _k_의 사후 평균이 대략 −1.5에서 1.5 범위에 있음을 보여줍니다. 이러한 값들은 고정 효과를 고려한 후에도 마을마다 양성 반응률에 상당한 차이가 있음을 나타냅니다. αk\=1.5인 마을 아동의 양성 판정 오즈(odds)는, 공변량은 같지만 αk\=0인 마을 아동보다 exp(1.5)\=4.48배 더 높습니다. [그림 4.2b](./12-chapter4.md#fig4_2)에서 _α_ _k_ 값이 큰 군집에 위치한 마을들을 조사하면, 이 지역들의 질병 유병률을 설명할 수 있는 누락된 요인들에 대한 가설을 생성하는 데 사용할 수 있을 것입니다.

__표 4.2 **잠비아 분석 요약**. 고정 효과 계수(_β_ _j_)와 임의 효과 표준편차(_τ_)에 대한 사후 요약. [본문으로 돌아가기.⏎](chapter4)__
| 평균 (Mean) | 표준편차 (SD) | 95% 구간 (95% Interval) | |
| ---------------- | ----- | ------------ | -------------- |
| 절편 (Intercept) | −0.64 | 0.13 | (−0.89, −0.39) |
| 연령 (Age) | 0.29 | 0.05 | ( 0.18, 0.39) |
| 모기장 사용 (Net use) | −0.20 | 0.07 | (−0.33, −0.05) |
| 처리됨 (Treated) | −0.18 | 0.10 | (−0.37, 0.01) |
| 녹지도 (Greenness) | 0.32 | 0.12 | ( 0.09, 0.55) |
| 보건소 (Health center) | −0.16 | 0.12 | (−0.40, 0.08) |
| 임의 효과 SD (Random effect SD) | 0.88 | 0.11 | ( 0.68, 1.12) |

사전 분포 민감도를 검증하기 위해, (a,b)를 (1,1), (5,1), (1,5) 혹은 (5,5)로 설정하고 τ2∼BP(a,b)로 모델을 다시 적합합니다. 임의 효과 표준편차 _τ_의 사후 평균 범위는 0.84에서 0.94였습니다. 따라서 사전 분포의 큰 변화는 사후 분포에 중간 정도의 변화를 생성하며 사전 분포에 약간의 민감도가 존재합니다.

## 4.6 상관관계가 있는 데이터의 선형 모델 (Linear models with correlated data)

4.4절, 특히 (4.42)는 임의 효과를 포함함으로써 관측치 간의 종속성을 파악할 수 있음을 보여줍니다. 이 접근 방식에서는 모델에 임의 효과를 포함하고, 임의 효과가 주어진 경우 관측치들 간의 조건부 독립성을 가정하였으며, 159\. (이 페이지 그림/표) 160\. 임의 효과에 대한 주변화를 통해 유도되는 상관관계를 연구했습니다. 대안으로 임의 효과를 포함하지 않고 관측치 간의 상관관계를 직접 모델링할 수도 있습니다. 예를 들어, 종속성을 캡처하기 위해 임의 효과를 포함하는 대신 (4.48)의 공분산 모델을 직접 사용하여 동등하게 종속성을 모델링할 수 있습니다. 공간 또는 시간 데이터와 같이 자연스러운 순서가 있는 데이터의 경우, 공간이나 시간의 거리에 대한 함수로 주변 상관관계를 직접 모델링하는 것이 종종 더 직관적인 접근 방식이 됩니다. 임의 효과와 마찬가지로 상관관계 모델은 여러 가지가 있으며, 타당한 추론을 얻고 정확한 예측을 수행하기 위해서는 합리적인 것을 찾는 것이 중요합니다.

상관된 오차를 갖는 베이지안 선형 모델은 다음과 같습니다.

Y|β∼Normal(Xβ,Σ).(4.52)

평균 구조 **Xβ**는 오차가 독립적인 선형 모델과 똑같이 해석됩니다. 즉, 다른 모든 공변량이 고정된 경우 한 공변량을 1만큼 증가시키면 평균이 β의 해당 원소만큼 증가합니다. 그러나 β의 사후 분포는 상관관계의 선택에 극적인 영향을 받을 수 있습니다(예: \[[78](./19-ref01.md#refbib78)\]). 상관된 데이터의 베이지안 분석은 예를 들어 공간 또는 시간의 상관관계를 포착하기 위해 상관 구조를 올바르게 지정하는 데 중점을 둡니다. 관측치 _i_가 공간 위치(경도/위도) si에서 획득되었다고 합시다. 그러면 위치 si와 sj의 관측치 사이의 상관관계는 위치 간의 거리인 dij\=||si−sj||에 따라 다음과 같이 감소한다고 가정할 수 있습니다.

Cor(Yi,Yj)\=C(dij)\=exp(−ϕdij)

여기서 ϕ\>0은 감소율을 결정하며 추정할 모수입니다. 그러면 Σ의 (i,j) 원소는 σ2C(dij)이므로, _σ_2은 **Y**의 분산입니다.

각 관측치 쌍 간의 상관관계를 명시적으로 모델링하는 대신, 거리를 입력받아 상관관계를 출력하는 상관 함수(correlation function) _C_에 의해 상관관계가 결정됩니다. 지수 상관 함수는 한 가지 간단한 예일 뿐이며, 예를 들어 제곱 지수 상관 함수(squared exponential correlation function)인 C(d)\=exp(−ϕd2)는 기계 학습(8.3.2절)에서 인기가 있습니다. 상관 함수는 일반적으로 유효한 상관 함수 목록에서 선택되는데, 이 목록은 Σ를 채우는 데 사용될 때 Σ가 대칭이고 양의 정부호(positive definite)임을 보장해 줍니다. \[[38](./19-ref01.md#refbib38)\]은 일반적인 공간 및 시간 상관 함수들의 목록을 제공합니다.

상관 구조와 상관관계 모수(이전 예제의 _ϕ_)에 대한 사전 분포가 주어지면 표준적인 베이지안 계산 도구를 사용하여 사후 분포를 요약할 수 있습니다. 상관 모수는 보통 켤레 사전 분포를 가지지 않으므로 메트로폴리스-헤이스팅스(Metropolis–Hastings) 표집이 사용됩니다. 상관된 데이터에 대한 베이지안 접근 방식의 장점은, MCMC 표집을 사용하면 예측 또는 다른 모수에 대한 추론 시 상관 모수의 불확실성을 설명할 수 있다는 것입니다. 반면 최대 가능도 분석은 종종 상관 모수의 플러그인 추정치(plug-in estimates)를 사용하므로 불확실성을 과소평가하게 됩니다.

**총기 규제 예제 (Gun control example)**: 이 분석의 데이터는 Kalesan 등(2016) \[[86](./19-ref01.md#refbib86)\]에서 가져온 것입니다. 응답 변수 _Y_ _i_는 2010년 상태 _i_(알래스카 및 하와이 제외)에서 인구 10,000명당 총기 관련 사망률의 로그값입니다. 이는 5가지 잠재적 교란 요인(confounders), 즉 2009년 인구 10,000명당 로그 총기 사망률, 총기 소유율 사분위수, 실업률 사분위수, 비총기 살인율 사분위수, 총기 수출률 사분위수에 회귀됩니다. 관심 있는 공변량은 해당 주에서 시행 중인 총기 규제법의 수입니다. 이는 p\=6개의 공변량을 제공합니다.

우리는 먼저 일반적인 베이지안 선형 회귀 모델을 적합합니다.

Yi\=β0+∑j\=1pXiβj+εi(4.53)

161\. 이 모델은 독립 오차 εi∼iid Normal(0,σ2) 및 무정보적 사전 분포를 가집니다. 총기법의 수에 해당하는 회귀 계수의 사후 밀도는 [그림 4.12](./12-chapter4.md#fig4_12)에 플롯되어 있습니다. 계수가 음수일 사후 확률은 0.96으로, 총기법의 수와 총기 관련 사망률 사이의 음의 관계를 시사합니다.

![A posterior density plot for Beta compares non spatial and spatial models. Both curves peak near Beta around minus 0.01. The non spatial curve is slightly taller and narrower, while the spatial curve is lower with heavier tails, reflecting greater uncertainty under the spatial model.](./images/fig4_12.jpg)

그림 4.12 **총기 규제법이 총기 관련 사망률에 미치는 영향**. 주(state) 총기 관련 사망률의 공간 및 비공간 모델에서 한 주의 총기 규제법 수와 관련된 계수의 사후 분포. [본문으로 돌아가기.⏎](chapter4)

독립적인 잔차 가정은 주(neighboring states)가 서로 인접해 있으면 상관관계가 있을 수 있으므로 미심쩍습니다. 공간적 상관관계는 총기가 주 경계를 넘어 반입되거나, 혹은 공간적으로 다양한 누락된 공변량(예: 총기에 대한 태도 및 사용)에서 발생할 수 있습니다. 연구에 따르면 잔차 종속성을 고려하는 것이 회귀 계수 추정치에 극적인 영향을 미칠 수 있습니다 \[[78](./19-ref01.md#refbib78)\].

우리는 잔차 공분산 Cov\[(ε1,...,εn)T\]\=Σ을 다음과 같이 분해합니다.

Σ\=τ2S+σ2In,(4.54)

여기서 τ2S는 공간 공분산이고 σ2In는 비공간 공분산입니다. 두 주 간의 상관관계가 두 주 간의 거리에 따라 감소하도록 허용하는 많은 공간 상관 모델(예: \[[11](./19-ref01.md#refbib11)\])이 있습니다. 예를 들어 주 간 상관관계가 그 사이의 거리에 따라 기하급수적으로 감소한다고 가정하는 것이 일반적인 모델입니다. 그러나 불규칙한 모양의 주 간의 거리를 정량화하는 것은 어렵기 때문에, 우리는 인접성(adjacencies)을 사용하여 공간 종속성을 모델링합니다. 주 _i_와 _j_가 국경을 공유하는 경우 Aij\=1이라 하고, i\=j이거나 주가 인접하지 않은 경우 Aij\=0이라 합시다. 공간 공분산은 조건부 자기회귀 모델(conditionally autoregressive model) S\=(M−ρA)−1을 따릅니다. 여기서 _A_는 (i,j) 원소가 Aij인 인접 행렬(adjacency matrix)이고 _M_은 i번째 대각 원소가 주 _i_와 이웃하는 주의 수와 동일한 대각 행렬(diagonal matrix)입니다. 모수 ρ∈(0,1)은 인접한 부지 간의 상관관계는 아니지만 공간 종속성의 강도를 결정하며 ρ\=0이면 독립에 해당합니다.

162\. 공간 종속성 모수 _ρ_의 사후 평균(표준편차)은 0.38 (0.25)이므로 이 데이터의 잔여 공간 종속성은 강하지 않습니다. 그러나 [그림 4.12](./12-chapter4.md#fig4_12)에서 관심 회귀 계수의 사후 분포는 비공간 모델보다 공간 모델에서 눈에 띄게 더 넓습니다. 계수가 음수일 사후 확률은 비공간 모델 하에서의 0.96에서 공간 모델의 0.93으로 낮아집니다. 따라서 잔차 종속성을 고려하는 것이 결과를 질적으로 변화시키지는 않았지만, 이 예시는 잔차에 대해 선택된 모델이 회귀 계수의 사후 분포에 영향을 줄 수 있음을 보여줍니다.

**턱뼈 밀도 예제 (Jaw bone density example)**: [그림 4.8](./12-chapter4.md#fig4_8)(왼쪽 위)의 종단 데이터에 대해 가능한 상관 구조는 방문 간의 시간에 따라 상관관계가 감소한다고 가정하는 것입니다. 1차 자기회귀 상관 구조(first-order autoregression correlation structure)는 Cor(Yij,Yik)\=ρ|j−k|입니다. 환자 _i_에 대한 _m_개 관측치의 벡터를 Yi\=(Yi1,...,Yim)T로 나타내면, m×m 공분산 행렬 Σ의 (j,k) 원소는 σ2ρ|j−k|과 같습니다. 행렬 표기법에서 평균에 대한 임의 기울기 모델은 E(Yi|αi)\=Xαi이며, 여기서 **X**는 첫 번째 열이 절편을 위한 벡터(vector of ones)이고 두 번째 열이 기울기를 위한 연령 X1,...,Xm인 m×2 행렬입니다. 그러면 가능도는 다음과 같습니다.

Yi|αi∼indepNormal (Xαi,Σ)(4.55)

임의 효과 분포는 αi∼iid Normal(β,Ω)입니다. 상관관계 모수에는 ρ∼Uniform(0,1) 사전 분포가 주어지며, 다른 모든 사전 분포는 다른 적합과 동일합니다. JAGS 코드는 [목록 4.15](./12-chapter4.md#list4_15)에 주어져 있습니다.

목록 4.15 JAGS에서의 자기회귀 종속성이 있는 임의 기울기 모델. [본문으로 돌아가기.⏎](chapter4)

```R
1 # Likelihood
2  for(i in 1:n){
3    Y[i,1:m] ˜ dmnorm(mn[i,1:m],SigmaInv)
4    for(j in 1:m){mn[i,j] <- alpha[i,1]+alpha[i,2]*age[j]}
5  }
6  SigmaInv[1:m,1:m] <- inverse(Sigma[1:m,1:m])
7  for(j in 1:m){for(k in 1:m){
8     Sigma[j,k] <- pow(rho,abs(k=j))/tau
9  }}
10 
11 # Random effects
12  for(i in 1:n){alpha[i,1:2] ˜ dmnorm(beta[1:2],Omega[1:2,1:2])}
13 
14 # Priors
15  tau ˜ dgamma(0.1,0.1)
16  for(j in 1:2){beta[j] ˜ dnorm(0,0.0001)}
17  rho ˜ dunif(0,1)
18  Omega[1:2,1:2] ˜ dwish(R[,],2.1)
19 
20   R[1,1]<-1/2.1
21   R[1,2]<-0
22   R[2,1]<-0
23   R[2,2]<-1/2.1
```

상관관계 모수 _ρ_의 사후 중앙값은 0.85이고 사후 95% 구간은 (0.46, 0.96)이므로 환자 고유의 선형 추세로는 설명할 수 없는 상관관계의 증거가 존재합니다. 자기회귀 상관관계를 포함하는 것은 고정 효과의 사후 분포에 약간의 영향만 미칩니다. 오차가 독립적인 임의 163\. 효과 모델에 대한 95% 사후 구간은 _β_1에 대해 (29.9, 38.3), _β_2에 대해 (1.33, 2.38)인 반면 자기회귀 모델의 경우는 _β_1에 대해 (30.0, 37.3), _β_2에 대해 (1.45, 2.31)이었습니다.

임의 절편, 임의 기울기, 자기회귀 상관 구조를 가진 이 모델은 이제 매우 복잡해졌습니다. 모수가 관측치 수만큼이나 많고, 종속성에 대한 여러 가지 설명(임의 효과 및 잔차 상관관계)이 존재합니다. 아마도 이 경우에는 이 모든 항이 필요하고 이 비교적 작은 데이터 셋에서도 추정할 수 있겠지만, 계산적인 목적에서나 더 단순한 모델이 설명하고 옹호하기 더 쉽다는 점 때문에 더 단순하면서도 충분한 모델이 선호됩니다. 모델 비교 및 모델의 적절성(adequacy) 테스트는 [6장](./14-chapter6.md)의 주제입니다.


## 4.7 연습문제 (Exercises)

1. 한 임상 시험에서 6명의 피험자에게 위약(placebo)을 투여하고 6명의 피험자에게는 새로운 체중 감량제를 투여했습니다. 응답 변수는 기준치(baseline) 대비 체중 변화(파운드)입니다(따라서 −2.0은 피험자가 2파운드를 감량했음을 의미합니다). 12명의 피험자에 대한 데이터는 다음과 같습니다.
| 위약 (Placebo) | 치료 (Treatment) |
| ------- | --------- |
| 2.0 | −3.5 |
| −3.1 | −1.6 |
| −1.0 | −4.6 |
| 0.2 | −0.9 |
| 0.3 | −5.1 |
| 0.4 | 0.1 |
이 두 그룹의 평균을 비교하기 위해 베이지안 분석을 수행하세요. 이 치료법이 효과적이라고 말할 수 있습니까? 결론이 사전 분포에 민감합니까?
2. R에서 고전적인 보스턴 주택 데이터(Boston Housing Data)를 로드하세요.
`> library(MASS)`
`> data(Boston)`
`> ?Boston`
응답 변수는 자가 소유 주택의 중간값(1,000달러 단위)인 medv이고, 나머지 13개 변수는 이웃 동네를 설명하는 공변량입니다.
   1. 회귀 계수에 대해 무정보적 가우스 사전 분포를 갖는 베이지안 선형 회귀 모델을 적합하세요. MCMC 표집기가 수렴했는지 확인하고 모든 회귀 계수의 사후 분포를 요약하세요.
   2. 고전적인 최소제곱 분석을 수행하세요(예: R의 lm 함수 사용). 결과를 베이지안 결과와 수치적 및 개념적으로 비교하세요.
   3. 회귀 계수에 대해 이중 지수(double exponential) 사전 분포를 사용하여 베이지안 모델을 다시 적합하고 결과가 무정보적 사전 분포를 사용한 분석과 어떻게 다른지 논의하세요.
   4. (a)의 베이지안 선형 회귀 모델을 처음 500개의 관측치만 사용하여 적합하고 마지막 6개의 관측치에 대한 사후 예측 분포를 계산하세요. 이 6개의 관측치에 대한 사후 예측 분포와 실제 값을 함께 플롯하고 예측이 합리적인지 의견을 제시하세요.
3. 164\. 책의 웹사이트에서 2016년 대통령 선거 데이터를 다운로드하세요. 카운티 _i_에 대한 응답 변수를 2016년과 2012년의 공화당 후보 득표율 차이로 하고 개체 **X**의 모든 변수를 공변량으로 사용하여 베이지안 선형 회귀를 수행하세요.
   1. 회귀 계수에 대해 무정보적 가우스 사전 분포를 갖는 베이지안 선형 회귀 모델을 적합하고 모든 회귀 계수의 사후 분포를 요약하세요.
   2. 잔차 Ri\=Yi−Xiβ^를 계산하세요. 여기서 β^는 회귀 계수의 사후 평균입니다. 잔차가 가우스 분포를 따릅니까? 어떤 카운티의 잔차가 가장 크고 가장 작은지, 그리고 이것이 해당 카운티에 대해 무엇을 의미할 수 있는지 설명하세요.
   3. 주(state)에 대한 임의 효과를 포함하세요. 즉 주 l\=1,...,50에 있는 카운티의 경우,
   Yi|αl∼Normal(Xiβ+αl,σ2)
   여기서 αl∼iidNormal(0,τ2)이고 _τ_2은 무정보적 사전 분포를 갖습니다. 임의 효과를 추가하는 것이 왜 필요할 수 있습니까? 임의 효과를 추가하면 회귀 계수의 사후 분포에 어떤 영향을 줍니까? 사후 평균 임의 효과가 가장 높은 주와 가장 낮은 주는 어디이며, 이는 이 주들에 대해 무엇을 시사할 수 있습니까?
4. 책의 웹사이트에서 미국 총기 규제 데이터를 다운로드하세요. 이 데이터는 \[[86](./19-ref01.md#refbib86)\]의 단면 연구(cross-sectional study)에서 가져온 것입니다. 주 _i_에 대해 _Y_ _i_를 살인 건수로, _N_ _i_를 인구로 둡시다.
   1. Yi|β∼Poisson(Niλi) 모델을 적합하세요. 단 log(λi)\=Xiβ입니다. 무정보적 사전 분포와 Xi 내의 p\=7개 공변량을 사용하세요: 절편, 5개의 교란 요인 Zi, 그리고 주 _i_의 총기법 총수. MCMC 표집기가 수렴했고 사후 분포를 충분히 탐색했다는 근거를 제공하고 β의 사후 분포를 요약하세요.
   2. 음이항 회귀 모델을 적합하고 푸아송 회귀의 결과와 비교하세요.
   3. (a)의 푸아송 모델에 대해, 총기법의 수를 0으로 설정하여 각 주의 사후 예측 분포를 계산하세요. 총기법의 수를 25(최대수)로 설정하여 이 작업을 반복하세요. 이러한 계산에 따르면 정책 변화가 전국 사망자 수에 어떤 영향을 미칠까요? 이 예측을 신뢰합니까?
5. R에서 titanic 데이터 세트를 다운로드하세요.
`library("titanic")`
`dat <- titanic_train`
`?titanic_train`
승객 _i_가 생존한 경우 Yi\=1로, 그렇지 않은 경우 Yi\=0으로 둡시다. 생존 확률을 승객의 나이, 성별(더미 변수) 및 객실 등급(2개의 더미 변수)에 회귀시키는 베이지안 로지스틱 회귀를 수행하세요. 각 공변량의 효과를 요약하세요.
6. [그림 3.10](./11-chapter3.md#fig3_10)에 플롯된 티라노사우루스(T. rex) 성장 차트 데이터에는 체중(kg)이 29.9, 1761, 1807, 2984, 3230, 5040, 5654이고 그에 해당하는 나이(년)가 2, 15, 14, 16, 18, 22, 28인 n\=6 관측치가 있습니다. 무게는 양수여야 하므로 감마 분포족은 이 데이터에 합리적인 모델입니다. 연령에 따라 선형으로 증가하는 로그 평균과 감마 가능도가 있는 모델을 설명하세요. 165\. MCMC를 사용하여 사후 분포를 근사하고, 모든 모델 모수의 사후 분포를 요약하며, 적합된 평균 곡선에 맞추어 데이터를 플롯하세요.
7. i\=1,...,n 및 j\=1,...,m에 대한 일원배치 임의 효과 모델 Yij|αi,σ2∼Normal(αi,σ2) 및 αi∼Normal(0,τ2)를 고려하세요. 켤레 사전 분포 σ2,τ2∼InvGamma(a,b)를 가정하여 _α_1, _σ_2 및 _τ_2의 완전 조건부 분포를 도출하고, 사후 분포에서 표집하기 위한 MCMC 알고리즘의 개요를 설명하세요(코드는 작성하지 마십시오).
8. R에서 잠비아(Gambia) 데이터를 로드하세요.
`> library(geoR)`
`> data(gambia)`
`> ?gambia`
응답 변수 _Y_ _i_는 아동 _i_가 말라리아 양성 판정(pos)을 받았음을 나타내는 이진 지표이고, 나머지 7개의 변수는 공변량입니다.
   1. _β_ _j_에 대한 무정보적 사전 분포가 있는 다음 로지스틱 회귀 모델을 적합하세요.
   logit\[Prob(Yi\=1)\]\=∑j\=1pXijβj
   MCMC 표집기가 수렴했는지 확인하고 공변량의 효과를 요약하세요.
   2. 이 데이터셋에서 2,035명의 어린이는 (데이터셋의 _x_ 및 _y_ 좌표로 정의된) L\=65개의 고유한 위치에 거주합니다. si∈{1,...,L}을 관측치 _i_의 위치 레이블로 둡시다. 다음 임의 효과 로지스틱 회귀 모델을 적합하세요.
   logit\[Prob(Yi\=1)\]\=∑j\=1pXijβj+αsi 여기서 αl∼iidNormal(0,τ2)
   그리고 _β_ _j_ 및 _τ_2는 무정보적 사전 분포를 갖습니다. MCMC 표집기가 수렴했는지 확인하십시오. 여기에 임의 효과가 왜 필요할 수 있는지 설명하십시오. 모델에 임의 효과가 추가될 때 발생하는 회귀 계수 사후 분포의 차이점을 논의하고 설명하십시오. 공간적 위치에 따른 _α_ _l_의 사후 평균을 플롯하고, 이 지도가 말라리아 연구자들에게 어떻게 유용할 수 있는지 제안하세요.
9. R에서 babynames 데이터를 다운로드하고 1950년 이후 매년 아기 이름이 "Sophia"로 지어질 로그 오즈(log odds)를 계산하세요.
`library(babynames)`
`dat <- babynames`
`dat <- dat[dat$name=="Sophia" &`
`           dat$sex=="F" &`
`           dat$year>1950,]`
`yr  <- dat$year`
`p   <- dat$prop`
`t   <- dat$year - 1950`
`Y   <- log(p/(1-p))`
_Y_ _t_를 t+1950년의 표본 로그 오즈로 둡시다. 이 데이터에 다음의 시계열(자기회귀 차수 1) 모델을 적합하세요.
Yt\=μt+ρ(Yt−1−μt−1)+εt
166\. 여기서 μt\=α+βt 및 εt∼iid Normal(0,σ2)입니다. 사전 분포는 α,β∼Normal(0,1002), ρ∼Uniform(−1,1), 및 σ2∼InvGamma(0.1,0.1)입니다.
   1. 4개의 모델 모수인 _α, β, ρ_ 및 _σ_2 각각에 대한 해석을 제공하세요.
   2. t\>1에 대해 JAGS를 사용하여 모델을 적합하고, 수렴을 확인하고, 각 모수에 대한 사후 평균과 95% 구간을 보고하세요.
   3. 2020년의 _Y_ _t_에 대한 사후 예측 분포를 플롯하세요.
10. 강의 웹사이트에 제공되고 [그림 3.4](./11-chapter3.md#fig3_4)에 플롯된 지구 평균 기온 이상(현재 연도와 1951-1980년 평균 간의 차이) 데이터를 고려하십시오. _X_ _i_와 _Y_ _i_를 관측치 _i_의 연도 및 기온 이상(anomaly) 섭씨온도(C)라고 합시다. [8.3절](./16-chapter8.md#sec8_3)에서처럼 비모수적 회귀 모델 Yi\=g(Xi)+εi을 사용하여 이 데이터를 분석할 것인데, 여기서 _g_는 J\=15개의 B-스플라인 기저 함수(B-spline basis functions)의 선형 결합으로 모델링됩니다.
   1. 모델을 적합하고 데이터와 함께 각 _X_ _i_ 값에 대한 g(Xi)의 사후 평균 및 95% 사후 구간을 플롯하세요.
   2. Δ\=g(2020)−g(1980)의 사후 분포를 요약하고 해석하세요.
   3. i\>1 각각에 대해 δi\=g(Xi)−g(Xi−1)의 사후 분포를 요약하고 해석하세요.
11. 응답 _Y_ _i_가 위치 i\=1,...,n에서 해양 생물 다양성의 척도인 생태학 연구를 고려해 보겠습니다. 공변량 Xi1,...,Xip는 수심, 해안과의 거리, 어업 규정 등 위치 _i_의 환경을 요약한 것입니다. 다중 선형 회귀 모델 Yi\=β0+Xi1β1+...+Xipβp+εi을 가정합니다. _β_ _j_에 대한 다음의 사전 분포가 유용할 수 있는 시나리오를 제시하세요.
   1. 제프리스 사전 분포 (Jeffreys prior)
   2. 베이지안 라쏘(LASSO) 사전 분포
   3. 작은 _v_에 대한 정보적 사전 분포 βj∼Normal(mj,v)
   시나리오에는 _n_ 및 _p_와 같은 수치 요약과 과학적 목적이 모두 포함되어야 합니다.
12. 우울증의 유전적 결정 요인에 대한 연구에서 연구자들은 1000명의 환자로부터 정보를 수집했습니다. 응답 변수는 (연속적인) CES-D 설문조사 점수이고 예측 변수는 10,000개의 유전자 마커입니다. 목적은 우울증과 연관된 마커가 있는지 확인하는 것입니다.
   1. 이 데이터에 대한 모델과 사전 분포를 설명하고 정당화하세요.
   2. 결과를 어떻게 요약하시겠습니까?
13. 아래 코드를 사용하여 R에서 은하계(galaxies) 데이터를 열고 플롯하세요.
`>   library(MASS)`
`>   data(galaxies)`
`>   ?galaxies`
`>   Y <- galaxies`
`>   hist(Y,breaks=25)`
K\=3개의 정규 분포 혼합(mixture of normal distributions)을 사용하여 관측치 Y1,...,Y82를 모델링하세요. 각 _S_개의 MCMC 반복에 대해 그리드 y∈{5000,5100,...,40000}(그리드의 351개 포인트) 상에서 밀도 함수를 평가하여 S×351 차원의 사후 표본 행렬을 도출하세요. 351개의 그리드 값 각각에서 밀도 함수의 사후 중앙값 및 95% 신용 집합을 플롯하세요. 이 혼합 모델이 데이터에 잘 맞습니까?
14. 167\. 10명의 생태학자로 구성된 그룹이 붉은가슴딱따구리(RCP, Red Cockaded Woodpeckers)를 찾기 위해 숲을 조사하고 있습니다. 각 생태학자는 다른 경로를 따라 걸으며 25번 정지합니다. 그들은 정지할 때마다 지역 조건(나무 밀도, 고도 등)과 RCP를 보거나 들었는지 여부를 기록합니다. 목적은 RCP에 가장 유리한 서식지 유형에 대한 모델을 구축하는 것입니다.
   1. 이 데이터에 대한 모델과 사전 분포를 설명하고 정당화하세요.
   2. 결과를 어떻게 요약하시겠습니까? 168은 비어있습니다.

