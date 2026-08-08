65\. 

# 3 계산적 접근법 (Computational approaches)

## 목차 (Contents)

* [3.1 결정론적 방법 (Deterministic methods)](./11-chapter3.md#sec3_1)  
   * [3.1.1 최대 사후 확률 추정 (Maximum a posteriori estimation)](./11-chapter3.md#sec3_1_1)  
   * [3.1.2 수치 적분 (Numerical integration)](./11-chapter3.md#sec3_1_2)  
   * [3.1.3 베이지안 중심 극한 정리 (Bayesian central limit theorem (CLT))](./11-chapter3.md#sec3_1_3)  
   * [3.1.4 변분 베이즈 방법 (Variational Bayesian (VB) Methods)](./11-chapter3.md#sec3_1_4)
* [3.2 확률적 방법 (Stochastic methods)](./11-chapter3.md#sec3_2)  
   * [3.2.1 MCMC 샘플링 (MCMC sampling)](./11-chapter3.md#sec3_2_1)  
         * [3.2.1.1 깁스 샘플링 (Gibbs sampling)](./11-chapter3.md#sec3_2_1_1)  
         * [3.2.1.2 메트로폴리스-해이스팅스 샘플링 (Metropolis–Hastings (MH) sampling)](./11-chapter3.md#sec3_2_1_2)  
   * [3.2.2 근사 베이지안 계산 (Approximate Bayesian Computing (ABC))](./11-chapter3.md#sec3_2_2)  
   * [3.2.3 시뮬레이션 기반 추론 (Simulation-based inference)](./11-chapter3.md#sec3_2_3)  
   * [3.2.4 순차 몬테카를로 샘플링 (Sequential Monte Carlo (SMC) sampling)](./11-chapter3.md#sec3_2_4)
* [3.3 R에서의 MCMC 소프트웨어 옵션 (MCMC software options in R )](./11-chapter3.md#sec3_3)
* [3.4 MCMC 수렴 진단 및 개선 (Diagnosing and improving MCMC convergence)](./11-chapter3.md#sec3_4)  
   * [3.4.1 초기값 선택 (Selecting initial values)](./11-chapter3.md#sec3_4_1)  
   * [3.4.2 수렴 진단 (Convergence diagnostics)](./11-chapter3.md#sec3_4_2)  
   * [3.4.3 수렴 개선 (Improving convergence)](./11-chapter3.md#sec3_4_3)  
   * [3.4.4 대용량 데이터 세트 처리 (Dealing with large datasets)](./11-chapter3.md#sec3_4_4)
* [3.5 연습문제 (Exercises)](./11-chapter3.md#sec3_5)

컴퓨팅은 현대의 모든 통계적 애플리케이션의 핵심 요소이며 베이지안(Bayesian) 분석도 예외는 아닙니다. 1980년대 후반까지 베이지안 방법의 적용은 켤레성(conjugacy) ([섹션 2.1](./10-chapter2.md#sec2_1))이 단순한 사후 분포로 이어지거나 파라미터 수가 수치 적분(예: [섹션 3.1.2](./11-chapter3.md#sec3_1_2))을 허용할 만큼 적은 작은 문제들로 제한되었습니다. 그러나 1950년대부터 1970년대에 걸쳐 개발된 마르코프 체인 몬테카를로(MCMC, Markov Chain Monte Carlo) 방법이 도입되면서 복잡한 데이터 세트에 현실적인 모델을 맞추는 것이 가능해졌고, 베이지안 방법의 인기는 폭발적으로 증가했습니다. 이 장은 이러한 부활을 이끈 핵심 베이지안 계산 도구들을 다룹니다.

베이지안 방법은 흔히 무거운 컴퓨팅과 연관되어 있다고 여겨집니다. 하지만 이는 베이지안 패러다임 고유의 속성이 아닙니다. 베이지안 컴퓨팅은 표본 분포(sampling distribution)에 대한 빈도주의적(frequentist) 추정과 유사한, 사후 분포를 추정하기 위해 설계되었습니다. 두 문제 모두 까다롭지만, 빈도주의자들은 종종 문제를 단순화하기 위해 대규모 표본 정규 근사(large-sample normal approximations)를 활용합니다. 베이지안이 이와 유사한 근사를 수행하지 못할 이유는 없으며, 이것이 [섹션 3.1.3](./11-chapter3.md#sec3_1_3)의 주제입니다. 그럼에도 이러한 근사가 일반적이지 않은 이유는, MCMC가 훨씬 더 풍부한 출력을 생성하고, 근사의 정확도가 데이터의 표본 크기에 의해 제한되지 않기 때문입니다. 오히려 계산 노력을 늘림으로써 비정규 사후 분포에 대해서도 MCMC 근사를 임의로 정밀하게 만들 수 있습니다.

66\. 베이지안 계산은 본질적으로 고차원 사후 분포([섹션 1.4](./09-chapter1.md#sec1_4))를 요약하는 과정입니다. 이 장에서는 점 추정치만을 생성하지만 계산 속도가 빠른 요약 방법(예: MAP 추정)부터 전체 불확실성 정량화가 가능하지만 속도가 더 느린 방법(예: MCMC)에 이르기까지 넓은 스펙트럼의 방법을 다룹니다. 최대 우도 분석(maximum likelihood analysis) 사용자가 반복적 가중 최소 제곱 최적화(iteratively reweighted least squares optimization)의 이론 및 실제에 대해 전문가일 필요가 없듯이, 사용자가 이면에 사용되는 알고리즘을 깊이 이해하지 않고도 대부분의 베이지안 분석을 구현할 수 있도록 도와주는 소프트웨어([섹션 3.3](./11-chapter3.md#sec3_3))가 준비되어 있습니다. 그러나 사용자는 최소한 산출물을 검사하여 그것이 타당한지 판단할 수 있을 정도로 기저 알고리즘에 대해 충분히 이해하고 있어야 합니다. [섹션 3.1](./11-chapter3.md#sec3_1)과 [3.2.1](./11-chapter3.md#sec3_2_1)에서는 베이지안 컴퓨팅에서 사용되는 기본 알고리즘을 개략적으로 설명하고 예시를 위해 몇 가지 작은 예제를 진행합니다. 하지만 책의 나머지 부분에서는 MCMC 코드를 직접 작성하는 것을 피하기 위해 컴퓨팅 목적으로 [섹션 3.3](./11-chapter3.md#sec3_3)에서 소개한 JAGS 패키지를 사용할 것입니다.

## 3.1 결정론적 방법 (Deterministic methods)

### 3.1.1 최대 사후 확률 추정 (Maximum a posteriori estimation)

이 장에서 논의되는 대부분의 계산 방법은 전체 사후 분포를 요약하려고 시도합니다. 이는 파라미터에 대한 불확실성을 완벽하게 정량화할 수 있게 하며 특정 가설이 참일 사후 확률과 같은 추론으로 이어집니다. 하지만 전체 사후 분포를 요약하는 것은 어려울 수 있으며 일부 설정에서는 불필요할 수도 있습니다. 예를 들어 머신러닝에서 통계적 추정은 현재 데이터로부터 도출된 추정치를 사용해 미래 값을 예측하는 것이 유일한 목적인 필터링 프로세스로 사용됩니다. 이때 불확실성을 고려하고 통계적 유의성에 접근하는 것은 우선순위가 높지 않습니다. 이러한 경우, 단일 점 추정치가 사후 분포의 충분한 요약이 될 수 있습니다.

MAP ([섹션 1.4.1](./09-chapter1.md#sec1_4_1)) 점 추정량(point estimator)은 다음과 같습니다.

θ^MAP\=argmaxθlog\[p(θ|Y)\]\=argmaxθlog\[f(Y|θ)\]+log\[π(θ)\].(3.1)

MAP 추정량은 적분이 필요한 사후 평균과 달리 사후 분포의 최적화(optimization)를 요구합니다. 일반적으로 최적화는 특히 고차원에서 적분보다 빠르며, 따라서 MAP 추정은 어려운 문제에도 적용될 수 있습니다.

MAP 해는 미적분학이나 수치적 최적화(numerical optimization) \[[117](./19-ref01.md#refbib117)\]를 사용하여 찾을 수 있습니다. 가장 간단한 최적화 알고리즘은 경사 상승법(gradient ascent, 혹은 함수 최소화를 위한 경사 하강법(gradient descent))으로, 초기값 θ(0)으로 시작해 반복(iteration) _t_ 에서 다음 규칙에 따라 파라미터를 업데이트하는 반복적 알고리즘입니다.

θ(t)\=θ(t−1)+γ∇(θ(t−1)),

여기서 스텝 크기 _γ_ 는 튜닝 파라미터이고, 로그 사후 분포의 경사 벡터(gradient vector)는 ∇(θ)\=\[∇1(θ),...,∇p(θ)\]T 이며 다음과 같이 주어집니다.

∇j(θ)\=∂∂θj{log\[f(Y|θ)\]+log\[π(θ)\]}.

67\. 이 단계는 수렴, 즉 θ(t)≈θ(t−1) 이 될 때까지 반복됩니다. 경사 상승법은 사후 분포가 볼록(convex)할 때 가장 효과적이며, 수렴을 보장하기 위해 일반적으로 여러 초기값이 필요합니다.

통계 컴퓨팅 최적화에 관해서는 방대한 문헌이 있습니다. 다른 유용한 방법으로는 뉴턴 방법(Newton's method), 기댓값 최대화(EM, expectation-maximization) 알고리즘, MM(majorize-minimization) 알고리즘, 유전 알고리즘(genetic algorithms) 등이 있습니다. R에는 다목적 알고리즘인 `optim`을 비롯하여 많은 최적화 루틴이 포함되어 있으며, 이는 데이터 Y\=(2.68,1.18,−0.97,−0.98,−1.03) 및 사전 분포 μ∼Normal(0,1002) 와 σ∼Unif(0,10) 를 갖는 모델 Yi|μ,σ∼iid Normal(μ,σ2) 에 대해 [리스트 3.1](./11-chapter3.md#list3_1)과 [그림 3.1](./11-chapter3.md#fig3_1)에서 설명되어 있습니다.

![A shaded density over a grid is plotted with mu on the horizontal axis and sigma on the vertical axis. Darkest shading appears near mu around 0 and sigma around 2, fading outward. Two points are marked: a point slightly above sigma equal to 2 and a point slightly below it. The background grid extends across mu from about minus 4 to 6 and sigma from 0 to 10.](./images/fig3_1.jpg)

그림 3.1 **수치 적분 및 최적화(Numerical integration and optimization)**. 작은 점들은 그리드 포인트 θj\=(μj∗,σj∗) 이며, 음영 색상은 데이터 Y\=(2.68,1.18,−0.97,−0.98,−1.03) 와 사전 분포 μ∼Normal(0,1002), σ∼Uniform(0,10) 를 가지는 모델 Yi|μ,σ∼Normal(μ,σ2) 의 사후 밀도를 나타냅니다. 큰 검은 점은 수치 적분을 사용하여 찾은 근사 사후 평균이고 큰 흰 점은 수치적 최적화를 사용하여 찾은 근사 사후 최빈값(posterior mode)입니다. [본문으로 돌아가기.⏎](chapter3)

리스트 3.1 사후 분포 요약을 위한 수치적 최적화 및 적분. [본문으로 돌아가기.⏎](chapter3)

```R
library(cubature)
Y <- c(2.68, 1.18, -0.97, -0.98, -1.03) # 데이터(Data)

# 플로팅을 위해 그리드에서 밀도 계산(Evaluate the density on the grid for plotting)
 m     <- 50
 mu    <- seq(-4,6,length=m)
 sigma <- seq(0,10,length=m)
 theta <- as.matrix(expand.grid(mu,sigma))
 D  <- dnorm(theta[,1],0,100)*dunif(theta[,2],0,10) # 사전 분포(Prior)
 for(i in 1:length(Y)){ # 우도(Likelihood)
   D  <- D * dnorm(Y[i],theta[,1],theta[,2])
 }
   W  <- matrix(D/sum(D),m,m)

# MAP 추정(MAP estimation)
 neg_log_post <- function(theta,Y){
   log_like <- sum(dnorm(Y,theta[1],theta[2],log=TRUE))
   log_prior <- dnorm(theta[1],0,100,log=TRUE)+
                 dunif(theta[2],0,10,log=TRUE)
 return(-log_like-log_prior)}

 inits <- c(mean(Y),sd(Y))
 MAP <- optim(inits,neg_log_post,Y=Y,
               method = "L-BFGS-B", # 사전 분포에 경계가 있으므로(Since the prior is bounded)
               lower = c(-Inf,0), upper = c(Inf,10))$par

# 사후 평균 계산(Compute the posterior mean)
 post <- function(theta,Y){
   like <- prod(dnorm(Y,theta[1],theta[2]))
   prior <- dnorm(theta[1],0,100)*dunif(theta[2],0,10)
 return(like*prior)}

 g0 <- function(theta,Y){post(theta,Y)}
 g1 <- function(theta,Y){theta[1]*post(theta,Y)}
 g2 <- function(theta,Y){theta[2]*post(theta,Y)}
 m0 <- adaptIntegrate(g0,c(-5,0.01),c(5,5),Y=Y)$int #상수(constant) m(Y)
 m1 <- adaptIntegrate(g1,c(-5,0.01),c(5,5),Y=Y)$int
 m2 <- adaptIntegrate(g2,c(-5,0.01),c(5,5),Y=Y)$int
 pm <- c(m1,m2)/m0

# 플롯 생성(Make the plot)
 image(mu,sigma,W,col=gray.colors(10,1,0),
       xlab=expression(mu),ylab=expression(sigma))
 points(theta,cex=0.1,pch=19)
 points(pm[1],pm[2],pch=19,cex=1.5)
 points(MAP[1],MAP[2],col="white",cex=1.5,pch=19)
 box()
```

### 3.1.2 수치 적분 (Numerical integration)

관심 있는 많은 사후 요약통계량(사후 평균, 공분산 및 가설 확률 포함)은 사후 분포에 대한 _p_ 변수 적분으로 작성될 수 있습니다. 예를 들어 _θ_1 에 대한 주변 사후 평균(marginal posterior mean), 분산 및 _θ_1 이 상수 _c_ 를 초과할 확률은 다음과 같습니다.

E(θ1|Y)\=∫θ1p(θ|Y)dθ
Var(θ1|Y)\=∫\[θ1−E(θ1|Y)\]2p(θ|Y)dθ
Prob(θ1\>c|Y)\=∫c∞∫...∫p(θ|Y)dθ1dθ2,...,dθp.(3.2)

68\. 69\. 이러한 요약이 사후 분포를 묘사하는 데 충분하다면 베이지안 계산을 위해 수치 적분을 사용할 수 있습니다.

(3.2)의 모든 요약은 어떤 함수 _g_ 에 대해 E\[g(θ)\] 로 작성할 수 있습니다. 예컨대 Prob(θ1\>c|Y) 는 g(θ)\=I(θ1\>c) 를 사용하며, 여기서 I(θ1\>c)\=1 은 θ1\>c 일 때 1이고 그렇지 않으면 0입니다. 점들의 그리드 θ1∗,...,θm∗ 가 사후 밀도가 0이 아닌 θ 의 범위를 포괄한다고 가정하겠습니다. 그러면 다음과 같습니다.

E\[g(θ)\]\=∫g(θ)p(θ|Y)dθ≈∑j\=1mg(θj)Wj(3.3)

여기서 _W_ _j_ 는 그리드 포인트 _j_ 에 부여된 가중치입니다. g(θ) 의 사후 평균을 근사화하기 위해 가중치는 사후 밀도 함수(PDF) Dj\=f(Y|θj∗)π(θj∗) 와 연관되어야 합니다. 정규화 상수 m(Y) 는 계산할 수 없으므로, 사후 분포는 그리드 포인트에서 Wj\=Dj/∑l\=1mDl 과 같이 정규화됩니다.

이 방식은 θ 의 사후 분포에 대해 매우 간단한 근사를 제공합니다. 물론, [리스트 3.1](./11-chapter3.md#list3_1)과 [그림 3.1](./11-chapter3.md#fig3_1)에 예시로 제시된 R 함수 `adaptIntegrate`처럼 더욱 정밀한 수치 적분을 사용할 수도 있습니다. 그러나 파라미터 수 _p_ 에 비례해 확장성이 좋지 않으므로 이러한 방법에 초점을 맞추지는 않습니다. 예를 들어 p\=10 인 각 파라미터에 대해 20개의 그리드 포인트를 사용하면 확장된 그리드에는 m\=2010 개의 포인트가 있으며 이는 1조 개가 넘는 수치입니다!

### 3.1.3 베이지안 중심 극한 정리 (Bayesian central limit theorem (CLT))

중심 극한 정리는 대규모 표본에서 표본 평균의 표본 분포가 비정규 데이터에 대해서도 근사적으로 정규 분포를 따른다고 명시합니다. 이는 MLE(최대우도추정량)의 표본 분포로 확장될 수 있습니다. 많은 빈도주의자 표준 오차 및 p-값 계산은 다음 근사에 의존합니다.

θ^MLE∼Normal(θ0,Σ^MLE),(3.4)

여기서 θ0 는 실제 값이고 p×p 공분산 행렬은 Σ^MLE\=(−H)−1 이며, 헤세(Hessian) 행렬 **H** 의 (j,k) 요소는 다음과 같습니다.

∂2∂θj∂θklog\[f( Y|θ)\]|θ\=θ^MLE.(3.5)

유사한 베이지안 근사를 정의하기 위해, i\=1,...,n 에 대해 Yi|θ∼indepf(Yi|θ) 이고 θ∼π(θ) 라고 가정해 보겠습니다. 일반적인 조건([섹션 9.2.2](./17-chapter9.md#sec9_2_2) 참고) 하에서 _n_ 이 크면 사후 분포는 근사적으로 다음과 같습니다.

θ|Y∼Normal(θ^MAP,Σ^MAP)(3.6)

여기서 Σ^MAP\=(−H)−1 은 헤세 행렬 **H** 가 다음과 같다는 점을 제외하면 Σ^MLE 와 동일한 방식으로 정의됩니다.

∂2∂θj∂θklog\[p(θ|Y)\]|θ\=θ^MAP,(3.7)

이는 표본 크기가 큰 경우 Σ^MLE 와 비슷해질 것입니다. 물론 파라미터가 이산형(discrete)인 경우에는 적절하지 않지만, 베이지안 CLT는 관측치가 종속적인 일부 상황에서도 활용될 수 있습니다.

70\. 예를 들어, 제프리스 사전 분포(Jeffreys prior) θ∼Beta(1/2,1/2) 를 가지는 베타-이항(beta-binomial) 모델 Y|θ∼Binomial(n,θ) 를 생각해 보겠습니다. MAP 추정치는 θ^MAP\=A/(A+B) 이며, 근사 사후 분산은 다음과 같습니다.

\[Aθ^MAP2+B(1−θ^MAP)2\]−1,

여기서 A\=Y−0.5 이고 B\=n−Y−0.5 입니다. 물론 이 단순한 경우에는 정확한 사후 분포가 θ|Y∼Beta(Y+1/2,n−Y+1/2) 이므로 근사가 필요하지 않지만, [그림 3.2](./11-chapter3.md#fig3_2)는 _n_ 이 클 때 가우스(Gaussian) 근사가 잘 작동함을 보여줍니다.

![Three panels compare Exact C L T and M A P posterior curves for theta. For Y equal to 3 n equal to 10 the curves peak near 0.3 with small shape differences. For Y equal to 9 n equal to 30 they align more closely. For Y equal to 30 n equal to 100 they almost coincide in a very narrow peak near 0.3.](./images/fig3_2.jpg)

그림 3.2 **베이지안 중심 극한 정리(CLT) 예시.** 다양한 _Y_ 와 _n_ 값들에 대해 사전 분포 θ∼Beta(1/2,1/2) 를 가진 모델 Y|θ∼Binomial(n,θ) 에서의 정확한 Beta(Y+1/2,n+1/2) 사후 분포와 가우스(Gaussian) 근사 간의 비교. [본문으로 돌아가기.⏎](chapter3)

[섹션 3.1.1](./11-chapter3.md#sec3_1_1)에서 논의된 바와 같이, 사전 분포가 무정보적(uninformative)인 경우 MAP와 MLE 추정치는 비슷해지며, 이러한 근사 기법은 사전 분포와 무관하게 표본 크기가 커지면 두 방법론이 비슷한 사후 분포를 가짐을 시사합니다. 이는 처음으로 접하는 반복되는 주제입니다: 베이지안 방법과 최대 우도 방법은 표본 크기가 클 경우 사전 분포와 상관없이 비슷한 결과를 도출합니다.

베이지안 CLT 근사의 장점은 적분([섹션 3.1.2](./11-chapter3.md#sec3_1_2))을 미분으로 대체하므로, 특히 _p_ 가 클 때 계산하기 더 쉬우며 방법이 확률적이기보다는 결정론적이라는 점입니다. 단점은 표본 크기가 작거나 보통인 경우에는 근사의 정확도가 떨어질 수 있다는 점입니다. 하지만 통합 내포 라플라스 근사(INLA, Integrated Nested Laplace Approximation) \[[135](./19-ref01.md#refbib135)\] ([부록 A.4](./18-appA.md#secA_4))와 같이 수치 적분과 가우스 근사를 결합한 좀 더 정교한 근사 방법들도 제안되었습니다. 혹은 변분 베이지안 추론(variational Bayesian inference) \[[85](./19-ref01.md#refbib85)\]에서는 사용자가 파라메트릭(잠재적으로 비정규) 사후 분포를 가정한 다음 가정한 사후 분포에서 실제 사후 분포에 가장 근접한 근사를 제공하는 파라미터의 값을 찾습니다.

### 3.1.4 변분 베이즈 방법 (Variational Bayesian (VB) Methods)

베이지안 CLT로 동기 부여된 결합 사후 분포에 대한 정규 근사는 표본 크기(_n_)가 크고 모델 크기(_p_)가 작고 고정되어 있을 때 이상적입니다. 이러한 조건 하에서는 이 근사 기법이 이론적으로 타당하고 계산적으로 효율적입니다. 그러나 크기가 작거나 대형 모델인 경우 사후 분포는 정규 분포로 잘 근사되지 않을 수 있으며 이로 인해 성능이 떨어집니다. 더 나아가, 모델 크기가 큰 경우 근사 사후 공분산 Σ^MAP 는 대략 _p_2 개의 요소가 계산되어야 하므로 계산이 느려집니다. p\=1,000 이라면 Σ^MAP 에는 500,500개의 고유한 요소가 존재합니다.

166\. 71\. 변분 베이즈(VB) 근사는 이러한 한계점을 모두 해결하여 고차원 문제에 대한 선도적인 계산 방법 중 하나로 부상했습니다. 베이지안 CLT 근사와 마찬가지로 VB는 사후 분포가 확률분포의 파라메트릭 계열에 있다고 가정합니다. 하지만 이 분포는 반드시 가우스적(Gaussian)이어야 할 필요는 없으며 심지어 모든 파라미터에 대해 같은 분포족일 필요도 없습니다. 예컨대 분산 파라미터의 사후 분포는 감마 분포를 따르고 평균 파라미터는 정규 분포를 따른다고 가정할 수 있습니다. 사후 공분산 항을 계산하지 않기 위해, 사후 분포에서 일부 또는 모든 파라미터가 독립이라고 가정할 수도 있습니다. 이러한 가정들은 엄청난 계산적 이득을 줄 수 있지만 가정을 심각하게 위반하면 성능이 매우 저하되므로 주의해서 적용해야 합니다.

VB 분석은 변분 사후 분포(variational posterior)를 지정하는 것으로 시작됩니다. 변분 사후 분포는 실제 사후 분포에 대한 좋은 근사를 포함한다고 가정된 분포의 파라메트릭 계열입니다. 변분 사후 밀도 함수를 q(θ|v) 로 나타내며, 여기서 v\=(v1,...,vq) 는 변분 파라미터(variational parameters)입니다. 예를 들어 지지 집합(support)이 \[0,1\] 인 단일 파라미터 _θ_ 가 있다면 q(θ|v) 를 Beta(v1,v2) 밀도 함수로 선택하고 일부 v\=(v1,v2) 값에 대해 p(θ|Y)≈q(θ|v) 라고 가정할 수 있습니다.

사후 분포가 파라메트릭 족인 _q_ 에 있다는 가정은, 문제 자체를 실제 분포를 가깝게 근사하게 해주는 변분 파라미터 **v** 의 값을 찾는 수준으로 단순화시킵니다. 이러한 파라미터는 변분 분포와 실제 사후 분포 간의 차이를 최소화하도록 선택됩니다. 이는 두 분포 간의 유사성 측정을 요구하며, 일반적으로 쿨백-라이블러 발산(Kullback–Leibler divergence, KLD)이 사용됩니다([그림 3.3](./11-chapter3.md#fig3_3)). KLD는 다음과 같이 정의됩니다.

KLD(v)\=Eq\[log{q(θ|v)p(θ|Y )}\]\=∫log{q(θ|v)p(θ|Y)}q(θ|v)dθ.(3.8)

![A set of density curves is plotted with theta on the horizontal axis and Density on the vertical axis. The true posterior forms a tall narrow peak near theta around 0.7. Three Beta approximations Beta six three Beta sixty thirty and Beta fifty five thirty three appear as smoother wider curves, with Beta six three spreading broadly, Beta sixty thirty matching the central peak more closely, and Beta fifty five thirty three lying slightly left of the true peak.](./images/fig3_3.jpg)

그림 3.3 **쿨백-라이블러 발산(Kullback–Leibler divergence, KLD) 예시.** (가상의) 실제 사후 분포(실선)와 세 개의 베타 분포(점선) 사이의 KLD는 Beta(6,3) 에 대해 0.74, Beta(60,30) 에 대해 0.31, Beta(55,33) 에 대해 0.08입니다. [본문으로 돌아가기.⏎](chapter3)

KLD는 임의의 **v** 에 대해 음수가 아니며 근사가 완벽하고 모든 θ 에 대해 q(θ|v)\=p(θ|Y) 이면 0이 됩니다. VB 근사는 KLD(v) 를 최소화하는 **v** 를 선택합니다.

사전 분포를 선택하는 것과 마찬가지로 유일하게 옳은 단 하나의 변분 사후 족(실제 사후 분포인 사소한 예외를 제외하고)은 없으나 파라미터와 동일한 지지 집합을 가지는 분포 족을 선택하는 것이 좋은 출발점입니다. 특히 고차원 문제의 경우 파라미터 간의 의존성을 포착하는 변분 분포를 가정하는 것이 어려우므로, 파라미터의 사후 분포가 서로 독립이라고 가정하는 평균장 근사(mean-field approximation)가 매력적입니다.

q(θ|v)\=∏j\=1pqj(θj|vj).

최적의 **v** 를 찾기 위해서는 (3.8)의 적분을 계산하거나 근사한 다음 **v** 에 대해 최적화해야 합니다. 일부 경우, [섹션 2.1](./10-chapter2.md#sec2_1)에서 켤레 사전 분포(conjugate priors)를 찾기 위해 사용된 것과 유사한 기법을 사용하여 분석적으로 적분을 풀 수 있으며, 다른 경우에는 수치적 근사법([3.1.2](./11-chapter3.md#sec3_1_2)에서 언급된 알고리즘 사용)이 필요합니다. 최적화 역시 작은 규모의 문제에서는 분석적으로 이루어질 수 있고 대부분의 경우 수치적으로 이루어질 수 있습니다([섹션 3.1.1](./11-chapter3.md#sec3_1_1)에서 언급된 알고리즘 사용). 순차적으로 **v** 의 단일 요소에 대해 최적화하는 켤레 경사 하강법(Conjugate gradient descent)은 보통 **v** 가 고차원일 때 널리 쓰입니다.

이 접근법을 설명하기 위해, [그림 3.4](./11-chapter3.md#fig3_4)에서 강좌 웹사이트(https://data.giss.nasa.gov/gistemp/ 에서 다운로드됨)의 지구 평균 온도 이상(현재 연도와 1951-1980 평균 간의 차이) 데이터를 고려해 보겠습니다. 우리는 1차 자기회귀 시계열 모델(first-order autoregressive time series model)을 사용하여 1980년 이전의 안정적인 기간의 데이터만을 분석합니다. 데이터가 정규화되었으므로 평균 반응은 0이라고 가정합니다. 첫 번째 관측치는 Y1∼Normal(0,σ2) 이고, 후속 관측치는 Yt|Yt−1∼Normal(ρYt−1,1−ρ2σ2) 이며 여기서 ρ∈(0,1) 는 시간적 상관관계를 제어하고 _σ_2 는 분산을 제어합니다. 이 모델에서 주변 분포는 모든 _t_ 에 대해 평균이 0이고 분산이 _σ_2 이며 Cor(Yt,Yt+h)\=ρh 입니다. 72\. 베이지안 모델은 파라미터의 지지 집합과 일치하는 사전 분포 σ2∼InvGamma(a,b) 및 a\=b\=0.1 이고 c\=d\=1 인 ρ∼Beta(c,d) 를 지정함으로써 완성됩니다. 사후 분포는 다음과 같습니다.

p(σ2,ρ|Y)∝(σ2)−n/2−a−1(1−ρ2)−(n−1)exp(−S(ρ)2σ2)×(σ2)a−1exp(−bσ2)ρc−1(1−ρ)d−1(3.9)

![A time series plot shows temperature anomaly in degrees Celsius on the vertical axis and year on the horizontal axis from about 1880 to 2020. Individual yearly anomalies appear as circles connected by thin lines. Values fluctuate around zero until the mid twentieth century, then rise steadily with increasing variability. The highest anomalies occur after 2000, reaching nearly 1.0.](./images/fig3_4.jpg)

그림 3.4 **지구 평균 온도 데이터(Global mean temperature data).** 연도별 지구 평균 온도 이상(현재 연도와 1951-1980 평균 간의 차이). [본문으로 돌아가기.⏎](chapter3)

여기서 S(ρ)\=Y12+∑t\=2n(Yt−ρYt−1)2/(1−ρ2) 입니다.

두 파라미터의 지지 집합을 일치시키기 위해, 변분 사후 분포를 ρ|Y∼Beta(v3,v4) 와 독립인 σ2|Y∼InvGamma(v1,v2) 로 둡니다. 그러면 변분 사후 분포는 다음과 같습니다.

q(σ2,ρ|v)\=v2v1Γ(v1)(σ2)−(v1+1)exp(−v2/σ2)×Γ(v3+v4)Γ(v3)Γ(v4)ρv3−1(1−ρ)v4−1.(3.10)

따라서 사후 분포 요약은 (3.10)의 _q_ 가 (3.9)의 _p_ 와 근사해지도록 만드는 v\=(v1,v2,v3,v4) 의 값을 찾는 과정으로 귀결됩니다. KLD의 적분은 _σ_2 와 _ρ_ 모두에 대해 이루어지며, _σ_2 에 대해서는 분석적으로 계산할 수 있지만 _ρ_ 에 대해서는 일변량 수치 적분이 필요합니다.

[그림 3.5](./11-chapter3.md#fig3_5)의 상단 좌측 패널에 플롯팅된 실제 사후 분포는 두 파라미터 간의 의존성을 명확히 보여주어, 사후 분포의 독립이라는 평균장 근사를 위배하고 있습니다. 그 결과 우측 상단의 VB 근사는 전반적으로 열악합니다. 확실히 더 나은 적합성을 보여줄 더 풍부한 변분 족이 있겠지만, 이러한 단순한 평균장 73\. 근사의 경우에도 [그림 3.5](./11-chapter3.md#fig3_5) 하단 행의 주변 분포들은 실제 사후 분포 근처에 중심을 두고 있으며 따라서 VB 점 추정치들은 실제 사후 분포에서 나온 값들과 유사합니다.

![Four plots compare true and variational Bayes posteriors. The true joint posterior forms a curved ridge in rho and sigma squared, while the variational version concentrates near rho around 0.9 and sigma squared near 0.05. Marginal posteriors below show the variational curves as narrower and more sharply peaked than the corresponding true marginals.](./images/fig3_5.jpg)

그림 3.5 **지구 온도 데이터의 변분 베이지안 분석(Variational Bayesian analysis of the global temperature data).** 첫 번째 행은 정확한(exact) 결합 사후 분포와 VB 이변량 사후 분포를 보여주며, 두 번째 행은 해당하는 주변 분포를 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

## 3.2 확률적 방법 (Stochastic methods)

210\. 몬테카를로(MC, Monte Carlo) 방법은 표본을 사용하여 모집단에 대해 추론을 한다는 기본적인 통계적 개념을 거울처럼 보여주기 때문에 통계학자들에게 매력적입니다. 통계학에서 정형화된 문제는 모집단 평균이나 표준편차와 같은 모집단의 요약통계량을 추정하는 것입니다. 대부분의 경우(즉, 전수조사가 아닌 경우) 우리는 평균이나 표준편차를 직접 계산하기 위해 모집단 전체를 관찰할 수는 없으며 그 대신 표본을 추출하여 이를 바탕으로 모집단을 추론합니다. 예를 들어 우리는 표본 평균을 이용해 모집단 평균을 근사화하고, 표본이 충분히 크다고 가정하면 대수의 법칙(law of large numbers)에 의해 근사가 신뢰할 수 있음을 보장받습니다.

사후 분포에서 MC 샘플링도 동일한 방식으로 작동합니다. 이 비유에서 관심의 대상이 되는 모집단은 사후 분포입니다. 사후 분포의 평균과 분산을 이용해 이를 요약하고 싶지만 대부분의 경우 이러한 사후 요약통계량은 직접 계산할 수 없기 때문에 우리는 사후 분포로부터 표본을 추출하고 MC 표본 평균을 사용하여 사후 평균을 근사화하며 MC 표본 분산을 이용해 사후 분산을 근사화합니다. MC 표본이 충분히 크다고 가정하면 근사는 신뢰할 수 있습니다. 이는 사실상 어떠한 사후 분포, 심지어 고차원 적분이 필요한 사후 분포 요약에 대해서도 유효합니다.

표본 _s_ 가 θ(s)\=(θ1(s),...,θp(s)) 일 때, _S_ 개의 사후 표본인 θ(1),...,θ(S)∼p(θ|Y) 를 생성했다고 가정해 보겠습니다. 이 추출값들은 [섹션 1.4](./09-chapter1.md#sec1_4)에서 논의된 사후 분포의 요약을 근사화하는 데 쓰일 수 있습니다. p\=1 인 일변량 모델의 경우, _S_ 표본들의 히스토그램으로 사후 밀도를, MC 표본 평균인 E(θ1|Y)≈∑s\=1Sθ1(s)/S 으로 사후 평균을 근사화할 수 있으며, 74\. (이 페이지에 그림 존재) 75\. _S_ 표본들의 MC 표본 분산으로 사후 분산을, θ1(s)\>0 인 MC 표본 비율로 θ1\>0 일 확률 등을 근사화할 수 있습니다.

MC 샘플링의 수렴을 예시로 들기 위해, 사전 분포 θ∼Gamma(a,b) 를 가진 포아송-감마 모델 Y|θ∼Poisson(Nθ) 를 고려해 보겠습니다. 사후 분포는 θ|Y∼Gamma(Y+a,N+b) 가 됩니다. [그림 3.6](./11-chapter3.md#fig3_6)은 N\=10, Y\=8, a\=b\=0.1 임을 가정하고 근사 사후 평균과 θ\>0.5 일 확률을 MC 반복 횟수에 대한 함수로 플롯팅했습니다. 표본 평균과 비율은 적은 수의 MC 표본에 대해서는 노이즈가 있지만 표본 수가 증가함에 따라 실제 사후 평균과 θ\>0.5 일 확률에 수렴합니다.

![Two trace plots display convergence over iterations S on the horizontal axis. The left plot shows sample mean on the vertical axis fluctuating sharply in early iterations then settling near a stable horizontal reference line around 0.8. The right plot shows sample probability with similar early volatility before stabilising close to the same reference value. Both plots extend to iteration 5000 with lines gradually flattening as S increases.](./images/fig3_6.jpg)

그림 3.6 **몬테카를로 근사의 수렴(Convergence of a Monte Carlo approximation).** θ(s)∼iidGamma(8.1,10.1) 이라 가정할 때 아래 플롯은 표본 수 _S_ 에 따른 표본 평균 ∑s\=1Sθ(s)/S 및 표본 비율 ∑s\=1SI(θ(s)\>0.5)/S 을 보여줍니다. 수평선은 실제 값을 나타냅니다. [본문으로 돌아가기.⏎](chapter3)

p\>1 개의 파라미터가 있는 다변량 모델의 경우, 표본 θj(1),...,θj(S) 는 _θ_ _j_ 의 주변 사후 분포 p(θj|Y) 를 따릅니다. 결정적으로, 우리는 p(θj|Y)\=∫f(θ|Y)dθ1...dθj−1dθj+1...dθp 를 분석적으로 적분할 필요가 없습니다. 왜냐하면 각 표본은 모든 파라미터에 대한 무작위 추출로 구성되기 때문에 MC 샘플링은 다른 파라미터의 불확실성을 감안하여 자동으로 주변 분포 _θ_ _j_ 로부터 샘플을 생성하기 때문입니다.

주변 분포 이외에도 MC 샘플링은 원래 파라미터의 변환으로 정의된 파라미터의 사후 분포를 근사화하는 데 쓰일 수 있습니다. 예를 들어, 어떤 함수 _g_ 에 대해 γ\=g(θ1) 의 사후 분포는 γ(s)\=g(θ1(s)) 인 MC 표본 γ(1),...,γ(S) 를 사용하여 근사할 수 있습니다. 즉, 우리는 그저 각 표본을 변환하면 되며, 변환된 표본은 _γ_ 의 사후 분포를 근사화합니다. 변환된 파라미터의 사후 분포는 다른 파라미터들과 마찬가지로 예를 들어 E(γ|Y)≈∑s\=1Sγ(s)/S 와 같이 요약됩니다. 또 다른 예로, 만약 Δ\=θ1−θ2\>0 인지 검정하고 싶다고 해보겠습니다. Δ 의 사후 분포는 _S_ 개의 표본 Δ(s)\=θ1(s)−θ2(s) 로 근사되며 따라서 Prob(Δ\>0|Y) 는 Δ(s)\>0 인 _S_ 개 표본의 비율을 사용하여 근사됩니다.

사후 표본을 가지고 있다면 사후 분포 혹은 복잡한 사후 분포의 함수를 요약하는 일은 간단하며 이것이 바로 MC 샘플링의 매력 중 하나입니다. 그러나 사후 분포로부터 유효한 표본을 만들어내는 일은 항상 간단한 것은 아닙니다.

### 3.2.1 76\. MCMC 샘플링 (MCMC sampling)

우리는 두 가지 샘플링 알고리즘인 깁스 샘플링(Gibbs sampling)과 메트로폴리스-해이스팅스 샘플링(Metropolis–Hastings sampling)에 초점을 맞출 것입니다. 각 파라미터의 조건부 분포에서 직접 샘플링할 수 있을 때에는 깁스 샘플링이 선호됩니다. 메트로폴리스-해이스팅스는 좀 더 복잡한 문제로 일반화된 형태입니다. 또한 지연 기각 적응형 메트로폴리스(DRAM, delayed rejection and adaptive Metropolis), 메트로폴리스 조정 랑주뱅 알고리즘(Metropolis-adjusted Langevin algorithm), 해밀토니안 MC(HMC) 및 슬라이스 샘플링(slice sampling)을 포함한 다른 더 발전된 알고리즘을 [부록 A.4](./18-appA.md#secA_4)에서 간략히 언급합니다.

#### 3.2.1.1 깁스 샘플링 (Gibbs sampling)

동기가 되는 예로 다음 가우스(Gaussian) 모델을 고려해 보겠습니다.

Y1,...,Yn|μ,σ2∼iid Normal(μ,σ2)(3.11)

여기서 사전 분포는 μ∼Normal(γ,τ2) 및 σ2∼InvGamma(a,b) 입니다. θ\=(μ,σ2) 의 사후 분포를 연구하기 위해, 우리는 결합 사후 분포 p(θ|Y) 로부터 _S_ 번의 추출을 시행하려 합니다. [섹션 2.1.3](./10-chapter2.md#sec2_1_3)은 μ|σ2,Y 가 가우스 분포를 따르기 때문에 _σ_2 를 아는 것으로 가정하고 _μ_ 의 사후 분포로부터 샘플을 추출하는 수단을 제공하며, [섹션 2.1.4](./10-chapter2.md#sec2_1_4)는 σ2|μ,Y 가 역-감마(inverse gamma) 분포를 따르기 때문에 _μ_ 가 알려진 것으로 가정하고 _σ_2 의 사후 분포에서 추출하는 수단을 제공합니다. 깁스 샘플링은 이 일변량 조건부 분포들만을 사용하여 원하는 θ 의 결합 사후 분포로부터 추출값을 생성합니다.

깁스 샘플링(\[[59](./19-ref01.md#refbib59)\]에 의해 제안됨)은 두 파라미터의 초기값으로 시작하여, _S_ 개의 샘플을 수집할 때까지 μ|σ2,Y 를 샘플링하는 것과 σ2|μ,Y 를 샘플링하는 과정을 번갈아 가며 진행합니다(_S_ 를 선택하는 방안에 대한 광범위한 논의는 [섹션 3.4](./11-chapter3.md#sec3_4)를 참조하세요). 예컨대 우리는 **Y** 의 표본 평균을 _μ_ 로 설정하고 **Y** 의 표본 분산을 _σ_2 로 설정하여 시작할 수 있습니다(초기화에 대한 광범위한 연구는 [섹션 3.4](./11-chapter3.md#sec3_4) 참고). 연속된 단계를 구현하려면 완전 조건부 사후 분포(full conditional posterior distributions), 즉 데이터와 다른 모든 파라미터를 조건으로 하는 어느 한 파라미터의 분포를 도출해야 합니다. (사전 분포를 약간 수정한) [섹션 2.1.3](./10-chapter2.md#sec2_1_3) 및 [2.1.4](./10-chapter2.md#sec2_1_4)를 따르면, 완전 조건부 사후 분포는 다음과 같습니다.

μ|σ2,Y∼Normal(∑inYi/σ2+γ/τ2n/σ2+1/τ2,1n/σ2+1/τ2)
σ2|μ,Y∼InvGamma(n/2+a,∑i\=1n(Yi−μ)2/2+b).

그 결과 사후 분포를 요약할 _S_ 개의 사후 표본 θ(1),...,θ(S) 이 도출되며 여기서 θ(s)\=(μ(s),σ2(s)) 입니다. [리스트 3.2](./11-chapter3.md#list3_2)는 [섹션 1.1.2](./09-chapter1.md#sec1_1_2)와 동일한 데이터를 사용하여 이 단계를 수행하는 R 코드를 제공합니다. [그림 3.7](./11-chapter3.md#fig3_7)(하단 행)은 모두 [그림 1.11](./09-chapter1.md#fig1_11)과 매우 유사한, 결합 사후 분포 및 _μ_ 의 주변 밀도로부터의 추출값을 플롯팅합니다.

리스트 3.2 미지의 평균 및 분산을 갖는 가우스 모델에 대한 깁스 샘플링. [본문으로 돌아가기.⏎](chapter3)

```R
 # 데이터 로드(Load the data)

 Y  <-  c(2.68,1.18,-0.97,-0.98,-1.03)
 n  <-  length(Y)

# MCMC 표본을 위한 빈 행렬 생성(Create an empty matrix for the MCMC samples)

  S                 <- 25000
  samples           <- matrix(NA,S,2)
  colnames(samples) <- c("mu","sigma")

# 초기값(Initial values)

 mu   <- mean(Y)
 sig2 <- var(Y)

# 사전 분포(priors): mu ~ N(gamma,tau), sig2 ~ InvG(a,b)

 gamma <- 0
 tau   <- 100^2
 a     <- 0.1
 b     <- 0.1

# 깁스 샘플링(Gibbs sampling)

 for(s in 1:S){
   P    <- n/sig2 + 1/tau
   M    <- sum(Y)/sig2 + gamma/tau
   mu   <- rnorm(1,M/P,1/sqrt(P))

   A    <- n/2 + a
   B    <- sum((Y-mu)^2)/2 + b
   sig2 <- 1/rgamma(1,A,B)

     samples[s,]<- c(mu,sqrt(sig2))
 }

# 결합 사후 분포와 mu의 주변 분포 플롯팅(Plot the joint posterior and marginal of mu)
 plot(samples,xlab=expression(mu),ylab=expression(sigma))
 hist(samples[,1],xlab=expression(mu))

 # 사후 평균, 중앙값 및 신뢰 구간(Posterior mean, median and credible intervals)
 apply(samples,2,mean)
 apply(samples,2,quantile,c(0.025,0.500,0.975))
```

```R
 Y  <-  c(2.68,1.18,-0.97,-0.98,-1.03)
 n  <-  length(Y)

# MCMC 표본을 위한 빈 행렬 생성(Create an empty matrix for the MCMC samples)

  S                 <- 25000
  samples           <- matrix(NA,S,2)
  colnames(samples) <- c("mu","sigma")

# 초기값(Initial values)

 mu   <- mean(Y)
 sig2 <- var(Y)

# 사전 분포(priors): mu ~ N(gamma,tau), sig2 ~ InvG(a,b)

 gamma <- 0
 tau   <- 100^2
 a     <- 0.1
 b     <- 0.1

# 깁스 샘플링(Gibbs sampling)

 for(s in 1:S){
   P    <- n/sig2 + 1/tau
   M    <- sum(Y)/sig2 + gamma/tau
   mu   <- rnorm(1,M/P,1/sqrt(P))

   A    <- n/2 + a
   B    <- sum((Y-mu)^2)/2 + b
   sig2 <- 1/rgamma(1,A,B)

     samples[s,]<- c(mu,sqrt(sig2))
 }

# 결합 사후 분포와 mu의 주변 분포 플롯팅(Plot the joint posterior and marginal of mu)
 plot(samples,xlab=expression(mu),ylab=expression(sigma))
 hist(samples[,1],xlab=expression(mu))

 # 사후 평균, 중앙값 및 신뢰 구간(Posterior mean, median and credible intervals)
 apply(samples,2,mean)
 apply(samples,2,quantile,c(0.025,0.500,0.975))
```

![Four panels show M C M C output for mu and sigma. Trace plots display mu and sigma across 25000 iterations with dense fluctuations around stable central regions. A scatter plot of sampled mu and sigma forms a concentrated cloud centred near mu about 0 and sigma about 2 with wider spread upward. A histogram of mu shows a tall narrow peak near zero with frequencies tapering toward the extremes.](./images/fig3_7.jpg)

그림 3.7 **미지의 평균과 분산을 가진 가우스 모델에 대한 MCMC 사후 표본 요약(Summary of posterior samples from MCMC for the Gaussian model with unknown mean and variance).** 첫 번째 행은 _μ_ 와 _σ_ 에 대한 표본의 트레이스 플롯(trace plots)을 제공합니다. 두 번째 행은 (μ,σ) 의 결합 사후 분포와 _μ_ 의 주변 사후 분포로부터 얻은 표본들을 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

[알고리즘 1](./11-chapter3.md#algo3_1)은 깁스 샘플링의 일반적인 레시피를 제공합니다. 알고리즘은 파라미터 벡터에 대한 초기값으로 시작하며, 각 반복(iteration)은 다른 모든 파라미터를 조건으로 한 상태에서 개별 파라미터의 완전 조건부 분포로부터 업데이트를 수행하는 시퀀스로 구성됩니다. 각 단계가 파라미터들을 순환하며 다른 모든 파라미터의 현재 값이 주어진 상태에서 업데이트하기 때문에, 이 표본들은 독립적이지 않습니다. 모든 샘플링은 이전 반복의 값을 조건으로 수행되므로, 이 표본들은 마르코프 체인(Markov chain)이라는 특정한 유형의 확률 과정을 형성하며 이런 이유로 마르코프 체인 몬테카를로(MCMC) 샘플링이라는 이름이 붙었습니다. 사후 분포로부터 독립적인 추출값을 샘플링하려는 다른 MC 샘플러들(예: 기각 샘플링(rejection sampling) 및 근사 베이지안 계산(ABC) \[[100](./19-ref01.md#refbib100)\])이 존재하지만 고차원 문제에는 MCMC가 선호됩니다. 77\. 78\. (이 페이지에 그림 존재)

알고리즘 1 깁스 샘플링(Gibbs Sampling) [본문으로 돌아가기.⏎](chapter3)

```
1: 초기화(Initialize) θ(0)=(θ1(0),...,θp(0))
2: for s=1,...,S do
3:        for j=1,...,p do
4:               sample θj(s)∼pj(θj|θ1(s),...,θj−1(s),θj+1(s−1),...,θp(s−1),Y)
5:         end for
6: end for
```

79\. 이 알고리즘의 묘미는 다변량 분포에서 샘플링하는 까다로운 문제를 더 단순한 일변량 문제들의 시퀀스로 줄여준다는 것입니다. 이는 완전 조건부 분포에서 쉽게 샘플링할 수 있다는 것을 전제로 하지만 다음 예시들에서 보게 되듯, _p_ 가 큰 고차원 문제라 하더라도 완전 조건부 분포는 종종 샘플링하기 좋은 익숙한 켤레 쌍(conjugacy pairs, [섹션 2.1](./10-chapter2.md#sec2_1))을 따릅니다.

다른 예제로 넘어가기에 앞서, 잠시 멈추고 이 알고리즘의 이론적 속성에 대해 논의해 보겠습니다. 왜 이것이 작동할까요? 다시 말해, 완전 조건부 분포로부터의 반복적인 샘플링이 결합 사후 분포의 표본들로 이어지는 이유는 무엇일까요? 확실히 초기값은 사용자에 의해 주관적으로 선택되기 때문에 사후 분포에서 나온 샘플이라고 믿을 수는 없습니다. 또한 첫 번째 무작위 샘플 역시 사후 분포에서 멀리 떨어져 있을 수 있는 초기값에 의존하기 때문에 사후 분포를 따른다고 믿는 것은 위험합니다. [부록 A.3](./18-appA.md#secA_3)은 확률 과정 이론(stochastic process theory)을 이용해 다음을 보여줍니다. (1) 일반적인 조건 하에서, 알고리즘이 생성한 표본은 어떤 초기값을 선택하든 사후 분포에 수렴하며, (2) 사후 분포에서 표본이 한 번 추출되면 그 후의 모든 표본들도 사후 분포로부터의 표본입니다.

이러한 이론적 수렴 논거는 깁스 샘플링이 실제로 어떻게 사용되는지를 결정합니다. 사용자는 체인이 수렴할 때까지 계속 따라가며, 이 번인(burn-in, 혹은 워밍업) 기간 동안의 모든 이전 표본들은 폐기합니다. 수렴은 종종 트레이스 플롯(즉, 반복 횟수에 따른 표본들의 플롯)을 시각적으로 검사하여 평가되지만, 형식적인 측정 방법들도 제공됩니다([섹션 3.4](./11-chapter3.md#sec3_4)). 수렴 이후 남은 모든 표본들은 사후 분포를 요약하는 데 사용됩니다. 다른 최적화 알고리즘들과 달리 우리는 깁스 샘플링이 단일 최적값에 수렴할 것으로 기대하지 않는다는 점을 기억해야 합니다. 그보다, 번인(burn-in) 후에 알고리즘이 사후 분포로부터 표본을 생성하기를 바랍니다. 즉, 알고리즘에 의해 생성된 표본들의 분포가 사후 분포로 수렴하고 어느 한 곳에 갇히지 않기를 바라는 것입니다. [그림 3.8](./11-chapter3.md#fig3_8)은 반복 횟수 1000회 부근에서 두 파라미터 모두 수렴이 뚜렷하게 일어나고, 이 지점 이후로는 나머지 표본들이 일관되게 동일한 분포에서 추출되어 트레이스 플롯이 "바코드" 또는 "애벌레"와 같은 형태를 띠는 이상화된 출력 결과를 제공합니다.

![Two trace plots show convergence for theta one and theta two over 5000 iterations. Theta one rises quickly from about minus 2 to around 4 then fluctuates tightly around that level after the marked burn in point near iteration 1000. Theta two begins near 20 then declines steeply to about 10 where it stabilises with small fluctuations after the same burn in point.](./images/fig3_8.jpg)

그림 3.8 **깁스 샘플링 트레이스 플롯(Gibbs sampling trace plots).** j\=1,2 에 대한 사후 표본 θj(s) 의 반복 횟수 _s_ 에 따른 이 트레이스 플롯들은 1000번째 반복에서 수렴함을 보여주며, 이 지점은 수직선으로 표시되어 있습니다. [본문으로 돌아가기.⏎](chapter3)

**이변량 정규 분포 예시(Bivariate normal example)**: 우리는 깁스 샘플링을 이용해 사후 분포로부터 추출값을 생성할 것이지만, 실제로는 (완전 조건부 분포에서 샘플링할 수만 있다면) 어떠한 분포에서 샘플링하는 데에도 사용할 수 있습니다. 예를 들어 θ\=(U,V) 가 평균이 0이고 두 파라미터의 분산이 1이며 파라미터 간의 상관관계가 _ρ_ 인 이변량 정규 분포를 따른다고 해 보겠습니다. 이는 데이터를 조건으로 하는 것이 아니므로 사후 분포는 아니지만, 이 장난감(toy) 예제에서는 알고리즘에 대한 감각과 이 알고리즘이 목표 분포에 수렴하는 방식을 이해하기 위해 깁스 샘플링을 사용해 이 이변량 정규 분포로부터 추출을 진행합니다. (1.26)에 제시된 바와 같이([그림 1.7](./09-chapter1.md#fig1_7) 참고), 완전 조건부 분포는 U|V∼Normal(ρV,1−ρ2) 및 V|U∼Normal(ρU,1−ρ2) 입니다. 초기값 (U(0),V(0)) 이 주어졌을 때, 첫 세 번의 반복은 다음과 같습니다.

* 1a. U(1)|V(0)∼Normal(ρV(0),1−ρ2) 를 추출합니다.
* 1b. V(1)|U(1)∼Normal(ρU(1),1−ρ2) 를 추출합니다.
* 2a. U(2)|V(1)∼Normal(ρV(1),1−ρ2) 를 추출합니다.
* 2b. 80\. V(2)|U(2)∼Normal(ρU(2),1−ρ2) 를 추출합니다.
* 3a. U(3)|V(2)∼Normal(ρV(2),1−ρ2) 를 추출합니다.
* 3b. V(3)|U(3)∼Normal(ρU(3),1−ρ2) 를 추출합니다.

이 알고리즘은 _S_ 번의 추출이 이루어질 때까지 계속됩니다.

_s_ 번의 반복 후의 분포는 다음과 같음을 보일 수 있습니다.

U(s)∼Normal(ρ2s−1V(0),1−ρ4s−2).

_U_ 의 실제 주변 분포는 Normal(0,1) 이고 큰 _s_ 에 대해 ρ2s−1≈0 이고 ρ4s−2≈0 이므로, 초기값에 관계없이 MCMC 사후 분포는 실제 사후 분포와 매우 가깝지만 결코 완전히 일치하지는 않습니다. ρ\=0 이면 수렴이 즉각적이고 |ρ|≈1 이면 수렴이 느리며, 이는 파라미터 간의 교차 상관관계(cross-correlation)가 MCMC 수렴을 방해할 수 있음을 보여줍니다(예: [그림 3.11](./11-chapter3.md#fig3_11) 및 아래의 블록형(blocked) 깁스 샘플링에 대한 논의 참조). 그럼에도 불구하고 어떤 _ρ_ 에 대해서든 큰 _S_ 를 취하면 근사는 확실히 충분하며 _S_ 를 증가시킴으로써 임의로 정밀해질 수 있습니다.

**NFL 뇌진탕 데이터에 대한 깁스 샘플러 구축(Constructing a Gibbs sampler for NFL concussion data)**: [섹션 2.1](./10-chapter2.md#sec2_1)에서 우리는 NFL 뇌진탕 데이터를 분석했습니다. 데이터는 2012년부터 2015년까지 발생한 뇌진탕 수인 Y1\=171, Y2\=152, Y3\=123, Y4\=199 로 구성됩니다. [섹션 2.1](./10-chapter2.md#sec2_1)에서 우리는 각각의 연도를 개별적으로 분석했지만, 여기서는 다음 모델을 사용하여 모든 연도를 동시에 분석합니다.

Yi|λi∼indepPoisson(Nλi) 
λi|γ∼indepGamma(1,γ) 
γ∼Gamma(a,b)(3.12)

여기서 N\=256 은 시즌당 경기 수입니다. 이 모델은 다섯 개의 미지 파라미터 θ\=(λ1,...,λ4,γ) 와 다음과 같은 사후 분포를 가집니다.

p(λ1,...,λ4,γ|Y)∝\[∏i\=14f(Yi|λi)π(λi|γ)\]π(γ),(3.13)

81\. 여기서 _f_ 는 포아송(Poisson) 확률 질량 함수(PMF)이고 _π_ 는 감마 확률 밀도 함수(PDF)입니다.

깁스 샘플러는 다섯 개 파라미터 각각의 완전 조건부 사후 분포를 요구합니다. 먼저 다른 모든 파라미터가 주어졌을 때 _λ_1 의 완전 조건부 분포를 계산합니다. 이 계산을 위해 우리는 _λ_1 에 의존하는 항들만 고려하면 되며, 나머지 모든 항은 정규화 상수로 흡수될 수 있습니다. _λ_1 에 대해 이는 다음을 도출합니다.

p(λ1,|λ2,λ3,λ4,γ,Y)∝f(Y1|λ1)π(λ1|γ).(3.14)

이것은 정확히 [섹션 2.1](./10-chapter2.md#sec2_1)에서 연구했던 포아송-감마 켤레 쌍의 사후 분포 형태이므로, _λ_1 의 완전 조건부 분포는 Gamma(Y1+1,N+γ) 가 됩니다. 다른 _λ_ _j_ 의 완전 조건부 분포도 유사합니다.

우도 f(Yi|λi) 내의 항들은 _γ_ 에 의존하지 않으므로 완전 조건부 분포는 다음과 같습니다.

p(γ|λ1,λ2,λ3,λ4,Y)∝\[∏i\=14π(λi|γ)\]π(γ).(3.15)

Gamma(1,γ) 사전 분포는 π(λi|γ)\=γexp(−γλi) 입니다. 따라서 완전 조건부 분포는 다음과 같이 단순화됩니다.

p(γ|λ1,λ2,λ3,λ4, Y)∝\[∏i\=14γexp(−γλi)\]γa−1exp(−γb)∝γ4+a−1exp\[−γ(∑i\=14λi+b)\],

그에 따라 _γ_ 에 대한 업데이트는 Gamma(4+a,∑i\=14λi+b) 입니다. 여기서 이 업데이트는 데이터 **Y** 에 의존하지 않는다는 점에 주목해야 합니다. 하지만 데이터는 _γ_ 에 간접적인 정보를 제공하는데, 왜냐하면 데이터가 _λ_ _j_ 의 사후 분포를 대부분 결정하고, 그것이 다시 _γ_ 의 사후 분포에 영향을 미치기 때문입니다.

리스트 3.3 NFL 뇌진탕 데이터에 대한 깁스 샘플링. [본문으로 돌아가기.⏎](chapter3)

```R
# 데이터 로드(Load data)

 Y <- c(171, 152, 123, 199)
 n <- 4
 N <- 256

# MCMC 표본을 위한 빈 행렬 생성(Create an empty matrix for the MCMC samples)

  S                 <- 25000
  samples           <- matrix(NA,S,5)
  colnames(samples) <- c("lam1","lam2","lam3","lam4","gamma")

# 초기값(Initial values)

 lambda <- log(Y/N)
  gamma <- 1/mean(lambda)

# 사전 분포(priors): lambda[i]|gamma ~ Gamma(1,gamma), gamma ~ InvG(a,b)

 a <- 0.1
 b <- 0.1

# 깁스 샘플링(Gibbs sampling)

 for(s in 1:S){
   for(i in 1:n){
      lambda[i] <- rgamma(1,Y[i]+1,N+gamma)
   }
   gamma       <- rgamma(1,4+a,sum(lambda)+b)
   samples[s,] <- c(lambda,gamma)
 }

 boxplot(samples[,1:4],outline=FALSE,
         ylab=expression(lambda),names=2012:2015)
 plot(samples[,5],type="l",xlab="Iteration",
      ylab=expression(gamma))

 # 사후 평균, 중앙값 및 신뢰 구간(Posterior mean, median and credible interval)
 apply(samples,2,mean)
 apply(samples,2,quantile,c(0.025,0.500,0.975))
```

[리스트 3.3](./11-chapter3.md#list3_3)은 [그림 3.9](./11-chapter3.md#fig3_9)의 결과를 생성하는 코드를 제공합니다. _γ_ ([그림 3.9](./11-chapter3.md#fig3_9)의 왼쪽 패널)에 대한 수렴은 (그림에는 표시되지 않았지만 _λ_ _i_ 와 마찬가지로) 즉각적이므로, 우리는 번인 기간이 필요 없으며 모든 표본을 사후 분포를 요약하는 데 사용합니다. 모든 연도를 고려한 이 분석의 사후 분포([그림 3.9](./11-chapter3.md#fig3_9)의 오른쪽 패널)는 사실 [그림 2.2](./10-chapter2.md#fig2_2)의 연도별 개별 분석과 꽤 유사합니다.

![Two panels summarise model output. The left panel shows boxplots of lambda for years 2012 to 2015, with medians decreasing from 2012 to 2014 and increasing again in 2015; each box has whiskers indicating spread. The right panel is a trace plot for gamma across 25000 iterations, forming a dense band of values mostly between about 1 and 5 with occasional higher spikes, showing persistent variability throughout the run.](./images/fig3_9.jpg)

그림 3.9 **NFL 뇌진탕 데이터에 대한 MCMC 분석(MCMC analysis of the NFL concussion data).** 왼쪽 패널은 각 연도별 뇌진탕 발생률 _λ_ _i_ 의 사후 분포를 나타내고, 오른쪽 패널은 초매개변수(hyperparameter) _γ_ 의 트레이스 플롯입니다. [본문으로 돌아가기.⏎](chapter3)

**티라노사우루스 렉스(T. rex) 성장 차트에 대한 블록 깁스 샘플러 구축(Constructing a blocked Gibbs sampler for a T. rex growth chart)**: \[[48](./19-ref01.md#refbib48)\]은 티라노사우루스 렉스(T. rex)를 포함한 여러 티라노사우루스과 공룡 종의 성장 차트를 연구합니다. [그림 3.10](./11-chapter3.md#fig3_10)(왼쪽 상단)에 플롯팅된 것처럼, n\=6 개인 T. rex 관측치의 몸무게(kg)는 29.9, 1761, 1807, 2984, 3230, 5040, 5654이며, 이에 대응하는 나이(세)는 각각 2, 15, 14, 16, 18, 22, 28입니다. 성장 곡선 데이터의 경우 비선형 모델이 더 적절할 가능성이 높지만([섹션 7.2.2](./15-chapter7.md#sec7_2_2)), 여기서는 설명을 위해 선형 모델을 적합시킵니다. 우리는 Yi\=β1+xiβ2+εi 로 가정하며, 여기서 _Y_ _i_ 는 공룡 _i_ 의 몸무게, _x_ _i_ 는 나이, _β_1 과 _β_2 는 회귀 절편 및 기울기이며 εi∼iid Normal(0,σ2) 입니다. τ\=10,0002 이고 a\=b\=0.1 인 무정보적(uninformative) 사전 분포 βj∼indepNormal(0,τ) 및 σ2∼InvGamma(a,b) 를 선택합니다. [그림 3.10](./11-chapter3.md#fig3_10)(오른쪽 상단)은 세 개의 파라미터 θ\=(β1,β2,σ) 에 대한 사후 분포를 플롯팅합니다. 오차 표준 편차 _σ_ 는 다른 파라미터에 강하게 의존하지 않지만, 회귀 계수 β\=(β1,β2) 는 Cor(β1,β2)\=−0.91 의 상관관계를 가집니다.

![A scatterplot shows body mass increasing with age alongside a fitted line. A second panel matrix shows posterior samples for Beta one Beta two and Sigma with strong negative correlation between the Betas and wide spread for Sigma. Two trace plots display Beta one and Beta two over 10000 iterations forming dense stable bands with occasional spikes, indicating mixing around consistent central values.](./images/fig3_10.jpg)

그림 3.10 **티라노사우루스 렉스(T. rex) 성장 차트 분석(Analysis of the T. rex growth chart).** 데이터는 왼쪽 상단 패널에 플롯팅되어 있으며 세 모델 파라미터의 결합 사후 분포의 표본들은 오른쪽 상단에 플롯팅되어 있습니다. 두 번째 행은 _β_1 과 _β_2 의 트레이스 플롯을 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

![A shaded joint density for Beta one and Beta two forms a diagonal band sloping downward from left to right. Over this surface a sequence of labelled points Beta zero through Beta three traces a stepwise path toward the high density region. Each step is shown with short arrows indicating successive moves in the parameter space, ending near the central darker area along the diagonal ridge.](./images/fig3_11.jpg)

그림 3.11 **상관관계가 있는 파라미터를 위한 깁스 샘플링(Gibbs sampling for correlated parameters).** 배경색은 β\=(β1,β2) 에 대한 가상의 이변량 사후 분포 확률 밀도 함수(PDF)를 나타내며(검은색은 높음, 흰색은 낮음을 의미), 점들은 초기값 β(0) 와 다섯 번의 가상 깁스 샘플링 업데이트인 β(1),...,β(5) 를 나타냅니다. [본문으로 돌아가기.⏎](chapter3)

깁스 샘플링은 파라미터들이 강한 사후 의존성을 가질 때 어려움을 겪습니다. 이를 확인하기 위해 [그림 3.11](./11-chapter3.md#fig3_11)에 표시된 가상의 예제를 생각해 보겠습니다. 두 파라미터 β\=(β1,β2) 는 평균이 0, 분산이 1, 상관관계가 -0.98인 이변량 정규 사후 분포를 따릅니다. 초기값은 β(0)\=(0,−3) 입니다. 첫 번째 단계는 β2\=−3 을 조건으로 _β_1 을 업데이트하는 것입니다. 이는 [그림 3.11](./11-chapter3.md#fig3_11)의 맨 아랫줄을 따라 표본을 추출하는 것에 해당하므로 β1≈3 이라는 샘플을 얻게 됩니다. 다음 업데이트는 β1≈3 을 조건으로 _β_2 의 조건부 분포에서 추출하는 것입니다. 두 변수는 음의 상관관계가 있으므로 β1≈3 일 때 82\. 83\. 필연적으로 β2≈−3 이 되어야 하며 따라서 _β_2 는 초기값에서 아주 미미하게만 변경됩니다. 이런 미세한 업데이트는 다음 네 번의 반복에서도 계속되며 β 는 우측 하단 사분면에 머뭅니다. 이 장난감(toy) 예제는 파라미터 간 강한 사후 의존성이 있을 때 한 번에 하나씩 추출하는(one-at-a-time) 깁스 샘플러가 파라미터 공간을 천천히 가로지르게 되어 결과적으로 열악한 수렴을 초래한다는 것을 보여줍니다.

수렴을 개선하는 한 가지 방법은 종속된 파라미터들을 묶어서(블록 단위로) 업데이트하는 것입니다. T. rex 예제로 돌아가서, _β_1 과 _β_2 는 서로 상관관계가 있지만 _σ_2 와는 그렇지 않으므로([그림 3.10](./11-chapter3.md#fig3_10)), 블록 θ1\=β 와 θ2\=σ2 를 설정하고 θ1|θ2,Y 과 θ2|θ1,Y 사이를 오가며 [알고리즘 1](./11-chapter3.md#algo3_1)을 적용할 수 있습니다. β 가 그 결합 사후 분포로부터 추출되기 때문에 하나씩 추출할 때처럼 모서리에 갇히지 않게 되어 수렴이 개선됩니다.

블록들의 완전 조건부 분포와 MCMC 코드를 도출하기 위해 우리는 행렬 표기법을 사용합니다. 데이터 벡터를 Y\=(Y1,...,Yn)T 로, i 번째 행이 (1,xi) 와 동일한 n×2 공변량 행렬을 **X** 로 둡니다. 선형 회귀 모델은 Y|β,σ2∼Normal(X β,σ2In) 로 작성할 수 있습니다. 완전 조건부 분포는 다음과 같습니다([부록 A.3](./18-appA.md#secA_3)에서 파생되었고 [섹션 2.1.6](./10-chapter2.md#sec2_1_6)에서 논의됨).

β|σ2,Y∼Normal(P−1W,P−1)
σ2|β,Y∼InvGamma(n/2+a,(Y−Xβ)T(Y−Xβ)/2+b)

여기서 P\=XTX/σ2+I/τ 이고 W\=XTY/σ2 입니다. 이 모델을 구현하는 코드는 [리스트 3.4](./11-chapter3.md#list3_4)에 제공되어 있습니다. [그림 3.10](./11-chapter3.md#fig3_10)(하단 행)은 훌륭한 수렴을 보여줍니다. 공변량이 두 개 이상이라면 **X** 에는 추가 열이 생기고 β 에는 추가 요소가 생기겠지만, 완전 조건부 분포와 깁스 샘플링 단계는 변하지 않을 것입니다. 84\. (이 페이지에 그림 존재) 85\.

리스트 3.4 T-rex 데이터에 적용된 선형 회귀에 대한 깁스 샘플링. [본문으로 돌아가기.⏎](chapter3)

```R

 library(mvtnorm)

# T-Rex 데이터 로드(Load T-Rex data)

  mass <- c(29.9, 1761, 1807, 2984, 3230, 5040, 5654)
  age  <- c(2, 15, 14, 16, 18, 22, 28)
  n  <- length(age)
  X  <- cbind(1,age)
  Y <- mass

# MCMC 표본을 위한 빈 행렬 생성(Create an empty matrix for the MCMC samples)

  S         <- 10000
  samples   <- matrix(NA,S,3)
  colnames(samples) <- c("Beta1","Beta2","Sigma")

# 초기값(Initial values)

  beta <- lm(mass~age)$coef
  sig2 <- var(lm(mass~age)$residuals)

# 사전 분포(priors): beta ~ N(0,tau I 2), sigma^2 ~ InvG(a,b)

  tau <- 10000^2
  a   <- 0.1
  b   <- 0.1

# 블록 깁스 샘플링(Blocked Gibbs sampling)

    V   <- diag(2)/tau
  tXX   <- t(X)%*%X
  tXY   <- t(X)%*%Y

  for(s in 1:S){
    P    <- tXX/sig2 + V
    W    <- tXY/sig2
    beta <- rmvnorm(1,solve(P)%*%W,solve(P))
    beta <- as.vector(beta)

      A    <- n/2 + a
      B    <- sum((Y-X%*%beta)^2)/2 + b
      sig2 <- 1/rgamma(1,A,B)

      samples[s,]<-c(beta,sqrt(sig2))
  }

 pairs(samples)
```

#### 3.2.1.2 86\. 메트로폴리스-해이스팅스 샘플링 (Metropolis–Hastings (MH) sampling)

깁스 샘플링의 각 단계에서는 다른 모든 파라미터를 조건으로 한 채, 한 파라미터(또는 파라미터들의 블록)의 완전 조건부 분포로부터 샘플을 취해야 합니다. 앞 절의 예제에서 모든 사전 분포들은 조건부 켤레(conditionally conjugate) 성질을 지녔으므로 완전 조건부 분포들이 우리에게 익숙한 확률분포족의 일원임을 보일 수 있었고 그에 따라 샘플링은 간단했습니다. 하지만 항상 조건부 켤레 사전 분포를 지정할 수 있는 것은 아닙니다.

예를 들어, NFL 뇌진탕 데이터([그림 3.12](./11-chapter3.md#fig3_12))로 돌아가서 다음과 같은 모델을 선택한다고 해봅시다.

Yi|β∼Poisson\[Nexp(β1+β2i)\](./3.16)

![Boxplots of fitted values for years 2012 to 2015 show increasing medians with occasional outliers. A short trace for Beta two over 200 iterations rises in steps toward about 0.1. Long trace plots for Beta one and Beta two over 25000 iterations form dense fluctuating bands around stable central values, indicating extensive mixing with no visible drift.](./images/fig3_12.jpg)

그림 3.12 **NFL 뇌진탕 데이터에 대한 포아송 회귀(Poisson regression for the NFL concussion data).** 첫 번째 패널은 각 연도에 대한 평균값 Nexp(β1+β2i) 의 사후 분포(박스플롯)와 관측된 뇌진탕 발생 횟수 _Y_ _i_ (점)를 플롯팅합니다. 나머지 세 패널은 회귀 계수 _β_1 과 _β_2 의 트레이스 플롯입니다. [본문으로 돌아가기.⏎](chapter3)

연도 i\=1,...,4 에 대해 로그 뇌진탕 발생률이 시간에 따라 선형적으로 변하도록 설정합니다. 그러면 우도(likelihood)는 다음과 같습니다.

f(Y|β)∝∏i\=14exp\[−Nexp(β1+β2i)\]exp\[Yi(β1+β2i)\].(3.17)


파라미터들은 지수 함수 내의 또 다른 지수 함수 안쪽에 나타나며, 이 특징을 갖는 _β_ 에 대해 널리 알려진 분포 족은 없습니다. 따라서 사후 분포는 알려진 분포에 속하지 않을 것이며, 어떻게 사후 분포에서 직접 표본을 추출할 수 있을지 명확하지 않습니다.

MH 샘플링 \[[74](./19-ref01.md#refbib74)\]은 정확한 완전 조건부 분포로부터의 표본 추출을, 제안/후보 분포(proposal/candidate distribution)로부터의 추출에 이은 수락/기각(accept/reject) 단계로 대체합니다. 즉, _β_1 의 깁스 업데이트는 단순한 표본 추출인 β1(s)|β2(s−1),Y 인 반면, MH는 _β_1 의 현재 값(그리고 잠재적으로 _β_2 및/또는 **Y** 에)을 조건으로 하는 후보 추출물 β1∗∼q(β1|β1(s−1)) 을 생성합니다. 당연하게도, 후보 분포는 87\. (이 페이지에 그림 존재) 88\. 사후 분포와 관련이 없을 수 있으므로 후보를 무조건 수락할 수는 없습니다. 이를 보정하기 위해 후보는 비율 _R_ 에 대해 min{1,R} 의 확률로 수락됩니다. 여기서 _R_ 은 다음과 같습니다.

R\=p(β1∗|β2,Y)p(β1(s−1)|β2,Y)q(β1(s−1)|β1∗)q(β1∗|β1(s−1)).(3.18)

동일하게, 수락/기각 단계는 U∼Uniform(0,1) 를 생성하고 U<R 이면 후보를 수락합니다. [리스트 2](./11-chapter3.md#list3_2)는 MH 샘플러를 공식적으로 설명하며, 이는 깁스 샘플링을 위해 [부록 A.3](./18-appA.md#secA_3)에 요약된 단계를 따름으로써 정당화될 수 있습니다.

알고리즘 2 메트로폴리스-해이스팅스 샘플링(Metropolis–Hastings Sampling) [본문으로 돌아가기.⏎](chapter3)

```
1:  초기화(Initialize) θ(0)=(θ1(0),...,θp(0))
2:  for s=1,...,S do
3:       for j=1,...,p do
4:                sample θj∗∼qj(θj|θj(s−1))
5:                set θ∗=(θ1(s),...,θj−1(s),θj∗,θj+1(s−1),...,θp(s−1))
6:                set R=f(Y|θ∗)π(θ∗)f(Y|θ(s−1))π(θ(s−1))⋅qj(θj(s−1)|θj∗)qj(θj∗|θj(s−1))
7:                sample U∼Uniform(0,1)
8:                if U<R then
9:                        set θj(s)=θj∗
10:               else 
11:                       set θj(s)=θj(s−1)
12:               end if
13:          end for
14:    end for
```

수락 비율 _R_ 은 후보 값과 현재 값에 대한 사후 분포의 비율에 의존합니다. 파라미터 _θ_ _j_ 를 업데이트할 때, _θ_ _j_ 를 포함하지 않는 우도나 사전 분포의 항들은 상쇄되므로 무시할 수 있습니다. 결정적으로 여기에는 종종 다루기 힘든 정규화 상수 m(Y) 가 포함됩니다. 다른 항들도 상쇄될 수 있습니다. 예를 들어, [리스트 3.3](./11-chapter3.md#list3_3)의 예제에서 전체 우도는 한 파라미터(γ)에 의존하지 않았으므로, 해당 파라미터에 대한 사후 비율에서 상쇄될 것입니다.

MH 샘플러는 깁스 샘플링보다 더 일반적으로 적용될 수 있지만, 각 파라미터(또는 파라미터 블록)에 대해 후보 분포를 선택하고 미세 조정(tune)해야 하는 대가가 따릅니다. 공통적인 선택은 무작위 보행(random-walk) 가우스 후보 분포입니다.

θj∗|θj(s−1)∼Normal(θj(s−1),cj2)(3.19)

여기서 _c_ _j_ 는 후보 표준 편차입니다. 이것은 단순히 체인의 현재 상태에 가우스 지터(Gaussian jitter)를 더하는 것이기 때문에 무작위 보행 제안 분포(random-walk proposal distribution)라고 불립니다. 가우스 후보 분포는 가우스 사전 분포가 없는 연속 파라미터라 하더라도 모든 연속 파라미터에 사용될 수 있지만 항상 이상적인 것은 아닙니다. 예를 들어 분산 파라미터에 대한 가우스 후보는 음수 값을 제안할 수 있는데, 이는 사전 분포의 확률 밀도와 그에 따른 수락 확률이 0이 될 것이므로 자동으로 폐기됩니다. 대신, 경계가 있는 지지 집합(예: σ\>0)을 가진 파라미터는 실수 전체를 지지 집합으로 가지도록 변환할 수 있으며(θ\=log(σ)), 이 변환된 공간에서 가우스를 사용할 수 있습니다. 그러나 이산 파라미터의 경우 가우스 분포에서 추출한 값이 이산 파라미터 사전 분포의 지지 집합에 포함되지 않을 확률이 거의 확실하므로, 가우스 후보 분포는 적절하지 않습니다.

89\. 후보 분포의 표준 편차 _c_ _j_ 는 튜닝 파라미터입니다. 경험 법칙(rule of thumb)에 따르면, 알고리즘이 후보의 30~50%를 수락하도록 조정하는 것이 좋습니다. 샘플링을 시작하기 전에 어떤 _c_ _j_ 값이 이 수락률로 이어질지 알기는 어렵기 때문에, 번인(burn-in) 기간 동안 표본의 배치를 기반으로 _c_ _j_ 를 조정하는 것이 일반적입니다. 예를 들어 최근 100명의 후보 중 30% 미만이 수락되었다면 _c_ _j_ 를 20% 감소시키고, 최근 100명의 후보 중 50% 초과가 수락되었다면 _c_ _j_ 를 20% 증가시킬 수 있습니다. 단, 고급 기법([부록 A.4](./18-appA.md#secA_4))을 고려하지 않는 이상 일단 번인이 완료되면 후보 분포를 고정해야 합니다.

무작위 보행 후보 분포의 두 가지 이점은 사후 분포의 형태에 대한 지식을 요구하지 않는다는 것과 수락 비율이 단순화된다는 것입니다. 만약 후보 확률 밀도 함수(PDF) _q_ 가 가우스 분포라면,

q(θj(s−1)|θj∗)\=qj(θj∗|θj(s−1))\=12πcjexp(−(θj(s−1)−θj∗)22cj2),(3.20)

MH 수락 확률에서 후보 분포의 비율이 상쇄됩니다. 이를 대칭 후보 분포(symmetric candidate distribution)라고 부릅니다. 만약 후보 분포가 대칭적이라면 MH 알고리즘은 [알고리즘 3](./11-chapter3.md#algo3_3)의 메트로폴리스 알고리즘 \[[104](./19-ref01.md#refbib104)\]으로 귀결됩니다.

알고리즘 3 메트로폴리스 샘플링(Metropolis Sampling) [본문으로 돌아가기.⏎](chapter3)

```
1: 초기화(Initialize) θ(0)=(θ1(0),...,θp(0))
2: for s=1,...,S do
3:        for j=1,...,p do
4:        대칭인 qj 에 대해 sample θj∗∼qj(θj|θj(s−1)) 
5:        set θ∗=(θ1(s),...,θj−1(s),θj∗,θj+1(s−1),...,θp(s−1))
6:        set R=f(Y|θ∗)π(θ∗)f(Y|θ(s−1))π(θ(s−1))
7:        sample U∼Uniform(0,1)
8:        if U<R then
9:                set θj(s)=θj∗
10:       else
11:               set θj(s)=θj(s−1)
12:       end if
13:    end for
14: end for
```

[리스트 3.5](./11-chapter3.md#list3_5)는 NFL 뇌진탕의 포아송 회귀 모델에 대한 R 코드를 제공합니다. 가우스 후보 분포의 표준 편차는 0.1로 설정되었습니다. 파라미터들이 평균 발생 건수의 지수 함수 내에 나타나므로 표준 편차 0.1의 정규 분포 지터를 추가하는 것은 평균 건수가 10% 변하는 것과 유사하며, 이는 β1 의 경우 0.42, β2 의 경우 0.18의 수락 확률로 이어집니다. 추가 튜닝을 통해 수락 확률을 높이기 위해 β2 의 후보 표준 편차를 줄일 수 있겠으나, [그림 3.12](./11-chapter3.md#fig3_12)의 트레이스 플롯은 양호한 수렴을 보여줍니다. [그림 3.12](./11-chapter3.md#fig3_12)의 우측 상단 패널은 메트로폴리스 알고리즘의 첫 몇 개의 표본들을 확대해서 보여주며, 후보가 수락될 때 β2 가 새로운 값으로 점프하기 전까지 여러 차례의 반복 동안 어떻게 일정한 값을 유지하는지 보여줍니다.

리스트 3.5 NFL 뇌진탕 데이터에 대한 메트로폴리스 샘플링. [본문으로 돌아가기.⏎](chapter3)

```R
# 데이터 로드(Load data)

 Y <- c(171, 152, 123, 199)
 t <- 1:4
 n <- 4
 N <- 256

# MCMC 표본을 위한 빈 행렬 생성(Create an empty matrix for the MCMC samples)

  S                  <- 25000
  samples            <- matrix(NA,S,2)
  colnames(samples)  <- c("beta1","beta2")
  fitted             <- matrix(NA,S,4)

# 초기값(Initial values)

  beta <- c(log(mean(Y/N)),0)

# 사전 분포(priors): beta[j] ~ N(0,tau^2)

  tau <- 10

# 메트로폴리스 샘플링 준비(Prep for Metropolis sampling)

  log_post  <- function(Y,N,t,beta,tau){
     mn     <- N*exp(beta[1]+beta[2]*t)
     like   <- sum(dpois(Y,mn,log=TRUE))
     prior  <- sum(dnorm(beta,0,tau,log=TRUE))
     post   <- like + prior
  return(post)}

  can_sd <- rep(0.1,2)

# 메트로폴리스 샘플링(Metropolis sampling)

  for(s in 1:S){
    for(j in 1:2){
       can              <- beta
       can[j]           <- rnorm(1,beta[j],can_sd[j])
       logR             <- log_post(Y,N,t,can,tau)-log_post(Y,N,t,beta,tau)
       if(log(runif(1))<logR){
          beta          <- can
       }
    }
    samples[s,] <- beta
    fitted[s,] <- N*exp(beta[1]+beta[2]*t)
  }

  boxplot(fitted,outline=FALSE,ylim=range(Y),
          xlab="Year",ylab="Fitted values",names=2012:2015)
  points(Y,pch=19)
```

[리스트 3.5](./11-chapter3.md#list3_5)의 라인 40-41에 있는 수락/기각(accept/reject) 단계의 경우, 계산을 로그(log) 척도로 수행하는 것이 중요합니다. 대규모 데이터 세트의 경우 분모의 아주 작은 값이 비율을 숫자로 표현 불가능한 무한대로 만들 수 있기 때문에 사후 분포의 비율은 수치적으로 불안정합니다. 따라서 비율을 로그 척도의 차이로 대체합니다. 원래의 90\. 91\. 기각 단계는 U<f(θ∗|Y)/f(θ(s−1)|Y) 인 경우에 기각하는 것이었습니다. 양변에 로그를 취하면 동치인 부등식 log(U)<log\[f(θ∗|Y)\]−log\[f(θ(s−1)|Y)\] 을 얻게 됩니다.

무작위 보행 가우스 분포는 블록 업데이트에도 사용할 수 있습니다. 깁스 샘플링에서와 마찬가지로 종속된 파라미터들을 동시에 업데이트하면 수렴이 개선될 수 있습니다. _p_ 개의 파라미터가 _q_ 개의 블록 θ\=(θ1,...,θq) 으로 파티셔닝(partitioned)되었다고 가정하고, 블록 _j_ 의 후보 분포를 θj∗∼Normal(θ(s−1),cjVj) 라 하겠습니다. 이전처럼 스칼라 _c_ _j_ 는 합리적인 수락 확률을 제공하도록 튜닝되어야 합니다. 행렬 Vj 도 튜닝해야 하지만, 이 행렬은 각 파라미터의 분산 및 모든 파라미터 쌍 간의 상관관계를 포함하고 있으며 이 모든 요소가 단일한 수락 확률에 기여하기 때문에 튜닝하기 어렵습니다. 한 가지 합리적인 선택은 Vj 를 θj 의 사후 공분산으로 설정하는 것이지만 안타깝게도 이는 알 수 없습니다. 일반적인 해결책은 짧은 번인(burn-in) 기간을 두고 Vj 를 번인 표본의 표본 공분산으로 설정한 후, 합리적인 수락을 얻을 수 있도록 스칼라 _c_ _j_ 를 조정하는 것입니다. 사전 MCMC 표본을 기반으로 제안 분포를 적응시킬 수도 있습니다([부록 A.4](./18-appA.md#secA_4)).

무작위 보행 후보 분포의 단점은 사후 분포와 밀접하게 근사하지 못할 경우 최적이 아닐 수 있다는 것입니다. 추가 정보를 사용하면 수렴을 개선할 수 있습니다. 예를 들어 베이지안 중심 극한 정리(CLT, [섹션 3.1.3](./11-chapter3.md#sec3_1_3)) 근사인 θ|Y≈Normal(θ^MAP,Σ^MAP) 을 사용해 샘플러가 후보 분포를 사후 분포에 맞추도록 튜닝할 수 있습니다. 이를 극한까지 몰고 가, 만일 우리가 완전 조건부 분포가 되도록 후보를 튜닝할 수 있다면 후보 분포는 사후 분포에 비례하게 되며(q(θj∗|θj(s−1))∝f(Y|θ∗)π(θ∗)), [알고리즘 2](./11-chapter3.md#algo3_2)의 6번째 줄에 있는 비율은 1로 약분됩니다. 이 알고리즘은 파라미터들을 순환하면서 각각 완전 조건부 분포에서 샘플링하고 모든 추출값을 보존합니다. 이것이 바로 [알고리즘 1](./11-chapter3.md#algo3_1)의 깁스 샘플링 알고리즘입니다! 이는 깁스 샘플링이 신중하게 선택된 후보 분포를 지닌 MH의 특수한 경우임을 보여줍니다. 또한 깁스 단계로 이어지는 잘 조율된 일부 후보 분포를 가지는 MH 샘플러 형태로 짜여질 수 있기 때문에 동일한 알고리즘에서 깁스와 MH 업데이트를 혼합할 수 있는 가능성도 열어줍니다.

**깁스 내의 메트로폴리스 예시(Metropolis-within-Gibbs example)**: 이 깁스 내의 메트로폴리스(Metropolis-within-Gibbs) 샘플링 알고리즘을 설명하기 위해 우리는 이진 데이터(binary data)를 위한 선형 회귀의 확장인 로지스틱 회귀 분석(logistic regression)을 사용합니다. 이진 데이터의 경우 E(Yi)\=Prob(Yi\=1) 인데, 공변량의 선형 결합(linear combination)인 Xi\=(1,Xi2...,Xip) 은 0과 1 사이에 속하지 않을 수 있으므로, 공변량의 선형 결합으로 평균을 직접 모델링할 수는 없습니다. 그에 따라 선형 회귀 모델은 Yi|β∼Bernoulli(pi) 로 수정되며, 여기서 ηi\=β1+∑j\=2pβjXij 일 때 pi\=1/\[1+exp(−ηi)\] 로 정의되어 평균 반응 값이 0과 1 사이에 있도록 보장합니다. 절편 β1∼Normal(0,102) 에 대해서는 무정보적 사전 분포를 선택하고 기울기 j\=2,...,p 에 대한 사전 분포는 다음과 같습니다.

βj|σ2∼Normal(0,σ2) 및 σ2∼InvGamma(a,b).(3.21)

회귀 계수 _β_ _j_ 는 비선형 함수 내부 우도에 나타나므로 정규 사전 분포는 조건부 켤레가 아닙니다. 반면, 사전 분산 _σ_2 는 조건부 켤레인 역-감마(inverse gamma) 사전 분포를 가지므로 역-감마 완전 조건부 분포를 갖습니다. 따라서 깁스 내의 메트로폴리스 알고리즘은 _β_ _j_ 들을 순회하며 가우스 후보 분포를 지닌 메트로폴리스 단계를 사용해 이들을 업데이트하고, _σ_2 를 그것의 역-감마 완전 조건부 분포에서 샘플링하기 위해 깁스 단계를 적용합니다.

이 모델은 p+1 개의 파라미터를 가지며, 알고리즘에는 다양한 유형의 업데이트와 _p_ 개의 튜닝 파라미터가 있습니다. 복잡성 때문에, 실제 데이터 분석에 사용하기 전에 충분히 검토해야 합니다. 코드를 검증하는 일반적인 방법은 시뮬레이션 연구(simulation study)입니다. 시뮬레이션 연구에서는 파라미터의 값을 고정하고 이 값들을 사용해 데이터를 생성합니다. 그런 다음 합성 데이터(synthetic data)를 분석하고, 도출된 사후 분포를 원래의 실제 값과 비교합니다. 실제 데이터 분석과 달리, 실제 값을 알고 있으므로 알고리즘이 해당 실제 값을 잘 복원할 수 있는지 검증할 수 있습니다.

92\. [리스트 3.6](./11-chapter3.md#list3_6)은 데이터를 시뮬레이션하고, 깁스 내의 메트로폴리스 알고리즘을 실행하며, 결과를 요약하는 코드를 제공합니다. 결과는 [그림 3.13](./11-chapter3.md#fig3_13)에 플롯팅되어 있습니다. 데이터를 생성하는 데 사용된 _β_ _j_ 의 실제 값은 예상대로 하나 혹은 두 개의 계수를 제외한 모든 계수에 대해 사후 분포 내에 포함됩니다. 따라서 알고리즘이 잘 작동하는 것으로 보입니다.

리스트 3.6 시뮬레이션된 로지스틱 회귀 데이터를 위한 깁스 내의 메트로폴리스 샘플링. [본문으로 돌아가기.⏎](chapter3)

```R
# 데이터 시뮬레이션(Simulate data)
  n           <- 100
  p           <- 20
  X           <- cbind(1,matrix(rnorm(n*(p-1)),n,p-1))
  beta_true   <- rnorm(p,0,.5)
  prob        <- 1/(1+exp(-X%*%beta_true))
  Y           <- rbinom(n,1,prob)

# 로그 사후 분포를 계산하는 함수(Function to compute the log posterior)
  log_post_beta <- function(Y,X,beta,sigma){
      prob  <- 1/(1+exp(-X%*%beta))
      like  <- sum(dbinom(Y,1,prob,log=TRUE))
      prior <- dnorm(beta[1],0,10,log=TRUE) + # 절편(Intercept)
                sum(dnorm(beta[-1],0,sigma,log=TRUE)) # 기울기(Slopes)
  return(like+prior)}

# MCMC 표본을 위한 빈 행렬 생성(Create empty matrix for the MCMC samples)
  S             <- 10000
  samples       <- matrix(NA,S,p+1)

# 초기값과 사전 분포(Initial values and priors)
  beta   <- rep(0,p)
  sigma  <- 1
  a      <- 0.1
  b      <- 0.1
  can_sd  <- 0.1

  for(s in 1:S){

    # beta에 대한 메트로폴리스(Metropolis for beta)
    for(j in 1:p){
      can <- beta
      can[j] <- rnorm(1,beta[j],can_sd)
      logR <- log_post_beta(Y,X,can,sigma)-
                  log_post_beta(Y,X,beta,sigma)
      if(log(runif(1))<logR){
         beta <- can
      }
    }

    # sigma에 대한 깁스(Gibbs for sigma)
    sigma <- 1/sqrt(rgamma(1,(p-1)/2+a,sum(beta[-1]^2)/2+b))

    samples[s,] <- c(beta,sigma)

   }

   boxplot(samples[,1:p],outline=FALSE,
            xlab="Index, j",ylab=expression(beta[j]))
   points(beta_true,pch=19,cex=1.25)
```

![A series of boxplots displays beta j across indices 1 to 20. Each box shows the posterior spread with whiskers extending to lower and upper extremes. Points mark corresponding observed or estimated values. The boxes vary in height and position, with some centred above zero and others below, indicating substantial variation in beta j across indices.](./images/fig3_13.jpg)

그림 3.13 **시뮬레이션 데이터의 로지스틱 회귀 분석(Logistic regression analysis of simulated data).** 박스플롯은 _β_ _j_ 의 사후 분포이며, 점은 데이터를 시뮬레이션하는 데 사용된 실제 값입니다. [본문으로 돌아가기.⏎](chapter3)

### 3.2.2 근사 베이지안 계산 (Approximate Bayesian Computing (ABC))

메트로폴리스-해이스팅스 샘플링은 우도(likelihood) 및 사전 분포의 PDF/PMF를 계산할 수 있는 모든 통계 모델에 적용할 수 있는 엄청나게 다재다능한 알고리즘입니다. 이는 전부는 아닐지라도 대다수의 애플리케이션에 적용됩니다. 반례로써 만일 개체군 동태 모델(population-dynamics model) 같은 확률 미분 방정식(stochastic differential equation)의 해로서 데이터가 발생한다고 가정한다면 우도가 너무 복잡해서 평가할 수 없을 수 있고, 다루기 힘든 우도 문제를 초래하게 됩니다. 근사 베이지안 계산(ABC; Approximate Bayesian Computing, \[[100](./19-ref01.md#refbib100)\])은 이런 간극을 메웁니다. ABC는 우도 함수가 평가될 수 있어야 한다고 요구하지 않으며, 그저 모델에서 무작위 표본을 생성하는 기능만을 요구합니다.

MCMC와 마찬가지로 ABC는 MCMC와 완전히 동일한 방식으로 사후 분포를 근사하는 데 사용할 수 있는 _S_ 개의 몬테카를로 표본 θ(1),...,θ(S) 을 생성합니다. 다시 말해 _S_ 개의 ABC 표본 평균이 사후 평균을 근사하는 식입니다. 마르코프 체인과 달리 ABC의 _S_ 개 표본들은 독립적이고 동일하게 분포(independent and identically distributed)합니다.

우리는 ABC의 가장 기본적인 버전만을 살펴볼 것이며, ABC를 MCMC와 결합하는 것을 포함한 확장 기능에 대해서는 관심 있는 독자들에게 \[[100](./19-ref01.md#refbib100)\]의 논평(review)을 참고하시길 권합니다. [알고리즘 4](./11-chapter3.md#algo3_4)에 간략히 설명된 바와 같이, ABC는 사전 분포에서 샘플링을 진행(θ∗∼π(θ))한 후, 파라미터 θ∗ 가 주어졌을 때의 시뮬레이션 데이터 세트 Y∗ 를 생성(Y∗|θ∗∼f(Y|θ∗))하는 것으로 시작합니다. 이 시뮬레이션된 데이터 세트는 원본 데이터 세트와 동일한 크기 및 설계(design)를 가져야 합니다. 예컨대 회귀 분석에서는, 시뮬레이션 데이터 내의 관측치 _i_ 가 실제 데이터의 관측치 _i_ 에 대응되도록 관측된 공변량 값들을 사용해 데이터를 시뮬레이션해야 합니다. 그런 다음, 이 시뮬레이션된 데이터 세트를 차이 척도(discrepancy measure)를 사용해 실제 데이터 세트와 93\. 94\. 비교합니다.

d∗\=D(Y,Y∗).

만약 _d_\* 가 임계값(threshold) _δ_ 보다 작다면 θ∗ 는 사후 표본으로 유지되며, 그렇지 않으면 폐기됩니다. 이 과정은 _S_ 개의 표본이 수집될 때까지 반복됩니다. 직관적으로, 관측치와 유사한 데이터를 도출하는 파라미터가 실제 값일 가능성이 더 높다는 것이며, 이는 최대 우도 추정(maximum likelihood estimation)의 이면에 깔린 직관과 유사합니다.

차이 척도 및 임계값의 선택은 알고리즘의 정확도에 매우 중요합니다. 만일 표본 θ∗ 가 오직 Y∗\=Y 일 때만 유지된다면, 이 알고리즘은 근사 없이 실제 사후 분포로부터 _S_ 번의 추출값을 산출합니다. 그러나 반응 변수가 연속형이거나 _n_ 이 크다면 Y∗ 가 정확히 **Y** 와 동일할 것으로 기대할 수 없습니다(실제로는 Y∗ 의 충분통계량이 **Y** 의 충분통계량과 일치하기만 하면 되지만 이조차 문제를 해결해 주지는 않습니다). 근사화된 사후 분포로부터 표본을 얻기 위해, 우리는 Y∗≠Y 이지만 차이 함수 _D_ 로 측정했을 때 유사한 표본들을 유지할 수 있습니다. 한 가지 옵션은 평균 제곱 차이인 D(Y,Y∗)\=∑i\=1n(Yi−Yi∗)2/n 입니다. 하지만 _n_ 이 큰 경우 Y∗ 가 모든 _n_ 개의 관측치에서 **Y** 와 유사할 확률은 낮으며, 파라미터들을 더 겨냥한(targeted) 차이 척도가 훨씬 더 효율적입니다. 예를 들어 파라미터들이 반응 분포의 평균과 표준 편차를 제어한다면 다음과 같은 차이 척도,

D(Y,Y∗)\=(Y¯−Y¯∗)2+w(s−s∗)2

가 표본 평균 Y¯ 와 Y¯∗, 표본 표준 편차 _s_ 와 _s_\*, 그리고 두 파라미터의 상대적 척도(scale)와 중요성을 반영하여 선택된 가중치 _w_ 를 통해 해당 파라미터들을 타겟팅합니다. 일반적으로 차이 척도에는 모델의 모든 파라미터를 식별할 수 있는 충분한 정보가 포함되어야 합니다. 이는 약간의 창의성을 요하며 파라미터가 많을 경우 어려울 수 있다는 것이 이 방법론의 한계점입니다.

[리스트 3.7](./11-chapter3.md#list3_7)은 시뮬레이션 데이터에 대한 ABC의 동작 예제를 제공합니다. n\=100 개의 관측치가 ηi\=β1+β2Xi 에 대해 pi\=1/\[1+exp(−ηi)\] 인 Yi|β∼Bernoulli(pi) 를 가지는 로지스틱 회귀 모델로부터 시뮬레이션되었습니다. 사전 분포는 β1,β2∼iid Normal(0,1) 입니다. 당연히 메트로폴리스-해이스팅스 샘플링이 위와 같이 이 모델에 적용될 수도 있겠으나, ABC 과정 중에는 우도 함수와 사전 분포가 계산될 필요가 없다는 점에 주목해 보세요. 데이터를 생성하는 데 사용된 실제 값은 β\=(2,−1) 이고 Xi∼Normal(0,1) 입니다. 이 모델이 Yi\=1 일 확률을 _X_ _i_ 의 함수로 추적하기 때문에, _D_ 의 한 가지 가능성은 관측치를 _X_ _i_ 로 그룹화(binning)하고 각 그룹(bin)에서 _Y_ _i_ 의 표본 비율을 계산하는 것입니다. 구체적으로 _X_ _i_ 의 등간격 분위수로 형성된 L\=4 개의 그룹을 생성하고 이를 q1,...,qL+1 로 표기합니다. 그룹 _l_ 의 표본 비율 Y¯l 은 ql≤Xi<ql+1 인 관측치들에 대한 _Y_ _i_ 의 표본 평균입니다. 따라서 차이 척도는 그룹 확률들 사이의 평균 제곱 오차(mean squared error)입니다.

D(Y,Y∗)\=∑l\=1L(Y¯l−Y¯l∗)2/L.

[그림 3.14](./11-chapter3.md#fig3_14)(좌측)는 β∗ 에 따른 D(Y,Y∗) 의 평균을 플롯팅한 것입니다. 실제 β 값 주변의 영역에서 확실히 차이가 더 낮으며, 따라서 진실값 근처의 β∗ 추출값들이 유지될 가능성이 더 높습니다. 또한 [그림 3.14](./11-chapter3.md#fig3_14)는 임계값 _δ_ 의 함수로서의 95% 신뢰 집합(credible sets)을 플롯팅합니다. _δ_ 값이 작을수록 사후 분포에 대해 더 나은 근사를 제공하지만 유지된 표본의 비율은 최소 임계값인 δ\=0.01 에 대해 0.02로 떨어집니다. 95\. 96\.

![A heatmap shows values over Beta one and Beta two, with darker shading on the left and lighter shading on the right, and a marked point in the light region. A line plot shows Beta one and Beta two against delta from 0 to 0.1, with Beta one slightly decreasing and Beta two increasing before flattening, illustrating how both parameters vary smoothly with delta.](./images/fig3_14.jpg)

그림 3.14 **ABC 시연(Illustration of ABC).** 시뮬레이션된 로지스틱 회귀 파라미터 β∗(좌측)의 함수로서 평균 차이(mean discrepancy) _D_ 와, 임계값 _δ_ 의 함수로서 _β_1 및 _β_2 에 대한 근사적인 사후 90% 신뢰 구간. β\=(2,−1) 의 실제 값은 좌측 패널 내의 점으로 표시되어 있습니다. [본문으로 돌아가기.⏎](chapter3)

알고리즘 4 근사 베이지안 계산 (ABC, Approximate Bayesian Computing) [본문으로 돌아가기.⏎](chapter3)

```
1: 초기화(initialize) s=0
2: while s<S do
3:           사전 분포로부터 파라미터 샘플링(sample parameters from prior) θ∗∼π(θ)
4:           우도로부터 데이터 샘플링(sample data from likelihood) Y∗|θ∗∼f(y|θ∗)
5:           차이 척도 계산(compute the discrepancy measure) d∗=D(Y∗,Y)
6:           if d∗<δ then
7:                set s=s+1
8:                set θ(s)=θ∗
9:           end if
10: end while
```

리스트 3.7 로지스틱 회귀에 대한 ABC. [본문으로 돌아가기.⏎](chapter3)

```R
# 차이 척도 정의(Define the discrepancy measure)
D <- function(X,Y_obs,Y_sim,q){
  L <- length(q)-1
  d <- 0
  for(l in 1:L){
     d <- d + (mean(Y_sim[X>=q[l] & X<q[l+1]]) -
                 mean(Y_obs[X>=q[l] & X<q[l+1]]))^2
  }
d/L}

# 차이 계산(Compute discrepancy)
deviation
 S          <- 1000000 # 사전 분포에서의 추출 횟수(Number of draws from the prior)
 b_sim      <- matrix(rnorm(S*2),S,2) # 사전 분포에서 추출(Draw from prior)
```


```R
  d          <- rep(0,S)
  q          <- quantile(X,c(0.00,0.25,0.50,0.75,1.00))

  for(s in 1:S){
    set.seed(s)
    eta   <- b_sim[s,1]+b_sim[s,2]*X
    p     <- 1/(1+exp(-eta))
    Y_sim <- rbinom(n,1,p)
    d[s]  <- D(X,Y,Y_sim,q)
  }

# beta의 함수로서 d 플롯팅(Plot d as a function of beta)
  quilt.plot(b_sim[,1],b_sim[,2],d,
              xlab=expression(beta[1]),ylab=expression(beta[2]))

# delta에 의한 사후 근사 계산(Compute posterior approximation by delta)
  m <- 20
  delta <- seq(0.01,0.10,length=m)
  qb1 <- qb2 <- matrix(0,m,3)
  for(i in 1:m){
    qb1[i,] <- quantile(b_sim[d<delta[i],1],c(0.05,0.50,0.95))
    qb2[i,] <- quantile(b_sim[d<delta[i],2],c(0.05,0.50,0.95))
  }
  matplot(delta,qb1,type="l",lty=c(2,1,2),col=1,lwd=2,
           xlab=expression(delta),ylab=expression(beta[1]))
  matplot(delta,qb2,type="l",lty=c(2,1,2),col=1,lwd=2,
           xlab=expression(delta),ylab=expression(beta[2]))
```

### 3.2.3 97\. 시뮬레이션 기반 추론 (Simulation-based inference)

시뮬레이션 기반 추론(SBI; Simulation-based inference, \[[37](./19-ref01.md#refbib37)\])은 ABC와 밀접하게 관련되어 있으며 사후 분포를 평가하기는 어렵지만 샘플링이 간단한 설정에서도 유용합니다. ABC와 마찬가지로, SBI는 사전 분포에서 추출하고 그런 다음 데이터 생성 과정으로부터 추출하는 것으로 시작합니다.

θ(s)∼π(θ) 및 Y(s)|θ(s)∼f(Y|θ(s))

여기서 s∈{1,...,S} 입니다. SBI가 ABC와 다른 점은, 사후 분포를 근사하기 위해 표본 θ(s) 의 하위 집합(subset)만을 유지하는 대신 _S_ 개의 표본 전체와 기계 학습(machine learning)을 사용해 사후 분포 p(θ|Y) 에 대한 근사치를 구축한다는 점입니다.

구조적으로 (θ(s),Y(s)) 는 파라미터와 데이터의 결합 분포(joint distribution)에서 추출된 표본입니다. 그러므로 이 표본들을 사용하여 θ(s) 를 Y(s) 에 회귀시키는 통계적 모델을 구축해 사후 분포 p(θ|Y) 를 근사할 수 있습니다. p\=1 인 일변량(univariate) 분석의 경우 매우 조악한 근사치는 다음과 같은 선형 회귀 분석(linear regression)입니다.

θ(s)|Y(s)∼Normal(b0+∑i\=1nYi(s)bj,τ2).

이런 (비현실적인) 가정하에 _S_ 개의 훈련 표본을 사용해 최소 제곱(least squares)으로 b\=(b0,...,bn) 와 _τ_2 를 추정하여 추정값 b^ 와 τ^2 를 얻습니다. 그 후 실제 데이터 세트 **Y** 에 대해 사후 분포는 θ|Y∼Normal(b^0+∑i\=1nYib^j,τ^2) 로 근사됩니다. 관측된 결과치 _Y_ _i_ 가 예측 변수(predictors)로 사용되는 이 회귀 분석에서의 역할 반전(role reversal)에 유의하세요. 이는 베이지안 파라미터 추론이 일반적인 정방향 예측 문제(forward prediction problem)가 아니라 역문제(inverse problem)임을 보여줍니다.

장난감(toy) 예제로서 사전 분포 θ∼Normal(0,1) 를 가지는 모델 Y|θ∼Normal(θ,1) 에 대해 SBI를 시연하는 [그림 3.15](./11-chapter3.md#fig3_15)를 고려해 보겠습니다. [섹션 2.1](./10-chapter2.md#sec2_1)에서 유도된 바와 같이 정확한 사후 분포는 θ|Y∼Normal(Y/2,1/2) 입니다. [그림 3.15](./11-chapter3.md#fig3_15)(좌측)는 (θ,Y) 의 결합 확률 밀도에서 추출된 S\=1,000 개의 표본을 플롯팅합니다. 수학적 유도 없이 사후 분포를 학습하기 위해 이 _S_ 개의 표본에 단순 선형 회귀 모델 θ|Y∼Normal(b0+Yb1,τ2) 을 적합(fit)시킵니다. 최소 제곱 분석은 b^0\=−0.045, b^1\=0.495 및 τ^2\=0.488 을 제공합니다. [그림 3.15](./11-chapter3.md#fig3_15)(우측)는 SBI가 실제 사후 분포에 밀접한 근사치를 제공함을 보여줍니다. 이 근사치가 정확한 주된 이유는 실제 사후 분포가 가우스 분포이고 관측치들에 선형적이라는 지식에 의존하기 때문인데, 이 두 가지는 간단하지 않은(non-trivial) 문제들에 대해서는 의심스러운 가정들입니다. 따라서 실제로는 사후 분포에 대한 더욱 유연한 통계 모델링과 더 많은 수의 표본이 요구됩니다.

![A scatterplot shows Y versus theta with points forming an upward sloping cloud bounded by three horizontal dashed lines at Y equal to minus 2, 0 and 2. Next to it, three posterior curves for theta are shown for Y equal to minus 2, Y equal to 0 and Y equal to 2, each forming a smooth peak that shifts rightward as Y increases, with all curves tapering to zero toward the extremes.](./images/fig3_15.jpg)

그림 3.15 **SBI 시연(Illustration of SBI).** (θ,Y) 의 결합 분포에서의 표본들(좌측) 및 세 가지 _Y_ 값에 대한 실제(두꺼운 선) 사후 분포 및 근사(얇은 선) 사후 분포(우측). [본문으로 돌아가기.⏎](chapter3)

이 접근법이 통계학자들에게 매력적인 이유는 무제한에 가까운 표본수 _S_ 에 힘입어 고급 컴퓨팅을 고급 통계 모델링으로 대체하기 때문입니다. 이 방대한 데이터 세트와 더불어 데이터 및 파라미터 간의 복잡한 비선형적·비가우스적 관계를 전제할 경우, 사후 분포 근사 모델을 구축하기 위해 위와 같은 선형 회귀 분석 대신 기계 학습, 그 중에서도 딥러닝(8.3.4장 참조)이 일반적으로 사용됩니다. ABC나 어떠한 통계적 모델 구축 과정과 마찬가지로 근사 사후 분포에서 요약 통계량(summary statistics)이 전체 데이터 세트를 대체할 수 있으며, 표본 생성을 위해 사전 분포 외의 다른 분포들을 사용할 수도 있습니다(다만 분석 후반부에 이에 대한 보정이 반드시 이루어져야 합니다).

SBI는 분할 상환 추론(amortized inference, 계산 비용을 미리 지불하고 나중에 재사용하는 추론)의 한 예입니다. 즉, _S_ 개의 표본을 시뮬레이션하고 파라미터와 데이터를 연관시키는 모델을 구축하는 데는 상당한 노력이 들지만, 이것이 완료되고 나면 주어진 데이터 세트에 대한 추론은 즉각적입니다. 이는 초기 분석의 동기가 된 데이터 세트뿐만 아니라 모델에서 생성된 어떤 데이터 세트에도 유효하므로 동일한 근사치를 어떤 잠재적 사용자도 사용할 수 있습니다. 따라서 데이터에서 사후 분포로 매핑하는 방법을 학습함으로써 미래의 사용자는 까다로운 연산 단계를 우회하고 즉시 근사적인 추론을 얻을 수 있습니다.

### 3.2.4 98\. 순차 몬테카를로 샘플링 (Sequential Monte Carlo (SMC) sampling)

MCMC, ABC, SBI와 마찬가지로, 순차 몬테카를로(SMC, Sequential Monte Carlo) 샘플링(파티클 필터링(particle filtering)이라고도 함)은 몬테카를로 샘플링을 사용하여 사후 분포를 근사합니다. 초기 분포(흔히 사전 분포)에서 표본을 만들고 데이터에 적합한 정도에 따라 이들을 평가한다는 점에서 SMC는 ABC와 유사합니다. 그러나 SMC는 ABC와 중요한 두 가지 측면에서 차이가 있습니다. 근사 우도가 아닌 정확한 우도를 사용한다는 점과 표본을 수락 혹은 기각하는 대신 가중치를 부여한다는 점입니다. SMC가 단일한 정적(static) 데이터 세트를 분석하는 데 사용될 수도 있지만, 그 이름은 실시간 스트리밍(streaming) 데이터의 순차적(sequential) 분석에 유용하다는 사실에서 유래했습니다. 예를 들어 매일 새로운 데이터 배치가 수집되면 이 새로운 정보를 기반으로 베이지안 분석이 업데이트됩니다.

ABC와 마찬가지로 여기서도 가장 간단한 버전만 제시합니다. 목적에 맞춘 초기 파티클 샘플링 및 잘못된 파티클 교체를 포함하는 더 진보된 알고리즘에 대해서는 SMC에 대한 풍부한 문헌(예: \[[45](./19-ref01.md#refbib45)\])을 참조하시기 바랍니다. 간단한 SMC 알고리즘은 사전 분포로부터 _S_ 개의 표본/파티클(particles)을 추출하는 것으로 시작합니다.

θ(1),...,θ(S)∼iidπ.

그런 다음 우도 함수인 u(s)\=f(Y|θ(s)) 에 기반해 파티클들에 가중치를 할당하고, 다음과 같이 합이 1이 되도록 정규화합니다.

w(s)\=u(s)∑t\=1Su(t).

그 후 사후 수치(posterior quantities)는 E(θj|Y)≈∑s\=1Sw(s)θj(s) 및 다음과 같은 가중 평균(weighted averages)을 사용하여 계산됩니다.

Prob(θj\>0|Y)≈∑s\=1Sw(s)I(θj(s)\>0),

99\. 즉, θj(s)\>0 인 파티클들에 대한 가중치의 합입니다. 데이터에 잘 맞지 않는 파티클은 가중치 w(s)≈0 을 가지게 되어 근사 사후 분포에서 "걸러짐(filtered out)"에 유의하세요. 따라서 사전 분포(혹은 다른 초기 분포)가 너무 퍼져 있어(diffuse) 대부분의 파티클이 필터링된다면 이 알고리즘은 비효율적이 될 것입니다.

위 알고리즘은 전체 데이터 세트를 사용하여 사후 분포를 근사합니다. 하지만 데이터가 순차적으로 도착하고 시간에 따라 독립적이라고 가정해 봅시다. 그러면 _n_ 개의 관측치를 관찰한 후의 정규화되지 않은 가중치는 다음과 같습니다.

un(s)\=∏i\=1nf(Yi|θ(s)).

추가 관측치가 도착하면 가중치는 다음과 같이 업데이트됩니다.

un+1(s)\=f(Yn+1|θ(s))un(s)

그리고 정규화된 가중치는 wn+1(s)\=un+1(s)/∑t\=1Sun+1(t) 가 됩니다. 각 시간 단계마다 데이터를 재분석하는 것과 비교할 때, 현재 파티클의 가중치를 얻기 위해 처음 _n_ 개의 관측치로부터의 우도를 다시 계산하거나 심지어 데이터를 저장할 필요가 없으므로 시간과 저장 공간이 절약됩니다.

이것이 왜 유효한 사후 근사를 제공하는지 알아보기 위해, 지지 집합이 θ∈{θ1,...,θK} 인 이산(discrete) 파라미터를 고려해 보겠습니다. θ\=θk 일 실제 사후 확률은 다음과 같습니다.

Prob(θ\=θk|Y)\=f(Y|θk)π(θk)∑l\=1Kf(Y|θl)π(θl).

SMC 근사는 다음과 같습니다.

Prob(θ\=θk|Y)≈∑s\=1SwsI(θ(s)\=θk)\=∑s\=1Sf(Y|θ(s))I(θ(s)\=θk)∑s\=1Sf(Y|θ(s)).

_S_ 개의 항들 중 θ(s)\=θk 인 것들의 비율을 _p_ _k_ 로 표기하고 항들을 모으면 다음을 도출합니다.

Prob(θ\=θk|Y)≈f(Y|θk)pk∑l\=1Kf(Y|θl)pl.

표본 크기 _S_ 가 증가함에 따라 _p_ _k_ 는 π(θk) 로 수렴하고 근사 사후 확률은 실제 확률로 수렴합니다. 이 논리는 θ 가 좁은 영역에 있을 확률을 평가함으로써 연속형 파라미터에도 확장 적용됩니다.

SMC는 일반적으로 많은 수의 파티클 _S_ 를 고려하지만, 이는 다수의 파티클이 사후 분포에 의해 지지될 때만 좋은 근사를 제공합니다. 예를 들어 S\=10000 이지만 가중치 중 2개만이 0이 아니라면, 사후 분포에 대한 유효 표본 크기(effective sample size)는 _S_ 가 아니라 2가 됩니다. 이런 직관은 유효 표본 크기 통계량으로 공식화됩니다.

ESS\=(∑s\=1Sws)2∑s\=1Sws2.

ESS는 1에서 _S_ 사이입니다. 최상의 경우에는 파티클들이 동등하게 가중치를 부여받아 모든 _s_ 에 대해 ws\=w 가 되며 ESS\=S 가 됩니다. 최악의 경우에는 한 _s_ 에 대해 ws\=1 이고 나머지 모든 _s_ 에 대해서는 ws\=0 이 되어 ESS\=1 이 됩니다. 일반적으로 _ESS_ 가 예를 들어 ESS\=1000 처럼 충분히 커질 때까지 _S_ 를 증가시킵니다.

[리스트 3.8](./11-chapter3.md#list3_8)과 [그림 3.16](./11-chapter3.md#fig3_16)은 베타-이항 모델(beta-binomial model) Y|θ∼Binomial(n,θ) 및 θ∼Beta(a,b) 에 대해, Y\=10, n\=40 이고 a,b\=1 인 조건에서의 SMC를 시연합니다. 먼저, MCMC 샘플링(이 경우에는 단지 100\. 베타 사후 분포로부터의 독립 표본)을 사용하여 사후 분포가 근사됩니다. 이것은 높은 사후 질량(posterior mass)을 가진 영역에 존재하며 사후 평균을 근사할 때 동등하게 가중치(ws\=1/S)가 부여된 _S_ 개의 표본을 산출합니다. 반면 SMC 표본들은 균등 사전 분포(uniform prior)를 따라 파라미터 공간 전반에 균등하게 분포되어 있으며 우도가 높은 표본들에 대해 높은 가중치가 부여됩니다. 다수의 파티클들이 낮은 가중치를 갖기 때문에, S\=50 개의 파티클의 유효 표본 크기(ESS)는 9.7에 불과합니다. 따라서 더 정밀한 근사를 제공하기 위해 추가적인 파티클들이 필요하게 됩니다.

![Two posterior density plots show theta on the horizontal axis and Posterior on the vertical axis. Each includes a smooth posterior curve w of s with sampled theta values plotted as vertical lines. In the left plot, samples cluster between about 0.2 and 0.4. In the right plot, a resampling scheme produces many repeated vertical lines near the posterior peak around 0.3, with occasional samples spread toward the tails.](./images/fig3_16.jpg)

그림 3.16 **MCMC 대비 SMC 비교 시연(Illustration of SMC versus MCMC).** 베타-이항 모델에 대한 MCMC의 S\=50 개의 동일 가중치 표본(좌측) 및 SMC의 가중치 부여 표본(우측). [본문으로 돌아가기.⏎](chapter3)

알고리즘 5 순차 몬테카를로 샘플링 (Sequential Monte Carlo sampling)

```
1: 사전 분포로부터 파티클 샘플링(Sample particles from the prior), θ(s)∼iidπ(θ)
2: 가중치 계산(Compute weights) us=∏i=1nf(yi|θ(s)) 그리고(and) ws=us/(∑t=1Sut)
3: 근사(Approximate) E{g(θ|Y)}≈∑s=1Swsg(θ(s)).
```

리스트 3.8 베타-이항 모델에 대한 MCMC 대비 SMC 샘플링. [본문으로 돌아가기.⏎](chapter3)

```R
  Y <- 10; n <- 40              # 데이터(Data)
  a <- b <- 1                   # 사전 분포(Prior)
  t <- seq(0,1,length=1000)            # 사후 분포(Posterior)
  p <- dbinom(Y,n,t)*dbeta(t,a,b)
  (Y+a)/(n+a+b)                               # 실제 사후 평균(True post mean)
  [1] 0.2619048

# MCMC
   set.seed(919)
  S     <- 50                      # MC 표본의 수(Number of MC samples)
  theta <- rbeta(S,Y+a,n-Y+b)
  w <- rep(1/S,S)

  plot(t,p,type="l",xlab=expression(theta),ylab="Posterior")
  points(theta,rep(0,S),cex=.6,pch=19)
  lines(theta,w,type="h")
  legend("topright",c(expression(theta^{(s)}),expression(w^{(s)})),
          pch=c(19,NA),lty=c(NA,1),bty="n")

 mean(theta) # E(theta|Y) 근사(Approximate E(theta|Y))
 [1] 0.2634136

# SMC
 set.seed(919)
 theta   <- rbeta(S,a,b)      # 사전 분포에서 추출(Sample from prior)
 u       <- dbinom(Y,n,theta)  # 우도에 의한 가중치 부여(Weight by likelihood)
 w       <- u/sum(u)

 plot(t,p,type="l",xlab=expression(theta),ylab="Posterior")
 points(theta,rep(0,S),cex=.6,pch=19)
 lines(theta,w,type="h")
 legend("topright",c(expression(theta^{(s)}),expression(w^{(s)})),
          pch=c(19,NA),lty=c(NA,1),bty="n")

 sum(w*theta) # E(theta|Y) 근사(Approximate E(theta|Y))
 [1] 0.2355692
 ESS <- (sum(w)^2)/sum(w^2)
 ESS
 [1] 9.706214
```

## 3.3 R에서의 MCMC 소프트웨어 옵션 (MCMC software options in R)

MCMC 코드를 한 줄씩(step-by-step) 작성하는 것은 훌륭한 연습이며 알고리즘을 진정으로 이해할 수 있는 유일한 방법입니다. 또한 특정 모델에 코드를 맞춤화하면 복잡한 모델에 대한 속도와 안정성을 향상시킬 수 있습니다. 하지만 기본 MCMC 샘플러를 직접 코딩하는 것은 점점 반복적인 작업이 됩니다. 깁스 샘플링의 경우, 각 파라미터의 완전 조건부 분포를 유도하고 코딩해야 하지만, 알려진 켤레 쌍(예: [부록 A.2](./18-appA.md#secA_2))은 수십 개에 불과하므로 깁스 샘플러 작성은 그저 올바른 완전 조건부 분포를 찾아 R에 입력하는 문제에 불과합니다. 파라미터의 완전 조건부 분포가 켤레 쌍 표에 없다면 가우스 무작위 보행(Gaussian random walk) 101\. 102\. 후보 분포가 있는 메트로폴리스 업데이트를 사용할 수 있습니다. 각각의 메트로폴리스 업데이트는 가우스 후보 제안과 그에 따른 수락 비율 계산으로 구성되며, 이는 오직 후보 표준 편차를 조절하고 그 모델에 대해 선택한 특정 우도와 사전 분포를 수락 비율에 끼워 넣는 것만을 요구합니다. 그러나 이 튜닝 단계는 모델 특화적인 부분이 아니며, 대부분의 모델들은 우도 및 사전 분포로 소수의 수십 가지 분포([부록 A.1](./18-appA.md#secA_1))만을 사용해 구축되므로 MH 코드를 작성하는 것은 단순히 올바른 분포들을 코딩하는 문제일 뿐입니다.

다행히도 이 과정은 자동화되어 왔습니다! R에서 호출할 수 있는 여러 범용 MCMC 패키지들이 있는데, 여기에는 Just Another Gibbs Sampler (JAGS), OpenBUGS (\[[148](./19-ref01.md#refbib148)\]), STAN (\[[28](./19-ref01.md#refbib28)\]), NIMBLE (\[[42](./19-ref01.md#refbib42)\])을 포함하여 파이썬(python)의 pymc 라이브러리(\[[121](./19-ref01.md#refbib121)\]) 및 SAS 프로시저인 PROC MCMC (\[[31](./19-ref01.md#refbib31)\]) 등이 있습니다. 이 패키지들은 단일 모델에 특화되어 있지 않습니다. 그 대신 우도와 사전 분포를 지정하는 스크립트를 입력으로 받아들이고, 이 정보를 이용해 지정된 모델에 대한 MCMC 샘플러를 구성합니다. 이 패키지들의 형식은 유사하므로 결과를 해석하기 위한 MCMC 샘플링에 대한 탄탄한 이해가 있다면 하나에서 다른 것으로 전환하는 것은 꽤 간단합니다.

이 책의 남은 부분에서는 MCMC 샘플링을 수행하기 위해 JAGS를 사용할 것입니다. 다른 패키지들에 비해 JAGS는 코딩하기 비교적 쉽고 이 책에서 다루는 모델들의 규모 및 복잡성을 고려할 때 충분히 빠릅니다. JAGS는 이 책에서 논의된 모든 모델과 그 밖의 더 많은 모델을 적합하는 데 사용할 수 있는 매우 범용적인 패키지입니다. 물론 특정 애플리케이션을 위해서는 더 집중적이고 그만큼 효율적인 코드가 있을 수 있습니다. 예를 들어 베이지안 선형 회귀의 경우 R의 BLR 패키지가 확실히 JAGS보다 성능이 더 낫습니다. 하지만 서로 다른 모델을 위해 수십 개의 패키지를 학습해야 하는 것을 피하고자 이 책 전체에 걸쳐 우리는 단순히 JAGS를 사용합니다.

먼저 <http://mcmc-jags.sourceforge.net/> 에서 JAGS를 다운로드하여 컴퓨터에 설치해야 합니다. R에서 JAGS와 통신하기 위해 rjags R 패키지도 다운로드하여 설치해야 합니다. JAGS를 사용한 MCMC 분석은 다음의 5단계를 거칩니다.

1. R에서 문자열로서 모델을 지정합니다.
2. 데이터를 업로드하고 `jags.model` 함수를 사용하여 모델을 컴파일합니다.
3. `update` 함수를 사용하여 번인 샘플링(burn-in sampling)을 수행합니다.
4. `coda.samples` 함수를 사용하여 사후 표본을 생성합니다.
5. `summarize` 및 `plot` 함수를 사용하여 결과를 요약합니다.

리스트 3.9 T-rex 데이터에 적용된 선형 회귀에 대한 JAGS 코드. [본문으로 돌아가기.⏎](chapter3)

```R
 library(rjags)
# T-Rex 데이터 로드(Load T-Rex data)
   mass <- c(29.9, 1761, 1807, 2984, 3230, 5040, 5654)
   age <- c(2, 15, 14, 16, 18, 22, 28)
   n   <- length(age)
   data <- list(mass=mass,age=age,n=n)

# (1) 모델을 문자열로 정의(Define the model as a string)
  model_string <- textConnection("model{
    # 우도 (dnorm은 분산이 아니라 정밀도를 사용함) (Likelihood (dnorm uses a precision, not variance))
     for(i in 1:n){
       mass[i] ~ dnorm(beta1 + beta2*age[i],tau)
     }
    # 사전 분포(Priors)
     tau ~ dgamma(0.1, 0.1)
     sigma <- 1/sqrt(tau)
     beta1 ~ dnorm(0, 0.001)
     beta2 ~ dnorm(0, 0.001)
  }")

# (2) 데이터를 로드하고, 초기값을 지정하며 MCMC 코드를 컴파일함 (Load the data, specify initial values and compile the MCMC code)
  inits <- list(beta1=rnorm(1),beta2=rnorm(1),tau=rgamma(1,1))
  model <- jags.model(model_string,data = data, inits=inits, n.chains=2)

# (3) 10000개 표본을 위한 번인(Burn-in for 10000 samples)
  update(model, 10000, progress.bar="none")

# (4) 20000개의 번인 후 표본 생성(Generate 20000 post-burn-in samples)
  params <- c("beta1","beta2","sigma")
  samples <- coda.samples(model,
            variable.names=params,
            n.iter=20000, progress.bar="none")

# (5) 출력 요약(Summarize the output)
  summary(samples)
  1. 각 변수에 대한 경험적 평균과 표준 편차, 그리고 평균의 표준 오차 (Empirical mean and standard deviation for each variable,
    plus standard error of the mean):

           Mean     SD  Naive SE Time-series SE
 beta1     2.512 31.61  0.1580       0.1580
 beta2    52.763 39.21  0.1961       0.3727
 sigma   2792.738 1177.88 5.8894     9.7678

 2. 각 변수에 대한 분위수 (Quantiles for each variable):

          2.5%     25%     50%   75%  97.5%
 beta1  -59.53  -18.87   2.601 23.57  64.61
 beta2  -21.36   25.71  51.531 78.34  134.17
 sigma 1083.16  1997.85 2601.864 3361.14 5622.69

  plot(samples)
```

[리스트 3.9](./11-chapter3.md#list3_9)에는 T. rex 성장 차트 데이터에 적용된 단순 선형 회귀에 대한 R 코드가 포함되어 있으며, 마지막 줄에 있는 plot 명령어의 출력값은 [그림 3.17](./11-chapter3.md#fig3_17)로 제공됩니다. 이 코드에 대한 몇 가지 주의 사항은 다음과 같습니다.

* 이 코드는 모두 R에서 실행됩니다. 컴퓨터에 JAGS를 설치해야 하지만, rjags 라이브러리가 R과 JAGS 사이에서 데이터 및 결과를 주고받기 때문에 사용자가 직접 JAGS를 열어볼 필요는 전혀 없습니다.
* 모델 사양 형식(model specification format)은 R 코드와 닮아 있지만 모든 R 명령어가 JAGS에서 적용되는 것은 아니며, 동일한 구문(syntax)을 가진 몇몇 명령어들도 다른 속성들을 갖습니다. 예를 들어, JAGS의 dnorm은 평균 및 표준 편차를 이용하는 동명의 R 명령어와 달리, 평균 및 정밀도(역-분산)를 통하여 정규 분포를 명시합니다. JAGS 명령어 목록과 그 의미는 \[[123](./19-ref01.md#refbib123)\]의 사용자 매뉴얼(user manual)을 참조하세요.
* “∼” 기호는 변수가 기호의 우측에 주어진 분포를 따름을 뜻하고, 결정적(deterministic) 연산들은 좌향 화살표 “<-”로 표기됩니다.
* 모델 정의 내에서 사전 분포는 정밀도(precision) tau에 놓이지만, 매 표본들은 표준 편차 sigma < - 1/sqrt(tau) 로 변환되며 결과적으로 sigma의 표본들이 반환됩니다. 즉, 이 줄은 JAGS에게 각각의 반복에서 σ(s)\=1/τ(s) 을 연산하고 그 표본 σ(s) 을 반환하도록 지시합니다. 103\.
* 104\. `model` 객체(object)는 데이터, 각 파라미터를 업데이트할 코드, 및 각각의 체인 내에 존재하는 각 파라미터들의 현재 상태를 포함합니다.
* 초기값 함수(initial values function)는 이 곳의 예시처럼 초기값을 위한 무작위 표본 추출값을 설정하거나 혹은 특정한 값(예컨대 beta1 = 0)을 부여할 수도 있습니다. `jags.model` 내에 inits 인자를 생략할 경우 JAGS가 자동으로 초기값을 세팅해줍니다.
* `update` 함수는 각각의 체인 내에서 매 파라미터의 상태들을 변경하지만 중간 표본들을 저장하지는 않습니다.
* `coda.samples` 함수는 모든 MCMC 표본을 보존하지만 variable.names에 열거된 파라미터들에 한해서만 그렇습니다. 이 함수는 표본들을 `samples` 객체 안으로 산출해내며, 이 객체는 수렴성을 연구하는 데 쓸 수 있는 `coda` 패키지에서 사용하는 형식을 따릅니다.
* JAGS는 빌트-인 `plot` 및 `summary` 기능들을 갖추고 있으며 [리스트 3.9](./11-chapter3.md#list3_9) 내에서도 쓰였지만, 체인 _c_ 에 대한 S×p 표본 행렬인 `samples[[c]]` 를 추출하여 자신만의 커스텀 플롯을 만들 수도 있습니다. 모든 표본들을 단일한 2S×p 행렬 안으로 한 데 모아  
`samps_matrix <- rbind(samples[[1]],samples[[2]])`  
예컨대 사후 분위수들은 다음과 같이 도출할 수도 있습니다.  
`apply(samps_matrix,2,quantile,`  
`      c(0.025,0.250,0.500,0.750,0.975))`
* `summary` 기능은 체인들에 걸친 전체 표본을 통합하며 사후 표본들의 표본 평균, 표준 편차, 그리고 분위수들을 줍니다.
* `summarize` 함수가 제공하는 각 파라미터 주변 사후 분포의 사후 평균과 95% 구간(즉, "Mean"과 "2.5%" 및 "97.5%" 항목들)을 사용하여 사후 분포를 요약할 수 있습니다.

리스트 3.10 NFL 뇌진탕 데이터에 대한 JAGS 코드. [본문으로 돌아가기.⏎](chapter3)

```R
 library(rjags)
# NFL 뇌진탕 데이터 로드(Load the NFL concussion data)
```


```R
  Y <- c(171, 152, 123, 199)
  n <- 4
  N <- 256

# (1) 모델을 문자열로 정의(Define the model as a string)
 model_string <- textConnection("model{
   # 우도(Likelihood)
    for(i in 1:n){
      Y[i] ~ dpois(N*lambda[i])
    }
   # 사전 분포(Priors)
    for(i in 1:n){
      lambda[i] ~ dgamma(1,gamma)
    }
    gamma ~ dgamma(a, b)
 }")

# (2) 데이터를 로드하고 MCMC 코드를 컴파일함(Load the data and compile the MCMC code)
 inits <- list(lambda=rgamma(n,1,1),gamma=1)
 data <- list(Y=Y,N=N,n=n,a=0.1,b=0.1)
 model <- jags.model(model_string,data = data, inits=inits, n.chains=2)

# (3) 10000개 표본을 위한 번인(Burn-in for 10000 samples)
 update(model, 10000, progress.bar="none")

# (4) 20000개의 번인 후 표본 생성(Generate 20000 post-burn-in samples)
 params <- c("lambda")
 samples <- coda.samples(model,
           variable.names=params,
           n.iter=20000, progress.bar="none")

# (5) 90% 신뢰 구간 계산(Compute 90% credible intervals)
 samps <- rbind(samples[[1]],samples[[2]]) #2S x 4 표본 행렬(2S x 4 matrix of samples)
 apply(samps,2,quantile,c(0.05,0.95))

       lambda[1] lambda[2] lambda[3] lambda[4]
 2.5% 0.5722272 0.5035036 0.4005522 0.6717104
97.5% 0.7704071 0.6925751 0.5685348 0.8878783
```

리스트 3.11 열악한 수렴을 보여주는 장난감(toy) 예제. [본문으로 돌아가기.⏎](chapter3)

```R
# 모델을 문자열로 정의(Define the model as a string)
> model_string <- textConnection("model{
> Y ~ dpois(exp(mu[1]+mu[2]))
> mu[1] ~ dnorm(0,0.001)
> mu[2] ~ dnorm(0,0.001)
> }")

# MCMC 표본 생성(Generate MCMC samples)
> inits <- list(mu=rnorm(2,0,5))
> data <- list(Y=1)
> model <- jags.model(model_string,data = data,
> inits=inits, n.chains=3)

>   update(model, 1000, progress.bar="none")
>   samples <- coda.samples(model,
>   variable.names=c("mu"),
>   n.iter=5000, progress.bar="none")

># 수렴성 진단 적용(Apply convergence diagnostics)

> # 플롯(Plots)
> plot(samples)
> autocorr.plot(samples)

> # 통계량(Statistics)
> autocorr(samples[[1]],lag=1)
, , mu[1]
          mu[1] mu[2]
Lag 1 0.9948544 -0.9926385
, , mu[2]
           mu[1] mu[2]
Lag 1 -0.9960286 0.9947489

> effectiveSize(samples)
   mu[1] mu[2]
22.90147 22.71505

> gelman.diag(samples)
      Point est. Upper C.I.
mu[1]      1.62   2.88
mu[2]      1.62   2.88

Multivariate psrf
1.48

> geweke.diag(samples[[1]])
 mu[1] mu[2]
-0.6555 0.6424
```

![Six panels summarise M C M C output for beta one beta two and sigma. Trace plots over iterations show dense fluctuations around stable ranges for each parameter, with sigma exhibiting larger spikes. Corresponding density estimates appear on the right: beta one and beta two form smooth bell shaped curves centred near zero, while sigma has a right skewed density with most mass near smaller values and a long tail toward larger values.](./images/fig3_17.jpg)

그림 3.17 **T. rex 분석에 대한 JAGS 출력 결과.** 이것은 T. rex 성장 차트 데이터의 선형 회귀에 대한 JAGS 내 `plot` 함수의 출력 결과입니다. 첫 번째 열의 플롯에 있는 회색 음영이 다른 두 선은 두 개의 체인(chains)으로부터 얻은 표본들입니다. 우측의 밀도 플롯(density plots)은 체인에 걸친 표본들을 결합하여 각 파라미터의 주변 분포(marginal distribution)를 요약합니다. [본문으로 돌아가기.⏎](chapter3)

[리스트 3.10](./11-chapter3.md#list3_10)은 포아송-감마 모델을 NFL 뇌진탕 데이터에 적합시키는 두 번째 예제를 제공합니다. 과정 및 코드는 이전 예제와 매우 비슷하지만, 빌트인된 요약 및 플롯 함수를 사용하는 대신 [리스트 3.10](./11-chapter3.md#list3_10)은 `rjags` 객체(object)인 `samples`에서 표본들을 추출하여 2S×4 표본 행렬 `samps`로 통합합니다. 이를 통해 결과를 설명하는 데 사용되는 사후 분포 요약에 훨씬 더 많은 유연성이 제공됩니다. 예컨대 [리스트 3.10](./11-chapter3.md#list3_10)은 90% 사후 구간(posterior intervals)을 계산합니다.

리스트 3.12 훌륭한 수렴을 보여주는 장난감(toy) 예제. [본문으로 돌아가기.⏎](chapter3)

```R
# 모델을 문자열로 정의(Define the model as a string)
> model_string <- textConnection("model{
> Y1 ~ dpois(exp(mu[1]))
> Y2 ~ dpois(exp(mu[2]))
> mu[1] ~ dnorm(0,0.001)
> mu[2] ~ dnorm(0,0.001)
> }")

# MCMC 표본 생성(Generate MCMC samples)
> inits <- list(mu=rnorm(2,0,5))
> data <- list(Y1=1,Y2=10)
> model <- jags.model(model_string,data = data,
>                           inits=inits, n.chains=3)

>   update(model, 1000, progress.bar="none")
>   samples <- coda.samples(model,
>                   variable.names=c("mu"),
>                   n.iter=5000, progress.bar="none")

># 수렴성 진단 적용(Apply convergence diagnostics)

> # 플롯(Plots)
> plot(samples)
> autocorr.plot(samples)

> # 통계량(Statistics)
> autocorr(samples[[1]],lag=1)
, , mu[1]
          mu[1] mu[2]
Lag 1 0.359733 0.02112005
, , mu[2]
            mu[1] mu[2]
Lag 1 0.002213494 0.2776712

> effectiveSize(samples)
   mu[1] mu[2]
6494.326 8227.748
> gelman.diag(samples)

      Point est. Upper C.I.
mu[1]         1         1
mu[2]         1         1
Multivariate psrf
1

> geweke.diag(samples[[1]])
 mu[1] mu[2]
-0.5217 -0.2353
```

책의 나머지 부분에서 우리는 모든 MCMC 코딩에 대해 JAGS를 사용할 것입니다. 우리는 모델 구문과 출력의 요약을 자주 보여주겠지만, 데이터를 로드하거나 표본을 생성하는 단계 등은 보고하지 않을 것인데 그 이유는 이러한 코드 블록들이 모든 모델에 대해 사실상 동일하기 때문입니다. 105\. (이 페이지에 그림 존재) 106\.

## 3.4 107\. MCMC 수렴성 진단 및 개선 (Diagnosing and improving MCMC convergence)

### 3.4.1 초기값 선택하기 (Selecting initial values)

이론상 깁스 및 메트로폴리스-해이스팅스 샘플링은 어떤 초기값이 주어지든 수렴해야 하지만 실제로는 초기값의 선택이 중요합니다. 초기화에는 두 가지 견해가 있습니다. 사후 최빈값(mode)에 가까운 초기값을 선택하여 긴 단일 체인을 하나 실행하는 것과, 알고리즘이 수렴했음을 검증하기 위해 고의적으로 넓게 퍼진 초기값을 사용하여 여러 체인을 병렬로 실행하는 것입니다.

좋은 초기값을 선택하고 긴 단일 체인을 하나 실행하는 것은 번인(burn-in) 기간을 단축하거나 아예 제거할 수 있다는 장점이 있습니다. 초기값을 선택하는 일반적인 방법은 수치 최적화(numerical optimization)를 통해 계산된 최대 우도 추정량(MLE) 또는 최대 사후 확률(MAP) 추정치를 사용하는 것인데, 이는 대개 MCMC 샘플링보다 계산하기 쉽습니다. 이 접근법의 단점은 체인이 국지적 최빈값(local mode)에 갇혀 사후 분포의 주요 부분을 완전히 놓칠 가능성을 배제하기 어렵다는 점입니다.

반면, 흩어진 초기값들로 다수(일반적으로 2~5개)의 체인을 시작하는 방법은 더 많은 번인 표본들을 요구하지만 모든 체인이 동일한 결과를 낸다면 그것이 알고리즘이 올바르게 수렴했다는 증거가 됩니다. [그림 3.18](./11-chapter3.md#fig3_18)은 반복 횟수 500회 부근에서 세 개 체인의 이상화된 수렴을 보여줍니다. 순차적 특성 때문에 MCMC는 쉽게 병렬화할 수 없지만, 여러 개의 독립적인 체인을 실행하는 것은 병렬 컴퓨팅을 활용하여 베이지안 계산을 향상시키는 한 방법입니다.

![Three M C M C chains are plotted with Sample on the vertical axis and Iteration on the horizontal axis from 0 to 5000. Chain one starts near 4.3 and rises toward about 5.1. Chain two begins near 6.1 and declines to the same region. Chain three starts near 5 and stabilises quickly. All three chains mix tightly around a common value after roughly 1500 iterations.](./images/fig3_18.jpg)

그림 3.18 **세 개 병렬 체인의 수렴(Convergence of three parallel chains).** 세 개의 트레이스 플롯은 병렬로 실행된 세 MCMC 체인 내 한 파라미터에 대한 표본을 나타냅니다. 체인들은 반복 횟수 1,000회 부근에서 수렴합니다. [본문으로 돌아가기.⏎](chapter3)

단일 체인과 병렬 체인 모두 각자의 장점이 있지만, 병렬 컴퓨팅 환경의 발전을 고려하면 최소한 2개의 체인을 실행하는 편이 낫습니다. 두 체인의 수렴이 수렴성에 대한 강력한 증거가 될 수 있도록 108\. 사전에 각 체인의 시작값들이 사후 분포 전반에 걸쳐 충분히 퍼져 있는지 세심하게 확인해야 합니다.

### 3.4.2 수렴성 진단 (Convergence diagnostics)

MCMC 체인이 수렴했는지, 그리고 사후 분포를 충분히 탐색할 만큼 길게 실행되었는지 검증하는 작업은 종종 [그림 3.18](./11-chapter3.md#fig3_18)에서처럼 트레이스 플롯에 대한 비공식적인 시각적 검사를 통해 이루어집니다. 그러나 미수렴(non-convergence)을 진단하기 위해 많은 형식적인 진단 도구들도 제안되었습니다. 이 절에서 우리는 R의 `coda` 패키지를 통해 JAGS가 생성하는 몇 가지 핵심 진단 기법들에 초점을 맞춥니다.

이 섹션 전반에 걸쳐 설명을 위해 두 개의 장난감(toy) 예제를 사용합니다.

열악한 수렴 모델(Poor convergence model): Y|μ∼Poisson\[exp(μ1+μ2)\]
훌륭한 수렴 모델(Good convergence model): Y1|μ∼Poisson\[exp(μ1)\], Y2|μ∼Poisson\[exp(μ2)\].

두 모델에서 사전 분포는 μj∼iidNormal(0,1000) 입니다. 첫 번째 모델에서 두 파라미터는 식별 불가능(unidentified)하며, 이는 [리스트 3.11](./11-chapter3.md#list3_11) 및 [그림 3.19](./11-chapter3.md#fig3_19)에 나타난 바와 같이 열악한 수렴을 초래합니다. 두 번째 모델에서 두 파라미터는 별개의 관측치와 관련되어 있으므로 양쪽 파라미터 모두 식별되어 [리스트 3.12](./11-chapter3.md#list3_12) 및 [그림 3.20](./11-chapter3.md#fig3_20)에 나타난 바와 같이 훌륭한 수렴을 이끕니다. 첫 번째 모델은 터무니없으며 데이터 적합용으로 절대 쓰이지 않겠지만 수렴성이 나쁜 경우에 대한 간단한 시연을 제공합니다. 물론 수렴 문제가 모두 식별 불가능한 파라미터들과 연관되어 있는 것은 아니지만 이것이 문제의 흔한 원인이긴 합니다.

![The plots shows M C M C diagnostics for mu one and mu two. Trace plots over iterations display fluctuating but stable paths. Density plots for each parameter form smooth peaks centred near moderate values with bandwidth about 2.6. Autocorrelation plots for lags up to 30 show slow decay, with bars remaining high for early lags, indicating strong correlation among successive samples for both mu one and mu two.](./images/fig3_19.jpg)

그림 3.19 **수렴성이 열악한 장난감(toy) 예제에 대한 수렴성 진단(Convergence diagnostics for a toy example with poor convergence).** 좌측 패널은 각 파라미터 및 각 체인(회색 음영으로 구분)의 트레이스 플롯을 보여주고, 우측 패널은 첫 번째 체인에 대한 자기상관(autocorrelation) 함수를 나타냅니다. [본문으로 돌아가기.⏎](chapter3)

![Trace plots for mu one and mu two show stable fluctuations over iterations. Density estimates display a sharp right skewed peak for mu one centred near zero and a symmetric peak for mu two around 2. Autocorrelation plots for lags up to 30 show low correlation beyond the first few lags, with bars dropping quickly toward zero for both parameters, indicating improved mixing compared with the previous example.](./images/fig3_20.jpg)

그림 3.20 **수렴성이 훌륭한 장난감(toy) 예제에 대한 수렴성 진단(Convergence diagnostics for a toy example with good convergence).** 좌측 패널은 각 파라미터 및 각 체인(회색 음영으로 구분)의 트레이스 플롯을 보여주고 우측 패널은 첫 번째 체인에 대한 자기상관 함수를 보여줍니다. [본문으로 돌아가기.⏎](chapter3)

첫 번째 진단 방법은 트레이스 플롯을 그려보고 모든 체인이 동일한 분포에 도달했으며 적절히 섞였는지(mixed)를 확인하는 것입니다. [그림 3.20](./11-chapter3.md#fig3_20)(좌측 상단)은 두 파라미터에 대한 체인들이 모두 바코드처럼 보이는 좋은 예시를 제공하는 반면, [그림 3.19](./11-chapter3.md#fig3_19)(좌측 상단)는 체인이 서서히 섞이고 있으며 사후 분포에 대한 좋은 근사치를 제공하기 위해 훨씬 더 많은 반복이 필요할 것 같은 문제가 있는 경우입니다. 이러한 작은 예제에서는 모든 파라미터에 대해 전체 체인을 플로팅하는 것이 가능하지만, 모델이 더 복잡해지면 대표성을 띠는 일부 파라미터들의 체인만 검사해야 할 수도 있습니다.

체인들의 자기상관성(Autocorrelation)은 체인이 얼마나 빠르게 섞이는지를 수치적으로 측정하는 방법을 제공합니다. 이상적으로는 반복들이 서로 독립적이어야 하겠지만 샘플러의 마르코프 성질(Markovian nature)은 연속된 반복들 사이에 의존성을 유발합니다. 파라미터 _θ_ _j_ 의 체인에 대한 시차-_l_ (lag-_l_) 자기상관성은 다음과 같이 정의됩니다.

ρj(l)\=Cor(θj(s),θj(s−l)),(3.22)

그리고 함수 ρj(l) 은 자기상관 함수(autocorrelation function)라고 불립니다. [그림 3.19](./11-chapter3.md#fig3_19)와 [3.20](./11-chapter3.md#fig3_20)의 우측 패널들은 두 예제에 대한 표본 자기상관 함수를 보여줍니다. 이상적으로는 모든 l\>0 에 대하여 ρj(l)≈0 을 얻기를 기대하지만, 어느 정도의 상관성은 나타나게 마련입니다. 예컨대 [그림 3.20](./11-chapter3.md#fig3_20)에서 _μ_1 에 대한 시차 1 자기상관은 약 0.4이지만, 체인은 훌륭하게 수렴합니다.

각 파라미터에 대한 전체 자기상관 함수를 보고하는 대신, 일반적으로 시차-1 자기상관만으로도 충분합니다. 전체 함수를 요약하는 또 다른 공통적인 단일 숫자 요약 지표는 유효 표본 크기(ESS, effective sample size)입니다. MCMC 표본들의 표본 평균은 단지 실제 사후 평균의 추정치일 뿐이며, 우리는 표본 평균의 표준 오차를 사용하여 이 추정치에 대한 불확실성을 정량화할 수 있음을 상기하십시오. 만약 표본들이 독립적이라면 표본 평균의 표준 오차는 sdj/S 인데 여기서 sdj 는 _θ_ _j_ 에 대한 추출값의 표본 표준 편차이고 _S_ 는 표본 수입니다. 이것은 [리스트 3.9](./11-chapter3.md#list3_9)의 JAGS 요약 출력 결과에서 "Naive SE"로 표시됩니다. 그러나 표본이 자기상관성을 가질 경우 109\. 110\. 이는 사후 평균 추정치 내의 불확실성을 과소평가하게 됩니다. 자기상관성을 반영한 실제 표준 오차는 sdj/ESSj 임을 입증할 수 있으며, 여기서

ESSj\=S1+2∑l\=1∞ρj(l).(3.23)

실무에서 이 무한급수는 어떤 큰 _L_ 에 대하여 ∑l\=1Lρj(l) 로 절단(truncated)되어야 합니다. 모든 _l_ 에 대하여 자기상관 함수 ρj(l) 가 음수가 아니라고 가정한다면, ESSj≤S 가 되며 따라서 표본 크기는 자기상관을 조정하기 위해 할인됩니다. 이 표준 오차 sdj/ESSj 는 JAGS 요약 출력 결과에서 "Time-series SE"로 표시됩니다. 표본의 수가 충분한지 판단하는 한 가지 방법은 이 표준 오차가 모든 파라미터에 대해 허용 가능할 만큼 낮아질 때까지 _S_ 를 증가시키는 것입니다. 또 다른 방법은 유효 표본 크기가 모든 파라미터에 대해 수용 가능하게 높아질 때까지, 예를 들어 ESSj\>1000 이 될 때까지 _S_ 를 늘리는 것입니다.

게베케(Geweke) 진단 지표(\[[61](./19-ref01.md#refbib61)\])는 미수렴(non-convergence)을 감지하는 데 사용됩니다. 이는 2표본 t-검정(two-sample t-test)을 통해 샘플러의 초기 배치(batch)와 후기 배치(batch) 사이에서 체인의 평균을 비교함으로써 수렴성을 테스트합니다. `coda`의 기본 설정은 처음 10%의 표본과 마지막 50% 표본 간 평균이 같은지를 검정하는 것입니다. 배치(batch) b\=1,2 에서 _θ_ _j_ 에 대한 표본 평균 및 표준 오차를 각각 θ¯jb 와 sejb 라고 둡시다. 이 표준 오차들은 체인 내 자기상관성((3.23)에서처럼)을 감안하여 산출됩니다. 그러면 게베케 통계량은 다음과 같습니다.

Z\=θ¯j1−θ¯j2sej12+sej22.(3.24)

각 배치에 대해 평균이 같다는 귀무가설(null hypothesis)하에서 (또한 각 배치가 큰 ESS를 지닌다고 가정할 때) _Z_ 는 표준 정규 분포를 따르므로, |Z|\>2 인 경우 우려할 만한 111\. 112\. 원인이 됩니다. [리스트 3.11](./11-chapter3.md#list3_11) 및 3.12 내 2개의 예시에서 |Z| 는 1보다 작으므로 이 통계량은 미수렴을 감지하지 못합니다.

게베케 통계량은 단 하나의 체인만을 사용합니다. 다중 체인에 대한 확장은 겔만-루빈(Gelman-Rubin) 통계량(\[[57](./19-ref01.md#refbib57)\])입니다. 겔만-루빈 통계량은 _C_ 개의 체인 사이에서 평균이 일치하는 정도를 측정합니다. 이는 기본적으로 체인들이 동일한 평균을 갖는지를 판단하는 분산 분석(ANOVA, Analysis of Variance) 검정이지만, 그 통계량은 1.0이 완벽한 수렴을 나타내고 1.1보다 큰 값은 미심쩍다는 의미가 되도록 스케일링됩니다. 이 통계량은 다음과 같습니다.

Rj\=S−1SW+(1S+1SC)BW(3.25)

여기서 _B_ 는 _θ_ _j_ 에 대한 _C_ 개의 MCMC 표본 평균 분산의 _S_ 배이고 _W_ 는 _θ_ _j_ 에 대한 _C_ 개의 MCMC 표본 분산들의 평균입니다. `coda` 패키지는 반복(iteration)의 함수로서 겔만-루빈 통계량을 플롯팅하므로, 통계량이 1에 도달하면 병렬 체인들 사이에 혼합(mixing)이 잘 이루어지고 있으며 체인들이 정상 분포(stationary distribution)에 도달했을 가능성이 있음을 나타냅니다. 겔만-루빈 통계량은 [리스트 11](./11-chapter3.md#list3_11)과 [리스트 3.12](./11-chapter3.md#list3_12)의 예제를 분명하게 구별해 줍니다. 열악한 수렴의 경우 Rj\=1.62 인 데 반해 훌륭한 수렴의 경우 Rj\=1.00 입니다.

### 3.4.3 수렴성 개선 (Improving convergence)

사용자가 이러한 진단 도구를 바탕으로 표본 출력 결과를 정직하게 점검한다면 대개 불량한 수렴 여부는 포착해 낼 수 있습니다. 이로 인해 수렴성을 어떻게 개선시킬 것인가라는 보다 까다로운 과제가 남게 됩니다! 언제나 수렴 문제를 해결해 주는 단 하나의 조치란 없지만, 아래 목록에서 몇 가지 제안 사항을 제공합니다.

1. 113\. **반복 횟수 늘리기(Increase the number of iterations)**: 이론적으로 MCMC 알고리즘은 _S_ 가 무한대로 감에 따라 사후 분포에 수렴하여 해당 분포를 모두 커버해야 합니다. 물론 시간이라는 한계가 있기 때문에 생성 가능한 표본 수가 무한할 수는 없지만, 때로는 다른 개선책을 모색하기보다는 _S_ 를 높이는 편이 더 빠를 수도 있습니다. 하지만 특히나 높은 자기상관성 때문에 체인 이동이 느린 경우, 이는 불량한 수렴 현상을 해결하기에 가장 만족스러운 조치라고 보기는 어렵습니다.
2. **MH 후보 분포 조정(Tune the MH candidate distribution)**: 번인 기간 동안 수락 확률이 대략 30~50% 부근이 되도록 조정할 수 있으나 체인이 진행되면서 수락 확률도 변할 수 있습니다. 만약 파라미터 그룹 간에 강한 교차 상관성(cross-correlation)이 존재하는 경우, 이들을 블록 단위로 묶어서 업데이트하게 되면 수렴성을 획기적으로 향상시킬 수 있습니다. 하지만 이 경우 후보 상관행렬이 사후 상관행렬에 근사하도록 블록 단계들을 세심히 조정해야 합니다.
3. **초기값 향상(Improve initial values)**: 샘플러는 사후 분포를 일단 찾게 되면 잘 구동하는 경우가 잦지만 사후 확률 밀도 함수(PDF)가 최빈값(mode)에서 너무 멀리 벗어나 있을 경우 평탄해져 샘플러가 중심부를 찾지 못할 가능성이 있습니다. 초기값을 개선하는 한 가지 방법은 최대 우도 추정치(MLE)를 사용하는 것입니다(θ(0)\=θ^MLE). 또 다른 옵션은 번인 기간 동안 담금질 기법(simulated annealing, 시뮬레이티드 어닐링)을 활용하는 것입니다. 시뮬레이티드 어닐링에서는 MH 수락 비율을 Ts∈(0,1\] 거듭제곱으로 올리며(즉, RTs), 여기서 온도(temperature) _T_ _s_ 는 번인 기간 내내 1까지 상승합니다. 예컨대 s<B 일 땐 Ts\=s/B 이며 s\>B 일 땐 Ts\=1 인 방식입니다(여기서 _B_ 는 번인 표본의 횟수). 이러한 변형의 바탕이 되는 직관적 원리는 수락 비율을 거듭제곱으로 올림으로써 알고리즘이 번인 기간 도중 큰 폭의 도약을 더 잘 이뤄내게끔 하고, 결과적으로 번인 기간이 끝날 무렵 사후 분포의 중심부에 천천히 정착하도록 유도한다는 데 있습니다.
4. **더 고급 알고리즘 활용(Use a more advanced algorithm)**: [부록 A.4](./18-appA.md#secA_4)에서는 유독 까다로운 문제를 해결하는 데 쓸 수 있는 여러 개의 진일보된 알고리즘들을 소개하고 있습니다. 예컨대 해밀토니안 몬테카를로(HMC, Hamiltonian Monte Carlo)는 영리하게 후보를 제안하기 위해 사후 분포의 미분(gradient)을 활용하는 메트로폴리스 샘플러이며, 적응형(adaptive) 메트로폴리스 추정(estimates)은 제안 분포가 반복을 거치며 스스로를 진화시킬 수 있도록 해줍니다.
5. **모델 단순화(Simplify the model)**: 당면한 데이터에 적용하기엔 모델이 과하게 복잡할 경우 흔히 불량한 수렴성이 나타납니다. 지나치게 복잡한 모델에는 식별되지 않아서 추정할 수 없는 파라미터들이 포함되어 있는 경우가 잦습니다. 예를 들어 Yi∼iidNormal(θ1+θ2,σ2) 모델에서는 두 평균 파라미터들이 식별되지 않으며, 이는 동일 우도를 낳는 조합들이 수없이 많다는 뜻입니다(예: θ\=(−10,10) 와 θ\=(10,−10) 모두 같은 평균치를 제공함). 물론 이렇듯 대놓고 식별되지 않는 형태의 모델을 데이터 적합용으로 고를 사람은 아무도 없겠지만 수십 개의 파라미터가 얽혀 있는 복잡한 모델 내에선 식별 불가능성을 쉽게 포착하기 어려울 수 있습니다. 공변량(covariates) 제거나 비선형 항(non-linear terms) 삭제, 독립을 가정한 공분산 구조(covariance structure) 축소 등을 활용해 모델을 더욱 단순하게 만들면 수렴성을 개선시킬 수 있습니다.
6. **더 많은 정보가 담긴 사전 분포 고르기(Pick more informative priors)**: 정보를 더 담고 있는 사전 분포를 취하는 행위는 앞선 모델 단순화와 유사한 효과를 불러옵니다. 어리석은 모델이라 할 수 있는 Yi∼iidNormal(θ1+θ2,σ2) 에서조차 θ1∼Normal(−3,1) 과 θ2∼Normal(3,1) 이라는 사전 분포가 주어지면 MCMC 알고리즘의 수렴성이 빠를 공산이 높습니다. 느슨한 수준의 사전 분포(weakly informative prior)조차도 사후 분포를 안정화시키고 수렴성을 높일 수 있습니다. 극단적인 예로, 경험적 베이지안 사전 분포([2장](./10-chapter2.md))에서는 귀찮은 파라미터(nuisance parameters)들을 MAP 추정치에 고정시킴으로써 불확실성 억제라는 대가를 치르더라도 수렴성을 드라마틱하게 끌어올릴 수 있습니다.
7. **시뮬레이션 연구 수행(Run a simulation study)**: 파라미터의 실제 값들을 알 수 없는 실제 데이터 분석의 경우에는 과연 체인이 수렴했는지 확인하는 과정 자체가 좌절스러울 수 있습니다. 이 경우 모델로부터 데이터를 미리 시뮬레이션해 본 다음 상이한 파라미터 값들과 여러 표본 크기 하에서 MCMC 알고리즘을 114\. (예컨대 [리스트 3.6](./11-chapter3.md#list3_6)과 같이) 적합시켜 보면 알고리즘이 띠는 여러 특성들을 파악하고 샘플러가 신뢰 가능한 결과치를 뽑아내고 있다는 믿음을 확립할 수 있습니다.

### 3.4.4 대규모 데이터 세트 다루기 (Dealing with large datasets)

빅데이터는 모든 종류의 통계적 기법들에 방대한 컴퓨팅 과제를 안겨주지만, 유독 MCMC에 미치는 파급력은 더 큽니다. 이는 MCMC가 수천 차례에 걸쳐 반복적으로 데이터를 관통해야 하기 때문입니다. 다행스럽게도 최근 수년에 걸쳐 거대한 데이터를 다루기 위한 베이지안 컴퓨팅 방법들이 급증하고 있습니다. 아마도 향후 몇 년간 이 분야 내에선 또 다른 괄목할 만한 추가 개발이 진행되겠지만 현재 통용되고 있는 가장 유용한 접근법들 중 일부를 개략적으로 아래와 같이 정리할 수 있습니다.

1. **MAP 추정(MAP estimation)**: MCMC는 전체 파라미터들의 결합 분포 전반에 걸친 추정치를 산출하기 때문에 베이지안 컴퓨팅 분야에서의 "표준(gold standard)" 격으로 꼽힙니다. 하지만 분석의 주된 목적이 그저 단일한 예측에 국한되어 있다면 파라미터 추정에 따른 불확실성 몫을 덜어내고 그저 MAP 추정치만 산출해 내는 편이 훨씬 빠릅니다.
2. **가우스 근사(Gaussian approximation)**: 베이지안 중심 극한 정리(CLT)는 표본의 크기가 충분히 크다면 사후 분포가 근사적인 정규성을 띠게 됨을 선언하고 있습니다. 그런 의미에서 이를테면 관측치가 n\=1,000,000, 공변량 p\=50 에 이르는 데이터에 무정보적 평탄 사전 분포(flat priors)를 취해 놓은 채 로지스틱 회귀를 위해 긴 MCMC 체인을 구동하는 행위는 정당화되기 어렵습니다. 그보다는 MAP 추정치를 중심으로 둔 상태로 정보 행렬(information matrix)이 지시하는 공분산을 지닌 가우스로써 사후 분포를 근사하는 쪽이 훌륭한 정확성을 지니면서도 훨씬 빠르기 때문입니다. 이러한 부류의 컴퓨팅 계산은 MLE 소프트웨어를 활용하여 수행되는 편이 일반적이며, 그에 따라 사후 분포 역시 MLE를 따르는 추출 분포 근사치와 대단히 유사한 꼴을 보일 것입니다. 그러나 이 속의 불확실성에 얽힌 해석만큼은 고스란히 베이지안의 몫으로 남습니다.
3. **변분 베이지안 컴퓨팅(Variational Bayesian computing)**: 만약에 사용자가 각각의 파라미터에 걸친 주변 분포에만 관심이 있다면 굳이 베이지안 CLT를 유도해 내는 과정 없이도 변분 베이지안 근사법(3.1.4장)을 구동하는 게 속도 면에선 더 유리할 수 있습니다. 환언컨대 사전 분포(prior)가 아니라 사후 분포(posterior)가 파라미터 전체에 걸쳐서 상호 독립적이라는 전제 하에 이런 형식의 구조를 가진 이상적이고 올바른 실제 사후 근사치를 적용할 수 있다면, 결합 분포를 연산하는 복잡한 과정 없이 근사화된 사후 분포를 얻어낼 수 있습니다. 이 과정 속엔 물론 "과연 이 상태에서 불확실성을 올바르게 계상해 낼 수 있는가"라는 쟁점이 야기되지만, 엄청난 컴퓨팅 비용 절감을 달성해 낼 수 있다는 장점이 있습니다.
4. **병렬 컴퓨팅(Parallel computing)**: MCMC는 그 자체로 선천적인 순차성을 갖습니다. 즉 여러분이 보통 첫 번째 파라미터를 갱신해 놓지 않고는 두 번째를 갱신할 도리가 없다는 점을 뜻합니다. 다만 몇몇 특정 상황하에서는 MCMC 루틴상 여러 단계들을 CPU나 GPU 클러스터상의 상이한 코어에서 동시다발적으로 구동해 볼 수 있습니다. 예컨대 (조건부) 독립성을 지닌 각각의 파라미터의 경우에는 이들을 병렬로 나눠 업데이트해 볼 수 있습니다. 대안적으로 데이터가 독립이고 우도가 _n_ 개 항의 곱셉으로 인수가 분해될 때면 각각의 MCMC 단계 중 이뤄지는 우도 연산을 서로 간섭하지 않게 분산 병렬화할 수 있습니다. 예를 들어 첫 번째 코어엔 _n_1 개 항, 두 번째 코어엔 _n_2 개 항을 배치하는 방식이 이에 해당합니다. 이론적으로 병렬화는 엄청난 효율을 낼 수 있지만 정교하게 수행되지 않는 이상 코어 사이에서 수시로 데이터를 통신하게 되며 도리어 성능 저하를 초래할 소지가 다분합니다.
5. **분할 정복법(Divide and conquer)**: 분할 및 정복 기법은 어마어마할 정도로 쪼개기가 쉬운 병렬성에 힘입어 사후 분포에 대한 훌륭한 근사치를 제시해 줍니다. 전체 데이터를 _T_ 개의 배치 Y1,...,YT 로 파티셔닝했다고 상정해 봅시다. 여기서 데이터는 각 모델 파라미터가 주어진 상태에서 독립적이라고 가정됩니다. 그렇게 되면 115\. 전체 사후 분포는 다음과 같이 표기됩니다.  
p(θ|Y)∝f(Y|θ)π(θ)\=∏t\=1T\[f(Yt|θ)π(θ)1/T\].(3.26)  
이 분해 과정은 _T_ 개의 병렬화된 독립형 베이지안 분석을 구동할 수 있음을 제시합니다. 이 과정에서 분석 단계 _t_ 가 데이터 세트 Yt 와 쪼개어진 사전 분포인 π(θ)1/T 를 가지고 구동된 후 결과값들을 하나로 합산하는 식입니다. 여기서 각각의 _T_ 번 분석은 기존 사전 분포 π(θ) 에 1/T 거듭제곱을 한 값을 가지게 되어 모든 배치들이 적절한 사전 정보량을 동등하게 나눠 갖게 해줍니다. 예를 들어 θ∼Normal(0,τ2) 인 조건이라면 이를 1/T 로 쪼갠 배치의 사전 분포는 Normal(0,Tτ2) 이 됩니다. 스콧(Scott) 등 \[[140](./19-ref01.md#refbib140)\]은 _T_ 개의 이 사후 분포 덩어리들을 결합할 수 있는 여러 가지 방식들을 논의하고 있습니다. 여기서 가장 단순한 길은 "사후 분포가 대략적으로 가우스를 따른다"는 전제를 차용하는 것입니다. 이렇게 되면 각각의 분석 _t_ 는 θ|Yt≈Normal(Mt,Vt) 라는 산출물을 내어 줍니다. 이런 상황 하에서 해당 사후 분포 결합은 다음과 같습니다.  
p(θ|Y)≈Normal(V∑t\=1TVt−1Mt,V)(3.27)  
여기서 V−1\=∑t\=1TVt−1 입니다.
6. **순차 몬테카를로(Sequential Monte Carlo)**: 추가적인 새로운 데이터 포인트 배치가 들어올 때마다 사후 근사에 쓰이는 입자 가중치를 갱신해 주는 스트리밍 데이터를 위한 SMC 메소드 기법은 3.2.4장에서 소개되었습니다. 이것은 방대한 대규모 데이터를 자잘한 꼬마 배치(미니배치) 크기로 일부러 분할한 뒤 차례대로 해당 미니배치를 분석하는 흐름을 거침으로써 대용량 데이터 환경에도 얼마든지 확장 적용 가능합니다. 방대한 데이터 세트를 마주할 시 이 방식이 가지는 장점은 언제나 오직 단일한 배치와 파티클 가중치만 메모리 내에 두면 된다는 점입니다.
7. **확률적 미분 기반(Stochastic gradient) MCMC**: 데이터를 작은 크기의 미니배치들로 쪼개는 방식은 특정 MCMC 기법들로까지 확장 적용할 수 있습니다\[[115](./19-ref01.md#refbib115)\]. 해밀토니안 몬테카를로 방법(부록 9.4장 참조)의 경우 사후 분포의 미분값을 활용해 보다 정밀하고 지능적인 수준의 메트로폴리스-해이스팅스 기반 후보 분포를 튜닝해 내는데, 확률적 기울기 방식을 차용한(Stochastic gradient) MCMC는 이를 바탕 삼아 원 데이터 중 일부분의 무작위 표본 덩어리만을 뽑은 후 기울기를 근사함으로써 덩치가 큰 대용량 데이터 세트상에서도 경이로운 효율성을 달성합니다.

## 3.5 연습 문제 (Exercises)

1. 다음 방법들의 장점과 단점을 제시하세요.
   1. 최대 사후 확률(MAP) 추정
   2. 수치 적분(Numerical integration)
   3. 베이지안 중심 극한 정리(Bayesian central limit theorem)
   4. 깁스 샘플링(Gibbs sampling)
   5. 메트로폴리스-해이스팅스 샘플링(Metropolis–Hastings sampling)
2. 모든 _μ_ 에 대하여 무정보적 사전 분포 π(μ)\=1 을 취하며, _σ_ _i_ 가 알려져 있고 i∈{1,...,n} 일 때 Yi|μ∼indepNormal(μ,σi2) 를 따른다고 가정합니다.
   1. _μ_ 에 대한 MAP 추정량 공식을 구하세요.
   2. n\=3, Y1\=12, Y2\=10, Y3\=22, σ1\=σ2\=3 이고 σ3\=10 을 관측했습니다. _μ_ 의 MAP 추정치를 구하세요.
```


   3. 116\. 수치 적분(numerical integration)을 사용하여 _μ_ 의 사후 평균(posterior mean)을 계산하세요.
   4. _μ_ 의 사후 분포를 플롯팅하고 해당 플롯에 MAP 및 사후 평균 추정치를 표시하세요.
3. i\=1,...,n 에 대해 Yi|λ∼indep Poisson(Niλ) 라고 가정합니다.
   1. _λ_ 에 대한 켤레 사전 분포(conjugate prior)를 식별하고, 이 사전 분포에 따르는 사후 분포를 유도하세요.
   2. λ∼Uniform(0,20) 사전 분포를 사용하여 _λ_ 의 MAP 추정량을 유도하세요.
   3. λ∼Uniform(0,20) 사전 분포를 사용하고 n\=2, N1\=50, N2\=100, Y1\=12 및 Y2\=25 임을 가정하여 _λ_ 의 그리드 상에 사후 분포를 플롯팅하고, MAP 추정치가 실제로 최대화점(maximizer)임을 보여주세요.
   4. (c)의 설정 하에서 베이지안 중심 극한 정리(CLT)를 사용하여 _λ_ 의 사후 분포를 근사하고, 이 근사 사후 분포를 플롯팅한 뒤 (c)의 플롯과 비교하세요.
4. i\=1,...,n 에 대하여 Yi|σi2∼indepNormal(0,σi2) 이고 여기서 σi2|b∼InvGamma(a,b) 및 b∼Gamma(1,1) 인 모델을 고려해 보세요.
   1. σ12 와 _b_ 에 대한 완전 조건부 사후 분포(full conditional posterior distributions)를 유도하세요.
   2. 깁스 샘플링을 위한 의사 코드(pseudocode)를 작성하세요. 즉 깁스 샘플링 알고리즘의 각 단계를 상세히 설명하세요.
   3. 자신만의 깁스 샘플링 코드(JAGS가 아님)를 작성하고, 각 파라미터에 대한 주변 사후 밀도(marginal posterior density)를 플롯팅하세요. n\=10, a\=10 이고 i\=1,...,10 에 대해 Yi\=i 라고 가정하세요.
   4. a\=1 로 설정하여 분석을 반복하고 MCMC 체인의 수렴성에 대해 논평하세요.
   5. JAGS를 사용하여 (c)의 이 모델을 구현하고 (c)의 결과와 비교하세요.
5. i\=1,...,n 에 대해 Yi|μ,σ2∼Normal(μ,σ2) 이고 i\=n+1,...,n+m 에 대해 Yi|μ,δ,σ2∼Normal(μ+δ,σ2) 인 모델을 고려해 봅니다. 여기서 μ,δ∼Normal(0,1002) 이고 σ2∼InvGamma(0.01,0.01) 입니다.
   1. 이 모델이 적절할 수 있는 실제 실험의 예를 하나 들어보세요.
   2. _μ, δ_ 및 _σ_2 에 대한 완전 조건부 사후 분포를 유도하세요.
   3. n\=m\=50, μ\=10, δ\=1 및 σ\=2 로 하여 이 모델에서 데이터 세트를 시뮬레이션하세요. 시뮬레이션된 데이터에 위 모델을 적합시키기 위한 자신만의 깁스 샘플링 코드(JAGS가 아님)를 작성하고 각 파라미터에 대한 주변 사후 밀도를 플롯팅하세요. 실제 값들을 합리적으로 잘 복원할 수 있나요?
   4. JAGS를 사용하여 이 모델을 구현하고 그 결과를 (c)의 결과와 비교해 보세요.
6. Y|θ,b∼Binomial(n,θ), θ|b∼Beta{exp(b),exp(b)} 및 b∼Normal(0,1) 모델을 고려해 봅니다.
   1. _θ_ 와 _b_ 에 대한 초기값을 지정하세요.
   2. _θ_ 의 완전 조건부 분포는 무엇인가요?
   3. _b_ 의 완전 조건부 분포는 훌륭한 형태(nice form)를 갖지 않으므로 깁스 샘플링을 사용하여 업데이트할 수 없습니다. (θ,b) 의 사후 분포에 대한 깁스 내의 메트로폴리스 샘플러(Metropolis-within-Gibbs sampler)를 개략적으로 작성하세요(sketch).
   4. 117\. n\=100 및 Y\=10 으로 여러분의 깁스 내의 메트로폴리스 샘플러를 구현하고 사후 분포를 요약하세요.
7. [1장](./09-chapter1.md)의 연습 문제 22번 표에 있는 NBA 자유투 데이터에 다음 모델을 적합시키세요.
Yi|θi∼Binomial(ni,θi) 및 θi|m∼Beta\[exp(m)qi,exp(m)(1−qi)\],
여기서 _Y_ _i_ 는 선수 i\=1,...,10 이 클러치 상황(clutch shots)에서 성공시킨 슛의 횟수, _n_ _i_ 는 시도한 클러치 슛의 횟수, qi∈(0,1) 는 전체 슛 성공 비율(overall proportion)이며, m∼Normal(0,10) 입니다.
   1. 이것이 왜 _θ_ _i_ 에 대한 합리적인 사전 분포인지 설명하세요.
   2. 사전 분포 내에서 _m_ 의 역할을 설명하세요.
   3. _θ_1 에 대한 완전 조건부 사후 분포를 유도하세요.
   4. 11개의 모든 모델 파라미터(θ1,...,θ10,m)들에 대하여 사후 평균과 95% 신뢰 구간 표를 계산하기 위한 자신만의 MCMC 알고리즘을 작성하세요. 주석이 달린 코드를 제출하세요.
   5. JAGS에서 동일한 모델을 적합시키세요. 주석이 달린 코드를 제출하고, 두 알고리즘이 동일한 결과를 산출했는지에 대해 논평하세요.
   6. 이 문제에서, 그리고 일반적으로 JAGS를 사용하는 대신 직접 코드를 작성하는 것의 장점과 단점은 무엇입니까?
8. 모델 Y|θ∼Gamma(θ,1) 와 사전 분포 θ∼Uniform(0,10) 를 가정합니다. Y\=5 를 관측했다고 칩시다. 이는 켤레 사전 분포가 아니므로 _θ_ 의 사후 분포를 근사하기 위해 메트로폴리스-해이스팅스 샘플링을 사용하게 될 것입니다.
   1. 참조용으로 _θ_ 의 사후 분포를 플롯팅하세요. 즉 (R에서),
   `> theta <- seq(0,10,0.01)`
   `> plot(theta,dgamma(5,theta,1),type="l")`
   2. _θ_ 에 대해 합리적인 후보 분포는 무엇인가요?
   3. 수락 확률 공식(가급적이면 R 코드로)을 제공하세요. (0,10) 범위를 벗어나는 후보들은 어떻게 처리할 계획인가요?
   4. 후보 분포를 어떻게 조정(tune)할 것입니까? 구체적으로 명시하세요.
   5. 사후 분포로부터 표본을 생성하여 플롯팅하고, 그 결과를 (a)의 곡선과 비교해 보세요.
9. R에서 아래 코드를 사용하여 은하계(galaxies) 데이터를 열고 플롯팅하세요.
`>   library(MASS)`
`>   data(galaxies)`
`>   ?galaxies`
`>   Y <- galaxies`
`>   n <- length(Y)`
`>   hist(Y,breaks=25)`
관측치 Y1,...,Y82 를 위치 _μ_, 척도 _σ_ 및 자유도 _k_ 를 가지는 스튜던트 t-분포(Student's t-distribution)를 사용해 모델링하세요. 사전 분포 μ∼Normal(0,100002), 1/σ2\=τ∼Gamma(0.01,0.01) 및 k∼Uniform(1,30) 를 가정합니다.
   1. 3개의 파라미터 각각에 대해 합리적인 초기값을 제공하세요.
   2. JAGS를 사용하여 모델을 적합시키세요. 각 파라미터의 트레이스 플롯을 보고하고 수렴성에 대해 논의하세요.
   3. 118\. 파라미터들이 그들의 사후 평균으로 설정된 t 분포를 관측된 데이터와 그래픽을 통해 비교해 보세요. 모델이 데이터에 잘 맞습니까?
10. 은하계 데이터 다운로드
`> library(MASS)`
`> data(galaxies)`
`> Y <- galaxies`
i\=1,…,n 에 대해 Yi|θ∼iid Laplace(μ,σ) 이고 여기서 θ\=(μ,σ) 라고 가정해 봅니다.
   1. 무정보적 사전 분포 σ∼Uniform(0,100000) 와 모든 μ∈(−∞,∞) 에 대해 π(μ)\=1 를 가정할 때, θ 의 결합 사후 분포와 _μ_ 및 _σ_ 의 주변 사후 분포를 플롯팅하세요.
   2. (a)의 분석에서 θ 의 사후 평균을 계산하고 이 값들을 사용한 라플라스(Laplace) 확률 밀도 함수(PDF)를 관측된 데이터와 대조하여 플롯팅하세요. 모델이 데이터에 잘 맞습니까?
   3. 새로운 관측치 Y∗|θ∼Laplace(μ,σ) 에 대한 사후 예측 분포(PPD)를 플롯팅하세요. PPD의 평균과 분산은 (b)의 "플러그인(plug-in)" 분포의 평균 및 분산과 비교하여 어떤 차이가 있습니까?
11. [섹션 2.5](./10-chapter2.md#sec2_5)에서 우리는 레지 잭슨(Reggie Jackson)의 정규 시즌과 월드 시리즈 홈런율을 비교했습니다. 그는 2820번의 정규 시즌 경기에서 563개의 홈런을 쳤고, 27번의 월드 시리즈 경기에서 10개의 홈런을 쳤습니다(선수는 한 경기에서 0, 1, 2, … 개의 홈런을 칠 수 있습니다). 두 홈런율 모두에 대해 Uniform(0,10) 사전 분포를 가정하고 JAGS를 사용하여 (i) 그의 정규 시즌 홈런율, (ii) 월드 시리즈 홈런율, (iii) 이 두 비율의 비(ratio)에 대한 사후 분포를 요약하세요. 3개의 파라미터 모두에 대한 트레이스 플롯을 제공하고 적절한 수렴성 진단 지표를 포함하여 MCMC 샘플러의 수렴성에 대해 논의하세요.
12. [섹션 1.6](./09-chapter1.md#sec1_6)에서 논의된 바와 같이 \[[47](./19-ref01.md#refbib47)\]은 2010년부터 2015년까지 매년 발견된 해양 이매패류(marine bivalve) 종의 수가 각각 64, 13, 33, 18, 30 및 20개였다고 보고합니다. 2009+t 년에 발견된 종의 수를 _Y_ _t_ 로 표기합니다(따라서 Y1\=64 는 2010년의 개수입니다). JAGS를 사용하여 다음 모델을 적합시키세요.
Yt|α,β∼indepPoisson(λt)
여기서 λt\=exp(α+βt) 이고 α,β∼indep Normal(0,102) 입니다. _α_ 와 _β_ 의 사후 분포를 요약하고 MCMC 샘플러가 수렴했는지 검증하세요. 이 분석 결과는 종의 발견율이 시간에 따라 변하고 있다는 증거를 제공합니까?
13. 이전 문제에 대해 여러분 자신만의 메트로폴리스 샘플러(즉 JAGS가 아닌)를 작성하세요. 주석이 달린 코드를 제출하고, 각 파라미터에 대한 후보 분포 및 대응되는 수락 비율을 보고하며 체인들이 수렴했음을 보여주기 위해 각 파라미터에 대한 트레이스 플롯을 사용하세요.
14. 한 임상 시험에서 100명의 환자를 각각 위약(placebo), 신약 저용량, 신약 고용량 그룹(총 300명의 환자)에 배정했습니다. 데이터는 아래 표에 주어져 있습니다.
| 치료(Treatment) | 긍정적 결과(Positive outcome) | 부정적 결과(Negative outcome) |
| --------- | ---------------- | ---------------- |
| 위약(Placebo)   | 52               | 48               |
| 저용량(Low dose)  | 60               | 40               |
| 고용량(High dose) | 54               | 46               |
119\. 각 치료 하에서 환자가 긍정적인 결과를 가질 확률에 대하여 균등 사전 분포(uniform priors)를 부여하여 JAGS를 이용한 베이지안 분석을 수행하세요.
   1. 각 치료 그룹에 대한 긍정적 결과 확률의 사후 평균과 95% 구간을 보고하세요.
   2. 세 가지 치료 옵션 중 저용량이 최고일 사후 확률을 계산하세요.
15. 정규 혼합 모델(normal mixture model)을 고려해 봅니다.
Yi|θ∼iidf(y|θ)\=12\[ϕ(y−θ)+ϕ(y)\],
i\=1,...,n 에 대하여 여기서 θ∈R 이고 ϕ(z)\=exp{−z2/2}/2π 는 표준 정규 분포의 밀도를 나타냅니다. 우리는 이 우도에 대해서는 무정보적 사전 분포를 사용할 수 없음을 보였습니다([섹션 2.4](./10-chapter2.md#sec2_4), 연습 문제 14번). 이 문제에서 우리는 모호하지만 정상적인(vague but proper) 사전 분포의 사용을 살펴봅니다. 다음 R 코드로 시뮬레이션된 데이터를 분석하세요.
`>   set.seed(27695)`
`>   theta_true <- 4`
`>   n          <- 30`
`>   B          <- rbinom(n,1,0.5)`
`>   Y          <- rnorm(n,B*theta_true,1)`
   1. 위 R 코드가 f(y|θ) 로부터 표본들을 생성함을 주장(Argue)하세요.
   2. 시뮬레이션된 데이터를 플롯팅하고, θ\={2,4,6} 에 대해 y∈\[−3,10\] 구간 내 f(y|θ) 를 별도로 그려보세요.
   3. 사전 분포 θ∼Normal(0,102) 를 가정하여, 다음 R 코드를 사용하여 _θ_ 의 MAP 추정치와 근사적인 사후 표준 편차(asymptotic posterior standard deviation)를 구하세요.
   `> nlp <- function(theta,Y){`
   `>    like <- 0.5*dnorm(Y,0,1)+`
   `>            0.5*dnorm(Y,theta,1)`
   `>    prior <- dnorm(theta,0,10)`
   `>    neg_log_post <- -sum(log(like))-sum(log(prior))`
   `> return(neg_log_post)}`
   ``
   ``
   `> map     <- optim(mean(Y),nlp,Y=Y,hessian=TRUE)`
   `> map_est <- map$par`
   `> map_sd  <- 1/sqrt(map$hessian)`
   4. 사전 분포가 θ∼ N(0,10k) 라고 가정합시다. k∈{0,1,2,3} 에 대해 _θ_ 의 사후 밀도를 플롯팅하고 이 사후 밀도들을 (c)에서 도출한 점근적 정규 분포와 (모든 5개의 밀도 곡선을 하나의 플롯에 겹쳐서 그림으로써) 비교해 보세요.
   5. 혼합 표현법 Yi|Bi,θ∼Normal(Biθ,1) 을 통하여 JAGS를 이용해 이 모델을 적합시키세요. 여기서 Bi∼iidBernoulli(0.5) 이고 θ∼Normal(0,102) 입니다. _θ_ 의 사후 분포를 (d) 파트의 결과와 비교하세요.
16. Y1,…,Yn 을 다음의 밀도 함수를 지닌 이동된 지수 분포(shifted exponential distribution)로부터의 무작위 표본이라고 합시다.
f(y|α,β)\={βexp−β(y−α)y≥α0y<α
여기서 α\>0 및 β\>0 이 주요 관심 파라미터입니다. 사전 분포 α∼Uniform(0,c) 와 β∼Gamma(a,b) 를 가정합니다.
   1. 120\. α\=2, β\=3 이고 y∈\[0,5\] 일 때의 f(y|α,β) 를 플롯팅하고, _α_ 와 _β_ 의 해석을 제공하며 이 모델이 적절하게 들어맞을 법한 실제 실험을 기술하세요.
   2. _α_ 와 _β_ 의 완전 조건부 분포를 제공하고 이들이 공통된 확률분포 족의 구성원인지 여부를 판단하세요.
   3. 초기값, 각 샘플링 단계의 상세 내용, 수렴성을 평가할 방법에 대한 계획 등을 모두 포괄하여 MCMC 샘플러를 위한 의사(pseudo) 코드를 작성하세요.
17. [리스트 3.9](./11-chapter3.md#list3_9)에 제공된 데이터(mass 및 age)를 사용하여 다음 비선형 회귀 모델을 적합시키세요.
massi∼Normal(μi,σ2) 이고 여기서 μi\=θ1+θ2ageiθ3,
사전 분포는 θ1∼Normal(0,1002), θ2∼Uniform(0,20000), θ3∼Normal(0,1), σ2∼InvGamma(0.01,0.01) 입니다.
   1. JAGS 내에서 이 모델을 적합시키고 데이터(age 대비 mass)와 _μ_ _i_ 의 사후 평균을 함께 플롯팅하여 해당 모델이 합리적으로 잘 들어맞는지 확인하세요.
   2. 수렴성에 대한 철저한 조사를 수행하세요.
   3. 수렴성을 향상시키기 위해 취할 수 있는 세 가지 조치를 제시하세요.
18. (과도하게 복잡한) 모델 Y|n,p∼Binomial(n,p) 와 사전 분포 n∼Poisson(λ) 및 p∼Beta(a,b) 를 가정합니다. 관측된 데이터는 Y\=10 입니다.
   1. 왜 이 모델의 수렴 속도가 느릴 수 있는지 설명하세요.
   2. λ\=10 과 a\=b\=1 조건에서 모델을 JAGS에 적합시키세요. 수렴성을 확인하고 _n, p_ 및 θ\=np 의 사후 분포를 요약하세요.
   3. a\=b\=10 으로 설정한 것을 제외하고 (b)의 분석을 반복하세요. 파라미터 _p_ 의 사전 분포가 세 가지 파라미터 전반의 수렴성에 어떤 효과를 미치는지 논평하세요.
19. (과도하게 복잡한) 모델을 고려해 봅니다.
Yi|μi,σi2∼Normal(μi,σi2)
여기서 μi∼iidNormal(0,θ1−1), σi2∼InvGamma(θ2,θ3) 이고 j\=1,2,3 에 대해 θj∼Gamma(ϵ,ϵ) 입니다. i\=1,...,n 에 대해 데이터가 Yi\=i 라고 가정하세요.
   1. 왜 이 모델의 수렴 속도가 느릴 수 있는지 설명하세요.
   2. n∈{5,25} 와 ϵ∈{0.1,10} 의 네 가지 모든 조합에 대해 모델을 JAGS에 적합시키고 이 네 차례의 적합 모두에 대해 _θ_1, _θ_2 및 _θ_3 의 유효 표본 크기를 보고하세요.
   3. _n_ 및 _ϵ_ 이 수렴성에 미치는 영향을 평가하세요.

