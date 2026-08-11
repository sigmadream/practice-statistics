211\.

# 7 계층적 모델링을 활용한 사례 연구

## 목차 (Contents)

- [7.1 계층적 모델링의 개요 (Overview of hierarchical modeling)](./15-chapter7.md#sec7_1)
  - [7.1.1 수학적/통계적 모델 (Mathematical/statistical models)](./15-chapter7.md#sec7_1_1)
  - [7.1.2 상태 공간 모델 (State-space models)](./15-chapter7.md#sec7_1_2)
  - [7.1.3 다수준 통계 모델 (Multilevel statistical models)](./15-chapter7.md#sec7_1_3)
  - [7.1.4 결측 및 중도절단 데이터를 위한 계층적 모델 (Hierarchical models for missing and censored data)](./15-chapter7.md#sec7_1_4)
- [7.2 사례 연구 (Case studies)](./15-chapter7.md#sec7_2)
  - [7.2.1 데이터 융합을 통한 종 분포 매핑 (Species distribution mapping via data fusion)](./15-chapter7.md#sec7_2_1)
  - [7.2.2 티라노사우루스과 성장 곡선 (Tyrannosaurid growth curves)](./15-chapter7.md#sec7_2_2)
  - [7.2.3 기후 재구성 (Climate reconstruction)](./15-chapter7.md#sec7_2_3)
- [7.3 연습문제 (Exercises)](./15-chapter7.md#sec7_3)

## 7.1 계층적 모델링의 개요

지금까지 우리는 표준 통계 모델의 맥락에서 베이즈 아이디어를 소개했습니다. 그러나 베이즈 방법론의 주요 이점 중 하나는 결측값, 중도절단(censored) 데이터, 오차를 포함하여 측정된 변수, 각각 뚜렷한 편향과 오차를 가진 여러 데이터 소스, 다른 특성을 가진 하위 모집단 등과 같은 불규칙성을 가진 비표준 사례를 처리할 수 있는 유연성입니다. 이러한 모든 기능을 분석에 통합하는 것은 막막해 보일 수 있지만, 종종 큰 모델을 관리 가능한 레이어(층)로 분할하고 이 레이어들을 계층적 모델(hierarchical model)로 결합함으로써 달성할 수 있습니다. 따라서 계층적 모델링(다수준 모델링(multilevel modeling)이라고도 함)은 필수적인 모델 구축 도구입니다.

복잡한 데이터에 대한 모델을 구축하는 것이 계층적으로 생각함으로써 어떻게 단순화될 수 있는지 알아보기 위해, 우리의 목표가 세 변수 _X, Y_ 및 *Z*의 결합 분포(joint distribution)를 지정하는 것이라고 가정해 봅시다. 다변량 결합 분포를 직접 피팅하는 것은 어려울 수 있으며 특히 세 변수가 서로 다른 지지대(support)를 갖는 경우 더욱 그렇습니다. 그러나 임의의 삼변량 분포(trivariate distribution)는 다음과 같이 작성할 수 있습니다.

f(x,y,z)\=f(x)f(y|x)f(z|x,y).(7.1)

세 변수의 순서를 정하고 *X*의 일변량 주변 분포(univariate marginal distribution)를 지정한 다음 Y|X 및 Z|X,Y에 대한 일변량 조건부 분포를 지정함으로써, 다변량 문제는 세 개의 일변량 문제로 축소됩니다. 변수들이 순서가 지정되어 있고 각 조건부 분포는 순서상 이전 변수들에만 의존하기 때문에 결과적인 결합 분포는 유효함이 보장됩니다. 또한 모든 다변량 분포는 이런 방식으로 분해될 수 있으므로, 이 접근 방식을 취함으로써 유연성의 손실은 없습니다.

212\. 이 삼변량 모델은 [그림 7.1a](./15-chapter7.md#fig7_1)에 방향성 비순환 그래프(DAG, Directed Acyclic Graph)로 표시되어 있습니다. DAG(베이즈 네트워크(Bayesian network)라고도 함)는 각 관측치와 매개변수를 노드(node)(즉, 그래프를 정의하는 점)로 하고, 조건부 의존성을 나타내는 엣지(edge)(즉, 노드 간의 연결)를 갖는 그래프로 모델을 나타냅니다. 유효한 확률 모델(stochastic model)을 정의하려면 그래프가 방향성이 있고 순환하지 않아야 합니다(directed and acyclic). 방향성 그래프는 각 엣지에 방향을 부여합니다. *X*에서 *Y*로 가는 화살표는 조건부 분포 *Y*를 *X*의 함수로 모델링하여 계층적 모델이 정의됨을 나타냅니다. [그림 7.1b](./15-chapter7.md#fig7_1)에서 *X*에서 *Z*로 가는 화살표가 없다는 것은 *Y*가 주어졌을 때 *Z*가 *X*에 의존하지 않는다는 선택을 전달합니다. 즉, f(z|x,y)\=f(z|y)입니다. 반대로, [그림 7.1a](./15-chapter7.md#fig7_1)는 *Y*가 주어졌을 때 *X*와 _Z_ 사이에 조건부 의존성이 있는 모델의 DAG입니다. 그래프는 또한 비순환적(acyclic)이어야 하며, 이는 그래프를 통해 노드에서 방향성 엣지를 따라가서 원래 노드로 돌아오는 것이 불가능하다는 것을 의미합니다. 이 두 조건은 p(x,y,z)\=p(x|y,z)p(y|z)p(z|y)와 같은 모델을 구축하는 것을 배제하며, 이는 유효한 결합 분포가 아닐 수 있습니다.

![Two causal diagrams compare relationships among X Y and Z. In panel A, X points to both Y and Z, and Y also points to Z, forming a converging structure with multiple arrows into Z. In panel B, X leads to Y which leads to Z in a simple chain, showing a single directional flow from X through Y to Z.](./images/fig7_1.jpg)

그림 7.1 **방향성 비순환 그래프 (DAGs, Directed acyclic graphs)**. 패널 (a)는 f(X,Y,Z)\=f(X)f(Y|X)f(Z|X,Y) 모델의 DAG를 보여주고 패널 (b)는 f(X,Y,Z)\=f(X)f(Y|X)f(Z|Y) 모델의 DAG를 보여줍니다. [Return to text.⏎](chapter7)

계층적 모델은 여러 형태를 취할 수 있지만, 모델을 구축하는 일반적인 방법은 데이터 레이어(data layer), 프로세스 레이어(process layer) 및 사전 확률 레이어(prior layer)를 생성하는 것입니다. 모델 구축은 관심 있는 근본적인 과학적 프로세스와 알려지지 않은 매개변수를 포함하는 *프로세스 레이어*에서 시작해야 합니다. 이 레이어를 구축하는 것은 도메인 전문가와 협의하여 이상적으로 수행됩니다. 이 레이어가 정의되면, 특정 매개변수를 추정하거나 특정 가설을 테스트하는 등의 통계적 목표를 명확히 할 수 있습니다. 이상적으로 이러한 목표는 분석을 위해 수집할 데이터를 결정합니다. *데이터 레이어*는 (가능도 함수(likelihood function)를 통해) 데이터를 프로세스와 연관시키고 데이터 수집 절차의 편향과 오차를 인코딩하며, 이는 데이터가 어떻게 수집되었는지에 대한 지식을 필요로 합니다. 마지막으로 *사전 확률 레이어*는 분석 시작 시 모델 매개변수에 대한 불확실성을 정량화합니다.

모델을 계층적으로 구축하는 것은 편리하지만, 우리가 이전에 고려했던 모델들과 근본적으로 다르지는 않습니다. 사실 우리는 [섹션 4.4](./12-chapter4.md#sec4_4)의 임의 효과 모델(random effects models)과 같은 많은 계층적 모델을 이미 접했습니다. 이것은 [3장](./11-chapter3.md)에서 설명한 사후 분포를 근사화하기 위한 계산 방법들이 계층적 모델에도 적용되며, 사후 분포를 요약하고 적합도를 확인하는 데 사용되는 그래픽 및 수치적 방법들도 마찬가지라는 것을 의미합니다. 아래에서는 계층적 모델의 여러 범주와 각 경우의 베이즈 방법 적용에 대해 논의합니다. 그런 다음 7.2장의 몇 가지 사례 연구를 통해 이러한 개념을 설명합니다.

### 7.1.1 213\. 수학적/통계적 모델

많은 과학 분야에서, 관심 프로세스는 일련의 미분 방정식(differential equations)으로 수학적으로 기술됩니다. 미분 방정식 모델은 시스템의 역학(dynamics)을 결정하기 위해 모든 변수의 현재 상태와 물리적 또는 생물학적 법칙을 사용합니다. 수학적 모델은 다음을 포함하여 많은 응용 분야에서 기본이 됩니다.

- 대서양을 가로지르는 허리케인의 경로,
- 주어진 설계 하에서 교량의 강도,
- 암세포의 형성 및 확산,
- 멸종 위기 종의 개체군 역학.

이상적으로는 수학적 모델이 시스템의 합리적인 표현이지만 이산화(discretization) 또는 단순화로 인해 불완전하며, 시스템의 초기 및 현재 상태와 모델 매개변수가 정확히 알려져 있지 않기 때문에 불확실성이 존재한다고 가정할 수 있습니다.

산불의 확산을 모델링하는 것을 생각해 보십시오 \[[49](./19-ref01.md#refbib49)\]. 수학적/미분 방정식 모델은 위상, 지형, 기상학 등의 함수로 화재의 분산을 기술할 것입니다. 모델의 예측에 오차가 있을 수 있는 데에는 많은 이유가 있습니다. 방정식이 시스템의 역학을 완벽하게 설명하더라도(이는 가능성이 낮지만), 실제로는 계산 상의 한계나 위성 이미지와 같은 모델 입력의 공간 해상도 때문에 방정식들을 이산화해야 할 것입니다. 또한 이 방정식들은 화재의 위치와 강도, 주변 식생의 상태, 현재 풍속 및 풍향을 포함한 시스템의 현재 상태에 조건부(contingent)입니다. 모델은 고지대 소나무 우점림으로의 확산 가능성과 같은 알려지지 않은 매개변수에도 의존합니다. 현재 상태도 모델 매개변수도 정확히 알려질 수는 없습니다.

모델 편향, 미지수 및 오차를 설명하는 통계 분석을 불확실성 정량화(UQ, Uncertainty Quantification; \[[143](./19-ref01.md#refbib143)\])라고 합니다. UQ 분석은 수학적 모델을 실제 데이터로 보정하여 예측을 개선하고, 이해관계자에게 불확실성을 전달하여 의사 결정을 개선하며, 모델 개선을 위한 미래 데이터 수집을 최적화하여 효율성을 높이고, 모든 불확실성 소스를 고려하면서 미래 시나리오 간의 차이를 공식적으로 테스트할 수 있습니다. 베이즈 접근법 \[[90](./19-ref01.md#refbib90)\]은 사용 가능할 때 사전 지식을 원활하게 통합할 수 있고 여러 불확실성 소스를 설명할 수 있기 때문에 이러한 작업에 자연스럽습니다.

도입부로서, 모델 구축 과정을 설명하기 위해 아래의 간단한 질병 진행 모델(disease progression model)을 고려해 보십시오. _S_ *t*와 _I_ *t*를 시간 *t*에서 모집단의 감수성 있는(susceptible) 개인과 감염된(infected) 개인의 수라고 합시다. 질병에 대한 과학적 이해는 질병 전파를 모델링하는 데 사용됩니다. 시간에 따른 _S_ *t*와 _I_ *t*의 변화는 미분 방정식으로 기술될 수 있습니다 \[[23](./19-ref01.md#refbib23)\]. 우리의 데이터는 일일 집계로 이산화되기 때문에 우리는 차분 방정식(difference equation)을 고려합니다. SIER 모델을 포함하여 많은 구획 모델(compartmental models)이 제안되었지만, 역학자(epidemiologist)와의 협의를 통해 다음과 같은 간단한 Reed–Frost 모델 \[[3](./19-ref01.md#refbib3)\]을 선택할 수 있습니다.

Process layer:It+1|St,It∼Binomial\[St,1−(1−q)It\]St+1\=St−It+1

여기서 모든 감염된 개인은 다음 시간 단계 전에 모집단에서 제거되며, *q*는 비감염자가 감염자와 접촉하여 질병에 걸릴 확률이라고 가정합니다.

역학적 프로세스 레이어 모델은 몇 가지 알려지지 않은 상태 변수 *I_1 및 \_S_1과 알려지지 않은 매개변수 \_q*까지 질병 역학을 표현합니다. 214\. 이러한 미지수를 추정하기 위해, _Y_ *t*로 표시되는 시간 *t*에서의 사례 수가 수집됩니다. 데이터 레이어는 프로세스 _I_ *t*를 측정하는 능력을 모델링합니다. 예를 들어, 도메인 전문가와 데이터 수집 절차에 대해 논의한 후, 우리는 위양성(false positives)(감염되지 않은 사람이 감염된 것으로 계산됨)은 없지만 잠재적으로 위음성(false negatives)(계산되지 않은 감염자)이 있다고 가정할 수 있으며, 따라서 다음과 같습니다.

Data layer: Yt|It∼Binomial(It,p)(7.2)

여기서 *p*는 감염된 개인을 검출할 확률입니다. 베이즈 모델은 알려지지 않은 상태 변수 및 매개변수에 대한 사전 확률을 사용하여 완성됩니다.

Prior layer:I1∼Poisson(λ1),S1∼Poisson (λ2)p,q∼beta(a,b).

[그림 7.2](./15-chapter7.md#fig7_2)는 이 모델에 해당하는 DAG를 도식화합니다.

![A hierarchical diagram shows three layers. The data layer contains Y two Y three and Y four, each receiving an arrow from corresponding process states I two S two I three S three and I four S four, which are linked sequentially. The process layer receives arrows from I one S one p and q, which sits below. The prior layer contains parameters lambda one lambda two a and b, with an arrow feeding into the process layer, indicating prior inputs to the system.](./images/fig7_2.jpg)

그림 7.2 Reed–Frost 감염병 모델에 대한 **방향성 비순환 그래프**. [Return to text.⏎](chapter7)

Reed-Frost 모델은 매우 단순하여 시스템 역학의 완벽한 표현을 제공할 가능성은 낮습니다. 진단 결과 모델 적합성에 의문이 생기면, 위에서 언급한 SIER 모델과 같이 더 정교한 모델을 선택하는 것이 해결책입니다. 그러나 선택한 모델이 실제 역학에 완벽하게 충실할 것이라고는 거의 확신할 수 없습니다. 따라서 계층적 모델에서 이러한 편향을 허용해야 합니다. 불일치(discrepancy) 항은 데이터 또는 프로세스 레이어에 들어갈 수 있습니다. 여기서는 데이터 레이어에 불일치 항 dt\>0을 추가합니다.

Data layer with discrepancy: Yt|It,dt∼Binomial(Itdt,p).(7.3)

이 매개변수화에서 dt\=1.1은 시간 *t*에서 실제로 감염된 사람의 수가 모델의 현재 상태보다 10% 더 많음을 의미합니다. 물론 불일치 항이 _t_ 전체에서 자유롭게 변할 수 있게 하면 데이터의 모든 신호를 흡수하여 실제 감염자 수를 추정할 수 없게 됩니다. 식별 가능성(identifiability) 문제는 215\. log(dt)\=∑k\=0Kαktk를 *t*의 저차 다항식 함수로 취하고 베이즈 계층적 모델에서 불일치 매개변수 _α_ *k*를 추정함으로써 줄일 수 있습니다.

Reed-Frost 모델은 간단하며 이 데이터에 대한 베이즈 방법을 사용한 분석은 표준 MCMC 방법을 활용할 수 있습니다. 그러나 일련의 미분 방정식의 해로 정의되는 더 복잡한 수학적 모델은 모델을 한 번 평가하는 데만 몇 시간이 걸릴 수 있기 때문에 계산 상의 어려움을 초래합니다. [섹션 8.6.5](./16-chapter8.md#sec8_6_5)에서는 계산을 개선하고 복잡한 수학적 모델의 베이즈 분석을 용이하게 하기 위해 기계 학습 도구를 사용하는 방법을 보여줍니다. 이 섹션에서는 도메인 지식을 기반으로 모델을 지정하는 대신 데이터에서 프로세스를 지배하는 방정식을 학습하는 베이즈 기계 학습 도구에 대해서도 논의합니다.

### 7.1.2 상태 공간 모델

상태 공간 모델(SSMs, State-space models)은 시간에 따른 프로세스의 진화를 연구하고 단기 예측을 하는 데 사용되는 계층적 모델의 한 부류입니다. 프로세스 레이어는 데이터 레이어의 관측치와 관련된 잠재(latent)(관측되지 않은) 상태의 진화를 정의합니다. 아래에서는 동적 선형 모델(DLM, Dynamic Linear Model)과 은닉 마르코프 모델(HMM, Hidden Markov Model)이라는 두 가지 SSM 예제에 대해 논의합니다. 두 모델 모두 시간이 이산화되어 있어서 t∈{1,…,n}인 시간에만 프로세스가 정의되며, 상태 분포가 이전 값이 주어졌을 때 기록(history)과 독립적인 마르코프 프로세스(Markov process)를 잠재 공간 변수가 따른다고 가정하여 설명됩니다. 연속적인 시간과 더 풍부한 시간 역학으로의 확장이 가능하지만 더 복잡합니다. 잠재 상태의 성질이 두 모델을 구분하는데, DLM은 이를 연속 확률 변수로 모델링하는 반면 HMM 모델은 잠재 상태를 이산 확률 변수로 모델링합니다.

**동적 선형 모델 (Dynamic linear models)**: DLM \[[163](./19-ref01.md#refbib163)\]에서, 시간 *t*에서의 *q*개 잠재 상태인 Xt\=(Xt1,...,Xtq)는 다음 선형 모델을 따라 진화합니다.

Xt\=AXt−1+ut

여기서 **A** (q×q)는 잠재 상태의 진화를 결정하고 ut∼iid Normal(0,Σu)입니다. 잠재 변수는 *p*개의 응답 변수 Yt\=(Yt1,...,Ytp)와 다음과 같이 관련됩니다.

Yt\=BXt+vt

여기서 **B** (p×q)는 잠재 상태와 결과를 연결하고 vt∼iid Normal(0,Σv)입니다.

구체적인 예로서, 응답이 *q*개의 잠재적 경제 요인에 의해 주도된다고 생각되는 *p*개의 주가라고 가정합시다. 또 다른 예로, 결과가 환자의 활력 징후(심박수, 혈압 등)이고 잠재 요인이 심장 기능의 측면이라고 가정해 봅시다. 두 경우 모두 응답이 시간에 걸쳐, 그리고 서로 상관되어 있으며 이러한 상관 관계의 대부분은 잠재 요인의 변화로 설명될 수 있다고 가정합니다. 잠재 변수를 사용하여 결과 간의 의존성을 모델링하는 이점은 소수의 잠재 요인(q<p)으로 의존성을 설명할 수 있다면 얻게 되는 효율성 이득입니다.

데이터가 시간 *n*까지 사용 가능한 경우, MCMC를 사용하여 사후 예측 분포(PPD, posterior prediction distribution) Yn+1에서 샘플링하는 것은 간단합니다. 반복 *s*의 값을 윗첨자로 표시하겠습니다(Xt(s)). 그런 다음 PPD의 샘플은 Xt+1(s)∼Normal(A(s)Xt(s),Σu(s))으로 생성된 다음 Yt+1(s)∼Normal(B(s)Xt+1(s),Σv(s))으로 생성됩니다. 샘플 Yt+1(1),...,Yt+1(S)의 분포는 매개변수(**A**, **B** 등)뿐만 아니라 잠재 상태의 불확실성도 설명합니다. MCMC는 단기 예측을 위한 간단한 방법을 제공하지만, 샘플링에 모든 *n*개의 훈련 관측치에 대해 잠재 프로세스 Xt를 업데이트해야 하므로 느릴 수 있습니다. 칼만 필터(Kalman filter) \[[32](./19-ref01.md#refbib32)\]를 포함하여 더 효율적인 방법을 사용할 수 있습니다.

216\. 라벨 전환(Label switching) \[[83](./19-ref01.md#refbib83)\]은 SSM(DLM 및 HMM 모두)을 피팅하고 해석할 때 어려움을 야기합니다. 잠재 요인을 구별할 사전 지식이 없는 한, 모델의 라벨을 변경해도 피팅에 영향을 미치지 않습니다. 예를 들어 q\=2이고 Xt\=(Xt1,Xt2)가 Xt\=(Xt2,Xt1)로 변경되고 **A**, **B** 및 Σu의 해당 행이 스왑되면 사후 분포는 변경되지 않습니다. 라벨 전환 문제를 해결하는 한 가지 방법은 알려진 인위적 제약을 삽입하여 잠재 상태를 식별하도록 보장하는 것입니다. 아래의 카지노 예제에서는 두 잠재 상태 중 하나의 분포가 알려져 있으며 이 정보가 라벨 전환을 해결합니다. 잠재 상태에 대해 알려진 것이 없으면 **A**, **B** 또는 Σu의 요소를 고정할 수 있습니다. 일반적인 제약 조건은 **A**가 양의 대각 요소가 있는 하삼각행렬(lower-triangular)이라는 것입니다. 또 다른 과제는 잠재 상태의 수를 선택하는 것입니다. DIC, WAIC 또는 교차 검증과 같은 모델 적합도 기준을 사용할 수 있습니다.

**은닉 마르코프 모델 (Hidden Markov models):** HMM \[[174](./19-ref01.md#refbib174)\]은 잠재 프로세스 Xt∈{1,...,q}가 오직 _q_ 레벨만 있는 이산적이라는 주요 차이점을 제외하면 DLM과 유사한 구조와 해석을 갖습니다. 이는 잠재 상태가 다양한 행동 모드를 나타낸다는 해석을 더합니다. 예를 들어, Yt가 시간 *t*에서 동물의 속도와 방향인 동물 이동 모델링에서 잠재 상태는 q\=3 레벨을 가질 수 있습니다. 즉, Xt\=1은 동물이 쉬고 있음을 나타내고 Xt\=2는 사냥 중임을 나타내며 Xt\=3은 이동 중임을 나타냅니다. 분명히 응답의 분포는 현재 상태에 따라 다를 것입니다. HMM 분석의 목표에는 잠재 상태를 식별하고 각 상태의 시간적 프로세스를 이해하는 것이 포함됩니다.

잠재 상태 _X_ *t*는 마르코프 프로세스라고 가정합니다. 즉, 모든 이전 상태 X1,...,Xt−1이 주어졌을 때 _X_ *t*는 오직 Xt−1에만 의존합니다. 이 가정 하에서 그것의 분포는 다음의 요소를 갖는 q×q 전이 확률 행렬(transition probability matrix)에 의해 완전히 결정됩니다.

Pij\=Prob(Xt\=j|Xt−1\=i).

이전 상태가 Xt−1\=i로 주어지면, 현재 상태의 확률 질량 함수(PMF)는 *q*개의 확률 Pi\=(Pi1,...,Piq)로 주어집니다. 확률 벡터에 대한 공액 사전 확률(conjugate prior)은 디리클레(Dirichlet) 사전 확률([섹션 2.1](./10-chapter2.md#sec2_1)) Pi∼Dirichlet(ai1,...,aiq)입니다. 대안적으로 전이 확률에 영향을 미칠 것으로 생각되는 외부 공변량(covariates)이 있는 경우 다항 로지스틱 회귀 모델([섹션 4.3.2](./12-chapter4.md#sec4_3_2))을 사용하여 Pi를 모델링할 수 있습니다.

응답 분포의 매개변수는 현재 상태에 의해 결정됩니다. 데이터가 실수 선에 지지대를 갖는 경우 가장 간단한 모델은 다음과 같습니다.

Yt|Xt\=k∼indepNormal(μk,Σ)

여기서 μk\=(μk1,...,μkp)T이고 Σ는 p×p 공분산 행렬입니다. 이 모델은 상태 특정 공분산, 상태 특정 회귀 계수를 갖는 평균의 공변량, 시간이 지남에 따른 응답의 의존성(체제 전환 모델(regime-switching model)이라고 함) 또는 비가우시안 데이터에 대한 상태 특정 계수를 갖는 일반화 선형 모델(generalized linear model)을 포함하여 여러 가지 방식으로 확장될 수 있습니다. 잠재 상태의 수가 적은 경우 상태 특정 매개변수에 대해 무정보(uninformative) 사전 확률을 사용하는 것이 합리적입니다. 잠재 상태의 수가 많은 경우 이를 임의 효과(random effects)로 처리할 수도 있습니다.

**부정직한 카지노 예제 (Dishonest casino example):** "부정직한 카지노" 데이터는 R 패키지 aphid \[[164](./19-ref01.md#refbib164)\]에서 가져왔습니다. 데이터는 n\=300번의 주사위 굴리기로 구성됩니다. 가정은 q\=2개의 잠재 상태가 있다는 것입니다. 하나는 6개 결과 모두에 대해 동일한 확률을 갖는 공정한 주사위이고, 다른 하나는 불균등한(그리고 알려지지 않은) 확률을 갖는 조작된 주사위입니다. 딜러는 게임 중 여러 시점에 조작된 주사위로 전환하며, 목표는 특정 시간에 조작된 주사위가 사용되고 있을 사후 확률을 계산하는 것입니다. 데이터는 [그림 7.3](./15-chapter7.md#fig7_3) (상단)에 도식화되어 있습니다. 6개 결과의 샘플 비율은 0.13, 0.14, 0.16, 0.12, 0.14 및 0.30이므로 공정한 주사위에서 예상되는 것보다 6의 비율이 훨씬 더 높습니다. 217\.

![A sequence of three hundred die rolls is plotted as points aligned with outcomes 1 through 6, showing scattered but occasionally frequent repeats. Below, a line plot shows the posterior probability the die is rigged given the observed results, fluctuating between 0 and 1 with several peaks near rolls exhibiting unusual streaks or irregularities.](./images/fig7_3.jpg)

그림 7.3 **"부정직한 카지노"를 위한 은닉 마르코프 모델**. 주사위 각 굴리기 결과(1-6) 도식화 및 (하단) 조작된 상태에 있을 사후 확률. 조작된 상태에 있을 확률이 최소 0.5인 굴리기 번호에 대해 결과가 꽉 찬 원으로 도식화됩니다. [Return to text.⏎](chapter7)

Yt∈{1,...,6}를 굴리기 횟수 t∈{1,...,n}의 결과라고 합시다. HMM은 시간 *t*에서 잠재 상태를 통해 정의되며, 공정한 주사위가 작동 중이면 Xt\=1이고 조작된 주사위가 작동 중이면 Xt\=2 값을 취합니다. _Y_ *t*의 분포는 공정한 주사위 하에서는 Prob(Yt\=y|Xt\=1)\=16이고 조작된 주사위 하에서는 Prob(Yt\=y|Xt\=2)\=py입니다. 잠재 상태의 분포는 Pij\=Prob(Xt\=j|Xt−1\=i)를 갖는 전이 확률 행렬에 의해 결정됩니다. 조작된 주사위 확률 (p1,...,p6)∼Dirichlet(1/6,...,1/6)과 전이 확률 행렬의 행 (P11,P12)∼Dirichlet(1/2,1/2) 및 (P21,P22)∼Dirichlet(1/2,1/2)에 무정보 디리클레 사전 확률이 주어집니다. 두 개의 상태만 있고 그 중 하나(공정한 주사위)는 알려진 데이터 분포를 가지므로 모델은 위에서 논의한 라벨 전환 문제를 겪지 않습니다.

Listing 7.1 부정직한 카지노 데이터의 HMM 분석을 위한 JAGS 모델 구문 (p2는 1/2 요소를 갖는 벡터; p6는 1/6 요소를 갖는 벡터입니다).

```jags
1 # Distribution of the data given the latent state
2  for(i in 1:n){
3    Y[i] ~ dcat(die_probs[state[i],1:6])
4  }
5
6 # Prior for the latent state
7  state[1] ~ dcat(p2[1:2])
8  for(i in 2:n){
9    state[i] ~ dcat(trans_probs[state[i-1],1:2])
10  }
11
12  #Prior distributions
13  die_probs[1,1:6] <- p6[1:6] # Probs for fair die
14  die_probs[2,1:6] ~ ddirch(p6[1:6]) # Probs for loaded die
15
16  trans_probs[1,1:2] ~ ddirch(p2[1:2]) # Probs from fair state
17  trans_probs[2,1:2] ~ ddirch(p2[1:2]) # Probs from rigged state
```

조작된 상태에서 6을 굴릴 확률의 사후 평균(표준 편차)은 0.80(0.13)이고 다른 5개 결과에 대해서는 0.10 미만이므로 조작된 주사위는 6에 가중치가 있는 것으로 보입니다. 시간 *t*에서 조작된 상태에 있을 사후 확률 Prob(Xt\=2|Y1,...,Yn)은 [그림 7.3](./15-chapter7.md#fig7_3) (하단)에 도식화되어 있습니다. 예를 들어 굴리기 횟수 60 부근과 같이 6의 비율이 높아서 조작된 주사위가 사용되고 있을 가능성이 높은 구간이 일부 존재합니다. 전이 확률 행렬 요소들의 사후 평균(표준 편차)은 다음과 같습니다.

(0.89 (0.05)0.11 (0.05)0.44 (0.18)0.56 (0.18)).

따라서 현재 상태가 공정하다고 주어졌을 때 다음 번 굴리기에서 조작된 주사위로 전환될 확률은 0.11로 추정되고, 현재 상태가 조작되었다고 주어졌을 때 공정한 주사위로 다시 전환될 확률은 0.44입니다. 요약하자면, 분석 결과 게임에서 짧은 간격 동안 사용되는 6에 가중치가 부여된 두 번째 주사위가 있음이 밝혀졌습니다. 218\.

### 7.1.3 다수준 통계 모델

많은 응용 분야에서 관심 시스템의 물리적 모델을 사용할 수 없으며 선형 추세 및 상관 관계와 같은 간단한 관계를 기반으로 구축된 순수한 통계 모델로 대체됩니다. 계층적 모델링은 풍부한 통계 모델을 구축하는 데에도 유용합니다. 데이터, 프로세스 및 사전 확률 레이어를 통해 계층적 모델을 구축한다는 일반적인 개념은 여전히 유효하지만, 프로세스 레이어에 대해 물리적 관점 대신 통계적 관점을 갖습니다. 예를 들어, 위의 전염병 분석에서 Reed-Frost 프로세스를 대체하는 순수한 통계 모델은 It+1x|St,It∼Binomial(St,qt)이고 logit(qt)\=β0+β1It입니다. 이 프로세스 레이어는 Reed-Frost 모델의 물리적 해석이 부족하지만 실제로는 잘 작동할 수 있는 유효한 통계 모델입니다.

다수준 모델을 구축하기 위한 일반적인 아이디어는 데이터를 동질적인 그룹으로 분할한 다음 사전 분포를 통해 그룹 전체에서 정보를 통합하는 것입니다. [섹션 4.4](./12-chapter4.md#sec4_4)에서 논의된 턱뼈 밀도 데이터([그림 4.8](./12-chapter4.md#fig4_8)에 표시됨)에 대한 임의 기울기 모델(random slopes model)이 이러한 방식으로 구축된 계층적 모델의 예입니다. 모델은 다음과 같습니다.

Data layer:Yij|βi∼Normal(Xjβi,σ2)Process layer:βi|μ,Σ∼Normal(μ,Σ)Prior layer:μ∼Normal (0,c2I2),Σ∼ InvWishart(ν,Ω)

여기서 i\=1,...,n\=20이고 j\=1,...,m\=4입니다. 20명의 환자 각각은 고유한 단순 선형 회귀 모델을 가지며, 이러한 회귀들은 환자 모집단의 개인 전체에 걸친 회귀 계수 분포를 지정하는 프로세스 레이어에 결합됩니다.

이 모델은 [그림 7.4](./15-chapter7.md#fig7_4)에서 DAG로 시각화됩니다. DAG는 정보가 계층적 모델을 통과하는 방식을 보여줍니다. 예를 들어 환자 1의 데이터가 환자 2의 다음 방문 시 뼈 밀도를 예측하는 데 어떻게 도움이 될까요? DAG에서 Y1에서 Y2로 이동하려면 모집단 매개변수 μ 및 Σ를 통과해야 합니다. 즉, Y1은 모델에게 β1에 대한 정보를 제공하여 β2 및 결국 Y2를 위한 모델에 들어가는 임의 효과 분포를 형성합니다. 만약 환자 2의 데이터만 있었다면, 우리는 아마도 β2에 대한 무정보 사전 확률에 의존했을 것이고 환자 2에 대한 관찰 횟수가 적어 사후 확률이 불안정했을 것입니다. 219\. 그러나 계층적 모델을 사용하면 결과를 안정화하기 위해 여러 환자의 힘을 빌릴 수 있습니다.

![A hierarchical model diagram is shown with three layers. The data layer contains grouped observations Y one one through Y one four Y two one through Y two four up to Y n one through Y n four. Each group is linked upward from corresponding beta parameters beta one through beta n in the process layer. These beta parameters all receive arrows from shared hyperparameters mu and Sigma. The prior layer below contains c nu and Omega, which feed into mu and Sigma as prior inputs.](./images/fig7_4.jpg)

그림 7.4 **방향성 비순환 그래프**. i\=1,...,n 및 j\=1,...,4에 대해 임의 효과 분포 βi|μ,Σ∼Normal(μ,Σ) 및 사전 확률 μ∼Normal(0,c2I2) 및 Σ∼InvWishart(ν,Ω)를 갖는 임의 기울기 모델 Yij|βi∼Normal(Xj βi,σ2)의 시각적 표현. [Return to text.⏎](chapter7)

MCMC는 계층적 모델을 맞추는 데 자연스러운 선택입니다. 계층적 모델이 간단한 조건부 분포를 계층화하여 복잡성을 구축하는 것처럼, MCMC는 간단한 전체 조건부 분포(full conditional distributions)에서 순차적으로 매개변수를 업데이트하여 복잡한 사후 분포에서 샘플링합니다. 사실, 계층적 모델을 DAG로 표시하는 것은 모델을 이해하는 데 도움이 될 뿐만 아니라, 매개변수에 대한 전체 조건부 분포가 DAG에서 매개변수의 노드로 들어오거나 나가는 화살표가 있는 항에만 의존하기 때문에 MCMC 샘플러를 코딩하는 데도 도움이 됩니다. [그림 7.4](./15-chapter7.md#fig7_4)에서 β2에 대한 전체 조건부 분포가 오직 Y21,...,Y2m|β2에 대한 데이터 레이어 항과 β2|μ,Σ에 대한 프로세스 레이어 항에만 의존한다는 것이 분명합니다. 모델을 이러한 용어만을 통해서 본다면 β2의 전체 조건부 분포가 정확히 표준 베이즈 선형 회귀([섹션 4.2](./12-chapter4.md#sec4_2))의 회귀 계수의 전체 조건부 분포와 같다는 것이 즉시 명확해집니다.

**언제 레이어 추가를 중단해야 할까요?** 이 섹션의 계층적 모델에는 데이터, 프로세스 및 사전 확률의 세 가지 수준이 있습니다. 그러나 사전 확률 레이어를 정의하는 값을 정확히 알 수 없을 가능성이 높으므로 이 불확실성을 설명하기 위해 네 번째(그리고 다섯 번째 등) 레이어를 추가하고 싶어집니다. 일반적인 경험 법칙은 매개변수를 추정할 복제본(replication)이 없을 때 레이어 추가를 중단하는 것입니다. 예를 들어, [그림 7.4](./15-chapter7.md#fig7_4)의 DAG를 참조하면, 임의 효과 평균 μ 및 공분산 Σ를 추정하는 레이어를 추가하는 것은 합리적입니다. 왜냐하면 이러한 매개변수를 추정하는 데 활용할 수 있는 반복되는 임의 효과 β1,...,βn이 있기 때문입니다. 그러나 임의 효과 공분산 Σ의 사전 평균 Ω를 추정하기 위해 추가 레이어를 추가하는 것은 모델에 Σ가 하나뿐이고 우리가 Σ를 정확히 안다고 해도 단일 샘플에서 그것의 분포를 추정할 수 없기 때문에 비합리적일 것입니다.

### 7.1.4 220\. 결측 및 중도절단 데이터를 위한 계층적 모델

결측 데이터(Missing data)와 중도절단(censoring)은 많은 분석에서 상황을 복잡하게 만드는 요인입니다. 베이즈 분석은 결측 또는 중도절단된 변수를 알려지지 않은 매개변수와 동일하게 처리함으로써 이러한 불규칙성을 원활하게 처리합니다. 매개변수와 마찬가지로 결측값에 대한 사전 분포를 할당해야 합니다. 결측 또는 중도절단된 응답의 경우 그 분포가 관측된 응답과 동일하므로 데이터가 무작위로 결측/중도절단되었다는 가정을 제외하고는 추가 가정이 필요하지 않습니다. 선형 회귀의 공변량과 같이 결측 또는 중도절단된 공변량 값을 처리하려면 결측값에 대한 새로운 사전 분포가 필요합니다. 왜냐하면 표준 분석에서는 공변량에 대해 분포 가정이 필요하지 않기 때문입니다. 합리적인 분포를 식별할 수 있고 모델링 가정이 충족된다고 가정하면, 결측 데이터가 있는 베이즈 분석은 평소와 같이 진행되며 매개변수의 사후 분포나 관심 있는 예측을 요약할 때 결측값의 불확실성을 고려할 수 있다는 이점이 있습니다.

**결측 데이터:** 결측 데이터 문제에 대한 가장 쉬운 해결책은 결측 데이터가 있는 관측치를 단순히 버리고 완전한 케이스 선형 회귀를 진행하는 것입니다. 그러나 이는 데이터 세트의 크기를 극적으로 줄일 수 있고 부분적으로 관찰된 응답에서 유용한 정보를 버릴 수 있습니다. 또 다른 간단한 접근법은 다른 변수를 공변량으로 사용하여 회귀를 사용해 결측값을 대치(impute)하는 것입니다. 단일 대치(single imputation)의 단점은 결측 관측치에 대한 불확실성을 고려하지 않으므로 결과적인 사후 추론이 의심스럽다는 것입니다. 다중 대치(Multiple imputation) 기술도 사용할 수 있으며 \[[134](./19-ref01.md#refbib134)\] (종종 베이즈 아이디어를 사용하여 동기가 부여됩니다).

계층적 모델을 사용한 베이즈 분석은 결측 데이터를 처리하는 자연스러운 방법입니다. 베이즈 접근법은 결측 데이터를 알려지지 않은 매개변수와 같은 방식으로 처리합니다. 우리는 계층적 베이즈 모델에서 무작위 변수(random variables)로 처리함으로써 이들에 대한 불확실성을 나타냅니다. 알려지지 않은 매개변수와 마찬가지로 알려지지 않은 결측 공변량에 대한 사후 추론은 사전 분포를 할당해야 합니다. 예를 들어, 4.2장의 다중 선형 회귀 모델을 고려해 봅시다.

Yi|Xi∼Normal(Xiβ,σ2) where Xi∼Normal(μ,Σ)(7.4)

여기서 μ는 길이가 *p*인 평균 벡터이고 Σ는 공변량의 p×p 공분산 행렬입니다. 주제-전문가 사전 정보가 없을 때, 하이퍼파라미터 μ와 Σ는 임의 효과 모델과 유사하게 베이즈 분석의 일부로 사전 확률이 부여되고 추정됩니다. 이런 식으로, 완전한 케이스는 (μ 및 Σ를 통해) 관측치 전체의 공변량 분포에 대해 모델에 알리고 이 정보는 결측값을 대치하는 데 사용됩니다.

물론, 이 접근 방식은 공변량 분포에 대한 합리적인 모델을 필요로 합니다. 이것은 *p*가 크거나 공변량이 비가우시안일 때 어렵습니다. 예를 들어 공변량이 연속 변수와 이진 변수의 혼합인 경우 결합 분포를 충분히 캡처하는 것이 어렵습니다. 간단한 접근법은 독립적인 사전 확률로 *p*개의 공변량을 모델링하는 것입니다. 이는 비효율적이지만 적어도 많은 상황에서 합리적인 근사치를 제공합니다. [8장](./16-chapter8.md)에 요약된 방법과 같은 더 정교한 모델링은 분석의 이 측면을 개선할 수 있습니다. 중요한 (그리고 흔히 검증할 수 없는) 가정은 결측 데이터에 체계적인 편향이 없다는 것입니다. 즉, 데이터가 완전히 무작위로 결측되었다는 것입니다. Xij가 결측될 확률이 _Y_ *i*에 의존하는 경우 Xij가 무작위로 결측되었다고 가정하면 _β_ *j*에 편향된 추정치가 나올 가능성이 높습니다.

공변량 모델이 정확하다고 가정할 때, 계층적 베이즈 분석은 결측값의 불확실성을 적절하게 고려합니다. 깁스 샘플러는 각 매개변수(β, _σ_ 등)를 업데이트한 다음 결측 관측치(_Y_ _i_, Xij)를 순환하고 모델 매개변수와 정확히 동일하게 취급하여 전체 조건부 분포에서 업데이트합니다. 따라서 회귀 계수 β의 각 샘플은 공변량의 전체 집합을 사용하여 업데이트되지만 대치된 공변량은 221\. 사후 분포를 따라 반복(iteration)마다 다릅니다. 따라서 결측값은 효과적으로 성가신 매개변수(nuisance parameters)로 취급되며 분석을 통해 결측값의 사후 예측 분포가 생성될 뿐만 아니라 더 중요한 것은 결측 관측치의 불확실성에 대한 한계적인(marginally) 회귀 계수의 사후 분포가 원하는 대로 생성된다는 것입니다.

**결측 데이터가 있는 마라톤 예제:** [섹션 2.1.7](./10-chapter2.md#sec2_1_7)의 2016년 보스턴 마라톤 데이터 분석에서 우리는 상위 여성 주자들의 처음 25마일의 속도의 함수로 마지막 마일(26마일)의 속도(마일당 분)를 예측하기 위한 선형 회귀 모델을 구축합니다. _Y_ *i*를 주자 i\=1,...,n\=149의 26마일 속도라고 하고 Xij를 주자 *i*의 j\=1,...,p\=25 마일의 속도라고 합시다. 속도 측정값의 약 8%가 결측되었습니다. 결측 관측치의 수는 주자에 걸쳐 0에서 19 사이이며 결측 관측치의 비율은 마일에 걸쳐 0%에서 55%(6마일 및 7마일) 사이입니다([그림 7.5a](./15-chapter7.md#fig7_5)). 하나라도 결측된 Xij를 가진 관측치를 버리면 샘플 크기가 149명의 주자에서 58명으로 줄어들 것입니다. 결측값의 대부분이 6마일과 7마일에 대한 것이고 이러한 공변량은 아마도 응답의 중요한 예측 변수가 아닐 것이라는 점을 감안할 때, 분석에서 이 모든 관측치를 버리는 것은 낭비일 것입니다.

![A missing data grid shows which miles lack observations for each runner. Imputed speeds for two runners plot observed points with boxplots at missing miles. Boxplots of posterior beta across miles compare the full data set and complete cases, with both sets centred near zero but complete cases showing wider intervals and greater variability, especially in later miles.](./images/fig7_5.jpg)

그림 7.5 **2016 보스턴 마라톤 데이터의 결측 데이터 분석**. 패널 (a)는 주자 (_i_) 및 마일 (_j_) 별로 결측(검정) 및 결측되지 않은(흰색) Xij를 보여줍니다. 패널 (b)는 두 주자에 대한 관측된 공변량(Xij)(점) 및 결측 공변량의 사후 분포(상자 수염 그림)를 표시합니다. 패널 (c)와 (d)는 결측 데이터 모델과 완전한 케이스 분석 각각에 대해 각 회귀 계수 _β_ *j*의 사후 분포를 도식화합니다. [Return to text.⏎](chapter7)

[섹션 2.1.7](./10-chapter2.md#sec2_1_7)에서 우리는 역 위샤트(inverse Wishart) 모델을 사용하여 공변량의 공분산(Σ)을 모델링했습니다. 이 모델은 마일 사이에 어떠한 구조도 가정하지 않았지만, 공분산 행렬의 사후 평균은 후속 마일이 높은 상관관계를 갖는다는 것을 보여주었습니다. 따라서 여기서는 1차 자기회귀(autoregressive) 시계열 모델 Xi1∼Normal(0,σ12)과 j≥1에 대해 Xij+1|Xij∼Normal(ρXij,σ22)을 사용하여 표준화된(평균이 0이고 분산이 1이 되도록) 공변량을 모델링합니다. 이 모델에 대한 JAGS 코드는 [목록 2](./15-chapter7.md#list7_2)에 나와 있습니다. 모든 하이퍼파라미터에는 무정보 사전 확률이 있습니다.

Listing 7.2 마라톤 데이터의 결측 데이터 분석을 위한 JAGS 모델 구문. [Return to text.⏎](chapter7)

```jags
1 # Likelihood
2  for(i in 1:n){
3    Y[i] ~ dnorm(alpha + inprod(X[i,],beta[]),taue)
4  }
5
6 # Missing-data model
7  for(i in 1:n){
8    X[i,1] ~ dnorm(0,tau1)
9    for(j in 2:p){
10       X[i,j] ~ dnorm(rho*X[i,j-1],tau2)
11    }
12  }
13
14 # Priors
15  alpha ~ dnorm(0,0.01)
16  for(j in 1:p){
17    beta[j] ~ dnorm(0,0.01)
18  }
19  taue ~ dgamma(0.1, 0.1)
20  tau1 ~ dgamma(0.1, 0.1)
21  tau2 ~ dgamma(0.1, 0.1)
22  rho ~ dnorm(0, 0.01)
```

[그림 7.5b](./15-chapter7.md#fig7_5)는 대표적인 두 주자에 대한 관찰된 공변량(점) 및 결측 공변량의 사후 분포(상자 수염 그림)를 플로팅합니다. 결측 공변량에 대한 시계열 모델 때문에, 결측된 Xij의 사후 분포는 두 주자 모두 인접 마일의 속도에 가깝습니다. 222\. 이 페이지의 그림. 223\. 공변량 _β_ *j*의 사후 분포는 [그림 7.5c](./15-chapter7.md#fig7_5)에 도식화되어 있습니다. 마일 24와 25만이 마지막 마일 속도의 유용한 예측 변수인 것으로 나타났습니다. 이 결측 데이터 분석에서의 사후 분산은 [그림 7.5d](./15-chapter7.md#fig7_5)에서 n\=58인 완전한 케이스 분석에서의 사후 분산보다 훨씬 작으며, 이는 결측 데이터 모델의 이점을 보여줍니다.

**공변량 측정 오차:** 변수 내 오차 모델(Error-in-variables models)(측정 오차 모델(measurement-error models)이라고도 함 \[[66](./19-ref01.md#refbib66)\])은 누락되지는 않았지만 오차와 함께 측정된 공변량을 처리하는 데 사용됩니다. 예를 들어 통제된 실험에서 사람의 혈액에서 측정한 환경 오염물질에 오차 표준 편차 *τ*가 있다고 가정해 보겠습니다. 그런 다음 _Y_ *i*로 측정된 사람의 건강 상태를 오염 물질 노출 수준 _X_ *i*에 회귀시키면 공변량 측정 오차가 발생합니다.

공변량 측정 오차의 분산이 알려져 있고 오차의 분포를 가정할 수 있는 경우(이것은 강력한 가정이지만) 베이즈 계층적 모델이 이 오차를 해결할 수 있습니다. 결합 모델은 다음과 같습니다.

Yi|β,Xi∗∼Normal(β0+β1Xi∗,σ2) and Xi|Xi∗∼Normal(Xi∗,τ2),

여기서 Xi∗는 공변량의 실제 값이며 사전 확률 Xi∗∼Normal(μ,σX2)을 갖습니다. MCMC에서 실제 값은 결측 데이터처럼 처리됩니다. 즉, 각 반복마다 각 관측치에 대해 대치되며 다른 모든 매개변수(β 및 \_σ_2)는 정규 선형 회귀에서와 같이 업데이트됩니다.

**중도절단(censored) 데이터:** 중도절단된 관측치는 부분적으로만 알려져 있고 정확하게는 알려져 있지 않습니다. 일반적으로 중도절단된 관측치는 특정 구간 내에서 발생한 것으로 알려져 있습니다(Yi∈Ii\=(Li,Ui)). 구간 중도절단의 특별한 경우로는 다음이 있습니다. Ii\=(−∞,Ui)이면 왼쪽-중도절단(left-censored); Ii\=(Li,∞)이면 오른쪽-중도절단(right-censored); Ii\=(−∞,∞)이면 결측(missing); Li\=Ui이면 중도절단되지 않음(uncensored)입니다. 중도절단된 데이터의 표준 예는 생존 분석(survival analysis)(\[[82](./19-ref01.md#refbib82)\])으로, 여기서 _Y_ *i*는 환자 *i*가 연구에 들어온 때부터 이벤트가 발생(질병 발생)할 때까지의 시간을 나타냅니다. 오른쪽-중도절단은 환자가 추적을 포기하거나(lost to follow-up) 시험 완료 후에만 이벤트가 발생하는 경우 기인합니다. 이 경우 시험 중 이벤트를 겪은 환자는 중도절단되지 않으며 Li\=Ui\=Yi이고, _L_ *i*까지 생존한 것으로 알려져 있지만 이후에 추적을 실패한 환자는 Ui\=∞인 오른쪽-중도절단 상태입니다.

오른쪽-중도절단된 데이터에 대한 편리한 모델은 공액 사전 확률 λ∼Gamma(a,b)를 갖는 Yi|λ∼iid Exponential(λ)입니다. 관측치 *i*가 가능도에 기여하는 바는 관측치가 중도절단되지 않은 경우 PDF λexp(−λYi)이고, _L_ *i*에서 오른쪽-중도절단된 경우 생존 시간 Prob(Yi\>Li|λ)\=exp(−λiLi)입니다. 관측치 *i*가 중도절단된 경우 δi\=0 및 Ti\=Li로 표시하고 그렇지 않은 경우 δi\=1 및 Ti\=Yi로 표시하면 다음이 성립합니다.

λ|δ1,...,δn,T1,...,Tn∼Gamma(∑i\=1nδi+a,∑i\=1nTi+b).

이 모델 하에서의 추론은 MCMC 샘플링을 필요로 하지 않습니다. 그러나 지수 모델(exponential model)은 일반적으로 너무 제한적이며, 공변량 추가(log(λi)\=Xiβ)는 공액성(conjugacy)을 깹니다. 따라서 더 유연한 모델이 필요합니다.

공변량을 포함하는 매력적인 모델은 로그 이벤트 발생 시간 Zi\=log(Yi)를 공변량 Zi\=Xiβ+εi에 회귀시키는 가속 실패 시간 모델(accelerated failure time model) \[[25](./19-ref01.md#refbib25)\]입니다. 이 모델에서 공변량 *j*가 1씩 증가하면 이벤트 발생까지의 중앙값(또는 다른 변위치)에 exp(βj)가 곱해집니다. 중도절단이 없고 εi∼iid Normal(0,σ2)이면, 이는 4.2장의 표준 다중 회귀 모델로 환원됩니다. 또한 β와 \_σ_2에 대해 조건부 공액 사전 확률을 선택하면 깁스 샘플링으로 이어집니다.

그러나 _Z_ *i*가 구간 \[Li,Ui\]로 중도절단되면 공액성은 깨집니다. _ε_ *i*에 대한 정규성을 가정하면, _Z_ *i*는 절단된 정규 분포(truncated normal distribution) Zi∼TruncatedNormal(μi,σ2,Li,Ui)를 따르며, μi\=Xiβ이고 밀도 함수는 다음과 같습니다.

1σ2πexp\[−(Zi−μi)22σ2\]1Φ\[(Ui−μi)/σ\]−Φ\[(Li−μi)/σ\](./7.5)

224\. 여기서 Φ는 표준 정규 CDF입니다. β가 정규 CDF 내에 포함되므로( _μ_ *i*를 통해) 사후 분포는 공액이 아닙니다.

MCMC 내에서 베이즈 대치(Bayesian imputation)를 사용하여 중도절단된 값 _Z_ *i*를 처리함으로써 공액성이 복원됩니다([목록 3](./15-chapter7.md#list7_3)에서 간단한 예제 참조). 매 반복마다 중도절단된 값 _Z_ *i*는 절단된 정규 분포 Zi∼TruncatedNormal(μi,σ2,Li,Ui)에서 도출되며, 이 분포는 \[Li,Ui\]에 대한 지지대를 가지므로 중도절단 정보를 유지합니다. 대치 후 응답은 완벽해지고, 일반적인 다중 선형 회귀에서와 같은 깁스 샘플링 방법을 사용하여 회귀 계수와 분산의 업데이트를 수행할 수 있습니다. 결측 데이터와 마찬가지로, 매 MCMC 반복마다 중도절단된 관측치의 새로운 샘플이 생성되므로 이러한 베이즈 대치 접근법은 (7.5)와 같은 항으로 구성된 가능도 함수를 사용하여 모델을 피팅하는 것과 동일합니다. 또한 베이즈 다중 대치는 중도절단된 값에 대한 불확실성이 회귀 매개변수의 사후 분포에 포함되도록 보장합니다. 이 대치 방법은 매우 일반적이며 정규성 및 선형성에 대한 가정을 완화하기 위해 [8장](./16-chapter8.md)의 유연한 회귀 방법에 적용될 수 있습니다.

Listing 7.3 중도절단이 포함된 깁스 샘플러의 스케치. [Return to text.⏎](chapter7)

`1 _#   The model is Z[i] ˜ Normal(mu,sigmaˆ2)._`
`2 _#   Observations with delta[i]=1 are observed_`
`3 _#   Observations with delta[i]=0 are censored to (L[i],U[i])_`
`4 _#   The improper prior for mu is pi(mu) = 1_`
`5 _#   Prior for sigmaˆ2 ˜ InvGamma(a,b)_`
`6 `
`7 _# Lung cancer survival data from the survival package_`
`8 `
`9   Z     <- log(lung$time) _# Final visit time_`
`10  n     <- length(Z)`
`11  delta <- lung$status=1 _# status is 1 for censored, 2 for died_`
`12  L     <- Z _# Data are right censored on (L[i],U[i])_`
`13  U     <- rep(Inf,n)`
`14 `
`15 _# Initial values_`
`16 `
`17  mu       <- mean(Z[delta==1])`
`18  s2       <- var(Z[delta==1])`
`19  keep mu  <- rep(0,10000) _# Store the samples_`
`20  a        <- b <- 0.1 _# Prior for s2_`
`21 `
`22 _# Gibbs sampling_`
`23  library(truncnorm)`
`24 `
`25  for(s in 1:10000){`
`26 `
`27     _# Impute censored values_`
`28      for(i in 1:n){if(delta[i]==0){`
`29         Z[i] <- rtruncnorm(1,L[i],U[i],mu,sqrt(s2))`
`30      }}`
`31 `
`32     _# Update parameters using the ‘‘complete’’ data_`
`33      mu <- rnorm(1,mean(Z),sqrt(s2/n))`
`34      s2 <- 1/rgamma(1,a+n/2,b+sum((Z=mu)ˆ2)/2)`
`35 `
`36     _# Store the results_`
`37      keep mu[s] <- mu`
`38 }`

## 7.2 사례 연구 (Case studies)

이 장의 나머지 부분은 계층적 모델링의 사례 연구(case studies) 시퀀스로 구성되어 있습니다. 세 가지 사례 연구는 각각 다른 과제를 제기합니다.

1. **데이터 융합을 통한 종 분포 매핑 (Species distribution mapping via data fusion)**: 편향과 불확실성을 설명하면서 여러 데이터 스트림의 정보를 결합합니다.
2. **티라노사우루스과 성장 곡선 (Tyrannosaurid growth curves)**: 하위 모집단(종) 전체에서 정보를 모으고 관측 횟수가 적은 비선형 모델에서 불확실성을 정량화합니다.
3. **대리 데이터를 사용한 기후 재구성 (Climate reconstruction using proxy data)**: 대리 데이터(proxy data)의 편향과 불확실성을 고려하여 최근 과거의 온도 분포를 재구성합니다.

이러한 분석에서 우리는 계층적 모델링의 유연성을 입증하고, 모델 및 사전 확률 지정, 모델 비교, 결과 발표를 포함한 완전한 베이즈 분석을 설명합니다.

### 7.2.1 데이터 융합을 통한 종 분포 매핑

이 사례 연구의 데이터는 \[[119](./19-ref01.md#refbib119)\]에서 가져왔습니다. 목적은 미국 남동부에 서식하는 작은 명금류인 갈색머리동고비(BHNU; _Sitta pusilla_)의 공간 분포를 매핑하는 것입니다. 강점이 서로 다른 두 가지 데이터 소스가 있습니다. 첫 번째 데이터 소스는 번식조류조사(BBS, Breeding Birds Survey)입니다. BBS는 1966년부터 활동해 온 수천 명의 자원봉사자가 조사하는 수백 개의 경로 네트워크입니다 \[[138](./19-ref01.md#refbib138)\]. 데이터는 훈련된 자원봉사자에 의해 체계적으로 수집되며, 변화를 모니터링하기 위해 사이트를 매년 방문합니다. 그러나 이 엄청난 샘플링 노력에도 불구하고 BBS 커버리지에는 공간적, 시간적 간격이 있습니다. 최근 부상하는 연구 방향은 매년 수천 명의 시민 과학자들로부터 수집된 수백만 개의 데이터 포인트로 구성된 Cornell Lab of Ornithology의 eBird 데이터베이스 \[[150](./19-ref01.md#refbib150)\]와 같은 방대한 시민 과학 데이터로 체계적인 조사 데이터를 보완하는 것입니다. 이 데이터는 훈련된 조류 관찰자가 수집한 것은 아니지만 공간적, 시간적 범위가 훨씬 더 넓습니다.

225\. 226\. 이 분석을 위해 미국 남동부를 n\=741개의 0.25×0.25도 위도/경도(lat/lon) 셀로 나누고, 2012년의 데이터를 분석합니다. N1i를 셀 *i*에서의 BBS 샘플링 횟수라고 하고 Y1i∈{0,1,...,N1i}를 BHNU를 목격한 횟수라고 합시다. 많은 셀이 BBS 경로를 갖지 않으므로 N1i\=Yi1\=0입니다. 유사하게, 셀 *i*에 대해 N2i를 eBird 시민 과학자가 기록한 시간 수로 정의하고 Y2i를 BHNU eBird 목격 횟수로 정의합니다. [그림 7.6](./15-chapter7.md#fig7_6)은 데이터를 매핑합니다. BBS 샘플링 노력은 상당히 균일한 반면 eBird 노력은 인구 밀집 지역에 더 집중되어 있습니다. 두 지도 모두 앨라배마, 조지아 및 캐롤라이나에서 더 많은 BHNU 목격을 보여줍니다.

![Multiple regional maps display bird survey information. Panels show B B S sampling occasions and sightings, Ebird square root effort and sightings, posterior mean abundance, and the probability occupancy exceeds 0.01. Shading intensities vary across states from Mississippi to Virginia, revealing spatial differences in survey effort, observed sightings, estimated abundance, and occupancy probability, with the final map highlighting high occupancy likelihood concentrated in central and eastern areas.](./images/fig7_6.jpg)

그림 7.6 **2012년 갈색머리동고비 데이터**. 패널 (a)와 (b)는 BBS 샘플링 횟수(N1i)와 BBS 목격 횟수(Y1i)를 도식화합니다. 패널 (c)와 (d)는 eBird 노력의 제곱근(N2i)과 eBird 목격 횟수의 제곱근(Y2i)을 도식화합니다. 패널 (e)는 사후 평균 풍부도(abundance) _λ_ *i*를 도식화하고, 패널 (f)는 점유 확률(occupancy probability)이 0.01을 초과할 사후 확률, 즉 Prob\[1−exp(−λi)\>0.01|Y\]를 도식화합니다. [Return to text.⏎](chapter7)

셀 *i*에서 관심 있는 진정한 프로세스는 풍부도(abundance) λi≥0이며, 조사 노력 한 단위 동안 지역에 존재하는 예상 새 수로 정의됩니다. BHNU가 서식하지 않는 셀의 경우 λi\=0입니다. 데이터 레이어는 프로세스를 관측 데이터와 연관시키며 두 데이터 소스의 장점을 신중하게 고려해야 합니다. 먼저 우리는 _λ_ *i*가 주어졌을 때 BBS와 eBird 데이터 세트가 독립적이라는 가정을 합니다. 대부분의 eBird 사용자가 BBS 업데이트를 따르지 않기 때문에 이는 합리적인 것 같습니다. BBS 데이터는 전문 조류 관찰자가 수집하므로 위양성이나 위음성이 없다고 가정합니다(비록 더 유연한 모델을 사용할 수 있고 아마도 더 선호될 것이지만, 예를 들어 \[[119](./19-ref01.md#refbib119)\] 참조). 조사 중 존재하는 새의 수가 Poisson(λi)로 분포된 경우 적어도 한 마리의 새가 존재할 확률은 1−exp(−λi)이므로 BBS 데이터를 Yi1|λi∼Binomial\[N1i,1−exp(−λi)\]로 모델링합니다.

eBird 데이터의 경우 위양성 및 위음성을 허용합니다. 평균은 N2iλ\~i이고 비율은 λ\~i\=θ1λi+θ2이며, 여기서 θ1\>0은 BBS 관찰자와 eBird 관찰자 사이의 관찰 비율의 차이를 제어하고 θ2\>0은 eBird 위양성 비율입니다. 따라서 셀이 진정으로 무인이고 λi\=0인 경우 E(Y2i)\=N2iθ2가 됩니다. 과산포(over-dispersion)를 허용하기 위해 다음 모델을 적용합니다.

Y2i|λi,θ∼NegBinomial(qi,m)(7.6)

여기서 확률 qi\=m/(λ\~i+m) 및 크기(size)는 *m*입니다. 가능도에 대한 이 두 가지 기여도를 결합하면 데이터 레이어는 다음과 같습니다.

Data layer:Y1i|λi,θ1,θ2∼Binomial\[N1i,1−exp(−λi)\]Y2i|λi,θ1,θ2∼NegBinomial(qi,m).

관심 있는 잠재 프로세스는 각 셀의 풍부도 _λ_ *i*입니다. 일부 셀에는 BBS 데이터가 없기 때문에 사전 지식 없이 모든 _λ_ *i*를 모델링하기는 어렵습니다. 다른 사전 지식이 없는 경우, 인근 셀이 비슷한 풍부도를 가질 것이라고 가정할 수 있습니다. 풍부도를 유도하는 기본 요인(기후, 서식지 등)이 공간적으로 다양하다면 이는 합리적인 가정이며 로컬에서 정보를 풀링하여 풍부도를 추정할 수 있습니다. 모델 풍부도를 위해 여러 공간 모델을 사용할 수 있지만 \[[53](./19-ref01.md#refbib53)\], 여기서는 [섹션 8.3](./16-chapter8.md#sec8_3)에서와 같이 스플라인 회귀(spline regression)를 사용합니다. 로그 풍부도(_λ_ *i*가 음수가 되지 않도록 로그 변환을 사용)는 그리드 셀의 공간적 위치 si\=(s1i,s2i)의 매끄러운(smooth) 함수입니다. 이것은 2차원 함수이므로 위도와 경도 모두에 스플라인 기저 전개(spline basis expansions)를 사용합니다.

Process layer:log(λi)\=∑j\=1J∑k\=1KBj(si1)Dk(si2)βjk(7.7)

여기서 _B_ *j*는 경도의 B-스플라인 기저 함수(B-spline basis functions)이고, _D_ *k*는 위도의 B-스플라인 기저 함수이며, βjk∼iidNormal(β0,σ2)입니다(여기서는 위도 및 경도 방향의 기저 함수가 서로 다른 기저 함수 수를 가지므로 서로 다른 형태를 갖기 때문에 다른 표기법을 사용합니다). 곱 Bj(si1)Dk(si2) 중 일부는 도메인의 모든 si에 대해 0에 가깝고 무시됩니다. 공간 도메인이 위도보다 더 넓은 범위의 경도에 걸쳐 있으므로 우리는 형태의 227\. 이 페이지의 그림. 228\. Xl\=Bj(si1)Dk(si2) 항 p\=2L2개를 위해 K\=2L 및 J\=L를 취하고 DIC를 사용하여 *L*을 선택합니다. 베이즈 계층적 모델을 완성하기 위해 우리는 무정보 사전 확률을 지정합니다.

Prior layer:θ1,θ2,m,σ−2∼Gamma(0.1,0.1) and β0∼Normal(0,100).(7.8)

이 모델을 구현하기 위한 JAGS 코드는 [목록 4](./15-chapter7.md#list7_4)에 나와 있습니다.

Listing 7.4 BHNU 풍부도를 위한 공간 데이터 융합 모델. [Return to text.⏎](chapter7)

```jags
1 # Data layer
2   for(i in 1:n){
3       Y1[i]  ~ dbin(phi[i],N1[i]) # BBS
4       phi[i] <- 1-exp(-lam[i])
5
6       Y2[i] ~ dnegbin(q[i],m) # eBird
7       q[i]  <- m/(m+N2[i]*(theta1*lam[i]+theta2))
8   }
9
10 # Process layer
11   for(j in 1:p){beta[j]~dnorm(beta0,tau)}
12   for(i in 1:n){
13     log(lam[i]) <- inprod(X[i,],beta[])
14   }
15
16  # Prior layer
17   theta1      ~ dgamma(0.1,0.1)
18   theta2      ~ dgamma(0.1,0.1)
19   m           ~ dgamma(0.1,0.1)
20   tau         ~ dgamma(0.1,0.1)
21   beta0       ~ dnorm(0,1)
```

두 개의 체인(chains)을 실행하며, 각각은 10,000번의 웜업(burn-in) 반복과 50,000번의 웜업 후 샘플로 구성됩니다. 샘플을 5로 솎아내어(thinning) 사후 분포를 근사하기 위한 20,000개의 샘플을 남깁니다. _DIC_ (_p_ _D_)는 L\=4의 경우 3107 (30), L\=6의 경우 3056 (58), L\=8의 경우 3015 (89), L\=10의 경우 2999 (127), L\=12의 경우 3014 (177) 및 L\=14의 경우 3009 (209)이므로 L\=10으로 진행합니다. L\=10일 때, 유효 샘플 크기는 모든 βjk에 대해 1,000을 초과하므로 샘플러가 잘 섞이고 사후 분포를 충분히 탐색했음을 나타냅니다.

[표 7.1](./15-chapter7.md#tbl7_1)은 하이퍼파라미터의 사후 분포를 나타냅니다. 주목할 점은 eBird 위양성 비율 *θ_2가 거의 0으로 추정되어 eBird 데이터가 신뢰할 수 있는 정보 소스로 보인다는 것입니다. *λ\_ *i*의 사후 평균과 셀이 점유될(즉, 최소 하나의 개체가 존재할) 사후 확률은 각각 [그림 7.6e](./15-chapter7.md#fig7_6) 및 [7.6f](./15-chapter7.md#fig7_6)에 매핑됩니다. 예상대로 조지아와 캐롤라이나에서 추정된 풍부도가 가장 크지만 점유 확률도 서쪽 루이지애나와 아칸소에서 높습니다. eBird 데이터를 제외했다면 이 서부 주들의 점유 확률은 낮아졌을 것입니다.

**표 7.1 **BHNU 분석에 대한 사후 분포**. L\=10인 최종 피팅에 대한 사후 중앙값 및 95% 구간. [Return to text.⏎](chapter7)**
| 중앙값 (Median) | 95% 구간 (95% Interval) | |
| ------------------------------ | ------------ | -------------- |
| 배율 인자(Scaling factor), *θ_1 | 11.5 | ( 9.2, 14.4) |
| 위양성률(False positive rate), *θ*2 | 0.00 | ( 0.00, 0.00) |
| 과산포(Over-dispersion) 매개변수, \_m* | 0.45 | ( 0.37, 0.55) |
| 평균 풍부도 매개변수, *β_0 | −5.81 | (−6.69, −4.86) |
| 스플라인(Spline) 표준 편차, *σ\_ | 5.58 | ( 4.52, 7.03) |

### 7.2.2 티라노사우루스과 성장 곡선

우리는 4가지 티라노사우루스과 종(Albertosaurus, Daspletosaurus, Gorgosaurus 및 Tyrannosaurus)의 성장 곡선을 추정하기 위해 화석 20개의 데이터를 분석합니다. 데이터는 \[[48](./19-ref01.md#refbib48)\]의 표 1에서 가져왔으며 [그림 7.7](./15-chapter7.md#fig7_7)에 도식화되어 있습니다. 목적은 각 종별로 연령에 따른 예상 체중(성장 곡선)을 수립하는 것입니다. 데이터는 연령과 체중 사이에 비선형적인 229\. 관계를 보여주며 종 간에 공통점이 있습니다. 따라서 우리는 비선형 계층적 모델을 추구합니다.

![Two panels compare dinosaur growth patterns. The left plot shows body mass versus age for Albertosaurus, Daspletosaurus, Gorgosaurus and Tyrannosaurus, each species following a distinct S shaped growth curve with Tyrannosaurus reaching the largest mass above 5000 kilograms. The right plot shows log body mass versus log age, where all species follow approximately linear trends with different slopes and intercepts, indicating similar allometric growth but differing overall size trajectories.](./images/fig7_7.jpg)

그림 7.7 **티라노사우루스과 성장 곡선 데이터**. 왼쪽 패널은 4개 티라노사우루스과 종의 샘플 20개의 추정 연령 및 체중(kg) 산점도를 제공합니다. 오른쪽 패널은 두 변수를 모두 로그 변환한 후 동일한 데이터를 플로팅합니다. 왼쪽 패널에 도식화된 곡선은 \[[48](./19-ref01.md#refbib48)\]의 적합된 로지스틱 곡선(fitted logistic curves)이며, 오른쪽 패널의 선은 최소 제곱 적합(least squares fits)입니다. [Return to text.⏎](chapter7)

이 데이터에 대한 원래 분석에서는 비선형 최소 제곱(왼쪽 패널 [그림 7.7](./15-chapter7.md#fig7_7)에 표시된 피팅 곡선)을 사용했습니다. 이 적합도에 대한 불확실성을 정량화하는 것은 어렵습니다. 추정량의 표본 분포는 비선형 평균 구조로 인해 닫힌 형식(closed form)을 갖지 않으며, 거의 같은 수의 매개변수를 추정할 몇 개의 관측치만 있는 경우 대규모 샘플에 대한 정규성 근사치가 유효하지 않고 붓스트랩(bootstrap)과 같은 재표본(resampling) 기술은 표본 분포를 근사화할 충분한 데이터가 없을 수 있습니다. 아래에 나와 있듯이 MCMC에 기반한 베이즈 분석은 사후 불확실성을 완전히 정량화합니다.

종 j\=1,...,4의 샘플 *i*의 체중과 나이를 각각 Yij와 Xij라고 합시다. 우리는 다음과 같이 데이터를 모델링합니다.

Yij\=fj(Xij)ϵij,(7.9)

여기서 _f_ *j*는 종 *j*의 참된(true) 성장 곡선이고 ϵij\>0는 평균이 1인 곱셈적 오차(multiplicative error)입니다. 인구의 변동이 질량/연령과 함께 증가할 가능성이 높기 때문에 우리는 가산적 오차(additive error) 대신 곱셈적 오차를 사용합니다. 오차가 log(ϵij)∼Normal(−σj2/2,σj2)로 로그 정규 분포(log-normal)를 따른다고 가정하면 요구되는 바와 같이 E(ϵij)\=1이며, 모델은 다음과 같이 됩니다.

log(Yij)∼Normal(log\[fj(Xij)\]−σj2/2,σj2),(7.10)

E(Yij)\=fj(Xij)에서 σj2는 종 *j*의 오차 분산을 제어합니다.

[그림 7.7](./15-chapter7.md#fig7_7) (왼쪽)의 데이터는 분명히 비선형성을 보입니다. 그러나 체중과 나이 모두 로그 변환을 수행하면 그 관계가 상당히 선형적이 됩니다([그림 7.7](./15-chapter7.md#fig7_7), 오른쪽). 따라서 우리가 고려할 수 있는 모델 중 하나는 로그 선형(log-linear) 모델입니다.

log\[fj(X)\]\=aj+bjlog(X)(7.11)

여기서 _a_ *j*와 _b_ *j*는 종 *j*에 대한 인터셉트(절편)와 기울기입니다. 원래의 스케일에서 대응되는 성장 곡선은 fj(X)\=exp(aj)Xbj입니다. 예상대로 _b_ *j*가 양수라면 성장 곡선은 무한히 증가하며 이는 현실적이지 않을 수 있습니다. 따라서 우리는 로그 선형 모델과 로지스틱 성장 곡선 모델을 비교합니다.

fj(X)\=aj+bjexp\[dj(x−cj)\]1+exp\[dj(x−cj)\],(7.12)

여기서 x\=log(X)입니다. 이 모델에는 다음 4가지 매개변수가 있습니다.

1. _a_ *j*는 0세 때의 예상 질량입니다.
2. _b_ *j*는 예상되는 평생 동안의 질량 증가입니다. 230\.
3. log(cj)는 종이 예상 증가량의 절반에 도달하는 연령입니다.
4. dj\>0는 연령에 따른 증가율을 결정합니다.

곡선의 형태는 (bj\>0라 가정할 때) 증가하고 나이가 들면서 계속 증가하는 것이 아니라 aj+bj에서 평탄해집니다(plateaus). 이것은 로그 나이로 변환하는 것을 제외하면 \[[48](./19-ref01.md#refbib48)\]에서 피팅한 함수와 동일합니다.

이러한 두 가지 형태의 성장 곡선을 비교하는 것 외에도 두 가지 사전 확률을 비교합니다. 첫 번째 사전 확률(“풀링되지 않음(unpooled)”)은 무정보 사전 확률을 사용하여 각 종을 개별적으로 피팅합니다. 로그 선형 모델의 경우 사전 확률은 aj,bj∼Normal(0,10) 및 σj2∼InvGamma(0.1,0.1)이고 로지스틱 모델의 경우 사전 확률은 log(aj),log(bj),cj,log(dj)∼Normal(0,10) 및 σj2∼InvGamma(0.1,0.1)입니다. aj,bj 및 _d_ *j*에 대해 정규 사전 확률을 로그 정규 사전 확률로 바꾸어 이들 매개변수가 양수가 되도록 보장하고 이에 따라 모든 *X*에 대해 fj(X)가 양수이고 증가하도록 합니다. 두 번째 사전 확률(“풀링됨(pooled)”)은 4개 종 전체에 걸쳐 정보를 차용하는 베이즈 계층적 모델입니다. 풀링된 분석에서는 모든 종의 분산이 동일하다고(σj2\=σ2) 가정하며 무정보 사전 확률 σ2∼InvGamma(0.1,0.1)을 취합니다. 인터셉트에 대한 로그 선형 모델의 사전 확률은 aj∼Normal(μa,σa2)이며 여기서 μa∼Normal(0,10) 및 σa2∼InvGamma(0.1,0.1)입니다. 동일한 계층적 모델이 로지스틱 모델의 log(aj), log(bj), _c_ _j_ 및 log(dj)에 적용됩니다. 이 모델의 JAGS 코드는 [목록 5](./15-chapter7.md#list7_5)에 나와 있습니다.

Listing 7.5 계층적 성장 곡선 모델링을 위한 JAGS 코드. [Return to text.⏎](chapter7)

```jags
1    # n is the total number of observations for all species
2    # x[i] is the log age of individual i
3    # y[i] is the log mass of individual i
4    # sp[i] is the species number (1, 2, 3, or 4) of individual i
5
6    # Data layer
7 for(i in 1:n){
8    y[i] ~ dnorm(muY[i],taue)
9    muY[i] <- log(a[sp[i]] + b[sp[i]]/(1+exp(-part[i]))) - 0.5/taue
10   part[i] <- (x[i]-c[sp[i]])/d[sp[i]]
11 }
12
13 # Process layer
14 for(j in 1:N){
15   a[j] <- exp(alpha[j,1])
16   b[j] <- exp(alpha[j,2])
17   c[j] <- alpha[j,3]
18   d[j] <- exp(alpha[j,4])
19
20     for(k in 1:4){alpha[j,k] ~ dnorm(mu[k],tau[k])}
21 }
22
23 # Prior layer
24 for(k in 1:4){
25   mu[k] ~ dnorm(0,0.1)
26   tau[k] ~ dgamma(0.1,0.1)
27 }
28 taue ~ dgamma(0.1,0.1)
```

이 계층적 모델은 4종의 매개변수를 임의 효과(random effects)로 취급하며 임의 효과 분포(즉, _μ_ _a_ 및 σa2)에 대해 학습하면 사전 확률을 통해 추가 정보를 제공하여 사후 분포를 안정화합니다. 이러한 매개변수가 진정한 임의 효과인지, 즉 이 연구를 위해 이 4종을 무작위로 선택한 교환 가능한(exchangeable) 종들의 무한한 분포가 존재하는지는 논쟁의 여지가 있습니다. 그러나 불확실성을 줄이기 위해 종 전체의 정보를 풀링하여 임의 효과 모델을 사용하는 것은 분명히 이 네 종의 데이터 231\. 232\. 분석 결과를 (아래에 표시된 것처럼) 개선합니다.

우리는 종별로 각각 분리된(풀링되지 않은) 것과 계층적 모델을 사용한(풀링된) 로그 선형 및 로지스틱 성장 곡선 모델을 피팅합니다. 4개 피팅의 _DIC_ (_p_ _D_)는 다음과 같습니다. 로그 선형 풀링되지 않은 경우 29 (25), 로그 선형 풀링된 경우 −3 (9), 로지스틱 풀링되지 않은 경우 64 (41), 로지스틱 풀링된 경우 −2 (12). 풀링된 모델은 ( _p_ *D*로 측정한) 모델 복잡성을 줄이고 이로 인해 더 작은(더 나은) *DIC*가 나타납니다. 로그 선형 및 로지스틱 성장 곡선의 *DIC*는 비슷합니다.

[그림 7.8](./15-chapter7.md#fig7_8), [그림 7.9](./15-chapter7.md#fig7_9), [그림 7.10](./15-chapter7.md#fig7_10)은 각 모델 및 종에 대한 _f_ *j*의 사후 평균 및 95% 신뢰 구간을 점별(pointwise)로 플로팅합니다(구간 추정치는 Yij가 아닌 _f_ *j*에 대한 것이므로 관측치의 95%를 포함하지 않아야 함). 4가지 방법의 사후 평균은 상당히 비슷하며 모두 데이터에 잘 맞습니다. 분석 간의 주요 차이점은 종 전체의 정보를 차용함으로써 풀링된 분석의 신뢰 구간이 더 좁다는 것입니다. 시각적으로, [그림 7.10](./15-chapter7.md#fig7_10)의 로그 선형 적합은 성장 곡선을 충분히 모델링하는 것처럼 보입니다. 그러나 로지스틱 곡선도 거의 맞고 노년기에 접어들면서 평평해지는(plateauing) 직관적 특성을 갖고 있음을 감안할 때 평생 과정을 고려하면 이 모델이 더 바람직하다고 주장할 수 있습니다.

![Four panels show body mass versus age for Albertosaurus, Daspletosaurus, Gorgosaurus and Tyrannosaurus. Each plot displays observed data points, the posterior mean growth curve, and dashed ninety five percent intervals. All species exhibit accelerating growth with age, but Tyrannosaurus reaches the greatest mass. The posterior mean curves track the data closely, while uncertainty bands widen at larger ages due to fewer observations, especially in Daspletosaurus and Gorgosaurus.](./images/fig7_8.jpg)

그림 7.8 **피팅된 로그 선형 성장 곡선 – 풀링되지 않음(unpooled)**. 풀링되지 않은 로그 선형 모델에 대한 티라노사우루스과 성장 곡선의 사후 평균(실선) 및 95% 구간(점선) 대 관측치(점). [Return to text.⏎](chapter7)

![Four small panels show dinosaur body-mass versus age with data points and fitted growth curves. Each panel corresponds to Albertosaurus, Daspletosaurus, Gorgosaurus, or Tyrannosaurus. Points mark observed masses, solid curves show posterior mean growth, and dashed curves give ninety-five percent intervals. All species display increasing mass with age, with Tyrannosaurus showing the steepest rise and widest uncertainty, while Albertosaurus and Gorgosaurus have smoother trajectories and fewer extreme values.](./images/fig7_9.jpg)

그림 7.9 **피팅된 로그 선형 성장 곡선 – 풀링됨(pooled)**. 풀링된(계층적) 로그 선형 모델에 대한 티라노사우루스과 성장 곡선의 사후 평균(실선) 및 95% 구간(점선) 대 관측치(점). [Return to text.⏎](chapter7)

![Four panels show body mass against age for Albertosaurus, Daspletosaurus, Gorgosaurus and Tyrannosaurus. Each plot includes observed points, a posterior mean growth curve and dashed ninety five percent intervals. Albertosaurus and Gorgosaurus show modest growth with wide uncertainty. Daspletosaurus has nearly flat fitted growth with large credible bands. Tyrannosaurus displays pronounced acceleration after age ten with tighter intervals, reaching the largest predicted mass among the four species.](./images/fig7_10.jpg)

그림 7.10 **피팅된 로그 선형 성장 곡선 – 로지스틱, 풀링되지 않음(unpooled)**. 풀링되지 않은 로지스틱 모델에 대한 티라노사우루스과 성장 곡선의 사후 평균(실선) 및 95% 구간(점선) 대 관측치(점). [Return to text.⏎](chapter7)

![A set of four panels shows body mass against age for Albertosaurus, Daspletosaurus, Gorgosaurus, and Tyrannosaurus. Each panel displays observed masses as points, a smooth posterior mean growth curve, and dashed ninety-five percent intervals. All species show strong positive growth with age, with Tyrannosaurus reaching the greatest mass and showing the steepest rise.](./images/fig7_11.jpg)

그림 7.11 **피팅된 로그 선형 성장 곡선 – 로지스틱, 풀링됨(pooled)**. 풀링된(계층적) 로지스틱 모델에 대한 티라노사우루스과 성장 곡선의 사후 평균(실선) 및 95% 구간(점선) 대 관측치(점).

### 7.2.3 기후 재구성

이 분석은 호주 브리즈번의 과거 연간 강수량을 재구성하기 위해 \[[39](./19-ref01.md#refbib39)\]의 데이터와 \[[27](./19-ref01.md#refbib27)\]의 통계적 접근 방식의 단순화된 버전을 사용합니다. 이 분석은 계층적 모델에서 여러 데이터 유형을 결합하고 각 데이터 유형의 누락된 값 비율이 높기 때문에 어렵습니다. 강수 지수(RFI)는 1889년부터 2017년까지 (mm 단위로) 측정되었으며 [그림 7.12](./15-chapter7.md#fig7_12)에 도식화되어 있습니다. 1612년까지 거슬러 올라가는 얼음 코어 및 나이테와 같은 대리(proxy) 데이터가 1889년 이전의 RFI를 재구성하는 데 사용됩니다. _Y_ *t*를 t+1611년에 관측된 RFI라고 합시다 233\. 이 페이지의 그림. 234\. (그래서 \_Y_1은 1612년에 해당합니다). 자기 회귀 시계열 모델은 Y1∼Normal(μ0,σ02)이고 다음과 같습니다.

Yt|Yt−1∼Normal(μ0+ϕ0(Yt−1−μ0),(1−ϕ02)σ02),

![A time-series plot shows reconstructed rainfall from 1600 to 2000 with highly variable modern observations beginning near 1880. A second plot overlays observed rainfall from 1860 to 1950 with posterior estimates, including a median curve and a wide ninety percent interval. Observations fluctuate around the posterior median, and uncertainty expands substantially before direct measurements begin.](./images/fig7_12.jpg)

그림 7.12 **기후 재구성 강수 지수 데이터**. 왼쪽 패널은 강수 지수(RFI)의 모든 값을 표시하고, 오른쪽 패널은 1850-1950년 동안의 데이터 및 사후 중앙값 및 90% 구간을 보여줍니다. [Return to text.⏎](chapter7)

여기서 *μ_0은 평균이고, σ02는 분산이며, *ϕ*0은 시간적 종속성을 제어합니다. 평균과 분산의 형태는 모든 \_t*에 대해 가장자리 (j≠t에 대한 _Y_ _j_ 전반에 걸쳐) 평균과 분산이 E(Yt)\=μ0 및 Var(Yt)\=σ02가 되도록 되어 있습니다.

안타깝게도, 강수량은 지난 129년 동안에만 측정되었습니다. 누락된 관측치를 대치하기 위해 우리는 [그림 7.13](./15-chapter7.md#fig7_13)에 표시된 여러 개의 대리(proxy) 변수를 사용합니다. 대리 변수들도 결측값의 비율이 높습니다. 그러나 대리가 있는 연도와 RFI 측정 연도 사이에 다소 겹치는 부분이 있으며, RFI와 대리 변수들 간의 상관 관계는 대리에 따라 최고 0.40까지 나타납니다. 따라서, 강수량과 대리 변수들 간의 관계를 활용하여 RFI를 1612년까지 추산해 낼 수 있습니다. 이 작업에는 다수의 파라미터와 결측 데이터가 존재하고, 이로 인한 불확실성을 시간에 걸친 강수량의 최종 분석에 전달(propagated)할 수 있기 때문에 계층적 베이즈 모델이 이를 처리하는 자연스러운 방법입니다.

![Six time-series panels display proxy climate records from 1600 to 2000. Each plot shows fluctuating values between about negative 4 and 4. Proxies 1, 3, and 5 span the full period, while Proxies 2, 4, and 6 begin later, around the late 1800s or early 1900s. All series show noisy variability without clear long-term trends, though amplitudes differ among proxies.](./images/fig7_13.jpg)

그림 7.13 **기후 재구성 대리 데이터**. 6개의 대리 변수가 표준화 이후 도식화되었습니다. [Return to text.⏎](chapter7)

t+1611년의 대리 변수 *j*를 Xtj로 표시합니다. 한 가지 옵션은 관측된 강수량에 대한 모델에서 이 대리 변수들을 공변량으로 사용하는 것입니다. 그러나, 물리적 현실에 보다 잘 맞는 계층적 모델은 강수량을 대리 변수들의 평균 내 공변량으로 두는 것입니다. 현실적으로 강수량이 이 대리 변수들의 동인(driver)으로 여겨지기 때문입니다. 이 대리 변수들은 선형 관계를 통해 강수량과 연관되어 있습니다.

E(Xjt)\=μjt\=αj+βjYt

235\. 여기서 인터셉트 _α_ *j*와 기울기 _β_ *j*는 가산적 및 승법적(additive and multiplicative) 편향을 설명합니다. 시간적 의존성을 고려하기 위해, Xj1∼Normal(μj1,σj2)를 갖는 또 다른 자기 회귀 모델을 사용합니다.

Xjt|Xjt−1∼Normal(μjt+ϕj(Xjt−1−μjt−1),(1−ϕj2)σj2).

JAGS 코드는 [목록 6](./15-chapter7.md#list7_6)에 주어져 있습니다.

Listing 7.6 브리즈번 기후 재구성을 위한 JAGS 모델 구문. [Return to text.⏎](chapter7)

```jags
1 # RFI observation model
2 Y[1] ~ dnorm(mu0,tau0)
3 for(t in 2:n){
4   Y[t] ~ dnorm(mu0+phi0*(Y[t-1]-mu0),tauAR0)
5 }
6
7 phi0    ~ dunif(0,1)
8 mu0     ~ dnorm(0,0.001)
9 tau0    ~ dgamma(0.1, 0.1)
```

`10 tauAR0 <- tau0/(1=phi0*phi0)`
`11 `
`12 _# Proxy model_`
`13 for(j in 1:6){`
`14   mu[1,j] <- Alpha[j] + Beta[j]*Y[1]`
`15   X[1,j] ˜ dnorm(mu[1,j],tau[j])`
`16   for(t in 2:n){`
`17      mu[t,j] <- Alpha[j] + Beta[j]*Y[t]`
`18      X[t,j] ˜ dnorm(mu[t,j] + phi[j]*(X[t=1,j]=mu[t=1,j]),`
`19                       tauAR[j])`
`20   }`
`21   Alpha[j] ˜ dnorm(0, 0.1)`
`22   Beta[j]  ˜ dnorm(0, 0.1)`
`23   phi[j]   ˜ dunif(0,1)`
`24   tau[j]   ˜ dgamma(0.1, 0.1)`
`25   tauAR[j] <- tau[j]/(1=phi[j]*phi[j])`
`}`

대리 데이터들은 매우 다른 스케일에 있기 때문에, 분석 전에 평균이 0이고 분산이 1이 되도록 별도로 표준화됩니다([그림 7.13](./15-chapter7.md#fig7_13) 참조). 베이즈 모델은 무정보 사전 확률(uninformative priors)로 완성됩니다([목록 6](./15-chapter7.md#list7_6) 참조). _Y_ *t*와 Xjt 모두 결측값이 있으며, 이는 [섹션 7.1.4](./15-chapter7.md#sec7_1_4)에 설명된 베이즈 다중 대치(Bayesian multiple imputation)를 사용하여 자연스럽게 처리됩니다. 따라서, 결측 데이터가 있는 연도에 대한 결과적인 _Y_ *t*의 사후 분포는 RFI 및 대리 데이터의 결측값뿐만 아니라 매개변수의 불확실성까지 설명합니다.

결측값이 많기 때문에 긴 MCMC 체인을 실행합니다. 우리는 JAGS에서 두 개의 MCMC 체인을 피팅하며, 각 체인은 20,000번의 웜업(burn-in) 반복 후 10으로 솎아낸 100,000번의 반복으로 구성됩니다. 유효 표본 크기(effective sample size)는 매개변수 및 대치된 RFI 값에 걸쳐 845에서 2626 사이입니다. MCMC 샘플링은 일반 PC에서 약 7분 정도 걸립니다.

대리 변수 *j*와 RFI 간의 관계 강도는 _β_ *j*에 의해 제어됩니다. 6개 대리 변수에 대한 _β_ *j*의 사후 중앙값(표준 편차)은 0.00077 (0.00026), 0.00135 (0.00034), −0.00008 (0.00023), 0.00089 (0.00030), 0.00100 (0.00030), 0.00156 (0.00033)입니다. 따라서 세 번째 대리 변수를 제외한 모든 대리 변수는 0을 포함하지 않는 90% 구간을 갖습니다. 모든 사후 중앙값이 작지만, 이는 대리 데이터가 표준화된 반면 RFI는 표준화되지 않고 0에서 2000 사이의 범위를 갖기 때문입니다. [그림 7.12](./15-chapter7.md#fig7_12)의 오른쪽 패널은 연구 대상 연도 중 일부에 대한 _Y_ *t*의 피팅된 값을 보여줍니다. 예상대로 과거로 갈수록 변동성이 커지지만, 대리 데이터와의 피팅된 관계 때문에 1800년대 중반에도 연도에 따라 추정된 RFI에 상당한 차이가 존재합니다.

236\. 이 페이지의 그림. 237\. 238\. RFI의 연도별 변동이 크기 때문에 시간에 따른 변화를 연구하는 것은 특정 기간 동안의 평균 RFI를 사용하는 것이 더 신뢰할 수 있습니다. 아래에서 우리는 1612-1649년, 1650-1699년, …, 1950-1999년 및 2000-2017년 기간의 평균 RFI를 연구합니다. _P_ *k*를 기간 *k*에 포함된 _n_ _k_ 연도의 인덱스라고 할 때, 기간 *k*의 평균 강수량은 Y¯k\=∑t∈PkYt/nk입니다. Y¯k의 사후 분포는 대치된 값들을 사용하여 각 MCMC 반복에 대해 Y¯k를 계산함으로써 MCMC를 사용하여 근사화되며, 이는 Y¯k의 사후 샘플을 생성합니다.

[표 7.2](./15-chapter7.md#tbl7_2)는 각 기간별 Y¯k의 사후 분포를 요약하고, 각 Y¯k가 2000-2017년의 Y¯k보다 클 사후 확률을 제공합니다. 최근 기간에는 누락된 데이터가 없으므로 Y¯k에 불확실성이 없는 반면, 먼 과거의 기간에 대해서는 Y¯k가 대리 데이터와의 의존성에만 기반하므로 불확실성이 큽니다. 현재 기간인 2000-2017년은 이전 기간인 1950-1999년보다 더 건조합니다. 그러나 사후 평균에서 기간 간에 상당한 차이가 있으며 결측 데이터로 인해 사후 표준 편차도 큽니다. 1750-1850년에 걸친 두 역사적 기간은 확률이 거의 1에 가깝게 현재 기간보다 건조한 것으로 나타났습니다.

**표 7.2 **시간 기간별 브리즈번의 평균 RFI 사후 분포**. 시간 기간별 평균 RFI Y¯k의 사후 평균, 표준 편차 및 90% 구간과 각 기간의 평균 RFI가 현재 기간(2000-2017)의 RFI보다 높을 사후 확률. [Return to text.⏎](chapter7)**
| 기간 (Period) | 평균 (Mean) | 표준 편차 (SD) | 90% 구간 (90% interval) | Prob > present |
| --------- | ---- | --- | ------------ | -------------- |
| 1612-1649 | 809 | 213 | ( 455, 1129) | 0.42 |
| 1650-1699 | 685 | 151 | ( 399, 888) | 0.09 |
| 1700-1749 | 731 | 149 | ( 460, 946) | 0.17 |
| 1750-1799 | 533 | 173 | ( 203, 746) | 0.00 |
| 1800-1849 | 600 | 164 | ( 305, 806) | 0.02 |
| 1850-1899 | 923 | 95 | ( 759, 1074) | 0.78 |
| 1900-1949 | 860 | 0 | ( 860, 860) | |
| 1950-1999 | 938 | 0 | ( 938, 938) | |
| 2000-2017 | 851 | 0 | ( 851, 851) | |

## 7.3 239\. 연습문제 (Exercises)

1. MCMC 알고리즘의 많은 부분에서 전체 조건부 분포(full conditional distributions)가 사용되므로, 조건부 분포를 통해 모델을 지정하고 싶어질 수 있습니다. 예를 들어, 다음과 같은 조건부 분포를 고려해 보십시오.  
   Y|X∼Normal(aX,1) 및 X|Y∼Normal(bY,1).
   1. 이러한 전체 조건부 분포가 양립할 수 없도록(incompatible) *a*와 _b_ 값을 선택하십시오. 즉, 이러한 전체 조건부 분포를 제공하는 _X_ 및 *Y*에 대해 유효한 결합 분포가 존재하지 않도록 하십시오. 전체 조건부 분포가 양립할 수 없다는 주장을 주장하되 공식적으로 증명하지는 마십시오.
   2. 유효한 DAG를 생성하는 모델을 구축하면 항상 유효한 결합 분포로 이어지는 이유를 설명하십시오.
2. [섹션 7.2.3](./15-chapter7.md#sec7_2_3)의 계층적 모델에 대한 DAG를 그리십시오.
3. 이 문제에서 우리는 메타 분석(meta analysis), 즉 여러 연구의 결과를 결합하는 분석을 수행할 것입니다. 데이터는 R의 rmeta 패키지에서 가져온 것입니다.  
   `> library(rmeta)`  
   `> data(cochrane)`  
   `> cochrane`  
   `           name ev.trt n.trt ev.ctrl n.ctrl`  
   `1     Auckland      36   532      60    538`  
   `2        Block       1    69       5     61`  
   `3        Doran       4    81      11     63`  
   `4        Gamsu      14   131      20    137`  
   `5     Morrison       3    67       7     59`  
   `6 Papageorgiou       1    71       7     75`  
   `7      Tauesch       8    56      10     71`  
   데이터는 코르티코스테로이드 요법이 신생아 사망에 미치는 영향을 평가하는 7개의 무작위 시험에서 나온 것입니다. 시험 i∈{1,...,7}에 대해 Ni0 통제군 환자 중 발생한 이벤트의 수를 Yi0로 표시하고, Ni1 치료군 환자 중 발생한 이벤트의 수를 Yi1로 표시합니다.
   1. θ0,θ1∼Uniform(0,1)과 함께 Yij|θj∼indep Binomial(Nij,θj) 모델을 피팅하십시오. 치료가 이벤트 비율을 줄인다고 결론 내릴 수 있습니까?
   2. logit(θij)\=αij이고 αi\=(αi0,αi1)T∼iidNormal(μ,Σ), μ∼Normal(0,102I2) 및 Σ∼InvWishart(3,I2)인 조건에서 Yij|θij∼indepBinomial(Nij,θij) 모델을 피팅하십시오. 치료가 사망률을 줄인다는 증거를 요약하십시오.
   3. 두 모델의 장단점을 논의하십시오.
   4. 이 데이터에 대해 선호되는 모델은 어느 것입니까?
4. 코스 웹페이지에서 [섹션 6.4](./14-chapter6.md#sec6_4)의 마라톤 데이터를 다운로드하십시오. Yij를 마일 *j*에서 주자 *i*의 속도라고 합시다. 계층적 모델 Yi1∼Normal(μi,σ02)을 피팅하고  
   Yij|Yij−1∼Normal(μi+ρi(Yij−1−μi),σi2),  
   여기서 μi∼iidNormal(θ1,θ2), ρi∼iidNormal(θ3,θ4) 및 σi2∼iid InvGamma(θ5,θ6)입니다. 240\.
   1. 이 모델에 대한 DAG를 그리고 모델 내 각 매개변수에 대한 해석을 제공하십시오.
   2. θ1,...,θ6에 대한 무정보 사전 분포를 선택하십시오.
   3. 각각 25,000번 반복되는 세 개의 체인을 사용하여 JAGS에서 모델을 피팅하고, _θ_ *j*에 대한 MCMC 수렴을 철저히 평가하십시오.
   4. 데이터가 _θ_ *j*에 대해 유용한 정보를 제공합니까? 즉, 사후 분포가 사전 분포보다 더 집중되어 있습니까?
   5. (c)와 (d)에 비추어 볼 때 고려할 만한 단순화(simplification)가 있습니까? 그렇다면 전체 모델과 단순화된 모델을 어떻게 비교하시겠습니까?
5. R의 survival 패키지에서 폐(lung) 데이터를 다운로드하십시오. _Y_ *i*를 대상자 i∈{1,...,n}의 생존 시간(일)이라고 합시다. 우리는 생존 시간이 지수 분포(exponential distribution) Yi|λ∼Exponential(λ)를 따른다는 비현실적인 가정을 할 것입니다. 연구는 시간 0에서 시간 *T*까지 실행되며 환자 *i*는 시간 T−Ti에 연구에 참여하여 _T_ *i*일 동안 연구의 일부가 됩니다. 분석은 중도절단으로 인해 복잡해집니다. 즉 일부 환자는 연구 결론 시점에서 이벤트를 겪지 않았습니다. Yi<Ti인 경우 이벤트가 관찰되고, Yi\>Ti인 경우 중도절단되며 δi\=I(Yi\>Ti)는 이벤트가 중도절단되었음을 나타내는 이진 지시자(binary indicator)를 나타냅니다. δi\=0인 중도절단되지 않은 환자의 경우 가능도 함수에 대한 기여도는 지수 PDF인 λexp(−λyi)입니다. δi\=1인 중도절단된 환자의 경우 가능도 함수에 대한 기여도는 _T_ *i*일 이상 생존할 확률인 exp(−λTi)입니다.
   1. 매개변수 *λ*와 데이터 (Y1,T1,δ1),...,(Yn,Tn,δn)를 사용하여 가능도 함수를 작성하십시오.
   2. *λ*에 대한 공액 사전 분포를 도출하고 무정보 사전 분포로 이어지는 하이퍼파라터를 제안하십시오.
   3. 전체 데이터 세트를 사용하여 몇 가지 공액 사전 확률(즉, 다른 하이퍼파라미터) 하에서 *λ*의 사후 분포를 요약하십시오. 결과가 사전 확률에 민감합니까?
   4. 무정보 공액 사전 확률을 사용하여 남성과 여성에 대해 각각 별도로 데이터를 분석하고, 생존 분포가 성별에 따라 다른지 여부를 테스트하십시오.
6. 감비아(Gambia) 데이터 다운로드  
   `> library(geoR)`  
   `> data(gambia)`  
   `> ?gambia`  
   데이터는 65개 마을에 거주하는 2,035명의 어린이들로 구성되어 있습니다. 마을 v∈{1,...,65}에 대해 샘플의 어린이 수를 _n_ _v_, 말라리아 양성 판정을 받은 어린이 수를 _Y_ _v_, 말라리아 양성 판정을 받을 실제 확률을 _p_ *v*로 표시하십시오. 우리는 공간 모델 αv\=logit(pv)를 사용하며, 여기서 α\=(α1,...,α65)T는 평균 E(αv)\=μ, 분산 Var(αv)\=σ2 및 상관관계 Cor(αu,αv)\=exp(−duv/ρ)를 갖는 다변량 정규 분포를 따르며, duv는 마을 *u*와 _v_ 사이의 거리입니다. 사전 확률의 경우 μ∼Normal(0,102), σ2∼InvGamma(0.1,0.1) 및 ρ∼Uniform(0,d∗)라고 가정하며, 여기서 _d_\*는 마을 간의 최대 거리입니다.
   1. 이 계층적 모델의 데이터 레이어, 프로세스 레이어 및 사전 확률 레이어를 지정하십시오.
   2. JAGS를 사용하여 모델을 피팅하고 수렴을 평가하십시오.
   3. 241\. 다섯 개의 지도, 즉 표본 크기 _n_ _v_, 표본 비율 Yv/nv, _p_ *v*의 사후 평균, _p_ *v*의 사후 표준 편차, _p_ *v*가 0.5를 초과할 사후 확률을 사용하여 데이터와 결과를 요약하십시오. 242페이지는 비어 있습니다.
