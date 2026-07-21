<a role="toc_link" id="chapter7"></a>
171\. 

# 7 Spectral Estimation

이제 스펙트럼 밀도(Spectral density)를 추정하는 방법에 대해 논의할 준비가 되었습니다. 일반적으로 비모수적(Nonparametric) 스펙트럼 추정과 모수적(Parametric) 스펙트럼 추정이라고 불리는 두 가지 주요 방법이 있습니다. 먼저 논의할 비모수적 방법에서는 스펙트럼 밀도가 존재한다는 점 외에는 아무런 가정을 하지 않습니다. 반면 모수적 방법에서는 스펙트럼 밀도가 특정 모수적 패밀리(Parametric family)에 속한다고 가정합니다.

## 7.1 Periodogram and Discrete Fourier Transform

먼저 이산 푸리에 변환(Discrete Fourier Transform, DFT)[1](#chapter7#fn7_1)을 정의하는 것으로 시작합니다.

Definition 7.1. [Return to text.⏎](chapter7) 주어진 데이터 x1,…,xn에 대해, 이산 푸리에 변환(Discrete Fourier Transform, DFT)은 다음과 같이 주어집니다.

d(ωj)\=n−1/2∑t\=1nxt e−2πiωjt

여기서 j\=0,1,…,n−1이며, 주파수 ωj\=j/n은 푸리에(Fourier) 또는 기본 주파수(Fundamental frequencies)입니다.

만약 _n_이 여러 소수의 곱으로 이루어진 합성수(Highly composite integer, 즉 _p,q,r_이 정수일 때 n\=2p 3q 5r 형태)라면, [Cooley and Tukey (1965)](#bibref1#refbib_13)에서 소개된 고속 푸리에 변환(Fast Fourier Transform, FFT)을 사용하여 DFT를 효율적으로 계산할 수 있습니다. 때로는 DFT의 역변환 결과를 활용하는 것이 도움이 되는데, 이는 선형 변환이 일대일(One-to-one) 대응임을 보여줍니다. 역 DFT(Inverse DFT)의 경우 다음과 같습니다.

xt\=n−1/2∑j\=0n−1d(ωj) e2πiωjt

여기서 t\=1,…,n입니다. 다음 예제는 데이터 셋 {1,2,3,4}에 대해 DFT와 그 역변환을 계산하는 방법을 보여줍니다.


`( dft = fft(1:4)/sqrt(4) )`
`  [1] 5+0i -1+1i -1+0i -1-1i`
`fft(dft, inverse=TRUE)/sqrt(4)`
`  [1] 1+0i 2+0i 3+0i 4+0i`

_________________ 1복소수는 [Appendix B](#appB#appB)에서 검토합니다. [Return to text.⏎](#chapter7#fn7_17b)

172\. 이제 피리오도그램(Periodogram)을 DFT의 절댓값 제곱으로 정의합니다.

Definition 7.2. 주어진 데이터 x1,…,xn에 대해 피리오도그램(Periodogram)은 다음과 같습니다.

I(ωj)\=|d(ωj)|2(7.1)

여기서 j\=0,1,2,…,n−1이며, d(ωj)는 [Definition 7.1](#chapter7#defi7_1)에 주어져 있습니다.

I(0)\=nx¯2임을 유의해 주십시오. 여기서 x¯는 표본 평균입니다. 이 값은 평균의 크기에 따라 상당히 커질 수 있으며, 이는 데이터의 주기적 패턴과는 아무런 관련이 없습니다. 따라서 스펙트럼 분석을 수행하기 전에 일반적으로 데이터에서 평균을 빼서 I(0)\=0이 되도록 만듭니다.

0이 아닌 주파수에 대해서는 다음을 보일 수 있습니다. (자세한 내용은 [Problem 7.4](#chapter7#question7_4) 참조)

I(ωj)\=∑h\=−(n−1)n−1γ^(h)e−2πiωjh,(7.2)

여기서 γ^(h)는 (2.12)에서 살펴본 γ(h)의 추정치입니다. (7.2)에 따르면, 피리오도그램 I(ωj)는 (6.14)에 주어진 f(ωj)의 표본 버전(Sample version)입니다. 즉, 피리오도그램을 _xt_의 표본 스펙트럼 밀도(Sample spectral density)로 생각할 수 있습니다. I(ωj)가 f(ω)의 합리적인 추정치처럼 보이지만, 결국 이것이 단지 시작점일 뿐이라는 사실을 깨닫게 될 것입니다.

DFT의 실수부와 허수부를 개별적으로 다루는 것이 유용할 때가 있습니다. 이를 위해 다음과 같은 변환을 정의합니다. \[3\]

Definition 7.3. 주어진 데이터 x1,…,xn에 대해 코사인 변환(Cosine transform)은 다음과 같습니다.

dc(ωj)\=n−1/2∑t\=1nxtcos(2πωjt)(7.3)

그리고 사인 변환(Sine transform)은 다음과 같습니다.

ds(ωj)\=n−1/2∑t\=1nxtsin(2πωjt)(7.4)

여기서 j\=0,1,…,n−1에 대해 ωj\=j/n입니다.

dc(ωj)와 ds(ωj)는 (x¯와 같이) 표본 평균이지만 사인파 형태의 가중치를 가집니다(표본 평균은 각 관측치에 대해 1/n의 가중치를 가집니다). 적절한 조건 하에서 이 값들에 대한 중심극한정리([Section A.9](#appA#secA_9) 참조)는 다음과 같이 주어집니다.

dc(ωj)∼⋅N(0,12f(ωj))andds(ωj)∼⋅N(0,12f(ωj)),(7.5)

173\. 여기서 ∼⋅는 _n_이 클 때 근사적으로 분포함을 의미합니다. 또한 _n_이 클 때 ωj≠ωk인 한 dc(ωj),ds(ωj),dc(ωk),ds(ωk)는 상호 독립임을 보일 수 있습니다. 만약 _xt_가 가우시안(Gaussian)이라면, (7.5)와 이어지는 독립성에 대한 설명은 표본 크기에 관계없이 정확히 성립합니다.

d(ωj)\=dc(ωj)−i ds(ωj)이므로, 피리오도그램은 다음과 같습니다.

I(ωj)\=|dc(ωj)−i ds(ωj)|2\=dc2(ωj)+ds2(ωj),

이는 _n_이 클 때 두 개의 독립적인 정규 확률변수(Normal random variables)의 제곱합이며, 적절히 정규화하면 카이제곱(_χ_2) 분포를 따릅니다([Section A.4](#appA#secA_4) 참조). 따라서 (7.5)에 의해 다음을 얻습니다.

2 I(ωj)f(ωj)∼⋅χ22,(7.6)

여기서 χ22는 자유도가 2인 카이제곱 분포입니다. χν2 분포의 평균과 분산은 각각 _ν_와 2ν이므로, (7.6)으로부터 다음이 성립합니다.

E(2 I(ωj)f(ωj))≈2andvar(2 I(ωj)f(ωj))≈4,

따라서 다음과 같습니다.

E\[I(ωj)\]≈f(ωj)andvar\[I(ωj)\]≈f2(ωj).(7.7)

이는 좋지 않은 소식입니다. 왜냐하면 피리오도그램이 근사적으로 편향되지 않은(Unbiased) 추정치라 할지라도 표본 크기가 증가함에 따라 분산이 0으로 수렴하지 않기 때문입니다. 따라서 피리오도그램은 관측치가 아무리 많아도 결코 실제 스펙트럼에 가까워지지 않습니다. 크기가 _n_인 확률 표본의 평균 x¯가 E(x¯)\=μ이고 n→∞일 때 var(x¯)\=σ2/n→0이 되는 것과 이 상황을 비교해 보십시오.

(7.6)의 분포 결과는 분산에 대해 수행하는 것과 같이 근사적인 신뢰구간을 유도하는 데 사용할 수 있습니다. χν2(α)를 자유도가 _ν_인 카이제곱 분포의 하위 _α_ 확률 꼬리라고 나타냅니다. 그러면 스펙트럼 밀도 함수에 대한 근사적인 100(1−α) % 신뢰구간은 다음과 같은 형태가 됩니다.

2 I(ωj)χ22(1−α/2)≤f(ω)≤2 I(ωj)χ22(α/2).

로그(Log) 변환은 분산 안정화 변환(Variance stabilizing transformation)입니다. 이 경우 신뢰구간의 형태는 다음과 같습니다.

\[logI(ωj)−log12χ22(1−α2), logI(ωj)−log12χ22(α2)\].

흔히 스펙트럼 분석 전에 제거해야 할 비정상성(Nonstationary) 추세가 존재합니다. 추세는 데이터의 높은 주파수 대역에서의 특징을 가릴 수 있는 극히 낮은 주파수 성분을 피리오도그램에 도입합니다. 이러한 이유로 174\. 스펙트럼 분석 이전에 0 성분을 제거하기 위해 xt−x¯ 형태의 평균 조정(Mean-adjusted) 데이터를 사용하거나 xt−β^1−β^2t와 같은 추세 제거(Detrended) 데이터를 사용하여 데이터의 중심을 맞추는 것이 일반적입니다. astsa와 stats 패키지의 스크립트는 기본적으로 이러한 방식으로 데이터의 추세를 제거함을 참고해 주십시오.

DFT와 나아가 피리오도그램을 계산할 때, 고속 푸리에 변환(FFT) 알고리즘이 사용됩니다. FFT는 _n_이 여러 소수의 곱, 즉 2, 3, 5의 인수를 가진 정수일 때 DFT 계산 과정의 중복성을 활용합니다. 이 속성을 수용하기 위해 데이터의 추세를 제거(또는 중심을 맞춘) 후 다음으로 큰 소수의 곱을 갖는 정수 _n_′까지 0으로 채워넣습니다(Padding). 이는 기본 주파수 좌표가 j/n 대신 ωj\=j/n′가 됨을 의미합니다. [Figure 1.5](#chapter1#fig1_5)에 표시된 SOI 및 Recruitment 시리즈의 피리오도그램을 고려하여 이를 설명합니다. 이들이 월별 데이터 시리즈이고 n\=453개월임을 상기해 주십시오. _n_′을 찾으려면 nextn(453) 명령어를 사용하여 기본적으로 스펙트럼 분석에서 n′\=480이 사용될 것임을 확인할 수 있습니다.

Example 7.4 Periodogram of SOI and Recruitment Series [Return to text.⏎](chapter7)

[Figure 7.1](#chapter7#fig7_1)은 각 시리즈의 피리오도그램을 보여주며, 여기서 주파수 축은 연(Year)의 배수로 레이블링되어 있습니다. 앞서 언급한 바와 같이, 중심이 맞춰진 데이터는 480 길이의 시리즈로 채워졌습니다. 우리는 175\. 명백한 연간 주기인 ω\=1에서 좁은 대역폭을 갖는 피크를 발견할 수 있습니다. 또한 ω\=1/4 부근의 4년 주기를 중심으로 낮은 주파수 영역(약 2~7년)의 넓은 대역폭에 상당한 스펙트럼이 나타납니다. 이는 잠재적인 엘니뇨(El Niño) 효과를 나타냅니다. 이러한 넓은 대역의 활동은 엘니뇨 주기가 불규칙적임을 시사합니다.


`par(mfrow=c(2,1))`
`**mvspec**(**soi**, col=4, lwd=2)`
`  rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.5,.2))`
`  abline(v=1/4, lty=2, col=4)`
`  mtext("1/4", side=1, line=0, at=.25, cex=.75)`
`**mvspec**(**rec**, col=4, lwd=2)`
`  rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.5,.2))`
`  abline(v=1/4, lty=2, col=4)`
`  mtext("1/4", side=1, line=0, at=.25, cex=.75)`

![Periodograms of SOI and Recruitment showing a large narrow peak at the annual cycle and a broad peak around the four year cycle](./images/fig7_1.jpg)

Figure 7.1: Periodogram of SOI and Recruitment: 주파수 축은 연(Year) 단위입니다. 공통 피크는 연 1주기인 ω\=1과, 4년마다 1주기를 가지는 ω\=1/4 근처의 값에 위치합니다. 회색 영역은 2년에서 7년 사이의 주기를 나타냅니다. [Return to text.⏎](chapter7)

**mvspec** 객체의 정보에서 신뢰구간을 구성할 수 있지만, 스펙트럼을 로그 척도로 플로팅하면 [Figure 7.2](#chapter7#fig7_2)에서 볼 수 있듯이 일반적인 신뢰구간(눈금에 "중심이 맞춰진")이 생성됩니다. 각 주파수에는 오직 2개의 자유도만 존재하기 때문에, 일반적인 신뢰구간이 너무 넓어 크게 유용하지 않음을 유의해 주십시오. 다음으로는 이 문제를 다루겠습니다.

![Logged periodograms of SOI and Recruitment showing a large narrow peak at the annual cycle and a broad peak around the four year cycle and displaying a generic confidence interval](./images/fig7_2.jpg)

Figure 7.2: Log-periodogram of SOI and Recruitment. 오른쪽 상단의 파란색 선으로 95% 신뢰구간이 표시됩니다. 원하는 주파수의 로그-피리오도그램 좌표에 수평 눈금을 배치한다고 가정해 보십시오. 그러면 수직선이 해당 구간을 나타냅니다. [Return to text.⏎](chapter7)

176\. 피리오도그램을 로그 척도로 표시하려면 **mvspec**() 호출에 log="y"를 추가합니다(또한 사각형 rect()의 ybottom 값을 1e-5로 변경합니다). 예를 들어,


`**mvspec**(**soi**, col=4, lwd=2, log="y")`
`  rect(1/7, 1e-5, 1/2, 1e5, density=NA, col=gray(.5,.2))`
`  abline(v=1/4, lty=2, col=4)`
`  mtext("1/4", side=1, line=0, at=.25, cex=.75)`

추정량으로서 피리오도그램은 불확실성에 크게 노출되기 쉽습니다. 이러한 현상이 발생하는 이유는 사용 가능한 관측치가 아무리 많아도 피리오도그램은 각 주파수에서 오직 2개의 정보만 사용하기 때문입니다.

## 7.2 Nonparametric Spectral Estimation

피리오도그램 딜레마에 대한 해결책은 평활화(Smoothing)이며, [Section 3.3](#chapter3#sec3_3)과 동일한 아이디어를 기반으로 합니다. 문제를 이해하기 위해 [Figure 7.3](#chapter7#fig7_3)에서 1000개의 독립적인 표준 정규분포(백색 정규 잡음) 데이터의 피리오도그램을 살펴보겠습니다. 실제 스펙트럼 밀도는 높이가 1인 균일 밀도(Uniform density)입니다. 피리오도그램의 변동성이 크지만, 평균화가 도움이 됩니다.[2](#chapter7#fn7_2)


`u = **mvspec**(rnorm(1000), col=8, **gg**=TRUE) _# periodogram_`
`abline(h=1, col=2, lwd=5)               _# true spectrum_`
`sm = filter(u$spec, filter=rep(1,101)/101, circular=TRUE) _# smooth_`
`lines(u$freq, sm, col=5, lwd=2)         _# add the smooth_`

![Periodogram of 1000 independent standard normals showing how variable the periodogram is, and the smoothed periodogram showing the advantages of smoothing](./images/fig7_3.jpg)

Figure 7.3: Periodogram of 1000 independent standard normals (white normal noise). 빨간색 직선은 이론적 스펙트럼(균일 밀도)이며 기복이 심한 파란색 선은 101개의 피리오도그램 좌표에 대한 이동 평균입니다. [Return to text.⏎](chapter7)

_________________ 2만약 dplyr 패키지가 로드되어 있다면, _Hints for Selected Exercises_ 의 경고를 확인하십시오. [Return to text.⏎](#chapter7#fn7_27b)

177\. 관심 주파수 _ω_ 근처에 선택된 주파수 ωj\=j/n을 중심으로, L≪n 개의 인접한 기본 주파수로 구성된 주파수 대역 B를 도입합니다.

B\={ωj+k/n: k\=0,±1,…,±m},(7.8)

여기서

L\=2m+1

은 홀수이며, 대역 B 구간에 위치한 스펙트럼 값, 즉

f(ωj+k/n),k\=−m,…,0,…,m

이 근사적으로 f(ω)와 같도록 선택됩니다.

이제 평균 피리오도그램(Averaged periodogram)을 대역 B 범위에서 피리오도그램 값들의 단순 평균으로 정의합니다.

f¯(ω)\=1L∑k\=−mmI(ωj+k/n),(7.9)

대역 B 내에서 스펙트럼 밀도가 상당히 일정하다는 가정 하에, (7.5)에 대한 논의에 따라 _n_이 클 때 다음을 보일 수 있습니다.

2Lf¯(ω)f(ω)∼⋅χ2L2.(7.10)

이제 우리는 다음을 얻습니다.

E\[f¯(ω)\]≈f(ω)andvar\[f¯(ω)\]≈f2(ω)/L,(7.11)

이 식은 (7.7)과 비교할 수 있습니다. 이 경우 n→∞일 때 L→∞가 되게 하면 var\[f¯(ω)\]→0이 되지만, _L_은 _n_보다 훨씬 더 느리게 커져야 합니다.

단순 평균을 통해 피리오도그램을 평활화할 때, (7.8)에 의해 정의된 주파수 구간의 너비

B\=Ln(7.12)

를 대역폭(Bandwidth)이라고 부릅니다.[3](#chapter7#fn7_3)

결과 (7.10)은 실제 스펙트럼 f(ω)에 대해 다음과 같은 형태의 근사적인 100(1−α)% 신뢰구간을 구하는 데 사용될 수 있습니다.

2Lf¯(ω)χ2L2(1−α/2)≤f(ω)≤2Lf¯(ω)χ2L2(α/2)(7.13)

_________________ 3대역폭에 대한 정의는 여러 가지가 있습니다. 여기서 제시한 정의는 대역의 너비라는 해석과 일치하므로 더 선호됩니다. [Return to text.⏎](#chapter7#fn7_37b)

178\. 앞서 논의했듯이, 스펙트럼의 로그(log) 값을 플로팅함으로써 스펙트럼 밀도 플롯의 특정 측면을 강조할 수 있습니다. 이 현상은 전체 파워 성분 중 일부 주력 파워 성분보다 훨씬 작은 피크가 존재하는 스펙트럼 영역에서 발생할 수 있습니다. 로그 스펙트럼에 대해 다음과 같은 신뢰구간을 얻습니다.

(logf¯(ω)−log12Lχ2L2(1−α2), logf¯(ω)−log12Lχ2L2(α2)).(7.14)

만약 스펙트럼 추정치를 계산하기 전에 데이터가 패딩(Padding)되었다면, 대가 없이는 아무것도 얻을 수 없기 때문에 자유도를 조정해야 합니다. 잘 작동하는 근사법은 2_L_을 2Ln/n′로 대체하는 것입니다. 따라서 조정된 자유도(Adjusted degrees of freedom)를 다음과 같이 정의합니다.

df\=2Lnn′(7.15)

이를 신뢰구간 (7.13) 및 (7.14)에서 2_L_ 대신 사용합니다. 예를 들어 (7.13)은 다음과 같이 변환됩니다.

dff¯(ω)χdf2(1−α/2)≤f(ω)≤dff¯(ω)χdf2(α/2).(7.16)

더 진행하기 전에 잠시 멈추고 [Figure 7.4](#chapter7#fig7_4)에 표시된 바와 같이 SOI 및 Recruitment 데이터 시리즈에 대한 평균 피리오도그램 계산을 고려해 봅니다.

![Averages periodograms  of  SOI and Recruitment showing the improvement of the spectral estimate](./images/fig7_4.jpg)

Figure 7.4: The averaged periodogram of the SOI and Recruitment series n\=453, n′\=480, L\=9, df\=17. 4년 주기의 ω\=1/4, 연간 주기 ω\=1, 그리고 이들의 배수 주파수(Harmonics)인 k\=2,3 일 때 ω\=k에서 공통된 피크를 보여줍니다. 회색 영역은 2년에서 7년 사이의 주기를 나타냅니다. [Return to text.⏎](chapter7)

Example 7.5 Averaged Periodogram for SOI and Recruitment [Return to text.⏎](chapter7)

일반적으로 피리오도그램에서 제안된 스펙트럼의 전반적인 형태와 호환되는 것으로 보이는 여러 대역폭을 시도해 보는 것이 좋은 접근법입니다. 앞서 [Figure 7.1](#chapter7#fig7_1)에서 계산된 SOI 및 Recruitment 시리즈의 피리오도그램은 주된 전체 주기를 식별하기 위해 낮은 주파수의 엘니뇨 대역폭 영역의 평활화가 필요함을 시사합니다. _L_ 값을 조정해 보면 합리적인 값으로 L\=9가 적절하게 도출되며 그 결과가 [Figure 7.4](#chapter7#fig7_4)에 나타나 있습니다.

[Figure 7.4](#chapter7#fig7_4)에 나타난 평활화된 스펙트럼은 [Figure 7.1](#chapter7#fig7_1)의 노이즈가 많은 버전과 일부 피크가 사라질 수 있는 지나치게 평활화된 스펙트럼 사이의 적절한 절충안을 제공합니다. 1년 주기 ω\=1에서 평균화의 원치 않는 영향을 발견할 수 있는데, [Figure 7.1](#chapter7#fig7_1)의 피리오도그램에 나타났던 좁은 대역폭의 피크들이 평평해져 근처 주파수로 퍼졌습니다. 또한 1년 주기의 배수 주파수(Harmonics), 즉 k\=1,2,…일 때 ω\=k 형태의 주파수가 나타나는 것을 볼 수 있습니다. 배수 주파수는 일반적으로 주기적 성분이 존재하지만 정현파(Sinusoidal) 형태가 아닐 때 발생합니다. [Example 7.6](#chapter7#exam7_6)을 참조해 주십시오.

[Figure 7.4](#chapter7#fig7_4)는 평균 스펙트럼 추정치를 보여줍니다. 평균 피리오도그램을 계산하기 위해 **mvspec** 호출에 L\=2m+1(이 예제에서는 L\=9 및 m\=4)을 지정합니다. 기본적으로 [Example 3.18](#chapter3#exam3_18)에서 했던 것처럼 스무더(Smoother)의 양 끝에 절반 가중치가 사용됨을 유의해 주십시오. 이는 (7.12)–(7.16)이 약간의 오차를 가짐을 의미하지만, 다른 스무더로 이동할 것이기 때문에 정확한 결과를 얻기 위해 모든 것을 다시 코딩할 가치는 없습니다. 스크립트는 또한 179\. 대역폭, 자유도 및 잠시 후 논의할 테이퍼링(Tapering) 정도를 출력합니다. SOI에 대한 R 코드는 아래에 주어져 있으며 Recruitment에 대한 해당 코드도 이와 유사합니다.


`soi_ave = **mvspec**(**soi**, spans=9, col=4, lwd=2)`
`    Bandwidth: 0.213 | Degrees of Freedom: 16.11 | split taper: 0%`
` rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.5,.2))`
` abline(v=.25, lty=2, col=4)`
` mtext("1/4", side=1, line=0, at=.25, cex=.75)`

최대 에너지를 가지는 것으로 식별된 두 주파수 대역에 대해 95% 신뢰구간을 살펴보고 하한이 인접한 기저선 스펙트럼(Baseline spectral) 수준보다 실질적으로 더 큰지 확인할 수 있습니다. 신뢰구간은 스펙트럼 추정치가 로그 척도로 플로팅될 때 나타남을 상기해 주십시오(이전과 마찬가지로 위 코드에 log="y"를 추가하고 사각형의 하단 부분을 \-1e5에서 1e-5로 변경합니다). 예를 들어, [Figure 7.5](#chapter7#fig7_5)에서 4년 주기의 엘니뇨 위치에 있는 피크는 만약 피크 없이 단순히 부드러운 스펙트럼 함수가 있는 경우의 스펙트럼 값보다 큰 하한 값을 가집니다.

![Logged averaged periodograms  of the  SOI and Recruitment and corresponding generic confidence intervals](./images/fig7_5.jpg)

Figure 7.5: [Figure 7.4](#chapter7#fig7_4) with the average periodogram ordinates plotted on a log scale. 오른쪽 상단 모서리에 표시된 항목은 일반적인 95% 신뢰구간을 나타내며 수평 선분의 너비는 대역폭을 나타냅니다. [Return to text.⏎](chapter7)

Example 7.6 180\. Harmonics [Return to text.⏎](chapter7)

이전 예제에서 연간 신호의 스펙트럼이 배수 주파수(Harmonics)에서 작은 피크를 나타내는 것을 확인했습니다. 즉, ω\=1 주기/년에서 큰 피크가 있었고, 이의 배수 주파수인 ω\=k (k\=2,3,…) (연간 2, 3회 등)에서 작은 피크들이 있었습니다. 대부분의 신호는 완벽한 정현파(Sinusoids) 또는 완벽한 주기성 형태가 아니기 때문에 종종 이러한 현상이 발생합니다. 이 경우, 신호의 비-정현파적인 거동(Non-sinusoidal behavior)을 포착하기 위해 배수 주파수가 필요합니다. 예로 [Figure 7.6](#chapter7#fig7_6)에 표시된 것처럼 20개의 포인트마다 1주기를 만드는 톱니파 신호(Sawtooth signal)를 고려해 봅니다. 이 시리즈는 순수한 신호(잡음이 추가되지 않음)이지만, 정현파 형태가 아니며 빠르게 상승한 다음 천천히 하강합니다. 톱니파 신호의 피리오도그램은 [Figure 7.6](#chapter7#fig7_6)에도 나타나 있으며 주된 주기의 배수 주파수에서 줄어드는 수준의 피크를 보여줍니다.


`y = ts(100:1 %% 20, freq=20)   _# sawtooth signal_`
`par(mfrow=2:1)`
`**tsplot**(1:100, y, ylab="sawtooth signal", col=4, **gg**=TRUE)`
`**mvspec**(y, main=NA, ylab="periodogram", col=5, **gg**=TRUE)`

![Harmonics demonstration using pure sawtooth signal and its corresponding periodogram showing decreasing peaks at the signal's harmonics](./images/fig7_6.jpg)

Figure 7.6: Harmonics: 20포인트마다 1주기를 만드는 순수한 톱니파 신호와 신호 주파수 및 배수 주파수에서 피크를 보여주는 해당 피리오도그램입니다. 주파수 척도는 20포인트 주기 단위입니다. [Return to text.⏎](chapter7)

[Example 7.5](#chapter7#exam7_5)는 피크가 유의미한지 여부를 결정하기 위한 비교적 체계적인 절차를 가져야 할 필요성을 지적합니다. 피크가 유의미한지에 대한 질문은 대개 스펙트럼의 기저선(Baseline) 수준으로 간주될 수 있는 값을 설정하는 것에 달려 있습니다. 다소 대략적으로 정의된 기저선 수준은 스펙트럼 피크가 존재하지 않을 때 예상할 수 있는 181\. 형태를 뜻합니다. 이 프로필은 대개 피크를 포함한 스펙트럼의 전체적인 형태를 살펴보면서 짐작할 수 있는데, 피크가 두드러지게 나타나는 일종의 기저선 수준이 시각적으로 드러나게 됩니다. 만약 스펙트럼 값의 하한 신뢰 한계가 미리 정해진 유의수준에서 여전히 기저선 수준보다 크다면, 그 주파수 값을 통계적으로 유의미한 피크라고 주장할 수 있습니다. 상한 한계에 대해서는 언급하지 않는 우리의 명시적인 무관심과 일치하도록 한쪽 꼬리 신뢰구간(One-sided confidence interval)을 사용할 수도 있습니다.

스펙트럼이 기본적으로 일정한 값을 가지게 되는 대역폭 B에 대해 결정을 내릴 때는 주의가 필요합니다. 너무 넓은 대역을 선택하면 대역에 걸쳐 일정한 분산 가정이 충족되지 않을 때 데이터의 유효한 피크마저 부드럽게 평활화시켜 버릴 수 있습니다. 반대로 너무 좁은 대역을 선택하면 신뢰구간이 너무 넓어져 피크가 더 이상 통계적으로 유의미하지 않게 됩니다.

따라서 대역폭 B를 늘림으로써 향상될 수 있는 분산 속성 혹은 대역폭 안정성(Bandwidth stability)과, B를 줄임으로써 향상될 수 있는 해상도(Resolution) 사이에 갈등이 존재함을 알 수 있습니다. 일반적인 접근법은 다양한 대역폭을 시도해 보고 각 경우에 대한 스펙트럼 추정치를 정성적으로 살펴보는 것입니다.

182\. 해상도 문제를 해결하기 위해, [Figure 7.4](#chapter7#fig7_4)와 [Figure 7.5](#chapter7#fig7_5)에서 발생한 피크의 평활화 현상이 (7.9)에 정의된 f¯(ω)를 계산할 때 단순 평균을 사용했기 때문임이 분명해져야 합니다. 단순 평균을 고집할 특별한 이유는 없으며 다음과 같이 가중 평균을 적용함으로써 추정치를 향상시킬 수 있습니다.

f^(ω)\=∑k\=−mmhk I(ωj+k/n),(7.17)

(7.9)와 동일한 정의를 사용하지만, 가중치(Weights) hk\>0은 다음을 만족합니다.

∑k\=−mmhk\=1.

특히, 중심 가중치 _h_0에서 멀어질수록 감소하는 가중치를 사용하면 추정치의 해상도(Resolution)가 향상될 것입니다. 이 아이디어에 대해서는 곧 다시 다루겠습니다. (7.17)의 평균 피리오도그램(Averaged periodogram) f¯(ω)를 얻기 위해 모든 _k_에 대해 hk\=1/L로 설정합니다. 여기서 L\=2m+1입니다. 우리는 다음과 같이 정의합니다.

Lh\=(∑k\=−mmhk2)−1,(7.18)

그리고 단순 평균에서와 같이 hk\=1/L이면 Lh\=L이 됨을 유의해 주십시오. (7.17)의 분포적 성질은 이제 더 까다로워졌는데, f^(ω)가 근사적으로 독립인 _χ_2 확률변수들의 가중 선형 결합이기 때문입니다. (온건한 조건 하에서) 잘 작동하는 것으로 보이는 근사법은 (7.10)의 _L_을 _Lh_로 대체하는 것입니다. 즉,

2Lhf^(ω)f(ω)∼⋅χ2Lh2.(7.19)

(7.12)와 유사하게, 이 경우 대역폭을 다음과 같이 정의합니다.

B\=Lhn.

(7.11)과 유사하게, _n_이 클 때 다음이 성립합니다.

E\[f^(ω)\]≈f(ω)andvar\[f^(ω)\]≈f2(ω)/Lh.

근사식 (7.19)를 사용하여 실제 스펙트럼 f(ω)에 대한 다음과 같은 형태의 근사적인 100(1−α)% 신뢰구간을 얻습니다.

2Lhf^(ω)χ2Lh2(1−α/2)≤f(ω)≤2Lhf^(ω)χ2Lh2(α/2)(7.20)

만약 데이터가 _n_′으로 패딩되었다면, (7.15)에서와 같이 (7.20)의 2Lh를 df\=2Lhn/n′로 대체하십시오.

183\. 기본적으로 스펙트럼을 추정하는 스크립트는 양 끝점에 절반 가중치를 사용하면서 평균화하는 수정된 다니엘 커널(Modified Daniell kernel)을 통해 피리오도그램을 평활화합니다. 예를 들어, m\=1(그리고 L\=2m+1\=3)일 때 가중치는 {hk}\={14,24,14}이며, 이를 숫자 시퀀스 {ut}에 적용하면 다음과 같은 결과를 얻습니다.

u^t\=14ut−1+12ut+14ut+1.

동일한 커널을 u^t에 다시 적용하면 다음을 산출합니다.

^^ut\=14u^t−1+12u^t+14u^t+1,

이를 단순화하면 다음과 같습니다.

^^ut\=116ut−2+416ut−1+616ut+416ut+1+116ut+2.

이 커널에 대한 추가적인 세부 사항은 [Example A.9](#appA#examA_9)에서 제공됩니다.

Example 7.7 Smoothed Periodogram for SOI and Recruitment [Return to text.⏎](chapter7)

이 예제에서는 (7.17)의 평활화된 피리오도그램 추정치를 사용하여 SOI 및 Recruitment 시리즈의 스펙트럼을 추정합니다. 우리는 두 번 모두 m\=3을 적용하여 수정된 다니엘 커널을 두 번 사용했습니다. 이로 인해 Lh\=1/∑hk2\=9.232가 도출되며, 이는 [Example 7.5](#chapter7#exam7_5)에서 사용된 L\=9 값과 비슷합니다. 가중치 _hk_는 다음과 같이 구하고 그래프로 나타낼 수 있습니다. [Figure 7.7](#chapter7#fig7_7)을 참조해 주십시오(오른쪽 플롯은 커널의 또 다른 적용을 추가합니다).


`(dm = kernel("modified.daniell", c(3,3)))       _# for a list_`
`par(mfrow=1:2)`
`plot(dm, ylab=bquote(h[~k]))                    _# for a plot_`
`plot(kernel("modified.daniell", c(3,3,3)), ylab=bquote(h[~k]))`

![Modified Daniell kernel using various amounts of smoothing and displaying how the kernel approaches a normal kernel](./images/fig7_7.jpg)

Figure 7.7: Modified Daniell kernel weights used in [Example 7.7](#chapter7#exam7_7). [Return to text.⏎](chapter7)

스펙트럼 추정치는 [Figure 7.8](#chapter7#fig7_8)에서 확인할 수 있으며, 이 추정치들이 [Figure 7.4](#chapter7#fig7_4)의 추정치들보다 더 호소력 있음을 알 수 있습니다. 아래 코드에서 spans는 커널의 너비 L\=2m+1로 주어지는 홀수들의 벡터(Vector)임을 주목하십시오. 개월 수 관점에서의 대역폭은 B\=9.232/480\=.019이지만, 184\. 시간의 한 단위는 1년이므로 그래프를 위한 대역폭은 9.232480cyclesmonths×12monthsyear\=.2308cyclesyear로 변환됩니다. **mvspec**을 사용할 때마다(plot=FALSE가 아닌 경우) 대역폭, 자유도, 그리고 테이퍼링 정도(Tapering amount)가 출력됩니다.


`par(mfrow=c(2,1))`
`sois = **mvspec**(**soi**, spans=c(7,7), taper=.1, col=4, lwd=2)`
` Bandwidth: 0.231 | Degrees of Freedom: 15.61 | split taper: 10%`
`  rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.5,.2))`
`  abline(v=.25, lty=2, col=4)`
`  mtext("1/4", side=1, line=0, at=.25, cex=.75)`
`recs = **mvspec**(**rec**, spans=c(7,7), taper=.1, col=4, lwd=2)`
`  rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.5,.2))`
`  abline(v=.25, lty=2, col=4)`
`  mtext("1/4", side=1, line=0, at=.25, cex=.75)`

![The estimates of the spectrum of the SOI and of the Recruitment time series based on smoothing the periodograms using a modified Daniell kernel](./images/fig7_8.jpg)

Figure 7.8: Smoothed (tapered) spectral estimates of the SOI and Recruitment series; see [Example 7.7](#chapter7#exam7_7) for details. [Return to text.⏎](chapter7)

이전과 마찬가지로, log="y"를 추가하여 **mvspec** 명령을 다시 실행하면 [Figure 7.5](#chapter7#fig7_5)와 유사한 그림이 생성됩니다(그리고 사각형의 하단 값을 1e-5로 변경하는 것을 잊지 마십시오). 스펙트럼 피크의 위치를 찾는 쉬운 방법은 피크 위치 근처의 몇 가지 값을 출력하는 것입니다. 이 예제에서는 피크가 시작 부분 근처에 있음을 알고 있으므로 해당 부분을 살펴봅니다.


`sois$details[1:45,]`
`      frequency period spectrum`
`             .      .       .`
` [5,]     0.125 8.0000   0.0320`
` [6,]     0.150 6.6667   0.0372 ∼ 7 year    period`
` [7,]     0.175 5.7143   0.0421`
` [8,]     0.200 5.0000   0.0461`
` [9,]     0.225 4.4444   0.0489`
`[10,]     0.250 4.0000   0.0502 <- 4 year   period`
`[11,]     0.275 3.6364   0.0490`
`[12,]     0.300 3.3333   0.0451`
`[13,]     0.325 3.0769   0.0403 ∼ 3 year    period`
`[14,]     0.350 2.8571   0.0361`
`             .      .       .`
`[38,]     0.950 1.0526   0.1253`
`[39,]     0.975 1.0256   0.1537`
`[40,]     1.000 1.0000   0.1675 <- 1 year   period`
`[41,]     1.025 0.9756   0.1538`
`[42,]     1.050 0.9524   0.1259`

185\. 마지막으로, [Figure 7.8](#chapter7#fig7_8)이 다음에 논의할 테이퍼(Taper)를 사용하여 생성되었음을 유의해 주십시오.

### Tapering

이제 테이퍼링(Tapering) 개념을 도입할 준비가 되었습니다. 더 자세한 논의는 [Bloomfield (2004)](#bibref1#refbib_3)에서 찾아볼 수 있습니다. _xt_가 스펙트럼 밀도 fx(ω)를 갖는 평균이 0인 정상성 과정(Stationary process)이라고 가정해 보십시오. 가중치 _at_를 지정하고, 원본 시리즈를 테이퍼링된 시리즈(Tapered series)

yt\=atxt,(7.21)

로 대체하여(여기서 t\=1,2,…,n), 수정된 DFT

dy(ωj)\=n−1/2∑t\=1natxte−2πiωjt,(7.22)

를 사용하고 Iy(ωj)\=|dy(ωj)|2로 둔다면, 다음을 얻게 됩니다.

E\[Iy(ωj)\]\=∫−1/21/2Wn(ωj−ω) fx(ω) dω.(7.23)

값 Wn(ω)는 스펙트럼 창(Spectral window)이라고 불립니다. 왜냐하면 (7.23)에서 볼 수 있듯이 이 값이 평균적으로 추정치 Iy(ωj)가 스펙트럼 밀도 fx(ω)의 어느 부분을 "보게(Seen)" 될지 결정하기 때문입니다. 모든 _t_에 대해 at\=1인 경우, Iy(ωj)\=Ix(ωj)는 단순히 데이터의 피리오도그램이며 이 때의 창은 다음과 같습니다.

Wn(ω)\=sin2(nπω)nsin2(πω)(7.24)

그리고 Wn(0)\=n을 가집니다.

186\. 테이퍼는 일반적으로 양 극단에 비해 데이터의 중심 부분을 강조하는 모양을 가집니다. 예를 들어 [Blackman and Tukey (1959)](#bibref1#refbib_2)가 선호한 다음과 같은 코사인 벨(Cosine bell) 형태가 있습니다.

at\=.5\[1+cos(2π(t−t―)n)\],(7.25)

여기서 t―\=(n+1)/2 입니다.

[Figure 7.9](#chapter7#fig7_9)에는 L\=9인 상태에서 (7.9)의 추정치 f¯(ω)를 사용할 때 n\=480에 대한 두 가지 스펙트럼 창 Wn(ω)의 형태가 그려져 있습니다. 그래픽의 왼쪽 부분은 테이퍼링이 없는 경우(at\=1)를 보여주며, 오른쪽 부분은 _at_가 (7.25)의 코사인 테이퍼인 경우를 나타냅니다. 두 경우 모두 대역폭은 B\=9/480\=.01875 포인트당 주기(Cycles per point)여야 하며, 이는 [Figure 7.9](#chapter7#fig7_9)에 표시된 창의 "너비"에 해당합니다. 두 창 모두 이 대역에서 적분된 평균 스펙트럼을 생성하지만, 왼쪽의 테이퍼링되지 않은 창은 대역 안팎에 걸쳐 상당한 물결무늬(Ripples)를 보여줍니다. 대역 외부의 이러한 물결을 사이드로브(Sidelobes)라고 부르며, 구간 외부의 주파수를 유입시켜 대역 내의 원하는 스펙트럼 추정치를 오염시키는 경향이 있습니다. 이러한 효과를 종종 누설(Leakage)이라고 부릅니다. [Figure 7.9](#chapter7#fig7_9)는 코사인 테이퍼를 사용할 때 사이드로브 억제 효과가 커짐을 강조합니다.

![Spectral windows with and without tapering displaying the concept of leakage](./images/fig7_9.jpg)

Figure 7.9: Spectral windows with and without tapering corresponding to the average periodogram with n\=480 and L\=9 as in [Example 7.5](#chapter7#exam7_5). The extra line and ticks on the abscissa exhibit the bandwidth. [Return to text.⏎](chapter7)

테이퍼링의 효과를 생각하는 한 가지 방법은 스펙트럼 창 모양의 유리창을 밖을 내다보는 상황을 상상하는 것입니다. 테이퍼링이 수행되지 않으면 유리에 많은 물결이 생겨 바깥 풍경이 흐릿하게 보일 것입니다. 이러한 효과는 창문을 손으로 만들었던 많은 오래된 건물에서 볼 수 있습니다. 그러나 만약 (현대식 유리창처럼) 테이퍼링이 적용된 스펙트럼 창과 같이 유리창이 만들어진다면 바깥 풍경이 선명하게 보일 것입니다.

Example 7.8 187\. The Effect of Tapering the SOI Series [Return to text.⏎](chapter7)

이 예제에서는 SOI 시리즈의 스펙트럼 추정치에 미치는 테이퍼링의 영향을 살펴봅니다. [Figure 7.10](#chapter7#fig7_10)은 로그 척도로 플로팅된 세 가지 스펙트럼 추정치의 일부를 보여줍니다. 여기서의 평활화 정도는 [Example 7.7](#chapter7#exam7_7)과 동일합니다. 세 가지 스펙트럼 추정치는 테이퍼링이 없는 경우, 각 끝단에 20%의 테이퍼링(즉, 데이터의 처음과 마지막 20%만 테이퍼링됨), 그리고 50%의 전체 테이퍼링(Full tapering)이 적용된 경우입니다. 테이퍼링된 스펙트럼이 연간 주기(ω\=1)와 엘니뇨 주기(ω\=1/4)를 분리하는 데 더 나은 성능을 발휘함을 유의해 주십시오.

![Smoothed spectral estimates of  SOI  without tapering and with  tapering](./images/fig7_10.jpg)

Figure 7.10: Display for [Example 7.8](#chapter7#exam7_8): Smoothed spectral estimates of SOI without tapering, with split tapering of 20%, and with a full (50%) cosine bell taper (7.25). The tapers are displayed on the right. [Return to text.⏎](chapter7)

다음 코드를 사용하여 [Figure 7.10](#chapter7#fig7_10)을 생성했습니다. mvspec은 기본적으로 테이퍼링을 수행하지 않음을 참고해 주십시오. 전체 테이퍼링을 적용하려면 인수를 taper=.5로 사용하여 데이터의 각 양 끝단에서 50%씩 테이퍼링하도록 mvspec에 지시합니다. 0과 0.5 사이의 어떠한 값이라도 허용됩니다.


`layout(matrix(1:2,1), widths=c(3,1))`
`s0 = **mvspec**(**soi**, spans=c(7,7), plot=FALSE)              _# no taper_`
`s10 = **mvspec**(**soi**, spans=c(7,7), taper=.2, plot=FALSE) _# 20%_`
`s50 = **mvspec**(**soi**, spans=c(7,7), taper=.5, plot=FALSE)   _# full taper_`
`r = 1:60`
`**tsplot**(s0$freq[r], log(s0$spec[r]), **gg**=TRUE, col=4, lwd=2, ylab="log-spectrum",`
`   xlab="frequency")`
`lines(s10$freq[r], log(s10$spec[r]), col=2, lwd=2)`
`lines(s50$freq[r], log(s50$spec[r]), col=3, lwd=2)`
`text(.7, -3.5, "leakage", cex=.8)`
`arrows(.7, -3.6, .7, -4.5, length=0.05, angle=30)`
`legend("bottomleft", legend=c("no taper", "20% taper", "50% taper"), lwd=2,`
`   col=c(4,2,3), bty="n")`
`_# tapers_`
`x = rep(1,100)`
`**tsplot**(1:100/100, cbind(spec.taper(x, p=.2), spec.taper(x, p=.5)), col=2:3,`
`   **gg**=TRUE, **spag**=TRUE, xlab="t / n", lwd=2, ylab="tapers")`

Example 7.9 188\. Tapering Recommendation

테이퍼 {at}를 사용하면 스펙트럼 추정량의 분산이 다음과 같이 주어지는 첨도 계수(Kurtosis factor)만큼 증가합니다.

κn\=1n∑tat4(1n∑tat2)2,(7.26)

이 식은 n→∞일 때 코시-슈바르츠 부등식(Cauchy-Schwarz inequality, A.3)에 의해 1보다 크거나 같습니다.

[Example 7.8](#chapter7#exam7_8)에서 논의된 바와 같이 스플릿(Split) 테이퍼링을 사용하는 (7.25)의 코사인 벨 테이퍼에 대해 [Bloomfield (2004, §9.5)](#bibref1#refbib_3)는 다음을 보였습니다. (0≤p≤.5)

κn≈128−186p2(8−10p)2.

p\=5%,10%,20%에 대해 _κn_의 값은 각각 1.06,1.12,1.26입니다. 또한 테이퍼링은 추정량의 자유도를 1/κn 배만큼 감소시킵니다. 따라서 이 수준에서의 스플릿 테이퍼링은 스펙트럼 추정량의 효율성을 아주 적게만 저하시키며, 이는 누설(Leakage)을 방지하기 위한 대가로 가치가 있습니다.

## 7.3 Parametric Spectral Estimation

[Section 7.2](#chapter7#sec7_2)의 방법들은 스펙트럼 밀도의 모수적 형태(Parametric form)에 대해 아무런 가정을 하지 않기 때문에 일반적으로 비모수적(Nonparametric) 스펙트럼 추정량으로 불립니다. Property 6.8에서 우리는 ARMA 과정의 스펙트럼을 제시했으며, 데이터에 맞춘 ARMA(_p,q_) 모델의 매개변수 추정치를 (6.15)에 주어진 스펙트럼 밀도 fx(ω) 공식에 대입하는 방식의 스펙트럼 추정량을 고려할 수 있습니다. 이러한 추정량을 모수적 스펙트럼 추정량(Parametric spectral estimator)이라고 부릅니다.

편의상 매개변수의 중복성을 피하기 위해 (Examples 4.9, 4.10, 4.11, 4.30 및 4.31에서처럼), AR(_p_) 모델을 데이터에 맞추어 모수적 스펙트럼 추정량을 얻으며 여기서 차수(Order) _p_는 (3.7)–(3.9)에 정의된 AIC, AICc 및 BIC와 같은 모델 선택 기준 중 하나에 의해 결정됩니다. 자기회귀(Autoregressive) 스펙트럼 추정량의 발전은 [Parzen (1983)](#bibref1#refbib_35)에 의해 요약되었습니다.

만약 ϕ^1,ϕ^2,…,ϕ^p와 σ^w2가 _xt_에 맞춰진 AR(_p_) 모델의 추정치들이라면, Property 6.8에 기반하여 이러한 추정치들을 (6.15)에 대입함으로써 fx(ω)의 모수적 스펙트럼 추정치를 얻을 수 있습니다. 즉,

f^x(ω)\=σ^w2|ϕ^(e−2πiω)|2,

189\. 여기서

ϕ^(z)\=1−ϕ^1z−ϕ^2z2−⋯−ϕ^pzp.(7.27)

안타깝게도 이 경우 스펙트럼에 대한 신뢰구간을 구하는 것은 어렵습니다. 대부분의 기술은 비현실적인 가정에 의존합니다.

(6.15) 형태 스펙트럼에 대한 흥미로운 사실은 어떠한 스펙트럼 밀도라도 AR 과정의 스펙트럼에 의해 임의로 가깝게 근사될 수 있다는 것입니다.

Property 7.10 (AR Spectral Approximation). [Return to text.⏎](chapter7) gx(ω)를 정상성 과정(Stationary process) xt의 스펙트럼 밀도라고 합시다. 그러면 ϵ\>0이 주어졌을 때, 다음에 해당하는 AR(p) 표현식이 존재합니다.

xt\=∑k\=1pϕkxt−k+wt

이때 해당 스펙트럼 fx(ω)는 다음을 만족합니다.

|fx(ω)−gx(ω)|<ϵfor all ω∈\[−1/2,1/2\].

그러나 한 가지 단점은, 이 속성이 근사치가 합리적이 되기 위해 _p_가 얼마나 커야 하는지 알려주지 않는다는 점입니다. 일부 상황에서는 _p_가 매우 커질 수 있습니다. [Property 7.10](#chapter7#prop7_10)은 일반적으로 MA 및 ARMA 과정에도 성립합니다. 다음 예제에서 이 기법을 시연합니다.

Example 7.11 Autoregressive Spectral Estimator for SOI

SOI 시리즈에 대해 [Figure 7.4](#chapter7#fig7_4)에 표시된 비모수적 추정치들과 비교할 수 있는 결과를 얻는 상황을 고려해 봅니다. 결과적으로 도출된 최소 AIC (p\=18) 및 최소 BIC (p\=15) 스펙트럼은 [Figure 7.11](#chapter7#fig7_11)에 나타나 있으며, 여기서 3~1/3년 및 1년 주기 부근의 광대역(Broadband) 에너지를 확인할 수 있습니다. 이는 [Section 7.2](#chapter7#sec7_2)에서 구한 비모수적 추정치와 유사합니다. [Figure 7.4](#chapter7#fig7_4)에는 2년에서 7년 사이의 주기가 강조되어 있습니다. 해당 주기 범위에서 BIC 스펙트럼이 더 넓게 퍼져 나타나며, 이는 ENSO 주기가 AIC 스펙트럼에서 볼 수 있는 것보다 더 불규칙적임을 시사합니다. 또한 연간 주기의 배수 주파수(Harmonics)가 추정된 스펙트럼에서 명백하게 나타납니다.

![Autoregressive spectral estimator for the SOI series based on the AIC and based on the BIC, where different orders are selected as best](./images/fig7_11.jpg)

Figure 7.11: Autoregressive spectral estimator for the SOI series after detrending by lowess. AIC selects an AR(18) model, whereas BIC selects an AR(15). The periods from 2 to 7 years are highlighted. [Return to text.⏎](chapter7)

_p_의 각 값에 대한 AIC와 BIC가 [Figure 7.12](#chapter7#fig7_12)에 나타나 있습니다. p\=0인 값은 둘 다 200을 초과하기 때문에 표시되지 않았습니다. [Figure 7.12](#chapter7#fig7_12)에서 볼 수 있듯이 BIC는 어떤 모델을 선택할지에 대해 매우 확고합니다. 즉, 최소 BIC가 매우 뚜렷합니다. 반면 AIC는 그렇게 확정적이지 않으며, p\=15,16,17,30에서의 값들이 최솟값인 p\=18에 가깝습니다.

![AIC and BIC as a function of order p for the AR models fitted to the SOI series](./images/fig7_12.jpg)

Figure 7.12: Model selection criteria AIC and BIC as a function of order _p_ for autoregressive models fitted to the SOI series. The values displayed are shifted by subtracting the values of the best model (hence the best values are zero). The p\=0 values are excluded in the plot because they both exceed 200. [Return to text.⏎](chapter7)

이 분석을 수행하기 위해 spec.ic 명령을 사용하여 AIC 또는 BIC를 적용한 최고(Best) 모델을 맞출 수 있습니다. 이 스크립트는 기본적으로 결과적인 최소 AIC (또는 BIC) 스펙트럼 추정치를 플로팅하며, AIC와 BIC 값을 반환합니다. 또한 이 스크립트는 스펙트럼 분석 전에 lowess를 통한 추세 제거를 포함하여 데이터의 추세를 제거할 수 있게 해주며 여기서는 이를 사용합니다.


`par(mfrow=2:1)`
`**spec.ic**(**soi**, col=5, lwd=2, lowess=TRUE) -> u    _# min AIC spec_`
` rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.6,.2))`
` mtext("3/10", side=1, line=0, at=.3, cex=.75)`
` abline(v=.3, lty=2, col=8)    _# approximate El Niño Cycle_`
`**spec.ic**(**soi**, col=6, lwd=2, lowess=TRUE, BIC=TRUE)    _# min BIC spec_`
` rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.6,.2))`
` mtext("3/10", side=1, line=0, at=.3, cex=.75)`
` abline(v=.3, lty=2, col=8)`
`_# AIC/BIC plots_`
`**tsplot**(u[[1]][-1,1], u[[1]][-1,2:3], type="o", xlab="order", col=5:6,`
`   pch=c(19,17), ylab="AIC / BIC", cex=1.05, **nxm**=5, **spag**=TRUE, **addLegend**=TRUE,`
`   **location**="topleft")`

## 7.4 190\. Coherence and Cross-Spectra \*

상관관계 분석(Correlation analysis)이 교차 상관관계 분석(Cross-correlation analysis)으로 확장되는 것과 같은 방식으로 스펙트럼 분석도 다변량 시리즈(Multiple series)로 확장됩니다. 예를 들어, _xt_와 _yt_가 결합 정상성(Jointly stationary) 시리즈라면, 다음과 같이 주파수 기반의 측정 지표인 결합도(Coherence)를 도입할 수 있습니다.

교차 공분산(Cross-covariance) 함수가 절대적으로 합산 가능(Absolutely summable)하다고 가정하면,

γxy(h)\=E\[(xt+h−μx)(yt−μy)\]

191\. 는 다음과 같은 스펙트럼 표현을 가집니다.

γxy(h)\=∫−1/21/2 fxy(ω)e2πiωh dω,h\=0,±1,±2,…,(7.28)

여기서 교차 스펙트럼(Cross-spectrum)은 푸리에 변환으로 정의됩니다.

fxy(ω)\=∑h\=−∞∞γxy(h) e−2πiωh−12≤ω≤12.(7.29)


교차 공분산은 반드시 대칭(Symmetric)일 필요가 없기 때문에, 교차 스펙트럼은 일반적으로 복소수 값(Complex-valued) 함수이며 종종 다음과 같이 쓰입니다.

fxy(ω)\=cxy(ω)−iqxy(ω),

여기서

cxy(ω)\=∑h\=−∞∞γxy(h) cos(2πωh)

그리고

qxy(ω)\=∑h\=−∞∞γxy(h) sin(2πωh)

는 각각 코스펙트럼(Cospectrum)과 쿼드스펙트럼(Quadspectrum)으로 정의됩니다. γyx(h)\=γxy(−h) 관계 때문에 (7.29)로부터 다음이 성립합니다.

fyx(ω)\=fxy(ω)―.(7.30)

192\. 이 결과는 코스펙트럼과 쿼드스펙트럼이 다음을 만족함을 의미합니다.

cyx(ω)\=cxy(ω)andqyx(ω)\=−qxy(ω).

교차 스펙트럼 적용의 중요한 예시는 선형 필터(Linear filter) 관계를 통해 입력(Input) 시리즈 _xt_로부터 출력(Output) 시리즈 _yt_를 예측하는 문제입니다. 이러한 관계의 강도를 나타내는 척도가 결합도 함수(Coherence function)이며, 다음과 같이 정의됩니다.

ρyx2(ω)\=|fyx(ω)|2fxx(ω)fyy(ω),(7.31)

여기서 fxx(ω)와 fyy(ω)는 각각 _xt_와 _yt_ 시리즈의 개별 스펙트럼입니다. (7.31)이 다음과 같은 형태를 취하는 기존의 상관계수 제곱(Squared correlation)과 유사함에 유의해 주십시오.

ρyx2\=σyx2σx2σy2,

여기서 확률변수들은 분산 σx2, σy2 및 공분산 σyx\=σxy를 가집니다. 이는 결합도(Coherence)를 주파수 _ω_에서의 두 시계열 간의 상관계수 제곱으로 해석하도록 동기를 부여합니다.

Example 7.12 Three-Point Moving Average

간단한 예시로서, _xt_와 3-포인트 이동 평균(Three-point moving average) yt\=(xt−1+xt+xt+1)/3 간의 교차 스펙트럼을 계산합니다. 여기서 _xt_는 스펙트럼 밀도 fxx(ω)를 갖는 정상성 입력 과정(Stationary input process)입니다. 먼저,

γxy(h)\=cov(xt+h,yt)\=13 cov(xt+h,xt−1+xt+xt+1)\=13(γxx(h+1)+γxx(h)+γxx(h−1))\=13∫−1/21/2(e2πiω+1+e−2πiω)e2πiωhfxx(ω) dω\=13∫−1/21/2\[1+2cos(2πω)\]fxx(ω)e2πiωh dω,

여기서 (6.13)을 사용했습니다. 푸리에 변환의 유일성(Uniqueness)을 사용하여 (7.28)의 스펙트럼 표현으로부터 다음을 논증합니다.

fxy(ω)\=13 \[1+2cos(2πω)\] fxx(ω)

따라서 이 경우 교차 스펙트럼은 실수(Real)입니다. [Example 6.9](#chapter6#exam6_9)에서와 같이 _yt_의 스펙트럼 밀도는 다음과 같습니다.

fyy(ω)\=19\[3+4cos(2πω)+2cos(4πω)\]fxx(ω)\=19 \[1+2cos(2πω)\]2 fxx(ω),

마지막 단계에서 항등식 cos(2α)\=2cos2(α)−1을 사용했습니다. 이를 193\. (7.31)에 대입하면 _xt_와 _yt_ 사이의 결합도 제곱(Squared coherence)이 모든 주파수에 대해 1임을 얻습니다.

ρxy(ω)\=1−1/2≤ω≤1/2.

이는 보다 일반적인 선형 필터에서 물려받는 특성입니다.

벡터 시리즈(Vector series) xt\=(xt1,xt2,…,xtp)′에 대해 DFT 벡터 d(ωj)\=(d1(ωj),d2(ωj),…,dp(ωj))′를 사용할 수 있으며 스펙트럼 행렬을 다음과 같이 추정할 수 있습니다.

f¯(ω)\=L−1∑k\=−mmIp(ωj+k/n)(7.32)

여기서

Ip(ωj)\=d(ωj) d∗(ωj)

는 p×p 복소 행렬(Complex matrix)이며 \*는 켤레 전치(Conjugate transpose) 연산을 나타냅니다.

마찬가지로 (7.32)에서 DFT를 취하기 전에 시리즈에 테이퍼(Taper)를 적용할 수 있으며 다음과 같은 가중 추정(Weighted estimation)을 사용할 수 있습니다.

f^(ω)\=∑k\=−mmhk Ip(ωj+k/n)

여기서 {hk}는 (7.17)에 정의된 가중치입니다. 두 시리즈 _yt_와 _xt_ 사이의 결합도 제곱에 대한 추정치는 다음과 같습니다.

ρ^yx2(ω)\=|f^yx(ω)|2f^xx(ω)f^yy(ω).(7.33)

만약 (7.33)의 스펙트럼 추정치가 동일한 가중치를 사용하여 얻어졌다면, 이 추정치를 ρ¯yx2(ω)로 쓰겠습니다.

일반적인 조건 하에서, 만약 ρyx2(ω)\>0 이라면

|ρ^yx(ω)|∼⋅N(|ρyx(ω)|,(1−ρyx2(ω))2/2Lh)(7.34)

여기서 _Lh_는 (7.18)에 정의되어 있습니다. 이 결과에 대한 자세한 내용은 [Brockwell and Davis (2013, Ch 11)](#bibref1#refbib_8)에서 찾을 수 있습니다. (7.34)를 사용하여 결합도 ρyx2(ω)에 대한 근사적인 신뢰구간을 얻을 수 있습니다.

L\>1로 두어 추정치로 ρ¯yx2(ω)를 사용하면 ρyx2(ω)\=0 이라는 가설을 검정할 수 있습니다.[4](#chapter7#fn7_4) 즉,

ρ¯yx2(ω)\=|f¯yx(ω)|2f¯xx(ω)f¯yy(ω).(7.35)

이 경우 귀무가설(Null hypothesis) 하에서 다음과 같은 통계량은

F\=ρ¯yx2(ω)(1−ρ¯y⋅x2(ω))(L−1)(7.36)

_________________ 4L\=1 이면 ρ¯yx2(ω)≡1 입니다. [Return to text.⏎](#chapter7#fn7_47b)

194\. 2와 2L−2의 자유도를 갖는 근사적인 _F_\-분포를 따릅니다. 시리즈가 길이 _n_′으로 확장되었을 때는 2L−2를 df−2로 대체하며, 여기서 _df_는 (7.15)에 정의되어 있습니다. 특정 유의수준 _α_에 대해 (7.36)을 풀면 다음과 같은 결과가 도출됩니다.

Cα\=F2,2L−2(α)L−1+F2,2L−2(α)(7.37)

이 값은 사전에(A priori) 지정된 주파수에서 ρyx2(ω)\=0 을 기각할 수 있기 위해 원래의 결합도 제곱이 초과해야 하는 근사치(Approximate value)입니다.

Example 7.13 Coherence Between SOI and Recruitment

[Figure 7.13](#chapter7#fig7_13)은 스펙트럼에 사용된 것보다 더 넓은 대역에서 SOI 및 Recruitment 시리즈 간의 결합도를 보여줍니다. 이 경우 우리는 유의수준 α\=.001에서 L\=19, df\=2(19)(453/480)≈36 및 F2,df−2(.001)≈8.53 을 사용했습니다. 따라서 C.001\=.32 를 초과하는 ρ¯yx2(ω) 값에 대해 결합도가 없다는 가설을 기각할 수 있습니다. 이 방법은 _F_\-통계량이 근사치라는 사실 외에도, 본페로니 부등식(Bonferroni inequality)을 염두에 두고 모든 주파수에 걸쳐 결합도 제곱을 조사하고 있기 때문에 대략적인 수준(Crude)임을 강조합니다. [Figure 7.13](#chapter7#fig7_13)은 신뢰대역(Confidence bands)도 보여줍니다. 이러한 대역은 ρyx2(ω)\>0 인 _ω_에 대해서만 유효함을 강조합니다.

![Squared coherency between the SOI and Recruitment series and a confidence band](./images/fig7_13.jpg)

Figure 7.13: Squared coherency between the SOI and Recruitment series; L\=19, n\=453, n′\=480, and α\=.001. The horizontal line is _C_.001. [Return to text.⏎](chapter7)

이 경우 계절 주파수(Seasonal frequency)와 약 3년에서 7년 주기의 엘니뇨 주파수는 강하게 결합되어(Coherent) 있습니다. 다른 주파수들도 강하게 결합되어 있지만, 이 더 높은 주파수에서의 기저 파워 스펙트럼이 꽤 작기 때문에 강한 결합도가 덜 인상적입니다. 마지막으로, 계절 배수 주파수(Seasonal harmonic frequencies)에서 결합도가 지속적으로 나타남에 유의해 주십시오.

이 예제는 다음 명령어를 사용하여 재현할 수 있습니다. 단순 평균을 수행하기 위해 가중치를 kernel("daniell",9)로 지정합니다. 그렇지 않으면 195\. 양 끝에서 절반 가중치를 사용하는 수정된 다니엘 커널(Modified Daniell kernel)이 기본적으로 사용됩니다. 스크립트 kernel()은 _m_을 지정해야 합니다. 여기서 m\=9를 사용하여 L\=2m+1\=19 가 되도록 합니다.


`sr = **mvspec**(cbind(**soi**,**rec**), kernel=kernel("daniell",9), plot.type="coh",`
`   main="SOI & Recruitment", col=5, lwd=2)`
` Bandwidth: 0.475 | Degrees of Freedom: 35.86 | split taper: 0%`
`(f = qf(.999, 2, sr$df-2) )`
`  [1] 8.529792`
`(C = f/(18+f) )`
`  [1] 0.3215175`
`abline(h = C)`

## Problems

* 7.1. 관측된 시리즈 _xt_가 주기적 신호와 잡음으로 구성되어 다음과 같이 쓰일 수 있다고 가정합니다.  
xt\=β1cos(2πωkt)+β2sin(2πωkt)+wt,  
여기서 _wt_는 분산이 σw2인 백색 잡음(White noise) 과정입니다. 주파수 ωk≠0,12 는 알려져 있고 k/n 형태라고 가정합니다. 주어진 데이터 x1,…,xn에 대해 최소제곱법(Least squares)으로 _β_1, _β_2 및 σw2를 추정하는 것을 고려해 보십시오.  
   1. 고정된 _ωk_에 대해 최소제곱 회귀(Least squares regression) 계수가 다음과 같음을 보이기 위해 [Example 6.2](#chapter6#exam6_2)의 회귀 공식을 사용하십시오.  
   β^1\=2n−1/2dc(ωk)andβ^2\=2n−1/2ds(ωk),  
   여기서 dc(⋅)와 ds(⋅)는 (7.3)과 (7.4)에 주어진 코사인(Cosine) 및 사인(Sine) 변환입니다.  
   2. 오차제곱합(Error sum of squares)이 다음과 같이 쓰일 수 있음을 증명하십시오.  
   SSE\=∑t\=1nxt2−2Ix(ωk)  
   따라서 오차 제곱을 최소화하는 _ωk_ 값은 피리오도그램 추정량 (7.1)인 Ix(ωk)를 최대화하는 값과 동일함을 보이십시오.  
   3. 회귀제곱합(Sum of squares for the regression)이 다음과 같이 주어짐을 보이십시오.  
   SSR\=2Ix(ωk).  
   4. 가우시안(Gaussian) 가정 및 고정된 _ωk_ 하에서, 회귀가 없다는 _F_\-검정(F-test of no regression)이 Ix(ωk)의 단조 함수(Monotone function)인 _F_\-통계량으로 이어짐을 보이십시오.  
196\.
* 7.2. [Figure 7.14](#chapter7#fig7_14)는 1749년 6월부터 1978년 12월까지 연 2회 측정된 n\=459개의 데이터 포인트를 바탕으로 2년 단위로 평활화된(12개월 이동 평균) 흑점 수를 보여줍니다. 이 데이터는 **sunspotz**에 포함되어 있습니다. [Example 7.4](#chapter7#exam7_4)를 참고하여 두드러진 주기를 식별하는 피리오도그램 분석을 수행하고 신뢰구간을 구하십시오. 결과에 대해 해석하십시오.  
![Biyearly smoothed number of sunspots from June 1749 to December 1978](./images/fig7_14.jpg)  
Figure 7.14: Smoothed 12-month sunspot numbers (sunspotz) sampled twice per year. [Return to text.⏎](chapter7)
* 7.3. 토양 과학(Soil science)에서 발생한 것으로 알려진 행별 염분 농도 수준(Levels of salt concentration over rows)과 이에 상응하는 평균 온도 수준(Average temperature levels) 데이터가 **salt**와 **saltemp**에 있습니다. 두 시리즈를 플로팅한 다음 각각 별도의 스펙트럼 분석을 수행하여 주된 주파수(Dominant frequencies)를 식별하십시오. 신뢰구간을 포함하고 결과를 해석하십시오.
* 7.4. \* 다음과 같이 (7.2)를 검증하십시오.  
   1. [Section B.4](#appB#secB_4)의 결과를 사용하여 다음을 보이십시오.  
   ∑t\=1nx¯ e−2πiωjt\=0.  
   2. 다음으로,  
   d(ωj)\=∑t\=1n(xt−x¯) e−2πiωjt  
   가 [Definition 7.1](#chapter7#defi7_1)의 DFT와 동일함을 논증하십시오.  
   3. 마지막으로, h\=t−s라고 두고  
   I(ωj)\=|d(ωj)|2\=n−1∑t\=1n∑s\=1n(xt−x¯)(xs−x¯)e−2πiωj(t−s)\=n−1∑h\=−(n−1)n−1∑t\=1n−|h|(xt+|h|−x¯)(xt−x¯)e−2πiωjh,  
   가 (7.2)에 주어진 피리오도그램의 형태임을 보이십시오.
* 7.5. 197\. 비모수적(Nonparametric) 스펙트럼 추정 절차를 사용하여 연어 가격(Salmon price) 데이터(**salmon**)를 분석하십시오. [Example 3.12](#chapter3#exam3_12)에서 발견된 뚜렷한 연간 주기 외에 어떤 다른 흥미로운 주기들이 드러납니까?
* 7.6. 비모수적 스펙트럼 추정 절차를 사용하여 [Problem 7.2](#chapter7#question7_2)를 반복하십시오. 결과를 자세히 논의하는 것 외에도, 평활화(Smoothing) 및 테이퍼링(Tapering)과 관련하여 귀하가 선택한 스펙트럼 추정량에 대해 설명하십시오.
* 7.7. 비모수적 스펙트럼 추정 절차를 사용하여 [Problem 7.3](#chapter7#question7_3)을 반복하십시오. 결과를 자세히 논의하는 것 외에도, 평활화 및 테이퍼링과 관련하여 귀하가 선택한 스펙트럼 추정량에 대해 설명하십시오.
* 7.8. 흔히 흑점(Sunspot) 시리즈의 주기성은 충분히 높은 차수(Order)의 자기회귀(Autoregressive) 스펙트럼을 맞춤으로써 조사됩니다. 주된 주기성은 대개 11년 부근이라고 알려져 있습니다. 선택한 모델 선택 방법(Model selection method)을 사용하여 흑점 데이터에 자기회귀 스펙트럼 추정량을 맞추십시오. 결과를 [Problem 7.6](#chapter7#question7_6)에서 찾은 일반적인 비모수적 스펙트럼 추정량과 비교하십시오.
* 7.9. 이 연습 문제에서는 파운드(Pound)당 미국 센트(U.S. cents)로 표시된 닭 통육 현물 가격(Whole bird spot price)인 **chicken** 파일의 데이터를 사용합니다.  
   1. 데이터 셋을 플로팅하고 관찰한 내용을 설명하십시오. 왜 여기서 차분(Differencing)이 합리적입니까?  
   2. 비모수적 스펙트럼 추정치를 사용하여 차분된 닭 가격 데이터를 분석하고 결과를 설명하십시오.  
   3. 모수적(Parametric) 스펙트럼 추정 절차를 사용하여 이전 부분을 반복하고 그 결과를 이전 부분과 비교하십시오.
* 7.10. Recruitment 시리즈에 자기회귀 스펙트럼 추정량을 맞추고 [Example 7.7](#chapter7#exam7_7)의 결과와 비교하십시오.
* 7.11. 반향(Echoes)에 의해 유발된 시계열의 주기적 거동은 시계열의 스펙트럼에서도 관찰될 수 있습니다. 이 사실은 [Problem 6.6](#chapter6#question6_6)에 언급된 결과에서 확인할 수 있습니다. 해당 문제의 표기법을 사용하여 xt\=st+Ast−D+nt 를 관측한다고 가정해 봅니다. 이는 스펙트럼이 fx(ω)\=\[1+A2+2Acos(2πωD)\]fs(ω)+fn(ω) 를 만족함을 암시합니다. 잡음을 무시할 수 있다면 (fn(ω)≈0), logfx(ω)는 주기적 성분 log\[1+A2+2Acos(2πωD)\] 와 logfs(ω) 의 합과 근사적으로 일치합니다. [Bogert et al. (1963)](#bibref1#refbib_4)는 추세가 제거된 로그 스펙트럼(Detrended log spectrum)을 가상의(Pseudo) 시계열로 간주하고 그것의 스펙트럼, 즉 켑스트럼(Cepstrum)을 계산하는 방법을 제안했습니다. 이 켑스트럼은 _D_에 해당하는 켑스트럼 상의 큐프렌시(Quefrency)에서 피크를 보여야 합니다. 켑스트럼은 큐프렌시의 함수로 플로팅될 수 있으며 이로부터 지연(Delay) _D_를 추정할 수 있습니다.  
**speech**에 제시된 음성 시리즈에 대해 다음과 같이 켑스트럴 분석(Cepstral analysis)을 사용하여 피치(Pitch) 주기를 추정하십시오.  
   1. 198\. 데이터의 로그-피리오도그램(Log-periodogram)을 계산하고 표시하십시오. 피리오도그램이 예상대로 주기적인가요?  
   2. 추세가 제거된 로그-피리오도그램에 대해 켑스트럴(스펙트럼) 분석을 수행하고, 그 결과를 사용하여 지연 _D_를 추정하십시오.
* 7.12.\* [Problem 7.3](#chapter7#question7_3)에서 논의된 온도와 염분 데이터 간의 결합도(Coherency)를 분석하십시오. 결과를 논의하십시오.
* 7.13.\* 다음과 같은 두 과정을 고려해 보십시오.  
xt\=wtandyt\=ϕxt−D+vt  
여기서 _wt_와 _vt_는 공통 분산 _σ_2를 갖는 독립적인 백색 잡음(White noise) 과정이며, _ϕ_는 상수이고, _D_는 고정된 정수 지연(Fixed integer delay)입니다.  
   1. _xt_와 _yt_ 사이의 결합도를 계산하십시오.  
   2. ϕ\=.9, σ2\=1, D\=0 인 경우에 대해 _xt_와 _yt_로부터 n\=1024 개의 정규 관측치(Normal observations)를 시뮬레이션(Simulate)하십시오. 그런 다음 _L_의 다음 값들에 대해 시뮬레이션된 시리즈 간의 결합도를 추정하여 플로팅하고 코멘트를 남기십시오. (i) L\=1, (ii) L\=3, (iii) L\=41, 및 (iv) L\=101.

---

