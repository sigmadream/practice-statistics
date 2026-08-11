# 9장. 고급 도구 및 다음 단계 (Chapter 9. Advanced Tools and Next Steps)

이 책은 Python과 R을 사용한 풋볼 분석의 기초에 중점을 두었습니다. 우리는 개인적으로 이 두 가지 언어를 정기적으로 사용합니다. 그러나 우리는 이 두 가지 프로그래밍 언어 외의 도구도 사용합니다. 계속해서 성장하고자 하는 사람들은 자신만의 안전지대(comfort zone)에서 벗어나야 합니다. 이 장에서는 우리가 사용하는 다른 도구들에 대한 개요를 제공합니다. 우리가 사용하지만 아직 언급하지 않은 모델링 도구부터 시작할 텐데, 그 이유는 주제가 너무 고급이거나 여러분이 코딩을 따라 할 수 있을 만큼 대중적인 데이터를 쉽게 찾을 수 없었기 때문입니다.

그런 다음 컴퓨터 도구로 넘어갑니다. 이 주제들은 분리되어 있으면서도 동시에 얽혀 있습니다. 한 가지 기술을 독립적으로 배울 수도 있지만, 종종 한 가지 기술을 사용할 때 다른 기술들과 함께 사용하는 것이 가장 효과적입니다. 풋볼에 비유하자면, 라인배커는 런 플레이를 방어하고, 패서를 압박하며, 패스 루트를 달리는 선수를 커버할 수 있어야 하는데, 이는 한 경기의 동일한 시리즈 내에서 이루어지는 경우가 많습니다. 일부 기술(플레이를 읽는 능력)과 선수의 특성(스피드)은 세 가지 라인배커 상황 모두에 도움이 되지만, 종종 개별적으로 훈련됩니다. 가장 가치 있는 선수는 이 세 가지 모두에 뛰어납니다.

이 장은 데이터 과학자로서의 우리 경험과 리차드가 천연자원 관리자를 위해 쓴 글([“Paths to Computational Fluency for Natural Resource Educators, Researchers, and Managers”](https://oreil.ly/oNokn))을 기반으로 합니다. 우리가 제시하는 순서대로 주제를 배우는 것을 제안하며, 그 이유는 <a href="#tbl-adv-tools" data-type="xref">표 9-1</a>에 나열되어 있습니다. 특정 기술에 대해 어느 정도 익숙해지면 다른 영역으로 넘어가세요. 결국 해당 기술 영역으로 다시 돌아와서 그 영역에서 어떻게 성장할지 알게 될 것입니다. 더 많은 기술을 배울수록 새로운 기술을 배우는 데 더 능숙해질 수 있습니다!

###### 팁 (Tip)

에밀리 로빈슨(Emily Robinson)과 재클린 놀리스(Jacqueline Nolis)는 _Build a Career in Data Science_ (Manning, 2020)에서 데이터 과학 경력을 위한 광범위한 기술을 다룹니다.

| 도구 (Tool)                                          | 이유 (Reason)                                                                                                                          | 예시 (Examples)                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 명령줄 (Command line)                                | 운영 체제(OS)에서 효율적이고 자동으로 작업하고, 명령줄 전용 도구를 사용합니다.                                                         | Microsoft PowerShell, bash, Zsh                                         |
| 버전 관리 (Version control)                          | 코드 변경 사항을 추적하고, 코드에 대해 협업하며, 코드를 공유 및 게시합니다.                                                            | Git, Apache Subversion (SVN), Mercurial                                 |
| 린팅 (Linting)                                       | 코드를 정리하고, 스타일의 내부적 일관성을 제공하며, 오류를 줄이고 품질을 향상시킵니다.                                                 | Pylint, lintr, Black                                                    |
| 패키지 생성 및 호스팅 (Package creation and hosting) | 자신만의 코드를 재사용하고, 내부 또는 외부에서 코드를 공유하며, 코드를 더 쉽게 유지 관리합니다.                                        | Conda, pip, CRAN                                                        |
| 환경 (Environment)                                   | 재현 가능한(reproducible) 결과를 제공하고, 모델을 프로덕션이나 클라우드로 가져가며, 협업 시 동일한 도구를 사용할 수 있도록 보장합니다. | Conda, Docker, Poetry                                                   |
| 대화형 도구 및 보고서 (Interactives and reports)     | 코딩 방법을 모르는 사람들도 데이터를 탐색할 수 있도록 하고, DevOps 팀에 넘기기 전에 도구의 프로토타입을 만듭니다.                      | Jupyter Notebook, Shiny, Quarto                                         |
| 클라우드 (Cloud)                                     | 도구와 고급 컴퓨팅 리소스를 배포하고, 데이터를 공유합니다.                                                                             | Amazon Web Services (AWS), Microsoft Azure, Google Cloud Platform (GCP) |

표 9-1. 고급 도구, 사용 이유 및 예시 제품 {#tbl-adv-tools style="width: 100%"}

###### 팁 (Tip)

우리가 언급하는 모든 고급 도구에는 도구에 포함되어 있거나 온라인에서 무료로 제공되는 설명서가 있습니다. 그러나 이 설명서를 찾아서 사용하는 것은 어려울 수 있으며, 종종 거친 원석 속에서 다이아몬드를 찾는 듯한 과정이 필요합니다. 따라서 유료 튜토리얼과 책 같은 리소스는 항상 그런 것은 아니지만 양질의 제품을 제공하는 경우가 많습니다. 만약 자금에 여유가 없는 대학원생이라면, 무료 리소스들을 파헤쳐 보석을 찾는 데 시간을 보내고 싶을 것입니다. 아이들이 있고 시간이 많지 않은 직장인이라면, 아마도 비용을 지불하고 학습 리소스를 이용하고 싶을 것입니다. 기본적으로 양질의 학습 자료를 찾는 것은 시간과 돈의 문제입니다.

# 고급 모델링 도구 (Advanced Modeling Tools)

이 책 내에서 우리는 다양한 범위의 모델을 다루었습니다. 많은 사람들에게 이 도구들만으로도 풋볼 분석 수준을 높이는 데 충분할 것입니다. 그러나 일부 사람들은 더 나아가 한계를 넘고 싶어 할 것입니다. 이 섹션에서는 우리가 정기적으로 사용하는 몇 가지 방법을 설명합니다. 이러한 주제들 중 상당수가 서로 얽혀 있다는 점에 유의하세요. 따라서 하나의 주제를 배우면 다른 주제도 배울 수 있습니다.

## 시계열 분석 (Time Series Analysis)

풋볼 데이터, 특히 경기 내의 특징(feature)이 풍부한 고해상도 데이터는 시간의 흐름에 따른 추세를 살펴보기에 적합합니다. _시계열 분석(Time series analysis)_ 은 시간에 따른 추세를 추정합니다. 이 방법론은 금융에서 흔히 사용되며, 생태학, 물리학, 사회 과학과 같은 다른 분야에서도 활용됩니다. 기본적으로 이러한 모델은 미래 예측을 위해 과거 관측치가 중요할 때(자기 상관(_auto-correlations_)이라고도 함) 더 나은 추정치를 제공할 수 있습니다. 다음은 우리가 유용하다고 생각한 리소스입니다.

- 로버트 H. 슘웨이(Robert H. Shumway)와 데이비드 S. 스토퍼(David S. Stoffer)의 _Time Series Analysis and Its Application_, 제4판(Springer, 2017). R을 활용한 시계열 분석에 대한 자세한 소개를 제공합니다.

- 에일린 닐슨(Aileen Nielsen)의 <a href="https://learning.oreilly.com/library/view/practical-time-series/9781492041641" class="orm:hideurl"><em>Practical Time Series Analysis</em></a> (O’Reilly, 2019)는 활용, 특히 기계 학습에 적용하는 부분에 초점을 맞추어 시계열 분석에 대한 더 부드러운 소개를 제공합니다.

- Facebook 핵심 데이터 과학 팀의 [Prophet](https://oreil.ly/nPbvF)은 올바르게 사용하면 매우 강력할 수 있는 시계열 모델링 도구입니다.

## PCA를 넘어서는 다변량 통계 (Multivariate Statistics Beyond PCA)

<a href="ch08.html#sec-pca-clus" data-type="xref">8장</a>에서는 PCA 및 클러스터링과 같은 다변량 방법에 대해 간략하게 소개했습니다. 이 두 가지 방법은 빙산의 일각에 불과합니다. 다변량 예측 변수와 반응 변수 모두를 허용하는 중복 분석(redundancy analysis, RDA)과 같은 다른 방법들도 존재합니다. 이러한 방법은 고유한 예측 그룹을 찾기 때문에 많은 입문용 비지도 학습(unsupervised learning) 방법의 기반을 형성합니다. 또한 PCA는 유클리드 거리(_Euclidean distance_)(피타고라스 정리에서 기억할지 모르겠지만 동일한 거리입니다. 예를 들어 2차원에서는 $\sqrt{a^{2} + b^{2}}$입니다)를 가정합니다. 다른 유형의 거리도 존재하며, 다변량 방법은 이를 다룹니다. 마지막으로 많은 분류 방법이 존재합니다. 예를 들어 일부 다변량 방법은 동적 요인 분석(dynamic factor analysis, DFA)과 같은 시계열 분석으로 확장됩니다.

이러한 도구를 직접 적용하는 것 외에도, 이 방법론을 이해하면 기계 학습 도구를 배우고자 할 때 탄탄한 기반을 마련할 수 있습니다. 우리가 배웠거나 도움이 될 것이라고 생각하는 책은 다음과 같습니다.

- 피에르 르장드르(Pierre Legendre)와 루이 르장드르(Louis Legendre)의 _Numerical Ecology_, 제3판(Elsevier, 2012)은 많은 다변량 방법에 대해 일반적으로 이해하기 쉬운 개요를 제공합니다.

- E. 홈스(E. Holmes), M. 슈어렐(M. Scheuerell), E. 워드(E. Ward)의 _Analysis of Multivariate Time Series Using the MARSS Package_ 비네트(vignette)는 MARSS 패키지와 함께 제공되며 [MARSS CRAN 페이지](https://oreil.ly/SQ2qZ)에서 찾을 수 있습니다. 이 자세한 입문서는 R을 사용하여 다변량 데이터에서 시계열 분석을 수행하는 방법을 설명합니다.

## 분위수 회귀 (Quantile Regression)

일반적으로 회귀(regression)는 평균적인(average or mean) 예상 가치를 모델링합니다. 분위수 회귀(_Quantile regression_)는 분포의 다른 부분, 특히 사용자가 지정한 분위수(quantile)를 모델링합니다. 상자 수염 그림(<a href="ch02.html#sec-boxplot" data-type="xref">“Boxplots”</a>에서 다룸)이 사전 정의된 분위수를 갖는 반면, 사용자는 분위수 회귀를 사용하여 원하는 분위수를 지정합니다. 예를 들어 NFL 스카우팅 콤바인 데이터를 살펴볼 때 시간이 지남에 따라 선수 속도가 어떻게 변하는지 궁금할 수 있습니다. 기존의 다중 회귀 분석(multiple regression)은 시간이 지남에 따라 평균적인 선수를 살펴봅니다. 반면 분위수 회귀 분석은 빠른 선수들이 시간이 지남에 따라 더 빨라지는지 확인하는 데 도움이 될 것입니다. 분위수 회귀는 <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a>의 NFL 드래프트 데이터를 살펴볼 때도 유용합니다. 분위수 회귀 학습을 위한 리소스로는 패키지 문서가 있습니다.

- R의 [`quantreg` 패키지](https://oreil.ly/diL03)에는 프로그래밍 언어와 상관없이 훌륭하게 작성된 21페이지 분량의 [비네트(vignette)](https://oreil.ly/P4ZVs)가 있습니다.

- Python의 `statsmodels` 패키지에도 분위수 회귀 [문서](https://oreil.ly/0ha7Z)가 있습니다.

## 베이지안 통계 및 계층적 모델 (Bayesian Statistics and Hierarchical Models)

지금까지 확률의 사용은 장기적 발생(long-term occurrence) 또는 사건이 발생하는 빈도에 따라 사건이 발생한다는 가정하에 이루어졌습니다. 이러한 유형의 확률을 빈도주의자(_frequentist_) 통계라고 합니다. 그러나 확률에 대한 다른 관점도 존재합니다.

특히, 베이지안 관점은 믿음의 정도 또는 확실성 측면에서 세상을 봅니다. 예를 들어, 평균에 대한 빈도주의자(frequentist)의 95% 신뢰 구간(confidence interval, CI)은 관찰을 아주아주 여러 번 반복할 경우 95%의 시간 동안 평균이 이 구간에 포함됨을 의미합니다. 반대로 베이지안 95% 신용 구간(credible interval, CrI)은 평균이 95% 확률로 포함될 것이라고 확신하는 범위를 나타냅니다. 미묘하지만 중요한 차이입니다.

베이지안 관점은 시스템에 대한 사전 지식(prior understanding)에서 시작하여 관측된 데이터를 사용하여 그 지식을 업데이트한 다음 사후 분포(posterior distribution)를 생성합니다. 실제로 베이지안 방법은 세 가지 주요 이점을 제공합니다.

- 다른 방법으로는 데이터가 충분하지 않을 때 더 복잡한 모델을 피팅할 수 있습니다.

- 여러 정보 출처를 더 쉽게 포함시킬 수 있습니다.

- 통계에 대한 베이지안의 관점은 통계의 이름은 몰라도 많은 사람이 가지고 있는 관점입니다.

예를 들어, 어느 팀이 이길지 선택한다고 가정해 보겠습니다. 사전 정보(prior information)는 다른 데이터, 가장 최선의 추측, 또는 기타 모든 출처에서 나올 수 있습니다. 만약 여러분이 겸손하다면 50%의 확률로 자신이 맞을 것이라 생각하고, 두 경기를 맞추고 두 경기는 틀릴 것이라 예상할 것입니다. 자신이 과신(overconfident)한다면 80%의 확률로 자신이 맞을 것이라 생각하고 여덟 경기를 맞추고 두 경기를 틀릴 것이라고 생각할 것입니다. 자신이 과소평가(underconfident)한다면 20%의 확률로 자신이 맞을 것이라 생각하고 여덟 경기를 틀리고 두 경기를 맞출 것이라고 생각할 것입니다. 이것이 여러분의 사전 분포(prior distribution)입니다.

이 예제에서 베타 분포(beta distribution)는 "성공" 횟수와 "실패" 횟수를 주어졌을 때 확률 분포를 나타냅니다. 그래픽으로 나타내면 <a href="#fig-prior-prob" data-type="xref">그림 9-1</a>이 나옵니다.

<figure>
<img src="assets/fapr_0901.png" />
<h6 id="figure-9-1.-prior-distribution-for-predicting-results-of-games">그림 9-1. 경기 결과 예측을 위한 사전 분포(prior distribution)</h6>
</figure>

50경기를 관찰한 결과, 30경기는 맞았고 20경기는 틀렸을 수 있습니다. 빈도주의자는 여러분이 60% 확률로(30/50) 맞는다고 말할 것입니다. 베이지안에게 이것은 관찰된 우도(observed likelihood)입니다. 베타 분포의 경우 60번의 성공과 40번의 실패가 될 것입니다. <a href="#fig-prior-likelihood" data-type="xref">그림 9-2</a>는 우도 확률(likelihood probability)을 보여줍니다.

다음으로, 베이지안은 사후 분포(posterior)인 <a href="#fig-prior-posterior" data-type="xref">그림 9-3</a>을 생성하기 위해 <a href="#fig-prior-prob" data-type="xref">그림 9-1</a>에 <a href="#fig-prior-likelihood" data-type="xref">그림 9-2</a>를 곱합니다. 세 가지 추측 모두 비슷하지만, 사전 분포가 사후 분포에 영향을 줍니다.

<figure>
<img src="assets/fapr_0902.png" />
<h6 id="figure-9-2.-likelihood-distribution-for-predicting-results-of-games">그림 9-2. 경기 결과 예측을 위한 우도 분포(likelihood distribution)</h6>
</figure>

<figure>
<img src="assets/fapr_0903.png" />
<h6 id="figure-9-3.-in-this-posterior-distribution-for-predicting-results-of-games-notice-the-influence-of-the-prior-distribution-on-the-posterior">그림 9-3. 경기 결과 예측을 위한 사후 분포(posterior distribution)에서 사전 분포가 사후 분포에 미치는 영향 확인</h6>
</figure>

이 단순한 예제는 쉬운 문제에 대해 베이지안 방법론이 어떻게 작동하는지를 보여줍니다. 그러나 베이지안 모델은 다수준(multi-level) 모델과 같이 훨씬 더 복잡한 모델을 피팅할 수도 있습니다(팀 단위 및 선수 단위 특성을 모두 포함하는 회귀 분석 검토). 또한 베이지안 모델의 사후 분포는 다른 추정 방법에는 없는 불확실성을 포착합니다. 베이지안처럼 생각하거나 베이지안 통계를 수행하는 방법에 대해 더 알고 싶은 분들을 위해 유용한 책 몇 권을 소개합니다.

- 샤론 버치 맥그레인(Sharon Bertsch McGrayne)의 _The Theory That Would Not Die: How Bayes’ Rule Cracked the Enigma Code, Hunted Down Russian Submarines, and Emerged Triumphant from Two Centuries of Controversy_ (Yale University Press, 2011)는 잃어버린 핵무기와 잠수함을 찾는 미 해군과 같은 세간의 이목을 끄는 예제를 통해 사람들이 어떻게 베이지안 통계를 사용하여 결정을 내렸는지 설명합니다.

- 레너드 J. 새비지(Leonard J. Savage)의 _The Foundations of Statistics_ (Dover Press, 1972)는 특히 베팅이나 관리와 같은 의사 결정의 맥락에서 베이지안처럼 생각하는 방법에 대한 개요를 제공합니다.

- 존 크루슈케(John Kruschke)의 _Doing Bayesian Data Analysis_, 제2판 (Elsevier, 2014)은 표지 때문에 강아지 책(_puppy book_)으로도 알려져 있습니다. 이 책은 베이지안 통계에 대한 부드러운 소개를 제공합니다.

- 앤드류 겔만(Andrew Gelman) 외 공저 _Bayesian Data Analysis_, 제3판(CRC Press, 2013). Stan 사용자들에게 *BDA3*로 자주 불리는 이 책은 베이지안 방법에 대해 엄밀하고 상세한 설명을 제공합니다. 리차드는 2년 과정의 대학원 입문 수준의 고급 학부 수학 과목을 수강하고 나서야 이 책을 읽을 수 있었습니다.

- 리차드 맥엘레스(Richard McElreath)의 _Statistical Rethinking: A Bayesian Course with Examples in R and Stan_, 제2판(CRC Press, 2020). 이 책은 내용의 엄밀성 측면에서 강아지 책(puppy book)과 BDA3 사이에 있으며, 베이지안 통계를 배우고 싶어 하는 사람들을 위한 중급 수준의 텍스트입니다.

## 생존 분석/사건 발생 시점 분석 (Survival Analysis/Time-to-Event)

쿼터백은 패스를 던지거나 색(sack)을 당할 때까지 포켓 안에서 얼마나 오래 버틸 수 있을까요? _사건 발생 시점(Time-to-event)_ 또는 _생존(survival)_ 분석이 이러한 질문에 답하는 데 도움을 줄 것입니다. 이 분석을 위한 퍼블릭 데이터를 찾을 수 없었기 때문에 이 책에서는 이 기법을 다루지 않았습니다. 그러나 더 자세한 시간 데이터를 보유한 사람들의 경우, 이 분석이 이벤트가 발생하기까지의 시간을 이해하는 데 도움이 될 것입니다. 이 주제와 관련하여 우리가 유용하다고 생각한 책은 다음과 같습니다.

- 프랭크 E. 하렐 주니어(Frank E. Harrell Jr.)의 _Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis_, 제2판(Springer, 2015). 이 책은 회귀(regression) 외에도 생존 분석에 대한 내용을 포함하고 있어 유용합니다.

- 앨런 B. 다우니(Allen B. Downey)의 <a href="https://learning.oreilly.com/library/view/think-stats-2nd/9781491907344/" class="orm:hideurl"><em>Think Stats</em>, 2nd edition</a> (O’Reilly, 2014)는 Python을 활용한 이해하기 쉬운 생존 분석 장을 포함하고 있습니다.

## 베이지안 네트워크/구조 방정식 모델링 (Bayesian Networks/Structural Equation Modeling)

<a href="ch08.html#sec-pca-clus" data-type="xref">8장</a>에서는 데이터의 상호 연결성(interconnectedness)에 대해 암시했습니다. 이를 한 단계 더 발전시켜 보면, 데이터에 명확한 원인과 결과가 없거나 원인과 결과 변수가 연결되어 있는 경우가 있습니다. 예를 들어 콤바인 드래프트 속성을 생각해 보세요. 선수의 몸무게는 선수의 달리기 속도와 연관이 있을 수 있습니다(더 가벼운 선수가 더 빨리 달립니다). 달리기 속도와 몸무게는 모두 러닝백의 러싱 야드와 관련이 있을 수 있습니다.

이러한 교란 변수(confounding variables)를 어떻게 분리할 수 있을까요? 구조 방정식 모델링(structural equation modeling)과 베이지안 네트워크(Bayesian networks) 같은 도구를 사용하면 이러한 관계를 추정할 수 있습니다. 다음은 우리가 유용하다고 생각한 책입니다.

- 주데아 펄(Judea Pearl)과 다나 맥켄지(Dana Mackenzie)의 _The Book of Why_ (Basic Books, 2018)는 네트워크 측면에서 세상을 생각하는 방법을 안내합니다. 이 책은 또한 네트워크 모델에 대한 훌륭한 개념적 소개를 제공합니다.

- 마르코 스쿠타리(Marco Scutari)와 장 밥티스트 드니스(Jean-Baptiste Denis)의 _Bayesian Networks With Examples in R_, 제2판(CRC Press, 2021)은 베이지안 네트워크에 대한 훌륭한 소개를 제공합니다.

- 제임스 B. 그레이스(James B. Grace)의 _Structural Equation Modeling and Natural Systems_ (Cambridge University Press, 2006)는 생태학 데이터를 활용한 이러한 모델에 대한 부드러운 소개를 제공합니다.

## 기계 학습 (Machine Learning)

_기계 학습(Machine learning)_ 은 단일 도구가 아니라 오히려 도구의 모음이자 데이터에 대해 사고하는 방식입니다. 우리 책의 대부분은 데이터의 통계적 이해에 중점을 두었습니다. 반면 기계 학습은 자동화된 방식으로 예측하기 위해 데이터를 사용하는 방법에 대해 생각합니다.

이 주제에 관한 훌륭한 책은 많지만, 특별히 추천할 만한 것은 없습니다. 대신 수학, 통계 및 프로그래밍에 대한 탄탄한 기초를 쌓으면 잘 이해할 수 있는 준비가 될 것입니다.

# 명령줄 도구 (Command Line Tools)

명령줄(_Command lines_)을 사용하면 코드를 통해 컴퓨터와 상호 작용할 수 있습니다. 명령줄에는 몇 가지 관련 명칭이 있으며, 각각 특정한 기술적 정의가 있지만 종종 번갈아 가며 사용됩니다. 그중 하나는 셸(_shell_)인데, 이는 인간(여러분과 같은)이 닿는 운영 체제의 외부 또는 "껍데기(shell)"이기 때문입니다. 다른 하나는 터미널(_terminal_)인데, 이는 입력 및 출력 텍스트를 사용하는 소프트웨어이기 때문입니다(역사적으로, 터미널은 사용자가 사용하는 하드웨어를 의미했습니다). 더 현대적인 정의로는 소프트웨어 자체를 지칭할 수도 있습니다. 예를 들어, 리차드의 Linux 컴퓨터는 명령줄 애플리케이션을 터미널(_terminal_)이라고 부릅니다. 마지막으로 콘솔(_console_)은 물리적 터미널을 지칭합니다. [Ask Ubuntu 사이트](https://oreil.ly/ask)는 몇 가지 그림 예시, 특히 [이 답변](https://oreil.ly/NR501)과 함께 이 주제에 대한 자세한 논의를 제공합니다.

이러한 명령줄 도구는 오래되었지만(예를 들어 Unix 개발은 1960년대 후반에 시작되었습니다), 그 강력함 덕분에 사람들은 여전히 이를 사용하고 있습니다. 예를 들어, 수천 개의 파일을 삭제하려면 마우스로 여러 번 클릭해야 하지만 명령줄에서는 한 줄의 코드만 있으면 됩니다.

처음 시작할 때 명령줄은 Python이나 R을 처음 시작할 때처럼 헷갈릴 수 있습니다. 마찬가지로 명령줄을 사용하는 것은 풋볼에서 러닝 훈련이나 협응(coordination) 훈련처럼 가장 기초적인 기술입니다. 명령줄은 우리가 나열하는 대부분의 고급 기술과 함께 사용되며, R이나 Python과 같은 언어에 대한 이해도 향상시킬 것입니다. 예를 들어 명령줄을 이해하면 컴퓨터 운영 체제가 작동하는 방식과 파일 구조에 대해 생각하게 함으로써 프로그래밍 언어에 대한 이해가 깊어질 것입니다. 그런데 어떤 명령줄을 사용해야 할까요?

우리는 두 가지 옵션을 고려해 볼 것을 제안합니다. 첫째, _본 어게인 셸(Bourne Again Shell)_ (이전의 본 셸을 대체한다는 의미에서 이름 붙여졌으며, 다시 셸의 창시자인 스티븐 본(Stephen Bourne)의 이름을 따서 명명됨. 줄여서 _bash_)은 전통적으로 Linux 및 macOS의 기본 셸이었습니다. 이 셸은 이제 Windows에서도 사용할 수 있으며 AWS, Microsoft Azure, GCP와 같은 클라우드 컴퓨터나 고성능 슈퍼컴퓨터의 기본 설정인 경우가 많습니다. 십중팔구는 여러분도 bash 셸로 시작하게 될 것입니다.

두 번째 옵션은 Microsoft PowerShell입니다. 역사적으로 이는 Windows 전용이었지만, 이제는 다른 운영 체제에서도 사용할 수 있습니다. 기업 환경에서 IT(정보 기술) 관련 업무를 많이 수행한다면 PowerShell을 배우는 것이 가장 좋은 선택일 것입니다. PowerShell에 포함된 도구는 보안 업데이트나 소프트웨어 설치와 같은 업무의 일부를 자동화하는 데 도움이 될 수 있습니다.

macOS나 Linux를 사용하는 경우 이미 bash 또는 bash 복제 터미널이 있습니다(macOS는 저작권 문제로 인해 Zsh 셸 언어를 사용하는 것으로 변경했지만, Zsh와 bash는 이 책의 기본 예제를 포함하여 많은 상황에서 상호 교환이 가능합니다). 컴퓨터에서 터미널 앱을 열고 따라 하기만 하면 됩니다. Windows를 사용하는 경우 가벼운(lightweight) bash 셸이 함께 제공되는 [Git for Windows](https://gitforwindows.org)를 다운로드하는 것이 좋습니다. bash의 유용성을 발견한 Windows 사용자는 결국 Linux용 Windows 하위 시스템(WSL, Windows Subsystem for Linux)으로 이동하고 싶을 수 있습니다. 이 프로그램은 Windows 컴퓨터에서 강력하고 완전한 버전의 Linux를 제공합니다.

## Bash 예제 (Bash Example)

터미널 인터페이스는 컴퓨터의 파일 구조에 대해 생각하도록 강제합니다. 터미널을 열 때 **`pwd`** 를 입력하여 현재 작업 디렉터리를 화면에 출력(print)합니다. 예를 들어 Pop!\_OS(Linux의 한 종류)를 실행하는 리차드의 Linux 컴퓨터에서는 다음과 같습니다.

```
(base) raerickson@pop-os:~$ pwd
/home/raerickson
```

여기서 */home/raerickson*은 현재 작업 디렉터리입니다. 작업 디렉터리에 있는 파일들을 확인하려면 목록 명령어인 **`ls`** 를 입력하세요(명령어를 기억하기 쉬운 방법으로 **`ls`** 가 *list stuff*의 줄임말이라고 생각할 수 있습니다).

raerickson@pop-os:~\$` `ls ` `Desktop` `Games` `Public ` `Documents` `R` `Untitled.ipynb ` `Downloads` `miniconda3` `Videos ` `Firefox_wallpaper.png` `Music` `Templates ` `Pictures` `test.py

리차드의 사용자 디렉터리에 있는 모든 디렉터리와 파일들을 볼 수 있습니다. 파일 경로 또한 중요합니다. 알아두어야 할 세 가지 기본 경로는 현재 디렉터리, 컴퓨터의 홈 디렉터리, 그리고 한 단계 위(상위) 디렉터리입니다.

- `./`는 현재 디렉터리입니다.

- `~/`는 컴퓨터의 기본 홈 디렉터리입니다.

- `../`는 이전 디렉터리입니다.

예를 들어 현재 디렉터리가 */home/raerickson*이라고 가정해 보겠습니다. 이 예시에서 디렉터리 구조는 다음과 같습니다.

- `../`는 _home_ 디렉터리가 됩니다.

- `./`는 _raerickson_ 디렉터리가 됩니다.

- `/`는 컴퓨터의 가장 낮은 레벨(루트 디렉터리)이 됩니다.

- `~/`는 리차드의 컴퓨터에서는 */home/raerickson*인 기본 홈 디렉터리가 됩니다.

###### 참고 (Note)

실용적인 목적에서 디렉터리(_directory_)와 폴더(_folder_)는 동일한 용어이며 이 책의 예제에서는 두 가지를 모두 사용할 수 있습니다.

디렉터리 변경 명령어인 `cd`를 사용하여 현재 작업 디렉터리를 변경할 수 있습니다. 예를 들어 홈 디렉터리로 이동하려면 다음과 같이 입력할 수 있습니다.

```nb
cd ../
```

또는 다음과 같이 입력할 수도 있습니다.

```nb
cd  /home/
```

첫 번째 옵션은 상대(_relative_) 경로를 사용합니다. 두 번째 옵션은 절대(_absolute_) 경로를 사용합니다. 일반적으로 특히 Python 및 R과 같은 언어에서 다른 사람이 여러 컴퓨터에서 코드를 재사용할 수 있는 경우 절대 경로보다 상대 경로가 더 좋습니다.

명령줄을 사용하여 파일과 디렉터리를 이동할 수도 있습니다. 예를 들어 *test.py*를 복사하려면 파일과 동일한 디렉터리에 있어야 합니다. 이를 위해 **`cd`** 를 사용하여 *test.py*가 있는 디렉터리로 이동합니다. **`ls`** 를 입력하여 파일이 보이는지 확인합니다. 그런 다음 복사 함수인 **`cp`** 를 사용하여 파일을 *Documents*로 복사합니다.

cp` `test.py` `./Documents

다른 파일 경로와 함께 `cp`를 사용할 수도 있습니다. 예를 들어, *Documents*에 있고 *test.py*를 *python_code*로 이동하고 싶다고 가정해 보겠습니다. `cp`와 함께 파일 경로를 사용할 수 있습니다.

cp` `../test.py` `./python_code

이 예제에서 여러분은 현재 */home/raerickson/Documents*에 있습니다. `./python_code`를 사용하여 `../test.py`를 `/home/raerickson/Documents/python_code` 디렉터리로 복사함으로써 _/home/raerickson/test.py/_ 의 파일을 가져올 수 있습니다.

디렉터리도 복사할 수 있습니다. 이를 위해서는 복사 명령어와 함께 재귀적(recursive) 옵션(Linux 전문 용어로는 플래그(_flag_)) `-r`을 사용합니다. 예를 들어 *python_code*를 복사하려면 `cp ./python_code new_location`을 사용합니다. 원래 객체를 남겨두지 않는 이동 기능도 존재합니다. 이동 명령어는 `mv`입니다.

###### 경고 (Warning)

명령줄로 삭제한 파일은 컴퓨터의 휴지통이나 쓰레기통 디렉터리로 가지 않습니다. 삭제는 영구적입니다.

마지막으로 터미널을 사용하여 디렉터리와 파일을 제거할 수 있습니다. 매우 주의하는 것을 권장합니다. 파일을 삭제하거나 제거하려면 *`rm file_name`*을 사용하며, 여기서 *`file_name`*은 삭제할 파일입니다. 디렉터리를 삭제하려면 *`rm -r directory`*를 사용하며, 여기서 *`directory`*는 제거하려는 디렉터리입니다. 시작하는 데 도움이 되도록, <a href="#tbl-common-bash" data-type="xref">표 9-2</a>에는 우리가 일상적으로 사용하는 일반적인 bash 명령어가 포함되어 있습니다.

| 명령어 (Command) | 이름 및 설명 (Name and description)                                       |
| ---------------- | ------------------------------------------------------------------------- |
| `pwd`            | 작업 디렉터리 인쇄(Print working directory), 컴퓨터 내 현재 위치를 표시함 |
| `cd`             | 디렉터리 변경(Change directory), 컴퓨터 내 위치를 변경함                  |
| `cp`             | 파일 복사                                                                 |
| `cp -r`          | 디렉터리 복사                                                             |
| `mv`             | 파일 이동                                                                 |
| `mv -r`          | 디렉터리 이동                                                             |
| `rm`             | 파일 삭제                                                                 |
| `rm -r`          | 디렉터리 삭제                                                             |

표 9-2. 일반적인 bash 명령어 {#tbl-common-bash style="width: 100%"}

## bash를 위한 권장 도서 (Suggested Readings for bash)

bash 셸은 컴퓨터와 상호 작용하는 방법일 뿐만 아니라 자체적인 프로그래밍 언어도 함께 제공합니다. 우리는 보통 일상적인 작업에서 그 도구들의 표면적인 부분만 다루지만, 일부 사람들은 그 언어로 광범위하게 프로그래밍하기도 합니다. 다음은 더 많은 것을 배울 수 있는 몇 가지 bash 리소스입니다.

- [Software Carpentry](https://software-carpentry.org)는 [Unix Shell](https://swcarpentry.github.io/shell-novice)에 대한 무료 튜토리얼을 제공합니다. 더 넓게 보면, 우리는 이 사이트를 이 책에서 다루는 많은 주제에 대한 일반적인 리소스로 추천합니다.

- 캐머런 뉴햄(Cameron Newham)과 빌 로젠블라트(Bill Rosenblatt)의 <a href="https://learning.oreilly.com/library/view/learning-the-bash/0596009658" class="orm:hideurl"><em>Learning the bash Shell</em>, 제3판</a> (O’Reilly, 2005)은 이 언어의 고급 도구 및 기능에 대한 소개와 관련 내용을 제공합니다.

- 예로엔 얀센스(Jeroen Janssens)의 <a href="https://learning.oreilly.com/library/view/data-science-at/9781492087908" class="orm:hideurl"><em>Data Science at the Command Line: Obtain, Scrub, Explore, and Model Data with Unix Power Tools</em>, 제2판</a> (O’Reilly, 2021)은 유용한 명령줄 도구를 사용하는 방법을 보여줍니다.

- 여러 책을 훑어보고 자신의 필요에 맞는 책을 찾는 것을 제안합니다. 리차드는 엘리 퀴글리(Ellie Quigley)의 _Unix Shells by Example_ 제4판(Pearson, 2004)으로 배웠습니다.

- 온라인 벤더들은 bash에 관한 코스를 제공합니다. 특별히 추천할 만한 것은 없습니다.

# 버전 관리 (Version Control)

정기적으로 코드 작업을 할 때, 우리는 _변경 사항을 어떻게 추적할 것인가?_ 또는 _코드를 어떻게 공유할 것인가?_ 와 같은 문제에 직면합니다. 이 문제에 대한 해결책이 바로 버전 관리 소프트웨어(_version control software_)입니다. 역사적으로 여러 프로그램이 존재했습니다. 이러한 프로그램은 다른 사람의 작업 내역을 확인하고 협업하며, 코드의 변경 사항을 추적해야 하는 필요성에서 등장했습니다. 현재 주요 버전 관리 프로그램은 Git입니다. 이 책의 출판 즈음에는 [70%](https://oreil.ly/6qO8r)에서 [90%](https://oreil.ly/nsZAX)에 이르는 높은 시장 점유율을 기록하고 있습니다.

Git이 등장하게 된 계기는 리누스 토르발스(Linus Torvalds)가 직접 개발한 운영 체제인 Linux에서 동일한 문제에 직면했기 때문입니다. 그는 전 세계에 퍼져 있는 수많은 자원봉사 프로그래머들의 변경 사항을 추적하기 위해 가볍고 효율적인 프로그램이 필요했습니다. 기존 프로그램들은 각 파일의 여러 버전을 보관했기 때문에 메모리를 너무 많이 사용했습니다. 대신 그는 파일 간의 변경 사항만 추적하는 프로그램을 만들었습니다. 그는 이 프로그램을 Git이라고 불렀습니다.

###### 참고 (Note)

_재미있는 사실:_ [리누스 토르발스는 반농담조로 자신의 두 소프트웨어 프로그램에 자기 이름을 붙였다고 말한 바 있습니다](https://oreil.ly/UUkZk). *Linux*는 Linux is not Unix(Linux는 Unix가 아니다)라는 재귀 약어(recursive acronym)이지만, 그의 이름과도 비슷합니다. *Git*은 영국 영어로 거만한 사람이나 얼간이를 뜻하는 속어입니다. 토르발스는 본인 스스로 인정하듯 같이 일하기 까다로운 사람일 수 있습니다. 예를 들어, 토르발스의 이미지를 검색해 보면 기자의 질문에 모욕적인 제스처로 응답하는 그의 모습을 볼 수 있습니다.

## Git

*Git*은 근본적으로 누구나 코드의 변경 사항을 추적할 수 있게 해주는 오픈 소스 프로그램입니다. 사용자들은 자신의 컴퓨터에서 Git을 사용하여 스스로 변경한 내용을 추적할 수 있습니다. 여기서는 Git의 몇 가지 기본 개념부터 시작해 보겠습니다. 먼저 Git을 다운로드해야 합니다.

- Windows 사용자에게는 [Git for Windows](https://www.gitforwindows.org)를 추천합니다.

- macOS 사용자는 터미널이 설치되어 있는지 확인하는 것이 좋습니다. Xcode를 설치하면 Git이 함께 포함되어 있지만 오래된 버전일 것입니다. 대신 [Git 프로젝트 홈 페이지](https://git-scm.com)에서 Git을 업그레이드할 것을 강력히 권장합니다.

- Linux 사용자는 안전을 위해 OS에 포함된 Git을 업그레이드할 것을 권장합니다.

- Windows나 macOS 시스템에서 GUI를 원하시는 분들은 [GitHub Desktop](https://oreil.ly/Ghub)을 확인해 보시기를 제안합니다. [Git 프로젝트 페이지](https://oreil.ly/bfYbS)에는 Linux, Windows, macOS용 GUI를 포함한 다른 많은 클라이언트가 나열되어 있습니다.

###### 팁 (Tip)

명령줄(Command-line) Git은 어떤 GUI보다 강력하지만 그만큼 더 어렵습니다. 명령줄을 사용해 개념을 설명하지만 GUI를 사용하시는 것을 권장합니다. 두 가지 좋은 옵션으로는 GitHub의 GUI와 Git에 함께 제공되는 기본 Git GUI가 있습니다.

Git을 다운로드한 후에는 코드를 추적할 위치를 Git에 알려주어야 합니다.

1.  터미널을 엽니다.

2.  **`cd path/to_my_code/`** 를 사용하여 작업 디렉터리를 프로젝트 디렉터리로 변경합니다.

3.  한 줄로 **`git init`** 을 입력한 다음 Enter/Return을 누릅니다. `git` 명령은 터미널에 Git 프로그램을 사용하라고 지시하고, `init`은 Git 프로그램에 `init` 명령을 사용하라고 지시합니다.

4.  Git에 어떤 코드를 추적할지 지시합니다. **`git add filename`** 을 통해 개별 파일에 대해 수행하거나 **`git add .`** 을 통해 모든 파일에 대해 수행할 수 있습니다(마침표는 현재 디렉터리의 모든 파일과 디렉터리에 대한 단축키입니다).

5.  **`git commit -m "initial commit"`** 을 사용하여 코드 변경 사항을 커밋합니다. 이 명령어에서 `git`은 터미널에 사용할 프로그램을 알려줍니다. `commit` 명령은 git에 커밋을 지시합니다. 플래그(_flag_) `-m`은 `commit`에 따옴표 안의 메시지인 _`"my changes"`_ 를 수락하도록 지시합니다. 향후 수정 시에는 여기에 설명이 포함된 용어를 사용하는 것이 좋습니다.

###### 경고 (Warning)

어떤 파일을 추적할지 주의하세요. 데이터 파일(_.csv_ 파일 등)이나 이미지나 표 같은 출력 파일을 추적하고 싶은 경우는 거의 없을 것입니다. GitHub과 같은 공용 저장소(public repositories)에 코드를 게시할 때는 특히 더 조심해야 합니다. _.gitignore_ 파일을 사용하여 `*.csv`와 같은 명령으로 모든 파일 유형에 대한 추적을 차단하여 CSV 파일 추적을 막을 수 있습니다.

이제 코드를 편집할 수 있습니다. 예를 들어, _my_code.R_ 파일을 편집한 다음 변경했다고 가정해 보겠습니다. **`git status`** 를 입력하면 해당 파일이 변경되었음을 확인할 수 있습니다. **`git add my_code.R`** 을 입력하여 파일의 변경 사항을 추가할 수 있습니다. 그런 다음 **`git commit -m "example changes"`** 를 통해 변경 사항을 커밋해야 합니다.

###### 팁 (Tip)

중요한 파일을 하나 이상 실수로 삭제한 적이 있다면 Git의 학습 곡선(learning curve)은 그만한 가치가 있습니다. 작업을 며칠, 몇 주, 몇 달 이상 잃어버리는 대신 Git에서 삭제를 취소하기 위해 검색하는 시간만큼만 잃게 됩니다. _저희를 믿으세요. 저희는 경험과 뼈아픈 시련을 통해 알게 되었습니다._

## GitHub 및 GitLab (GitHub and GitLab)

Git에 익숙해지면(적어도 자신의 코드를 다른 사람과 공유하기 시작할 정도로 익숙해진다면), 코드를 백업하고 공유하고 싶어질 것입니다. 리차드가 2007년쯤 대학원에 다닐 때, 그는 원격으로 지도 교수의 컴퓨터에 접속하기 위해 터미널을 사용해야 했고 박사 학위 프로젝트 코드를 얻기 위해 Git을 사용해야 했습니다. 코드 공유를 위한 상용 솔루션(GitHub와 같이 사용하기 쉬운)이 아직 존재하지 않았기 때문입니다. 다행히도 이제는 상용 서비스에서 Git 저장소(repositories)를 호스팅합니다.

가장 큰 제공업체는 [GitHub](https://github.com)입니다. 이 회사와 서비스는 현재 Microsoft가 소유하고 있습니다. 비즈니스 모델은 무료 호스팅을 허용하지만 비즈니스 사용자 및 추가 기능에 대해서는 요금을 부과합니다. 두 번째로 큰 제공업체는 [GitLab](https://gitlab.com)입니다. 유사한 비즈니스 모델을 가지고 있지만 개발자 중심적인 성향이 더 강합니다. GitLab은 오픈 소스 소프트웨어를 사용하여 무료로 자체 호스팅할 수 있는 옵션도 포함하고 있습니다. 예를 들어, O’Reilly Media와 우리의 고용주 중 한 곳은 모두 자체 GitLab 저장소를 자체 호스팅(self-host)하고 있습니다.

어떤 상용 플랫폼을 사용하든, 모두 동일한 기본 Git 기술 및 명령줄 도구를 사용합니다. 비록 제공업체가 서로 다른 웹사이트, GUI 도구 및 부가 기능들을 제공하더라도, 기본 Git 프로그램은 동일합니다. 우리의 기본 추천은 GitHub이지만, Microsoft를 피하고 GitLab을 선호하는 사람들도 있다는 것을 알고 있습니다. 또 다른 선택지로 [Bitbucket](https://bitbucket.org)이 있지만, 우리는 이 플랫폼에 덜 익숙합니다.

원격 저장소의 목적은 코드가 백업되고 다른 사람들도 액세스할 수 있도록 하는 것입니다. 원한다면 여러분의 코드를 다른 사람들과 공유할 수 있습니다. 오픈 소스 소프트웨어의 경우 사용자들이 버그를 보고하고 소프트웨어의 새로운 기능과 버그 수정을 기여할 수 있습니다. 또한 누구나 코드 업데이트나 수정 내역을 확인할 수 있어 GitHub나 GitLab의 온라인 GUI를 유용하게 사용합니다. 이러한 웹 페이지의 또 다른 장점은 Jupyter Notebook 및 마크다운(Markdown) 파일을 정적(static) 웹 페이지로 렌더링해준다는 점입니다.

## GitHub 웹 페이지 및 이력서 (GitHub Web Pages and Résumés)

Git과 GitHub에 대해 배우는 재미있는 방법은 이력서(résumé)를 작성해 보는 것입니다. 'GitHub 이력서'를 검색하면 온라인 튜토리얼을 찾는 데 도움이 될 것입니다(이러한 페이지는 지속적으로 변경되므로 링크는 포함하지 않습니다). Git 기반의 이력서를 사용하면 자신의 기술을 보여주고 그 기술을 입증하는 시장성 있는 제품(marketable product)을 만들 수 있습니다. 재미로든 구직 활동의 일환으로든 자신이 만든 풋볼 제품을 뽐내기 위해 이를 사용할 수도 있습니다. 예를 들어, 우리는 인턴들이 배운 내용을 기록하는 동시에 Git을 더 잘 배울 수 있는 방법으로 이러한 페이지를 만들게 할 것입니다. 이전 인턴이었던 존 올리버(John Oliver)의 이력서 예시를 <a href="https://oreil.ly/JOliv" class="bare"><em>https://oreil.ly/JOliv</em></a>에서 확인할 수 있습니다.

# Git 권장 도서 (Suggested Reading for Git)

학습 스타일에 따라 Git에 대해 배우는 데 도움이 될 수 있는 리소스가 많이 있습니다. 우리가 유용하다고 생각한 리소스는 다음과 같습니다.

- 배경지식이 없는 사람들에게 Git 기술에 대한 개요를 제공하는 비디오를 포함하여, [Git 프로젝트 홈 페이지](https://git-scm.com)에서 Git 튜토리얼을 제공합니다.

- Software Carpentry는 [Git 튜토리얼](https://swcarpentry.github.io/git-novice)을 제공합니다.

- GitHub의 교육 자료가 리소스를 제공합니다. 이 내용은 시간이 지남에 따라 변경되므로 직접 링크를 제공하지는 않습니다.

- 프렘 쿠마르 포누토라이(Prem Kumar Ponuthorai)와 존 로엘리거(Jon Loeliger) 공저의 <a href="https://learning.oreilly.com/library/view/version-control-with/9781492091189/" class="orm:hideurl"><em>Version Control with Git</em>, 제3판</a> (2022)과 같은 O’Reilly의 Git 책들은 한 곳에서 여러 리소스를 제공할 수 있습니다.

# 스타일 가이드 및 린팅 (Style Guides and Linting)

글을 쓸 때 우리는 다양한 스타일을 사용합니다. 우리가 상점에 있다는 파트너에게 보내는 문자 메시지는 _"나 가게야, 이따 봐(At store, see u)."_ 일 수 있고, 이어서 _"ㅇㅋ. 우유 좀 사줘(k. plz buy milk)"_ 라는 응답이 올 수 있습니다. 반면 상사에게 보고서를 작성할 때는 매우 다를 것이며, 외부 고객을 위한 보고서는 훨씬 더 공식적일 것입니다. 코딩에도 다양한 스타일이 있을 수 있습니다. 일관성 있는 코드를 작성하기 위해 스타일 가이드(style guides)가 존재합니다. 그러나 프로그래머들은 창의적이고 실용적인 사람들이기 때문에, 그들 스스로 스타일을 따르는 데 도움이 되는 도구들을 만들었습니다. 대체로 이러한 도구를 린팅(_linting_)이라고 합니다.

###### 참고 (Note)

린팅(_linting_)이라는 용어는 옷에서 보풀(lint)을 제거하는 것, 즉 먼지 부스러기를 제거하기 위해 스웨터에 보풀 제거기를 사용하는 것에서 유래되었습니다.

언어마다 다른 표준이 존재합니다. Python의 경우, 다른 스타일 가이드도 존재하지만 PEP 8이 가장 일반적인 스타일 가이드일 것입니다. R의 경우 Tidyverse/Google 스타일 가이드가 가장 일반적인 스타일일 것입니다.

###### 참고 (Note)

오픈 소스 프로젝트는 종종 분할(split)되었다가 다시 합쳐져 결국 하나로 엮이게 됩니다. R 스타일 가이드도 예외는 아닙니다. 처음에 Google에서 R 스타일 가이드를 만들었습니다. 그 후 Tidyverse 스타일 가이드가 Google의 R 스타일 가이드를 기반으로 만들어졌습니다. 그러나 그 후, Google은 R을 위한 Tidyverse 스타일 가이드를 자체적으로 수정하여 채택했습니다. 이렇게 얽히고설킨 역사는 [Tidyverse 스타일 가이드 페이지](https://style.tidyverse.org)와 [Google의 R 스타일 가이드 페이지](https://oreil.ly/RGuide)에 설명되어 있습니다.

스타일에 대해 더 자세히 알아보려면 [PEP 8 스타일 가이드](https://oreil.ly/PEP8), [Tidyverse 스타일 홈 페이지](https://style.tidyverse.org) 또는 [다양한 언어에 대한 Google의 스타일 가이드](https://oreil.ly/YLnE6)를 방문하시기 바랍니다.

###### 참고 (Note)

Google의 스타일 가이드는 틀림없이 Git을 사용하여 추적되는 마크다운 언어를 사용해서 GitHub에 호스팅되어 있습니다.

<a href="#tbl-adv-tools" data-type="xref">표 9-1</a>에는 몇 가지 예시 린팅 프로그램이 나열되어 있습니다. 또한 코드 편집기의 설명서를 살펴보는 것을 권장합니다. 여기에는 코드를 작성하는 동안 린팅할 수 있는 애드온(add-on)이나 플러그인(plug-in)이 포함되어 있는 경우가 많습니다.

# 패키지 (Packages)

Python이나 R로 프로그래밍하다 보면 결국 사용자 정의 함수(custom functions)를 작성하게 되는 경우가 많습니다. 우리는 이러한 함수들을 쉽게 재사용하고 다른 사람들과 공유할 수 있는 방법이 필요합니다. 함수를 패키지에 넣음으로써 이를 수행할 수 있습니다. 예를 들어, 리차드는 R을 통해 호출되는 Stan 언어를 사용하여 어업 분석에 사용되는 베이지안 모델을 만들었습니다. 그는 이 모델들을 R 패키지인 [`fishStan`](https://oreil.ly/iJagB)으로 출시했습니다. 그런 다음 이 모델들의 출력은 어업 모델에 사용되며, 이는 Python [패키지](https://oreil.ly/XCwTS)로 배포되었습니다.

패키지를 사용하면 모든 함수를 한곳에 보관할 수 있습니다. 이를 통해 재사용이 가능할 뿐만 아니라, 버그 하나를 수정하기 위해 동일한 파일의 여러 버전을 뒤지지 않아도 됩니다. 또한 함수를 업데이트하거나 변경한 후에도 예상대로 작동하는지 확인하는 테스트를 포함할 수 있습니다. 따라서 패키지를 통해 재사용이 가능하고 유지 관리가 쉬운 코드를 만들 수 있습니다.

패키지를 사용하여 다른 사람들과 코드를 공유할 수 있습니다. 패키지를 배포하는 가장 일반적인 방법은 아마도 GitHub 저장소(repos)를 이용하는 것일 겁니다. 진입 장벽이 낮기 때문에 누구나 패키지를 배포할 수 있습니다. Python에는 사용자가 패키지를 제출할 수 있는 `pip` 및 `conda-forge`를 포함한 여러 패키지 관리자가 있습니다. 마찬가지로 R에도 현재 하나의 주요 패키지 관리자(과거에는 더 많았음)인 CRAN(Comprehensive R Archive Network)이 있습니다. 이러한 저장소들은 제출 전 품질 기준의 수준이 각기 다르기 때문에, GitHub와 같은 사이트에 직접 배포하는 것과 비교할 때 어느 정도 검열(gatekeeping)이 이루어집니다.

## 패키지 권장 도서 (Suggested Readings for Packages)

- R 패키지에 대해 배우기 위해 우리는 해들리 위컴(Hadley Wickham)과 제니퍼 브라이언(Jennifer Bryan)의 [온라인 책](https://r-pkgs.org)을 사용했으며, 이 책은 <a href="https://learning.oreilly.com/library/view/r-packages-2nd/9781098134938" class="orm:hideurl">O’Reilly</a>에서 종이책 버전으로도 제공됩니다.

- Python 패키지에 대해 배우기 위해 우리는 공식 Python 튜토리얼인 [“Packaging Python Projects.”](https://oreil.ly/ySvav)를 활용했습니다.

# 컴퓨터 환경 (Computer Environments)

R이나 Python 패키지를 실행했는데, 다음 세션에서 패키지가 실행되지 않는다고 상상해 보세요. 몇 시간 후, 결국 패키지 소유자가 패키지를 업데이트하여 코드를 업데이트해야 한다는 사실을 알아냅니다(네, 우리도 비슷한 상황을 겪었습니다). 이 문제를 예방하는 한 가지 방법은 컴퓨터 환경(environment)을 계속 추적하는 것입니다. 마찬가지로 컴퓨터 사용자는 다른 사람들과 협업할 때 문제가 발생할 수 있습니다. 예를 들어, 에릭은 Windows 컴퓨터로 이 책을 집필한 반면 리차드는 Linux 컴퓨터를 사용했습니다.

컴퓨터의 환경은 컴퓨터의 소프트웨어와 하드웨어 모음입니다. 예를 들어, 하드웨어로 2022년형 Dell XPS 13인치 노트북을 사용 중일 수 있습니다. 소프트웨어 환경에는 Windows 11 릴리스 22H2(10.0.22621.1105)와 같은 운영 체제(OS)와 R 4.1.3 및 `ggplot2` 버전 3.4.0과 같은 R, Python 및 해당 패키지의 버전이 포함될 수 있습니다. 일반적으로 대부분의 사람들은 컴퓨팅 환경을 위한 프로그램에 관심을 둡니다. 사용자 간에(리차드와 에릭) 또는 시간에 따라(2021년 에릭의 컴퓨터와 2023년 에릭의 컴퓨터) 환경이 일치하지 않는 경우, 종종 <a href="#fig-conda" data-type="xref">그림 9-4</a>에 표시된 문제와 같이 프로그램이 실행되지 않을 수 있습니다.

###### 팁 (Tip)

이 책에서는 Conda와 같은 가상 환경(virtual environments)을 충분히 다룰 수 없습니다. 그러나 많은 프로그래머와 데이터 과학자들은 이것의 사용이 경험 많은 전문가와 아마추어를 구별하는 데 도움이 된다고 주장할 것입니다.

[Conda](https://oreil.ly/Conda)와 같은 도구를 사용하면 컴퓨터 환경을 잠그고(lock down) 사용된 특정 프로그램을 공유할 수 있습니다. [Docker](https://www.docker.com)와 같은 도구는 한 단계 더 나아가 환경뿐만 아니라 운영 체제까지 제어합니다. 이 두 프로그램 모두 사용자가 터미널을 이해할 때 가장 효과적으로 작동합니다.

<figure>
<img src="assets/fapr_0904.png" />
<h6 id="figure-9-4.-example-of-computer-environments-and-how-versions-may-vary-across-users-and-machines">그림 9-4. 컴퓨터 환경 예제 및 버전이 사용자와 컴퓨터에 따라 어떻게 달라질 수 있는지에 대한 예시</h6>
</figure>

# 데이터 공유를 위한 대화형 도구 및 보고서 도구 (Interactives and Report Tools to Share Data)

대부분의 사람은 코딩을 하지 않습니다. 그러나 많은 사람이 데이터에 접근하고 싶어 하며, 다행스럽게도 여러분 역시 코드를 공유하기 원할 것입니다. 데이터 및 모델을 공유하기 위한 도구에는 대화형 애플리케이션(interactive applications), 즉 _대화형 도구(interactives)_ 가 포함됩니다. 이를 통해 사람들은 여러분의 코드와 결과물에 상호 작용할 수 있습니다. 일부 독자들이 이 책을 다 마친 후 공유하고 싶어 할 만한 소규모 프로젝트의 경우, [Posit의 Shiny](https://shiny.posit.co) 또는 웹에서 호스팅되는 [위젯이 포함된 Jupyter 노트북](https://jupyter.org/widgets)과 같은 프로그램이 요구 사항을 충족시킬 수 있습니다. 에릭과 같이 데이터 과학 산업에 종사하는 사람들도 이러한 도구를 사용하여 개념 증명(proof-of-concept) 도구를 컴퓨터 과학자 팀에 넘겨 프로덕션 등급(production-grade) 제품을 만들기 전에 모델을 프로토타이핑합니다.

대화형 도구는 데이터를 보는 동적 도구로 아주 잘 작동합니다. 때로는 보고서를 작성하고 싶거나 작성해야 할 때도 있습니다. 마크다운(Markdown) 기반 도구를 사용하면 코드, 데이터, 그림 및 텍스트를 모두 하나로 병합할 수 있습니다. 예를 들어 에릭은 [R Markdown](https://rmarkdown.rstudio.com)으로 클라이언트에게 보고서를 작성하고, 리차드는 [Jupyter Notebook](https://jupyter.org)으로 소프트웨어 설명서를 작성하며, [LaTeX](https://www.latex-project.org)로 과학 논문을 작성합니다. 그리고 이 책은 [Quarto](https://quarto.org)로 작성되었습니다. 처음 시작하는 경우라면 Quarto를 추천합니다. Quarto는 Python 및 다른 언어에서도 작동하도록 R Markdown 언어를 확장했기 때문입니다(R Markdown 자체는 LaTeX를 더 사용하기 쉽게 만들기 위해 만들어졌습니다). Jupyter Notebook은 보고서나 긴 문서에도 유용할 수 있지만(예를 들어, Jupyter Notebook으로 쓴 책도 있습니다) 대화형 도구 같은 동적 애플리케이션에 더 잘 작동하는 경향이 있습니다.

# 인공 지능 도구 (Artificial Intelligence Tools)

현재 사람들의 코딩을 돕는 인공 지능(AI)이나 이와 유사한 도구들이 존재합니다. 예를 들어, 많은 코드 편집기에는 자동 완성(autocompletion) 도구가 있습니다. 기본적으로 이러한 도구들은 기능적으로 AI입니다. 이 책을 쓰는 동안 사람들의 코딩을 돕는 데 큰 잠재력을 가진 새로운 AI 도구들이 등장했습니다. 예를 들어, ChatGPT를 사용하여 사용자의 프롬프트(prompt) 입력에 따라 코드를 생성할 수 있습니다. 마찬가지로 GitHub Copilot과 같은 프로그램은 입력 프롬프트에 기반하여 코딩을 돕고, Google은 이와 경쟁하는 프로그램인 Codey를 출시했습니다.

하지만 AI 도구는 아직 새롭기 때문에 사용할 때 몇 가지 과제가 존재합니다. 예를 들어 이 도구들은 [사실관계 오류(factual errors)](https://oreil.ly/nJOpQ)나 [잘 알려진 편향성(well-documented biases)](https://oreil.ly/K1IlO)을 생성하기도 합니다. 이런 사실관계 오류와 편향성 외에도, 이 프로그램들은 사용자 데이터를 소비합니다. 이를 통해 피드백을 받아 더 나은 프로그램을 만드는 데 도움을 주지만, 사람들이 의도치 않게 데이터를 유출할 수도 있습니다. 예를 들어 삼성 직원이 우발적으로 반도체 소프트웨어와 독점적인(proprietary) 데이터를 [ChatGPT에 유출](https://oreil.ly/-rgCg)한 사건이 있었습니다. 마찬가지로 [Copilot for Business Privacy Statement(비즈니스용 코파일럿 개인정보 취급방침)](https://oreil.ly/Nflm7)에는 "서비스를 제공하기 위해 데이터를 수집하며, 그중 일부는 추가 분석 및 제품 개선을 위해 저장된다"라고 명시되어 있습니다.

###### 경고 (Warning)

서비스가 여러분의 데이터와 코드를 어떻게 사용하고 저장하는지 완전히 이해하지 못했다면, AI 서비스에 데이터나 코드를 업로드하지 마십시오.

우리는 AI 기반 코딩 도구가 코딩 능력을 크게 향상시킬 것이라 예상하지만, 숙련된 작업자도 필요할 것입니다. 예를 들어, 맞춤법 검사기나 문법 검사기가 편집자의 필요성을 없애지는 못했습니다. 그것들은 단지 편집자의 업무 중 한 부분을 덜어주었을 뿐입니다.

# 결론 (Conclusion)

미식축구는 미국에서 가장 인기 있는 스포츠이며 전 세계에서 가장 인기 있는 스포츠 중 하나입니다. 수백만 명의 팬들이 매년 자신이 좋아하는 팀을 보기 위해 셀 수 없이 많은 거리를 이동하며, 갱신될 때마다 10억 달러가 넘는 TV 중계권 계약이 체결됩니다. 미식축구는 오락, 여가, 자부심 또는 투자 등 모든 종류의 참여를 위한 훌륭한 매개체입니다. 이제 우리는 미식축구가 수학을 위한 매개체가 되기를 바랍니다.

이 책 전체에 걸쳐 우리는 수학에 관심이 있는 사람이 전 세계의 수많은 학부 프로그램에서 배우는 통계 및 컴퓨터 도구를 통해 게임을 더 잘 이해할 수 있는 다양한 방법을 제시했습니다. 바로 이와 동일한 접근법이 하나의 산업으로서 대화를 새로운 영역으로 이끌고 나아가는 데 도움이 되었습니다. 이러한 분석 주도형(analytically driven) 접근법은 우리가 해결해야 할 새로운 문제를 만들어내고 있으며, 이는 의심할 여지 없이 미래에 이 책의 독자인 여러분이 해결해야 할 추가적인 문제를 만들어낼 것입니다.

지난 10년 동안의 풋볼 분석은 네이트 실버(Nate Silver)가 대중화한 "신호와 노이즈(signal and the noise)" 프레임워크 쪽으로 대화를 옮겨갔습니다. 일례로 에릭과 그의 전 직장 동료인 조지 차로우리(George Chahrouri)는 [PFF 기사](https://oreil.ly/JnD78)에서 "만약 우리가 미래를 예측하고 싶다면, 쿼터백이 압박을 받을 때와 클린 포켓(clean pocket) 상황일 때 중 어디에 더 비중을 두어야 할까?"라는 질문을 던졌습니다.

우리는 또한 포지션별 선수 가치에서도 극적인 변화를 목격했는데, 이는 주로 포지션 가치를 평가하여 [로스터 구성을 돕는 PFF](https://oreil.ly/fkg1U)와 같은 분석 회사들의 작업과 궤를 같이 합니다. 마찬가지로 _<a href="https://rbsdm.com" class="bare"><em>https://rbsdm.com</em></a>_ 과 같은 웹사이트는 팬, 분석가 및 기자들이 데이터를 사용하여 그들이 사랑하고 담당하는 경기를 맥락화(contextualize)할 수 있게 해주었습니다.

비슷한 맥락에서 미국 전역에 걸친 스포츠 베팅의 합법화는 오해의 소지가 있는 정보에서 의미 있는 정보를 가려내는 능력의 필요성을 증가시켰습니다. 한때 풋볼 시즌에서 뒷전으로 밀려났던 NFL 드래프트조차 큰 판돈이 걸린 포커 게임이 되었고, 선수 선발과 자산 배분을 최대한 효율적으로 만들기 위해 일하는 게임 속 최고의 두뇌들을 끌어모았습니다.

풋볼 분석의 미래도 밝습니다. 최근 선수 추적(player-tracking) 데이터의 확산과 함께, 이 책에서 얻은 통찰력은 경기를 더욱 즐겁게 만들어줄 문제들이 끊임없이 늘어나는 이 분야에서 도약대가 될 것입니다. 결국 경기의 거의 모든 분석적 발전(더 많은 패스, 더 많은 포스 다운(fourth-down) 시도)은 경기를 더 재밌게 만들었습니다. 우리는 그러한 추세가 계속될 것이라고 예측합니다.

게다가, 스포츠 분석 전반과 특히 풋볼 분석은 이전 세대보다 훨씬 더 많은 사람이 스포츠에 참여하고 적극적으로 관여할 수 있는 길을 열어주었습니다. 예를 들어 2023년 5월 현재, 에릭의 인턴십 프로그램은 4명을 NFL 프런트 오피스에 보냈으며 앞으로 더 많은 사람이 나오기를 바라고 있습니다. 누가 참여할 수 있는지를 확대하고 게임에 가치를 더함으로써, 미식축구는 이제 미래 세대에게 훨씬 더 매력적인 스포츠가 될 기회를 얻었습니다.

이 책을 통해 여러분이 풋볼과 풋볼 분석에 대한 흥미를 키우셨기를 바랍니다. 이 떠오르는 분야에 그저 발을 담그고 싶은 사람이라면, 아마도 여러분이 필요한 모든 것이 들어 있을 것입니다. 판타지 풋볼이나 오피스 풀(office pool)에서 우위를 점하고 싶은 분들은 올해 데이터로 우리 예제를 업데이트할 수 있습니다. 더 깊이 파고들고자 하는 분들에게는 각 장의 참고문헌이 향후 탐구를 위한 도약대가 될 것입니다.

마지막으로 풋볼에 대한 자세한 정보를 얻을 수 있는 웹사이트 몇 곳을 소개합니다.

- 에릭의 현재 직장, SumerSports: _<a href="https://sumersports.com/" class="bare"><em>https://sumersports.com/</em></a>_

- 에릭의 이전 직장, PFF: _<a href="https://www.pff.com" class="bare"><em>https://www.pff.com</em></a>_

- 고급 NFL 통계에 초점을 맞춘 웹사이트, Football Outsiders: _<a href="https://www.footballoutsiders.com" class="bare"><em>https://www.footballoutsiders.com</em></a>_

- 다른 유용한 자료를 소개하는 벤 볼드윈(Ben Baldwin)의 페이지: _<a href="https://rbsdm.com" class="bare"><em>https://rbsdm.com</em></a>_

풋볼 데이터의 세계로 깊이 빠져들면서 즐거운 코딩하시길 바랍니다!
