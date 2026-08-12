# 서문

삶은 이야기에 관한 것입니다. 많은 사람들에게 이 이야기는 스포츠, 특히 풋볼을 중심으로 전개됩니다. 우리 중 많은 사람들은 가족 및 친구들과 뒷마당에서 경기를 했던 것을 기억하며, 그들 중 일부는 조명 아래에서 팬들 앞에서 경기를 하는 행운을 누리기도 했습니다. 우리는 우리의 영웅을 사랑하고 라이벌을 혐오합니다. 우리는 과거를 이해하고 미래를 예측하기 위해 이러한 이야기를 합니다. 풋볼 분석(Football analytics)의 핵심은, 우리가 사랑하는 게임에 대해 더 정확한 이야기를 전달하기 위해 정보를 사용하는 데 있습니다. 분석은 경기의 난해한 질문들에 대한 답을 제시해 줍니다.

예를 들어, 챔피언십 우승 횟수가 많은 쿼터백이 최고의 쿼터백일까요? 아니면 패싱 통계가 좋은 선수가 최고일까요? "좋은 패싱 통계"라는 것은 과연 무엇을 의미할까요? 선수들이 수학을 초월하여 수행하는 것들이 있을까요? 만약 그렇다면, 그들은 단순히 현재 우리가 게임을 수학적으로 이해하는 수준을 넘어서는 것일 뿐, 새롭고 더 나은 정보를 얻게 되었을 때 비로소 그 탁월함이 입증되는 것일까요? 만약 여러분이 응원하는 팀이 올해 우승을 목표로 한다면 누구를 드래프트해야 할까요? 3년 후를 본다면 어떨까요? 여러분의 할머니는 판타지 풋볼에서 누구를 드래프트해야 할까요? 일요일 풋볼 경기 동안 라스베이거스에서 경유할 때 어느 팀에 베팅해야 할까요?

야구에서 _머니볼(Moneyball)_ 접근 방식(2003년 W. W. Norton & Company에서 출간된 마이클 루이스(Michael Lewis)의 책에 소개됨)이 성공을 거둔 이후, 사람들은 점점 더 수학과 통계를 사용하여 위와 같은 질문을 적절하게 공식화하고 답변하는 데 도움을 받고 있습니다. 100야드 필드에서 22명의 선수가 상호작용하는 풋볼은 투수와 타자 간의 독립적인 대결이 많은 야구보다 통계 분석을 적용하기 어려워 보일 수 있지만, 많은 이들이 게임을 이해하고 더 나은 방향으로 바꾸는 데 상당한 진전을 이루어냈습니다.

우리의 목표는 풋볼에 통계 분석을 어떻게 적용할 수 있는지 여러분이 이해하기 시작하도록 돕는 것입니다. 이 책에서 배운 내용을 바탕으로 우리가 사랑하는 게임의 위대한 데이터 혁명에 여러분 중 일부가 기여할 수 있기를 바랍니다. 이 책의 각 장은 미식축구의 특정 문제에 초점을 맞추면서, 그 문제를 해결하기 위한 기술을 다룹니다.

- <a href="ch01.html#sec-introduction" data-type="xref" data-xrefstyle="chap-num-title">1장, "풋볼 분석(Football Analytics)"</a>에서는 최근 몇 년 동안 해결된 문제들을 포함하여 지금까지의 풋볼 분석에 대한 개요를 제공합니다. 그런 다음 공개적으로 이용 가능한 플레이 단위(play-by-play) 데이터를 탐색하여 내셔널 풋볼 리그(NFL) 쿼터백의 평균 타겟 깊이(average depth of target, aDOT)를 살펴봄으로써 그들의 공격성을 측정합니다.

- <a href="ch02.html#sec-EDA-stable" data-type="xref" data-xrefstyle="chap-num-title">2장, "탐색적 데이터 분석: 안정적인 쿼터백 통계 대 불안정적인 쿼터백 통계(Exploratory Data Analysis: Stable Versus Unstable Quarterback Statistics)"</a>에서는 탐색적 데이터 분석(EDA)을 소개하여 쿼터백 패싱 데이터의 하위 집합(짧은 패스와 긴 패스) 중 어느 것이 해마다 더 안정적인지 조사하고, 이러한 분석을 사용하여 해마다 또는 주마다 회귀(regression) 후보를 살펴보는 방법을 알아봅니다.

- <a href="ch03.html#sec-lm-ryoa" data-type="xref" data-xrefstyle="chap-num-title">3장, "단순 선형 회귀: 기대 대비 러싱 야드(Simple Linear Regression: Rushing Yards Over Expected)"</a>에서는 선형 회귀를 사용하여 NFL 볼 캐리어의 러싱 데이터를 정규화합니다. 데이터를 정규화하면 퍼스트 다운을 얻기 위해 팀에 필요한 야드 수와 같은 컨텍스트를 조정하는 데 도움이 되며, 이는 선수가 생산하는 원시 데이터 결과에 영향을 미칠 수 있습니다.

- <a href="ch04.html#sec-mr-ryoe2" data-type="xref" data-xrefstyle="chap-num-title">4장, "다중 회귀: 기대 대비 러싱 야드(Multiple Regression: Rushing Yards Over Expected)"</a>에서는 정규화할 더 많은 변수를 포함하기 위해 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>의 작업을 확장합니다. 예를 들어, 다운(down)과 거리(distance)는 모두 볼 캐리어에 대한 기대치에 영향을 미치므로 러싱 야드 모델에 둘 다 포함되어야 합니다. 그런 다음 이러한 정규화가 러싱 데이터의 안정성을 더해주는지 여부를 확인합니다. 이 장에서는 "러닝백이 중요할까?"라는 질문도 탐구합니다.

- <a href="ch05.html#sec-lr-pass" data-type="xref" data-xrefstyle="chap-num-title">5장, "일반화 선형 모델: 기대 대비 패스 성공률(Generalized Linear Models: Completion Percentage over Expected)"</a>에서는 쿼터백의 패스 성공률을 모델링하기 위해 로지스틱 회귀를 사용하는 방법을 보여줍니다.

- <a href="ch06.html#sec-pos-td" data-type="xref" data-xrefstyle="chap-num-title">6장, "스포츠 베팅을 위한 데이터 과학 활용: 푸아송 회귀 및 패싱 터치다운(Using Data Science for Sports Betting: Poisson Regression and Passing Touchdowns)"</a>에서는 푸아송 회귀를 사용하여 경기 결과를 모델링하는 방법과 이러한 모델이 베팅 시장에 어떻게 적용되는지 보여줍니다.

- <a href="ch07.html#sec-webscrap-draft" data-type="xref" data-xrefstyle="chap-num-title">7장, "웹 스크래핑: 드래프트 픽 수집 및 분석(Web Scraping: Obtaining and Analyzing Draft Picks)"</a>에서는 웹 스크래핑 기술을 사용하여 2000년대 초반 이후의 NFL 드래프트 데이터를 가져옵니다. 그런 다음 드래프트 픽에 대한 기대치를 조정한 후 선수 선발에 있어서 어느 팀이 기대보다 더 낫거나 못한지 분석합니다.

- <a href="ch08.html#sec-pca-clus" data-type="xref" data-xrefstyle="chap-num-title">8장, "주성분 분석 및 클러스터링: 선수 속성(Principal Component Analysis and Clustering: Player Attributes)"</a>에서는 주성분 분석(PCA)과 클러스터링을 사용하여 비지도 학습을 통해 선수 유형을 결정하기 위해 NFL 스카우팅 컴바인(NFL Scouting Combine) 데이터를 분석합니다.

- <a href="ch09.html#sec-advanced-tool" data-type="xref" data-xrefstyle="chap-num-title">9장, "고급 도구 및 다음 단계(Advanced Tools and Next Steps)"</a>에서는 분석 기술을 한 단계 끌어올리려는 분들을 위한 고급 도구를 설명합니다.

일반적으로 한 장에 제시된 도구가 나중에 사용될 수 있기 때문에 이 장들은 서로를 기반으로 구성되어 있습니다. 이 책에는 또한 세 개의 부록이 포함되어 있습니다.

- <a href="app01.html#sec-appendix-1" data-type="xref" data-xrefstyle="app-num-title">부록 A, "Python 및 R 기초(Python and R Basics)"</a>에서는 이 프로그램들을 처음 접하고 설치에 대한 안내가 필요한 분들을 위해 Python과 R을 소개합니다.

- <a href="app02.html#sec-ssdw-pass" data-type="xref" data-xrefstyle="app-num-title">부록 B, "요약 통계 및 데이터 랭글링: 볼 패싱(Summary Statistics and Data Wrangling: Passing the Ball)"</a>에서는 패싱 야드를 시연하는 예제를 통해 요약 통계 및 데이터 랭글링을 소개합니다.

- <a href="app03.html#sec-app-dw" data-type="xref" data-xrefstyle="app-num-title">부록 C, "데이터 랭글링 기초(Data-Wrangling Fundamentals)"</a>에서는 더 많은 데이터 랭글링 기술에 대한 개요를 제공합니다.

책의 마지막에는 용어를 정의하기 위한 <a href="glossary01.html#glossary" data-type="xref">용어집(Glossary)</a>도 포함되어 있습니다.

이 책에서는 사례 연구를 중점적으로 다룹니다. <a href="#case-study-table" data-type="xref">표 P-1</a>은 장별 사례 연구와 각 사례 연구가 다루는 기술을 보여줍니다.

| 사례 연구                          | 기법                                                       | 위치                                                                                                                                                         |
| ---------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 동기 부여 예제: 홈 필드 어드밴티지 | 문제 구성 예제                                             | <a href="#sec-home-field" data-type="xref">"풋볼 예제(A Football Example)"</a>                                                                               |
| 쿼터백의 패스 깊이                 | R과 Python에서 NFL 데이터 가져오기                         | <a href="ch01.html#sec-pass-deep" data-type="xref">"예제 데이터: 누가 깊게 던지는가?(Example Data: Who Throws Deep?)"</a>                                    |
| 여러 시즌에 걸친 패싱 야드         | EDA를 통한 안정성 분석 소개                                | <a href="ch02.html#sec-eda-stable" data-type="xref">"시도당 패싱 야드의 선수별 안정성(Player-Level Stability of Passing Yards per Attempt)"</a>              |
| 러싱 야드 예측 인자                | 기대 대비 러싱 야드(RYOE)를 추정하기 위한 간단한 모델 구축 | <a href="ch03.html#sec-best-RYOE" data-type="xref">"누가 RYOE에서 최고였는가?(Who Was the Best in RYOE?)"</a>                                                |
| 러싱 야드의 다중 예측 인자         | RYOE를 추정하기 위한 다중 회귀 모델 구축                   | <a href="ch04.html#sec-amlr" data-type="xref">"다중 선형 회귀 적용(Applying Multiple Linear Regression)"</a>                                                 |
| 패스 성공률                        | 패스 성공률을 추정하기 위해 로지스틱 회귀 사용             | <a href="ch05.html#sec-glm-pass-completion" data-type="xref">"성공률에 대한 GLM 적용(GLM Application to Completion Percentage)"</a>                          |
| 프로포지션(또는 프롭) 베팅         | 베팅을 이해하기 위해 푸아송 회귀 사용                      | <a href="ch06.html#sec-poisson-prop" data-type="xref">"푸아송 회귀의 적용: 프롭 시장(Application of Poisson Regression: Prop Markets)"</a>                   |
| 제츠/콜츠 2018년 트레이드 정량화   | 드래프트 트레이드 평가                                     | <a href="ch07.html#sec-draft-jets-colts" data-type="xref">"제츠/콜츠 2018년 트레이드 평가(The Jets/Colts 2018 Trade Evaluated)"</a>                          |
| 모든 팀의 드래프트 평가            | 모든 NFL 팀의 드래프트 결과 비교                           | <a href="ch07.html#sec-all-team-draft" data-type="xref">"선수 드래프트를 더 잘하는 팀이 있는가?(Are Some Teams Better at Drafting Players Than Others?)"</a> |
| 선수의 NFL 스카우팅 컴바인 속성    | 다변량 분석을 사용하여 선수 속성 분류                      | <a href="ch08.html#sec-cluster-combine" data-type="xref">"컴바인 데이터 클러스터링(Clustering Combine Data)"</a>                                             |

표 P-1. 이 책의 사례 연구 {#case-study-table}

# 이 책의 대상 독자

저희 책은 두 부류의 독자를 대상으로 합니다. 첫째, 풋볼 분석을 _직접 해보면서_ 풋볼 분석에 대해 배우고 싶은 사람들을 위해 이 책을 썼습니다. 여러분이 직면할 수 있는 문제들을 헤쳐 나가는 데 도움이 되는 예제와 연습 문제를 공유합니다. 이러한 예제와 연습 문제를 통해 저희가 풋볼 데이터에 대해 어떻게 생각하고 데이터를 어떻게 분석하는지 보여드립니다. 여러분은 자신이 응원하는 팀에 대해 더 알고 싶은 팬이거나, 판타지 풋볼 플레이어이거나, 매주 어느 팀이 이길지 관심이 있는 사람이거나, 또는 풋볼 데이터 분석가를 지망하는 사람일 수 있습니다. 둘째, 데이터 과학에 대한 입문을 원하지만 1930년대의 꽃 측정 데이터나 1912년의 _타이타닉호_ 생존자 표와 같은 고전적인 데이터셋으로 배우고 싶지 않은 사람들을 위해 이 책을 썼습니다. 직장에서 위젯에 데이터 과학을 적용하게 되더라도, 최소한 미식축구와 같이 즐거운 주제를 활용하여 배울 수 있습니다.

저희는 여러분이 고등학교 수준의 수학 배경지식을 갖추고 있지만, 아마도 기억이 조금 가물가물한 상태(즉, 미적분학 이전 과정을 마친 상태)라고 가정합니다. 여러분은 고등학생일 수도 있고 30년 동안 수학 수업을 듣지 않은 사람일 수도 있습니다. 저희는 진행하면서 개념을 설명해 드릴 것입니다. 또한 풋볼이 어떻게 재미있는 수학 스토리 문제를 제공할 수 있는지 여러분이 깨닫도록 돕는 데 중점을 둡니다. 이 책은 풋볼 분석가들이 매일 사용하는 기본적인 기술들을 이해하는 데 도움이 될 것입니다. 팬들에게는 이것만으로도 데이터 과학 기술로 충분할 것입니다. 풋볼 분석가를 꿈꾸는 분들에게는 저희 책이 여러분의 꿈과 평생 학습을 위한 도약대가 되기를 바랍니다.

학습을 돕기 위해 이 책은 공개 데이터를 사용합니다. 이를 통해 여러분은 저희의 모든 분석을 재현할 수 있을 뿐만 아니라 향후 시즌을 위해 데이터셋을 업데이트할 수도 있습니다. 예를 들어, 저희는 이 책의 집필을 마치기 전 마지막으로 완료된 시즌인 2022년 시즌까지의 데이터만 사용합니다. 하지만 저희가 가르쳐드리는 도구를 사용하면 향후 연도를 포함하도록 예제를 업데이트할 수 있습니다. 또한 데이터 형식을 지정하는 방법을 보여드리기 위해 모든 데이터 랭글링 방법을 제시합니다. 때로는 다소 지루할 수도 있지만, 데이터를 다루는 방법을 배우면 궁극적으로 더 많은 자유를 얻게 될 것입니다. 즉, 깔끔한 데이터를 위해 다른 사람에게 의존하지 않아도 됩니다.

# 이 책이 적합하지 않은 독자

저희는 초보자를 위해 이 책을 썼으며 프로그래밍 경험이 거의 없거나 전혀 없는 사람들을 위한 부록을 포함했습니다. R이나 Python의 프로그래밍 및 통계에 대한 폭넓은 경험이 있는 사람들은 (풋볼 분석에 존재하는 입문적인 문제의 종류를 살펴보는 것 외에는) 이 책을 통해 큰 도움을 얻지 못할 것입니다. 대신 그런 분들은 R에 대해 더 배우려면 해들리 위컴(Hadley Wickham) 등이 쓴 <a href="https://learning.oreilly.com/library/view/r-for-data/9781492097396/" class="orm:hideurl"><em>R for Data Science</em>, 2판</a> (O’Reilly, 2023)을, Python에 대해 더 배우려면 웨스 맥키니(Wes McKinney)가 쓴 <a href="https://learning.oreilly.com/library/view/python-for-data/9781098104023/" class="orm:hideurl"><em>Python for Data Analysis</em>, 3판</a> (O’Reilly, 2022)과 같은 더 수준 높은 책으로 넘어가야 합니다. 또는 다변량 통계, 회귀 분석, 또는 Posit Shiny 애플리케이션과 같이 이 책에서 다루는 주제에 대한 더 심도 있는 책으로 넘어가기를 원할 수도 있습니다.

저희는 복잡한 분석보다는 간단한 예제에 초점을 맞춥니다. 마찬가지로, 계산적으로 효율적인 코드보다는 더 단순하고 이해하기 쉬운 코드에 중점을 둡니다. 여러분이 빠르게 시작하고 실제 데이터와 연결될 수 있도록 돕고자 합니다. 앙투안 드 생텍쥐페리(Antoine de Saint-Exupéry)의 말로 알려진 인용구를 빌리자면 다음과 같습니다.

> 배를 만들고 싶다면 사람들을 팀으로 나누어 숲으로 나무를 베러 보내지 마라. 대신 그들에게 넓고 끝없는 바다를 동경하도록 가르쳐라.

따라서 저희는 여러분을 풋볼 데이터에 빠르게 연결하여, 이러한 연결이 영감을 주고 도구들을 더 깊이 있게 계속 배우도록 격려하기를 바랍니다.

# 데이터에 대한 사고 방식과 이 책의 활용법

코드만 읽을 것이 아니라 코드를 실행하고, 연습 문제를 풀며, 데이터를 통해 여러분만의 풋볼 관련 질문을 던져보면서 이 책을 학습하시길 권합니다. 저희 예제를 풀어보는 것 외에도 자유롭게 질문을 추가하고 아이디어를 만들어 보세요. 블로그나 GitHub 페이지를 만들어 새로운 기술을 자랑하거나 배운 내용을 공유해 보세요. 친구 한두 명과 함께 책을 공부하세요. 막혔을 때 서로 도와주고 데이터를 통해 발견한 것에 대해 이야기하세요. 마지막 단계가 특히 중요합니다. 전문 데이터 과학자로서 일하면서 저희는 데이터셋을 공유하는 방법에 대해 정기적으로 고민하고 세부적으로 조정합니다.

저희의 본업은 사람들이 데이터를 사용하여 의사 결정을 내리도록 돕는 것입니다. 이 책에서 저희는 우리의 도구와 사고 과정을 여러분과 공유하고자 합니다. 저희의 정규 학문적 교육은 수학과 통계를 다루었지만, 자연계의 지저분한 생태 및 환경 데이터를 분석해야만 했을 때 비로소 진정한 데이터 과학 능력을 키울 수 있었습니다. 저희는 정제되지 않은 데이터를 청소하고 병합하고 조작해야 했으며, 동시에 저희에게 주어진 정보의 빈틈을 어떻게 처리해야 할지 알아내야 했습니다. 그리고 나서 그 지저분한 데이터셋 속에 숨겨진 의미를 설명하려고 노력해야 했습니다.

지난 10년의 중반 무렵, 에릭은 프로 풋볼 포커스(Pro Football Focus, PFF)라는 회사에서 처음에는 작가로, 나중에는 분석가로 풋볼에 자신의 기술을 적용하기 시작했습니다. 결국 그는 학계를 떠나 PFF에 합류하여 회사의 첫 번째 데이터 과학 그룹을 운영하는 데 기여했습니다. PFF에서 근무하는 동안 그는 수머스포츠(SumerSports)에서 새로운 역할을 맡기 전까지 32개의 NFL 팀과 130개 이상의 대학 팀과 함께 일했습니다. 한편, 리처드는 생태 데이터를 계속 다루면서 사람들이 이 데이터를 사용하여 의사 결정을 내리는 것을 돕고 있습니다. 예를 들어, 원치 않는 종을 통제하기 위해 어디서 얼마나 많은 물고기를 수확해야 하는지와 같은 문제입니다.

저희 두 사람 모두 고학력을 가지고 있지만, 정규 교육보다 분명하게 사고하고 데이터를 탐색하는 능력이 더 중요합니다. 알베르트 아인슈타인(Albert Einstein)의 말로 알려진 인용구에 따르면, "상상력이 지식보다 중요합니다." 저희는 이것이 풋볼 분석에도 똑같이 적용된다고 생각합니다. 올바른 질문을 던지고 충분히 좋은 답을 찾는 것이 여러분이 현재 알고 있는 것보다 더 중요합니다. 저희는 다양한 질문과 데이터셋을 살펴보고 생각하는 능력을 키워줌으로써 정량적 도구들이 어떻게 우리의 상상력을 넓히는 데 도움을 주는지 매일 목격합니다. 따라서 저희는 분석의 사용을 이끌어 줄 중요한 질문을 상상해야 합니다.

# 풋볼 예제

그린베이 패커스(Green Bay Packers)가 홈 필드 어드밴티지를 가지고 있는지 알고 싶다고 가정해 봅시다. 꽁꽁 얼어붙은 툰드라(Frozen Tundra)가 정말로 사람들이 말하는 만큼의 이점인지, 아니면 팬들이 동상에 걸릴 더 큰 위험을 감수하면서 피 같은 돈을 낭비하고 있는 것인지에 대해 친구와 의견이 다를 수 있습니다. 개념적으로 우리는 다음 단계를 거칩니다.

1.  질문에 답하는 데 도움이 되는 데이터를 찾습니다.

2.  질문에 답하는 데 도움이 되는 형식으로 데이터를 랭글링(조작)합니다.

3.  플롯을 그리고 요약 통계를 계산하여 데이터를 탐색합니다.

4.  관찰 결과를 정량화하고 확인하는 데 도움이 되는 모델을 맞춥니다.

5.  마지막으로 중요한 것은, 우리의 결과를 공유하는 것입니다 (선택 사항이지만 아마도 중요하게, 우리는 데이터로 답한 "질문"에 대한 내기를 결판냅니다).

패커스 홈 필드 어드밴티지 예제의 경우, 구체적인 단계는 다음과 같습니다.

1.  무료로 사용할 수 있는 데이터를 가져오기 위해 [`nflreadr` 패키지](https://nflreadr.nflverse.com)를 사용합니다.

2.  각 게임의 점수 차이를 제공하도록 데이터를 랭글링합니다.

3.  데이터 시각화를 돕기 위해 <a href="#img-gb_points" data-type="xref">그림 P-1</a>과 같은 플롯을 만듭니다.

4.  모델을 사용하여 홈 경기와 원정 경기의 평균 점수 차이를 추정합니다.

5.  이 주제에 대해 논쟁을 벌였던 친구들과 결과를 공유합니다.

데이터에 따르면, 패커스는 일반적으로 원정 경기에 비해 홈 경기에서 점수 차이가 2점 더 높습니다. 이 숫자는 시간이 지남에 따라 크게 변했지만, 이는 리그 전반에서 홈 필드 어드밴티지가 어느 정도일 것이라고 가정하는 바와 일치합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_Preface01.png" alt="2016년부터 2022년까지 홈 및 원정 경기 동안의 그린베이 점수 차이." />
<h6 id="figure-p-1.-green-bay-score-differential-in-home-and-away-games-from-2016-to-2022">그림 P-1. 2016년부터 2022년까지 홈 및 원정 경기에서의 그린베이 점수 차이</h6>
</figure>

이 관찰 결과가 여러분에게 더 많은 질문을 제기하기를 바랍니다. 예를 들어, 이 추정치 주변에는 얼마나 많은 변동성이 존재할까요? 팬들이 패커스 홈 경기장에 참석할 수 없었던 2020년을 제외하면 어떻게 될까요? 홈 필드 어드밴티지는 전반전이나 후반전 중 어디에 더 많은 영향을 미칠까요? 경기 수준과 스케줄 차이에 대해서는 어떻게 조정할까요? 이동 거리가 홈 필드 어드밴티지에 어떤 영향을 미칠까요? 친숙함은 어떨까요? 아니면 친숙함과 근접성이 관련되어 있기 때문에 둘 다 영향을 미칠까요? 날씨 차이는 홈 필드 어드밴티지에 어떤 영향을 미칠까요? 홈 필드 어드밴티지가 경기를 지는 것과 비교하여 경기를 이기는 데 어떤 영향을 미칠까요?

패커스가 승리한 경기와 패배한 경기의 홈 필드 어드밴티지를 빠르게 살펴보겠습니다. <a href="#img-gb_points_2" data-type="xref">그림 P-2</a>를 참조하세요. 이 그림의 데이터 요약에 따르면, 패커스는 홈 경기에 비해 원정 경기에서 1점을 더 차이 나게 지지만, 홈 경기에서는 원정 경기에서보다 무려 6점 더 차이 나게 승리합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_Preface02.png" alt="2016년부터 2022년까지 승리 및 패배한 경기에서 홈 및 원정 경기 동안의 그린베이 점수 차이." />
<h6 id="figure-p-2.-green-bay-score-differential-in-home-and-away-games-from-2016-to-2022-for-winning-and-losing-games">그림 P-2. 2016년부터 2022년까지 승리 및 패배한 경기에 대한 홈 및 원정 경기에서의 그린베이 점수 차이</h6>
</figure>

그렇긴 하지만, 그린베이 패커스는 좋은 팀이며 어떤 상황에서든 양의 점수 차이를 가질 가능성이 높습니다. 홈 필드 어드밴티지에 대한 질문은 사소한 문제가 아니며 수십 년 동안 분석가와 베팅자 모두를 당혹스럽게 만들었습니다. 바라건대 이 예제가 여러분에게 더 많은 호기심을 자극했기를 바랍니다. 그렇다면 여러분은 제대로 된 책을 읽고 있는 것입니다.

1단계인 풋볼 데이터 수집은 <a href="ch01.html#sec-introduction" data-type="xref">1장</a>에서 다룹니다. 2단계인 데이터 탐색은 <a href="ch02.html#sec-EDA-stable" data-type="xref">2장</a>에서 다룹니다. 3단계인 데이터 랭글링은 책 전체의 사례 연구와 더불어 부록 <a href="app02.html#sec-ssdw-pass" data-type="xref" data-xrefstyle="select:labelnumber">B</a> 및 <a href="app03.html#sec-app-dw" data-type="xref" data-xrefstyle="select:labelnumber">C</a>에서 다룹니다. 4단계에 관해서는 <a href="ch02.html#sec-EDA-stable" data-type="xref">2장</a>과 <a href="app02.html#sec-ssdw-pass" data-type="xref">부록 B</a>에서 기초 통계와 함께 설명하고, 이어서 모델에 대해서는 <a href="ch03.html#sec-lm-ryoa" data-type="xref" data-xrefstyle="select:labelnumber">3장</a>부터 <a href="ch08.html#sec-pca-clus" data-type="xref" data-xrefstyle="select:labelnumber">8장</a>까지 다룹니다. 5단계는 저희가 발견한 내용에 대해 설명하면서 다양한 장 전체에서 다룹니다. 마지막으로, 저희가 매일 사용하는 몇 가지 고급 도구를 설명하는 <a href="ch09.html#sec-advanced-tool" data-type="xref">9장</a>으로 책을 마무리합니다.

# 이 책에서 배울 내용

풋볼 분석에 대한 여정을 시작하는 데 도움이 되는 자료를 이 책에 포함했습니다. 열정적인 팬에게는 우리 책이 시작하고 실행하기에 충분할 수 있습니다. 정량적 풋볼 분석가가 되기를 열망하는 사람들에게 우리 책이 도약대가 되기를 바랍니다. 데이터 과학자가 되거나 데이터 과학 기술을 향상시키고자 하는 사람들을 위해 우리 책은 질문에 답하기 위해 데이터를 사용하는 방법에 대한 실용적인 예를 제공합니다. 우리는 구체적으로 다음을 가르칩니다.

- 데이터를 시각화하는 방법

- 데이터를 요약하는 방법

- 데이터를 모델링하는 방법

- 데이터 분석 결과를 발표하는 방법

- 이전의 기법들을 사용하여 스토리를 전달하는 방법

# 이 책에서 사용되는 표기 규칙

이 책에서는 다음의 표기 규칙을 사용합니다.

_이탤릭체(Italic)_  
새로운 용어, URL, 이메일 주소, 파일 이름, 파일 확장자를 나타냅니다.

`고정 너비(Constant width)`  
프로그램 리스팅뿐만 아니라 변수나 함수 이름, 데이터베이스, 데이터 유형, 환경 변수, 문(statement), 키워드와 같은 프로그램 요소를 나타내기 위해 단락 내에서도 사용됩니다.

**`고정 너비 굵게(Constant width bold)`**  
사용자가 문자 그대로 입력해야 하는 명령어나 기타 텍스트를 보여줍니다.

_고정 너비 이탤릭체(Constant width italic)_  
사용자가 제공하는 값이나 컨텍스트에 따라 결정되는 값으로 대체되어야 하는 텍스트를 보여줍니다.

###### 팁(Tip)

이 요소는 팁이나 제안을 의미합니다.

###### 참고(Note)

이 요소는 일반적인 참고 사항을 의미합니다.

###### 경고(Warning)

이 요소는 경고나 주의를 나타냅니다.

# 코드 예제 사용하기

보충 자료(코드 예제, 연습 문제 등)는 <a href="https://github.com/raerickson/football_book_code" class="bare"><em>https://github.com/raerickson/football_book_code</em></a>에서 다운로드할 수 있습니다.

코드 예제 사용에 있어 기술적인 질문이나 문제가 있다면 <a href="mailto:support@oreilly.com" class="email"><em>support@oreilly.com</em></a>으로 이메일을 보내주세요.

이 책은 여러분이 작업을 완수하는 데 도움을 주기 위해 존재합니다. 일반적으로 이 책과 함께 예제 코드가 제공되는 경우, 여러분의 프로그램과 문서에 그것을 사용할 수 있습니다. 코드의 상당 부분을 복제하는 경우가 아니라면 권한을 얻기 위해 저희에게 연락할 필요가 없습니다. 예를 들어, 이 책의 여러 코드 청크를 사용하는 프로그램을 작성하는 데는 권한이 필요하지 않습니다. O'Reilly 책의 예제를 판매하거나 배포하려면 권한이 필요합니다. 이 책을 인용하고 예제 코드를 인용하여 질문에 답하는 데는 권한이 필요하지 않습니다. 제품 문서에 이 책의 예제 코드를 상당 부분 포함하려면 권한이 필요합니다.

저희는 출처 표기를 감사하게 생각하지만 일반적으로 필수로 요구하지는 않습니다. 출처 표기에는 일반적으로 제목, 저자, 출판사, ISBN이 포함됩니다. 예: "_Football Analytics with Python and R_ by Eric A. Eager and Richard A. Erickson (O’Reilly). Copyright 2023 Eric A. Eager and Richard A. Erickson, 978-1-492-09962-8."

예제 코드의 사용이 공정 사용(fair use) 또는 위에 명시된 허용 범위를 벗어난다고 생각되면 <a href="mailto:permissions@oreilly.com" class="email"><em>permissions@oreilly.com</em></a>으로 자유롭게 연락해 주세요.

# O’Reilly 온라인 학습

###### 참고(Note)

40년 이상 동안 <a href="https://oreilly.com" class="orm:hideurl"><em>O’Reilly Media</em></a>는 기업이 성공할 수 있도록 기술 및 비즈니스 교육, 지식, 통찰력을 제공해 왔습니다.

전문가와 혁신가로 구성된 고유의 네트워크는 책, 기사, 온라인 학습 플랫폼을 통해 지식과 전문성을 공유합니다. O’Reilly의 온라인 학습 플랫폼을 사용하면 라이브 교육 과정, 심층 학습 경로, 대화형 코딩 환경, 그리고 O’Reilly와 200개 이상의 기타 출판사가 제공하는 방대한 텍스트 및 비디오 모음에 온디맨드로 액세스할 수 있습니다. 자세한 내용은 <a href="https://oreilly.com" class="orm:hideurl"><em>https://oreilly.com</em></a>을 방문하세요.

# 연락처

이 책에 관한 의견 및 질문은 출판사로 보내주시기 바랍니다.

- O’Reilly Media, Inc.
- 1005 Gravenstein Highway North
- Sebastopol, CA 95472
- 800-889-8969 (미국 또는 캐나다 내)
- 707-829-7019 (국제 또는 지역)
- 707-829-0104 (팩스)
- <a href="mailto:support@oreilly.com" class="email"><em>support@oreilly.com</em></a>
- [_https://www.oreilly.com/about/contact.html_](https://www.oreilly.com/about/contact.html)

저희는 정오표, 예제 및 기타 추가 정보를 나열하는 이 책의 웹 페이지를 마련했습니다. <a href="https://oreil.ly/football-analytics" class="bare"><em>https://oreil.ly/football-analytics</em></a>에서 이 페이지에 접속할 수 있습니다.

저희 책과 과정에 대한 뉴스와 정보를 원하시면 <a href="https://oreilly.com" class="bare"><em>https://oreilly.com</em></a>을 방문하세요.

LinkedIn에서 찾기: <a href="https://linkedin.com/company/oreilly-media" class="bare"><em>https://linkedin.com/company/oreilly-media</em></a>

Twitter 팔로우: <a href="https://twitter.com/oreillymedia" class="bare"><em>https://twitter.com/oreillymedia</em></a>

YouTube에서 시청하기: <a href="https://youtube.com/oreillymedia" class="bare"><em>https://youtube.com/oreillymedia</em></a>

# 감사의 글

지원해주신 Michelle Smith, Corbin Collins, Clare Laylock, Aleeya Rahman을 포함한 O’Reilly의 편집자들에게 감사드립니다. 저희 책을 꼼꼼하게 편집해 주신 Sharon Wilkey, 교정을 도와주신 Amnet Systems LLC의 Larry Baker, 인덱스를 만들어 주신 nSight, Inc.의 Cheryl Lenser에게도 감사를 표합니다. O'Reilly Atlas 시스템과 관련하여 기술적 지원을 제공해주신 Nick Adams와 Danny Elfanbaum에게 감사합니다. 기술적인 피드백을 주신 Boyan Angelov, Richie Cotton, Matthew Coller, Molly Creagar, Ryan Day, Haley English, Chester Ismay, Kaelen Medeiros, George Mound, John Oliver, Tobias Zwingmann에게 감사를 전합니다. 또한 성공적인 책 제안서를 작성하는 데 팁을 주신 Richie Cotton에게도 감사의 말씀을 전합니다.

에릭은 아무리 터무니없더라도 그의 꿈을 항상 지지하고 인내해 준 아내 스테파니에게 감사를 전하고 싶습니다. 또한 PFF를 설립하여 그가 지금까지 해온 모든 일의 바탕이 되는 플랫폼을 제공해 준 Neil Hornsby의 선견지명에 감사드리며, 2020년 가을에 그의 이메일에 답장을 주었던 Thomas Dimitroff에게도 감사의 마음을 표합니다. 아울러 2022년 SumerSports를 창립한 Paul과 Jack Jones의 비전에도 감사를 드립니다. 마지막으로 에릭은 본인들은 큰 풋볼 팬이 아님에도 불구하고 어린 시절 내내 그의 열정에 불을 지펴 주신 부모님께 감사드립니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with 부록 &amp; R/assets/fapr_Preface03.png" alt="걷고 있는 마고(왼쪽)와 세이디(오른쪽). 누가 누구를 산책시키고 있는 걸까요? 사진 제공 리처드 에릭슨(Richard Erickson)" />
<h6 id="figure-p-3.-margo-left-and-sadie-right-walking.-who-is-walking-whom-photo-by-richard-erickson">그림 P-3. 걷고 있는 마고(왼쪽)와 세이디(오른쪽). 누가 누구를 산책시키고 있는 걸까요? (사진: 리처드 에릭슨)</h6>
</figure>

리처드는 딸 마고가 잠든 후에 이 글을 쓸 수 있도록 잘 자준 것에 감사합니다. 또한 글을 쓰는 동안 산책을 포기하고 스트레칭 휴식을 취하도록 참을성 있게(그리고 조바심 내며) 일깨워 준 세이디에게도 감사드립니다(<a href="#img-m_and_s" data-type="xref">그림 P-3</a>). 리처드는 또한 그를 호기심 많게 키워주신 부모님과, 그가 프로그래밍을 배우도록 자극하고 이 책의 제안서 작성을 도와준 동생에게도 감사를 전합니다. 마지막으로 리처드는 이 책을 쓰는 동안 지원을 아끼지 않은 Hale, Skemp, Hanson, Skemp, & Sleik의 Tom Horvath와 다른 분들께 감사드립니다.
