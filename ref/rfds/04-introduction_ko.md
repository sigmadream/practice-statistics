# 서론

데이터 과학은 원시 데이터를 이해(understanding), 통찰(insight), 지식(knowledge)으로 변환할 수 있게 해주는 흥미로운 분야입니다. *R for Data Science*의 목표는 효율적이고 재현 가능한 데이터 과학을 수행하고, 그 과정에서 재미를 느낄 수 있게 해주는 중요한 R 도구들을 배우도록 돕는 것입니다! 이 책을 읽고 나면, R의 좋은 부분들을 사용하여 다양하고 광범위한 데이터 과학 문제들을 해결할 수 있는 도구들을 갖게 될 것입니다.

# 제2판 서문

_R for Data Science (R4DS)_ 제2판에 오신 것을 환영합니다! 이번 판은 1판의 대대적인 개정판으로, 더 이상 유용하지 않다고 생각되는 자료를 삭제하고 1판에 포함되었으면 했던 자료를 추가했으며, 모범 사례의 변화를 반영하여 텍스트와 코드를 전반적으로 업데이트했습니다. 또한 새로운 공동 저자인 미네 체틴카야-룬델(Mine Çetinkaya-Rundel)을 맞이하게 되어 매우 기쁩니다. 그녀는 저명한 데이터 과학 교육자이자 Posit(구 RStudio)의 동료입니다.

큰 변경 사항에 대한 간략한 요약은 다음과 같습니다.

- 책의 첫 번째 부분의 이름이 "전체 게임(Whole Game)"으로 변경되었습니다. 이 섹션의 목표는 세부 사항에 들어가기 전에 데이터 과학의 "전체 게임"에 대한 대략적인 세부 정보를 제공하는 것입니다.

- 책의 두 번째 부분은 "시각화(Visualize)"입니다. 이 부분은 1판에 비해 데이터 시각화 도구와 모범 사례를 더 자세히 다룹니다. 모든 세부 사항을 파악하기에 좋은 곳은 여전히 [ggplot2 book](https://oreil.ly/HNIie)이지만, 이제 R4DS에서 더 중요한 기술들을 대부분 다룹니다.

- 책의 세 번째 부분은 이제 "변환(Transform)"이라고 불리며 숫자, 논리형 벡터, 결측값에 대한 새로운 장이 추가되었습니다. 이 내용들은 이전에는 데이터 변환 장의 일부였지만 모든 세부 사항을 다루기 위해 훨씬 더 많은 공간이 필요했습니다.

- 책의 네 번째 부분은 "가져오기(Import)"라고 합니다. 일반 텍스트 파일을 읽는 것을 넘어 스프레드시트 작업, 데이터베이스에서 데이터 가져오기, 빅데이터 작업, 계층적 데이터를 직사각형 형태로 만들기(rectangling), 웹사이트에서 데이터 스크래핑하기 등을 다루는 새로운 장들로 구성되었습니다.

- "프로그래밍(Program)" 부분은 유지되었지만 함수 작성과 반복(iteration)의 중요한 부분에 집중하기 위해 완전히 다시 작성되었습니다. 지난 몇 년 동안 이 과정이 훨씬 쉬워지고 중요해짐에 따라, 함수 작성에는 이제 tidyverse 함수들을 래핑하는 방법(tidy evaluation의 문제를 다루는 방법)에 대한 세부 정보가 포함됩니다. 또한 야생에서 발견되는(?) 실제 R 코드에서 자주 볼 수 있는 중요한 기본 R 함수들에 대한 새로운 장을 추가했습니다.

- "모델링(Modeling)" 부분은 삭제되었습니다. 모델링을 충분히 제대로 다룰 공간이 없었고, 이제 훨씬 더 좋은 리소스들이 많이 있기 때문입니다. 일반적으로 [tidymodels 패키지](https://oreil.ly/0giAa)를 사용하고 Max Kuhn과 Julia Silge가 쓴 [_Tidy Modeling with R_](https://oreil.ly/9Op9s) (O’Reilly)을 읽을 것을 권장합니다.

- "소통(Communicate)" 부분은 유지되었지만 R Markdown 대신 [Quarto](https://oreil.ly/_6LNH)를 특징으로 하도록 전면 업데이트되었습니다. 이 책의 이번 판은 Quarto로 작성되었으며, 의심할 여지 없이 미래의 도구입니다.

# 무엇을 배울 것인가

데이터 과학은 방대한 분야이며 단 한 권의 책을 읽고 모든 것을 마스터할 수는 없습니다. 이 책은 중요한 도구들에 대한 탄탄한 기초를 제공하고, 필요할 때 더 배울 수 있는 리소스를 찾을 수 있는 충분한 지식을 제공하는 것을 목표로 합니다. 전형적인 데이터 과학 프로젝트의 단계 모델은 <a href="#fig-ds-diagram" data-type="xref">그림 I-1</a>과 같습니다.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0001.png" alt="A diagram displaying the data science cycle: Import -&gt; Tidy -&gt; Understand (which has the phases Transform -&gt; Visualize -&gt; Model in a cycle) -&gt; Communicate. Surrounding all of these is Communicate. " /></p>
<h6 id="figure-i-1.-in-our-model-of-the-data-science-process-you-start-with-data-import-and-tidying.-next-you-understand-your-data-with-an-iterative-cycle-of-transforming-visualizing-and-modeling.-you-finish-the-process-by-communicating-your-results-to-other-humans.">그림 I-1. 우리가 제시하는 데이터 과학 프로세스 모델에서는 데이터 가져오기와 깔끔하게 정리하기(tidying)로 시작합니다. 다음으로 변환, 시각화, 모델링의 반복적인 주기를 통해 데이터를 이해합니다. 마지막으로 그 결과를 다른 사람들에게 전달하는 것으로 프로세스를 마칩니다.</h6>
</figure>

먼저 데이터를 R로 _가져와야(import)_ 합니다. 이는 일반적으로 파일, 데이터베이스 또는 웹 애플리케이션 프로그래밍 인터페이스(API)에 저장된 데이터를 가져와 R의 데이터 프레임으로 로드하는 것을 의미합니다. 데이터를 R로 가져올 수 없다면 데이터 과학을 수행할 수 없습니다!

데이터를 가져온 후에는 _깔끔하게 정리(tidy)_ 하는 것이 좋습니다. 데이터를 깔끔하게 정리한다는 것은 데이터셋의 의미 구조와 데이터가 저장된 방식이 일치하도록 일관된 형태로 저장하는 것을 의미합니다. 간단히 말해, 데이터가 깔끔할 때 각 열은 변수이고 각 행은 관측치입니다. 깔끔한 데이터가 중요한 이유는 일관된 구조 덕분에 데이터를 다른 함수들에 맞게 변형하려 애쓰는 대신, 데이터에 대한 질문에 답하는 데 노력을 집중할 수 있게 해주기 때문입니다.

깔끔한 데이터를 확보한 후의 일반적인 다음 단계는 데이터를 _변환(transform)_ 하는 것입니다. 변환에는 관심 있는 관측치(한 도시의 모든 사람 또는 작년의 모든 데이터)로 좁히기, 기존 변수들의 함수인 새로운 변수(거리와 시간으로 속도 계산) 생성하기, 그리고 요약 통계량(개수 또는 평균) 계산하기가 포함됩니다. 깔끔하게 정리하고 변환하는 것을 합쳐서 _랭글링(wrangling)_ 이라고 부르는데, 데이터를 작업하기 자연스러운 형태로 만드는 과정이 마치 싸움처럼 느껴질 때가 많기 때문입니다!

필요한 변수가 있는 깔끔한 데이터를 얻고 나면, 지식 생성의 두 가지 주요 엔진인 시각화와 모델링이 있습니다. 이들은 상호 보완적인 장단점을 가지고 있으므로, 실제 데이터 분석에서는 두 과정 사이를 여러 번 반복하게 됩니다.

_시각화(Visualization)_ 는 근본적으로 인간의 활동입니다. 훌륭한 시각화는 예상치 못한 것들을 보여주거나 데이터에 대한 새로운 질문을 제기합니다. 또한 잘못된 질문을 하고 있거나 다른 데이터를 수집해야 한다는 점을 암시할 수도 있습니다. 시각화는 놀라움을 줄 수 있지만 인간의 해석이 필요하기 때문에 확장성(scale)이 특별히 좋지는 않습니다.

_모델(Model)_ 은 시각화를 보완하는 도구입니다. 질문이 충분히 구체화되면 모델을 사용하여 답을 얻을 수 있습니다. 모델은 근본적으로 수학적 또는 계산적 도구이므로 일반적으로 확장성이 좋습니다. 그렇지 않더라도, 사람의 뇌를 더 사는 것보다 컴퓨터를 더 사는 것이 보통 더 저렴합니다! 하지만 모든 모델은 가정을 바탕으로 하며, 그 본질상 모델 스스로가 자신의 가정을 의심할 수는 없습니다. 즉, 모델은 근본적으로 여러분에게 놀라움을 줄 수는 없습니다.

데이터 과학의 마지막 단계는 _소통(communication)_ 이며, 이는 모든 데이터 분석 프로젝트에서 절대적으로 중요한 부분입니다. 모델과 시각화가 데이터를 이해하는 데 아무리 큰 도움을 주었다 해도, 그 결과를 다른 사람에게 전달할 수 없다면 아무 소용이 없습니다.

이 모든 도구들을 둘러싸고 있는 것이 _프로그래밍(programming)_ 입니다. 프로그래밍은 데이터 과학 프로젝트의 거의 모든 부분에서 사용되는 교차적(cross-cutting) 도구입니다. 성공적인 데이터 과학자가 되기 위해 전문가 수준의 프로그래머가 될 필요는 없지만, 프로그래밍에 대해 더 많이 배울수록 가치가 있습니다. 더 나은 프로그래머가 되면 일반적인 작업을 자동화하고 새로운 문제를 더 쉽게 해결할 수 있기 때문입니다.

모든 데이터 과학 프로젝트에서 이러한 도구들을 사용하게 되지만, 대부분의 프로젝트에서 이것만으로는 충분하지 않습니다. 대략적인 80/20 법칙이 작용합니다. 이 책에서 배울 도구들을 사용하여 모든 프로젝트의 약 80%를 해결할 수 있지만, 나머지 20%를 해결하려면 다른 도구들이 필요합니다. 이 책 전반에 걸쳐 더 많은 것을 배울 수 있는 리소스를 안내해 드릴 것입니다.

# 이 책의 구성

앞서 설명한 데이터 과학 도구들은 대체로 분석에서 사용하는 순서에 따라 구성되어 있습니다(물론 실제로는 이 단계들을 여러 번 반복하게 됩니다). 하지만 우리의 경험상, 데이터를 가져오고 깔끔하게 정리하는 방법을 먼저 배우는 것은 최적이 아닙니다. 80%의 시간은 일상적이고 지루하며, 나머지 20%의 시간은 이상하고 좌절감을 주기 때문입니다. 이는 새로운 주제를 배우기 시작하기에 좋은 방법이 아닙니다! 대신 우리는 이미 가져와서 정리된 데이터의 시각화와 변환부터 시작할 것입니다. 그렇게 하면 나중에 여러분 자신의 데이터를 수집하고 정리할 때, 그 고통이 가치 있다는 것을 알기 때문에 동기 부여를 높게 유지할 수 있습니다.

각 장 내에서는 큰 그림을 볼 수 있도록 동기를 부여하는 예제로 시작한 다음 세부 사항으로 들어가는 일관된 패턴을 고수하려고 노력했습니다. 책의 각 섹션에는 배운 내용을 연습할 수 있는 연습 문제가 짝을 이룹니다. 연습 문제를 건너뛰고 싶은 유혹이 들 수 있지만, 실제 문제를 통해 연습하는 것보다 더 나은 학습 방법은 없습니다.

# 배우지 않을 내용

이 책에서 다루지 않는 중요한 주제들이 몇 가지 있습니다. 우리는 가능한 한 빨리 실전에 돌입할 수 있도록 필수적인 내용에만 무자비하게 집중하는 것이 중요하다고 믿습니다. 즉, 이 책에서 모든 중요한 주제를 다룰 수는 없습니다.

## 모델링

모델링은 데이터 과학에서 매우 중요하지만 방대한 주제이므로 불행히도 여기서는 충분히 다룰 공간이 없습니다. 모델링에 대해 더 자세히 알아보려면 Max Kuhn과 Julia Silge(O’Reilly)가 쓴 [_Tidy Modeling with R_](https://oreil.ly/9Op9s)을 강력히 추천합니다. 이 책은 tidymodels 패키지 제품군을 가르쳐 주며, 이름에서 짐작할 수 있듯이 이 책에서 사용하는 tidyverse 패키지들과 많은 규칙을 공유합니다.

## 빅데이터

이 책은 작고 메모리에 적재 가능한(in-memory) 데이터셋에 자랑스럽고 주로 초점을 맞춥니다. 작은 데이터에 대한 경험이 없으면 빅데이터를 다룰 수 없기 때문에 이것이 시작하기에 올바른 방법입니다. 이 책의 대부분에서 배우는 도구들은 수백 메가바이트의 데이터를 쉽게 처리할 수 있으며, 조금만 주의를 기울이면 일반적으로 몇 기가바이트의 데이터를 다루는 데에도 사용할 수 있습니다. 또한 데이터베이스와 파케이(parquet) 파일에서 데이터를 가져오는 방법도 보여드릴 텐데, 둘 다 빅데이터를 저장하는 데 자주 사용됩니다. 반드시 전체 데이터셋으로 작업할 수 있는 것은 아니지만, 관심 있는 질문에 답하기 위해 부분 집합(subset)이나 하위 표본(subsample)만 필요하므로 문제되지 않습니다.

정기적으로 더 큰 데이터(10~100GB)를 다루는 경우, [data.table](https://oreil.ly/GG4Et)에 대해 자세히 알아보는 것을 추천합니다. tidyverse와 다른 인터페이스를 사용하고 다른 규칙들을 몇 가지 배워야 하기 때문에 여기서는 가르치지 않습니다. 하지만 엄청나게 빠르며, 대용량 데이터를 다루는 경우 시간을 투자하여 배울 만한 성능상의 이점이 있습니다.

## Python, Julia, 그리고 기타 언어들

이 책에서는 Python, Julia, 또는 데이터 과학에 유용한 기타 프로그래밍 언어에 대해서는 전혀 배우지 않습니다. 이는 이러한 도구들이 나쁘다고 생각해서가 아닙니다. 전혀 그렇지 않습니다! 그리고 실제로 대부분의 데이터 과학 팀은 여러 언어를 혼합하여 사용하며, 보통 최소한 R과 Python을 함께 사용합니다. 하지만 우리는 한 번에 하나의 도구를 마스터하는 것이 최선이며, R이 시작하기에 좋은 언어라고 굳게 믿습니다.

# 사전 지식

이 책을 최대한 활용하기 위해 여러분이 이미 알고 있을 것이라고 몇 가지 가정을 했습니다. 기본적으로 숫자 감각이 있어야 하며, 기본적인 프로그래밍 경험이 이미 있다면 도움이 됩니다. 프로그래밍을 한 번도 해본 적이 없다면, Garrett Grolemund(O'Reilly)의 [Hands-On Programming with R](https://oreil.ly/8uiH5)이 이 책의 유용한 보조 자료가 될 수 있습니다.

이 책의 코드를 실행하려면 네 가지가 필요합니다. R, RStudio, *tidyverse*라는 R 패키지 모음, 그리고 몇 가지 다른 패키지들입니다. 패키지는 재현 가능한 R 코드의 기본 단위입니다. 여기에는 재사용 가능한 함수, 사용 방법을 설명하는 문서, 샘플 데이터가 포함되어 있습니다.

## R

R을 다운로드하려면 *C*omprehensive _R_ *A*rchive *N*etwork인 [CRAN](https://oreil.ly/p3_RG)으로 이동하세요. R의 새로운 주요 버전은 1년에 한 번 나오며, 매년 2~3번의 마이너 릴리스가 있습니다. 정기적으로 업데이트하는 것이 좋습니다. 업그레이드는, 특히 모든 패키지를 다시 설치해야 하는 주요 버전의 경우 약간 번거로울 수 있지만 미루면 오히려 더 힘들어집니다. 이 책에서는 R 4.2.0 이상을 권장합니다.

## RStudio

RStudio는 R 프로그래밍을 위한 통합 개발 환경(IDE)으로, [RStudio 다운로드 페이지](https://oreil.ly/pxF-k)에서 다운로드할 수 있습니다. RStudio는 1년에 몇 번 업데이트되며 새 버전이 나오면 자동으로 알려주므로 다시 확인할 필요가 없습니다. 새롭고 훌륭한 최신 기능들을 활용하려면 정기적으로 업그레이드하는 것이 좋습니다. 이 책을 위해서는 최소한 RStudio 2022.02.0을 설치해야 합니다.

RStudio를 시작하면(<a href="#fig-rstudio-console" data-type="xref">그림 I-2</a>) 콘솔(console) 창과 출력(output) 창이라는 두 가지 주요 영역이 인터페이스에 보입니다. 지금으로서는 콘솔 창에 R 코드를 입력하고 Enter 키를 눌러 실행한다는 것만 알면 됩니다. 진행하면서 더 자세히 배우게 될 것입니다!<sup><a href="introduction01.html#idm44771332141712" id="idm44771332141712-marker" data-type="noteref">1</a></sup>

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0002.png" alt="The RStudio IDE with the panes Console and Output highlighted." /></p>
<h6 id="figure-i-2.-the-rstudio-ide-has-two-key-regions-type-r-code-in-the-console-pane-on-the-left-and-look-for-plots-in-the-output-pane-on-the-right.">그림 I-2. RStudio IDE에는 두 가지 핵심 영역이 있습니다. 왼쪽의 콘솔 창에 R 코드를 입력하고, 오른쪽의 출력 창에서 그래프를 확인합니다.</h6>
</figure>

## Tidyverse

R 패키지도 몇 개 설치해야 합니다. R 패키지는 기본 R의 기능을 확장하는 함수, 데이터, 문서의 모음입니다. 패키지를 사용하는 것은 R을 성공적으로 사용하는 데 있어 핵심입니다. 이 책에서 배우게 될 대부분의 패키지들은 소위 tidyverse의 일부입니다. tidyverse의 모든 패키지들은 데이터와 R 프로그래밍에 대한 공통된 철학을 공유하며, 서로 함께 작동하도록 설계되었습니다.

코드 한 줄로 전체 tidyverse를 설치할 수 있습니다.

`install.packages("tidyverse")`

컴퓨터의 콘솔에 위 코드 줄을 입력한 다음 Enter 키를 눌러 실행하세요. R이 CRAN에서 패키지를 다운로드하여 컴퓨터에 설치합니다.

패키지를 로드할 때까지는 해당 패키지의 함수, 객체 또는 도움말 파일을 사용할 수 없습니다. 패키지를 설치하고 나면 <a href="https://rdrr.io/r/base/library.html" class="orm:hideurl"><code>library()</code></a> 함수를 사용하여 패키지를 로드할 수 있습니다.

```r
library(tidyverse)
#> ── Attaching core tidyverse packages ───────────────────── tidyverse 2.0.0 ──
#> ✔ dplyr     1.1.0.9000     ✔ readr     2.1.4
#> ✔ forcats   1.0.0          ✔ stringr   1.5.0
#> ✔ ggplot2   3.4.1          ✔ tibble    3.1.8
#> ✔ lubridate 1.9.2          ✔ tidyr     1.3.0
#> ✔ purrr     1.0.1
#> ── Conflicts ─────────────────────────────────────── tidyverse_conflicts() ──
#> ✖ dplyr::filter() masks stats::filter()
#> ✖ dplyr::lag()    masks stats::lag()
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all
#>   conflicts to become errors
```

이는 tidyverse가 dplyr, forcats, ggplot2, lubridate, purrr, readr, stringr, tibble, tidyr 등 9개의 패키지를 로드함을 알려줍니다. 이것들은 거의 모든 분석에서 사용하게 되므로 tidyverse의 _핵심(core)_ 으로 간주됩니다.

tidyverse의 패키지들은 꽤 자주 변경됩니다. <a href="https://tidyverse.tidyverse.org/reference/tidyverse_update.html" class="orm:hideurl"><code>tidyverse_update()</code></a>를 실행하여 업데이트가 있는지 확인할 수 있습니다.

## 기타 패키지들

다른 도메인의 문제를 해결하거나 근본적으로 다른 원칙을 바탕으로 설계되었기 때문에 tidyverse의 일부가 아닌 훌륭한 패키지들도 많이 있습니다. 이로 인해 더 좋거나 더 나쁘게 되는 것은 아니며, 단지 다를 뿐입니다. 다시 말해, tidyverse의 반대말은 messyverse(지저분한 세계)가 아니라, 상호 관련된 패키지들의 다른 많은 우주(universes)들입니다. R을 사용하여 더 많은 데이터 과학 프로젝트를 해결함에 따라 새로운 패키지와 데이터에 대한 새로운 사고 방식을 배우게 될 것입니다.

이 책에서는 tidyverse 외부의 패키지들도 많이 사용할 것입니다. 예를 들어, R을 배우는 과정에서 활용해볼 흥미로운 데이터셋들을 제공하므로 다음 패키지들을 사용할 것입니다.

`install.packages(c("arrow", "babynames", "curl", "duckdb", "gapminder", "ggrepel", "ggridges", "ggthemes", "hexbin", "janitor", "Lahman", "leaflet", "maps", "nycflights13", "openxlsx", "palmerpenguins", "repurrrsive", "tidymodels", "writexl"))`

또한 일회성 예제를 위해 선별된 다른 패키지들도 사용할 것입니다. 지금 설치할 필요는 없지만 다음과 같은 에러가 표시될 때를 기억하세요.

`library(ggrepel)`
`#> Error in library(ggrepel) : there is no package called ‘ggrepel’`

이는 패키지를 설치하기 위해 `install.packages("ggrepel")`을 실행해야 함을 의미합니다.

# R 코드 실행하기

이전 섹션에서는 R 코드를 실행하는 몇 가지 예를 보여주었습니다. 이 책의 코드는 다음과 같이 보입니다.

`1 + 2`
`#> [1] 3`

로컬 콘솔에서 같은 코드를 실행하면 다음과 같이 보입니다.

`> 1 + 2`
`[1] 3`

두 가지 주요 차이점이 있습니다. 콘솔에서는 _프롬프트(prompt)_ 라고 하는 `>` 뒤에 입력하지만, 책에서는 프롬프트를 표시하지 않습니다. 책에서는 결과가 `#>`로 주석 처리되어 있지만, 콘솔에서는 코드 바로 뒤에 나타납니다. 이 두 가지 차이점은 전자책 버전으로 작업하는 경우 책에서 코드를 쉽게 복사하여 콘솔에 붙여넣을 수 있음을 의미합니다.

책 전체에 걸쳐 코드를 참조할 때 일관된 표기 규칙을 사용합니다.

- 함수는 <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a> 또는 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>처럼 코드 글꼴로 표시되고 뒤에 괄호가 붙습니다.

- 다른 R 객체(데이터나 함수 인자 등)는 `flights` 또는 `x`처럼 괄호 없는 코드 글꼴로 표시됩니다.

- 때로는 객체가 어떤 패키지에서 왔는지 명확히 하기 위해 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>dplyr::mutate()</code></a> 또는 <a href="https://rdrr.io/pkg/nycflights13/man/flights.html" class="orm:hideurl"><code>nycflights13::flights</code></a>처럼 패키지 이름 뒤에 두 개의 콜론을 사용합니다. 이것도 유효한 R 코드입니다.

# 이 책에서 사용된 기타 규칙

이 책에서는 다음과 같은 타이포그래피 규칙을 사용합니다.

_이탤릭체(Italic)_  
URL과 이메일 주소를 나타냅니다.

`고정 폭(Constant width)`  
프로그램 코드뿐만 아니라 변수나 함수 이름, 데이터베이스, 데이터 유형, 환경 변수, 구문, 키워드, 파일 이름과 같은 프로그램 요소를 지칭할 때 단락 내에서 사용됩니다.

**`고정 폭 굵게(Constant width bold)`**  
사용자가 문자 그대로 입력해야 하는 명령어 또는 기타 텍스트를 나타냅니다.

_`고정 폭 이탤릭체(Constant width italic)`_  
사용자가 제공하는 값이나 컨텍스트에 따라 결정되는 값으로 대체되어야 하는 텍스트를 나타냅니다.

###### 참고 (Note)

이 요소는 일반적인 참고 사항을 의미합니다.

###### 경고 (Warning)

이 요소는 경고 또는 주의를 나타냅니다.

# O’Reilly 온라인 학습

###### 참고

40년 이상 동안 <a href="https://oreilly.com" class="orm:hideurl"><em>O’Reilly Media</em></a>는 기술 및 비즈니스 교육, 지식, 통찰력을 제공하여 기업의 성공을 도왔습니다.

우리의 고유한 전문가 및 혁신가 네트워크는 도서, 기사, 온라인 학습 플랫폼을 통해 그들의 지식과 전문성을 공유합니다. O'Reilly의 온라인 학습 플랫폼은 실시간 교육 과정, 심도 있는 학습 경로, 대화형 코딩 환경, O'Reilly 및 200개 이상의 다른 출판사가 제공하는 방대한 텍스트 및 비디오 컬렉션에 대한 온디맨드(on-demand) 액세스를 제공합니다. 더 자세한 정보는 <a href="https://oreilly.com" class="orm:hideurl"><em>https://oreilly.com</em></a>을 방문하세요.

# 연락처

이 책과 관련된 의견이나 질문은 출판사로 보내주시기 바랍니다.

- O’Reilly Media, Inc.
- 1005 Gravenstein Highway North
- Sebastopol, CA 95472
- 800-889-8969 (미국 또는 캐나다)
- 707-829-7019 (국제 전화 또는 현지)
- 707-829-0104 (팩스)
- <a href="mailto:support@oreilly.com" class="email"><em>support@oreilly.com</em></a>
- [_https://www.oreilly.com/about/contact.html_](https://www.oreilly.com/about/contact.html)

이 책의 웹 페이지를 마련해 두었으며, 오탈자, 예제 및 기타 추가 정보를 나열하고 있습니다. 이 페이지는 [_https://oreil.ly/r-for-data-science-2e_](https://oreil.ly/r-for-data-science-2e)에서 접속할 수 있습니다.

당사의 책과 과정에 대한 뉴스와 정보는 [_https://oreilly.com_](https://oreilly.com)을 방문하세요.

LinkedIn: [_https://linkedin.com/company/oreilly-media_](https://linkedin.com/company/oreilly-media)

Twitter: [_https://twitter.com/oreillymedia_](https://twitter.com/oreillymedia)

YouTube: [_https://www.youtube.com/oreillymedia_](https://www.youtube.com/oreillymedia)

# 감사의 말

이 책은 단지 해들리(Hadley), 미네(Mine), 개럿(Garrett)의 산물이 아니라 우리가 R 커뮤니티의 많은 사람들과 (직접 또는 온라인으로) 나눈 수많은 대화의 결과입니다. 여러분과 나눈 모든 대화에 진심으로 감사드립니다. 정말 감사합니다!

소중한 피드백을 주신 기술 리뷰어 분들께 감사드립니다. Ben Baumer, Lorna Barclay, Richard Cotton, Emma Rand, Kelly Bodwin.

이 책은 오픈 소스 방식으로 쓰여졌으며, 많은 분들이 pull request를 통해 기여해 주셨습니다. GitHub pull request를 통해 개선 사항을 기여해 주신 259분 모두에게 특별히 감사드립니다(사용자 이름 알파벳순): @a-rosenberg, Tim Becker (@a2800276), Abinash Satapathy (@Abinashbunty), Adam Gruer (@adam-gruer), adi pradhan (@adidoit), A. s. (@Adrianzo), Aep Hidyatuloh (@aephidayatuloh), Andrea Gilardi (@agila5), Ajay Deonarine (@ajay-d), @AlanFeder, Daihe Sui (@alansuidaihe), @alberto-agudo, @AlbertRapp, @aleloi, pete (@alonzi), Alex (@ALShum), Andrew M. (@amacfarland), Andrew Landgraf (@andland), @andyhuynh92, Angela Li (@angela-li), Antti Rask (@AnttiRask), LOU Xun (@aquarhead), @ariespirgel, @august-18, Michael Henry (@aviast), Azza Ahmed (@azzaea), Steven Moran (@bambooforest), Brian G. Barkley (@BarkleyBG), Mara Averick (@batpigandme), Oluwafemi OYEDELE (@BB1464), Brent Brewington (@bbrewington), Bill Behrman (@behrman), Ben Herbertson (@benherbertson), Ben Marwick (@benmarwick), Ben Steinberg (@bensteinberg), Benjamin Yeh (@bentyeh), Betul Turkoglu (@betulturkoglu), Brandon Greenwell (@bgreenwell), Bianca Peterson (@BinxiePeterson), Birger Niklas (@BirgerNi), Brett Klamer (@bklamer), @boardtc, Christian (@c-hoh), Caddy (@caddycarine), Camille V Leonard (@camillevleonard), @canovasjm, Cedric Batailler (@cedricbatailler), Christina Wei (@christina-wei), Christian Mongeau (@chrMongeau), Cooper Morris (@coopermor), Colin Gillespie (@csgillespie), Rademeyer Vermaak (@csrvermaak), Chloe Thierstein (@cthierst), Chris Saunders (@ctsa), Abhinav Singh (@curious-abhinav), Curtis Alexander (@curtisalexander), Christian G. Warden (@cwarden), Charlotte Wickham (@cwickham), Kenny Darrell (@darrkj), David Kane (@davidkane9), David (@davidrsch), David Rubinger (@davidrubinger), David Clark (@DDClark), Derwin McGeary (@derwinmcgeary), Daniel Gromer (@dgromer), @Divider85, @djbirke, Danielle Navarro (@djnavarro), Russell Shean (@DOH-RPS1303), Zhuoer Dong (@dongzhuoer), Devin Pastoor (@dpastoor), @DSGeoff, Devarshi Thakkar (@dthakkar09), Julian During (@duju211), Dylan Cashman (@dylancashman), Dirk Eddelbuettel (@eddelbuettel), Edwin Thoen (@EdwinTh), Ahmed El-Gabbas (@elgabbas), Henry Webel (@enryH), Ercan Karadas (@ercan7), Eric Kitaif (@EricKit), Eric Watt (@ericwatt), Erik Erhardt (@erikerhardt), Etienne B. Racine (@etiennebr), Everett Robinson (@evjrob), @fellennert, Flemming Miguel (@flemmingmiguel), Floris Vanderhaeghe (@florisvdh), @funkybluehen, @gabrivera, Garrick Aden-Buie (@gadenbuie), Peter Ganong (@ganong123), Gerome Meyer (@GeroVanMi), Gleb Ebert (@gl-eb), Josh Goldberg (@GoldbergData), bahadir cankardes (@gridgrad), Gustav W Delius (@gustavdelius), Hao Chen (@hao-trivago), Harris McGehee (@harrismcgehee), @hendrikweisser, Hengni Cai (@hengnicai), Iain (@Iain-S), Ian Sealy (@iansealy), Ian Lyttle (@ijlyttle), Ivan Krukov (@ivan-krukov), Jacob Kaplan (@jacobkap), Jazz Weisman (@jazzlw), John Blischak (@jdblischak), John D. Storey (@jdstorey), Gregory Jefferis (@jefferis), Jeffrey Stevens (@JeffreyRStevens), 蒋雨蒙 (@JeldorPKU), Jennifer (Jenny) Bryan (@jennybc), Jen Ren (@jenren), Jeroen Janssens (@jeroenjanssens), @jeromecholewa, Janet Wesner (@jilmun), Jim Hester (@jimhester), JJ Chen (@jjchern), Jacek Kolacz (@jkolacz), Joanne Jang (@joannejang), @johannes4998, John Sears (@johnsears), @jonathanflint, Jon Calder (@jonmcalder), Jonathan Page (@jonpage), Jon Harmon (@jonthegeek), JooYoung Seo (@jooyoungseo), Justinas Petuchovas (@jpetuchovas), Jordan (@jrdnbradford), Jeffrey Arnold (@jrnold), Jose Roberto Ayala Solares (@jroberayalas), Joyce Robbins (@jtr13), @juandering, Julia Stewart Lowndes (@jules32), Sonja (@kaetschap), Kara Woo (@karawoo), Katrin Leinweber (@katrinleinweber), Karandeep Singh (@kdpsingh), Kevin Perese (@kevinxperese), Kevin Ferris (@kferris10), Kirill Sevastyanenko (@kirillseva), Jonathan Kitt (@KittJonathan), @koalabearski, Kirill Müller (@krlmlr), Rafał Kucharski (@kucharsky), Kevin Wright (@kwstat), Noah Landesberg (@landesbergn), Lawrence Wu (@lawwu), @lindbrook, Luke W Johnston (@lwjohnst86), Kara de la Marck (@MarckK), Kunal Marwaha (@marwahaha), Matan Hakim (@matanhakim), Matthias Liew (@MatthiasLiew), Matt Wittbrodt (@MattWittbrodt), Mauro Lepore (@maurolepore), Mark Beveridge (@mbeveridge), @mcewenkhundi, mcsnowface, PhD (@mcsnowface), Matt Herman (@mfherman), Michael Boerman (@michaelboerman), Mitsuo Shiota (@mitsuoxv), Matthew Hendrickson (@mjhendrickson), @MJMarshall, Misty Knight-Finley (@mkfin7), Mohammed Hamdy (@mmhamdy), Maxim Nazarov (@mnazarov), Maria Paula Caldas (@mpaulacaldas), Mustafa Ascha (@mustafaascha), Nelson Areal (@nareal), Nate Olson (@nate-d-olson), Nathanael (@nateaff), @nattalides, Ned Western (@NedJWestern), Nick Clark (@nickclark1000), @nickelas, Nirmal Patel (@nirmalpatel), Nischal Shrestha (@nischalshrestha), Nicholas Tierney (@njtierney), Jakub Nowosad (@Nowosad), Nick Pullen (@nstjhp), @olivier6088, Olivier Cailloux (@oliviercailloux), Robin Penfold (@p0bs), Pablo E. Garcia (@pabloedug), Paul Adamson (@padamson), Penelope Y (@penelopeysm), Peter Hurford (@peterhurford), Peter Baumgartner (@petzi53), Patrick Kennedy (@pkq), Pooya Taherkhani (@pooyataher), Y. Yu (@PursuitOfDataScience), Radu Grosu (@radugrosu), Ranae Dietzel (@Ranae), Ralph Straumann (@rastrau), Rayna M Harris (@raynamharris), @ReeceGoding, Robin Gertenbach (@rgertenbach), Jajo (@RIngyao), Riva Quiroga (@rivaquiroga), Richard Knight (@RJHKnight), Richard Zijdeman (@rlzijdeman), @robertchu03, Robin Kohrs (@RobinKohrs), Robin (@Robinlovelace), Emily Robinson (@robinsones), Rob Tenorio (@robtenorio), Rod Mazloomi (@RodAli), Rohan Alexander (@RohanAlexander), Romero Morais (@RomeroBarata), Albert Y. Kim (@rudeboybert), Saghir (@saghirb), Hojjat Salmasian (@salmasian), Jonas (@sauercrowd), Vebash Naidoo (@sciencificity), Seamus McKinsey (@seamus-mckinsey), @seanpwilliams, Luke Smith (@seasmith), Matthew Sedaghatfar (@sedaghatfar), Sebastian Kraus (@sekR4), Sam Firke (@sfirke), Shannon Ellis (@ShanEllis), @shoili, Christian Heinrich (@Shurakai), S’busiso Mkhondwane (@sibusiso16), SM Raiyyan (@sm-raiyyan), Jakob Krigovsky (@sonicdoe), Stephan Koenig (@stephan-koenig), Stephen Balogun (@stephenbalogun), Steven M. Mortimer (@StevenMMortimer), Stéphane Guillou (@stragu), Sulgi Kim (@sulgik), Sergiusz Bleja (@svenski), Tal Galili (@talgalili), Alec Fisher (@Taurenamo), Todd Gerarden (@tgerarden), Tom Godfrey (@thomasggodfrey), Tim Broderick (@timbroderick), Tim Waterhouse (@timwaterhouse), TJ Mahr (@tjmahr), Thomas Klebel (@tklebel), Tom Prior (@tomjamesprior), Terence Teo (@tteo), @twgardner2, Ulrik Lyngs (@ulyngs), Shinya Uryu (@uribo), Martin Van der Linden (@vanderlindenma), Walter Somerville (@waltersom), @werkstattcodes, Will Beasley (@wibeasley), Yihui Xie (@yihui), Yiming (Paul) Li (@yimingli), @yingxingwu, Hiroaki Yutani (@yutannihilation), Yu Yu Aung (@yuyu-aung), Zach Bogart (@zachbogart), @zeal626, and Zeki Akyol (@zekiakyol).

# 온라인 에디션

이 책의 온라인 버전은 책의 [GitHub 리포지토리](https://oreil.ly/8GLe7)에서 볼 수 있습니다. 실물 책의 재인쇄 사이에도 계속해서 발전할 것입니다. 책의 소스는 [_https://oreil.ly/Q8z_O_](https://oreil.ly/Q8z_O)에서 볼 수 있습니다. 이 책은 텍스트와 실행 가능한 코드를 결합한 책을 쓰기 쉽게 해주는 [Quarto](https://oreil.ly/_6LNH)를 기반으로 합니다.

<sup>[1](introduction01.html#idm44771332141712-marker)</sup> RStudio의 모든 기능에 대한 포괄적인 개요를 보려면 [RStudio 사용 설명서](https://oreil.ly/pRhEK)를 참조하세요.
