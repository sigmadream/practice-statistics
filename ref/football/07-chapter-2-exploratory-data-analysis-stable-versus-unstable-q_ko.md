# 2장. 탐색적 데이터 분석: 안정적인 쿼터백 통계 대 불안정적인 쿼터백 통계

어떤 학문 분야에서든 평범한 사람들과 진정으로 위대한 주제 전문가들, 또는 이제 막 경력을 시작한 전문가들과 평균적인 전문가들, 혹은 초보자들과 신입 전문가들을 구분 짓는 직관(흔히 *직감(gut feeling)*이라고 알려진) 수준이 존재할 수 있습니다. 풋볼에서 그러한 기술은 선수 평가에서 나타난다고들 합니다. 일부 스카우트들은 수년간 갈고닦은 훌륭한 직관을 통해 재능을 식별하는 요령을 가지고 있다고 여겨지기 때문입니다. 대학 풋볼에서 프로 무대로, 혹은 한 코치의 전술(scheme)에서 다른 코치의 전술로 상황이 바뀌어도 변하지 않는 선수의 특성은 인식하고 추가 조사를 수행해야 하는 반면, 측정할 수 없는(적어도 현재의 데이터와 도구를 사용하여) 선수의 결과물은 버려집니다. 선수 평가의 전문가들은 또한 자신들의 노력에 대한 결실로 최대한의 집단적 이익을 얻기 위해 이를 어떻게 적절하게 소통해야 하는지도 알고 있습니다.

전통적인 스카우트 방식과 풋볼 분석은 종종 서로 상충하는 것으로 여겨지지만, 선수의 통계적 평가에는 본질적으로 동일한 과정이 필요합니다. 훌륭한 풋볼 분석가는 선수(또는 여러 선수)의 데이터를 평가할 때 질문을 던질 올바른 데이터 사양, 사용할 생산성 지표, 통제해야 할 상황적 요인, 그리고 버려야 할 정보를 찾을 수 있습니다. 그러한 통찰력은 어떻게 얻을 수 있을까요? 스카우트가 하는 방식과 똑같습니다. 수년간의 의도적인 연습과 개선을 통해 분석가는 선수, 팀, 전술, 그리고 경기를 평가하는 도구 세트뿐만 아니라 적절한 시기에 적절한 질문을 던지는 요령도 얻게 됩니다.

###### 팁(Tip)

유명한 통계학자인 [존 튜키(John Tukey)는 다음과 같이 지적했습니다](https://oreil.ly/RVyMp). "잘못된 질문에 대한 정확한 답보다는, 흔히 모호하긴 하지만 올바른 질문에 대한 근사적인 답이 훨씬 낫습니다. 잘못된 질문에 대한 답은 항상 정밀하게 만들 수 있기 때문입니다." 실질적으로 이 인용구는 풋볼이나 더 넓은 데이터 과학에서, 기존의 데이터와 모델을 정확하게 사용하는 것보다 여러분의 필요에 부합하는 질문을 던지는 것이 더 중요하다는 것을 설명합니다.

통계적 접근 방식이 전통적인 방법보다 가지는 한 가지 장점은 확장이 가능하다는 것입니다. 분석가가 시도해 보고 입증된 선수 평가 방법을 개발하고 나면, 컴퓨팅의 힘을 사용하여 그 분석을 여러 선수에게 한 번에 실행할 수 있습니다. 이는 전통적인 스카우트 방법으로는 엄청나게 번거로운 작업입니다.

이 장에서는 통계를 사용하여 풋볼을 평가하는 요령을 개발하는 데 필요한 첫 번째 도구들 중 일부를 제공합니다. 이 주제에서 살펴볼 첫 번째 개념은 *안정성(stability)*입니다. 특히 스포츠에서 무엇인가를 평가할 때 안정성은 중요합니다. 안정성 척도는 특정 설정에서 일어난 일 중 얼마나 많은 부분이 선수에게 내재된 *근본적인(fundamental)* 기술(다른 설정으로 전이 가능한 부분)이고, 과거 성과 중 얼마나 많은 부분이 *분산(variance)*에 기인하는지 비교하여 결정할 수 있는 방법을 제공하기 때문입니다. FiveThirtyEight의 설립자인 네이트 실버(Nate Silver)의 말을 빌리자면, 안정성 분석은 무엇이 *신호(signal)*이고 무엇이 *소음(noise)*인지 구분하는 데 도움을 줍니다. 만약 어떤 선수가 풋볼의 안정적인 구성 요소에서는 매우 잘하지만 불안정한 요소에서는 부진하다면, 그 선수는 시장에서 저평가된 *저점 매수(buy-low)* 후보일 수 있습니다. 반대로 불안정한 지표에서는 성과가 좋지만 안정적인 지표에서는 부진한 선수는 *고점 매도(sell-high)* 선수일 수 있습니다.

안정성에 대한 정확한 정의는 특정 맥락에 따라 다르지만, 이 책에서는 평가 지표의 안정성을 미리 정해진 기간 동안 해당 지표의 일관성으로 언급합니다. 예를 들어 판타지 풋볼 분석가에게 그 기간은 매주가 될 수 있고, 팀을 위해 드래프트 모델을 구축하는 분석가에게는 선수의 대학 마지막 몇 시즌부터 프로 첫 몇 시즌까지가 될 수 있습니다. 풋볼 업계에서는 일반적으로 피어슨 상관계수(Pearson’s correlation coefficient)나 그것의 제곱인 결정계수(coefficient of determination)를 사용하여 안정성을 측정합니다. *피어슨 상관계수*의 범위는 -1에서 1 사이입니다. 0의 값은 상관관계가 없음을 나타내고, 1의 값은 완벽한 양의 상관관계(두 변수의 값이 함께 증가함)를 나타내며, -1의 값은 완벽한 음의 상관관계(한 변수가 증가함에 따라 다른 변수가 감소함)를 나타냅니다.

어떤 지표가 안정적인 것으로 간주되기 위한 정확한 숫자 기준은 일반적으로 상황이나 시대에 따라 다르지만, 계수가 높을수록 더 안정적인 통계임을 의미합니다. 예를 들어 패스 러싱 통계는 일반적으로 꽤 안정적이지만 패스 방어(coverage) 통계는 그렇지 않습니다. 에릭은 최근 [슬론 스포츠 분석 컨퍼런스(Sloan Sports Analytics Conference) 논문](https://oreil.ly/_GOBP)에서 이에 대해 더 자세히 다루었습니다. 만약 두 개의 패스 러싱 지표를 비교한다면, 덜 안정적인 지표가 더 안정적인 방어 지표보다 상관계수가 낮을 수 있습니다.

###### 참고(Note)

판타지 팬들은 *안정성 분석(stability analysis)*을 다른 용어인 *끈적한 스탯(sticky stats)*으로 알게 될 것입니다. 이 용어는 특정 통계적 추정치가 시간이 지나도 변하지 않고 "달라붙어(stick)" 일관성을 유지하기 때문에 생겨났습니다.

안정성 분석은 미국 통계학자인 존 튜키가 만든 *탐색적 데이터 분석(exploratory data analysis, EDA)*이라는 통계 분석의 하위 집합의 일부입니다. 형식적인 모델링 및 가설 검정과 대조적으로, EDA는 데이터 세트를 분석하여 흔히 통계 그래픽 및 기타 데이터 시각화 방법을 사용하여 주요 특성을 요약하는 접근 방식입니다. EDA는 이 분야의 초보자와 베테랑 모두가 풋볼 게임을 이해하기 위해 통계 분석을 사용하는 과정에서 종종 간과하는 단계이지만, 그 이유는 서로 다릅니다.

###### 참고(Note)

존 튜키는 이 책이 끝날 때쯤 여러분이 알게 되거나 알기를 바라는 다른 용어들도 만들었습니다. *상자 수염 그림(boxplot)* (그래프의 일종), *분산 분석(analysis of variance)* (줄여서 ANOVA; 통계 검정의 일종), *소프트웨어(software)* (컴퓨터 프로그램), *비트(bit)* (가장 작은 컴퓨터 데이터 단위로 대개 0/1로 표시됨; 여러분은 아마 8비트인 바이트나 기가바이트 같은 더 큰 단위에 더 익숙할 것입니다) 등이 그 예입니다. 튜키와 그의 제자들은 또한 20년 이상의 풋볼 데이터를 조사함으로써 프린스턴 대학교(Princeton University) 풋볼 팀이 기본 통계 방법을 사용한 데이터 분석을 구현하도록 도왔습니다. 하지만 현대 컴퓨터의 부재는 그의 작업에 한계를 가져왔고, 여러분이 이 책에서 배우는 도구의 대부분은 그가 접근할 수 있었던 방법보다 더 발전된 것입니다. 예를 들어, 그의 옛 제자 중 한 명인 [그레그 랑주(Gregg Lange)](https://oreil.ly/L3MCF)는 어떻게 단순한 실수로 인해 컴퓨터에 100파운드의 데이터 카드를 다시 집어넣어야 했는지 회상했습니다. 튜키의 삶과 공헌에 대해 더 자세히 읽어보고 싶다면 [*통계학 연보(Annals of Statistics)*](https://oreil.ly/7Wb0a)에 실린 데이비드 브릴린저(David Brillinger)의 "존 W. 튜키: 그의 삶과 직업적 공헌(John W. Tukey: His Life and Professional Contributions)"을 확인해 보세요.

# 질문 정의하기

올바른 질문을 던지는 것은 문제를 해결하는 것만큼이나 중요합니다. 사실, 튜키의 인용구가 강조하듯이 잘못된 질문에 대한 올바른 답은 그 자체로 쓸모가 없는 반면, 올바른 질문은 정답에 미치지 못하더라도 만족스러운 결과로 이끌 수 있습니다. 올바른 질문을 던지는 법을 배우는 것은 잘못된 질문을 던짐으로써 배우면서 다듬어지는 과정입니다. 긍정적인 결과는 수많은 부정적인 결과를 헤쳐 나가며 얻은 전리품입니다.

과학적이기 위해 질문은 검증 가능하고 반증 가능한 가설에 관한 것이어야 합니다. 예를 들어, "멀리 패스를 던지는 것이 짧은 패스보다 더 가치 있지만, 쿼터백이 깊은 패스를 잘 던지는지 여부를 말하기는 어렵다"는 타당한 가설입니다. 하지만 이를 과학적으로 만들려면, 여러분은 "가치 있다"는 것이 무엇을 의미하는지, 그리고 어떤 선수가 깊은 패스를 "잘한다"(또는 "못한다")고 말할 때 그 의미가 무엇인지 정의해야 합니다. 이를 위해 데이터가 필요합니다.

# 데이터 가져오기 및 필터링하기

패싱 데이터의 안정성을 연구하기 위해, R에서는 `nflfastR` 패키지를 사용하고 Python에서는 `nfl_data_py` 패키지를 사용합니다. <a href="ch01.html#sec-introduction" data-type="xref">1장</a>에서 배운 도구를 사용하여 2016년부터 2022년까지의 데이터를 플레이 단위(play-by-play), 즉 `pbp`로 로드하는 것부터 시작합니다.

2016년을 사용하는 것은 대체로 임의적인 선택입니다. 이 경우에는 경기에 영향을 미친 실질적인 규칙 변경(킥오프 터치백(kickoff touchback)을 25야드 라인으로 옮김)이 있었던 마지막 해입니다. 다른 시즌들, 예를 들어 2002년(리그가 마지막으로 확장된 해), 2011년(리그의 단체 협약에 대한 마지막으로 영향력 있었던 변경), 2015년(리그가 엑스트라 포인트(extra point)를 15야드 라인으로 후퇴시킨 해), 2020년(COVID-19 팬데믹, 그리고 리그가 플레이오프를 확대한 해), 2021년(리그가 정규 시즌 경기를 16경기에서 17경기로 늘린 해) 등은 자연스러운 분기점입니다.

Python에서는 `nfl_data_py` 패키지뿐만 아니라 `pandas`와 `numpy` 패키지를 로드합니다.

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl
```

###### 경고(Warning)

Python은 넘버링을 0부터 시작합니다. R은 1부터 시작합니다. 두 언어를 모두 사용하는 데이터 과학자 지망생이라면 이 부분에서 자주 걸려 넘어집니다. 이 때문에 이 예제에서는 `range()`의 마지막 값 입력에 **`+ 1`**을 더해야 합니다.

다음으로, Python에게 `range()`를 사용하여 로드할 연도를 알려줍니다. 그런 다음 해당 시즌의 NFL 데이터를 가져옵니다.

```
## Python
seasons = range(2016, 2022 + 1)
pbp_py = nfl.import_pbp_data(seasons)
```

R에서는 필요한 패키지를 먼저 로드합니다. 패키지 모음인 `tidyverse`는 데이터를 랭글링(조작)하고 그리는 데 도움을 줍니다. `nflfastR` 패키지는 데이터를 제공합니다. `ggthemes` 패키지는 플롯 서식 지정을 돕습니다.

```
## R
library("tidyverse")
library("nflfastR")
library("ggthemes")
```

R에서는 `2016:2022`라는 단축키를 사용하여 2016년부터 2022년까지의 범위를 지정할 수 있습니다.

```
## R
pbp_r <- load_pbp(2016:2022)
```

###### 경고(Warning)

어떤 데이터세트든 *메타데이터(metadata)*, 즉 데이터에 대한 데이터를 이해해야 합니다. 예를 들어, 0과 1은 무엇을 의미할까요? 어느 것이 '예'이고 어느 것이 '아니오'일까요? 아니면 작성자가 수준(levels)을 나타내기 위해 1과 2를 사용할까요? 데이터 분석가와 과학자들이 메타데이터를, 그리고 표준적인 0과 1 대신 1과 2의 사용을 오해하여 과학적 연구가 철회되었다는 이야기를 들은 적이 있습니다. 결과적으로 과학자들은 그들 자신의 데이터 구조를 이해하지 못했기 때문에 그들의 연구에 결함이 있다고 사람들에게 알려야 했습니다. 예를 들어, [*Significance*의 2021년 기사](https://oreil.ly/9kORC)에서 이러한 실수가 발생한 사례를 설명합니다.

이 분석에 필요한 데이터의 하위 집합(subset)을 얻으려면 다음 코드로 할 수 있는 패싱 플레이만으로 필터링합니다.

```
## Python
pbp_py_p = \
    pbp_py\
    .query("play_type == 'pass' & air_yards.notnull()")\
    .reset_index()
```

R에서는 동일한 기준을 사용하여 데이터를 `filter()`(필터링)합니다.

```
## R
pbp_r_p <-
    pbp_r |>
    filter(play_type == "pass" & !is.na(air_yards))
```

여기서 `play_type`이 `pass`와 같으면 러닝 플레이와 페널티로 인해 취소된 플레이가 모두 제거됩니다. 때로는 페널티가 있는 플레이를 포함하고 싶을 수도 있습니다 (예를 들어, PFF와 같은 등급 기반 시스템을 사용하는 경우). 등급 기반 시스템은 플레이의 최종 통계와 무관하게 플레이어가 플레이에서 얼마나 잘 수행했는지 측정하려고 시도하므로 `play_type == no_play`인 데이터를 유지하는 것이 가치가 있을 수 있습니다.

하지만 이 연습을 위해, 저희는 그러한 플레이를 생략하도록 할 것입니다. 또한 R에서는 `air_yards`가 `NA`이고 Python에서는 `NULL`인 플레이도 생략합니다. 이러한 플레이는 패스가 스크리미지(scrimmage) 라인에서 쳐내지거나(batted down), 버려지거나(thrown away), 스파이크(spiked)되어 의도한 리시버를 향하지 않을 때 발생합니다. 이러한 패스들이 패서의 최종 통계에 포함되고 선수로서 그가 누구인지 보여주는 근본적인 부분이긴 하지만, 여기서 던지는 질문과는 반드시 관련이 있는 것은 아닙니다.

다음으로 몇 가지 데이터 정리 및 랭글링 작업을 수행해야 합니다.

첫째, *긴(long)* 패스는 에어 야드가 20야드 이상인 패스로 정의하고, *짧은(short)* 패스는 에어 야드가 20야드 미만인 패스로 정의합니다. NFL 데이터에는 패스 길이(`pass_length`)에 대한 범주형 변수가 있지만 관찰자에게 분류 기준이 명확하지 않습니다 (이 장 끝에 있는 연습 문제를 참조하세요). 다행히도 여러분은 이 기준을 스스로 쉽게 계산할 수 있습니다 (원한다면 15야드나 25야드 등 다른 기준을 사용할 수도 있습니다).

둘째, 불완전(incomplete) 패스의 패싱 야드는 R에서는 `NA`, Python에서는 `NULL`로 기록되지만 이 분석을 위해서는 0으로 설정해야 합니다 (앞에서 제대로 필터링했다면 말이죠).

Python에서는 (별칭 `np`로 가져온) `numpy` 패키지의 `where()` 함수가 이 변경 작업을 돕습니다. 먼저 필터링 기준을 만듭니다.

```
## Python
pbp_py_p["pass_length_air_yards"] = np.where(
    pbp_py_p["air_yards"] >= 20, "long", "short"
)
```

그런 다음 필터링 기준을 사용하여 결측값을 대체합니다.

```
## Python
pbp_py_p["passing_yards"] = \
    np.where(
        pbp_py_p["passing_yards"].isnull(), 0, pbp_py_p["passing_yards"]
        )
```

R에서는 `mutate()` 안의 `ifelse()` 함수를 통해 동일한 변경이 가능합니다.

```
## R
pbp_r_p <-
    pbp_r_p |>
    mutate(
        pass_length_air_yards = ifelse(air_yards >= 20, "long", "short"),
        passing_yards = ifelse(is.na(passing_yards), 0, passing_yards)
    )
```

<a href="app02.html#sec-ssdw-pass" data-type="xref">부록 B</a>는 필터링과 같은 데이터 조작 주제를 매우 자세히 다룹니다. 데이터 랭글링을 더 잘 이해하는 데 도움이 필요하다면 이 자료를 참조하세요. 여기서는 흥미로운 질문과 함께 데이터를 바로 살펴볼 수 있도록 이러한 세부 사항을 간략하게 넘어갑니다.

###### 팁(Tip)

프로그래밍할 때 객체의 이름을 짓는 것은 놀라울 정도로 어려울 수 있습니다. 입력하기 쉬운 짧고 간단한 이름과 길고 더 많은 정보를 담고 있는 이름 사이에서 균형을 맞추도록 노력하세요. 이는 특히 이름이 긴 스크립트를 작성하기 시작할 때 중요할 수 있습니다. 이름 짓기에서 가장 중요한 부분은 다른 사람들과 미래의 자신 모두가 이해할 수 있는 이름을 만드는 것입니다.

# 데이터 요약하기

`passing_yards` 데이터를 설명하는 데 사용되는 몇 가지 기본 숫자를 간단히 살펴보겠습니다. Python에서 `passing_yards` 열을 선택한 다음 `describe()` 함수를 사용합니다.

```
## Python
pbp_py_p["passing_yards"]\
    .describe()
```

결과는 다음과 같습니다.

count 131606.000000 mean 7.192111 std 9.667021 min -20.000000 25% 0.000000 50% 5.000000 75% 11.000000 max 98.000000 Name: passing_yards, dtype: float64

R에서는 데이터프레임을 가져와서 `passing_yards` 열을 선택(또는 `pull()`)한 다음 `summary()` 통계를 계산합니다.

```
## R
pbp_r_p |>
    pull(passing_yards) |>
    summary()
```

그러면 다음과 같은 결과가 나타납니다.

Min. 1st Qu. Median Mean 3rd Qu. Max. -20.000 0.000 5.000 7.192 11.000 98.000

출력값에서 다음의 이름들이 설명하는 바는 이렇습니다 (<a href="app02.html#sec-ssdw-pass" data-type="xref">부록 B</a>에서 이 값들을 계산하는 방법을 보여줍니다).

- `count` (Python에서만)는 데이터의 레코드(행) 개수입니다.

- Python의 `mean` (R에서는 `Mean`)은 산술 평균입니다.

- `std` (Python에서만)는 표준편차입니다.

- Python의 `min` 또는 R의 `Min.`은 가장 낮거나 최솟값입니다.

- Python의 `25%` 또는 R의 `1st Qu.`는 제1사분위수로, 전체 값의 4분의 1이 이보다 작습니다.

- `Median` (R에서) 또는 `50%` (Python에서)는 중앙값으로, 값의 절반이 이보다 크고 절반이 이보다 작습니다.

- Python의 `75%` 또는 R의 `3rd Qu.`는 제3사분위수로, 전체 값의 4분의 3이 이보다 작습니다.

- Python의 `max` 또는 R의 `Max.`는 가장 크거나 최댓값입니다.

여러분이 진정으로 보고 싶은 것은 `pass_length_air_yards`의 다양한 값 아래에서 요약된 데이터입니다. 짧은 패스의 경우, Python에서 긴 패스를 걸러낸 다음 요약합니다.

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "short"')["passing_yards"]\
    .describe()
```

결과는 다음과 같습니다.

count 116087.000000 mean 6.526812 std 7.697057 min -20.000000 25% 0.000000 50% 5.000000 75% 10.000000 max 95.000000 Name: passing_yards, dtype: float64

그리고 R에서는 다음과 같습니다.

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "short") |>
    pull(passing_yards) |>
    summary()
```

그러면 다음과 같은 결과가 나타납니다.

Min. 1st Qu. Median Mean 3rd Qu. Max. -20.000 0.000 5.000 6.527 10.000 95.000

마찬가지로 Python에서 긴 패스만 선택하도록 필터링할 수 있습니다.

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "long"')["passing_yards"]\
    .describe()
```

결과는 다음과 같습니다.

count 15519.000000 mean 12.168761 std 17.923951 min 0.000000 25% 0.000000 50% 0.000000 75% 26.000000 max 98.000000 Name: passing_yards, dtype: float64

그리고 R에서는 다음과 같습니다.

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "long") |>
    pull(passing_yards) |>
    summary()
```

결과는 다음과 같습니다.

Min. 1st Qu. Median Mean 3rd Qu. Max. 0.00 0.00 0.00 12.17 26.00 98.00

여기서 주목해야 할 점은 최대 패싱 야드는 거의 같더라도 제1사분위수와 제3사분위수의 차이인 *사분위범위(interquartile range)*가 짧은 패스에 비해 긴 패스에서 훨씬 크다는 것입니다. 공중으로 20야드 이상 날아간 패스에서 음수의 야드를 얻는 것은 거의 불가능하기 때문에 긴 패스의 최솟값이 더 높을 수밖에 없습니다.

<a href="ch01.html#sec-introduction" data-type="xref">1장</a>에서 소개된 기대 점수 추가(EPA)에 대해서도 동일한 분석을 수행할 수 있습니다. EPA는 각 플레이에 점수 값을 할당하기 위해 상황 요인을 사용하는 플레이 성공의 더 연속적인 측정값임을 기억하십시오. Python에서는 다음과 같이 수행할 수 있습니다.

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "short"')["epa"]\
    .describe()
```

결과는 다음과 같습니다.

count 116086.000000 mean 0.119606 std 1.426238 min -13.031219 25% -0.606135 50% -0.002100 75% 0.959107 max 8.241420 Name: epa, dtype: float64

그리고 R에서는 다음과 같습니다.

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "short") |>
    pull(epa) |>
    summary()
```

그러면 다음과 같은 결과가 나타납니다.

Min. 1st Qu. Median Mean 3rd Qu. Max. NA's -13.0312 -0.6061 -0.0021 0.1196 0.9591 8.2414 1

마찬가지로 Python에서 긴 패스에 대해서도 이렇게 할 수 있습니다.

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "long"')["epa"]\
    .describe()
```

결과는 다음과 같습니다.

count 15519.000000 mean 0.382649 std 2.185551 min -10.477922 25% -0.827421 50% -0.465344 75% 2.136431 max 8.789743 Name: epa, dtype: float64

또는 R에서 다음과 같이 할 수 있습니다.

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "long") |>
    pull(epa) |>
    summary()
```

결과는 다음과 같습니다.

Min. 1st Qu. Median Mean 3rd Qu. Max. -10.4779 -0.8274 -0.4653 0.3826 2.1364 8.7897

여기서도 동일한 역학 관계를 볼 수 있습니다. 짧은 패스보다 긴 패스에서 더 폭넓은 결과가 나타납니다. 긴 패스는 짧은 패스보다 더 *변동성이 큽니다(variable)*.

더 나아가 긴 패스에 대한 시도당 평균 패싱 야드(YPA)와 시도당 EPA를 살펴보면, 두 지표 모두 짧은 패스보다 더 높습니다 (하지만 중앙값의 경우 관계가 역전됩니다. 그 이유는 무엇일까요?). 따라서 평균적으로 이 장의 가이드 가설의 첫 번째 부분인 "멀리 패스를 던지는 것이 짧은 패스보다 더 가치 있지만, 쿼터백이 깊은 패스를 잘 던지는지 여부를 말하기는 어렵다"를 비공식적으로 확인할 수 있습니다.

###### 팁(Tip)

코딩에 있어서 줄 바꿈과 공백(white space)은 중요합니다. 이러한 끊어주기는 여러분의 코드를 더 읽기 쉽게 만들어 줍니다. Python과 R은 줄 바꿈을 다루는 방식이 다르지만 때때로 두 언어 모두 줄 바꿈을 특별한 명령으로 취급합니다. 두 언어 모두에서 읽기 쉽도록 짧은 줄을 만들기 위해 함수 입력을 자주 분할합니다. 예를 들어 함수 이름을 띄우고 코드를 더 읽기 쉽게 만들기 위해 다음과 같이 함수의 간격을 둘 수 있습니다.

```
## Python or R
my_plot(data=big_name_data_frame,
        x="long_x_name",
        y="long_y_name")
```

R에서는 쉼표가 이전 줄에 유지되도록 해야 합니다. Python에서는 줄 바꿈을 위해 `\`를 사용해야 할 수도 있습니다.

```
## Python
x =\
    2 + 4
```

또는 괄호 안에 전체 명령을 넣을 수 있습니다.

```
## Python
x = (
    2 + 4
    )
```

다음과 같이 한 줄에 하나의 Python 함수를 작성할 수도 있습니다.

```
## Python
my_out = \
    my_long_long_long_data\
    .function_1()\
    .function_2()
```

# 데이터 시각화하기(Plotting Data)

데이터의 수치적 요약은 유용하고, 또 많은 사람들이 기하학적으로 사고하기보다는 대수학적으로 사고하지만(에릭이 이런 경우입니다), 많은 사람들이 숫자가 아닌 다른 것을 시각화할 필요를 느낍니다. 우리가 데이터를 그리는(plot) 것을 좋아하는 이유는 다음과 같습니다.

- 데이터가 괜찮아 보이는지 확인하기 위해. 예를 들어 너무 큰 값이 있습니까? 너무 작은 값이 있습니까? 다른 이상한 데이터 점들이 존재합니까?

- 데이터에 이상치(outliers)가 있습니까? 그것들이 자연스럽게 발생합니까(예: 거의 모든 패스 효율성 차트에 등장하는 패트릭 마홈스(Patrick Mahomes)), 아니면 부자연스럽게 발생합니까(예: 0 미만이거나 1을 초과하는 확률)?

- 첫눈에 나타나는 넓은 경향이 있습니까?

## 히스토그램(Histograms)

플롯의 한 종류인 *히스토그램*은 데이터 점들의 개수를 막대로 합산하여 데이터를 볼 수 있게 해 줍니다. 이 막대들을 *빈(bins)*이라고 부릅니다.

###### 경고(Warning)

이 책에서 사용된 패키지의 이전 버전을 설치한 경우, 저희의 코드 예제가 작동하지 않는다면 업그레이드해야 할 수 있습니다. 반대로 이 책에서 사용된 패키지의 향후 버전은 함수의 작동 방식을 업데이트할 수 있습니다. 이 책의 GitHub 페이지(github.com/raerickson/football_book_code)에 업데이트된 코드가 있을 수 있습니다.

Python에서는 이 책의 대부분의 시각화에 `seaborn` 패키지를 사용합니다. 먼저 `sns`라는 별칭을 사용하여 `seaborn`을 가져옵니다. 그런 다음 `displot()` 함수를 사용하여 <a href="#fig-sns_hist_1" data-type="xref">그림 2-1</a>에 표시된 그림을 만듭니다.

```
## Python
import seaborn as sns
import matplotlib.pyplot as plt

sns.displot(data=pbp_py, x="passing_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0201.png" />
<h6 id="figure-2-1.-a-histogram-in-python-using-seaborn-for-the-passing_yards-variable">그림 2-1. <code>passing_yards</code> 변수에 대해 <code>seaborn</code>을 사용한 Python 히스토그램</h6>
</figure>

###### 경고(Warning)

macOS에서는 다른 패키지를 로드할 때 `import matplotlib.pyplot as plt`도 포함해야 합니다. 마찬가지로 macOS 사용자들도 플롯 코드를 작성한 후 플롯이 표시되게 하려면 `plt.show()`를 포함해야 합니다. 저희는 또한 Linux의 경우 일부 편집기(예: Microsoft Visual Studio Code)에서는 `plt.show()`를 사용해야 하지만 다른 편집기(예: JupyterLab)에서는 그렇지 않다는 것을 알게 되었습니다. 의심스러운 경우 이 선택적 코드를 포함하세요. `plt.show()`를 실행한다고 해서 해가 되지는 않지만 그림을 표시하는 데 필요할 수 있습니다. Windows에서는 필요할 수도 있고 아닐 수도 있습니다.

마찬가지로 R에서도 히스토그램을 쉽게 만들 수 있습니다.

###### 참고(Note)

기본 R(base R)은 자체 시각화 도구를 함께 제공하지만 이 책에서는 `ggplot2`를 사용합니다. `ggplot2` 도구는 자체적인 언어를 가지고 있는데, 이는 릴런드 윌킨슨(Leland Wilkinson)의 *그래픽의 문법(The Grammar of Graphics)* (Springer, 2005)을 바탕으로 하며 해들리 위컴(Hadley Wickham)이 아이오와 주립대학교(Iowa State University) 박사 과정 중에 R로 구현한 것입니다. 교육적인 관점에서, 저희는 데이비드 로빈슨(David Robinson)이 "초보자들에게 내장된 시각화(기본 R)를 가르치지 마세요(ggplot2를 가르치세요)"([“Don’t Teach Built-in Plotting to Beginners (Teach ggplot2)”](https://oreil.ly/QDtpo))라는 제목의 블로그 게시물에서 기본 R 대신 `ggplot2`를 사용한 시각화를 가르치는 이유를 설명한 것에 동의합니다.

R에서는 `ggplot()` 함수와 함께 R의 `ggplot2`를 사용하여 <a href="#fig-ggplot2_hist_1" data-type="xref">그림 2-2</a>에 표시된 히스토그램을 만듭니다. 이 함수에서 `pbp_r_p` 데이터세트를 사용하고 `x`에 대한 미학(aesthetic)을 `passing_yards`가 되도록 설정합니다. 그런 다음 `geom_histogram()`이라는 기하학(geometry)을 추가합니다.

```
## R
ggplot(pbp_r, aes(x = passing_yards)) +
    geom_histogram()
```

결과는 다음과 같습니다.

\`stat_bin()\` using \`bins = 30\`. Pick better value with \`binwidth\`.

Warning: Removed 257229 rows containing non-finite values (\`stat_bin()\`).

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0202.png" />
<h6 id="figure-2-2.-a-histogram-in-r-using-ggplot2-for-the-passing_yards-variable">그림 2-2. <code>passing_yards</code> 변수에 대해 <code>ggplot2</code>를 사용한 R 히스토그램</h6>
</figure>

###### 경고(Warning)

데이터의 중요한 속성을 숨기기 위해 고의로 잘못된 빈 개수를 사용하는 것은 광범위한 통계 커뮤니티에서 사기로 간주됩니다. 히스토그램의 빈 개수를 선택할 때는 사려 깊고 신중하게 결정하십시오. 이 과정은 다양한 히스토그램 빈 개수를 탐색함에 따라 여러 번의 반복이 필요합니다.

그림 <a href="#fig-sns_hist_1" data-type="xref" data-xrefstyle="select:labelnumber">2-1</a> 및 <a href="#fig-ggplot2_hist_1" data-type="xref" data-xrefstyle="select:labelnumber">2-2</a>를 통해 저희 데이터의 기준을 이해할 수 있습니다. 획득한 패싱 야드는 약 -10야드에서 약 75야드 사이이며 대부분의 플레이에서 0(종종 패스 실패)에서 10야드 사이를 얻습니다. R은 `binwidth`와 빈 수에 주의하라는 경고를 표시하며 누락된 값(결측치) 제거에 대해서도 경고합니다. 기본값을 사용하는 대신 각 빈의 너비가 1야드가 되도록 설정하세요. 결측값에 대한 두 번째 경고를 무시하거나 시각화 전에 결측값을 필터링하여 경고를 피할 수 있습니다. 이런 빈 너비에서는 패스 실패가 너무 많기 때문에 데이터가 더 이상 정규 분포처럼 보이지 않습니다.

다음으로 각 `pass_depth_air_yards` 값에 대한 히스토그램을 만듭니다. 저희는 짧은 패스를 Python으로 만드는 방법(<a href="#fig-sns_hist_pass_short" data-type="xref">그림 2-3</a>)과 긴 패스를 R로 만드는 방법(<a href="#fig-ggplot2_hist_py_long" data-type="xref">그림 2-4</a>)을 보여드리겠습니다.

Python에서는 팔레트 옵션에 대한 테마를 `colorblind`로 변경하고 `whitegrid` 옵션을 사용하여 `ggplot2`의 흑백 테마와 유사한 플롯을 만듭니다.

```
## Python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="colorblind")
```

다음으로 짧은 패스만 선택하도록 필터링합니다.

```
## Python
pbp_py_p_short = \
    pbp_py_p\
    .query('pass_length_air_yards == "short"')
```

그런 다음 히스토그램을 만들고 `set_axis_labels`를 사용하여 플롯의 레이블을 변경하여 <a href="#fig-sns_hist_pass_short" data-type="xref">그림 2-3</a>과 같이 시각적으로 더 나은 형태로 만듭니다.

```
## Python
# Plot, change labels, and then show the output
pbp_py_hist_short = \
    sns.displot(data=pbp_py_p_short,
                binwidth=1,
                x="passing_yards");
pbp_py_hist_short\
    .set_axis_labels(
        "Yards gained (or lost) during a passing play", "Count"
        );
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0203.png" />
<h6 id="figure-2-3.-refined-histogram-in-python-using-seaborn-for-the-passing_yards-variable">그림 2-3. <code>passing_yards</code> 변수에 대해 <code>seaborn</code>을 사용한 개선된 Python 히스토그램</h6>
</figure>

R에서는 긴 패스만 선택하도록 필터링하고 x축 및 y축에 레이블을 추가하며 흑백 테마(`theme_bw()`)를 사용하여 시각적으로 더 나은 형태의 플롯을 만들어 <a href="#fig-ggplot2_hist_py_long" data-type="xref">그림 2-4</a>를 생성합니다.

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "long") |>
    ggplot(aes(passing_yards)) +
    geom_histogram(binwidth = 1) +
    ylab("Count") +
    xlab("Yards gained (or lost) during passing plays on long passes") +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0204.png" />
<h6 id="figure-2-4.-refined-histogram-in-r-using-ggplot2-for-the-passing_yards-variable">그림 2-4. <code>passing_yards</code> 변수에 대해 <code>ggplot2</code>를 사용한 개선된 R 히스토그램</h6>
</figure>

###### 참고(Note)

책의 나머지 부분에서는 R 플롯의 기본값으로 흑백 테마인 `theme_bw()`를 사용하고, Python 플롯의 기본값으로 `sns.set_theme(style="whitegrid", palette="colorblind")`를 사용할 것입니다. 종이에 인쇄할 때 더 멋져 보인다고 생각하기 때문에 이 테마들을 선호합니다.

이 히스토그램들은 <a href="#sec-chp2-sub-sum-data" data-type="xref">"데이터 요약하기"</a>에서 숫자상으로 본 내용을 그림으로 나타냅니다. 구체적으로 짧은 패스가 긴 패스보다 결과의 변동성이 작습니다. EPA에 대해서도 동일한 작업을 수행하여 유사한 결과를 찾을 수 있습니다. 이 장의 나머지 부분에서는 예제에 패싱 YPA를 계속 사용할 것이며, 스스로 해볼 수 있도록 패스 시도당 EPA를 연습 문제 안에 포함할 것입니다.

###### 참고(Note)

`ggplot2`, 더 넓게는 R이 객체를 파이핑(piping)하는 데 잘 작동하고 중간 객체를 피한다는 점을 알아두세요. 대조적으로 Python은 중간 객체를 저장하는 데 잘 작동합니다. 두 접근 방식 모두 장단점이 있습니다. 예를 들어 중간 객체를 저장하면 시각화의 중간 단계 출력을 확인할 수 있습니다. 대조적으로 동일한 객체 이름을 다시 쓰는 것은 지루할 수 있습니다. 이러한 대조적인 접근 방식은 두 언어 간의 철학적 차이를 나타냅니다. 어느 쪽이 본질적으로 옳거나 그르지 않으며 둘 다 장단점을 가지고 있습니다.

## 상자 수염 그림(Boxplots)

히스토그램을 사용하면 사람들이 데이터 점의 분포를 *볼(see)* 수 있습니다. 하지만 히스토그램은 특히 많은 변수를 탐색할 때 번거로울 수 있습니다. 상자 수염 그림은 히스토그램과 수치적 요약 사이의 절충안입니다(숫자 값은 <a href="#tbl-boxplot-parts" data-type="xref">표 2-1</a> 참조). *상자 수염 그림(Boxplots)*은 정렬된 데이터의 중간 50%를 포함하는 직사각형 *상자(box)*에서 그 이름을 얻었습니다. 상자 중간의 선은 중앙값이므로 정렬된 데이터의 절반은 선 위에 있고 데이터의 절반은 선 아래에 있습니다.

상자 위아래로 선이 연장되어 있기 때문에 어떤 사람들은 이를 *상자 수염(box-and-whisker) 그림*이라고 부릅니다. 이러한 수염(whiskers)에는 이상치를 제외한 나머지 데이터가 포함됩니다. `seaborn`과 `ggplot` 모두 상자 수염 그림에서 *이상치(outliers)*를 *사분위범위(interquartile range)* (25번째 백분위수와 75번째 백분위수 사이의 범위)의 1.5배 이상인 점(제3사분위수보다 크거나 제1사분위수보다 작음)으로 기본 설정합니다. 이러한 이상치는 점으로 그려집니다.

| 부분 이름                     | 데이터 범위                             |
|-------------------------------|-----------------------------------------|
| 위쪽 점들                     | 데이터 위의 이상치                      |
| 위쪽 수염                     | 이상치를 제외한 데이터의 100%에서 75%까지|
| 상자의 윗부분                 | 데이터의 75%에서 50%까지                |
| 상자 중간의 선                | 데이터의 50% (중앙값)                   |
| 상자의 아랫부분               | 데이터의 50%에서 25%까지                |
| 아래쪽 수염                   | 이상치를 제외한 데이터의 25%에서 0%까지 |
| 아래쪽 점들                   | 데이터 아래의 이상치                    |

표 2-1. 상자 수염 그림의 부분들 {#tbl-boxplot-parts}


다양한 유형의 이상치가 존재합니다. 이상치는 문제가 있는 데이터 점(예를 들어 누군가 10야드를 의도했는데 -10야드를 입력한 경우)일 수도 있지만, 종종 데이터의 일부로 존재하기도 합니다. 이러한 데이터 점 이면의 이유를 이해하면 최상의 결과와 최악의 결과를 반영하고 그 이면에 흥미로운 이야기가 있을 수 있기 때문에 데이터에 대한 예리한 통찰력을 얻는 경우가 많습니다. 오류(예: 잘못된 데이터 입력)로 인해 존재하는 것이 아니라면, 이상치는 일반적으로 모델 훈련에 사용되는 데이터에 포함되어야 합니다.

###### 참고(Note)

저희는 플롯의 텍스트 설명을 억제하기 위해 Python 플롯 명령 뒤에 세미콜론(`;`)을 배치합니다. 이러한 세미콜론은 선택 사항이며 단순히 저자의 선호입니다.

Python에서는 `seaborn`의 `boxplot()` 함수를 사용하고 축 레이블을 변경하여 <a href="#fig-sns-boxplot" data-type="xref">그림 2-5</a>를 만듭니다.

```
## Python
pass_boxplot = \
    sns.boxplot(data=pbp_py_p,
                x="pass_length_air_yards",
                y="passing_yards");
pass_boxplot.set(
    xlabel="Pass length (long >= 20 yards, short < 20 yards)",
    ylabel="Yards gained (or lost) during a passing play",
);
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0205.png" />
<h6 id="figure-2-5.-boxplot-of-yards-gained-from-long-and-short-air-yard-passes-seaborn">그림 2-5. 에어 야드가 길거나 짧은 패스에서 획득한 야드의 상자 수염 그림 (<code>seaborn</code>)</h6>
</figure>

R에서는 `ggplot2`와 함께 `geom_boxplot()`을 사용하여 <a href="#fig-ggplot2-boxplot" data-type="xref">그림 2-6</a>을 만듭니다.

```
## R
ggplot(pbp_r_p, aes(x = pass_length_air_yards, y = passing_yards)) +
    geom_boxplot() +
    theme_bw() +
    xlab("Pass length in yards (long >= 20 yards, short < 20 yards)") +
    ylab("Yards gained (or lost) during a passing play")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0206.png" />
<h6 id="figure-2-6.-boxplot-of-yards-gained-from-long-and-short-air-yard-passes-ggplot2">그림 2-6. 에어 야드가 길거나 짧은 패스에서 획득한 야드의 상자 수염 그림 (<code>ggplot2</code>)</h6>
</figure>

# 선수 수준에서의 시도당 패싱 야드의 안정성

이제 데이터에 익숙해졌으므로 이를 선수 평가에 사용할 차례입니다. 가장 먼저 해야 할 일은 미리 정해진 기간에 걸쳐 집계하여 각 선수에 대한 값을 얻는 것입니다. 주(week) 수준의 결과도 확실히 중요하지만(특히 판타지 풋볼과 베팅의 경우, <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a> 참조), 팀이 선수를 영입하려고 생각할 때 대부분의 경우 시즌 수준 데이터(때로는 여러 시즌에 걸친 데이터)를 사용합니다.

따라서 여기서는 Python의 `grouby()` 구문이나 R의 `group_by()` 구문을 사용하여 시즌 수준에서 집계합니다. *그룹화(group by)* 개념은 SQL 유형의 데이터베이스 언어에서 차용한 것입니다. 여기서의 과정을 생각할 때 *group by*는 동사로 간주될 수 있습니다. 예를 들어, 플레이 단위(play-by-play) 데이터를 사용한 다음 *시즌별로* *그룹화*하고 쿼터백의 평균 패싱 YPA를 계산하기 위해 *집계*(Python)하거나 *요약*(R)합니다.

이 문제를 위해 플레이 단위 데이터프레임(`pbp_py` 또는 `pbp_r`)을 가져온 다음 `passer_player_name`, `passer_player_id`, `season` 및 `pass_length`를 기준으로 *그룹화*합니다. 동명이인(또는 최소한 이름의 첫 글자와 성이 같은 선수)이 일부 존재하지만, 이름은 분석 결과를 연구하는 데 중요하므로 선수 ID와 선수 이름 열을 모두 기준으로 그룹화합니다. 하위 집합(subsets)으로 넘어가기 전에 먼저 전체 데이터세트로 시작하세요.

Python에서는 그룹화하려는 변수 목록(Python 구문에서는 `["item1", "item2"]`)과 함께 `groupby()`를 사용합니다. 그런 다음 `mean`을 위해 `passing_yards`에 대한 데이터를 집계합니다.

```
## Python
pbp_py_p_s = \
    pbp_py_p\
    .groupby(["passer_id", "passer", "season"])\
    .agg({"passing_yards": ["mean", "count"]})
```

Python을 사용할 때 열(columns)을 축소하여 데이터프레임을 다루기 더 쉽게 만듭니다 (`list()`는 목록을 만들고, `map()`은 반복문 구문이 없는 `for` 루프처럼 항목을 반복합니다 — `for` 루프에 대한 자세한 내용은 <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a> 참조).

```
## Python
pbp_py_p_s.columns = list(map("_".join, pbp_py_p_s.columns.values))
```

다음으로 다루기 쉽도록 더 짧고 직관적인 이름으로 열의 이름을 바꿉니다.

```
pbp_py_p_s \
    .rename(columns={'passing_yards_mean': 'ypa',
                     'passing_yards_count': 'n'},
            inplace=True)
```

R에서는 `pbp_p`를 `group_by()` 함수로 파이프(pipe)한 다음 `summarize()` 함수를 사용하여 `passing_yards`의 `mean()`을 계산하고 각 시즌의 각 선수에 대한 패스 시도 횟수 `n()`을 계산합니다. 결과 데이터프레임에서 그룹화를 삭제하도록 R에 지시하려면 `.groups = "drop"`을 포함시킵니다. 계산된 `passing_yards`의 평균은 쿼터백의 플레이당 평균 패스 거리인 YPA입니다. 계산 결과를 새로운 데이터프레임 `pbp_r_p_s`로 저장하기 위해 `<-` 함수를 사용합니다.

```
## R
pbp_r_p_s <-
    pbp_r_p |>
    group_by(passer_player_name, passer_player_id, season) |>
    summarize(
        ypa = mean(passing_yards, na.rm = TRUE),
        n = n(),
        .groups = "drop"
    )
```

이제 Python에서 `head()`를 사용하여 결과 데이터프레임의 상단을 살펴보고 `ypa`를 기준으로 `sort()`하여 결과를 더 잘 볼 수 있도록 합니다. `ascending=False` 옵션은 낮은 값에서 높은 값(예: 값을 7, 8, 9로 배열)이 아니라 높은 값에서 낮은 값(예: 값을 9, 8, 7로 배열)으로 정렬하도록 Python에 지시합니다.

```
## Python
pbp_py_p_s\
    .sort_values(by=["ypa"], ascending=False)\
    .head()
```

결과는 다음과 같습니다.

ypa n passer_id passer season 00-0035544 T.Kennedy 2021 75.0 1 00-0033132 K.Byard 2018 66.0 1 00-0031235 O.Beckham 2018 53.0 2 00-0030669 A.Wilson 2018 52.0 1 00-0029632 M.Sanu 2017 51.0 1

R에서는 `ypa`와 함께 `arrange()`를 사용하여 출력을 정렬합니다. 음수 기호(`-`)는 R에 순서를 반대로 하라고 지시합니다(예를 들어, 정렬할 때 7, 8, 9가 9, 8, 7이 됩니다).

```
## R
pbp_r_p_s |>
    arrange(-ypa) |>
    print()
```

결과는 다음과 같습니다.

\# A tibble: 746 × 5 passer_player_name passer_player_id season ypa n \<chr\> \<chr\> \<dbl\> \<dbl\> \<int\> 1 T.Kennedy 00-0035544 2021 75 1 2 K.Byard 00-0033132 2018 66 1 3 O.Beckham 00-0031235 2018 53 2 4 A.Wilson 00-0030669 2018 52 1 5 M.Sanu 00-0029632 2017 51 1 6 C.McCaffrey 00-0033280 2018 50 1 7 W.Snead 00-0030663 2016 50 1 8 T.Boyd 00-0033009 2021 46 1 9 R.Golden 00-0028954 2017 44 1 10 J.Crowder 00-0031941 2020 43 1 \# ℹ 736 more rows

YPA 값이 가장 높은 선수들은 주로 큰 야드(big yardage)를 획득하기 위해 한두 번의 패스(일반적으로 트릭 플레이)를 던져 성공시킨 선수들이기 때문에 이는 아직 큰 정보를 주지 못합니다. 이 문제를 해결하려면 한 시즌 내 일정 패스 시도 횟수(예: 100번)를 기준으로 필터링하고 결과를 확인해 보세요.

이 코드에서 어떤 일이 일어나고 있는지 더 자세히 이해하는 데 도움이 필요하다면 <a href="app03.html#sec-app-dw" data-type="xref">부록 C</a>에 데이터 랭글링에 대한 더 많은 팁과 요령이 포함되어 있습니다. Python에서는 이전 코드와 함께 `pbp_py_p_s`를 재사용하되 `'n >= 100'`을 사용하여 100번 이상의 패스 시도를 한 선수를 대상으로 `query()`를 포함하세요.

```
## Python
pbp_py_p_s_100 = \
    pbp_py_p_s\
    .query("n >= 100")\
    .sort_values(by=["ypa"], ascending=False)
```

이제 데이터의 상단(head)을 살펴봅니다.

```
## Python
pbp_py_p_s_100.head()
```

결과는 다음과 같습니다.

ypa n passer_id passer season 00-0023682 R.Fitzpatrick 2018 9.617886 246 00-0026143 M.Ryan 2016 9.442155 631 00-0029701 R.Tannehill 2019 9.069971 343 00-0033537 D.Watson 2020 8.898524 542 00-0036212 T.Tagovailoa 2022 8.892231 399

R에서는 동일한 변수들을 기준으로 *그룹화*한 다음 이번에는 `n()`을 사용하여 그룹당 관찰 횟수를 포함하여 *요약*합니다. 계속해서 결과를 파이프(pipe)하고 100번 이상의 패스(`n >= 100`)를 던진 패서들만 *필터링*한 다음 출력을 *정렬(arrange)*합니다.

```
## R
pbp_r_p_100 <-
    pbp_r_p |>
    group_by(passer_id, passer, season) |>
    summarize(
        n = n(), ypa = mean(passing_yards),
        .groups = "drop"
    ) |>
    filter(n >= 100) |>
    arrange(-ypa)
```

그런 다음 상위 20개의 결과를 인쇄(print)합니다.

```
## R
pbp_r_p_100 |>
    print(n = 20)
```

그러면 다음과 같은 결과가 나타납니다.

\# A tibble: 300 × 5 passer_id passer season n ypa \<chr\> \<chr\> \<dbl\> \<int\> \<dbl\> 1 00-0023682 R.Fitzpatrick 2018 246 9.62 2 00-0026143 M.Ryan 2016 631 9.44 3 00-0029701 R.Tannehill 2019 343 9.07 4 00-0033537 D.Watson 2020 542 8.90 5 00-0036212 T.Tagovailoa 2022 399 8.89 6 00-0031345 J.Garoppolo 2017 176 8.86 7 00-0033873 P.Mahomes 2018 651 8.71 8 00-0036442 J.Burrow 2021 659 8.67 9 00-0026498 M.Stafford 2019 289 8.65 10 00-0031345 J.Garoppolo 2021 511 8.50 11 00-0033319 N.Mullens 2018 270 8.43 12 00-0033537 D.Watson 2017 202 8.41 13 00-0033077 D.Prescott 2020 221 8.40 14 00-0034869 S.Darnold 2022 137 8.34 15 00-0037834 B.Purdy 2022 233 8.34 16 00-0029604 K.Cousins 2020 513 8.31 17 00-0031345 J.Garoppolo 2019 532 8.28 18 00-0025708 M.Moore 2016 122 8.28 19 00-0033873 P.Mahomes 2019 596 8.28 20 00-0020531 D.Brees 2017 606 8.26 \# i 280 more rows

가장 눈치 빠른 독자라 할지라도 제이미스 윈스턴(Jameis Winston)의 백업으로서 하버드대 출신 라이언 피츠패트릭(Ryan Fitzpatrick)이 보낸 시즌이 이 목록의 맨 위에 등장할 것이라고는 예상하지 못했을 것입니다. 여러분은 맷 라이언(Matt Ryan)(2016년)과 패트릭 마홈스(Patrick Mahomes)(2018년)의 MVP 시즌과 위대한 카일 섀너핸(Kyle Shanahan)이 지도한 (맷 라이언을 포함한) 여러 쿼터백들을 볼 수 있습니다.

## 긴 패스 대 짧은 패스

이제 가설의 두 번째 부분인 "멀리 패스를 던지는 것이 짧은 패스보다 더 가치 있지만, 쿼터백이 깊은 패스를 잘 던지는지 여부를 말하기는 어렵다"를 테스트하는 이 장의 본론으로 들어갑니다. 이 안정성 분석을 위해 다음 단계를 수행하세요.

1.  각 시즌의 각 패서에 대한 YPA를 계산합니다.

2.  각 패서의 이전 시즌에 대한 YPA를 계산합니다.

3.  1단계와 2단계에서 계산된 값의 상관관계를 살펴보고 안정성을 확인합니다.

이전과 비슷한 코드를 사용하되 패스 야드를 포함하려면 *group by* 명령과 함께 `pass_length_air_yards`를 포함하세요. 이 작업에서는 이름 지정이 어려워집니다.

저희는 데이터세트(*플레이 단위(play-by-play)*, `pbp`), 언어(Python일 경우 `_py`, R일 경우 `_r`), 패싱 플레이(`_p`), 시즌 데이터(`_s`), 그리고 마지막으로 패스 길이(`_pl`)를 사용하도록 합니다.

두 언어 모두에서 데이터프레임의 복사본을 만든 다음 1을 더하여 연도를 이동시킵니다. 그런 다음 새 데이터프레임을 원래 데이터프레임과 병합(merge)합니다. 이렇게 하면 현재 연도와 이전 연도의 값을 가질 수 있습니다.

###### 팁(Tip)

긴 이름은 지루하지만 고유한 이름이 중요합니다. 그래야(대부분의 코드 편집기에서 볼 수 있는) '찾기 및 바꾸기'와 같은 도구를 사용하여 코드를 빠르게 검색하여(찾기로) 코드에서 어떤 일이 일어나고 있는지 확인하거나(바꾸기로) 이름을 변경할 수 있습니다.

Python에서는 여러 단계를 거쳐 `pbp_r_p_s_pl`을 만듭니다. 먼저 *그룹화*하고 *집계*하여 평균과 개수를 구합니다.

```
## Python
pbp_py_p_s_pl = \
    pbp_py_p\
    .groupby(["passer_id", "passer", "season", "pass_length_air_yards"])\
    .agg({"passing_yards": ["mean", "count"]})
```

다음으로 작업하기 더 쉬운 짧은 이름을 갖기 위해 열 이름을 평탄화(flatten)하고 `passing_yards_mean`을 **`ypa`**로, `passing_yards_count`를 **`n`**으로 바꿉니다.

```
## Python
pbp_py_p_s_pl.columns =\
    list(map("_".join, pbp_py_p_s_pl.columns.values))
pbp_py_p_s_pl\
    .rename(columns={'passing_yards_mean': 'ypa',
                     'passing_yards_count': 'n'},
            inplace=True)
```

다음으로 인덱스를 재설정(reset)합니다.

```
## Python
pbp_py_p_s_pl.reset_index(inplace=True)
```

짧은 패스 플레이가 100번 넘는 패서들의 데이터와 긴 패스 플레이가 30번 넘는 패서들의 데이터만 선택합니다.

```
## Python
q_value = (
    '(n >= 100 & ' +
     'pass_length_air_yards == "short") | ' +
     '(n >= 30 & ' +
     'pass_length_air_yards == "long")'
)
pbp_py_p_s_pl = pbp_py_p_s_pl.query(q_value).reset_index()
```

그런 다음 저장할 열의 목록(`cols_save`)과 이 열들만 있는 새 데이터프레임(`air_yards_py`)을 만듭니다. 편집한 내용이 원래 데이터프레임으로 다시 전달되지 않도록 `.copy()`를 포함하세요.

```
## Python
cols_save =\
    ["passer_id", "passer", "season",
     "pass_length_air_yards", "ypa"]
air_yards_py =\
    pbp_py_p_s_pl[cols_save].copy()
```

다음으로 `air_yards_py`를 복사하여 `air_yards_lag_py`를 만듭니다. 현재 시즌 값을 가져와 단축 명령 `+=`를 사용하여 1을 더하고, `passing_yards_mean`의 이름에 `lag`(두 연도 사이의 1년 오프셋 또는 지연을 의미함)를 포함하도록 변경합니다.

```
## Python
air_yards_lag_py =\
    air_yards_py\
    .copy()
air_yards_lag_py["season"] += 1
```


```
## Python
air_yards_lag_py\
    .rename(columns={'ypa': 'ypa_last'},
    inplace=True)
```

마지막으로 `merge()`를 사용하여 두 데이터프레임을 하나로 합쳐 `air_yards_both_py`를 만듭니다. 공유되는 연도만 저장되도록 *내부 조인(inner join)*을 사용하고 `passer_id`, `passer`, `season`, `pass_length_air_yards`를 기준으로 조인합니다.

```
## Python
pbp_py_p_s_pl =\
    air_yards_py\
    .merge(air_yards_lag_py,
           how='inner',
           on=['passer_id', 'passer',
               'season', 'pass_length_air_yards'])
```

톰 브래디(`T.Brady`)와 애런 로저스(`A.Rodgers`) 등 여러분이 선택한 몇몇 쿼터백을 살펴보고, 보기 쉬운 데이터프레임을 만들기 위해 필요한 열만 포함하여 Python에서의 선택 결과를 확인하세요.

```
## Python
print(
    pbp_py_p_s_pl[["pass_length_air_yards", "passer",
                    "season", "ypa", "ypa_last"]]\
    .query('passer == "T.Brady" | passer == "A.Rodgers"')\
    .sort_values(["passer", "pass_length_air_yards", "season"])\
    .to_string()
)
```

결과는 다음과 같습니다.

pass_length_air_yards passer season ypa ypa_last 47 long A.Rodgers 2019 12.092593 12.011628 49 long A.Rodgers 2020 16.097826 12.092593 51 long A.Rodgers 2021 14.302632 16.097826 53 long A.Rodgers 2022 10.312500 14.302632 45 short A.Rodgers 2017 6.041475 6.693523 46 short A.Rodgers 2018 6.697446 6.041475 48 short A.Rodgers 2019 6.207224 6.697446 50 short A.Rodgers 2020 6.718447 6.207224 52 short A.Rodgers 2021 6.777083 6.718447 54 short A.Rodgers 2022 6.239130 6.777083 0 long T.Brady 2017 13.264706 15.768116 2 long T.Brady 2018 10.232877 13.264706 4 long T.Brady 2019 10.828571 10.232877 6 long T.Brady 2020 12.252101 10.828571 8 long T.Brady 2021 12.242424 12.252101 10 long T.Brady 2022 10.802469 12.242424 1 short T.Brady 2017 7.071429 7.163022 3 short T.Brady 2018 7.356452 7.071429 5 short T.Brady 2019 6.048276 7.356452 7 short T.Brady 2020 6.777600 6.048276 9 short T.Brady 2021 6.634697 6.777600 11 short T.Brady 2022 5.832168 6.634697

###### 팁(Tip)

코드를 확인할 때는 최소 두 명의 선수를 사용할 것을 권장합니다. 예를 들어, 톰 브래디는 `passer_id` 순으로 첫 번째 선수이며, 그의 값만 살펴본다면 데이터프레임의 첫 번째 선수에게 영향을 미치지 않는 실수를 발견하지 못할 수 있습니다.

R에서도 유사한 단계를 거쳐 `pbp_r_p_s_pl`을 만듭니다. 먼저 필요한 열을 선택하고 데이터프레임을 정렬하여 `air_yards_r`을 만듭니다.

```
## R
air_yards_r <-
    pbp_r_p |>
    select(passer_id, passer, season,
           pass_length_air_yards, passing_yards) |>
    arrange(passer_id, season,
            pass_length_air_yards) |>
    group_by(passer_id, passer,
             pass_length_air_yards, season) |>
    summarize(n = n(),
              ypa = mean(passing_yards),
              .groups = "drop") |>
    filter((n >= 100 & pass_length_air_yards == "short") |
           (n >= 30 & pass_length_air_yards == "long")) |>
    select(-n)
```

다음으로 시즌에 1을 더하는 `mutate`를 포함하여 지연(lag) 데이터프레임을 만듭니다.

```
## R
air_yards_lag_r <-
    air_yards_r |>
    mutate(season = season + 1) |>
    rename(ypa_last = ypa)
```

마지막으로 데이터프레임들을 조인하여 `pbp_r_p_s_pl`을 만듭니다.

```
## R
pbp_r_p_s_pl <-
    air_yards_r |>
    inner_join(air_yards_lag_r,
              by = c("passer_id", "pass_length_air_yards",
                     "season", "passer"))
```

톰 브래디(`T.Brady`)와 애런 로저스(`A.Rodgers`) 등 여러분이 선택한 패서들을 조사하여 R에서의 결과를 확인하세요.

```
## R
pbp_r_p_s_pl |>
    filter(passer %in% c("T.Brady", "A.Rodgers")) |>
    print(n = Inf)
```

그러면 다음과 같은 결과가 나타납니다.

\# A tibble: 22 × 6 passer_id passer pass_length_air_yards season ypa ypa_last \<chr\> \<chr\> \<chr\> \<dbl\> \<dbl\> \<dbl\> 1 00-0019596 T.Brady long 2017 13.3 15.8 2 00-0019596 T.Brady long 2018 10.2 13.3 3 00-0019596 T.Brady long 2019 10.8 10.2 4 00-0019596 T.Brady long 2020 12.3 10.8 5 00-0019596 T.Brady long 2021 12.2 12.3 6 00-0019596 T.Brady long 2022 10.8 12.2 7 00-0019596 T.Brady short 2017 7.07 7.16 8 00-0019596 T.Brady short 2018 7.36 7.07 9 00-0019596 T.Brady short 2019 6.05 7.36 10 00-0019596 T.Brady short 2020 6.78 6.05 11 00-0019596 T.Brady short 2021 6.63 6.78 12 00-0019596 T.Brady short 2022 5.83 6.63 13 00-0023459 A.Rodgers long 2019 12.1 12.0 14 00-0023459 A.Rodgers long 2020 16.1 12.1 15 00-0023459 A.Rodgers long 2021 14.3 16.1 16 00-0023459 A.Rodgers long 2022 10.3 14.3 17 00-0023459 A.Rodgers short 2017 6.04 6.69 18 00-0023459 A.Rodgers short 2018 6.70 6.04 19 00-0023459 A.Rodgers short 2019 6.21 6.70 20 00-0023459 A.Rodgers short 2020 6.72 6.21 21 00-0023459 A.Rodgers short 2021 6.78 6.72 22 00-0023459 A.Rodgers short 2022 6.24 6.78

###### 팁(Tip)

저희는 "자신의 코드가 맞다고 확신하기 전까지는 코드가 틀렸다고 가정하라"는 철학을 따릅니다. 따라서 저희는 우리가 코드가 수행할 것이라고 생각하는 것과 코드가 실제로 수행하는 것을 확실히 이해하기 위해 종종 코드를 엿봅니다. 실질적으로 이는 전 미국 대통령 로널드 레이건(Ronald Reagan)의 "신뢰하되 검증하라(Trust but verify)"는 조언을 당신의 코드에 따르는 것을 의미합니다.

이제 생성하신 데이터프레임(Python의 경우 `pbp_py_p_s_pl`, R의 경우 `pbp_r_p_s_pl`)에는 여섯 개의 열이 포함되어 있습니다. Python에서 데이터프레임의 `info()`를 살펴보세요.

```
## Python
pbp_py_p_s_pl\
    .info()
```

결과는 다음과 같습니다.

\<class 'pandas.core.frame.DataFrame'\> RangeIndex: 317 entries, 0 to 316 Data columns (total 6 columns): \# Column Non-Null Count Dtype --- ------ -------------- ----- 0 passer_id 317 non-null object 1 passer 317 non-null object 2 season 317 non-null int64 3 pass_length_air_yards 317 non-null object 4 ypa 317 non-null float64 5 ypa_last 317 non-null float64 dtypes: float64(2), int64(1), object(3) memory usage: 15.0+ KB

또는 R에서 데이터프레임을 `glimpse()`(훑어보기) 해보세요.

```
## R
pbp_r_p_s_pl |>
    glimpse()
```

결과는 다음과 같습니다.

Rows: 317 Columns: 6 \$ passer_id \<chr\> "00-0019596", "00-0019596", "00-0019596", "00-00… \$ passer \<chr\> "T.Brady", "T.Brady", "T.Brady", "T.Brady", "T.B… \$ pass_length_air_yards \<chr\> "long", "long", "long", "long", "long", "long", … \$ season \<dbl\> 2017, 2018, 2019, 2020, 2021, 2022, 2017, 2018, … \$ ypa \<dbl\> 13.264706, 10.232877, 10.828571, 12.252101, 12.2… \$ ypa_last \<dbl\> 15.768116, 13.264706, 10.232877, 10.828571, 12.2…

여섯 개의 열에는 다음과 같은 데이터가 포함되어 있습니다.

- `passer_id`는 해당 선수의 고유한 패서 식별 번호입니다.

- `passer`는 패서의 이름 첫 글자와 성(고유하지 않을 수도 있음)입니다.

- `pass_length_air_yards`는 여러분이 이전에 정의한 패스 유형(길거나 짧음)입니다.

- `season`은 두 시즌 쌍 중 마지막 시즌입니다 (예를 들어 `season`이 2017이라는 것은 2016년과 2017년을 비교하고 있음을 의미합니다).

- `ypa`는 명시된 시즌(예: 이전 예제의 2017년) 동안의 시도당 야드입니다.

- `ypa_last`는 명시된 시즌의 이전 시즌(예: 이전 예제의 2016년) 동안의 시도당 야드입니다.

이제 데이터에 무엇이 포함되어 있는지 상기했으니, 데이터를 파헤쳐보고 몇 명의 쿼터백이 있는지 확인해 봅시다. Python에서는 `passer_id` 열을 사용하여 고유한(`unique()`) 값들을 찾은 다음 이 객체의 길이(length)를 구합니다.

```
## Python
len(pbp_py_p_s_pl.passer_id.unique())
```

결과는 다음과 같습니다.

65

R에서는 `passer_id`와 함께 `distinct`(고유한) 함수를 사용한 다음 얼마나 많은 행이 존재하는지 확인합니다.

```
## R
pbp_r_p_s_pl |>
    distinct(passer_id) |>
    nrow()
```

결과는 다음과 같습니다.

\[1\] 65

이제 괜찮은 표본 크기의 쿼터백 데이터를 얻었습니다. 산점도(scatterplot)를 사용하여 이 데이터를 시각화할 수 있습니다. *산점도*는 그림에 점을 표시하는 것으로, 데이터 빈을 막대로 그리는 히스토그램이나 중앙값과 같은 데이터 요약을 표시하는 상자 수염 그림과 대조됩니다. 산점도를 사용하면 데이터를 직접 "볼(see)" 수 있습니다. 가로축은 *x축(x-axis)*이라고 불리며 일반적으로 예측 변수(또는 원인 변수)가 있을 경우 이를 포함합니다. 세로축은 *y축(y-axis)*이라고 불리며 일반적으로 반응 변수(또는 결과 변수)가 있을 경우 이를 포함합니다. 이번 예제에서는 전년도 YPA를 올해 YPA의 예측 변수로 사용할 것입니다. R에서는 `geom_point()`를 사용하여 이를 그리고 이 플롯을 `scatter_ypa_r`이라고 부른 다음 `scatter_ypa_r`을 인쇄하여 <a href="#fig-ypa-ggplot2" data-type="xref">그림 2-7</a>을 만듭니다.

```
## R
scatter_ypa_r <-
    ggplot(pbp_r_p_s_pl, aes(x = ypa_last, y = ypa)) +
    geom_point() +
    facet_grid(cols = vars(pass_length_air_yards)) +
    labs(
        x = "Yards per Attempt, Year n",
        y = "Yards per Attempt, Year n + 1"
    ) +
    theme_bw() +
    theme(strip.background = element_blank())

print(scatter_ypa_r)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0207.png" />
<h6 id="figure-2-7.-stability-of-ypa-plotted-with-ggplot2.-notice-that-both-sub-plots-have-the-same-x-and-y-scales">그림 2-7. <code>ggplot2</code>로 그린 YPA의 안정성. 두 하위 플롯(sub-plots)이 동일한 x 및 y 축 척도를 가지고 있음을 주목하세요</h6>
</figure>

<a href="#fig-ypa-ggplot2" data-type="xref">그림 2-7</a>은 짧은 패스에 대해 고무적입니다. 한 해에 짧은 패스를 잘 던지는 쿼터백은 다음 해에도 짧은 패스를 잘 던지며 그 반대도 마찬가지인 것으로 보입니다. 반면에 긴 패스는 훨씬 다루기 힘든(무작위적인) 형태임을 눈여겨보세요. 이러한 추세를 더 잘 조사할 수 있도록 데이터에 최적 적합선(line of best fit)을 포함시켜 <a href="#fig-ypa-ggplot2-trend" data-type="xref">그림 2-8</a>을 만드세요 (여기서 플롯을 재사용할 수 있도록 앞서 `scatter_ypa_r`을 저장하게 했던 것입니다).

```
## R
# add geom_smooth() to the previously saved plot
scatter_ypa_r +
    geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0208.png" />
<h6 id="figure-2-8.-stability-of-ypa-plotted-with-ggplot2-and-including-a-trend-line">그림 2-8. <code>ggplot2</code>로 그리고 추세선을 포함한 YPA의 안정성</h6>
</figure>

두 패스 유형 모두에 대해 <a href="#fig-ypa-ggplot2-trend" data-type="xref">그림 2-8</a>의 선들은 약간 양의 기울기를 갖지만(플롯 전체에서 선이 증가하고 있음), 이를 시각적으로 확인하기는 어렵습니다. 상관관계를 사용하여 이 추정치를 얻으려면 숫자 값들을 살펴보세요.

```
## R
pbp_r_p_s_pl |>
    filter(!is.na(ypa) & !is.na(ypa_last)) |>
    group_by(pass_length_air_yards) |>
    summarize(correlation = cor(ypa, ypa_last))
```

결과는 다음과 같습니다.

\# A tibble: 2 × 2 pass_length_air_yards correlation \<chr\> \<dbl\> 1 long 0.234 2 short 0.438

Python에서도 이러한 그림과 분석을 반복하여 <a href="#fig-ypa-sns-trend" data-type="xref">그림 2-9</a>를 만들 수 있습니다.

```
## Python
sns.lmplot(data=pbp_py_p_s_pl,
           x="ypa",
           y="ypa_last",
           col="pass_length_air_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0209.png" />
<h6 id="figure-2-9.-stability-of-ypa-plotted-with-seaborn-and-including-a-trend-line">그림 2-9. <code>seaborn</code>으로 그리고 추세선을 포함한 YPA의 안정성</h6>
</figure>

마찬가지로 `pandas`를 사용하여 상관관계를 얻을 수도 있습니다.

```
## Python
pbp_py_p_s_pl\
    .query("ypa.notnull() & ypa_last.notnull()")\
    .groupby("pass_length_air_yards")[["ypa", "ypa_last"]]\
    .corr()
```

결과는 다음과 같습니다.

ypa ypa_last pass_length_air_yards long ypa 1.000000 0.233890 ypa_last 0.233890 1.000000 short ypa 1.000000 0.438479 ypa_last 0.438479 1.000000

피어슨 상관계수는 그림 <a href="#fig-ypa-ggplot2-trend" data-type="xref" data-xrefstyle="select:labelnumber">2-8</a>과 <a href="#fig-ypa-sns-trend" data-type="xref" data-xrefstyle="select:labelnumber">2-9</a>에서 보여주는 것을 숫자로 잡아냅니다.

두 데이터 세트 모두 꽤 많은 노이즈를 포함하고 있지만, 피어슨 상관계수와 비교해 볼 때 쿼터백의 짧은 패스 성과가 긴 패스 성과보다 두 배나 더 안정적입니다. 따라서 이 장의 가이드 가설의 두 번째 부분인 "멀리 패스를 던지는 것이 짧은 패스보다 더 가치 있지만, 쿼터백이 깊은 패스를 잘 던지는지 여부를 말하기는 어렵다"를 확인할 수 있습니다.

###### 참고(Note)

피어슨 상관계수는 -1에서 1까지 다양할 수 있습니다. 안정성의 경우 숫자가 +1에 가까울수록 강하고 긍정적인 상관관계와 높은 안정성을 의미하며, 0에 가까울수록 기껏해야 약한 상관관계(그리고 불안정한 척도)를 의미합니다. -1의 피어슨 상관계수는 감소하는 상관관계를 의미하며 안정성의 경우에는 존재하지 않지만, 만약 존재한다면 올해의 높은 값이 내년의 낮은 값과 상관관계가 있을 것임을 의미합니다.

## 그렇다면, 우리는 이 통찰력으로 무엇을 해야 할까요?

일반적으로 노이즈가 많은 데이터는 다시 반복될 가능성이 없는 깜짝 시즌(pop-up seasons)을 보내는 선수(또는 팀이나 팀 내 특정 유닛)를 찾을 수 있는 곳입니다. 어느 해 인플레이 타구 타율(babip)이 높아져 타율이 2할(20-point) 껑충 뛴 야구 선수는 판타지나 실제 야구에서 여러분이 로스터에 포함시키는 것을 피하고 싶어 하는 사람일 수 있습니다. 마찬가지로, 상대적으로 안정적인 짧은 패스에서 지표의 증가 없이 어느 해에 깊은 패스에서 높은 YPA(또는 패스 시도당 EPA)를 생성한 기량이 부족한 쿼터백은 분석가들이 *회귀 후보(regression candidate)*라고 부르는 선수일 수 있습니다.


예를 들어, Python에서 2017년 깊은 패스 YPA의 리더보드를 살펴보겠습니다.

```
## Python
pbp_py_p_s_pl\
    .query(
        'pass_length_air_yards == "long" & season == 2017'
        )[["passer_id", "passer", "ypa"]]\
    .sort_values(["ypa"], ascending=False)\
    .head(10)
```

결과는 다음과 같습니다.

passer_id passer ypa 41 00-0023436 A.Smith 19.338235 79 00-0026498 M.Stafford 17.830769 12 00-0020531 D.Brees 16.632353 191 00-0032950 C.Wentz 13.555556 33 00-0022942 P.Rivers 13.347826 0 00-0019596 T.Brady 13.264706 129 00-0029604 K.Cousins 12.847458 114 00-0029263 R.Wilson 12.738636 203 00-0033077 D.Prescott 12.585366 109 00-0028986 C.Keenum 11.904762

이 목록에는 훌륭한 이름들(드류 브리스(Drew Brees), 톰 브래디(Tom Brady), 러셀 윌슨(Russell Wilson))도 있지만 그저 그런 이름들도 있습니다. 2018년의 동일한 목록을 살펴보겠습니다.

```
## Python
pbp_py_p_s_pl\
    .query(
        'pass_length_air_yards == "long" & season == 2018'
        )[["passer_id", "passer", "ypa"]]\
    .sort_values(["ypa"], ascending=False)\
    .head(10)
```

결과는 다음과 같습니다.

passer_id passer ypa 116 00-0029263 R.Wilson 15.597403 14 00-0020531 D.Brees 14.903226 205 00-0033077 D.Prescott 14.771930 214 00-0033106 J.Goff 14.445946 35 00-0022942 P.Rivers 14.357143 157 00-0031280 D.Carr 14.339286 188 00-0032268 M.Mariota 13.941176 64 00-0026143 M.Ryan 13.465753 193 00-0032950 C.Wentz 13.222222 24 00-0022803 E.Manning 12.941176

오랫동안 짧은 패스(dink-and-dunk) 전문가로 여겨졌던 알렉스 스미스(Alex Smith)가 이 목록에서 완전히 사라졌습니다. 그는 2017년에 패서 레이팅(passer rating) 부문 리그 1위를 차지한 후 캔자스시티에서 워싱턴으로 트레이드되어 3라운드 픽과 스타 코너백 켄달 풀러(Kendall Fuller)를 받아왔습니다(고점에 팔 줄 아는 팀이 여기 있네요!).

이 목록에는 깊은 패스에서의 YPA가 반복되는 선수도 일부 포함되어 있지만, 많은 새로운 이름이 등장합니다. 특히 데이터 세트에서 맷 라이언(Matt Ryan)의 이름을 필터링해 보면, 그가 2016년(NFL MVP 수상 연도) 깊은 패스에서 평균 17.7 YPA를 기록했음을 알 수 있습니다. 2017년에는 그 수치가 8.5로 떨어졌다가 2018년에 13.5로 다시 올랐습니다. 라이언의 능력이 이 3년 동안 급격하게 변한 것일까요, 아니면 그가 상당한 통계적 변동성에 영향을 받은 것일까요? 수학은 후자를 시사합니다. 따라서 판타지 풋볼이나 베팅에서 그는 2017년에는 *고점 매도(sell-high)* 후보, 2018년에는 *저점 매수(buy-low)* 후보였을 것입니다.

# 이 장에서 사용된 데이터 과학 도구

이 장에서는 다음 주제를 다루었습니다.

- R에서 직접 `nflfastR` 패키지를 사용하거나 Python에서 `nfl_data_py` 패키지를 사용하여 여러 시즌의 데이터 얻기

- Python에서 `where`나 R에서 `ifelse()` 문을 사용하여 조건에 따라 열 변경하기

- 데이터에 대해 Python에서는 `pandas`와 함께 `describe()`를 사용하거나 R에서는 `summarize()`를 사용하기

- Python에서는 `sort_by()`를 사용하거나 R에서는 `arrange()`를 사용하여 값 재정렬하기

- Python에서는 `merge()`를 사용하거나 R에서는 `join()`을 사용하여 연도 간의 차이 계산하기

# 연습 문제

1.  <a href="#sec-eda-hist" data-type="xref">"히스토그램"</a>과 동일한 히스토그램을 작성하되, 패스 시도당 EPA에 대해 작성하세요.

2.  <a href="#sec-eda-hist" data-type="xref">"히스토그램"</a>과 동일한 상자 수염 그림을 작성하되, 패스 시도당 EPA에 대해 작성하세요.

3.  <a href="#sec-eda-stable" data-type="xref">"선수 수준에서의 시도당 패싱 야드의 안정성"</a>과 동일한 안정성 분석을 수행하되, 패스 시도당 EPA에 대해 수행하세요. YPA를 사용할 때와 질적으로 동일한 결과를 확인할 수 있나요? 연도별로 YPA 수치는 비슷하지만 연도별로 패스 시도당 EPA 수치가 크게 다른 선수가 있나요? 이는 어디에서 비롯될 수 있을까요?

4.  긴 패스 시도에 대한 데이터가 짧은 패스 시도에 대한 데이터보다 덜 안정적인 이유 중 하나는 데이터 수가 적기 때문이며, 이는 20야드를 긴 패스로 임의로 자르는 기준(PFF와 같은 회사에 의해)이 크게 작용한 결과입니다. 데이터를 절반으로 균등하게 분할하는 기준점을 찾아 동일한 분석을 수행하세요. 결과가 동일하게 유지되나요?

# 추천 도서

그리기에 대해 더 자세히 알고 싶다면 저희가 유용하다고 생각한 몇 가지 자료를 참조하세요.

- [*The Visual Display of Quantitative Information* by Edward Tufte](https://oreil.ly/BYBhX) (Graphics Press, 2001). 이 책은 데이터에 대해 생각하는 방법에 관한 고전입니다. 이 책에는 코드가 포함되어 있지 않지만, 데이터를 위한 정보를 보는 방법을 보여줍니다. 이 책의 지침은 매우 귀중합니다.

- [`ggplot2` package documentation](https://ggplot2.tidyverse.org). R을 사용하는 경우 이 문서는 `ggplot2`에 대해 더 자세히 알아보기 위해 가장 먼저 시작해야 할 곳입니다. 이 페이지에는 초보자를 위한 자료와 고급 자료로 연결되는 링크가 포함되어 있습니다. 또한 둘러보기에 좋은 예제들도 포함되어 있습니다.

- [`seaborn` package documentation](https://seaborn.pydata.org). Python을 사용하는 경우 이 문서는 `seaborn`에 대해 더 자세히 알아보기 위해 가장 먼저 시작해야 할 곳입니다. 이 페이지에는 초보자를 위한 자료와 고급 자료로 연결되는 링크가 포함되어 있습니다. 또한 둘러보기에 좋은 예제들도 포함되어 있습니다. 이 페이지의 갤러리는 데이터를 시각화하는 방법에 대해 생각할 때 특히 유용합니다.

- [*ggplot2: Elegant Graphics for Data Analysis*](https://ggplot2-book.org), 3rd edition, by Hadley Wickham et al. (Springer). 3판은 현재 개발 중이며 온라인에서 액세스할 수 있습니다. 이 책은 `ggplot2`가 어떻게 작동하는지 매우 자세히 설명할 뿐만 아니라, 단어를 사용하여 데이터 그리기에 대해 생각하는 좋은 방법을 제공합니다. 제시된 코드의 각 줄을 분석하고 수정하면서 이 책을 읽으면 `ggplot2`의 전문가가 될 수 있습니다. 하지만 이 길이 반드시 쉬운 길은 아닙니다.

