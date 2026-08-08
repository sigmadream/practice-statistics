# 3장. 단순 선형 회귀: 기대 대비 러싱 야드(Rushing Yards Over Expected)

미식축구는 맥락적인 스포츠입니다. 패스가 성공할지 여부를 고려해 보세요. 이는 여러 가지 요인에 따라 달라집니다. 쿼터백이 압박을 받고 있었는가(성공하기 더 어렵게 만듦)? 수비가 패스를 예상하고 있었는가(성공하기 더 어렵게 만듦)? 패스의 깊이는 어느 정도였는가(타겟의 깊이가 깊어질수록 성공률은 떨어짐)?

사람들이 미식축구 분석에 흥미를 잃게 만드는 것은, 그들이 보기에 게임에 대한 맥락적 이해가 부족한 결론입니다. "원시 수치(Raw numbers)"는 오해를 불러일으킬 수 있습니다. 샘 브래드퍼드(Sam Bradford)는 2016년에 미네소타 바이킹스(Minnesota Vikings) 소속으로 한 시즌 NFL 패스 성공률 기록을 세운 적이 있습니다. 시즌 초반 트레이드를 통해 팀에 합류하여 새로운 환경에 빠르게 적응해야 했기 때문에 이는 인상적이었습니다. 그것이 인상적이긴 했지만, 그렇다고 해서 그가 그해 NFL에서 최고의 쿼터백이었다거나 가장 정확한 쿼터백이었다는 것을 의미하지는 않습니다. 한 가지 예를 들면, 그해 그는 평균 타겟 깊이(aDOT)가 단 6.6야드였으며, PFF에 따르면 이는 NFL에서 37위에 불과했습니다. 이로 인해 그의 패스 시도당 야드는 비교적 평균적인 7.0으로 떨어졌고, 이는 리그에서 공동 20위에 불과했습니다. <a href="ch04.html#sec-mr-ryoe2" data-type="xref">4장</a>에서는 해당 수치에 대해 더 많은 맥락을 제공하고, 이를 스스로 조정하는 방법을 보여줍니다.

다행히도, `nflfastR`을 지원하는 사람들의 훌륭한 작업 덕분에, 여러분은 *회귀(regression)*라는 통계적 도구를 적용하여 메트릭에 대한 자신만의 맥락을 제공할 수 있습니다. 회귀를 통해 여러분은 선수의 생산성에 영향을 미치는 것으로 나타난 변수(또는 *특성(features)*)를 *정규화(normalize)*하거나 *통제(control for)*할 수 있습니다. 어떤 특성이 선수의 생산성을 예측하는지 여부는 실생활에서 증명하기가 매우 어렵습니다. 또한 캔자스시티 치프스(Kansas City Chiefs)의 패트릭 마홈스(Patrick Mahomes)나 테네시 타이탄스(Tennessee Titans)의 데릭 헨리(Derrick Henry)처럼 이와 관련한 우리의 가정을 깨뜨리는 선수들이 등장하기도 합니다. 더욱이 데이터는 종종 성과에 영향을 미치는 많은 요소를 포착하지 못합니다. 인생과 마찬가지로 모든 것을 고려할 수는 없지만 다행히 가장 중요한 것들은 포착할 수 있습니다. 텍사스 공과대학교(Texas Tech University)에 있는 리처드의 교수님 중 한 명인 캐서린 롱(Katharine Long)은 이러한 접근 방식을 믹 재거 정리(Mick Jagger theorem)로 정의하는 것을 좋아합니다. "항상 원하는 것을 얻을 수는 없지만, 가끔 노력하다 보면 필요한 것을 얻을 수 있을지도 모릅니다(You can't always get what you want, but if you try sometimes, you just might get what you need)."

공공 및 민간 미식축구 분석 분야 모두에서 정규화 과정은 일반적으로 이 장에서 다루는 단순 선형 회귀보다 더 복잡한 모델을 필요로 합니다. 하지만 어딘가에서는 시작해야 합니다. 그리고 단순 선형 회귀는 이해하기 쉬우면서도 여러 다른 유형의 분석을 위한 토대가 되기 때문에 모델링을 시작하기에 좋은 출발점을 제공합니다.

###### 참고(Note)

많은 분야에서 단순 선형 회귀를 사용하며, 이로 인해 여러 용어가 사용됩니다. 수학적으로 예측 변수는 주로 *x*이고 반응 변수는 주로 *y*입니다. *x*의 동의어로는 *예측 변수(predictor variable)*, *특성(feature)*, *설명 변수(explanatory variable)*, *독립 변수(independent variable)* 등이 있습니다. *y*의 동의어로는 *반응 변수(response variable)*, *대상(target)*, *종속 변수(dependent variable)* 등이 있습니다. 마찬가지로 의학 연구에서는 학력, 연령 또는 기타 사회경제적 데이터와 같은 외생 데이터나 교란 데이터(통계학자에게는 *변수(variables)*, 데이터 과학자에게는 *특성(features)*)를 *보정(correct for)*하는 경우가 많습니다. 이 장과 <a href="ch04.html#sec-mr-ryoe2" data-type="xref">4장</a>에서 *정규화(normalize)* 및 *통제(control for)*라는 용어를 통해 동일한 개념을 배우게 됩니다.

*단순 선형 회귀(Simple linear regression)*는 단일 종속 변수(또는 *특성*)와 선형 관계에 있다고 가정되는 단일 설명 변수가 있는 모델로 구성됩니다. *단순 선형 회귀*는 하나의 독립 예측 변수를 사용하여 반응 변수를 예측 변수의 함수로 추정함으로써 통계적으로 "최적인(best)" 직선을 맞춥니다(fit). *단순(Simple)*이라는 말은 예측 변수가 하나뿐이며 절편(intercept)이 있다는 것을 의미하는데, 이 가정은 <a href="ch04.html#sec-mr-ryoe2" data-type="xref">4장</a>에서 완화하는 방법을 보여줍니다. *선형(Linear)*이라는 말은 직선(고등학교 대수를 기억하는 분들을 위해 곡선이나 다항식 선과 비교하여)을 의미합니다.

[1877년 프랜시스 골턴(Francis Galton)](https://oreil.ly/5hyWI)이 지적했듯이, *회귀(Regression)*는 원래 관찰값들이 시간이 지남에 따라 평균으로 돌아가거나 *회귀할* 것이라는 개념을 의미했습니다. 예를 들어, 러닝백(running back)이 어느 한 해에 평균 이상의 캐리당 러싱 야드를 기록했다면, 다른 모든 조건이 동일할 때, 향후 몇 년 동안 통계적으로 그들이 리그 평균으로 되돌아오거나 *회귀(regress)*할 것으로 예상합니다. 많은 모델에서 이루어지는 선형 가정이 부담스러운 경우가 많지만 일반적으로 첫 시도로서는 괜찮습니다.

단순 선형 회귀 적용을 시작하기 위해, 2020 빅 데이터 보울(Big Data Bowl) 기간 동안 공개 공간에서 이미 해결된 문제에 대해 작업해 볼 것입니다. 이 이벤트 참가자들은 *추적 데이터(tracking data)* (0.1초마다 경기장에 있는 모든 22명 선수의 위치, 방향, 오리엔테이션)를 사용하여 플레이에서 예상되는 획득 러싱 야드를 모델링했습니다. 그런 다음 이 값을 플레이에서 선수의 실제 러싱 야드에서 빼서 *기대 대비 러싱 야드(rushing yards over expected, RYOE)*를 결정했습니다. 1장에서 이야기했듯이, 이러한 종류의 잔차(residual) 분석은 모든 스포츠 분석의 초석이 되는 작업입니다.

이후 RYOE 메트릭은 NFL 경기 방송에 도입되었습니다. Tej Seth가 PFF에서 [RYOE용 R Shiny 앱](https://oreil.ly/ZD2V_)을 사용하여 수행한 것처럼, 추적 데이터 대신 스카우팅 데이터를 사용하는 버전을 만드는 것을 포함하여 이 메트릭을 개선하기 위한 추가 작업이 수행되었습니다. 모델의 메커니즘에 관계없이, 넓은 의미에서의 아이디어는 러셔(rusher)가 야드를 얻기 위해 겪어야 하는 상황을 조정하는 것입니다.

###### 참고(Note)

[빅 데이터 보울(Big Data Bowl)](https://oreil.ly/XAXTJ)은 NFL의 데이터 및 분석 디렉터인 마이클 로페즈(Michael Lopez)의 아이디어입니다. 에릭과 마찬가지로 로페즈도 이전에 교수였으며, 로페즈의 경우 스키드모어 대학(Skidmore College)에서 통계학 교수로 재직했습니다. 로페즈의 [홈페이지](https://statsbylopez.com)에는 스포츠뿐만 아니라 커리어에 대한 유용한 팁, 통찰력, 조언이 포함되어 있습니다.

RYOE를 모방하되 훨씬 작은 규모로 모방하기 위해, 특정 플레이에 *남은 야드(yards to go)*를 사용할 것입니다. 각 미식축구 플레이에는 다운(down)과 거리(distance)가 있는데, 여기서 *다운*은 팀이 10야드를 얻거나 터치다운 또는 필드골을 기록하기 위해 주어지는 4번의 다운 시퀀스 중의 위치를 의미합니다. *거리* 또는 *남은 야드*는 해당 목표를 달성하기 위해 남은 거리를 말하며 데이터에 `ydstogo`로 코딩됩니다.

합리적인 사람이라면 특정한 다운과 남은 야드가 RYOE에 영향을 미칠 것이라고 예상할 것입니다. 이러한 관찰이 발생하는 이유는 남은 야드가 많을수록 수비가 보통 롱 플레이(longer plays)를 막으려 하기 때문에 공을 들고 뛰기가 더 쉽기 때문입니다. 예를 들어, 공격팀이 서드 다운(third down)에 남은 거리가 10야드인 상황일 때 수비팀은 빅 플레이를 피하기 위해 뒤로 물러나 플레이합니다. 반대로 세컨드 다운(second down)에 1야드가 남았을 때, 수비팀은 퍼스트 다운이나 터치다운을 막기 위해 전진해서 플레이합니다.

수년 동안 팀들은 숏 야디지 백(short-yardage back)(보통 체격이 더 큰 러닝백)을 배치해 왔는데, 이들은 퍼스트 다운이나 터치다운에 1~2야드만 필요한 3번째나 4번째 다운에서 (적은) 야드를 획득하는 임무를 맡았습니다. 이 선수들은 판타지 풋볼에서, 팀이 득점하기 위해 팀의 선발 러닝백이 종종 많은 역할을 한 터치다운을 가로채는(또는 공로를 차지하는) 능력으로 귀중하게 여겨졌습니다. 그러나 숏 야디지 백의 캐리당 야드 수치는 선발 러닝백에 비해 인상적이지 않았습니다. 이 숏 야디지 백의 캐리당 야드는 선발 러닝백에 비해 맥락이 결여되어 있었습니다. 따라서 RYOE와 같은 메트릭은 러닝백 플레이의 맥락을 정규화하는 데 도움이 됩니다.

NFL 역사상 많은 사례 선수들이 존재합니다. 1996년 탬파베이 버커니어스(Tampa Bay Buccaneers)의 2라운드 픽이었던 마이크 알스토트(Mike Alstott)는 1990년대 후반/2000년대 초반 급부상한 벅스 팀에서 숏 야디지 백으로 활약하는 경우가 많았습니다. 대조적으로 1997년 팀의 1라운드 지명자였던 백필드 동료 워릭 던(Warrick Dunn)은 "얼리 다운(early-down)" 역할을 맡았습니다. 그 결과 같은 팀의 멤버로서 이들의 캐리당 야드 수치는 서로 달랐습니다(알스토트는 3.7야드, 던은 4.0야드). 따라서 회귀는 이를 고려하고 더 나은 비교를 통해 RYOE와 같은 메트릭을 만드는 데 도움이 될 수 있습니다.

###### 참고(Note)

상위 2라운드에서 러닝백을 드래프트하는 것의 지혜는, 2년 연속은 차치하고라도 한 번만으로도, 미식축구 분석에서 완전히 별개의 주제입니다. 드래프트에 대해서는 <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a>에서 아주 자세히 이야기합니다.

이 장에서 단순 선형 회귀를 이해하는 것은 다른 장에서 다루는 기술의 토대 역할을 하기도 합니다. 예를 들어 <a href="ch04.html#sec-mr-ryoe2" data-type="xref">4장</a>의 더 복잡한 RYOE 모델, <a href="ch05.html#sec-lr-pass" data-type="xref">5장</a>의 패싱 게임에서 기대 대비 패스 성공률, <a href="ch06.html#sec-pos-td" data-type="xref">6장</a>의 경기당 터치다운 패스, <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a>의 드래프트 데이터 평가에 사용되는 모델 등이 있습니다. 저자들을 포함하여 많은 사람들은 선형 모델을 응용 통계와 데이터 과학을 위한 '일꾼'이자 '토대'라고 부릅니다.

# 탐색적 데이터 분석

단순 선형 회귀를 실행하기 전에, 모델링 과정의 일부로서 <a href="ch02.html#sec-EDA-stable" data-type="xref">2장</a>에서 배운 탐색적 데이터 분석(EDA) 기술을 사용하여 데이터를 플롯(plot)하는 것이 항상 좋습니다. Python의 `seaborn`이나 R의 `ggplot2`를 사용하여 이 작업을 수행할 것입니다. RYOE를 계산하기 전에 데이터를 로드하고 변환해야(wrangle) 합니다. 2016년부터 2022년까지의 데이터를 사용할 것입니다. 먼저 패키지와 데이터를 로드하세요.

###### 팁(Tip)

터미널에서 **`pip install statsmodels`**를 사용하여 `statsmodels` 패키지를 설치했는지 확인하세요.

Python을 사용하는 경우 이 코드를 사용하여 데이터를 로드하세요.

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

seasons = range(2016, 2022 + 1)
pbp_py = nfl.import_pbp_data(seasons)
```

R을 사용하는 경우 이 코드를 사용하여 데이터를 로드하세요.

```
## R
library(tidyverse)
library(nflfastR)

pbp_r <- load_pbp(2016:2022)
```

데이터를 로드한 후 러닝 플레이를 선택하세요. 필터링 기준 `play_type == "run"`을 사용하세요. 또한 러셔가 없는 플레이를 제거하고 누락된 러싱 야드를 `0`으로 바꿉니다.

Python에서는 쿼리의 일부로 `rusher_id.notnull()`을 사용한 다음 누락된 `rushing_yards` 값을 `0`으로 바꿉니다.

```
## Python
pbp_py_run =\
    pbp_py.query('play_type == "run" & rusher_id.notnull()')\
    .reset_index()
pbp_py_run\
    .loc[pbp_py_run.rushing_yards.isnull(), "rushing_yards"] = 0
```

R에서는 `filter()` 단계의 일부로 `!is.na(rusher_id)`를 사용한 다음 `ifelse()` 함수와 함께 `mutate()`를 사용하여 누락된 값을 바꿉니다.

```
## R
pbp_r_run <-
    pbp_r |>
    filter(play_type == "run" & !is.na(rusher_id)) |>
    mutate(rushing_yards = ifelse(is.na(rushing_yards), 0, rushing_yards))
```

다음으로, 모델을 만들기 전에 원시 데이터를 시각화합니다. Python에서는 `seaborn`의 `displot()`을 사용하여 <a href="#fig-py-ytg-ry" data-type="xref">그림 3-1</a>을 만듭니다.

```
## Python
sns.set_theme(style="whitegrid", palette="colorblind")
sns.scatterplot(data=pbp_py_run, x="ydstogo", y="rushing_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0301.png" />
<h6 id="figure-3-1.-yards-to-go-plotted-against-rushing-yards-using-seaborn">그림 3-1. <code>seaborn</code>을 사용하여 그린 남은 야드(Yards to go) 대 획득 러싱 야드(rushing yards)</h6>
</figure>

R에서는 `ggplot2`의 `geom_point()`를 사용하여 <a href="#fig-r-ytg-ry" data-type="xref">그림 3-2</a>를 만듭니다.

```
ggplot(pbp_r_run, aes(x = ydstogo, y = rushing_yards)) +
    geom_point() +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0302.png" />
<h6 id="figure-3-2.-yards-to-go-plotted-against-rushing-yards-using-ggplot2">그림 3-2. <code>ggplot2</code>를 사용하여 그린 남은 야드(Yards to go) 대 획득 러싱 야드(rushing yards)</h6>
</figure>

그림 <a href="#fig-py-ytg-ry" data-type="xref" data-xrefstyle="select:labelnumber">3-1</a>과 <a href="#fig-r-ytg-ry" data-type="xref" data-xrefstyle="select:labelnumber">3-2</a>는 점들이 빽빽하게 모여 있는 그래프여서, 남은 야드와 플레이에서 획득한 러싱 야드 수 사이에 관계가 있는지 알아보기 어렵습니다. 플롯을 읽기 쉽게 만들기 위해 몇 가지 작업을 수행할 수 있습니다. 먼저 추세선을 추가하여 데이터가 위로 향하는지, 아래로 향하는지, 아니면 어느 쪽도 아닌지 확인합니다.

Python에서는 `regplot()`을 사용하여 <a href="#fig-py-ytg-ry-tl" data-type="xref">그림 3-3</a>을 만듭니다.

```
## Python
sns.regplot(data=pbp_py_run, x="ydstogo", y="rushing_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0303.png" />
<h6 id="figure-3-3.-yards-to-go-plotted-against-rushing-yards-with-a-trend-line-seaborn">그림 3-3. 추세선을 포함하여 <code>seaborn</code>으로 그린 남은 야드 대 러싱 야드</h6>
</figure>

R에서는 <a href="#fig-r-ytg-ry" data-type="xref">그림 3-2</a>의 코드와 함께 `stat_smooth(method = "lm")`을 사용하여 <a href="#fig-r-ytg-ry-tl" data-type="xref">그림 3-4</a>를 만듭니다.

```
ggplot(pbp_r_run, aes(x = ydstogo, y = rushing_yards)) +
    geom_point() +
    theme_bw() +
    stat_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0304.png" />
<h6 id="figure-3-4.-yards-to-go-plotted-against-rushing-yards-with-a-trendline-ggplot2">그림 3-4. 추세선을 포함하여 <code>ggplot2</code>로 그린 남은 야드 대 러싱 야드</h6>
</figure>

그림 <a href="#fig-py-ytg-ry-tl" data-type="xref" data-xrefstyle="select:labelnumber">3-3</a>과 <a href="#fig-r-ytg-ry-tl" data-type="xref" data-xrefstyle="select:labelnumber">3-4</a>에서, 비록 매우 작긴 하지만 양의 기울기를 볼 수 있습니다. 이것은 남은 야드가 증가함에 따라 러싱 획득 야드도 약간 증가함을 보여줍니다. 데이터를 살펴보기 위해 시도해 볼 수 있는 또 다른 접근 방식은 *구간화 및 평균화(binning and averaging)*입니다. 이는 (<a href="ch02.html#sec-eda-hist" data-type="xref">"히스토그램"</a>에서 다룬) 히스토그램의 아이디어를 차용한 것이지만, 각 구간에 개수(count)를 사용하는 대신 각 구간에 평균(average)을 사용합니다. 이 경우 구간은 정의하기 쉽습니다. 정수인 ydstogo 값이 구간이 됩니다.

이제 각 구간에서 획득한 캐리당 야드의 값을 평균 냅니다. Python에서는 데이터를 집계(aggregate)한 다음 이를 시각화하여 <a href="#fig-ypc-py" data-type="xref">그림 3-5</a>를 만듭니다.

```
## Python
pbp_py_run_ave =\
    pbp_py_run.groupby(["ydstogo"])\
    .agg({"rushing_yards": ["mean"]})

pbp_py_run_ave.columns = \
    list(map("_".join, pbp_py_run_ave.columns))
pbp_py_run_ave\
    .reset_index(inplace=True)

sns.regplot(data=pbp_py_run_ave, x="ydstogo", y="rushing_yards_mean");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0305.png" />
<h6 id="figure-3-5.-average-yards-per-carry-plotted-with-seaborn">그림 3-5. <code>seaborn</code>으로 그린 평균 캐리당 야드</h6>
</figure>

R에서는 새로운 변수인 *캐리당 야드(yards per carry)* (`ypc`)를 만든 다음 결과를 시각화하여 <a href="#fig-ypc-r" data-type="xref">그림 3-6</a>을 만듭니다.

```
## R
pbp_r_run_ave <-
    pbp_r_run |>
    group_by(ydstogo) |>
    summarize(ypc = mean(rushing_yards))

ggplot(pbp_r_run_ave, aes(x = ydstogo, y = ypc)) +
    geom_point() +
    theme_bw() +
    stat_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0306.png" />
<h6 id="figure-3-6.-average-yards-per-carry-plotted-with-ggplot2">그림 3-6. <code>ggplot2</code>로 그린 평균 캐리당 야드</h6>
</figure>

###### 팁(Tip)

그림 <a href="#fig-ypc-py" data-type="xref" data-xrefstyle="select:labelnumber">3-5</a>와 <a href="#fig-ypc-r" data-type="xref" data-xrefstyle="select:labelnumber">3-6</a>을 통해 평균 획득 야드와 남은 야드 사이의 양의 선형 관계를 확인할 수 있습니다. 구간화 및 평균화가 전체 데이터세트에 대해 회귀를 수행하는 것을 대체할 수는 없지만, 이 접근 방식은 애초에 그러한 노력을 할 가치가 있는지에 대한 통찰력을 제공하고 데이터를 더 잘 "볼(see)" 수 있도록 도와줍니다.

# 단순 선형 회귀

이제 데이터를 랭글링(wrangled)하고 심문(interrogated)했으므로 단순 선형 회귀를 실행할 준비가 되었습니다. 이 책에서 보여드리는 함수에 대해 Python과 R은 동일한 수식 표기법을 사용합니다. 예를 들어, `ydstogo`가 `rushing_yards`를 예측하는 단순 선형 회귀를 구축하려면 `rushing_yards ~ 1 + ydstogo` 수식을 사용합니다.

###### 팁(Tip)

수식의 왼쪽에는 대상 변수, 즉 반응 변수가 포함됩니다. 수식의 오른쪽에는 반응 변수, 즉 예측 변수들이 포함됩니다. <a href="ch04.html#sec-mr-ryoe2" data-type="xref">4장</a>에서는 `+`로 구분된 여러 예측 변수를 사용하는 방법을 보여줍니다.

이 수식은 `rushing_yards`가 절편(`1`)과 남은 야드(`ydstogo`)의 기울기 매개변수에 *의해 예측된다*(물결표인 `~`로 표시되며, 이는 미국 키보드의 `1` 키 옆에 있고 다른 키보드에서는 위치가 다를 수 있음)고 읽을 수 있습니다. `1`은 모델이 어디에 절편을 포함하는지 명시적으로 알려주기 위한 선택적 값입니다. 저희의 코드나 저희가 읽는 다른 대부분 사람들의 코드에서는 일반적으로 수식에 절편을 포함하지 않지만, 모델에서 이 항(term)에 대해 명시적으로 생각하는 데 도움이 되도록 여기서는 포함했습니다.

###### 참고(Note)

`statsmodels`를 사용한 수식은 일반적으로 R과 유사하거나 동일합니다. 이는 컴퓨터 언어가 종종 다른 컴퓨터 언어에서 차용하기 때문입니다. Python의 `pandas`가 R에서 데이터프레임을 차용한 것과 유사하게, `statsmodels`는 R에서 수식을 차용했습니다. R 또한 아이디어를 차용하며, 사실 R은 S 언어를 오픈 소스로 재창조한 것입니다. 또 다른 예로 R의 `tidyverse`와 Python의 `pandas` 모두 데이터를 정제하기 위한 구문과 아이디어를 `SQL` 유형의 언어에서 차용했습니다.

머신 러닝에 더 적합하고 더 대중적인 Python 패키지인 `scikit-learn`에 비해, 통계적 추론에 더 적합하기 때문에 저희는 `statsmodels` 패키지를 사용합니다. 또한 `statsmodels`는 R과 유사한 구문을 사용하므로 두 언어를 더 쉽게 비교할 수 있습니다.

Python에서는 `smf`로 가져온(imported) `statsmodels` 패키지의 `formula.api`를 사용하여 *상소 최소 제곱 회귀(ordinary least-squares regression)*, 즉 `ols()`를 실행합니다. 모델을 구축하려면 Python에 회귀를 맞추는(fit) 방법을 알려주어야 합니다. 이를 위해, 플레이의 러싱 야드 수(`rushing_yards`)가 절편(`1`)과 플레이에 남은 야드 수(`ydstogo`)에 의해 예측되는 모델을 구축하는데, 이는 `rushing_yards ~ 1 + ydstogo`라는 수식으로 작성됩니다.

Python을 사용하여 모델을 구축하고(build) 맞춘(fit) 다음, 모델의 요약(summary)을 살펴보세요.

```
## Python
import statsmodels.formula.api as smf

yard_to_go_py =\
    smf.ols(formula='rushing_yards ~ 1 + ydstogo', data=pbp_py_run)

print(yard_to_go_py.fit().summary())
```

결과는 다음과 같습니다.

OLS Regression Results ============================================================================== Dep. Variable: rushing_yards R-squared: 0.007 Model: OLS Adj. R-squared: 0.007 Method: Least Squares F-statistic: 623.7 Date: Sun, 04 Jun 2023 Prob (F-statistic): 3.34e-137 Time: 09:35:30 Log-Likelihood: -3.0107e+05 No. Observations: 92425 AIC: 6.021e+05 Df Residuals: 92423 BIC: 6.022e+05 Df Model: 1 Covariance Type: nonrobust ============================================================================== coef std err t P\>\|t\| \[0.025 0.975\] ------------------------------------------------------------------------------ Intercept 3.2188 0.047 68.142 0.000 3.126 3.311 ydstogo 0.1329 0.005 24.974 0.000 0.122 0.143 ============================================================================== Omnibus: 81985.726 Durbin-Watson: 1.994 Prob(Omnibus): 0.000 Jarque-Bera (JB): 4086040.920 Skew: 4.126 Prob(JB): 0.00 Kurtosis: 34.511 Cond. No. 20.5 ============================================================================== Notes: \[1\] Standard Errors assume that the covariance matrix of the errors is correctly specified.

이 요약 출력에는 모델에 대한 설명이 포함됩니다. 종속 변수(`Dep. Variable`), 날짜(`Date`), 시간(`Time`)과 같은 요약 항목의 상당수는 직관적일 것입니다. 관찰 횟수(`No. Observations`)는 자유도(degrees of freedom)와 관련이 있습니다. *자유도*는 적합된 매개변수의 수와 비교하여 존재하는 "추가" 관찰의 수를 나타냅니다.

이 모델의 경우 `rushing_yards`에 대한 기울기와 절편인 `Intercept`, 두 가지 매개변수가 적합되었습니다. 따라서 잔차 자유도(`Df Residuals`)는 관찰 횟수(`No. Observations`) – 2와 같습니다. $`R^{2}`$ 값은 모델이 데이터에 얼마나 잘 들어맞는지를 나타냅니다. $`R^{2} = 1.0`$이면 모델이 데이터에 완벽하게 들어맞습니다. 반대로 $`R^{2} = 0`$이면 모델이 데이터를 전혀 예측하지 못합니다. 이 경우 0.007이라는 낮은 $`R^{2}`$는 이 단순 모델이 데이터를 잘 예측하지 못함을 보여줍니다.

관심 있는 다른 출력으로는 `Intercept`와 `ydstogo`에 대한 계수(coefficient) 추정치가 있습니다. `Intercept`는 퍼스트 다운이나 터치다운에 남은 야드가 0야드일 때(실제로는 결코 일어나지 않음) 예상되는 획득 러싱 야드 수입니다. `ydstogo`의 기울기는 남은 야드가 1야드 추가될 때마다 예상되는 추가 획득 러싱 야드 수에 해당합니다. 예를 들어 2야드가 남은 러싱 플레이는 평균적으로 3.2(절편) + 0.1(기울기) × 2(남은 야드 수) = 3.4야드를 생산할 것으로 예상됩니다.

계수와 함께 점 추정치(`coef`)와 표준 오차(`std err`)가 존재합니다. *표준 오차(SE)*는 계수 추정치 주변의 불확실성을 포착하며, <a href="app02.html#sec-ssdw-pass" data-type="xref">부록 B</a>에서 더 자세히 설명합니다. `t`-값은 통계 분포(구체적으로는 *t*-분포)에서 나오며 SE 및 신뢰 구간(CI)을 생성하는 데 사용됩니다. *p*-값은 계수가 0이라는 귀무가설이 참이라고 가정할 때 관찰된 *t*-값을 얻을 확률을 제공합니다.

*p*-값은 귀무가설 유의성 검정(NHST)과 연관되어 있는데, 대부분의 통계학 입문 과정에서 다루지만 실제 통계학자들 사이에서는 점차 사용되지 않는 추세입니다. 마지막으로, 요약에는 계수에 대한 95% CI가 포함됩니다. 더 낮은 CI는 `[0.025` 열이고, 더 높은 CI는 `0.975]` 열입니다 (97.5 – 2.5 = 95%). *95% CI*는 관찰 프로세스가 여러 번 반복된다고 가정할 때 시간의 95%는 계수에 대한 진정한 추정치를 포함해야 합니다. 그러나 여러분이 *어느* 5%의 시간에 틀렸는지는 결코 알 수 없습니다.

R을 사용하여 비슷한 선형 모델(`lm()`)을 맞춘 다음 요약 결과를 인쇄할 수 있습니다.

```
## R
yard_to_go_r <-
    lm(rushing_yards ~ 1 + ydstogo, data = pbp_r_run)

summary(yard_to_go_r)
```

결과는 다음과 같습니다.

Call: lm(formula = rushing_yards ~ 1 + ydstogo, data = pbp_r_run) Residuals: Min 1Q Median 3Q Max -33.079 -3.352 -1.415 1.453 94.453 Coefficients: Estimate Std. Error t value Pr(\>\|t\|) (Intercept) 3.21876 0.04724 68.14 \<2e-16 \*\*\* ydstogo 0.13287 0.00532 24.97 \<2e-16 \*\*\* --- Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1 Residual standard error: 6.287 on 92423 degrees of freedom Multiple R-squared: 0.006703, Adjusted R-squared: 0.006692 F-statistic: 623.7 on 1 and 92423 DF, p-value: \< 2.2e-16

회귀 출력의 일반적인 구조는 R과 Python 간에 다릅니다. 그러나 제공되는 항목은 유사하며 주된 차이점은 형식에서 나타납니다. R은 모델 수식 또는 `Call`을 제공하고 그 뒤에 잔차(residuals) 요약을 제공합니다. *잔차*는 데이터가 모델의 적합도(fit)와 얼마나 잘 비교되는지를 나타냅니다. RYOE를 위해서는 실제로 나중에 잔차를 사용할 것입니다. 그런 다음 요약은 계수(`Coefficients`)와 그 불확실성을 제공합니다. 마지막으로, Python의 세부 정보와 유사한 모델의 세부 정보가 인쇄됩니다.

###### 경고(Warning)

자유도를 확인하는 것은 모델링을 막 시작하는 사람들에게는 이상해 보일 수 있습니다. 그러나 이는 모델이 모든 입력을 올바르게 사용하고 있는지, 값이 유실되지 않았는지 확인하기 위해 데이터를 점검하는 훌륭한 방법이 될 수 있습니다. 저희 친구 중 한 명인 바브 베니(Barb Bennie)는 통계학 대학원생들에게 모델 전반에 걸쳐 자유도를 비교하는 방법을 가르치는 데 한 학기의 대부분을 할애합니다. 이 책을 쓸 때 Python과 R 버전의 `nflfastR` 데이터가 모델 추정치에 대해 서로 다른 값을 반환하고 있었는데, 자유도는 저희의 컴퓨터에서 업데이트가 필요한 패키지를 알아내는 단서가 되었습니다. 자유도를 이해하는 것의 유용성과 힘을 과소평가하지 마세요.

마지막으로, RYOE를 살펴보기 전에 데이터에 `RYOE` 열을 만들기 위해 잔차를 저장해야 합니다. *잔차(Residuals)*는 모델의 예상(또는 예측) 출력과 관찰된 데이터 간의 차이입니다. Python에서 `pandas`를 사용하여 모델의 잔차로부터 `pbp_py_run` 데이터프레임에 새로운 `RYOE` 열을 만듭니다.

```
## Python
pbp_py_run["ryoe"] =\
    yard_to_go_py\
    .fit()\
    .resid
```

###### 팁(Tip)

Python과 R의 선형 모델에는 이 책에서는 겉핥기식으로만 다루는 기능과 도구들이 있습니다. <a href="#sec-chp3-fr" data-type="xref">"추천 도서"</a>에 나열된 것과 같은 자료를 사용하여 이러한 도구의 세부 사항을 학습하면 선형 모델의 힘을 더 잘 활용하는 데 도움이 될 것입니다.

마찬가지로 R에서는 `pbp_r_run` 데이터를 변경(mutate)하여 새로운 열 `ryoe`를 만듭니다.

```
## R
pbp_r_run <-
    pbp_r_run |>
    mutate(ryoe = resid(yard_to_go_r))
```

###### 참고(Note)

통계 교육을 위해 만들어진 R은 S 언어를 기반으로 합니다. 이러한 역사와 1990년대 초반의 통계학 현황을 고려할 때, R은 언어 내에 선형 모델이 잘 통합되어 있습니다. 대조적으로, Python에는 통계적 추론을 위한 선형 모델 측면에서 R의 복제본, 구체적으로 `statsmodels` 패키지가 있습니다. Python의 주요 모델 패키지인 `scikit-learn`(`sklearn`)은 통계적 추론보다는 머신 러닝에 초점을 맞추고 있습니다. R과 Python의 역사를 이해하면 이 언어들이 *왜* 현재와 같은 형태로 존재하는지에 대한 통찰력을 얻을 수 있을 뿐만 아니라, 각 언어의 강점을 활용할 수 있습니다. 또한 저희는 단순히 통계적 추론을 위해 회귀 모델을 맞추는 것이 필요하고 또 하고자 하는 전부라면, 소프트웨어 선택으로서 R이 더 낫다고 주장하고 싶습니다.

# RYOE에서 누가 최고였을까요?

이제 2016년부터 2022년까지 총 RYOE(total yards over expected)와 캐리당 평균 RYOE(average yards over expected per carry)의 관점에서 RYOE의 리더보드를 살펴보겠습니다. <a href="ch02.html#sec-EDA-stable" data-type="xref">2장</a>의 패서 데이터에서와 마찬가지로, 일부 선수는 성과 이름 첫 글자가 같기 때문에 `rusher`와 `rusher_id` 모두로 그룹화해야 합니다.

`pandas`를 사용하여 `seasons`, `rusher_id`, `rusher`별로 그룹화하세요. 그런 다음 `ryoe`를 개수(count), 합계(sum), 평균(mean)으로 집계하고, `rushing_yards`를 평균으로 집계합니다. 그러면 다음과 같은 열이 생성됩니다.

- RYOE의 `count`(개수)는 *러셔가 캐리한 횟수*입니다.

- RYOE의 `sum`(합계)은 *총 RYOE*입니다.

- RYOE의 `mean`(평균)은 *캐리당 RYOE*입니다.

- 러싱 야드의 `mean`(평균)은 *캐리당 야드*입니다.

데이터프레임을 다루기 더 쉽게 만들기 위해 열을 평탄화(flatten)하고 인덱스를 재설정(reset)하세요. 다음으로, 열 이름을 미식축구에 특화된 이름으로 바꿉니다. 마지막으로, 캐리 횟수가 50회 이상인 선수만 인쇄(print)하도록 결과를 쿼리(query)합니다.

```
## Python
ryoe_py =\
    pbp_py_run\
    .groupby(["season", "rusher_id", "rusher"])\
    .agg({
        "ryoe": ["count", "sum", "mean"],
        "rushing_yards": "mean"})

ryoe_py.columns = \
    list(map("_".join, ryoe_py.columns))
ryoe_py.reset_index(inplace=True)

ryoe_py =\
    ryoe_py\
    .rename(columns={
        "ryoe_count": "n",
        "ryoe_sum": "ryoe_total",
        "ryoe_mean": "ryoe_per",
        "rushing_yards_mean": "yards_per_carry",
    }
).query("n > 50")

print(ryoe_py.sort_values("ryoe_total", ascending=False))
```

결과는 다음과 같습니다.

season rusher_id rusher n ryoe_total ryoe_per yards_per_carry 1989 2021 00-0036223 J.Taylor 332 417.501295 1.257534 5.454819 1440 2020 00-0032764 D.Henry 397 362.768406 0.913774 5.206549 1258 2019 00-0034796 L.Jackson 135 353.652105 2.619645 6.800000 1143 2019 00-0032764 D.Henry 387 323.921354 0.837006 5.131783 1474 2020 00-0033293 A.Jones 222 288.358241 1.298911 5.540541 ... ... ... ... ... ... ... ... 419 2017 00-0029613 D.Martin 139 -198.461432 -1.427780 2.920863 122 2016 00-0029613 D.Martin 144 -199.156646 -1.383032 2.923611 675 2018 00-0027325 L.Blount 155 -247.528360 -1.596957 2.696774 1058 2019 00-0030496 L.Bell 245 -286.996618 -1.171415 3.220408 267 2016 00-0032241 T.Gurley 278 -319.803875 -1.150374 3.183453 \[534 rows x 7 columns\]

전체 테이블을 Python에서 한 번에 인쇄하려면 `print(ryoe_py.query("n > 50").to_string())`을 실행하세요. 저희는 공간을 절약하기 위해 이 작업을 수행하지 않았습니다. 대안으로 `pandas`의 `set_option()` 함수를 사용하여 전체 세션에서의 인쇄 방식을 변경할 수 있습니다. 예를 들어 `pd.set_option("display.min_rows", 10)`은 항상 10개의 행을 인쇄합니다.

`R`에서는 `pbp_r_run` 데이터를 사용하고 `season`, `rusher_id`, `rusher`별로 그룹화합니다. 그런 다음 `summarize()`를 수행하여 그룹당 개수, 총 RYOE, 평균 RYOE, 캐리당 야드를 구합니다. 마지막으로 필터링하여 캐리 횟수가 50회 이상인 선수만 포함합니다.

```
## R
ryoe_r <-
    pbp_r_run |>
    group_by(season, rusher_id, rusher) |>
    summarize(
        n = n(),
        ryoe_total = sum(ryoe),
        ryoe_per = mean(ryoe),
        yards_per_carry = mean(rushing_yards)
    ) |>
    arrange(-ryoe_total) |>
    filter(n > 50)

print(ryoe_r)
```

결과는 다음과 같습니다.

\# A tibble: 534 × 7 \# Groups: season, rusher_id \[534\] season rusher_id rusher n ryoe_total ryoe_per yards_per_carry \<dbl\> \<chr\> \<chr\> \<int\> \<dbl\> \<dbl\> \<dbl\> 1 2021 00-0036223 J.Taylor 332 418. 1.26 5.45 2 2020 00-0032764 D.Henry 397 363. 0.914 5.21 3 2019 00-0034796 L.Jackson 135 354. 2.62 6.8 4 2019 00-0032764 D.Henry 387 324. 0.837 5.13 5 2020 00-0033293 A.Jones 222 288. 1.30 5.54 6 2019 00-0031687 R.Mostert 190 282. 1.48 5.83 7 2016 00-0033045 E.Elliott 344 279. 0.810 5.10 8 2021 00-0034791 N.Chubb 228 276. 1.21 5.52 9 2022 00-0034796 L.Jackson 73 276. 3.78 7.82 10 2020 00-0034791 N.Chubb 221 254. 1.15 5.48 \# ℹ 524 more rows

저희는 공간을 절약하기 위해 여러분에게 테이블 전체를 출력하도록 시키지 않았습니다. 하지만 끝에 `|> print(n = Inf)`를 사용하면 R에서 전체 테이블을 볼 수 있습니다. 대안으로 `options(pillar.print_min = n)`을 실행하여 R 세션에 대한 모든 인쇄 방식을 변경할 수도 있습니다.

필터링된 목록에 대해서는 이상치를 배제하고 페이지 공간을 절약하기 위해 볼을 50회 이상 캐리한 선수의 목록만 인쇄하도록 했습니다. 총 야드의 경우 캐리 횟수가 그토록 적은 선수는 어차피 그렇게 많은 RYOE를 누적하지 못할 것이므로 굳이 그렇게 할 필요는 없습니다.

총 RYOE를 기준으로 볼 때 2021년 조나단 테일러(Jonathan Taylor)는 2016년 이후 미식축구 최고의 러닝백이었으며 400 이상의 RYOE를 창출했습니다. 그다음은 앞서 언급한 헨리로, 2,000야드를 질주했던 2020년 시즌 동안 374 RYOE를 창출했습니다. 목록의 세 번째 선수인 라마 잭슨(Lamar Jackson)은 쿼터백으로, 2019년 1,200야드 이상을 러싱하고(쿼터백으로서 NFL 기록) 패스 터치다운 부문 리그 1위를 차지하여 NFL MVP를 수상했습니다. 2022년 4월, 잭슨은 이러한 노력에 힘입어 NFL 역사상 가장 큰 규모의 계약을 체결했습니다.

NFL 데이터의 흥미로운 특징 중 하나는 쿼터백을 위해 *설계된 런(designed runs)* (쿼터백이 공을 내리고 달리는 깨진(broken-down) 패싱 플레이가 아닌 실제 러닝 플레이)만이 이 데이터 세트에 포함된다는 것입니다. 따라서 잭슨이 자신이 달린 런 중 일부만으로 이렇게 많은 RYOE를 창출했다는 것은 놀라울 정도로 인상적입니다.

다음으로 캐리당 RYOE를 기준으로 데이터를 정렬합니다. R에 대한 코드만 포함하지만 이전 Python 코드도 쉽게 적용할 수 있습니다.

```
## R
ryoe_r |>
    arrange(-ryoe_per)
```

결과는 다음과 같습니다.

\# A tibble: 534 × 7 \# Groups: season, rusher_id \[534\] season rusher_id rusher n ryoe_total ryoe_per yards_per_carry \<dbl\> \<chr\> \<chr\> \<int\> \<dbl\> \<dbl\> \<dbl\> 1 2022 00-0034796 L.Jackson 73 276. 3.78 7.82 2 2019 00-0034796 L.Jackson 135 354. 2.62 6.8 3 2019 00-0035228 K.Murray 56 122. 2.17 6.5 4 2020 00-0034796 L.Jackson 121 249. 2.06 6.26 5 2021 00-0034750 R.Penny 119 229. 1.93 6.29 6 2022 00-0036945 J.Fields 85 160. 1.88 6 7 2022 00-0033357 T.Hill 96 178. 1.86 5.99 8 2021 00-0034253 D.Hilliard 56 101. 1.80 6.25 9 2022 00-0034750 R.Penny 57 99.2 1.74 6.07 10 2019 00-0034400 J.Wilkins 51 87.8 1.72 6.02 \# ℹ 524 more rows

캐리당 RYOE를 보면 잭슨의 기록이 4위 안에 3년 치나 포함되어 있으며, 그 사이에 또 다른 쿼터백인 카일러 머레이(Kyler Murray)의 기록도 한 해 껴있습니다. 머레이는 2019년 NFL 드래프트 전체 1순위 지명 선수로, 러닝과 패싱의 조합을 통해 2018년 미식축구 최악이었던 애리조나 카디널스(Arizona Cardinals)의 공격력을 2019년에 훨씬 더 훌륭한 수준으로 끌어올렸습니다. 2018년 시애틀 시호크스의 1라운드 픽이었던 라샤드 페니(Rashaad Penny)는 마침내 2021년에 두각을 나타내어 전체 캐리당 야드(6.3)에서 NFL 1위를 차지하면서 캐리당 기대 대비 약 2야드를 더 획득했습니다. 이로 인해 페니는 다음 오프시즌에 시애틀과 두 번째 계약을 맺었습니다.

우리가 던져야 할 합리적인 질문은 총 야드 수와 캐리당 야드 수 중 어느 것이 선수의 능력을 나타내는 더 나은 척도인지입니다. 판타지 풋볼 분석가와 드래프트 분석가 모두 "볼륨은 얻어지는 것이다(volume is earned)"라는 일반적인 합의를 가지고 있습니다. 어떤 선수가 많은 캐리를 창출할 만큼 충분히 플레이할 때 그 데이터 안에 숨겨진 신호가 있다는 생각입니다.

데이터는 현실을 불완전하게 표현한 것이며, 선수들은 포착되지 않는 행동들을 합니다. 코치가 선수를 많이 출전시킨다면, 이는 그 선수가 훌륭하다는 것을 보여주는 좋은 지표입니다. 더욱이 같은 이유로 볼륨(volume)과 효율성(efficiency) 사이에는 일반적으로 음의 관계가 존재합니다. 어떤 선수가 많이 출전할 만큼 훌륭하다면 수비팀은 그에게 더 쉽게 집중하여 그의 효율성을 감소시킬 수 있습니다. 이것이 저희가 주전 선수에 비해 캐리당 야드 수치가 높은 백업 러닝백을 자주 보게 되는 이유입니다 (예를 들어, 2019-2022년 댈러스 카우보이스의 토니 폴라드(Tony Pollard)와 에제키엘 엘리엇(Ezekiel Elliott)을 살펴보세요). 반드시 볼륨을 최우선으로 고려해야 하는 것은 아닌 또 다른 요소들(지크(Zeke)의 드래프트 상태와 계약 등)도 작용하지만, 검토하는 것은 중요합니다.

# RYOE가 더 나은 메트릭인가요?

미식축구에서 선수나 팀 평가를 위한 새로운 메트릭을 만들 때마다 그 예측력을 테스트해야 합니다. 다른 사람과 마찬가지로 새로운 메트릭에 많은 생각을 담을 수 있고, 이 장에서와 같이 조정(adjustments)은 타당할 수 있습니다. 하지만 이 메트릭이 이전 반복 평가(iterations of the evaluation)보다 더 안정적이지 않다면, 그 작업이 헛수고였다고 결론 내리거나, 아니면 선수의 성과를 둘러싼 근본적인 상황이 실제로 그 신호를 전달하는 주체라고 결론을 내려야 합니다. 따라서 생산성 측면에서 일어나고 있는 일에 대해 개별 선수에게 너무 많은 부분을 귀속시키게 됩니다.

<a href="ch02.html#sec-EDA-stable" data-type="xref">2장</a>에서는 패스 데이터에 대한 안정성 분석을 수행했습니다. 이제 캐리가 50회 이상인 선수들의 기존 캐리당 야드 값과 비교하여 캐리당 RYOE를 살펴볼 것입니다. 총 RYOE나 러싱 야드는 저희가 살펴시보도록 지시하지 않을 텐데, 왜냐하면 두 가지 성과 척도(볼륨과 효율성)가 결합되어 있기 때문입니다. 두 가지 척도 모두에 볼륨이 포함되어 있으면 상황이 모호해집니다.

이 코드는 <a href="ch02.html#sec-chp2dpsp" data-type="xref">"긴 패스 대 짧은 패스"</a>의 코드와 유사하므로, 여기서는 자세한 설명 없이 주석이 달린 코드만 제공합니다. 따라서 자세한 내용은 해당 섹션을 참조하시기 바랍니다.

Python에서는 이 코드를 사용하세요.

```
## Python
#  keep only columns needed
cols_keep =\
    ["season", "rusher_id", "rusher",
     "ryoe_per", "yards_per_carry"]

# create current dataframe
ryoe_now_py =\
    ryoe_py[cols_keep].copy()

# create last-year's dataframe
ryoe_last_py =\
    ryoe_py[cols_keep].copy()

# rename columns
ryoe_last_py\
    .rename(columns = {'ryoe_per': 'ryoe_per_last',
                       'yards_per_carry': 'yards_per_carry_last'},
                       inplace=True)

# add 1 to season
ryoe_last_py["season"] += 1

# merge together
ryoe_lag_py =\
    ryoe_now_py\
    .merge(ryoe_last_py,
           how='inner',
           on=['rusher_id', 'rusher',
               'season'])
```

마지막으로 캐리당 야드에 대한 상관관계를 조사합니다.

```
## Python
ryoe_lag_py[["yards_per_carry_last", "yards_per_carry"]].corr()
```

결과는 다음과 같습니다.

yards_per_carry_last yards_per_carry yards_per_carry_last 1.00000 0.32261 yards_per_carry 0.32261 1.00000

RYOE에 대해서도 반복합니다.

```
## Python
ryoe_lag_py[["ryoe_per_last", "ryoe_per"]].corr()
```

그러면 다음과 같은 결과가 나타납니다.

ryoe_per_last ryoe_per ryoe_per_last 1.000000 0.348923 ryoe_per 0.348923 1.000000

R에서는 이 코드를 사용하세요.

```
## R
# create current dataframe
ryoe_now_r <-
    ryoe_r |>
    select(-n, -ryoe_total)

# create last-year's dataframe
# and add 1 to season
ryoe_last_r <-
    ryoe_r |>
    select(-n, -ryoe_total) |>
    mutate(season = season + 1) |>
    rename(ryoe_per_last = ryoe_per,
           yards_per_carry_last = yards_per_carry)

# merge together
ryoe_lag_r <-
    ryoe_now_r |>
    inner_join(ryoe_last_r,
               by = c("rusher_id", "rusher", "season")) |>
    ungroup()
```

그런 다음 두 개의 캐리당 야드 열을 선택하고 상관관계를 조사합니다.

```
## R
ryoe_lag_r |>
    select(yards_per_carry, yards_per_carry_last) |>
    cor(use = "complete.obs")
```

결과는 다음과 같습니다.

yards_per_carry yards_per_carry_last yards_per_carry 1.0000000 0.3226097 yards_per_carry_last 0.3226097 1.0000000

RYOE 열로 상관관계를 반복합니다.

```
## R
ryoe_lag_r |>
    select(ryoe_per, ryoe_per_last) |>
    cor(use = "complete.obs")
```

결과는 다음과 같습니다.

ryoe_per ryoe_per_last ryoe_per 1.0000000 0.3489235 ryoe_per_last 0.3489235 1.0000000

이러한 결과는 2년 연속 러싱 시도가 50회 이상인 선수들의 경우, 이 버전의 캐리당 RYOE가 캐리당 야드보다 해마다 약간 더 안정적이라는 것(상관계수가 더 크기 때문에)을 보여줍니다. 따라서 캐리당 야드에는 러닝백이 공을 들고 뛰는 특정 플레이에 내재된 정보가 포함되어 있습니다. 게다가 이 정보는 해마다 다를 수 있습니다. 이를 추출한 후, 러닝백 성과에 대한 저희의 새로운 메트릭은 해가 거듭될수록 예측력이 약간 더 높아졌습니다.

러닝백이 중요한가(또는 더 정확하게 말하자면 얼마나 중요한가)라는 질문에 대해서, 우리의 상관계수 차이는 아직 이렇다 할 결론을 도출할 만한 내용이 많지 않다는 것을 시사합니다. 이 장 이전에 여러분은 정말로 하나의 포지션(쿼터백)에 대한 통계만 살펴보았고, 해당 포지션의 프로필에서 더 안정적인 메트릭(짧은 패스의 패스 시도당 야드, *r* 값은 거의 0.5에 달함)은 캐리당 야드 및 RYOE보다 훨씬 더 안정적입니다. 더욱이 여러분은 패싱 게임과 비교한 러닝 게임에 대한 철저한 분석을 수행하지 않았는데, 이는 러닝백이 중요하다거나 중요하지 않다는 주장을 완성하는 데 필수적입니다.

미식축구 분석가들이 러닝백에 대해 묻는 핵심 질문은 다음과 같습니다.

1.  러닝백의 기여는 가치가 있는가?

2.  그들의 기여는 여러 해에 걸쳐 반복될 수 있는가?

<a href="ch04.html#sec-mr-ryoe2" data-type="xref">4장</a>에서는 러닝 게임에 영향을 미치는 다른 요인들을 통제하기 위해 변수를 추가할 것입니다. 예를 들어, 다운 앤 디스턴스(down and distance) 중 *다운(down)* 부분은 확실히 중요한데, 수비진이 같은 서드 다운 1야드 남은 상황보다 포스 다운 1야드 남은 상황에서 훨씬 더 타이트하게 플레이할 것이기 때문입니다. 14점 차이로 뒤처져(또는 지고) 있는 팀은 14점 차이로 앞서 있는 팀보다 공을 들고 뛰는 것이 훨씬 수월할 텐데(다른 모든 조건이 동일하다면), 왜냐하면 앞서고 있는 팀이 상대팀에게 *너무(too)* 많은 야드만 아니라면 야드 전진을 허용할 의향이 있는 "프리벤트(prevent)" 수비로 플레이할 수도 있기 때문입니다.

# 이 장에서 사용된 데이터 과학 도구

이 장에서는 다음 주제를 다루었습니다.

- Python에서는 `OLS()`를 사용하거나 R에서는 `lm()`을 사용하여 단순 선형 회귀 맞추기

- 단순 선형 회귀의 계수 이해하고 읽기

- Python에서는 `seaborn`의 `regplot()`을 사용하거나 R에서는 `geom_smooth()`를 사용하여 단순 선형 회귀 시각화하기

- Python과 R의 `corr()` 함수와 함께 상관관계를 사용하여 안정성 분석 수행하기

# 연습 문제

1.  상관관계 분석을 임계값인 100캐리로 반복하면 어떻게 될까요? *r* 값의 차이는 어떻게 될까요?

2.  알스토트의 모든 캐리는 서드 다운 1야드 남은 상황에서 이루어졌고, 던의 모든 캐리는 퍼스트 다운 10야드 남은 상황에서 이루어졌다고 가정해 보세요. 이것으로 그들의 캐리당 야드 수치의 불일치(3.7 대 4.0)를 설명하기에 충분할까요? 이 장에 있는 단순 선형 모델의 계수를 사용하여 이 질문을 이해해 보세요.

3.  엔드존까지 남은 야드(`yardline_100`)를 특성(feature)으로 하여 이 장의 분석을 반복하면 어떻게 될까요?

4.  리시버와 패싱 게임을 대상으로 이 장의 과정을 반복하세요. 이렇게 하려면 `play_type == "pass"`이고 `receiver_id`가 `NA`나 `NULL`이 아닌 조건을 기준으로 필터링해야 합니다.

# 추천 도서

에릭은 PFF에 근무하는 동안 러닝백과 러닝백의 가치에 대한 여러 기사를 썼습니다. 예시는 다음과 같습니다.

- [“The NFL’s Best Running Backs on Perfectly and Non-perfectly Blocked Runs in 2021”](https://oreil.ly/IArDE)

- [“Are NFL Running Backs Easily Replaceable: The Story of the 2018 NFL Season”](https://oreil.ly/x5dAk)

- [“Explaining Dallas Cowboys RB Ezekiel Elliott’s 2018 PFF Grade”](https://oreil.ly/aYhfj)

이 외에도 회귀 및 통계 입문에 관한 많은 책이 있으며, <a href="app02.html#sec-apdx-2-fr" data-type="xref">"추천 도서"</a>에는 몇 가지 통계 입문서가 나열되어 있습니다. 회귀와 관련하여 저희가 유용하다고 생각한 책은 다음과 같습니다.

- *Regression and Other Stories* by Andrew Gelman et al. (Cambridge University Press, 2020). 이 책은 회귀 분석을 실제 문제에 적용하는 방법을 보여줍니다. 실전 사례 연구를 더 찾아보고 계신 분들께, 저희는 회귀 적용에 대해 생각하는 방법을 배우는 데 도움이 되는 이 책을 추천합니다.

- *Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis*, 2nd edition, by Frank E. Harrell Jr. (Springer, 2015). 이 책은 저자 중 한 명이 회귀 모델링의 세계를 통해 생각하는 데 도움이 되었습니다. 심화 단계의 책이지만 회귀 분석에 대한 훌륭한 통찰력을 제공합니다. 이 책은 고급 학부 수준 또는 입문 대학원 수준으로 쓰여 있습니다. 어렵긴 하지만, 이 책을 끝까지 공부하면 회귀 분석을 마스터할 수 있습니다.

