# 6장. 스포츠 베팅에 데이터 과학 활용하기: 푸아송 회귀와 패스 터치다운 (Chapter 6. Using Data Science for Sports Betting: Poisson Regression and Passing Touchdowns)

미국 내, 그리고 광범위하게는 전 세계적으로 스포츠 베팅 분야에서 많은 발전이 이루어졌습니다. 2023년 2월 에릭이 슈퍼볼에 갔을 때 시야에 보이는 거의 모든 텔레비전 및 라디오 프로그램이 도박 관련 단체의 후원을 받고 있었습니다. 불과 5년 전만 해도 스포츠 베팅은 가장자리에서만 논의되는 금기시된 주제였으며, 네바다주를 제외한 어느 주에서도 합법이 아니었습니다. 그러나 2018년 봄, 미국 연방대법원이 프로 및 아마추어 스포츠 보호법(PASPA)을 폐지하면서 각 주가 자국 내에서 스포츠 베팅을 합법화할지 여부와 그 방법을 결정할 수 있게 되면서 모든 것이 바뀌었습니다.

스포츠 베팅은 뉴저지와 펜실베이니아를 얼리 어답터로 시작하여 미국 전역에서 서서히 합법화되기 시작했으며 일리노이와 애리조나를 포함한 서부 지역으로 확산되었습니다. 현재 거의 3분의 2에 달하는 주에서 어떤 형태로든 합법화된 베팅이 허용되고 있으며, 이로 인해 레크리에이션 베터와 전문 베터 모두를 위한 새롭고 다양한 상품을 제공하는 골드러시가 발생했습니다.

베팅 시장은 어느 주말 풋볼 경기장에서 어떤 일이 일어날지 예측하는 단일한 최고 예측 수단입니다. 그 이유는 군중의 지혜(wisdom of the crowds) 때문입니다. 이 주제는 제임스 수로위키(James Surowiecki)의 저서 _The Wisdom of the Crowds_ (Doubleday, 2004)에서 다루어집니다. 피나클(Pinnacle), 벳크리스(Betcris), 서카 스포츠(Circa Sports)와 같은 마켓 메이킹 _북스(books)_ (도박에서 *북스*는 베팅을 받는 회사를 의미하고 *부키(bookies)*는 동일한 일을 하는 개인을 뜻합니다)에는 오리지네이션(origination)이라는 과정을 사용하여 게임의 초기 가격을 생성함으로써 *라인을 설정(set the line)*하는 오즈메이커(oddsmaker)가 있습니다.

주초에 레크리에이션 베터와 전문 베터 모두 베팅을 통해 자신의 의견을 냅니다. 이러한 베팅은 일주일이 진행되고(날씨 및 부상 같은 정보가 업데이트됨에 따라) 증가할 수 있습니다. *클로징 라인(closing line)*에는 이론적으로 라인을 이동시킬 만큼 충분한 영향력을 가진 모든 베터의 의견이 베팅을 통해 포함되어 있습니다.

시장은 효율성을 향해 나아가는 경향이 있다고 가정하기 때문에 게임의 최종 가격(다음 섹션에서 설명)은 우리가 이용할 수 있는 정확한 (공개) 예측입니다. 베팅 시장에서 이기려면 *우위(edge)*가 필요하며, 이는 다른 베터는 이용할 수 없는 이점을 제공하는 정보를 의미합니다.

우위(edge)는 일반적으로 시장보다 더 나은 데이터 또는 시장보다 더 나은 데이터 합성 방법이라는 두 가지 출처에서 파생됩니다. 전자는 대개 시장의 다른 사람들보다 정보(부상 정보)를 더 빨리 입수하거나 (초창기 PFF가 그랬던 것처럼) 다른 사람들은 굳이 수집하지 않는 _종단적(longitudinal)_ 데이터(이러한 게임 수준 데이터에서 시간에 따른 자세한 관찰 데이터)를 수집하여 생성됩니다. 후자는 통계 기법을 사용하여 데이터를 처리하고 모델을 생성하여 자신의 가격을 설정한 후 자신의 가격(_내부 가격_)과 시장 가격의 차이에 베팅하는 대부분의 베터가 취하는 접근 방식입니다. 이것이 이 장의 주요 주제가 될 것입니다.

# 풋볼의 주요 시장 (The Main Markets in Football)

미식축구에서 세 가지 주요 시장은 전통적으로 스프레드(spread), 토탈(total), 머니라인(moneyline)이었습니다. 인기 있는 시장인 *스프레드(spread)*는 이해하기 매우 쉽습니다. 스프레드는 많은 경기 샘플을 기반으로 결과를 절반으로 나누기 위한 점수 값입니다. 예를 하나 들어보겠습니다. 워싱턴 커맨더스(Washington Commanders)가 뉴욕 자이언츠(New York Giants)를 상대로 동일한 조건 하에 무한한 횟수의 경기를 펼칠 수 있는 이론적 세계에서는 필드에서 평균 4점 더 우수합니다. 따라서 오즈메이커는 커맨더스와 자이언츠의 스프레드를 4점으로 정할 것입니다. 이 예에서는 베팅에 대한 5가지 결과가 발생할 수 있습니다.

- 어떤 사람이 커맨더스의 승리에 베팅합니다. 커맨더스가 5점 차 이상으로 이기면 그 사람은 베팅에서 승리합니다.

- 어떤 사람이 커맨더스의 승리에 베팅합니다. 커맨더스가 완전히 패하거나 5점 차 이상으로 이기지 못하면 그 사람은 베팅에서 패배합니다.

- 어떤 사람이 자이언츠의 승리에 베팅합니다. 자이언츠가 게임에서 완전히 이기거나 3점 이하 차이로 지면 그 사람은 베팅에서 승리합니다.

- 어떤 사람이 자이언츠의 승리에 베팅합니다. 자이언츠가 게임에서 완전히 패하면 그 사람은 베팅에서 패배합니다.

- 어떤 사람이 한 팀의 승리에 베팅하고 경기가 커맨더스에게 유리한 4점 차이로 _결정됩니다_(최종 점수). 이 경기는 *푸시(push)*로 간주되어 베터는 돈을 돌려받게 됩니다.

스포츠북에 대해 이점이 없는 포인트 스프레드 베터는 베팅의 약 50%를 이길 것으로 예상됩니다. 이 50%는 확률에 대한 이해에서 비롯된 것으로 스프레드가 시스템에 대한 모든 정보를 (이론적으로) 포착한다고 가정합니다. 예를 들어 동전 던지기에서 앞면에 베팅하는 것은 오랜 기간 동안 절반은 맞고 절반은 틀릴 것입니다. 스포츠북이 이 플레이어로부터 돈을 벌기 위해 각 베팅마다 _비고리시(vigorish)_ 또는 *비그(vig)*라는 수수료를 부과합니다. 포인트 스프레드 베팅의 경우 일반적으로 1달러당 10센트입니다.

따라서 워싱턴(–4)에 베팅하여 100달러를 따고 싶다면 100달러를 얻기 위해 110달러를 _걸어야(lay)_ 합니다. 이러한 베팅은 장기적으로 손익분기점에 도달하기 위해 무승부(푸시)가 아닌 베팅에서 52.38%(110 / (110 + 100)으로 계산)의 성공률을 요구합니다. 따라서 포인트 스프레드 베팅에서 승리하려면 스포츠북보다 대략 2.5포인트의 우위(52.38% - 50%는 약 2.5%)가 필요한데, 스포츠 베터의 99% 이상이 돈을 잃는다는 사실을 고려하면 이는 생각보다 훨씬 어려운 일입니다.

###### 팁 (Tip)

"카지노가 항상 이긴다(The house always wins)"라는 옛말이 나오는 이유는 하우스(카지노)가 장기전을 벌이고, 내재된 이점을 가지고 있으며, 잃는 것보다 최소한 조금 더 이기기 때문입니다. 예를 들어 룰렛 테이블의 경우 카지노의 우위는 녹색 0번 숫자가 포함되어 있다는 데서 비롯됩니다(그래서 검정색이나 빨간색을 얻을 확률이 50-50 미만입니다). 스포츠 베팅의 경우 카지노(하우스)는 비그(vig)를 얻기 때문에 오즈(배당률)가 일관되고 체계적으로 잘못되지 않는 한 사실상 거의 이익을 보장받을 수 있습니다.

*토탈(total)*에 베팅하려면 단순히 두 팀 점수의 합계가 지정된 금액을 초과하는지(over) 또는 미달하는지(under)에 베팅합니다. 예를 들어 커맨더스와 자이언츠의 시장 토탈이 43.5점(–110)이라고 가정해 봅시다. 언더(under)에 베팅하려면 100달러를 얻기 위해 110달러를 걸어야 하며 커맨더스와 자이언츠의 합산 점수가 43점 이하가 되기를 바라야 합니다. 44점 이상 득점하면 초기 판돈 110달러를 잃게 됩니다. 스프레드나 토탈에 0.5가 붙어있는 경우에는 무승부(푸시)가 불가능합니다. 일부 베터들은 경기 전체 토탈과 특정 경기 구간(전반전 또는 1쿼터)에만 적용되는 토탈을 모두 포함한 토탈 베팅에 특화되어 있습니다.

미식축구 전통 베팅 중 마지막은 *머니라인 베팅(moneyline bet)*입니다. 본질적으로 한 팀이 정면 승부에서 이기는 것에 베팅하는 것입니다. 경기가 진정한 50-50(pick’em) 확률이 되는 경우는 드물기 때문에 머니라인에 베팅하려면 적은 돈을 따기 위해 많은 돈을 걸거나(한 팀이 *탑독(favorite)*일 때), 많은 돈을 따기 위해 적은 돈을 걸게 됩니다(한 팀이 *언더독(underdog)*일 때). 예를 들어 커맨더스가 자이언츠를 상대로 이길 확률이 60%로 여겨진다면, 커맨더스의 머니라인 가격(북미 배당률 사용, 다른 국가는 소수점 배당률 사용)은 -150이 됩니다. 베터는 100달러를 따기 위해 150달러를 걸어야 합니다. 이 베팅의 소수점 배당률은 (100 + 150) / 150 = 1.67이며, 이는 투자 대비 총 수익률을 나타냅니다. –150이라는 수치는 부분적으로는 탑독 배당률 앞에 붙는 마이너스 기호라는 관례와 $`100 \times \frac{0.6}{(1 - 0.6)} = 150`$이라는 계산을 통해 도출됩니다. 여기에서 소수점 배당률은 (100 + 150) / 100 = 2.5입니다.

자이언츠의 배당률(가격)은 +150이 되며, 이는 100달러를 걸어 성공하면 원래 판돈에 추가로 150달러를 받는다는 의미입니다. 이 값은 역방향 계산 방식인 $`100 \times \frac{1 - 0.4}{0.4} = 150`$을 통해 도출되며, 언더독 가격 앞에는 플러스 기호를 붙인다는 관례를 따릅니다.

###### 참고 (Note)

카지노(book)는 머니라인 베팅에서도 비고리시(수수료)를 약간 취하기 때문에 워싱턴이 -150, 뉴욕이 +150이 되는 대신 워싱턴은 -160에 가깝고 뉴욕은 +140에 가까운 값을 보게 될 수도 있습니다(수수료는 카지노마다 다름). -160과 +140의 절댓값 사이의 차이는 슈퍼볼과 같은 큰 경기에 대한 드문 프로모션 시장을 제외한 모든 시장에 존재합니다.

# 푸아송 회귀 적용: 프롭 시장 (Application of Poisson Regression: Prop Markets)

회귀를 사용한 3대 풋볼 베팅 시장에서 제대로 작동하는 모델은 각 팀의 공격, 수비 및 스페셜 팀에 대한 평가와 날씨 및 부상 같은 상황 조정이 필요하기 때문에 이 책의 범위를 벗어납니다. 이 시장들은 많은 베터와 베팅 _핸들(handle)_ (베터가 베팅한 총 금액)을 끌어들이기 때문에 미국에서 효율적인 베팅 시장이자 전 세계적으로도 효율적인 베팅 시장으로 꼽힙니다.

그러나 PASPA가 폐지된 이후 스포츠북 운영자들은 풋볼 베팅 시장의 스프레드, 토탈, 머니라인이라는 심해에서 헤엄치고 싶지 않은 베터들을 위한 대안을 서둘러 만들었습니다. 그 결과 _프로포지션(proposition)_ (또는 _프롭(prop)_) 시장이 확산되었습니다. 역사적으로 슈퍼볼과 같은 대형 이벤트에만 국한되었던 이제 모든 NFL 경기와 대부분의 대학 풋볼 경기에서 베터들은 모든 종류의 이벤트(_프롭_)에 베팅할 수 있는 기회를 갖게 되었습니다. 누가 첫 터치다운을 기록할 것인가? 패트릭 마홈스는 가로채기를 몇 번이나 당할 것인가? 타이릭 힐은 리셉션을 몇 번이나 기록할 것인가? 여기서 베팅할 수 있는 종류가 워낙 방대하기 때문에 스포츠북이 이 가격들을 각각 정확히 책정하는 것은 매우, 매우 어려우며 이 프롭 시장에는 베터를 위한 더 큰 기회가 존재합니다.

이 장에서는 NFL 쿼터백의 터치다운 패스 시장을 살펴보겠습니다. 일반적으로 선발 출전하는 쿼터백의 프롭 시장은 터치다운 패스 0.5개 초과/미만(over/under), 1.5개 초과/미만(over/under), 그리고 최고의 쿼터백을 위한 2.5개 초과/미만(over/under)을 제공합니다. 여기서 터치다운 패스 횟수를 *인덱스(index)*라고 부릅니다. 쿼터백이 경기에서 던지는 터치다운 패스의 횟수는 매우 이산적(discrete)이기 때문에, 프롭 제공의 중요한 측면은 오버 및 언더에 대한 가격이며, 베팅 시장은 베터의 의견에 반응하여 제공하는 것을 연속체(continuum)로 생성합니다.

따라서 인기 있는 인덱스인 1.5 터치다운 패스 초과/미만(over/under)의 경우 한 선수에 대한 가격이 오버에 -140(100달러를 따기 위해 140달러 걸기)이 될 수 있는 반면 다른 선수는 동일한 터치다운 수 오버에 대한 가격이 +140(140달러를 따기 위해 100달러 베팅)이 될 수 있습니다. 전자는 1.5 터치다운 패스 오버에 대한 *탑독(favorite)*이고 후자는 이에 대한 언더독(underdog)입니다. 이러한 값이 어떻게 결정되는지, 그리고 베팅해야 하는지 여부는 분석(analytics)에 의해 크게 좌우됩니다.

# 푸아송 분포 (The Poisson Distribution)

프로포지션(proposition) 베팅을 생성/베팅하거나 기타 베팅 시장에 참여하려면 이벤트가 발생할 가능성, 즉 *확률(probability)*을 추정할 수 있어야 합니다. 이 장의 표준적인 예에서는 이것이 바로 특정 쿼터백이 특정 경기에서 던진 터치다운 패스 수입니다.

이러한 확률을 지정하는 간단한 방법은 각 결과(0개 터치다운 패스, 1개 터치다운 패스, 2개 터치다운 패스 등)의 발생 빈도(frequencies)를 경험적으로 살펴보는 것입니다. 2016년부터 2022년까지 매주 10회 이상의 패스 플레이를 한 NFL 쿼터백을 살펴보고 다양한 터치다운 패스 결과의 빈도를 확인해 보겠습니다. 우리는 10회 패스 플레이 기준을 해당 팀의 선발 선수임을 나타내는 척도로 사용합니다. 완벽하지는 않지만 현재로서는 충분할 것입니다. 일반적으로 패싱 터치다운 프롭은 특정 경기의 선발 선수에게만 제공됩니다. 패스 플레이가 아닌 플레이를 제거하기 위해 <a href="ch05.html#sec-lr-pass" data-type="xref">5장</a>과 동일한 필터를 사용할 것입니다. 먼저 Python에서 데이터를 로드합니다.

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

seasons = range(2016, 2022 + 1)
pbp_py =\
    nfl.import_pbp_data(seasons)

pbp_py_pass = \
  pbp_py.\
    query('passer_id.notnull()')\
    .reset_index()
```

또는 R에서 데이터를 로드합니다.

```
## R
library(nflfastR)
library(tidyverse)

pbp_r <-
    load_pbp(2016:2022)

pbp_r_pass <-
  pbp_r |>
  filter(!is.na(passer_id))
```

그런 다음 `pass_touchdown`에 대해 `NULL` 또는 `NA` 값을 `0`으로 대체합니다. 또한 Python에서는 데이터가 올바르게 요약되도록 `passer_id` 및 `passer`가 없는 플레이를 `none`으로 설정해야 합니다.

다음으로 `season`, `week`, `passer_id`, `passer`별로 집계하여 주당 패스 수와 주당 터치다운 패스 수를 계산합니다. 그런 다음 필터링을 거쳐 매주 패서로 10번 미만의 플레이를 한 선수를 제외합니다. 다음으로 쿼터백당 주간 터치다운 패스 횟수를 계산합니다.

마지막으로 나중에 사용할 `total_line`을 저장합니다. 이것은 이 장의 앞부분에서 논의했던 총 득점 시장을 가리키는 `nflfastR`의 명칭일 뿐입니다. 우리는 다른 토탈을 가진 경기는 터치다운 패스의 기회도 다를 것이라고 가정합니다(토탈 점수가 높을수록 평균적으로 터치다운 패스가 많아집니다). 게임 전반에 걸쳐 `total_line`은 동일하므로 Python 또는 R이 게임에 대한 값을 집계할 수 있도록 함수를 사용해야 합니다. `mean()` 또는 `max()` 같은 함수는 게임의 값을 제공하며, 우리는 `mean()`을 사용했습니다. Python에서는 다음 코드를 사용합니다.

```
## Python
pbp_py_pass\
    .loc[pbp_py_pass.pass_touchdown.isnull(), "pass_touchdown"] = 0

pbp_py_pass\
    .loc[pbp_py_pass.passer.isnull(), "passer"] = 'none'

pbp_py_pass\
    .loc[pbp_py_pass.passer_id.isnull(), "passer_id"] = 'none'

pbp_py_pass_td_y = \
    pbp_py_pass\
    .groupby(["season", "week", "passer_id", "passer"])\
    .agg({"pass_touchdown": ["sum"],
          "total_line": ["count", "mean"]})

pbp_py_pass_td_y.columns =\
    list(map("_".join, pbp_py_pass_td_y.columns))

pbp_py_pass_td_y.reset_index(inplace=True)

pbp_py_pass_td_y\
    .rename(columns={
        "pass_touchdown_sum": "pass_td_y",
        "total_line_mean": "total_line",
        "total_line_count": "n_passes"
    },
    inplace=True
)

pbp_py_pass_td_y =\
    pbp_py_pass_td_y\
   .query("n_passes >= 10")

pbp_py_pass_td_y\
    .groupby("pass_td_y")\
    .agg({"n_passes": "count"})
```

결과는 다음과 같습니다.

n_passes pass_td_y 0.0 902 1.0 1286 2.0 1050 3.0 506 4.0 186 5.0 31 6.0 4

또는 R에서 이 코드를 사용합니다.

```
## R
pbp_r_pass_td_y <-
    pbp_r_pass |>
    mutate(
        pass_touchdown = ifelse(is.na(pass_touchdown), 0,
                                pass_touchdown)) |>
    group_by(season, week, passer_id, passer) |>
    summarize(
        n_passes = n(),
        pass_td_y = sum(pass_touchdown),
        total_line = mean(total_line)
    ) |>
    filter(n_passes >= 10)

pbp_r_pass_td_y |>
    group_by(pass_td_y) |>
    summarize(n = n())
```

결과는 다음과 같습니다.

\# A tibble: 7 × 2 pass_td_y n \<dbl\> \<int\> 1 0 902 2 1 1286 3 2 1050 4 3 506 5 4 186 6 5 31 7 6 4

###### 팁 (Tip)

팀당 일주일에 한 경기만 있기 때문에 `season` 및 `week`별로 그룹화할 수 있습니다. `passer_id`가 고유하기 때문에 `passer_id` 및 `passer`별로 그룹화합니다(일부 쿼터백은 이름이 같거나 최소한 이름의 첫 글자와 성이 같을 수 있습니다). 데이터를 더 잘 이해하는 데 도움이 되도록 `passer`를 포함시켰습니다. 새 데이터에서 이와 같은 그룹화를 사용할 때는 특정 요구사항에 맞는 고유한 그룹을 만드는 방법을 생각해 보세요.

이제 경험적 분포의 알맹이가 1개의 터치다운 패스를 중심으로 집중되어 있으며 10번 이상의 패스 시도를 한 선수가 터치다운 패스를 하나도 못 던질 확률보다 터치다운 패스를 2개 이상 던질 가능성이 높다는 점을 고려하면 왜 인기 있는 지표가 1.5인지 알 수 있습니다. Python에서 확인할 수 있듯이 분포의 평균은 1.48 터치다운 패스입니다.

```
## Python
pbp_py_pass_td_y\
    .describe()
```

결과는 다음과 같습니다.

season week pass_td_y n_passes total_line count 3965.000000 3965.000000 3965.000000 3965.000000 3965.000000 mean 2019.048928 9.620177 1.469609 38.798487 45.770618 std 2.008968 5.391064 1.164085 10.620958 4.409124 min 2016.000000 1.000000 0.000000 10.000000 32.000000 25% 2017.000000 5.000000 1.000000 32.000000 42.500000 50% 2019.000000 10.000000 1.000000 39.000000 45.500000 75% 2021.000000 14.000000 2.000000 46.000000 48.500000 max 2022.000000 22.000000 6.000000 84.000000 63.500000

R의 경우 다음과 같습니다.

```
pbp_r_pass_td_y |>
    ungroup() |>
    select(-passer, -passer_id) |>
    summary()
```

결과는 다음과 같습니다.

season week n_passes pass_td_y total_line Min. :2016 Min. : 1.00 Min. :10.0 Min. :0.00 Min. :32.00 1st Qu.:2017 1st Qu.: 5.00 1st Qu.:32.0 1st Qu.:1.00 1st Qu.:42.50 Median :2019 Median :10.00 Median :39.0 Median :1.00 Median :45.50 Mean :2019 Mean : 9.62 Mean :38.8 Mean :1.47 Mean :45.77 3rd Qu.:2021 3rd Qu.:14.00 3rd Qu.:46.0 3rd Qu.:2.00 3rd Qu.:48.50 Max. :2022 Max. :22.00 Max. :84.0 Max. :6.00 Max. :63.50

값의 개수는 좋은 출발점이지만 종종 그 이상의 정보가 필요합니다. 일반적으로 추론과 예측을 위해 이러한 카운트에만 의존하는 것은 수많은 문제를 수반합니다. 중요하게 발생하는 문제는 일반화(generalization)의 문제입니다. 바로 여기서 확률 분포(probability distributions)가 진가를 발휘합니다.

터치다운 패스만이 베팅을 원하는 유일한 프롭 시장은 아닙니다. 가로채기(interceptions), 색(sacks) 등 발생 빈도가 낮은 기타 시장은 모두 유사한 정량적 특성을 가질 수 있으며, 도구 상자에 사용 가능한 작은 도구 모음을 갖추는 것이 유용할 것입니다. 또한 패스 야드 같이 10배 많은 이산적 결과가 존재하는 다른 시장은 종종 한 리그의 역사, 더 나아가 선수의 역사에서 실제로 발생했던 결과보다 더 많은 잠재적 결과를 산출할 수 있습니다. 여기에서는 분명히 일반적인 프레임워크가 필요합니다.

바로 여기서 확률 분포가 유용하게 쓰입니다. *확률 분포(probability distribution)*는 각각의 가능한 결과에 0과 1 사이의 값을 할당하는 수학적 객체이며, 이를 *확률(probability)*이라고 합니다. 경기에서의 터치다운 패스와 같은 이산형 결과의 경우 이는 이해하기가 상당히 쉬우며 각 결과에 대해 계산하기 위해 공식을 요구할 수도 있지만 대개 "_X_ = 0일 확률은 얼마인가?"라는 질문에 대한 답을 쉽게 얻을 수 있습니다. 신장(키)처럼 연속적인 결과를 위해서는 미적분학의 도구가 필요하여 작업이 조금 번거롭습니다. 우리는 이 책에서 이산 확률 분포(discrete probability distributions) 사용을 고수할 것입니다.

인기 있는 이산 확률 분포 중 하나는 *푸아송 분포(Poisson distribution)*입니다. 이 분포는 정수(즉, 이산 값) $`x`$($`x = 0,1,2,3,...`$)를 얻을 확률을 $`\frac{e^{\lambda}\lambda^{x}}{x!}`$ 값으로 정의합니다. 이 방정식에서 그리스 문자 $`\lambda`$(람다)는 모집단의 평균 값이고 !는 팩토리얼(factorial) 함수입니다. 푸아송 분포는 고정된 시간 또는 공간 간격 내에서 주어진 횟수의 이벤트가 발생할 가능성을 모델링합니다.

###### 참고 (Note)

팩토리얼의 정의는 _n_! = _n_ × (_n_ − 1) × (_n_ − 2) × (_n_ − 3)…​× 2 × 1 이고 0! = 1 입니다. 또한 순열과 함께 수학 수업에서 사용된 것을 기억할 수도 있습니다. 예를 들어 a, b, c 세 글자를 배열할 수 있는 방법은 몇 가지일까요? 3! = 6 즉, aba, acb, bac, bcb, cab, cba입니다.

푸아송 분포의 핵심 가정은 다음과 같습니다.

- 이벤트는 동일한 확률로 발생합니다.

- 이벤트는 마지막 이벤트가 발생한 이후의 시간과 독립적입니다.

이러한 가정들은 풋볼에서 정확히 충족되지는 않습니다. 한 경기에서 터치다운을 한 번 득점한 팀이 필드 반대편에 있는 수비진을 "파악"했을 수도 있고 그렇지 않을 수도 있기 때문입니다. 하지만 적어도 쿼터백의 한 경기 터치다운 패스를 모델링할 때 고려해 볼 만한 분포입니다.

###### 팁 (Tip)

Python과 R에는 모두 통계 분포 작업을 위한 강력한 도구가 있습니다. 이 책에서는 이러한 주제를 간략하게만 다룹니다. 우리는 벤저민 M. 볼커(Benjamin M. Bolker)의 _Ecological Models and Data in R_ (Princeton University Press, 2008)과 같은 책이 응용 통계 분포와 그 응용 분야에 대한 훌륭한 리소스라는 것을 알게 되었습니다.

푸아송이 합리적인지 확인하려면 빈도수(frequencies)에 대한 막대 그래프를 살펴보고 이를 평균이 $`\lambda`$인 동일한 푸아송 분포와 비교해 보겠습니다. Python 코드를 사용하여 <a href="#fig-py-hist-pois" data-type="xref">그림 6-1</a>을 만들어보세요.

```
## Python
pass_td_y_mean_py =\
    pbp_py_pass_td_y\
    .pass_td_y\
    .mean()

plot_pos_py =\
    pd.DataFrame(
        {"x": range(0, 7),
        "expected": [poisson.pmf(x, pass_td_y_mean_py) for x in range(0, 7)]
        }
    )

sns.histplot(pbp_py_pass_td_y["pass_td_y"], stat="probability");
plt.plot(plot_pos_py.x, plot_pos_py.expected);
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0601.png" />
<h6 id="figure-6-1.-histogram-vertical-bars-of-normalized-observed-touchdowns-per-game-per-quarterback-with-at-least-10-games-plotted-with-seaborn">그림 6-1. 최소 10경기 이상 출전한 쿼터백의 경기당 정규화된 관측 터치다운 히스토그램(수직 막대, <code>seaborn</code>으로 플롯함)</h6>
</figure>

이 히스토그램에서 *정규화(normalized)*라는 용어는 모든 막대의 합이 1이라는 의미입니다. 곡선은 푸아송 분포에 따른 이론적 기대값을 보여줍니다.

또는 다음 R 코드를 사용하여 <a href="#fig-r-hist-pois" data-type="xref">그림 6-2</a>를 만들어보세요.

```
## R
pass_td_y_mean_r <-
    pbp_r_pass_td_y |>
    pull(pass_td_y) |>
    mean()

plot_pos_r <-
    tibble(x = seq(0, 7)) |>
    mutate(expected = dpois(
        x = x,
        lambda = pass_td_y_mean_r
    ))

ggplot() +
    geom_histogram(
        data = pbp_r_pass_td_y,
        aes(
            x = pass_td_y,
            y = after_stat(count / sum(count))
        ),
        binwidth = 0.5
    ) +
    geom_line(
        data = plot_pos_r, aes(x = x, y = expected),
        color = "red", linewidth = 1
    ) +
    theme_bw() +
    xlab("Touchdown passes per player per game for 2016 to 2022") +
    ylab("Probability")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0602.png" />
<h6 id="figure-6-2.-histogram-vertical-bars-of-normalized-observed-touchdowns-per-game-per-quarterback-with-at-least-10-games-plotted-with-ggplot2">그림 6-2. 최소 10경기 이상 출전한 쿼터백의 경기당 정규화된 관측 터치다운 히스토그램(수직 막대, <code>ggplot2</code>로 플롯함)</h6>
</figure>

이 히스토그램에서 *정규화(normalized)*라는 것은 모든 막대의 합이 1이라는 의미입니다. 곡선은 푸아송 분포에 따른 이론적 기대값을 보여줍니다. 푸아송 분포는 터치다운 패스 1개의 가능성을 약간 과대평가하는 것으로 보이며, 이로 인해 경기에서 0개, 2개 또는 그 이상의 터치다운 패스가 발생할 확률을 약간 과소평가합니다.

비록 차이가 크지 않을지라도 스포츠 베팅에서는 이러한 차이가 승패를 가르는 요인이 될 수 있습니다. 따라서 본인의 판단에 따라 베팅하려면 약간의 조정이 필요합니다. 대안으로, 초과 산포(over dispersion)를 설명하는 (음이항 분포(negative binomial distribution)나 유사 푸아송(quasi-Poisson)과 같은) 완전히 다른 분포를 사용할 수도 있습니다. 하지만 이 책에서는 푸아송 분포가 이 장에서 터치다운 패스의 확률에 핸디캡을 부여하기에 충분하다고 가정할 것입니다.

# 개별 선수 시장 및 모델링 (Individual Player Markets and Modeling)

각 쿼터백과 그 선수의 상대는 퀄리티에 따라 차이가 있으므로 각 선수는 경기마다 터치다운 패스에 관한 서로 다른 시장을 갖게 됩니다. 예를 들어 캔자스시티 치프스(Kansas City Chiefs)와 필라델피아 이글스(Philadelphia Eagles) 간의 슈퍼볼 LVII에서 패트릭 마홈스(Patrick Mahomes)의 터치다운 패스 프롭은 2.5개(DraftKings Sportsbook 기준)였으며 오버 가격은 +150(150달러를 따기 위해 100달러 베팅), 언더 가격은 -185(100달러를 따기 위해 185달러 베팅)였습니다.

앞서 논의한 것처럼 이러한 가격은 여러분이 베팅하는 대상의 확률을 반영합니다. 오버의 경우 가격이 > 0이므로 손익분기점 확률을 계산하는 공식은 100 / (100 + 150) = 0.4입니다. 즉, 오버에 베팅하려면 마홈스가 2022 시즌 NFL 1위를 차지했던 이글스 수비진을 상대로 3개 이상의 터치다운 패스를 기록할 확률이 40% 이상이어야 한다는 확신이 있어야 합니다.

언더의 경우 손익분기점 확률은 185 / (100 + 185) = 0.649이므로, 언더에 베팅하려면 마홈스가 2개 이하의 터치다운을 던질 것이라는 확신이 64.9% 이상이어야 합니다. 이러한 확률을 더하면 104.9%가 되며, 여기서 4.9%는 앞에서 설명한 바와 같이 카지노가 수수료(vig)를 부과하여 생성된 카지노의 우위(house edge) 또는 *홀드(hold)*를 나타냅니다.

마홈스의 이러한 확률을 지정하기 위해 마홈스가 출전한 경기의 이력을 살펴보고 2개 이하의 터치다운과 3개 이상의 터치다운이 나온 경기 비율을 비교해 볼 수 있습니다. 그러나 이 방법에는 몇 가지 이유로 결함이 있습니다. 첫 번째는 상대의 강점, 날씨, 조력자의 변화 또는 기타 유사한 요인과 같은 외생적 요인(exogenous factors)을 고려하지 않는다는 것입니다. 또한 관중 소음의 부재가 공격에 큰 도움이 되었던 코로나19 기간 동안 발생한 일과 같이 리그 전반의 환경 변화도 고려하지 않습니다.

이제 다양한 방법을 통해 이를 모델에 통합할 수 있으며, 실제로 베팅을 하는 사람이라면 가능한 한 많은 합리적인 요소들을 고려해야 합니다. 여기서는 베팅 시장에서 득점할 것으로 예상되는 포인트의 수인, 앞서 언급한 경기에 대한 *토탈(total)*을 사용할 것입니다. 이것은 수비력, 속도, 날씨를 하나의 숫자로 함께 고려한 것입니다. 이 숫자와 더불어 해당 쿼터백의 이전 두 시즌 동안 터치다운 패스 평균 횟수를 특성(feature)으로 사용할 것입니다.

###### 팁 (Tip)

`for` 루프는 프로그래밍에서 강력한 도구입니다. `for` 루프, 특히 중첩된 `for` 루프를 구축하기 시작할 때 인덱스만 출력하는 것으로 시작할 것입니다. 예를 들어 Python 코드를 사용하고 각 인덱스에 대한 세부 정보를 채울 것입니다.

```
## Python
for season_idx in range(2017, 2022 + 1):
    print(season_idx)
    for week_idx in range(1, 22 + 1):
        print(week_idx)
```

두 가지 이유로 이렇게 합니다. 첫째, 인덱싱이 작동하는지 확인하기 위해서입니다. 둘째, 이제 메모리에 `season_idx`와 `week_idx`가 저장됩니다. 이 두 인덱스 예제에서 코드가 작동한다면 `for` 루프의 나머지 인덱스 값에 대해서도 코드가 작동할 가능성이 높습니다.

이것이 바로 모델 학습에서 *x*라고 부를 대상입니다. Python에서는 다음 코드를 사용합니다.

```
## Python
# 주당 pass_td_y 가 10 이상인 경우
pbp_py_pass_td_y_geq10 =\
    pbp_py_pass_td_y.query("n_passes >= 10")

# 이전 시즌과 현재 시즌의 현재 경기 전까지 각 QB의 평균 터치다운 패스 가져오기
x_py = pd.DataFrame()
for season_idx in range(2017, 2022 + 1):
    for week_idx in range(1, 22 + 1):
        week_calc_py = (
            pbp_py_pass_td_y_geq10\
                .query("(season == " +
                       str(season_idx - 1) +
                       ") |" +
                       "(season == " +
                       str(season_idx) +
                       "&" +
                       "week < " +
                       str(week_idx) +
                       ")")\
            .groupby(["passer_id", "passer"])\
            .agg({"pass_td_y": ["count", "mean"]})
        )
        week_calc_py.columns =\
            list(map("_".join, week_calc_py.columns))
        week_calc_py.reset_index(inplace=True)
        week_calc_py\
            .rename(columns={
                "pass_td_y_count": "n_games",
                "pass_td_y_mean": "pass_td_rate"},
            inplace=True)
        week_calc_py["season"] = season_idx
        week_calc_py["week"] = week_idx
        x_py = pd.concat([x_py, week_calc_py])
```

###### 경고 (Warning)

중첩된 루프는 컴퓨팅 시간을 빠르게 증가시키고 코드의 가독성을 떨어뜨릴 수 있습니다. 중첩된 루프를 많이 사용하고 있다면 벡터화(vectorization) 같은 다른 코딩 방법론을 배우는 것을 고려해 보세요. 여기서는 루프를 이해하기 더 쉽고 컴퓨터의 성능이 중요하지 않기 때문에 루프를 사용합니다.

또는 R에서는 다음 코드를 사용합니다.

```
## R
# 주당 pass_td_y 가 10 이상인 경우
pbp_r_pass_td_y_geq10 <-
    pbp_r_pass_td_y |>
    filter(n_passes >= 10)

# 이전 시즌과 현재 시즌의 현재 경기 전까지 각 QB의 평균 터치다운 패스 가져오기
x_r <- tibble()

for (season_idx in seq(2017, 2022)) {
    for (week_idx in seq(1, 22)) {
        week_calc_r <-
            pbp_r_pass_td_y_geq10 |>
            filter((season == (season_idx - 1)) |
                (season == season_idx & week < week_idx)) |>
            group_by(passer_id, passer) |>
            summarize(
                n_games = n(),
                pass_td_rate = mean(pass_td_y),
                .groups = "keep"
            ) |>
            mutate(season = season_idx, week = week_idx)

        x_r <- bind_rows(x_r, week_calc_r)
    }
}
```

###### 팁 (Tip)

역사적인 관례에 따라 많은 사람들이 `for i in …​` 과 같이 `for` 루프의 인덱스로 `i`, `j`, `k`를 사용합니다. 리처드(Richard)는 세 가지 이유로 `season_idx` 또는 `week_idx`와 같은 긴 용어를 사용하는 것을 선호합니다. 첫째, 단어가 더 서술적이어서 코드에서 무슨 일이 일어나고 있는지 파악하는 데 도움이 됩니다. 둘째, `Find(찾기)` 도구를 사용하여 단어를 더 쉽게 검색할 수 있습니다. 셋째, 코드의 다른 곳에서 단어가 반복될 가능성이 줄어듭니다.

여기서 매주 출전하는 모든 선수에 대해 평균 터치다운 패스 횟수가 있는 것을 볼 수 있습니다.

###### 팁 (Tip)

이 책에서는 개념적으로 사용하고 이해하기 쉽기 때문에 `for` 루프를 사용합니다. Python의 `map()`이나 R의 `lapply()` 또는 `apply()`와 같은 _apply_ 함수 등 다른 도구들도 존재합니다. 이러한 함수들은 고급 사용자가 이해하고 읽기에 더 빠르고 쉬우며 오류가 발생할 가능성도 적습니다. 하지만 우리는 고급 데이터 과학 프로그래밍이 아닌 입문용 풋볼 분석(football analytics)에 대한 책을 집필했습니다. 따라서 이 책에서는 주로 `for` 루프를 고수합니다. 이러한 메서드가 무엇이고 왜 사용해야 하는지에 대한 설명은 해들리 위컴(Hadley Wickham)의 _Advanced R_ 2판(CRC Press, 2019)에 있는 [9장](https://adv-r.hadley.nz/functionals.html)과 같은 리소스를 참조하세요.

Python으로 슈퍼볼 LVII에 출전하는 패트릭 마홈스(Patrick Mahomes)를 살펴보겠습니다.

```
## Python
x_py.query('passer == "P.Mahomes"').tail()
```

결과는 다음과 같습니다.

passer_id passer n_games pass_td_rate season week 39 00-0033873 P.Mahomes 36 2.444444 2022 18 40 00-0033873 P.Mahomes 37 2.405405 2022 19 40 00-0033873 P.Mahomes 37 2.405405 2022 20 40 00-0033873 P.Mahomes 38 2.394737 2022 21 40 00-0033873 P.Mahomes 39 2.384615 2022 22

또는 R에서 살펴보겠습니다.

```
## R
x_r |>
    filter(passer == "P.Mahomes") |>
    tail()
```

결과는 다음과 같습니다.

\# A tibble: 6 × 6 passer_id passer n_games pass_td_rate season week \<chr\> \<chr\> \<int\> \<dbl\> \<int\> \<int\> 1 00-0033873 P.Mahomes 35 2.43 2022 17 2 00-0033873 P.Mahomes 36 2.44 2022 18 3 00-0033873 P.Mahomes 37 2.41 2022 19 4 00-0033873 P.Mahomes 37 2.41 2022 20 5 00-0033873 P.Mahomes 38 2.39 2022 21 6 00-0033873 P.Mahomes 39 2.38 2022 22

카지노(books)가 꽤 괜찮은 숫자를 설정한 것 같습니다. 2021년과 2022년 22주차까지의 데이터를 사용한 해당 경기 이전 마홈스의 평균은 경기당 2.38 터치다운 패스였습니다. 이제 추가된 경기 토탈과 함께 이전 코드로 만든 데이터프레임 `pbp_pass_td_y_geq10`인 종속 변수(response variable)를 생성해야 합니다. Python에서는 `merge()` 함수를 사용합니다.

```
## Python
pbp_py_pass_td_y_geq10 =\
    pbp_py_pass_td_y_geq10.query("season != 2016")\
    .merge(x_py,
           on=["season", "week", "passer_id", "passer"],
           how="inner")
```

R에서는 `inner_join()` 함수를 사용합니다(조인에 대한 자세한 내용은 <a href="app03.html#sec-app-dw" data-type="xref">부록 C</a> 참조).

```
### R
pbp_r_pass_td_y_geq10 <-
    pbp_r_pass_td_y_geq10 |>
    inner_join(x_r,
        by = c(
            "season", "week",
            "passer_id", "passer"
        )
    )
```

이제 데이터 세트를 병합하여 모델에 대한 학습 데이터 세트를 얻었습니다. 데이터를 모델링하기 전에 R에서 `ggplot2`를 사용하여 빠르게 살펴보겠습니다. 먼저 `passer_id` 열을 사용하여( `passer` 열 대신) 각 패서의 매 경기 패스 터치다운을 선으로 플롯합니다. 플롯에서 시즌별로 `facet`을 지정하고 의미 있는 캡션을 추가합니다. 이것을 `_weekly_passing_id_r_plot_`으로 저장하고 <a href="#fig-tl_r" data-type="xref">그림 6-3</a>을 살펴봅니다.

```
## R
weekly_passing_id_r_plot <-
    pbp_r_pass_td_y_geq10 |>
    ggplot(aes(x = week, y = pass_td_y, group = passer_id)) +
    geom_line(alpha = 0.25) +
    facet_wrap(vars(season), nrow = 3) +
    theme_bw() +
    theme(strip.background = element_blank()) +
    ylab("Total passing touchdowns") +
    xlab("Week of season")
weekly_passing_id_r_plot
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0603.png" />
<h6 id="figure-6-3.-weekly-passing-touchdowns-throughout-the-2017-to-2022-seasons.-each-line-corresponds-to-an-individual-passer.">그림 6-3. 2017년부터 2022년 시즌까지 주간 패싱 터치다운. 각 선은 개별 패서를 나타냅니다.</h6>
</figure>

<a href="#fig-tl_r" data-type="xref">그림 6-3</a>은 경기당 패스 터치다운의 변동성을 보여줍니다. 값은 시간에 따라 일정한 것으로 보이며 뚜렷한 추세는 나타나지 않습니다. 플롯에 푸아송 회귀 추세선을 추가하여 <a href="#fig-tl_trend_r" data-type="xref">그림 6-4</a>를 만듭니다.

```
## R
weekly_passing_id_r_plot +
    geom_smooth(method = 'glm', method.args = list("family" = "poisson"),
                se=FALSE,
                linewidth = 0.5, color = 'blue',
                alpha = 0.25)
```

언뜻 보기에 <a href="#fig-tl_trend_r" data-type="xref">그림 6-4</a>에서는 뚜렷한 추세가 나타나지 않습니다. 선수들은 보통 시즌 내내 경기당 평균 예상 패싱 터치다운이 일정하게 유지되지만 주마다 상당한 변동이 있습니다. 다음으로 모델을 사용하여 이 데이터를 조사해 보겠습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0604.png" />
<h6 id="figure-6-4.-weekly-passing-touchdowns-throughout-the-2017-to-2022-seasons.-each-line-corresponds-to-an-individual-passer.-the-trendline-is-from-a-poisson-regression.">그림 6-4. 2017년부터 2022년 시즌까지 주간 패싱 터치다운. 각 선은 개별 패서를 나타냅니다. 추세선은 푸아송 회귀 분석에서 파생되었습니다.</h6>
</figure>

###### 참고 (Note)

그림 <a href="#fig-tl_r" data-type="xref" data-xrefstyle="select:labelnumber">6-3</a>과 <a href="#fig-tl_trend_r" data-type="xref" data-xrefstyle="select:labelnumber">6-4</a>는 많은 통찰력을 제공하지는 않습니다. 탐색적 데이터 분석(EDA)을 사용하여 데이터를 확인하는 과정을 보여드리기 위해 이 책에 포함했습니다. 아마도 고객이 시간에 따른 추세가 존재하는지 구체적으로 묻거나, 긴 기술 보고서 또는 과제를 작성하는 경우가 아니라면 이러한 수치는 커뮤니케이션에 사용되지 않을 것입니다.

선수가 경기에서 던진 터치다운 패스 횟수에 대해 푸아송 분포를 가정하고 있으므로 모델로 *푸아송 회귀(Poisson regression)*를 사용합니다. 푸아송 회귀를 적합시키는 코드는 <a href="ch05.html#sec-lr-pass" data-type="xref">5장</a>의 로지스틱 회귀와 유사합니다. 그러나 이제 `family`는 binomial(이항)이 아닌 Poisson입니다. 푸아송 회귀를 수행하려면 Poisson이 필요합니다. Python에서 다음 코드를 사용하여 모델을 적합시키고 출력을 데이터의 `exp_pass_td` 열에 저장한 다음 요약을 살펴봅니다.

```
## Python
pass_fit_py = \
    smf.glm(
        formula="pass_td_y ~ pass_td_rate + total_line",
        data=pbp_py_pass_td_y_geq10,
        family=sm.families.Poisson())\
    .fit()

pbp_py_pass_td_y_geq10["exp_pass_td"] = \
    pass_fit_py\
    .predict()

print(pass_fit_py.summary())
```

결과는 다음과 같습니다.

Generalized Linear Model Regression Results ============================================================================== Dep. Variable: pass_td_y No. Observations: 3297 Model: GLM Df Residuals: 3294 Model Family: Poisson Df Model: 2 Link Function: Log Scale: 1.0000 Method: IRLS Log-Likelihood: -4873.8 Date: Sun, 04 Jun 2023 Deviance: 3395.2 Time: 09:41:29 Pearson chi2: 2.83e+03 No. Iterations: 5 Pseudo R-squ. (CS): 0.07146 Covariance Type: nonrobust ================================================================================ coef std err z P\>\|z\| \[0.025 0.975\] -------------------------------------------------------------------------------- Intercept -0.9851 0.148 -6.641 0.000 -1.276 -0.694 pass_td_rate 0.3066 0.029 10.706 0.000 0.251 0.363 total_line 0.0196 0.003 5.660 0.000 0.013 0.026 ================================================================================

마찬가지로 R에서는 다음 코드를 사용하여 모델을 적합시키고 출력을 데이터의 `exp_pass_td` 열에 저장한 후(계수/모델 척도가 아닌 데이터 척도에 출력을 배치하려면 `type = "response"`를 사용해야 함) 요약을 살펴봅니다.

```
## R
pass_fit_r <-
    glm(pass_td_y ~ pass_td_rate + total_line,
        data = pbp_r_pass_td_y_geq10,
        family = "poisson"
    )

pbp_r_pass_td_y_geq10 <-
    pbp_r_pass_td_y_geq10 |>
    ungroup() |>
    mutate(exp_pass_td = predict(pass_fit_r, type = "response"))

summary(pass_fit_r) |>
    print()
```

결과는 다음과 같습니다.

Call: glm(formula = pass_td_y ~ pass_td_rate + total_line, family = "poisson", data = pbp_r_pass_td_y_geq10) Coefficients: Estimate Std. Error z value Pr(\>\|z\|) (Intercept) -0.985076 0.148333 -6.641 3.12e-11 \*\*\* pass_td_rate 0.306646 0.028643 10.706 \< 2e-16 \*\*\* total_line 0.019598 0.003463 5.660 1.52e-08 \*\*\* --- Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1 (Dispersion parameter for poisson family taken to be 1) Null deviance: 3639.6 on 3296 degrees of freedom Residual deviance: 3395.2 on 3294 degrees of freedom AIC: 9753.5 Number of Fisher Scoring iterations: 5

###### 경고 (Warning)

푸아송 회귀의 계수 및 예측값은 <a href="ch05.html#sec-odd-ratios" data-type="xref">"승산비에 대한 간단한 입문서(A Brief Primer on Odds Ratios)"</a>의 로지스틱 회귀처럼 수학적 연결 함수(link function)의 척도(scale)에 따라 달라집니다. <a href="#sec-pos-coef" data-type="xref">"푸아송 회귀 계수(Poisson Regression Coefficients)"</a>에서는 푸아송 회귀의 출력값에 대해 간략하게 설명합니다.

계수를 살펴보고 여기에서 해석해 보겠습니다. 푸아송 회귀의 경우 계수는 지수 척도에 있습니다(자세한 내용은 이 주제에 대한 이전 경고 참조). Python에서는 모델의 파라미터에 접근한 다음 NumPy 라이브러리(`np.exp()`)를 사용하여 지수를 취합니다.

```
## Python
np.exp(pass_fit_py.params)
```

결과는 다음과 같습니다.

Intercept 0.373411 pass_td_rate 1.358860 total_line 1.019791 dtype: float64

R에서는 `tidy()` 함수를 사용하여 계수를 살펴봅니다.

```
## R
library(broom)
tidy(pass_fit_r, exponentiate = TRUE, conf.int = TRUE)
```

결과는 다음과 같습니다.

\# A tibble: 3 × 7 term estimate std.error statistic p.value conf.low conf.high \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 0.373 0.148 -6.64 3.12e-11 0.279 0.499 2 pass_td_rate 1.36 0.0286 10.7 9.55e-27 1.28 1.44 3 total_line 1.02 0.00346 5.66 1.52e- 8 1.01 1.03

먼저 `pass_td_rate` 계수를 살펴보세요. 이 계수의 경우 선수 이력의 추가 터치다운 패스마다 1.36을 곱하여 예상 터치다운 패스 수를 구합니다. 두 번째로 `total_line` 계수를 살펴보세요. 이 계수의 경우 총 라인에 1.02를 곱합니다. 이 경우 총 라인은 꽤 효율적이며 기대값의 2% 이내입니다(1 – 1.02 = -0.02). 두 계수 모두 (가산적) 모델 척도에서 0과 통계적으로 다르거나 (승수적) 데이터 척도에서 1과 통계적으로 다릅니다.

이제 Python에서 현재 결과를 생략하고(공간을 절약하기 위해 `passer_id`도 생략) 슈퍼볼 LVII의 마홈스 데이터를 살펴보겠습니다.

```
## Python
# 공간을 위해 필터 기준을 자체 줄에 지정
filter_by = 'passer == "P.Mahomes" & season == 2022 & week == 22'
# 공간을 위해 열을 자체 줄에 지정
cols_look = [
    "season",
    "week",
    "passer",
    "total_line",
    "n_games",
    "pass_td_rate",
    "exp_pass_td",
]

pbp_py_pass_td_y_geq10.query(filter_by)[cols_look]
```

결과는 다음과 같습니다.

season week passer total_line n_games pass_td_rate exp_pass_td 3295 2022 22 P.Mahomes 51.0 39 2.384615 2.107833

또는 R에서 데이터를 살펴보겠습니다.

```
## R
pbp_r_pass_td_y_geq10 |>
    filter(passer == "P.Mahomes",
           season == 2022, week == 22) |>
    select(-pass_td_y, -n_passes, -passer_id, - week, -season, -n_games)
```

결과는 다음과 같습니다.

\# A tibble: 1 × 4 passer total_line pass_td_rate exp_pass_td \<chr\> \<dbl\> \<dbl\> \<dbl\> 1 P.Mahomes 51 2.38 2.11

자, 이 숫자들은 무엇을 의미할까요?

- `n_games`는 우리가 고려하고 있는 마홈스 샘플에 총 39경기(2022 시즌 21경기, 이전 시즌 18경기)가 있음을 보여줍니다.

- `pass_td_rate`는 우리가 고려하고 있는 샘플에서 마홈스의 경기당 평균 터치다운 패스 횟수입니다.

- `exp_pass_td`는 모델에서 예측한 슈퍼볼에서의 마홈스의 예상 터치다운 패스 횟수입니다.

그리고 이 숫자들은 무엇을 의미할까요? 이 수치들은 마홈스의 경기 총점이 51점으로 비교적 높음에도 불구하고 이전 평균보다 밑돌 것으로 예상된다는 것을 보여줍니다. 예상 _터치다운 패스(touchdown passes)_ 개념의 일부는 _평균으로의 회귀(regression toward the mean)_ 개념에서 기인했을 가능성이 높습니다. 사람들이 평균보다 높을 때 통계 모델은 평균에 더 가까워지도록 감소할 것으로 예상합니다(반대의 경우도 마찬가지입니다. 평균보다 낮은 선수는 평균에 더 가까워지도록 증가할 것으로 예상됩니다). 마홈스는 현시점에서 게임 내 최고의 쿼터백이기 때문에 모델은 그가 증가하기보다는 감소할 가능성이 더 높다고 예측합니다. *평균으로의 회귀*에 대한 자세한 논의는 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>을 참조하세요.

평균 수치 자체는 사실 많은 것을 알려주지 않습니다. 2.5 미만이지만 베팅 시장은 이미 2.5 언더를 가능성 있는(favorite) 결과로 만들었습니다. 여러분이 질문해야 할 것은 "탑독(favorite) 배당이 너무 높은가, 아니면 너무 낮은가?"입니다. 이를 수행하기 위해 `exp_pass_td`를 마홈스의 $`\lambda`$ 값으로 사용하고 확률 질량 함수(PMF) 및 누적 분포 함수(CDF)를 사용하여 명시적으로 이러한 확률 계산할 수 있습니다.

*확률 질량 함수(probability mass function, PMF)*는 통계적 분포를 가정할 때 이산형 이벤트가 발생할 확률을 제공합니다. 우리의 예에서 이는 패서가 터치다운 패스 1개 또는 3개 등 경기당 단일 개수의 터치다운 패스를 완료할 확률입니다. *누적 분포 함수(cumulative density function, CDF)*는 통계적 분포를 가정할 때 여러 이벤트가 발생할 확률의 합을 제공합니다. 우리의 예에서 이는 *X*개 이하의 터치다운 패스를 완료할 확률입니다. 예를 들어 _X_ = 2를 사용하면 특정 주에 터치다운 패스를 0개, 1개, 2개 완료할 확률입니다.

Python은 PMF와 CDF에 대해 비교적 직관적인 이름을 사용합니다. 단순히 분포에 이름을 추가하기만 하면 됩니다. Python에서 쿼터백이 한 경기에서 0개, 1개 또는 2개의 터치다운 패스를 기록할 확률을 확인하려면 `poisson.pmf()`를 사용합니다. 쿼터백이 한 경기에서 2개를 초과하는 터치다운 패스를 기록할 확률을 계산하려면 `1 - poisson.cdf()`를 사용합니다.

```
## Python
pbp_py_pass_td_y_geq10["p_0_td"] =     poisson.pmf(k=0,
                mu=pbp_py_pass_td_y_geq10["exp_pass_td"])

pbp_py_pass_td_y_geq10["p_1_td"] =     poisson.pmf(k=1,
                mu=pbp_py_pass_td_y_geq10["exp_pass_td"])

pbp_py_pass_td_y_geq10["p_2_td"] =     poisson.pmf(k=2,
                mu=pbp_py_pass_td_y_geq10["exp_pass_td"])

pbp_py_pass_td_y_geq10["p_g2_td"] =     1 - poisson.cdf(k=2,
                    mu=pbp_py_pass_td_y_geq10["exp_pass_td"])
```

Python에서 "빅 게임"(또는 풋볼에 익숙하지 않은 독자를 위한 슈퍼볼)에 출전하는 마홈스에 대한 출력을 살펴보겠습니다.

```
## Python
# 공간을 위해 필터 기준을 자체 줄에 지정
filter_by = 'passer == "P.Mahomes" & season == 2022 & week == 22'

# 공간을 위해 열을 자체 줄에 지정
cols_look = [
    "passer",
    "total_line",
    "n_games",
    "pass_td_rate",
    "exp_pass_td",
    "p_0_td",
    "p_1_td",
    "p_2_td",
    "p_g2_td",
]

pbp_py_pass_td_y_geq10    .query(filter_by)[cols_look]
```

결과는 다음과 같습니다.

passer total_line n_games ... p_1_td p_2_td p_g2_td 3295 P.Mahomes 51.0 39 ... 0.256104 0.269912 0.352483 \[1 rows x 9 columns\]

R은 통계 분포에 더 혼란스러운 이름을 사용합니다. `dpois()` 함수는 PMF를 제공하며, `d`는 연속 분포(정규 분포)가 질량(mass) 대신 밀도(density)를 갖기 때문에 밀도(_density_)에서 유래했습니다. `ppois()` 함수는 CDF를 제공합니다. R에서 쿼터백이 한 경기에서 0개, 1개 또는 2개의 터치다운 패스를 기록할 확률을 계산하려면 `dpois()` 함수를 사용합니다. 쿼터백이 한 경기에서 2개를 초과하는 터치다운 패스를 기록할 확률을 계산하려면 `ppois()` 함수를 사용합니다.

```
## R
pbp_r_pass_td_y_geq10 <-
    pbp_r_pass_td_y_geq10 |>
    mutate(
        p_0_td = dpois(x = 0,
                       lambda = exp_pass_td),
        p_1_td = dpois(x = 1,
                       lambda = exp_pass_td),
        p_2_td = dpois(x = 2,
                       lambda = exp_pass_td),
        p_g2_td = ppois(q = 2,
                        lambda = exp_pass_td,
                        lower.tail = FALSE)
    )
```

그런 다음 R에서 빅 게임에 출전하는 마홈스에 대한 출력 결과를 살펴보겠습니다.

```
## R
pbp_r_pass_td_y_geq10 |>
    filter(passer == "P.Mahomes", season == 2022, week == 22) |>
    select(-pass_td_y, -n_games, -n_passes,
           -passer_id, -week, -season)
```

결과는 다음과 같습니다.

\# A tibble: 1 × 8 passer total_line pass_td_rate exp_pass_td p_0_td p_1_td p_2_td p_g2_td \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 P.Mahomes 51 2.38 2.11 0.122 0.256 0.270 0.352

###### 팁 (Tip)

이러한 예제에서는 마홈스의 $`\lambda`$ 값을 하드코딩하는 대신 해당 값을 직접 사용합니다. 여기에는 몇 가지 이유가 있습니다. 첫째, 출처를 알 수 없는 *신비한 숫자(magical mystery number)*가 되는 대신 숫자가 어디서 왔는지 알 수 있습니다. 둘째, 데이터가 변경되면 숫자가 자체적으로 업데이트될 수 있습니다. 이러한 추상화는 우리가 코드를 재사용해야 하거나 원할 때 코드를 다른 상황에 더 쉽게 일반화할 수 있도록 합니다. 이 예제에서 볼 수 있듯이 Python( `numpy` 사용)과 R에서의 벡터화(vectorization)가 가진 아름다움 덕분에 단일 숫자를 계산할 때도 종종 동일한 양의 코드가 필요할 수 있습니다. *벡터화(vectorization)*에서는 컴퓨터 언어가 단일 값(스칼라)이 아닌 벡터(열)에 함수를 적용합니다.

좋습니다. 마홈스가 터치다운 패스를 3개 이상 던질 확률을 35.2%로 추정했으며, 이는 오버에 베팅하는 데 필요한 40% 미만입니다.<sup><a href="ch06.html#id759" id="id759-marker" data-type="noteref">1</a></sup> 마홈스가 2개 이하의 터치다운 패스를 던질 확률 64.8%는 언더에 베팅하는 데 필요한 64.9%보다 약간 낮습니다.

예리한 독자라면 이것이 바로 대부분의 스포츠 베터들이 장기적으로 이기지 못하는 이유라는 것을 눈치챘을 것입니다. 적어도 DraftKings Sportsbook의 경우, 수학적으로는 베팅을 하지 말거나 다른 스포츠북을 찾으라고 말하고 있습니다! 미국의 또 다른 주요 스포츠북 중 하나인 FanDuel은 인덱스가 1.5 터치다운으로 달랐으며 오버(손익분기점 67.2%)에 –205를, 언더(37.9%)에 +164를 제공했습니다. 이는 1개 이하의 터치다운 패스 확률을 37.8%, 2개 이상 확률을 62.2%로 만들었으며 역시 언더의 확률이 베팅의 손익분기점 확률보다 약간 낮았으므로 어떠한 가치도 제공하지 못했습니다.

관계자들에게는 다행스럽게도 마홈스는 MVP로 가는 과정에서 1.5개와 2.5개 터치다운 패스를 모두 오버(초과)했습니다. 따라서 이 이야기의 교훈은 일반적으로 "몇 가지 예외적인 경우를 제외하고는 베팅하지 말라"는 것입니다.

# 푸아송 회귀 계수 (Poisson Regression Coefficients)

<a href="ch05.html#sec-odd-ratios" data-type="xref">"승산비에 대한 간단한 입문서(A Brief Primer on Odds Ratios)"</a>에서 다룬 로지스틱 회귀와 유사하게 GLM의 계수는 연결 함수(link function)에 따라 달라집니다. Python과 R에서 기본적으로 푸아송 회귀는 데이터와 동일한 척도를 유지하기 위해 지수 함수를 적용해야 합니다. 그러나 이로 인해 연결 척도(link scale)의 계수가 (선형 회귀 계수와 같은) 가산적(additive) 특성에서 (로지스틱 회귀 계수와 같은) 승수적(multiplicative) 특성으로 변경됩니다.

이 속성을 입증하기 위해 먼저 시뮬레이션을 사용하여 푸아송 회귀에 대한 이해를 돕겠습니다. 먼저 평균이 1인 푸아송 분포에서 10개의 *표본(draws)*을 고려하고 이것을 객체 `x`에 저장합니다. 그런 다음 `print()`를 사용하여 `x`의 값과 평균을 살펴봅니다.

###### 참고 (Note)

두 가지 언어를 모두 사용하는 분들의 경우, Python과 R 예제는 대개 서로 다른 난수(우연히 같은 값일 경우를 제외하고)를 생성합니다. 두 언어가 서로 다른 난수 생성기를 사용하기 때문이며 두 언어가 어떤 식으로든 똑같은 난수 생성기 함수를 사용하더라도 난수를 생성하는 함수 호출 방식은 다를 것입니다.

Python에서는 이 코드를 사용하여 난수를 생성합니다.

```
## Python
from scipy.stats import poisson

x = poisson.rvs(mu=1, size=10)
```

그리고 숫자를 출력합니다.

```
## Python
print(x)
```

결과는 다음과 같습니다.

\[1 1 1 2 1 1 1 4 1 1\]

그리고 평균을 출력합니다.

```
## Python
print(x.mean())
```

결과는 다음과 같습니다.

1.4

R에서는 이 코드를 사용하여 숫자를 생성합니다.

```
## R
x <- rpois(n = 10, lambda = 1)
```

그리고 숫자를 출력합니다.

```
## R
print(x)
```

결과는 다음과 같습니다.

\[1\] 2 0 6 1 1 1 3 0 1 1

그리고 그들의 평균을 살펴봅니다.

```
## R
print(mean(x))
```

결과는 다음과 같습니다.

\[1\] 1.6

다음으로 전역 절편(global intercept)을 포함한 GLM을 적합시키고 모델 척도와 지수 척도의 계수를 살펴봅니다. Python에서는 이 코드를 사용합니다.

```
# Python
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd

# glm을 위한 데이터프레임 생성
df_py = pd.DataFrame({"x": x})

# GLM 적합
glm_out_py =     smf.glm(formula="x ~ 1", data=df_py, family=sm.families.Poisson()).fit()

# 모델 척도의 출력 살펴보기
print(glm_out_py.params)

# 지수 척도의 출력 살펴보기
```

결과는 다음과 같습니다.

Intercept 0.336472 dtype: float64

```nb
print(np.exp(glm_out_py.params))
```

결과는 다음과 같습니다.

Intercept 1.4 dtype: float64

R에서는 이 코드를 사용합니다.

```
## R
library(broom)

# GLM 적합
glm_out_r <-
    glm(x ~ 1, family = "poisson")

# 모델 척도의 출력 살펴보기
print(tidy(glm_out_r))
```

결과는 다음과 같습니다.

\# A tibble: 1 × 5 term estimate std.error statistic p.value \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 0.470 0.250 1.88 0.0601

```
# 지수 척도의 출력 살펴보기
print(tidy(glm_out_r, exponentiate = TRUE))
```

결과는 다음과 같습니다.

\# A tibble: 1 × 5 term estimate std.error statistic p.value \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 1.60 0.250 1.88 0.0601

지수 척도( `exponentiate = TRUE`를 사용한 두 번째 출력 테이블)의 계수가 시뮬레이션 데이터의 평균과 같다는 점에 유의하세요. 그러나 기울기와 절편처럼 두 개의 계수가 있는 경우는 어떨까요?

구체적인 예로 볼티모어 레이븐스(Baltimore Ravens)의 경기당 터치다운 수를 살펴보겠습니다. 안타깝게도 이번 시즌 레이븐스의 쿼터백은 13번째 경기에서 부상을 당했습니다. 따라서 시즌이 진행됨에 따라 레이븐스의 패스 횟수가 감소할 것으로 합리적으로 예상할 수 있습니다. 이를 테스트하는 공식적인 방법은 "경기당 평균(또는 예상) 터치다운 횟수가 시즌이 진행됨에 따라 변하는가?"라는 질문을 한 다음 푸아송 회귀를 사용하여 이를 통계적으로 평가하는 것입니다.

이를 테스트하기 위해 먼저 데이터를 랭글링한 다음 데이터가 어떻게 돌아가는지 확인하는 데 도움이 되도록 데이터를 플롯합니다. 주(week)에서 1을 빼서 주(week)를 이동시킵니다. 이렇게 하면 1주차가 모델의 절편이 됩니다. 또한 축 눈금을 설정하고 축 레이블도 변경합니다. Python에서 이 코드를 사용하여 <a href="#fig-py-td-per-game" data-type="xref">그림 6-5</a>를 만듭니다.

```
## Python
# 데이터 부분 집합(subset) 만들기
bal_td_py = (
    pbp_py
    .query('posteam=="BAL" & season == 2022')
    .groupby(["game_id", "week"])
    .agg({"touchdown": ["sum"]})
)

# 열 형식 재구성
bal_td_py.columns = list(map("_".join, bal_td_py.columns))
bal_td_py.reset_index(inplace=True)

# 절편 0 = 1주차가 되도록 주(week)를 이동
bal_td_py["week"] = bal_td_py["week"] - 1

# 플롯을 위한 주 목록 만들기
weeks_plot = np.linspace(start=0, stop=18, num=10)
weeks_plot

# 데이터 플롯하기
```

결과는 다음과 같습니다.

array(\[ 0., 2., 4., 6., 8., 10., 12., 14., 16., 18.\])

```
ax = sns.regplot(data=bal_td_py, x="week", y="touchdown_sum");
ax.set_xticks(ticks = weeks_plot, labels = weeks_plot);
plt.xlabel("Week")
plt.ylabel("Touchdowns per game")

plt.show();
```

<figure>
<img src="D:\sd\Practicesny2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0605.png" />
<h6 id="figure-6-5.-ravens-touchdowns-per-game-during-the-2022-season-plotted-with-seaborn.-notice-the-use-of-a-linear-regression-trendline.">그림 6-5. 2022 시즌 레이븐스의 경기당 터치다운 (<code>seaborn</code>으로 플롯함). 선형 회귀 추세선을 사용한 점을 눈여겨보세요.</h6>
</figure>

###### 팁 (Tip)

대부분의 Python 플로팅 도구는 `matplotlib` 패키지의 래퍼(wrapper)이며 `seaborn`도 예외는 아닙니다. 따라서 `seaborn`을 사용자 정의하려면 대개 `matplotlib` 명령을 사용해야 하므로 Python 플로팅의 전문가가 되려면 번거로운 `matplotlib` 명령에 익숙해져야 할 것입니다.

R에서는 이 코드를 사용하여 <a href="#fig-r-td-per-game" data-type="xref">그림 6-6</a>을 만듭니다.

```
## r
bal_td_r <-
    pbp_r |>
    filter(posteam == "BAL" & season == 2022) |>
    group_by(game_id, week) |>
    summarize(
        td_per_game =
            sum(touchdown, na.rm = TRUE),
        .groups = "drop"
    ) |>
    mutate(week = week - 1)

ggplot(bal_td_r, aes(x = week, y = td_per_game)) +
    geom_point() +
    theme_bw() +
    stat_smooth(
        method = "glm", formula = "y ~ x",
        method.args = list(family = "poisson")
    ) +
    xlab("Week") +
    ylab("Touchdowns per game") +
    scale_y_continuous(breaks = seq(0, 6)) +
    scale_x_continuous(breaks = seq(1, 20, by = 2))
```

<figure>
<img src="D:\sd\Practicesny2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0606.png" />
<h6 id="figure-6-6.-ravens-touchdowns-per-game-during-the-2022-season-plotted-with-ggplot2.-notice-the-use-of-a-poisson-regression-trend-line.">그림 6-6. 2022 시즌 레이븐스의 경기당 터치다운 (<code>ggplot2</code>로 플롯함). 푸아송 회귀 추세선을 사용한 점을 눈여겨보세요.</h6>
</figure>

그림 <a href="#fig-py-td-per-game" data-type="xref" data-xrefstyle="select:labelnumber">6-5</a>와 <a href="#fig-r-td-per-game" data-type="xref" data-xrefstyle="select:labelnumber">6-6</a>을 비교할 때, `seaborn`은 선형 모델만 허용하는 반면 `ggplot2`는 추세선에 푸아송 회귀를 사용할 수 있게 해준다는 점을 유의하세요. 두 그림 모두 평균적으로 볼티모어의 경기당 터치다운 횟수가 감소하고 있음을 보여줍니다. 이제 모델을 만들고 계수를 살펴보겠습니다. 다음은 Python 코드입니다.

```
## Python
glm_bal_td_py =     smf.glm(formula="touchdown_sum ~ week",
            data=bal_td_py,
            family=sm.families.Poisson())    .fit()
```

그런 다음 연결(또는 로그) 척도에서 계수를 살펴봅니다.

```
## Python
print(glm_bal_td_py.params)
```

결과는 다음과 같습니다.

Intercept 1.253350 week -0.063162 dtype: float64

그리고 지수(또는 데이터) 척도를 살펴봅니다.

```
## Python
print(np.exp(glm_bal_td_py.params))
```

결과는 다음과 같습니다.

Intercept 3.502055 week 0.938791 dtype: float64

또는 R을 사용하여 모델을 적합시킵니다.

```
## R
glm_bal_td_r <-
    glm(td_per_game ~ week,
        data = bal_td_r,
        family = "poisson"
    )
```

그런 다음 연결(또는 로그) 척도에서 계수를 살펴봅니다.

```
## R
print(tidy(glm_bal_td_r))
```

결과는 다음과 같습니다.

\# A tibble: 2 × 5 term estimate std.error statistic p.value \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 1.25 0.267 4.70 0.00000260 2 week -0.0632 0.0300 -2.10 0.0353

그리고 지수(또는 데이터) 척도를 살펴봅니다.

```
## R
print(tidy(glm_bal_td_r, exponentiate = TRUE))
```

결과는 다음과 같습니다.

\# A tibble: 2 × 5 term estimate std.error statistic p.value \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 3.50 0.267 4.70 0.00000260 2 week 0.939 0.0300 -2.10 0.0353

이제 계수의 의미를 살펴보겠습니다. 먼저 절편 `1.253`과 기울기 `-0.063`이라는 연결 척도(link-scale)의 값을 조사하는 것은 문맥상 많은 정보를 제공하지 않는 것 같습니다. 그러나 기울기 항은 매주의 _위험비(risk ratio)_ 또는 *상대 위험(relative risk)*으로도 알려져 있습니다. 계수를 지수화하면 3.502의 절편 값과 0.939의 기울기 값을 얻게 됩니다.

이 숫자들은 이해하기 쉬운 값을 가지고 있습니다. 절편은 첫 번째 경기 중 예상 패스 수(이것이 1주차가 0이자 절편이 되도록 `week`에서 1을 빼게 한 이유입니다)이며 그 값은 3.502입니다. 그림 <a href="#fig-py-td-per-game" data-type="xref" data-xrefstyle="select:labelnumber">6-5</a>와 <a href="#fig-r-td-per-game" data-type="xref" data-xrefstyle="select:labelnumber">6-6</a>을 보면 0주차는 3, 1주차와 2주차는 5였기 때문에 이는 합리적인 것으로 보입니다.

###### 참고 (Note)

로그 척도(log scale)에서 곱셈을 사용하는 것은 비변환 숫자에서 덧셈을 사용하는 것과 같습니다. 예를 들어 $`x + y = \log\left( e^{x} 	imes e^{y} 
ight)`$입니다. 실수를 사용하여 설명하자면 1 + 3 = 4를 생각해 보세요. 그리고 (반올림 오차 무시) $`e^{1} = 2\!.\! 718`$이고 $`e^{3} = 20\!.\! 09`$입니다. 이 결과들을 사용하면 $`2\!.\! 718 	imes 20\!.\! 09 = 54\!.\! 60`$입니다. 마지막으로 역변환(back transforming)하면 다음을 얻게 됩니다. $`\log\left( 54\!.\! 60 
ight) = 4\!.\! 00`$.

그 다음 주간의 예상 터치다운 횟수는 $`3\!.\! 502 	imes 0\!.\! 939 = 3\!.\! 288`$이 됩니다. 그 다음 주는 $`3\!.\! 502 	imes 0\!.\! 939^{2} = 3\!.\! 088`$입니다. 따라서 볼티모어 레이븐스의 2022 시즌 중 특정 주간의 예상 터치다운 수를 추정하는 방정식은 로그 척도로 $`3\!.\! 502 	imes 0\!.\! 939^{week}`$가 됩니다. 또는 이를 $`e^{1\!.\! 253 + - 0\!.\! 063 	imes {week}}`$로 다시 작성할 수 있습니다.

푸아송 회귀를 설명하기 위해 이 예제를 임의로 선택(cherry-pick)해야 했지만, 이 예제가 그 목적에 부합하여 여러분이 푸아송 회귀 항을 해석하는 방법을 이해하는 데 도움이 되었기를 바랍니다.

# GLM에 대한 마무리 생각 (Closing Thoughts on GLMs)

이 장과 <a href="ch05.html#sec-lr-pass" data-type="xref">5장</a>에서는 GLM과 이러한 모델이 데이터를 이해하고 예측하는 데 어떻게 도움이 될 수 있는지 소개했습니다. 사실, 이 두 장에서 수행된 간단한 예측은 간단한 머신러닝 방법의 기초를 이해하는 데 도움이 될 것입니다. GLM을 사용할 때, 계수가 데이터와 비교하여 다른 척도에서 추정되기 때문에 두 예제 모델 모두 연결 함수(link function)를 다뤄야만 했다는 점에 유의하세요. 또한 이 책의 <a href="ch05.html#sec-lr-pass" data-type="xref">5장</a>에서는 이항(binomial), 이 장에서는 푸아송(Poisson)이라는 두 가지 유형의 오류 계열(error families)을 보았습니다.

이 장의 연습문제에서는 예상보다 많은 혹은 적은 0이 있을 때 산포를 고려하기 위해 유사 푸아송(quasi-Poisson)을 살펴보도록 합니다. 카운트 데이터를 모델링하기 위한 푸아송의 또 다른 대안으로 음이항(negative binomial)을 살펴보는 것도 좋습니다. 다른 두 가지 일반적인 GLM 유형으로는 감마(gamma) 회귀와 대수정규(lognormal) 회귀가 있습니다. 감마 회귀와 대수정규 회귀는 유사합니다. 패스 데이터(분포의 우측에 더 많거나 더 큰 양수 값이 있고 분포의 좌측에 더 작거나 음수 값이 있음)처럼 양수이지만 _오른쪽 꼬리가 긴(right skew)_ 데이터를 분석할 때는 대수정규 분포가 더 적합할 수 있습니다.

선형 모델과 GLM에는 무수히 많은 확장 기능이 존재합니다. 계층적, 다수준 또는 임의 효과(random-effect) 모델을 사용하면 동일한 개인이나 그룹의 반복적인 관찰과 같은 기능으로 불확실성을 모델링할 수 있습니다. 일반화 가산 모델(GAM)을 사용하면 모델을 직선이 아닌 곡선으로 피팅할 수 있습니다. 일반화 가산 혼합 효과 모델(GAMM)은 GAM과 임의 효과 모델을 결합한 것입니다. 이처럼 모델 선택 등의 도구는 어떤 유형의 모델을 사용할지 결정하는 데 도움이 될 수 있습니다.

기본적으로 선형 모델은 통계 및 과학 분야에서 오랜 역사와 활용도를 가지고 있기 때문에 다양한 옵션을 제공합니다. 우리 동료 중에는 회귀의 가정에 대해 몇 시간이고 이야기할 수 있는 (또는 이야기할) 사람도 있습니다. 마찬가지로 우리가 몇 장에 걸쳐 다룬 모델에 대해서도 1년 과정의 대학원 수준 강의가 개설되어 있습니다. 하지만 회귀 모델의 기본을 이해하는 것만으로도 여러분의 풋볼 분석 능력을 크게 향상시키는 데 도움이 될 것입니다.

# 이 장에서 사용된 데이터 과학 도구 (Data Science Tools Used in This Chapter)

이 장에서는 다음 주제를 다루었습니다.

- Python과 R에서 `glm()`을 사용하여 푸아송 회귀 적합시키기

- 상대 위험(relative risk)을 포함한 푸아송 회귀의 계수 이해하고 읽기

- Python의 `merge()` 또는 R의 `inner_join()`을 사용하여 데이터세트 연결하기

- 이전 장에서 배운 데이터 랭글링 도구 재적용하기

- 베팅 확률(betting odds) 알아보기

# 연습문제 (Exercises)

1. 경기의 총점(total)을 포함하지 않으면 터치다운 패스 모델에 어떤 일이 일어납니까? 슈퍼볼에서 패트릭 마홈스의 터치다운 패스에 대한 베팅을 권장할 만큼 확률에 변화가 있습니까?

2. 가로채기 횟수에 대해서도 이 장의 작업을 반복하세요. 오버 가격은 –120이고 언더 가격은 –110이었던 슈퍼볼 LVII에서 마홈스의 0.5가로채기 프롭을 조사하세요.

3. 슈퍼볼 LVII의 또 다른 쿼터백인 제일런 허츠(Jalen Hurts)의 경우는 어떻습니까? 터치다운 프롭은 1.5(오버 -115, 언더 -115)였고 가로채기 프롭은 0.5(+105/–135)였습니다. 두 시장 중 어느 하나에 베팅할 가치가 있었습니까?

4. 터치다운과 가로채기 모델에 추가할 기능이 있는지 `nflfastR` 데이터 세트를 살펴보세요. 경기 전 날씨가 전체 경기 점수에 영향을 미칩니까? 포인트 스프레드의 크기는 어떻습니까?

5. 이 장의 GLM을 반복하되, 푸아송 분포 대신 산포 매개변수를 추정하는 유사 푸아송(quasi-Poisson) 분포를 사용하세요.

6. 이 장의 GLM을 반복하되 푸아송 분포가 아닌 음이항 분포를 사용합니다.

# 추천 도서 (Suggested Readings)

[76페이지](ch03.html#sec-chp3-fr), [112페이지](ch04.html#sec-chp4-sr), [136페이지](ch05.html#sec-chp5-sr)의 "추천 도서(Suggested Readings)"에서 제안된 자료들은 푸아송 회귀와 같은 일반화 선형 모델을 더 잘 이해하는 데 도움이 될 것입니다. 마지막으로 도움이 될 만한 회귀 관련 서적은 다음과 같습니다. _Applied Generalized Linear Models and Multilevel Models in R_ (Paul Roback 및 Julie Legler 저, CRC Press, 2021). 이 책은 회귀 분석과 GLM과 같은 고급 도구에 대해 자세히 알고 싶은 분들을 위해 접근하기 쉬운 소개를 제공합니다.

베팅 및 더 넓은 세계에 대한 확률의 적용에 대해 더 자세히 알아보려면 다음 자료를 확인하세요.

- _The Logic of Sports Betting_ (Ed Miller 및 Matthew Davidow 저, 자가 출판, 2019)은 더 자세한 내용을 원하는 사람들을 위해 베팅이 어떻게 작동하는지에 대한 세부 정보를 제공합니다.

- _The Wisdom of the Crowds_ (James Surowiecki 저, Doubleday, 2004)는 군중의 "시장" 또는 집단적 추측이 어떻게 결과를 예측할 수 있는지 설명합니다.

- _Sharp Sports Betting_ (Stanford Wong 저, Huntington Press, 2021)은 독자가 베팅을 더 잘 이해할 수 있도록 돕고 해당 주제에 대한 소개를 제공합니다.

- _Sharper: a Guide to Modern Sports Betting_ (True Pokerjo 저, 자가 출판, 2016)은 통찰력을 얻고자 하는 사람들을 위해 스포츠 베팅에 대해 이야기합니다.

- _The Foundations of Statistics_, 제2 개정판 (Leonard J. Savage 저, Dover Press, 1972)은 통계를 사용하여 결정을 내리는 방법에 대한 고전적인 통계 텍스트입니다. 이 책의 전반부는 실제 응용에서의 통계를 잘 설명하고 있습니다. 후반부는 수학적으로 엄격한 기초를 원하는 사람들을 위한 이론적인 내용입니다. (리처드는 후반부를 훑어보고 나서 대부분을 건너뛰었습니다.)

- [The Good Judgment Project](https://goodjudgment.com)는 현실 세계의 문제를 예측하기 위해 "군중의 지혜"를 적용하고자 합니다. 처음에 미국 정보기관의 자금 지원을 받은 이 프로젝트의 웹사이트에는 "예측이 너무 정확해서 기밀 데이터에 접근할 수 있는 정보 분석가보다 실적이 좋았습니다"라고 명시되어 있습니다.

<sup>[1](ch06.html#id759-marker)</sup> 이 장의 모든 베팅 확률(odds)은 [Betstamp](https://betstamp.app)에서 가져왔습니다.
