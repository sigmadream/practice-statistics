# 부록 C. 데이터 랭글링 기본 (Appendix C. Data-Wrangling Fundamentals)

> 깔끔한(Tidy) 데이터셋은 모두 비슷하지만 지저분한 데이터셋은 각자의 방식으로 지저분합니다.
>
> 해들리 위컴(Hadley Wickham)

이 부록에서는 *데이터 랭글링(data wrangling)*, 즉 데이터를 사용하기 전에 포맷을 지정하고 정리하는 프로세스의 몇 가지 기본 사항에 초점을 맞춥니다. 여기에는 우리가 정기적으로 사용하지만 때로는 혼란스러울 수 있는 몇 가지 일반적인 도구가 포함됩니다. 해들리 위컴이 언급했듯이, 각각의 지저분한 데이터에는 고유한 병리(pathologies)가 있으므로 큰 도구 상자가 필요합니다. Python과 R에 대한 보다 심층적인 비교를 보려면 릭 J. 스카베타(Rick J. Scavetta)와 보얀 안젤로프(Boyan Angelov)가 쓴 <a href="https://learning.oreilly.com/library/view/python-and-r/9781492093398" class="orm:hideurl"><em>Python and R for the Modern Data Scientist</em></a> (O’Reilly, 2021)의 부록을 확인하십시오.

###### 참고 (Note)

데이터로 작업하는 거의 모든 사람이 데이터를 정리(clean)해야 하기 때문에 *데이터 랭글링* 에는 많은 동의어가 있습니다. 다른 용어로는 *데이터 클리닝(data cleaning)*, *데이터 포맷팅(data formatting)*, *데이터 타이딩(data tidying)*, *데이터 변환(data transformation)*, *데이터 조작(data manipulation)*, *데이터 먼징(data munging)* 및 *데이터 뮤테이팅(data mutating)* 이 있습니다. 기본적으로 사람들은 다양한 용어를 사용하므로 여러 출처에서 서로 다른 용어를 보더라도 놀라지 마십시오. 또한 우리의 경험에 비추어 볼 때 사람들은 이러한 용어를 일관성 없이 사용합니다. 핵심은 여러분 스스로도 언젠가는 자신의 데이터를 정리, 포맷 지정, 변환 또는 변경해야 한다는 것입니다. 따라서 우리는 이 부록을 포함했습니다.

# 논리 연산자 (Logic Operators)

논리 연산자는 Python과 R을 포함한 대부분의 언어에서 동일합니다. 다음 <a href="#tbl-logic" data-type="xref">표 C-1</a>에는 몇 가지 일반적인 연산자가 나열되어 있습니다. R에서 벡터를 생성하여 이러한 연산자를 탐색해 보십시오.

```
## R
score <- c(21, 7, 0, 14)
team <- c("GB", "DEN", "KC", "NYJ")
```

또는 Python에서 `numpy`를 사용하여 배열(arrays)을 생성합니다.

```
## Python
import numpy as np
score = np.array([21, 7, 0, 14])
team = np.array(['GB', 'DEN', 'KC', 'NYJ'])
```

###### 경고 (Warning)

Python의 `numpy` 배열은 기본 Python의 리스트와 다르며 수학 함수에 대해 다르게 동작합니다.

초과(`>`) 또는 미만(`<`)과 같은 기본 연산자는 쉽게 파악할 수 있습니다. 예를 들어 Python에서 7보다 큰 요소(elements)가 무엇인지 확인할 수 있습니다.

```
## Python
score > 7
```

결과는 다음과 같습니다.

array([ True, False, False,  True])

보시다시피, 배열과 함께 이러한 연산자를 사용하면 작업이 각 요소에 대해 개별적으로 수행되고 모든 결과가 새 배열에 배치됩니다. 이것은 곧 살펴볼 R에서 작업이 수행되는 방식과 유사합니다. 예를 들어 R에서 15보다 작은 요소가 무엇인지 확인할 수 있습니다.

```
## R
score < 15
```

결과는 다음과 같습니다.

[1] FALSE  TRUE  TRUE  TRUE

이하 및 이상은 등호와 연산자를 사용합니다. `>=`는 이상이고 `<=`는 이하입니다. 예를 들어 다음 코드 예시를 이전 예시와 비교해 보십시오.

```
## Python
score <= 14
```

결과는 다음과 같습니다.

array([False,  True,  True,  True])

다른 연산자들은 덜 명확합니다. 이미 객체를 정의하는 데 `=`를 사용하기 때문에 *같음(equals)* 에는 `==`가 사용됩니다. 예를 들어 `team`의 요소 중 `GB`와 같은 요소를 찾을 수 있습니다. `team`을 인용 부호(**`"GB"`**)로 묶어야 합니다. 그렇지 않으면 컴퓨터는 당신이 `GB`라는 이름의 객체를 사용하려고 한다고 생각합니다.

```
## Python
team == "GB"
```

결과는 다음과 같습니다.

array([ True, False, False, False])

`in` 유형의 연산자를 사용하는 것은 비슷한 포지션에서 뛰는 선수를 여러 가지 방법으로 차트화할 때 정말 유용합니다. 예를 들어 `DE`(수비 엔드), `OLB`(아웃사이드 라인배커) 및 `ED`(에지 디펜더)는 풋볼에서 비슷한 의미이며, 이 세 가지 용어를 모두 사용하여 데이터셋을 필터링하는 것은 분석에서 자주 하는 일입니다.

`numpy`에서는 `.isin()` 함수를 사용하여 이 작업을 수행할 수 있습니다.

```
## Python
position = np.array(['QB', 'DE', 'OLB', 'ED'])
np.isin(position, ['DE', 'OLB', 'ED'])
```

결과는 다음과 같습니다.

array([False,  True,  True,  True])

`pandas` 패키지에는 데이터프레임에 대해 유사한 함수가 있으며, 이는 <a href="#sec-ap-3-sort" data-type="xref">“데이터 필터링 및 정렬 (Filtering and Sorting Data)”</a>에서 다룹니다.

R에는 약간 다른 연산자인 `%in%` 함수가 있습니다.

```
## R
position <- c("QB", "DE", "OLB", "ED")
position %in% c("DE", "OLB", "ED")
```

결과는 다음과 같습니다.

[1] FALSE  TRUE  TRUE  TRUE

`%in%`을 사용할 때는 순서에 유의하십시오. 예를 들어 이전 예제의 `position %in% c("DE", "OLB", "ED")`와 `c("DE", "OLB", "ED") %in% position`을 비교해 보십시오.

```
## R
c("DE", "OLB", "ED") %in% position
```

결과는 다음과 같습니다.

[1] TRUE TRUE TRUE

###### 팁 (Tip)

`in` 연산자를 사용하는 것은 어려울 수 있습니다. 우리는 종종 <a href="#sec-logic-ops" data-type="xref">“논리 연산자 (Logic Operators)”</a>에 있는 것과 같은 작은 테스트용 하위 집합(subset) 데이터를 가져와 코드가 예상대로 작동하는지 확인합니다. 더 넓게 말하면, 코드가 예상대로 작동한다고 스스로 확신할 때까지 코드를 신뢰하지 마십시오. 평소에는 `print()` 문을 사용하여 코드를 살짝 살펴보고 생각한 대로 수행되는지 확인하십시오. 우리는 일회성 프로젝트의 경우 이 작업을 수행합니다. 공식적으로 단위 테스트(unit-testing)는 코드를 테스트하는 방법으로 존재합니다. Python에는 `unittest` 패키지가 함께 제공되며, R에는 공식적인 테스트를 위한 `testthat` 패키지가 있습니다. 우리는 재사용하려는 코드나 오류 발생 시 비용이 많이 드는 코드를 가져올 때 단위 테스트를 사용합니다.

또한 *and* 연산자(`&`) 또는 *or* 연산자(`|`)를 사용하여 연산자를 연결할 수도 있습니다. 여러 연산자를 사용하려면 값이 쌍으로 정렬되어 있어야 합니다. 우리의 예제는 `score`가 `team`에 해당함을 의미합니다. 예제에서 두 벡터의 길이는 모두 4입니다.

예를 들어 `score`에 대해 값이 `7` 이상이고 `team` 값이 `DEN`인 항목을 확인할 수 있습니다. `numpy` 배열을 사용할 때는 `where()` 함수를 사용해야 하지만 이 논리는 이후 이 장의 `pandas`에서도 동일하며 유사한 표기법을 사용할 것입니다. 기준을 충족하는 항목이 결과로 나타납니다.

```
## Python
np.where((score >= 7) & (team == "DEN"))
```

결과는 다음과 같습니다.

(array([1]),)

또한 *or* 연산자를 사용하여 이와 유사한 비교를 통해 `score` 값이 `7`보다 큰 요소 *또는* `team` 값이 `DEN`과 같은 요소를 확인할 수 있습니다.

```
## R
score > 7 | team == "DEN"
```

결과는 다음과 같습니다.

[1]  TRUE  TRUE FALSE  TRUE

괄호를 사용하여 여러 조건을 함께 연결할 수 있습니다. 예를 들어 `score` 값이 `7` 이상이면서 *and* `team`이 `DEN`인 요소 *또는* `score`가 `0`과 같은 요소가 무엇인지 확인할 수 있습니다.

```
## Python
np.where((score >= 7) & (team == "DEN") | (score == 0))
```

결과는 다음과 같습니다.

(array([1, 2]),)

마찬가지로 R에서도 유사한 표기법을 사용할 수 있습니다.

```
## R
(score >= 7 & team == "DEN") | (score == 0)
```

결과는 다음과 같습니다.

[1] FALSE  TRUE  TRUE FALSE

<table id="tbl-logic" style="width: 100%">
<caption>표 C-1. 일반적인 논리 연산자. <sup><a href="app03.html#id1131" id="id1131-marker" data-type="noteref">a</a></sup></caption>
<thead>
<tr>
<th>기호 (Symbol)</th>
<th>예시 (Example)</th>
<th>이름 (Name)</th>
<th>질문 (Question)</th>
</tr>
</thead>
<tbody>
<tr>
<td><p><code>==</code></p></td>
<td><p><code>score == 2</code></p></td>
<td><p>같음 (Equals)</p></td>
<td><p><code>score</code>가 <code>2</code>와 같은가?</p></td>
</tr>
<tr>
<td><p><code>!=</code></p></td>
<td><p><code>score != 2</code></p></td>
<td><p>같지 않음 (Not equals)</p></td>
<td><p><code>score</code>가 <code>2</code>와 같지 않은가?</p></td>
</tr>
<tr>
<td><p><code>&gt;</code></p></td>
<td><p><code>score &gt; 2</code></p></td>
<td><p>초과 (Greater than)</p></td>
<td><p><code>score</code>가 <code>2</code>보다 큰가?</p></td>
</tr>
<tr>
<td><p><code>&lt;</code></p></td>
<td><p><code>score &lt; 2</code></p></td>
<td><p>미만 (Less than)</p></td>
<td><p><code>score</code>가 <code>2</code>보다 작은가?</p></td>
</tr>
<tr>
<td><p><code>&gt;=</code></p></td>
<td><p><code>score &gt;= 2</code></p></td>
<td><p>이상 (Greater than or equal to)</p></td>
<td><p><code>score</code>가 <code>2</code>보다 크거나 같은가?</p></td>
</tr>
<tr>
<td><p><code>&lt;=</code></p></td>
<td><p><code>score &lt;= 2</code></p></td>
<td><p>이하 (Less than or equal to)</p></td>
<td><p><code>score</code>가 <code>2</code>보다 작거나 같은가?</p></td>
</tr>
<tr>
<td><p><code>|</code></p></td>
<td><p><code>(score &gt; 2) | (team =="GB")</code></p></td>
<td><p>또는 (Or)</p></td>
<td><p><code>score</code>가 <code>2</code>보다 크거나, <code>team</code>이 GB와 같은가?</p></td>
</tr>
<tr>
<td><p><code>&amp;</code></p></td>
<td><p><code>(score &gt; 2) &amp; (team =="GB")</code></p></td>
<td><p>그리고 (And)</p></td>
<td><p><code>score</code>가 <code>2</code>보다 크고, <code>team</code>이 <code>GB</code>와 같은가?</p></td>
</tr>
</tbody>
<tbody>
<tr class="footnotes">
<td colspan="4"><p><sup><a href="app03.html#id1131-marker">a</a></sup> <code>pandas</code>는 때때로 일부 상황에서 <em>not</em>에 대해 <code>!</code> 대신 <code>~</code>를 사용합니다.</p></td>
</tr>
</tbody>
</table>

# 데이터 필터링 및 정렬 (Filtering and Sorting Data)

이전 섹션에서는 논리 연산자에 대해 배웠습니다. 이러한 함수는 데이터를 필터링하는 기초 역할을 합니다. 실제로 필터링에 문제가 발생하면 데이터를 이해하고 필터가 작동하는(또는 때때로 작동하지 않는) 방식을 이해하기 위해 종종 <a href="#sec-logic-ops" data-type="xref">“논리 연산자 (Logic Operators)”</a>의 경우와 같은 작은 테스트 사례를 구축합니다.

###### 팁 (Tip)

필터링은 어려울 수 있습니다. 작게 시작하여 필터링 명령에 복잡성을 점진적으로 추가해 보십시오. 문제를 해결할 수 있을 때까지 세부 정보를 계속 추가하십시오. 때로는 하나의 큰 필터 연산 대신 두 개 이상의 더 작은 필터를 사용해야 할 수도 있습니다. 이것도 괜찮습니다. 최적화에 대해 걱정하기 전에 코드가 제대로 작동하게 만드십시오.

2020 시즌 2주차 그린베이-디트로이트(Green Bay–Detroit) 데이터를 사용하게 될 것입니다. 먼저, 데이터를 읽어오고 간단한 필터를 수행하여 캐치 후 전진 야드(yards-after-catch) 값이 15야드보다 큰 플레이를 찾아 큰 플레이가 어디에서 발생했는지 파악합니다.

R에서는 `tidyverse` 및 `nflfastR` 패키지를 로드한 다음 2020년 데이터를 로드합니다.

```
## R
library(tidyverse)
library(nflfastR)

# 모든 데이터 로드(Load all data)
pbp_r <- load_pbp(2020)
```

Python에서는 `pandas`, `numpy` 및 `nfl_data_py` 패키지를 임포트(import)한 다음 2020년 데이터를 로드합니다.

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# 모든 데이터 로드(Load all data)

pbp_py = nfl.import_pbp_data([2020])
```

결과는 다음과 같습니다.

2020 done. Downcasting floats.

R에서는 다음으로 `filter()` 함수를 사용합니다. 필터에 들어가는 첫 번째 인수(argument)는 `data`입니다. 두 번째 인수는 `filter` 기준(criteria)입니다. 그린베이에서 열린 디트로이트의 경기 데이터를 필터링하고 몇 가지 패싱 관련 열을 선택합니다.

```
# 게임 데이터 필터링(Filter out game data)
gb_det_2020_r_pass <-
    pbp_r |>
    filter(home_team == 'GB' & away_team == 'DET') |>
    select(posteam, yards_after_catch, air_yards,
           pass_location, qb_scramble)
```

다음으로 `yards_after_catch`가 `15`보다 큰 플레이를 `filter()`로 걸러냅니다.

```
gb_det_2020_r_pass |>
filter(yards_after_catch > 15)
```

결과는 다음과 같습니다.

# A tibble: 5 × 5
  posteam yards_after_catch air_yards pass_location qb_scramble
  <chr>               <dbl>     <dbl> <chr>               <dbl>
1 DET                    16        13 left                    0
2 GB                     19         3 right                   0
3 GB                     19         6 right                   0
4 DET                    16         1 middle                  0
5 DET                    20        16 middle                  0

###### 팁 (Tip)

R과 Python에서는 항상 인수(argument) 이름을 사용할 필요는 없습니다. 대신 언어는 사전에 정의된 순서와 인수를 일치시킵니다. 이 순서는 도움말 파일에 나열되어 있습니다. 예를 들어 `gb_det_2020_r_pass |> filter(yards_after_catch > 15)`의 경우 `gb_det_2020_r_pass |> filter(filter = yards_after_catch > 15)`로 작성할 수도 있었습니다. 우리는 일반적으로 더 복잡한 함수이거나 명확하게 하려는 경우 인수 이름을 정의합니다. 이를 통해 코드를 더 읽기 쉽게 만들 수 있기 때문에, 명시적으로(explicit) 작성하고 인수 이름을 사용하는 쪽으로 실수를 범하는 것이 낫습니다.
