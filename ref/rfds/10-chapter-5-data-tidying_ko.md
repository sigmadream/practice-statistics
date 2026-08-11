# 5장. 데이터 정돈(Data Tidying)

# 들어가며

> “행복한 가정은 모두 엇비슷하지만, 불행한 가정은 제각기 다른 이유로 불행하다.”\
> — 레프 톨스토이(Leo Tolstoy)

> “정돈된 데이터셋(Tidy datasets)은 모두 엇비슷하지만, 지저분한 데이터셋은 제각기 다른 이유로 지저분하다.”\
> — 해들리 위컴(Hadley Wickham)

이 장에서는 *정돈된 데이터(tidy data)*라는 시스템을 사용하여 R에서 데이터를 일관된 방식으로 구성하는 방법을 배웁니다. 데이터를 이 형식으로 만드는 데는 초기에 약간의 작업이 필요하지만, 장기적으로는 그 노력이 보상을 받습니다. 일단 정돈된 데이터와 tidyverse의 패키지들이 제공하는 정돈 도구들을 갖추고 나면, 한 표현에서 다른 표현으로 데이터를 변환(munging)하는 데 훨씬 적은 시간을 쓰게 되어, 여러분이 관심을 가지는 데이터 질문에 더 많은 시간을 할애할 수 있습니다.

이 장에서는 먼저 정돈된 데이터의 정의를 배우고 간단한 장난감 데이터셋에 적용된 것을 볼 것입니다. 그런 다음 데이터를 정돈하는 데 사용할 주요 도구인 피벗(pivoting)에 대해 깊이 파고들 것입니다. 피벗을 사용하면 값을 변경하지 않고도 데이터의 형태를 바꿀 수 있습니다.

## 사전 준비

이 장에서는 지저분한 데이터셋을 정돈하는 데 도움이 되는 여러 도구를 제공하는 패키지인 tidyr에 중점을 둘 것입니다. tidyr은 핵심 tidyverse의 멤버입니다.

`library``(``tidyverse``)`

이 장부터는 <a href="https://tidyverse.tidyverse.org" class="orm:hideurl"><code>library(tidyverse)</code></a>의 로딩 메시지를 숨기겠습니다.

# 정돈된 데이터(Tidy Data)

동일한 기본 데이터를 여러 가지 방식으로 표현할 수 있습니다. 다음 예제는 동일한 데이터가 세 가지 다른 방식으로 구성된 것을 보여줍니다. 각 데이터셋은 _country_(국가), _year_(연도), _population_(인구), 결핵(TB)의 기록된 _cases_(환자 수)라는 네 가지 변수의 동일한 값을 보여주지만, 각 데이터셋은 다른 방식으로 값을 구성합니다.

 

`table1` `#> # A tibble: 6 × 4` `#> country year cases population` `#> <chr> <dbl> <dbl> <dbl>` `#> 1 Afghanistan 1999 745 19987071` `#> 2 Afghanistan 2000 2666 20595360` `#> 3 Brazil 1999 37737 172006362` `#> 4 Brazil 2000 80488 174504898` `#> 5 China 1999 212258 1272915272` `#> 6 China 2000 213766 1280428583` `table2` `#> # A tibble: 12 × 4` `#> country year type count` `#> <chr> <dbl> <chr> <dbl>` `#> 1 Afghanistan 1999 cases 745` `#> 2 Afghanistan 1999 population 19987071` `#> 3 Afghanistan 2000 cases 2666` `#> 4 Afghanistan 2000 population 20595360` `#> 5 Brazil 1999 cases 37737` `#> 6 Brazil 1999 population 172006362` `#> # … with 6 more rows` `table3` `#> # A tibble: 6 × 3` `#> country year rate ` `#> <chr> <dbl> <chr> ` `#> 1 Afghanistan 1999 745/19987071 ` `#> 2 Afghanistan 2000 2666/20595360 ` `#> 3 Brazil 1999 37737/172006362 ` `#> 4 Brazil 2000 80488/174504898 ` `#> 5 China 1999 212258/1272915272` `#> 6 China 2000 213766/1280428583`

이들은 모두 동일한 기본 데이터를 표현한 것이지만, 사용하기에 똑같이 쉽지는 않습니다. 그 중 하나인 `table1`은 _정돈되어(tidy)_ 있기 때문에 tidyverse 내에서 작업하기가 훨씬 쉬울 것입니다.

데이터셋을 정돈되게 만드는 데에는 서로 연관된 세 가지 규칙이 있습니다.

1.  각 변수는 열(column)입니다; 각 열은 변수입니다.
2.  각 관측치는 행(row)입니다; 각 행은 관측치입니다.
3.  각 값은 셀(cell)입니다; 각 셀은 단일 값입니다.

<a href="#fig-tidy-structure" data-type="xref">그림 5-1</a>은 이 규칙들을 시각적으로 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0501.png" alt="정돈된 데이터 프레임을 나타내는 3개의 패널. 첫 번째 패널은 각 변수가 열임을 보여줍니다. 두 번째 패널은 각 관측치가 행임을 보여줍니다. 세 번째 패널은 각 값이 셀임을 보여줍니다." />
<h6 id="figure-5-1.-three-rules-make-a-dataset-tidy-variables-are-columns-observations-are-rows-and-values-are-cells.">그림 5-1. 데이터셋을 정돈되게 만드는 세 가지 규칙: 변수는 열, 관측치는 행, 값은 셀입니다.</h6>
</figure>

왜 데이터를 정돈되게 보장해야 할까요? 두 가지 주요 이점이 있습니다.

1.  데이터를 저장하는 일관된 방식을 하나 선택하는 데서 오는 일반적인 이점이 있습니다. 일관된 데이터 구조를 가지면, 근본적인 통일성이 있기 때문에 그와 작동하는 도구를 배우기 쉽습니다.

2.  변수를 열에 배치하면 R의 벡터화(vectorized) 특성이 빛을 발할 수 있다는 특별한 이점이 있습니다. <a href="ch03.html#sec-mutate" data-type="xref">“mutate()”</a>와 <a href="ch03.html#sec-summarize" data-type="xref">“summarize()”</a>에서 배웠듯이, 대부분의 내장 R 함수는 값의 벡터와 함께 작동합니다. 이는 정돈된 데이터를 변환하는 것을 특히 자연스럽게 느껴지게 합니다.

dplyr, ggplot2 및 tidyverse의 다른 모든 패키지들은 정돈된 데이터와 함께 작동하도록 설계되었습니다.

여기 `table1`로 어떻게 작업할 수 있는지 보여주는 몇 가지 작은 예시가 있습니다.

`# 10,000명당 비율 계산` `table1` `|>` `mutate``(``rate` `=` `cases` `/` `population` `*` `10000``)` `#> # A tibble: 6 × 5` `#> country year cases population rate` `#> <chr> <dbl> <dbl> <dbl> <dbl>` `#> 1 Afghanistan 1999 745 19987071 0.373` `#> 2 Afghanistan 2000 2666 20595360 1.29 ` `#> 3 Brazil 1999 37737 172006362 2.19 ` `#> 4 Brazil 2000 80488 174504898 4.61 ` `#> 5 China 1999 212258 1272915272 1.67 ` `#> 6 China 2000 213766 1280428583 1.67` `# 연도별 총 환자 수 계산` `table1` `|>` `group_by``(``year``)` `|>` `summarize``(``total_cases` `=` `sum``(``cases``))` `#> # A tibble: 2 × 2` `#> year total_cases` `#> <dbl> <dbl>` `#> 1 1999 250740` `#> 2 2000 296920` `# 시간에 따른 변화 시각화` `ggplot``(``table1``,` `aes``(``x` `=` `year``,` `y` `=` `cases``))` `+` `geom_line``(``aes``(``group` `=` `country``),` `color` `=` `"grey50"``)` `+` `geom_point``(``aes``(``color` `=` `country``,` `shape` `=` `country``))` `+` `scale_x_continuous``(``breaks` `=` `c``(``1999``,` `2000``))` `# x축 눈금을 1999와 2000으로 설정`

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_05in01.png" alt="이 그림은 아프가니스탄, 브라질, 중국의 1999년과 2000년 결핵 환자 수를 x축은 연도, y축은 환자 수로 나타냅니다." />
</figure>

## 연습문제

1.  각 샘플 테이블에 대해 각 관측치와 각 열이 무엇을 나타내는지 설명하세요.

2.  `table2`와 `table3`에 대한 `rate`(비율)를 계산하기 위해 사용할 프로세스의 스케치를 그려보세요. 다음 네 가지 작업을 수행해야 합니다.
    1.  국가 및 연도별 결핵 환자 수를 추출합니다.
    2.  국가 및 연도별 일치하는 인구를 추출합니다.
    3.  환자 수를 인구로 나누고 10,000을 곱합니다.
    4.  적절한 위치에 다시 저장합니다.

    이러한 작업을 실제로 수행하는 데 필요한 모든 함수를 아직 배우지는 않았지만, 여전히 필요한 변환 과정을 생각할 수 있어야 합니다.

# 데이터 길게 만들기 (Lengthening Data)

정돈된 데이터의 원칙이 너무나 당연해 보여서 정돈되지 않은 데이터셋을 마주칠 일이 있을까 궁금할 수 있습니다. 하지만 불행히도 대부분의 실제 데이터는 정돈되어 있지 않습니다. 그 이유는 두 가지 주요 원인이 있습니다.

1.  데이터는 분석 이외의 어떤 목표를 촉진하기 위해 종종 구성됩니다. 예를 들어, 분석이 아닌 데이터 입력을 쉽게 만들도록 데이터가 구조화되는 것이 일반적입니다.

2.  대부분의 사람들은 정돈된 데이터의 원칙에 익숙하지 않으며, 데이터 작업을 하면서 많은 시간을 보내지 않는 한 스스로 이 원칙을 이끌어내기 어렵습니다.

이는 대부분의 실제 분석이 최소한 약간의 정돈 작업을 필요로 함을 의미합니다. 먼저 기본 변수와 관측치가 무엇인지 파악하는 것으로 시작할 것입니다. 때로는 이것이 쉽지만, 다른 때에는 원래 데이터를 생성한 사람들과 상의해야 할 수도 있습니다. 다음으로, 변수를 열에, 관측치를 행에 두는 정돈된 형태로 데이터를 *피벗(pivot)*할 것입니다.

tidyr은 데이터를 피벗하기 위한 두 가지 함수를 제공합니다. <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>. <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>가 가장 일반적인 경우이기 때문에 먼저 시작할 것입니다. 몇 가지 예제를 살펴보겠습니다.

## 열 이름에 포함된 데이터

`billboard` 데이터셋은 2000년의 빌보드 노래 순위를 기록합니다.

`billboard` `#> # A tibble: 317 × 79` `#> artist track date.entered wk1 wk2 wk3 wk4 wk5` `#> <chr> <chr> <date> <dbl> <dbl> <dbl> <dbl> <dbl>` `#> 1 2 Pac Baby Don't Cry (Ke… 2000-02-26 87 82 72 77 87` `#> 2 2Ge+her The Hardest Part O… 2000-09-02 91 87 92 NA NA` `#> 3 3 Doors Down Kryptonite 2000-04-08 81 70 68 67 66` `#> 4 3 Doors Down Loser 2000-10-21 76 76 72 69 67` `#> 5 504 Boyz Wobble Wobble 2000-04-15 57 34 25 17 17` `#> 6 98^0 Give Me Just One N… 2000-08-19 51 39 34 26 26` `#> # … with 311 more rows, and 71 more variables: wk6 <dbl>, wk7 <dbl>,` `#> # wk8 <dbl>, wk9 <dbl>, wk10 <dbl>, wk11 <dbl>, wk12 <dbl>, wk13 <dbl>, …`

이 데이터셋에서 각 관측치는 노래입니다. 처음 세 열(`artist`, `track`, `date.entered`)은 노래를 설명하는 변수입니다. 그런 다음 매주 노래의 순위를 설명하는 76개의 열(`wk1`-`wk76`)이 있습니다.<sup><a href="ch05.html#idm44771326722336" id="idm44771326722336-marker" data-type="noteref">1</a></sup> 여기서 열 이름은 하나의 변수(`week`)이고 셀 값은 다른 변수(`rank`)입니다.

이 데이터를 정돈하기 위해 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>를 사용할 것입니다.

`billboard` `|>` `pivot_longer``(` `cols` `=` `starts_with``(``"wk"``),` `names_to` `=` `"week"``,` `values_to` `=` `"rank"` `)` `#> # A tibble: 24,092 × 5` `#> artist track date.entered week rank` `#> <chr> <chr> <date> <chr> <dbl>` `#> 1 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk1 87` `#> 2 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk2 82` `#> 3 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk3 72` `#> 4 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk4 77` `#> 5 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk5 87` `#> 6 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk6 94` `#> 7 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk7 99` `#> 8 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk8 NA` `#> 9 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk9 NA` `#> 10 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk10 NA` `#> # … with 24,082 more rows`

데이터 뒤에 세 가지 주요 인수가 있습니다.

`cols`  
피벗해야 할 열(즉, 변수가 아닌 열)을 지정합니다. 이 인수는 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>와 동일한 구문을 사용하므로, 여기서 `!c(artist, track, date.entered)` 또는 `starts_with("wk")`를 사용할 수 있습니다.

`names_to`  
열 이름에 저장될 변수의 이름을 지정합니다; 우리는 그 변수를 `week`라고 명명했습니다.

`values_to`  
셀 값에 저장될 변수의 이름을 지정합니다; 우리는 그 변수를 `rank`라고 명명했습니다.

코드에서 `"week"`와 `"rank"`가 인용 부호로 묶인 이유는 우리가 새로 생성하는 변수이기 때문입니다; <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a> 호출을 실행할 때 이 변수들은 데이터에 아직 존재하지 않습니다.

이제 결과인 더 긴 데이터 프레임에 주의를 돌려봅시다. 만약 어떤 노래가 76주 미만 동안 탑 100에 머물렀다면 어떻게 될까요? 2 Pac의 “Baby Don’t Cry”를 예로 들어보겠습니다. 이전 출력은 이 곡이 7주 동안만 탑 100에 있었고 남은 모든 주는 결측치로 채워져 있음을 시사합니다. 이러한 `NA`들은 사실 알려지지 않은 관측치를 나타내는 것이 아니라, 데이터셋의 구조에 의해 강제로 존재하게 된 것입니다.<sup><a href="ch05.html#idm44771328141216" id="idm44771328141216-marker" data-type="noteref">2</a></sup> 따라서 `values_drop_na = TRUE`로 설정하여 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>가 그것들을 제거하도록 할 수 있습니다.

`billboard` `|>` `pivot_longer``(` `cols` `=` `starts_with``(``"wk"``),` `names_to` `=` `"week"``,` `values_to` `=` `"rank"``,` `values_drop_na` `=` `TRUE` `)` `#> # A tibble: 5,307 × 5` `#> artist track date.entered week rank` `#> <chr> <chr> <date> <chr> <dbl>` `#> 1 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk1 87` `#> 2 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk2 82` `#> 3 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk3 72` `#> 4 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk4 77` `#> 5 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk5 87` `#> 6 2 Pac Baby Don't Cry (Keep... 2000-02-26 wk6 94` `#> # … with 5,301 more rows`

이제 행 수가 훨씬 적어졌으며, `NA`가 있는 많은 행이 드롭되었음을 나타냅니다.

또한 노래가 76주 이상 상위 100위 안에 들면 어떻게 되는지 궁금할 수도 있습니다. 이 데이터로는 알 수 없지만, `wk77`, `wk78` 등과 같은 추가 열이 데이터 세트에 추가되었을 것이라고 짐작할 수 있습니다.

이제 이 데이터는 정돈되었지만, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>readr::parse_number()</code></a>를 사용하여 `week`의 값을 문자열에서 숫자로 변환하면 향후 계산을 좀 더 쉽게 할 수 있습니다. <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>parse_number()</code></a>는 다른 모든 텍스트를 무시하고 문자열에서 첫 번째 숫자를 추출하는 유용한 함수입니다.

`billboard_longer` `<-` `billboard` `|>` `pivot_longer``(` `cols` `=` `starts_with``(``"wk"``),` `names_to` `=` `"week"``,` `values_to` `=` `"rank"``,` `values_drop_na` `=` `TRUE` `)` `|>` `mutate``(` `week` `=` `parse_number``(``week``)` `)` `billboard_longer` `#> # A tibble: 5,307 × 5` `#> artist track date.entered week rank` `#> <chr> <chr> <date> <dbl> <dbl>` `#> 1 2 Pac Baby Don't Cry (Keep... 2000-02-26 1 87` `#> 2 2 Pac Baby Don't Cry (Keep... 2000-02-26 2 82` `#> 3 2 Pac Baby Don't Cry (Keep... 2000-02-26 3 72` `#> 4 2 Pac Baby Don't Cry (Keep... 2000-02-26 4 77` `#> 5 2 Pac Baby Don't Cry (Keep... 2000-02-26 5 87` `#> 6 2 Pac Baby Don't Cry (Keep... 2000-02-26 6 94` `#> # … with 5,301 more rows`

이제 하나의 변수에 모든 주(week) 번호를, 다른 변수에는 모든 순위 값을 갖게 되어, 시간에 따라 곡의 순위가 어떻게 변하는지 시각화하기에 좋은 상태가 되었습니다. 코드는 여기에 표시되며 결과는 <a href="#fig-billboard-ranks" data-type="xref">그림 5-2</a>에 있습니다. 20주 이상 탑 100에 머무는 곡이 매우 적다는 것을 볼 수 있습니다.

`billboard_longer` `|>` `ggplot``(``aes``(``x` `=` `week``,` `y` `=` `rank``,` `group` `=` `track``))` `+` `geom_line``(``alpha` `=` `0.25``)` `+` `scale_y_reverse``()`

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0502.png" alt="x축에 주, y축에 순위를 나타내는 선형 그래프로, 각 선은 노래를 나타냅니다." />
<h6 id="figure-5-2.-a-line-plot-showing-how-the-rank-of-a-song-changes-over-time.">그림 5-2. 시간에 따른 곡 순위 변화를 보여주는 선 그래프.</h6>
</figure>

## 피벗팅은 어떻게 작동할까요?

데이터 형태를 재구성(reshape)하기 위해 피벗을 어떻게 사용할 수 있는지 살펴보았으니, 이제 피벗이 데이터에 어떤 작업을 하는지에 대한 직관을 얻는 시간을 조금 가져봅시다. 어떻게 되는지 보기 쉽게 하기 위해 간단한 데이터 세트로 시작해 보겠습니다. `id`가 A, B, C인 세 명의 환자가 있고 각 환자마다 혈압 측정을 두 번 한다고 가정해 봅시다. 작은 티블을 수작업으로 만드는 데 유용한 함수인 <a href="https://tibble.tidyverse.org/reference/tribble.html" class="orm:hideurl"><code>tribble()</code></a>로 데이터를 만들어 보겠습니다.

`df` `<-` `tribble``(` `~``id``,` `~``bp1``,` `~``bp2``,` `"A"``,` `100``,` `120``,` `"B"``,` `140``,` `115``,` `"C"``,` `120``,` `125` `)`

우리는 새 데이터 세트에 `id`(이미 존재함), `measurement`(열 이름), `value`(셀 값)라는 세 가지 변수를 갖기를 원합니다. 이를 달성하려면 `df`를 더 길게 피벗해야 합니다.

`df` `|>` `pivot_longer``(` `cols` `=` `bp1``:``bp2``,` `names_to` `=` `"measurement"``,` `values_to` `=` `"value"` `)` `#> # A tibble: 6 × 3` `#> id measurement value` `#> <chr> <chr> <dbl>` `#> 1 A bp1 100` `#> 2 A bp2 120` `#> 3 B bp1 140` `#> 4 B bp2 115` `#> 5 C bp1 120` `#> 6 C bp2 125`

형태 재구성은 어떻게 이루어지나요? 열 단위로 생각하면 더 쉽게 볼 수 있습니다. <a href="#fig-pivot-variables" data-type="xref">그림 5-3</a>에 표시된 것처럼 원본 데이터 세트(`id`)에서 이미 변수였던 열의 값은 피벗되는 각 열에 대해 한 번씩 반복되어야 합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0503.png" alt="간단한 데이터셋을 pivot_longer()가 어떻게 변환하는지 보여주는 다이어그램입니다." />
<h6 id="figure-5-3.-columns-that-are-already-variables-need-to-be-repeated-once-for-each-column-that-is-pivoted.">그림 5-3. 이미 변수인 열은 피벗되는 열당 한 번씩 반복되어야 합니다.</h6>
</figure>

열 이름은 `names_to`에 의해 정의된 이름을 가진 새 변수의 값이 되며, 이는 <a href="#fig-pivot-names" data-type="xref">그림 5-4</a>에 나와 있습니다. 원본 데이터 세트의 각 행에 대해 한 번씩 반복되어야 합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0504.png" alt="간단한 데이터 세트를 pivot_longer()가 어떻게 변환하는지 보여주는 다이어그램입니다." />
<h6 id="figure-5-4.-the-column-names-of-pivoted-columns-become-values-in-a-new-column.-the-values-need-to-be-repeated-once-for-each-row-of-the-original-dataset.">그림 5-4. 피벗된 열의 이름은 새 열의 값이 됩니다. 값은 원본 데이터 세트의 각 행에 대해 한 번씩 반복되어야 합니다.</h6>
</figure>

셀 값 또한 `values_to`로 정의된 이름의 새 변수의 값이 됩니다. 이들은 행별로 풀립니다(unwound). <a href="#fig-pivot-values" data-type="xref">그림 5-5</a>는 이 과정을 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0505.png" alt="셀 값(혈압 측정값)이 새로운 value 열의 값이 되는 것을 강조하는 다이어그램입니다." />
<h6 id="figure-5-5.-the-number-of-values-is-preserved-not-repeated-but-unwound-row-by-row.">그림 5-5. 값의 개수는 유지되며(반복되지 않음) 행별로 풀립니다.</h6>
</figure>

## 열 이름에 포함된 여러 변수들

열 이름에 여러 정보 조각이 밀집되어 있고 이들을 각각 별도의 새 변수에 저장하고 싶을 때 더 까다로운 상황이 발생합니다. 예를 들어 앞서 보았던 `table1`의 출처인 `who2` 데이터셋을 살펴보겠습니다.

`who2` `#> # A tibble: 7,240 × 58` `#> country year sp_m_014 sp_m_1524 sp_m_2534 sp_m_3544 sp_m_4554` `#> <chr> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>` `#> 1 Afghanistan 1980 NA NA NA NA NA` `#> 2 Afghanistan 1981 NA NA NA NA NA` `#> 3 Afghanistan 1982 NA NA NA NA NA` `#> 4 Afghanistan 1983 NA NA NA NA NA` `#> 5 Afghanistan 1984 NA NA NA NA NA` `#> 6 Afghanistan 1985 NA NA NA NA NA` `#> # … with 7,234 more rows, and 51 more variables: sp_m_5564 <dbl>,` `#> # sp_m_65 <dbl>, sp_f_014 <dbl>, sp_f_1524 <dbl>, sp_f_2534 <dbl>, …`

세계보건기구(WHO)에서 수집한 이 데이터셋은 결핵 진단에 관한 정보를 기록합니다. 이미 변수이며 해석하기 쉬운 열 두 개가 있습니다. `country`와 `year`. 이 뒤에는 `sp_m_014`, `ep_m_4554`, `rel_m_3544`와 같은 56개의 열이 이어집니다. 이 열들을 오랫동안 들여다보면 패턴이 있다는 것을 알게 될 것입니다. 각 열 이름은 `_`로 분리된 세 조각으로 구성됩니다. 첫 번째 조각인 `sp`/`rel`/`ep`는 진단에 사용된 방법을 설명하고, 두 번째 조각인 `m`/`f`는 `gender`(성별, 이 데이터셋에서는 이진 변수로 코딩됨)이며, 세 번째 조각인 `014`/`1524`/`2534`/`3544`/`4554`/`65`는 `age`(연령) 범위입니다(예를 들어 `014`는 0–14세를 나타냄).

그래서 이 경우 `who2`에는 6가지 정보가 기록되어 있습니다. 국가와 연도(이미 열임), 진단 방법, 성별 범주, 연령대 범주(다른 열 이름에 포함됨), 해당 범주의 환자 수(셀 값). 이 6가지 정보를 6개의 별도 열로 구성하기 위해, `names_to`에 대한 열 이름 벡터와 원본 변수 이름을 여러 조각으로 분리하기 위한 지시자 `names_sep`, 그리고 `values_to`를 위한 열 이름과 함께 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>를 사용합니다.

`who2` `|>` `pivot_longer``(` `cols` `=` `!``(``country``:``year``),` `names_to` `=` `c``(``"diagnosis"``,` `"gender"``,` `"age"``),` `names_sep` `=` `"_"``,` `values_to` `=` `"count"` `)` `#> # A tibble: 405,440 × 6` `#> country year diagnosis gender age count` `#> <chr> <dbl> <chr> <chr> <chr> <dbl>` `#> 1 Afghanistan 1980 sp m 014 NA` `#> 2 Afghanistan 1980 sp m 1524 NA` `#> 3 Afghanistan 1980 sp m 2534 NA` `#> 4 Afghanistan 1980 sp m 3544 NA` `#> 5 Afghanistan 1980 sp m 4554 NA` `#> 6 Afghanistan 1980 sp m 5564 NA` `#> # … with 405,434 more rows`

`names_sep`의 대안으로 `names_pattern`이 있는데, <a href="ch15.html#chp-regexps" data-type="xref">15장</a>에서 정규 표현식을 배운 후에 더 복잡한 이름 지정 시나리오에서 변수를 추출하는 데 사용할 수 있습니다.

개념적으로 이것은 이미 본 단순한 사례의 약간의 변형일 뿐입니다. <a href="#fig-pivot-multiple-names" data-type="xref">그림 5-6</a>은 기본 아이디어를 보여줍니다. 이제 열 이름이 단일 열로 피벗되는 대신 여러 열로 피벗됩니다. 이 작업이 두 단계(먼저 피벗한 다음 분리)로 일어난다고 상상할 수 있지만 내부적으로는 더 빠르기 때문에 단일 단계로 발생합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0506.png" alt="names_sep 및 다중 names_to를 제공하여 출력에 여러 변수를 만드는 방법을 설명하는 다이어그램입니다." />
<h6 id="figure-5-6.-pivoting-columns-with-multiple-pieces-of-information-in-the-names-means-that-each-column-name-now-fills-in-values-in-multiple-output-columns.">그림 5-6. 이름에 여러 정보 조각이 있는 열을 피벗하면 이제 각 열 이름이 여러 출력 열의 값을 채우게 됨을 의미합니다.</h6>
</figure>

## 열 헤더의 데이터 및 변수 이름

복잡성이 한 단계 더 올라가는 경우는 열 이름에 변수 값과 변수 이름이 섞여 있을 때입니다. 예를 들어 `household` 데이터셋을 살펴봅시다.

`household` `#> # A tibble: 5 × 5` `#> family dob_child1 dob_child2 name_child1 name_child2` `#> <int> <date> <date> <chr> <chr> ` `#> 1 1 1998-11-26 2000-01-29 Susan Jose ` `#> 2 2 1996-06-22 NA Mark <NA> ` `#> 3 3 2002-07-11 2004-04-05 Sam Seth ` `#> 4 4 2004-10-10 2009-08-27 Craig Khai ` `#> 5 5 2000-12-05 2005-02-28 Parker Gracie`

이 데이터셋에는 최대 두 명의 자녀에 대한 이름과 생년월일이 포함된 5개 가족에 대한 데이터가 들어 있습니다. 이 데이터셋의 새로운 과제는 열 이름에 두 개의 변수(`dob`, `name`) 이름과 또 다른 변수(`child`, 값은 1 또는 2)의 값이 포함되어 있다는 것입니다. 이 문제를 해결하기 위해 다시 `names_to`에 벡터를 제공해야 하지만 이번에는 특수 `".value"` 센티넬(sentinel)을 사용합니다. 이것은 변수의 이름이 아니라 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>에게 다른 작업을 하도록 지시하는 고유한 값입니다. 이는 일반적인 `values_to` 인수를 재정의하여 피벗된 열 이름의 첫 번째 구성 요소를 출력의 변수 이름으로 사용합니다.

`household` `|>` `pivot_longer``(` `cols` `=` `!``family``,` `names_to` `=` `c``(``".value"``,` `"child"``),` `names_sep` `=` `"_"``,` `values_drop_na` `=` `TRUE` `)` `#> # A tibble: 9 × 4` `#> family child dob name ` `#> <int> <chr> <date> <chr>` `#> 1 1 child1 1998-11-26 Susan` `#> 2 1 child2 2000-01-29 Jose ` `#> 3 2 child1 1996-06-22 Mark ` `#> 4 3 child1 2002-07-11 Sam ` `#> 5 3 child2 2004-04-05 Seth ` `#> 6 4 child1 2004-10-10 Craig` `#> # … with 3 more rows`

입력의 형태가 명시적 누락 변수(자녀가 한 명뿐인 가족)의 생성을 강제하기 때문에 우리는 여기서도 `values_drop_na = TRUE`를 사용합니다.

<a href="#fig-pivot-names-and-values" data-type="xref">그림 5-7</a>은 더 간단한 예로 기본 아이디어를 설명합니다. `names_to`에서 `".value"`를 사용하면 입력의 열 이름이 출력의 값과 변수 이름 모두에 기여합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0507.png" alt="특수 .value 센티넬이 어떻게 작동하는지 보여주는 다이어그램입니다." />
<h6 id="figure-5-7.-pivoting-with-names_to-c.value-num-splits-the-column-names-into-two-components-the-first-part-determines-the-output-column-name-x-or-y-and-the-second-part-determines-the-value-of-the-num-column.">그림 5-7. <code>names_to = c(".value", "num")</code>을 사용한 피벗은 열 이름을 두 구성 요소로 나눕니다. 첫 번째 부분은 출력 열 이름(<code>x</code> 또는 <code>y</code>)을 결정하고 두 번째 부분은 <code>num</code> 열의 값을 결정합니다.</h6>
</figure>

# 데이터 넓게 만들기 (Widening Data)

지금까지는 값이 열 이름에 들어가게 된 일반적인 종류의 문제를 해결하기 위해 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot*longer()</code></a>를 사용했습니다. 다음으로 우리는 데이터셋의 열을 늘리고 행을 줄여 *더 넓게(wider)\_ 만들어 주며 하나의 관측치가 여러 행에 분산되어 있을 때 도움이 되는 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>로 피벗(농담입니다)해 볼 것입니다. 이 경우는 실제 환경에서는 덜 흔하게 발생하는 것 같지만, 정부 데이터를 다룰 때는 꽤 자주 등장하는 것 같습니다.

환자 경험에 대한 데이터를 수집하는 메디케어 및 메디케이드 서비스 센터(Centers of Medicare and Medicaid services)의 데이터셋인 `cms_patient_experience`를 살펴봄으로써 시작하겠습니다.

`cms_patient_experience` `#> # A tibble: 500 × 5` `#> org_pac_id org_nm measure_cd measure_title prf_rate` `#> <chr> <chr> <chr> <chr> <dbl>` `#> 1 0446157747 USC CARE MEDICAL GROUP INC CAHPS_GRP_1 CAHPS for MIPS… 63` `#> 2 0446157747 USC CARE MEDICAL GROUP INC CAHPS_GRP_2 CAHPS for MIPS… 87` `#> 3 0446157747 USC CARE MEDICAL GROUP INC CAHPS_GRP_3 CAHPS for MIPS… 86` `#> 4 0446157747 USC CARE MEDICAL GROUP INC CAHPS_GRP_5 CAHPS for MIPS… 57` `#> 5 0446157747 USC CARE MEDICAL GROUP INC CAHPS_GRP_8 CAHPS for MIPS… 85` `#> 6 0446157747 USC CARE MEDICAL GROUP INC CAHPS_GRP_12 CAHPS for MIPS… 24` `#> # … with 494 more rows`

연구 대상이 되는 핵심 단위는 조직(organization)이지만, 설문 조사 기관에서 측정한 측정값마다 한 행씩 사용하여 각 조직은 6개의 행에 걸쳐 분산되어 있습니다. <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>를 사용하여 `measure_cd`와 `measure_title`에 대한 전체 값 집합을 볼 수 있습니다.

`cms_patient_experience` `|>` `distinct``(``measure_cd``,` `measure_title``)` `#> # A tibble: 6 × 2` `#> measure_cd measure_title ` `#> <chr> <chr> ` `#> 1 CAHPS_GRP_1 CAHPS for MIPS SSM: Getting Timely Care, Appointments, and In…` `#> 2 CAHPS_GRP_2 CAHPS for MIPS SSM: How Well Providers Communicate ` `#> 3 CAHPS_GRP_3 CAHPS for MIPS SSM: Patient's Rating of Provider ` `#> 4 CAHPS_GRP_5 CAHPS for MIPS SSM: Health Promotion and Education ` `#> 5 CAHPS_GRP_8 CAHPS for MIPS SSM: Courteous and Helpful Office Staff ` `#> 6 CAHPS_GRP_12 CAHPS for MIPS SSM: Stewardship of Patient Resources`

이 두 열 모두 특별히 훌륭한 변수 이름이 되지는 못합니다. `measure_cd`는 변수의 의미를 힌트해주지 않고, `measure_title`은 공백이 포함된 긴 문장입니다. 일단은 새 열 이름의 소스로 `measure_cd`를 사용하겠지만, 실제 분석에서는 짧고 의미 있는 자신만의 변수 이름을 만들고 싶을 것입니다.

<a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>는 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>와 반대되는 인터페이스를 가집니다. 새 열 이름을 선택하는 대신 값(`values_from`)과 열 이름(`names_from`)을 정의하는 기존 열을 제공해야 합니다.

`cms_patient_experience` `|>` `pivot_wider``(` `names_from` `=` `measure_cd``,` `values_from` `=` `prf_rate` `)` `#> # A tibble: 500 × 9` `#> org_pac_id org_nm measure_title CAHPS_GRP_1 CAHPS_GRP_2` `#> <chr> <chr> <chr> <dbl> <dbl>` `#> 1 0446157747 USC CARE MEDICAL GROUP … CAHPS for MIPS… 63 NA` `#> 2 0446157747 USC CARE MEDICAL GROUP … CAHPS for MIPS… NA 87` `#> 3 0446157747 USC CARE MEDICAL GROUP … CAHPS for MIPS… NA NA` `#> 4 0446157747 USC CARE MEDICAL GROUP … CAHPS for MIPS… NA NA` `#> 5 0446157747 USC CARE MEDICAL GROUP … CAHPS for MIPS… NA NA` `#> 6 0446157747 USC CARE MEDICAL GROUP … CAHPS for MIPS… NA NA` `#> # … with 494 more rows, and 4 more variables: CAHPS_GRP_3 <dbl>,` `#> # CAHPS_GRP_5 <dbl>, CAHPS_GRP_8 <dbl>, CAHPS_GRP_12 <dbl>`

출력이 아주 정확해 보이지는 않습니다. 각 조직에 대해 여전히 여러 행이 있는 것 같습니다. 이는 또한 각 행을 고유하게 식별하는 값이 있는 열 또는 열들이 무엇인지 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>에 알려주어야 하기 때문입니다. 이 경우에는 `"org"`로 시작하는 변수들입니다.

`cms_patient_experience` `|>` `pivot_wider``(` `id_cols` `=` `starts_with``(``"org"``),` `names_from` `=` `measure_cd``,` `values_from` `=` `prf_rate` `)` `#> # A tibble: 95 × 8` `#> org_pac_id org_nm CAHPS_GRP_1 CAHPS_GRP_2 CAHPS_GRP_3 CAHPS_GRP_5` `#> <chr> <chr> <dbl> <dbl> <dbl> <dbl>` `#> 1 0446157747 USC CARE MEDICA… 63 87 86 57` `#> 2 0446162697 ASSOCIATION OF … 59 85 83 63` `#> 3 0547164295 BEAVER MEDICAL … 49 NA 75 44` `#> 4 0749333730 CAPE PHYSICIANS… 67 84 85 65` `#> 5 0840104360 ALLIANCE PHYSIC… 66 87 87 64` `#> 6 0840109864 REX HOSPITAL INC 73 87 84 67` `#> # … with 89 more rows, and 2 more variables: CAHPS_GRP_8 <dbl>,` `#> # CAHPS_GRP_12 <dbl>`

이렇게 하면 우리가 찾고 있는 출력을 얻을 수 있습니다.

## pivot_wider()는 어떻게 작동할까요?

<a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>가 어떻게 작동하는지 이해하기 위해 다시 간단한 데이터셋으로 시작해보겠습니다. 이번에는 `id`가 A와 B인 두 명의 환자가 있습니다. 환자 A에 대해 세 번의 혈압 측정이 있고, 환자 B에 대해 두 번의 측정이 있습니다.

`df` `<-` `tribble``(` `~``id``,` `~``measurement``,` `~``value``,` `"A"``,` `"bp1"``,` `100``,` `"B"``,` `"bp1"``,` `140``,` `"B"``,` `"bp2"``,` `115``,` `"A"``,` `"bp2"``,` `120``,` `"A"``,` `"bp3"``,` `105` `)`

우리는 `value` 열에서 값을 가져오고 `measurement` 열에서 이름을 가져올 것입니다.

`df` `|>` `pivot_wider``(` `names_from` `=` `measurement``,` `values_from` `=` `value` `)` `#> # A tibble: 2 × 4` `#> id bp1 bp2 bp3` `#> <chr> <dbl> <dbl> <dbl>` `#> 1 A 100 120 105` `#> 2 B 140 115 NA`

프로세스를 시작하기 위해 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>는 먼저 무엇이 행과 열에 들어갈지 파악해야 합니다. 새 열 이름은 `measurement`의 고유 값이 될 것입니다.

`df` `|>` `distinct``(``measurement``)` `|>` `pull``()` `#> [1] "bp1" "bp2" "bp3"`

기본적으로 출력의 행은 새 이름이나 값으로 들어가지 않는 모든 변수에 의해 결정됩니다. 이들을 `id_cols`라고 부릅니다. 여기에는 열이 하나뿐이지만 일반적으로는 임의의 개수일 수 있습니다.

`df` `|>` `select``(``-``measurement``,` `-``value``)` `|>` `distinct``()` `#> # A tibble: 2 × 1` `#> id ` `#> <chr>` `#> 1 A ` `#> 2 B`

그런 다음 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>는 이러한 결과를 결합하여 빈 데이터 프레임을 생성합니다.

`df` `|>` `select``(``-``measurement``,` `-``value``)` `|>` `distinct``()` `|>` `mutate``(``x` `=` `NA``,` `y` `=` `NA``,` `z` `=` `NA``)` `#> # A tibble: 2 × 4` `#> id x y z ` `#> <chr> <lgl> <lgl> <lgl>` `#> 1 A NA NA NA ` `#> 2 B NA NA NA`

그런 다음 입력의 데이터를 사용하여 누락된 모든 값을 채웁니다. 이 경우 출력의 모든 셀이 입력의 해당 값을 갖는 것은 아닙니다. 환자 B에 대한 세 번째 혈압 측정값이 없으므로 해당 셀은 누락된 채로 남아 있습니다. <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>가 누락된 값을 "만들" 수 있다는 이 아이디어에 대해서는 <a href="ch18.html#chp-missing-values" data-type="xref">18장</a>에서 다시 다룰 것입니다.

입력에 출력의 한 셀에 해당하는 여러 행이 있는 경우 어떻게 되는지 궁금할 수도 있습니다. 다음 예에는 `id A` 및 `measurement bp1`에 해당하는 두 개의 행이 있습니다.

`df` `<-` `tribble``(` `~``id``,` `~``measurement``,` `~``value``,` `"A"``,` `"bp1"``,` `100``,` `"A"``,` `"bp1"``,` `102``,` `"A"``,` `"bp2"``,` `120``,` `"B"``,` `"bp1"``,` `140``,` `"B"``,` `"bp2"``,` `115` `)`

이 데이터를 피벗하려고 하면 리스트 열(list-columns)이 포함된 출력을 얻게 됩니다. 이에 대해서는 <a href="ch23.html#chp-rectangling" data-type="xref">23장</a>에서 자세히 알아볼 것입니다.

`df` `|>` `pivot_wider``(` `names_from` `=` `measurement``,` `values_from` `=` `value` `)` ``#> Warning: Values from `value` are not uniquely identified; output will contain`` `#> list-cols.` ``#> • Use `values_fn = list` to suppress this warning.`` ``#> • Use `values_fn = {summary_fun}` to summarise duplicates.`` `#> • Use the following dplyr code to identify duplicates.` `#> {data} %>%` `#> dplyr::group_by(id, measurement) %>%` `#> dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%` `#> dplyr::filter(n > 1L)` `#> # A tibble: 2 × 3` `#> id bp1 bp2 ` `#> <chr> <list> <list> ` `#> 1 A <dbl [2]> <dbl [1]>` `#> 2 B <dbl [1]> <dbl [1]>`

아직 이런 종류의 데이터로 작업하는 방법을 모르기 때문에, 어디에 문제가 있는지 파악하기 위해 경고 메시지의 힌트를 따르고 싶을 것입니다.

`df` `|>` `group_by``(``id``,` `measurement``)` `|>` `summarize``(``n` `=` `n``(),` `.groups` `=` `"drop"``)` `|>` `filter``(``n` `>` `1``)` `#> # A tibble: 1 × 3` `#> id measurement n` `#> <chr> <chr> <int>` `#> 1 A bp1 2`

그런 다음 데이터에 무엇이 잘못되었는지 파악하고 근본적인 손상을 복구하거나 그룹화 및 요약 기술을 사용하여 행과 열 값의 각 조합이 단일 행만 갖도록 보장하는 것은 여러분의 몫입니다.

# 요약 (Summary)

이 장에서는 변수가 열에 있고 관측치가 행에 있는 데이터인 정돈된 데이터(tidy data)에 대해 배웠습니다. 정돈된 데이터는 대부분의 함수에서 이해되는 일관된 구조이기 때문에 tidyverse에서의 작업을 더 쉽게 만듭니다. 주요 과제는 전달받은 어떤 구조의 데이터든 정돈된 형식으로 변환하는 것입니다. 이를 위해 많은 정돈되지 않은 데이터셋을 정돈할 수 있게 해주는 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>에 대해 배웠습니다. 여기서 제시한 예제는 <a href="https://tidyr.tidyverse.org/articles/pivot.html" class="orm:hideurl"><code>vignette("pivot", package = "tidyr")</code></a>에서 선택한 것이므로, 이 장에서 도움을 받지 못한 문제에 직면한다면 다음에 시도해 볼 좋은 자료가 바로 그 비네트(vignette)입니다.

또 다른 과제는, 주어진 데이터셋에 대해 더 길거나 넓은 버전을 "정돈된(tidy)" 것이라고 명명하는 것이 불가능할 수 있다는 점입니다. 이는 정돈된 데이터가 각 열에 하나의 변수를 가진다고 말했지만 실제로 변수가 무엇인지는 정의하지 않았던(그리고 정의하기가 놀랍도록 어렵습니다) 우리의 정돈된 데이터 정의를 일부 반영하는 것입니다. 분석을 가장 쉽게 만들어주는 것이라면 무엇이든 변수라고 말하는 등 실용적으로 접근해도 전혀 문제가 없습니다. 따라서 어떤 계산을 어떻게 수행할지 알아내는 데 막혔다면 데이터 구성을 전환하는 것을 고려해 보세요. 필요에 따라 정돈을 해제하고, 변환하고, 다시 정돈하는 것을 두려워하지 마세요!

이 장을 즐겁게 읽었고 기본 이론에 대해 더 알고 싶다면, *Journal of Statistical Software*에 게시된 [“Tidy Data” 논문](https://oreil.ly/86uxw)에서 그 역사와 이론적 토대에 대해 자세히 알아볼 수 있습니다.

이제 상당한 양의 R 코드를 작성하고 있으므로 코드를 파일 및 디렉터리로 구성하는 방법에 대해 더 자세히 알아볼 때입니다. 다음 장에서는 스크립트와 프로젝트의 이점과 여러분의 삶을 편하게 만들어 줄 많은 도구에 대해 모두 알아볼 것입니다.

<sup>[1](ch05.html#idm44771326722336-marker)</sup> 2000년 어느 시점에 탑 100에 포함되었고, 차트에 등장한 후 최대 72주까지 추적된 노래라면 포함됩니다.

<sup>[2](ch05.html#idm44771328141216-marker)</sup> 이 아이디어에 대해서는 <a href="ch18.html#chp-missing-values" data-type="xref">18장</a>에서 다시 다루겠습니다.
