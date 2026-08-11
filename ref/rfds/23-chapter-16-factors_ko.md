# 16장. 팩터(Factors)

# 소개

팩터(Factors)는 고정되고 알려진 가능한 값들의 집합을 갖는 변수인 범주형 변수(categorical variables)에 사용됩니다. 또한 문자 벡터를 알파벳 순서가 아닌 순서로 표시하고 싶을 때도 유용합니다.

데이터 분석을 위해 팩터가 왜 필요한지 동기를 부여하고<sup><a href="ch16.html#idm44771290852512" id="idm44771290852512-marker" data-type="noteref">1</a></sup>, <a href="https://rdrr.io/r/base/factor.html" class="orm:hideurl"><code>factor()</code></a>로 어떻게 생성할 수 있는지부터 시작해 보겠습니다. 그런 다음 실험해 볼 수 있는 많은 범주형 변수를 포함하는 `gss_cat` 데이터셋을 소개합니다. 순서형 팩터(ordered factors)에 대한 논의로 마무리하기 전에, 그 데이터셋을 사용하여 팩터의 순서와 값을 수정하는 연습을 하게 될 것입니다.

## 사전 준비

기본 R은 팩터를 생성하고 조작하기 위한 몇 가지 기본 도구를 제공합니다. 우리는 이것들을 핵심 tidyverse의 일부인 forcats 패키지로 보완할 것입니다. 이 패키지는 팩터 작업을 위한 광범위한 도우미들을 사용하여 범주형 변수(*cat*egorical variables)를 다루는 도구를 제공합니다 (그리고 이 패키지 이름은 factors의 철자를 바꾼 애너그램입니다!).

```
library(tidyverse)
```

# 팩터 기초

월(month)을 기록하는 변수가 있다고 상상해 봅시다.

```
x1 <- c("Dec", "Apr", "Jan", "Mar")
```

이 변수를 기록하기 위해 문자열을 사용하면 두 가지 문제가 있습니다.

1. 가능한 월은 12개뿐이며, 오타를 방지해 줄 수 있는 것이 없습니다.

   ```
   x2 <- c("Dec", "Apr", "Jam", "Mar")
   ```

2. 유용한 방식으로 정렬되지 않습니다.

   ```
   sort(x1)
   #> [1] "Apr" "Dec" "Jan" "Mar"
   ```

팩터를 사용하면 이 두 가지 문제를 모두 해결할 수 있습니다. 팩터를 생성하려면 먼저 유효한 *수준(levels)*의 리스트를 생성하는 것부터 시작해야 합니다.

```
month_levels <- c(
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
)
```

이제 팩터를 생성할 수 있습니다.

```
y1 <- factor(x1, levels = month_levels)
y1
#> [1] Dec Apr Jan Mar
#> Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec

sort(y1)
#> [1] Jan Mar Apr Dec
#> Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
```

수준(level)에 없는 모든 값은 아무런 메시지 없이 `NA`로 변환됩니다.

```
y2 <- factor(x2, levels = month_levels)
y2
#> [1] Dec  Apr  <NA> Mar
#> Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
```

이것은 위험해 보이므로, 대신 <a href="https://forcats.tidyverse.org/reference/fct.html" class="orm:hideurl"><code>forcats::fct()</code></a>를 사용하고 싶을 수 있습니다.

```
y2 <- fct(x2, levels = month_levels)
#> Error in `fct()`:
#> ! All values of `x` must appear in `levels` or `na`
#> ℹ Missing level: "Jam"
```

수준을 생략하면, 데이터에서 알파벳 순서로 가져옵니다.

```
factor(x1)
#> [1] Dec Apr Jan Mar
#> Levels: Apr Dec Jan Mar
```

알파벳 순으로 정렬하는 것은 모든 컴퓨터가 동일한 방식으로 문자열을 정렬하지는 않기 때문에 약간 위험합니다. 그래서 <a href="https://forcats.tidyverse.org/reference/fct.html" class="orm:hideurl"><code>forcats::fct()</code></a>는 처음 나타나는 순서대로 정렬합니다.

```
fct(x1)
#> [1] Dec Apr Jan Mar
#> Levels: Dec Apr Jan Mar
```

만약 유효한 수준의 집합에 직접 접근해야 할 필요가 있다면, <a href="https://rdrr.io/r/base/levels.html" class="orm:hideurl"><code>levels()</code></a>로 그렇게 할 수 있습니다.

```
levels(y2)
#>  [1] "Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"
```

readr로 데이터를 읽을 때 <a href="https://readr.tidyverse.org/reference/parse_factor.html" class="orm:hideurl"><code>col_factor()</code></a>를 사용하여 팩터를 생성할 수도 있습니다.

```
csv <- "
month,value
Jan,12
Feb,56
Mar,12"

df <- read_csv(csv, col_types = cols(month = col_factor(month_levels)))
df$month
#> [1] Jan Feb Mar
#> Levels: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
```

# 종합 사회 조사(General Social Survey)

이 장의 나머지 부분에서는 <a href="https://forcats.tidyverse.org/reference/gss_cat.html" class="orm:hideurl"><code>forcats::gss_cat</code></a>을 사용할 것입니다. 이것은 시카고 대학의 독립 연구 기관인 NORC에서 수행하는 장기 미국 설문 조사인 [종합 사회 조사(General Social Survey)](https://oreil.ly/3qBI5)의 데이터 샘플입니다. 설문 조사에는 수천 개의 질문이 있으므로, `gss_cat`에서 해들리(Hadley)는 팩터 작업 시 직면하게 될 몇 가지 일반적인 문제들을 설명해 줄 수 있는 소수의 질문을 선택했습니다.

```
gss_cat
#> # A tibble: 21,483 × 9
#>    year marital         age race  rincome        partyid
#>   <int> <fct>         <int> <fct> <fct>          <fct>
#> 1  2000 Never married    26 White $8000 to 9999  Ind,near rep
#> 2  2000 Divorced         48 White $8000 to 9999  Not str republican
#> 3  2000 Widowed          67 White Not applicable Independent
#> 4  2000 Never married    39 White Not applicable Ind,near rep
#> 5  2000 Divorced         25 White Not applicable Not str democrat
#> 6  2000 Married          25 White $20000 - 24999 Strong democrat
#> # … with 21,477 more rows, and 3 more variables: relig <fct>, denom <fct>,
#> #   tvhours <int>
```

(이 데이터셋은 패키지에서 제공되므로 <a href="https://forcats.tidyverse.org/reference/gss_cat.html" class="orm:hideurl"><code>?gss_cat</code></a>을 통해 변수에 대한 더 많은 정보를 얻을 수 있다는 것을 기억하세요.)

팩터가 티블(tibble)에 저장되어 있을 때는 그 수준(levels)을 쉽게 볼 수 없습니다. 수준을 보는 한 가지 방법은 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>를 사용하는 것입니다.

```
gss_cat |>
  count(race)
#> # A tibble: 3 × 2
#>   race      n
#>   <fct> <int>
#> 1 Other  1959
#> 2 Black  3129
#> 3 White 16395
```

팩터로 작업할 때 가장 일반적인 두 가지 연산은 수준의 순서를 변경하는 것과 수준의 값을 변경하는 것입니다. 이러한 작업은 다음 섹션에 설명되어 있습니다.

## 연습문제

1. `rincome`(보고된 소득)의 분포를 탐색해 보세요. 기본 막대 차트를 이해하기 어렵게 만드는 요인은 무엇인가요? 플롯을 어떻게 개선할 수 있을까요?

2. 이 설문 조사에서 가장 흔한 `relig`(종교)는 무엇인가요? 가장 흔한 `partyid`(정당 식별)는 무엇인가요?

3. `denom`(교파)은 어떤 `relig`에 적용되나요? 표로 어떻게 알아낼 수 있나요? 시각화로 어떻게 알아낼 수 있나요?

# 팩터 순서 수정하기

시각화할 때 팩터 수준의 순서를 변경하는 것이 종종 유용합니다. 예를 들어, 종교별로 하루에 TV를 시청하는 데 보내는 평균 시간을 탐색하고 싶다고 상상해 보세요.

```
relig_summary <- gss_cat |>
  group_by(relig) |>
  summarize(
    tvhours = mean(tvhours, na.rm = TRUE),
    n = n()
  )

ggplot(relig_summary, aes(x = tvhours, y = relig)) +
  geom_point()
```

![x축에 tvhours가 있고 y축에 religion이 있는 산점도. y축이 임의로 정렬된 것처럼 보여 전반적인 패턴을 파악하기 어렵습니다.](D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_16in01.png)

전반적인 패턴이 없기 때문에 이 플롯을 읽기 어렵습니다. <a href="https://forcats.tidyverse.org/reference/fct_reorder.html" class="orm:hideurl"><code>fct_reorder()</code></a>를 사용하여 `relig`의 수준을 재정렬함으로써 이를 개선할 수 있습니다. <a href="https://forcats.tidyverse.org/reference/fct_reorder.html" class="orm:hideurl"><code>fct_reorder()</code></a>는 세 가지 인수를 취합니다.

- `f`: 수준을 수정하려는 팩터.
- `x`: 수준을 재정렬하는 데 사용할 숫자형 벡터.
- 선택적으로 `fun`: `f`의 각 값에 대해 `x` 값이 여러 개 있을 경우 사용되는 함수. 기본값은 `median`입니다.

```
ggplot(relig_summary, aes(x = tvhours, y = fct_reorder(relig, tvhours))) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_16in02.png" alt="위와 동일한 산점도이지만, 이제 종교가 tvhours의 오름차순으로 표시됩니다. &quot;Other eastern(기타 동양 종교)&quot;이 2시간 미만으로 가장 적고, &quot;Don't know(모름)&quot;가 5시간 이상으로 가장 높습니다." />
</figure>

종교를 재정렬하면 "Don't know(모름)" 범주에 있는 사람들이 TV를 훨씬 더 많이 시청하고, 힌두교 및 기타 동양 종교(other Eastern religions) 사람들은 훨씬 적게 시청한다는 것을 파악하기가 훨씬 쉬워집니다.

더 복잡한 변환을 시작할 때는, 이를 <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a> 외부로 빼내어 별도의 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 단계로 옮기는 것을 권장합니다. 예를 들어, 이전 플롯을 다음과 같이 다시 작성할 수 있습니다.

```
relig_summary |>
  mutate(
    relig = fct_reorder(relig, tvhours)
  ) |>
  ggplot(aes(x = tvhours, y = relig)) +
  geom_point()
```

보고된 소득 수준에 따라 평균 연령이 어떻게 달라지는지 보는 유사한 플롯을 만들면 어떨까요?

```
rincome_summary <- gss_cat |>
  group_by(rincome) |>
  summarize(
    age = mean(age, na.rm = TRUE),
    n = n()
  )

ggplot(rincome_summary, aes(x = age, y = fct_reorder(rincome, age))) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_16in03.png" alt="x축에 연령이 있고 y축에 소득이 있는 산점도. 소득이 평균 연령 순서로 재정렬되어 있는데 이는 별로 이치에 맞지 않습니다. y축의 한 부분은 $6000-6999, 그 다음은 &lt;$1000, 그 다음은 $8000-9999로 이어집니다." />
</figure>

여기서는 수준을 임의로 재정렬하는 것이 좋은 생각이 아닙니다! 그 이유는 `rincome`에는 이미 건드리지 말아야 할 원칙적인 순서가 있기 때문입니다. <a href="https://forcats.tidyverse.org/reference/fct_reorder.html" class="orm:hideurl"><code>fct_reorder()</code></a>는 수준이 임의로 정렬된 팩터를 위해 남겨두세요.

하지만 "Not applicable(해당 없음)"을 다른 특별한 수준들과 함께 맨 앞으로 끌어오는 것은 이치에 맞습니다. <a href="https://forcats.tidyverse.org/reference/fct_relevel.html" class="orm:hideurl"><code>fct_relevel()</code></a>을 사용할 수 있습니다. 이 함수는 팩터 `f`를 취하고, 그 다음 줄의 맨 앞으로 이동시킬 수준들을 원하는 만큼 취합니다.

```
ggplot(rincome_summary, aes(x = age, y = fct_relevel(rincome, "Not applicable"))) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_16in04.png" alt="동일한 산점도이지만 이제 &quot;Not Applicable&quot;이 y축 하단에 표시됩니다. 전반적으로 소득과 연령 사이에는 양의 상관관계가 있으며, 평균 연령이 가장 높은 소득 구간은 &quot;Not applicable&quot;입니다." />
</figure>

왜 "Not applicable"의 평균 연령이 그렇게 높다고 생각하시나요?

플롯의 선에 색을 입힐 때 또 다른 유형의 재정렬이 유용합니다. `fct_reorder2(f, x, y)`는 가장 큰 `x` 값과 관련된 `y` 값에 따라 팩터 `f`를 재정렬합니다. 이렇게 하면 플롯의 맨 오른쪽에 있는 선의 색상이 범례와 정렬되어 플롯을 읽기가 더 쉬워집니다.

```
by_age <- gss_cat |>
  filter(!is.na(age)) |>
  count(age, marital) |>
  group_by(age) |>
  mutate(
    prop = n / sum(n)
  )
```

```
ggplot(by_age, aes(x = age, y = prop, color = marital)) +
  geom_line(linewidth = 1) +
  scale_color_brewer(palette = "Set1")

ggplot(by_age, aes(x = age, y = prop, color = fct_reorder2(marital, age, prop))) +
  geom_line(linewidth = 1) +
  scale_color_brewer(palette = "Set1") +
  labs(color = "marital")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_16in05.png" alt="x축에 연령이 있고 y축에 비율이 있는 선형 플롯. 혼인 상태의 각 범주(no answer, never married, separated, divorced, widowed, married)마다 하나의 선이 있습니다. 범례의 순서가 플롯의 선과 관련이 없기 때문에 플롯을 읽기가 약간 어렵습니다. 범례를 재정렬하면 이제 범례 색상이 플롯의 맨 오른쪽에 있는 선의 순서와 일치하므로 플롯을 읽기가 더 쉬워집니다. 놀랍지 않은 몇 가지 패턴을 볼 수 있습니다. 결코 결혼하지 않은(never married) 비율은 나이가 들면서 감소하고, 결혼한(married) 비율은 거꾸로 된 U자 모양을 형성하며, 사별한(widowed) 비율은 낮게 시작하지만 60세 이후에 급격히 증가합니다." />
</figure>

마지막으로 막대 플롯의 경우 <a href="https://forcats.tidyverse.org/reference/fct_inorder.html" class="orm:hideurl"><code>fct_infreq()</code></a>를 사용하여 수준을 빈도 감소 순으로 정렬할 수 있습니다. 추가 변수가 필요하지 않기 때문에 이것이 가장 간단한 유형의 재정렬입니다. 막대 플롯에서 가장 큰 값이 왼쪽이 아닌 오른쪽에 오도록 빈도 증가 순으로 정렬하려면 <a href="https://forcats.tidyverse.org/reference/fct_rev.html" class="orm:hideurl"><code>fct_rev()</code></a>와 결합하세요.

```
gss_cat |>
  mutate(marital = marital |> fct_infreq() |> fct_rev()) |>
  ggplot(aes(x = marital)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_16in06.png" alt="혼인 상태의 빈도를 가장 적은 것부터 가장 많은 것 순으로 정렬한 막대 차트: no answer (~0), separated (~1,000), widowed (~2,000), divorced (~3,000), never married (~5,000), married (~10,000)." />
</figure>

## 연습문제

1. `tvhours`에 의심스러울 정도로 높은 숫자가 몇 개 있습니다. 평균이 좋은 요약일까요?

2. `gss_cat`의 각 팩터에 대해 수준의 순서가 임의적인지 아니면 원칙적인지 파악하세요.

3. "Not applicable(해당 없음)"을 수준의 맨 앞으로 이동시켰을 때 왜 그것이 플롯의 맨 아래로 이동했을까요?

# 팩터 수준 수정하기

수준의 순서를 변경하는 것보다 더 강력한 것은 그 값을 변경하는 것입니다. 이를 통해 출판물을 위한 레이블을 명확히 하고, 상위 수준의 디스플레이를 위해 수준을 축소(collapse)할 수 있습니다. 가장 일반적이고 강력한 도구는 <a href="https://forcats.tidyverse.org/reference/fct_recode.html" class="orm:hideurl"><code>fct_recode()</code></a>입니다. 이 함수는 각 수준의 값을 다시 코딩하거나 변경할 수 있게 해줍니다. 예를 들어 `gss_cat` 데이터 프레임에서 `partyid` 변수를 가져와 보겠습니다.

```
gss_cat |> count(partyid)
#> # A tibble: 10 × 2
#>   partyid                n
#>   <fct>              <int>
#> 1 No answer            154
#> 2 Don't know             1
#> 3 Other party          393
#> 4 Strong republican   2314
#> 5 Not str republican  3032
#> 6 Ind,near rep        1791
#> # … with 4 more rows
```

수준들이 간결하고 일관성이 없습니다. 이것들을 더 길게 다듬고 병렬 구조를 사용해 봅시다. tidyverse의 대부분의 이름 바꾸기 및 다시 코딩하기 함수와 마찬가지로, 새로운 값은 왼쪽에 오고 이전 값은 오른쪽에 옵니다.

```
gss_cat |>
  mutate(
    partyid = fct_recode(partyid,
      "Republican, strong"    = "Strong republican",
      "Republican, weak"      = "Not str republican",
      "Independent, near rep" = "Ind,near rep",
      "Independent, near dem" = "Ind,near dem",
      "Democrat, weak"        = "Not str democrat",
      "Democrat, strong"      = "Strong democrat"
    )
  ) |>
  count(partyid)
#> # A tibble: 10 × 2
#>   partyid                   n
#>   <fct>                 <int>
#> 1 No answer               154
#> 2 Don't know                1
#> 3 Other party             393
#> 4 Republican, strong     2314
#> 5 Republican, weak       3032
#> 6 Independent, near rep  1791
#> # … with 4 more rows
```

<a href="https://forcats.tidyverse.org/reference/fct_recode.html" class="orm:hideurl"><code>fct_recode()</code></a>는 명시적으로 언급되지 않은 수준은 그대로 두고, 존재하지 않는 수준을 실수로 참조하면 경고를 표시합니다.

그룹을 결합하려면 동일한 새 수준에 여러 개의 이전 수준을 할당할 수 있습니다.

```
gss_cat |>
  mutate(
    partyid = fct_recode(partyid,
      "Republican, strong"    = "Strong republican",
      "Republican, weak"      = "Not str republican",
      "Independent, near rep" = "Ind,near rep",
      "Independent, near dem" = "Ind,near dem",
      "Democrat, weak"        = "Not str democrat",
      "Democrat, strong"      = "Strong democrat",
      "Other"                 = "No answer",
      "Other"                 = "Don't know",
      "Other"                 = "Other party"
    )
  )
```

이 기법을 주의해서 사용하세요. 만약 완전히 다른 수준들을 그룹화한다면, 결국 오해의 소지가 있는 결과를 얻게 될 것입니다.

많은 수준을 축소하고 싶다면 <a href="https://forcats.tidyverse.org/reference/fct_collapse.html" class="orm:hideurl"><code>fct_collapse()</code></a>가 <a href="https://forcats.tidyverse.org/reference/fct_recode.html" class="orm:hideurl"><code>fct_recode()</code></a>의 유용한 변형입니다. 각 새로운 변수에 대해 이전 수준들의 벡터를 제공할 수 있습니다.

```
gss_cat |>
  mutate(
    partyid = fct_collapse(partyid,
      "other" = c("No answer", "Don't know", "Other party"),
      "rep" = c("Strong republican", "Not str republican"),
      "ind" = c("Ind,near rep", "Independent", "Ind,near dem"),
      "dem" = c("Not str democrat", "Strong democrat")
    )
  ) |>
  count(partyid)
#> # A tibble: 4 × 2
#>   partyid     n
#>   <fct>   <int>
#> 1 other     548
#> 2 rep      5346
#> 3 ind      8409
#> 4 dem      7180
```

때로는 플롯이나 표를 더 단순하게 만들기 위해 작은 그룹들을 하나로 묶고(lump) 싶을 수 있습니다. 이것이 `fct_lump_*()` 함수 계열이 하는 일입니다. <a href="https://forcats.tidyverse.org/reference/fct_lump.html" class="orm:hideurl"><code>fct_lump_lowfreq()</code></a>는 항상 "Other(기타)"를 가장 작은 범주로 유지하면서 가장 작은 그룹의 범주들을 점진적으로 "Other"로 묶는 간단한 출발점입니다.

```
gss_cat |>
  mutate(relig = fct_lump_lowfreq(relig)) |>
  count(relig)
#> # A tibble: 2 × 2
#>   relig          n
#>   <fct>      <int>
#> 1 Protestant 10846
#> 2 Other      10637
```

이 경우에는 그다지 유용하지 않습니다. 이 설문조사에서 미국인의 대다수가 개신교(Protestant)인 것은 사실이지만, 우리는 아마도 더 많은 세부 사항을 보고 싶을 것입니다! 대신 <a href="https://forcats.tidyverse.org/reference/fct_lump.html" class="orm:hideurl"><code>fct_lump_n()</code></a>을 사용하여 정확히 10개의 그룹을 원한다고 지정할 수 있습니다.

```
gss_cat |>
  mutate(relig = fct_lump_n(relig, n = 10)) |>
  count(relig, sort = TRUE)
#> # A tibble: 10 × 2
#>   relig          n
#>   <fct>      <int>
#> 1 Protestant 10846
#> 2 Catholic    5124
#> 3 None        3523
#> 4 Christian    689
#> 5 Other        458
#> 6 Jewish       388
#> # … with 4 more rows
```

다른 경우에 유용한 <a href="https://forcats.tidyverse.org/reference/fct_lump.html" class="orm:hideurl"><code>fct_lump_min()</code></a> 및 <a href="https://forcats.tidyverse.org/reference/fct_lump.html" class="orm:hideurl"><code>fct_lump_prop()</code></a>에 대해 알아보려면 문서를 읽어보세요.

## 연습문제

1. 자신을 민주당원, 공화당원, 무당파(Independent)로 식별하는 사람들의 비율은 시간이 지남에 따라 어떻게 변했나요?

2. `rincome`을 적은 수의 범주 세트로 어떻게 축소할 수 있을까요?

3. 이전의 `fct_lump` 예제에서 "Other"를 제외하고 9개의 그룹이 있는 것에 주목하세요. 왜 10개가 아닐까요? (힌트: <a href="https://forcats.tidyverse.org/reference/fct_lump.html" class="orm:hideurl"><code>?fct_lump</code></a>를 입력하고, 인수 `other_level`의 기본값이 "Other"인지 찾아보세요.)

# 순서형 팩터(Ordered Factors)

넘어가기 전에, 간단히 언급해야 할 특별한 유형의 팩터가 있습니다. 바로 순서형 팩터(ordered factors)입니다. <a href="https://rdrr.io/r/base/factor.html" class="orm:hideurl"><code>ordered()</code></a>로 생성되는 순서형 팩터는 엄격한 순서와 수준 간의 동일한 거리를 암시합니다. 즉, 첫 번째 수준은 두 번째 수준보다 "작고", 그 작은 정도는 두 번째 수준이 세 번째 수준보다 "작은" 정도와 같습니다. 이를 출력해 보면 팩터 수준들 사이에 `<`를 사용하므로 인식할 수 있습니다.

```
ordered(c("a", "b", "c"))
#> [1] a b c
#> Levels: a < b < c
```

실제로는 <a href="https://rdrr.io/r/base/factor.html" class="orm:hideurl"><code>ordered()</code></a> 팩터는 일반 팩터와 유사하게 동작합니다. 다른 동작을 눈치챌 수 있는 곳은 두 군데뿐입니다.

- 만약 ggplot2에서 순서형 팩터를 color나 fill에 매핑하면, 순위를 암시하는 색상 스케일인 `scale_color_viridis()`/`scale_fill_viridis()`가 기본값으로 적용됩니다.
- 선형 모델(linear model)에서 순서형 팩터를 사용하면 "다항 대비(polygonal contrasts)"를 사용합니다. 이는 다소 유용하지만, 통계학 박사 학위가 없는 한 들어본 적이 없을 가능성이 높고, 있더라도 아마 일상적으로 해석하지는 않을 것입니다. 더 배우고 싶다면 Lisa DeBruine의 `vignette("contrasts", package = "faux")`를 추천합니다.

이러한 차이점들의 유용성에 논란의 여지가 있으므로, 우리는 일반적으로 순서형 팩터 사용을 권장하지 않습니다.

# 요약

이 장에서는 가장 일반적으로 사용되는 기능들을 설명하면서 팩터 작업을 위해 편리한 forcats 패키지를 소개했습니다. forcats에는 여기에 논의할 공간이 없었던 다른 광범위한 도우미들이 포함되어 있습니다. 따라서 이전에 접해본 적 없는 팩터 분석 문제에 직면할 때마다 [참조 인덱스](https://oreil.ly/J_IIg)를 훑어보면서 문제를 해결하는 데 도움이 되는 미리 만들어진 함수가 있는지 확인해 볼 것을 적극 권장합니다.

이 장을 읽은 후 팩터에 대해 더 배우고 싶다면, Amelia McNamara와 Nicholas Horton의 논문인 ["Wrangling categorical data in R"](https://oreil.ly/zPh8E)을 읽어보시길 권장합니다. 이 논문은 ["stringsAsFactors: An unauthorized biography"](https://oreil.ly/Z9mkP)와 ["stringsAsFactors = \<sigh\>"](https://oreil.ly/phWQo)에서 논의된 일부 역사를 제시하고, 이 책에 요약된 범주형 데이터에 대한 단정한(tidy) 접근 방식과 기본 R 메서드를 비교합니다. 이 논문의 초기 버전은 forcats 패키지의 동기를 부여하고 범위를 정하는 데 도움이 되었습니다. 고마워요, Amelia와 Nick!

다음 장에서는 R의 날짜와 시간(dates and times)에 대해 배우기 위해 방향을 전환할 것입니다. 날짜와 시간은 속기 쉬울 정도로 단순해 보이지만, 곧 알게 되시겠지만 그것에 대해 더 많이 배울수록 점점 더 복잡해 보입니다!

<sup>[1](ch16.html#idm44771290852512-marker)</sup> 모델링을 위해서도 정말 중요합니다.
