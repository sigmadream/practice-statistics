# 9장. 레이어 (Layers)

# 소개 (Introduction)

<a href="ch01.html#chp-data-visualize" data-type="xref">1장</a>에서는 단순한 산점도(scatterplots), 막대 차트(bar charts), 상자 그림(boxplots)을 만드는 방법보다 훨씬 더 많은 것을 배웠습니다. ggplot2로 _모든_ 종류의 플롯을 만드는 데 사용할 수 있는 기본 토대를 배웠습니다.

이 장에서는 그래픽의 레이어 문법(layered grammar of graphics)에 대해 배우면서 그 토대를 확장할 것입니다. 먼저 심미적 매핑(aesthetic mappings), 기하학적 객체(geometric objects), 패싯(facets)에 대해 더 깊이 파고드는 것으로 시작합니다. 그런 다음 ggplot2가 플롯을 만들 때 내부적으로(under the hood) 수행하는 통계적 변환(statistical transformations)에 대해 알아봅니다. 이러한 변환은 막대 플롯의 막대 높이나 상자 그림의 중앙값과 같이 플롯에 그릴 새로운 값을 계산하는 데 사용됩니다. 또한 플롯에서 기하학적 객체(geom)가 표시되는 방식을 수정하는 위치 조정(position adjustments)에 대해서도 알아봅니다. 마지막으로 좌표계(coordinate systems)를 간략하게 소개합니다.

이러한 각 레이어에 대한 모든 단일 기능과 옵션을 다루지는 않지만, ggplot2에서 제공하는 가장 중요하고 흔히 사용되는 기능을 단계별로 안내하고 ggplot2를 확장하는 패키지들을 소개할 것입니다.

## 사전 준비 (Prerequisites)

이 장은 ggplot2에 중점을 둡니다. 이 장에서 사용되는 데이터셋, 도움말 페이지 및 함수에 액세스하려면 다음 코드를 실행하여 tidyverse를 로드하세요.

```
library(tidyverse)
```

# 심미적 매핑 (Aesthetic Mappings)

> “그림의 가장 큰 가치는 우리가 전혀 볼 것이라 예상하지 못했던 것을 주목하게 만들 때 나타난다.” — 존 튜키(John Tukey)

ggplot2 패키지에 번들로 포함된 `mpg` 데이터 프레임에는 38개 자동차 모델에 대한 234개의 관측치가 포함되어 있다는 것을 기억하세요.

```
mpg
#> # A tibble: 234 × 11
#>   manufacturer model displ  year   cyl trans      drv     cty   hwy fl
#>   <chr>        <chr> <dbl> <int> <int> <chr>      <chr> <int> <int> <chr>
#> 1 audi         a4      1.8  1999     4 auto(l5)   f        18    29 p
#> 2 audi         a4      1.8  1999     4 manual(m5) f        21    29 p
#> 3 audi         a4      2    2008     4 manual(m6) f        20    31 p
#> 4 audi         a4      2    2008     4 auto(av)   f        21    30 p
#> 5 audi         a4      2.8  1999     6 auto(l5)   f        16    26 p
#> 6 audi         a4      2.8  1999     6 manual(m5) f        18    26 p
#> # … with 228 more rows, and 1 more variable: class <chr>
```

`mpg`의 변수 중에는 다음이 있습니다.

`displ`  
자동차의 엔진 크기, 리터 단위입니다. 수치형 변수(numerical variable)입니다.

`hwy`  
자동차의 고속도로 연비, 갤런당 마일(mpg) 단위입니다. 연비가 낮은 자동차는 같은 거리를 주행할 때 연비가 높은 자동차보다 더 많은 연료를 소비합니다. 수치형 변수입니다.

`class`  
자동차 유형. 범주형 변수(categorical variable)입니다.

다양한 자동차 `class`(클래스)에 대해 `displ`과 `hwy`의 관계를 시각화하는 것으로 시작해 보겠습니다. 수치형 변수는 `x`와 `y` 심미성(aesthetic)에 매핑되고 범주형 변수는 `color`나 `shape`와 같은 심미성에 매핑되는 산점도를 사용해 이 작업을 수행할 수 있습니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy, color = class)) +
  geom_point()

# Right
ggplot(mpg, aes(x = displ, y = hwy, shape = class)) +
  geom_point()
#> Warning: The shape palette can deal with a maximum of 6 discrete values
#> because more than 6 becomes difficult to discriminate; you have 7.
#> Consider specifying shapes manually if you must have them.
#> Warning: Removed 62 rows containing missing values (`geom_point()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in01.png" alt="왼쪽 그림은 색상을 활용하고 오른쪽 그림은 모양을 활용한 산점도." />
</figure>

`class`가 `shape`에 매핑될 때 두 가지 경고가 표시됩니다.

> 1: The shape palette can deal with a maximum of 6 discrete values because more than 6 becomes difficult to discriminate; you have 7. Consider specifying shapes manually if you must have them.
>
> 2: Removed 62 rows containing missing values (<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>).

ggplot2는 기본적으로 한 번에 6개의 모양(shape)만 사용하기 때문에 모양 심미성을 사용할 때 추가 그룹은 플롯에 표시되지 않습니다. 두 번째 경고는 이와 관련이 있습니다. 데이터셋에 62대의 SUV가 있는데 그것들이 그려지지 않은 것입니다.

마찬가지로 점의 모양과 투명도를 각각 제어하는 `size`나 `alpha` 심미성에도 `class`를 매핑할 수 있습니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy, size = class)) +
  geom_point()
#> Warning: Using size for a discrete variable is not advised.

# Right
ggplot(mpg, aes(x = displ, y = hwy, alpha = class)) +
  geom_point()
#> Warning: Using alpha for a discrete variable is not advised.
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in02.png" alt="왼쪽 그림은 크기를 활용하고 오른쪽 그림은 알파값을 활용한 산점도." />
</figure>

이 둘 모두 다음과 같은 경고를 생성합니다.

> Using alpha for a discrete variable is not advised. (이산형 변수에 alpha를 사용하는 것은 권장되지 않습니다.)

순서가 없는 이산형(범주형) 변수(`class`)를 순서가 있는 심미성(`size`나 `alpha`)에 매핑하는 것은 실제로는 존재하지 않는 순위가 있음을 암시하므로 일반적으로 좋은 생각이 아닙니다.

일단 심미성을 매핑하면 나머지는 ggplot2가 알아서 처리합니다. 심미성과 함께 사용할 합리적인 척도(scale)를 선택하고 레벨과 값 사이의 매핑을 설명하는 범례(legend)를 구성합니다. x 및 y 심미성에 대해 ggplot2는 범례를 만들지 않지만, 눈금 마크(tick marks)와 레이블이 있는 축선(axis line)을 만듭니다. 축선은 범례와 동일한 정보를 제공합니다; 즉, 위치와 값 사이의 매핑을 설명합니다.

모양(appearance)을 결정하기 위해 변수 매핑에 의존하는 대신 geom 함수의 인수(<a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a> _외부_)로서 geom의 시각적 속성을 수동으로 설정할 수도 있습니다. 예를 들어, 플롯의 모든 점을 파란색으로 만들 수 있습니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(color = "blue")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in03.png" alt="모든 점이 파란색인 산점도." />
</figure>

여기서 색상은 변수에 대한 정보를 전달하지 않고 플롯의 모양만 바꿉니다. 해당 심미성에 적합한 값을 선택해야 합니다.

- 색상의 이름은 문자열로 지정합니다 (`color = "blue"`)
- 점의 크기는 mm 단위로 지정합니다 (`size = 1`)
- 점의 모양은 숫자로 지정합니다 (`shape = 1` - <a href="#fig-shapes" data-type="xref">그림 9-1</a> 참조)

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0901.png" alt="모양과 이를 나타내는 숫자 간의 매핑 다이어그램." />
<h6 id="figure-9-1.-r-has-25-built-in-shapes-that-are-identified-by-numbers.-there-are-some-seeming-duplicates-for-example-0-15-and-22-are-all-squares.-the-difference-comes-from-the-interaction-of-the-color-and-fill-aesthetics.-the-hollow-shapes-014-have-a-border-determined-by-color-the-solid-shapes-1520-are-filled-with-color-and-the-filled-shapes-2124-have-a-border-of-color-and-are-filled-with-fill.-shapes-are-arranged-to-keep-similar-shapes-next-to-each-other.">그림 9-1. R에는 숫자로 식별되는 25개의 기본 제공 모양이 있습니다. 겉보기에는 중복되는 것들이 있습니다. 예를 들어 0, 15, 22는 모두 사각형입니다. 차이점은 <code>color</code>와 <code>fill</code> 심미성의 상호 작용에서 비롯됩니다. 속이 빈 모양(0~14)은 <code>color</code>로 테두리가 결정되고, 단색 모양(15~20)은 <code>color</code>로 채워지며, 채워진 모양(21~24)은 <code>color</code>로 테두리가 있고 <code>fill</code>로 채워집니다. 비슷한 모양끼리 서로 옆에 있도록 배열되어 있습니다.</h6>
</figure>

지금까지 점 geom을 사용할 때 산점도에서 매핑하거나 설정할 수 있는 심미성에 대해 논의했습니다. [심미성 명세 비네트(aesthetic specifications vignette)](https://oreil.ly/SP6zV)에서 가능한 모든 심미성 매핑에 대해 더 자세히 알아볼 수 있습니다.

플롯에 사용할 수 있는 특정 심미성은 데이터를 표현하는 데 사용하는 geom에 따라 다릅니다. 다음 섹션에서는 geom에 대해 더 깊이 살펴봅니다.

## 연습문제

1.  점이 분홍색으로 채워진 삼각형인 `hwy` 대 `displ` 산점도를 만드세요.

2.  다음 코드가 파란색 점이 있는 플롯을 생성하지 않은 이유는 무엇입니까?

    ```
    ggplot(mpg) +
      geom_point(aes(x = displ, y = hwy, color = "blue"))
    ```

3.  `stroke` 심미성은 무엇을 합니까? 어떤 모양에 작동합니까? (힌트: <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>?geom_point</code></a>를 사용하세요.)

4.  심미성을 변수 이름이 아닌 다른 것(`aes(color = displ < 5)`)에 매핑하면 어떻게 됩니까? 참고로 x와 y도 지정해야 합니다.

# 기하학적 객체 (Geometric Objects)

이 두 플롯은 어떻게 비슷할까요?

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in04.png" alt="왼쪽 그림은 산점도이고 오른쪽 그림은 평활 곡선을 보여줍니다." />
</figure>

두 플롯 모두 동일한 x 변수와 동일한 y 변수를 포함하며 둘 다 동일한 데이터를 설명합니다. 그러나 플롯이 동일하지는 않습니다. 각 플롯은 데이터를 표현하기 위해 서로 다른 기하학적 객체(geometric object), 즉 geom을 사용합니다. 왼쪽 플롯은 점 geom을 사용하고 오른쪽 플롯은 데이터에 적합된 부드러운 선인 평활(smooth) geom을 사용합니다.

플롯의 geom을 변경하려면 <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>에 추가하는 geom 함수를 변경합니다. 예를 들어 이전 플롯을 만들려면 다음 코드를 사용할 수 있습니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point()

# Right
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_smooth()
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

ggplot2의 모든 geom 함수는 geom 레이어에 로컬로 정의되거나 <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> 레이어에 전역으로 정의된 `mapping` 인수를 사용합니다. 하지만 모든 심미성이 모든 geom에서 작동하는 것은 아닙니다. 점의 모양은 설정할 수 있지만 선의 "모양"은 설정할 수 없습니다. 시도하면 ggplot2는 해당 심미적 매핑을 조용히 무시할 것입니다. 반면에 선의 선종류(linetype)는 설정*할 수 있습니다*. <a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>는 선종류에 매핑하는 변수의 각 고유 값에 대해 서로 다른 선종류를 가진 다른 선을 그립니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy, shape = drv)) +
  geom_smooth()

# Right
ggplot(mpg, aes(x = displ, y = hwy, linetype = drv)) +
  geom_smooth()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in05.png" alt="왼쪽 그림은 모양에 매핑을 시도했고 오른쪽 그림은 선종류(linetype)에 매핑한 결과." />
</figure>

여기서 <a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>는 자동차의 구동계를 설명하는 `drv` 값을 기준으로 자동차를 세 개의 선으로 분리합니다. 하나의 선은 값이 `4`인 모든 점을 설명하고, 다른 하나는 값이 `f`인 모든 점을 설명하며, 마지막 선은 값이 `r`인 모든 점을 설명합니다. 여기서 `4`는 4륜 구동(four-wheel drive), `f`는 전륜 구동(front-wheel drive), `r`은 후륜 구동(rear-wheel drive)을 의미합니다.

이것이 이상하게 들린다면 원시 데이터(raw data) 위에 선을 오버레이(overlaying)한 다음 모든 것을 `drv`에 따라 색칠하여 더 명확하게 만들 수 있습니다.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  geom_smooth(aes(linetype = drv))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in06.png" alt="점과 평활 곡선이 함께 그려져 구동계별로 구분된 플롯." />
</figure>

이 플롯에는 동일한 그래프에 두 개의 geom이 포함되어 있습니다.

<a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>와 같은 많은 geom은 단일 기하학적 객체를 사용하여 여러 데이터 행을 표시합니다. 이러한 geom의 경우 `group` 심미성을 범주형 변수로 설정하여 여러 객체를 그릴 수 있습니다. ggplot2는 그룹화 변수(grouping variable)의 각 고유 값에 대해 별도의 객체를 그립니다. 실제로는 `linetype` 예제에서처럼 이산형 변수에 심미성을 매핑할 때마다 ggplot2가 이러한 geom에 대해 자동으로 데이터를 그룹화합니다. `group` 심미성 자체는 geom에 범례나 식별 특징을 추가하지 않기 때문에 이 기능에 의존하는 것이 편리합니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_smooth()

# Middle
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_smooth(aes(group = drv))

# Right
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_smooth(aes(color = drv), show.legend = FALSE)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in07.png" alt="그룹 매핑에 따른 평활 곡선의 분리를 보여주는 세 개의 플롯." />
</figure>

geom 함수 내에 매핑을 배치하면 ggplot2는 해당 매핑을 레이어에 대한 로컬 매핑으로 취급합니다. 해당 매핑을 사용하여 _해당 레이어에 대해서만_ 전역 매핑을 확장하거나 덮어씁니다(overwrite). 이를 통해 서로 다른 레이어에 서로 다른 심미성을 표시할 수 있습니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in08.png" alt="점은 차종별로 색칠하고 평활 곡선은 전체 데이터에 적용한 산점도." />
</figure>

동일한 아이디어를 사용하여 각 레이어에 서로 다른 `data`를 지정할 수도 있습니다. 여기서는 2인승 자동차(2seater cars)를 강조하기 위해 속이 빈 원(open circles)과 함께 빨간색 점을 사용합니다. <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>의 로컬 데이터 인수는 해당 레이어에 대해서만 <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>의 전역 데이터 인수를 재정의(override)합니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  geom_point(
    data = mpg |> filter(class == "2seater"),
    color = "red"
  ) +
  geom_point(
    data = mpg |> filter(class == "2seater"),
    shape = "circle open", size = 3, color = "red"
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in09.png" alt="특정 클래스의 차들을 빨간 점과 속이 빈 원으로 강조한 산점도." />
</figure>

<a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>는 데이터셋에서 고유한 모든 행을 찾으므로 기술적인 관점에서는 주로 행에 작용합니다. 하지만 대부분의 경우 일부 변수들의 고유한 조합을 원할 것이므로, 선택적으로 열 이름을 제공할 수도 있습니다.

```
# Remove duplicate rows, if any
flights |>
  distinct()
#> # A tibble: 336,776 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …

# Find all unique origin and destination pairs
flights |>
  distinct(origin, dest)
#> # A tibble: 224 × 2
#>   origin dest
#>   <chr>  <chr>
#> 1 EWR    IAH
#> 2 LGA    IAH
#> 3 JFK    MIA
#> 4 JFK    BQN
#> 5 LGA    ATL
#> 6 EWR    ORD
#> # … with 218 more rows
```

대안으로, 고유한 행을 필터링할 때 다른 열을 유지하고 싶다면 `.keep_all = TRUE` 옵션을 사용할 수 있습니다.

```
flights |>
  distinct(origin, dest, .keep_all = TRUE)
#> # A tibble: 224 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 218 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

고유한 항공편이 모두 1월 1일에 있다는 것은 우연이 아닙니다. <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>는 데이터셋에서 고유한 행의 첫 번째 발생을 찾고 나머지는 버리기 때문입니다.

대신 발생 횟수를 찾으려면 <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>를 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>로 바꾸는 것이 좋으며, `sort = TRUE` 인자를 사용하면 발생 횟수의 내림차순으로 정렬할 수 있습니다. <a href="ch13.html#sec-counts" data-type="xref">"개수(Counts)"</a>에서 개수에 대해 더 배울 것입니다.

```
flights |>
  count(origin, dest, sort = TRUE)
#> # A tibble: 224 × 3
#>   origin dest      n
#>   <chr>  <chr> <int>
#> 1 JFK    LAX   11262
#> 2 LGA    ATL   10263
#> 3 LGA    ORD    8857
#> 4 JFK    SFO    8204
#> 5 LGA    CLT    6168
#> 6 EWR    ORD    6100
#> # … with 218 more rows
```

## 연습 문제

1.  각 조건에 대해 단일 파이프라인에서 해당 조건을 충족하는 모든 항공편을 찾으세요.
    - 도착 지연이 2시간 이상이었습니다.
    - 휴스턴(`IAH` 또는 `HOU`)으로 비행했습니다.
    - United, American 또는 Delta 항공이 운항했습니다.
    - 여름(7월, 8월, 9월)에 출발했습니다.
    - 도착은 2시간 이상 늦었지만 출발은 늦지 않았습니다.
    - 1시간 이상 지연되었지만 비행 중 30분 이상을 만회했습니다.

2.  `flights`를 정렬하여 출발 지연이 가장 긴 항공편을 찾으세요. 아침에 가장 일찍 출발한 항공편을 찾으세요.

3.  `flights`를 정렬하여 가장 빠른 항공편을 찾으세요. (힌트: 함수 내부에 수학 계산을 포함해 보세요.)

4.  2013년 매일 항공편이 있었나요?

5.  가장 먼 거리를 이동한 항공편은 무엇인가요? 가장 짧은 거리를 이동한 항공편은 무엇인가요?

6.  <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>를 둘 다 사용하는 경우 어떤 순서로 사용하든 상관이 있나요? 왜 그런가요/아닌가요? 결과와 함수가 해야 할 작업량에 대해 생각해 보세요.

# 열 (Columns)

행을 변경하지 않고 열에 영향을 미치는 4가지 중요한 동사가 있습니다. <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>는 기존 열에서 파생된 새 열을 생성하고, <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>는 존재하는 열을 변경하며, <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename()</code></a>은 열의 이름을 변경하고, <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a>는 열의 위치를 변경합니다.

## mutate()

<a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>의 역할은 기존 열에서 계산된 새 열을 추가하는 것입니다. 변환 관련 장들에서 여러 유형의 변수를 조작하는 데 사용할 수 있는 많은 함수 집합을 배울 것입니다. 지금은 지연된 비행이 공중에서 시간을 얼마나 만회했는지를 나타내는 `gain`과 시간당 마일 단위의 `speed`를 계산할 수 있게 해주는 기본 대수(algebra)를 고수하겠습니다.

```
flights |>
  mutate(
    gain = dep_delay - arr_delay,
    speed = distance / air_time * 60
  )
#> # A tibble: 336,776 × 21
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 13 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

기본적으로 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>는 데이터셋의 오른쪽에 새 열을 추가하므로 여기서는 무슨 일이 일어나고 있는지 확인하기 어렵습니다. `.before` 인자를 사용하여 대신 왼쪽에 변수를 추가할 수 있습니다.<sup><a href="ch03.html#idm44771330064672" id="idm44771330064672-marker" data-type="noteref">2</a></sup>

```
flights |>
  mutate(
    gain = dep_delay - arr_delay,
    speed = distance / air_time * 60,
    .before = 1
  )
#> # A tibble: 336,776 × 21
#>    gain speed  year month   day dep_time sched_dep_time dep_delay arr_time
#>   <dbl> <dbl> <int> <int> <int>    <int>          <int>     <dbl>    <int>
#> 1    -9  370.  2013     1     1      517            515         2      830
#> 2   -16  374.  2013     1     1      533            529         4      850
#> 3   -31  408.  2013     1     1      542            540         2      923
#> 4    17  517.  2013     1     1      544            545        -1     1004
#> 5    19  394.  2013     1     1      554            600        -6      812
#> 6   -16  288.  2013     1     1      554            558        -4      740
#> # … with 336,770 more rows, and 12 more variables: sched_arr_time <int>,
#> #   arr_delay <dbl>, carrier <chr>, flight <int>, tailnum <chr>, …
```

`.`은 `.before`가 함수에 대한 인자이며, 우리가 생성하는 세 번째 새 변수의 이름이 아니라는 것을 나타내는 표시입니다. `.after`를 사용하여 변수 뒤에 추가할 수도 있으며, `.before`와 `.after` 모두 위치 대신 변수 이름을 사용할 수 있습니다. 예를 들어 `day` 뒤에 새 변수를 추가할 수 있습니다.

```
flights |>
  mutate(
    gain = dep_delay - arr_delay,
    speed = distance / air_time * 60,
    .after = day
  )
```

또는 `.keep` 인자를 사용하여 유지할 변수를 제어할 수 있습니다. 특히 유용한 인자는 `"used"`로, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 단계에 관여했거나 생성된 열만 유지하도록 지정합니다. 예를 들어 다음 출력에는 `dep_delay`, `arr_delay`, `air_time`, `gain`, `hours`, `gain_per_hour` 변수만 포함됩니다.

```
flights |>
  mutate(
    gain = dep_delay - arr_delay,
    hours = air_time / 60,
    gain_per_hour = gain / hours,
    .keep = "used"
  )
```

이전 계산 결과를 `flights`에 다시 할당하지 않았기 때문에 새 변수 `gain`, `hours`, `gain_per_hour`는 출력만 되고 데이터 프레임에 저장되지 않는다는 점에 유의하세요. 그리고 나중에 사용할 수 있도록 데이터 프레임에 포함되게 하려면 결과를 `flights`에 다시 할당하여 원래 데이터 프레임을 훨씬 더 많은 변수로 덮어쓸 것인지, 아니면 새 객체에 할당할 것인지 신중하게 생각해야 합니다. 대개는 올바른 정답이 그 내용을 나타내는 정보 제공용 이름(`delay_gain`)으로 지정된 새 객체이지만, `flights`를 덮어쓸 합당한 이유가 있을 수도 있습니다.

## select()

수백 개나 심지어 수천 개의 변수를 가진 데이터셋을 얻는 것은 흔한 일입니다. 이 상황에서 첫 번째 과제는 보통 관심 있는 변수에만 집중하는 것입니다. <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>를 사용하면 변수의 이름에 기반한 작업을 사용하여 유용한 부분 집합(subset)을 빠르게 확대할 수 있습니다.

- 이름으로 열 선택:

  ```
  flights |>
    select(year, month, day)
  ```

- year와 day 사이의 모든 열 선택 (포함):

  ```
  flights |>
    select(year:day)
  ```

- year부터 day까지(포함)를 제외한 모든 열 선택:

  ```
  flights |>
    select(!year:day)
  ```

  `!` 대신 `-`를 사용할 수도 있습니다(그리고 실무에서 그렇게 쓰는 것을 자주 볼 수 있습니다). 하지만 우리는 `!`를 추천하는데, "아니다(not)"로 읽히며 `&` 및 `|`와 잘 결합되기 때문입니다.

- 문자인 모든 열 선택:

  ```
  flights |>
    select(where(is.character))
  ```

<a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a> 내에서 사용할 수 있는 헬퍼(helper) 함수들이 많이 있습니다.

`starts_with("abc")`  
"abc"로 시작하는 이름과 일치

`ends_with("xyz")`  
"xyz"로 끝나는 이름과 일치

`contains("ijk")`  
"ijk"를 포함하는 이름과 일치

`num_range("x", 1:3)`  
`x1`, `x2`, `x3`와 일치

자세한 내용은 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>?select</code></a>를 참조하세요. 정규 표현식(<a href="ch15.html#chp-regexps" data-type="xref">15장</a>의 주제)을 알게 되면 <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>matches()</code></a>를 사용하여 패턴과 일치하는 변수를 선택할 수도 있습니다.

`=`를 사용하여 변수를 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>하면서 변수의 이름을 바꿀 수 있습니다. 새 이름은 `=`의 왼쪽에 나타나고 기존 변수는 오른쪽에 나타납니다.

```
flights |>
  select(tail_num = tailnum)
#> # A tibble: 336,776 × 1
#>   tail_num
#>   <chr>
#> 1 N14228
#> 2 N24211
#> 3 N619AA
#> 4 N804JB
#> 5 N668DN
#> 6 N39463
#> # … with 336,770 more rows
```

## rename()

기존 변수를 모두 유지하고 몇 가지 이름만 바꾸고 싶다면 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a> 대신 <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename()</code></a>을 사용할 수 있습니다.

```
flights |>
  rename(tail_num = tailnum)
#> # A tibble: 336,776 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
```

# Left

ggplot(mpg, aes(x = drv, color = drv)) +
geom_bar()

# Right

ggplot(mpg, aes(x = drv, fill = drv)) +
geom_bar()

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in20.png" alt="자동차의 구동계 유형을 나타내는 두 개의 막대 차트. 첫 번째 플롯에서 막대는 색상 테두리를 가집니다. 두 번째 플롯에서는 색상으로 채워져 있습니다. 막대의 높이는 각 구동계 범주의 자동차 수에 해당합니다." />
</figure>

`fill` 심미성을 `class`와 같은 다른 변수에 매핑하면 어떻게 되는지 확인해 보세요. 막대가 자동으로 누적(stacked)됩니다. 각 색상 직사각형은 `drv`와 `class`의 조합을 나타냅니다.

```

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar()

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in21.png" alt="자동차의 구동계 유형을 나타내는 분할된(누적) 막대 차트로, 각 막대는 자동차 클래스 색상으로 채워져 있습니다. 막대의 높이는 각 구동 범주의 자동차 수에 해당하고, 색상 세그먼트의 높이는 주어진 구동계 유형 수준 내에서 특정 클래스 수준을 가진 자동차의 수에 비례합니다." />
</figure>

누적은 `position` 인수로 지정된 *위치 조정(position adjustment)*을 사용하여 자동으로 수행됩니다. 누적 막대 차트를 원하지 않으면 `"identity"`, `"dodge"`, `"fill"`의 세 가지 다른 옵션 중 하나를 사용할 수 있습니다.

- `position = "identity"`는 각 객체를 그래프 문맥에 맞는 바로 그 위치에 정확히 배치합니다. 막대 차트의 경우 막대들이 서로 겹치게(overlap) 되므로 별로 유용하지 않습니다. 겹치는 것을 보려면 `alpha`를 작은 값으로 설정하여 막대를 약간 투명하게 만들거나 `fill = NA`로 설정하여 완전히 투명하게 만들어야 합니다.

```

# Left

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar(alpha = 1/5, position = "identity")

# Right

ggplot(mpg, aes(x = drv, color = class)) +
geom_bar(fill = NA, position = "identity")

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in22.png" alt="위치 조정이 identity인 막대 차트로 인해 겹쳐 보이는 플롯." />
</figure>

identity 위치 조정은 2D geom(점)에 더 유용하며 이것이 점의 기본값입니다.

- `position = "fill"`은 누적과 동일하게 작동하지만 누적된 각 막대 세트의 높이를 같게 만듭니다. 이렇게 하면 그룹 간 비율(proportions)을 더 쉽게 비교할 수 있습니다.

- `position = "dodge"`는 겹치는 객체를 서로 *나란히(beside)* 배치합니다. 이렇게 하면 개별 값을 더 쉽게 비교할 수 있습니다.

```

# Left

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar(position = "fill")

# Right

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar(position = "dodge")

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in23.png" alt="왼쪽 플롯은 비율을 보여주는 fill 위치 조정을, 오른쪽 플롯은 병렬로 배치된 dodge 위치 조정을 보여줍니다." />
</figure>

막대 차트에는 유용하지 않지만 산점도에는 매우 유용할 수 있는 또 다른 유형의 조정이 있습니다. 첫 번째 산점도를 떠올려보세요. 데이터셋에 234개의 관측치가 있음에도 불구하고 플롯에는 126개의 점만 표시된다는 것을 눈치채셨나요?

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in24.png" alt="음의 상관관계를 보여주는 자동차의 고속도로 연비 대 엔진 크기 산점도." />
</figure>

`hwy` 및 `displ`의 기본 값은 점이 그리드에 나타나도록 반올림되어 있으며 많은 점이 서로 겹칩니다. 이 문제를 *오버플로팅(overplotting)*이라고 합니다. 이러한 배치로 인해 데이터의 분포를 파악하기 어렵습니다. 데이터 점이 그래프 전체에 고르게 퍼져 있는지, 아니면 109개의 값을 포함하는 `hwy`와 `displ`의 하나의 특별한 조합이 있는지 알 수 없습니다.

위치 조정을 `"jitter"`로 설정하면 이 그리드 현상(gridding)을 피할 수 있습니다. `position = "jitter"`를 사용하면 각 점에 약간의 무작위 노이즈(random noise)를 추가합니다. 두 점이 동일한 양의 임의 노이즈를 받을 가능성이 없기 때문에 점들이 넓게 퍼지게 됩니다.

```

ggplot(mpg, aes(x = displ, y = hwy)) +
geom_point(position = "jitter")

````

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in25.png" alt="음의 상관관계를 보여주는 자동차의 고속도로 연비 대 엔진 크기의 지터 처리된(Jittered) 산점도." />
</figure>

무작위성을 추가하는 것은 플롯을 개선하는 이상한 방법처럼 보이지만, 작은 축척(small scales)에서는 그래프의 정확성을 떨어뜨리는 대신 큰 축척(large scales)에서는 그래프가 *더 많은 것*을 보여주게 만듭니다. 이것은 매우 유용한 작업이기 때문에 ggplot2는 `geom_point(position = "jitter")`에 대한 단축어인 <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>를 함께 제공합니다.

위치 조정에 대해 더 자세히 알아보려면 각 조정과 연관된 도움말 페이지를 찾아보세요.

- <a href="https://ggplot2.tidyverse.org/reference/position_dodge.html" class="orm:hideurl"><code>?position_dodge</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_stack.html" class="orm:hideurl"><code>?position_fill</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_identity.html" class="orm:hideurl"><code>?position_identity</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_jitter.html" class="orm:hideurl"><code>?position_jitter</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_stack.html" class="orm:hideurl"><code>?position_stack</code></a>

## 연습문제

1.  다음 플롯의 문제점은 무엇입니까? 어떻게 개선할 수 있습니까?

    ```
    ggplot(mpg, aes(x = cty, y = hwy)) +
      geom_point()
    ```

2.  두 플롯 간에 차이점이 있다면 무엇입니까? 그 이유는 무엇입니까?

    ```
    ggplot(mpg, aes(x = displ, y = hwy)) +
      geom_point()
    ggplot(mpg, aes(x = displ, y = hwy)) +
      geom_point(position = "identity")
    ```

3.  <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>의 어떤 매개변수가 지터링(jittering)의 양을 제어합니까?

4.  <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>를 <a href="https://ggplot2.tidyverse.org/reference/geom_count.html" class="orm:hideurl"><code>geom_count()</code></a>와 비교 및 대조해 보세요.

5.  <a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>의 기본 위치 조정은 무엇입니까? 이를 입증하는 `mpg` 데이터셋의 시각화를 만드세요.

# 좌표계 (Coordinate Systems)

좌표계는 아마도 ggplot2에서 가장 복잡한 부분일 것입니다. 기본 좌표계는 데카르트 좌표계(Cartesian coordinate system)로, 여기서 x와 y 위치는 각 점의 위치를 결정하기 위해 독립적으로 작용합니다. 가끔 유용하게 쓰일 수 있는 두 가지 다른 좌표계가 있습니다.

- <a href="https://ggplot2.tidyverse.org/reference/coord_map.html" class="orm:hideurl"><code>coord_quickmap()</code></a>은 지리적 지도의 가로세로 비율(aspect ratio)을 올바르게 설정합니다. 이는 ggplot2로 공간 데이터를 그릴 때 중요합니다. 이 책에서는 지도를 다룰 공간이 부족하지만, *ggplot2: Elegant Graphics for Data Analysis* (Springer)의 [지도 챕터(Maps chapter)](https://oreil.ly/45GHE)에서 자세히 알아볼 수 있습니다.

````

nz <- map_data("nz")

ggplot(nz, aes(x = long, y = lat, group = group)) +
geom_polygon(fill = "white", color = "black")

ggplot(nz, aes(x = long, y = lat, group = group)) +
geom_polygon(fill = "white", color = "black") +
coord_quickmap()

```

![뉴질랜드 경계의 두 지도. 첫 번째 플롯에서는 종횡비가 올바르지 않고 두 번째 플롯에서는 올바릅니다.](D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in26.png)

![뉴질랜드 경계의 두 지도. 첫 번째 플롯에서는 종횡비가 올바르지 않고 두 번째 플롯에서는 올바릅니다.](D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in27.png)

- <a href="https://ggplot2.tidyverse.org/reference/coord_polar.html" class="orm:hideurl"><code>coord_polar()</code></a>는 극좌표(polar coordinates)를 사용합니다. 극좌표는 막대 차트와 콕스콤(Coxcomb) 차트 사이의 흥미로운 연관성을 보여줍니다.

```

bar <- ggplot(data = diamonds) +
geom_bar(
mapping = aes(x = clarity, fill = clarity),
show.legend = FALSE,
width = 1
) +
theme(aspect.ratio = 1)

bar + coord_flip()
bar + coord_polar()

````

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in28.png" alt="왼쪽에는 다이아몬드 투명도의 막대 차트가 있고 오른쪽에는 동일한 데이터의 Coxcomb 차트가 있습니다." />
</figure>

## 연습문제

1.  <a href="https://ggplot2.tidyverse.org/reference/coord_polar.html" class="orm:hideurl"><code>coord_polar()</code></a>를 사용하여 누적 막대 차트를 원형 차트(pie chart)로 바꾸세요.

2.  <a href="https://ggplot2.tidyverse.org/reference/coord_map.html" class="orm:hideurl"><code>coord_quickmap()</code></a>과 <a href="https://ggplot2.tidyverse.org/reference/coord_map.html" class="orm:hideurl"><code>coord_map()</code></a>의 차이점은 무엇입니까?

3.  다음 플롯은 도시(city)와 고속도로(highway) 연비(mpg) 간의 관계에 대해 무엇을 알려줍니까? 왜 <a href="https://ggplot2.tidyverse.org/reference/coord_fixed.html" class="orm:hideurl"><code>coord_fixed()</code></a>가 중요합니까? <a href="https://ggplot2.tidyverse.org/reference/geom_abline.html" class="orm:hideurl"><code>geom_abline()</code></a>은 어떤 역할을 합니까?

  ```
  ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) +
    geom_point() +
    geom_abline() +
    coord_fixed()
  ```

# 그래픽의 레이어 문법 (The Layered Grammar of Graphics)

위치 조정, stat, 좌표계 및 패싯 분할을 추가하여 <a href="ch01.html#sec-ggplot2-calls" data-type="xref">"ggplot2 호출"</a>에서 배운 그래프 템플릿을 확장할 수 있습니다.

  ggplot(data = <데이터>) +
    <GEOM_함수>(
       mapping = aes(<매핑>),
       stat = <STAT>,
       position = <위치>
    ) +
    <좌표계_함수> +
    <패싯_함수>

우리의 새로운 템플릿은 템플릿에 나타나는 대괄호로 묶인 단어인 7개의 매개변수를 사용합니다. 실제로 그래프를 만들기 위해 7개의 매개변수를 모두 제공해야 하는 경우는 드문데, 이는 ggplot2가 데이터, 매핑, geom 함수를 제외한 모든 항목에 대해 유용한 기본값을 제공하기 때문입니다.

템플릿에 있는 이 7개의 매개변수는 플롯을 작성하기 위한 공식 시스템(formal system)인 그래픽의 문법(grammar of graphics)을 구성합니다. 그래픽 문법은 *어떠한* 플롯이든 데이터셋, geom, 매핑 세트, stat, 위치 조정, 좌표계, 패싯 분할 체계, 테마의 조합으로 고유하게 설명할 수 있다는 통찰에 기반합니다.

이것이 어떻게 작동하는지 보려면 처음부터 기본 플롯을 작성하는 방법을 생각해 보세요. 먼저 데이터셋으로 시작한 다음 표시하려는 정보로 변환(stat을 사용하여)할 수 있습니다. 다음으로 변환된 데이터에서 각 관측치를 표현할 기하학적 객체를 선택할 수 있습니다. 그런 다음 geom의 심미적 속성을 사용하여 데이터의 변수를 표현할 수 있습니다. 각 변수의 값을 심미성의 수준(levels)에 매핑하게 됩니다. 이러한 단계들은 <a href="#fig-visualization-grammar" data-type="xref">그림 9-3</a>에 예시되어 있습니다. 그런 다음 geom을 배치할 좌표계를 선택하고, 객체의 위치(위치 자체도 심미적 속성임)를 사용하여 x 및 y 변수의 값을 표시합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0903.png" alt="원시 데이터에서 빈도 표로 이동한 다음 막대의 높이가 빈도를 나타내는 막대 플롯으로 이동하는 단계를 보여주는 그림." />
<h6 id="figure-9-3.-these-are-the-steps-for-going-from-raw-data-to-a-table-of-frequencies-to-a-bar-plot-where-the-heights-of-the-bar-represent-the-frequencies.">그림 9-3. 이것은 원시 데이터에서 빈도 표를 거쳐 막대의 높이가 빈도를 나타내는 막대 플롯을 만들기까지의 단계입니다.</h6>
</figure>

이 시점에는 완전한 그래프가 완성되지만, 좌표계 내에서 geom의 위치를 추가로 조정(위치 조정)하거나 그래프를 하위 플롯(패싯 분할)으로 나눌 수 있습니다. 또한 하나 이상의 추가 레이어를 더하여 플롯을 확장할 수 있으며, 이 경우 각각의 추가 레이어는 자체 데이터셋, geom, 매핑 세트, stat, 위치 조정을 사용하게 됩니다.

이 방법을 사용하면 상상하는 *모든* 플롯을 작성할 수 있습니다. 즉, 이 장에서 배운 코드 템플릿을 사용하여 수십만 개의 고유한 플롯을 만들 수 있습니다.

ggplot2의 이론적 근간에 대해 자세히 알아보고 싶다면, ggplot2의 이론을 자세히 설명하는 과학 논문인 ["A Layered Grammar of Graphics(그래픽의 레이어 문법)"](https://oreil.ly/8fZzE)를 읽어보시는 것을 즐기실 수 있을 것입니다.

# 요약 (Summary)

이 장에서는 간단한 플롯을 만들기 위한 심미성과 기하 구조, 플롯을 부분 집합으로 나누는 패싯, geom이 어떻게 계산되는지 이해하기 위한 통계량(statistics), geom이 겹칠 수 있을 때 위치의 세부 사항을 제어하기 위한 위치 조정, 그리고 `x`와 `y`가 의미하는 바를 근본적으로 변경할 수 있게 해주는 좌표계로 시작하여 그래픽의 레이어 문법(layered grammar of graphics)에 대해 배웠습니다. 우리가 아직 다루지 않은 한 가지 레이어는 테마(theme)인데, 이는 <a href="ch11.html#sec-themes" data-type="xref">"테마(Themes)"</a>에서 소개할 것입니다.

전체 ggplot2 기능에 대한 개요를 얻을 수 있는 매우 유용한 두 가지 리소스는 [ggplot2 치트시트(cheatsheet)](https://oreil.ly/NlKZF)와 [ggplot2 패키지 웹사이트](https://oreil.ly/W6ci8)입니다.

이 장에서 얻어야 할 중요한 교훈은, ggplot2에서 제공하지 않는 geom이 필요하다고 느낄 때 해당 geom을 제공하는 ggplot2 확장 패키지를 만들어 다른 누군가가 당신의 문제를 이미 해결했는지 항상 확인해보는 것이 좋다는 것입니다.
````
