# Chapter 1. Data Visualization

# 서론

> “단순한 그래프는 다른 어떤 장치보다 데이터 분석가의 마음에 더 많은 정보를 가져다주었습니다.” —존 튜키(John Tukey)

R에는 그래프를 만들기 위한 여러 시스템이 있지만, ggplot2는 우아하고 다재다능한 시스템 중 하나입니다. ggplot2는 그래프를 설명하고 구축하기 위한 일관된 시스템인 _그래픽 문법(grammar of graphics)_ 을 구현합니다. ggplot2를 사용하면 하나의 시스템을 배워 여러 곳에 적용함으로써 더 빠르고 더 많은 작업을 수행할 수 있습니다.

이 장에서는 ggplot2를 사용하여 데이터를 시각화하는 방법을 배웁니다. 간단한 산점도를 만드는 것으로 시작하여 ggplot2의 기본 구성 요소인 심미적 매핑(aesthetic mappings)과 기하학적 객체(geometric objects)를 소개하는 데 사용합니다. 그런 다음 단일 변수의 분포를 시각화하고 두 개 이상의 변수 간의 관계를 시각화하는 과정을 안내합니다. 마지막으로 그래프를 저장하는 방법과 문제 해결 팁으로 마무리하겠습니다.

## 사전 준비

이 장은 tidyverse의 핵심 패키지 중 하나인 ggplot2에 중점을 둡니다. 이 장에서 사용하는 데이터셋, 도움말 페이지 및 함수에 액세스하려면 다음을 실행하여 tidyverse를 로드하세요.

```
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

이 코드 한 줄로 거의 모든 데이터 분석에 사용할 패키지들인 핵심 tidyverse가 로드됩니다. 또한 tidyverse의 어떤 함수가 기본 R(또는 로드했을 수 있는 다른 패키지)의 함수와 충돌하는지 알려줍니다.<sup><a href="ch01.html#idm44771333724368" id="idm44771333724368-marker" data-type="noteref">1</a></sup>

이 코드를 실행했을 때 `there is no package called 'tidyverse'`라는 에러 메시지가 표시되면, 먼저 설치한 다음 <a href="https://rdrr.io/r/base/library.html" class="orm:hideurl"><code>library()</code></a>를 다시 한 번 실행해야 합니다.

```
install.packages("tidyverse")
library(tidyverse)
```

패키지는 한 번만 설치하면 되지만, 새 세션을 시작할 때마다 로드해야 합니다.

tidyverse 외에도 Palmer 군도의 세 섬에 사는 펭귄의 신체 측정값을 포함하는 `penguins` 데이터셋이 있는 palmerpenguins 패키지와, 색맹에게 안전한(colorblind safe) 색상 팔레트를 제공하는 ggthemes 패키지를 사용할 것입니다.

```
library(palmerpenguins)
library(ggthemes)
```

# 첫걸음

지느러미(flipper)가 더 긴 펭귄은 더 짧은 펭귄보다 무게가 더 많이 나갈까요, 적게 나갈까요? 이미 답을 알고 있을 수도 있지만, 답을 정확하게 내려보세요. 지느러미 길이와 체질량 사이의 관계는 어떻게 보이나요? 양의 관계인가요? 음의 관계인가요? 선형적인가요? 비선형적인가요? 펭귄의 종(species)에 따라 관계가 다른가요? 펭귄이 사는 섬(island)에 따라서는 어떨까요? 이러한 질문에 답하는 데 사용할 수 있는 시각화를 만들어 봅시다.

## penguins 데이터 프레임

palmerpenguins 패키지에 있는 `penguins` 데이터 프레임(일명 <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>palmerpenguins::penguins</code></a>)을 사용하여 이 질문들에 대한 답을 테스트할 수 있습니다. 데이터 프레임은 (열에) 변수와 (행에) 관측치를 모아 놓은 직사각형 모음입니다. `penguins`에는 Kristen Gorman 박사와 남극 팔머 기지 LTER(Long Term Ecological Research)가 수집하여 제공한 344개의 관측치가 포함되어 있습니다.<sup><a href="ch01.html#idm44771333851472" id="idm44771333851472-marker" data-type="noteref">2</a></sup>

논의를 쉽게 하기 위해 몇 가지 용어를 정의해 보겠습니다.

변수 (Variable)  
측정할 수 있는 수량, 품질 또는 속성입니다.

값 (Value)  
측정할 때의 변수 상태입니다. 변수의 값은 측정할 때마다 다를 수 있습니다.

관측치 (Observation)  
유사한 조건에서 수행된 측정값의 집합입니다(일반적으로 관측치 내의 모든 측정은 동일한 객체에 대해 동시에 이루어집니다). 관측치에는 각각 다른 변수와 연관된 여러 값이 포함됩니다. 때때로 관측치를 _데이터 포인트(data point)_ 라고 부르기도 합니다.

테이블 형식 데이터 (Tabular data)  
각각 변수 및 관측치와 연관된 값들의 집합입니다. 각 값이 고유한 "셀(cell)"에 배치되고, 각 변수가 고유한 열(column)에 배치되며, 각 관측치가 고유한 행(row)에 배치되는 경우 테이블 형식 데이터는 _깔끔(tidy)_ 하다고 합니다.

이 맥락에서 변수는 모든 펭귄의 속성을 의미하고, 관측치는 펭귄 한 마리의 모든 속성을 의미합니다.

콘솔에 데이터 프레임의 이름을 입력하면 R이 내용의 미리보기를 출력합니다. 미리보기 상단에 `tibble`이라고 적혀 있는 것에 주목하세요. tidyverse에서는 곧 배우게 될 *tibble*이라는 특수한 데이터 프레임을 사용합니다.

```
penguins
#> # A tibble: 344 × 8
#>   species island    bill_length_mm bill_depth_mm flipper_length_mm
#>   <fct>   <fct>              <dbl>         <dbl>             <int>
#> 1 Adelie  Torgersen           39.1          18.7               181
#> 2 Adelie  Torgersen           39.5          17.4               186
#> 3 Adelie  Torgersen           40.3          18                 195
#> 4 Adelie  Torgersen           NA            NA                  NA
#> 5 Adelie  Torgersen           36.7          19.3               193
#> 6 Adelie  Torgersen           39.3          20.6               190
#> # … with 338 more rows, and 3 more variables: body_mass_g <int>, sex <fct>,
#> #   year <int>
```

이 데이터 프레임에는 8개의 열이 포함되어 있습니다. 모든 변수와 각 변수의 처음 몇 개 관측치를 볼 수 있는 대안적인 보기를 원한다면 <a href="https://pillar.r-lib.org/reference/glimpse.html" class="orm:hideurl"><code>glimpse()</code></a>를 사용하세요. 또는 RStudio에 있다면 `View(penguins)`를 실행하여 대화형 데이터 뷰어를 엽니다.

```
glimpse(penguins)
#> Rows: 344
#> Columns: 8
#> $ species           <fct> Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, A…
#> $ island            <fct> Torgersen, Torgersen, Torgersen, Torgersen, Torge…
#> $ bill_length_mm    <dbl> 39.1, 39.5, 40.3, NA, 36.7, 39.3, 38.9, 39.2, 34.…
#> $ bill_depth_mm     <dbl> 18.7, 17.4, 18.0, NA, 19.3, 20.6, 17.8, 19.6, 18.…
#> $ flipper_length_mm <int> 181, 186, 195, NA, 193, 190, 181, 195, 193, 190, …
#> $ body_mass_g       <int> 3750, 3800, 3250, NA, 3450, 3650, 3625, 4675, 347…
#> $ sex               <fct> male, female, female, NA, female, male, female, m…
#> $ year              <int> 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2…
```

`penguins`에 포함된 변수 중 일부는 다음과 같습니다.

`species`  
펭귄의 종 (Adelie, Chinstrap, 또는 Gentoo)

`flipper_length_mm`  
펭귄의 지느러미 길이 (밀리미터 단위)

`body_mass_g`  
펭귄의 체질량 (그램 단위)

`penguins`에 대해 더 알아보려면 <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>?penguins</code></a>를 실행하여 도움말 페이지를 여세요.

## 궁극적인 목표

이 장에서의 최종 목표는 펭귄의 종을 고려하여 이 펭귄들의 지느러미 길이와 체질량 사이의 관계를 보여주는 다음 시각화를 재현하는 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in01.png" alt="A scatterplot of body mass vs. flipper length of penguins, with a best fit line of the relationship between these two variables overlaid. The plot displays a positive, fairly linear, and relatively strong relationship between these two variables. Species (Adelie, Chinstrap, and Gentoo) are represented with different colors and shapes. The relationship between body mass and flipper length is roughly the same for these three species, and Gentoo penguins are larger than penguins from the other two species." />
</figure>

## ggplot 만들기

이 그래프를 단계별로 재현해 보겠습니다.

ggplot2를 사용하면 <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> 함수로 그래프를 시작하여, 나중에 _레이어(layers)_ 를 추가할 플롯 객체를 정의합니다. <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>의 첫 번째 인자는 그래프에 사용할 데이터셋이므로, `ggplot(data = penguins)`는 `penguins` 데이터를 표시할 준비가 된 빈 그래프를 생성합니다. 하지만 아직 데이터를 시각화하는 방법을 알려주지 않았으므로 현재는 비어 있습니다. 이것은 그다지 흥미로운 그래프는 아니지만, 나머지 그래프 레이어를 그릴 빈 캔버스라고 생각하면 됩니다.

```
ggplot(data = penguins)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in02.png" alt="A blank, gray plot area." />
</figure>

다음으로 데이터의 정보가 어떻게 시각적으로 표현될지 <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>에 알려주어야 합니다. <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> 함수의 `mapping` 인자는 데이터셋의 변수가 그래프의 시각적 속성(_심미성, aesthetics_)에 어떻게 매핑되는지 정의합니다. `mapping` 인자는 항상 <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a> 함수 내에 정의되며, <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a>의 `x` 및 `y` 인자는 x축 및 y축에 매핑할 변수를 지정합니다. 지금은 지느러미 길이만 `x` 심미성에 매핑하고 체질량을 `y` 심미성에 매핑해 보겠습니다. ggplot2는 `data` 인자(이 경우 `penguins`)에서 매핑된 변수를 찾습니다.

다음 그래프는 이러한 매핑을 추가한 결과를 보여줍니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in03.png" alt="The plot shows flipper length on the x-axis, with values that range from 170 to 230, and body mass on the y-axis, with values that range from 3000 to 6000." />
</figure>

이제 빈 캔버스에 더 많은 구조가 생겼습니다. 지느러미 길이가 표시될 위치(x축)와 체질량이 표시될 위치(y축)가 명확해졌습니다. 하지만 펭귄 자체는 아직 그래프에 없습니다. 데이터 프레임의 관측치를 그래프에 어떻게 표현할지 아직 코드에 명시하지 않았기 때문입니다.

이를 수행하려면 그래프가 데이터를 나타내는 데 사용하는 기하학적 객체인 *geom*을 정의해야 합니다. 이러한 기하학적 객체는 `geom_`으로 시작하는 함수들을 통해 ggplot2에서 사용할 수 있습니다. 사람들은 그래프가 사용하는 geom 유형으로 그래프를 설명하는 경우가 많습니다. 예를 들어, 막대 차트는 막대(bar) geom(<a href="https://ggplot2.tidyverse.org/reference/geom_bar.html" class="orm:hideurl"><code>geom_bar()</code></a>)을 사용하고, 선 차트는 선(line) geom(<a href="https://ggplot2.tidyverse.org/reference/geom_path.html" class="orm:hideurl"><code>geom_line()</code></a>)을 사용하고, 박스 플롯은 상자(boxplot) geom(<a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>)을 사용하고, 산점도는 점(point) geom(<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>)을 사용하는 식입니다.

<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a> 함수는 그래프에 점 레이어를 추가하여 산점도를 만듭니다. ggplot2에는 많은 geom 함수들이 제공되며, 각각은 그래프에 서로 다른 유형의 레이어를 추가합니다. 이 책 전반에 걸쳐, 특히 <a href="ch09.html#chp-layers" data-type="xref">9장</a>에서 아주 다양한 geom을 배우게 될 것입니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point()
#> Warning: Removed 2 rows containing missing values (`geom_point()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in04.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, linear, and relatively strong relationship between these two variables." />
</figure>

이제 우리가 생각하는 "산점도"와 비슷한 모양을 갖추었습니다. 아직 "궁극적인 목표" 그래프와 일치하지는 않지만, 이 그래프를 사용하여 우리의 탐구를 촉발한 질문, 즉 "지느러미 길이와 체질량 사이의 관계는 어떻게 보이는가?"에 답하기 시작할 수 있습니다. 그 관계는 양의 관계(지느러미 길이가 증가할수록 체질량도 증가함), 꽤 선형적인 관계(점들이 곡선이 아닌 직선 주위에 군집함), 그리고 적당히 강한 관계(직선 주위에 흩어짐이 너무 많지 않음)인 것으로 보입니다. 지느러미가 더 긴 펭귄은 일반적으로 체질량 측면에서 더 큽니다.

이 그래프에 더 많은 레이어를 추가하기 전에 잠시 멈추고 우리가 받은 경고 메시지를 살펴보겠습니다.

> Removed 2 rows containing missing values (<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>).

이 메시지가 나타나는 이유는 데이터셋에 체질량 및/또는 지느러미 길이 값이 누락된 펭귄이 두 마리 있으며, ggplot2는 이 두 값 모두 없이는 이들을 그래프에 나타낼 방법이 없기 때문입니다. R과 마찬가지로 ggplot2는 결측값이 조용히 사라지게 내버려두면 안 된다는 철학을 따릅니다. 이런 유형의 경고는 실제 데이터를 다룰 때 아마도 흔하게 볼 수 있는 경고 유형 중 하나일 것입니다. 결측값은 흔한 문제이며, 책 전체, 특히 <a href="ch18.html#chp-missing-values" data-type="xref">18장</a>에서 이에 대해 더 자세히 배울 것입니다. 이 장의 나머지 그래프에서는 우리가 그리는 모든 그래프마다 이 경고가 출력되지 않도록 숨길 것입니다.

## 심미성과 레이어 추가하기

산점도는 두 수치형 변수 간의 관계를 표시하는 데 유용하지만, 두 변수 사이에 나타나는 명백한 관계에 대해 회의적인 태도를 취하고, 이 피상적인 관계의 성격을 설명하거나 바꿀 수 있는 다른 변수가 있는지 물어보는 것이 항상 좋은 생각입니다. 예를 들어, 지느러미 길이와 체질량 간의 관계가 종에 따라 다를까요? 종(species)을 그래프에 포함하여 변수들 사이의 명백한 관계에 대한 추가적인 통찰력을 드러내는지 확인해 봅시다. 종을 다른 색상의 점으로 표현하여 이를 수행할 것입니다.

이를 달성하려면 심미성(aesthetic)과 geom 중 어느 것을 수정해야 할까요? "심미성 매핑에서, <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a> 안에서"라고 추측했다면, 이미 ggplot2를 사용한 데이터 시각화 생성의 요령을 터득하고 있는 것입니다! 그렇지 않더라도 걱정하지 마세요. 책을 진행하면서 더 많은 ggplot을 만들게 될 것이며, 만들면서 직관을 확인할 기회가 더 많이 있을 것입니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g, color = species)
) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in05.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, fairly linear, and relatively strong relationship between these two variables. Species (Adelie, Chinstrap, and Gentoo) are represented with different colors." />
</figure>

범주형 변수가 심미성에 매핑될 때, ggplot2는 변수의 각 고유한 수준(여기서는 세 종 각각)에 고유한 심미성 값(여기서는 고유한 색상)을 자동으로 할당하는데, 이 과정을 _스케일링(scaling)_ 이라고 합니다. ggplot2는 어떤 값이 어떤 수준에 해당하는지 설명하는 범례도 추가합니다.

이제 한 가지 레이어를 더 추가해 보겠습니다. 체질량과 지느러미 길이 사이의 관계를 보여주는 부드러운 곡선(smooth curve)입니다. 계속하기 전에 이전 코드를 참조하여 기존 그래프에 이것을 어떻게 추가할 수 있을지 생각해 보세요.

이것은 데이터를 나타내는 새로운 기하학적 객체이므로, 점 geom 위에 새로운 geom을 레이어로 추가할 것입니다. <a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>. 그리고 `method = "lm"`을 사용하여 선형 모델(`l`inear `m`odel)을 기반으로 최적의 적합선을 그리도록 지정할 것입니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g, color = species)
) +
  geom_point() +
  geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in06.png" alt="A scatterplot of body mass vs. flipper length of penguins. Overlaid on the scatterplot are three smooth curves displaying the relationship between these variables for each species (Adelie, Chinstrap, and Gentoo). Different penguin species are plotted in different colors for the points and the smooth curves." />
</figure>

성공적으로 선을 추가했지만, 이 그래프는 <a href="#sec-ultimate-goal" data-type="xref">"궁극적인 목표"</a> 섹션의 그래프와 같지 않습니다. 그 그래프는 각각의 펭귄 종에 대해 별도의 선이 있는 것이 아니라 전체 데이터셋에 대해 하나의 선만 가지고 있습니다.

<a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> 내에서 심미성 매핑이 정의되면(전역, _global_ 수준에서), 그래프의 후속적인 각 geom 레이어로 전달됩니다. 그러나 ggplot2의 각 geom 함수는 전역 수준에서 상속된 것에 더해 _로컬(local)_ 수준의 심미성 매핑을 허용하는 `mapping` 인자를 받을 수도 있습니다. 점은 종(species)에 따라 색상이 지정되길 원하지만 선은 종별로 분리되길 원하지 않으므로, <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>에 대해서만 `color = species`를 지정해야 합니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point(mapping = aes(color = species)) +
  geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in07.png" alt="A scatterplot of body mass vs. flipper length of penguins. Overlaid on the scatterplot is a single line of best fit displaying the relationship between these variables for each species (Adelie, Chinstrap, and Gentoo). Different penguin species are plotted in different colors for the points only." />
</figure>

짜잔! 완벽하지는 않지만 우리의 궁극적인 목표와 매우 유사한 것이 완성되었습니다. 여전히 각 펭귄 종마다 다른 모양(shape)을 사용하고 레이블을 개선해야 합니다.

그래프에서 색상만을 사용하여 정보를 나타내는 것은 일반적으로 좋은 생각이 아닙니다. 색맹이나 기타 색각 차이로 인해 사람들이 색상을 다르게 인식하기 때문입니다. 따라서 색상 외에도 `species`를 `shape` 심미성에 매핑할 수 있습니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point(mapping = aes(color = species, shape = species)) +
  geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in08.png" alt="A scatterplot of body mass vs. flipper length of penguins. Overlaid on the scatterplot is a single line of best fit displaying the relationship between these variables for each species (Adelie, Chinstrap, and Gentoo). Different penguin species are plotted in different colors and shapes for the points only." />
</figure>

범례는 점의 다양한 모양(shape)을 반영하기 위해 자동으로 업데이트된다는 점에 유의하세요.

마지막으로 새로운 레이어에 <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a> 함수를 사용하여 그래프의 레이블을 개선할 수 있습니다. <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>의 일부 인자들은 자명할 수 있습니다. `title`은 그래프에 제목을 추가하고, `subtitle`은 부제목을 추가합니다. 다른 인자들은 심미성 매핑과 일치합니다. `x`는 x축 레이블, `y`는 y축 레이블이며, `color`와 `shape`는 범례의 레이블을 정의합니다. 게다가 ggthemes 패키지의 <a href="https://rdrr.io/pkg/ggthemes/man/colorblind.html" class="orm:hideurl"><code>scale_color_colorblind()</code></a> 함수를 사용하여 색상 팔레트를 색맹에게 안전하도록 개선할 수 있습니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point(aes(color = species, shape = species)) +
  geom_smooth(method = "lm") +
  labs(
    title = "Body mass and flipper length",
    subtitle = "Dimensions for Adelie, Chinstrap, and Gentoo Penguins",
    x = "Flipper length (mm)", y = "Body mass (g)",
    color = "Species", shape = "Species"
  ) +
  scale_color_colorblind()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in09.png" alt="A scatterplot of body mass vs. flipper length of penguins, with a line of best fit displaying the relationship between these two variables overlaid. The plot displays a positive, fairly linear, and relatively strong relationship between these two variables. Species (Adelie, Chinstrap, and Gentoo) are represented with different colors and shapes. The relationship between body mass and flipper length is roughly the same for these three species, and Gentoo penguins are larger than penguins from the other two species." />
</figure>

드디어 "궁극적인 목표"와 완벽하게 일치하는 그래프를 얻었습니다!

## 연습 문제

1.  `penguins`에는 행이 몇 개 있나요? 열은 몇 개인가요?

2.  `penguins` 데이터 프레임의 `bill_depth_mm` 변수는 무엇을 설명하나요? 알아보기 위해 <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>?penguins</code></a>의 도움말을 읽어보세요.

3.  `bill_length_mm`에 대한 `bill_depth_mm`의 산점도를 만들어 보세요. 즉, y축에 `bill_depth_mm`이 있고 x축에 `bill_length_mm`이 있는 산점도를 만듭니다. 이 두 변수 간의 관계를 설명해 보세요.

4.  `bill_depth_mm`에 대한 `species`의 산점도를 만들면 어떻게 되나요? 더 나은 geom 선택은 무엇일까요?

5.  다음 코드에서 에러가 발생하는 이유는 무엇이며, 어떻게 고칠 수 있을까요?

    ```
    ggplot(data = penguins) +
      geom_point()
    ```

6.  <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>에서 `na.rm` 인자는 무슨 역할을 하나요? 인자의 기본값은 무엇인가요? 이 인자를 `TRUE`로 성공적으로 설정한 산점도를 만들어 보세요.

7.  이전 연습 문제에서 만든 그래프에 다음 캡션을 추가하세요. “Data come from the palmerpenguins package.” 힌트: <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>의 문서를 살펴보세요.

8.  다음 시각화를 재현해 보세요. `bill_depth_mm`은 어떤 심미성에 매핑되어야 할까요? 그리고 전역(global) 수준에 매핑해야 할까요, 아니면 geom 수준에 매핑해야 할까요?

    <figure>
    <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in10.png" alt="A scatterplot of body mass vs. flipper length of penguins, colored by bill depth. A smooth curve of the relationship between body mass and flipper length is overlaid. The relationship is positive, fairly linear, and moderately strong." />
    </figure>

9.  이 코드를 머릿속에서 실행해보고 출력이 어떻게 보일지 예측해 보세요. 그런 다음 R에서 코드를 실행하고 예측을 확인해 보세요.

    ```
    ggplot(
      data = penguins,
      mapping = aes(x = flipper_length_mm, y = body_mass_g, color = island)
    ) +
      geom_point() +
      geom_smooth(se = FALSE)
    ```

10. 이 두 그래프가 다르게 보일까요? 그 이유는 무엇이거나 무엇이 아닌가요?

    ```
    ggplot(
      data = penguins,
      mapping = aes(x = flipper_length_mm, y = body_mass_g)
    ) +
      geom_point() +
      geom_smooth()

    ggplot() +
      geom_point(
        data = penguins,
        mapping = aes(x = flipper_length_mm, y = body_mass_g)
      ) +
      geom_smooth(
        data = penguins,
        mapping = aes(x = flipper_length_mm, y = body_mass_g)
      )
    ```

# ggplot2 호출하기

이러한 입문 섹션을 넘어가면서, ggplot2 코드를 좀 더 간결하게 표현하는 방식으로 전환할 것입니다. 지금까지는 다음과 같이 매우 명시적이었습니다. 이는 학습할 때 도움이 됩니다.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point()
```

일반적으로 함수의 처음 한두 개 인자는 너무 중요해서 외우고 있어야 합니다. <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>의 처음 두 인자는 `data`와 `mapping`입니다. 책의 나머지 부분에서는 이러한 이름을 입력하지 않겠습니다. 그러면 타이핑이 줄어들고 추가 텍스트 양이 줄어들어 그래프 간의 차이점을 더 쉽게 확인할 수 있습니다. 이는 우리가 <a href="ch25.html#chp-functions" data-type="xref">25장</a>에서 다시 다루게 될 정말 중요한 프로그래밍 고려 사항입니다.

이전 그래프를 더 간결하게 다시 작성하면 다음과 같습니다.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point()
```

향후에는 다음과 같이 그래프를 만들 수 있게 해주는 파이프 `|>`에 대해서도 배울 것입니다.

```
penguins |>
  ggplot(aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point()
```

# 분포 시각화하기

변수의 분포를 시각화하는 방법은 변수 유형이 범주형인지 수치형인지에 따라 다릅니다.

## 범주형 변수

변수가 작은 값 집합 중 하나만 취할 수 있는 경우 _범주형(categorical)_ 변수입니다. 범주형 변수의 분포를 조사하려면 막대 차트(bar chart)를 사용할 수 있습니다. 막대의 높이는 각 `x` 값에서 관측치가 얼마나 발생했는지를 보여줍니다.

```
ggplot(penguins, aes(x = species)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in11.png" alt="A bar chart of frequencies of species of penguins: Adelie (approximately 150), Chinstrap (approximately 90), Gentoo (approximately 125)." />
</figure>

앞서 나온 펭귄의 `species`처럼 순서가 없는 수준(level)을 가진 범주형 변수의 막대 그래프에서는 종종 빈도를 기준으로 막대를 재정렬하는 것이 바람직합니다. 이를 위해서는 변수를 팩터(factor, R이 범주형 데이터를 처리하는 방식)로 변환한 다음 해당 팩터의 수준을 재정렬해야 합니다.

```
ggplot(penguins, aes(x = fct_infreq(species))) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in12.png" alt="A bar chart of frequencies of species of penguins, where the bars are ordered in decreasing order of their heights (frequencies): Adelie (approximately 150), Gentoo (approximately 125), Chinstrap (approximately 90)." />
</figure>

팩터와 팩터를 다루기 위한 함수(<a href="https://forcats.tidyverse.org/reference/fct_inorder.html" class="orm:hideurl"><code>fct_infreq()</code></a>)에 대해서는 <a href="ch16.html#chp-factors" data-type="xref">16장</a>에서 자세히 알아볼 것입니다.

## 수치형 변수

광범위한 수치 값을 가질 수 있고 해당 값으로 덧셈, 뺄셈 또는 평균을 구하는 것이 합리적인 경우, 변수는 _수치형(numerical)_ (또는 양적) 변수입니다. 수치형 변수는 연속형이거나 이산형일 수 있습니다.

연속형 변수의 분포에 일반적으로 사용되는 시각화 중 하나는 히스토그램입니다.

```
ggplot(penguins, aes(x = body_mass_g)) +
  geom_histogram(binwidth = 200)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in13.png" alt="A histogram of body masses of penguins. The distribution is unimodal and right skewed, ranging between approximately 2500 to 6500 grams." />
</figure>

히스토그램은 x축을 동일한 간격의 구간(bin)으로 나눈 다음 막대의 높이를 사용하여 각 구간에 속하는 관측치의 수를 표시합니다. 이전 그래프에서 높은 막대는 39개의 관측치가 막대의 왼쪽과 오른쪽 가장자리인 3,500에서 3,700그램 사이의 `body_mass_g` 값을 가지고 있음을 보여줍니다.

`x` 변수의 단위로 측정되는 `binwidth` 인자를 사용하여 히스토그램의 구간 너비를 설정할 수 있습니다. 히스토그램을 작업할 때는 `binwidth` 값이 다르면 다른 패턴이 나타날 수 있으므로 항상 다양한 `binwidth` 값을 탐색해야 합니다. 다음 그래프들에서 `binwidth` 20은 너무 좁아서 막대가 너무 많아져 분포의 모양을 파악하기 어렵게 만듭니다. 마찬가지로 2,000의 `binwidth`는 너무 커서 모든 데이터가 단 3개의 막대에 모이게 되어 분포의 모양을 파악하기 어렵게 만듭니다. `binwidth` 200은 합리적인 균형을 제공합니다.

```
ggplot(penguins, aes(x = body_mass_g)) +
  geom_histogram(binwidth = 20)
ggplot(penguins, aes(x = body_mass_g)) +
  geom_histogram(binwidth = 2000)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in14.png" alt="Two histograms of body masses of penguins, one with binwidth of 20 (left) and one with binwidth of 2000 (right). The histogram with binwidth of 20 shows lots of ups and downs in the heights of the bins, creating a jagged outline. The histogram with binwidth of 2000 shows only three bins." />
</figure>

수치형 변수의 분포를 위한 대안적인 시각화는 밀도 플롯(density plot)입니다. 밀도 플롯은 히스토그램을 매끄럽게 만든 버전이며 특히 근본적으로 매끄러운 분포에서 나오는 연속형 데이터의 경우 실용적인 대안입니다. <a href="https://ggplot2.tidyverse.org/reference/geom_density.html" class="orm:hideurl"><code>geom_density()</code></a>가 밀도를 추정하는 방법은 다루지 않겠지만(해당 내용에 대해 함수 문서에서 자세히 읽어볼 수 있습니다), 비유를 통해 밀도 곡선이 그려지는 방법을 설명해 보겠습니다. 나무 블록으로 만든 히스토그램을 상상해 보세요. 그런 다음 그 위에 익힌 스파게티 가닥을 떨어뜨린다고 상상해 보세요. 블록 위에 늘어진 스파게티의 모양을 밀도 곡선의 모양으로 생각할 수 있습니다. 히스토그램보다 세부 사항은 덜 보여주지만, 특히 최빈값(mode)과 왜도(skewness)와 관련하여 분포의 형태를 신속하게 파악하는 데 더 쉽게 만들어 줍니다.

```
ggplot(penguins, aes(x = body_mass_g)) +
  geom_density()
#> Warning: Removed 2 rows containing non-finite values (`stat_density()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in15.png" alt="A density plot of body masses of penguins. The distribution is unimodal and right skewed, ranging between approximately 2500 to 6500 grams." />
</figure>

## 연습 문제

1.  `penguins`의 `species`에 대한 막대 그래프를 만들되, `species`를 `y` 심미성에 매핑하세요. 이 그래프는 어떻게 다른가요?

2.  다음 두 그래프는 어떻게 다른가요? 막대의 색상을 변경하는 데 `color`와 `fill` 중 어떤 심미성이 더 유용한가요?

    ```
    ggplot(penguins, aes(x = species)) +
      geom_bar(color = "red")

    ggplot(penguins, aes(x = species)) +
      geom_bar(fill = "red")
    ```

3.  <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_histogram()</code></a>의 `bins` 인자는 어떤 역할을 하나요?

4.  tidyverse 패키지를 로드할 때 사용할 수 있는 `diamonds` 데이터셋에 있는 `carat` 변수의 히스토그램을 만들어 보세요. 다양한 `binwidth` 값으로 실험해 보세요. 어떤 값이 흥미로운 패턴을 드러내나요?

# 관계 시각화하기

관계를 시각화하려면 그래프의 심미성에 매핑된 변수가 적어도 두 개 이상 있어야 합니다. 다음 섹션에서는 두 개 이상의 변수 사이의 관계를 시각화하는 데 흔히 사용되는 그래프와 이를 생성하는 데 사용되는 geom에 대해 배웁니다.

## 수치형 및 범주형 변수

수치형 변수와 범주형 변수 간의 관계를 시각화하려면 나란히 배치된 박스 플롯(side-by-side box plots)을 사용할 수 있습니다. _박스 플롯(boxplot)_ 은 분포를 설명하는 위치 측정값(백분위수)에 대한 시각적 속기(shorthand)의 한 유형입니다. 잠재적인 이상치(outlier)를 식별하는 데에도 유용합니다. <a href="#fig-eda-boxplot" data-type="xref">그림 1-1</a>에 나와 있듯이 각 박스 플롯은 다음으로 구성됩니다.

- 분포의 25번째 백분위수에서 75번째 백분위수까지 뻗어 있는, _사분위수 범위(interquartile range, IQR)_ 로 알려진 거리인 데이터 중간 절반의 범위를 나타내는 상자(box). 상자 중간에는 분포의 중앙값, 즉 50번째 백분위수를 표시하는 선이 있습니다. 이 세 개의 선은 분포의 퍼짐 정도와 분포가 중앙값을 중심으로 대칭인지 한쪽으로 치우쳐(skewed) 있는지 알려줍니다.

- 상자의 양쪽 가장자리에서 IQR의 1.5배 이상 벗어난 곳에 떨어지는 관측치를 표시하는 시각적 점(points). 이러한 이상치 점들은 비정상적이므로 개별적으로 그려집니다.

- 상자의 각 끝에서 분포 내 멀리 떨어진 비이상치(non-outlier) 점까지 뻗어 있는 선(또는 수염, whisker).

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0101.png" alt="A diagram depicting how a boxplot is created following the steps outlined above." /></p>
<h6 id="figure-1-1.-diagram-depicting-how-a-boxplot-is-created.">그림 1-1. 박스 플롯이 생성되는 과정을 묘사한 다이어그램.</h6>
</figure>

<a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>을 사용하여 종별 체질량 분포를 살펴봅시다.

```
ggplot(penguins, aes(x = species, y = body_mass_g)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in16.png" alt="Side-by-side box plots of distributions of body masses of Adelie, Chinstrap, and Gentoo penguins. The distribution of Adelie and Chinstrap penguins&#39; body masses appear to be symmetric with medians around 3750 grams. The median body mass of Gentoo penguins is much higher, around 5000 grams, and the distribution of the body masses of these penguins appears to be somewhat right skewed." />
</figure>

대안적으로 <a href="https://ggplot2.tidyverse.org/reference/geom_density.html" class="orm:hideurl"><code>geom_density()</code></a>를 사용하여 밀도 플롯을 만들 수 있습니다.

```
ggplot(penguins, aes(x = body_mass_g, color = species)) +
  geom_density(linewidth = 0.75)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in17.png" alt="A density plot of body masses of penguins by species of penguins. Each species (Adelie, Chinstrap, and Gentoo) is represented with different colored outlines for the density curves." />
</figure>

또한 선이 배경과 대비되어 좀 더 눈에 띄게 만들기 위해 `linewidth` 인자를 사용하여 선의 두께를 사용자 지정했습니다.

추가적으로 `species`를 `color`와 `fill` 심미성 모두에 매핑하고 `alpha` 심미성을 사용하여 채워진 밀도 곡선에 투명도를 추가할 수 있습니다. 이 심미성은 0(완전히 투명함)과 1(완전히 불투명함) 사이의 값을 가집니다. 다음 그래프에서는 0.5로 설정되어 있습니다.

```
ggplot(penguins, aes(x = body_mass_g, color = species, fill = species)) +
  geom_density(alpha = 0.5)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in18.png" alt="A density plot of body masses of penguins by species of penguins. Each species (Adelie, Chinstrap, and Gentoo) is represented in different colored outlines for the density curves. The density curves are also filled with the same colors, with some transparency added." />
</figure>

여기서 우리가 사용한 용어에 주목하세요.

- 해당 심미성으로 표현되는 시각적 속성이 변수의 값에 따라 변하게 하려면, 변수를 심미성에 _매핑(map)_ 합니다.
- 그렇지 않은 경우 심미성의 값을 _설정(set)_ 합니다.

## 두 개의 범주형 변수

누적 막대 그래프(stacked bar plots)를 사용하여 두 범주형 변수 간의 관계를 시각화할 수 있습니다. 예를 들어, 다음 두 개의 누적 막대 그래프는 모두 `island`와 `species` 간의 관계를 표시하거나, 구체적으로는 각 섬 내의 `species` 분포를 시각화합니다.

첫 번째 그래프는 각 섬에 서식하는 각 펭귄 종의 빈도를 보여줍니다. 빈도 그래프는 각 섬에 동수의 Adelie가 있음을 보여주지만, 각 섬 내의 백분율 균형에 대해서는 좋은 감을 주지 못합니다.

```
ggplot(penguins, aes(x = island, fill = species)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in19.png" alt="Bar plots of penguin species by island (Biscoe, Dream, and Torgersen)" />
</figure>

두 번째 그래프는 geom에서 `position = "fill"`을 설정하여 만든 상대 빈도 그래프이며, 섬마다 다른 펭귄 수의 영향을 받지 않기 때문에 섬 간의 종 분포를 비교하는 데 더 유용합니다. 이 그래프를 사용하면 Gentoo 펭귄은 모두 Biscoe 섬에 살며 해당 섬 펭귄의 약 75%를 차지하고, Chinstrap은 모두 Dream 섬에 살며 해당 섬 펭귄의 약 50%를 차지하며, Adelie는 3개 섬 모두에 살고 Torgersen에 사는 펭귄의 전부를 차지한다는 것을 알 수 있습니다.

```
ggplot(penguins, aes(x = island, fill = species)) +
  geom_bar(position = "fill")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in20.png" alt="Bar plots of penguin species by island (Biscoe, Dream, and Torgersen) the bars are scaled to the same height, making it a relative frequencies plot" />
</figure>

이러한 막대 차트를 만들 때 막대로 나눌 변수를 `x` 심미성에 매핑하고, 막대 안의 색상을 변경할 변수를 `fill` 심미성에 매핑합니다.

## 두 개의 수치형 변수

지금까지 두 수치형 변수 간의 관계를 시각화하기 위해 산점도(<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>로 만듦)와 매끄러운 곡선(<a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>로 만듦)에 대해 배웠습니다. 산점도는 아마도 두 수치형 변수 간의 관계를 시각화하는 데 일반적으로 사용되는 그래프일 것입니다.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in21.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, linear, relatively strong relationship between these two variables." />
</figure>

## 3개 이상의 변수

<a href="#sec-adding-aesthetics-layers" data-type="xref">"심미성과 레이어 추가하기"</a>에서 보았듯이, 그래프에 더 많은 변수를 추가 심미성에 매핑하여 통합할 수 있습니다. 예를 들어, 다음 산점도에서 점의 색상은 종(species)을 나타내고 점의 모양은 섬(island)을 나타냅니다.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point(aes(color = species, shape = island))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in22.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, linear, relatively strong relationship between these two variables. The points are colored based on the species of the penguins and the shapes of the points represent islands (round points are Biscoe island, triangles are Dream island, and squared are Torgersen island). The plot is very busy and it&#39;s difficult to distinguish the shapes of the points." />
</figure>

하지만 그래프에 너무 많은 심미적 매핑을 추가하면 혼란스러워져 이해하기 어려워집니다. 특히 범주형 변수에 유용한 또 다른 옵션은 그래프를 _패싯(facets)_ 으로 분할하는 것입니다. 패싯은 각각 데이터의 부분 집합(subset) 하나를 표시하는 하위 그래프입니다.

단일 변수를 기준으로 그래프를 패싯하려면 <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a>을 사용합니다. <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a>의 첫 번째 인자는 공식(formula)<sup><a href="ch01.html#idm44771330671200" id="idm44771330671200-marker" data-type="noteref">3</a></sup>으로, `~` 뒤에 변수 이름을 붙여 만듭니다. <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a>에 전달하는 변수는 범주형이어야 합니다.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point(aes(color = species, shape = species)) +
  facet_wrap(~island)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in23.png" alt="A scatterplot of body mass vs. flipper length of penguins. The shapes and colors of points represent species. Penguins from each island are on a separate facet. Within each facet, the relationship between body mass and flipper length is positive, linear, relatively strong." />
</figure>

변수의 분포와 이들 간의 관계를 시각화하기 위한 다른 많은 geom들에 대해서는 <a href="ch09.html#chp-layers" data-type="xref">9장</a>에서 배우게 될 것입니다.

## 연습 문제

1.  ggplot2 패키지에 번들로 포함된 `mpg` 데이터 프레임에는 미국 환경보호국(US Environmental Protection Agency)이 수집한 38개 자동차 모델에 대한 234개의 관측치가 포함되어 있습니다. `mpg`의 어떤 변수가 범주형인가요? 어떤 변수가 수치형인가요? (힌트: <a href="https://ggplot2.tidyverse.org/reference/mpg.html" class="orm:hideurl"><code>?mpg</code></a>를 입력하여 데이터셋에 대한 문서를 읽어보세요.) `mpg`를 실행할 때 이 정보를 어떻게 볼 수 있나요?

2.  `mpg` 데이터 프레임을 사용하여 `displ`에 대한 `hwy`의 산점도를 만들어 보세요. 다음으로 세 번째 수치형 변수를 `color`에 매핑하고, 그다음에는 `size`에, 그다음에는 `color`와 `size` 모두에, 그다음에는 `shape`에 매핑해 보세요. 범주형 변수와 비교하여 수치형 변수에 대해 이러한 심미성들이 어떻게 다르게 작동하나요?

3.  `displ`에 대한 `hwy` 산점도에서, 세 번째 변수를 `linewidth`에 매핑하면 어떻게 될까요?

4.  동일한 변수를 여러 심미성에 매핑하면 어떻게 되나요?

5.  `bill_length_mm`에 대한 `bill_depth_mm`의 산점도를 만들고 `species`에 따라 점의 색상을 지정하세요. 종별 색상 지정을 추가하면 이 두 변수 간의 관계에 대해 무엇이 드러나나요? 종별로 패싯을 나누면 어떨까요?

6.  다음은 왜 두 개의 개별 범례를 생성하나요? 두 범례를 결합하려면 어떻게 고쳐야 할까요?

    ```
    ggplot(
      data = penguins,
      mapping = aes(
        x = bill_length_mm, y = bill_depth_mm,
        color = species, shape = species
      )
    ) +
      geom_point() +
      labs(color = "Species")
    ```

7.  다음 두 개의 누적 막대 그래프를 생성하세요. 첫 번째 그래프로 어떤 질문에 답할 수 있나요? 두 번째 그래프로 어떤 질문에 답할 수 있나요?

    ```
    ggplot(penguins, aes(x = island, fill = species)) +
      geom_bar(position = "fill")
    ggplot(penguins, aes(x = species, fill = island)) +
      geom_bar(position = "fill")
    ```

# 그래프 저장하기

그래프를 다 만든 후에는 다른 곳에서 사용할 수 있도록 이미지로 저장하여 R 밖으로 가져오고 싶을 수 있습니다. 이는 최근에 생성된 그래프를 디스크에 저장하는 <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a>의 역할입니다.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point()
ggsave(filename = "penguin-plot.png")
```

이렇게 하면 작업 디렉토리(working directory)에 그래프가 저장됩니다. 작업 디렉토리는 <a href="ch06.html#chp-workflow-scripts" data-type="xref">6장</a>에서 자세히 알아볼 개념입니다.

`width`와 `height`를 지정하지 않으면 현재 플로팅 디바이스의 차원에서 가져옵니다. 재현 가능한 코드를 위해 이를 지정하는 것이 좋습니다. 설명서에서 <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a>에 대해 자세히 알아볼 수 있습니다.

하지만 일반적으로는 코드와 텍스트를 끼워 넣고 작성한 문서에 그래프를 자동으로 포함할 수 있는 재현 가능한 저작 시스템인 Quarto를 사용하여 최종 보고서를 조합할 것을 권장합니다. Quarto에 대해서는 <a href="ch28.html#chp-quarto" data-type="xref">28장</a>에서 자세히 알아볼 것입니다.

## 연습 문제

1.  다음 코드 줄들을 실행해 보세요. 두 그래프 중 어느 것이 `mpg-plot.png`로 저장되나요? 이유는 무엇인가요?

    ```
    ggplot(mpg, aes(x = class)) +
      geom_bar()
    ggplot(mpg, aes(x = cty, y = hwy)) +
      geom_point()
    ggsave("mpg-plot.png")
    ```

2.  그래프를 PNG 대신 PDF로 저장하려면 이전 코드에서 무엇을 변경해야 하나요? <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a>에서 어떤 유형의 이미지 파일이 작동하는지 어떻게 알 수 있나요?

# 일반적인 문제

R 코드 실행을 시작하면 문제에 부딪힐 가능성이 높습니다. 걱정하지 마세요. 누구에게나 일어나는 일입니다. 우리 모두 수년간 R 코드를 작성해왔지만 매일 여전히 첫 시도에서 작동하지 않는 코드를 작성합니다!

실행 중인 코드를 책에 있는 코드와 주의 깊게 비교하는 것부터 시작하세요. R은 극도로 까다로워서 잘못 배치된 문자 하나가 큰 차이를 만들 수 있습니다. 모든 `(`가 `)`와 짝을 이루고 모든 `"`가 다른 `"`와 짝을 이루는지 확인하세요. 때로는 코드를 실행해도 아무 일도 일어나지 않을 수 있습니다. 콘솔의 왼쪽을 확인하세요. `+`가 있으면 R은 아직 완전한 표현식을 입력하지 않았다고 생각하고 기다리고 있는 것입니다. 이 경우 обычно Escape를 눌러 현재 명령어 처리를 중단하고 처음부터 다시 시작하는 것이 쉽습니다.

ggplot2 그래픽을 만들 때 한 가지 흔한 문제는 `+`를 잘못된 곳에 놓는 것입니다. 줄의 시작이 아니라 끝에 와야 합니다. 즉, 다음과 같은 코드를 실수로 작성하지 않았는지 확인하세요.

```
ggplot(data = mpg)
+ geom_point(mapping = aes(x = displ, y = hwy))
```

여전히 막혀 있다면 도움말을 시도해 보세요. 콘솔에서 `?함수_이름`을 실행하거나 함수 이름을 강조 표시하고 RStudio에서 F1을 눌러 R 함수에 대한 도움말을 얻을 수 있습니다. 도움말이 그다지 도움이 되지 않는 것 같더라도 걱정하지 마세요. 대신 예제로 내려가서 여러분이 하려는 작업과 일치하는 코드를 찾아보세요.

그것도 도움이 되지 않는다면 에러 메시지를 주의 깊게 읽어보세요. 때로는 답이 거기에 묻혀 있을 것입니다! 하지만 R이 처음일 때는 에러 메시지에 답이 있더라도 그것을 어떻게 이해해야 할지 아직 모를 수 있습니다. 또 다른 훌륭한 도구는 Google입니다. 에러 메시지를 Google에서 검색해 보세요. 다른 사람도 같은 문제를 겪고 온라인에서 도움을 받았을 가능성이 높습니다.

# 요약

이 장에서는 ggplot2를 사용한 데이터 시각화의 기본을 배웠습니다. 시각화란 데이터의 변수에서 위치, 색상, 크기, 모양 등과 같은 심미적 속성으로의 매핑이라는 ggplot2를 뒷받침하는 기본 아이디어에서 시작했습니다. 그런 다음 복잡성을 높이고 레이어별로 그래프의 표현을 개선하는 방법을 배웠습니다. 또한 추가적인 심미적 매핑을 활용하고/하거나 패싯을 사용하여 작은 다중 그래프로 분할함으로써 단일 변수의 분포뿐만 아니라 두 개 이상의 변수 간의 관계를 시각화하는 데 일반적으로 사용되는 그래프에 대해서도 배웠습니다.

이 책 전반에 걸쳐 시각화를 반복해서 사용할 것이며, 필요에 따라 새로운 기술을 소개하고 <a href="ch09.html#chp-layers" data-type="xref">9장</a>부터 <a href="ch11.html#chp-communication" data-type="xref">11장</a>까지 ggplot2를 사용한 시각화 생성에 대해 더 깊이 들어갈 것입니다.

이제 시각화의 기본 사항을 이해했으므로, 다음 장에서는 기어를 조금 바꿔 실용적인 워크플로에 대한 조언을 제공하겠습니다. 책의 이 부분 전반에 걸쳐 데이터 과학 도구들과 워크플로에 대한 조언을 교차 배치할 것입니다. 이는 더 많은 양의 R 코드를 작성할 때 체계적으로 유지하는 데 도움이 되기 때문입니다.

<sup>[1](ch01.html#idm44771333724368-marker)</sup> 더 많은 패키지를 로드할수록 중요해지는 conflicted 패키지를 사용하면 해당 메시지를 제거하고 요구에 따라 충돌 해결이 발생하도록 강제할 수 있습니다. conflicted에 대해 더 알아보려면 [패키지 웹사이트](https://oreil.ly/01bKz)를 참조하세요.

<sup>[2](ch01.html#idm44771333851472-marker)</sup> Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package version 0.1.0. [_https://oreil.ly/ncwc5_](https://oreil.ly/ncwc5). doi: 10.5281/zenodo.3960218.

<sup>[3](ch01.html#idm44771330671200-marker)</sup> 여기서 "공식(formula)"은 "방정식(equation)"의 동의어가 아니라 `~`에 의해 생성된 것의 이름입니다.
