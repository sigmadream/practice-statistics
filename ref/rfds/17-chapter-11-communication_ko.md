# 제11장. 소통 (Communication)

# 소개 (Introduction)

<a href="ch10.html#chp-EDA" data-type="xref">제10장</a>에서는 플롯(plots)을 *탐색(exploration)*을 위한 도구로 사용하는 방법을 배웠습니다. 탐색적 플롯을 만들 때는, 살펴보기도 전에 플롯에 어떤 변수가 표시될지 이미 알고 있습니다. 여러분은 목적을 가지고 각 플롯을 만들었고, 그것을 빠르게 살펴본 다음 다음 플롯으로 넘어갈 수 있었습니다. 대부분의 분석 과정에서 수십 또는 수백 개의 플롯을 만들게 되며, 그 중 대부분은 즉시 버려집니다.

이제 데이터를 이해했으므로, 여러분의 이해를 다른 사람들에게 *소통(communicate)*해야 합니다. 여러분의 청중은 여러분의 배경지식을 공유하지 않을 것이며 데이터에 깊이 투자하지도 않았을 것입니다. 다른 사람들이 데이터에 대한 훌륭한 멘탈 모델을 빠르게 구축하도록 돕기 위해, 여러분은 플롯을 가능한 한 자명하게 만들기 위해 상당한 노력을 투자해야 합니다. 이 장에서는 ggplot2가 이를 수행하기 위해 제공하는 몇 가지 도구를 배울 것입니다.

이 장은 좋은 그래픽을 만들기 위해 필요한 도구에 중점을 둡니다. 여러분이 원하는 것을 이미 알고 있고, 단지 그것을 어떻게 수행하는지 알 필요만 있다고 가정합니다. 그러한 이유로 이 장을 좋은 일반적인 시각화 서적과 함께 읽는 것을 적극 권장합니다. 우리는 알베르토 카이로(Albert Cairo)가 쓴 [_The Truthful Art_](https://oreil.ly/QIr_w) (New Riders 출판)를 특히 좋아합니다. 이 책은 시각화를 만드는 메커니즘을 가르치지는 않지만, 대신 효과적인 그래픽을 만들기 위해 생각해야 할 것에 중점을 둡니다.

## 사전 준비 (Prerequisites)

이 장에서는 다시 한번 ggplot2에 집중할 것입니다. 또한 데이터 조작을 위해 약간의 dplyr을 사용하고, 기본 눈금(breaks), 레이블(labels), 변환(transformations), 팔레트(palettes)를 재정의(override)하기 위해 *scales*를 사용하며, Kamil Slowikowski의 [ggrepel](https://oreil.ly/IVSL4)과 Thomas Lin Pedersen의 [patchwork](https://oreil.ly/xWxVV)를 포함한 몇 가지 ggplot2 확장 패키지들을 사용할 것입니다. 해당 패키지들이 아직 없다면 <a href="https://rdrr.io/r/utils/install.packages.html" class="orm:hideurl"><code>install.packages()</code></a>로 설치해야 한다는 점을 잊지 마세요.

```
library(tidyverse)
library(scales)
library(ggrepel)
library(patchwork)
```

# 레이블 (Labels)

탐색적 그래픽을 설명적(expository) 그래픽으로 바꿀 때 가장 시작하기 쉬운 곳은 좋은 레이블을 사용하는 것입니다. <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a> 함수로 레이블을 추가합니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  labs(
    x = "Engine displacement (L)",
    y = "Highway fuel economy (mpg)",
    color = "Car type",
    title = "Fuel efficiency generally decreases with engine size",
    subtitle = "Two seaters (sports cars) are an exception because of their light weight",
    caption = "Data from fueleconomy.gov"
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in01.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, where points are colored according to the car class. A smooth curve following the trajectory of the relationship between highway fuel efficiency versus engine size of cars is overlaid. The x-axis is labelled &quot;Engine displacement (L)&quot; and the y-axis is labelled &quot;Highway fuel economy (mpg)&quot;. The legend is labelled &quot;Car type&quot;. The plot is titled &quot;Fuel efficiency generally decreases with engine size&quot;. The subtitle is &quot;Two seaters (sports cars) are an exception because of their light weight&quot; and the caption is &quot;Data from fueleconomy.gov&quot;." />
</figure>

플롯 제목(title)의 목적은 주요 발견 사항을 요약하는 것입니다. "엔진 배기량과 연비의 산점도(A scatterplot of engine displacement vs. fuel economy)"와 같이 플롯이 무엇인지 설명하기만 하는 제목은 피하십시오.

더 많은 텍스트를 추가해야 하는 경우, 두 가지 다른 유용한 레이블이 있습니다. `subtitle`은 제목 아래에 작은 글꼴로 추가적인 세부 정보를 덧붙이고, `caption`은 주로 데이터의 출처를 설명하는 단위를 포함하는 것이 일반적으로 좋은 방법입니다.

텍스트 문자열 대신 수학 공식을 사용하는 것도 가능합니다. 단순히 `""`를 <a href="https://rdrr.io/r/base/substitute.html" class="orm:hideurl"><code>quote()</code></a>로 바꾸고, 사용 가능한 옵션에 대해서는 <a href="https://rdrr.io/r/grDevices/plotmath.html" class="orm:hideurl"><code>?plotmath</code></a>를 읽어보세요.

```
df <- tibble(
  x = 1:10,
  y = cumsum(x^2)
)

ggplot(df, aes(x, y)) +
  geom_point() +
  labs(
    x = quote(x[i]),
    y = quote(sum(x[i] ^ 2, i == 1, n))
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in02.png" alt="Scatterplot with math text on the x and y axis labels. X-axis label says x_i, y-axis label says sum of x_i squared, for i from 1 to n." />
</figure>

## 연습문제 (Exercises)

1. 커스터마이즈된 `title`, `subtitle`, `caption`, `x`, `y`, `color` 레이블을 사용하여 연료 경제성(fuel economy) 데이터에 하나의 플롯을 만드세요.
2. 연료 경제성 데이터를 사용하여 다음 플롯을 다시 만드세요. 점의 색상과 모양이 모두 구동 방식(drivetrain) 유형에 따라 다르다는 점에 유의하세요.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in03.png" alt="Scatterplot of highway versus city fuel efficiency. Shapes and colors of points are determined by type of drivetrain." />
</figure>

3. 지난 한 달 동안 만든 탐색적 그래픽을 가져와서 다른 사람들이 이해하기 쉽도록 유익한 제목을 추가하세요.

# 주석 (Annotations)

플롯의 주요 구성 요소에 레이블을 지정하는 것 외에도, 개별 관측치나 관측치 그룹에 레이블을 지정하는 것이 유용할 때가 많습니다. 마음대로 사용할 수 있는 첫 번째 도구는 <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>입니다. <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>는 <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>와 유사하지만 `label`이라는 미적 매핑을 추가로 가지고 있습니다. 이를 통해 플롯에 텍스트 레이블을 추가할 수 있습니다.

레이블의 출처(sources)는 두 가지가 가능합니다. 첫째, 레이블을 제공하는 티블(tibble)을 가질 수 있습니다. 다음 플롯에서는 각 구동 유형에서 엔진 크기가 가장 큰 자동차를 뽑아 그 정보를 `label_info`라는 새로운 데이터 프레임으로 저장합니다.

```
label_info <- mpg |>
  group_by(drv) |>
  arrange(desc(displ)) |>
  slice_head(n = 1) |>
  mutate(
    drive_type = case_when(
      drv == "f" ~ "front-wheel drive",
      drv == "r" ~ "rear-wheel drive",
      drv == "4" ~ "4-wheel drive"
    )
  ) |>
  select(displ, hwy, drv, drive_type)

label_info
#> # A tibble: 3 × 4
#> # Groups:   drv [3]
#>   displ   hwy drv   drive_type
#>   <dbl> <int> <chr> <chr>
#> 1   6.5    17 4     4-wheel drive
#> 2   5.3    25 f     front-wheel drive
#> 3   7      24 r     rear-wheel drive
```

그런 다음 이 새 데이터 프레임을 사용하여 세 그룹에 직접 레이블을 지정하고, 범례를 플롯에 직접 배치된 레이블로 바꿉니다. `fontface`와 `size` 인자를 사용하여 텍스트 레이블의 모양을 사용자 지정할 수 있습니다. 그것들은 플롯의 나머지 텍스트보다 크고 굵게 표시됩니다. (`theme(legend.position = "none")`은 모든 범례를 끕니다—이에 대해서는 곧 더 이야기하겠습니다.)

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  geom_text(
    data = label_info,
    aes(x = displ, y = hwy, label = drive_type),
    fontface = "bold", size = 5, hjust = "right", vjust = "bottom"
  ) +
  theme(legend.position = "none")
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in04.png" alt="Scatterplot of highway mileage versus engine size where points are colored by drive type. Smooth curves for each drive type are overlaid. Text labels identify the curves as front-wheel, rear-wheel, and 4-wheel." />
</figure>

레이블의 정렬을 제어하기 위해 `hjust` (가로 정렬) 및 `vjust` (세로 정렬)를 사용한 점에 유의하세요.

그러나 방금 만든 주석이 달린 플롯은 레이블들이 서로 겹치고 점들과 겹치기 때문에 읽기 어렵습니다. ggrepel 패키지의 <a href="https://rdrr.io/pkg/ggrepel/man/geom_text_repel.html" class="orm:hideurl"><code>geom_label_repel()</code></a> 함수를 사용하여 이 두 가지 문제를 모두 해결할 수 있습니다. 이 유용한 패키지는 겹치지 않도록 레이블을 자동으로 조정합니다.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  geom_label_repel(
    data = label_info,
    aes(x = displ, y = hwy, label = drive_type),
    fontface = "bold", size = 5, nudge_y = 2
  ) +
  theme(legend.position = "none")
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in05.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, where points are colored according to the car class. Some points are labelled with the car&#39;s name. The labels are box with white, transparent background and positioned to not overlap." />
</figure>

ggrepel 패키지의 <a href="https://rdrr.io/pkg/ggrepel/man/geom_text_repel.html" class="orm:hideurl"><code>geom_text_repel()</code></a>을 사용하여 플롯의 특정 점을 강조하는 동일한 아이디어를 사용할 수도 있습니다. 여기에 사용된 또 다른 편리한 기술에 주목하세요. 우리는 레이블이 지정된 점을 더 강조하기 위해 크고 속이 빈(hollow) 점들의 두 번째 레이어를 추가했습니다.

```
potential_outliers <- mpg |>
  filter(hwy > 40 | (hwy > 20 & displ > 5))

ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  geom_text_repel(data = potential_outliers, aes(label = model)) +
  geom_point(data = potential_outliers, color = "red") +
  geom_point(
    data = potential_outliers,
    color = "red", size = 3, shape = "circle open"
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in06.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars. Points where highway mileage is above 40 as well as above 20 with engine size above 5 are red, with a hollow red circle, and labelled with model name of the car." />
</figure>

<a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a> 및 <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_label()</code></a> 외에도, ggplot2에는 플롯에 주석을 다는 데 도움이 되는 다른 많은 기하 객체(geoms)가 있다는 것을 기억하세요. 몇 가지 아이디어:

- 참조선(reference lines)을 추가하려면 <a href="https://ggplot2.tidyverse.org/reference/geom_abline.html" class="orm:hideurl"><code>geom_hline()</code></a> 및 <a href="https://ggplot2.tidyverse.org/reference/geom_abline.html" class="orm:hideurl"><code>geom_vline()</code></a>을 사용하세요. 우리는 종종 그것들을 굵게(`linewidth = 2`)하고 희게(`color = white`) 만들어 주요 데이터 레이어 밑에 그립니다. 그렇게 하면 데이터에서 주의를 분산시키지 않으면서도 선을 쉽게 볼 수 있습니다.
- 관심 있는 점 주위에 직사각형을 그리려면 <a href="https://ggplot2.tidyverse.org/reference/geom_tile.html" class="orm:hideurl"><code>geom_rect()</code></a>를 사용하세요. 직사각형의 경계는 미적 매핑 `xmin`, `xmax`, `ymin`, `ymax`로 정의됩니다. 대안으로, [ggforce 패키지](https://oreil.ly/DZtL1)를 살펴보세요. 특히 <a href="https://ggforce.data-imaginist.com/reference/geom_mark_hull.html" class="orm:hideurl"><code>geom_mark_hull()</code></a>은 껍질(hulls)을 사용하여 점들의 하위 집합에 주석을 달 수 있게 해줍니다.
- 점에 화살표로 주의를 끌려면 `arrow` 인자와 함께 <a href="https://ggplot2.tidyverse.org/reference/geom_segment.html" class="orm:hideurl"><code>geom_segment()</code></a>를 사용하세요. 시작 위치를 정의하려면 미적 매핑 `x`와 `y`를 사용하고, 끝 위치를 정의하려면 `xend`와 `yend`를 사용하세요.

플롯에 주석을 추가하는 또 다른 편리한 함수는 <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a>입니다. 경험칙상, geoms는 일반적으로 데이터의 하위 집합을 강조하는 데 유용한 반면, <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a>는 플롯에 하나 또는 몇 개의 주석 요소를 추가하는 데 유용합니다.

<a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a> 사용을 시연하기 위해, 플롯에 추가할 텍스트를 만들어 보겠습니다. 텍스트가 약간 길기 때문에 줄당 원하는 글자 수에 따라 자동으로 줄바꿈을 추가해 주는 <a href="https://stringr.tidyverse.org/reference/str_wrap.html" class="orm:hideurl"><code>stringr::str_wrap()</code></a>을 사용할 것입니다.

```
trend_text <- "Larger engine sizes tend to\nhave lower fuel economy." |>
  str_wrap(width = 30)
trend_text
#> [1] "Larger engine sizes tend to\nhave lower fuel economy."
```

그런 다음, 우리는 두 층의 주석을 추가합니다. 하나는 라벨 geom이고 다른 하나는 세그먼트(선분) geom입니다. 둘의 `x`와 `y` 미적 매핑은 주석이 시작되는 위치를 정의하고, 세그먼트 주석의 `xend`와 `yend` 미적 매핑은 세그먼트의 끝 위치가 시작되는 곳을 정의합니다. 세그먼트가 화살표 모양으로 지정된 것도 주목하세요.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  annotate(
    geom = "label", x = 3.5, y = 38,
    label = trend_text,
    hjust = "left", color = "red"
  ) +
  annotate(
    geom = "segment",
    x = 3, y = 35, xend = 5, yend = 25, color = "red",
    arrow = arrow(type = "closed")
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in07.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars. A red arrow pointing down follows the trend of the points and the annotation placed next to the arrow reads &quot;Larger engine sizes tend to have lower fuel economy&quot;. The arrow and the annotation text is red." />
</figure>

주석은 시각화의 주요 요점(takeaways)과 흥미로운 기능(features)을 소통하는 강력한 도구입니다. 유일한 한계는 여러분의 상상력(그리고 미적으로 보기 좋게 주석을 배치하는 것에 대한 여러분의 인내심)뿐입니다!

## 연습문제 (Exercises)

1. 위치를 무한대(infinite positions)로 설정하여 플롯의 네 모서리에 텍스트를 배치하려면 <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>를 사용해 보세요.
2. 티블을 만들지 않고도 마지막 플롯의 한가운데에 포인트 geom을 추가하려면 <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a>를 사용해 보세요. 점의 모양, 크기 또는 색상을 사용자 지정하세요.
3. <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>를 사용한 레이블은 패싯 분할(faceting)과 어떻게 상호 작용하나요? 단일 패싯에 레이블을 추가하려면 어떻게 해야 하나요? 각 패싯에 다른 레이블을 넣으려면 어떻게 해야 하나요? (힌트: <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>로 전달되는 데이터세트에 대해 생각해 보세요.)
4. <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_label()</code></a>의 어떤 인자가 배경 상자(background box)의 모양을 제어하나요?
5. <a href="https://rdrr.io/r/grid/arrow.html" class="orm:hideurl"><code>arrow()</code></a>의 네 가지 인자는 무엇입니까? 그것들은 어떻게 작동하나요? 가장 중요한 옵션을 시연하는 일련의 플롯을 만드세요.

# 척도 (Scales)

소통을 위해 플롯을 더 좋게 만드는 세 번째 방법은 척도(scales)를 조정하는 것입니다. 척도는 미적 매핑이 시각적으로 어떻게 나타날지 제어합니다.

## 기본 척도 (Default Scales)

일반적으로 ggplot2는 백그라운드에서 자동으로 척도를 추가합니다. 예를 들어 다음과 같이 입력할 때:

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class))
```

ggplot2는 보이지 않게 자동으로 기본 척도를 추가합니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  scale_x_continuous() +
  scale_y_continuous() +
  scale_color_discrete()
```

척도의 명명 규칙(naming scheme)에 유의하세요. `scale_` 뒤에 미적 매핑의 이름, 그 다음에 `_`, 그리고 척도의 이름이 옵니다. 기본 척도는 척도가 정렬되는 변수 유형(연속형(continuous), 이산형(discrete), 날짜-시간(date-time) 또는 날짜(date))에 따라 명명됩니다. <a href="https://ggplot2.tidyverse.org/reference/scale_continuous.html" class="orm:hideurl"><code>scale_x_continuous()</code></a>는 `displ`의 숫자 값을 x축의 연속 숫자 선상에 놓고, <a href="https://ggplot2.tidyverse.org/reference/scale_colour_discrete.html" class="orm:hideurl"><code>scale_color_discrete()</code></a>는 자동차의 각 `class`에 대한 색상을 선택하는 식입니다. 다음으로 배울 기본이 아닌 많은 척도들이 있습니다.

기본 척도는 광범위한 입력에 대해 잘 작동하도록 신중하게 선택되었습니다. 그럼에도 불구하고 두 가지 이유로 기본값을 재정의(override)하고 싶을 수 있습니다.

- 기본 척도의 일부 매개변수(parameters)를 조정하고 싶을 수 있습니다. 이를 통해 축의 눈금을 변경하거나 범례의 주요 레이블을 변경하는 등의 작업을 할 수 있습니다.
- 척도 전체를 대체하고 완전히 다른 알고리즘을 사용하고 싶을 수 있습니다. 데이터에 대해 더 많이 알고 있기 때문에 종종 기본값보다 더 나은 작업을 수행할 수 있습니다.

## 축 눈금과 범례 키 (Axis Ticks and Legend Keys)

축과 범례를 통틀어 *가이드(guides)*라고 부릅니다. 축은 `x`와 `y` 미적 매핑에 사용되고, 범례는 다른 모든 것에 사용됩니다.

축의 눈금(ticks)과 범례의 키(keys) 모양에 영향을 주는 두 가지 주요 인자(arguments)가 있습니다. `breaks`와 `labels`입니다. `breaks` 인자는 눈금의 위치 또는 키와 연관된 값을 제어합니다. `labels` 인자는 각 눈금/키와 관련된 텍스트 레이블을 제어합니다. `breaks`의 가장 일반적인 용도는 기본 선택을 재정의(override)하는 것입니다.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  scale_y_continuous(breaks = seq(15, 40, by = 5))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in08.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, colored by drive. The y-axis has breaks starting at 15 and ending at 40, increasing by 5." />
</figure>

`labels`도 동일한 방식(`breaks`와 길이가 같은 문자형 벡터)으로 사용할 수 있지만, `NULL`로 설정하여 레이블을 완전히 표시하지 않도록 할 수도 있습니다. 이는 지도나 절대 수치를 공유할 수 없는 게시용 플롯에 유용할 수 있습니다. `breaks`와 `labels`를 사용하여 범례의 모양을 제어할 수도 있습니다. 범주형 변수의 이산형(discrete) 척도의 경우, `labels`는 기존 수준(levels) 이름과 그에 대해 원하는 레이블로 구성된 이름 있는 리스트(named list)가 될 수 있습니다.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  scale_x_continuous(labels = NULL) +
  scale_y_continuous(labels = NULL) +
  scale_color_discrete(labels = c("4" = "4-wheel", "f" = "front", "r" = "rear"))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in09.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, colored by drive. The x and y-axes do not have any labels at the axis ticks. The legend has custom labels: 4-wheel, front, rear." />
</figure>

scales 패키지의 레이블링 함수와 결합된 `labels` 인자는 숫자를 통화, 백분율 등으로 서식 지정(formatting)하는 데에도 유용합니다. 왼쪽 플롯은 달러 기호와 천 단위 구분 쉼표를 추가하는 <a href="https://scales.r-lib.org/reference/label_dollar.html" class="orm:hideurl"><code>label_dollar()</code></a>를 사용한 기본 레이블링을 보여줍니다. 오른쪽 플롯은 달러 값을 1,000으로 나누고 접미사 "K"("수천"을 의미)를 추가할 뿐만 아니라 사용자 지정 눈금(breaks)을 추가하여 시각화를 더욱 세밀하게 제어합니다. `breaks`는 원래 데이터 척도를 기준으로 한다는 점에 유의하세요.

```
# Left
ggplot(diamonds, aes(x = price, y = cut)) +
  geom_boxplot(alpha = 0.05) +
  scale_x_continuous(labels = label_dollar())

# Right
ggplot(diamonds, aes(x = price, y = cut)) +
  geom_boxplot(alpha = 0.05) +
  scale_x_continuous(
    labels = label_dollar(scale = 1/1000, suffix = "K"),
    breaks = seq(1000, 19000, by = 6000)
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in10.png" alt="Two side-by-side box plots of price versus cut of diamonds. The outliers are transparent. On both plots the x-axis labels are formatted as dollars. The x-axis labels on the plot start at $0 and go to $15,000, increasing by $5,000. The x-axis labels on the right plot start at $1K and go to $19K, increasing by $6K." />
</figure>

또 다른 편리한 레이블 함수는 <a href="https://scales.r-lib.org/reference/label_percent.html" class="orm:hideurl"><code>label_percent()</code></a>입니다.

```
ggplot(diamonds, aes(x = cut, fill = clarity)) +
  geom_bar(position = "fill") +
  scale_y_continuous(name = "Percentage", labels = label_percent())
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in11.png" alt="Segmented bar plots of cut, filled with levels of clarity. The y-axis labels start at 0% and go to 100%, increasing by 25%. The y-axis label name is &quot;Percentage&quot;." />
</figure>

`breaks`의 또 다른 용도는 데이터 포인트가 비교적 적고 관측치가 정확히 어디에서 발생하는지 강조하고 싶을 때입니다. 예를 들어, 역대 미국 대통령의 임기 시작과 끝을 보여주는 이 플롯을 살펴보세요.

```
presidential |>
  mutate(id = 33 + row_number()) |>
  ggplot(aes(x = start, y = id)) +
  geom_point() +
  geom_segment(aes(xend = end, yend = id)) +
  scale_x_date(name = NULL, breaks = presidential$start, date_labels = "'%y")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in12.png" alt="Line plot of id number of presidents versus the year they started their presidency. Start year is marked with a point and a segment that starts there and ends at the end of the presidency. The x-axis labels are formatted as two digit years starting with an apostrophe, e.g., &#39;53." />
</figure>

이 인자에 대해서는 미적 매핑을 수행할 수 없기 때문에, `breaks` 인자를 위해 `start` 변수를 `presidential$start`라는 벡터로 추출했다는 점에 주목하세요. 또한 날짜 및 날짜-시간 척도에 대한 눈금과 레이블을 지정하는 방식이 약간 다릅니다.

- `date_labels`는 <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>parse_datetime()</code></a>과 동일한 형식으로 형식 지정자(format specification)를 사용합니다.
- `date_breaks` (여기서는 표시되지 않음)는 "2 days(2일)" 또는 "1 month(1개월)"과 같은 문자열을 취합니다.

## 범례 레이아웃 (Legend Layout)

축을 조정할 때 `breaks`와 `labels`를 가장 자주 사용하게 될 것입니다. 이 둘은 범례에서도 모두 작동하지만, 더 자주 사용할 만한 몇 가지 다른 기술이 있습니다.

범례의 전체적인 위치를 제어하려면 <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a> 설정을 사용해야 합니다. 테마(themes)에 대해서는 이 장의 끝에서 다시 다루겠지만, 간단히 말해서 데이터가 아닌 플롯의 부분을 제어합니다. 테마 설정 `legend.position`은 범례가 그려지는 위치를 제어합니다.

```
base <- ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class))

base + theme(legend.position = "right") # the default
base + theme(legend.position = "left")
base +
  theme(legend.position = "top") +
  guides(col = guide_legend(nrow = 3))
base +
  theme(legend.position = "bottom") +
  guides(col = guide_legend(nrow = 3))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in13.png" alt="Four scatterplots of highway fuel efficiency versus engine size of cars where points are colored based on class of car. Clockwise, the legend is placed on the right, left, top, and bottom of the plot." />
</figure>

플롯이 짧고 넓은 경우에는 범례를 위쪽이나 아래쪽에 배치하고, 높고 좁은 경우에는 왼쪽이나 오른쪽에 배치하세요. `legend.position = "none"`을 사용하여 범례의 표시를 완전히 억제할 수도 있습니다.

개별 범례의 표시를 제어하려면 <a href="https://ggplot2.tidyverse.org/reference/guides.html" class="orm:hideurl"><code>guides()</code></a>와 함께 <a href="https://ggplot2.tidyverse.org/reference/guide_legend.html" class="orm:hideurl"><code>guide_legend()</code></a> 또는 <a href="https://ggplot2.tidyverse.org/reference/guide_colourbar.html" class="orm:hideurl"><code>guide_colorbar()</code></a>를 사용하세요. 다음 예제는 두 가지 중요한 설정을 보여줍니다. `nrow`로 범례가 사용하는 행 수를 제어하는 것, 그리고 점을 더 크게 만들기 위해 미적 매핑 중 하나를 재정의(override)하는 것입니다. 이 기능은 많은 점을 플롯에 표시하기 위해 낮은 `alpha` 값을 사용했을 때 특히 유용합니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2, override.aes = list(size = 4)))
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in14.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars where points are colored based on class of car. Overlaid on the plot is a smooth curve. The legend is in the bottom and classes are listed horizontally in two rows. The points in the legend are larger than the points in the plot." />
</figure>

<a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>에서와 마찬가지로, <a href="https://ggplot2.tidyverse.org/reference/guides.html" class="orm:hideurl"><code>guides()</code></a>에 있는 인자의 이름은 미적 매핑의 이름과 일치해야 한다는 점에 유의하세요.

## 척도 대체하기 (Replacing a Scale)

세부 사항을 조금 조정하는 대신, 척도를 완전히 교체할 수도 있습니다. 가장 자주 바꾸고 싶어 할 두 가지 유형의 척도는 연속형 위치 척도(continuous position scales)와 색상 척도(color scales)입니다. 다행히 동일한 원리가 다른 모든 미적 매핑에도 적용되므로, 위치와 색상을 숙달하고 나면 다른 척도 대체 방법도 빠르게 익힐 수 있습니다.

변수의 변환(transformations)을 플롯하는 것이 유용할 때가 있습니다. 예를 들어, `carat`과 `price`에 로그 변환(log transform)을 적용하면 둘 사이의 정확한 관계를 더 쉽게 볼 수 있습니다.

```
# Left
ggplot(diamonds, aes(x = carat, y = price)) +
  geom_bin2d()

# Right
ggplot(diamonds, aes(x = log10(carat), y = log10(price))) +
  geom_bin2d()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in15.png" alt="Two plots of price versus carat of diamonds. Data binned and the color of the rectangles representing each bin based on the number of points that fall into that bin. In the plot on the right, price and carat values are logged and the axis labels shows the logged values." />
</figure>

그러나 이 변환의 단점은 축이 이제 변환된 값으로 레이블이 지정되어 플롯을 해석하기 어려워진다는 것입니다. 미적 매핑에서 변환을 수행하는 대신, 척도(scale)를 통해 변환을 수행할 수 있습니다. 이것은 시각적으로 완전히 동일하지만, 축은 원래의 데이터 척도로 레이블이 지정됩니다.

```
ggplot(diamonds, aes(x = carat, y = price)) +
  geom_bin2d() +
  scale_x_log10() +
  scale_y_log10()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in16.png" alt="Plot of price versus carat of diamonds. Data binned and the color of the rectangles representing each bin based on the number of points that fall into that bin. The axis labels are on the original data scale." />
</figure>

자주 커스터마이즈되는 또 다른 척도는 색상(color)입니다. 기본 범주형 척도는 색상환(color wheel)에서 균등하게 간격을 둔 색상들을 선택합니다. 유용한 대안은 일반적인 색맹(color blindness)을 가진 사람들에게 더 잘 보이도록 수작업으로 조정된 ColorBrewer 척도입니다. 다음의 두 플롯은 비슷해 보이지만, 빨간색과 초록색 색조에 충분한 차이가 있어 적록 색맹인 사람도 오른쪽의 점들을 구별할 수 있습니다.<sup><a href="ch11.html#idm44771304642976" id="idm44771304642976-marker" data-type="noteref">1</a></sup>

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv))

ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  scale_color_brewer(palette = "Set1")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in17.png" alt="Two scatterplots of highway mileage versus engine size where points are colored by drive type. The plot on the left uses the default ggplot2 color palette and the plot on the right uses a different color palette." />
</figure>

접근성(accessibility)을 개선하기 위한 더 단순한 기술들을 잊지 마세요. 만약 몇 가지 색상만 있다면, 중복되는 모양(shape) 매핑을 추가할 수 있습니다. 이는 플롯이 흑백에서도 해석 가능하게 하는 데 도움이 됩니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv, shape = drv)) +
  scale_color_brewer(palette = "Set1")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in18.png" alt="Two scatterplots of highway mileage versus engine size where both color and shape of points are based on drive type. The color palette is not the default ggplot2 palette." />
</figure>

ColorBrewer 척도는 [온라인에 문서화되어 있으며](https://oreil.ly/LNHAy), Erich Neuwirth가 만든 RColorBrewer 패키지를 통해 R에서 사용할 수 있습니다. <a href="#fig-brewer" data-type="xref">그림 11-1</a>은 모든 팔레트의 전체 목록을 보여줍니다. 순차적(sequential, 위쪽) 및 발산적(diverging, 아래쪽) 팔레트는 범주형 값에 순서가 있거나 "중간(middle)"이 있을 때 특히 유용합니다. 이는 연속형 변수를 범주형 변수로 만들기 위해 <a href="https://rdrr.io/r/base/cut.html" class="orm:hideurl"><code>cut()</code></a>을 사용했을 때 종종 나타납니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1101.png" alt="All ColorBrewer scales. One group goes from light to dark colors. Another group is a set of non ordinal colors. And the last group has diverging scales (from dark to light to dark again). Within each set there are a number of palettes." />
<h6 id="figure-11-1.-all-colorbrewer-scales.">그림 11-1. 모든 ColorBrewer 척도.</h6>
</figure>

값과 색상 사이에 사전 정의된 매핑이 있는 경우 <a href="https://ggplot2.tidyverse.org/reference/scale_manual.html" class="orm:hideurl"><code>scale_color_manual()</code></a>을 사용하세요. 예를 들어, 대통령의 정당을 색상에 매핑한다면 공화당에는 빨간색, 민주당에는 파란색을 표준 매핑으로 사용하고 싶을 것입니다. 이러한 색상을 할당하는 한 가지 방법은 16진수 색상 코드(hex color codes)를 사용하는 것입니다.

```
presidential |>
  mutate(id = 33 + row_number()) |>
  ggplot(aes(x = start, y = id, color = party)) +
  geom_point() +
  geom_segment(aes(xend = end, yend = id)) +
  scale_color_manual(values = c(Republican = "#E81B23", Democratic = "#00AEF3"))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in19.png" alt="Line plot of id number of presidents versus the year they started their presidency. Start year is marked with a point and a segment that starts there and ends at the end of the presidency. Democratic presidents are represented in blue and Republicans in red." />
</figure>

연속형 색상의 경우 기본 내장된 <a href="https://ggplot2.tidyverse.org/reference/scale_gradient.html" class="orm:hideurl"><code>scale_color_gradient()</code></a> 또는 <a href="https://ggplot2.tidyverse.org/reference/scale_gradient.html" class="orm:hideurl"><code>scale_fill_gradient()</code></a>를 사용할 수 있습니다. 발산적(diverging) 척도가 있다면 <a href="https://ggplot2.tidyverse.org/reference/scale_gradient.html" class="orm:hideurl"><code>scale_color_gradient2()</code></a>를 사용할 수 있습니다. 이것은 예를 들어, 양수 값과 음수 값에 각각 다른 색상을 부여할 수 있게 해줍니다. 때로는 평균(mean) 위나 아래의 점을 구별하고 싶을 때도 유용합니다.

또 다른 옵션은 viridis 색상 척도를 사용하는 것입니다. 디자이너인 Nathaniel Smith와 Stéfan van der Walt는 컬러와 흑백 모두에서 지각적으로(perceptually) 균일할 뿐만 아니라 다양한 형태의 색맹인 사람들도 지각할 수 있는 연속 색상표를 신중하게 맞춤 제작했습니다. 이 척도들은 ggplot2에서 연속형(`c`), 이산형(`d`), 구간형(`b`) 팔레트로 제공됩니다.

```
df <- tibble(
  x = rnorm(10000),
  y = rnorm(10000)
)

ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed() +
  labs(title = "Default, continuous", x = NULL, y = NULL)

ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed() +
  scale_fill_viridis_c() +
  labs(title = "Viridis, continuous", x = NULL, y = NULL)

ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed() +
  scale_fill_viridis_b() +
  labs(title = "Viridis, binned", x = NULL, y = NULL)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in20.png" alt="Three hex plots where the color of the hexes show the number of observations that fall into that hex bin. The first plot uses the default, continuous ggplot2 scale. The second plot uses the viridis, continuous scale, and the third plot uses the viridis, binned scale." />
</figure>

모든 색상 척도는 각각 `color`와 `fill` 미적 매핑을 위한 두 가지 변형(varieties), 즉 `scale_color_*()`와 `scale_fill_*()`로 제공된다는 점에 유의하세요 (색상 척도는 영국식 철자(colour)와 미국식 철자(color) 모두 사용할 수 있습니다).

## 줌 (Zooming)

플롯의 한계(limits)를 제어하는 데는 세 가지 방법이 있습니다.

- 어떤 데이터가 플롯될지 조정하기
- 각 척도의 한계 설정하기
- <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>에서 `xlim`과 `ylim` 설정하기

우리는 일련의 플롯에서 이러한 옵션들을 시연할 것입니다. 왼쪽 플롯은 엔진 크기와 연비 사이의 관계를 구동 방식(drivetrain) 유형별로 색상을 다르게 하여 보여줍니다. 오른쪽 플롯은 동일한 변수를 보여주지만 플롯되는 데이터의 하위 집합(subsets)을 취합니다. 데이터의 하위 집합을 취하는 것은 매끄러운 곡선(smooth curve)뿐만 아니라 x축과 y축 척도에도 영향을 미쳤습니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth()

# Right
mpg |>
  filter(displ >= 5 & displ <= 6 & hwy >= 10 & hwy <= 25) |>
  ggplot(aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in21.png" alt="On the left, scatterplot of highway mileage vs. displacement, with displacement. The smooth curve overlaid shows a decreasing, and then increasing, trend like a hockey stick. On the right, same variables are plotted with displacement ranging only from 5 to 6 and highway mileage ranging only from 10 to 25. The smooth curve overlaid shows a trend that&#39;s slightly increasing first and then decreasing." />
</figure>

이들을 다음의 두 플롯과 비교해 봅시다. 왼쪽 플롯은 개별 척도에 `limits`를 설정하고, 오른쪽 플롯은 <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>에서 그것들을 설정합니다. 우리는 한계를 줄이는 것이 데이터의 하위 집합을 취하는 것과 동일하다는 것을 볼 수 있습니다. 따라서 플롯의 특정 영역을 확대(zoom in)하려면 일반적으로 <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>을 사용하는 것이 가장 좋습니다.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth() +
  scale_x_continuous(limits = c(5, 6)) +
  scale_y_continuous(limits = c(10, 25))

# Right
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth() +
  coord_cartesian(xlim = c(5, 6), ylim = c(10, 25))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in22.png" alt="On the left, scatterplot of highway mileage vs. displacement, with displacement ranging from 5 to 6 and highway mileage ranging from 10 to 25. The smooth curve overlaid shows a trend that&#39;s slightly increasing first and then decreasing. On the right, same variables are plotted with the same limits; however, the smooth curve overlaid shows a relatively flat trend with a slight increase at the end." />
</figure>

반면에, 개별 척도에 `limits`를 설정하는 것은 한계를 *확장(expand)*하고 싶을 때(다른 플롯들 간에 척도를 일치시키기 위해) 일반적으로 더 유용합니다. 예를 들어, 두 종류의 자동차를 추출하여 별도로 플롯하는 경우, 세 가지 척도(x축, y축, 색상 미적 매핑)가 모두 다른 범위를 가지기 때문에 플롯을 비교하기가 어렵습니다.

```
suv <- mpg |> filter(class == "suv")
compact <- mpg |> filter(class == "compact")

# Left
ggplot(suv, aes(x = displ, y = hwy, color = drv)) +
  geom_point()

# Right
ggplot(compact, aes(x = displ, y = hwy, color = drv)) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in23.png" alt="On the left, a scatterplot of highway mileage vs. displacement of SUVs. On the right, a scatterplot of the same variables for compact cars. Points are colored by drive type for both plots. Among SUVs, more of the cars are 4-wheel drive and the others are rear-wheel drive, while among compact cars more of the cars are front-wheel drive and the others are 4-wheel drive. SUV plot shows a clear negative relationship between highway mileage and displacement, while in the compact cars plot, the relationship is much flatter." />
</figure>

이 문제를 극복하는 한 가지 방법은 전체 데이터의 `limits`로 척도를 훈련(training)시켜, 여러 플롯 간에 척도를 공유하는 것입니다.

```
x_scale <- scale_x_continuous(limits = range(mpg$displ))
y_scale <- scale_y_continuous(limits = range(mpg$hwy))
col_scale <- scale_color_discrete(limits = unique(mpg$drv))

# Left
ggplot(suv, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  x_scale +
  y_scale +
  col_scale

# Right
ggplot(compact, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  x_scale +
  y_scale +
  col_scale
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in24.png" alt="On the left, a scatterplot of highway mileage vs. displacement of SUVs. On the right, a scatterplot of the same variables for compact cars. Points are colored by drive type for both plots. Both plots are plotted on the same scale for highway mileage, displacement, and drive type, resulting in the legend showing all three types (front, rear, and 4-wheel drive) for both plots even though there are no front-wheel drive SUVs and no rear-wheel drive compact cars. Since the x and y scales are the same, and go well beyond minimum or maximum highway mileage and displacement, the points do not take up the entire plotting area." />
</figure>

이 특별한 경우에는 단순히 패싯 분할(faceting)을 사용할 수도 있었겠지만, 이 기술은 예를 들어 보고서의 여러 페이지에 플롯을 펼치고 싶을 때 더 널리 유용합니다.

## 연습문제 (Exercises)

1. 다음 코드가 기본 척도를 재정의하지 못하는 이유는 무엇입니까?

   ```
   df <- tibble(
     x = rnorm(10000),
     y = rnorm(10000)
   )

   ggplot(df, aes(x, y)) +
     geom_hex() +
     scale_color_gradient(low = "white", high = "red") +
     coord_fixed()
   ```

2. 모든 척도에 대한 첫 번째 인자는 무엇입니까? 그것은 <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>와 어떻게 다릅니까?
3. 다음을 통해 대통령 임기의 디스플레이를 변경하세요.
   1. 색상과 x축 눈금을 사용자 정의하는 두 가지 변형(variants) 결합하기
   2. y축의 디스플레이 개선하기
   3. 각 임기에 대통령 이름으로 레이블 지정하기
   4. 정보가 담긴 플롯 레이블 추가하기
   5. 4년마다 눈금 배치하기 (이것은 생각보다 까다롭습니다!)
4. 먼저, 다음 플롯을 만드세요. 그런 다음 `override.aes`를 사용하여 범례를 보기 쉽게 만들도록 코드를 수정하세요.

   ```
   ggplot(diamonds, aes(x = carat, y = price)) +
     geom_point(aes(color = cut), alpha = 1/20)
   ```

# 테마 (Themes)

마지막으로, 테마(theme)를 사용하여 플롯의 데이터가 아닌 요소를 사용자 지정할 수 있습니다.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in25.png" alt="Scatterplot of highway mileage vs. displacement of cars, colored by class of car. The plot background is white, with gray grid lines." />
</figure>

ggplot2에는 <a href="#fig-themes" data-type="xref">그림 11-2</a>에 표시된 8가지 테마가 포함되어 있으며, <a href="https://ggplot2.tidyverse.org/reference/ggtheme.html" class="orm:hideurl"><code>theme_gray()</code></a>가 기본값입니다.<sup><a href="ch11.html#idm44771303447760" id="idm44771303447760-marker" data-type="noteref">2</a></sup> Jeffrey Arnold가 만든 [ggthemes](https://oreil.ly/F1nga)와 같은 추가 기능(add-on) 패키지에는 더 많은 테마가 포함되어 있습니다. 특정 기업이나 학술지의 스타일을 일치시키려는 경우 자신만의 테마를 만들 수도 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1102.png" alt="Eight barplots created with ggplot2, each with one of the eight built-in themes: theme_bw() - White background with grid lines, theme_light() - Light axes and grid lines, theme_classic() - Classic theme, axes but no grid lines, theme_linedraw() - Only black lines, theme_dark() - Dark background for contrast, theme_minimal() - Minimal theme, no background, theme_gray() - Gray background (default theme), theme_void() - Empty theme, only geoms are visible." />
<h6 id="figure-11-2.-the-eight-themes-built-in-to-ggplot2.">그림 11-2. ggplot2에 내장된 8가지 테마.</h6>
</figure>

y축에 사용되는 글꼴의 크기와 색상 등 각 테마의 개별 구성 요소를 제어하는 것도 가능합니다. 우리는 `legend.position`이 범례가 그려지는 위치를 제어한다는 것을 이미 보았습니다. <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a>을 사용하여 사용자 정의할 수 있는 범례의 다른 측면들이 많습니다. 예를 들어, 다음 플롯에서는 범례의 방향을 변경하고 그 주위에 검은색 테두리를 넣습니다. 범례 상자 및 플롯 제목 테마 요소의 사용자 지정은 `element_*()` 함수를 통해 수행된다는 점에 유의하세요. 이 함수들은 데이터가 아닌 구성 요소의 스타일을 지정합니다. 예컨대, 제목 텍스트는 <a href="https://ggplot2.tidyverse.org/reference/element.html" class="orm:hideurl"><code>element_text()</code></a>의 `face` 인자에서 굵게(bolded) 설정되고, 범례 테두리 색상은 <a href="https://ggplot2.tidyverse.org/reference/element.html" class="orm:hideurl"><code>element_rect()</code></a>의 `color` 인자에 정의됩니다. 제목과 캡션의 위치를 제어하는 테마 요소는 각각 `plot.title.position`과 `plot.caption.position`입니다. 다음 플롯에서는 플롯 패널(기본값) 대신 이러한 요소들이 전체 플롯 영역에 정렬되어 있음을 나타내기 위해 이 값들이 `"plot"`으로 설정되어 있습니다. 제목과 캡션 텍스트의 형식을 지정하기 위한 배치를 변경하기 위해 몇 가지 다른 유용한 <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a> 구성 요소가 사용됩니다.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  labs(
    title = "Larger engine sizes tend to have lower fuel economy",
    caption = "Source: https://fueleconomy.gov."
  ) +
  theme(
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal",
    legend.box.background = element_rect(color = "black"),
    plot.title = element_text(face = "bold"),
    plot.title.position = "plot",
    plot.caption.position = "plot",
    plot.caption = element_text(hjust = 0)
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in26.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, colored by drive. The plot is titled &#39;Larger engine sizes tend to have lower fuel economy&#39; with the caption pointing to the source of the data, fueleconomy.gov. The caption and title are left justified, the legend is inside of the plot with a black border." />
</figure>

모든 <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a> 구성 요소에 대한 개요는 <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>?theme</code></a>의 도움말을 참조하세요. [ggplot2 책](https://oreil.ly/T4Jxn)도 테마 설정에 대한 전체 세부 정보를 얻기에 좋은 곳입니다.

## 연습문제 (Exercises)

1. ggthemes 패키지에서 제공하는 테마를 하나 골라 방금 만든 마지막 플롯에 적용하세요.
2. 플롯의 축 레이블을 파란색 굵은 글씨로 만드세요.

# 레이아웃 (Layout)

지금까지 우리는 단일 플롯을 생성하고 수정하는 방법에 대해 이야기했습니다. 만약 특정 방식으로 레이아웃을 배치하고 싶은 여러 개의 플롯이 있다면 어떨까요? patchwork 패키지를 사용하면 개별 플롯을 동일한 그래픽으로 결합할 수 있습니다. 우리는 이 장의 앞부분에서 이 패키지를 로드했습니다.

두 개의 플롯을 나란히 배치하려면, 단순히 그것들을 서로 더하기만 하면 됩니다. 먼저 플롯을 만들고 객체(다음 예제에서는 `p1`과 `p2`라고 부름)로 저장해야 한다는 점에 유의하세요. 그런 다음, `+`를 사용하여 그것들을 나란히 배치합니다.

```
p1 <- ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  labs(title = "Plot 1")
p2 <- ggplot(mpg, aes(x = drv, y = hwy)) +
  geom_boxplot() +
  labs(title = "Plot 2")
p1 + p2
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in27.png" alt="Two plots (a scatterplot of highway mileage versus engine size and a side-by-side boxplots of highway mileage versus drivetrain) placed next to each other." />
</figure>

이전 코드 청크에서 patchwork 패키지의 새로운 함수를 사용하지 않았다는 점에 유의하는 것이 중요합니다. 대신, 이 패키지는 `+` 연산자에 새로운 기능을 추가했습니다.

patchwork를 사용하여 복잡한 플롯 레이아웃을 만들 수도 있습니다. 다음에서 `|`는 `p1`과 `p3`를 나란히 배치하고, `/`는 `p2`를 다음 줄로 이동시킵니다.

```
p3 <- ggplot(mpg, aes(x = cty, y = hwy)) +
  geom_point() +
  labs(title = "Plot 3")
(p1 | p3) / p2
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in28.png" alt="Three plots laid out such that first and third plot are next to each other and the second plot stretched beneath them. The first plot is a scatterplot of highway mileage versus engine size, third plot is a scatterplot of highway mileage versus city mileage, and the third plot is side-by-side boxplots of highway mileage versus drivetrain) placed next to each other." />
</figure>

게다가 patchwork를 사용하면 여러 플롯의 범례를 하나의 공통 범례로 모으고, 범례의 위치와 플롯의 크기를 사용자 지정하며, 플롯에 공통 제목, 부제목, 캡션 등을 추가할 수 있습니다. 여기서는 다섯 개의 플롯을 만들었습니다. 상자 그림과 산점도의 범례를 끄고, 밀도 플롯(density plots)의 범례를 `& theme(legend.position = "top")`을 사용하여 플롯 상단에 모았습니다. 여기서 일반적인 `+` 대신 `&` 연산자가 사용된 것에 주목하세요. 이는 개별 ggplot이 아니라 patchwork 플롯 전체에 대한 테마를 수정하고 있기 때문입니다. 범례는 플롯 상단의 <a href="https://patchwork.data-imaginist.com/reference/guide_area.html" class="orm:hideurl"><code>guide_area()</code></a> 안쪽에 배치됩니다. 마지막으로, patchwork의 다양한 구성 요소들의 높이(heights)도 사용자 지정했습니다. 가이드(guide)의 높이는 1, 상자 그림은 3, 밀도 플롯은 2, 그리고 분할된(faceted) 산점도는 4입니다. patchwork는 이 척도를 사용하여 플롯에 할당한 영역을 나누고 그에 따라 구성 요소를 배치합니다.

```
p1 <- ggplot(mpg, aes(x = drv, y = cty, color = drv)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Plot 1")

p2 <- ggplot(mpg, aes(x = drv, y = hwy, color = drv)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "Plot 2")

p3 <- ggplot(mpg, aes(x = cty, color = drv, fill = drv)) +
  geom_density(alpha = 0.5) +
  labs(title = "Plot 3")

p4 <- ggplot(mpg, aes(x = hwy, color = drv, fill = drv)) +
  geom_density(alpha = 0.5) +
  labs(title = "Plot 4")

p5 <- ggplot(mpg, aes(x = cty, y = hwy, color = drv)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~drv) +
  labs(title = "Plot 5")

(guide_area() / (p1 + p2) / (p3 + p4) / p5) +
  plot_annotation(
    title = "City and highway mileage for cars with different drivetrains",
    caption = "Source: https://fueleconomy.gov."
  ) +
  plot_layout(
    guides = "collect",
    heights = c(1, 3, 2, 4)
    ) &
  theme(legend.position = "top")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in29.png" alt="Five plots laid out such that first two plots are next to each other. Plots three and four are underneath them. And the fifth plot stretches under them. The patchworked plot is titled &quot;City and highway mileage for cars with different drivetrains&quot; and captioned &quot;Source: https://fueleconomy.gov&quot;. The first two plots are side-by-side box plots. Plots 3 and 4 are density plots. And the fifth plot is a faceted scatterplot. Each of these plots show geoms colored by drivetrain, but the patchworked plot has only one legend that applies to all of them, above the plots and beneath the title." />
</figure>

patchwork를 사용하여 여러 플롯을 결합하고 배치하는 방법에 대해 더 자세히 알아보고 싶다면, [패키지 웹사이트](https://oreil.ly/xWxVV)의 가이드를 살펴볼 것을 권장합니다.

## 연습문제 (Exercises)

1. 다음 플롯 레이아웃에서 괄호를 생략하면 어떻게 되나요? 왜 그런 현상이 발생하는지 설명할 수 있나요?

   ```
   p1 <- ggplot(mpg, aes(x = displ, y = hwy)) +
     geom_point() +
     labs(title = "Plot 1")
   p2 <- ggplot(mpg, aes(x = drv, y = hwy)) +
     geom_boxplot() +
     labs(title = "Plot 2")
   p3 <- ggplot(mpg, aes(x = cty, y = hwy)) +
     geom_point() +
     labs(title = "Plot 3")

   (p1 | p2) / p3
   ```

이전 연습문제의 세 가지 플롯을 사용하여 다음 patchwork를 다시 만드세요.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in30.png" alt="Three plots: Plot 1 is a scatterplot of highway mileage versus engine size. Plot 2 is side-by-side box plots of highway mileage versus drivetrain. Plot 3 is side-by-side box plots of city mileage versus drivetrain. Plots 1 is on the first row. Plots 2 and 3 are on the next row, each span half the width of Plot 1. Plot 1 is labelled &quot;Fig. A&quot;, Plot 2 is labelled &quot;Fig. B&quot;, and Plot 3 is labelled &quot;Fig. C&quot;." />
</figure>

# 요약 (Summary)

이 장에서는 제목, 부제목, 캡션과 같은 플롯 레이블을 추가하는 것뿐만 아니라 기본 축 레이블을 수정하는 방법, 플롯에 정보 텍스트를 추가하거나 특정 데이터 포인트를 강조하기 위해 주석을 사용하는 방법, 축 척도를 사용자 지정하고 플롯의 테마를 변경하는 방법에 대해 배웠습니다. 또한 단순한 플롯 레이아웃과 복잡한 플롯 레이아웃 모두를 사용하여 여러 플롯을 단일 그래프로 결합하는 방법도 배웠습니다.

지금까지 여러 가지 다양한 유형의 플롯을 만드는 방법과 다양한 기술을 사용하여 이를 사용자 지정하는 방법에 대해 배웠지만, ggplot2로 만들 수 있는 것의 겉핥기만 했을 뿐입니다. ggplot2에 대한 포괄적인 이해를 원하신다면, [_ggplot2: Elegant Graphics for Data Analysis_](https://oreil.ly/T4Jxn) (Springer 출판) 책을 읽어보실 것을 권장합니다. 다른 유용한 리소스로는 Winston Chang의 [_R Graphics Cookbook_](https://oreil.ly/CK_sd) (O’Reilly 출판)과 Claus Wilke의 [_Fundamentals of Data Visualization_](https://oreil.ly/uJRYK) (O’Reilly 출판)가 있습니다.

<sup>[1](ch11.html#idm44771304642976-marker)</sup> 색맹을 시뮬레이션하여 이러한 이미지를 테스트하려면 [SimDaltonism](https://oreil.ly/i11yd)과 같은 도구를 사용할 수 있습니다.

<sup>[2](ch11.html#idm44771303447760-marker)</sup> 많은 사람들이 기본 테마의 배경이 회색인 이유를 궁금해합니다. 이는 눈금선(grid lines)을 여전히 보이게 하면서 데이터를 돋보이게 하기 위한 의도적인 선택이었습니다. 흰색 눈금선은 (위치 판단에 크게 도움이 되기 때문에 중요한데) 눈에 보이기는 하지만 시각적 영향은 거의 없어서 우리가 쉽게 무시할 수 있습니다. 회색 배경은 플롯에 텍스트와 유사한 타이포그래피 색상을 부여하여, 그래픽이 밝은 흰색 배경으로 튀지 않으면서 문서의 흐름과 잘 어우러지도록 합니다. 마지막으로, 회색 배경은 연속적인 색상 영역을 생성하여 플롯이 단일 시각적 개체로 인식되도록 합니다.
