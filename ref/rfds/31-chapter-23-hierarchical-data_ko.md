# 제23장. 계층적 데이터

# 소개

이 장에서는 본질적으로 계층적이거나 트리(tree) 모양인 데이터를 행과 열로 구성된 직사각형 데이터 프레임으로 변환하는 데이터 _직사각형화(rectangling)_ 기술을 배울 것입니다. 계층적 데이터는 특히 웹에서 가져온 데이터로 작업할 때 놀라울 정도로 흔하기 때문에 이는 중요합니다.

직사각형화에 대해 배우려면 먼저 계층적 데이터를 가능하게 하는 데이터 구조인 리스트(list)에 대해 배워야 합니다. 그런 다음 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>tidyr::unnest_longer()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>tidyr::unnest_wider()</code></a>라는 두 가지 중요한 tidyr 함수에 대해 배울 것입니다. 이어서 실제 문제를 해결하기 위해 이 간단한 함수들을 반복적으로 적용하는 몇 가지 사례 연구를 보여줄 것입니다. 마지막으로 계층적 데이터 세트의 빈번한 출처이자 웹에서 데이터 교환의 일반적인 형식인 JSON에 대해 이야기하며 마무리하겠습니다.

## 사전 준비

이 장에서는 tidyverse의 핵심 멤버인 tidyr의 많은 함수를 사용할 것입니다. 또한 직사각형화 연습을 위한 흥미로운 데이터 세트를 제공하기 위해 *repurrrsive*를 사용하고, JSON 파일을 R 리스트로 읽기 위해 *jsonlite*를 사용하는 것으로 마무리할 것입니다.

```
library(tidyverse)
library(repurrrsive)
library(jsonlite)
```

# 리스트

지금까지 여러분은 정수, 숫자, 문자, 날짜-시간, 팩터(factor)와 같은 단순한 벡터가 포함된 데이터 프레임으로 작업했습니다. 이러한 벡터는 동질적(homogeneous), 즉 모든 요소가 동일한 데이터 유형이기 때문에 단순합니다. 서로 다른 유형의 요소를 같은 벡터에 저장하려면 *리스트(list)*가 필요하며, 이는 <a href="https://rdrr.io/r/base/list.html" class="orm:hideurl"><code>list()</code></a>로 생성합니다.

```
x1 <- list(1:4, "a", TRUE)
x1
#> [[1]]
#> [1] 1 2 3 4
#>
#> [[2]]
#> [1] "a"
#>
#> [[3]]
#> [1] TRUE
```

티블의 열 이름을 지정하는 것과 같은 방식으로 리스트의 구성 요소, 즉 *자식(children)*에 이름을 지정하는 것이 편리한 경우가 많습니다.

```
x2 <- list(a = 1:2, b = 1:3, c = 1:4)
x2
#> $a
#> [1] 1 2
#>
#> $b
#> [1] 1 2 3
#>
#> $c
#> [1] 1 2 3 4
```

이러한 간단한 리스트의 경우에도 출력은 꽤 많은 공간을 차지합니다. 유용한 대안은 내용보다는 *구조(structure)*를 강조하여 간결하게 표시하는 <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>입니다.

```
str(x1)
#> List of 3
#>  $ : int [1:4] 1 2 3 4
#>  $ : chr "a"
#>  $ : logi TRUE

str(x2)
#> List of 3
#>  $ a: int [1:2] 1 2
#>  $ b: int [1:3] 1 2 3
#>  $ c: int [1:4] 1 2 3 4
```

보시다시피 <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>은 리스트의 각 자식을 각자의 줄에 표시합니다. 이름이 있으면 이름을 표시하고, 그다음 유형의 약어, 그리고 처음 몇 개의 값을 표시합니다.

## 계층 구조

리스트는 다른 리스트를 포함하여 어떤 유형의 객체든 포함할 수 있습니다. 이는 계층적(트리 모양) 구조를 나타내는 데 적합하게 만듭니다.

```
x3 <- list(list(1, 2), list(3, 4))
str(x3)
#> List of 2
#>  $ :List of 2
#>   ..$ : num 1
#>   ..$ : num 2
#>  $ :List of 2
#>   ..$ : num 3
#>   ..$ : num 4
```

이것은 평면적인 벡터를 생성하는 <a href="https://rdrr.io/r/base/c.html" class="orm:hideurl"><code>c()</code></a>와 현저히 다릅니다.

```
c(c(1, 2), c(3, 4))
#> [1] 1 2 3 4

x4 <- c(list(1, 2), list(3, 4))
str(x4)
#> List of 4
#>  $ : num 1
#>  $ : num 2
#>  $ : num 3
#>  $ : num 4
```

리스트가 더 복잡해질수록 계층 구조를 한눈에 볼 수 있게 해주는 <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>이 더 유용해집니다.

```
x5 <- list(1, list(2, list(3, list(4, list(5)))))
str(x5)
#> List of 2
#>  $ : num 1
#>  $ :List of 2
#>   ..$ : num 2
#>   ..$ :List of 2
#>   .. ..$ : num 3
#>   .. ..$ :List of 2
#>   .. .. ..$ : num 4
#>   .. .. ..$ :List of 1
#>   .. .. .. ..$ : num 5
```

리스트가 훨씬 더 커지고 복잡해지면 <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>은 결국 실패하기 시작할 것이며, <a href="https://rdrr.io/r/utils/View.html" class="orm:hideurl"><code>View()</code></a>로 전환해야 할 것입니다.<sup><a href="ch23.html#idm44771276868272" id="idm44771276868272-marker" data-type="noteref">1</a></sup> <a href="#fig-view-collapsed" data-type="xref">그림 23-1</a>은 `View(x5)`를 호출한 결과를 보여줍니다. 뷰어는 리스트의 최상위 레벨만 보여주는 것으로 시작하지만, <a href="#fig-view-expand-1" data-type="xref">그림 23-2</a>와 같이 구성 요소를 대화식으로 확장하여 더 많은 것을 볼 수 있습니다. 또한 <a href="#fig-view-expand-2" data-type="xref">그림 23-3</a>과 같이 RStudio는 해당 요소에 액세스하는 데 필요한 코드를 보여줍니다. 이 코드가 어떻게 작동하는지는 <a href="ch27.html#sec-subset-one" data-type="xref">“$와 [[를 사용하여 단일 요소 선택하기”</a>에서 다시 설명하겠습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2301.png" alt="A screenshot of RStudio showing the list-viewer. It shows the two children of x4: the first child is a double vector and the second child is a list. A rightward facing triable indicates that the second child itself has children but you can&#39;t see them. " />
<h6 id="figure-23-1.-the-rstudio-view-lets-you-interactively-explore-a-complex-list.-the-viewer-opens-showing-only-the-top-level-of-the-list.">그림 23-1. RStudio 뷰를 사용하면 복잡한 리스트를 대화식으로 탐색할 수 있습니다. 뷰어는 리스트의 최상위 레벨만 보여주며 열립니다.</h6>
</figure>

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2302.png" alt="Another screenshot of the list-viewer having expand the second child of x2. It also has two children, a double vector and another list. " />
<h6 id="figure-23-2.-clicking-the-right-facing-triangle-expands-that-component-of-the-list-so-that-you-can-also-see-its-children.">그림 23-2. 오른쪽을 가리키는 삼각형을 클릭하면 리스트의 해당 구성 요소가 확장되어 그 자식들도 볼 수 있습니다.</h6>
</figure>

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2303.png" alt="Another screenshot, having expanded the grandchild of x4 to see its two children, again a double vector and a list. " />
<h6 id="figure-23-3.-you-can-repeat-this-operation-as-many-times-as-needed-to-get-to-the-data-youre-interested-in.-note-the-bottom-left-corner-if-you-click-an-element-of-the-list-rstudio-will-give-you-the-subsetting-code-needed-to-access-it-in-this-case-x5222.">그림 23-3. 관심 있는 데이터에 도달할 때까지 이 작업을 필요한 만큼 반복할 수 있습니다. 왼쪽 하단 모서리에 주목하세요. 리스트의 요소를 클릭하면 RStudio가 접근에 필요한 부분집합(subsetting) 코드(이 경우 <code>x5[[2]][[2]][[2]]</code>)를 알려줍니다.</h6>
</figure>

## 리스트 열

리스트는 티블 안에도 존재할 수 있으며, 이 경우 우리는 이를 리스트 열(list column)이라고 부릅니다. 리스트 열은 일반적으로 티블에 속하지 않는 객체를 티블에 배치할 수 있게 해주기 때문에 유용합니다. 특히 모델 출력이나 재표본(resample) 같은 것을 데이터 프레임에 저장할 수 있게 해주기 때문에 [tidymodels 생태계](https://oreil.ly/0giAa)에서 리스트 열이 많이 사용됩니다.

리스트 열의 간단한 예는 다음과 같습니다.

```
df <- tibble(
  x = 1:2,
  y = c("a", "b"),
  z = list(list(1, 2), list(3, 4, 5))
)
df
#> # A tibble: 2 × 3
#>       x y     z
#>   <int> <chr> <list>
#> 1     1 a     <list [2]>
#> 2     2 b     <list [3]>
```

티블의 리스트에는 특별한 것이 없습니다. 다른 열과 똑같이 작동합니다.

```
df |> filter(x == 1)
#> # A tibble: 1 × 3
#>       x y     z
#>   <int> <chr> <list>
#> 1     1 a     <list [2]>
```

리스트 열로 계산하는 것은 더 어렵지만, 이는 리스트로 계산하는 것 자체가 일반적으로 더 어렵기 때문입니다. 이에 대해서는 <a href="ch26.html#chp-iteration" data-type="xref">26장</a>에서 다시 다루겠습니다. 이 장에서는 기존 도구를 사용할 수 있도록 리스트 열을 일반 변수로 언네스팅(unnesting, 중첩 해제)하는 데 중점을 둘 것입니다.

기본 출력(print) 메서드는 내용의 대략적인 요약만 표시합니다. 리스트 열은 임의로 복잡할 수 있으므로 이를 출력하는 좋은 방법이 없습니다. 이를 보려면 리스트 열 하나만 추출하여 `df |> pull(z) |> str()` 또는 `df |> pull(z) |> View()`와 같이 이전에 배운 기술 중 하나를 적용해야 합니다.

# Base R

`data.frame`의 열에 리스트를 넣을 수는 있지만 <a href="https://rdrr.io/r/base/data.frame.html" class="orm:hideurl"><code>data.frame()</code></a>이 리스트를 열들의 리스트로 취급하기 때문에 훨씬 더 까다롭습니다.

```
data.frame(x = list(1:3, 3:5))
#>   x.1.3 x.3.5
#> 1     1     3
#> 2     2     4
#> 3     3     5
```

리스트를 <a href="https://rdrr.io/r/base/AsIs.html" class="orm:hideurl"><code>I()</code></a>로 감싸면 <a href="https://rdrr.io/r/base/data.frame.html" class="orm:hideurl"><code>data.frame()</code></a>이 리스트를 행들의 리스트로 취급하도록 강제할 수 있지만 결과가 그리 잘 출력되지는 않습니다.

```
data.frame(
  x = I(list(1:2, 3:5)),
  y = c("1, 2", "3, 4, 5")
)
#>         x       y
#> 1    1, 2    1, 2
#> 2 3, 4, 5 3, 4, 5
```

<a href="https://tibble.tidyverse.org/reference/tibble.html" class="orm:hideurl"><code>tibble()</code></a>은 리스트를 벡터처럼 취급하고 출력 메서드가 리스트를 염두에 두고 설계되었기 때문에 티블과 함께 리스트 열을 사용하는 것이 더 쉽습니다.

# 언네스팅

이제 리스트와 리스트 열의 기본을 배웠으니 이들을 다시 일반 행과 열로 되돌리는 방법을 알아보겠습니다. 여기서는 기본 개념을 익히기 위해 간단한 샘플 데이터를 사용하고, 다음 섹션에서 실제 데이터로 전환하겠습니다.

리스트 열은 일반적으로 이름이 있는(named) 형태와 이름이 없는(unnamed) 형태의 두 가지 기본 형태를 갖습니다. 자식들에게 *이름이 있을 때*는 모든 행에서 동일한 이름을 갖는 경향이 있습니다. 예를 들어 `df1`에서 리스트 열 `y`의 모든 요소는 `a`와 `b`라는 두 개의 요소를 갖습니다. 이름이 있는 리스트 열은 자연스럽게 열로 언네스팅됩니다. 즉, 이름이 있는 각 요소가 이름이 있는 새로운 열이 됩니다.

```
df1 <- tribble(
  ~x, ~y,
   1, list(a = 11, b = 12),
   2, list(a = 21, b = 22),
   3, list(a = 31, b = 32),
)
```

자식들에게 *이름이 없을 때*는 요소의 수가 행마다 다른 경향이 있습니다. 예를 들어 `df2`에서 리스트 열 `y`의 요소들은 이름이 없고 길이가 1에서 3까지 다양합니다. 이름이 없는 리스트 열은 자연스럽게 행으로 언네스팅됩니다. 즉, 각 자식당 하나의 행을 얻게 됩니다.

```
df2 <- tribble(
  ~x, ~y,
   1, list(11, 12, 13),
   2, list(21),
   3, list(31, 32),
)
```

tidyr는 이 두 가지 경우를 위해 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>라는 두 가지 함수를 제공합니다. 다음 섹션에서는 이들이 어떻게 작동하는지 설명합니다.

## unnest_wider()

`df1`과 같이 각 행이 같은 이름의 같은 개수의 요소를 가질 때, <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>를 사용하여 각 구성 요소를 자체 열에 넣는 것이 자연스럽습니다.

```
df1 |> unnest_wider(y)
#> # A tibble: 3 × 3
#>       x     a     b
#>   <dbl> <dbl> <dbl>
#> 1     1    11    12
#> 2     2    21    22
#> 3     3    31    32
```

기본적으로 새 열의 이름은 오로지 리스트 요소의 이름에서만 가져오지만, `names_sep` 인자를 사용하여 열 이름과 요소 이름을 결합하도록 요청할 수 있습니다. 이는 중복되는 이름을 명확히 하는 데 유용합니다.

```
df1 |> unnest_wider(y, names_sep = "_")
#> # A tibble: 3 × 3
#>       x   y_a   y_b
#>   <dbl> <dbl> <dbl>
#> 1     1    11    12
#> 2     2    21    22
#> 3     3    31    32
```

## unnest_longer()

각 행에 이름이 없는 리스트가 포함되어 있을 때 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>를 사용하여 각 요소를 자체 행에 넣는 것이 자연스럽습니다.

```
df2 |> unnest_longer(y)
#> # A tibble: 6 × 2
#>       x     y
#>   <dbl> <dbl>
#> 1     1    11
#> 2     1    12
#> 3     1    13
#> 4     2    21
#> 5     3    31
#> 6     3    32
```

`y` 내부의 각 요소에 대해 `x`가 어떻게 복제되는지 주목하세요. 즉, 리스트 열 내부의 각 요소에 대해 하나의 출력 행을 얻습니다. 하지만 다음 예제처럼 요소 중 하나가 비어 있으면 어떻게 될까요?

```
df6 <- tribble(
  ~x, ~y,
  "a", list(1, 2),
  "b", list(3),
  "c", list()
)
df6 |> unnest_longer(y)
#> # A tibble: 3 × 2
#>   x         y
#>   <chr> <dbl>
#> 1 a         1
#> 2 a         2
#> 3 b         3
```

출력에 0개의 행이 표시되므로 해당 행은 사실상 사라집니다. 해당 행을 보존하려면 `y`에 `NA`를 추가하고 `keep_empty = TRUE`를 설정하세요.

## 일관되지 않은 유형

다양한 유형의 벡터를 포함하는 리스트 열을 언네스팅하면 어떻게 될까요? 예를 들어 리스트 열 `y`에 일반적으로 단일 열에 섞일 수 없는 숫자 2개, 문자 1개, 논리값 1개가 포함된 다음 데이터 세트를 살펴보세요.

```
df4 <- tribble(
  ~x, ~y,
  "a", list(1),
  "b", list("a", TRUE, 5)
)
```

<a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>는 항상 열의 집합을 변경하지 않고 행의 수만 변경합니다. 그럼 어떻게 될까요? <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>는 어떻게 `y` 안의 모든 것을 유지하면서 4개의 행을 생성할까요?

```
df4 |> unnest_longer(y)
#> # A tibble: 4 × 2
#>   x     y
#>   <chr> <list>
#> 1 a     <dbl [1]>
#> 2 b     <chr [1]>
#> 3 b     <lgl [1]>
#> 4 b     <dbl [1]>
```

보시다시피 출력에는 리스트 열이 포함되어 있지만, 리스트 열의 각 요소에는 단일 요소만 포함되어 있습니다. <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>가 벡터의 공통 유형을 찾을 수 없기 때문에 원래 유형을 리스트 열에 유지합니다. 열의 모든 요소는 같은 유형이어야 한다는 계명을 위반하는 것이 아닌지 궁금할 수 있습니다. 그렇지 않습니다. 내용물이 서로 다른 유형이더라도 모든 요소는 리스트입니다.

일관되지 않은 유형을 다루는 것은 까다로우며 세부 사항은 문제의 정확한 특성과 목표에 따라 다르지만, 가능성 높은 것은 <a href="ch26.html#chp-iteration" data-type="xref">26장</a>의 도구가 필요할 것이라는 점입니다.

## 다른 함수들

tidyr에는 이 책에서 다루지 않을 몇 가지 유용한 직사각형화 함수가 더 있습니다.

- <a href="https://tidyr.tidyverse.org/reference/unnest_auto.html" class="orm:hideurl"><code>unnest_auto()</code></a>는 리스트 열의 구조를 기반으로 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a> 중 하나를 자동으로 선택합니다. 빠른 탐색에는 좋지만 데이터가 어떻게 구조화되어 있는지 이해하도록 강제하지 않고 코드를 이해하기 어렵게 만들 수 있으므로 궁극적으로는 좋은 생각이 아닙니다.
- <a href="https://tidyr.tidyverse.org/reference/unnest.html" class="orm:hideurl"><code>unnest()</code></a>는 행과 열을 모두 확장합니다. 데이터 프레임과 같은 2D 구조를 포함하는 리스트 열이 있을 때 유용합니다. 이 책에서는 볼 수 없지만 [tidymodels 생태계](https://oreil.ly/ytJvP)를 사용하면 접할 수 있습니다.

이러한 함수들은 다른 사람의 코드를 읽거나 더 드문 직사각형화 문제를 스스로 해결할 때 만날 수 있으므로 알아두는 것이 좋습니다.

## 연습문제

1.  `df2`와 같은 이름 없는 리스트 열에 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>를 사용하면 어떻게 되나요? 이제 어떤 인자가 필요한가요? 결측값(missing value)은 어떻게 되나요?

2.  `df1`과 같은 이름 있는 리스트 열에 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>를 사용하면 어떻게 되나요? 출력에서 어떤 추가 정보를 얻나요? 그 추가 세부 정보를 어떻게 억제할 수 있나요?

3.  때때로 정렬된(aligned) 값을 가진 여러 리스트 열이 있는 데이터 프레임을 만나게 됩니다. 예를 들어 다음 데이터 프레임에서 `y`와 `z`의 값은 정렬되어 있습니다(즉, `y`와 `z`는 행 내에서 항상 같은 길이를 가지며 `y`의 첫 번째 값은 `z`의 첫 번째 값에 대응합니다). 이 데이터 프레임에 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a> 호출을 두 번 적용하면 어떻게 되나요? `x`와 `y` 사이의 관계를 어떻게 보존할 수 있나요? (힌트: 문서를 주의 깊게 읽어보세요.)

    ```
    df4 <- tribble(
      ~x, ~y, ~z,
      "a", list("y-a-1", "y-a-2"), list("z-a-1", "z-a-2"),
      "b", list("y-b-1", "y-b-2", "y-b-3"), list("z-b-1", "z-b-2", "z-b-3")
    )
    ```

# 사례 연구

앞서 사용한 간단한 예제와 실제 데이터의 주요 차이점은 실제 데이터는 일반적으로 여러 번의 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a> 및/또는 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a> 호출이 필요한 여러 수준의 중첩을 포함한다는 것입니다. 그 과정을 보여주기 위해 이 섹션에서는 repurrrsive 패키지의 데이터 세트를 사용하여 3가지 실제 직사각형화 문제를 살펴봅니다.

## 매우 넓은 데이터

`gh_repos`부터 시작하겠습니다. 이것은 GitHub API를 사용하여 검색한 GitHub 리포지토리(repository) 모음에 대한 데이터를 포함하는 리스트입니다. 이 리스트는 매우 깊게 중첩되어 있어서 이 책에서 구조를 보여주기가 어렵습니다. 계속하기 전에 `View(gh_repos)`를 사용하여 여러분 스스로 조금 탐색해 볼 것을 권장합니다.

`gh_repos`는 리스트이지만 우리의 도구는 리스트 열로 작동하므로 먼저 티블에 넣는 것으로 시작하겠습니다. 나중에 설명할 이유 때문에 이 열을 `json`이라고 부르겠습니다.

```
repos <- tibble(json = gh_repos)
repos
#> # A tibble: 6 × 1
#>   json
#>   <list>
#> 1 <list [30]>
#> 2 <list [30]>
#> 3 <list [30]>
#> 4 <list [26]>
#> 5 <list [30]>
#> 6 <list [30]>
```

이 티블에는 `gh_repos`의 각 자식마다 하나씩, 총 6개의 행이 포함되어 있습니다. 각 행에는 26개 또는 30개의 행이 있는 이름 없는 리스트가 포함되어 있습니다. 이름이 없기 때문에 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>로 시작하여 각 자식을 자체 행에 넣겠습니다.

```
repos |> unnest_longer(json)
#> # A tibble: 176 × 1
#>   json
#>   <list>
#> 1 <named list [68]>
#> 2 <named list [68]>
#> 3 <named list [68]>
#> 4 <named list [68]>
#> 5 <named list [68]>
#> 6 <named list [68]>
#> # … with 170 more rows
```

언뜻 보면 상황이 나아지지 않은 것처럼 보일 수 있습니다. 행은 6개에서 176개로 늘어났지만 `json`의 각 요소는 여전히 리스트입니다. 하지만 중요한 차이점이 있습니다. 이제 각 요소가 _이름이 있는(named)_ 리스트이므로 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>를 사용하여 각 요소를 자체 열에 넣을 수 있습니다.

```
repos |>
  unnest_longer(json) |>
  unnest_wider(json)
#> # A tibble: 176 × 68
#>       id name        full_name owner private html_url
#>    <int> <chr>       <chr>     <list>  <lgl> <chr>
#> 1 6.12e7 after       gaborcsa… <name…  FALSE https://github.co…
#> 2 4.05e7 argufy      gaborcsa… <name…  FALSE https://github.co…
#> 3 3.64e7 ask         gaborcsa… <name…  FALSE https://github.co…
#> 4 3.49e7 baseimports gaborcsa… <name…  FALSE https://github.co…
#> 5 6.16e7 citest      gaborcsa… <name…  FALSE https://github.co…
#> 6 3.39e7 clisymbols  gaborcsa… <name…  FALSE https://github.co…
#> # … with 170 more rows, and 62 more variables: description <chr>,
#> #   fork <lgl>, url <chr>, forks_url <chr>, keys_url <chr>, …
```

효과가 있었지만 결과가 약간 압도적입니다. 열이 너무 많아서 티블이 모두 출력하지도 못합니다! <a href="https://rdrr.io/r/base/names.html" class="orm:hideurl"><code>names()</code></a>를 사용하여 모두 볼 수 있으며, 여기서는 처음 10개를 살펴봅니다.

```
repos |>
  unnest_longer(json) |>
  unnest_wider(json) |>
  names() |>
  head(10)
#>  [1] "id"          "name"        "full_name"   "owner"       "private"
#>  [6] "html_url"    "description" "fork"        "url"         "forks_url"
```

흥미로워 보이는 몇 가지를 추출해 보겠습니다.

```
repos |>
  unnest_longer(json) |>
  unnest_wider(json) |>
  select(id, full_name, owner, description)
#> # A tibble: 176 × 4
#>         id full_name               owner             description
#>      <int> <chr>                   <list>            <chr>
#> 1 61160198 gaborcsardi/after       <named list [17]> Run Code in the Backgro…
#> 2 40500181 gaborcsardi/argufy      <named list [17]> Declarative function ar…
#> 3 36442442 gaborcsardi/ask         <named list [17]> Friendly CLI interactio…
#> 4 34924886 gaborcsardi/baseimports <named list [17]> Do we get warnings for …
#> 5 61620661 gaborcsardi/citest      <named list [17]> Test R package and repo…
#> 6 33907457 gaborcsardi/clisymbols  <named list [17]> Unicode symbols for CLI…
#> # … with 170 more rows
```

이를 사용하여 거꾸로 작업하여 `gh_repos`가 어떻게 구성되었는지 이해할 수 있습니다. 각 자식은 그들이 생성한 최대 30개의 GitHub 리포지토리 리스트를 포함하는 GitHub 사용자였습니다.

`owner`는 또 다른 리스트 열이며 이름이 있는 리스트를 포함하므로 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>를 사용하여 값을 얻을 수 있습니다.

```
repos |>
  unnest_longer(json) |>
  unnest_wider(json) |>
  select(id, full_name, owner, description) |>
  unnest_wider(owner)
#> Error in `unnest_wider()`:
#> ! Can't duplicate names between the affected columns and the original
#>   data.
#> ✖ These names are duplicated:
#>   ℹ `id`, from `owner`.
#>   ℹ Use `names_sep` to disambiguate using the column name.
#>   ℹ Or use `names_repair` to specify a repair strategy.
```

이런, 이 리스트 열에도 `id` 열이 포함되어 있는데 동일한 데이터 프레임에 두 개의 `id` 열을 가질 수는 없습니다. 제안된 대로 `names_sep`을 사용하여 문제를 해결해 보겠습니다.

```
repos |>
  unnest_longer(json) |>
  unnest_wider(json) |>
  select(id, full_name, owner, description) |>
  unnest_wider(owner, names_sep = "_")
#> # A tibble: 176 × 20
#>         id full_name               owner_login owner_id owner_avatar_url
#>      <int> <chr>                   <chr>          <int> <chr>
#> 1 61160198 gaborcsardi/after       gaborcsardi   660288 https://avatars.g…
#> 2 40500181 gaborcsardi/argufy      gaborcsardi   660288 https://avatars.g…
#> 3 36442442 gaborcsardi/ask         gaborcsardi   660288 https://avatars.g…
#> 4 34924886 gaborcsardi/baseimports gaborcsardi   660288 https://avatars.g…
#> 5 61620661 gaborcsardi/citest      gaborcsardi   660288 https://avatars.g…
#> 6 33907457 gaborcsardi/clisymbols  gaborcsardi   660288 https://avatars.g…
#> # … with 170 more rows, and 15 more variables: owner_gravatar_id <chr>,
#> #   owner_url <chr>, owner_html_url <chr>, owner_followers_url <chr>, …
```

이것은 또 다른 넓은 데이터 세트를 제공하지만, `owner`에 리포지토리를 "소유"한 사람에 대한 추가 데이터가 많이 포함되어 있는 것 같다는 것을 알 수 있습니다.

## 관계형 데이터

중첩된 데이터는 대개 여러 데이터 프레임에 분산시킬 데이터를 나타내는 데 때때로 사용됩니다. 예를 들어 _왕좌의 게임(Game of Thrones)_ 책과 TV 시리즈에 등장하는 등장인물에 대한 데이터가 포함된 `got_chars`를 살펴보겠습니다. `gh_repos`와 마찬가지로 리스트이므로 티블의 리스트 열로 바꾸는 것부터 시작합니다.

```
chars <- tibble(json = got_chars)
chars
#> # A tibble: 30 × 1
#>   json
#>   <list>
#> 1 <named list [18]>
#> 2 <named list [18]>
#> 3 <named list [18]>
#> 4 <named list [18]>
#> 5 <named list [18]>
#> 6 <named list [18]>
#> # … with 24 more rows
```

`json` 열에는 이름이 있는 요소가 포함되어 있으므로 이를 넓히는(widening) 것으로 시작하겠습니다.

```
chars |> unnest_wider(json)
#> # A tibble: 30 × 18
#>   url                  id name  gender culture born
#>   <chr>             <int> <chr> <chr>  <chr>   <chr>
#> 1 https://www.anap…  1022 Theon… Male   "Ironb… "In 2…
#> 2 https://www.anap…  1052 Tyrio… Male   ""      "In 2…
#> 3 https://www.anap…  1074 Victa… Male   "Ironb… "In 2…
#> 4 https://www.anap…  1109 Will  Male   ""      ""
#> 5 https://www.anap…  1166 Areo … Male   "Norvo… "In 2…
#> 6 https://www.anap…  1267 Chett Male   ""      "At H…
#> # … with 24 more rows, and 12 more variables: died <chr>, alive <lgl>,
#> #   titles <list>, aliases <list>, father <chr>, mother <chr>, …
```

그런 다음 읽기 쉽도록 몇 개의 열을 선택합니다.

```
characters <- chars |>
  unnest_wider(json) |>
  select(id, name, gender, culture, born, died, alive)
characters
#> # A tibble: 30 × 7
#>      id name              gender culture    born                    died
#>   <int> <chr>             <chr>  <chr>      <chr>                   <chr>
#> 1  1022 Theon Greyjoy     Male   "Ironborn" "In 278 AC or 279 AC, … ""
#> 2  1052 Tyrion Lannister  Male   ""         "In 273 AC, at Casterl… ""
#> 3  1074 Victarion Greyjoy Male   "Ironborn" "In 268 AC or before, … ""
#> 4  1109 Will              Male   ""         ""                      "In 2…
#> 5  1166 Areo Hotah        Male   "Norvoshi" "In 257 AC or before, … ""
#> 6  1267 Chett             Male   ""         "At Hag's Mire"         "In 2…
#> # … with 24 more rows, and 1 more variable: alive <lgl>
```

이 데이터 세트에는 많은 리스트 열도 포함되어 있습니다.

```
chars |>
  unnest_wider(json) |>
  select(id, where(is.list))
#> # A tibble: 30 × 8
#>      id titles    aliases    allegiances books     povBooks  tvSeries  playedBy
#>   <int> <list>    <list>     <list>      <list>    <list>    <list>    <list>
#> 1  1022 <chr [2]> <chr [4]>  <chr [1]>   <chr [3]> <chr [2]> <chr [6]> <chr [1]>
#> 2  1052 <chr [2]> <chr [11]> <chr [1]>   <chr [2]> <chr [4]> <chr [6]> <chr [1]>
#> 3  1074 <chr [2]> <chr [1]>  <chr [1]>   <chr [3]> <chr [2]> <chr [1]> <chr [1]>
#> 4  1109 <chr [1]> <chr [1]>  <NULL>      <chr [1]> <chr [1]> <chr [1]> <chr [1]>
#> 5  1166 <chr [1]> <chr [1]>  <chr [1]>   <chr [3]> <chr [2]> <chr [2]> <chr [1]>
#> 6  1267 <chr [1]> <chr [1]>  <NULL>      <chr [2]> <chr [1]> <chr [1]> <chr [1]>
#> # … with 24 more rows
```

`titles` 열을 탐색해 보겠습니다. 이름이 없는 리스트 열이므로 행으로 언네스팅하겠습니다.

```
chars |>
  unnest_wider(json) |>
  select(id, titles) |>
  unnest_longer(titles)
#> # A tibble: 59 × 2
#>      id titles
#>   <int> <chr>
#> 1  1022 Prince of Winterfell
#> 2  1022 Lord of the Iron Islands (by law of the green lands)
#> 3  1052 Acting Hand of the King (former)
#> 4  1052 Master of Coin (former)
#> 5  1074 Lord Captain of the Iron Fleet
#> 6  1074 Master of the Iron Victory
#> # … with 53 more rows
```

이 데이터를 필요에 따라 등장인물 데이터에 쉽게 조인(join)할 수 있도록 별도의 테이블에서 보고 싶을 수 있습니다. 그렇게 해 보겠습니다. 빈 문자열이 있는 행을 제거하고 이제 각 행에 하나의 칭호만 포함되므로 `titles`의 이름을 `title`로 바꾸는 약간의 정리가 필요합니다.

```
titles <- chars |>
  unnest_wider(json) |>
  select(id, titles) |>
  unnest_longer(titles) |>
  filter(titles != "") |>
  rename(title = titles)
titles
#> # A tibble: 52 × 2
#>      id title
#>   <int> <chr>
#> 1  1022 Prince of Winterfell
#> 2  1022 Lord of the Iron Islands (by law of the green lands)
#> 3  1052 Acting Hand of the King (former)
#> 4  1052 Master of Coin (former)
#> 5  1074 Lord Captain of the Iron Fleet
#> 6  1074 Master of the Iron Victory
#> # … with 46 more rows
```

각 리스트 열마다 이와 같은 테이블을 생성한 다음 필요에 따라 조인을 사용하여 등장인물 데이터와 결합하는 것을 상상할 수 있습니다.

## 깊게 중첩된

매우 깊게 중첩되어 풀기 위해 여러 번의 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>를 반복해야 하는 리스트 열인 `gmaps_cities`로 이 사례 연구를 마무리하겠습니다. 이것은 5개의 도시 이름과 위치를 확인하기 위해 구글의 [지오코딩 API](https://oreil.ly/cdBWZ)를 사용한 결과를 포함하는 2열 티블입니다.

```
gmaps_cities
#> # A tibble: 5 × 2
#>   city       json
#>   <chr>      <list>
#> 1 Houston    <named list [2]>
#> 2 Washington <named list [2]>
#> 3 New York   <named list [2]>
#> 4 Chicago    <named list [2]>
#> 5 Arlington  <named list [2]>
```

`json`은 내부 이름이 있는 리스트 열이므로 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>로 시작합니다.

```
gmaps_cities |> unnest_wider(json)
#> # A tibble: 5 × 3
#>   city       results    status
#>   <chr>      <list>     <chr>
#> 1 Houston    <list [1]> OK
#> 2 Washington <list [2]> OK
#> 3 New York   <list [1]> OK
#> 4 Chicago    <list [1]> OK
#> 5 Arlington  <list [2]> OK
```

이것은 `status`와 `results`를 제공합니다. `status` 열은 모두 `OK`이므로 삭제하겠습니다. 실제 분석에서는 `status != "OK"`인 모든 행을 캡처하여 무엇이 잘못되었는지 파악하고 싶을 것입니다. `results`는 하나 또는 두 개의 요소(이유는 곧 알게 될 것입니다)가 있는 이름 없는 리스트이므로 행으로 언네스팅하겠습니다.

```
gmaps_cities |>
  unnest_wider(json) |>
  select(-status) |>
  unnest_longer(results)
#> # A tibble: 7 × 2
#>   city       results
#>   <chr>      <list>
#> 1 Houston    <named list [5]>
#> 2 Washington <named list [5]>
#> 3 Washington <named list [5]>
#> 4 New York   <named list [5]>
#> 5 Chicago    <named list [5]>
#> 6 Arlington  <named list [5]>
#> # … with 1 more row
```

이제 `results`는 이름이 있는 리스트이므로 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>를 사용하겠습니다.

```
locations <- gmaps_cities |>
  unnest_wider(json) |>
  select(-status) |>
  unnest_longer(results) |>
  unnest_wider(results)
locations
#> # A tibble: 7 × 6
#>   city       address_compone…¹ formatted_address geometry place_id
#>   <chr>      <list>            <chr>             <list>   <chr>
#> 1 Houston    <list [4]>        Houston, TX, USA  <named … ChIJAYWN…
#> 2 Washington <list [2]>        Washington, USA   <named … ChIJ-bDD…
#> 3 Washington <list [4]>        Washington, DC, … <named … ChIJW-T2…
#> 4 New York   <list [3]>        New York, NY, USA <named … ChIJOwg_…
#> 5 Chicago    <list [4]>        Chicago, IL, USA  <named … ChIJ7cv0…
#> 6 Arlington  <list [4]>        Arlington, TX, U… <named … ChIJ05gI…
#> # … with 1 more row, 1 more variable: types <list>, and abbreviated variable
#> #   name ¹​address_components
```

이제 왜 두 도시가 두 개의 결과를 얻었는지 알 수 있습니다. 워싱턴은 워싱턴 주(Washington state)와 워싱턴 DC(Washington, DC) 모두 일치했고, 알링턴은 버지니아주 알링턴(Arlington, Virginia)과 텍사스주 알링턴(Arlington, Texas) 모두 일치했습니다.

여기서 나아갈 수 있는 몇 가지 다른 방향이 있습니다. `geometry` 리스트 열에 저장된 일치 항목의 정확한 위치를 판별하고 싶을 수 있습니다.

```
locations |>
  select(city, formatted_address, geometry) |>
  unnest_wider(geometry)
#> # A tibble: 7 × 6
#>   city       formatted_address bounds     location   location_type
#>   <chr>      <chr>             <list>     <list>     <chr>
#> 1 Houston    Houston, TX, USA  <named li… <named li… APPROXIMATE
#> 2 Washington Washington, USA   <named li… <named li… APPROXIMATE
#> 3 Washington Washington, DC, USA <named li… <named li… APPROXIMATE
#> 4 New York   New York, NY, USA <named li… <named li… APPROXIMATE
#> 5 Chicago    Chicago, IL, USA  <named li… <named li… APPROXIMATE
#> 6 Arlington  Arlington, TX, USA <named li… <named li… APPROXIMATE
#> # … with 1 more row, and 1 more variable: viewport <list>
```

그것은 우리에게 새로운 `bounds`(직사각형 영역)와 `location`(점)을 제공합니다. `location`을 언네스팅하여 위도(`lat`)와 경도(`lng`)를 볼 수 있습니다.

```
locations |>
  select(city, formatted_address, geometry) |>
  unnest_wider(geometry) |>
  unnest_wider(location)
#> # A tibble: 7 × 7
#>   city       formatted_address bounds     lat   lng location_type
#>   <chr>      <chr>             <list>   <dbl> <dbl> <chr>
#> 1 Houston    Houston, TX, USA  <named …  29.8 -95.4 APPROXIMATE
#> 2 Washington Washington, USA   <named …  47.8 -121. APPROXIMATE
#> 3 Washington Washington, DC, USA <named …  38.9 -77.0 APPROXIMATE
#> 4 New York   New York, NY, USA <named …  40.7 -74.0 APPROXIMATE
#> 5 Chicago    Chicago, IL, USA  <named …  41.9 -87.6 APPROXIMATE
#> 6 Arlington  Arlington, TX, USA <named …  32.7 -97.1 APPROXIMATE
#> # … with 1 more row, and 1 more variable: viewport <list>
```

`bounds`를 추출하려면 몇 가지 단계가 더 필요합니다.

```
locations |>
  select(city, formatted_address, geometry) |>
  unnest_wider(geometry) |>
  # focus on the variables of interest
  select(!location:viewport) |>
  unnest_wider(bounds)
#> # A tibble: 7 × 4
#>   city       formatted_address northeast        southwest
#>   <chr>      <chr>             <list>           <list>
#> 1 Houston    Houston, TX, USA  <named list [2]> <named list [2]>
#> 2 Washington Washington, USA   <named list [2]> <named list [2]>
#> 3 Washington Washington, DC, USA <named list [2]> <named list [2]>
#> 4 New York   New York, NY, USA <named list [2]> <named list [2]>
#> 5 Chicago    Chicago, IL, USA  <named list [2]> <named list [2]>
#> 6 Arlington  Arlington, TX, USA <named list [2]> <named list [2]>
#> # … with 1 more row
```

그런 다음 `names_sep`을 사용하여 짧지만 기억하기 쉬운 이름을 만들 수 있도록 `southwest`와 `northeast`(직사각형의 모서리)의 이름을 바꿉니다.

```
locations |>
  select(city, formatted_address, geometry) |>
  unnest_wider(geometry) |>
  select(!location:viewport) |>
  unnest_wider(bounds) |>
  rename(ne = northeast, sw = southwest) |>
  unnest_wider(c(ne, sw), names_sep = "_")
#> # A tibble: 7 × 6
#>   city       formatted_address ne_lat ne_lng sw_lat sw_lng
#>   <chr>      <chr>              <dbl>  <dbl>  <dbl>  <dbl>
#> 1 Houston    Houston, TX, USA    30.1  -95.0   29.5  -95.8
#> 2 Washington Washington, USA     49.0 -117.    45.5 -125.
#> 3 Washington Washington, DC, USA   39.0  -76.9   38.8  -77.1
#> 4 New York   New York, NY, USA   40.9  -73.7   40.5  -74.3
#> 5 Chicago    Chicago, IL, USA    42.0  -87.5   41.6  -87.9
#> 6 Arlington  Arlington, TX, USA   32.8  -97.0   32.6  -97.2
#> # … with 1 more row
```

<a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>에 변수 이름의 벡터를 제공하여 두 열을 동시에 언네스팅하는 방식에 주목하세요.

관심 있는 구성 요소에 도달하는 경로를 발견하면 또 다른 tidyr 함수인 <a href="https://tidyr.tidyverse.org/reference/hoist.html" class="orm:hideurl"><code>hoist()</code></a>를 사용하여 이를 직접 추출할 수 있습니다.

```
locations |>
  select(city, formatted_address, geometry) |>
  hoist(
    geometry,
    ne_lat = c("bounds", "northeast", "lat"),
    sw_lat = c("bounds", "southwest", "lat"),
    ne_lng = c("bounds", "northeast", "lng"),
    sw_lng = c("bounds", "southwest", "lng"),
  )
```

이러한 사례 연구로 인해 실제 직사각형화에 대한 흥미가 생겼다면 `vignette("rectangling", package = "tidyr")`에서 몇 가지 예제를 더 볼 수 있습니다.

## 연습문제

1.  `gh_repos`가 언제 생성되었는지 대략적으로 추정해 보세요. 왜 대략적으로만 날짜를 추정할 수 있나요?

2.  각 소유자는 여러 리포지토리를 가질 수 있으므로 `gh_repos`의 `owner` 열에는 많은 중복 정보가 포함되어 있습니다. 각 소유자당 하나의 행이 포함된 `owners` 데이터 프레임을 만들 수 있나요? (힌트: <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>가 `list-cols`와 작동하나요?)

3.  `titles`에 사용된 단계를 따라 _왕좌의 게임_ 등장인물의 별칭(aliases), 충성 대상(allegiances), 책(books), TV 시리즈(TV series)에 대해 유사한 테이블을 만드세요.

4.  다음 코드를 한 줄씩 설명하세요. 이 코드가 흥미로운 이유는 무엇인가요? `got_chars`에는 작동하지만 일반적으로는 작동하지 않을 수 있는 이유는 무엇인가요?

    ```
    tibble(json = got_chars) |>
      unnest_wider(json) |>
      select(id, where(is.list)) |>
      pivot_longer(
        where(is.list),
        names_to = "name",
        values_to = "value"
      ) |>
      unnest_longer(value)
    ```

5.  `gmaps_cities`에서 `address_components`에는 무엇이 포함되어 있나요? 행마다 길이가 다른 이유는 무엇인가요? 적절하게 언네스팅하여 이를 파악해 보세요. (힌트: `types`는 항상 두 개의 요소를 포함하는 것처럼 보입니다. <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>보다 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>를 사용하는 것이 작업하기 더 쉽나요?)

# JSON

이전 섹션의 모든 사례 연구는 실제 환경에서 수집한 JSON을 출처로 합니다. JSON은 JavaScript Object Notation의 약자로 대부분의 웹 API가 데이터를 반환하는 방식입니다. JSON과 R의 데이터 유형은 꽤 비슷하지만 완벽한 일대일 매핑이 없기 때문에 문제가 발생할 때를 대비해 JSON에 대해 조금 알아두는 것이 좋으므로 이를 이해하는 것은 중요합니다.

## 데이터 유형

JSON은 사람이 아닌 기계가 쉽게 읽고 쓸 수 있도록 설계된 간단한 형식입니다. 6가지 핵심 데이터 유형이 있습니다. 그중 4가지는 스칼라(scalar)입니다.

- 단순한 유형은 null(`null`)로, R의 `NA`와 같은 역할을 합니다. 데이터가 없음을 나타냅니다.
- 문자열(string)은 R의 문자열과 매우 유사하지만 항상 큰따옴표를 사용해야 합니다.
- 숫자(number)는 R의 숫자와 비슷합니다. 정수(123), 소수(123.45), 과학적 표기법(1.23e3)을 사용할 수 있습니다. JSON은 `Inf`, `-Inf`, `NaN`을 지원하지 않습니다.
- 불리언(boolean)은 R의 `TRUE`, `FALSE`와 비슷하지만 소문자 `true`와 `false`를 사용합니다.

JSON의 문자열, 숫자, 불리언은 R의 문자(character), 숫자(numeric), 논리(logical) 벡터와 꽤 비슷합니다. 주요 차이점은 JSON의 스칼라가 단일 값만 나타낼 수 있다는 점입니다. 여러 값을 나타내려면 나머지 두 가지 유형 중 하나인 배열(array)이나 객체(object)를 사용해야 합니다.

배열과 객체 모두 R의 리스트와 유사하며, 차이점은 이름이 지정되어 있는지 여부입니다. *배열(array)*은 이름이 없는 리스트와 같으며 `[]`로 작성합니다. 예를 들어 `[1, 2, 3]`은 세 개의 숫자를 포함하는 배열이고, `[null, 1, "string", false]`는 null, 숫자, 문자열, 불리언을 포함하는 배열입니다. *객체(object)*는 이름이 있는 리스트와 같으며 <a href="https://rdrr.io/r/base/Paren.html" class="orm:hideurl"><code>{}</code></a>로 작성합니다. 이름(JSON 용어로는 키(key))은 문자열이므로 따옴표로 묶어야 합니다. 예를 들어 `{"x": 1, "y": 2}`는 `x`를 1에, `y`를 2에 매핑하는 객체입니다.

JSON은 날짜나 날짜-시간을 나타내는 기본 방법이 없기 때문에 문자열로 저장되는 경우가 많으며, 이를 올바른 데이터 구조로 바꾸려면 <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>readr::parse_date()</code></a> 또는 <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>readr::parse_datetime()</code></a>을 사용해야 합니다. 마찬가지로 부동 소수점(floating-point) 숫자를 나타내는 JSON의 규칙은 약간 부정확해서 때로는 숫자가 문자열에 저장된 것을 발견할 수 있습니다. 필요에 따라 <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>readr::parse_double()</code></a>을 적용하여 올바른 변수 유형을 얻으세요.

## jsonlite

JSON을 R 데이터 구조로 변환하려면 Jeroen Ooms가 만든 jsonlite 패키지를 추천합니다. 우리는 <a href="https://rdrr.io/pkg/jsonlite/man/read_json.html" class="orm:hideurl"><code>read_json()</code></a>과 <a href="https://rdrr.io/pkg/jsonlite/man/read_json.html" class="orm:hideurl"><code>parse_json()</code></a>이라는 두 가지 jsonlite 함수만 사용할 것입니다. 실제로는 디스크에서 JSON 파일을 읽을 때 <a href="https://rdrr.io/pkg/jsonlite/man/read_json.html" class="orm:hideurl"><code>read_json()</code></a>을 사용할 것입니다. 예를 들어 repurrsive 패키지는 JSON 파일로서 `gh_user`의 소스도 제공하며 <a href="https://rdrr.io/pkg/jsonlite/man/read_json.html" class="orm:hideurl"><code>read_json()</code></a>으로 이를 읽을 수 있습니다.

```
# A path to a json file inside the package:
gh_users_json()
#> [1] "/Users/hadley/Library/R/arm64/4.2/library/repurrrsive/extdata/gh_users.json"

# Read it with read_json()
gh_users2 <- read_json(gh_users_json())

# Check it's the same as the data we were using previously
identical(gh_users, gh_users2)
#> [1] TRUE
```

이 책에서는 JSON을 포함하는 문자열을 인자로 취하여 간단한 예제를 생성하는 데 적합한 <a href="https://rdrr.io/pkg/jsonlite/man/read_json.html" class="orm:hideurl"><code>parse_json()</code></a>도 사용할 것입니다. 시작하기 위해 다음은 숫자부터 시작해서 몇 개의 숫자를 배열에 넣고 그 배열을 객체에 넣는 3가지 간단한 JSON 데이터 세트입니다.

```
str(parse_json('1'))
#>  int 1
str(parse_json('[1, 2, 3]'))
#> List of 3
#>  $ : int 1
#>  $ : int 2
#>  $ : int 3
str(parse_json('{"x": [1, 2, 3]}'))
#> List of 1
#>  $ x:List of 3
#>   ..$ : int 1
#>   ..$ : int 2
#>   ..$ : int 3
```

jsonlite에는 <a href="https://rdrr.io/pkg/jsonlite/man/fromJSON.html" class="orm:hideurl"><code>fromJSON()</code></a>이라는 또 다른 중요한 함수가 있습니다. 자동 단순화(`simplifyVector = TRUE`)를 수행하기 때문에 여기서는 사용하지 않습니다. 이것은 종종, 특히 단순한 경우에 잘 작동하지만, 무슨 일이 일어나고 있는지 정확히 알고 복잡한 중첩 구조를 더 쉽게 처리할 수 있도록 직접 직사각형화(rectangling)를 수행하는 편이 더 낫다고 생각합니다.

## 직사각형화 프로세스 시작하기

대부분의 경우 JSON 파일은 단일 최상위 배열을 포함합니다. 이는 여러 페이지, 여러 레코드, 또는 여러 결과와 같이 여러 "것(things)"에 대한 데이터를 제공하도록 설계되었기 때문입니다. 이 경우 각 요소가 행이 되도록 `tibble(json)`을 사용하여 직사각형화를 시작합니다.

```
json <- '[
  {"name": "John", "age": 34},
  {"name": "Susan", "age": 27}
]'
df <- tibble(json = parse_json(json))
df
#> # A tibble: 2 × 1
#>   json
#>   <list>
#> 1 <named list [2]>
#> 2 <named list [2]>

df |> unnest_wider(json)
#> # A tibble: 2 × 2
#>   name    age
#>   <chr> <int>
#> 1 John     34
#> 2 Susan    27
```

더 드문 경우로 JSON 파일이 하나의 "것"을 나타내는 단일 최상위 JSON 객체로 구성됩니다. 이 경우 티블에 넣기 전에 리스트로 감싸서 직사각형화 프로세스를 시작해야 합니다.

```
json <- '{
  "status": "OK",
  "results": [
    {"name": "John", "age": 34},
    {"name": "Susan", "age": 27}
  ]
}
'
df <- tibble(json = list(parse_json(json)))
df
#> # A tibble: 1 × 1
#>   json
#>   <list>
#> 1 <named list [2]>

df |>
  unnest_wider(json) |>
  unnest_longer(results) |>
  unnest_wider(results)
#> # A tibble: 2 × 3
#>   status name    age
#>   <chr>  <chr> <int>
#> 1 OK     John     34
#> 2 OK     Susan    27
```

또는 구문 분석된 JSON 내부에 접근하여 실제로 관심 있는 부분부터 시작할 수도 있습니다.

```
df <- tibble(results = parse_json(json)$results)
df |> unnest_wider(results)
#> # A tibble: 2 × 2
#>   name    age
#>   <chr> <int>
#> 1 John     34
#> 2 Susan    27
```

## 연습문제

1.  다음 `df_col`과 `df_row`를 직사각형화하세요. 이는 데이터 프레임을 JSON으로 인코딩하는 두 가지 방법을 나타냅니다.

    ```
    json_col <- parse_json('
      {
        "x": ["a", "x", "z"],
        "y": [10, null, 3]
      }
    ')
    json_row <- parse_json('
      [
        {"x": "a", "y": 10},
        {"x": "x", "y": null},
        {"x": "z", "y": 3}
      ]
    ')

    df_col <- tibble(json = list(json_col))
    df_row <- tibble(json = json_row)
    ```

# 요약

이 장에서는 리스트가 무엇인지, JSON 파일에서 리스트를 어떻게 생성하는지, 그리고 이를 어떻게 직사각형 형태의 데이터 프레임으로 바꾸는지 배웠습니다. 놀랍게도 우리에게는 리스트 요소를 행으로 넣는 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>와 리스트 요소를 열로 넣는 <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a>라는 두 가지 새로운 함수만 필요합니다. 리스트 열이 얼마나 깊게 중첩되어 있는지는 중요하지 않습니다. 여러분이 해야 할 일은 이 두 가지 함수를 반복적으로 호출하는 것뿐입니다.

JSON은 웹 API가 반환하는 일반적인 데이터 형식입니다. 웹사이트에 API가 없지만 웹사이트에서 원하는 데이터를 볼 수 있는 경우에는 어떻게 해야 할까요? 그것이 다음 장의 주제인 웹 스크래핑(web scraping), 즉 HTML 웹 페이지에서 데이터를 추출하는 것입니다.

<sup>[1](ch23.html#idm44771276868272-marker)</sup> 이것은 RStudio의 기능입니다.
