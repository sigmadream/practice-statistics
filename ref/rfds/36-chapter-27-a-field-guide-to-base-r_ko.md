# 27장. 기본 R 현장 가이드 (A Field Guide to Base R)

# 소개

프로그래밍 섹션을 마무리하기 위해, 이 책의 다른 곳에서는 다루지 않은 가장 중요한 기본 R 함수들에 대해 간단히 살펴보겠습니다. 이 도구들은 프로그래밍을 더 많이 할 때 특히 유용하며 야생에서 마주치는 코드를 읽는 데 도움이 될 것입니다.

여기가 데이터 과학 문제를 해결하는 데 tidyverse만이 유일한 방법은 아니라는 점을 상기시키기에 좋은 곳입니다. 이 책에서 tidyverse를 가르치는 이유는 tidyverse 패키지들이 공통된 설계 철학을 공유하여 함수 간의 일관성을 높이고 각각의 새로운 함수나 패키지를 배우고 사용하기 조금 더 쉽게 만들기 때문입니다. 기본 R(base R)을 사용하지 않고 tidyverse를 사용하는 것은 불가능하므로, 패키지를 로드하는 <a href="https://rdrr.io/r/base/library.html" class="orm:hideurl"><code>library()</code></a>, 숫자 요약을 위한 <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>과 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>, 팩터, 날짜 및 POSIXct 데이터 유형, 그리고 물론 `+`, `-`, `/`, `*`, `|`, `&`, `!`와 같은 모든 기본 연산자를 포함하여 이미 *많은* 기본 R 함수를 가르쳤습니다. 지금까지 우리가 집중하지 않았던 것은 기본 R 워크플로우이므로 이 장에서는 그 중 몇 가지를 강조하겠습니다.

이 책을 읽은 후에는 기본 R, data.table 및 다른 패키지를 사용하여 동일한 문제에 대한 다른 접근 방식을 배우게 될 것입니다. 다른 사람들이 작성한 R 코드를 읽기 시작할 때, 특히 StackOverflow를 사용하는 경우 의심할 여지 없이 이러한 다른 접근 방식을 만나게 될 것입니다. 접근 방식을 혼합하여 사용하는 코드를 작성하는 것은 100% 괜찮습니다. 다른 사람의 말에 흔들리지 마세요!

이 장에서는 네 가지 주요 주제인 `[`를 사용한 하위 집합(subsetting), `[[`와 `$`를 사용한 하위 집합, apply 함수 제품군 사용, `for` 루프 사용에 중점을 둘 것입니다. 마지막으로 두 가지 필수 플로팅 함수에 대해 간단히 논의하겠습니다.

## 사전 준비

이 패키지는 기본 R에 초점을 맞추고 있으므로 실제 사전 준비 사항은 없지만, 몇 가지 차이점을 설명하기 위해 tidyverse를 로드하겠습니다.

```
library(tidyverse)
```

# \[로 다중 요소 선택하기 (Selecting Multiple Elements with \[)

`[`는 벡터와 데이터 프레임에서 하위 구성 요소를 추출하는 데 사용되며 `x[i]` 또는 `x[i, j]`와 같이 호출됩니다. 이 섹션에서는 먼저 벡터와 함께 사용하는 방법을 보여준 다음 데이터 프레임과 같은 2D 구조에 동일한 원리가 간단한 방식으로 어떻게 확장되는지 보여줌으로써 `[`의 힘을 소개합니다. 그런 다음 다양한 dplyr 동사(verb)가 어떻게 `[`의 특수한 경우인지 보여주어 그 지식을 굳건히 하는 데 도움을 줄 것입니다.

## 벡터 하위 집합 만들기 (Subsetting Vectors)

벡터의 하위 집합을 만들 수 있는, 즉 `x[i]`에서 `i`가 될 수 있는 것에는 5가지 주요 유형이 있습니다:

- *양의 정수 벡터*. 양의 정수로 하위 집합을 만들면 해당 위치의 요소를 유지합니다.

  ```
  x <- c("one", "two", "three", "four", "five")
  x[c(3, 2, 5)]
  #> [1] "three" "two"   "five"
  ```

  위치를 반복하여 실제로는 입력보다 더 긴 출력을 만들 수 있으므로 "하위 집합 만들기(subsetting)"라는 용어는 약간 잘못된 이름이 됩니다.

  ```
  x[c(1, 1, 5, 5, 5, 2)]
  #> [1] "one"  "one"  "five" "five" "five" "two"
  ```

- *음의 정수 벡터*. 음수 값은 지정된 위치의 요소를 제외합니다.

  ```
  x[c(-1, -3, -5)]
  #> [1] "two"  "four"
  ```

- *논리형 벡터*. 논리형 벡터로 하위 집합을 만들면 `TRUE` 값에 해당하는 모든 값을 유지합니다. 이것은 대부분 비교 함수와 함께 사용할 때 유용합니다.

  ```
  x <- c(10, 3, NA, 5, 8, 1, NA)

  # x의 결측값이 아닌 모든 값
  x[!is.na(x)]
  #> [1] 10  3  5  8  1

  # x의 모든 짝수(또는 결측!) 값
  x[x %% 2 == 0]
  #> [1] 10 NA  8 NA
  ```

  <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 달리 `NA` 인덱스는 출력에 `NA`로 포함됩니다.

- *문자형 벡터*. 명명된 벡터가 있는 경우 문자형 벡터로 하위 집합을 만들 수 있습니다.

  ```
  x <- c(abc = 1, def = 2, xyz = 5)
  x[c("xyz", "def")]
  #> xyz def 
  #>   5   2
  ```

  양의 정수로 하위 집합을 만드는 것과 마찬가지로 문자형 벡터를 사용하여 개별 항목을 복제할 수 있습니다.

- *아무것도 없음(Nothing)*. 마지막 하위 집합 유형은 아무것도 없는 `x[]`이며, 이는 완전한 `x`를 반환합니다. 벡터의 하위 집합을 만드는 데는 유용하지 않지만, 곧 보게 되겠지만 티블(tibble)과 같은 2D 구조의 하위 집합을 만들 때 유용합니다.

## 데이터 프레임 하위 집합 만들기 (Subsetting Data Frames)

데이터 프레임에 `[`를 사용할 수 있는 방법은 꽤 많이 있지만,<sup><a href="ch27.html#idm44771263328096" id="idm44771263328096-marker" data-type="noteref">1</a></sup> 가장 중요한 방법은 `df[rows, cols]`를 사용하여 행과 열을 독립적으로 선택하는 것입니다. 여기서 `rows`와 `cols`는 앞서 설명한 벡터입니다. 예를 들어, `df[rows, ]`와 `df[, cols]`는 빈 하위 집합을 사용하여 다른 차원을 보존하면서 행이나 열만 선택합니다.

다음은 몇 가지 예입니다:

```
df <- tibble(
  x = 1:3, 
  y = c("a", "e", "f"), 
  z = runif(3)
)

# 첫 번째 행과 두 번째 열 선택
df[1, 2]
#> # A tibble: 1 × 1
#>   y    
#>   <chr>
#> 1 a

# 모든 행과 x 및 y 열 선택
df[, c("x" , "y")]
#> # A tibble: 3 × 2
#>       x y    
#>   <int> <chr>
#> 1     1 a    
#> 2     2 e    
#> 3     3 f

# `x`가 1보다 큰 행과 모든 열 선택
df[df$x > 1, ]
#> # A tibble: 2 × 3
#>       x y         z
#>   <int> <chr> <dbl>
#> 1     2 e     0.834
#> 2     3 f     0.601
```

잠시 후 `$`에 대해 다시 살펴보겠지만, `df$x`가 문맥상 무엇을 하는지 짐작할 수 있어야 합니다: `df`에서 `x` 변수를 추출합니다. 여기서 이 방법을 사용해야 하는 이유는 `[`가 tidy evaluation을 사용하지 않으므로 `x` 변수의 출처를 명시해야 하기 때문입니다.

`[`와 관련하여 티블(tibble)과 데이터 프레임 사이에는 중요한 차이점이 있습니다. 이 책에서는 주로 데이터 프레임의 일종이지만 삶을 조금 더 편하게 만들어주기 위해 일부 동작을 조정한 티블을 사용해 왔습니다. 대부분의 경우 "티블"과 "데이터 프레임"을 서로 바꾸어 사용할 수 있으므로 R의 내장 데이터 프레임에 특별한 주의를 환기시키고 싶을 때는 `data.frame`이라고 쓰겠습니다. `df`가 `data.frame`인 경우 `df[, cols]`는 `col`이 단일 열을 선택하면 벡터를 반환하고, 두 개 이상의 열을 선택하면 데이터 프레임을 반환합니다. `df`가 티블인 경우 `[`는 항상 티블을 반환합니다.

```
df1 <- data.frame(x = 1:3)
df1[, "x"]
#> [1] 1 2 3

df2 <- tibble(x = 1:3)
df2[, "x"]
#> # A tibble: 3 × 1
#>       x
#>   <int>
#> 1     1
#> 2     2
#> 3     3
```

`data.frame`에서 이 모호성을 피하는 한 가지 방법은 명시적으로 `drop = FALSE`를 지정하는 것입니다:

```
df1[, "x" , drop = FALSE]
#>   x
#> 1 1
#> 2 2
#> 3 3
```

## dplyr 등가물 (dplyr Equivalents)

몇 가지 dplyr 동사는 `[`의 특수한 경우입니다:

- <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>는 논리형 벡터로 행의 하위 집합을 만드는 것과 같으며, 결측값을 제외하도록 주의합니다:

  ```
  df <- tibble(
    x = c(2, 3, 1, 1, NA), 
    y = letters[1:5], 
    z = runif(5)
  )
  df |> filter(x > 1)

  # same as
  df[!is.na(df$x) & df$x > 1, ]
  ```

  야생에서 볼 수 있는 또 다른 일반적인 기술은 결측값을 삭제하는 부작용 때문에 <a href="https://rdrr.io/r/base/which.html" class="orm:hideurl"><code>which()</code></a>를 사용하는 것입니다: `df[which(df$x > 1), ]`.

- <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>는 정수형 벡터(보통 <a href="https://rdrr.io/r/base/order.html" class="orm:hideurl"><code>order()</code></a>로 생성됨)로 행의 하위 집합을 만드는 것과 같습니다:

  ```
  df |> arrange(x, y)

  # same as
  df[order(df$x, df$y), ]
  ```

  `order(decreasing = TRUE)`를 사용하여 모든 열을 내림차순으로 정렬하거나 `-rank(col)`을 사용하여 열을 개별적으로 내림차순으로 정렬할 수 있습니다.

- <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a> 모두 문자형 벡터로 열의 하위 집합을 만드는 것과 유사합니다:

  ```
  df |> select(x, z)

  # same as
  df[, c("x", "z")]
  ```

기본 R은 또한 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>의 기능이 결합된<sup><a href="ch27.html#idm44771262898928" id="idm44771262898928-marker" data-type="noteref">2</a></sup> <a href="https://rdrr.io/r/base/subset.html" class="orm:hideurl"><code>subset()</code></a>이라는 함수를 제공합니다:

```
df |> 
  filter(x > 1) |> 
  select(y, z)
#> # A tibble: 2 × 2
#>   y           z
#>   <chr>   <dbl>
#> 1 a     0.157  
#> 2 b     0.00740
```

```
# same as
df |> subset(x > 1, c(y, z))
```

이 함수는 dplyr 구문의 많은 부분에 영감을 주었습니다.

## 연습문제 (Exercises)

1.  벡터를 입력으로 취하고 다음을 반환하는 함수를 만드세요:
    1.  짝수 번째 위치의 요소
    2.  마지막 값을 제외한 모든 요소
    3.  짝수 값만 (결측값 제외)
2.  왜 `x[-which(x > 0)]`은 `x[x <= 0]`과 같지 않을까요? <a href="https://rdrr.io/r/base/which.html" class="orm:hideurl"><code>which()</code></a> 문서를 읽고 이를 파악하기 위한 몇 가지 실험을 해보세요.

# \$ 및 \[\[로 단일 요소 선택하기 (Selecting a Single Element with \$ and \[\[)

여러 요소를 선택하는 `[`는 단일 요소를 추출하는 `[[` 및 `$`와 쌍을 이룹니다. 이 섹션에서는 `[[` 및 `$`를 사용하여 데이터 프레임에서 열을 추출하는 방법을 보여주고, `data.frames`와 티블(tibble) 간의 몇 가지 차이점에 대해 논의하며, 리스트와 함께 사용할 때 `[`와 `[[`의 몇 가지 중요한 차이점을 강조하겠습니다.

## 데이터 프레임 (Data Frames)

`[[` 및 `$`를 사용하여 데이터 프레임에서 열을 추출할 수 있습니다. `[[`는 위치나 이름으로 접근할 수 있으며 `$`는 이름으로 접근하는 데 특화되어 있습니다:

```
tb <- tibble(
  x = 1:4,
  y = c(10, 4, 1, 21)
)

# 위치로
tb[[1]]
#> [1] 1 2 3 4

# 이름으로
tb[["x"]]
#> [1] 1 2 3 4
tb$x
#> [1] 1 2 3 4
```

이들은 또한 새로운 열을 생성하는 데 사용될 수 있는데, 이는 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에 해당하는 기본 R입니다:

```
tb$z <- tb$x + tb$y
tb
#> # A tibble: 4 × 3
#>       x     y     z
#>   <int> <dbl> <dbl>
#> 1     1    10    11
```
```
#> 2     2     4     6
#> 3     3     1     4
#> 4     4    21    25
```

<a href="https://rdrr.io/r/base/transform.html" class="orm:hideurl"><code>transform()</code></a>, <a href="https://rdrr.io/r/base/with.html" class="orm:hideurl"><code>with()</code></a> 및 <a href="https://rdrr.io/r/base/with.html" class="orm:hideurl"><code>within()</code></a>을 포함하여 새 열을 만드는 다른 여러 가지 기본 R 접근 방식이 있습니다. Hadley는 몇 가지 [예제](https://oreil.ly/z6vyT)를 수집했습니다.

빠른 요약을 수행할 때 `$`를 직접 사용하는 것이 편리합니다. 예를 들어, 가장 큰 다이아몬드의 크기나 `cut`의 가능한 값만 찾고 싶다면 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>를 사용할 필요가 없습니다:

```
max(diamonds$carat)
#> [1] 5.01

levels(diamonds$cut)
#> [1] "Fair"      "Good"      "Very Good" "Premium"   "Ideal"
```

dplyr은 또한 <a href="ch03.html#chp-data-transform" data-type="xref">3장</a>에서 언급하지 않았던 `[[`/`$`에 해당하는 <a href="https://dplyr.tidyverse.org/reference/pull.html" class="orm:hideurl"><code>pull()</code></a>을 제공합니다. <a href="https://dplyr.tidyverse.org/reference/pull.html" class="orm:hideurl"><code>pull()</code></a>은 변수 이름이나 변수 위치를 취하여 해당 열만 반환합니다. 즉, 파이프를 사용하도록 이전 코드를 다시 작성할 수 있습니다:

```
diamonds |> pull(carat) |> mean()
#> [1] 0.7979397

diamonds |> pull(cut) |> levels()
#> [1] "Fair"      "Good"      "Very Good" "Premium"   "Ideal"
```

## 티블 (Tibbles)

`$`와 관련하여 티블(tibble)과 기본 `data.frame` 간에는 몇 가지 중요한 차이점이 있습니다. 데이터 프레임은 변수 이름의 접두사와 일치하며(소위 *부분 일치(partial matching)*), 열이 존재하지 않아도 불평하지 않습니다:

```
df <- data.frame(x1 = 1)
df$x
#> Warning in df$x: partial match of 'x' to 'x1'
#> [1] 1
df$z
#> NULL
```

티블은 더 엄격합니다. 항상 변수 이름과 정확히 일치하며 접근하려는 열이 존재하지 않으면 경고를 생성합니다:

```
tb <- tibble(x1 = 1)

tb$x
#> Warning: Unknown or uninitialised column: `x`.
#> NULL
tb$z
#> Warning: Unknown or uninitialised column: `z`.
#> NULL
```

이러한 이유로 우리는 종종 티블이 게으르고 퉁명스럽다고 농담합니다. 일은 덜 하고 불평은 더 많이 합니다.

## 리스트 (Lists)

`[[`와 `$`는 리스트 작업에도 정말 중요하며, `[`와 어떻게 다른지 이해하는 것이 중요합니다. `l`이라는 이름의 리스트로 차이점을 설명해 보겠습니다:

```
l <- list(
  a = 1:3, 
  b = "a string", 
  c = pi, 
  d = list(-1, -5)
)
```

- `[`는 하위 리스트를 추출합니다. 몇 개의 요소를 추출하든 결과는 항상 리스트입니다.

  ```
  str(l[1:2])
  #> List of 2
  #>  $ a: int [1:3] 1 2 3
  #>  $ b: chr "a string"

  str(l[1])
  #> List of 1
  #>  $ a: int [1:3] 1 2 3

  str(l[4])
  #> List of 1
  #>  $ d:List of 2
  #>   ..$ : num -1
  #>   ..$ : num -5
  ```

  벡터와 마찬가지로 논리형, 정수형 또는 문자형 벡터로 하위 집합을 만들 수 있습니다.

- `[[`와 `$`는 리스트에서 단일 구성 요소를 추출합니다. 그들은 리스트에서 계층 구조의 수준을 제거합니다.

  ```
  str(l[[1]])
  #>  int [1:3] 1 2 3

  str(l[[4]])
  #> List of 2
  #>  $ : num -1
  #>  $ : num -5

  str(l$a)
  #>  int [1:3] 1 2 3
  ```

`[`와 `[[`의 차이점은 리스트에서 특히 중요한데, `[[`는 리스트의 더 깊은 곳을 파고드는 반면, `[`는 새롭고 더 작은 리스트를 반환하기 때문입니다. 차이점을 기억하는 데 도움이 되도록 <a href="#fig-pepper" data-type="xref">그림 27-1</a>에 표시된 특이한 후추병(pepper shaker)을 살펴보세요. 이 후추병이 여러분의 리스트 `pepper`라면, `pepper[1]`은 단일 후추 패킷이 들어 있는 후추병입니다. `pepper[2]`는 똑같이 생겼지만 두 번째 패킷이 들어 있습니다. `pepper[1:2]`는 두 개의 후추 패킷이 들어 있는 후추병이 됩니다. `pepper[[1]]`은 후추 패킷 자체를 추출합니다.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2701.png" alt="Three photos. On the left is a photo of a glass pepper shaker. Instead of the pepper shaker containing pepper, it contains a single packet of pepper. In the middle is a photo of a single packet of pepper. On the right is a photo of the contents of a packet of pepper." /></p>
<h6 id="figure-27-1.-left-a-pepper-shaker-that-hadley-once-found-in-his-hotel-room.-middle-pepper1.-right-pepper1.">그림 27-1. (왼쪽) 해들리(Hadley)가 언젠가 자신의 호텔 방에서 발견한 후추병. (가운데) <code>pepper[1]</code>. (오른쪽) <code>pepper[[1]]</code>.</h6>
</figure>

이 동일한 원리는 데이터 프레임과 함께 1D `[`를 사용할 때도 적용됩니다: `df["x"]`는 단일 열 데이터 프레임을 반환하고, `df[["x"]]`는 벡터를 반환합니다.

## 연습문제 (Exercises)

1.  벡터의 길이보다 큰 양의 정수와 함께 `[[`를 사용하면 어떻게 됩니까? 존재하지 않는 이름으로 하위 집합을 만들면 어떻게 됩니까?

2.  `pepper[[1]][1]`은 무엇이 됩니까? `pepper[[1]][[1]]`은 어떻습니까?

# Apply 함수 제품군 (Apply Family)

<a href="ch26.html#chp-iteration" data-type="xref">26장</a>에서는 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>dplyr::across()</code></a> 및 map 함수 제품군과 같은 반복을 위한 tidyverse 기술을 배웠습니다. 이 섹션에서는 기본 R에 해당하는 *apply 함수 제품군(apply family)*에 대해 배울 것입니다. 이 맥락에서 apply와 map은 동의어입니다. 왜냐하면 "벡터의 각 요소에 함수를 매핑(map)한다"를 다르게 표현하면 "벡터의 각 요소에 함수를 적용(apply)한다"이기 때문입니다. 여기서는 여러분이 야생에서 이 제품군을 인식할 수 있도록 빠른 개요를 제공할 것입니다.

이 제품군의 가장 중요한 멤버는 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a>과 유사한 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>입니다.<sup><a href="ch27.html#idm44771262249040" id="idm44771262249040-marker" data-type="noteref">3</a></sup> 실제로 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>의 더 고급 기능은 사용하지 않았기 때문에 <a href="ch26.html#chp-iteration" data-type="xref">26장</a>의 모든 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> 호출을 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>로 바꿀 수 있습니다.

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>와 정확히 일치하는 기본 R 함수는 없지만 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>와 함께 `[`를 사용하면 근접할 수 있습니다. 내부적으로 데이터 프레임은 열의 리스트이므로 데이터 프레임에 대해 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>를 호출하면 각 열에 함수가 적용되기 때문입니다.

```
df <- tibble(a = 1, b = 2, c = "a", d = "b", e = 4)

# First find numeric columns
num_cols <- sapply(df, is.numeric)
num_cols
#>     a     b     c     d     e 
#>  TRUE  TRUE FALSE FALSE  TRUE

# Then transform each column with lapply() then replace the original values
df[, num_cols] <- lapply(df[, num_cols, drop = FALSE], \(x) x * 2)
df
#> # A tibble: 1 × 5
#>       a     b c     d         e
#>   <dbl> <dbl> <chr> <chr> <dbl>
#> 1     2     4 a     b         8
```

이전 코드는 새로운 함수인 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a>를 사용합니다. 이는 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>와 유사하지만 항상 결과를 단순화하려고 시도하며(이름의 `s`는 이를 나타냅니다), 여기서는 리스트 대신 논리형 벡터를 생성합니다. 단순화가 실패하여 예기치 않은 유형을 반환할 수 있으므로 프로그래밍에 사용하는 것은 권장하지 않지만 대화형 사용에는 대개 괜찮습니다. purrr에는 <a href="ch26.html#chp-iteration" data-type="xref">26장</a>에서 언급하지 않은 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map_vec()</code></a>라는 유사한 함수가 있습니다.

기본 R은 *v*ector apply의 줄임말인 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>vapply()</code></a>라는 더 엄격한 버전의 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a>를 제공합니다. 이는 예상되는 유형을 지정하는 추가 인자를 취하여 입력에 관계없이 동일한 방식으로 단순화가 발생하도록 합니다. 예를 들어 이전의 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a> 호출을 다음 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>vapply()</code></a>로 바꿀 수 있습니다. 여기서는 <a href="https://rdrr.io/r/base/numeric.html" class="orm:hideurl"><code>is.numeric()</code></a>이 길이 1의 논리형 벡터를 반환할 것으로 예상한다고 지정합니다:

```
vapply(df, is.numeric, logical(1))
#>     a     b     c     d     e 
#>  TRUE  TRUE FALSE FALSE  TRUE
```

<a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a>와 <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>vapply()</code></a> 사이의 구분은 이들이 함수 내부에 있을 때 정말 중요하지만(특이한 입력에 대한 함수의 견고성에 큰 차이를 만들기 때문입니다), 데이터 분석에서는 보통 중요하지 않습니다.

apply 제품군의 또 다른 중요한 구성원은 단일 그룹화된 요약을 계산하는 <a href="https://rdrr.io/r/base/tapply.html" class="orm:hideurl"><code>tapply()</code></a>입니다:

```
diamonds |> 
  group_by(cut) |> 
  summarize(price = mean(price))
#> # A tibble: 5 × 2
#>   cut       price
#>   <ord>     <dbl>
#> 1 Fair      4359.
#> 2 Good      3929.
#> 3 Very Good 3982.
#> 4 Premium   4584.
#> 5 Ideal     3458.

tapply(diamonds$price, diamonds$cut, mean)
#>      Fair      Good Very Good   Premium     Ideal 
#>  4358.758  3928.864  3981.760  4584.258  3457.542
```

불행히도 <a href="https://rdrr.io/r/base/tapply.html" class="orm:hideurl"><code>tapply()</code></a>는 명명된 벡터로 결과를 반환하므로 여러 요약 및 그룹화 변수를 데이터 프레임으로 수집하려면 약간의 체조(gymnastics)가 필요합니다(이렇게 하지 않고 자유롭게 떠다니는 벡터로만 작업하는 것도 확실히 가능하지만 우리의 경험상 이는 작업을 지연시킬 뿐입니다). 다른 그룹화된 요약을 수행하기 위해 <a href="https://rdrr.io/r/base/tapply.html" class="orm:hideurl"><code>tapply()</code></a> 또는 기타 기본 기술을 어떻게 사용할 수 있는지 알고 싶다면 Hadley가 [gist에](https://oreil.ly/evpcw) 몇 가지 기술을 모아두었습니다.

apply 제품군의 마지막 구성원은 이름과 같은 <a href="https://rdrr.io/r/base/apply.html" class="orm:hideurl"><code>apply()</code></a>로, 행렬과 배열에서 작동합니다. 특히 `apply(df, 2, something)`을 주의하세요. 이는 `lapply(df, something)`을 수행하는 느리고 잠재적으로 위험한 방법입니다. 데이터 과학에서는 행렬이 아닌 데이터 프레임으로 주로 작업하기 때문에 이런 경우는 거의 발생하지 않습니다.

# for 루프 (for Loops)

`for` 루프는 apply 및 map 제품군 모두가 내부적으로 사용하는 반복의 기본 구성 요소입니다. `for` 루프는 숙련된 R 프로그래머가 되기 위해 배워야 할 중요하고 강력하며 일반적인 도구입니다. `for` 루프의 기본 구조는 다음과 같습니다:

```
for (element in vector) {
  # do something with element
}
```

`for` 루프의 가장 직접적인 사용은 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>walk()</code></a>와 동일한 효과를 얻는 것입니다: 리스트의 각 요소에 부작용이 있는 함수를 호출하는 것입니다. 예를 들어, <a href="ch26.html#sec-save-database" data-type="xref">“데이터베이스에 쓰기”</a>에서 `walk()`를 사용하는 대신:

```
paths |> walk(append_file)
```

우리는 `for` 루프를 사용할 수 있었습니다:

```
for (path in paths) {
  append_file(path)
}
```

예를 들어 <a href="ch26.html#chp-iteration" data-type="xref">26장</a>에서 했던 것처럼 디렉터리의 모든 Excel 파일을 읽는 것과 같이 `for` 루프의 출력을 저장하려는 경우 상황이 조금 더 까다로워집니다:

```
paths <- dir("data/gapminder", pattern = "\\.xlsx$", full.names = TRUE)
files <- map(paths, readxl::read_excel)
```

사용할 수 있는 몇 가지 다양한 기술이 있지만, 처음에 출력이 어떤 모습일지 명시적으로 지정하는 것이 좋습니다. 이 경우 <a href="https://rdrr.io/r/base/vector.html" class="orm:hideurl"><code>vector()</code></a>를 사용하여 생성할 수 있는 `paths`와 길이가 같은 리스트를 원할 것입니다:

```
files <- vector("list", length(paths))
```

그런 다음 `paths`의 요소를 반복하는 대신 <a href="https://rdrr.io/r/base/seq.html" class="orm:hideurl"><code>seq_along()</code></a>을 사용하여 `paths`의 각 요소에 대해 하나의 인덱스를 생성하여 해당 인덱스를 반복합니다:

```
seq_along(paths)
#>  [1]  1  2  3  4  5  6  7  8  9 10 11 12
```

입력의 각 위치를 출력의 해당 위치와 연결할 수 있기 때문에 인덱스를 사용하는 것이 중요합니다:

```
for (i in seq_along(paths)) {
  files[[i]] <- readxl::read_excel(paths[[i]])
}
```

티블 리스트를 단일 티블로 결합하려면 <a href="https://rdrr.io/r/base/do.call.html" class="orm:hideurl"><code>do.call()</code></a> + <a href="https://rdrr.io/r/base/cbind.html" class="orm:hideurl"><code>rbind()</code></a>를 사용할 수 있습니다:

```
do.call(rbind, files)
#> # A tibble: 1,704 × 5
#>   country     continent lifeExp      pop gdpPercap
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 Afghanistan Asia         28.8  8425333      779.
#> 2 Albania     Europe       55.2  1282697     1601.
#> 3 Algeria     Africa       43.1  9279525     2449.
#> 4 Angola      Africa       30.0  4232095     3521.
#> 5 Argentina   Americas     62.5 17876956     5911.
#> 6 Australia   Oceania      69.1  8691212    10040.
#> # … with 1,698 more rows
```

리스트를 만들고 결과를 저장하는 것보다 더 간단한 방법은 데이터 프레임을 하나씩 쌓아가는 것입니다:

```
out <- NULL
for (path in paths) {
  out <- rbind(out, readxl::read_excel(path))
}
```

이 패턴은 벡터가 길 때 속도가 느려질 수 있으므로 피하는 것이 좋습니다. 이것이 `for` 루프가 느리다는 끈질긴 헛소문의 근원입니다. `for` 루프 자체는 느리지 않지만 벡터를 반복적으로 키우는 것은 느립니다.

# 플롯 (Plots)

그렇지 않으면 tidyverse를 사용하지 않는 많은 R 사용자는 현명한 기본값, 자동 범례 및 현대적인 모양과 같은 유용한 기능 때문에 플로팅에 ggplot2를 선호합니다. 그러나 기본 R 플로팅 함수는 매우 간결하기 때문에 여전히 유용할 수 있습니다. 기본 탐색 플롯을 수행하는 데 타이핑이 거의 필요하지 않습니다.

야생에서 볼 수 있는 기본 플롯에는 산점도와 히스토그램의 두 가지 주요 유형이 있으며, 각각 <a href="https://rdrr.io/r/graphics/plot.default.html" class="orm:hideurl"><code>plot()</code></a> 및 <a href="https://rdrr.io/r/graphics/hist.html" class="orm:hideurl"><code>hist()</code></a>로 생성됩니다. 다음은 `diamonds` 데이터 세트의 간단한 예입니다:

```
# Left
hist(diamonds$carat)

# Right
plot(diamonds$carat, diamonds$price)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_27in01.png" alt="On the left, histogram of carats of diamonds, ranging from 0 to 5 carats. The distribution is unimodal and right-skewed. On the right, scatterplot of price versus carat of diamonds, showing a positive relationship that fans out as both price and carat increases. The scatterplot shows very few diamonds bigger than 3 carats compared to diamonds between 0 to 3 carats." />
</figure>

기본 플로팅 함수는 벡터와 함께 작동하므로 `$` 또는 다른 기술을 사용하여 데이터 프레임에서 열을 빼내야 합니다.

# 요약 (Summary)

이 장에서는 하위 집합을 만들고 반복하는 데 유용한 기본 R 함수를 엄선하여 보여드렸습니다. 책의 다른 곳에서 논의된 접근 방식과 비교할 때 이러한 함수는 데이터 프레임과 일부 열 사양을 취하는 대신 개별 벡터를 취하는 경향이 있기 때문에 "데이터 프레임" 특징보다는 "벡터" 특징을 더 많이 가지는 경향이 있습니다. 이것은 종종 프로그래밍을 쉽게 만들어주므로 더 많은 함수를 작성하고 자신만의 패키지를 작성하기 시작할 때 더욱 중요해집니다.

이 장으로 이 책의 프로그래밍 섹션을 마칩니다. 여러분은 R을 사용하는 데이터 과학자뿐만 아니라 R로 *프로그래밍*할 수 있는 데이터 과학자가 되기 위한 여정을 순조롭게 시작했습니다. 이 장들이 여러분의 프로그래밍에 대한 흥미를 불러일으키고 이 책 외에도 더 많은 것을 배울 수 있기를 바랍니다.

<sup>[1](ch27.html#idm44771263328096-marker)</sup> 데이터 프레임을 1D 객체인 것처럼 부분집합을 지정하는 방법과 행렬로 부분집합을 지정하는 방법에 대해 알아보려면 *Advanced R*의 [여러 요소 선택 섹션](https://oreil.ly/VF0sY)을 읽어보세요.

<sup>[2](ch27.html#idm44771262898928-marker)</sup> 하지만 그룹화된 데이터 프레임을 다르게 처리하지 않으며 <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>starts_with()</code></a>와 같은 선택 도우미 함수를 지원하지 않습니다.

<sup>[3](ch27.html#idm44771262249040-marker)</sup> 오류가 있는 경우 어떤 요소가 문제를 일으켰는지 보고하거나 진행률 표시줄과 같은 편리한 기능이 없을 뿐입니다.
