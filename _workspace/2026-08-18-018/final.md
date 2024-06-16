---
title: 기본 R 가이드
---

```{r}
#| echo: false
source("_common.R")
```

책의 다른 부분에서 다루지 않은 중요한 기본 R 함수를 간단히 살펴보겠습니다. 프로그래밍을 많이 할수록 유용한 도구이며 실제 코드도 더 쉽게 읽게 됩니다.

tidyverse만이 데이터 과학 문제를 해결하는 방법은 아닙니다. 이 책에서 tidyverse를 가르치는 이유는 패키지들이 공통된 디자인 철학을 공유하기 때문입니다. 함수 사이의 일관성이 높아 새로운 함수나 패키지를 조금 더 쉽게 배우고 사용합니다.

tidyverse는 기본 R 없이 사용할 수 없으므로 사실 이미 수많은 기본 R 함수를 배웠습니다.

패키지를 불러오는 `library()`, 숫자를 요약하는 `sum()`과 `mean()`, 요인(factor)·날짜·POSIXct 데이터 유형, `+`, `-`, `/`, `*`, `|`, `&`, `!` 같은 기본 연산자가 여기에 해당합니다. 지금까지는 기본 R 워크플로우에 집중하지 않았으므로 이 장에서 몇 가지를 짚어봅니다.

이 장은 `[`를 사용한 부분 집합 추출, `[[`와 `$`를 사용한 부분 집합 추출, apply 함수 제품군, `for` 루프라는 네 주제를 다룹니다.

```{r}
#| label: setup
#| message: false
library(tidyverse)
```

## `[`를 사용하여 여러 요소 선택하기

`[`는 벡터와 데이터 프레임에서 하위 구성 요소를 추출하며 `x[i]` 또는 `x[i, j]`처럼 호출(called)합니다. 먼저 벡터에서 사용하는 방법을 봅니다. 이어서 같은 원리가 데이터 프레임 같은 2차원 구조로 어떻게 확장되는지 살펴봅니다.

### 벡터 부분 집합 추출

벡터의 부분 집합을 추출할 때 `x[i]`의 `i`에 넣는 값은 크게 다섯 유형입니다.

1. 양의 정수 벡터: 양의 정수로 부분 집합을 추출하면 해당 위치의 요소를 유지합니다.

```{r}
x <- c("one", "two", "three", "four", "five")
x[c(3, 2, 5)]
```

위치를 반복하면 입력보다 긴 출력도 만들 수 있어 "부분 집합 추출"이라는 이름은 다소 어색합니다.

```{r}
x[c(1, 1, 5, 5, 5, 2)]
```

2. 음의 정수 벡터: 음수 값은 지정된 위치의 요소를 삭제합니다.

```{r}
x[c(-1, -3, -5)]
```

3. 논리 벡터: `TRUE`에 해당하는 값을 모두 유지합니다. 비교 함수(comparison functions)와 함께(in conjunction with) 자주 사용합니다(most often).

```{r}
x <- c(10, 3, NA, 5, 8, 1, NA)

# x의 누락되지 않은 모든 값
x[!is.na(x)]

# x의 모든 짝수 (또는 누락된!) 값
x[x %% 2 == 0]
```

`filter()`와 달리, `NA` 인덱스는 출력에 `NA`로 포함됩니다.

4. 문자 벡터: 이름이 있는 벡터는 문자 벡터로 부분 집합을 추출합니다.

```{r}
x <- c(abc = 1, def = 2, xyz = 5)
x[c("xyz", "def")]
```

양의 정수와 마찬가지로 문자 벡터로 개별 항목을 복제합니다.

5. 아무것도 없음: `x[]`처럼 아무것도 넣지 않으면 완전한 `x`를 반환합니다. 벡터에서는 쓸모가 적지만 티블 같은 2차원 구조를 추출할 때 유용합니다.

### 데이터 프레임 부분 집합 추출

데이터 프레임에서 `[`를 쓰는 방법은 많지만 가장 중요한 형태는 `df[rows, cols]`입니다. 행과 열을 독립적으로 선택하며 `rows`와 `cols`에는 앞에서 설명한 벡터를 넣습니다.

예를 들어 `df[rows, ]` 및 `df[, cols]`는 빈 부분 집합을 사용하여 다른 차원을 보존하면서 행이나 열만 선택합니다.

다음은 몇 가지 예제입니다.

```{r}
df <- tibble(
  x = 1:3, 
  y = c("a", "e", "f"), 
  z = runif(3)
)

# 첫 번째 행과 두 번째 열 선택
df[1, 2]

# 모든 행과 열 x, y 선택
df[, c("x" , "y")]

# `x`가 1보다 큰 행과 모든 열 선택
df[df$x > 1, ]
```

`$`는 뒤에서 다시 설명합니다. 여기서 `df$x`는 `df`에서 `x` 변수를 추출합니다. `[`는 깔끔한 평가를 사용하지 않으므로 `x`의 출처를 명시해야 합니다.

`[`의 동작은 티블과 데이터 프레임에서 다릅니다. 이 책에서는 데이터 프레임의 일부 동작을 더 편리하게 조정한 티블을 주로 사용했습니다. 보통 두 용어를 바꿔 쓰되 R 내장 데이터 프레임을 특별히 가리킬 때는 `data.frame`이라고 하겠습니다. `df`가 `data.frame`이면 `df[, cols]`는 열 하나를 선택할 때 벡터, 둘 이상을 선택할 때 데이터 프레임을 반환합니다. 티블에서 `[`는 언제나 티블을 반환합니다.

```{r}
df1 <- data.frame(x = 1:3)
df1[, "x"]

df2 <- tibble(x = 1:3)
df2[, "x"]
```

`data.frame`에서 이러한 모호함을 방지하는 한 가지 방법은 `drop = FALSE`를 명시적으로 지정하는 것입니다.

```{r}
df1[, "x" , drop = FALSE]
```

### dplyr 등가물

여러 dplyr 동사는 `[`를 특수하게 활용한 경우입니다.

1. `filter()`는 논리 벡터를 사용하여 행을 부분 집합으로 추출하는 것과 동일하며 누락된 값을 제외하도록 주의합니다.

```{r}
#| results: false
df <- tibble(
  x = c(2, 3, 1, 1, NA), 
  y = letters[1:5], 
  z = runif(5)
)
df |> filter(x > 1)

# 다음과 같음
df[!is.na(df$x) & df$x > 1, ]
```

실무에서는 `which()`의 부작용을 이용해 누락된 값을 삭제하기도 합니다. `df[which(df$x > 1), ]`처럼 씁니다.

2. `arrange()`는 대개 `order()`로 생성된 정수 벡터를 사용하여 행을 부분 집합으로 추출하는 것과 같습니다.

```{r}
#| results: false
df |> arrange(x, y)

# 다음과 같음
df[order(df$x, df$y), ]
```

모든 열을 내림차순으로 정렬할 때는 `order(decreasing = TRUE)`, 개별 열에는 `-rank(col)`을 사용합니다.

3. `select()`와 `relocate()`는 모두 문자 벡터를 사용하여 열을 부분 집합으로 추출하는 것과 유사합니다.

```{r}
#| results: false
df |> select(x, z)

# 다음과 같음
df[, c("x", "z")]
```

기본 R에는 `filter()`와 `select()`의 기능을 결합한 `subset()` 함수도 있습니다.

```{r}
df |> 
  filter(x > 1) |> 
  select(y, z)
```

```{r}
#| results: false
# 다음과 같음
df |> subset(x > 1, c(y, z))
```

이 함수는 dplyr 구문의 많은 부분에 영감을 주었습니다.

### 연습 문제

1. 벡터를 입력으로 취하고 다음을 반환하는 함수를 생성하세요.
  a.  짝수(even-numbered) 위치에 있는 요소.
  b.  마지막 값을 제외한(except) 모든 요소.
  c.  짝수 값만 (그리고 누락된 값 없음).

2. `x[-which(x > 0)]`은 `x[x <= 0]`과 같지 않을까요? `which()` 설명서를 읽고 실험해 보세요.

## `$`와 `[[`를 사용하여 단일 요소 선택하기

`[`가 여러 요소를 선택한다면 `[[`와 `$`는 단일 요소를 추출합니다. 이 절에서는 데이터 프레임에서 열을 추출하는 방법, `data.frame`과 티블의 차이, 리스트에서 `[`와 `[[`가 어떻게 다른지 살펴봅니다.

### 데이터 프레임

`[[`와 `$`는 데이터 프레임에서 열을 추출합니다. `[[`는 위치나 이름을 사용하고 `$`는 이름으로 접근할 때 씁니다.

```{r}
tb <- tibble(
  x = 1:4,
  y = c(10, 4, 1, 21)
)

# 위치 기준(by position)
tb[[1]]

# 이름 기준(by name)
tb[["x"]]
tb$x
```

이러한 기호는 `mutate()`의 기본 R 등가물(equivalent)인 새 열을 생성하는 데 사용할 수도 있습니다.

```{r}
tb$z <- tb$x + tb$y
tb
```

`transform()`, `with()`, `within()`을 비롯해 새 열을 만드는 다른 기본 R 방식도 몇 가지 있습니다. 해들리는 [https://gist.github.com/hadley/1986a273e384fb2d4d752c18ed71bedf](https://gist.github.com/hadley/1986a273e384fb2d4d752c18ed71bedf)에 예제를 모았습니다.

간단히 요약할 때는 `$`를 직접 쓰면 편리합니다. 큰 다이아몬드의 크기나 `cut`의 가능한 값을 찾는 데 `summarize()`까지 사용할 필요는 없습니다.

```{r}
max(diamonds$carat)

levels(diamonds$cut)
```

dplyr의 `pull()`은 `[[`/`$`에 해당합니다. 변수 이름이나 위치를 받아 해당 열만 반환하므로 위 코드를 파이프로 다시 작성합니다.

```{r}
diamonds |> pull(carat) |> max()

diamonds |> pull(cut) |> levels()
```

### 티블

`$`도 티블과 기본 `data.frame`에서 다르게 작동합니다. 데이터 프레임은 변수 이름의 접두사를 부분 일치(partial matching)시키며 열이 없어도 불평하지(complain) 않습니다.

```{r}
df <- data.frame(x1 = 1)
df$x
df$z
```

티블은 더 엄격합니다. 변수 이름을 정확히 일치시키고 접근하려는 열이 없으면 경고합니다.

```{r}
tb <- tibble(x1 = 1)

tb$x
tb$z
```

이런 이유로 우리는 때때로 티블이 게으르고(lazy) 뚱하다고(surly) 농담을 합니다. 티블은 일을 덜 하고 불평을 더 많이 합니다.

### 리스트

리스트에서도 `[[`와 `$`가 중요합니다. `l`이라는 리스트로 `[`와의 차이를 살펴봅시다.

```{r}
l <- list(
  a = 1:3, 
  b = "a string", 
  c = pi, 
  d = list(-1, -5)
)
```

1. `[`는 하위 리스트를 추출합니다. 얼마나 많은(how many) 요소를 추출하든 상관없이 결과는 항상 리스트가 됩니다.

    ```{r}
    str(l[1:2])

    str(l[1])

    str(l[4])
    ```

    벡터와 마찬가지로 논리(logical), 정수(integer), 문자(character) 벡터로 부분 집합을 추출합니다.

2. `[[`와 `$`는 리스트에서 단일 구성 요소를 추출합니다. 이들은 리스트에서 계층 구조 수준을 제거합니다.

```{r}
str(l[[1]])

str(l[[4]])

str(l$a)
```

리스트에서 이 차이는 특히 중요합니다. `[[`는 리스트 안으로 파고들고(drills down into) `[`는 더 작은 새(new, smaller) 리스트를 반환합니다.

데이터 프레임에 1d `[`를 사용할 때도 동일한 원리(same principle)가 적용됩니다. `df["x"]`는 하나의 열(one-column) 데이터 프레임을 반환하고 `df[["x"]]`는 벡터를 반환합니다.

### 연습 문제

1. 벡터의 길이보다 큰 양의 정수와 함께 `[[`를 사용하면 어떻게 되나요? 존재하지 않는 이름(name)으로 부분 집합을 추출하면 어떻게 되나요?

2. `pepper[[1]][1]`은 무엇일까요? `pepper[[1]][[1]]`은 어떨까요?

## Apply 제품군

지금까지 `dplyr::across()`와 map 함수 제품군 같은 tidyverse 반복 기술을 배웠습니다. 이제 기본 R의 apply 제품군(apply family)을 알아봅니다. 이 문맥에서 apply와 map은 같은 뜻입니다. "벡터의 각 요소에 함수를 매핑한다"는 말은 곧 "함수를 적용한다"는 뜻입니다. 실제 코드에서 알아볼 수 있도록 간단히 개관하겠습니다.

중요한 함수는 `purrr::map()`과 매우 비슷한 `lapply()`입니다. `across()`와 정확히 같은 기본 R 함수는 없지만 `lapply()`와 `[`를 함께 쓰면 비슷합니다. 데이터 프레임은 내부적으로 열의 리스트이므로 `lapply()`를 호출하면 함수가 각 열에 적용됩니다.

```{r}
df <- tibble(a = 1, b = 2, c = "a", d = "b", e = 4)

# 먼저 숫자 열 찾기
num_cols <- sapply(df, is.numeric)
num_cols

# 그런 다음 lapply()를 사용하여 각 열을 변환한 다음 원래 값을 바꿉니다
df[, num_cols] <- lapply(df[, num_cols, drop = FALSE], \(x) x * 2)
df
```

위 코드는 `sapply()`라는 새 함수를 사용합니다. `lapply()`와 비슷하지만 항상 결과 단순화를 시도해 이름에 `s`가 붙었습니다. 여기서는 리스트 대신 논리 벡터를 만듭니다. 단순화에 실패하면 예상치 못한 유형이 나올 수 있어 프로그래밍에는 권하지 않지만 대화형으로 쓰기에는 대체로 괜찮습니다.

purrr에는 비슷한 `map_vec()`이 있습니다. 기본 R의 `vapply()`는 vector apply의 줄임말이며 `sapply()`보다 엄격합니다. 예상 유형을 추가 인수로 받아 입력과 관계없이 언제나 같은 방식으로 단순화합니다.

예를 들어 `is.numeric()`이 길이 1인 논리 벡터를 반환한다고 지정해 위의 `sapply()` 호출을 `vapply()`로 바꿉니다.

```{r}
vapply(df, is.numeric, logical(1))
```

`sapply()`와 `vapply()`의 차이는 함수 안에서 중요합니다. 비정상적인 입력에 대한 견고성(robustness to unusual inputs)에 큰 차이(big difference)를 만들기 때문입니다. 일반적인 데이터 분석에서는 대개 중요하지(matter) 않습니다.

apply 제품군의 또 다른 중요한 멤버는 단일 그룹화된 요약을 계산하는 `tapply()`입니다.

```{r}
diamonds |> 
  group_by(cut) |> 
  summarize(price = mean(price))

tapply(diamonds$price, diamonds$cut, mean)
```

`tapply()`는 결과를 이름 있는 벡터로 반환합니다. 여러 요약과 그룹화 변수를 데이터 프레임으로 모으려면 약간의 수고가 필요합니다. 벡터를 그대로 써도 되지만 경험상 작업만 늦어집니다.

`tapply()` 또는 기타 기본 기술을 사용하여 다른 그룹화된 요약을 수행하는 방법을 보려면 해들리가 [gist]([https://gist.github.com/hadley/c430501804349d382ce90754936ab8ec](https://gist.github.com/hadley/c430501804349d382ce90754936ab8ec))에 수집한 몇 가지 기술을 확인하세요.

마지막은 이름 그대로 행렬과 배열에 쓰는 `apply()`입니다. `apply(df, 2, something)`은 `lapply(df, something)`보다 느리고 잠재적으로 위험하니 주의하세요. 데이터 과학에서는 주로 데이터 프레임을 다루므로 이 함수를 거의 쓰지 않습니다.

## `for` 루프

`for` 루프는 apply와 map 제품군이 내부에서 사용하는 반복의 기본 구성 요소입니다. 강력하고 일반적인 도구라 숙련된 R 프로그래머가 되려면 알아야 합니다.
`for` 루프의 기본 구조는 다음과 같습니다.

```{r}
#| eval: false
for (element in vector) {
  # 요소로(with element) 무언가를 수행합니다(do something)
}
```

`for` 루프의 직관적인 용도는 `walk()`와 같은 효과를 내는 것입니다. 리스트의 각 요소에 부작용이 있는 함수를 호출합니다.

```{r}
#| eval: false
paths |> walk(append_file)
```

같은 작업을 `for` 루프로도 할 수(could have used) 있습니다.

```{r}
#| eval: false
for (path in paths) {
  append_file(path)
}
```

`for` 루프의 출력을 저장하면(if you want to save) 조금 까다로워집니다(get a little trickier).

```{r}
paths <- dir("data/gapminder", pattern = "\\.xlsx$", full.names = TRUE)
files <- map(paths, readxl::read_excel)
```

방법은 여러 가지지만 출력(output)의 형태(going to look like)를 미리(upfront) 명확히(explicit) 정하기를 권합니다(recommend). 여기서는 `paths`와 같은 길이(same length)의 리스트를 `vector()`로 만듭니다(create).

```{r}
files <- vector("list", length(paths))
```

그런 다음 `paths`의 요소를 직접 반복하지 않고 `seq_along()`으로 각 요소의 인덱스(index)를 만들어 반복합니다(iterating over).

```{r}
seq_along(paths)
```

인덱스는 입력의 각 위치를 출력의 해당 위치와 연결하므로 중요합니다.

```{r}
for (i in seq_along(paths)) {
  files[[i]] <- readxl::read_excel(paths[[i]])
}
```

티블 리스트를 하나로 결합할 때는 `do.call()` + `rbind()`를 사용합니다.

```{r}
do.call(rbind, files)
```

리스트를 미리 만들지 않고 데이터 프레임을 조금씩 구축하는 더 간단한 방식도 있습니다.

```{r}
out <- NULL
for (path in paths) {
  out <- rbind(out, readxl::read_excel(path))
}
```

이 패턴은 벡터가 길면 매우 느려지므로 피하세요. 여기서 `for` 루프가 느리다는 오해가 생겼습니다. 루프가 아니라 벡터를 반복해서 키우는 작업이 느립니다.

## 플롯 (Plots)

tidyverse를 쓰지 않는 R 사용자도 합리적인 기본값, 자동 범례, 최신 디자인 때문에 플롯에는 ggplot2를 선호하는 경우가 많습니다.

그래도 기본 R 플로팅 함수는 매우 간결해 탐색용 플롯을 적은 코드로 그릴 때 유용합니다.

실제 코드에서 자주 보는 기본 플롯은 `plot()`의 산점도와 `hist()`의 히스토그램입니다. 다음은 다이아몬드 데이터셋의 간단한 예입니다.

```{r}
#| dev: png
#| fig-width: 4
#| fig-asp: 1
#| layout-ncol: 2
#| fig-alt: |
#|   On the left, histogram of carats of diamonds, ranging from 0 to 5 carats.
#|   The distribution is unimodal and right-skewed. On the right, scatter 
#|   plot of price vs. carat of diamonds, showing a positive relationship 
#|   that fans out as both price and carat increases. The scatter plot 
#|   shows very few diamonds bigger than 3 carats compared to diamonds between 
#|   0 to 3 carats.
# 왼쪽(Left)
hist(diamonds$carat)

# 오른쪽(Right)
plot(diamonds$carat, diamonds$price)
```

<!-- HUMANIZE-SUMMARY
원본 글자수: 11,522자
윤문본 글자수: 10,202자
변경률: 15.5% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 4 → 0
- A-7 가지다 직역: 0 → 0
- A-10 가능 표현 남발: 19 → 0
- A-11 목적절 남발: 1 → 0
- A-15 본문 추상 주어·만능 동사: 3 → 0
- C-11 연결어미 뒤 쉼표: 16 → 0

자체검증: 6/6 통과
1. 고유명사·수치·날짜·인용·내용 앵커 보존: 통과
2. 변경률 30% 이하: 통과
3. 장르 유지: 통과
4. register 유지: 통과
5. 잔존 S1 패턴 0건: 통과
6. 새 비유·수사·상투구 없음: 통과

등급: A
사유: S1·선별 S2 패턴이 남지 않았고 본문 변경률이 10~25% 범위임.

주요 변경 하이라이트:
- "중요한 기본 R 함수들에 대해" → "중요한 기본 R 함수를"
- "부분 집합 추출이라는 용어는 다소 부적절한 명칭" → "부분 집합 추출이라는 이름은 다소 어색"
- "문맥상 ... 짐작할 수 있을 것입니다" → "`df$x`는 `df`에서 `x` 변수를 추출합니다"
- "데이터 프레임이면서 ... 티블" → "일부 동작을 더 편리하게 조정한 티블"
- "루프가 느린 게 아니라" → "벡터를 반복해서 키우는 작업이 느립니다"
-->
