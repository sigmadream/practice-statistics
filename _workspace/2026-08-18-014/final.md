---
title: "결측값"
---

```{r}
#| echo: false
source("_common.R")
```

```{r}
#| label: setup
#| message: false
library(tidyverse)
```

## 명시적 결측값

먼저 `NA`가 표시된 셀, 즉 명시적 결측값을 만들거나 제거할 때 유용한(handy) 도구 몇 가지를 살펴보겠습니다.

### 마지막 관측값 이월

결측값은 데이터 입력을 편리하게(data entry convenience) 하려고 자주 사용합니다. 데이터를 수동으로(by hand) 입력할 때는 이전 행의 값이 반복되거나 이월(carried forward)되었음을 결측값으로 나타내기도 합니다.

```{r}
treatment <- tribble(
  ~person,           ~treatment, ~response,
  "Derrick Whitmore", 1,         7,
  NA,                 2,         10,
  NA,                 3,         NA,
  "Katherine Burke",  1,         4
)
```

이런 결측값은 `tidyr::fill()`로 채웁니다(fill in). 열 집합을 받아 `select()`처럼 작동하는 함수입니다.

```{r}
treatment |>
  fill(everything())
```

이 처리를 "마지막 관측값 이월(last observation carried forward)", 줄여서 locf라고도 합니다. 더 특이한(exotic) 방식으로 생긴 결측값을 채울 때는 `.direction` 인수를 사용합니다.

### 고정값

결측값이 고정되고 알려진 값, 보통 0을 나타내기도 합니다. 이때는 `dplyr::coalesce()`로 대체합니다(replace).

```{r}
x <- c(1, 4, 5, 7, NA)
coalesce(x, 0)
```

반대로 구체적인(concrete) 값이 실제로는 결측값을 뜻하는 경우도 있습니다. 보통 결측값을 제대로 나타내지 못하는 구형 소프트웨어가 만든 데이터에서 발생합니다(arises). 이런 소프트웨어는 99나 -999 같은 특별한(special) 값을 대신 사용합니다.

가능하면 데이터를 읽을 때 `readr::read_csv()`의 `na` 인수로 처리하세요(`read_csv(path, na = "99")`). 문제를 나중에 발견했거나 데이터 소스가 읽기 단계에서 처리할 방법을 지원하지 않는다면 `dplyr::na_if()`를 사용합니다.

```{r}
x <- c(1, 4, 5, 7, -99)
na_if(x, -99)
```

### NaN

다음으로 때때로(from time to time) 마주치는 특별한 결측값을 살펴봅시다. `NaN`("난(nan)"이라고 발음함), 즉 숫자가 아님(not a number)입니다. 보통 `NA`와 똑같이 작동하므로 아주 중요하지는 않습니다.

```{r}
x <- c(NA, NaN)
x * 10
x == 1
is.na(x)
```

드물게(In the rare case) `NA`와 `NaN`을 구별(distinguish)해야 한다면 `is.nan(x)`를 사용합니다. 결과를 결정할 수 없는(indeterminate) 수학 연산에서 주로 `NaN`이 나옵니다.

```{r}
0 / 0 
0 * Inf
Inf - Inf
sqrt(-1)
```

## 암묵적 결측값 (Implicit missing values) {#sec-missing-implicit}

지금까지는 데이터에 `NA`로 나타나는 명시적(explicitly) 결측값을 다뤘습니다(missing). 하지만 행 전체가 데이터에 없다면 암묵적(implicitly) 결측값이 됩니다. 어떤 주식의 가격을 분기마다 기록한 간단한 데이터셋으로 차이를 살펴보겠습니다.

```{r}
stocks <- tibble(
  year  = c(2020, 2020, 2020, 2020, 2021, 2021, 2021),
  qtr   = c(   1,    2,    3,    4,    2,    3,    4),
  price = c(1.88, 0.59, 0.35,   NA, 0.92, 0.17, 2.66)
)
```

이 데이터셋에는 결측 관측값이 두 개 있습니다.

1. 2020년 4분기 `price`는 값이 `NA`이므로 명시적 결측값입니다.
2. 2021년 1분기 `price`는 데이터셋에 아예 나타나지 않으므로 암묵적 결측값입니다.

두 값의 차이는 다음 선(Zen)의 화두(koan)로 생각해 볼 수 있습니다.

1. 명시적 결측값은 부재(absence)의 존재(presence)이다.
2. 암묵적 결측값은 존재의 부재이다.

작업할 물리적인(physical) 대상을 마련하려고 암묵적 결측값을 명시적으로 만들 때가 있습니다. 반대로 데이터 구조 때문에 명시적 결측값이 강제로(forced upon) 생겨 이를 없애야 할 때도 있습니다. 다음 절에서는 암묵적 누락(missingness)과 명시적 누락 사이를 오가는(moving between) 도구 몇 가지를 설명합니다.

### 피벗

암묵적 결측값과 명시적 결측값을 서로 바꾸는 도구인 피벗은 이미 살펴봤습니다. 데이터를 더 넓게(wider) 만들면 행과 새 열의 모든 조합(combination)에 값이 필요하므로 암묵적 결측값이 명시적으로 바뀝니다. 예를 들어 분기(`qtr`)가 열에 오도록 `stocks`를 피벗(pivot)하면 두 결측값이 모두 명시적으로 나타납니다.

```{r}
stocks |>
  pivot_wider(
    names_from = qtr, 
    values_from = price
  )
```

기본적으로(By default) 데이터를 길게(longer) 만들어도 명시적 결측값은 남습니다. 다만 데이터가 깔끔하지 않아서만(not tidy) 생긴 구조적(structurally) 결측값이라면 `values_drop_na = TRUE`로 삭제해 암묵적으로 바꿉니다.

### 완료

`tidyr::complete()`는 존재해야 하는 행의 조합을 변수로 지정해 명시적 결측값을 만듭니다. 예를 들어 `stocks` 데이터에는 `year`와 `qtr`의 모든 조합이 있어야 합니다.

```{r}
stocks |>
  complete(year, qtr)
```

보통 기존 변수 이름으로 `complete()`를 호출해(call) 누락된 조합을 채웁니다. 개별 변수 자체가 불완전하다면 데이터를 직접 지정합니다. 예를 들어 `stocks` 데이터셋의 범위가 2019년부터 2021년까지라면 `year`에 해당 값을 명시합니다.

```{r}
stocks |>
  complete(year = 2019:2021, qtr)
```

변수의 범위는 정확하지만 일부 값이 없다면 `full_seq(x, 1)`로 `min(x)`부터 `max(x)`까지 1씩 간격을 둔(spaced out) 값을 모두 만듭니다.

경우에 따라(In some cases) 변수의 단순한 조합만으로는 완전한(complete) 관측값 세트를 만들지 못합니다. 이때는 `complete()`의 작업을 직접 수행합니다. 필요한 기법을 조합해 존재해야 할 행을 모두 담은 데이터 프레임을 만든 뒤 `dplyr::full_join()`으로 원본 데이터셋과 결합하세요.

### 조인

조인(joins)도 암묵적 결측 관측값을 드러내는(revealing) 중요한 방법입니다. 한 데이터셋을 다른 데이터셋과 비교해야만 값이 빠졌음을 아는 경우가 많아 여기서 간단히 언급합니다(mention). `dplyr::anti_join(x, y)`는 `y`에 일치하는 항목(match)이 없는 `x`의 행만 골라내므로 특히 유용합니다. 두 번의 `anti_join()`으로 `flights`에 언급된(mentioned) 4개 공항과 722대 비행기의 정보 누락을 확인합니다.

```{r}
library(nycflights13)

flights |> 
  distinct(faa = dest) |> 
  anti_join(airports)

flights |> 
  distinct(tailnum) |> 
  anti_join(planes)
```

### 연습 문제

1. 항공사(carrier)와 `planes`에서 누락된 것으로 보이는(appear to be missing) 행 사이의 관계(relationship)를 찾을 수 있습니까?

## 요인과 빈 그룹

마지막 누락(missingness) 유형은 빈 그룹(empty group)입니다. 요인(factors)으로 작업할 때 생기며 관측값이 하나도 없는 그룹을 뜻합니다. 사람들의 건강 정보가 담긴 데이터셋을 예로 들어보겠습니다.

```{r}
health <- tibble(
  name   = c("Ikaia", "Oletta", "Leriah", "Dashay", "Tresaun"),
  smoker = factor(c("no", "no", "no", "no", "no"), levels = c("yes", "no")),
  age    = c(34, 88, 75, 47, 56),
)
```

`dplyr::count()`로 흡연자(smokers) 수를 세어(count) 봅시다.

```{r}
health |> count(smoker)
```

이 데이터셋에는 비흡연자만 있지만 실제로는 흡연자도 존재합니다. 흡연자 그룹이 비어 있는 셈입니다. `count()`에 `.drop = FALSE`를 지정하면 데이터에 나타나지 않은 그룹까지 모두 유지합니다(request).

```{r}
health |> count(smoker, .drop = FALSE)
```

같은 원칙(principle)은 ggplot2의 이산형(discrete) 축에도 적용됩니다. 값이 없는 수준은 삭제됩니다(drop). 해당 이산형 축에 `drop = FALSE`를 지정하면 강제로(force) 표시됩니다.

```{r}
#| layout-ncol: 2
#| fig-width: 3
#| fig-alt: 
#|   - x축에 "no"라는 단일 값이 있는 막대 차트.
#|   - 마지막 플롯과 동일한 막대 차트이지만 
#|     이제 x축에 "yes"와 "no"라는 두 개의 값이 있습니다. "yes" 범주에는 막대가 없습니다.
ggplot(health, aes(x = smoker)) +
  geom_bar() +
  scale_x_discrete()

ggplot(health, aes(x = smoker)) +
  geom_bar() +
  scale_x_discrete(drop = FALSE)
```

같은 문제는 `dplyr::group_by()`에서 더 흔히 발생합니다(comes up). 여기서도 `.drop = FALSE`로 모든 요인 수준을 보존합니다.

```{r}
#| warning: false
health |> 
  group_by(smoker, .drop = FALSE) |> 
  summarize(
    n = n(),
    mean_age = mean(age),
    min_age = min(age),
    max_age = max(age),
    sd_age = sd(age)
  )
```

빈 그룹을 요약(summarizing)하면 요약 함수가 길이 0인(zero-length) 벡터에 적용돼 흥미로운 결과가 나옵니다. 길이가 0인 빈(empty) 벡터와 길이가 1인 결측값은 서로 다릅니다(distinction).

```{r}
# 두 개의 결측값을 포함하는 벡터
x1 <- c(NA, NA)
length(x1)

# 아무것도 포함하지 않는 벡터
x2 <- numeric()
length(x2)
```

모든 요약 함수는 길이 0인 벡터에도 작동하지만 뜻밖의 결과를 반환하기도 합니다. 여기서 `mean(age)`는 `NaN`입니다. `mean(age)` = `sum(age)/length(age)`이고 이 경우에는 0/0이기 때문입니다. `max()`와 `min()`은 빈 벡터에 각각 -Inf와 Inf를 반환합니다. 따라서 결과를 비어 있지 않은 새 데이터 벡터와 결합(combine)해 다시 계산하면(recompute) 새 데이터의 최솟값 또는 최댓값이 나옵니다.

더 간단하게는 먼저 요약한 뒤 `complete()`로 암묵적 결측값을 명시적으로 만듭니다.

```{r}
health |> 
  group_by(smoker) |> 
  summarize(
    n = n(),
    mean_age = mean(age),
    min_age = min(age),
    max_age = max(age),
    sd_age = sd(age)
  ) |> 
  complete(smoker)
```

이 방식의 주요 단점(drawback)은 개수(count)가 0이어야 한다고 알면서도(even though you know) `NA`가 나온다는 점입니다.

<!-- HUMANIZE-SUMMARY
원본 글자수: 7,017자
윤문본 글자수: 6,369자
변경률: 14.9% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 5 → 0
- A-7 가지다 직역: 0 → 0
- A-10 가능 표현 남발: 14 → 0
- A-11 목적절 남발: 0 → 0
- A-15 본문 추상 주어·만능 동사: 0 → 0
- C-11 연결어미 뒤 쉼표: 2 → 0

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
- "결측값의 일반적인 용도는 데이터 입력 편의를 위한 것입니다" → "결측값은 데이터 입력을 편리하게 하려고 자주 사용합니다"
- "작업할 물리적인 것을 갖기 위해" → "작업할 물리적인 대상을 마련하려고"
- "조인으로 이어집니다" → "조인도 ... 중요한 방법입니다"
- "관측값이 포함되지 않은 그룹" → "관측값이 하나도 없는 그룹"
- "개수에 대해 `NA`를 얻게 된다는 것입니다" → "개수에 `NA`가 나온다는 점입니다"
-->
