---
title: "데이터 불러오기"
---

```r
#| echo: false
source("_common.R")
```

## 스프레드시트 (Spreadsheets)

`.csv`와 `.tsv` 같은 일반 텍스트 파일에서 데이터를 가져오는 방법을 배웠습니다. 이번에는 Excel이나 Google 스프레드시트에서 데이터를 불러와 보겠습니다.

스프레드시트 데이터를 다룰 때 따로 살펴야 할 점과 그에 따른 복잡성도 함께 짚습니다.

여러분이나 동료가 스프레드시트로 데이터를 구성한다면 Karl Broman과 Kara Woo의 "스프레드시트의 데이터 구성(Data Organization in Spreadsheets)" 논문([https://doi.org/10.1080/00031305.2017.1375989](https://doi.org/10.1080/00031305.2017.1375989))을 꼭 읽어보길 권합니다.

이 논문이 제시하는 모범 사례(best practices)를 따르면 분석과 시각화에 쓸 데이터를 스프레드시트에서 R로 옮길 때 생기는 골칫거리(headache)를 크게 줄일 수 있습니다.

Microsoft Excel은 데이터를 파일 안의 워크시트에 정리하는 대표적인 스프레드시트 프로그램입니다.

```r
#| message: false
library(readxl)
library(tidyverse)
library(writexl)
```

readxl에는 Excel 스프레드시트를 R로 불러오는 함수가 여럿 있습니다.

1. `read_xls()`는 `xls` 형식의 Excel 파일을 읽습니다.
2. `read_xlsx()`는 `xlsx` 형식의 Excel 파일을 읽습니다.
3. `read_excel()`은 `xls`와 `xlsx` 형식을 모두 읽습니다. 입력값을 보고 파일 유형을 추정합니다.

이 함수들의 구문(syntax)은 앞서 다른 파일을 읽을 때 쓴 `read_csv()`, `read_table()` 등과 비슷합니다. 여기서는 `read_excel()`을 중심으로 살펴봅니다.

### 엑셀 스프레드시트 읽기

먼저 R로 읽을 스프레드시트가 Excel에서 어떻게 보이는지 확인해 봅시다. 이 스프레드시트는 [https://docs.google.com/spreadsheets/d/1V1nPp1tzOuutXFLb3G9Eyxi3qxeEhnOXUzL5_BcCQ0w/](https://docs.google.com/spreadsheets/d/1V1nPp1tzOuutXFLb3G9Eyxi3qxeEhnOXUzL5_BcCQ0w/)에서 Excel 파일로 내려받을 수 있습니다.

`read_excel()`의 첫 번째 인수는 읽을 파일의 경로입니다.

```r
students <- read_excel("data/students.xlsx")
```

`read_excel()`은 파일을 읽어 티블로 만듭니다.

```r
students
```

데이터에는 학생 6명과 학생별 변수 5개가 들어 있습니다.
다만 몇 군데는 손볼 필요가 있습니다.

1. 열 이름이 제각각입니다. `col_names` 인수에 새 이름을 지정해 일관된 형식으로 맞출 수 있습니다. 여기서는 `snake_case`를 권합니다.

```r
read_excel(
  "data/students.xlsx",
  col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age")
)
```

아직 결과가 완벽하지는 않습니다. 변수 이름은 원하는 대로 바뀌었습니다. 그러나 원래 머리글이던 행이 첫 번째 관측값으로 들어왔습니다. `skip` 인수로 이 행을 건너뜁니다.

```r
read_excel(
  "data/students.xlsx",
  col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
  skip = 1
)
```

2.  `favourite_food` 열의 `N/A`는 "이용할 수 없음(not available)"이라는 뜻이지만 지금은 `NA`로 인식되지 않았습니다. 목록에서 네 번째 학생의 나이와 비교해 보세요. `na` 인수에는 `NA`로 처리할 문자열을 지정합니다. 기본값은 `""`뿐입니다. 스프레드시트에서는 빈 셀이나 수식 `=NA()`가 있는 셀이 여기에 해당합니다.

```r
read_excel(
  "data/students.xlsx",
  col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
  skip = 1,
  na = c("", "N/A")
)
```

3. `age`가 문자형으로 읽힌 점도 문제입니다. 실제로는 숫자형이어야 합니다. 플랫 파일을 읽는 `read_csv()` 계열과 마찬가지로 `read_excel()`도 `col_types` 인수에서 각 변수의 열 유형을 지정합니다. 다만 구문은 조금 다릅니다. 선택지는 `"skip"`, `"guess"`, `"logical"`, `"numeric"`, `"date"`, `"text"`, `"list"`입니다.

```r
read_excel(
  "data/students.xlsx",
  col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
  skip = 1,
  na = c("", "N/A"),
  col_types = c("numeric", "text", "text", "text", "numeric")
)
```

이번에도 원하는 결과가 바로 나오지는 않았습니다. `age`를 숫자형으로 지정하자 숫자가 아닌 `five`가 든 셀이 `NA`로 바뀌었습니다. 이런 경우에는 나이를 `"text"`로 읽습니다. R에 불러온 뒤 값을 고칩니다.

```r
students <- read_excel(
  "data/students.xlsx",
  col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
  skip = 1,
  na = c("", "N/A"),
  col_types = c("numeric", "text", "text", "text", "text")
)

students <- students |>
  mutate(
    age = if_else(age == "five", "5", age),
    age = parse_number(age)
  )

students
```

원하는 형식으로 데이터를 정확히 불러오기까지 여러 단계와 시행착오(trial-and-error)를 거쳤습니다. 특별한 일은 아닙니다. 데이터 과학은 본래 반복적인(iterative) 과정(process)입니다. 더구나 스프레드시트는 데이터 저장뿐 아니라 공유와 의사소통에도 쓰입니다. 사람이 직접 입력하고 다루는 만큼, 스프레드시트 데이터를 읽는 작업은 일반 텍스트로 된 사각형(rectangular) 데이터 파일보다 훨씬 번거로울(tedious) 수 있습니다.

데이터를 불러와 살펴보기 전에는 정확한 모양을 알기 어렵습니다. 물론 Excel에서 파일을 직접 열어볼(take a peek) 수도 있습니다. 그럴 때는 원본 데이터 파일을 그대로 보존하고 Excel 파일의 복사본을 만드세요. 복사본은 자유롭게 열어 탐색하되, R에서는 손대지 않은 원본을 읽는 편이 좋습니다.

그러면 스프레드시트를 살피다가 원본을 실수로 덮어쓰는(overwrite) 일을 막습니다. 데이터를 불러와 확인한 다음 코드를 고쳐 다시 불러옵니다. 결과가 마음에 들 때까지 이 과정을 되풀이해도 괜찮습니다.

### 워크시트 읽기

스프레드시트가 플랫 파일과 다른 중요한 특징(feature)은 워크시트라고 부르는 여러 시트를 담을 수 있다는 점입니다. `read_excel()`의 `sheet` 인수로 원하는 워크시트 하나를 읽습니다. 지금까지는 기본값인 첫 번째 시트를 읽었습니다.

```r
read_excel("data/penguins.xlsx", sheet = "Torgersen Island")
```

숫자형으로 보이는 일부 변수는 `"NA"` 문자열이 실제 `NA`로 처리되지 않아 문자형으로 읽혔습니다.

```r
penguins_torgersen <- read_excel("data/penguins.xlsx", sheet = "Torgersen Island", na = "NA")

penguins_torgersen
```

다른 방법도 있습니다. `excel_sheets()`로 Excel 스프레드시트의 워크시트 정보를 모두 확인한 뒤 필요한 시트를 읽습니다.

```r
excel_sheets("data/penguins.xlsx")
```

워크시트 이름을 확인했다면 `read_excel()`로 하나씩 읽을 수 있습니다.

```r
penguins_biscoe <- read_excel("data/penguins.xlsx", sheet = "Biscoe Island", na = "NA")
penguins_dream  <- read_excel("data/penguins.xlsx", sheet = "Dream Island", na = "NA")
```

이 펭귄 데이터셋은 스프레드시트의 워크시트 3개에 나뉘어 있습니다.
각 워크시트에는 동일한 수의 열이 있지만 행 수는 다릅니다.

```r
dim(penguins_torgersen)
dim(penguins_biscoe)
dim(penguins_dream)
```

`bind_rows()`로 세 워크시트를 하나로 합칩니다.

```r
penguins <- bind_rows(penguins_torgersen, penguins_biscoe, penguins_dream)
penguins
```

### 시트의 일부 읽기

Excel 스프레드시트는 데이터 저장뿐 아니라 발표 자료로도 자주 쓰입니다. 그래서 R로 읽으려는 데이터와 무관한 셀이 섞인 경우가 흔합니다. 시트 가운데에는 데이터 프레임 모양의 표가 있습니다. 위아래 셀에는 불필요한(extraneous) 텍스트가 들어 있기도 합니다.

이 스프레드시트는 readxl 패키지가 제공하는 예제입니다.
`readxl_example()` 함수로 패키지 설치 디렉터리에 있는 파일의 위치를 찾을(locate) 수 있습니다.

이 함수가 돌려주는 경로는 평소처럼 `read_excel()`에 넣으면 됩니다.

```r
deaths_path <- readxl_example("deaths.xlsx")
deaths <- read_excel(deaths_path)
deaths
```

위쪽 3개 행과 아래쪽 4개 행은 데이터 프레임에 속하지 않습니다. `skip`과 `n_max` 인수로 불필요한(extraneous) 행을 뺄 수도 있습니다. 하지만 여기서는 셀 범위를 지정하는 편이 낫습니다.
Excel에서 왼쪽 상단 셀은 `A1`입니다.

열을 가로질러 오른쪽으로 이동하면 셀 레이블이 알파벳순으로 이동합니다(`B1`, `C1` 등). 그리고 열 아래로 이동하면 셀 레이블의 숫자가 증가합니다(`A2`, `A3` 등).

읽으려는 데이터는 `A5`에서 시작해 `F15`에서 끝납니다. 스프레드시트 표기법(notation)으로는 `A5:F15`입니다. 이 값을 `range` 인수에 넣습니다.

```r
read_excel(deaths_path, range = "A5:F15")
```

### 데이터 유형

CSV 파일의 값은 모두 문자열(strings)입니다. 데이터의 성격을 충실히(true) 담지는 못하지만 구조는 단순합니다.

Excel 스프레드시트의 내부(underlying) 데이터는 더 복잡합니다. 셀은 다음 네 유형 가운데 하나입니다.

- `TRUE`, `FALSE` 또는 `NA`와 같은 부울(boolean).
- "10" 또는 "10.5"와 같은 숫자(number).
- "11/1/21" 또는 "11/1/21 3:00 PM"처럼 시간을 포함하기도 하는 날짜-시간(datetime).
- "ten"과 같은 텍스트 문자열(text string).

스프레드시트에서는 내부 데이터와 셀에 보이는 값이 크게 다를 수 있다는 점을 기억해야 합니다. 예를 들어 Excel에는 정수(integer)라는 유형이 없습니다.

모든 숫자는 부동 소수점(floating points)으로 저장됩니다. 표시할 소수점 이하 자릿수(decimal points)는 사용자가 정합니다.

날짜도 실제로는 숫자, 구체적으로 1900년 1월 1일 이후의 일수로 저장됩니다. Excel 서식으로 날짜 표시 방식을 바꿉니다.

혼란스럽게도(Confusingly), 숫자처럼 보이지만 실제로는 문자열인 것도 가능합니다(Excel 셀에 `'10`을 입력).

이처럼 저장 방식과 표시 방식이 달라서 데이터를 R로 불러올 때 예상 밖의 결과가 나올 수 있습니다.

기본적으로 readxl은 주어진 열의 데이터 유형을 추측합니다.

먼저 readxl이 열 유형을 추정하게 두고 결과가 적절한지 확인하세요. 맞지 않으면 `col_types`를 지정해 다시 불러오는(re-import) 흐름을 권합니다.

또 다른 과제는 Excel 스프레드시트에 이러한 유형이 혼합된 열이 있는 경우입니다(일부 셀은 숫자, 일부는 텍스트, 일부는 날짜).

데이터를 R로 가져올 때 readxl은 몇 가지 결정을 내려야 합니다.

이때 열 유형을 `"list"`로 정하면 길이 1인 벡터의 목록으로 불러옵니다. 각 벡터 요소의 유형은 따로 추정됩니다.

### 엑셀에 쓰기

파일로 내보낼(write out) 작은 데이터 프레임을 만들어 봅시다.
`item`은 요인(factor)이고 `quantity`는 정수입니다.

```r
bake_sale <- tibble(
  item     = factor(c("brownie", "cupcake", "cookie")),
  quantity = c(10, 5, 8)
)

bake_sale
```

[writexl 패키지]([https://docs.ropensci.org/writexl/](https://docs.ropensci.org/writexl/))의 `write_xlsx()` 함수는 데이터를 Excel 파일에 써서 디스크에 저장합니다.

```r
#| eval: false

write_xlsx(bake_sale, path = "data/bake-sale.xlsx")
```

열 이름은 파일에 포함되며 굵게 표시됩니다. 이 서식은 `col_names`와 `format_headers` 인수를 `FALSE`로 정하면 끌(turned off) 수 있습니다.

CSV와 마찬가지로 Excel도 데이터를 다시 읽으면 유형 정보가 사라집니다. 따라서 Excel 파일은 중간(interim) 결과를 캐싱(caching)하는 용도로도 믿기 어렵습니다(unreliable).

```r
read_excel("data/bake-sale.xlsx")
```

### 형식이 지정된 출력

writexl 패키지는 간단한 Excel 스프레드시트를 만들기에 알맞습니다. 기존 스프레드시트의 시트에 쓰거나 스타일을 지정(styling)하는 기능까지 필요하다면 [openxlsx 패키지]([https://ycphs.github.io/openxlsx](https://ycphs.github.io/openxlsx))가 좋습니다.

여기서는 이 패키지의 자세한 사용법까지 다루지 않습니다. openxlsx로 R 데이터를 Excel에 쓸 때 적용하는 여러 서식은 [https://ycphs.github.io/openxlsx/articles/Formatting.html](https://ycphs.github.io/openxlsx/articles/Formatting.html)에 잘 정리되어 있습니다.

이 패키지는 tidyverse에 속하지 않아 함수와 작업 흐름이 낯설게(unfamiliar) 느껴질 수 있습니다.
예를 들어 함수 이름은 카멜 케이스(camelCase)입니다. 여러 함수를 파이프라인으로 구성(composed)하지 못하고 인수 순서도 tidyverse의 관례와 다릅니다.

하지만 괜찮습니다.

이 책을 넘어 R을 더 폭넓게 배우고 쓰다 보면, 목적에 따라 여러 패키지와 저마다의 스타일을 만나게(encounter) 됩니다.

새 패키지의 코딩 스타일이 낯설다면 함수 설명서의 예제를 직접 실행해 보세요. 구문과 출력 형식을 익힌 뒤 패키지가 제공하는 비네트(vignettes)를 읽으면 한결 수월합니다.

### 연습 문제

1. Excel 파일에서 다음 데이터셋을 만들고 `survey.xlsx`로 저장하세요. 또는 [https://docs.google.com/spreadsheets/d/1yc5gL-a2OOBr8M7B3IsDNX5uR17vBHOyWZq6xSTG2G8](https://docs.google.com/spreadsheets/d/1yc5gL-a2OOBr8M7B3IsDNX5uR17vBHOyWZq6xSTG2G8)에서 Excel 파일로 내려받으세요. 그런 다음 `survey_id`는 문자형, `n_pets`는 숫자형 변수로 지정해 R로 읽습니다.

```r
#| echo: false
read_excel("data/survey.xlsx", na = c("", "N/A"), col_types = c("text", "text")) |>
  mutate(
    n_pets = case_when(
      n_pets == "none" ~ "0",
      n_pets == "two"  ~ "2",
      TRUE             ~ n_pets
    ),
    n_pets = as.numeric(n_pets)
  )
```

2.  다른 Excel 파일에서 다음 데이터셋을 만들고 `roster.xlsx`로 저장하세요. 또는 [https://docs.google.com/spreadsheets/d/1LgZ0Bkg9d_NK8uTdP2uHXm07kAlwx8-Ictf8NocebIE](https://docs.google.com/spreadsheets/d/1LgZ0Bkg9d_NK8uTdP2uHXm07kAlwx8-Ictf8NocebIE)에서 Excel 파일로 내려받으세요. 그런 다음 R로 읽습니다. 결과 데이터 프레임의 이름은 `roster`이며 아래 모양이어야 합니다.

```r
#| echo: false
#| message: false
read_excel("data/roster.xlsx") |>
  fill(group, subgroup) |>
  print(n = 12)
```

3.  새 Excel 파일에서 다음 데이터셋을 만들고 `sales.xlsx`로 저장하세요. 또는 [https://docs.google.com/spreadsheets/d/1oCqdXUNO8JR3Pca8fHfiz_WXWxMuZAp3YiYFaKze5V0](https://docs.google.com/spreadsheets/d/1oCqdXUNO8JR3Pca8fHfiz_WXWxMuZAp3YiYFaKze5V0)에서 Excel 파일로 내려받으세요.
  a. `sales.xlsx`를 읽어 들이고 `sales`로 저장합니다. 데이터 프레임은 다음과 같아야 하며 `id`와 `n`이 열 이름이고 9개의 행이 있어야 합니다.

  ```r
  #| echo: false
  #| message: false
  read_excel("data/sales.xlsx", skip = 3, col_names = c("id", "n")) |>
    print(n = 9)
  ```

  b. `sales`를 추가로(further) 수정(Modify)하여 3개의 열(`brand`, `id`, `n`)과 7개의 데이터 행이 있는 다음의 깔끔한(tidy) 형식으로 만드세요. `id`와 `n`은 숫자형이고 `brand`는 문자형 변수라는 점에 유의하세요.

  ```r
  #| echo: false
  #| message: false
  read_excel("data/sales.xlsx", skip = 3, col_names = c("id", "n")) |>
    mutate(brand = if_else(str_detect(id, "Brand"), id, NA)) |>
    fill(brand) |>
    filter(n != "n") |>
    relocate(brand) |>
    mutate(
      id = as.numeric(id),
      n = as.numeric(n)
    ) |>
    print(n = 7)
  ```

4.`bake_sale` 데이터 프레임을 다시 만들고 openxlsx 패키지의 `write.xlsx()` 함수를 사용하여 Excel 파일로 출력하세요.

5. 열 이름을 스네이크 케이스(snake case)로 바꾸는 `janitor::clean_names()` 함수를 배웠습니다. 이 절 앞부분의 `students.xlsx` 파일을 읽고 이 함수로 열 이름을 "정리(clean)"하세요.

6. `.xlsx` 확장자(extension)를 가진 파일을 `read_xls()`로 읽으려고 하면 어떻게 됩니까?

## 구글 스프레드시트

Google 스프레드시트도 널리 쓰이는 스프레드시트 프로그램입니다. 무료로 쓸 수 있으며 웹에서 작동합니다.
Excel과 마찬가지로 데이터는 파일 안의 워크시트(또는 시트)에 정리됩니다.

```r
library(googlesheets4)
library(tidyverse)
```

패키지 이름에 붙은 4는 [Sheets API v4]([https://developers.google.com/sheets/api/](https://developers.google.com/sheets/api/))를 뜻합니다. googlesheets4는 이 API를 이용한 Google 스프레드시트용 R 인터페이스입니다.

googlesheets4의 주요 함수는 URL이나 파일 ID로 Google 스프레드시트를 읽는 `read_sheet()`입니다. `range_read()`라는 이름으로도 씁니다. `gs4_create()`는 새 시트를 만듭니다. `sheet_write()` 계열 함수는 기존 시트에 데이터를 씁니다.

Excel과 Google 스프레드시트의 작동 방식이 완전히(exactly) 같지는 않습니다. 작업에 따라 함수 호출을 조금 더 조정해야 할 수도 있습니다.

### 구글 스프레드시트 읽기

`read_sheet()`의 첫 번째 인수에는 읽을 파일의 URL을 넣습니다. 결과는 티블(tibble)로 반환됩니다. 예를 들어 [https://docs.google.com/spreadsheets/d/1V1nPp1tzOuutXFLb3G9Eyxi3qxeEhnOXUzL5_BcCQ0w](https://docs.google.com/spreadsheets/d/1V1nPp1tzOuutXFLb3G9Eyxi3qxeEhnOXUzL5_BcCQ0w)처럼 URL이 긴 경우가 많아서 보통은 ID로 시트를 지정합니다.

```r
students_sheet_id <- "1V1nPp1tzOuutXFLb3G9Eyxi3qxeEhnOXUzL5_BcCQ0w"
students <- read_sheet(students_sheet_id)
students
```

`read_excel()`과 마찬가지로 `read_sheet()`에도 열 이름, NA 문자열, 열 유형을 지정합니다.

```r
students <- read_sheet(
  students_sheet_id,
  col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
  skip = 1,
  na = c("", "N/A"),
  col_types = "dcccc"
)

students
```

여기서는 짧은 코드(short codes)로 열 유형을 지정했습니다. 예를 들어 "dcccc"는 "double, character, character, character, character"를 뜻합니다.

Google 스프레드시트에서 개별 시트를 읽는 것도 가능합니다. [펭귄 구글 스프레드시트(penguins Google Sheet)](https://pos.it/r4ds-penguins)에서 "Torgersen Island" 시트를 읽어봅시다.

```r
penguins_sheet_id <- "1aFu8lnD_g0yjF5O-K6SFgSEWiHPpgvFCF0NY9D6LXnY"
read_sheet(penguins_sheet_id, sheet = "Torgersen Island")
```

`sheet_names()`는 Google 스프레드시트에 든 모든 시트의 목록을 돌려줍니다.

```r
sheet_names(penguins_sheet_id)
```

마지막으로 `read_excel()`처럼 `read_sheet()`에도 `range`를 정해 Google 스프레드시트의 일부만 읽을 수 있습니다. 아래에서는 `gs4_example()` 함수로 googlesheets4 패키지가 제공하는 예제 Google 스프레드시트의 위치를 찾습니다.

```r
deaths_url <- gs4_example("deaths")
deaths <- read_sheet(deaths_url, range = "A5:F15")
deaths
```

### 구글 스프레드시트에 쓰기

`write_sheet()`는 R의 데이터를 Google 스프레드시트에 씁니다. 첫 번째 인수에는 데이터 프레임을, 두 번째 인수에는 Google 스프레드시트의 이름이나 다른 식별자를 넣습니다.

```r
#| eval: false
write_sheet(bake_sale, ss = "bake-sale")
```

특정 워크시트에 쓰려면 `sheet` 인수로 시트를 지정합니다.

```r
#| eval: false
write_sheet(bake_sale, ss = "bake-sale", sheet = "Sales")
```

### 인증

공개 Google 스프레드시트는 Google 계정을 인증하지 않고도 읽을 수 있습니다. 이때 `gs4_deauth()`를 씁니다. 반면 비공개 시트를 읽거나 시트에 데이터를 쓰려면 googlesheets4가 Google 스프레드시트를 보고 관리하도록 인증해야 합니다.

인증이 필요한 시트를 읽으면 googlesheets4가 웹 브라우저를 엽니다. Google 계정에 로그인한 뒤, 패키지가 Google 스프레드시트를 대신 조작하도록 권한을 부여하라는 안내가 나옵니다.

특정 Google 계정이나 인증 범위를 지정하려면 `gs4_auth()`를 씁니다(`gs4_auth(email = "mine@example.com")`). 그러면 해당 이메일에 연결된 토큰을 사용합니다.

인증 방법은 googlesheets4의 인증 비네트(auth vignette)([https://googlesheets4.tidyverse.org/articles/auth.html](https://googlesheets4.tidyverse.org/articles/auth.html))에 자세히 나와 있습니다.

### 연습 문제

1. 이 장 앞부분의 `students` 데이터셋을 `read_excel()`과 `read_sheet()`에 추가 인수를 넣지 않고 Excel과 Google 스프레드시트에서 각각 읽으세요. R에서 얻은 데이터 프레임이 정확히(exactly) 같습니까? 다르다면 무엇이 다릅니까?

2. [https://pos.it/r4ds-survey](https://pos.it/r4ds-survey)에서 survey라는 제목의 Google 스프레드시트를 읽고 `survey_id`는 문자형 변수로, `n_pets`는 숫자형 변수로 설정합니다.

3. [https://pos.it/r4ds-roster](https://pos.it/r4ds-roster)에서 roster라는 제목의 Google 스프레드시트를 읽습니다. 결과 데이터 프레임의 이름은 `roster`여야 하며 다음과 같아야 합니다.

```r
#| echo: false
#| message: false
read_sheet("https://docs.google.com/spreadsheets/d/1LgZ0Bkg9d_NK8uTdP2uHXm07kAlwx8-Ictf8NocebIE/") |>
  fill(group, subgroup) |>
  print(n = 12)
```

## Arrow

CSV 파일은 사람이 읽기 쉽게 설계됐습니다. 구조가 단순하고 거의 모든 도구에서 읽을 수 있어 교환 형식으로 훌륭합니다. 다만 효율은 높지 않습니다. R로 데이터를 읽으려면 꽤 많은 작업(work)이 필요합니다.

이 장에서는 강력한 대안인 [parquet 형식]([https://parquet.apache.org/](https://parquet.apache.org/))을 배웁니다. 개방형 표준에 기반하며 빅데이터 시스템에서 널리 쓰이는 형식입니다.

parquet 파일은 [Apache Arrow](https://arrow.apache.org)와 함께 사용하겠습니다. Apache Arrow는 대규모 데이터셋을 효율적으로 분석하고 전송하도록 설계된 다국어 도구 모음입니다.

[arrow 패키지](https://arrow.apache.org/docs/r/)를 통해 Apache Arrow를 사용합니다. 이 패키지는 dplyr 백엔드를 제공하므로 익숙한 dplyr 구문으로 메모리보다 큰 데이터셋도 분석합니다.

더구나 arrow는 매우 빠릅니다. arrow와 dbplyr 모두 dplyr 백엔드를 제공하니 어느 때 무엇을 써야 할지 궁금할 것입니다.

데이터가 이미 데이터베이스나 parquet 파일에 있고 그대로(as is) 작업한다면 선택은 자연스럽게 정해집니다. 하지만 CSV 파일처럼 직접 가진 데이터에서 시작한다면 데이터베이스에 넣거나 parquet로 바꿀 수 있습니다. 어느 쪽이 더 나을지는 미리 알기 어렵습니다. 분석 초기에 둘 다 시험해 보고 잘 맞는 쪽을 고르는 편이 좋습니다.

```r
#| message: false
#| warning: false
library(tidyverse)
library(arrow)
library(dbplyr, warn.conflicts = FALSE)
library(duckdb)
```

### 데이터 가져오기

먼저 이 도구에 걸맞은 시애틀 공공 도서관의 자료 대출 데이터셋을 불러옵니다. [data.seattle.gov/Community/Checkouts-by-Title/tmmm-ytt6]([https://data.seattle.gov/Community/Checkouts-by-Title/tmmm-ytt6](https://data.seattle.gov/Community/Checkouts-by-Title/tmmm-ytt6))에서 온라인으로 볼 수 있습니다.

이 데이터셋에는 2005년 4월부터 2022년 10월까지 매월 각 도서가 얼마나 대출되었는지 알려주는 41,389,465개의 행이 포함되어 있습니다.

다음 코드는 캐시해 둔 데이터 복사본을 내려받습니다. 9GB짜리 CSV 파일이라 시간이 걸립니다. `curl::multi_download()`는 이런 대용량 파일에 알맞습니다. 진행률 표시줄(progress bar)이 나오고 도중에 끊겨도 이어받아서 적극 권합니다.

```r
#| eval: !expr "!file.exists('data/seattle-library-checkouts.csv')"
dir.create("data", showWarnings = FALSE)

curl::multi_download(
  "https://r4ds.s3.us-west-2.amazonaws.com/seattle-library-checkouts.csv",
  "data/seattle-library-checkouts.csv",
  resume = TRUE
)
```

### 데이터셋 열기

먼저 데이터를 살펴봅시다. 이 파일은 9GB라 메모리에 통째로 올리기에는 큽니다. 경험상 데이터 크기의 두 배가 넘는 메모리가 필요한데, 많은 노트북의 최대 메모리는 16GB입니다. 따라서 `read_csv()` 대신 `arrow::open_dataset()`을 쓰는 편이 좋습니다.

```r
seattle_csv <- open_dataset(
  sources = "data/seattle-library-checkouts.csv", 
  col_types = schema(ISBN = string()),
  format = "csv"
)
```

`open_dataset()`은 처음 몇 천 행을 훑어 데이터셋 구조를 파악합니다. `ISBN` 열은 처음 80,000개 행이 비어 있으므로 arrow가 구조를 올바르게 파악하도록 열 유형을 지정해야 합니다. 스캔이 끝나면 알아낸 내용을 기록하고 멈춥니다. 사용자가 명시적으로 요청해야 다음 행을 읽습니다. `seattle_csv`를 출력하면 이 메타데이터가 보입니다.

```r
seattle_csv
```

출력 첫 줄을 보면 `seattle_csv`가 로컬 디스크의 CSV 파일 하나에 저장되어 있습니다. 필요할 때만 메모리에 올라온다는 사실도 알 수 있습니다. 나머지 부분에는 arrow가 각 열에 추정(imputed)한 유형이 나옵니다.

`glimpse()`로 실제 내용을 들여다볼 수 있습니다. 약 4,100만 행과 12개 열이 있다는 정보와 몇 가지 값이 표시됩니다.

```r
#| cache: true
seattle_csv |> glimpse()
```

이제 dplyr 동사로 데이터셋을 다룰 수 있습니다. `collect()`를 호출하면 arrow가 연산을 실행하고 일부 데이터를 반환합니다. 다음 코드는 연도별 총 대출 횟수를 구합니다.

```r
#| cache: true
seattle_csv |> 
  group_by(CheckoutYear) |> 
  summarise(Checkouts = sum(Checkouts)) |> 
  arrange(CheckoutYear) |> 
  collect()
```

arrow를 쓰면 원본 데이터셋의 크기와 관계없이 이 코드가 작동합니다. 다만 지금은 조금 느립니다. 해들리의 컴퓨터에서는 약 10초가 걸렸습니다. 데이터 양을 생각하면 심한 수준은 아닙니다. 그래도 형식을 바꾸면 훨씬 빨라집니다.

### parquet 형식

데이터를 더 쉽게 다루도록 parquet 형식으로 바꾸고 여러 파일로 나눠 봅시다. 먼저 parquet와 파티셔닝을 알아본 뒤 시애틀 도서관 데이터에 적용합니다.

#### parquet의 장점

parquet도 CSV처럼 사각형 데이터에 쓰입니다. 다만 텍스트가 아니라 빅데이터에 맞춰 설계한 바이너리 형식입니다. 여기에는 몇 가지 중요한 의미가 있습니다.

- parquet 파일은 보통 같은 데이터를 담은 CSV 파일보다 작습니다. [효율적인 인코딩(efficient encodings)]([https://parquet.apache.org/docs/file-format/data-pages/encodings/](https://parquet.apache.org/docs/file-format/data-pages/encodings/))을 사용하고 압축도 지원합니다. 디스크에서 메모리로 옮길 데이터가 적어 읽는 속도도 빨라집니다.

- parquet 파일에는 풍부한 유형 체계가 있습니다. CSV 파일은 열 유형 정보를 주지 않으므로, CSV 리더는 `"08-10-2022"`를 문자열로 읽을지 날짜로 읽을지 추정해야 합니다. 반면 parquet 파일은 데이터와 함께 유형도 기록합니다.

- parquet 파일은 "열 지향적"입니다. R의 데이터 프레임처럼 열 단위로 구성됩니다. 보통 행 단위로 구성된 CSV보다 데이터 분석 성능이 좋습니다.

- parquet 파일은 여러 청크로 나뉩니다(chunked). 파일의 서로 다른 부분을 동시에 처리합니다. 조건이 맞으면 일부 청크는 아예 읽지 않아도 됩니다.

주된 단점은 사람이 직접 읽을 수 없다는 점입니다. `readr::read_file()`로 parquet 파일을 열면 알아볼 수 없는 문자만 보입니다.

#### 파티셔닝

데이터셋이 커질수록 모든 데이터를 파일 하나에 저장하기가 버거워집니다. 이럴 때는 큰 데이터셋을 여러 파일로 나누는(split) 편이 유용합니다.

파일을 영리하게 나누면 성능도 크게 좋아집니다. 많은 분석이 전체 파일이 아니라 일부 파일만 필요로 하기 때문입니다.

데이터셋을 파티셔닝하는 절대적인 규칙은 없습니다. 데이터와 접근 패턴, 데이터를 읽는 시스템에 따라 결과가 달라집니다.
자신의 상황에 알맞은 구성을 찾으려면 몇 차례 실험해야 할 가능성이 큽니다.

대략적인 기준으로 arrow는 20MB보다 작거나 2GB보다 큰 파일을 피하라고 권합니다. 파일을 10,000개 넘게 만드는 파티션도 피하는 편이 좋습니다.

가능하면 필터링 기준으로 자주 쓰는 변수에 따라 파티셔닝하세요. 잠시 뒤 확인하듯, 그러면 arrow가 관련 파일만 읽고 불필요한 작업을 건너뜁니다.

#### 시애틀 도서관 데이터 다시 쓰기

이제 시애틀 도서관 데이터에 적용해 실제로 어떻게 작동하는지(play out) 살펴봅시다.
분석에 따라 최근 데이터만 보고 싶을 때가 많습니다. 연도별로 나누면 적당한(reasonable) 크기의 청크(chunks) 18개가 나오므로 `CheckoutYear`를 기준으로 파티셔닝하겠습니다.

데이터를 다시 쓸(rewrite) 때는 `dplyr::group_by()`로 파티션을 정하고 `arrow::write_dataset()`으로 디렉터리에 저장합니다. `write_dataset()`의 핵심 인수는 파일을 만들 디렉터리와 저장 형식입니다.

```r
pq_path <- "data/seattle-library-checkouts"
```

```r
#| eval: !expr "!file.exists(pq_path)"
seattle_csv |>
  group_by(CheckoutYear) |>
  write_dataset(path = pq_path, format = "parquet")
```

실행에는 1분 정도 걸립니다. 이후 작업 시간을 크게 줄여 주는 초기 투자입니다.

방금 생성한 항목을 살펴보겠습니다.

```r
tibble(
  files = list.files(pq_path, recursive = TRUE),
  size_MB = file.size(file.path(pq_path, files)) / 1024^2
)
```

9GB짜리 CSV 파일 하나가 parquet 파일 18개로 바뀌었습니다. 파일 이름에는 [Apache Hive]([https://hive.apache.org](https://hive.apache.org)) 프로젝트의 "자체 설명적" 명명 규칙을 따릅니다.

Hive 스타일 파티션은 "key=value" 규칙으로 폴더 이름을 지정하므로 짐작하셨겠지만 `CheckoutYear=2005` 디렉터리에는 `CheckoutYear`가 2005인 모든 데이터가 포함되어 있습니다.

각 파일은 100~300MB이고 전체 크기는 약 4GB입니다. 원본 CSV의 절반을 조금 넘는 수준입니다. parquet이 훨씬 효율적인 형식이므로 예상한 결과입니다.

### arrow와 함께 dplyr 사용하기

parquet 파일을 만들었으니 다시 읽어봅시다. 이번에는 `open_dataset()`에 디렉터리를 지정합니다.

```r
seattle_pq <- open_dataset(pq_path)
```

이제 dplyr 파이프라인을 작성합니다. 다음 예에서는 지난 5년간 매달 대출된(checked out) 도서의 총수를 셉니다.

```r
query <- seattle_pq |> 
  filter(CheckoutYear >= 2018, MaterialType == "BOOK") |>
  group_by(CheckoutYear, CheckoutMonth) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(CheckoutYear, CheckoutMonth)
```

arrow 데이터에 dplyr 코드를 쓰는 방식은 개념적으로 dbplyr와 비슷합니다. 작성한 코드는 Apache Arrow C++ 라이브러리가 이해하는 쿼리로 자동 변환됩니다. `collect()`를 호출하면 실행됩니다.

`query` 객체를 출력하면 쿼리가 실행될 때 Arrow가 반환할 결과의 정보를 간단히 볼 수 있습니다.

```r
query
```
```

이어서 `collect()`를 호출해 결과를 받습니다.

```r
query |> collect()
```

dbplyr와 마찬가지로 arrow가 이해하는 R 표현식에는 제한이 있습니다. 평소 쓰던 코드를 그대로 작성하지 못할 수도 있습니다.

그래도 지원하는 연산과 함수가 상당히 많습니다(fairly extensive). 그 목록도 계속 늘고 있습니다. 현재 지원하는 함수는 `?acero`에 모두 나옵니다.

#### 성능

CSV를 parquet로 바꿨을 때 성능이 얼마나 달라지는지 간단히 살펴봅시다. 먼저 대용량 CSV 파일 하나에서 2021년 월별 도서 대출 건수를 계산하는 시간을 잽니다.

```r
#| cache: true

seattle_csv |> 
  filter(CheckoutYear == 2021, MaterialType == "BOOK") |>
  group_by(CheckoutMonth) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(desc(CheckoutMonth)) |>
  collect() |> 
  system.time()
```

이번에는 시애틀 도서관 대출 데이터를 작은 parquet 파일 18개로 나눈 버전을 사용합니다.

```r
#| cache: true

seattle_pq |> 
  filter(CheckoutYear == 2021, MaterialType == "BOOK") |>
  group_by(CheckoutMonth) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(desc(CheckoutMonth)) |>
  collect() |> 
  system.time()
```

약 100배 빨라진(speedup) 데에는 여러 파일로 나눈 방식과 개별 파일 형식, 두 가지 요인이 작용했습니다.

1. 파티셔닝은 성능을 높입니다. 이 쿼리는 `CheckoutYear == 2021`로 데이터를 거릅니다. arrow는 parquet 파일 18개 가운데 하나만 읽으면 된다는 사실을 알아챕니다.

2. parquet은 데이터를 메모리로 더 직접 읽을 수 있는 바이너리(binary) 형식이라 빠릅니다. 열 단위(column-wise) 구조와 풍부한 메타데이터(metadata) 덕분에 arrow는 쿼리에 쓰인 네 열(`CheckoutYear`, `MaterialType`, `CheckoutMonth`, `Checkouts`)만 읽습니다.

이처럼 큰 성능 차이 때문에 대용량 CSV를 parquet로 바꾸는 수고가 보상을 받습니다(pays off)!

#### arrow와 함께 duckdb 사용하기

parquet과 arrow에는 장점이 하나 더 있습니다. `arrow::to_duckdb()`만 호출하면 arrow 데이터셋을 DuckDB 데이터베이스로 손쉽게 바꿀 수 있습니다.

```r
seattle_pq |> 
  to_duckdb() |>
  filter(CheckoutYear >= 2018, MaterialType == "BOOK") |>
  group_by(CheckoutYear) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(desc(CheckoutYear)) |>
  collect()
```

`to_duckdb()`는 메모리를 복사하지 않습니다. 한 컴퓨팅 환경에서 다른 환경으로 매끄럽게 옮겨 간다는 arrow 생태계(ecosystem)의 목표가 잘 드러납니다.

### 연습 문제

1. 매년 가장 인기 있는 책은 무엇입니까?
2. 시애틀 도서관 시스템에서 가장 많은 책을 보유한 저자는 누구입니까?
3. 종이책과 전자책의 대출은 지난 10년 동안 어떻게 변했습니까?

## 데이터베이스

엄청난 양의(huge amount of) 데이터가 데이터베이스에 들어 있습니다. 따라서 데이터베이스에 접근하는 법은 꼭 알아야 합니다. 다른 사람에게 스냅샷(snapshot)을 `.csv`로 내려 달라고 부탁할 수도 있습니다. 하지만 수정할 때마다 다시 요청해야 해서 금세 번거로워집니다.

필요한 데이터를 제때 얻으려면 데이터베이스에 직접 접근할 줄 알아야 합니다.

먼저 DBI 패키지의 기본을 익힙니다. DBI로 데이터베이스에 연결하고 SQL 쿼리로 데이터를 검색해 보겠습니다.

구조화된 질의어(Structured Query Language)의 줄임말인 SQL은 데이터베이스의 공통어이며 모든 데이터 과학자가 배워야 할 중요한 언어입니다.

그렇다고 처음부터 SQL을 쓰지는 않습니다. dplyr 코드를 SQL로 바꾸는 dbplyr로 시작하겠습니다.

dbplyr를 발판 삼아 SQL의 주요 기능을 배웁니다. 이 장을 마친다고 SQL 전문가가 되지는 않습니다. 그래도 중요한 구성 요소(components)를 알아보고 각각의 역할을 이해하게 될 것입니다.

```r
#| label: setup
#| message: false
library(DBI)
library(dbplyr)
library(tidyverse)
```

간단히 말해 데이터베이스는 데이터 프레임의 모음(collection)입니다. 데이터베이스에서는 이 데이터 프레임을 테이블(tables)이라고 부릅니다.

데이터 프레임과 마찬가지로 데이터베이스 테이블도 이름이 붙은 열로 구성됩니다. 한 열의 값은 모두 같은 유형(type)입니다.

데이터 프레임과 데이터베이스 테이블 간에는 세 가지 높은 수준의(high level) 차이점이 있습니다.

1. 데이터베이스 테이블은 디스크에 저장하므로 크기에 사실상 제한이 없습니다. 데이터 프레임은 메모리에 저장되어 근본적인(fundamentally) 한계가 있습니다. 물론 많은 문제를 다루기에는 그 한계도 충분히 큽니다.

2. 데이터베이스 테이블에는 거의 언제나 인덱스(indexes)가 있습니다. 책의 찾아보기와 비슷해서 모든 행을 훑지 않고도 원하는 행을 빠르게 찾습니다. 데이터 프레임과 티블(tibbles)에는 인덱스가 없지만 data.table에는 있습니다. data.table이 빠른 이유 중 하나입니다.

3. 전통적인(classical) 데이터베이스는 기존 데이터 분석보다 빠른 데이터 수집(collecting)에 최적화되어 있습니다. R처럼 열 단위(column-by-column)가 아니라 행 단위(row-by-row)로 저장하므로 행 지향형(row-oriented)이라고 합니다. 최근에는 기존 데이터를 훨씬 빠르게 분석하는 열 지향형(column-oriented) 데이터베이스도 활발히 개발되고 있습니다.

데이터베이스는 데이터베이스 관리 시스템(DBMS)이 운영합니다. 크게 세 가지 형태로 나뉩니다.

1. 클라이언트-서버(Client-server) DBMS는 강력한 중앙 서버에서 실행됩니다. 사용자는 자신의 컴퓨터(클라이언트)로 접속합니다. 조직 안의 여러 사람이 데이터를 공유하기 좋습니다. PostgreSQL, MariaDB, SQL Server, Oracle이 대표적입니다.


2. Snowflake, Amazon의 RedShift, Google의 BigQuery 같은 클라우드(Cloud) DBMS는 클라이언트-서버 DBMS와 비슷하지만 클라우드에서 실행됩니다. 매우 큰 데이터셋도 쉽게 처리합니다. 필요하면 컴퓨팅 자원을 자동으로 늘립니다.

3. SQLite나 duckdb 같은 인-프로세스(In-process) DBMS는 사용자 컴퓨터에서만 실행됩니다. 주로 혼자 쓰는 큰 데이터셋을 다룰 때 유용합니다.

### 데이터베이스에 연결하기

R에서 데이터베이스에 연결할 때는 다음 두 종류의 패키지를 함께 씁니다.

1. DBI(database interface)는 항상 사용합니다. 데이터베이스 연결, 데이터 업로드, SQL 쿼리 실행 등에 필요한 공통 함수가 들어 있습니다.

2. 연결할 DBMS에 맞춘 패키지도 필요합니다. 이 패키지가 DBI의 공통 명령을 해당 DBMS가 요구하는 구체적인 명령으로 바꿉니다. 대개 DBMS마다 패키지가 하나씩 있습니다. PostgreSQL에는 RPostgres, MySQL에는 RMariaDB를 씁니다.

DBMS 전용 패키지가 없다면 보통 odbc 패키지로 대신합니다.
odbc는 여러 DBMS가 지원하는 ODBC 프로토콜을 사용합니다. ODBC 드라이버를 설치하고 그 위치를 odbc 패키지에 알려야 해서 설정(setup)이 조금 더 필요합니다.

데이터베이스 연결은 `DBI::dbConnect()`로 만듭니다.
첫 번째 인수로 DBMS를 고릅니다. 그다음 인수에는 연결 방법을 적습니다. DBMS의 위치와 접속에 필요한 자격 증명(credentials) 등이 여기에 들어갑니다.
다음은 흔히 볼 수 있는 예시 두 가지입니다.

```r
#| eval: false
con <- DBI::dbConnect(
  RMariaDB::MariaDB(), 
  username = "foo"
)
con <- DBI::dbConnect(
  RPostgres::Postgres(), 
  hostname = "databases.mycompany.com", 
  port = 1234
)
```

연결에 필요한 세부 정보는 DBMS마다 크게 달라 여기서 모두 다룰 수는 없습니다. 어느 정도는 직접 찾아봐야 합니다.

팀의 다른 데이터 과학자나 DBA(database administrator, 데이터베이스 관리자)에게 물어보는 것이 보통 가장 빠릅니다. 초기 설정에는 약간의 시행착오(fiddling)와 검색이 필요합니다. 다만 대개 한 번만 마치면 됩니다.

클라이언트-서버나 클라우드 DBMS 설정은 이 책에서 다루기 번거롭습니다. 대신 R 패키지 안에서 모두 작동하는 인-프로세스 DBMS인 duckdb를 사용하겠습니다.

DBI 덕분에 duckdb와 다른 DBMS의 차이는 연결 방법뿐입니다. 코드를 쉽게 실행합니다. 여기서 배운 내용도 다른 DBMS에 옮겨 적용하기 좋아 교육용으로 알맞습니다.

duckdb 연결은 특히 간단합니다. 기본 설정에서는 R을 종료할 때 지워지는 임시(temporary) 데이터베이스를 만듭니다. R을 다시 시작할 때마다 빈 상태(clean slate)에서 출발하므로 학습하기 좋습니다.

```r
con <- DBI::dbConnect(duckdb::duckdb())
```

duckdb는 데이터 과학자의 필요에 맞춰 설계된 고성능(high-performance) 데이터베이스입니다. 시작하기 쉽고 기가바이트 단위의 데이터도 매우 빠르게 처리해서 여기서 사용합니다.

실제 데이터 분석 프로젝트에서는 영구적인(persistent) 데이터베이스를 만들어야 합니다. `dbdir` 인수에 duckdb를 저장할 위치를 지정하세요.
프로젝트를 쓴다면 현재 프로젝트의 `duckdb` 디렉터리가 적당합니다.

```r
#| eval: false
con <- DBI::dbConnect(duckdb::duckdb(), dbdir = "duckdb")
```

#### 일부 데이터 로드하기

새 데이터베이스이므로 먼저 데이터를 넣어야 합니다. 여기서는 `DBI::dbWriteTable()`로 ggplot2의 `mpg`와 `diamonds` 데이터셋을 추가합니다. 기본적인 `dbWriteTable()` 호출에는 데이터베이스 연결, 만들 테이블의 이름, 데이터 프레임 형식의 데이터라는 세 인수가 필요합니다.

```r
dbWriteTable(con, "mpg", ggplot2::mpg)
dbWriteTable(con, "diamonds", ggplot2::diamonds)
```

실제 프로젝트에서 duckdb를 쓴다면 `duckdb_read_csv()`와 `duckdb_register_arrow()`도 꼭 살펴보세요. 데이터를 먼저 R에 올리지 않고 duckdb로 직접 빠르게 불러오는 강력한 방법입니다. 여러 파일을 데이터베이스에 넣는 유용한 기법도 담겨 있습니다.

#### DBI 기본

몇 가지 DBI 함수로 데이터가 제대로 들어갔는지 확인합니다. `dbListTables()`는 데이터베이스의 모든 테이블을 나열하고 `dbReadTable()`은 테이블의 내용을 읽습니다.

```r
dbListTables(con)

con |> 
  dbReadTable("diamonds") |> 
  as_tibble()
```

`dbReadTable()`은 `data.frame`을 반환합니다. 보기 좋게 출력하려고 `as_tibble()`로 티블로 바꿉니다.

SQL을 안다면 `dbGetQuery()`로 데이터베이스에서 쿼리를 실행하고 결과를 받을 수 있습니다.

```r
sql <- "
  SELECT carat, cut, clarity, color, price 
  FROM diamonds 
  WHERE price > 15000
"
as_tibble(dbGetQuery(con, sql))
```

SQL을 처음 보더라도 걱정하지 마세요. 곧 자세히 살펴봅니다. 코드를 찬찬히 읽으면 diamonds 데이터셋에서 다섯 열과 `price`가 15,000보다 큰 모든 행을 고른다는 뜻이 보입니다.

### dbplyr 기초

데이터베이스에 연결해 데이터를 넣었으니 이제 dbplyr를 살펴봅시다. dbplyr는 dplyr 백엔드(backend)입니다. 사용자는 평소처럼 dplyr 코드를 씁니다. 백엔드가 그 코드를 다른 방식으로 실행합니다.

dbplyr는 dplyr 코드를 SQL로 바꿉니다. 다른 백엔드로는 코드를 data.table([https://r-datatable.com](https://r-datatable.com))로 바꾸는 dtplyr([https://dtplyr.tidyverse.org](https://dtplyr.tidyverse.org))와 여러 코어에서 실행하는 multidplyr([https://multidplyr.tidyverse.org](https://multidplyr.tidyverse.org))가 있습니다.

dbplyr를 쓰려면 먼저 `tbl()`로 데이터베이스 테이블을 가리키는 객체(object)를 만듭니다.

```r
diamonds_db <- tbl(con, "diamonds")
diamonds_db
```

이 객체는 지연(lazy) 상태입니다. dplyr 동사(verbs)를 적용해도 바로 계산하지 않습니다. 실행할 작업의 순서만 기록해 두었다가 필요할 때 처리합니다. 다음 파이프라인(pipeline)을 봅시다.

```r
big_diamonds_db <- diamonds_db |> 
  filter(price > 15000) |> 
  select(carat:clarity, price)

big_diamonds_db
```

출력 맨 위에 DBMS 이름이 나오므로 이 객체가 데이터베이스 쿼리를 나타낸다는 사실을 알 수 있습니다. 열 수는 보여주지만 보통 행 수는 알지 못합니다. 전체 행 수를 알려면 완전한 쿼리를 실행해야 하는데, 되도록 피하려는 작업이기 때문입니다.

dplyr 함수 `show_query()`를 쓰면 생성된 SQL 코드를 볼 수 있습니다. dplyr를 안다면 SQL을 배우기에 아주 좋은 방법입니다. dplyr 코드를 작성해 dbplyr로 SQL로 바꾼 뒤, 두 언어가 어떻게 대응하는지 살펴보세요.

```r
big_diamonds_db |>
  show_query()
```

모든 데이터를 R로 가져오려면 `collect()`를 호출합니다. 내부에서는 SQL을 만들고 `dbGetQuery()`로 데이터를 받아 온 다음 결과를 티블로 바꿉니다.

```r
big_diamonds <- big_diamonds_db |> 
  collect()
big_diamonds
```

보통 dbplyr로 데이터베이스에서 필요한 데이터를 고르고 기본적인 필터링(filtering)과 집계(aggregation)를 처리합니다. R 고유의 함수로 분석할 단계가 되면 `collect()`로 데이터를 불러옵니다. 그러면 메모리 안의 티블을 얻어 순수한 R 코드로 작업을 이어갑니다.

### SQL

이 장의 나머지 부분에서는 dbplyr를 통해 SQL을 조금 배웁니다.
다소 색다른(non-traditional) 접근이지만 SQL의 기초를 빠르게 익히는 데 도움이 될 것입니다.

dplyr를 이해한다면 이미 좋은 출발점에 서 있습니다. 두 언어가 공유하는 개념이 많아 SQL을 빠르게 익힐 수 있습니다.

nycflights13 패키지에서 익숙한 `flights`와 `planes`를 이용해 dplyr과 SQL의 관계를 살펴보겠습니다.

dbplyr에는 nycflights13의 테이블을 학습용 데이터베이스로 복사하는 함수가 있어 두 데이터셋을 쉽게 가져올 수 있습니다.

```r
dbplyr::copy_nycflights13(con)
flights <- tbl(con, "flights")
planes <- tbl(con, "planes")
```

```r
#| echo: false
options(dplyr.strict_sql = TRUE)
```

#### SQL 기초

SQL의 최상위 구성 요소는 명령문(statements)입니다. 흔히 쓰는 명령문으로는 새 테이블을 정의하는 `CREATE`, 데이터를 추가하는 `INSERT`, 데이터를 검색하는 `SELECT`가 있습니다. 여기서는 데이터 과학자가 거의 전적으로 사용하는 `SELECT` 명령문에 집중합니다. `SELECT` 명령문은 쿼리(queries)라고도 합니다.

쿼리는 여러 절(clauses)로 이루어집니다. 중요한 절은 `SELECT`, `FROM`, `WHERE`, `ORDER BY`, `GROUP BY` 다섯 가지입니다. 모든 쿼리에는 `SELECT`와 `FROM`이 있어야 합니다. `SELECT * FROM table`은 지정한 테이블에서 모든 열을 고르는 간단한 쿼리입니다. dbplyr가 가공하지 않은 테이블에 생성하는 코드이기도 합니다.

문맥에 따라 `SELECT`가 명령문을 가리킬 때도 있고 절을 가리킬 때도 있어 혼동하기 쉽습니다. 이 책에서는 둘을 구분하려고 보통 `SELECT` 명령문 대신 `SELECT` 쿼리라는 표현을 씁니다.

엄밀히 말하면 `SELECT 1+1`처럼 기본 계산만 수행할 수도 있어 필수 절은 `SELECT`뿐입니다. 하지만 실제 데이터로 작업하려면 `FROM`도 필요합니다.

```r
flights |> show_query()
planes |> show_query()
```

`WHERE`는 포함할 행을, `ORDER BY`는 행의 정렬 순서를 정합니다.

```r
flights |> 
  filter(dest == "IAH") |> 
  arrange(dep_delay) |>
  show_query()
```

`GROUP BY`는 쿼리를 요약(summary) 형태로 바꾸어 집계(aggregation)가 일어나게 합니다.

```r
flights |> 
  group_by(dest) |> 
  summarize(dep_delay = mean(dep_delay, na.rm = TRUE)) |> 
  show_query()
```

dplyr 동사와 SELECT 절 사이에는 두 가지 중요한 차이점(differences)이 있습니다.

1. SQL은 대소문자를 구분하지 않습니다. `select`, `SELECT`, `SeLeCt`가 모두 같습니다. 이 책에서는 SQL 키워드를 테이블이나 변수 이름과 구별하려고 대문자로 쓰는 일반적인 관례를 따릅니다.

2. SQL에서는 절의 순서가 중요합니다. 항상 `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY` 순서로 써야 합니다. 실제 평가 순서는 이와 달리 `FROM`, `WHERE`, `GROUP BY`, `SELECT`, `ORDER BY`입니다.

##### SELECT

`SELECT` 절은 쿼리의 핵심입니다. `select()`, `mutate()`, `rename()`, `relocate()`에 해당하는 작업을 맡습니다. 다음 절에서 배울 `summarize()` 역할도 합니다.

`select()`, `rename()`, `relocate()`는 열의 표시 여부와 위치, 이름만 바꾸므로 `SELECT`로 곧바로 변환됩니다.

```r
planes |> 
  select(tailnum, type, manufacturer, model, year) |> 
  show_query()

planes |> 
  select(tailnum, type, manufacturer, model, year) |> 
  rename(year_built = year) |> 
  show_query()

planes |> 
  select(tailnum, type, manufacturer, model, year) |> 
  relocate(manufacturer, model, .before = type) |> 
  show_query()
```

이 예제에서는 SQL에서 이름을 바꾸는 방법도 확인합니다.
SQL에서는 이름 바꾸기를 별칭(aliasing)이라고 하며 `AS`를 사용합니다.
`mutate()`와 달리 기존 이름이 왼쪽, 새 이름이 오른쪽에 옵니다.

`mutate()`의 변환도 비슷하게 단순합니다. 각 변수는 `SELECT` 안의 새 표현식(expression)이 됩니다.

```r
flights |> 
  mutate(
    speed = distance / (air_time / 60)
  ) |> 
  show_query()
```

`/` 같은 개별(individual) 구성 요소의 변환은 @sec-sql-expressions에서 다시 설명하겠습니다.

##### FROM

`FROM` 절은 데이터 소스를 정의합니다.단일 테이블만 사용하기 때문에 당분간(for a little while)은 다소 흥미롭지 않을(uninteresting) 것입니다. 조인 함수에 도달(hit)하면 더 복잡한 예제를 볼 수 있습니다.

##### GROUP BY

`group_by()`는 `GROUP BY` 절로 변환되고 `summarize()`는 `SELECT` 절로 변환됩니다.

```r
diamonds_db |> 
  group_by(cut) |> 
  summarize(
    n = n(),
    avg_price = mean(price, na.rm = TRUE)
  ) |> 
  show_query()
```

`n()`과 `mean()`의 변환에 무슨 일이 일어나는지 다시 설명(come back to)하겠습니다.

##### WHERE

`filter()`는 `WHERE` 절로 변환됩니다.

```r
flights |> 
  filter(dest == "IAH" | dest == "HOU") |> 
  show_query()

flights |> 
  filter(arr_delay > 0 & arr_delay < 20) |> 
  show_query()
```

여기서 유의해야 할 몇 가지 중요한 세부 사항(details)이 있습니다.

1. `|`는 `OR`이 되고 `&`는 `AND`가 됩니다.
2. SQL은 값을 비교할 때 `==`가 아닌 `=`를 사용합니다. SQL에는 할당(assignment)이 없으므로 혼동할 여지가 없습니다.
3. SQL은 문자열(strings)에 `""`가 아닌 `''`만 사용합니다. SQL에서 `""`는 R의 backticks 처럼 변수를 식별(identify)하는 데 사용됩니다.

또 다른 유용한 SQL 연산자는 `IN`으로, R의 `%in%`과 매우 유사(close)합니다.

```r
flights |> 
  filter(dest %in% c("IAH", "HOU")) |> 
  show_query()
```

SQL은 `NA` 대신 `NULL`을 사용합니다. `NULL`은 `NA`와 비슷하게 동작합니다. 주된 차이는 비교와 산술(arithmetic)에서는 "전염성(infectious)"이 있지만 요약할 때는 조용히 빠진다는 점입니다. dbplyr는 이 동작을 처음 만나면 안내 메시지를 띄웁니다.

```r
flights |> 
  group_by(dest) |> 
  summarize(delay = mean(arr_delay))
```

`NULL`이 어떻게 작동(work)하는지 더 알고 싶으시다면 Markus Winand의 "[The Three-Valued Logic of SQL](https://modern-sql.com/concept/three-valued-logic)"을 읽어보시면 좋을 것입니다(enjoy).

일반적으로 R에서 `NA`에 쓰는 함수로 `NULL`도 다룹니다.

```r
flights |> 
  filter(!is.na(dep_delay)) |> 
  show_query()
```

이 SQL 쿼리에서 dbplyr의 단점(drawbacks) 하나가 드러납니다. SQL은 정확하지만 손으로 쓰기에는 지나치게 복잡합니다.

이 경우 괄호(parentheses)를 빼고 더 읽기 쉬운 특수 연산자를 씁니다.

``` sql
WHERE "dep_delay" IS NOT NULL
```

요약(summarize)을 사용하여 만든 변수를 `filter()`하는 경우, dbplyr는 `WHERE` 절이 아닌 `HAVING` 절을 생성한다는 점에 유의(Note)하세요. 이것은 SQL의 특이한 점(idiosyncrasies) 중 하나입니다. `WHERE`는 `SELECT` 및 `GROUP BY` 전에 평가(evaluated)되므로, SQL은 나중에(afterwards) 평가될 다른 절이 필요합니다.

```r
diamonds_db |> 
  group_by(cut) |> 
  summarize(n = n()) |> 
  filter(n > 100) |> 
  show_query()
```

##### ORDER BY

행 정렬(ordering rows)에는 `arrange()`에서 `ORDER BY` 절로의 간단한(straightforward) 변환이 포함(involves)됩니다.

```r
flights |> 
  arrange(year, month, day, desc(dep_delay)) |> 
  show_query()
```

`desc()`가 `DESC`로 어떻게 변환되는지 주목하세요(Notice). 이 함수는 이름이 SQL에서 직접 영감을 받은(directly inspired by) 많은 dplyr 함수 중 하나입니다.

##### 서브쿼리

때로는(Sometimes) dplyr 파이프라인(pipeline)을 단일 `SELECT` 명령문으로 변환할 수 없어서 서브쿼리를 사용해야 하는 경우가 있습니다. 서브쿼리(subquery)는 일반적인 테이블 대신 `FROM` 절에서 데이터 소스로 사용되는 쿼리일 뿐입니다.

dbplyr는 보통 서브쿼리로 SQL의 한계(limitations)를 우회합니다. 예를 들어 `SELECT` 절의 표현식(expressions)은 방금 만든 열을 참조하지 못합니다. 따라서 다음의 단순한 dplyr 파이프라인도 두 단계로 처리해야 합니다. 첫 번째 내부 쿼리가 `year1`을 계산한 다음 두 번째 외부 쿼리가 `year2`를 계산합니다.

```r
flights |> 
  mutate(
    year1 = year + 1,
    year2 = year1 + 1
  ) |> 
  show_query()
```

방금 생성한 변수를 `filter()`하려고 시도한(attempted) 경우에도 이것을 볼 수 있습니다. `WHERE`는 `SELECT` 이후에 작성되지만 그 전에 평가되므로 이 (어리석은) 예제에서는 서브쿼리가 필요합니다(Remember).

```r
flights |> 
  mutate(year1 = year + 1) |> 
  filter(year1 == 2014) |> 
  show_query()
```

때로는 dbplyr가 해당 변환을 최적화(optimize)하는 방법을 아직 모르기 때문에 필요하지 않은 곳에(where it's not needed) 서브쿼리를 생성(create)하기도 합니다. 시간이 지나면서(over time) dbplyr가 개선(improves)됨에 따라 이러한 경우는 드물어지겠지만(get rarer) 아마 완전히 없어지지는(never go away) 않을 것입니다.

##### 조인

dplyr의 조인에 익숙하다면 SQL 조인도 매우 유사합니다. 다음은 간단한 예시입니다.

```r
flights |> 
  left_join(planes |> rename(year_built = year), join_by(tailnum)) |> 
  show_query()
```

여기서 주목해야 할 주요(main) 사항은 구문(syntax)입니다. SQL 조인은 `FROM` 절의 하위 절(sub-clauses)을 사용하여 추가 테이블을 가져오고 `ON`을 사용하여 테이블 간의 관계(related)를 정의합니다.

이 함수들의 dplyr 이름은 SQL과 밀접하게 이어져 있습니다. 따라서 `inner_join()`, `right_join()`, `full_join()`에 해당하는 SQL도 쉽게 짐작됩니다.

``` sql
SELECT flights.*, "type", manufacturer, model, engines, seats, speed
FROM flights
INNER JOIN planes ON (flights.tailnum = planes.tailnum)

SELECT flights.*, "type", manufacturer, model, engines, seats, speed
FROM flights
RIGHT JOIN planes ON (flights.tailnum = planes.tailnum)

SELECT flights.*, "type", manufacturer, model, engines, seats, speed
FROM flights
FULL JOIN planes ON (flights.tailnum = planes.tailnum)
```

데이터베이스의 데이터를 다루다 보면 조인을 자주 쓰게 됩니다. 데이터베이스 테이블은 대개 매우 정규화된 형태(highly normalized form)로 저장됩니다. 각 "사실(fact)"은 한 곳에만 있습니다. 완전한 분석 데이터셋을 만들려면 기본 키(primary keys)와 외래 키(foreign keys)로 이어진 복잡한 테이블 관계를 탐색해야 합니다.

이러한 상황에 부닥친다면(hit this scenario), Tobias Schieferdecker, Kirill Müller, Darko Bergant가 만든 [dm 패키지]([https://cynkra.github.io/dm/](https://cynkra.github.io/dm/))가 구세주(life saver)가 될 것입니다.

이 패키지는 DBA가 제공하는 제약 조건(constraints)으로 테이블 간 연결을 자동으로 판단합니다. 관계를 시각화하며 한 테이블을 다른 테이블에 잇는 조인도 생성합니다.

##### 다른 동사들

dbplyr는 `distinct()`, `slice_*()`, `intersect()`와 같은 다른 동사(verbs)와 `pivot_longer()`, `pivot_wider()`와 같이 점점 더 다양해지는(growing selection) tidyr 함수도 변환합니다.

현재 사용 가능한 전체 세트(full set)를 확인하는 쉬운 방법은 dbplyr 웹사이트([https://dbplyr.tidyverse.org/reference/](https://dbplyr.tidyverse.org/reference/))를 방문(visit)하는 것입니다.

### 연습 문제

1. `distinct()`는 무엇으로 변환됩니까?`head()`는 어떻습니까?

2. 다음 SQL 쿼리가 각각 무엇을 하는지 설명하고 dbplyr를 사용하여 다시 만들어(recreate) 보세요.

``` sql
SELECT * 
FROM flights
WHERE dep_delay < arr_delay

SELECT *, distance / (air_time / 60) AS speed
FROM flights
```

### 함수 변환

지금까지 우리는 dplyr 동사가 쿼리의 절(clauses)로 변환되는 과정의 큰 그림(big picture)에 초점을 맞췄습니다.

이제 범위를 좁혀 개별 열(individual columns)에 작동하는 R 함수의 변환을 살펴보겠습니다. 예를 들어 `summarize()`에서 `mean(x)`를 쓰면 어떻게 될까요?

무슨 일이 일어나는지(what's going on) 확인하는 데 도움이 되도록 `summarize()` 또는 `mutate()`를 실행하고 생성된 SQL을 보여주는 몇 가지 작은 도우미(helper) 함수를 사용하겠습니다.

이렇게 하면 몇 가지 변형(variations)을 탐구(explore)하고 요약(summaries)과 변형(transformations)이 어떻게 다를(differ) 수 있는지 조금 더 쉽게 볼 수 있습니다.

```r
summarize_query <- function(df, ...) {
  df |> 
    summarize(...) |> 
    show_query()
}
mutate_query <- function(df, ...) {
  df |> 
    mutate(..., .keep = "none") |> 
    show_query()
}
```

몇 가지 요약으로 뛰어들어(dive in) 봅시다! 아래 코드를 보면 `mean()`과 같은 일부 요약 함수는 비교적(relatively) 간단한(simple) 변환을 갖는 반면 `median()`과 같은 다른 함수는 훨씬 더 복잡하다는(complex) 것을 알 수 있습니다.

통계(statistics)에서는 일반적(common)이지만 데이터베이스에서는 덜 일반적인 연산일수록 일반적으로 복잡성이 더 높습니다(higher).

```r
flights |> 
  group_by(year, month, day) |>  
  summarize_query(
    mean = mean(arr_delay, na.rm = TRUE),
    median = median(arr_delay, na.rm = TRUE)
  )
```

요약 함수를 `mutate()` 내부에서 사용하면 소위(so-called) 윈도우(window) 함수로 전환되어야 하기 때문에 요약 함수의 변환이 더 복잡(complicated)해집니다. SQL에서는 그 뒤에 `OVER`를 추가하여 일반(ordinary) 집계 함수를 윈도우 함수로 전환(turn)합니다.

```r
flights |> 
  group_by(year, month, day) |>  
  mutate_query(
    mean = mean(arr_delay, na.rm = TRUE),
  )
```

SQL에서 `GROUP BY` 절은 요약 전용(exclusively for summaries)으로 사용되므로 여기서는 그룹화(grouping)가 `GROUP BY` 절에서 `OVER`로 이동(moved)한 것을 볼 수 있습니다.

윈도우 함수에는 각각 "이전(previous)" 또는 "다음(next)" 값을 보는(look at) `lead()` 및 `lag()`와 같이 앞을 내다보거나(look forward) 뒤를 돌아보는(look backwards) 모든 함수가 포함(include)됩니다.

```r
flights |> 
  group_by(dest) |>  
  arrange(time_hour) |> 
  mutate_query(
    lead = lead(arr_delay),
    lag = lag(arr_delay)
  )
```

SQL 테이블에는 내재된(intrinsic) 순서가 없기 때문에 여기서 데이터를 `arrange()`하는 것이 중요합니다.

사실 `arrange()`를 사용하지 않으면 매번 다른 순서로 행을 반환받을 수도 있습니다! 윈도우 함수의 경우 순서 정보(ordering information)가 반복된다는(repeated) 점에 유의하세요(Notice): 메인 쿼리의 `ORDER BY` 절이 윈도우 함수에 자동으로 적용(apply)되지는 않습니다.

또 다른 중요한 SQL 함수는 `CASE WHEN`입니다. `if_else()`와 `case_when()`의 변환에 쓰입니다. 이 함수는 dplyr 함수에 직접 영감을 주었습니다(directly inspired).

다음은 몇 가지 간단한 예시입니다.

```r
flights |> 
  mutate_query(
    description = if_else(arr_delay > 0, "delayed", "on-time")
  )
flights |> 
  mutate_query(
    description = 
      case_when(
        arr_delay < -5 ~ "early", 
        arr_delay < 5 ~ "on-time",
        arr_delay >= 5 ~ "late"
      )
  )
```

`CASE WHEN`은 R에서 SQL로 직접 변환(direct translation)되지 않는 다른 일부 함수에도 사용됩니다. 이것의 좋은 예시가 `cut()`입니다.

```r
flights |> 
  mutate_query(
    description =  cut(
      arr_delay, 
      breaks = c(-Inf, -5, 5, Inf), 
      labels = c("early", "on-time", "late")
    )
  )
```

dbplyr는 또한 일반적인 문자열 및 날짜-시간 조작(manipulation) 함수도 변환하는데, 이는 `vignette("translation-function", package = "dbplyr")`에서 배울 수 있습니다.

dbplyr의 변환이 완벽하지는 않고 아직 변환되지 않은 R 함수도 많습니다. 그래도 자주 쓰는 함수는 놀라울 만큼 잘 다룹니다.

## 웹 스크래핑

이 장에서는 [rvest]([https://rvest.tidyverse.org](https://rvest.tidyverse.org))로 웹을 스크래핑하는 기본 방법을 소개합니다.

웹 스크래핑은 웹페이지에서 데이터를 뽑아낼 때 매우 유용합니다. 일부 웹사이트는 구조화된 HTTP 요청(requests) 모음인 API를 운영합니다. 그 결과는 JSON으로 돌아오며 @sec-rectangling의 기술로 처리합니다.
API가 있다면 API를 쓰는 편이 좋습니다. 보통 더 안정적인 데이터를 주기 때문입니다. 다만 웹 API 프로그래밍은 이 책의 범위를 벗어납니다.

여기서는 API 제공 여부와 상관없이 적용하는 스크래핑을 배웁니다.

```r
#| label: setup
#| message: false
library(tidyverse)
library(rvest)
```

### 스크래핑 윤리 및 합법성

코드를 살펴보기 전에 웹 스크래핑의 합법성과 윤리부터 짚어야 합니다. 두 문제 모두 상황이 복잡합니다.

합법성은 거주 지역에 따라 크게 달라집니다. 일반적으로 데이터가 공개되어 있고 개인 정보가 아니며 사실에 해당한다면 문제가 없을 가능성이 큽니다.

이 세 가지 요소가 중요한 이유는 아래에서 논의할 사이트의 이용 약관, 개인 식별 정보 및 저작권과 관련이 있기 때문입니다.

물론 우리는 변호사가 아닙니다. 이 내용도 법률 자문이 아닙니다. 다만 이 주제를 폭넓게 검토한 뒤 정리했습니다.

데이터가 공개되지 않았거나 개인 정보 또는 사실이 아닌 내용을 담고 있다면, 혹은 수익을 목적으로 스크래핑한다면 변호사와 상담해야 합니다. 어떤 경우든 대상 페이지를 호스팅하는 서버의 자원을 존중하세요. 여러 페이지를 긁을 때는 요청 사이에 반드시 잠시 간격을 둬야 합니다.

간단한 방법은 Dmytro Perepolkin의 [polite]([https://dmi3kno.github.io/polite/](https://dmi3kno.github.io/polite/)) 패키지를 쓰는 것입니다. 요청 사이에 자동으로 멈추고 결과를 캐시해서 같은 페이지를 두 번 요청하지 않습니다.

#### 서비스 약관

많은 웹사이트에는 "이용 약관(terms and conditions)"이나 "서비스 약관(terms of service)" 링크가 있습니다. 자세히 읽어보면 웹 스크래핑을 명시적으로 금지한 곳도 적지 않습니다. 이런 약관은 기업이 매우 넓은 권리를 주장하는 법적 영역인 경우가 많습니다. 가능한 한 약관을 존중하는 것이 예의입니다. 다만 모든 주장을 그대로 받아들일 필요는 없습니다.

미국 법원은 웹사이트 바닥글에 서비스 약관을 게시한 것만으로는 사용자에게 구속력이 생기지 않는다고 판단하는 경우가 많습니다(HiQ Labs v. LinkedIn([https://en.wikipedia.org/wiki/HiQ_Labs_v._LinkedIn](https://en.wikipedia.org/wiki/HiQ_Labs_v._LinkedIn)). 

일반적으로 서비스 약관에 구속되려면 계정 생성 또는 확인란 선택과 같은 명시적인 행동을 취해야 합니다. 이것이 데이터의 공개 여부가 중요한 이유입니다. 데이터에 액세스하는 데 계정이 필요하지 않다면 서비스 약관에 구속될 가능성이 낮습니다.

유럽의 사정은 상당히 다릅니다. 명시적으로 동의하지 않은 서비스 약관도 집행 대상이라는 법원 판결이 있습니다.

#### 개인 식별 정보

공개 데이터라도 이름, 이메일 주소, 전화번호, 생년월일 같은 개인 식별 정보를 스크래핑할 때는 극도로 조심해야 합니다. 유럽에는 이런 데이터의 수집과 저장을 엄격히 규제하는 GDPR이 있습니다. 거주지와 무관하게 윤리 문제에 빠질 위험도 큽니다. 2016년 한 연구진은 데이팅 사이트 OkCupid에서 공개 프로필 70,000건을 스크래핑했습니다. 사용자 이름, 나이, 성별, 위치 등이 담긴 데이터를 비식별화하지 않은 채 공개했습니다.

연구진은 이미 공개된 데이터라 문제가 없다고 여겼습니다. 그러나 데이터셋에 포함된 사용자의 식별 가능성을 둘러싼 윤리적 우려 때문에 큰 비판을 받았습니다. 개인 식별 정보를 스크래핑해야 한다면 OkCupid 연구[^webscraping-3]와 비슷한 사례를 꼭 읽어보세요. 개인 정보의 획득과 공개를 둘러싼 연구 윤리 문제를 미리 살펴야 합니다.

OkCupid 연구에 대한 기사의 한 예는 Wired([https://www.wired.com/2016/05/okcupid-study-reveals-perils-big-data-science](https://www.wired.com/2016/05/okcupid-study-reveals-perils-big-data-science))에 게시되었습니다.

#### 저작권

마지막으로 저작권법도 고려해야 합니다. 법은 복잡합니다. 그래도 보호 대상을 설명한 미국법을 살펴볼 만합니다.

"... 어떤 형태의 유형적인 표현 매체에 고정된 저자의 독창적인 저작물, ..."

그런 다음 어문 저작물, 음악 저작물, 영화 등 적용되는 범주를 설명합니다. 이 목록에는 데이터가 빠져 있습니다. 따라서 사실만 스크래핑한다면 저작권 보호가 적용되지 않습니다. 다만 유럽에는 데이터베이스를 별도로 보호하는 "독자적(sui generis)" 권리가 있습니다.

간단한 예로, 미국에서는 재료 및 지침 목록이 저작권의 대상이 아니므로 저작권을 사용하여 레시피를 보호할 수 없습니다. 그러나 해당 레시피 목록에 실질적이고 새로운 문학적 콘텐츠가 함께 제공되는 경우에는 저작권이 인정됩니다. 이것이 인터넷에서 레시피를 찾을 때 항상 사전에 내용이 너무 많은 이유입니다.

텍스트나 이미지 같은 원본 콘텐츠를 스크래핑해야 할 때도 공정 이용 원칙(doctrine of fair use)([https://en.wikipedia.org/wiki/Fair_use](https://en.wikipedia.org/wiki/Fair_use))의 보호를 받을 수 있습니다. 공정 이용은 단일한 규칙이 아니라 여러 요소를 함께 따집니다. 연구나 비상업적 목적으로 데이터를 모으고 꼭 필요한 범위만 스크래핑할수록 적용 가능성이 높습니다.

### HTML의 기초

웹페이지를 스크래핑하려면 페이지를 기술하는 언어인 HTML을 조금 알아야 합니다. HTML은 HyperText Markup Language의 약자이며 다음처럼 생겼습니다.

``` html
<html>
<head>
  <title>Page title</title>
</head>
<body>
  <h1 id='first'>A heading</h1>
  <p>Some text &amp; <b>some bold text.</b></p>
  <img src='myimg.png' width='100' height='100'>
</body>
```

HTML은 요소(elements)가 겹겹이 이어진 계층 구조입니다. 요소는 시작 태그(tag, 예: `<tag>`), 선택적인 속성(attributes)(`id='first'`), 종료 태그(`</tag>`), 그리고 두 태그 사이의 콘텐츠(contents)로 구성됩니다.

`<`와 `>`는 태그를 나타내는 기호라 본문에 직접 쓸 수 없습니다.
대신 HTML 이스케이프(escapes)인 `&gt;`(보다 큼, greater than)와 `&lt;`(보다 작음, less than)을 씁니다.

이스케이프가 `&`로 시작하므로 앰퍼샌드(ampersand) 자체는 `&amp;`로 적습니다. HTML 이스케이프는 매우 다양하지만 rvest가 자동으로 처리하니 모두 외울 필요는 없습니다.

웹 스크래핑이 가능한 까닭은 대개 대상 페이지가 일관된 구조로 데이터를 담고 있기 때문입니다.

#### 요소

HTML 요소는 100개가 넘습니다. 그중 중요한 몇 가지를 살펴봅시다.

1. 모든 HTML 페이지는 `<html>` 요소 안에 있고 두 자식(children)을 둡니다. `<head>`에는 페이지 제목 같은 문서 메타데이터(document metadata)가, `<body>`에는 브라우저에 표시할 콘텐츠가 들어갑니다.

2. `<h1>`(제목 1, heading 1), `<section>`(섹션, section), `<p>`(단락, paragraph), `<ol>`(순서가 있는 리스트, ordered list) 같은 블록(Block) 태그는 페이지의 전체 구조를 만듭니다.

3. `<b>`(굵게, bold), `<i>`(기울임꼴, italics), `<a>`(링크, link) 같은 인라인(Inline) 태그는 블록 태그 안의 텍스트 서식을 지정합니다.

낯선 태그를 만나면 간단한 검색으로 기능을 알아볼 수 있습니다. 웹 프로그래밍의 거의 모든 내용을 설명하는 [MDN Web Docs]([https://developer.mozilla.org/en-US/docs/Web/HTML](https://developer.mozilla.org/en-US/docs/Web/HTML))도 좋은 출발점입니다.

대부분의 요소는 시작 태그와 종료 태그 사이에 콘텐츠를 담습니다. 콘텐츠는 텍스트일 수도 있고 다른 요소일 수도 있습니다. 다음 HTML에는 단어 하나를 굵게 표시한 텍스트 단락이 있습니다.

```         
<p>
  Hi! My <b>name</b> is Hadley.
</p>
```

다른 요소 안에 든 요소를 자식(children)이라고 합니다. 위의 `<p>`에는 `<b>`라는 자식 하나가 있습니다. `<b>`에는 자식이 없지만 "name"이라는 텍스트가 들어 있습니다.

#### 속성

태그에는 `name1='value1' name2='value2'`처럼 이름이 붙은 속성(attributes)을 둘 수 있습니다. 중요한 속성은 `id`와 `class`입니다. CSS(Cascading Style Sheets)와 함께 써서 페이지의 시각적 모양을 조절합니다.

두 속성은 페이지에서 데이터를 스크래핑할 때 자주 유용합니다. 속성은 링크 대상(`<a>`의 `href`)이나 이미지 출처(`<img>`의 `src`)를 기록하는 데도 쓰입니다.

### 데이터 추출

스크래핑을 시작하려면 대상 페이지의 URL이 필요합니다. 보통 웹 브라우저에서 복사하면 됩니다. `read_html()`로 페이지의 HTML을 R에 읽어 들이세요. 이 함수는 `xml_document` 객체를 반환합니다. 이후 rvest 함수로 이 객체를 다룹니다.

```r
html <- read_html("http://rvest.tidyverse.org/")
html
```

rvest에는 HTML을 코드 안에 직접 쓰는 함수도 있습니다. 여러 rvest 함수의 작동 방식을 간단한 예제로 설명할 때 자주 사용하겠습니다.

```r
html <- minimal_html("
  <p>This is a paragraph</p>
  <ul>
    <li>This is a bulleted list</li>
  </ul>
")
html
```

이제 R에 HTML을 불러왔으니 필요한 데이터를 추출할 차례입니다. 먼저 관심 있는 요소를 찾는 CSS 선택자(selectors)와 그 요소에서 데이터를 꺼내는 rvest 함수를 배웁니다. 이어서 전용 도구가 있는 HTML 표(tables)를 간단히 다룹니다.

#### 요소 찾기

CSS는 cascading style sheets의 약자로, HTML 문서의 시각적 스타일을 정의하는 도구입니다. CSS에는 페이지 요소를 선택하는 작은 언어가 들어 있는데 이를 CSS 선택자(CSS selectors)라고 합니다. 선택자는 HTML 요소를 찾는 패턴입니다. 추출할 요소를 간결하게 표현해서 스크래핑에 유용합니다.

CSS 선택자는 뒤에서 더 자세히 다룹니다. 우선 세 가지만 알아도 많은 일을 해냅니다.

- `p`는 모든 `<p>` 요소를 선택합니다.
- `.title`은 `class`가 "title"인 모든 요소를 선택합니다.
- `#title`은 "title"과 같은(equals) `id` 속성을 가진 요소를 선택합니다.

Id 속성은 문서 안에서 고유해야 하므로 `#title`은 요소 하나만 선택합니다. 간단한 예제로 세 선택자를 시험해 봅시다.

```r
html <- minimal_html("
  <h1>This is a heading</h1>
  <p id='first'>This is a paragraph</p>
  <p class='important'>This is an important paragraph</p>
")
```

선택자와 일치하는 모든 요소는 `html_elements()`로 찾습니다.

```r
html |> html_elements("p")
html |> html_elements(".important")
html |> html_elements("#first")
```

`html_element()`도 중요한 함수입니다. 언제나 입력과 같은 개수의 출력을 반환합니다. 문서 전체에 적용하면 처음 일치한 항목을 돌려줍니다.

```r
html |> html_element("p")
```

선택자와 일치하는 요소가 없을 때 두 함수의 차이가 드러납니다. `html_elements()`는 길이 0인 벡터를, `html_element()`는 결측값을 반환합니다. 이 차이는 곧 중요해집니다.

```r
html |> html_elements("b")
html |> html_element("b")
```

#### 선택자 중첩

대개 `html_elements()`와 `html_element()`를 함께 씁니다. 먼저 `html_elements()`로 관측치(observations)가 될 요소를 찾습니다. 이어서 `html_element()`로 변수(variables)가 될 요소를 고릅니다.

간단한 예제로 작동 방식을 살펴봅시다. 다음은 StarWars 캐릭터 네 명의 정보가 담긴 순서 없는 목록(unordered list, `<ul>`)입니다. 캐릭터마다 목록 항목(`<li>`)이 하나씩 있습니다.

```r
html <- minimal_html("
  <ul>
    <li><b>C-3PO</b> is a <i>droid</i> that weighs <span class='weight'>167 kg</span></li>
    <li><b>R4-P17</b> is a <i>droid</i></li>
    <li><b>R2-D2</b> is a <i>droid</i> that weighs <span class='weight'>96 kg</span></li>
    <li><b>Yoda</b> weighs <span class='weight'>66 kg</span></li>
  </ul>
  ")
```

`html_elements()`로 캐릭터마다 요소 하나를 담은 벡터를 만듭니다.

```r
characters <- html |> html_elements("li")
characters
```

각 캐릭터의 이름은 `html_element()`로 추출합니다. `html_elements()`의 출력에 적용하면 요소마다 응답 하나를 돌려주기 때문입니다.

```r
characters |> html_element("b")
```

두 함수의 차이는 이름을 추출할 때는 중요하지 않지만 체중을 다룰 때는 중요합니다. 체중 `<span>`이 없는 캐릭터도 결과에서 자리를 유지해야 합니다. `html_element()`가 바로 그렇게 작동합니다.

```r
characters |> html_element(".weight")
```

`html_elements()`는 `characters`의 자식 가운데 체중 `<span>`을 모두 찾습니다. 세 개뿐이라 이름과 체중의 대응 관계가 끊어집니다.

```r
characters |> html_elements(".weight")
```

이제 관심 있는 요소를 선택했으므로 텍스트 콘텐츠(text contents) 또는 일부 속성(attributes)에서 데이터를 추출해야 합니다.

#### 텍스트와 속성

`html_text2()`는 HTML 요소의 일반 텍스트(plain text) 콘텐츠를 추출합니다. 

```r
characters |> 
  html_element("b") |> 
  html_text2()

characters |> 
  html_element(".weight") |> 
  html_text2()
```

이스케이프는 자동으로 처리됩니다. HTML 이스케이프는 rvest가 반환한 데이터가 아니라 원본(source) HTML에서만 보입니다.

`html_attr()`은 속성에서 데이터를 추출합니다.

```r
html <- minimal_html("
  <p><a href='https://en.wikipedia.org/wiki/Cat'>cats</a></p>
  <p><a href='https://en.wikipedia.org/wiki/Dog'>dogs</a></p>
")

html |> 
  html_elements("p") |> 
  html_element("a") |> 
  html_attr("href")
```

`html_attr()`은 언제나 문자열을 반환합니다. 숫자나 날짜를 추출했다면 후처리(post-processing)가 조금 필요합니다.

#### 표 (Tables)

운이 좋으면 데이터가 이미 HTML 표(table)에 담겨 있어 그대로 읽기만 하면 됩니다. 브라우저에서 표를 알아보기는 어렵지 않습니다. 행과 열로 된 직사각형 구조이며 Excel 같은 도구에 복사해 붙여 넣을 수 있습니다.

HTML 표는 주로 네 요소로 구성됩니다. `<table>`, `<tr>`(표 행, table row), `<th>`(표 제목, table heading), `<td>`(표 데이터, table data)입니다. 다음은 2열 3행짜리 간단한 HTML 표입니다.

```r
html <- minimal_html("
  <table class='mytable'>
    <tr><th>x</th>   <th>y</th></tr>
    <tr><td>1.5</td> <td>2.7</td></tr>
    <tr><td>4.9</td> <td>1.3</td></tr>
    <tr><td>7.2</td> <td>8.1</td></tr>
  </table>
  ")
```

rvest의 `html_table()`은 이런 데이터를 읽는 함수입니다. 페이지에서 찾은 표마다 티블(tibble) 하나를 만들어 리스트로 반환합니다. 원하는 표는 `html_element()`로 고릅니다.

```r
html |> 
  html_element(".mytable") |> 
  html_table()
```

`x`와 `y`는 자동으로 숫자로 바뀌었습니다. 자동 변환이 늘 성공하는 것은 아닙니다. 더 복잡한 상황에서는 `convert = FALSE`로 기능을 끄고 직접 변환하는 편이 좋습니다.

#### 올바른 선택자 찾기

데이터에 맞는 선택자를 찾는 일이 보통 가장 어렵습니다. 관심 없는 항목은 제외하면서(specific) 필요한 항목은 모두 포함하는(sensitive) 선택자를 찾으려면 여러 번 시험해야 합니다.

시행착오(trial and error)는 자연스러운 과정입니다. SelectorGadget과 브라우저 개발자 도구(developer tools)가 큰 도움이 됩니다.

[SelectorGadget]([https://rvest.tidyverse.org/articles/selectorgadget.html](https://rvest.tidyverse.org/articles/selectorgadget.html))은 사용자가 지정한 긍정적(positive)·부정적(negative) 예시를 바탕으로 CSS 선택자를 자동 생성하는 자바스크립트 북마크릿(javascript bookmarklet)입니다.

항상 성공하지는 않지만 제대로 작동할 때는 놀라울 만큼 편리합니다.
설치와 사용법은 [https://rvest.tidyverse.org/articles/selectorgadget.html]([https://rvest.tidyverse.org/articles/selectorgadget.html](https://rvest.tidyverse.org/articles/selectorgadget.html)) 또는 Mine의 [동영상](https://www.youtube.com/watch?v=PetWV5g1Xsc)에서 배울 수 있습니다.

최신 브라우저에는 모두 개발자 도구가 있습니다. 평소 다른 브라우저를 쓰더라도 이 작업에는 Chrome을 권합니다. Chrome의 웹 개발자 도구는 완성도가 높고 바로 쓸 수 있습니다.

페이지의 요소를 마우스 오른쪽 버튼으로 클릭(Right click)하고 `Inspect`(검사)를 클릭합니다.
그러면 방금 클릭한 요소를 중심(centered on)으로 전체 HTML 페이지의 확장 가능(expandable)한 뷰(view)가 열립니다.

이 화면에서 페이지 구조를 살피며 알맞은 선택자를 가늠합니다.

class와 id 속성을 눈여겨보세요. 페이지의 시각적 구조를 만드는 데 자주 쓰이므로 원하는 데이터를 추출할 때 좋은 실마리가 됩니다.

요소(Elements) 화면에서 항목을 마우스 오른쪽 버튼으로 누르고 `Copy as Selector`(선택자로 복사)를 고르면 해당 요소를 고유하게 가리키는 선택자가 만들어집니다.

SelectorGadget이나 Chrome DevTools가 이해할 수 없는(don't understand) CSS 선택자를 생성한 경우 CSS 선택자를 쉬운 영어(plain English)로 번역(translates)해 주는 Selectors Explained[https://kittygiraudel.github.io/selectors-explained/](https://kittygiraudel.github.io/selectors-explained/)를 사용해 보세요.

스크래핑을 자주 한다면 CSS 선택자를 체계적으로 배워두는 편이 좋습니다. 재미있는 [CSS dinner](https://flukeout.github.io/) 튜토리얼로 시작한 뒤 [MDN 웹 문서](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Selectors)를 참고하길 권합니다.

### 종합하기

이제 배운 내용을 종합해 몇몇 웹사이트를 스크래핑해 봅시다. 예제를 실행할 시점에는 코드가 더는 작동하지 않을 수도 있습니다. 웹 스크래핑이 안고 있는 근본적인 문제입니다. 사이트 구조가 바뀌면 스크래핑 코드도 고쳐야 합니다.

#### StarWars

rvest의 `vignette("starwars")`에는 아주 간단한 예제가 있습니다. HTML을 최소화한 페이지라 시작하기 좋습니다.

해당 페이지로 이동해 "요소 검사(Inspect Element)"로 스타워즈 영화 제목(headings) 하나를 살펴보세요.

키보드나 마우스로 HTML 계층을 오가며 영화마다 되풀이되는 구조를 찾아보세요.

각 영화는 아래와 같은 공통 구조를 지닙니다.

``` html
<section>
  <h2 data-id="1">The Phantom Menace</h2>
  <p>Released: 1999-05-19</p>
  <p>Director: <span class="director">George Lucas</span></p>
  
  <div class="crawl">
    <p>...</p>
    <p>...</p>
    <p>...</p>
  </div>
</section>
```

목표는 이 데이터를 7행짜리 데이터 프레임으로 바꾸는 것입니다. 변수는 `title`(제목), `year`(연도), `director`(감독), `intro`(소개)입니다. 먼저 HTML을 읽고 `<section>` 요소를 모두 추출합니다.

```r
url <- "https://rvest.tidyverse.org/articles/starwars.html"
html <- read_html(url)

section <- html |> html_elements("section")
section
```

페이지의 영화 7편과 일치하는 요소 7개가 나왔습니다. `section`이 적절한 선택자라는 뜻입니다. 데이터는 모두 텍스트에 있으므로 개별 요소를 꺼내는 일은 어렵지 않습니다. 알맞은 선택자만 찾으면 됩니다.

```r
section |> html_element("h2") |> html_text2()

section |> html_element(".director") |> html_text2()
```

구성 요소마다 같은 작업을 마친 뒤 결과를 티블로 묶습니다.
```r
tibble(
  title = section |> 
    html_element("h2") |> 
    html_text2(),
  released = section |> 
    html_element("p") |> 
    html_text2() |> 
    str_remove("Released: ") |> 
    parse_date(),
  director = section |> 
    html_element(".director") |> 
    html_text2(),
  intro = section |> 
    html_element(".crawl") |> 
    html_text2()
)
```

나중에 분석하기 편하도록 `released`를 조금 더 가공했습니다.

#### IMDB 최고 영화

다음은 조금 더 까다로운 작업입니다. 인터넷 영화 데이터베이스(IMDb)에서 상위 영화 250편을 추출해 보겠습니다.

```r
#| label: fig-scraping-imdb
#| echo: false
#| fig-cap: | 
#|   2022년 12월 5일에 캡처한(taken on) IMDb 상위 영화 웹페이지의 스크린샷.
#| fig-alt: |
#|   스크린샷에는 "Rank and Title", "IMDb Rating", 
#|   "Your Rating" 열이 있는 표가 표시되어 있습니다. 상위 250개 중
#|   9개의 영화가 표시됩니다. 상위 5개는 쇼생크 탈출(Shawshank Redemption), 
#|   대부(The Godfather), 다크 나이트(The Dark Knight), 대부 2(The Godfather: Part II),
#|   12명의 성난 사람들(12 Angry Men)입니다.
knitr::include_graphics("screenshots/scraping-imdb.png", dpi = 300)
```

데이터가 뚜렷한 표 구조(tabular structure)를 이루므로 `html_table()`부터 써볼 만합니다.

```r
url <- "https://web.archive.org/web/20220201012049/https://www.imdb.com/chart/top/"
html <- read_html(url)

table <- html |> 
  html_element("table") |> 
  html_table()
table
```

빈 열이 몇 개 있지만 표의 정보는 대체로 잘 들어왔습니다. 쓰기 편한 형태로 만들려면 조금 더 가공해야 합니다. 먼저 열 이름을 바꾸고 순위(rank)와 제목(title)에 낀 불필요한 공백(whitespace)을 없앱니다.
두 열을 한 단계에서 고르고 이름도 바꾸려고 `rename()` 대신 `select()`를 씁니다. 이어서 줄바꿈과 여분의 공백을 제거한 뒤 `separate_wider_regex()`로 제목, 연도, 순위를 각각의 변수로 추출합니다.

```r
ratings <- table |>
  select(
    rank_title_year = `Rank & Title`,
    rating = `IMDb Rating`
  ) |> 
  mutate(
    rank_title_year = str_replace_all(rank_title_year, "\n +", " ")
  ) |> 
  separate_wider_regex(
    rank_title_year,
    patterns = c(
      rank = "\\d+", "\\. ",
      title = ".+", " +\\(",
      year = "\\d+", "\\)"
    )
  )
ratings
```

데이터 대부분이 표의 셀(cells)에서 나오더라도 원본(raw) HTML을 살펴볼 가치가 있습니다. 속성 하나에서 추가 데이터를 얻을 수도 있습니다. 페이지 소스를 동굴 탐험하듯 훑다 보면 새로운 데이터를 발견하거나 더 쉬운 구문 분석(parsing) 경로를 찾기도 합니다.

```r
html |> 
  html_elements("td strong") |> 
  head() |> 
  html_attr("title")
```

이 값을 표 데이터와 합친 뒤 `separate_wider_regex()`를 다시 적용해 필요한 데이터를 추출합니다.

```r
ratings |>
  mutate(
    rating_n = html |> html_elements("td strong") |> html_attr("title")
  ) |> 
  separate_wider_regex(
    rating_n,
    patterns = c(
      "[0-9.]+ based on ",
      number = "[0-9,]+",
      " user ratings"
    )
  ) |> 
  mutate(
    number = parse_number(number)
  )
```

#### 동적 사이트

지금까지는 `html_elements()`가 브라우저 화면과 같은 내용을 반환하는 웹사이트를 다뤘습니다. 반환값을 구문 분석하고 깔끔한(tidy) 데이터 프레임으로 정리하는 방법도 살펴봤습니다.

하지만 `html_elements()` 계열 함수가 브라우저 화면과 전혀 다른 내용을 돌려주는 사이트도 있습니다.

대개 자바스크립트(javascript)로 페이지 콘텐츠를 동적으로 만드는 사이트에서 이런 일이 생깁니다.

rvest는 원본 HTML을 다운로드하고 자바스크립트를 실행하지 않기 때문에 현재는 rvest로 작동하지 않습니다.

이런 사이트도 스크래핑은 가능합니다. 다만 rvest가 자바스크립트 실행까지 포함해 웹 브라우저 전체를 흉내 내야 하므로 비용이 큽니다.

집필 당시에는 이 기능을 쓸 수 없지만 적극적으로 개발 중입니다. 독자가 이 글을 읽을 때쯤에는 제공될지도 모릅니다.

이 기능은 백그라운드에서 실제 Chrome 브라우저를 실행하는 [chromote 패키지](https://rstudio.github.io/chromote/index.html)를 사용합니다. 사람이 텍스트를 입력하고 버튼을 누르듯 사이트와 상호 작용하는 도구도 들어 있습니다.

자세한 내용은 [rvest 웹사이트](http://rvest.tidyverse.org/)에서 확인하세요.

## 계층형 데이터

계층형(hierarchical) 또는 트리(tree-like) 형태의 데이터를 행과 열로 된 직사각형(rectangular) 데이터 프레임으로 바꾸는 직사각형화(rectangling) 기법을 배웁니다.

계층형 데이터는 생각보다 흔합니다. 특히 웹에서 가져온 데이터를 다룰 때 자주 만나므로 중요한 기술입니다.

```r
#| label: setup
#| message: false
library(tidyverse)
library(repurrrsive)
library(jsonlite)
```

### 리스트

지금까지는 정수, 숫자, 문자, 날짜-시간, 요인(factors)처럼 단순한 벡터를 담은 데이터 프레임을 다뤘습니다. 이런 벡터는 모든 요소의 데이터 유형이 같은 동질적(homogeneous) 구조입니다. 한 벡터에 서로 다른 유형의 요소를 저장하려면 `list()`로 만드는 리스트(list)가 필요합니다.

```r
x1 <- list(1:4, "a", TRUE)
x1
```

티블 열에 이름을 붙이듯 리스트의 구성 요소(components), 즉 자식(children)에도 이름을 붙이면 편리합니다.

```r
x2 <- list(a = 1:2, b = 1:3, c = 1:4)
x2
```

아주 단순한 리스트도 그대로 출력하면 공간을 꽤 많이 차지합니다. `str()`은 내용을 줄이고 구조를 간결하게 보여주는 유용한 대안입니다.

```r
str(x1)
str(x2)
```

`str()`은 리스트의 자식을 한 줄에 하나씩 표시합니다.
이름이 있다면 이름과 유형의 약어(abbreviation), 처음 몇 개 값도 함께 표시합니다.

#### 계층 구조

리스트에는 다른 리스트를 비롯해 어떤 유형의 객체든 담을 수 있습니다. 그래서 계층형, 즉 트리 구조를 나타내기에 알맞습니다.

```r
x3 <- list(list(1, 2), list(3, 4))
str(x3)
```

이것은 평면(flat) 벡터를 생성하는 `c()`와 현저하게(notably) 다릅니다.

```r
c(c(1, 2), c(3, 4))

x4 <- c(list(1, 2), list(3, 4))
str(x4)
```

리스트가 복잡해질수록 계층 구조를 한눈에 보여주는 `str()`이 더 유용합니다.

```r
x5 <- list(1, list(2, list(3, list(4, list(5)))))
str(x5)
```

리스트가 더 크고 복잡해져 `str()`으로 보기 어려워지면 `View()`로 전환합니다.

뷰어(viewer)는 처음에 리스트의 최상위 수준만 표시합니다. 구성 요소를 대화형으로 펼치며 세부 내용을 확인합니다.

#### 리스트 열

리스트는 티블 안에도 들어갑니다. 이를 리스트-열(list-columns)이라고 합니다. 보통 티블에 넣기 어려운 객체까지 열에 담아서 유용합니다.

특히 리스트-열은 모델 출력이나 재샘플(resamples) 같은 객체를 데이터 프레임에 저장해 줍니다. 그래서 [tidymodels]([https://www.tidymodels.org](https://www.tidymodels.org)) 생태계에서 많이 사용됩니다.

리스트 열의 간단한 예시(simple example)는 다음과 같습니다.

```r
df <- tibble(
  x = 1:2, 
  y = c("a", "b"),
  z = list(list(1, 2), list(3, 4, 5))
)
df
```

티블의 리스트도 다른 열과 똑같이 작동합니다.

```r
df |> 
  filter(x == 1)
```

리스트-열 계산이 더 까다로운 까닭은 리스트 자체를 계산하기가 어렵기 때문입니다. 

기본 출력 메서드는 내용을 대략적으로만 요약합니다. 리스트-열은 얼마든지 복잡해질 수 있어 깔끔하게 출력할 방법이 없습니다.
자세히 보려면 리스트-열 하나를 꺼내 위에서 배운 방법을 적용하세요(`df |> pull(z) |> str()` 또는 `df |> pull(z) |> View()`).

#### 중첩 해제

리스트와 리스트-열의 기초를 익혔으니 일반적인 행과 열로 되돌리는 방법을 살펴봅시다. 

리스트-열은 크게 이름이 있는(named) 형태와 없는(unnamed) 형태로 나뉩니다.

자식에 이름이 있으면 보통 모든 행에서 같은 이름을 씁니다. `df1`의 리스트-열 `y`에는 행마다 `a`와 `b`라는 요소가 있습니다. 이름이 있는 리스트-열은 열 방향으로 펼치는 것이 자연스럽습니다. 이름이 붙은 요소마다 새 열이 하나씩 생깁니다.

```r
df1 <- tribble(
  ~x, ~y,
  1, list(a = 11, b = 12),
  2, list(a = 21, b = 22),
  3, list(a = 31, b = 32),
)
```

자식에 이름이 없으면 요소 수가 행마다 다를 수 있습니다. `df2`의 리스트-열 `y`는 이름 없는 요소를 담고 있으며 길이가 1~3개로 제각각입니다. 이런 리스트-열은 행 방향으로 펼치는 편이 자연스럽습니다. 자식마다 행 하나가 만들어집니다.

```r
df2 <- tribble(
  ~x, ~y,
  1, list(11, 12, 13),
  2, list(21),
  3, list(31, 32),
)
```

tidyr에는 두 경우에 맞춘 `unnest_wider()`와 `unnest_longer()`가 있습니다.

#### `unnest_wider()`

`df1`처럼 행마다 같은 이름의 요소가 같은 수만큼 있다면 `unnest_wider()`로 구성 요소를 각각의 열에 넣습니다.

```r
df1 |> 
  unnest_wider(y)
```

기본적으로 새 열 이름은 리스트 요소의 이름을 그대로 따릅니다. `names_sep` 인수를 쓰면 원래 열 이름과 요소 이름을 합칠 수 있습니다. 같은 이름이 반복될 때 구분하기 좋습니다.

```r
df1 |> 
  unnest_wider(y, names_sep = "_")
```

#### `unnest_longer()`

행마다 이름 없는 리스트가 들어 있다면 `unnest_longer()`로 요소를 각각의 행에 넣습니다.

```r
df2 |> 
  unnest_longer(y)
```

`y`의 요소마다 `x`가 복제되는 모습을 확인하세요. 리스트-열 안의 요소 하나당 출력 행 하나가 생깁니다. 그렇다면 다음 예처럼 요소 하나가 비어 있으면 어떻게 될까요?

```r
df6 <- tribble(
  ~x, ~y,
  "a", list(1, 2),
  "b", list(3),
  "c", list()
)
df6 |> unnest_longer(y)
```

빈 요소에서는 출력 행이 나오지 않아 원래 행도 사라집니다. `y`에 `NA`를 넣어 해당 행을 보존하려면 `keep_empty = TRUE`로 설정하세요.

#### 일관성 없는 유형

서로 다른 유형의 벡터가 든 리스트-열을 펼치면 어떻게 될까요? 다음 데이터셋의 리스트-열 `y`에는 숫자 두 개, 문자 하나, 논리값(logical) 하나가 있습니다. 보통은 한 열에 섞을 수 없는 유형입니다.

```r
df4 <- tribble(
  ~x, ~y,
  "a", list(1),
  "b", list("a", TRUE, 5)
)
```

`unnest_longer()`는 열 구성을 유지하면서 행 수를 바꿉니다. 그렇다면 `y`의 항목을 모두 보존한 채 행 다섯 개를 어떻게 만들까요?

```r
df4 |> 
  unnest_longer(y)
```

출력에는 여전히 리스트-열이 있습니다. 하지만 각 리스트에는 요소 하나만 들어 있습니다. `unnest_longer()`가 벡터의 공통 유형을 찾지 못해 원래 유형을 리스트-열에 보존한 것입니다. 열의 모든 요소가 같은 유형이어야 한다는 원칙을 어긴 것은 아닙니다. 내용물의 유형은 달라도 각 요소 자체는 모두 리스트입니다.

일관되지 않은 유형을 처리하기는 까다롭습니다. 구체적인 방법은 문제의 성격과 목표에 따라 달라집니다. 대개 반복과 순회 도구가 필요합니다.

#### 다른 함수들

tidyr에는 몇 가지 다른 유용한 직사각형화(rectangling) 함수가 있습니다.

1. `unnest_auto()`는 리스트-열 구조를 보고 `unnest_longer()`와 `unnest_wider()` 가운데 하나를 자동으로 고릅니다. 빠르게 탐색할 때는 유용하지만 최종 코드에는 권하지 않습니다. 데이터 구조를 이해하지 않고도 쓸 수 있는 데다 코드를 파악하기 어려워지기 때문입니다.

2. `unnest()`는 행과 열을 모두 펼칩니다. 이 책에서는 쓰지 않습니다. 하지만 데이터 프레임 같은 2차원 구조가 든 리스트-열에 유용합니다. [tidymodels]([https://www.tmwr.org/base-r.html#combining-base-r-models-and-the-tidyverse](https://www.tmwr.org/base-r.html#combining-base-r-models-and-the-tidyverse)) 생태계에서 만나게 될 수 있습니다.

다른 사람의 코드를 읽거나 드문 직사각형화 문제를 다룰 때 이런 함수를 만날 수 있으니 알아두면 좋습니다.

#### 연습 문제

1. `df2`와 같이 명명되지 않은 리스트-열과 함께 `unnest_wider()`를 사용하면 어떻게 됩니까? 이제 어떤 인수가 필요(necessary)합니까? 결측값(missing values)은 어떻게 됩니까?

2. `df1`과 같이 명명된 리스트-열과 함께 `unnest_longer()`를 사용하면 어떻게 됩니까? 출력에서 어떤 추가 정보(additional information)를 얻습니까? 그 추가적인(extra) 세부 정보를 억제(suppress)하려면 어떻게 해야 합니까?

3. 때로는 정렬된 값(aligned values)이 든 여러 리스트-열을 가진 데이터 프레임을 만납니다. 다음 데이터 프레임에서 `y`와 `z`의 값은 서로 정렬되어 있습니다. 두 열의 길이는 행 안에서 늘 같고 `y`의 첫 번째 값은 `z`의 첫 번째 값과 대응합니다. 이 데이터 프레임에 `unnest_longer()`를 두 번 적용하면 어떻게 됩니까? `x`와 `y`의 관계는 어떻게 보존합니까? (힌트: 문서(docs)를 주의 깊게 읽어보세요.)

```r
df4 <- tribble(
  ~x, ~y, ~z,
  "a", list("y-a-1", "y-a-2"), list("z-a-1", "z-a-2"),
  "b", list("y-b-1", "y-b-2", "y-b-3"), list("z-b-1", "z-b-2", "z-b-3")
)
```

#### 사례 연구

앞의 간단한 예시와 실제 데이터의 가장 큰 차이는 중첩 단계의 수입니다. 실제 데이터에는 보통 여러 단계가 있어 `unnest_longer()`나 `unnest_wider()`를 여러 번 호출해야 합니다.

repurrrsive 패키지의 데이터셋으로 실제 작동 방식을 살펴보겠습니다. 세 가지 직사각형화 과제를 차례로 풉니다.

##### 매우 넓은 데이터

`gh_repos`부터 시작합니다. GitHub API에서 가져온 리포지토리 모음(collection)을 담은 리스트입니다. 중첩이 너무 깊어 책에서 구조를 보여주기 어렵습니다. 계속하기 전에 `View(gh_repos)`로 직접 살펴보길 권합니다.

`gh_repos`는 리스트지만 여기서 쓸 도구는 리스트-열을 다룹니다. 먼저 리스트를 티블에 넣겠습니다. 뒤에서 설명할 이유로 열 이름은 `json`이라고 붙입니다.

```r
repos <- tibble(json = gh_repos)
repos
```

이 티블에는 `gh_repos`의 자식마다 행 하나씩, 모두 6개 행이 있습니다. 각 행은 요소 26개 또는 30개짜리 이름 없는 리스트를 담습니다. 이름이 없으므로 `unnest_longer()`로 자식을 각각의 행에 넣습니다.

```r
repos |> 
  unnest_longer(json)
```

언뜻 보면 나아진 점이 없어 보입니다. 행은 6개에서 176개로 늘었지만 `json`의 각 요소가 여전히 리스트입니다.
중요한 차이는 이제 요소마다 이름이 생겼다는 점입니다. `unnest_wider()`로 각 요소를 별도 열에 넣을 수 있습니다.

```r
repos |> 
  unnest_longer(json) |> 
  unnest_wider(json) 
```

작업은 잘됐지만 결과가 다소 압도적입니다. 열이 너무 많아 티블이 전부 출력하지 못합니다. `names()`로 전체 이름을 확인하며 여기서는 처음 10개만 봅니다.

```r
repos |> 
  unnest_longer(json) |> 
  unnest_wider(json) |> 
  names() |> 
  head(10)
```

흥미로워 보이는 몇 가지를 뽑아보겠습니다(pull out):

```r
repos |> 
  unnest_longer(json) |> 
  unnest_wider(json) |> 
  select(id, full_name, owner, description)
```

이 결과를 거꾸로 따라가면 `gh_repos`의 구조가 보입니다. 각 자식은 GitHub 사용자이며 사용자가 만든 리포지토리를 최대 30개까지 담고 있었습니다.

`owner`도 이름 있는 리스트를 담은 리스트-열입니다. `unnest_wider()`로 값을 펼칩니다.

```r
#| error: true
repos |> 
  unnest_longer(json) |> 
  unnest_wider(json) |> 
  select(id, full_name, owner, description) |> 
  unnest_wider(owner)
```

이 리스트-열에도 `id`가 있어 기존 `id`와 이름이 겹칩니다. 한 데이터 프레임에 같은 이름의 열을 둘 수 없으므로 안내대로 `names_sep`을 써서 해결해 봅시다.

```r
repos |> 
  unnest_longer(json) |> 
  unnest_wider(json) |> 
  select(id, full_name, owner, description) |> 
  unnest_wider(owner, names_sep = "_")
```

결과는 다시 넓은 데이터셋입니다. `owner`에 리포지토리 소유자에 관한 추가 정보가 많이 들어 있음을 알 수 있습니다.

##### 관계형 데이터

중첩 데이터는 여러 데이터 프레임에 흩어진 정보를 나타낼 때 자주 쓰입니다. 왕좌의 게임(Game of Thrones) 책과 TV 시리즈의 등장인물 데이터인 `got_chars`를 살펴봅시다. `gh_repos`처럼 리스트이므로 먼저 티블의 리스트-열로 바꿉니다.

```r
chars <- tibble(json = got_chars)
chars
```

`json` 열의 요소에는 이름이 있으므로 열 방향으로 펼치겠습니다.

```r
chars |> 
  unnest_wider(json)
```

읽기 쉽도록 열 몇 개만 고릅니다.

```r
characters <- chars |> 
  unnest_wider(json) |> 
  select(id, name, gender, culture, born, died, alive)
characters
```

이 데이터셋에는 여러(many) 리스트 열도 포함되어 있습니다.

```r
chars |> 
  unnest_wider(json) |> 
  select(id, where(is.list))
```

`titles` 열을 살펴봅시다. 이름 없는 리스트-열이므로 행 방향으로 펼칩니다.

```r
chars |> 
  unnest_wider(json) |> 
  select(id, titles) |> 
  unnest_longer(titles)
```

이 데이터는 별도 테이블로 두면 필요할 때 캐릭터 데이터에 쉽게 조인됩니다. 빈 문자열이 든 행을 빼고 각 행에 칭호가 하나씩만 남았으므로 `titles`를 `title`로 바꾸겠습니다.

```r
titles <- chars |> 
  unnest_wider(json) |> 
  select(id, titles) |> 
  unnest_longer(titles) |> 
  filter(titles != "") |> 
  rename(title = titles)
titles
```

각 리스트-열마다 이런 테이블을 만든 뒤 필요할 때 캐릭터 데이터와 조인하면 됩니다.

##### 깊은 중첩

마지막 사례는 아주 깊게 중첩된 리스트-열 `gmaps_cities`입니다. 구조를 풀려면 `unnest_wider()`와 `unnest_longer()`를 여러 차례 호출해야 합니다.

도시 이름 5개와 Google [geocoding API]([https://developers.google.com/maps/documentation/geocoding](https://developers.google.com/maps/documentation/geocoding))로 위치를 조회한 결과를 담은 2열짜리 티블입니다.

```r
gmaps_cities
```

`json`은 내부(internal) 이름이 있는 리스트-열이므로 `unnest_wider()`로 시작합니다.

```r
gmaps_cities |> 
  unnest_wider(json)
```

결과로 `status`와 `results`가 나옵니다. `status`가 모두 `OK`라 이 열은 삭제하겠습니다. 실제 분석이라면 `status != "OK"`인 행을 따로 모아 오류 원인을 확인해야 합니다. `results`는 요소가 한두 개인 이름 없는 리스트이므로 행 방향으로 펼칩니다. 도시마다 요소 수가 다른 이유는 곧 드러납니다.

```r
gmaps_cities |> 
  unnest_wider(json) |> 
  select(-status) |> 
  unnest_longer(results)
```

이제 `results`가 이름이 지정된(named) 리스트이므로 `unnest_wider()`를 사용합니다.

```r
locations <- gmaps_cities |> 
  unnest_wider(json) |> 
  select(-status) |> 
  unnest_longer(results) |> 
  unnest_wider(results)
locations
```

이제 두 도시에서 결과가 두 개씩 나온 이유를 알 수 있습니다. 워싱턴(Washington)은 워싱턴주와 워싱턴 DC에 모두 일치했습니다. 알링턴(Arlington)은 버지니아주와 텍사스주의 알링턴에 모두 일치했습니다.

여기서 여러 방향으로 분석을 이어갈 수 있습니다. 일치한 항목의 정확한 위치는 `geometry` 리스트-열에 저장되어 있습니다.

```r
locations |> 
  select(city, formatted_address, geometry) |> 
  unnest_wider(geometry)
```

새로 `bounds`(직사각형 영역)와 `location`(점) 열이 생겼습니다. 위도(latitude, `lat`)와 경도(longitude, `lng`)를 보려면 `location`을 펼칩니다.

```r
locations |> 
  select(city, formatted_address, geometry) |> 
  unnest_wider(geometry) |> 
  unnest_wider(location)
```

경계를 추출하려면 몇 단계를 더 거칩니다.

```r
locations |> 
  select(city, formatted_address, geometry) |> 
  unnest_wider(geometry) |> 
  # 관심 변수에 초점을 맞춤 (focus on the variables of interest)
  select(!location:viewport) |>
  unnest_wider(bounds)
```

그런 다음 직사각형 모서리를 뜻하는 `southwest`와 `northeast`의 이름을 바꿉니다. `names_sep`으로 짧고 의미가 분명한 이름을 만들기 위해서입니다.

```r
locations |> 
  select(city, formatted_address, geometry) |> 
  unnest_wider(geometry) |> 
  select(!location:viewport) |>
  unnest_wider(bounds) |> 
  rename(ne = northeast, sw = southwest) |> 
  unnest_wider(c(ne, sw), names_sep = "_") 
```

변수 이름 벡터를 `unnest_wider()`에 넣어 두 열을 한꺼번에 펼친 점을 눈여겨보세요.

필요한 구성 요소까지 이어지는 경로를 찾았다면 tidyr의 `hoist()`로 곧바로 추출합니다.

```r
#| results: false
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

실제 직사각형화 사례를 더 보고 싶다면 `vignette("rectangling", package = "tidyr")`를 참고하세요.

##### 연습 문제

1.`gh_repos`가 생성된 시기(when)를 대략 추정해 봅니다. 날짜를 어림잡을 수밖에 없는 이유는 무엇입니까?

2. `gh_repos`의 `owner` 열에는 소유자마다 리포지토리가 여러 개라 중복 정보가 많습니다. 소유자마다 행 하나만 담은 `owners` 데이터 프레임을 만들어 보세요. (힌트: `distinct()`가 `list-cols`와 함께 작동합니까?)

3.`titles`에 사용된 단계를 따라(Follow the steps) 왕좌의 게임 캐릭터의 별칭(aliases), 충성도(allegiances), 책(books) 및 TV 시리즈(TV series)에 대한 유사한 테이블을 만드세요.

4.다음 코드를 한 줄씩(line-by-line) 설명하세요. 이것이 왜 흥미롭습니까? 이것이 `got_chars`에는 작동하지만 일반적으로는 작동하지 않을 수 있는 이유는 무엇입니까?

```r
#| results: false
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

5. `gmaps_cities`에서 `address_components`에는 무엇이 포함되어 있습니까? 행마다(between rows) 길이가 다른 이유는 무엇입니까? 알아내기 위해(figure it out) 적절하게(appropriately) 중첩 해제하세요(Unnest it). (힌트: `types`는 항상 두 개의 요소를 포함하는 것처럼 보입니다. `unnest_wider()`가 `unnest_longer()`보다 작업하기 쉽게(easier to work with) 만들어 줍니까?)

### JSON

JSON은 javascript object notation의 약자입니다. 대부분의 웹 API가 데이터를 반환할 때 쓰는 형식입니다.

JSON과 R의 데이터 유형은 꽤 비슷하지만 완벽하게 일대일로 대응하지는 않습니다. 문제가 생겼을 때 원인을 찾으려면 JSON의 기본 구조를 알아두는 편이 좋습니다.

#### 데이터 유형

JSON은 기계가 쉽게 읽고 쓰도록 설계된 단순한 형식입니다.
핵심 데이터 유형은 여섯 가지입니다. 그중 네 가지는 스칼라(scalars)입니다.

1. 가장 단순한 유형은 null(`null`)입니다. R의 `NA`처럼 값이 없음을 나타냅니다.

2. 문자열(string)은 R의 문자열과 비슷하지만 언제나 큰따옴표(double quotes)를 써야 합니다.

3. 숫자(number)는 R과 비슷합니다. 정수(123), 소수(decimal, 예: 123.45), 과학적 표기법(scientific, 예: 1.23e3)을 쓸 수 있습니다. 단, JSON은 `Inf`, `-Inf`, `NaN`을 지원하지 않습니다.

4. 부울(boolean)은 R의 `TRUE`, `FALSE`와 비슷하지만 소문자 `true`, `false`로 적습니다.

JSON의 문자열, 숫자, 부울은 R의 문자, 숫자, 논리(logical) 벡터와 상당히 비슷합니다. 큰 차이는 JSON의 스칼라가 값 하나만 나타낸다는 점입니다. 여러 값을 담으려면 나머지 두 유형인 배열(arrays)이나 객체(objects)를 써야 합니다.

배열과 객체는 모두 R의 리스트와 닮았습니다. 요소에 이름이 있는지 없는지가 둘의 차이입니다. 배열(array)은 이름 없는 리스트와 같고 `[]`로 씁니다.
예를 들어 `[1, 2, 3]`은 3개의 숫자를 포함하는 배열이고 `[null, 1, "string", false]`는 null, 숫자, 문자열 및 부울을 포함하는 배열입니다.

객체(object)는 이름 있는 리스트와 같으며 `{}`로 씁니다. JSON에서 이름은 키(keys)라고 하며 문자열이므로 따옴표로 묶어야 합니다. 예를 들어 `{"x": 1, "y": 2}`는 `x`를 1에, `y`를 2에 대응시킨 객체입니다.

JSON에는 날짜(dates)나 날짜-시간(date-times)을 나타내는 기본 유형이 없습니다. 흔히 문자열로 저장되므로 `readr::parse_date()`나 `readr::parse_datetime()`으로 알맞은 데이터 구조로 바꿔야 합니다.

부동 소수점 숫자(floating point numbers)를 나타내는 규칙도 조금 부정확해서 숫자가 문자열로 저장된 경우가 있습니다.

필요하면 `readr::parse_double()`을 적용해 올바른 변수 유형으로 바꿉니다.

#### jsonlite

JSON을 R 데이터 구조로 바꿀 때는 Jeroen Ooms의 jsonlite 패키지를 권합니다.
여기서는 jsonlite의 `read_json()`과 `parse_json()`만 사용합니다.
실제 작업에서는 디스크의 JSON 파일을 `read_json()`으로 읽습니다.
예를 들어 repurrsive 패키지는 `gh_user`의 원본을 JSON 파일로도 제공하므로 `read_json()`으로 읽을 수 있습니다.

```r
# 패키지 내부의 json 파일에 대한 경로:
gh_users_json()

# read_json()으로 읽기
gh_users2 <- read_json(gh_users_json())

# 이전에 사용하던 데이터와 동일한지 확인
identical(gh_users, gh_users2)
```

이 책에서는 JSON 문자열로 간단한 예제를 만들기 좋아 `parse_json()`도 사용합니다.

다음은 간단한 JSON 데이터셋 세 가지입니다. 숫자 하나에서 시작해 숫자 배열을 만듭니다. 마지막에는 그 배열을 객체에 넣습니다.

```r
str(parse_json('1'))
str(parse_json('[1, 2, 3]'))
str(parse_json('{"x": [1, 2, 3]}'))
```

jsonlite에는 `fromJSON()`이라는 또 다른 중요한 함수가 있습니다.
자동 단순화(automatic simplification)(`simplifyVector = TRUE`)를 수행하므로 여기서는 사용하지 않습니다.

간단한 데이터에서는 이 기능이 잘 작동합니다. 그래도 직사각형화는 직접 하는 편이 낫습니다. 어떤 변환이 일어나는지 정확히 알게 되고 복잡한 중첩 구조도 더 쉽게 처리하기 때문입니다.

#### 직사각형화 과정 시작하기

대부분의 JSON 파일에는 최상위 배열 하나가 있습니다. 여러 페이지나 레코드, 결과처럼 여러 항목을 제공하도록 설계됐기 때문입니다.

이때는 요소마다 행 하나가 생기도록 `tibble(json)`으로 직사각형화를 시작합니다.

```r
json <- '[
  {"name": "John", "age": 34},
  {"name": "Susan", "age": 27}
]'
df <- tibble(json = parse_json(json))
df

df |> 
  unnest_wider(json)
```

드물게 JSON 파일이 항목 하나를 나타내는 최상위 객체 하나로만 구성되기도 합니다.

이때는 객체를 리스트로 감싼 뒤 티블에 넣어 직사각형화를 시작합니다.

```r
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

df |> 
  unnest_wider(json) |> 
  unnest_longer(results) |> 
  unnest_wider(results)
```

또는 파싱한 JSON 안으로 들어가 실제로 필요한 부분부터 시작할 수도 있습니다.

```r
df <- tibble(results = parse_json(json)$results)
df |> 
  unnest_wider(results)
```

#### 연습 문제

1. 아래의 `df_col`과 `df_row`를 직사각형화(Rectangle)하세요. 이들은 JSON에서 데이터 프레임을 인코딩(encoding)하는 두 가지 방법을 나타냅니다(represent).

```r
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

<!-- HUMANIZE-SUMMARY
원본 글자수: 83,971자
윤문본 글자수: 68,967자
변경률: 20.6% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 38 → 0
- A-7 가지다 직역: 6 → 0
- A-10 가능 표현 남발: 93 → 0
- A-11 목적절 남발: 16 → 0
- A-15 본문 추상 주어·만능 동사: 12 → 0
- C-11 연결어미 뒤 쉼표: 49 → 0

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
- "스프레드시트 데이터로 작업할 때 추가로 고려해야 할 사항" → "스프레드시트 데이터를 다룰 때 따로 살펴야 할 점"
- "데이터 과학은 반복적인 과정이며" → "데이터 과학은 본래 반복적인 과정입니다"
- "엄청난 양의 데이터가 데이터베이스에 존재하므로" → "엄청난 양의 데이터가 데이터베이스에 들어 있습니다"
- "웹 스크래핑을 수행하는 데 필요한 코드에 대해 논의하기 전에" → "코드를 살펴보기 전에 웹 스크래핑의 합법성과 윤리부터 짚어야 합니다"
- "직사각형화 과제를 해결해 봅니다" → "세 가지 직사각형화 과제를 차례로 풉니다"
-->
