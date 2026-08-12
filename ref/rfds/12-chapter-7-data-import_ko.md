# 7장. 데이터 가져오기(Data Import)

# 소개 (Introduction)

R 패키지에서 제공하는 데이터로 작업하는 것은 데이터 과학 도구를 배우는 좋은 방법이지만, 어느 시점에는 배운 내용을 자신의 데이터에 적용하고 싶어질 것입니다. 이 장에서는 R로 데이터 파일을 읽어들이는 기본 사항을 배울 것입니다.

구체적으로 이 장에서는 일반 텍스트로 된 직사각형 파일(plain-text rectangular files)을 읽는 데 중점을 둘 것입니다. 열 이름, 유형 및 누락된 데이터와 같은 특징을 처리하기 위한 실용적인 조언부터 시작하겠습니다. 그런 다음 한 번에 여러 파일에서 데이터를 읽고 R에서 파일로 데이터를 쓰는 방법에 대해 배울 것입니다. 마지막으로 R에서 데이터 프레임을 직접 수작업으로 만드는 방법을 배울 것입니다.

## 사전 준비 (Prerequisites)

이 장에서는 핵심 tidyverse의 일부인 readr 패키지를 사용하여 R에서 플랫 파일(flat files)을 로드하는 방법을 배웁니다.

`library``(``tidyverse``)`

# 파일에서 데이터 읽기 (Reading Data from a File)

시작하기 위해 일반적인 직사각형 데이터 파일 유형인 CSV(comma-separated values의 약자)에 중점을 두겠습니다. 다음은 간단한 CSV 파일의 모습입니다. 보통 *헤더 행(header row)*이라고 불리는 첫 번째 행에는 열 이름이 있고, 다음 6개 행에는 데이터가 있습니다. 열은 쉼표로 구분(또는 _delimited_)됩니다.

`Student ID,Full Name,favourite.food,mealPlan,AGE 1,Sunil Huffmann,Strawberry yoghurt,Lunch only,4 2,Barclay Lynn,French fries,Lunch only,5 3,Jayendra Lyne,N/A,Breakfast and lunch,7 4,Leon Rossini,Anchovies,Lunch only, 5,Chidiegwu Dunkel,Pizza,Breakfast and lunch,five 6,Güvenç Attila,Ice cream,Lunch only,6`

<a href="#tbl-students-table" data-type="xref">표 7-1</a>은 동일한 데이터를 표 형태로 나타낸 것입니다.

| Student ID | Full Name        | favourite.food     | mealPlan            | AGE  |
| ---------: | :--------------- | :----------------- | :------------------ | :--- |
|          1 | Sunil Huffmann   | Strawberry yoghurt | Lunch only          | 4    |
|          2 | Barclay Lynn     | French fries       | Lunch only          | 5    |
|          3 | Jayendra Lyne    | N/A                | Breakfast and lunch | 7    |
|          4 | Leon Rossini     | Anchovies          | Lunch only          | NA   |
|          5 | Chidiegwu Dunkel | Pizza              | Breakfast and lunch | five |
|          6 | Güvenç Attila    | Ice cream          | Lunch only          | 6    |

표 7-1. students.csv 파일의 데이터를 테이블로 나타낸 것 {#tbl-students-table .table .table-sm .table-striped}

<a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>를 사용하여 이 파일을 R로 읽어들일 수 있습니다. 첫 번째 인수가 중요한데, 바로 파일의 경로(path)입니다. 경로를 파일의 주소라고 생각할 수 있습니다. 파일의 이름은 `students.csv`이고 `data` 폴더 안에 있습니다.

`students` `<-` `read_csv``(``"data/students.csv"``)` `#> Rows: 6 Columns: 5` `#> ── Column specification ─────────────────────────────────────────────────────` `#> Delimiter: ","` `#> chr (4): Full Name, favourite.food, mealPlan, AGE` `#> dbl (1): Student ID` `#> ` ``#> ℹ Use `spec()` to retrieve the full column specification for this data.`` ``#> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.``

프로젝트의 `data` 폴더에 `students.csv` 파일이 있다면 위 코드가 작동할 것입니다. [`students.csv` 파일](https://oreil.ly/GDubb)을 다운로드하거나 다음 코드를 사용하여 해당 URL에서 직접 읽을 수 있습니다.

`students` `<-` `read_csv``(``"https://pos.it/r4ds-students-csv"``)`

<a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>를 실행하면 데이터의 행과 열 수, 사용된 구분 기호, 그리고 열 명세(열에 포함된 데이터 유형별로 구성된 열 이름)를 알려주는 메시지가 출력됩니다. 전체 열 명세를 검색하는 방법과 이 메시지를 숨기는 방법에 대한 일부 정보도 인쇄됩니다. 이 메시지는 readr의 핵심적인 부분이므로 <a href="#sec-col-types" data-type="xref">“열 유형 제어하기(Controlling Column Types)”</a>에서 다시 다룰 것입니다.

## 실용적인 조언 (Practical Advice)

데이터를 읽어온 후 첫 번째 단계는 대개 남은 분석 과정에서 더 쉽게 작업할 수 있도록 데이터를 어떤 방식으로든 변환하는 것입니다. 이를 염두에 두고 `students` 데이터를 다시 살펴보겠습니다.

`students` `#> # A tibble: 6 × 5` ``#> `Student ID` `Full Name` favourite.food mealPlan AGE `` `#> <dbl> <chr> <chr> <chr> <chr>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4 ` `#> 2 2 Barclay Lynn French fries Lunch only 5 ` `#> 3 3 Jayendra Lyne N/A Breakfast and lunch 7 ` `#> 4 4 Leon Rossini Anchovies Lunch only <NA> ` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch five ` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

`favourite.food` 열에는 여러 음식 항목이 있고, 그 다음에 문자열 `N/A`가 있는데, 이것은 R이 "이용할 수 없음(not available)"으로 인식하는 진짜 `NA`였어야 합니다. 이는 `na` 인수를 사용하여 해결할 수 있습니다. 기본적으로 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>는 이 데이터 세트의 빈 문자열(`""`)만을 `NA`로 인식합니다. 우리는 문자열 `"N/A"`도 인식하도록 만들고 싶습니다.

`students` `<-` `read_csv``(``"data/students.csv"``,` `na` `=` `c``(``"N/A"``,` `""``))` `students` `#> # A tibble: 6 × 5` ``#> `Student ID` `Full Name` favourite.food mealPlan AGE `` `#> <dbl> <chr> <chr> <chr> <chr>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4 ` `#> 2 2 Barclay Lynn French fries Lunch only 5 ` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7 ` `#> 4 4 Leon Rossini Anchovies Lunch only <NA> ` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch five ` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

또한 `Student ID`와 `Full Name` 열이 백틱(backticks)으로 둘러싸여 있다는 것을 눈치채셨을 것입니다. 이는 공백이 포함되어 있어 변수 이름에 대한 R의 일반적인 규칙을 위반하기 때문입니다. 이러한 이름들은 _비문법적(nonsyntactic)_ 이름입니다. 이러한 변수를 참조하려면 변수를 백틱(`` ` ``)으로 둘러싸야 합니다.

`students` `|>` `rename``(` `student_id` `=` `` `Student ID` ```,` `full_name` `=` `` `Full Name` `` `)` `#> # A tibble: 6 × 5` `#> student_id full_name favourite.food mealPlan AGE ` `#> <dbl> <chr> <chr> <chr> <chr>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4 ` `#> 2 2 Barclay Lynn French fries Lunch only 5 ` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7 ` `#> 4 4 Leon Rossini Anchovies Lunch only <NA> ` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch five ` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

또 다른 대안적 접근 방식은 <a href="https://rdrr.io/pkg/janitor/man/clean_names.html" class="orm:hideurl"><code>janitor::clean_names()</code></a>를 사용하여 일종의 휴리스틱(heuristics)으로 모두 한 번에 스네이크 케이스(snake case)로 바꾸는 것입니다.<sup><a href="ch07.html#idm44771312718064" id="idm44771312718064-marker" data-type="noteref">1</a></sup>

`students` `|>` `janitor``::``clean_names``()` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age ` `#> <dbl> <chr> <chr> <chr> <chr>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4 ` `#> 2 2 Barclay Lynn French fries Lunch only 5 ` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7 ` `#> 4 4 Leon Rossini Anchovies Lunch only <NA> ` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch five ` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

데이터를 읽어들인 후 흔히 하는 또 다른 작업은 변수 유형을 고려하는 것입니다. 예를 들어, `meal_plan`은 알려진 가능한 값 집합을 가진 범주형 변수이며, R에서는 요인(factor)으로 표현되어야 합니다.

`students` `|>` `janitor``::``clean_names``()` `|>` `mutate``(``meal_plan` `=` `factor``(``meal_plan``))` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age ` `#> <dbl> <chr> <chr> <fct> <chr>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4 ` `#> 2 2 Barclay Lynn French fries Lunch only 5 ` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7 ` `#> 4 4 Leon Rossini Anchovies Lunch only <NA> ` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch five ` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

`meal_plan` 변수의 값은 동일하게 유지되었지만 변수 이름 아래에 표시된 변수 유형이 문자(`<chr>`)에서 요인(`<fct>`)으로 변경된 것을 확인하세요. 요인에 대해서는 <a href="ch16.html#chp-factors" data-type="xref">16장</a>에서 자세히 알아볼 것입니다.

이 데이터를 분석하기 전에 `age`와 `id` 열을 수정하고 싶을 것입니다. 현재 `age`는 문자 변수인데, 관측치 중 하나가 숫자 `5` 대신 문자 `five`로 입력되어 있기 때문입니다. 이 문제를 해결하는 자세한 내용은 <a href="ch20.html#chp-spreadsheets" data-type="xref">20장</a>에서 논의합니다.

`students` `<-` `students` `|>` `janitor``::``clean_names``()` `|>` `mutate``(` `meal_plan` `=` `factor``(``meal_plan``),` `age` `=` `parse_number``(``if_else``(``age` `==` `"five"``,` `"5"``,` `age``))` `)` `students` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age` `#> <dbl> <chr> <chr> <fct> <dbl>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4` `#> 2 2 Barclay Lynn French fries Lunch only 5` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7` `#> 4 4 Leon Rossini Anchovies Lunch only NA` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch 5` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

여기서 새로운 함수는 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>로, 세 개의 인수를 가집니다. 첫 번째 인수 `test`는 논리형 벡터(logical vector)여야 합니다. 결과는 `test`가 `TRUE`일 때는 두 번째 인수 `yes`의 값을, `FALSE`일 때는 세 번째 인수 `no`의 값을 포함하게 됩니다. 여기서 우리는 `age`가 문자열 `"five"`인 경우 `"5"`로 만들고, 그렇지 않으면 `age` 그대로 남겨두라고 지시하는 것입니다. <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>와 논리 벡터에 대해서는 <a href="ch12.html#chp-logicals" data-type="xref">12장</a>에서 자세히 배울 것입니다.

## 기타 인수 (Other Arguments)

언급해야 할 몇 가지 중요한 다른 인수가 있는데, 유용한 요령을 먼저 보여드리면 설명하기가 더 쉬울 것입니다. <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>는 CSV 파일처럼 작성하고 포맷한 텍스트 문자열을 읽을 수 있습니다.

`read_csv``(` `"a,b,c` ` 1,2,3` ` 4,5,6"` `)` `#> # A tibble: 2 × 3` `#> a b c` `#> <dbl> <dbl> <dbl>` `#> 1 1 2 3` `#> 2 4 5 6`

일반적으로 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>는 일반적인 규칙에 따라 데이터의 첫 번째 줄을 열 이름으로 사용합니다. 하지만 파일의 위쪽에 몇 줄의 메타데이터가 포함되어 있는 경우도 드물지 않습니다. `skip = n`을 사용하여 처음 `n` 줄을 건너뛰거나, `comment = "#"`을 사용하여 `#`로 시작하는 모든 줄을 삭제할 수 있습니다.

`read_csv``(` `"The first line of metadata` ` The second line of metadata` ` x,y,z` ` 1,2,3"``,` `skip` `=` `2` `)` `#> # A tibble: 1 × 3` `#> x y z` `#> <dbl> <dbl> <dbl>` `#> 1 1 2 3` `read_csv``(` `"# A comment I want to skip` ` x,y,z` ` 1,2,3"``,` `comment` `=` `"#"` `)` `#> # A tibble: 1 × 3` `#> x y z` `#> <dbl> <dbl> <dbl>` `#> 1 1 2 3`

데이터에 열 이름이 없는 경우도 있습니다. `col_names = FALSE`를 사용하여 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>에게 첫 번째 행을 제목으로 처리하지 말고 대신 `X1`에서 `Xn`까지 순차적으로 레이블을 지정하도록 지시할 수 있습니다.

`read_csv``(` `"1,2,3` ` 4,5,6"``,` `col_names` `=` `FALSE` `)` `#> # A tibble: 2 × 3` `#> X1 X2 X3` `#> <dbl> <dbl> <dbl>` `#> 1 1 2 3` `#> 2 4 5 6`

또는 `col_names`에 문자형 벡터를 전달하여 이를 열 이름으로 사용할 수도 있습니다.

`read_csv``(` `"1,2,3` ` 4,5,6"``,` `col_names` `=` `c``(``"x"``,` `"y"``,` `"z"``)` `)` `#> # A tibble: 2 × 3` `#> x y z` `#> <dbl> <dbl> <dbl>` `#> 1 1 2 3` `#> 2 4 5 6`

이러한 인수들만 알고 있으면 실무에서 접하게 될 대부분의 CSV 파일을 읽어 들일 수 있습니다. (나머지의 경우에는 `.csv` 파일을 주의 깊게 검사하고 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code>의</a> 수많은 기타 인수에 대한 문서를 읽어야 합니다.)

## 다른 파일 유형 (Other File Types)

<a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>를 마스터하고 나면 readr의 다른 함수들을 사용하는 것은 간단합니다. 어떤 함수를 사용할지만 알면 됩니다.

<a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv2()</code></a>  
세미콜론으로 구분된 파일을 읽습니다. 이 파일들은 쉼표(`,`) 대신 세미콜론(`;`)을 사용하여 필드를 구분하며, 소수점 마커로 쉼표(`,`)를 사용하는 국가에서 흔히 볼 수 있습니다.

<a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_tsv()</code></a>  
탭(tab)으로 구분된 파일을 읽습니다.

<a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_delim()</code></a>  
모든 구분 기호로 구분된 파일을 읽으며, 구분 기호를 지정하지 않으면 자동으로 추측하려고 시도합니다.

<a href="https://readr.tidyverse.org/reference/read_fwf.html" class="orm:hideurl"><code>read_fwf()</code></a>  
고정 너비(fixed-width) 파일을 읽습니다. <a href="https://readr.tidyverse.org/reference/read_fwf.html" class="orm:hideurl"><code>fwf_widths()</code></a>로 필드의 너비를 지정하거나 <a href="https://readr.tidyverse.org/reference/read_fwf.html" class="orm:hideurl"><code>fwf_positions()</code></a>로 필드의 위치를 지정할 수 있습니다.

<a href="https://readr.tidyverse.org/reference/read_table.html" class="orm:hideurl"><code>read_table()</code></a>  
고정 너비 파일의 흔한 변형으로 열이 공백으로 구분된 파일을 읽습니다.

<a href="https://readr.tidyverse.org/reference/read_log.html" class="orm:hideurl"><code>read_log()</code></a>  
Apache 스타일 로그 파일을 읽습니다.

## 연습문제

1.  필드가 \|로 구분된 파일을 읽으려면 어떤 함수를 사용하겠습니까?

2.  `file`, `skip`, `comment` 외에 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>와 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_tsv()</code></a>가 공통적으로 가지는 다른 인수는 무엇입니까?

3.  <a href="https://readr.tidyverse.org/reference/read_fwf.html" class="orm:hideurl"><code>read_fwf()</code></a>의 중요한 인수는 무엇입니까?

4.  때때로 CSV 파일의 문자열에 쉼표가 포함되어 있습니다. 이것이 문제를 일으키는 것을 방지하기 위해 `"` 또는 `'`와 같은 인용 문자(quoting character)로 둘러싸여 있어야 합니다. 기본적으로 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>는 인용 문자가 `"`일 것이라고 가정합니다. 다음 텍스트를 데이터 프레임으로 읽어 들이려면 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>에 어떤 인수를 지정해야 합니까?

    `"x,y\n1,'a,b'"`

5.  다음의 각 인라인 CSV 파일에서 무엇이 잘못되었는지 식별하세요. 코드를 실행하면 어떻게 됩니까?

    `read_csv``(``"a,b\n1,2,3\n4,5,6"``)` `read_csv``(``"a,b,c\n1,2\n1,2,3,4"``)` `read_csv``(``"a,b\n\"1"``)` `read_csv``(``"a,b\n1,2\na,b"``)` `read_csv``(``"a;b\n1;3"``)`

6.  다음 데이터 프레임에서 비문법적(nonsyntactic) 이름을 참조하는 연습을 하세요. 다음의 작업을 수행합니다.
    1.  `1`이라는 변수를 추출합니다.
    2.  `1` 대 `2`의 산점도(scatterplot)를 그립니다.
    3.  `2`를 `1`로 나눈 값인 `3`이라는 새 열을 생성합니다.
    4.  열의 이름을 `one`, `two`, `three`로 변경합니다.

    `annoying` `<-` `tibble``(` `` `1` `` `=` `1``:``10``,` `` `2` `` `=` `` `1` `` `*` `2` `+` `rnorm``(``length``(``` `1` ```))` `)`

# 열 유형 제어하기 (Controlling Column Types)

CSV 파일에는 각 변수의 유형(논리형, 숫자형, 문자열 등)에 대한 정보가 포함되어 있지 않으므로 readr이 그 유형을 추측하려고 시도합니다. 이 섹션에서는 추측 프로세스가 어떻게 작동하는지, 이 프로세스가 실패하게 만드는 몇 가지 일반적인 문제를 해결하는 방법, 필요한 경우 열 유형을 직접 제공하는 방법에 대해 설명합니다. 마지막으로 readr이 치명적으로 실패하여 파일 구조에 대한 더 깊은 통찰력이 필요한 경우 유용한 몇 가지 일반적인 전략에 대해 언급하겠습니다.

## 유형 추측하기 (Guessing Types)

readr은 열 유형을 파악하기 위해 휴리스틱을 사용합니다. 각 열에 대해 결측치를 무시하고 첫 번째 행부터 마지막 행까지 일정한 간격으로 1,000개<sup><a href="ch07.html#idm44771312159040" id="idm44771312159040-marker" data-type="noteref">2</a></sup> 행의 값을 가져옵니다. 그런 다음 다음 질문을 통해 확인 작업을 수행합니다.

- 대소문자 구분 없이 `F`, `T`, `FALSE`, `TRUE`만 포함되어 있습니까? 그렇다면 논리형(logical)입니다.
- 숫자(`1`, `-4.5`, `5e6`, `Inf`)만 포함되어 있습니까? 그렇다면 숫자형(number)입니다.
- ISO8601 표준과 일치합니까? 그렇다면 날짜(date) 또는 날짜-시간(date-time)입니다. (날짜-시간에 대해서는 <a href="ch17.html#sec-creating-datetimes" data-type="xref">"날짜/시간 생성하기"</a>에서 더 자세히 다룰 것입니다.)
- 그렇지 않으면 문자열(string)이어야 합니다.

이 간단한 예제에서 그 동작을 실제로 볼 수 있습니다.

`read_csv``(``"` ` logical,numeric,date,string` ` TRUE,1,2021-01-15,abc` ` false,4.5,2021-02-15,def` ` T,Inf,2021-02-16,ghi` `"``)` `#> # A tibble: 3 × 4` `#> logical numeric date string` `#> <lgl> <dbl> <date> <chr> ` `#> 1 TRUE 1 2021-01-15 abc ` `#> 2 FALSE 4.5 2021-02-15 def ` `#> 3 TRUE Inf 2021-02-16 ghi`

이러한 휴리스틱은 깔끔한 데이터셋이 있을 때 잘 작동하지만, 실제로는 기이하고 다채로운 여러 실패 사례와 마주하게 될 것입니다.

## 결측치, 열 유형 및 문제 (Missing Values, Column Types, and Problems)

열 감지가 실패하는 흔한 방식은 열에 예상치 못한 값이 포함되어 있어서, 더 구체적인 유형 대신 문자(character) 열을 얻게 되는 경우입니다. 이에 대한 흔한 원인 중 하나는 readr이 예상하는 `NA`가 아닌 다른 어떤 것을 사용하여 기록된 결측치(missing value)입니다.

단일 열로 된 이 간단한 CSV 파일을 예로 들어 보겠습니다.

`simple_csv` `<-` `"` ` x` ` 10` ` .` ` 20` ` 30"`

아무런 추가 인수 없이 이를 읽어들이면, `x`는 문자 열이 됩니다.

`read_csv``(``simple_csv``)` `#> # A tibble: 4 × 1` `#> x ` `#> <chr>` `#> 1 10 ` `#> 2 . ` `#> 3 20 ` `#> 4 30`

이 작은 예제에서는 결측치 `.`을 쉽게 볼 수 있습니다. 하지만 수천 개의 행 사이에 `.`로 표시된 결측치가 몇 개 흩어져 있다면 어떻게 될까요? 한 가지 방법은 readr에게 `x`가 숫자형(numeric) 열이라고 알려준 다음 어디서 실패하는지 보는 것입니다. CSV 파일의 열 이름과 일치하는 이름을 가진 리스트(named list)를 취하는 `col_types` 인수를 사용하여 그렇게 할 수 있습니다.

`df` `<-` `read_csv``(` `simple_csv``,` `col_types` `=` `list``(``x` `=` `col_double``())` `)` ``#> Warning: One or more parsing issues, call `problems()` on your data frame for`` `#> details, e.g.:` `#> dat <- vroom(...)` `#> problems(dat)`

이제 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>는 문제가 있었다고 보고하며 <a href="https://readr.tidyverse.org/reference/problems.html" class="orm:hideurl"><code>problems()</code></a>를 통해 더 자세히 알아볼 수 있다고 알려줍니다.

`problems``(``df``)` `#> # A tibble: 1 × 5` `#> row col expected actual file ` `#> <int> <int> <chr> <chr> <chr> ` `#> 1 3 1 a double . /private/tmp/RtmpAYlSop/file392d445cf269`

이것은 3행 1열에서 문제가 발생했음을 알려줍니다. 이 위치에서 readr은 배정밀도 실수(double)를 예상했지만 `.`을 얻었습니다. 이는 이 데이터셋이 결측치에 `.`을 사용한다는 것을 시사합니다. 따라서 `na = "."`로 설정하면 자동 추측이 성공하여 우리가 원하는 숫자형 열을 얻을 수 있습니다.

`read_csv``(``simple_csv``,` `na` `=` `"."``)` `#> # A tibble: 4 × 1` `#> x` `#> <dbl>` `#> 1 10` `#> 2 NA` `#> 3 20` `#> 4 30`

## 열 유형 (Column Types)

readr은 여러분이 사용할 수 있도록 총 9가지 열 유형을 제공합니다.

- <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>col_logical()</code></a>과 <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>col_double()</code></a>은 논리형과 실수를 읽습니다. 이들은 readr이 보통 여러분을 위해 자동으로 추측해 주므로 (앞서 보여드린 경우를 제외하고) 비교적 드물게 필요합니다.
- <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>col_integer()</code></a>는 정수를 읽습니다. 정수와 배정밀도 실수는 기능적으로 동등하기 때문에 이 책에서는 이를 거의 구별하지 않지만, 정수를 명시적으로 읽는 것은 배정밀도 실수의 절반에 해당하는 메모리만 차지하기 때문에 때때로 유용할 수 있습니다.
- <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>col_character()</code></a>는 문자열을 읽습니다. 숫자로 된 식별자, 즉 객체를 식별하지만 수학 연산을 적용하는 것이 무의미한 긴 숫자 시리즈인 열이 있을 때 명시적으로 지정하는 데 유용할 수 있습니다. 예로는 전화번호, 사회 보장 번호, 신용카드 번호 등이 있습니다.
- <a href="https://readr.tidyverse.org/reference/parse_factor.html" class="orm:hideurl"><code>col_factor()</code></a>, <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>col_date()</code></a>, <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>col_datetime()</code></a>은 각각 요인(factor), 날짜, 날짜-시간을 생성합니다. 이에 대해서는 <a href="ch16.html#chp-factors" data-type="xref">16장</a>과 <a href="ch17.html#chp-datetimes" data-type="xref">17장</a>에서 해당 데이터 유형에 도달할 때 더 자세히 배울 것입니다.
- <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>col_number()</code></a>는 비숫자 구성 요소를 무시하는 허용적인(permissive) 숫자 파서(parser)로, 특히 통화(currency)에 유용합니다. 이에 대해서는 <a href="ch13.html#chp-numbers" data-type="xref">13장</a>에서 자세히 배울 것입니다.
- <a href="https://readr.tidyverse.org/reference/col_skip.html" class="orm:hideurl"><code>col_skip()</code></a>은 결과에 포함되지 않도록 열을 건너뜁니다. 대용량 CSV 파일이 있고 일부 열만 사용하려는 경우 데이터 읽기 속도를 높이는 데 유용할 수 있습니다.

<a href="https://rdrr.io/r/base/list.html" class="orm:hideurl"><code>list()</code></a>에서 <a href="https://readr.tidyverse.org/reference/cols.html" class="orm:hideurl"><code>cols()</code></a>로 전환하고 `.default`를 지정하여 기본 열을 재정의(override)하는 것도 가능합니다.

`another_csv` `<-` `"` `x,y,z` `1,2,3"` `read_csv``(` `another_csv``,` `col_types` `=` `cols``(``.default` `=` `col_character``())` `)` `#> # A tibble: 1 × 3` `#> x y z ` `#> <chr> <chr> <chr>` `#> 1 1 2 3`

또 다른 유용한 헬퍼는 지정한 열만 읽어 들이는 <a href="https://readr.tidyverse.org/reference/cols.html" class="orm:hideurl"><code>cols_only()</code></a>입니다.

`read_csv``(` `another_csv``,` `col_types` `=` `cols_only``(``x` `=` `col_character``())` `)` `#> # A tibble: 1 × 1` `#> x ` `#> <chr>` `#> 1 1`

# 여러 파일에서 데이터 읽기 (Reading Data from Multiple Files)

데이터가 단일 파일에 포함되어 있지 않고 여러 파일에 나뉘어져 있는 경우가 있습니다. 예를 들어, 1월은 `01-sales.csv`, 2월은 `02-sales.csv`, 3월은 `03-sales.csv`와 같이 각 달의 데이터가 별도의 파일로 되어 있는 여러 달의 판매 데이터가 있을 수 있습니다. <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>를 사용하면 이러한 데이터를 한 번에 읽어 단일 데이터 프레임에 층층이 쌓을 수(stack) 있습니다.

`sales_files` `<-` `c``(``"data/01-sales.csv"``,` `"data/02-sales.csv"``,` `"data/03-sales.csv"``)` `read_csv``(``sales_files``,` `id` `=` `"file"``)` `#> # A tibble: 19 × 6` `#> file month year brand item n` `#> <chr> <chr> <dbl> <dbl> <dbl> <dbl>` `#> 1 data/01-sales.csv January 2019 1 1234 3` `#> 2 data/01-sales.csv January 2019 1 8721 9` `#> 3 data/01-sales.csv January 2019 1 1822 2` `#> 4 data/01-sales.csv January 2019 2 3333 1` `#> 5 data/01-sales.csv January 2019 2 2156 9` `#> 6 data/01-sales.csv January 2019 2 3987 6` `#> # … with 13 more rows`

다시 한 번 말씀드리지만, 프로젝트의 `data` 폴더에 CSV 파일이 있다면 위 코드가 작동할 것입니다. [_https://oreil.ly/jVd8o_](https://oreil.ly/jVd8o), [_https://oreil.ly/RYsgM_](https://oreil.ly/RYsgM), [_https://oreil.ly/4uZOm_](https://oreil.ly/4uZOm)에서 이러한 파일들을 다운로드하거나, 다음 코드로 직접 읽을 수 있습니다.

`sales_files` `<-` `c``(` `"https://pos.it/r4ds-01-sales"``,` `"https://pos.it/r4ds-02-sales"``,` `"https://pos.it/r4ds-03-sales"` `)` `read_csv``(``sales_files``,` `id` `=` `"file"``)`

`id` 인수는 데이터가 어떤 파일에서 왔는지를 식별하는 `file`이라는 이름의 새 열을 결과 데이터 프레임에 추가합니다. 이는 읽어들이는 파일에 관측치를 원래 소스로 역추적(trace)하는 데 도움이 될 수 있는 식별 열이 없는 경우에 특히 유용합니다.

읽고자 하는 파일이 많은 경우, 그 이름들을 리스트로 모두 작성하는 것은 번거로울 수 있습니다. 대신 기본 함수인 <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a>를 사용하여 파일 이름의 패턴을 일치시켜 파일을 찾게 할 수 있습니다. 이러한 패턴에 대해서는 <a href="ch15.html#chp-regexps" data-type="xref">15장</a>에서 자세히 알아볼 것입니다.

`sales_files` `<-` `list.files``(``"data"``,` `pattern` `=` `"sales\\.csv$"``,` `full.names` `=` `TRUE``)` `sales_files` `#> [1] "data/01-sales.csv" "data/02-sales.csv" "data/03-sales.csv"`

# 파일에 쓰기 (Writing to a File)

readr에는 디스크에 데이터를 쓰기 위한 두 가지 유용한 함수가 함께 제공됩니다. <a href="https://readr.tidyverse.org/reference/write_delim.html" class="orm:hideurl"><code>write_csv()</code></a>와 <a href="https://readr.tidyverse.org/reference/write_delim.html" class="orm:hideurl"><code>write_tsv()</code></a>입니다. 이 함수들의 중요한 인수는 `x`(저장할 데이터 프레임)와 `file`(저장할 위치)입니다. 또한 `na`를 사용하여 결측치를 어떻게 작성할지 지정할 수 있고, 기존 파일에 `append`(추가)할지 여부도 지정할 수 있습니다.

`write_csv``(``students``,` `"students.csv"``)`

이제 그 CSV 파일을 다시 읽어봅시다. 일반 텍스트 파일에서 데이터를 읽는 것부터 다시 시작하기 때문에, CSV로 저장할 때 방금 설정한 변수 유형 정보는 손실된다는 점에 유의하세요.

`students` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age` `#> <dbl> <chr> <chr> <fct> <dbl>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4` `#> 2 2 Barclay Lynn French fries Lunch only 5` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7` `#> 4 4 Leon Rossini Anchovies Lunch only NA` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch 5` `#> 6 6 Güvenç Attila Ice cream Lunch only 6` `write_csv``(``students``,` `"students-2.csv"``)` `read_csv``(``"students-2.csv"``)` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age` `#> <dbl> <chr> <chr> <chr> <dbl>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4` `#> 2 2 Barclay Lynn French fries Lunch only 5` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7` `#> 4 4 Leon Rossini Anchovies Lunch only NA` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch 5` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

이로 인해 CSV는 중간 결과를 캐시하는 데 다소 불안정합니다. 로드할 때마다 열 명세를 다시 만들어야 하기 때문입니다. 두 가지 주요 대안이 있습니다.

- <a href="https://readr.tidyverse.org/reference/read_rds.html" class="orm:hideurl"><code>write*rds()</code></a>와 <a href="https://readr.tidyverse.org/reference/read_rds.html" class="orm:hideurl"><code>read_rds()</code></a>는 기본 함수 <a href="https://rdrr.io/r/base/readRDS.html" class="orm:hideurl"><code>readRDS()</code></a>와 <a href="https://rdrr.io/r/base/readRDS.html" class="orm:hideurl"><code>saveRDS()</code></a>를 감싸는 일관된 래퍼(wrapper)입니다. 이들은 RDS라는 R의 사용자 정의 이진(binary) 형식으로 데이터를 저장합니다. 즉, 객체를 다시 로드할 때, 저장했던 *정확히 동일한\_ R 객체를 로드하게 됩니다.
  `write_rds``(``students``,` `"students.rds"``)` `read_rds``(``"students.rds"``)` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age` `#> <dbl> <chr> <chr> <fct> <dbl>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4` `#> 2 2 Barclay Lynn French fries Lunch only 5` `#> 3 3 Jayendra Lyne <NA> Breakfast and lunch 7` `#> 4 4 Leon Rossini Anchovies Lunch only NA` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch 5` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`
- arrow 패키지를 사용하면 프로그래밍 언어 간에 공유할 수 있는 빠른 이진 파일 형식인 parquet(파케이) 파일을 읽고 쓸 수 있습니다. arrow에 대해서는 <a href="ch22.html#chp-arrow" data-type="xref">22장</a>에서 더 깊이 다루겠습니다.
  `library``(``arrow``)` `write_parquet``(``students``,` `"students.parquet"``)` `read_parquet``(``"students.parquet"``)` `#> # A tibble: 6 × 5` `#> student_id full_name favourite_food meal_plan age` `#> <dbl> <chr> <chr> <fct> <dbl>` `#> 1 1 Sunil Huffmann Strawberry yoghurt Lunch only 4` `#> 2 2 Barclay Lynn French fries Lunch only 5` `#> 3 3 Jayendra Lyne NA Breakfast and lunch 7` `#> 4 4 Leon Rossini Anchovies Lunch only NA` `#> 5 5 Chidiegwu Dunkel Pizza Breakfast and lunch 5` `#> 6 6 Güvenç Attila Ice cream Lunch only 6`

Parquet는 RDS보다 훨씬 빠르고 R 외부에서도 사용할 수 있는 경향이 있지만 arrow 패키지가 필요합니다.

# 데이터 직접 입력 (Data Entry)

가끔 R 스크립트에서 작은 데이터를 직접 입력하여 티블(tibble)을 "수작업"으로 조합해야 할 때가 있습니다. 이를 돕는 두 가지 유용한 함수가 있으며, 열을 기준으로 티블을 배치하느냐 행을 기준으로 배치하느냐에 따라 다릅니다. <a href="https://tibble.tidyverse.org/reference/tibble.html" class="orm:hideurl"><code>tibble()</code></a>은 열을 기준으로 작동합니다.

`tibble``(` `x` `=` `c``(``1``,` `2``,` `5``),` `y` `=` `c``(``"h"``,` `"m"``,` `"g"``),` `z` `=` `c``(``0.08``,` `0.83``,` `0.60``)` `)` `#> # A tibble: 3 × 3` `#> x y z` `#> <dbl> <chr> <dbl>` `#> 1 1 h 0.08` `#> 2 2 m 0.83` `#> 3 5 g 0.6`

열을 기준으로 데이터를 배치하면 행이 어떻게 연관되어 있는지 확인하기 어려울 수 있으므로, 데이터를 행별로 배치할 수 있게 해주는 *tr*ansposed t*ibble*의 줄임말인 <a href="https://tibble.tidyverse.org/reference/tribble.html" class="orm:hideurl"><code>tribble()</code></a>이 그 대안입니다. <a href="https://tibble.tidyverse.org/reference/tribble.html" class="orm:hideurl"><code>tribble()</code></a>은 코드에서의 데이터 입력에 맞게 커스터마이징되어 있습니다. 열 제목은 `~`로 시작하고 항목은 쉼표로 구분됩니다. 이렇게 하면 적은 양의 데이터를 읽기 쉬운 형태로 배치할 수 있습니다.

`tribble``(` `~``x``,` `~``y``,` `~``z``,` `1``,` `"h"``,` `0.08``,` `2``,` `"m"``,` `0.83``,` `5``,` `"g"``,` `0.60` `)` `#> # A tibble: 3 × 3` `#> x y z` `#> <chr> <dbl> <dbl>` `#> 1 1 h 0.08` `#> 2 2 m 0.83` `#> 3 5 g 0.6`

# 요약 (Summary)

이 장에서는 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>를 사용하여 CSV 파일을 로드하는 방법과 <a href="https://tibble.tidyverse.org/reference/tibble.html" class="orm:hideurl"><code>tibble()</code></a> 및 <a href="https://tibble.tidyverse.org/reference/tribble.html" class="orm:hideurl"><code>tribble()</code></a>을 사용하여 자신만의 데이터를 직접 입력하는 방법을 배웠습니다. CSV 파일이 어떻게 작동하는지, 직면할 수 있는 몇 가지 문제와 이를 극복하는 방법을 배웠습니다. 이 책에서는 데이터 가져오기에 대해 몇 번 더 다룰 것입니다. <a href="ch20.html#chp-spreadsheets" data-type="xref">20장</a>에서는 Excel 및 Google 스프레드시트에서, <a href="ch21.html#chp-databases" data-type="xref">21장</a>에서는 데이터베이스에서, <a href="ch22.html#chp-arrow" data-type="xref">22장</a>에서는 parquet 파일에서, <a href="ch23.html#chp-rectangling" data-type="xref">23장</a>에서는 JSON에서, <a href="ch24.html#chp-webscraping" data-type="xref">24장</a>에서는 웹사이트에서 데이터를 로드하는 방법을 보여줄 것입니다.

우리는 이 책의 해당 섹션의 끝에 거의 다다랐지만, 다루어야 할 중요한 마지막 주제가 하나 남아 있습니다. 바로 도움을 구하는 방법입니다. 다음 장에서는 도움을 찾기 좋은 곳, 좋은 도움을 받을 확률을 최대화하기 위해 reprex(재현 가능한 예제)를 만드는 방법, 그리고 R 세계를 계속 따라잡기 위한 몇 가지 일반적인 조언을 배울 것입니다.

<sup>[1](ch07.html#idm44771312718064-marker)</sup> [janitor 패키지](https://oreil.ly/-J8GX)는 tidyverse의 일부가 아니지만, 데이터 정리를 위한 유용한 함수를 제공하며 `|>`를 사용하는 데이터 파이프라인 내에서 잘 작동합니다.

<sup>[2](ch07.html#idm44771312159040-marker)</sup> `guess_max` 인수를 사용하여 기본값인 1,000을 재정의(override)할 수 있습니다.
