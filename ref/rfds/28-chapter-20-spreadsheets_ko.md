# 제20장. 스프레드시트

# 소개

<a href="ch07.html#chp-data-import" data-type="xref">7장</a>에서는 `.csv`나 `.tsv` 같은 일반 텍스트 파일에서 데이터를 가져오는 방법에 대해 배웠습니다. 이제 엑셀 스프레드시트나 구글 시트 같은 스프레드시트에서 데이터를 가져오는 방법을 배울 차례입니다. 이는 <a href="ch07.html#chp-data-import" data-type="xref">7장</a>에서 배운 많은 내용을 바탕으로 하지만, 스프레드시트 데이터 작업을 할 때 고려해야 할 추가적인 사항과 복잡성에 대해서도 논의할 것입니다.

여러분이나 공동 작업자가 데이터 정리를 위해 스프레드시트를 사용한다면, Karl Broman과 Kara Woo의 논문 [“Data Organization in Spreadsheets”](https://oreil.ly/Ejuen)를 읽어보시기를 강력히 권장합니다. 이 논문에서 제시하는 모범 사례들은 나중에 데이터를 분석하고 시각화하기 위해 스프레드시트에서 R로 데이터를 가져올 때 발생할 수 있는 많은 골칫거리를 덜어줄 것입니다.

# 엑셀

마이크로소프트 엑셀(Microsoft Excel)은 데이터가 스프레드시트 파일 내의 워크시트에 체계적으로 정리되어 있는, 널리 사용되는 스프레드시트 소프트웨어 프로그램입니다.

## 사전 준비

이 섹션에서는 readxl 패키지를 사용하여 R에서 엑셀 스프레드시트의 데이터를 불러오는 방법을 배웁니다. 이 패키지는 tidyverse의 핵심 패키지는 아니므로 명시적으로 불러와야 하지만, tidyverse 패키지를 설치할 때 자동으로 설치됩니다. 나중에 엑셀 스프레드시트를 생성할 수 있게 해주는 writexl 패키지도 사용할 예정입니다.

```
library(readxl)
library(tidyverse)
library(writexl)
```

## 시작하기

readxl의 대부분의 함수는 엑셀 스프레드시트를 R로 불러올 수 있게 해줍니다:

- <a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_xls()</code></a>는 `XLS` 형식의 엑셀 파일을 읽습니다.
- <a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_xlsx()</code></a>는 `XLSX` 형식의 엑셀 파일을 읽습니다.
- <a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a>은 `XLS` 및 `XLSX` 형식의 파일을 모두 읽을 수 있습니다. 입력값을 바탕으로 파일 유형을 추측합니다.

이 함수들은 이전에 다른 유형의 파일을 읽기 위해 소개한 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>, <a href="https://readr.tidyverse.org/reference/read_table.html" class="orm:hideurl"><code>read_table()</code></a> 등과 같은 다른 함수들과 모두 비슷한 구문을 가지고 있습니다. 이 장의 나머지 부분에서는 <a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a> 사용에 집중할 것입니다.

## 엑셀 스프레드시트 읽기

<a href="#fig-students-excel" data-type="xref">그림 20-1</a>은 우리가 R로 읽어 들일 스프레드시트가 엑셀에서 어떻게 보이는지 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2001.png" alt="A look at the students spreadsheet in Excel. The spreadsheet contains information on 6 students, their ID, full name, favorite food, meal plan, and age." />
<h6 id="figure-20-1.-spreadsheet-called-students.xlsx-in-excel.">그림 20-1. 엑셀의 <code>students.xlsx</code>라는 스프레드시트.</h6>
</figure>

<a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a>의 첫 번째 인자는 읽어 들일 파일의 경로입니다.

```
students <- read_excel("data/students.xlsx")
```

<a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a>은 파일을 티블(tibble) 형태로 읽어옵니다.

```
students
#> # A tibble: 6 × 5
#>   `Student ID` `Full Name`      favourite.food     mealPlan            AGE  
#>          <dbl> <chr>            <chr>              <chr>               <chr>
#> 1            1 Sunil Huffmann   Strawberry yoghurt Lunch only          4    
#> 2            2 Barclay Lynn     French fries       Lunch only          5    
#> 3            3 Jayendra Lyne    N/A                Breakfast and lunch 7    
#> 4            4 Leon Rossini     Anchovies          Lunch only          <NA> 
#> 5            5 Chidiegwu Dunkel Pizza              Breakfast and lunch five 
#> 6            6 Güvenç Attila    Ice cream          Lunch only          6
```

데이터에는 6명의 학생과 각 학생에 대한 5개의 변수가 있습니다. 하지만 이 데이터 세트에서 처리하고 싶은 몇 가지 사항이 있습니다:

1.  열 이름이 제각각입니다. 일관된 형식을 따르는 열 이름을 제공할 수 있습니다. `col_names` 인자를 사용하여 `snake_case`를 사용하는 것을 권장합니다.

    ```
    read_excel(
      "data/students.xlsx",
      col_names = c(
        "student_id", "full_name", "favourite_food", "meal_plan", "age")
    )
    #> # A tibble: 7 × 5
    #>   student_id full_name        favourite_food     meal_plan           age  
    #>   <chr>      <chr>            <chr>              <chr>               <chr>
    #> 1 Student ID Full Name        favourite.food     mealPlan            AGE  
    #> 2 1          Sunil Huffmann   Strawberry yoghurt Lunch only          4    
    #> 3 2          Barclay Lynn     French fries       Lunch only          5    
    #> 4 3          Jayendra Lyne    N/A                Breakfast and lunch 7    
    #> 5 4          Leon Rossini     Anchovies          Lunch only          <NA> 
    #> 6 5          Chidiegwu Dunkel Pizza              Breakfast and lunch five 
    #> 7 6          Güvenç Attila    Ice cream          Lunch only          6
    ```

    안타깝게도 이것만으로는 문제가 해결되지 않았습니다. 이제 우리가 원하는 변수 이름은 얻었지만, 이전에 헤더 행이었던 것이 이제 데이터의 첫 번째 관측치로 나타납니다. `skip` 인자를 사용하여 해당 행을 명시적으로 건너뛸 수 있습니다.

    ```
    read_excel(
      "data/students.xlsx",
      col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
      skip = 1
    )
    #> # A tibble: 6 × 5
    #>   student_id full_name        favourite_food     meal_plan           age  
    #>        <dbl> <chr>            <chr>              <chr>               <chr>
    #> 1          1 Sunil Huffmann   Strawberry yoghurt Lunch only          4    
    #> 2          2 Barclay Lynn     French fries       Lunch only          5    
    #> 3          3 Jayendra Lyne    N/A                Breakfast and lunch 7    
    #> 4          4 Leon Rossini     Anchovies          Lunch only          <NA> 
    #> 5          5 Chidiegwu Dunkel Pizza              Breakfast and lunch five 
    #> 6          6 Güvenç Attila    Ice cream          Lunch only          6
    ```

2.  `favourite_food` 열에서 관측치 중 하나는 "not available"을 나타내는 `N/A`이지만, 현재는 `NA`로 인식되지 않고 있습니다(이 `N/A`와 목록에 있는 4번째 학생의 나이 값을 비교해 보세요). `na` 인자를 사용하여 어떤 문자열을 `NA`로 인식할지 지정할 수 있습니다. 기본적으로는 `""`(빈 문자열, 또는 스프레드시트에서 읽을 때 빈 셀이나 `=NA()` 수식이 있는 셀)만 `NA`로 인식됩니다.

    ```
    read_excel(
      "data/students.xlsx",
      col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
      skip = 1,
      na = c("", "N/A")
    )
    #> # A tibble: 6 × 5
    #>   student_id full_name        favourite_food     meal_plan           age  
    #>        <dbl> <chr>            <chr>              <chr>               <chr>
    #> 1          1 Sunil Huffmann   Strawberry yoghurt Lunch only          4    
    #> 2          2 Barclay Lynn     French fries       Lunch only          5    
    #> 3          3 Jayendra Lyne    <NA>               Breakfast and lunch 7    
    #> 4          4 Leon Rossini     Anchovies          Lunch only          <NA> 
    #> 5          5 Chidiegwu Dunkel Pizza              Breakfast and lunch five 
    #> 6          6 Güvenç Attila    Ice cream          Lunch only          6
    ```

3.  남은 또 다른 문제는 `age`가 문자형 변수로 읽히지만 실제로는 숫자형이어야 한다는 점입니다. 플랫 파일(flat file)에서 데이터를 읽기 위해 사용하는 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>와 그 친구들과 마찬가지로, <a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a>에 `col_types` 인자를 제공하고 읽어 들일 변수의 열 유형을 지정할 수 있습니다. 하지만 구문이 조금 다릅니다. 가능한 옵션은 `"skip"`, `"guess"`, `"logical"`, `"numeric"`, `"date"`, `"text"`, 또는 `"list"`입니다.

    ```
    read_excel(
      "data/students.xlsx",
      col_names = c("student_id", "full_name", "favourite_food", "meal_plan", "age"),
      skip = 1,
      na = c("", "N/A"),
      col_types = c("numeric", "text", "text", "text", "numeric")
    )
    #> Warning: Expecting numeric in E6 / R6C5: got 'five'
    #> # A tibble: 6 × 5
    #>   student_id full_name        favourite_food     meal_plan             age
    #>        <dbl> <chr>            <chr>              <chr>               <dbl>
    #> 1          1 Sunil Huffmann   Strawberry yoghurt Lunch only              4
    #> 2          2 Barclay Lynn     French fries       Lunch only              5
    #> 3          3 Jayendra Lyne    <NA>               Breakfast and lunch     7
    #> 4          4 Leon Rossini     Anchovies          Lunch only             NA
    #> 5          5 Chidiegwu Dunkel Pizza              Breakfast and lunch    NA
    #> 6          6 Güvenç Attila    Ice cream          Lunch only              6
    ```

    하지만 이 역시 원하는 결과를 정확히 만들어내지는 못했습니다. `age`가 숫자형이어야 한다고 지정함으로써, 숫자형 항목이 아닌 유일한 셀(`five`라는 값을 가진 셀)을 `NA`로 만들어 버렸습니다. 이 경우, 나이를 `"text"`로 읽은 다음 데이터가 R에 로드된 후에 변경을 적용해야 합니다.

    ```
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
    #> # A tibble: 6 × 5
    #>   student_id full_name        favourite_food     meal_plan             age
    #>        <dbl> <chr>            <chr>              <chr>               <dbl>
    #> 1          1 Sunil Huffmann   Strawberry yoghurt Lunch only              4
    #> 2          2 Barclay Lynn     French fries       Lunch only              5
    #> 3          3 Jayendra Lyne    <NA>               Breakfast and lunch     7
    #> 4          4 Leon Rossini     Anchovies          Lunch only             NA
    #> 5          5 Chidiegwu Dunkel Pizza              Breakfast and lunch     5
    #> 6          6 Güvenç Attila    Ice cream          Lunch only              6
    ```

원하는 형식으로 데이터를 로드하는 데 여러 단계와 시행착오를 거쳤으며, 이는 예상 밖의 일이 아닙니다. 데이터 과학은 반복적인 과정이며, 사람들이 스프레드시트에 데이터를 입력하고 그것을 단지 데이터 저장 목적뿐만 아니라 공유와 의사소통을 위해 사용하는 경향이 있기 때문에 다른 일반 텍스트, 직사각형 구조의 데이터 파일과 비교할 때 스프레드시트에서 데이터를 읽어오는 과정에서 이런 반복 작업이 훨씬 더 번거로울 수 있습니다.

데이터를 로드해서 살펴보기 전까지는 데이터가 정확히 어떻게 생겼는지 알 수 있는 방법은 없습니다. 글쎄요, 사실 한 가지 방법이 있기는 합니다. 엑셀에서 파일을 열어서 살짝 엿보는 것입니다. 그렇게 하려면 원본 데이터 파일은 그대로 두고 상호작용 방식으로 열어서 살펴볼 엑셀 파일의 복사본을 만든 다음, R에서는 그대로 둔 원본 파일에서 데이터를 읽는 것을 권장합니다. 이렇게 하면 데이터를 살펴보는 중에 실수로 스프레드시트의 내용을 덮어쓰는 일을 방지할 수 있습니다. 또한 여기서 우리가 했던 방식을 두려워해서는 안 됩니다: 데이터를 로드하고, 살펴보고, 코드를 수정하고, 다시 로드하고, 결과에 만족할 때까지 반복하세요.

## 워크시트 읽기

스프레드시트를 플랫 파일과 구분하는 중요한 특징은 워크시트(worksheet)라고 불리는 다중 시트 개념입니다. <a href="#fig-penguins-islands" data-type="xref">그림 20-2</a>는 다중 워크시트가 있는 엑셀 스프레드시트를 보여줍니다. 데이터는 palmerpenguins 패키지에서 가져왔습니다. 각 워크시트에는 데이터가 수집된 여러 섬의 펭귄에 대한 정보가 포함되어 있습니다.

<a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a>의 `sheet` 인자를 사용하여 스프레드시트에서 단일 워크시트를 읽을 수 있습니다. 지금까지 우리가 의존해 온 기본값은 첫 번째 시트입니다.

```
read_excel("data/penguins.xlsx", sheet = "Torgersen Island")
#> # A tibble: 52 × 8
#>   species island    bill_length_mm     bill_depth_mm      flipper_length_mm
#>   <chr>   <chr>     <chr>              <chr>              <chr>            
#> 1 Adelie  Torgersen 39.1               18.7               181              
#> 2 Adelie  Torgersen 39.5               17.399999999999999 186              
#> 3 Adelie  Torgersen 40.299999999999997 18                 195              
#> 4 Adelie  Torgersen NA                 NA                 NA               
#> 5 Adelie  Torgersen 36.700000000000003 19.3               193              
#> 6 Adelie  Torgersen 39.299999999999997 20.6               190              
#> # … with 46 more rows, and 3 more variables: body_mass_g <chr>, sex <chr>,
#> #   year <dbl>
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2002.png" alt="A look at the penguins spreadsheet in Excel. The spreadsheet contains has three worksheets: Torgersen Island, Biscoe Island, and Dream Island." />
<h6 id="figure-20-2.-spreadsheet-called-penguins.xlsx-in-excel-containing-three-worksheets.">그림 20-2. 세 개의 워크시트를 포함하는 엑셀의 <code>penguins.xlsx</code>라는 스프레드시트.</h6>
</figure>

숫자 데이터를 포함하는 것처럼 보이는 일부 변수들은 `"NA"`라는 문자열이 진짜 `NA`로 인식되지 않기 때문에 문자형으로 읽혀집니다.

```
penguins_torgersen <- read_excel(
  "data/penguins.xlsx", sheet = "Torgersen Island", na = "NA"
)

penguins_torgersen
#> # A tibble: 52 × 8
#>   species island    bill_length_mm bill_depth_mm flipper_length_mm
#>   <chr>   <chr>              <dbl>         <dbl>             <dbl>
#> 1 Adelie  Torgersen           39.1          18.7               181
#> 2 Adelie  Torgersen           39.5          17.4               186
#> 3 Adelie  Torgersen           40.3          18                 195
#> 4 Adelie  Torgersen           NA            NA                  NA
#> 5 Adelie  Torgersen           36.7          19.3               193
#> 6 Adelie  Torgersen           39.3          20.6               190
#> # … with 46 more rows, and 3 more variables: body_mass_g <dbl>, sex <chr>,
#> #   year <dbl>
```

대안으로, <a href="https://readxl.tidyverse.org/reference/excel_sheets.html" class="orm:hideurl"><code>excel_sheets()</code></a>를 사용하여 엑셀 스프레드시트의 모든 워크시트에 대한 정보를 얻은 다음 관심 있는 워크시트(들)를 읽을 수 있습니다.

```
excel_sheets("data/penguins.xlsx")
#> [1] "Torgersen Island" "Biscoe Island"    "Dream Island"
```

워크시트의 이름을 알게 되면, <a href="https://readxl.tidyverse.org/reference/read_excel.html" class="orm:hideurl"><code>read_excel()</code></a>을 사용하여 그것들을 개별적으로 읽을 수 있습니다.

```
penguins_biscoe <- read_excel("data/penguins.xlsx", sheet = "Biscoe Island", na = "NA")
penguins_dream  <- read_excel("data/penguins.xlsx", sheet = "Dream Island", na = "NA")
```

이 경우, 전체 펭귄 데이터 세트는 스프레드시트 내의 세 개의 워크시트에 분산되어 있습니다. 각 워크시트는 열의 개수는 같지만 행의 개수는 다릅니다.

```
dim(penguins_torgersen)
#> [1] 52  8
dim(penguins_biscoe)
#> [1] 168   8
dim(penguins_dream)
#> [1] 124   8
```
#> [1] "ppl" "pr"  "bnn"
```

데이터 정리를 할 때 이러한 함수들은 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 자연스럽게 짝을 이루며, 일관성 없는 형식의 레이어를 벗겨내기 위해 이들을 반복적으로 적용하는 경우가 많습니다.

## 변수 추출하기

우리가 논의할 마지막 함수는 정규 표현식을 사용하여 한 열의 데이터를 하나 이상의 새로운 열로 추출하는 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>입니다. 이 함수는 <a href="ch14.html#sec-string-columns" data-type="xref">"열로 분리하기(Separating into Columns)"</a>에서 배웠던 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_position()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_delim()</code></a> 함수와 동급입니다. 이 함수들은 개별 벡터가 아닌 데이터 프레임(의 열)에서 작동하기 때문에 tidyr에 속해 있습니다.

이 함수가 어떻게 작동하는지 보여주기 위해 간단한 데이터셋을 만들어 보겠습니다. 여기 `babynames`에서 파생된 데이터가 있는데, 이름, 성별, 나이가 다소 이상한 형식으로 되어 있습니다:<sup><a href="ch15.html#idm44771293255808" id="idm44771293255808-marker" data-type="noteref">5</a></sup>

```
df <- tribble(
  ~str,
  "<Sheryl>-F_34",
  "<Kisha>-F_45", 
  "<Brandon>-N_33",
  "<Sharon>-F_38", 
  "<Penny>-F_58",
  "<Justin>-M_41", 
  "<Patricia>-F_84", 
)
```

<a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>를 사용하여 이 데이터를 추출하려면 각 조각과 일치하는 정규 표현식 시퀀스를 구성하기만 하면 됩니다. 해당 조각의 내용이 출력에 나타나기를 원한다면, 거기에 이름을 부여합니다:

```
df |> 
  separate_wider_regex(
    str,
    patterns = c(
      "<", 
      name = "[A-Za-z]+", 
      ">-", 
      gender = ".", "_", 
      age = "[0-9]+"
    )
  )
#> # A tibble: 7 × 3
#>   name    gender age  
#>   <chr>   <chr>  <chr>
#> 1 Sheryl  F      34   
#> 2 Kisha   F      45   
#> 3 Brandon N      33   
#> 4 Sharon  F      38   
#> 5 Penny   F      58   
#> 6 Justin  M      41   
#> # … with 1 more row
```

일치가 실패할 경우 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_delim()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/separate_wider_position.html" class="orm:hideurl"><code>separate_wider_position()</code></a>처럼 `too_short = "debug"`를 사용하여 무엇이 잘못되었는지 파악할 수 있습니다.

## 연습문제

1. 가장 모음이 많은 아기 이름은 무엇인가요? 모음의 비율이 가장 높은 이름은 무엇인가요? (힌트: 분모는 무엇일까요?)

2. `"a/b/c/d/e"`의 모든 슬래시를 백슬래시로 변경하세요. 백슬래시를 다시 슬래시로 변경하여 변환을 되돌리려고 하면 어떤 일이 발생하나요? (이 문제에 대해서는 곧 논의할 것입니다.)

3. <a href="https://stringr.tidyverse.org/reference/str_replace.html" class="orm:hideurl"><code>str_replace_all()</code></a>을 사용하여 간단한 버전의 <a href="https://stringr.tidyverse.org/reference/case.html" class="orm:hideurl"><code>str_to_lower()</code></a>를 구현하세요.

4. 여러분의 나라에서 흔히 표기되는 전화번호 형식과 일치하는 정규 표현식을 작성하세요.

# 패턴 세부 사항

이제 패턴 언어의 기본과 이를 stringr 및 tidyr 함수와 함께 사용하는 방법을 이해했으므로, 세부 사항을 더 파고들 시간입니다. 먼저, 특별하게 취급되는 메타문자들을 일치시킬 수 있게 해주는 *이스케이핑(escaping)*으로 시작할 것입니다. 그 다음에는 문자열의 시작이나 끝부분과 일치시키는 *앵커(anchors)*에 대해 배울 것입니다. 이어서 집합 내의 임의의 문자와 일치시킬 수 있는 *문자 클래스(character classes)*와 그 단축키에 대해 배웁니다. 다음으로, 패턴이 일치할 수 있는 횟수를 제어하는 *수량자(quantifiers)*의 마지막 세부 사항을 다룹니다. 그런 다음, 중요하지만 복잡한 주제인 *연산자 우선순위(operator precedence)*와 괄호에 대해 다루어야 합니다. 그리고 패턴의 요소를 *그룹화(grouping)*하는 몇 가지 세부 사항으로 마무리할 것입니다.

여기서 사용하는 용어들은 각 구성 요소에 대한 기술적인 이름입니다. 이 용어들이 항상 그 목적을 잘 떠오르게 하는 것은 아니지만, 나중에 더 자세한 내용을 구글링하고 싶을 때 올바른 용어를 아는 것이 유용합니다.

## 이스케이핑

리터럴 `.`과 일치시키려면, 정규 표현식에 메타문자들을<sup><a href="ch15.html#idm44771293083536" id="idm44771293083536-marker" data-type="noteref">6</a></sup> 문자 그대로 일치시키라고 알려주는 *이스케이프(escape)*가 필요합니다. 문자열과 마찬가지로 정규 표현식은 이스케이핑을 위해 백슬래시를 사용합니다. 따라서 `.`과 일치시키려면 정규 표현식 `\.`이 필요합니다. 안타깝게도 이것은 문제를 발생시킵니다. 우리는 정규 표현식을 나타내기 위해 문자열을 사용하고, `\`는 문자열에서도 이스케이프 기호로 사용됩니다. 따라서 정규 표현식 `\.`을 생성하려면, 다음 예제에서 보여주듯이 문자열 `"\\."`이 필요합니다:

```
# 정규 표현식 \. 을 생성하려면 \\. 을 사용해야 합니다.
dot <- "\\."

# 하지만 표현식 자체는 \ 를 하나만 포함합니다.
str_view(dot)
#> [1] │ \.

# 그리고 이것은 R에게 명시적인 . 을 찾으라고 지시합니다.
str_view(c("abc", "a.c", "bef"), "a\\.c")
#> [2] │ <a.c>
```

이 책에서 우리는 대개 `\.`처럼 정규 표현식을 따옴표 없이 작성할 것입니다. 만약 실제로 입력해야 할 내용을 강조할 필요가 있다면, 따옴표로 감싸고 추가적인 이스케이프를 더하여 `"\\."`처럼 작성할 것입니다.

정규 표현식에서 `\`가 이스케이프 문자로 사용된다면, 리터럴 `\`와 어떻게 일치시킬 수 있을까요? 글쎄요, 이를 이스케이프하여 정규 표현식 `\\`를 생성해야 합니다. 그 정규 표현식을 생성하려면 문자열을 사용해야 하고, 이 문자열 또한 `\`를 이스케이프해야 합니다. 이는 리터럴 `\`와 일치시키기 위해 `"\\\\"`를 작성해야 한다는 것을 의미합니다 — 한 개의 백슬래시와 일치시키기 위해 네 개가 필요합니다!

```
x <- "a\\b"
str_view(x)
#> [1] │ a\b
str_view(x, "\\\\")
#> [1] │ a<\>b
```

대안으로, <a href="ch14.html#sec-raw-strings" data-type="xref">"원시 문자열(Raw Strings)"</a>에서 배운 원시 문자열을 사용하는 것이 더 쉬울 수도 있습니다. 이를 통해 한 단계의 이스케이핑을 피할 수 있습니다:

```
str_view(x, r"{\\}")
#> [1] │ a<\>b
```

만약 리터럴 `.`, `$`, `|`, `*`, `+`, `?`, `{`, `}`, `(`, `)`와 일치시키려 한다면, 백슬래시 이스케이프를 사용하는 것의 대안이 있습니다. 문자 클래스를 사용할 수 있습니다: `[.]`, `[$]`, `[|]`, ... 은 모두 리터럴 값과 일치합니다:

```
str_view(c("abc", "a.c", "a*c", "a c"), "a[.]c")
#> [2] │ <a.c>
str_view(c("abc", "a.c", "a*c", "a c"), ".[*]c")
#> [3] │ <a*c>
```

## 앵커

기본적으로 정규 표현식은 문자열의 어떤 부분과도 일치합니다. 시작이나 끝에서 일치시키고 싶다면, 시작과 일치하는 `^`나 끝과 일치하는 `$`를 사용하여 정규 표현식을 *고정(anchor)*해야 합니다:

```
str_view(fruit, "^a")
#> [1] │ <a>pple
#> [2] │ <a>pricot
#> [3] │ <a>vocado
str_view(fruit, "a$")
#>  [4] │ banan<a>
#> [15] │ cherimoy<a>
#> [30] │ feijo<a>
#> [36] │ guav<a>
#> [56] │ papay<a>
#> [74] │ satsum<a>
```

달러 금액을 표기하는 방식 때문에 `$`가 문자열의 시작과 일치해야 한다고 생각하기 쉽지만, 정규 표현식에서는 그렇지 않습니다.

정규 표현식이 전체 문자열과만 일치하도록 강제하려면, `^`와 `$` 둘 다 사용하여 고정합니다:

```
str_view(fruit, "apple")
#>  [1] │ <apple>
#> [62] │ pine<apple>
str_view(fruit, "^apple$")
#> [1] │ <apple>
```

단어 간의 경계(즉, 단어의 시작이나 끝)도 `\b`와 일치시킬 수 있습니다. 이것은 RStudio의 찾기 및 바꾸기 도구를 사용할 때 특히 유용할 수 있습니다. 예를 들어, <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>의 모든 사용을 찾으려면, `summarize`, `summary`, `rowsum` 등과 일치하는 것을 피하기 위해 `\bsum\b`를 검색할 수 있습니다:

```
x <- c("summary(x)", "summarize(df)", "rowsum(x)", "sum(x)")
str_view(x, "sum")
#> [1] │ <sum>mary(x)
#> [2] │ <sum>marize(df)
#> [3] │ row<sum>(x)
#> [4] │ <sum>(x)
str_view(x, "\\bsum\\b")
#> [4] │ <sum>(x)
```

단독으로 사용될 때, 앵커는 길이가 0인 일치를 생성합니다:

```
str_view("abc", c("$", "^", "\\b"))
#> [1] │ abc<>
#> [2] │ <>abc
#> [3] │ <>abc<>
```

이것은 독립적인 앵커를 교체할 때 무슨 일이 일어나는지 이해하는 데 도움이 됩니다:

```
str_replace_all("abc", c("$", "^", "\\b"), "--")
#> [1] "abc--"   "--abc"   "--abc--"
```

## 문자 클래스

*문자 클래스(character class)* 또는 문자 *집합(set)*을 사용하면 집합 내의 임의의 문자와 일치시킬 수 있습니다. 앞서 논의했듯이, `[]`를 사용하여 자신만의 집합을 구성할 수 있으며, 여기서 `[abc]`는 "a", "b", "c" 중 하나와 일치하고 `[^abc]`는 "a", "b", "c"를 제외한 모든 문자와 일치합니다. `^` 외에도 `[]` 안에서 특별한 의미를 갖는 두 개의 문자가 더 있습니다:

- `-`는 범위를 정의합니다; 예: `[a-z]`는 소문자, `[0-9]`는 숫자와 일치합니다.
- `\`는 특수 문자를 이스케이프하므로, `[\^\-\]]`는 `^`, `-`, 또는 `]`와 일치합니다.

몇 가지 예시입니다:

```
x <- "abcd ABCD 12345 -!@#%."
str_view(x, "[abc]+")
#> [1] │ <abc>d ABCD 12345 -!@#%.
str_view(x, "[a-z]+")
#> [1] │ <abcd> ABCD 12345 -!@#%.
str_view(x, "[^a-z0-9]+")
#> [1] │ abcd< ABCD >12345< -!@#%.>

# [] 안에서 원래 특별한 문자와 일치시키려면 이스케이프가 필요합니다.
str_view("a-b-c", "[a-c]")
#> [1] │ <a>-<b>-<c>
str_view("a-b-c", "[a\\-c]")
#> [1] │ <a><->b<-><c>
```

일부 문자 클래스들은 너무 자주 사용되어 그들만의 단축키를 갖게 되었습니다. 여러분은 이미 줄바꿈을 제외한 모든 문자와 일치하는 `.`을 보았습니다. 특히 유용한 세 가지 다른 쌍이 있습니다:<sup><a href="ch15.html#idm44771292559200" id="idm44771292559200-marker" data-type="noteref">7</a></sup>

- `\d`는 숫자와 일치합니다.\
  `\D`는 숫자가 아닌 모든 것과 일치합니다.
- `\s`는 공백 문자(예: 스페이스, 탭, 줄바꿈)와 일치합니다.\
  `\S`는 공백 문자가 아닌 모든 것과 일치합니다.
- `\w`는 "단어" 문자(즉, 문자와 숫자)와 일치합니다.\
  `\W`는 "단어" 문자가 아닌 모든 것과 일치합니다.

다음 코드는 문자와 숫자, 구두점 문자를 선택하여 여섯 가지 단축키를 시연합니다:

```
x <- "abcd ABCD 12345 -!@#%."
str_view(x, "\\d+")
#> [1] │ abcd ABCD <12345> -!@#%.
str_view(x, "\\D+")
#> [1] │ <abcd ABCD >12345< -!@#%.>
str_view(x, "\\s+")
#> [1] │ abcd< >ABCD< >12345< >-!@#%.
str_view(x, "\\S+")
#> [1] │ <abcd> <ABCD> <12345> <-!@#%.>
str_view(x, "\\w+")
#> [1] │ <abcd> <ABCD> <12345> -!@#%.
str_view(x, "\\W+")
#> [1] │ abcd< >ABCD< >12345< -!@#%.>
```

## 수량자

*수량자(Quantifiers)*는 패턴이 일치하는 횟수를 제어합니다. <a href="#sec-reg-basics" data-type="xref">"패턴 기초(Pattern Basics)"</a>에서 우리는 `?` (0번 또는 1번 일치), `+` (1번 이상 일치), 그리고 `*` (0번 이상 일치)에 대해 배웠습니다. 예를 들어, `colou?r`는 미국식 또는 영국식 철자와 일치할 것이고, `\d+`는 하나 이상의 숫자와 일치하며, `\s?`는 단일 공백 항목과 선택적으로 일치할 것입니다. <a href="https://rdrr.io/r/base/Paren.html" class="orm:hideurl"><code>{}</code></a>를 사용하여 일치 횟수를 정확하게 지정할 수도 있습니다:

- `{n}`는 정확히 n번 일치합니다.
- `{n,}`는 적어도 n번 일치합니다.
- `{n,m}`는 n번에서 m번 사이로 일치합니다.

## 연산자 우선순위와 괄호

`ab+`는 무엇과 일치할까요? "a" 뒤에 하나 이상의 "b"가 오는 것과 일치할까요, 아니면 임의의 횟수로 반복되는 "ab"와 일치할까요? `^a|b$`는 무엇과 일치할까요? 완전한 문자열 a나 완전한 문자열 b와 일치할까요, 아니면 a로 시작하는 문자열이나 b로 끝나는 문자열과 일치할까요?

이러한 질문에 대한 답은 학교에서 배웠을 PEMDAS나 BEDMAS 규칙과 유사한 연산자 우선순위에 의해 결정됩니다. `*`가 더 높은 우선순위를 갖고 `+`가 더 낮은 우선순위를 갖기 때문에(즉, `*`를 `+`보다 먼저 계산하기 때문에) `a + b * c`가 `(a + b) * c`가 아니라 `a + (b * c)`와 동등하다는 것을 알고 있을 것입니다.

마찬가지로 정규 표현식에도 자체 우선순위 규칙이 있습니다. 수량자는 높은 우선순위를 갖고, 대체(alternation)는 낮은 우선순위를 갖습니다. 즉, `ab+`는 `a(b+)`와 동등하고, `^a|b$`는 `(^a)|(b$)`와 동등합니다. 대수학에서처럼 괄호를 사용하여 일반적인 순서를 재정의할 수 있습니다. 하지만 대수학과 달리 정규 표현식의 우선순위 규칙을 기억할 가능성은 낮으므로, 필요할 때마다 괄호를 자유롭게 사용하세요.

## 그룹화와 캡처링

연산자 우선순위를 재정의하는 것 외에도 괄호에는 또 다른 중요한 효과가 있습니다. 일치 항목의 하위 구성 요소를 사용할 수 있게 해주는 *캡처 그룹(capturing groups)*을 생성합니다.

캡처 그룹을 사용하는 첫 번째 방법은 *역참조(back reference)*를 사용하여 일치 항목 내에서 참조하는 것입니다. `\1`은 첫 번째 괄호에 포함된 일치 항목을, `\2`는 두 번째 괄호를 참조하는 식입니다. 예를 들어, 다음 패턴은 반복되는 두 글자 쌍을 가진 모든 과일을 찾습니다:

```
str_view(fruit, "(..)\\1")
#>  [4] │ b<anan>a
```
전체 개수에 의해 결정되는 높이가 `cut`에 따라 너무 많이 달라서 분포 모양의 차이를 보기 어렵기 때문에, 여기서는 <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a>의 기본 모양이 그다지 유용하지 않습니다.

비교를 더 쉽게 하려면 y축에 표시되는 것을 바꿔야 합니다. 개수를 표시하는 대신, 각 도수 다각형(frequency polygon) 아래의 면적이 1이 되도록 표준화된 개수인 *밀도(density)*를 표시할 것입니다:

```
ggplot(diamonds, aes(x = price, y = after_stat(density))) + 
  geom_freqpoly(aes(color = cut), binwidth = 500, linewidth = 0.75)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in08.png" alt="A frequency polygon of densities of prices of diamonds where each cut of carat (Fair, Good, Very Good, Premium, and Ideal) is represented with a different color line. The x-axis ranges from 0 to 20000. The lines overlap a great deal, suggesting similar density distributions of prices of diamonds. One notable feature is that all but Fair diamonds have high peaks around a price of 1500 and Fair diamonds have a higher mean than others." />
</figure>

우리는 밀도를 `y`에 매핑하고 있지만, `density`는 `diamonds` 데이터세트의 변수가 아니기 때문에 먼저 계산해야 한다는 점에 유의하세요. 이를 위해 <a href="https://ggplot2.tidyverse.org/reference/aes_eval.html" class="orm:hideurl"><code>after_stat()</code></a> 함수를 사용합니다.

이 플롯에는 꽤 놀라운 점이 있습니다: (품질이 가장 낮은) Fair 다이아몬드가 가장 높은 평균 가격을 가지는 것처럼 보입니다! 하지만 이는 도수 다각형을 해석하기가 약간 어렵기 때문일 수 있습니다. 이 플롯에서는 많은 일이 벌어지고 있습니다.

이 관계를 탐색하기 위한 시각적으로 더 간단한 플롯은 나란히 놓인 상자 그림(side-by-side boxplots)을 사용하는 것입니다:

```
ggplot(diamonds, aes(x = cut, y = price)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in09.png" alt="Side-by-side boxplots of prices of diamonds by cut. The distribution of prices is right skewed for each cut (Fair, Good, Very Good, Premium, and Ideal). The medians are close to each other, with the median for Ideal diamonds lowest and that for Fair highest." />
</figure>

우리는 분포에 대한 정보는 훨씬 덜 보게 되지만, 상자 그림은 훨씬 더 간결하기 때문에 더 쉽게 비교할 수 있습니다 (그리고 하나의 플롯에 더 많이 들어맞게 할 수 있습니다). 이는 품질이 좋은 다이아몬드가 일반적으로 더 저렴하다는 직관에 반하는 발견을 뒷받침합니다! 연습문제에서 여러분은 왜 그런지 알아내는 도전을 받게 될 것입니다.

`cut`은 순서형 요인(ordered factor)입니다: Fair는 Good보다 나쁘고, Good은 Very Good보다 나쁜 식입니다. 많은 범주형 변수는 이와 같은 내재적 순서를 가지고 있지 않으므로, 더 많은 정보를 제공하는 디스플레이를 만들기 위해 그것들을 다시 정렬하고 싶을 수 있습니다. 이를 수행하는 한 가지 방법은 <a href="https://forcats.tidyverse.org/reference/fct_reorder.html" class="orm:hideurl"><code>fct_reorder()</code></a>를 사용하는 것입니다. 그 함수에 대해서는 <a href="ch16.html#sec-modifying-factor-order" data-type="xref">“팩터 순서 수정하기(Modifying Factor Order)”</a>에서 자세히 배우겠지만, 너무 유용하기 때문에 여기서 간단히 미리 보여드리고자 합니다. 예를 들어, `mpg` 데이터세트의 `class` 변수를 생각해 보세요. 클래스에 따라 고속도로 연비(highway mileage)가 어떻게 달라지는지 알고 싶을 수 있습니다:

```
ggplot(mpg, aes(x = class, y = hwy)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in10.png" alt="Side-by-side boxplots of highway mileages of cars by class. Classes are on the x-axis (2seaters, compact, midsize, minivan, pickup, subcompact, and suv)." />
</figure>

추세를 더 쉽게 보기 위해, `hwy`의 중앙값(median)을 기준으로 `class`를 다시 정렬할 수 있습니다:

```
ggplot(mpg, aes(x = fct_reorder(class, hwy, median), y = hwy)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in11.png" alt="Side-by-side boxplots of highway mileages of cars by class. Classes are on the x-axis and ordered by increasing median highway mileage (pickup, suv, minivan, 2seater, subcompact, compact, and midsize)." />
</figure>

변수 이름이 긴 경우, <a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>을 90° 뒤집으면 더 잘 작동할 것입니다. x와 y의 미적 매핑(aesthetic mappings)을 서로 교환하여 그렇게 할 수 있습니다:

```
ggplot(mpg, aes(x = hwy, y = fct_reorder(class, hwy, median))) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in12.png" alt="Side-by-side boxplots of highway mileages of cars by class. Classes are on the y-axis and ordered by increasing median highway mileage." />
</figure>

### 연습문제 (Exercises)

1. 배운 내용을 사용하여 취소된 항공편과 취소되지 않은 항공편의 출발 시간 시각화를 개선해 보세요.
2. EDA를 기반으로 할 때, 다이아몬드 데이터세트에서 다이아몬드의 가격을 예측하는 데 가장 중요한 변수는 무엇으로 보이나요? 해당 변수는 cut과 어떤 상관관계가 있나요? 두 관계의 조합이 품질이 낮은 다이아몬드가 더 비싼 이유로 이어지는 이유는 무엇일까요?
3. x와 y 변수를 교환하는 대신, 수직 상자 그림에 <a href="https://ggplot2.tidyverse.org/reference/coord_flip.html" class="orm:hideurl"><code>coord_flip()</code></a>을 새 레이어로 추가하여 수평 상자 그림을 만들어 보세요. 변수를 교환하는 것과 비교하면 어떻습니까?
4. 상자 그림의 한 가지 문제점은 훨씬 더 작은 데이터세트 시대에 개발되었기 때문에, "이상치 값(outlying values)"을 지나치게 많이 표시하는 경향이 있다는 것입니다. 이 문제를 해결하기 위한 한 가지 접근 방식은 문자값 플롯(letter value plot)입니다. lvplot 패키지를 설치하고, `geom_lv()`를 사용하여 cut 대 price의 분포를 표시해 보세요. 무엇을 알게 되나요? 플롯을 어떻게 해석하나요?
5. <a href="https://ggplot2.tidyverse.org/reference/geom_violin.html" class="orm:hideurl"><code>geom_violin()</code></a>을 사용하여 다이아몬드 가격 대 `diamonds` 데이터세트의 범주형 변수의 시각화를 생성하고, 그 다음으로 분할된(faceted) <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_histogram()</code></a>, 색상이 지정된 <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a>, 그리고 색상이 지정된 <a href="https://ggplot2.tidyverse.org/reference/geom_density.html" class="orm:hideurl"><code>geom_density()</code></a>를 차례로 생성하세요. 네 개의 플롯을 비교하고 대조해 보세요. 범주형 변수의 수준(levels)을 기준으로 수치형 변수의 분포를 시각화하는 각 방법의 장단점은 무엇입니까?
6. 데이터세트가 작을 경우, 겹쳐 그리는 것(overplotting)을 방지하여 연속형 변수와 범주형 변수 사이의 관계를 더 쉽게 보기 위해 <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>를 사용하는 것이 때때로 유용합니다. ggbeeswarm 패키지는 <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>와 유사한 여러 가지 방법을 제공합니다. 그것들을 나열하고 각각이 무엇을 하는지 간략하게 설명하세요.

## 두 개의 범주형 변수 (Two Categorical Variables)

범주형 변수 간의 공변동을 시각화하려면 이 범주형 변수들의 수준(levels)의 각 조합에 대한 관측치 수를 계산해야 합니다. 이를 수행하는 한 가지 방법은 내장된 <a href="https://ggplot2.tidyverse.org/reference/geom_count.html" class="orm:hideurl"><code>geom_count()</code></a>에 의존하는 것입니다:

```
ggplot(diamonds, aes(x = cut, y = color)) +
  geom_count()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in13.png" alt="A scatterplot of color vs. cut of diamonds. There is one point for each combination of levels of cut (Fair, Good, Very Good, Premium, and Ideal) and color (D, E, F, G, G, I, and J). The sizes of the points represent the number of observations for that combination. The legend indicates that these sizes range between 1000 and 4000." />
</figure>

플롯에서 각 원의 크기는 값의 각 조합에서 얼마나 많은 관측치가 발생했는지를 표시합니다. 공변동은 특정 x 값과 특정 y 값 사이의 강한 상관관계로 나타날 것입니다.

이러한 변수 간의 관계를 탐색하는 또 다른 접근 방식은 dplyr을 사용하여 개수를 계산하는 것입니다:

```
diamonds |> 
  count(color, cut)
#> # A tibble: 35 × 3
#>   color cut           n
#>   <ord> <ord>     <int>
#> 1 D     Fair        163
#> 2 D     Good        662
#> 3 D     Very Good  1513
#> 4 D     Premium    1603
#> 5 D     Ideal      2834
#> 6 E     Fair        224
#> # … with 29 more rows
```

그런 다음 <a href="https://ggplot2.tidyverse.org/reference/geom_tile.html" class="orm:hideurl"><code>geom_tile()</code></a>과 fill 미적 매핑을 사용하여 시각화합니다:

```
diamonds |> 
  count(color, cut) |>  
  ggplot(aes(x = color, y = cut)) +
  geom_tile(aes(fill = n))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in14.png" alt="A tile plot of cut vs. color of diamonds. Each tile represents a cut/color combination and tiles are colored according to the number of observations in each tile. There are more Ideal diamonds than other cuts, with the highest number being Ideal diamonds with color G. Fair diamonds and diamonds with color I are the lowest in frequency." />
</figure>

만약 범주형 변수가 순서가 없다면(unordered), seriation 패키지를 사용하여 흥미로운 패턴을 더 명확하게 드러내기 위해 행과 열을 동시에 다시 정렬하고 싶을 수 있습니다. 더 큰 플롯의 경우 대화형 플롯을 생성하는 heatmaply 패키지를 사용해 볼 수 있습니다.

### 연습문제 (Exercises)

1. 특정 color 내의 cut 분포, 또는 특정 cut 내의 color 분포를 더 명확하게 보여주기 위해 이전 개수 데이터세트의 척도를 어떻게 재조정(rescale)할 수 있을까요?
2. color가 `x` 미적 매핑에 연결되고 `cut`이 `fill` 미적 매핑에 연결된 분할 막대 차트(segmented bar chart)를 통해 어떤 다른 데이터 통찰력을 얻을 수 있나요? 각 세그먼트에 속하는 개수를 계산해 보세요.
3. 평균 비행 출발 지연이 목적지(destination)와 연도별 월(month of year)에 따라 어떻게 달라지는지 탐색하기 위해 <a href="https://ggplot2.tidyverse.org/reference/geom_tile.html" class="orm:hideurl"><code>geom_tile()</code></a>을 dplyr과 함께 사용해 보세요. 이 플롯을 읽기 어렵게 만드는 것은 무엇입니까? 어떻게 개선할 수 있을까요?

## 두 개의 수치형 변수 (Two Numerical Variables)

두 수치형 변수 간의 공변동을 시각화하는 훌륭한 방법을 이미 보았습니다: <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>를 사용하여 산점도(scatterplot)를 그리는 것입니다. 점들의 패턴으로 공변동을 볼 수 있습니다. 예를 들어, 다이아몬드의 캐럿 크기와 가격 사이의 양의 관계를 볼 수 있습니다: 캐럿이 높은 다이아몬드가 가격이 높습니다. 이 관계는 기하급수적(exponential)입니다.

```
ggplot(smaller, aes(x = carat, y = price)) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in15.png" alt="A scatterplot of price vs. carat. The relationship is positive, somewhat strong, and exponential." />
</figure>

(이 섹션에서는 3캐럿보다 작은 대다수의 다이아몬드에 집중하기 위해 `smaller` 데이터세트를 사용할 것입니다.)

데이터세트의 크기가 커짐에 따라, 점들이 겹쳐 그려지고(overplot) 균일한 검은색 영역으로 쌓이기 시작하므로 산점도의 유용성이 떨어집니다. 이는 2차원 공간 전체에서 데이터 밀도의 차이를 판단하기 어렵게 만들고 추세를 파악하기 어렵게 만듭니다. 우리는 이 문제를 해결하는 한 가지 방법을 이미 보았습니다: `alpha` 미적 매핑을 사용하여 투명도(transparency)를 추가하는 것입니다.

```
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_point(alpha = 1 / 100)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in16.png" alt="A scatterplot of price vs. carat. The relationship is positive, somewhat strong, and exponential. The points are transparent, showing clusters where the number of points is higher than other areas, The most obvious clusters are for diamonds with 1, 1.5, and 2 carats." />
</figure>

하지만 매우 큰 데이터세트에서는 투명도를 사용하는 것이 까다로울 수 있습니다. 또 다른 해결책은 구간(bins)을 사용하는 것입니다. 이전에 <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_histogram()</code></a>과 <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a>를 사용하여 1차원으로 구간을 나누었습니다. 이제 <a href="https://ggplot2.tidyverse.org/reference/geom_bin_2d.html" class="orm:hideurl"><code>geom_bin2d()</code></a>와 <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a>를 사용하여 2차원으로 구간을 나누는 방법을 배울 것입니다.

<a href="https://ggplot2.tidyverse.org/reference/geom_bin_2d.html" class="orm:hideurl"><code>geom_bin2d()</code></a>와 <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a>는 좌표 평면을 2D 구간으로 나누고 채우기 색상(fill color)을 사용하여 각 구간에 몇 개의 점이 속하는지 표시합니다. <a href="https://ggplot2.tidyverse.org/reference/geom_bin_2d.html" class="orm:hideurl"><code>geom_bin2d()</code></a>는 직사각형 구간을 만듭니다. <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a>는 육각형 구간을 만듭니다. <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a>를 사용하려면 hexbin 패키지를 설치해야 합니다.

```
ggplot(smaller, aes(x = carat, y = price)) +
  geom_bin2d()

# install.packages("hexbin")
ggplot(smaller, aes(x = carat, y = price)) +
  geom_hex()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in17.png" alt="Plot 1: A binned density plot of price vs. carat. Plot 2: A hexagonal bin plot of price vs. carat. Both plots show that the highest density of diamonds have low carats and low prices." />
</figure>

또 다른 선택지는 하나의 연속형 변수를 구간으로 나누어 범주형 변수처럼 작동하게 만드는 것입니다. 그런 다음, 여러분이 배웠던 범주형 변수와 연속형 변수의 조합을 시각화하는 기술 중 하나를 사용할 수 있습니다. 예를 들어, `carat`을 구간으로 나눈 다음 각 그룹에 대해 상자 그림을 표시할 수 있습니다:

```
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_boxplot(aes(group = cut_width(carat, 0.1)))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in18.png" alt="Side-by-side box plots of price by carat. Each box plot represents diamonds that are 0.1 carats apart in weight. The box plots show that as carat increases the median price increases as well. Additionally, diamonds with 1.5 carats or lower have right skewed price distributions, 1.5 to 2 have roughly symmetric price distributions, and diamonds that weigh more have left skewed distributions. Cheaper, smaller diamonds have outliers on the higher end, more expensive, bigger diamonds have outliers on the lower end." />
</figure>

여기에 사용된 `cut_width(x, width)`는 `x`를 너비가 `width`인 구간으로 나눕니다. 기본적으로 상자 그림은 관측치가 얼마나 많든 간에 대략 동일하게 보이기 때문에(이상치의 수는 제외하고), 각 상자 그림이 서로 다른 수의 점을 요약하고 있다는 것을 알기 어렵습니다. 이를 보여주는 한 가지 방법은 `varwidth = TRUE`를 사용하여 상자 그림의 너비를 점의 개수에 비례하도록 만드는 것입니다.

### 연습문제 (Exercises)

1. 상자 그림으로 조건부 분포를 요약하는 대신, 도수 다각형(frequency polygon)을 사용할 수 있습니다. <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_width()</code></a>와 <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_number()</code></a>를 사용할 때 무엇을 고려해야 하나요? 이것이 `carat`과 `price`의 2D 분포 시각화에 어떤 영향을 미치나요?
2. `price`로 분할된 `carat`의 분포를 시각화하세요.
3. 매우 큰 다이아몬드의 가격 분포는 작은 다이아몬드와 어떻게 비교됩니까? 예상한 대로인가요, 아니면 놀라운가요?
4. 배운 두 가지 기술을 결합하여 cut, carat, price의 결합 분포(combined distribution)를 시각화하세요.
5. 2차원 플롯은 1차원 플롯에서는 보이지 않는 이상치를 드러냅니다. 예를 들어, 다음 플롯의 일부 점들은 이례적인 `x`와 `y` 값의 조합을 가지고 있어, 개별적으로 검토할 때 `x`와 `y` 값이 정상으로 보임에도 불구하고 이 점들을 이상치로 만듭니다. 이 경우 구간화된 플롯(binned plot)보다 산점도가 더 나은 디스플레이인 이유는 무엇입니까?

    ```
    diamonds |> 
      filter(x >= 4) |> 
      ggplot(aes(x = x, y = y)) +
      geom_point() +
      coord_cartesian(xlim = c(4, 11), ylim = c(4, 11))
    ```

6. <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_width()</code></a>를 사용하여 너비가 같은 상자를 만드는 대신, <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_number()</code></a>를 사용하여 대략 같은 수의 점을 포함하는 상자를 만들 수 있습니다. 이 접근 방식의 장점과 단점은 무엇입니까?

    ```
    ggplot(smaller, aes(x = carat, y = price)) + 
      geom_boxplot(aes(group = cut_number(carat, 20)))
    ```

# 패턴과 모델 (Patterns and Models)

두 변수 사이에 체계적인 관계가 존재한다면, 이는 데이터에 패턴으로 나타날 것입니다. 패턴을 발견하면 스스로에게 물어보세요:

- 이 패턴이 우연(즉, 무작위 확률)에 의한 것일 수 있는가?
- 패턴이 암시하는 관계를 어떻게 설명할 수 있는가?
- 패턴이 암시하는 관계는 얼마나 강한가?
- 다른 어떤 변수가 관계에 영향을 미칠 수 있는가?
- 데이터의 개별 하위 그룹(subgroups)을 살펴보면 관계가 바뀌는가?

데이터의 패턴은 관계에 대한 단서를 제공합니다. 즉, 공변동을 드러냅니다. 변동(variation)을 불확실성을 만들어내는 현상으로 생각한다면, 공변동(covariation)은 그것을 줄여주는 현상입니다. 두 변수가 공변동한다면, 한 변수의 값을 사용하여 두 번째 변수의 값에 대해 더 나은 예측을 할 수 있습니다. 만약 공변동이 인과 관계(특별한 경우) 때문이라면, 한 변수의 값을 사용하여 두 번째 변수의 값을 통제할 수 있습니다.

모델(Models)은 데이터에서 패턴을 추출하기 위한 도구입니다. 예를 들어 다이아몬드 데이터를 생각해 보겠습니다. cut과 carat, carat과 price가 밀접하게 관련되어 있기 때문에 cut과 price 사이의 관계를 이해하기 어렵습니다. 모델을 사용하여 price와 carat 사이의 매우 강한 관계를 제거하여 남아있는 미묘한 차이를 탐색하는 것이 가능합니다. 다음 코드는 `carat`으로부터 `price`를 예측하는 모델을 피팅(fit)한 다음, 잔차(residuals; 예측된 값과 실제 값의 차이)를 계산합니다. 잔차는 carat의 효과가 제거된 후의 다이아몬드 가격을 보여줍니다. `price`와 `carat`의 원래 값(raw values)을 사용하는 대신, 먼저 로그 변환을 수행하고 로그 변환된 값에 모델을 피팅한다는 점에 유의하세요. 그런 다음, 우리는 잔차를 원래 가격 척도로 되돌려놓기 위해 지수화(exponentiate)합니다.

```
library(tidymodels)

diamonds <- diamonds |>
  mutate(
    log_price = log(price),
    log_carat = log(carat)
  )

diamonds_fit <- linear_reg() |>
  fit(log_price ~ log_carat, data = diamonds)

diamonds_aug <- augment(diamonds_fit, new_data = diamonds) |>
  mutate(.resid = exp(.resid))

ggplot(diamonds_aug, aes(x = carat, y = .resid)) + 
  geom_point()
```
