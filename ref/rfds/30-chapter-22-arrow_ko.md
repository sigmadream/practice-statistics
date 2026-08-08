# 제22장. Arrow

# 소개

CSV 파일은 사람이 읽기 쉽게 설계되었습니다. 간단하고 세상의 모든 도구에서 읽을 수 있기 때문에 훌륭한 교환 형식입니다. 하지만 CSV 파일은 효율적이지 않습니다. 데이터를 R로 읽어 들이려면 꽤 많은 작업을 해야 합니다. 이 장에서는 강력한 대안에 대해 배울 것입니다. 바로 빅 데이터 시스템에서 널리 사용되는 개방형 표준 기반 형식인 [파케이(parquet) 형식](https://oreil.ly/ClE7D)입니다.

우리는 파케이 파일을 대규모 데이터 세트의 효율적인 분석과 전송을 위해 설계된 다국어 툴박스인 [Apache Arrow](https://oreil.ly/TGrH5)와 결합할 것입니다. 우리는 익숙한 dplyr 구문을 사용하여 메모리보다 큰 데이터 세트를 분석할 수 있게 해주는 dplyr 백엔드를 제공하는 [arrow 패키지](https://oreil.ly/g60F8)를 통해 Apache Arrow를 사용할 것입니다. 추가적인 이점으로 arrow는 매우 빠릅니다. 이 장 후반부에서 몇 가지 예를 보게 될 것입니다.

arrow와 dbplyr 모두 dplyr 백엔드를 제공하므로 언제 어느 것을 사용해야 할지 궁금할 수 있습니다. 데이터가 이미 데이터베이스나 파케이 파일에 있어서 그대로 작업하고 싶은 경우처럼 많은 경우에 선택은 이미 정해져 있습니다. 하지만 자신의 데이터(아마도 CSV 파일)로 시작한다면 데이터베이스로 로드하거나 파케이로 변환할 수 있습니다. 일반적으로 어떤 것이 가장 잘 작동할지 알기 어렵기 때문에 분석의 초기 단계에서는 둘 다 시도해 보고 자신에게 가장 잘 맞는 것을 선택하시길 권장합니다.

(이 장의 초기 버전을 기고해 주신 Danielle Navarro에게 큰 감사를 드립니다.)

## 사전 준비

이 장에서도 계속해서 tidyverse, 특히 dplyr를 사용하겠지만, 이를 대규모 데이터 작업용으로 특별히 설계된 arrow 패키지와 결합할 것입니다:

```
library(tidyverse)
library(arrow)
```

이 장의 후반부에서는 arrow와 duckdb 사이의 몇 가지 연결성도 살펴볼 것이므로 dbplyr와 duckdb도 필요합니다:

```
library(dbplyr, warn.conflicts = FALSE)
library(duckdb)
#> Loading required package: DBI
```

# 데이터 가져오기

우리는 이러한 도구에 걸맞은 데이터 세트, 즉 온라인([Seattle Open Data](https://oreil.ly/u56DR))에서 이용 가능한 시애틀 공공 도서관의 대출 항목 데이터 세트를 가져오는 것으로 시작합니다. 이 데이터 세트에는 2005년 4월부터 2022년 10월까지 매달 각 책이 몇 번 대출되었는지를 알려주는 41,389,465개의 행이 포함되어 있습니다.

다음 코드를 사용하면 데이터의 캐시된 복사본을 얻을 수 있습니다. 데이터가 9GB 크기의 CSV 파일이므로 다운로드하는 데 시간이 좀 걸릴 것입니다. 매우 큰 파일을 받을 때는 정확히 이 목적으로 만들어진 `curl::multi_download()`를 사용하는 것을 강력히 권장합니다. 이 함수는 진행률 표시줄을 제공하며 중단된 경우 다운로드를 재개할 수 있습니다.

```
dir.create("data", showWarnings = FALSE)

curl::multi_download(
  "https://r4ds.s3.us-west-2.amazonaws.com/seattle-library-checkouts.csv",
  "data/seattle-library-checkouts.csv",
  resume = TRUE
)
```

# 데이터 세트 열기

데이터를 살펴보는 것으로 시작해 보겠습니다. 9GB 크기인 이 파일은 꽤 커서 아마 전체를 메모리에 로드하고 싶지는 않을 것입니다. 일반적으로 데이터 크기의 최소 두 배 이상의 메모리가 필요하다는 것이 좋은 경험 법칙이며, 많은 노트북 컴퓨터는 최대 16GB의 메모리를 갖추고 있습니다. 이는 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>read_csv()</code></a>를 피하고 대신 <a href="https://arrow.apache.org/docs/r/reference/open_dataset.html" class="orm:hideurl"><code>arrow::open_dataset()</code></a>을 사용해야 함을 의미합니다:

```
seattle_csv <- open_dataset(
  sources = "data/seattle-library-checkouts.csv", 
  format = "csv"
)
```

이 코드가 실행되면 무슨 일이 일어날까요? <a href="https://arrow.apache.org/docs/r/reference/open_dataset.html" class="orm:hideurl"><code>open_dataset()</code></a>은 데이터 세트의 구조를 파악하기 위해 몇 천 개의 행을 스캔합니다. 그런 다음 찾은 것을 기록하고 멈춥니다. 여러분이 구체적으로 요청할 때만 더 많은 행을 읽을 것입니다. `seattle_csv`를 출력하면 이러한 메타데이터를 볼 수 있습니다:

```
seattle_csv
#> FileSystemDataset with 1 csv file
#> UsageClass: string
#> CheckoutType: string
#> MaterialType: string
#> CheckoutYear: int64
#> CheckoutMonth: int64
#> Checkouts: int64
#> Title: string
#> ISBN: null
#> Creator: string
#> Subjects: string
#> Publisher: string
#> PublicationYear: string
```

출력의 첫 번째 줄은 `seattle_csv`가 디스크에 단일 CSV 파일로 로컬에 저장되어 있음을 알려줍니다. 이 파일은 필요할 때만 메모리에 로드됩니다. 출력의 나머지 부분은 arrow가 각 열에 대해 추정한 열 유형을 알려줍니다.

<a href="https://pillar.r-lib.org/reference/glimpse.html" class="orm:hideurl"><code>glimpse()</code></a>를 사용하면 실제로 무엇이 들어있는지 볼 수 있습니다. 약 4,100만 개의 행과 12개의 열이 있음을 밝혀주고 몇 가지 값을 보여줍니다.

```
seattle_csv |> glimpse()
#> FileSystemDataset with 1 csv file
#> 41,389,465 rows x 12 columns
#> $ UsageClass      <string> "Physical", "Physical", "Digital", "Physical", "Ph…
#> $ CheckoutType    <string> "Horizon", "Horizon", "OverDrive", "Horizon", "Hor…
#> $ MaterialType    <string> "BOOK", "BOOK", "EBOOK", "BOOK", "SOUNDDISC", "BOO…
#> $ CheckoutYear     <int64> 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 20…
#> $ CheckoutMonth    <int64> 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,…
#> $ Checkouts        <int64> 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 3, 2, 1, 3, 2,…
#> $ Title           <string> "Super rich : a guide to having it all / Russell S…
#> $ ISBN            <string> "", "", "", "", "", "", "", "", "", "", "", "", ""…
#> $ Creator         <string> "Simmons, Russell", "Barclay, James, 1965-", "Tim …
#> $ Subjects        <string> "Self realization, Conduct of life, Attitude Psych…
#> $ Publisher       <string> "Gotham Books,", "Pyr,", "Random House, Inc.", "Di…
#> $ PublicationYear <string> "c2011.", "2010.", "2015", "2005.", "c2004.", "c20…
```

arrow가 계산을 수행하고 약간의 데이터를 반환하도록 강제하는 <a href="https://dplyr.tidyverse.org/reference/compute.html" class="orm:hideurl"><code>collect()</code></a>를 사용함으로써, dplyr 동사들과 함께 이 데이터 세트를 사용하기 시작할 수 있습니다. 예를 들어 다음 코드는 연도별 총 대출 횟수를 알려줍니다:

```
seattle_csv |> 
  count(CheckoutYear, wt = Checkouts) |> 
  arrange(CheckoutYear) |> 
  collect()
#> # A tibble: 18 × 2
#>   CheckoutYear       n
#>          <int>   <int>
#> 1         2005 3798685
#> 2         2006 6599318
#> 3         2007 7126627
#> 4         2008 8438486
#> 5         2009 9135167
#> 6         2010 8608966
#> # … with 12 more rows
```

arrow 덕분에 이 코드는 기본 데이터 세트가 아무리 크더라도 작동할 것입니다. 하지만 현재는 다소 느립니다. Hadley의 컴퓨터에서는 실행하는 데 약 10초가 걸렸습니다. 우리가 가진 데이터의 양을 고려하면 끔찍한 수준은 아니지만 더 나은 형식으로 전환하면 훨씬 더 빠르게 만들 수 있습니다.

# 파케이 형식

이 데이터로 작업하기 더 쉽게 만들기 위해 파케이 파일 형식으로 전환하고 여러 파일로 나누어 보겠습니다. 다음 섹션에서는 먼저 파케이와 파티셔닝(partitioning)을 소개하고 배운 것을 시애틀 도서관 데이터에 적용할 것입니다.

## 파케이의 장점

CSV와 마찬가지로 파케이는 직사각형 구조의 데이터에 사용되지만, 아무 파일 편집기로나 읽을 수 있는 텍스트 형식 대신 빅 데이터의 요구에 맞게 특별히 설계된 사용자 정의 바이너리 형식입니다. 이는 다음을 의미합니다:

- 파케이 파일은 일반적으로 동일한 CSV 파일보다 작습니다. 파케이는 파일 크기를 줄이기 위해 [효율적인 인코딩](https://oreil.ly/OzpFo)에 의존하며 파일 압축을 지원합니다. 디스크에서 메모리로 이동할 데이터가 적기 때문에 이는 파케이 파일을 빠르게 만드는 데 도움이 됩니다.

- 파케이 파일은 풍부한 타입 시스템을 가지고 있습니다. <a href="ch07.html#sec-col-types" data-type="xref">“열 유형 제어하기”</a>에서 이야기했듯이 CSV 파일은 열 유형에 대한 어떤 정보도 제공하지 않습니다. 예를 들어 CSV 리더는 `"08-10-2022"`를 문자열로 구문 분석해야 할지 아니면 날짜로 구문 분석해야 할지 추측해야 합니다. 이와 대조적으로 파케이 파일은 데이터와 함께 타입을 기록하는 방식으로 데이터를 저장합니다.

- 파케이 파일은 "열 지향(column-oriented)"입니다. 즉, R의 데이터 프레임과 마찬가지로 열별로 정리되어 있습니다. 이는 일반적으로 행별로 정리된 CSV 파일에 비해 데이터 분석 작업에서 더 나은 성능을 이끌어냅니다.

- 파케이 파일은 "청크(chunk)"로 나뉘어져 있어 파일의 여러 부분에서 동시에 작업할 수 있으며, 운이 좋으면 일부 청크를 완전히 건너뛸 수도 있습니다.

## 파티셔닝

데이터 세트가 점점 커짐에 따라 모든 데이터를 단일 파일에 저장하는 것은 점점 더 고통스러워지며, 대규모 데이터 세트를 여러 파일로 분할하는 것이 유용한 경우가 많습니다. 이 구조화가 지능적으로 수행되면 많은 분석에서 파일의 하위 집합만 필요로 하기 때문에 성능이 크게 향상될 수 있습니다.

데이터 세트를 분할하는 방법에 대한 엄격하고 빠른 규칙은 없습니다. 결과는 여러분의 데이터, 접근 패턴, 데이터를 읽는 시스템에 따라 달라질 것입니다. 여러분의 상황에 이상적인 파티셔닝을 찾기 전에 약간의 실험을 해야 할 가능성이 높습니다. 대략적인 지침으로 arrow는 20MB보다 작거나 2GB보다 큰 파일을 피하고 10,000개 이상의 파일을 생성하는 파티셔닝을 피할 것을 제안합니다. 또한 필터링에 사용할 변수 기준으로 파티셔닝을 시도해야 합니다. 곧 보게 되겠지만 그렇게 하면 arrow가 관련 파일만 읽음으로써 많은 작업을 건너뛸 수 있습니다.

## 시애틀 도서관 데이터 다시 쓰기

이러한 아이디어를 시애틀 도서관 데이터에 적용하여 실제로 어떻게 작동하는지 살펴보겠습니다. 일부 분석에서는 최근 데이터만 보고 싶어 할 가능성이 높고, 연도별로 파티셔닝하면 적당한 크기의 18개 청크가 생성되므로 `CheckoutYear`를 기준으로 파티셔닝할 것입니다.

데이터를 다시 쓰기 위해 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>dplyr::group_by()</code></a>를 사용하여 파티션을 정의한 다음 <a href="https://arrow.apache.org/docs/r/reference/write_dataset.html" class="orm:hideurl"><code>arrow::write_dataset()</code></a>을 사용하여 파티션들을 디렉토리에 저장합니다. <a href="https://arrow.apache.org/docs/r/reference/write_dataset.html" class="orm:hideurl"><code>write_dataset()</code></a>에는 파일을 생성할 디렉토리와 사용할 형식이라는 두 가지 중요한 인자가 있습니다.

```
pq_path <- "data/seattle-library-checkouts"

seattle_csv |>
  group_by(CheckoutYear) |>
  write_dataset(path = pq_path, format = "parquet")
```

이를 실행하는 데 1분 정도 걸립니다. 곧 보게 되겠지만 이는 향후 작업을 훨씬 훨씬 빠르게 만들어주는 초기 투자입니다.

우리가 방금 무엇을 만들어냈는지 살펴보겠습니다:

```
tibble(
  files = list.files(pq_path, recursive = TRUE),
  size_MB = file.size(file.path(pq_path, files)) / 1024^2
)
#> # A tibble: 18 × 2
#>   files                            size_MB
#>   <chr>                              <dbl>
#> 1 CheckoutYear=2005/part-0.parquet    109.
#> 2 CheckoutYear=2006/part-0.parquet    164.
#> 3 CheckoutYear=2007/part-0.parquet    178.
#> 4 CheckoutYear=2008/part-0.parquet    195.
#> 5 CheckoutYear=2009/part-0.parquet    214.
#> 6 CheckoutYear=2010/part-0.parquet    222.
#> # … with 12 more rows
```

9GB짜리 단일 CSV 파일이 18개의 파케이 파일로 다시 작성되었습니다. 파일 이름은 [Apache Hive 프로젝트](https://oreil.ly/kACzC)에서 사용하는 "자기 설명적(self-describing)" 규칙을 사용합니다. Hive 스타일 파티션은 "key=value" 규칙으로 폴더 이름을 지정하므로 짐작할 수 있듯이 `CheckoutYear=2005` 디렉토리에는 `CheckoutYear`가 2005인 모든 데이터가 포함됩니다. 각 파일은 100~300MB 사이이며 현재 총 크기는 약 4GB로 원래 CSV 파일 크기의 절반보다 조금 더 큽니다. 파케이가 훨씬 더 효율적인 형식이므로 이는 우리가 예상한 바입니다.

# Arrow와 함께 dplyr 사용하기

이제 파케이 파일들을 만들었으니 이들을 다시 읽어 들여야 합니다. <a href="https://arrow.apache.org/docs/r/reference/open_dataset.html" class="orm:hideurl"><code>open_dataset()</code></a>을 다시 사용하지만 이번에는 디렉토리를 제공합니다:

```
seattle_pq <- open_dataset(pq_path)
```

이제 dplyr 파이프라인을 작성할 수 있습니다. 예를 들어 지난 5년 동안 매달 대출된 책의 총 개수를 셀 수 있습니다:

```
query <- seattle_pq |> 
  filter(CheckoutYear >= 2018, MaterialType == "BOOK") |>
  group_by(CheckoutYear, CheckoutMonth) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(CheckoutYear, CheckoutMonth)
```

arrow 데이터용 dplyr 코드를 작성하는 것은 <a href="ch21.html#chp-databases" data-type="xref">21장</a>에서 논의한 것처럼 개념적으로 dbplyr와 유사합니다. dplyr 코드를 작성하면 Apache Arrow C++ 라이브러리가 이해할 수 있는 쿼리로 자동 변환되며, <a href="https://dplyr.tidyverse.org/reference/compute.html" class="orm:hideurl"><code>collect()</code></a>를 호출할 때 실행됩니다. `query` 객체를 출력하면 실행 시 Arrow가 무엇을 반환할 것으로 예상되는지에 대한 약간의 정보를 볼 수 있습니다:

```
query
#> FileSystemDataset (query)
#> CheckoutYear: int32
#> CheckoutMonth: int64
#> TotalCheckouts: int64
#> 
#> * Grouped by CheckoutYear
#> * Sorted by CheckoutYear [asc], CheckoutMonth [asc]
#> See $.data for the source Arrow object
```

그리고 <a href="https://dplyr.tidyverse.org/reference/compute.html" class="orm:hideurl"><code>collect()</code></a>를 호출하여 결과를 얻을 수 있습니다:

```
query |> collect()
#> # A tibble: 58 × 3
#> # Groups:   CheckoutYear [5]
#>   CheckoutYear CheckoutMonth TotalCheckouts
#>          <int>         <int>          <int>
#> 1         2018             1         355101
#> 2         2018             2         309813
#> 3         2018             3         344487
#> 4         2018             4         330988
#> 5         2018             5         318049
#> 6         2018             6         341825
#> # … with 52 more rows
```

dbplyr와 마찬가지로 arrow는 일부 R 표현식만 이해하므로 평소 작성하던 코드와 정확히 똑같이 작성하지 못할 수도 있습니다. 하지만 지원되는 작업 및 함수 목록은 꽤 방대하며 계속해서 증가하고 있습니다. 현재 지원되는 함수의 전체 목록은 <a href="https://arrow.apache.org/docs/r/reference/acero.html" class="orm:hideurl"><code>?acero</code></a>에서 찾아보세요.

## 성능

CSV에서 파케이로 전환할 때의 성능 영향을 간단히 살펴보겠습니다. 먼저 데이터가 하나의 대형 CSV 파일로 저장되어 있을 때 2021년 매달 대출된 책의 수를 계산하는 데 걸리는 시간을 측정해 보겠습니다:

```
seattle_csv |> 
  filter(CheckoutYear == 2021, MaterialType == "BOOK") |>
  group_by(CheckoutMonth) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(desc(CheckoutMonth)) |>
  collect() |> 
  system.time()
#>    user  system elapsed 
#>  11.997   1.189  11.343
```

이제 시애틀 도서관 대출 데이터가 18개의 더 작은 파케이 파일로 분할된 새 버전의 데이터 세트를 사용해 보겠습니다:

```
seattle_pq |> 
  filter(CheckoutYear == 2021, MaterialType == "BOOK") |>
  group_by(CheckoutMonth) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(desc(CheckoutMonth)) |>
  collect() |> 
  system.time()
#>    user  system elapsed 
#>   0.272   0.063   0.063
```

약 100배의 성능 향상은 두 가지 요인 때문입니다: 다중 파일 파티셔닝과 개별 파일의 형식입니다:

- 파티셔닝은 성능을 향상시킵니다. 왜냐하면 이 쿼리는 `CheckoutYear == 2021`을 사용하여 데이터를 필터링하며, arrow는 18개의 파케이 파일 중 하나만 읽으면 된다는 것을 인식할 만큼 똑똑하기 때문입니다.
- 파케이 형식은 메모리로 더 직접적으로 읽을 수 있는 바이너리 형식으로 데이터를 저장함으로써 성능을 향상시킵니다. 열 단위 형식과 풍부한 메타데이터 덕분에 arrow는 쿼리에 실제로 사용되는 네 개의 열(`CheckoutYear`, `MaterialType`, `CheckoutMonth`, `Checkouts`)만 읽으면 됩니다.

이 엄청난 성능 차이가 바로 큰 CSV 파일을 파케이로 변환하는 것이 가치 있는 이유입니다!

## Arrow와 함께 dbplyr 사용하기

파케이와 arrow의 마지막 이점이 하나 더 있습니다. <a href="https://arrow.apache.org/docs/r/reference/to_duckdb.html" class="orm:hideurl"><code>arrow::to_duckdb()</code></a>를 호출하여 arrow 데이터 세트를 DuckDB 데이터베이스(<a href="ch21.html#chp-databases" data-type="xref">21장</a>)로 쉽게 전환할 수 있다는 것입니다:

```
seattle_pq |> 
  to_duckdb() |>
  filter(CheckoutYear >= 2018, MaterialType == "BOOK") |>
  group_by(CheckoutYear) |>
  summarize(TotalCheckouts = sum(Checkouts)) |>
  arrange(desc(CheckoutYear)) |>
  collect()
#> Warning: Missing values are always removed in SQL aggregation functions.
#> Use `na.rm = TRUE` to silence this warning
#> This warning is displayed once every 8 hours.
#> # A tibble: 5 × 2
#>   CheckoutYear TotalCheckouts
#>          <int>          <dbl>
#> 1         2022        2431502
#> 2         2021        2266438
#> 3         2020        1241999
#> 4         2019        3931688
#> 5         2018        3987569
```

<a href="https://arrow.apache.org/docs/r/reference/to_duckdb.html" class="orm:hideurl"><code>to_duckdb()</code></a>의 깔끔한 점은 전송에 메모리 복사가 포함되지 않으며 한 컴퓨팅 환경에서 다른 환경으로의 원활한 전환을 가능하게 한다는 arrow 생태계의 목표를 잘 보여준다는 것입니다.

# 요약

이 장에서는 디스크에 있는 대규모 데이터 세트 작업을 위한 dplyr 백엔드를 제공하는 arrow 패키지의 맛을 보았습니다. CSV 파일과도 작동할 수 있지만 데이터를 파케이로 변환하면 훨씬 훨씬 더 빠릅니다. 파케이는 최신 컴퓨터에서 데이터 분석을 위해 특별히 설계된 바이너리 데이터 형식입니다. CSV에 비해 파케이 파일로 작업할 수 있는 도구는 훨씬 적지만 파티셔닝되고 압축된 열 형태의 구조 덕분에 분석이 훨씬 더 효율적입니다.

다음으로 tidyr 패키지가 제공하는 도구를 사용하여 처리할 첫 번째 비직사각형 데이터 소스에 대해 배울 것입니다. JSON 파일에서 가져온 데이터에 초점을 맞추겠지만 일반적인 원칙은 출처에 관계없이 트리(tree) 형태의 데이터에 적용됩니다.
