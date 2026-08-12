# 제21장. 데이터베이스

# 소개

엄청난 양의 데이터가 데이터베이스에 존재하기 때문에, 데이터베이스에 접근하는 방법을 아는 것은 필수적입니다. 때로는 누군가에게 특정 시점의 데이터를 `.csv` 파일로 다운로드해 달라고 부탁할 수 있지만, 이는 금세 고통스러운 작업이 됩니다. 변경 사항을 적용해야 할 때마다 다른 사람과 소통해야 하기 때문입니다. 여러분은 필요할 때 필요한 데이터를 얻기 위해 데이터베이스에 직접 접근할 수 있기를 원할 것입니다.

이 장에서는 먼저 DBI 패키지의 기본 사항을 배웁니다. 데이터베이스에 연결하고 SQL<sup><a href="ch21.html#idm44771280827536" id="idm44771280827536-marker" data-type="noteref">1</a></sup> 쿼리를 사용하여 데이터를 검색하는 방법입니다. 구조화된 질의어(Structured Query Language)의 약자인 *SQL*은 데이터베이스의 공용어이며 모든 데이터 과학자가 배워야 할 중요한 언어입니다. 그렇지만 SQL로 시작하지 않고, 대신 dplyr 코드를 SQL로 번역해 줄 수 있는 dbplyr를 가르칠 것입니다. 이를 통해 SQL의 중요한 기능 몇 가지를 가르치는 방법으로 사용할 것입니다. 이 장이 끝날 때쯤 여러분이 SQL의 달인이 되지는 않겠지만, 중요한 구성 요소를 식별하고 그것들이 어떤 역할을 하는지 이해할 수는 있을 것입니다.

## 사전 준비

이 장에서는 DBI와 dbplyr를 소개합니다. DBI는 데이터베이스에 연결하고 SQL을 실행하는 저수준 인터페이스이며, dbplyr는 dplyr 코드를 SQL 쿼리로 번역한 다음 DBI를 사용하여 실행하는 고수준 인터페이스입니다.

```
library(DBI)
library(dbplyr)
library(tidyverse)
```

# 데이터베이스 기초

단순한 수준에서 데이터베이스는 데이터베이스 용어로 *테이블(table)*이라고 불리는 데이터 프레임의 모음이라고 생각할 수 있습니다. `data.frame`과 마찬가지로 데이터베이스 테이블은 이름이 지정된 열의 모음이며, 열의 모든 값은 동일한 유형입니다. 데이터 프레임과 데이터베이스 테이블 간에는 세 가지 주요 차이점이 있습니다.

- 데이터베이스 테이블은 디스크에 저장되며 임의로 커질 수 있습니다. 데이터 프레임은 메모리에 저장되며 근본적으로 제한이 있습니다(물론 그 제한도 많은 문제에 대해 여전히 충분히 크긴 하지만요).

- 데이터베이스 테이블은 거의 항상 인덱스(index)를 가지고 있습니다. 책의 색인과 마찬가지로 데이터베이스 인덱스는 모든 행을 일일이 살펴보지 않고도 관심 있는 행을 빠르게 찾을 수 있게 해줍니다. 데이터 프레임과 티블에는 인덱스가 없지만, 데이터베이스 테이블에는 인덱스가 있으며, 이것이 그들이 매우 빠른 이유 중 하나입니다.

- 대부분의 고전적인 데이터베이스는 기존 데이터를 분석하는 것이 아니라 데이터를 빠르게 수집하는 데 최적화되어 있습니다. 이러한 데이터베이스는 R처럼 열 단위가 아니라 행 단위로 데이터를 저장하기 때문에 *행 지향(row-oriented)*이라고 불립니다. 최근에는 기존 데이터 분석을 훨씬 더 빠르게 해주는 _열 지향(column-oriented)_ 데이터베이스가 많이 개발되었습니다.

데이터베이스는 데이터베이스 관리 시스템(간단히 _DBMS_)에 의해 운영되며, 크게 세 가지 기본 형태가 있습니다.

- _클라이언트-서버(Client-server)_ DBMS는 여러분이 컴퓨터(클라이언트)에서 연결하는 강력한 중앙 서버에서 실행됩니다. 조직 내의 여러 사람과 데이터를 공유하는 데 적합합니다. 널리 사용되는 클라이언트-서버 DBMS로는 PostgreSQL, MariaDB, SQL Server, Oracle 등이 있습니다.
- Snowflake, Amazon의 RedShift, Google의 BigQuery와 같은 _클라우드(Cloud)_ DBMS는 클라이언트-서버 DBMS와 유사하지만 클라우드에서 실행됩니다. 이는 매우 큰 데이터 세트를 쉽게 처리할 수 있고 필요에 따라 자동으로 더 많은 컴퓨팅 리소스를 제공할 수 있음을 의미합니다.
- SQLite 또는 duckdb와 같은 _인-프로세스(In-process)_ DBMS는 전적으로 여러분의 컴퓨터에서 실행됩니다. 여러분이 주요 사용자인 대규모 데이터 세트 작업에 아주 좋습니다.

# 데이터베이스 연결하기

R에서 데이터베이스에 연결하려면 패키지 쌍을 사용해야 합니다.

- 데이터베이스에 연결하고, 데이터를 업로드하고, SQL 쿼리를 실행하는 등의 일반적인(generic) 기능 집합을 제공하는 DBI(*d*ata*b*ase *i*nterface)는 항상 사용하게 됩니다.

- 또한 연결하려는 DBMS에 맞춤화된 패키지도 사용합니다. 이 패키지는 일반적인 DBI 명령을 특정 DBMS에 필요한 구체적인 명령으로 번역합니다. PostgreSQL의 경우 RPostgres, MySQL의 경우 RMariaDB와 같이 대개 각 DBMS마다 하나의 패키지가 있습니다.

특정 DBMS를 위한 패키지를 찾을 수 없다면 보통 odbc 패키지를 대신 사용할 수 있습니다. 이는 많은 DBMS에서 지원하는 ODBC 프로토콜을 사용합니다. odbc는 ODBC 드라이버도 설치하고 odbc 패키지에게 해당 드라이버의 위치를 알려주어야 하므로 약간의 추가 설정이 필요합니다.

구체적으로 <a href="https://dbi.r-dbi.org/reference/dbConnect.html" class="orm:hideurl"><code>DBI::dbConnect()</code></a>를 사용하여 데이터베이스 연결을 생성합니다. 첫 번째 인자는 DBMS를 선택하고,<sup><a href="ch21.html#idm44771280761328" id="idm44771280761328-marker" data-type="noteref">2</a></sup> 두 번째 및 이후 인자들은 데이터베이스에 연결하는 방법(즉, 어디에 있는지와 접근에 필요한 자격 증명)을 설명합니다. 다음 코드는 두 가지 전형적인 예를 보여줍니다.

```
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

연결에 대한 정확한 세부 사항은 DBMS마다 크게 다르기 때문에 아쉽게도 여기에서 모든 세부 사항을 다룰 수는 없습니다. 이는 여러분이 스스로 약간의 조사를 해야 함을 의미합니다. 일반적으로 팀의 다른 데이터 과학자에게 묻거나 DBA(데이터베이스 관리자)와 상의할 수 있습니다. 초기 설정은 올바르게 하기 위해 약간의 조정(그리고 약간의 구글링)이 필요한 경우가 많지만, 보통 한 번만 수행하면 됩니다.

## 이 책에서는

이 책을 위해 클라이언트-서버 또는 클라우드 DBMS를 설정하는 것은 고통스러운 일일 것이므로, 대신 R 패키지 내에 전적으로 존재하는 인-프로세스 DBMS인 duckdb를 사용할 것입니다. DBI의 마법 덕분에 duckdb를 사용하는 것과 다른 DBMS를 사용하는 것 사이의 유일한 차이점은 데이터베이스에 연결하는 방법뿐입니다. 쉽게 이 코드를 실행해 볼 수 있고, 배운 것을 쉽게 다른 곳에 적용할 수 있기 때문에 교육용으로 아주 훌륭합니다.

duckdb에 연결하는 것은 특별히 간단한데, 기본적으로 R을 종료할 때 삭제되는 임시 데이터베이스를 생성하기 때문입니다. R을 다시 시작할 때마다 백지 상태에서 시작할 수 있도록 보장해 주므로 학습에 매우 좋습니다.

```
con <- DBI::dbConnect(duckdb::duckdb())
```

duckdb는 데이터 과학자의 요구에 매우 잘 맞게 설계된 고성능 데이터베이스입니다. 쉽게 시작할 수 있으면서도 기가바이트 단위의 데이터를 엄청난 속도로 처리할 수 있는 능력이 있기 때문에 여기서 사용합니다. 실제 데이터 분석 프로젝트에 duckdb를 사용하고 싶다면, 영구적인 데이터베이스를 만들고 duckdb에게 어디에 저장할지 알려주기 위해 `dbdir` 인자도 제공해야 합니다. 프로젝트(<a href="ch06.html#chp-workflow-scripts" data-type="xref">6장</a>)를 사용하고 있다고 가정할 때, 현재 프로젝트의 `duckdb` 디렉토리에 저장하는 것이 합리적입니다.

```
con <- DBI::dbConnect(duckdb::duckdb(), dbdir = "duckdb")
```

## 데이터 로드하기

이것은 새로운 데이터베이스이므로 일부 데이터를 추가하는 것으로 시작해야 합니다. 여기서는 <a href="https://dbi.r-dbi.org/reference/dbWriteTable.html" class="orm:hideurl"><code>DBI::dbWriteTable()</code></a>을 사용하여 ggplot2의 `mpg`와 `diamonds` 데이터 세트를 추가하겠습니다. <a href="https://dbi.r-dbi.org/reference/dbWriteTable.html" class="orm:hideurl"><code>dbWriteTable()</code></a>의 간단한 사용법은 데이터베이스 연결, 데이터베이스에 생성할 테이블의 이름, 데이터의 데이터 프레임이라는 세 가지 인자를 필요로 합니다.

```
dbWriteTable(con, "mpg", ggplot2::mpg)
dbWriteTable(con, "diamonds", ggplot2::diamonds)
```

실제 프로젝트에서 duckdb를 사용하는 경우 `duckdb_read_csv()`와 `duckdb_register_arrow()`에 대해 알아보기를 강력히 권장합니다. 이들은 데이터를 먼저 R로 로드할 필요 없이 직접 duckdb로 빠르게 로드할 수 있는 강력하고 성능 좋은 방법을 제공합니다. <a href="ch26.html#sec-save-database" data-type="xref">“데이터베이스에 쓰기”</a>에서 여러 파일을 데이터베이스로 로드하는 유용한 기술도 보여드리겠습니다.

## DBI 기초

두 가지 다른 DBI 함수를 사용하여 데이터가 올바르게 로드되었는지 확인할 수 있습니다. `dbListTable()`은 데이터베이스의 모든 테이블 목록을 표시하며,<sup><a href="ch21.html#idm44771280576144" id="idm44771280576144-marker" data-type="noteref">3</a></sup> <a href="https://dbi.r-dbi.org/reference/dbReadTable.html" class="orm:hideurl"><code>dbReadTable()</code></a>은 테이블의 내용을 가져옵니다.

```
dbListTables(con)
#> [1] "diamonds" "mpg"

con |>
  dbReadTable("diamonds") |>
  as_tibble()
#> # A tibble: 53,940 × 10
#>    carat cut       color clarity depth table price     x     y     z
#>    <dbl> <fct>     <fct> <fct>   <dbl> <dbl> <int> <dbl> <dbl> <dbl>
#>  1 0.23  Ideal     E     SI2      61.5    55   326  3.95  3.98  2.43
#>  2 0.21  Premium   E     SI1      59.8    61   326  3.89  3.84  2.31
#>  3 0.23  Good      E     VS1      56.9    65   327  4.05  4.07  2.31
#>  4 0.290 Premium   I     VS2      62.4    58   334  4.2   4.23  2.63
#>  5 0.31  Good      J     SI2      63.3    58   335  4.34  4.35  2.75
#>  6 0.24  Very Good J     VVS2     62.8    57   336  3.94  3.96  2.48
#> # … with 53,934 more rows
```

<a href="https://dbi.r-dbi.org/reference/dbReadTable.html" class="orm:hideurl"><code>dbReadTable()</code></a>은 `data.frame`을 반환하므로 <a href="https://tibble.tidyverse.org/reference/as_tibble.html" class="orm:hideurl"><code>as_tibble()</code></a>을 사용하여 예쁘게 출력되도록 티블로 변환합니다.

이미 SQL을 알고 있다면, <a href="https://dbi.r-dbi.org/reference/dbGetQuery.html" class="orm:hideurl"><code>dbGetQuery()</code></a>를 사용하여 데이터베이스에서 쿼리를 실행한 결과를 얻을 수 있습니다.

```
sql <- "
  SELECT carat, cut, clarity, color, price
  FROM diamonds
  WHERE price > 15000
"
as_tibble(dbGetQuery(con, sql))
#> # A tibble: 1,655 × 5
#>    carat cut       clarity color price
#>    <dbl> <fct>     <fct>   <fct> <int>
#>  1  1.54 Premium   VS2     E     15002
#>  2  1.19 Ideal     VVS1    F     15005
#>  3  2.1  Premium   SI1     I     15007
#>  4  1.69 Ideal     SI1     D     15011
#>  5  1.5  Very Good VVS2    G     15013
#>  6  1.73 Very Good VS1     G     15014
#> # … with 1,649 more rows
```

이전에 SQL을 본 적이 없더라도 걱정하지 마세요! 잠시 후에 더 자세히 배우게 될 것입니다. 하지만 쿼리를 주의 깊게 읽어보면 `diamonds` 데이터 세트에서 5개의 열을 선택하고 `price`가 15,000보다 큰 모든 행을 선택한다는 것을 추측할 수 있을 것입니다.

# dbplyr 기초

데이터베이스에 연결하고 일부 데이터를 로드했으므로, 이제 dbplyr에 대해 배울 수 있습니다. dbplyr는 dplyr *백엔드(backend)*로, 여러분이 계속 dplyr 코드를 작성하면 백엔드가 이를 다르게 실행한다는 의미입니다. 여기서 dbplyr는 코드를 SQL로 번역합니다. 다른 백엔드로는 [data.table](https://oreil.ly/k3EaP)로 번역하는 [dtplyr](https://oreil.ly/9Dq5p)와 다중 코어에서 코드를 실행하는 [multidplyr](https://oreil.ly/gmDpk)가 있습니다.

dbplyr를 사용하려면 먼저 <a href="https://dplyr.tidyverse.org/reference/tbl.html" class="orm:hideurl"><code>tbl()</code></a>을 사용하여 데이터베이스 테이블을 나타내는 객체를 만들어야 합니다.

```
diamonds_db <- tbl(con, "diamonds")
diamonds_db
#> # Source:   table<diamonds> [?? x 10]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#>    carat cut       color clarity depth table price     x     y     z
#>    <dbl> <fct>     <fct> <fct>   <dbl> <dbl> <int> <dbl> <dbl> <dbl>
#>  1 0.23  Ideal     E     SI2      61.5    55   326  3.95  3.98  2.43
#>  2 0.21  Premium   E     SI1      59.8    61   326  3.89  3.84  2.31
#>  3 0.23  Good      E     VS1      56.9    65   327  4.05  4.07  2.31
#>  4 0.290 Premium   I     VS2      62.4    58   334  4.2   4.23  2.63
#>  5 0.31  Good      J     SI2      63.3    58   335  4.34  4.35  2.75
#>  6 0.24  Very Good J     VVS2     62.8    57   336  3.94  3.96  2.48
#> # … with more rows
```

###### 참고

데이터베이스와 상호작용하는 다른 두 가지 일반적인 방법이 있습니다. 첫째, 많은 기업용 데이터베이스는 매우 커서 모든 테이블을 체계적으로 유지하기 위한 어떤 계층 구조가 필요합니다. 이 경우 관심 있는 테이블을 선택하기 위해 스키마(schema), 또는 카탈로그(catalog)와 스키마를 제공해야 할 수도 있습니다.

```
diamonds_db <- tbl(con, in_schema("sales", "diamonds"))
diamonds_db <- tbl(
  con,
  in_catalog("north_america", "sales", "diamonds")
)
```

다른 경우에는 여러분 자신의 SQL 쿼리를 시작점으로 사용하고 싶을 수도 있습니다.

```
diamonds_db <- tbl(con, sql("SELECT * FROM diamonds"))
```

이 객체는 _지연된(lazy)_ 객체입니다. 이 객체에 dplyr 동사를 사용하면 dplyr는 아무 작업도 수행하지 않습니다. 단지 여러분이 수행하려는 작업의 순서를 기록할 뿐이며 필요할 때만 작업을 수행합니다. 예를 들어 다음 파이프라인을 보겠습니다.

```
big_diamonds_db <- diamonds_db |>
  filter(price > 15000) |>
  select(carat:clarity, price)

big_diamonds_db
#> # Source:   SQL [?? x 5]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#>    carat cut       color clarity price
#>    <dbl> <fct>     <fct> <fct>   <int>
#>  1  1.54 Premium   E     VS2     15002
#>  2  1.19 Ideal     F     VVS1    15005
#>  3  2.1  Premium   I     SI1     15007
#>  4  1.69 Ideal     D     SI1     15011
#>  5  1.5  Very Good G     VVS2    15013
#>  6  1.73 Very Good G     VS1     15014
#> # … with more rows
```

이 객체가 데이터베이스 쿼리를 나타낸다는 것을 알 수 있는데, 상단에 DBMS 이름이 출력되고, 열의 수는 알려주지만 행의 수는 일반적으로 알지 못하기 때문입니다. 전체 행의 수를 찾으려면 대개 전체 쿼리를 실행해야 하는데, 우리는 이를 피하려고 하기 때문입니다.

<a href="https://dplyr.tidyverse.org/reference/explain.html" class="orm:hideurl"><code>show_query()</code></a> 함수를 사용하면 dplyr가 생성한 SQL 코드를 볼 수 있습니다. dplyr를 안다면 이것은 SQL을 배우는 훌륭한 방법입니다! dplyr 코드를 작성하고 dbplyr가 그것을 SQL로 번역하게 한 다음, 두 언어가 어떻게 매치되는지 파악해 보세요.

```
big_diamonds_db |> show_query()
#> <SQL>
#> SELECT carat, cut, color, clarity, price
#> FROM diamonds
#> WHERE (price > 15000.0)
```

모든 데이터를 다시 R로 가져오려면 <a href="https://dplyr.tidyverse.org/reference/compute.html" class="orm:hideurl"><code>collect()</code></a>를 호출합니다. 이면에서 이 함수는 SQL을 생성하고, <a href="https://dbi.r-dbi.org/reference/dbGetQuery.html" class="orm:hideurl"><code>dbGetQuery()</code></a>를 호출하여 데이터를 가져온 다음 그 결과를 티블로 변환합니다.

```
big_diamonds <- big_diamonds_db |> collect()
big_diamonds
#> # A tibble: 1,655 × 5
#>    carat cut       color clarity price
#>    <dbl> <fct>     <fct> <fct>   <int>
#>  1  1.54 Premium   E     VS2     15002
#>  2  1.19 Ideal     F     VVS1    15005
#>  3  2.1  Premium   I     SI1     15007
#>  4  1.69 Ideal     D     SI1     15011
#>  5  1.5  Very Good G     VVS2    15013
#>  6  1.73 Very Good G     VS1     15014
#> # … with 1,649 more rows
```

일반적으로 dbplyr를 사용하여 다음에 설명할 번역을 통해 기본적인 필터링과 집계를 수행하며 데이터베이스에서 원하는 데이터를 선택합니다. 그런 다음 R만의 고유한 함수를 사용하여 데이터를 분석할 준비가 되면, <a href="https://dplyr.tidyverse.org/reference/compute.html" class="orm:hideurl"><code>collect()</code></a>를 사용하여 데이터를 가져와 메모리 내 티블을 얻고 순수 R 코드로 작업을 계속합니다.

# SQL

이 장의 나머지 부분에서는 dbplyr의 렌즈를 통해 SQL을 조금 배우게 될 것입니다. SQL에 대한 다소 비전통적인 소개이지만, 기본 사항을 빠르게 파악할 수 있기를 바랍니다. 다행히도 여러분이 dplyr를 이해하고 있다면, 많은 개념이 동일하기 때문에 SQL을 빠르게 습득할 수 있는 좋은 위치에 있는 것입니다.

nycflights13 패키지의 오랜 친구인 `flights`와 `planes`라는 두 데이터 세트를 사용하여 dplyr와 SQL의 관계를 탐구해 보겠습니다. dbplyr에는 nycflights13의 테이블을 우리의 데이터베이스로 복사하는 함수가 포함되어 있기 때문에 이 데이터 세트들을 학습용 데이터베이스에 쉽게 넣을 수 있습니다.

```
dbplyr::copy_nycflights13(con)
#> Creating table: airlines
#> Creating table: airports
#> Creating table: flights
#> Creating table: planes
#> Creating table: weather

flights <- tbl(con, "flights")
planes <- tbl(con, "planes")
```

## SQL 기초

SQL의 최상위 수준 구성 요소를 *문(statement)*이라고 합니다. 일반적인 문으로는 새 테이블을 정의하는 `CREATE`, 데이터를 추가하는 `INSERT`, 데이터를 검색하는 `SELECT`가 있습니다. 여러분은 데이터 과학자로서 거의 이것만 사용하게 될 것이므로 `SELECT` 문(또는 *쿼리*라고도 함)에 집중할 것입니다.

쿼리는 *절(clause)*들로 구성됩니다. 다섯 가지 중요한 절이 있습니다. `SELECT`, `FROM`, `WHERE`, `ORDER BY`, `GROUP BY`. 모든 쿼리에는 `SELECT`<sup><a href="ch21.html#idm44771280084352" id="idm44771280084352-marker" data-type="noteref">4</a></sup>와 `FROM`<sup><a href="ch21.html#idm44771280052864" id="idm44771280052864-marker" data-type="noteref">5</a></sup> 절이 있어야 하며 단순한 쿼리는 지정된 테이블의 모든 열을 선택하는 `SELECT * FROM table`입니다. 이것이 조작되지 않은 테이블에 대해 dbplyr가 생성하는 것입니다.

```
flights |> show_query()
#> <SQL>
#> SELECT *
#> FROM flights

planes |> show_query()
#> <SQL>
#> SELECT *
#> FROM planes
```

`WHERE`와 `ORDER BY`는 포함될 행과 그 정렬 방식을 제어합니다.

```
flights |>
  filter(dest == "IAH") |>
  arrange(dep_delay) |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM flights
#> WHERE (dest = 'IAH')
#> ORDER BY dep_delay
```

`GROUP BY`는 쿼리를 요약으로 변환하여 집계가 발생하도록 합니다.

```
flights |>
  group_by(dest) |>
  summarize(dep_delay = mean(dep_delay, na.rm = TRUE)) |>
  show_query()
#> <SQL>
#> SELECT dest, AVG(dep_delay) AS dep_delay
#> FROM flights
#> GROUP BY dest
```

dplyr 동사와 `SELECT` 절 사이에는 두 가지 중요한 차이점이 있습니다.

- SQL에서는 대소문자를 구별하지 않습니다. `select`, `SELECT` 또는 심지어 `SeLeCt`라고 써도 됩니다. 이 책에서는 테이블이나 변수 이름과 구분하기 위해 SQL 키워드를 대문자로 쓰는 일반적인 규칙을 따를 것입니다.
- SQL에서는 순서가 중요합니다. 항상 `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY` 순서로 절을 작성해야 합니다. 헷갈리게도 이 순서는 절이 실제로 평가되는 순서와 일치하지 않는데, 실제 평가는 `FROM`, `WHERE`, `GROUP BY`, `SELECT`, `ORDER BY` 순서로 이루어집니다.

다음 섹션에서는 각 절에 대해 더 자세히 탐구합니다.

###### 참고

SQL이 표준이긴 하지만 매우 복잡하고 표준을 정확히 따르는 데이터베이스는 없다는 점에 유의하세요. 이 책에서 우리가 초점을 맞출 주요 구성 요소는 DBMS 간에 비슷하지만 많은 사소한 변형이 있습니다. 다행히 dbplyr는 이 문제를 처리하도록 설계되었으며 다양한 데이터베이스에 대해 서로 다른 번역을 생성합니다. 완벽하지는 않지만 지속적으로 개선되고 있으며, 문제에 부딪히면 저희가 더 잘할 수 있도록 [GitHub](https://oreil.ly/xgmg8)에 이슈를 남겨주세요.

## SELECT

`SELECT` 절은 쿼리의 핵심(workhorse)이며 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>, <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename()</code></a>, <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a> 그리고 다음 섹션에서 배울 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 동일한 작업을 수행합니다.

<a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename()</code></a>, <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a>는 단지 열이 나타나는 위치와 열의 이름에 영향을 미치기 때문에 `SELECT`로 아주 직접적으로 번역됩니다.

```
planes |>
  select(tailnum, type, manufacturer, model, year) |>
  show_query()
#> <SQL>
#> SELECT tailnum, "type", manufacturer, model, "year"
#> FROM planes

planes |>
  select(tailnum, type, manufacturer, model, year) |>
  rename(year_built = year) |>
  show_query()
#> <SQL>
#> SELECT tailnum, "type", manufacturer, model, "year" AS year_built
#> FROM planes

planes |>
  select(tailnum, type, manufacturer, model, year) |>
  relocate(manufacturer, model, .before = type) |>
  show_query()
#> <SQL>
#> SELECT tailnum, manufacturer, model, "type", "year"
#> FROM planes
```

이 예제는 또한 SQL이 이름 바꾸기를 어떻게 수행하는지 보여줍니다. SQL 용어로 이름 바꾸기를 *별칭 지정(aliasing)*이라고 하며 `AS`를 사용하여 수행됩니다. <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와는 달리 이전 이름이 왼쪽에, 새 이름이 오른쪽에 있다는 점에 유의하세요.

###### 참고

이전 예제에서 `"year"`와 `"type"`이 큰따옴표로 묶여 있는 것에 유의하세요. 이 단어들은 duckdb에서 *예약어(reserved words)*이기 때문에 열/테이블 이름과 SQL 연산자 간의 잠재적 혼동을 피하기 위해 dbplyr가 이들을 따옴표로 묶습니다.
다른 데이터베이스로 작업할 때는 모든 변수 이름이 따옴표로 묶인 것을 보게 될 것입니다. duckdb와 같이 소수의 클라이언트 패키지만 어떤 단어가 예약어인지 모두 알고 있으므로, 안전을 기하기 위해 모든 것을 따옴표로 묶기 때문입니다.

```
SELECT "tailnum", "type", "manufacturer", "model", "year" FROM "planes"
```

다른 일부 데이터베이스 시스템은 따옴표 대신 백틱(backtick)을 사용합니다.

```
SELECT `tailnum`, `type`, `manufacturer`, `model`, `year` FROM `planes`
```

<a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에 대한 번역도 비슷하게 간단합니다. 각 변수는 `SELECT` 내에서 새로운 표현식이 됩니다.

```
flights |>
  mutate(
    speed = distance / (air_time / 60)
  ) |>
  show_query()
#> <SQL>
#> SELECT *, distance / (air_time / 60.0) AS speed
#> FROM flights
```

<a href="#sec-sql-expressions" data-type="xref">“함수 번역”</a>에서 개별 구성 요소(`/`)의 번역으로 다시 돌아올 것입니다.

## FROM

`FROM` 절은 데이터 소스를 정의합니다. 단일 테이블만 사용하고 있기 때문에 한동안은 다소 흥미롭지 않을 것입니다. 조인(join) 함수에 도달하면 더 복잡한 예제를 보게 될 것입니다.

## GROUP BY

<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>는 `GROUP BY`<sup><a href="ch21.html#idm44771279537344" id="idm44771279537344-marker" data-type="noteref">6</a></sup> 절로 번역되고, <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>는 `SELECT` 절로 번역됩니다.

```
diamonds_db |>
  group_by(cut) |>
  summarize(
    n = n(),
    avg_price = mean(price, na.rm = TRUE)
  ) |>
  show_query()
#> <SQL>
#> SELECT cut, COUNT(*) AS n, AVG(price) AS avg_price
#> FROM diamonds
#> GROUP BY cut
```

<a href="#sec-sql-expressions" data-type="xref">“함수 번역”</a>에서 <a href="https://dplyr.tidyverse.org/reference/context.html" class="orm:hideurl"><code>n()</code></a>과 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>을 번역할 때 무슨 일이 일어나는지 다시 살펴보겠습니다.

## WHERE

<a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>는 `WHERE` 절로 번역됩니다.

```
flights |>
  filter(dest == "IAH" | dest == "HOU") |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM flights
#> WHERE (dest = 'IAH' OR dest = 'HOU')

flights |>
  filter(arr_delay > 0 & arr_delay < 20) |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM flights
#> WHERE (arr_delay > 0.0 AND arr_delay < 20.0)
```

여기서 주의해야 할 몇 가지 중요한 세부 사항이 있습니다.

- `|`는 `OR`가 되고, `&`는 `AND`가 됩니다.
- SQL은 비교에 `==`가 아니라 `=`를 사용합니다. SQL에는 할당(assignment)이 없으므로 혼동할 여지가 없습니다.
- SQL은 문자열에 `""`가 아니라 `''`만 사용합니다. SQL에서 `""`는 R의 백틱(` `` `)처럼 변수를 식별하는 데 사용됩니다.

또 다른 유용한 SQL 연산자는 `IN`으로, 이는 R의 `%in%`과 유사합니다.

```
flights |>
  filter(dest %in% c("IAH", "HOU")) |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM flights
#> WHERE (dest IN ('IAH', 'HOU'))
```

SQL은 `NA` 대신 `NULL`을 사용합니다. `NULL`은 `NA`와 비슷하게 동작합니다. 주요 차이점은 비교 및 산술 연산에서는 "전염성(infectious)"을 가지지만, 요약할 때는 조용히 무시된다는 점입니다. dbplyr는 여러분이 처음 이 동작에 부딪힐 때 이에 대해 상기시켜 줄 것입니다.

```
flights |>
  group_by(dest) |>
  summarize(delay = mean(arr_delay))
#> Warning: Missing values are always removed in SQL aggregation functions.
#> Use `na.rm = TRUE` to silence this warning
#> This warning is displayed once every 8 hours.
#> # Source:   SQL [?? x 2]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#>    dest   delay
#>    <chr>  <dbl>
#>  1 ATL   11.3
#>  2 ORD    5.88
#>  3 RDU   10.1
#>  4 IAD   13.9
#>  5 DTW    5.43
#>  6 LAX    0.547
#> # … with more rows
```

`NULL`이 어떻게 작동하는지에 대해 더 자세히 알고 싶다면, Markus Winand의 [“The Three-Valued Logic of SQL”](https://oreil.ly/PTwQz)을 즐겁게 읽으실 수 있을 것입니다.

일반적으로 R에서 `NA`에 사용하는 함수를 사용하여 `NULL`로 작업할 수 있습니다.

```
flights |>
  filter(!is.na(dep_delay)) |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM flights
#> WHERE (NOT((dep_delay IS NULL)))
```

이 SQL 쿼리는 dbplyr의 단점 중 하나를 보여줍니다. SQL은 올바르지만 손으로 작성하는 것만큼 간단하지는 않습니다. 이 경우 괄호를 없애고 읽기 쉬운 특수 연산자를 사용할 수 있습니다.

```
WHERE "dep_delay" IS NOT NULL
```

`summarize`를 사용하여 만든 변수를 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>로 거르면 dbplyr가 `WHERE` 절 대신 `HAVING` 절을 생성한다는 점에 유의하세요. 이것은 SQL의 특성 중 하나입니다. `WHERE`는 `SELECT`와 `GROUP BY` 전에 평가되므로, SQL은 나중에 평가되는 다른 절이 필요합니다.

```
diamonds_db |>
  group_by(cut) |>
  summarize(n = n()) |>
  filter(n > 100) |>
  show_query()
#> <SQL>
#> SELECT cut, COUNT(*) AS n
#> FROM diamonds
#> GROUP BY cut
#> HAVING (COUNT(*) > 100.0)
```

## ORDER BY

행을 정렬하는 것은 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>에서 `ORDER BY` 절로 직관적으로 번역됩니다.

```
flights |>
  arrange(year, month, day, desc(dep_delay)) |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM flights
#> ORDER BY "year", "month", "day", dep_delay DESC
```

<a href="https://dplyr.tidyverse.org/reference/desc.html" class="orm:hideurl"><code>desc()</code></a>가 `DESC`로 번역되는 방식에 주목하세요. 이는 이름이 SQL에서 직접 영감을 받은 많은 dplyr 함수 중 하나입니다.

## 서브쿼리

때로는 dplyr 파이프라인을 단일 `SELECT` 문으로 번역하는 것이 불가능하여 서브쿼리(subquery)를 사용해야 할 수도 있습니다. *서브쿼리*는 일반적인 테이블 대신 `FROM` 절에서 데이터 소스로 사용되는 쿼리일 뿐입니다.

dbplyr는 일반적으로 SQL의 한계를 우회하기 위해 서브쿼리를 사용합니다. 예를 들어 `SELECT` 절의 표현식은 방금 생성된 열을 참조할 수 없습니다. 즉, 다음 (바보 같은) dplyr 파이프라인은 두 단계로 발생해야 합니다. 첫 번째(내부) 쿼리는 `year1`을 계산하고, 그런 다음 두 번째(외부) 쿼리가 `year2`를 계산할 수 있습니다.

```
flights |>
  mutate(
    year1 = year + 1,
    year2 = year1 + 1
  ) |>
  show_query()
#> <SQL>
#> SELECT *, year1 + 1.0 AS year2
#> FROM (
#>   SELECT *, "year" + 1.0 AS year1
#>   FROM flights
#> ) q01
```

방금 만든 변수를 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>로 거르려고 시도한 경우에도 이를 보게 될 것입니다. `WHERE`는 `SELECT` 뒤에 작성되지만 그보다 먼저 평가되므로, 이 (바보 같은) 예제에서는 서브쿼리가 필요하다는 것을 기억하세요.

```
flights |>
  mutate(year1 = year + 1) |>
  filter(year1 == 2014) |>
  show_query()
#> <SQL>
#> SELECT *
#> FROM (
#>   SELECT *, "year" + 1.0 AS year1
#>   FROM flights
#> ) q01
#> WHERE (year1 = 2014.0)
```

dbplyr는 아직 해당 번역을 최적화하는 방법을 모르기 때문에 필요하지 않은 서브쿼리를 생성할 때도 있습니다. dbplyr가 시간이 지남에 따라 개선되면서 이러한 경우는 드물어지겠지만 아마 완전히 사라지지는 않을 것입니다.

## 조인

dplyr의 조인(join)에 익숙하다면 SQL 조인도 비슷합니다. 다음은 간단한 예제입니다.

```
flights |>
  left_join(planes |> rename(year_built = year), by = "tailnum") |>
  show_query()
#> <SQL>
#> SELECT
#>   flights.*,
#>   planes."year" AS year_built,
#>   "type",
#>   manufacturer,
#>   model,
#>   engines,
#>   seats,
#>   speed,
#>   engine
#> FROM flights
#> LEFT JOIN planes
#>   ON (flights.tailnum = planes.tailnum)
```

여기서 주목해야 할 주요 사항은 구문입니다. SQL 조인은 `FROM` 절의 하위 절(subclause)을 사용하여 추가 테이블을 가져오고, `ON`을 사용하여 테이블이 어떻게 관련되어 있는지 정의합니다.

이 함수들에 대한 dplyr의 이름은 SQL과 매우 밀접하게 연결되어 있어 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>inner_join()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>right_join()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>full_join()</code></a>에 해당하는 SQL을 쉽게 추측할 수 있습니다.

```
SELECT flights.*, "type", manufacturer, model, engines, seats, speed
FROM flights
INNER JOIN planes
  ON (flights.tailnum = planes.tailnum)

SELECT flights.*, "type", manufacturer, model, engines, seats, speed
FROM flights
RIGHT JOIN planes
  ON (flights.tailnum = planes.tailnum)

SELECT flights.*, "type", manufacturer, model, engines, seats, speed
FROM flights
FULL JOIN planes
  ON (flights.tailnum = planes.tailnum)
```

데이터베이스의 데이터로 작업할 때는 많은 조인이 필요할 가능성이 큽니다. 데이터베이스 테이블은 각 "사실(fact)"이 단일 장소에 저장되는 고도로 정규화된(normalized) 형태로 저장되는 경우가 많기 때문입니다. 따라서 분석을 위한 완전한 데이터 세트를 유지하려면 기본 키(primary key)와 외래 키(foreign key)로 연결된 복잡한 테이블 네트워크를 탐색해야 합니다. 이러한 상황에 부딪힌다면 Tobias Schieferdecker, Kirill Müller, Darko Bergant가 만든 [dm 패키지](https://oreil.ly/tVS8h)가 구원자가 될 것입니다. DBA가 흔히 제공하는 제약 조건을 사용하여 테이블 간의 연결을 자동으로 결정하고, 상황을 파악할 수 있도록 연결을 시각화하며, 한 테이블을 다른 테이블과 연결하는 데 필요한 조인을 생성할 수 있습니다.

## 다른 동사들

dbplyr는 <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>, `slice_*()`, <a href="https://generics.r-lib.org/reference/setops.html" class="orm:hideurl"><code>intersect()</code></a>와 같은 다른 동사뿐만 아니라 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>와 같이 점점 더 다양해지는 tidyr 함수들도 번역합니다. 현재 사용 가능한 전체 집합을 보는 쉬운 방법은 [dbplyr 웹사이트](https://oreil.ly/A8OGW)를 방문하는 것입니다.

## 연습문제

1.  <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>는 무엇으로 번역되나요? <a href="https://rdrr.io/r/utils/head.html" class="orm:hideurl"><code>head()</code></a>는 어떤가요?

2.  다음 SQL 쿼리들이 각각 무엇을 하는지 설명하고 dbplyr를 사용하여 그것들을 다시 만들어 보세요.

    ```
    SELECT *
    FROM flights
    WHERE dep_delay < arr_delay

    SELECT *, distance / (airtime / 60) AS speed
    FROM flights
    ```

# 함수 번역

지금까지는 dplyr 동사가 쿼리의 여러 절로 어떻게 번역되는지에 대한 큰 그림에 초점을 맞추었습니다. 이제 좀 더 자세히 들여다보면서 개별 열에 작동하는 R 함수의 번역에 대해 이야기해 보겠습니다. 예를 들어 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>에서 `mean(x)`를 사용할 때 무슨 일이 일어날까요?

무슨 일이 일어나고 있는지 확인하는 데 도움이 되도록, <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a> 또는 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>를 실행하고 생성된 SQL을 표시하는 두 개의 작은 헬퍼 함수를 사용하겠습니다. 이렇게 하면 몇 가지 변형을 탐구하고 요약과 변환이 어떻게 다를 수 있는지 보는 것이 조금 더 쉬워질 것입니다.

```
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

몇 가지 요약부터 살펴보겠습니다! 다음 코드를 살펴보면 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>과 같은 일부 요약 함수는 비교적 간단한 번역을 가지고 있는 반면 <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>과 같은 다른 함수는 훨씬 더 복잡하다는 것을 알 수 있습니다. 데이터베이스에서는 덜 일반적이지만 통계에서는 일반적인 연산일수록 대개 복잡도가 더 높습니다.

```
flights |>
  group_by(year, month, day) |>
  summarize_query(
    mean = mean(arr_delay, na.rm = TRUE),
    median = median(arr_delay, na.rm = TRUE)
  )
#> `summarise()` has grouped output by "year" and "month". You can override
#> using the `.groups` argument.
#> <SQL>
#> SELECT
#>   "year",
#>   "month",
#>   "day",
#>   AVG(arr_delay) AS mean,
#>   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY arr_delay) AS median
#> FROM flights
#> GROUP BY "year", "month", "day"
```

요약 함수를 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 내에서 사용할 때는 소위 _윈도우(window)_ 함수로 변해야 하기 때문에 번역이 더 복잡해집니다. SQL에서는 일반적인 집계 함수 뒤에 `OVER`를 추가하여 윈도우 함수로 변환합니다.

```
flights |>
  group_by(year, month, day) |>
  mutate_query(
    mean = mean(arr_delay, na.rm = TRUE),
  )
#> <SQL>
#> SELECT
#>   "year",
#>   "month",
#>   "day",
#>   AVG(arr_delay) OVER (PARTITION BY "year", "month", "day") AS mean
#> FROM flights
```

SQL에서 `GROUP BY` 절은 요약 목적으로만 사용되므로, 여기서는 그룹화가 `PARTITION BY` 인자에서 `OVER`로 이동한 것을 볼 수 있습니다.

윈도우 함수에는 앞이나 뒤를 보는 모든 함수가 포함됩니다. 각각 "이전" 또는 "다음" 값을 보는 <a href="https://dplyr.tidyverse.org/reference/lead-lag.html" class="orm:hideurl"><code>lead()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/lead-lag.html" class="orm:hideurl"><code>lag()</code></a> 같은 함수들입니다.

```
flights |>
  group_by(dest) |>
  arrange(time_hour) |>
  mutate_query(
    lead = lead(arr_delay),
    lag = lag(arr_delay)
  )
#> <SQL>
#> SELECT
#>   dest,
#>   LEAD(arr_delay, 1, NULL) OVER (PARTITION BY dest ORDER BY time_hour) AS lead,
#>   LAG(arr_delay, 1, NULL) OVER (PARTITION BY dest ORDER BY time_hour) AS lag
#> FROM flights
#> ORDER BY time_hour
```

여기서 데이터를 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>하는 것이 중요한데, SQL 테이블에는 본질적인 순서가 없기 때문입니다. 사실 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>를 사용하지 않으면 매번 다른 순서로 행을 반환받을 수 있습니다! 윈도우 함수의 경우 정렬 정보가 반복된다는 점에 유의하세요. 메인 쿼리의 `ORDER BY` 절이 윈도우 함수에 자동으로 적용되지는 않습니다.

또 다른 중요한 SQL 함수는 `CASE WHEN`입니다. 이것은 자신에게 직접적인 영감을 준 dplyr 함수인 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a>의 번역으로 사용됩니다. 다음은 몇 가지 간단한 예제입니다.

```
flights |>
  mutate_query(
    description = if_else(arr_delay > 0, "delayed", "on-time")
  )
#> <SQL>
#> SELECT CASE WHEN
#>   (arr_delay > 0.0) THEN 'delayed'
#>   WHEN NOT (arr_delay > 0.0) THEN 'on-time' END AS description
#> FROM flights

flights |>
  mutate_query(
    description = case_when(
      arr_delay < -5 ~ "early",
      arr_delay < 5 ~ "on-time",
      arr_delay >= 5 ~ "late"
    )
  )
#> <SQL>
#> SELECT CASE
#> WHEN (arr_delay < -5.0) THEN 'early'
#> WHEN (arr_delay < 5.0) THEN 'on-time'
#> WHEN (arr_delay >= 5.0) THEN 'late'
#> END AS description
#> FROM flights
```

`CASE WHEN`은 R에서 SQL로 직접 번역되지 않는 다른 일부 함수들에도 사용됩니다. 이에 대한 좋은 예가 <a href="https://rdrr.io/r/base/cut.html" class="orm:hideurl"><code>cut()</code></a>입니다.

```
flights |>
  mutate_query(
    description = cut(
      arr_delay,
      breaks = c(-Inf, -5, 5, Inf),
      labels = c("early", "on-time", "late")
    )
  )
#> <SQL>
#> SELECT CASE
#> WHEN (arr_delay <= -5.0) THEN 'early'
#> WHEN (arr_delay <= 5.0) THEN 'on-time'
#> WHEN (arr_delay > 5.0) THEN 'late'
#> END AS description
#> FROM flights
```

dbplyr는 일반적인 문자열 및 날짜-시간 조작 함수들도 번역하며, 이에 대해서는 <a href="https://dbplyr.tidyverse.org/articles/translation-function.html" class="orm:hideurl"><code>vignette("translation-function", package = "dbplyr")</code></a>에서 배울 수 있습니다. dbplyr의 번역이 완벽하지는 않고 아직 번역되지 않은 R 함수도 많이 있지만, dbplyr는 대부분의 경우 여러분이 사용할 함수들을 놀라울 정도로 잘 다룹니다.

# 요약

이 장에서는 데이터베이스에서 데이터에 접근하는 방법을 배웠습니다. 익숙한 dplyr 코드를 작성하면 이를 자동으로 SQL로 번역해 주는 dplyr "백엔드"인 dbplyr에 초점을 맞추었습니다. 우리는 그 번역을 사용하여 여러분에게 SQL을 약간 가르쳤습니다. SQL은 데이터 작업에 널리 사용되는 언어이며 SQL을 조금 아는 것이 R을 사용하지 않는 다른 데이터 전문가들과 소통하는 것을 더 쉽게 만들어 줄 것이기 때문에 SQL을 조금 배우는 것은 중요합니다. 이 장을 마치고 SQL에 대해 더 배우고 싶다면 두 가지를 추천합니다.

- Renée M. P. Teate의 [_SQL for Data Scientists_](https://oreil.ly/QfAat)는 데이터 과학자의 요구에 특별히 맞춰 설계된 SQL 입문서이며 실제 조직에서 마주칠 가능성이 높은 종류의 고도로 상호 연결된 데이터의 예를 포함하고 있습니다.
- Anthony DeBarros의 [_Practical SQL_](https://oreil.ly/-0Usp)은 데이터 저널리스트(설득력 있는 이야기를 전달하는 데 특화된 데이터 과학자)의 관점에서 작성되었으며 데이터를 데이터베이스에 넣고 자체 DBMS를 실행하는 것에 대해 더 자세히 다룹니다.

다음 장에서는 대규모 데이터를 다루기 위한 또 다른 dplyr 백엔드인 arrow에 대해 배울 것입니다. arrow 패키지는 디스크의 대용량 파일 작업용으로 설계되었으며 데이터베이스를 자연스럽게 보완해 줍니다.

<sup>[1](ch21.html#idm44771280827536-marker)</sup> SQL은 "에스-큐-엘" 또는 "시퀄"이라고 발음합니다.

<sup>[2](ch21.html#idm44771280761328-marker)</sup> 일반적으로 이것이 클라이언트 패키지에서 사용할 유일한 함수이므로, <a href="https://rdrr.io/r/base/library.html" class="orm:hideurl"><code>library()</code></a>로 전체 패키지를 로드하는 대신 `::`를 사용하여 해당 함수 하나만 가져오는 것을 권장합니다.

<sup>[3](ch21.html#idm44771280576144-marker)</sup> 적어도 여러분이 볼 권한이 있는 모든 테이블을 표시합니다.

<sup>[4](ch21.html#idm44771280084352-marker)</sup> 혼란스럽게도 문맥에 따라 `SELECT`는 문(statement)이기도 하고 절(clause)이기도 합니다. 이러한 혼동을 피하기 위해 일반적으로 `SELECT` 문 대신 `SELECT` 쿼리라는 용어를 사용할 것입니다.

<sup>[5](ch21.html#idm44771280052864-marker)</sup> 기술적으로는 `SELECT 1+1`과 같은 쿼리를 작성하여 기본적인 계산을 수행할 수 있으므로 `SELECT`만 필요합니다. 하지만 (여러분이 항상 그렇듯!) 데이터 작업을 원한다면 `FROM` 절도 필요할 것입니다.

<sup>[6](ch21.html#idm44771279537344-marker)</sup> 이것은 우연이 아닙니다. 이 dplyr 함수 이름은 SQL 절에서 영감을 받은 것입니다.
