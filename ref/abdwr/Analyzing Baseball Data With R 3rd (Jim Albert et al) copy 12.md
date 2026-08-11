### 12 대용량 데이터 다루기

12.1 서론

11장에서는 R의 RMariaDB 패키지를 사용하여 MySQL 데이터베이스를 설정하고 데이터를 채웠습니다. 해당 데이터베이스에는 Retrosheet에서 수집한 게임 로그와 플레이별 데이터가 포함되어 있었습니다. 이 장에서는 이 작업을 확장하여 Baseball Savant에서 다운로드한 피치별 Statcast 데이터의 데이터베이스를 구축합니다. 그 과정에서 MySQL 외에도 4년간의 플레이(2020-2023)에 걸친 데이터 세트를 저장하고 표현하는 여러 가지 다른 방법을 살펴봅니다. 마지막으로 이러한 다양한 접근 방식의 장단점을 비교 및 대조하고 몇 가지 테스트 드라이브를 위해 이러한 데이터베이스를 활용합니다.

부록 C.10에서는 baseballr 패키지의 statcast_search() 함수를 사용하여 Baseball Savant에서 Statcast 데이터를 다운로드하는 방법을 보여줍니다. 메이저 리그 베이스볼에서 던진 각 투구에 대한 데이터 행이 있기 때문에 이러한 데이터는 작업 흐름을 복잡하게 만들 수 있을 정도로 빠르게 커질 수 있습니다. 이 책에서 사용한 데이터는 대부분 크기가 작았습니다. 즉, 몇 킬로바이트나 메가바이트에 불과하며, 이는 대부분의 개인용 컴퓨터에 있는 물리적 메모리(즉, 랜덤 액세스 메모리) 양보다 훨씬 적습니다. 모든 플레이를 다루는 Retrosheet 데이터조차도 전체 시즌에 대해 100MB 미만입니다. 그러나 아래에서 볼 수 있듯이 Statcast 데이터는 규모가 한 자릿수 더 크며 시즌당 수백 메가바이트를 차지하고 여러 시즌에 걸쳐 수 기가바이트를 차지합니다. 이 정도 규모의 데이터로 작업하면 주의하지 않을 경우 개인용 컴퓨터가 방해를 받거나 과부하가 걸릴 수 있습니다. 따라서 이 장에서는 다년간의 Statcast를 수집한 다음 해당 데이터를 효율적으로 저장하기 위한 새로운 도구를 소개합니다. 다행스럽게도 우리가 어떤 데이터 저장 형식을 선택하든 dplyr과 dbplyr의 마법은 데이터를 분석하는 과정을 동일하게 만듭니다.

DOI: 10.1201/9781032668239-12 280

### 12 대용량 데이터 다루기

- 12.1 서론

11장에서는 R의 RMariaDB 패키지를 사용하여 MySQL 데이터베이스를 설정하고 데이터를 채웠습니다. 해당 데이터베이스에는 Retrosheet에서 수집한 게임 로그와 플레이별 데이터가 포함되어 있었습니다. 이 장에서는 이 작업을 확장하여 Baseball Savant에서 다운로드한 피치별 Statcast 데이터의 데이터베이스를 구축합니다. 그 과정에서 MySQL 외에도 4년간의 플레이(2020-2023)에 걸친 데이터 세트를 저장하고 표현하는 여러 가지 다른 방법을 살펴봅니다. 마지막으로 이러한 다양한 접근 방식의 장단점을 비교 및 대조하고 몇 가지 테스트 드라이브를 위해 이러한 데이터베이스를 활용합니다.

부록 C.10에서는 baseballr 패키지의 statcast_search() 함수를 사용하여 Baseball Savant에서 Statcast 데이터를 다운로드하는 방법을 보여줍니다. 메이저 리그 베이스볼에서 던진 각 투구에 대한 데이터 행이 있기 때문에 이러한 데이터는 작업 흐름을 복잡하게 만들 수 있을 정도로 빠르게 커질 수 있습니다. 이 책에서 사용한 데이터는 대부분 크기가 작았습니다. 즉, 몇 킬로바이트나 메가바이트에 불과하며, 이는 대부분의 개인용 컴퓨터에 있는 물리적 메모리(즉, 랜덤 액세스 메모리) 양보다 훨씬 적습니다. 모든 플레이를 다루는 Retrosheet 데이터조차도 전체 시즌에 대해 100MB 미만입니다. 그러나 아래에서 볼 수 있듯이 Statcast 데이터는 규모가 한 자릿수 더 크며 시즌당 수백 메가바이트를 차지하고 여러 시즌에 걸쳐 수 기가바이트를 차지합니다. 이 정도 규모의 데이터로 작업하면 주의하지 않을 경우 개인용 컴퓨터가 방해를 받거나 과부하가 걸릴 수 있습니다. 따라서 이 장에서는 다년간의 Statcast를 수집한 다음 해당 데이터를 효율적으로 저장하기 위한 새로운 도구를 소개합니다. 다행스럽게도 우리가 어떤 데이터 저장 형식을 선택하든 dplyr과 dbplyr의 마법은 데이터를 분석하는 과정을 동일하게 만듭니다.

DOI: 10.1201/9781032668239-12 280

1년 치 Statcast 데이터 수집 281

12.2 1년 치 Statcast 데이터 수집

많은 세이버메트릭스 아이디어를 더 자세히 탐구하려면 특정 시즌의 모든 플레이에 대한 인플레이 데이터를 수집하는 것이 유용합니다. baseballr에서 제공하는 statcast_search() 함수를 통해 관련 데이터를 수집할 수 있습니다. 그러나 이 함수는 최대 25,000개의 관측치만 반환하고 2023년에 700,000개 이상의 투구가 있었기 때문에 전체 시즌 데이터를 수집하는 데는 약간의 노력이 필요합니다. 이를 위해 매일 Statcast 데이터를 다운로드하는 프로세스를 반복한 다음 해당 데이터를 이 책에서 사용하는 전체 시즌 데이터 세트로 결합하는 방법을 보여줍니다.

정규 시즌의 일반적인 하루에는 약 4,500개의 투구가 발생합니다. 즉, 일반적인 주에는 30,000개 이상의 투구가 발생하며 이는 statcast_search()가 반환하는 25,000개 제한을 초과합니다. 즉, 매주 데이터를 안전하게 다운로드할 수 없습니다. 차선책으로, 단일 일자에 대한 Statcast 데이터를 다운로드하고 지정된 디렉토리 dir에 적절한 이름의 CSV로 해당 데이터를 작성하는 다음 함수를 작성합니다. statcast_daily() 함수에 대한 코드는 abdwr3edata 패키지에 포함되어 있으며 여기에 재현되어 있습니다.

abdwr3edata::statcast_daily

function(the_date = lubridate::now(), dir = getwd()) { if (!dir.exists(dir)) { dir.create(dir, recursive = TRUE)

} filename <- paste0("sc\_", lubridate::as_date(the_date), ".csv") file_path <- fs::path(dir, filename) # if the file already exists, read it. if (file.exists(file_path)) {

x <- file_path |> readr::read_csv() |> suppressMessages()

if (nrow(x) > 0) {

message( paste(

"Found", nrow(x), "observations in", file_path, "..." )

)

} return(NULL)

}

# the file doesn't exist or doesn't have data, get it message(paste("Retrieving data from", the_date)) x <- baseballr::statcast_search(

start_date = lubridate::as_date(the_date), end_date = lubridate::as_date(the_date), player_type = "batter"

) |> dplyr::filter(game_type == "R")

if (nrow(x) > 0) { message(paste("Writing", file_path, "...")) x |>

readr::write_csv(file = fs::path(dir, filename))

} return(NULL)

} <bytecode: 0x582ac35576e0> <environment: namespace:abdwr3edata>

이 경우 작업 디렉토리에 데이터를 저장하고 싶지 않습니다. statcast_csv라는 디렉토리에 저장하고 싶으며, 디렉토리가 아직 없으면 생성해야 합니다. fs 패키지의 path() 함수를 사용하면 우리가 만드는 파일 경로가 모든 운영 체제에서 유효하도록 보장합니다.

library(fs) data_dir <- here::here("data_large") statcast_dir <- path(data_dir, "statcast_csv") if (!dir.exists(statcast_dir)) {

dir.create(statcast_dir) }

이제 Statcast 데이터를 다운로드할 날짜의 벡터를 만듭니다. 예를 들어 2023 시즌 정규 시즌 경기는 3월 30일과 11월 6일 사이에 열렸습니다. 이는 각각 연중 89번째와 274번째 날이었습니다. statcast_daily()를 전체 시즌으로 확장하려면 일련의 날짜에 대해 함수를 반복해야 합니다. lubridate 패키지의 parse_date_time() 함수를 사용하여 정수 벡터를 날짜 벡터로 변환할 수 있습니다.

mlb_2023_dates <- 89:274 |> parse_date_time("%j") |> as_date()

head(mlb_2023_dates)

# the file doesn't exist or doesn't have data, get it message(paste("Retrieving data from", the_date)) x <- baseballr::statcast_search(

start_date = lubridate::as_date(the_date), end_date = lubridate::as_date(the_date), player_type = "batter"

) |> dplyr::filter(game_type == "R")

if (nrow(x) > 0) { message(paste("Writing", file_path, "...")) x |>

readr::write_csv(file = fs::path(dir, filename))

} return(NULL)

} <bytecode: 0x582ac35576e0> <environment: namespace:abdwr3edata>

이 경우 작업 디렉토리에 데이터를 저장하고 싶지 않습니다. statcast_csv라는 디렉토리에 저장하고 싶으며, 디렉토리가 아직 없으면 생성해야 합니다. fs 패키지의 path() 함수를 사용하면 우리가 만드는 파일 경로가 모든 운영 체제에서 유효하도록 보장합니다.

library(fs) data_dir <- here::here("data_large") statcast_dir <- path(data_dir, "statcast_csv") if (!dir.exists(statcast_dir)) {

dir.create(statcast_dir) }

이제 Statcast 데이터를 다운로드할 날짜의 벡터를 만듭니다. 예를 들어 2023 시즌 정규 시즌 경기는 3월 30일과 11월 6일 사이에 열렸습니다. 이는 각각 연중 89번째와 274번째 날이었습니다. statcast_daily()를 전체 시즌으로 확장하려면 일련의 날짜에 대해 함수를 반복해야 합니다. lubridate 패키지의 parse_date_time() 함수를 사용하여 정수 벡터를 날짜 벡터로 변환할 수 있습니다.

mlb_2023_dates <- 89:274 |> parse_date_time("%j") |> as_date()

head(mlb_2023_dates)

1년 치 Statcast 데이터 수집 283

[1] "2024-03-29" "2024-03-30" "2024-03-31" "2024-04-01" [5] "2024-04-02" "2024-04-03"

purrr 패키지의 walk() 함수를 사용하면 이 186일 각각에 대해 statcast_daily() 함수를 연속적으로 적용할 수 있습니다. walk()는 map()과 유사하지만 아무것도 반환하지 않습니다. statcast_daily()는 항상 NULL을 반환하므로 효과는 동일합니다. statcast_daily()는 매일 CSV 파일을 작성하므로 작업을 복제하지 않고 이 함수를 반복해서 안전하게 실행할 수 있습니다. 아래에 표시된 statcast_season() 함수에 대한 코드는 이러한 단계를 결합합니다. 여러 해에 걸쳐 이 함수를 사용하려면 시작일과 종료일 선택을 약간 더 보수적으로 해야 합니다. 이러한 날짜는 해마다 다르지만 일반적으로 4월 1일과 11월 1일에 가까울 것입니다.

abdwr3edata::statcast_season

function(

year = lubridate::year(lubridate::now()), dir = getwd() ) {

if (!dir.exists(dir)) { dir.create(dir, recursive = TRUE)

} mlb_days <- 80:280 mlb_dates <- mlb_days |>

paste(year) |> lubridate::parse_date_time("%j %Y") |> lubridate::as_date()

mlb_dates |> purrr::walk(statcast_daily, dir)

} <bytecode: 0x582ac71bac90> <environment: namespace:abdwr3edata>

디렉토리가 일일 CSV 파일로 채워지면 read_csv()를 호출하여 모두 하나의 큰 데이터 프레임으로 읽어들일 수 있습니다. 이 기능은 statcast_read_csv() 함수에 캡슐화되어 있습니다.

abdwr3edata::statcast_read_csv

function(dir = getwd(), pattern = "\*.csv") { dir |> list.files(pattern = pattern, full.names = TRUE) |>

readr::read_csv() |> dplyr::bind_rows()

} <bytecode: 0x582ac739aac0> <environment: namespace:abdwr3edata>

모두 합쳐서 statcast_season() 함수를 실행하여 2023 시즌에 대한 모든 데이터를 다운로드했는지 확인하고, 완료되면 statcast_read_csv() 함수를 실행하여 다운로드한 다양한 CSV 파일을 읽습니다.

library(abdwr3edata)

# skip this step while building the book! # statcast_season(2023, dir = statcast_dir)

sc2023 <- statcast_dir |> statcast_read_csv(pattern = "2023.+\\.csv")

데이터의 유효성을 확인하기 위해 알려진 값과 비교하여 특정 통계를 부분적으로 확인할 수 있습니다. 먼저 30개 팀이 서로 162경기를 치르면 총 2,430경기가 표시되어야 합니다. 둘째, Baseball-Reference에 따르면 2023년에 홈런이 5,868개 기록되었습니다. 데이터에서 몇 개가 표시됩니까?

sc2023 |> group_by(game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

)

# A tibble: 1 x 4

game_type num_games num_pitches num_hr <chr> <int> <int> <int>

1 R 2430 717945 5868 데이터는 전체 시즌에 대해 정확한 것으로 보입니다. 여러 해의 Statcast 데이터를 얻으려면 statcast_season() 함수를 여러 해에 걸쳐 반복하면 됩니다. 이 장의 나머지 부분에서는 2020-2023 시즌의 데이터를 분석합니다.

readr::read_csv() |> dplyr::bind_rows()

} <bytecode: 0x582ac739aac0> <environment: namespace:abdwr3edata>

모두 합쳐서 statcast_season() 함수를 실행하여 2023 시즌에 대한 모든 데이터를 다운로드했는지 확인하고, 완료되면 statcast_read_csv() 함수를 실행하여 다운로드한 다양한 CSV 파일을 읽습니다.

library(abdwr3edata)

# skip this step while building the book! # statcast_season(2023, dir = statcast_dir)

sc2023 <- statcast_dir |> statcast_read_csv(pattern = "2023.+\\.csv")

데이터의 유효성을 확인하기 위해 알려진 값과 비교하여 특정 통계를 부분적으로 확인할 수 있습니다. 먼저 30개 팀이 서로 162경기를 치르면 총 2,430경기가 표시되어야 합니다. 둘째, Baseball-Reference에 따르면 2023년에 홈런이 5,868개 기록되었습니다. 데이터에서 몇 개가 표시됩니까?

sc2023 |> group_by(game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

)

# A tibble: 1 x 4

game_type num_games num_pitches num_hr <chr> <int> <int> <int>

1 R 2430 717945 5868 데이터는 전체 시즌에 대해 정확한 것으로 보입니다. 여러 해의 Statcast 데이터를 얻으려면 statcast_season() 함수를 여러 해에 걸쳐 반복하면 됩니다. 이 장의 나머지 부분에서는 2020-2023 시즌의 데이터를 분석합니다.

12.3 대용량 데이터를 효율적으로 저장하기

데이터의 무결성에 만족하면 나중에 작업하기 쉬운 형식으로 데이터를 저장해야 합니다. 매일 데이터를 다운로드하는 데는 긴 시간이 걸렸으며 데이터 사본이 이미 확보되면 Baseball Savant 서버에 계속 핑을 보낼 이유가 없습니다.

위에서 언급했듯이 Statcast 데이터의 전체 시즌에는 700,000개 이상의 행과 거의 100개의 변수가 포함되어 있습니다. 이는 R 메모리 공간의 약 0.5기가바이트를 차지합니다.

sc2023 |> object.size() |> print(units = "MB")

500.7 Mb

요즘 대부분의 컴퓨터에는 최소 16GB의 메모리가 있지만 이 정도 규모의 데이터로 작업하면 (특히 여러 시즌에 걸쳐 데이터로 작업하려는 경우) 빠르게 부담이 될 수 있습니다. 데이터를 디스크에 쓰고 효율적인 형식으로 저장하면 도움이 될 수 있습니다.

위에서 전체 시즌 Statcast 데이터가 포함된 데이터 프레임이 메모리에서 약 0.5기가바이트를 차지한다는 것을 확인했습니다. 우리는 각각 하루 분량의 데이터가 포함된 많은 작은 CSV를 결합하여 해당 데이터 프레임을 구축했습니다. 해당 CSV 파일은 디스크에서 어느 정도의 공간을 차지합니까?

statcast_dir |> list.files(full.names = TRUE) |> str_subset("2023") |> file.size() |> sum() / 1024^2

[1] 373

이 경우 CSV는 데이터가 메모리로 읽혀질 때 디스크에서 데이터가 차지하는 공간의 약 75%를 차지합니다. 나쁘지 않습니다만 더 개선할 수 있습니다.

CSV의 주요 장점은 사람이 읽을 수 있고 편집할 수 있어 이해하기 쉽다는 것입니다. 널리 사용되기 때문에 데이터 작업을 위해 설계된 거의 모든 프로그램에서 읽고 쓸 수 있습니다. 그러나 CSV는 공간 효율적인 파일 형식이 아닙니다. 몇 가지 다른 옵션을 살펴보겠습니다.

12.3.1 R 내부 데이터 형식 사용하기

이제 여러 해의 Statcast 데이터로 작업하려고 한다고 가정해 보겠습니다. 예를 들어 2020-2023 시즌의 데이터를 사용하여 팬데믹 시대의 추세를 조사하려고 한다고 가정해 보겠습니다. 이 데이터를 4개의 파일로 저장해야 할까요, 아니면 1개로 저장해야 할까요? 데이터를 4개의 파일(각 연도에 하나씩)에 보관하면 데이터를 R로 개별적으로 읽어들인 다음 결과 데이터 프레임을 하나의 큰 데이터 프레임으로 결합해야 하므로 여러 해에 걸쳐 분석하는 것이 번거로울 것입니다. 그 큰 데이터 프레임은 수 기가바이트의 메모리를 차지하여 성능이 저하될 수 있습니다. 또는 이 작업을 한 번 수행한 다음 4년 치 데이터가 모두 포함된 하나의 큰 \*.rds 파일을 작성할 수 있습니다. 그러나 그렇게 하면 파일이 너무 커지고 새로운 데이터가 들어올 때마다 해당 프로세스를 다시 수행해야 합니다.

데이터를 R로 읽어들인 후에는 write_rds() 함수를 사용하여 전체 시즌 데이터 프레임을 디스크에 쓰고 안전하게 보관할 수 있습니다. 이는 R의 내부 이진 데이터 저장 형식을 사용하며 CSV보다 공간 효율성이 훨씬 뛰어납니다. 그러나 .rds 형식은 R과 작동하도록 설계되었으므로 R에서만 독점적으로 작업하려는 경우에는 좋지만 데이터를 다른 애플리케이션(Python) 사용자와 공유하려는 경우에는 그다지 유용하지 않습니다.

statcast_write_rds() 함수는 write_rds() 함수를 래핑하지만 먼저 데이터를 연도 기준으로 그룹으로 분할합니다. 전달하는 데이터 프레임에 있는 각 연도에 대해 적절하게 이름이 지정된 다른 \*.rds 파일을 작성합니다.

abdwr3edata::statcast_write_rds

function(x, dir = getwd(), ...) { tmp <- x |> dplyr::group_by(year = lubridate::year(game_date))

years <- tmp |> dplyr::group_keys() |> dplyr::pull(year)

tmp |> dplyr::group_split() |> rlang::set_names(years) |> purrr::map(

~readr::write_rds(

.x, file = fs::path(

dir, paste0(

"statcast\_", max(lubridate::year(dplyr::pull(.x, game_date))), ".rds"

12.3.1 R 내부 데이터 형식 사용하기

이제 여러 해의 Statcast 데이터로 작업하려고 한다고 가정해 보겠습니다. 예를 들어 2020-2023 시즌의 데이터를 사용하여 팬데믹 시대의 추세를 조사하려고 한다고 가정해 보겠습니다. 이 데이터를 4개의 파일로 저장해야 할까요, 아니면 1개로 저장해야 할까요? 데이터를 4개의 파일(각 연도에 하나씩)에 보관하면 데이터를 R로 개별적으로 읽어들인 다음 결과 데이터 프레임을 하나의 큰 데이터 프레임으로 결합해야 하므로 여러 해에 걸쳐 분석하는 것이 번거로울 것입니다. 그 큰 데이터 프레임은 수 기가바이트의 메모리를 차지하여 성능이 저하될 수 있습니다. 또는 이 작업을 한 번 수행한 다음 4년 치 데이터가 모두 포함된 하나의 큰 \*.rds 파일을 작성할 수 있습니다. 그러나 그렇게 하면 파일이 너무 커지고 새로운 데이터가 들어올 때마다 해당 프로세스를 다시 수행해야 합니다.

데이터를 R로 읽어들인 후에는 write_rds() 함수를 사용하여 전체 시즌 데이터 프레임을 디스크에 쓰고 안전하게 보관할 수 있습니다. 이는 R의 내부 이진 데이터 저장 형식을 사용하며 CSV보다 공간 효율성이 훨씬 뛰어납니다. 그러나 .rds 형식은 R과 작동하도록 설계되었으므로 R에서만 독점적으로 작업하려는 경우에는 좋지만 데이터를 다른 애플리케이션(Python) 사용자와 공유하려는 경우에는 그다지 유용하지 않습니다.

statcast_write_rds() 함수는 write_rds() 함수를 래핑하지만 먼저 데이터를 연도 기준으로 그룹으로 분할합니다. 전달하는 데이터 프레임에 있는 각 연도에 대해 적절하게 이름이 지정된 다른 \*.rds 파일을 작성합니다.

abdwr3edata::statcast_write_rds

function(x, dir = getwd(), ...) { tmp <- x |> dplyr::group_by(year = lubridate::year(game_date))

years <- tmp |> dplyr::group_keys() |> dplyr::pull(year)

tmp |> dplyr::group_split() |> rlang::set_names(years) |> purrr::map(

~readr::write_rds(

.x, file = fs::path(

dir, paste0(

"statcast\_", max(lubridate::year(dplyr::pull(.x, game_date))), ".rds"

)

), compress = "xz",

# ...

) )

list.files(dir, pattern = "\*.rds", full.names = TRUE)

} <bytecode: 0x582ac79e6c38> <environment: namespace:abdwr3edata>

이 경우 CSV 파일 디렉토리는 동일한 데이터가 포함된 단일 .rds 파일(64Mb)보다 거의 6배 더 많은 공간(2023 파일의 경우 373Mb)을 차지합니다.

disk_space_rds <- data_dir |> path("statcast_rds") |> dir_info(regexp = "\*.rds") |> select(path, size) |> mutate(

path = path_file(path), format = "rds"

) disk_space_rds

# A tibble: 4 x 3 path size format <chr> <fs::bytes> <chr>

- 1 statcast_2020.rds 23.4M rds
- 2 statcast_2021.rds 63.1M rds
- 3 statcast_2022.rds 63M rds
- 4 statcast_2023.rds 63.8M rds

  12.3.2 Apache Arrow와 Apache Parquet 사용하기

Apache Parquet은 소프트웨어 프레임워크인 Apache Arrow와 결합하여 데이터 저장 방법에 대해 위에서 제기한 문제에 대한 매끄럽고 확장 가능한 솔루션을 제공하는 파일 형식입니다. Parquet 형식은 \*.rds 형식만큼 공간 효율적이지는 않지만 그룹화 변수(이 경우 연도)를 기반으로 데이터를 파티션으로 자동 분할한다는 점에서 교차 플랫폼이며 확장 가능합니다. R용 arrow 패키지는 Parquet 형식의 데이터를 매우 쉽게 다룰 수 있는 dplyr 호환 인터페이스를 제공합니다. Arrow는 행 지향이 아닌 열 지향이므로 매우 빠를 수 있습니다. Arrow에 대한 자세한 내용은 Wickham, Cetinkaya-Rundel 및 Grolemund(2023)의 Arrow 장을 참조하십시오.

arrow의 좋은 기능 중 하나는 open_dataset() 함수를 사용하여 CSV의 전체 디렉토리를 읽을 수 있다는 것입니다.

library(arrow) sc_arrow <- statcast_dir |>

open_dataset(format = "csv") dim(sc_arrow)

[1] 2399921 92

Arrow 객체 sc_arrow는 데이터 프레임처럼 작동하지만(이 경우 거의 240만 행 포함!) 메모리에서 공간을 거의 차지하지 않습니다.

sc_arrow |> object.size()

504 bytes

이것이 가능한 이유는 데이터가 여전히 CSV 형식으로 디스크에 있기 때문입니다. 아직 R의 메모리로 읽히지 않았습니다. 하지만 그렇다고 데이터를 쿼리할 수 없는 것은 아닙니다.

summary_arrow <- sc_arrow |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

) summary_arrow |> collect()

# A tibble: 4 x 5 # Groups: year [4]

year game_type num_games num_pitches num_hr <int> <chr> <int> <int> <int>

- 1 2020 R 898 263584 2304
- 2 2021 R 2429 709852 5944
- 3 2022 R 2430 708540 5215
- 4 2023 R 2430 717945 5868

지금까지 우리는 CSV 디렉토리로 백업되는 Arrow 객체로 작업했습니다. write_dataset() 함수를 사용하여 Parquet 형식의 Arrow 데이터 프레임을 작성할 수 있습니다. 먼저 group_by() 함수를 사용했기 때문에 각 연도마다 하나의 Parquet 파일을 얻게 됩니다. 이것은 파일 기반

arrow의 좋은 기능 중 하나는 open_dataset() 함수를 사용하여 CSV의 전체 디렉토리를 읽을 수 있다는 것입니다.

library(arrow) sc_arrow <- statcast_dir |>

open_dataset(format = "csv") dim(sc_arrow)

[1] 2399921 92

Arrow 객체 sc_arrow는 데이터 프레임처럼 작동하지만(이 경우 거의 240만 행 포함!) 메모리에서 공간을 거의 차지하지 않습니다.

sc_arrow |> object.size()

504 bytes

이것이 가능한 이유는 데이터가 여전히 CSV 형식으로 디스크에 있기 때문입니다. 아직 R의 메모리로 읽히지 않았습니다. 하지만 그렇다고 데이터를 쿼리할 수 없는 것은 아닙니다.

summary_arrow <- sc_arrow |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

) summary_arrow |> collect()

- # A tibble: 4 x 5 # Groups: year [4]

year game_type num_games num_pitches num_hr <int> <chr> <int> <int> <int>

- 1 2020 R 898 263584 2304
- 2 2021 R 2429 709852 5944
- 3 2022 R 2430 708540 5215
- 4 2023 R 2430 717945 5868

지금까지 우리는 CSV 디렉토리로 백업되는 Arrow 객체로 작업했습니다. write_dataset() 함수를 사용하여 Parquet 형식의 Arrow 데이터 프레임을 작성할 수 있습니다. 먼저 group_by() 함수를 사용했기 때문에 각 연도마다 하나의 Parquet 파일을 얻게 됩니다. 이것은 상당한 성능 이점을 제공할 수 있는 파일 기반

파티셔닝의 한 형태입니다(Wickham, Cetinkaya-Rundel, and Grolemund 2023).

statcast_parquet <- path(data_dir, "statcast_parquet") if (!dir.exists(statcast_parquet)) {

dir.create(statcast_parquet)

} sc_arrow |>

group_by(year = year(game_date)) |> write_dataset(path = statcast_parquet, format = "parquet")

write_dataset() 함수는 데이터를 각 전체 시즌에 대해 약 100MB의 개별 파일로 분할하는 디렉토리 구조를 자동으로 생성합니다. 이것은 CSV가 차지하는 디스크 공간의 4분의 1보다 조금 더 큽니다.

disk_space_parquet <- statcast_parquet |> dir_info(recurse = TRUE, glob = "\*.parquet") |> select(path, size) |> mutate(

format = "parquet", path = path_rel(path, start = statcast_parquet)

) disk_space_parquet

# A tibble: 4 x 3 path size format <fs::path> <fs::bytes> <chr>

- 1 year=2020/part-0.parquet 37.4M parquet
- 2 year=2021/part-0.parquet 101.5M parquet
- 3 year=2022/part-0.parquet 101.1M parquet
- 4 year=2023/part-0.parquet 102.4M parquet

  12.3.3 DuckDB 사용하기

Arrow는 R에서 Arrow 객체를 데이터 프레임처럼 매끄럽게 사용할 수 있는 dplyr 인터페이스를 제공하지만 Arrow는 SQL 기반이 아닙니다. 따라서 Arrow와 Parquet은 교차 플랫폼이지만 이에 대해 SQL 쿼리를 작성하려면 다른 인터페이스가 필요합니다.

SQL 기반인 또 다른 빠르고 교차 플랫폼 대안은 DuckDB입니다. SQLite와 마찬가지로 DuckDB에는 데이터를 메모리에 저장하거나 디스크에 데이터베이스 파일을 로컬로 쓸 수 있는 서버리스 아키텍처가 있습니다. 따라서 SQL 인터페이스를 원하지만 SQL 서버를 설정하거나 유지 관리하고 싶지 않은 사람에게 훌륭한 옵션입니다. DuckDB에 대한 자세한 내용은 Wickham, Cetinkaya-Rundel 및 Grolemund(2023)에서 알아볼 수 있습니다.

DuckDB도 dplyr 인터페이스를 구현하므로 DBI 호환 SQL 데이터베이스 연결을 설정하는 것과 동일한 방식으로 즉, dbConnect()를 사용하여 데이터베이스 연결을 설정합니다. 그러나 이 경우에는 향후에 다시 사용할 수 있고 다른 저장 형식과 크기를 비교할 수 있도록 디스크에 데이터베이스 파일을 쓰려고 합니다. dbdir 인수는 데이터베이스 파일이 아직 없는 경우 생성될 DuckDB 데이터베이스 파일의 경로를 지정합니다.

statcast_duckdb <- path(data_dir, "statcast_duckdb") if (!dir.exists(statcast_duckdb)) {

dir.create(statcast_duckdb)

} library(duckdb) con_duckdb <- dbConnect(

drv = duckdb(), dbdir = path(statcast_duckdb, "statcast.ddb")

)

초기에 DuckDB 데이터베이스에는 테이블이 없으므로 dbWriteTable() 함수를 사용하여 Arrow 객체의 내용을 DuckDB 객체로 복사합니다.1

con_duckdb |> dbWriteTable("events", collect(sc_arrow), overwrite = TRUE)

이제 친숙한 dplyr 인터페이스를 사용하여 DuckDB 데이터베이스에 액세스할 수 있습니다.

sc_ddb <- con_duckdb |> tbl("events")

summary_duckdb <- sc_ddb |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(as.numeric(events == "home_run"), na.rm = TRUE)

)

summary_duckdb

# Source: SQL [4 x 5] # Database: DuckDB v0.9.2 [bbaumer@Linux:R 4.3.2//statcast.ddb] # Groups: year

1arrow 패키지에는 기존 Arrow 객체에서 DuckDB 객체를 생성하는 to_duckdb()라는 함수가 포함되어 있지만 이 경우에는 데이터를 복사하고 싶으므로 사용하지 않습니다.

DuckDB도 dplyr 인터페이스를 구현하므로 DBI 호환 SQL 데이터베이스 연결을 설정하는 것과 동일한 방식으로 즉, dbConnect()를 사용하여 데이터베이스 연결을 설정합니다. 그러나 이 경우에는 향후에 다시 사용할 수 있고 다른 저장 형식과 크기를 비교할 수 있도록 디스크에 데이터베이스 파일을 쓰려고 합니다. dbdir 인수는 데이터베이스 파일이 아직 없는 경우 생성될 DuckDB 데이터베이스 파일의 경로를 지정합니다.

statcast_duckdb <- path(data_dir, "statcast_duckdb") if (!dir.exists(statcast_duckdb)) {

dir.create(statcast_duckdb)

} library(duckdb) con_duckdb <- dbConnect(

drv = duckdb(), dbdir = path(statcast_duckdb, "statcast.ddb")

)

초기에 DuckDB 데이터베이스에는 테이블이 없으므로 dbWriteTable() 함수를 사용하여 Arrow 객체의 내용을 DuckDB 객체로 복사합니다.1

con_duckdb |> dbWriteTable("events", collect(sc_arrow), overwrite = TRUE)

이제 친숙한 dplyr 인터페이스를 사용하여 DuckDB 데이터베이스에 액세스할 수 있습니다.

sc_ddb <- con_duckdb |> tbl("events")

summary_duckdb <- sc_ddb |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(as.numeric(events == "home_run"), na.rm = TRUE)

)

summary_duckdb

# Source: SQL [4 x 5] # Database: DuckDB v0.9.2 [bbaumer@Linux:R 4.3.2//statcast.ddb] # Groups: year

1arrow 패키지에는 기존 Arrow 객체에서 DuckDB 객체를 생성하는 to_duckdb()라는 함수가 포함되어 있지만 이 경우에는 데이터를 복사하고 싶으므로 사용하지 않습니다.

year game_type num_games num_pitches num_hr <dbl> <chr> <dbl> <dbl> <dbl>

- 1 2020 R 898 263584 2304
- 2 2022 R 2430 708540 5215
- 3 2021 R 2429 709852 5944
- 4 2023 R 2430 717945 5868

arrow 및 duckdb 패키지 모두 dplyr 인터페이스를 제공하지만 dbGetQuery()와 같은 SQL 도구와 함께 작동하는 것은 duckdb뿐입니다.

con_duckdb |>

dbGetQuery(" SELECT game_date, pitch_type, release_speed, pitcher FROM events WHERE release_speed > 100 AND events = 'home_run' LIMIT 6;

")

game_date pitch_type release_speed pitcher

- 1 2021-04-10 FF 100 594798
- 2 2021-05-18 SI 100 621237
- 3 2021-06-27 FF 100 543037
- 4 2021-06-29 SI 101 621237
- 5 2021-07-09 FC 100 661403
- 6 2021-07-16 FC 100 661403

DuckDB 데이터베이스의 저장 공간은 CSV의 저장 공간과 비슷하지만 성능은 탁월하다는 것을 섹션 12.4에서 확인할 수 있습니다.

disk_space_duckdb <- statcast_duckdb |> dir_info(recurse = TRUE, glob = "\*.ddb") |> select(path, size) |> mutate(

format = "duckdb", path = path_rel(path, start = statcast_duckdb)

) disk_space_duckdb

# A tibble: 1 x 3 path size format <fs::path> <fs::bytes> <chr>

1 statcast.ddb 1.28G duckdb

12.3.4 MySQL 사용하기

마지막으로 섹션 11.2에서 설정한 MariaDB(MySQL) 데이터베이스를 사용할 수 있습니다.

library(dbplyr) library(RMariaDB) con_mariadb <- dbConnect(MariaDB(), group = "abdwr")

DuckDB에서 했던 것과 마찬가지로 먼저 dbWriteTable()을 사용하여 데이터를 MySQL 서버에 복사합니다.

con_mariadb |> dbWriteTable("events", collect(sc_arrow), overwrite = TRUE)

이제 dplyr 인터페이스를 사용하거나 SQL 쿼리를 작성하여 데이터베이스를 쿼리할 수 있습니다.

sc_maria <- con_mariadb |> tbl("events")

summary_maria <- sc_maria |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

) summary_maria

# Source: SQL [4 x 5] # Database: mysql [abdwr@localhost:NA/abdwr] # Groups: year

year game_type num_games num_pitches num_hr <int> <chr> <int64> <int64> <dbl>

- 1 2020 R 898 263584 2304
- 2 2021 R 2429 709852 5944
- 3 2022 R 2430 708540 5215
- 4 2023 R 2430 717945 5868

MySQL 데이터베이스의 저장 공간을 확인하는 것은 약간 더 복잡하지만 아래 출력의 events.ibd 및 events.frm 파일 크기는 events 테이블 크기의 하한을 제공합니다. 여기서의 저장 공간은 원래 CSV 파일 및 DuckDB의 저장 공간과 거의 같습니다.

- 12.3.4 MySQL 사용하기

마지막으로 섹션 11.2에서 설정한 MariaDB(MySQL) 데이터베이스를 사용할 수 있습니다.

library(dbplyr) library(RMariaDB) con_mariadb <- dbConnect(MariaDB(), group = "abdwr")

DuckDB에서 했던 것과 마찬가지로 먼저 dbWriteTable()을 사용하여 데이터를 MySQL 서버에 복사합니다.

con_mariadb |> dbWriteTable("events", collect(sc_arrow), overwrite = TRUE)

이제 dplyr 인터페이스를 사용하거나 SQL 쿼리를 작성하여 데이터베이스를 쿼리할 수 있습니다.

disk_space_mariadb <- "/var/lib/mysql/abdwr/" |> dir_info(glob = "_events._") |> select(path, size) |> mutate(

format = "mariadb", path = path_rel(path, start = "/var/lib/mysql/abdwr/")

) disk_space_mariadb

# A tibble: 2 x 3 path size format <fs::path> <fs::bytes> <chr>

- 1 events.frm 3.92K mariadb
- 2 events.ibd 1.22G mariadb

sc_maria <- con_mariadb |> tbl("events")

summary_maria <- sc_maria |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

) summary_maria

# Source: SQL [4 x 5] # Database: mysql [abdwr@localhost:NA/abdwr] # Groups: year

year game_type num_games num_pitches num_hr

<int> <chr> <int64> <int64> <dbl> 1 2020 R 898 263584 2304 2 2021 R 2429 709852 5944 3 2022 R 2430 708540 5215 4 2023 R 2430 717945 5868

MySQL 데이터베이스의 저장 공간을 확인하는 것은 약간 더 복잡하지만 아래 출력의 events.ibd 및 events.frm 파일 크기는 events 테이블 크기의 하한을 제공합니다. 여기서의 저장 공간은 원래 CSV 파일 및 DuckDB의 저장 공간과 거의 같습니다.

### 12.4 성능 비교

이 장에서는 5가지 데이터 저장 형식(CSV, \*.rds, Parquet, DuckDB, MariaDB)과 해당 R 객체 인터페이스를 살펴보았으며, 이들은 모두 dplyr 패키지의 형식과 호환됩니다. 컴퓨팅 성능은 종종 계산 속도, 메모리 공간 및 디스크 저장 공간의 세 가지 양으로 측정됩니다. 이러한 세 가지 기준을 차례로 고려합니다.

12.4.1 계산 속도

먼저 쿼리 속도 측면에서 성능을 비교합니다. bench 패키지의 mark() 함수를 사용하여 2020-2023 데이터에 대한 Statcast 요약 통계를 계산하는 데 걸리는 시간을 비교합니다. 객체의 5가지 클래스는 다음과 같습니다. 1) 데이터를 메모리에 저장하는 tbl(데이터 프레임), 2) CSV 파일로 백업되는 Arrow 객체, 3) 연도별로 분할된 Parquet 파일로 백업되는 Arrow 객체, 4) DuckDB 객체, 5) MariaDB 객체.

먼저 tbl 인터페이스를 설정하고 전체 4년 치 데이터를 쿼리합니다.

sc_tbl <- statcast_dir |> statcast_read_csv()

summary_tbl <- sc_tbl |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

)

둘째, 앞서 만든 Parquet 파일에서 읽어오고 연도 기준의 파일 기반 파티셔닝을 활용하도록 arrow 객체를 설정합니다.

sc_arrow_part <- statcast_parquet |> open_dataset(partitioning = "year")

summary_arrow_part <- sc_arrow_part |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

)

이제 쿼리 실행 시간을 벤치마킹할 수 있습니다.

library(bench) res <- mark(

tbl = summary_tbl, arrow_csv = summary_arrow |> collect(), arrow_part = summary_arrow_part |> collect(), duckdb = summary_duckdb |> collect(), mariadb = summary_maria |> collect(), check = FALSE

) |>

arrange(median) res

# A tibble: 5 x 6

expression min median `itr/sec` mem_alloc `gc/sec` <bch:expr> <bch:tm> <bch:tm> <dbl> <bch:byt> <dbl>

- 1 tbl 18.04ns 22ns 45777998. 0B 0
- 2 duckdb 59.18ms 59.38ms 16.6 446.1KB 5.52
- 3 arrow_part 143.9ms 145.94ms 6.79 47.6KB 0
- 4 arrow_csv 991.09ms 991.09ms 1.01 47.9KB 0
- 5 mariadb 3.54s 3.54s 0.282 154.8KB 0

성능은 사용 가능한 하드웨어 및 컴퓨터의 소프트웨어 구성에 따라 크게 다를 수 있습니다. 12개의 CPU와 32기가바이트의 RAM을 갖춘 이 기계에서 결과는 duckdb가 다른 데이터베이스보다 훨씬 빠르다는 것을 나타냅니다. tbl 인터페이스에는 이미 데이터가 메모리에 있으므로

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

)

둘째, 앞서 만든 Parquet 파일에서 읽어오고 연도 기준의 파일 기반 파티셔닝을 활용하도록 arrow 객체를 설정합니다.

sc_arrow_part <- statcast_parquet |> open_dataset(partitioning = "year")

summary_arrow_part <- sc_arrow_part |> group_by(year = year(game_date), game_type) |> summarize(

num_games = n_distinct(game_pk), num_pitches = n(), num_hr = sum(events == "home_run", na.rm = TRUE)

)

이제 쿼리 실행 시간을 벤치마킹할 수 있습니다.

library(bench) res <- mark(

tbl = summary_tbl, arrow_csv = summary_arrow |> collect(), arrow_part = summary_arrow_part |> collect(), duckdb = summary_duckdb |> collect(), mariadb = summary_maria |> collect(), check = FALSE

) |>

arrange(median) res

- # A tibble: 5 x 6 expression min median `itr/sec` mem_alloc `gc/sec` <bch:expr> <bch:tm> <bch:tm> <dbl> <bch:byt> <dbl>

- 1 tbl 18.04ns 22ns 45777998. 0B 0
- 2 duckdb 59.18ms 59.38ms 16.6 446.1KB 5.52
- 3 arrow_part 143.9ms 145.94ms 6.79 47.6KB 0
- 4 arrow_csv 991.09ms 991.09ms 1.01 47.9KB 0
- 5 mariadb 3.54s 3.54s 0.282 154.8KB 0

성능은 사용 가능한 하드웨어 및 컴퓨터의 소프트웨어 구성에 따라 크게 다를 수 있습니다. 12개의 CPU와 32기가바이트의 RAM을 갖춘 이 기계에서 결과는 duckdb가 다른 데이터베이스보다 훨씬 빠르다는 것을 나타냅니다. tbl 인터페이스에는 이미 데이터가 메모리에 있으므로

물론 단연코 가장 빠릅니다. 이 경우 쿼리가 4개 파티션 전체에서 실행되었으므로 Parquet의 파티셔닝 체계는 그다지 유용하지 않았습니다. 그럼에도 불구하고 Parquet 파일로 백업된 Arrow 객체는 CSV 파일로 백업된 Arrow 객체보다 약 7배 더 빨랐습니다. 그러나 DuckDB 인스턴스는 여전히 Arrow/Parquet 객체보다 약 2배 더 빨랐습니다. CSV 파일로 백업된 Arrow 객체조차도 RMariaDB보다 약 4배 더 빠르며, RMariaDB는 단연 가장 성능이 낮았습니다.

섹션 12.5에서는 개별 시즌의 데이터를 개별적으로 쿼리할 때 Arrow/Parquet 객체의 성능이 향상되는지 살펴봅니다.

- 12.4.2 메모리 공간

둘째, 다른 객체는 메모리에서 차지하는 공간이 미미한 반면, tbl 객체는 RAM을 1.6Gb나 차지합니다! 위에서 언급했듯이 tbl 객체는 데이터를 메모리에 저장하는 반면, 다른 객체는 데이터를 디스크에 남겨두고 요청이 있을 때만 관련 데이터를 쿼리하기 때문입니다. 이러한 객체의 크기(바이트)는 아래에 나와 있습니다. 따라서 tbl 객체의 우수한 성능에는 대가가 따릅니다.

list( "tbl" = sc_tbl, "arrow" = sc_arrow, "duckdb" = sc_ddb, "mariadb" = sc_maria

) |> map_int(object.size)

tbl arrow duckdb mariadb 1752447264 504 44384 12672

- 12.4.3 디스크 저장 공간

셋째, 데이터 저장 공간 측면에서 _.rds 형식이 가장 작습니다. 그러나 고유한 제한 사항이 있습니다. 즉, _.rds는 R에서만 작동합니다. Parquet 파일은 \*.rds 파일보다 약 50% 더 많은 공간을 차지하지만 여러 플랫폼에서 작동하며 원활한 파티셔닝을 구현합니다. DuckDB 파일은 Parquet 파일보다 약 3배의 디스크 공간을 차지하지만(아직 인덱스를 작성하기 전입니다!) 쿼리 속도가 더 빠릅니다.

disk_space <- bind_rows( disk_space_csv, disk_space_rds, disk_space_parquet, disk_space_duckdb,

disk_space_mariadb

) disk_space |>

group_by(format) |> summarize(footprint = sum(size)) |> arrange(desc(footprint))

# A tibble: 5 x 2

format footprint <chr> <fs::bytes>

- 1 duckdb 1.28G
- 2 mariadb 1.22G
- 3 csv 1.22G
- 4 parquet 342.46M
- 5 rds 213.36M

- 12.4.4 전체 지침

이 데이터를 사용하여 이 특정 컴퓨터에서 240만 행 데이터 세트를 사용한 실험에서 메모리의 tbl 객체로 데이터를 읽어들이면 메모리 공간이 가장 많이 필요한 반면 계산 성능이 가장 빠르다는 명백한 사실을 확인했습니다. 디스크에서 데이터를 읽는 인터페이스 중에서는 duckdb가 가장 빠른 계산 성능을 제공한 반면 RMariaDB는 가장 느린 성능을 제공했습니다. 둘 다 원본 CSV 파일에서 디스크의 가장 큰 공간을 크게 줄이지 못했습니다. Parquet 저장 형식을 사용하는 Arrow는 디스크에서 중간 정도의 공간을 차지하면서 중간 정도의 성능을 제공했습니다. R의 내부 \*.rds 저장 형식은 가장 작았지만 활용도가 가장 낮았습니다.

이는 더 광범위한 실제 적용에서 이러한 옵션을 고려할 때 다음 지침으로 이어집니다.

- • 데이터가 작은 경우(즉, 몇 백 메가바이트 미만) 쉽고 교차 플랫폼이며 다재다능하므로 CSV를 사용하십시오. 공간 효율적이지 않다는 사실은 어쨌든 데이터가 작기 때문에 중요하지 않습니다.
- • 데이터가 몇 백 메가바이트보다 크고 R에서만 작업하는 경우(혼자 또는 소수의 동료와 함께) 공간 효율적이고 R에 최적화된 .rds를 사용하십시오. (이것이 이 책에서 사용하는 많은 Retrosheet 및 Statcast 데이터를 저장하기로 선택한 방법입니다.) 이러한 파일을 tbl 객체로 읽어들이면 빠른 성능으로 이어지며, 데이터는 아마도 컴퓨터 메모리를 눈에 띄게 소모할 만큼 크지 않을 것입니다.
- • 데이터가 1기가바이트 이상이고 다양한 플랫폼(즉, R뿐만 아니라 Python 등)에서 데이터 파일을 공유해야 하며 SQL 기반 RDBMS를 사용하고 싶지 않은 경우 데이터를 Parquet 형식으로 저장하고 arrow 패키지를 사용하십시오. Parquet은 교차 플랫폼이며 Arrow는

disk_space_mariadb

) disk_space |>

group_by(format) |> summarize(footprint = sum(size)) |> arrange(desc(footprint))

# A tibble: 5 x 2

format footprint <chr> <fs::bytes>

- 1 duckdb 1.28G
- 2 mariadb 1.22G
- 3 csv 1.22G
- 4 parquet 342.46M
- 5 rds 213.36M

- .rds보다 더 잘 확장됩니다. 성능과 저장 공간 모두 필요를 충족할 가능성이 높습니다. 파일 기반 파티셔닝 체계는 데이터베이스를 쿼리하는 방법에 따라 도움이 될 수도 있고 그렇지 않을 수도 있습니다.
- • 로컬 데이터 저장소와 함께 SQL에서 작업하려면 RSQLite보다 더 많은 기능과 더 나은 성능을 제공하고 설정 및 유지 관리가 번거로울 수 있는 서버-클라이언트 아키텍처가 필요하지 않은 DuckDB를 사용하십시오.
- • RDBMS 서버에 액세스할 수 있는 경우(전문 데이터베이스 관리자가 유지 관리하기를 바랍니다) 적절한 DBI 인터페이스(RMariaDB, RPostgreSQL 등)를 사용하여 연결하십시오. 리소스가 충분하고 잘 관리되는 서버는 개인용 컴퓨터에서 할 수 있는 모든 작업을 쉽게 능가합니다.

상황에 맞는 적절한 선택을 하려면 이러한 요소를 신중하게 평가해야 합니다.

- 12.4.4 전체 지침

이 데이터를 사용하여 이 특정 컴퓨터에서 240만 행 데이터 세트를 사용한 실험에서 메모리의 tbl 객체로 데이터를 읽어들이면 메모리 공간이 가장 많이 필요한 반면 계산 성능이 가장 빠르다는 명백한 사실을 확인했습니다. 디스크에서 데이터를 읽는 인터페이스 중에서는 duckdb가 가장 빠른 계산 성능을 제공한 반면 RMariaDB는 가장 느린 성능을 제공했습니다. 둘 다 원본 CSV 파일에서 디스크의 가장 큰 공간을 크게 줄이지 못했습니다. Parquet 저장 형식을 사용하는 Arrow는 디스크에서 중간 정도의 공간을 차지하면서 중간 정도의 성능을 제공했습니다. R의 내부 \*.rds 저장 형식은 가장 작았지만 활용도가 가장 낮았습니다.

이는 더 광범위한 실제 적용에서 이러한 옵션을 고려할 때 다음 지침으로 이어집니다.

- • 데이터가 작은 경우(즉, 몇 백 메가바이트 미만) 쉽고 교차 플랫폼이며 다재다능하므로 CSV를 사용하십시오. 공간 효율적이지 않다는 사실은 어쨌든 데이터가 작기 때문에 중요하지 않습니다.
- • 데이터가 몇 백 메가바이트보다 크고 R에서만 작업하는 경우(혼자 또는 소수의 동료와 함께) 공간 효율적이고 R에 최적화된 .rds를 사용하십시오. (이것이 이 책에서 사용하는 많은 Retrosheet 및 Statcast 데이터를 저장하기로 선택한 방법입니다.) 이러한 파일을 tbl 객체로 읽어들이면 빠른 성능으로 이어지며, 데이터는 아마도 컴퓨터 메모리를 눈에 띄게 소모할 만큼 크지 않을 것입니다.
- • 데이터가 1기가바이트 이상이고 다양한 플랫폼(즉, R뿐만 아니라 Python 등)에서 데이터 파일을 공유해야 하며 SQL 기반 RDBMS를 사용하고 싶지 않은 경우 데이터를 Parquet 형식으로 저장하고 arrow 패키지를 사용하십시오. Parquet은 교차 플랫폼이며 Arrow는

### 12.5 발사 각도 및 타구 속도, 재고찰

이 책의 이전 판에서는 2017 시즌의 모든 인플레이 타구에 대해 wOBA가 발사 각도와 타구 속도의 함수로 어떻게 달라지는지 보여주는 데이터 그래픽을 만들었습니다. 이 데이터 그래픽은 두 번째 판의 표지에 실렸습니다. 여기에서는 4년간의 데이터에 걸쳐 유사한 그래픽을 생성합니다. 그렇게 함으로써 가장 빠른 데이터 인터페이스의 성능을 재검토하고 실제로 얼마나 잘 작동하는지 비교합니다.

DuckDB는 데이터를 하나의 큰 파일에 저장하지만 Arrow는 데이터를 각 연도에 대해 별도의 파일로 쓰는 파일 기반 파티셔닝 체계를 사용한다는 점을 상기하십시오. 데이터베이스에 특정 플레이어(피트 알론소)에 대한 특정 연도(2020년)의 모든 데이터를 제공하도록 요청할 때 어떤 일이 발생하는지 생각해 보십시오. DuckDB는 이 데이터를 찾기 위해 전체 큰 파일을 찾아야 하지만 Arrow는 2020년 파일만 찾으면 되며, 이 경우 모든 파일을 합친 크기의 1/4도 되지 않습니다. 그런 다음 해당 파일에서 알론소의 데이터만 찾으면 됩니다. 파일이 더 작기 때문에 관련 데이터를 더 빨리 찾을 수 있습니다.

반대로 4년 동안 알론소의 데이터를 요청한다면 어쨌든 모든 파일을 참조해야 하므로 파일 기반 파티셔닝은 아무 소용이 없습니다.

다음 함수는 특정 타자 및 연도 세트에 필요한 데이터를 가져옵니다.

read_bip_data <- function(tbl, begin, end = begin, batter_id = 624413) {

x <- tbl |> mutate(year = year(game_date)) |> group_by(year) |> filter(type == "X", year >= begin, year <= end) |> select(

year, game_date, batter, launch_speed, launch_angle, estimated_ba_using_speedangle, estimated_woba_using_speedangle

) if (!is.null(batter_id)) { x <- x |> filter(batter == batter_id)

}

} x |>

collect() }

먼저 피트 알론소의 단일 시즌인 2020년 인플레이 타구를 추출하기 위한 계산 성능을 비교합니다.

mark( tbl = nrow(read_bip_data(sc_tbl, 2020)), arrow = nrow(read_bip_data(sc_arrow_part, 2020)), duckdb = nrow(read_bip_data(sc_ddb, 2020)), iterations = 5

) |> arrange(median)

# A tibble: 3 x 6

expression min median `itr/sec` mem_alloc `gc/sec` <bch:expr> <bch:tm> <bch:tm> <dbl> <bch:byt> <dbl>

1 duckdb 91.8ms 95.6ms 9.70 1.07MB 3.88 2 arrow 218.3ms 221ms 4.44 408.14KB 5.33 3 tbl 390.6ms 394.9ms 2.49 280.27MB 0.498

놀랍게도 duckdb와 arrow 객체 모두 tbl 객체보다 뛰어난 성능을 발휘합니다. tbl 객체는 R 메모리에 저장되고 다른 객체는 디스크에 저장된 파일을 읽고 있다는 점을 감안할 때 놀라운 결과입니다. 이 결과는 이렇게 고도로 최적화된 기술이 실제로 얼마나 잘 작동하는지 보여줍니다. 또한 duckdb 객체가 여전히 arrow 객체보다 약 2배 더 빠르지만 arrow 객체는 이전 비교에 비해 성능이 크게 향상되었다는 점도 주목할 가치가 있습니다. 이 경우 특정 연도의 데이터만 쿼리했기 때문에 파일 기반 파티셔닝 체계가 유용했기 때문입니다.

대신 여러 연도에 걸쳐 쿼리하는 경우 이 성능 향상은 사라지고 다시 한 번 duckdb가 arrow보다 몇 배 더 빨라집니다.

x <- tbl |> mutate(year = year(game_date)) |> group_by(year) |> filter(type == "X", year >= begin, year <= end) |> select(

year, game_date, batter, launch_speed, launch_angle, estimated_ba_using_speedangle, estimated_woba_using_speedangle

) if (!is.null(batter_id)) { x <- x |> filter(batter == batter_id)

} x |>

collect() }

mark( tbl = nrow(read_bip_data(sc_tbl, 2021, 2023)), arrow = nrow(read_bip_data(sc_arrow_part, 2021, 2023)), duckdb = nrow(read_bip_data(sc_ddb, 2021, 2023)), iterations = 5

) |> arrange(median)

# A tibble: 3 x 6

expression min median `itr/sec` mem_alloc `gc/sec` <bch:expr> <bch:tm> <bch:tm> <dbl> <bch:byt> <dbl>

- 1 duckdb 89.7ms 100ms 9.35 983KB 3.74
- 2 arrow 398.7ms 407ms 2.39 375KB 2.87
- 3 tbl 548.1ms 552ms 1.76 520MB 0.352

먼저 피트 알론소의 단일 시즌인 2020년 인플레이 타구를 추출하기 위한 계산 성능을 비교합니다.

mark( tbl = nrow(read_bip_data(sc_tbl, 2020)), arrow = nrow(read_bip_data(sc_arrow_part, 2020)), duckdb = nrow(read_bip_data(sc_ddb, 2020)), iterations = 5

) |> arrange(median)

12.5.1 시간에 따른 발사 각도

이제 플롯을 만들고 wOBA가 발사 각도 및 타구 속도와 어떤 관계가 있는지 비교할 준비가 되었지만 이번에는 시간에 따른 변화를 살펴봅니다. 두 번째 판에서 했던 것처럼 플롯에 몇 가지 유용한 가이드라인을 추가합니다.

guidelines <- tibble( launch_angle = c(10, 25, 50), launch_speed = 40, label = c("Ground balls", "Line drives", "Flyballs")

)

# A tibble: 3 x 6

expression min median `itr/sec` mem_alloc `gc/sec` <bch:expr> <bch:tm> <bch:tm> <dbl> <bch:byt> <dbl>

1 duckdb 91.8ms 95.6ms 9.70 1.07MB 3.88 2 arrow 218.3ms 221ms 4.44 408.14KB 5.33 3 tbl 390.6ms 394.9ms 2.49 280.27MB 0.498

놀랍게도 duckdb와 arrow 객체 모두 tbl 객체보다 뛰어난 성능을 발휘합니다. tbl 객체는 R 메모리에 저장되고 다른 객체는 디스크에 저장된 파일을 읽고 있다는 점을 감안할 때 놀라운 결과입니다. 이 결과는 이렇게 고도로 최적화된 기술이 실제로 얼마나 잘 작동하는지 보여줍니다. 또한 duckdb 객체가 여전히 arrow 객체보다 약 2배 더 빠르지만 arrow 객체는 이전 비교에 비해 성능이 크게 향상되었다는 점도 주목할 가치가 있습니다. 이 경우 특정 연도의 데이터만 쿼리했기 때문에 파일 기반 파티셔닝 체계가 유용했기 때문입니다.

대신 여러 연도에 걸쳐 쿼리하는 경우 이 성능 향상은 사라지고 다시 한 번 duckdb가 arrow보다 몇 배 더 빨라집니다.

duckdb가 가장 성능이 좋은 것으로 입증되었으므로 이를 사용하여 데이터를 가져와 플롯을 그립니다. 데이터를 모두 플로팅하지 않으려면 slice_sample() 함수를 사용합니다.

ev_plot <- sc_ddb |> read_bip_data(2020, 2023, batter_id = NULL) |> # for speed slice_sample(prop = 0.2) |> ggplot(

aes(

- x = launch_speed,
- y = launch_angle, color = estimated_woba_using_speedangle

) ) + geom_hline(

data = guidelines, aes(yintercept = launch_angle), color = "black", linetype = 2

) + geom_text(

data = guidelines, aes(label = label, y = launch_angle - 4), color = "black", hjust = "left"

) + geom_point(alpha = 0.05) + scale_color_viridis_c("BA") +

- scale_x_continuous( "Exit velocity (mph)", limits = c(40, 120)

) +

- scale_y_continuous( "Launch angle (degrees)", breaks = seq(-75, 75, 25)

) + facet_wrap(vars(year))

그림 12.1에서 몇 가지 관찰 결과를 확인할 수 있습니다. 첫째, 라인 드라이브는 안타가 될 확률이 매우 높지만 그 가능성은 공이 배트에서 얼마나 세게 떨어지는지와 얼마나 높이 가는지에 따라 다릅니다. 둘째, 거의 모든 타구가 안타가 되는 "스위트 스팟"이 있습니다. 이들은 발사 각도 약 25도와 타구 속도 100mph 이상을 중심으로 흰색 주머니를 형성합니다. 나중에 살펴보겠지만 이는 종종 홈런입니다. 타자들이 이러한 특성을 가진 타구를 만들기 위해 스윙을 최적화하고 있다는 주장입니다.

ev_plot + guides(color = guide_colorbar(title = "wOBA"))

여러 해에 걸쳐 차이점이 보이십니까? 팬데믹으로 인해 시즌이 단축되어 2020년에 데이터가 적다는 사실 외에는 관계가 거의 동일한 것으로 나타납니다.

12.6 추가 읽을거리

abdwr3edata 패키지에는 이 장에 표시된 모든 함수가 포함되어 있습니다. 이 책의 11장에서는 파크 팩터를 탐색하기 위해 MySQL 데이터베이스를 사용하는 방법을 다룹니다. 11.2절에서는 MySQL 서버를 설정하고 사용하는 방법을 설명하고 11.6절에서는 자신만의 야구 데이터베이스를 구축하는 방법을 설명합니다.

Wickham, Cetinkaya-Rundel 및 Grolemund(2023)의 21장에서는 데이터베이스를 다루고 duckdb 및 기타

) + geom_text(

data = guidelines, aes(label = label, y = launch_angle - 4), color = "black", hjust = "left"

) + geom_point(alpha = 0.05) + scale_color_viridis_c("BA") +

- scale_x_continuous( "Exit velocity (mph)", limits = c(40, 120)

) +

- scale_y_continuous( "Launch angle (degrees)", breaks = seq(-75, 75, 25)

) + facet_wrap(vars(year))

그림 12.1에서 몇 가지 관찰 결과를 확인할 수 있습니다. 첫째, 라인 드라이브는 안타가 될 확률이 매우 높지만 그 가능성은 공이 배트에서 얼마나 세게 떨어지는지와 얼마나 높이 가는지에 따라 다릅니다. 둘째, 거의 모든 타구가 안타가 되는 "스위트 스팟"이 있습니다. 이들은 발사 각도 약 25도와 타구 속도 100mph 이상을 중심으로 흰색 주머니를 형성합니다. 나중에 살펴보겠지만 이는 종종 홈런입니다. 타자들이 이러한 특성을 가진 타구를 만들기 위해 스윙을 최적화하고 있다는 주장입니다.

ev_plot + guides(color = guide_colorbar(title = "wOBA"))

여러 해에 걸쳐 차이점이 보이십니까? 팬데믹으로 인해 시즌이 단축되어 2020년에 데이터가 적다는 사실 외에는 관계가 거의 동일한 것으로 나타납니다.

- 12.6 추가 읽을거리

abdwr3edata 패키지에는 이 장에 표시된 모든 함수가 포함되어 있습니다. 이 책의 11장에서는 파크 팩터를 탐색하기 위해 MySQL 데이터베이스를 사용하는 방법을 다룹니다. 11.2절에서는 MySQL 서버를 설정하고 사용하는 방법을 설명하고 11.6절에서는 자신만의 야구 데이터베이스를 구축하는 방법을 설명합니다.

Wickham, Cetinkaya-Rundel 및 Grolemund(2023)의 21장에서는 데이터베이스를 다루고 duckdb 및 기타

추가 읽을거리 301

![image 85](images/imageFile85.png)

그림 12.1 발사 각도 및 타구 속도의 함수로서 추정된 wOBA, 2020-2023.

DBI 호환 데이터베이스에 대한 dplyr 인터페이스를 설명합니다. 22장에서는 Arrow와 Parquet에 대해 자세히 설명합니다.

12.7 연습 문제

- 1. 빠른 공에 대한 홈런

이 장에서 구성한 데이터베이스를 사용하여 2020-2023 4시즌 동안 100mph를 초과하는 패스트볼에서 나온 홈런의 총수를 구하십시오. 각 시즌에 발생한 홈런 총수 중 이러한 패스트볼에서 나온 홈런의 비율은 얼마입니까?

- 2. 구속별 도루 비율

이 장에서 구성한 데이터베이스를 사용하여 1mph 단위로 반올림하여 모든 구속에 대한 도루 성공률을 계산하십시오. 2루 도루와 3루 도루에 따라 분석을 구분하십시오. 구속이 도루 성공률과 상관 관계가 있는 것으로 나타납니까?

DBI 호환 데이터베이스에 대한 dplyr 인터페이스를 설명합니다. 22장에서는 Arrow와 Parquet에 대해 자세히 설명합니다.

- 12.7 연습 문제

- 1. 빠른 공에 대한 홈런

이 장에서 구성한 데이터베이스를 사용하여 2020-2023 4시즌 동안 100mph를 초과하는 패스트볼에서 나온 홈런의 총수를 구하십시오. 각 시즌에 발생한 홈런 총수 중 이러한 패스트볼에서 나온 홈런의 비율은 얼마입니까?

- 2. 구속별 도루 비율

이 장에서 구성한 데이터베이스를 사용하여 1mph 단위로 반올림하여 모든 구속에 대한 도루 성공률을 계산하십시오. 2루 도루와 3루 도루에 따라 분석을 구분하십시오. 구속이 도루 성공률과 상관 관계가 있는 것으로 나타납니까?
