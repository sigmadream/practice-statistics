11

데이터베이스를 사용하여 파크 팩터 계산하기

- 11.1 소개

이 책의 앞부분에서는 전적으로 R에 로드된 야구 데이터셋을 사용하여 분석을 수행했습니다. 이는 비교적 적은 수의 행을 가진 데이터셋을 다루었기 때문에 가능했습니다. 그러나 여러 시즌의 플레이 바이 플레이(또는 피치 바이 피치) 데이터를 분석하고자 할 때는 모든 데이터를 R 내부에서 관리하기가 더 어려워집니다.1 레트로시트(Retrosheet) 게임 로그는 약 25만 개의 기록으로 구성되지만, 레트로시트 플레이 바이 플레이 이벤트는 1,000만 개가 넘고, 스탯캐스트(Statcast)는 MLB 경기에 대해 연간 약 80만 개의 투구 데이터를 제공합니다.

이러한 빅데이터 문제의 해결책은 관계형 데이터베이스 관리 시스템(RDBMS)에 데이터를 저장하고, 이를 R과 연결하여 특정 분석에 필요한 데이터만 가져오는 것입니다. 이 장에서는 이러한 접근 방식에 대한 지침을 제공합니다. RDBMS로는 MySQL의 포크이자 인기 있는 오픈소스 RDBMS2인 MariaDB를 선택했습니다. 그러나 다른 소프트웨어(PostgreSQL, SQLite)에 익숙한 독자라면 선호하는 RDBMS에서 비슷한 해결책을 찾을 수 있습니다. 12장에서 대체 빅데이터 전략을 비교합니다.

여기에서는 MySQL을 사용하여 야구의 구장 효과를 이해해 보겠습니다. 다른 대부분의 팀 스포츠와 달리, 야구는 경기장의 크기와 형태가 구장마다 크게 다릅니다. 펜웨이 파크(보스턴 레드삭스의 홈구장)의 좌측 외야 펜스는 홈 플레이트에서 310피트 거리에 있는 것으로 기록되어 있는 반면, 리글리 필드(시카고 컵스의 홈구장)의 좌측 외야 펜스는 355피트 거리에 있습니다. 그린 몬스터로 잘 알려진 보스턴의 좌측 외야 펜스는 37피트 높이인 반면, 로스앤젤레스 다저 스타디움의 좌측 외야 펜스는 4피트에 불과합니다. 이러한 구장 형태와 크기의 차이, 그리고

- 1기본적으로 R은 데이터를 메모리(RAM)로 읽어 들이므로, 읽을 수 있는 데이터셋의 크기에 제한이 있습니다.
- 2MariaDB는 MySQL을 대체할 수 있도록 설계되었습니다. 실제로 MariaDB 애플리케이션은 mysql이라고 불립니다. 어떤 경우에는 두 용어를 혼용해서 사용할 수도 있습니다.

###### DOI: 10.1201/9781032668239-11 257

지역 날씨 조건은 경기와 관련된 선수 성적 지표에 큰 영향을 미칩니다.

먼저 MariaDB 서버를 구하고 설정하는 방법을 살펴본 후, 데이터를 삽입하고 검색하기 위해 R을 데이터베이스에 연결하는 방법을 설명합니다. 이 인터페이스를 사용하여 쿠어스 필드(콜로라도 로키스의 홈구장)가 득점에 미치는 영향에 대한 증거를 제시합니다. 또한 MySQL로 가져올 준비가 된 야구 데이터(시즌별 및 피치 바이 피치 형태)를 제공하는 온라인 리소스를 소개합니다. 마지막으로 독자들에게 구장 요인을 계산하고 이를 사용하여 선수의 통계에 적절한 조정을 가하는 기본적인 방법을 제공하며 이 장을 마무리합니다.

- 11.2 MySQL 설치 및 데이터베이스 생성

이 책은 R에 초점을 맞추고 있으므로, R과 함께 MariaDB를 사용하는 것을 강조합니다. 사용자는 https://mari adb.com/downloads/ 에서 직접 MariaDB 서버를 설치할 수 있습니다. 이 책의 제3판에서는 거의 완벽하게 호환되는 MySQL의 오픈소스 포크인 MariaDB의 사용을 시연할 것입니다. 다소 혼란스럽게도 MariaDB는 MySQL이라는 이름의 서버를 설치합니다. 독자들은 MariaDB나 MySQL 중 무엇을 사용하든 이 지침을 따를 수 있을 것입니다.3

많은 사람들이 이미 자신의 컴퓨터에 MySQL 서버를 실행하고 있습니다. 이를 확인하는 쉬운 방법은 터미널을 사용하여 실행 중인 프로세스를 확인하는 것입니다.

ps -ax | grep "mariadb"

404058 ? SNsl 2:04 /usr/sbin/mariadbd 490520 pts/0 SN+ 0:00 sh -c 'ps' -ax | grep 'mariadb' 490522 pts/0 SN+ 0:00 grep mariadb

여기서 /usr/sbin/mariadbd 프로세스가 MySQL 서버입니다.

MySQL 서버가 실행되면 이에 접속하여 새 데이터베이스를 생성합니다. 명령줄 MySQL 클라이언트를 사용하여 이를 달성하는 방법을 보여줍니다. 이 경우 root 사용자로 MySQL에 로그인하고 해당 비밀번호를 입력합니다. 이 장의 작업을 복제하려면 새 사용자와 데이터베이스를 생성할 수 있는 충분한 권한을 가진 서버 계정에 액세스해야 합니다. 문제가 발생하면 MariaDB 문서를 참조하시기 바랍니다.

3Benjamin S. Baumer, Kaplan, and Horton (2021b)의 부록 F (https://mdsr-book.github.io/mdsr3e/F-dbsetup.html 참조)에서도 SQL 서버 설정에 대한 단계별 지침을 제공합니다.

지역 날씨 조건은 경기와 관련된 선수 성적 지표에 큰 영향을 미칩니다.

먼저 MariaDB 서버를 구하고 설정하는 방법을 살펴본 후, 데이터를 삽입하고 검색하기 위해 R을 데이터베이스에 연결하는 방법을 설명합니다. 이 인터페이스를 사용하여 쿠어스 필드(콜로라도 로키스의 홈구장)가 득점에 미치는 영향에 대한 증거를 제시합니다. 또한 MySQL로 가져올 준비가 된 야구 데이터(시즌별 및 피치 바이 피치 형태)를 제공하는 온라인 리소스를 소개합니다. 마지막으로 독자들에게 구장 요인을 계산하고 이를 사용하여 선수의 통계에 적절한 조정을 가하는 기본적인 방법을 제공하며 이 장을 마무리합니다.

###### 11.2 MySQL 설치 및 데이터베이스 생성

이 책은 R에 초점을 맞추고 있으므로, R과 함께 MariaDB를 사용하는 것을 강조합니다. 사용자는 https://mari adb.com/downloads/ 에서 직접 MariaDB 서버를 설치할 수 있습니다. 이 책의 제3판에서는 거의 완벽하게 호환되는 MySQL의 오픈소스 포크인 MariaDB의 사용을 시연할 것입니다. 다소 혼란스럽게도 MariaDB는 MySQL이라는 이름의 서버를 설치합니다. 독자들은 MariaDB나 MySQL 중 무엇을 사용하든 이 지침을 따를 수 있을 것입니다.3

많은 사람들이 이미 자신의 컴퓨터에 MySQL 서버를 실행하고 있습니다. 이를 확인하는 쉬운 방법은 터미널을 사용하여 실행 중인 프로세스를 확인하는 것입니다.

ps -ax | grep "mariadb"

404058 ? SNsl 2:04 /usr/sbin/mariadbd 490520 pts/0 SN+ 0:00 sh -c 'ps' -ax | grep 'mariadb' 490522 pts/0 SN+ 0:00 grep mariadb

여기서 /usr/sbin/mariadbd 프로세스가 MySQL 서버입니다.

MySQL 서버가 실행되면 이에 접속하여 새 데이터베이스를 생성합니다. 명령줄 MySQL 클라이언트를 사용하여 이를 달성하는 방법을 보여줍니다. 이 경우 root 사용자로 MySQL에 로그인하고 해당 비밀번호를 입력합니다. 이 장의 작업을 복제하려면 새 사용자와 데이터베이스를 생성할 수 있는 충분한 권한을 가진 서버 계정에 액세스해야 합니다. 문제가 발생하면 MariaDB 문서를 참조하시기 바랍니다.

3Benjamin S. Baumer, Kaplan, and Horton (2021b)의 부록 F (https://mdsr-book.github.io/mdsr3e/F-dbsetup.html 참조)에서도 SQL 서버 설정에 대한 단계별 지침을 제공합니다.

MySQL 설치 및 데이터베이스 생성 259 문제가 발생할 경우.

mysql -u root -p

MySQL에 들어가면 다음 명령어를 사용하여 abdwr라는 새 데이터베이스를 생성할 수 있습니다.

CREATE DATABASE abdwr;

마찬가지로 spahn이라는 비밀번호를 사용하는 새 사용자 abdwr를 생성합니다.

CREATE USER 'abdwr'@'localhost' IDENTIFIED BY 'spahn';

다음으로, abdwr 사용자에게 abdwr 데이터베이스의 모든 권한을 부여하고, 서버가 권한 데이터베이스를 다시 불러오도록 강제합니다.

GRANT ALL ON abdwr.\* TO 'abdwr'@'localhost' WITH GRANT OPTION; FLUSH PRIVILEGES;

- 11.2.1 옵션 파일 설정

이전 섹션에서는 MariaDB 서버에 abdwr라는 사용자를 생성하고 그 사용자에게 spahn이라는 비밀번호를 부여했습니다. 일반적으로 일반 텍스트로 비밀번호를 노출하는 것은 좋은 생각이 아니며, 이제 MariaDB/MySQL 데이터베이스 자격 증명을 저장하기 위해 옵션 파일을 사용하는 방법을 설명하겠습니다.

옵션 파일은 데이터베이스 연결 옵션이 포함된 일반 텍스트 파일입니다. 이 파일이 ~/.my.cnf에 저장되어 있으면 MariaDB/MySQL 데이터베이스에 로그인하려고 할 때 비밀번호(또는 다른 연결 매개변수)를 입력할 필요 없이 자동으로 읽어 들입니다.

이 책에서 필요한 옵션 파일은 다음과 같습니다.

cat ~/.my.cnf

[abdwr] database="abdwr" user="abdwr" password="spahn"

이 옵션 파일을 사용하면 다음 명령만으로 명령줄을 통해 MariaDB 서버에 연결할 수 있습니다.

mysql

###### 11.3 R과 MySQL 연결

R에서 SQL 데이터베이스로의 연결은 여러 다른 RDBMS에 대한 공통 인터페이스를 제공하는 DBI 패키지를 통해 잘 관리됩니다. DBI를 준수하는 많은 패키지들이 다양한 데이터베이스로의 직접 연결을 제공합니다. 예를 들어, RMariaDB 패키지는 MariaDB 데이터베이스에 연결하고, RPostgreSQL 패키지는 PostgreSQL 데이터베이스에 연결하며, RSQLite 패키지는 SQLite 데이터베이스에 연결하고, odbc 패키지는 ODBC를 지원하는 모든 데이터베이스에 연결합니다. 라이선스의 미묘한 변경으로 인해 이 책의 제2판 이후 MySQL 서버에 연결하는 데이터베이스 도구의 개발이 RMySQL에서 RMariaDB 패키지로 이동했습니다. R에 선호하는 데이터베이스를 연결하는 것에 대한 최신 정보는 https://solutions.posit.co/connections/db/ 를 참조하시기 바랍니다.

RMariaDB 패키지는 R 사용자에게 MySQL 데이터베이스에 연결하는 함수를 제공합니다. MySQL 연결을 광범위하게 사용할 계획인 독자들은 RMariaDB를 설치하기 위해 필요한 노력을 기울일 것을 권장합니다.

- 11.3.1 RMariaDB를 사용한 연결

DBI 함수 dbConnect()는 데이터베이스 서버에 대한 연결을 생성하고 연결 정보를 저장하는 객체를 반환합니다.

비밀번호 보안에 신경 쓰지 않는다면 다음 코드와 같이 매개변수를 명시적으로 지정할 수 있습니다. user 및 password 인수는 MySQL 데이터베이스에 액세스하기 위한 사용자 이름과 비밀번호를 나타내며(데이터베이스 생성 시 지정된 경우), dbname은 R이 연결될 기본 데이터베이스(섹션 11.2에서 생성한 abdwr)를 나타냅니다.

library(RMariaDB)
con <- dbConnect(
MariaDB(), dbname = "abdwr", user = "abdwr", password = "spahn"
)

또는 섹션 11.2.1에서 설명한 옵션 파일과 함께 group 인수를 사용하여 연결할 수 있습니다.

library(RMariaDB)
con <- dbConnect(MariaDB(), group = "abdwr")

여러 함수에서 인수로 필요하므로 연결이 R 객체(con)에 할당된다는 점에 유의하시기 바랍니다.

###### 11.3 R과 MySQL 연결

R에서 SQL 데이터베이스로의 연결은 여러 다른 RDBMS에 대한 공통 인터페이스를 제공하는 DBI 패키지를 통해 잘 관리됩니다. DBI를 준수하는 많은 패키지들이 다양한 데이터베이스로의 직접 연결을 제공합니다. 예를 들어, RMariaDB 패키지는 MariaDB 데이터베이스에 연결하고, RPostgreSQL 패키지는 PostgreSQL 데이터베이스에 연결하며, RSQLite 패키지는 SQLite 데이터베이스에 연결하고, odbc 패키지는 ODBC를 지원하는 모든 데이터베이스에 연결합니다. 라이선스의 미묘한 변경으로 인해 이 책의 제2판 이후 MySQL 서버에 연결하는 데이터베이스 도구의 개발이 RMySQL에서 RMariaDB 패키지로 이동했습니다. R에 선호하는 데이터베이스를 연결하는 것에 대한 최신 정보는 https://solutions.posit.co/connections/db/ 를 참조하시기 바랍니다.

RMariaDB 패키지는 R 사용자에게 MySQL 데이터베이스에 연결하는 함수를 제공합니다. MySQL 연결을 광범위하게 사용할 계획인 독자들은 RMariaDB를 설치하기 위해 필요한 노력을 기울일 것을 권장합니다.

- 11.3.1 RMariaDB를 사용한 연결

DBI 함수 dbConnect()는 데이터베이스 서버에 대한 연결을 생성하고 연결 정보를 저장하는 객체를 반환합니다.

비밀번호 보안에 신경 쓰지 않는다면 다음 코드와 같이 매개변수를 명시적으로 지정할 수 있습니다. user 및 password 인수는 MySQL 데이터베이스에 액세스하기 위한 사용자 이름과 비밀번호를 나타내며(데이터베이스 생성 시 지정된 경우), dbname은 R이 연결될 기본 데이터베이스(섹션 11.2에서 생성한 abdwr)를 나타냅니다.

library(RMariaDB)
con <- dbConnect(
MariaDB(), dbname = "abdwr", user = "abdwr", password = "spahn"
)

또는 섹션 11.2.1에서 설명한 옵션 파일과 함께 group 인수를 사용하여 연결할 수 있습니다.

library(RMariaDB)
con <- dbConnect(MariaDB(), group = "abdwr")

여러 함수에서 인수로 필요하므로 연결이 R 객체(con)에 할당된다는 점에 유의하시기 바랍니다.

class(con)

[1] "MariaDBConnection"
attr(,"package")
[1] "RMariaDB"

연결을 해제하려면 dbDisconnect()를 사용합니다.

- 11.3.2 다른 SQL 백엔드와 R 연결

다른 SQL 백엔드와 R을 연결하는 과정은 위에서 설명한 MySQL의 경우와 매우 유사합니다. 연결은 모두 DBI에 의해 관리되므로 데이터베이스 백엔드만 변경하면 됩니다. 예를 들어 MySQL 대신 PostgreSQL 서버에 연결하려면 RMariaDB 대신 RPostgreSQL 패키지를 로드하고 dbConnect() 호출에서 MariaDB() 함수 대신 PostgreSQL() 함수를 사용합니다. 나머지 과정은 동일합니다. 결과적으로 생성된 PostgreSQL 연결은 아래에서 MySQL 연결이 사용되는 것과 동일한 방식으로 사용할 수 있습니다.

11.4 R에서 MySQL 게임 로그 데이터베이스 채우기

게임 로그 데이터 파일은 현재 레트로시트 웹페이지(https://www.retrosheet.org/gamelogs/index.html)에서 제공됩니다. 단일 연도를 클릭하면 해당 시즌의 게임 로그 텍스트 파일이 포함된 압축(.zip) 파일을 얻을 수 있습니다. 여기서는 먼저 시즌 게임 로그를 R로 로드하는 함수를 만든 다음 데이터를 MySQL 테이블에 추가하는 방법을 보여줍니다. 그런 다음 여러 시즌 동안 이 과정을 반복하여 레트로시트에서 게임 로그를 다운로드하고 MySQL 테이블에 추가합니다.

레트로시트에서 다운로드한 게임 로그 파일에는 열 헤더가 없으므로 이 책의 다른 곳에서 한 것처럼 game_log_header.csv 파일에 저장된 의미 있는 이름이 포함된 결과 데이터 프레임 gl2012가 생성됩니다.

- 11.4.1 레트로시트에서 R로

아래에 표시된 retrosheet_gamelog() 함수는 시즌의 연도를 입력으로 받아 다음 작업을 수행합니다.

- • 열 헤더 파일 가져오기
- • 레트로시트에서 시즌 zip 파일 다운로드
- • 다운로드한 zip 파일에 포함된 텍스트 파일 추출
- • 알려진 열 헤더를 사용하여 텍스트 파일을 R로 읽어들이기
- • 압축 파일과 추출된 파일 모두 제거

###### • 결과 데이터 프레임을 반환합니다.

retrosheet_gamelog <- function(season) {
require(abdwr3edata)
require(fs)
dir <- tempdir()
glheaders <- retro_gl_header
remote <- paste0(
"http://www.retrosheet.org/gamelogs/gl", season, ".zip"
)
local <- path(dir, paste0("gl", season, ".zip"))
download.file(url = remote, destfile = local)
unzip(local, exdir = dir)
local_txt <- gsub(".zip", ".txt", local)
gamelog <- here::here(local_txt) |>
read_csv(col_names = names(glheaders))
file.remove(local)
file.remove(local_txt)
return(gamelog)
}

###### 이 함수를 R로 읽어들인 후, 다음 명령어를 입력하여 한 시즌의 게임 로그(예를 들어, 2012년)를 R로 읽어들일 수 있습니다.

gl2012 <- retrosheet_gamelog(2012)

###### 11.4.2 R에서 MySQL로

다음으로, gl2012 데이터 프레임의 데이터를 abdwr MySQL 데이터베이스로 전송합니다. 이어지는 코드 줄에서 우리는 dbWriteTable() 함수를 사용하여 MySQL 데이터베이스의 테이블(존재할 수도 있고 존재하지 않을 수도 있음)에 데이터를 추가합니다.

다음은 dbWriteTable() 함수의 인수에 대한 몇 가지 참고 사항입니다.

- • conn 인수는 열린 연결이 필요합니다. 여기서는 이전에 정의된(con) 연결이 지정됩니다.
- • name 인수는 데이터가 추가될 테이블(데이터베이스 내)의 이름을 나타내는 문자열이 필요합니다.
- • value 인수는 MySQL 데이터베이스의 테이블에 추가할 R 데이터 프레임의 이름이 필요합니다.
- • append를 TRUE로 설정하면 "gamelogs"라는 이름의 테이블이 이미 존재하는 경우 gl2012의 데이터가 해당 테이블에 추가됩니다. append가 FALSE로 설정되면 "gamelogs" 테이블(존재하는 경우)을 덮어씁니다.

###### • 결과 데이터 프레임을 반환합니다.

retrosheet_gamelog <- function(season) {
require(abdwr3edata)
require(fs)
dir <- tempdir()
glheaders <- retro_gl_header
remote <- paste0(
"http://www.retrosheet.org/gamelogs/gl", season, ".zip"
)
local <- path(dir, paste0("gl", season, ".zip"))
download.file(url = remote, destfile = local)
unzip(local, exdir = dir)
local_txt <- gsub(".zip", ".txt", local)
gamelog <- here::here(local_txt) |>
read_csv(col_names = names(glheaders))
file.remove(local)
file.remove(local_txt)
return(gamelog)
}

이 함수를 R로 읽어들인 후, 다음 명령어를 입력하여 한 시즌의 게임 로그(예를 들어, 2012년)를 R로 읽어들일 수 있습니다.

gl2012 <- retrosheet_gamelog(2012)

- 11.4.2 R에서 MySQL로

다음으로, gl2012 데이터 프레임의 데이터를 abdwr MySQL 데이터베이스로 전송합니다. 이어지는 코드 줄에서 우리는 dbWriteTable() 함수를 사용하여 MySQL 데이터베이스의 테이블(존재할 수도 있고 존재하지 않을 수도 있음)에 데이터를 추가합니다.

다음은 dbWriteTable() 함수의 인수에 대한 몇 가지 참고 사항입니다.

- • conn 인수는 열린 연결이 필요합니다. 여기서는 이전에 정의된(con) 연결이 지정됩니다.
- • name 인수는 데이터가 추가될 테이블(데이터베이스 내)의 이름을 나타내는 문자열이 필요합니다.
- • value 인수는 MySQL 데이터베이스의 테이블에 추가할 R 데이터 프레임의 이름이 필요합니다.
- • append를 TRUE로 설정하면 "gamelogs"라는 이름의 테이블이 이미 존재하는 경우 gl2012의 데이터가 해당 테이블에 추가됩니다. append가 FALSE로 설정되면 "gamelogs" 테이블(존재하는 경우)을 덮어씁니다.

• field.types 인수는 MySQL 열에 대한 데이터 유형의 명명된 벡터를 제공합니다. 이 인수를 비워두면 dbWriteTable()이 최적의 값을 추측하려고 시도합니다. 이 경우, 우리는 여러 연도에 걸쳐 일관성을 유지하고자 특정 변수에 대한 값을 지정하기로 선택했습니다.

if (dbExistsTable(con, "gamelogs")) {
dbRemoveTable(con, "gamelogs")
}
con |>
dbWriteTable(
name = "gamelogs", value = gl2012, append = FALSE,
field.types = c(
CompletionInfo = "varchar(50)",
AdditionalInfo = "varchar(255)",
HomeBatting1Name = "varchar(50)",
HomeBatting2Name = "varchar(50)",
HomeBatting3Name = "varchar(50)",
HomeBatting4Name = "varchar(50)",
HomeBatting5Name = "varchar(50)",
HomeBatting6Name = "varchar(50)",
HomeBatting7Name = "varchar(50)",
HomeBatting8Name = "varchar(50)",
HomeBatting9Name = "varchar(50)",
HomeManagerName = "varchar(50)",
VisitorStartingPitcherName = "varchar(50)",
VisitorBatting1Name = "varchar(50)",
VisitorBatting2Name = "varchar(50)",
VisitorBatting3Name = "varchar(50)",
VisitorBatting4Name = "varchar(50)",
VisitorBatting5Name = "varchar(50)",
VisitorBatting6Name = "varchar(50)",
VisitorBatting7Name = "varchar(50)",
VisitorBatting8Name = "varchar(50)",
VisitorBatting9Name = "varchar(50)",
VisitorManagerName = "varchar(50)",
HomeLineScore = "varchar(30)",
VisitorLineScore = "varchar(30)",
SavingPitcherName = "varchar(50)",
ForfeitInfo = "varchar(10)",
ProtestInfo = "varchar(10)",
UmpireLFID = "varchar(8)",
UmpireRFID = "varchar(8)",
UmpireLFName = "varchar(50)",
UmpireRFName = "varchar(50)"
)
)

이제 데이터가 MySQL 서버에 존재하는지 확인하기 위해 dplyr을 사용하여 쿼리할 수 있습니다.

gamelogs <- con |>
tbl("gamelogs")
head(gamelogs)

# Source: SQL [6 x 161]

# Database: mysql [abdwr@localhost:NA/abdwr]

Date DoubleHeader DayOfWeek VisitingTeam VisitingTeamLeague
<dbl> <dbl> <chr> <chr> <chr>
1 2.01e7 0 Wed SEA AL
2 2.01e7 0 Thu SEA AL
3 2.01e7 0 Wed SLN NL
4 2.01e7 0 Thu TOR AL
5 2.01e7 0 Thu BOS AL
6 2.01e7 0 Thu WAS NL

# i 156 more variables: VisitingTeamGameNumber <dbl>,

# HomeTeam <chr>, HomeTeamLeague <chr>,

# HomeTeamGameNumber <dbl>, VisitorRunsScored <dbl>,

# HomeRunsScore <dbl>, LengthInOuts <dbl>, DayNight <chr>,

# CompletionInfo <chr>, ForfeitInfo <chr>, ProtestInfo <chr>,

# ParkID <chr>, Attendance <dbl>, Duration <dbl>,

# VisitorLineScore <chr>, HomeLineScore <chr>, ...

11.4.1절에서 하나의 시즌 게임 로그를 MySQL 테이블에 추가하는 코드를 제공했습니다. 하지만 이전 장들에서 보여드린 바와 같이 단일 시즌의 게임 로그를 다루기 위해 R을 사용하는 것은 직관적입니다. 관계형 데이터베이스 관리 시스템(RDBMS)에 데이터를 저장할 때의 이점을 충분히 이해하기 위해, 우리는 야구 역사상 과거의 게임 로그들을 MySQL 테이블에 채워넣을 것입니다. 과거 데이터베이스와 R 연결을 통해, 여러 시즌에 걸쳐 분석을 수행하기 위해 R을 활용하는 방법을 시연하겠습니다.

우리는 게임 로그 데이터를 R로 가져오는 앞의 두 단계를 결합한 다음 해당 데이터를 MySQL 테이블로 전송하는 간단한 함수 append_game_logs()를 작성합니다.4 전체 과정은 몇 분 정도 걸릴 수 있습니다. 1871년까지 거슬러 올라가는 파일을 다운로드하는 것에 관심이 없다면, 1995년 이후의 시즌들만으로도 다음 섹션의 예제를 재현하기에 충분합니다.

함수 append_game_logs()는 다음 매개변수를 입력으로 받습니다.

• conn은 데이터베이스에 대한 DBI 연결입니다.

4레트로시트에서 데이터를 다운로드하는 작업은 이전에 제시된 retrosheet_gamelog() 함수에 의해 수행되므로, 독자들은 이 섹션의 코드가 작동하려면 해당 함수가 로드되어 있는지 확인해야 합니다.

이제 데이터가 MySQL 서버에 존재하는지 확인하기 위해 dplyr을 사용하여 쿼리할 수 있습니다.

gamelogs <- con |>
tbl("gamelogs")
head(gamelogs)

# Source: SQL [6 x 161]

# Database: mysql [abdwr@localhost:NA/abdwr]

Date DoubleHeader DayOfWeek VisitingTeam VisitingTeamLeague
<dbl> <dbl> <chr> <chr> <chr>

- 1 2.01e7 0 Wed SEA AL
- 2 2.01e7 0 Thu SEA AL
- 3 2.01e7 0 Wed SLN NL
- 4 2.01e7 0 Thu TOR AL
- 5 2.01e7 0 Thu BOS AL
- 6 2.01e7 0 Thu WAS NL

# i 156 more variables: VisitingTeamGameNumber <dbl>,

# HomeTeam <chr>, HomeTeamLeague <chr>,

# HomeTeamGameNumber <dbl>, VisitorRunsScored <dbl>,

# HomeRunsScore <dbl>, LengthInOuts <dbl>, DayNight <chr>,

# CompletionInfo <chr>, ForfeitInfo <chr>, ProtestInfo <chr>,

# ParkID <chr>, Attendance <dbl>, Duration <dbl>,

# VisitorLineScore <chr>, HomeLineScore <chr>, ...

11.4.1절에서 하나의 시즌 게임 로그를 MySQL 테이블에 추가하는 코드를 제공했습니다. 하지만 이전 장들에서 보여드린 바와 같이 단일 시즌의 게임 로그를 다루기 위해 R을 사용하는 것은 직관적입니다. 관계형 데이터베이스 관리 시스템(RDBMS)에 데이터를 저장할 때의 이점을 충분히 이해하기 위해, 우리는 야구 역사상 과거의 게임 로그들을 MySQL 테이블에 채워넣을 것입니다. 과거 데이터베이스와 R 연결을 통해, 여러 시즌에 걸쳐 분석을 수행하기 위해 R을 활용하는 방법을 시연하겠습니다.

우리는 게임 로그 데이터를 R로 가져오는 앞의 두 단계를 결합한 다음 해당 데이터를 MySQL 테이블로 전송하는 간단한 함수 append_game_logs()를 작성합니다.4 전체 과정은 몇 분 정도 걸릴 수 있습니다. 1871년까지 거슬러 올라가는 파일을 다운로드하는 것에 관심이 없다면, 1995년 이후의 시즌들만으로도 다음 섹션의 예제를 재현하기에 충분합니다.

함수 append_game_logs()는 다음 매개변수를 입력으로 받습니다.

• conn은 데이터베이스에 대한 DBI 연결입니다.

• season은 레트로시트에서 다운로드하여 MySQL 데이터베이스에 추가하고자 하는 시즌을 나타냅니다. 기본적으로 이 함수는 1871년부터 2022년까지의 시즌에 대해 작동합니다.

append_game_logs <- function(conn, season) {
message(paste("Working on", season, "season..."))
one_season <- retrosheet_gamelog(season)
conn |>
dbWriteTable(
name = "gamelogs", value = one_season, append = TRUE
)
}

다음으로, TRUNCATE TABLE SQL 명령어를 사용하여 이전 경기들을 제거하고 map()을 활용해 append_game_logs 과정을 반복하여 테이블을 채웁니다.

dbSendQuery(con, "TRUNCATE TABLE gamelogs;")
map(1995:2017, append_game_logs, conn = con)

이제 수년 치의 게임 로그가 있습니다.

gamelogs |>
group_by(year = str_sub(Date, 1, 4)) |>
summarize(num_games = n())

# Source: SQL [?? x 2]

# Database: mysql [abdwr@localhost:NA/abdwr]

year num_games
<chr> <int64>

- 1 1995 2017
- 2 1996 2267
- 3 1997 2266
- 4 1998 2432
- 5 1999 2428
- 6 2000 2429
- 7 2001 2429
- 8 2002 2426

- 9 2003 2430
- 10 2004 2428

# i more rows

4레트로시트에서 데이터를 다운로드하는 작업은 이전에 제시된 retrosheet_gamelog() 함수에 의해 수행되므로, 독자들은 이 섹션의 코드가 작동하려면 해당 함수가 로드되어 있는지 확인해야 합니다.

###### 11.5 R에서 데이터 쿼리하기

11.5.1 SQL에서 데이터 검색하기

모든 DBI 백엔드는 데이터베이스에서 SQL 쿼리의 결과를 검색하는 dbGetQuery()를 지원합니다. 특정 분석을 위해 R로 데이터를 선택적으로 가져오기 위해 데이터를 MySQL 데이터베이스에 저장하므로, 일반적으로 데이터베이스의 하나 이상의 테이블을 쿼리하여 R로 데이터를 선택적으로 읽어들입니다.

예를 들어, 2006 시즌 이후 두 시카고 팀의 요일별 관중 수를 비교하는 데 관심이 있다고 가정해 보겠습니다. 다음 코드는 원시 데이터를 R로 검색합니다.

query <- "
SELECT date, hometeam, dayofweek, attendance
FROM gamelogs
WHERE Date > 20060101
AND HomeTeam IN ('CHN', 'CHA');
"
chi_attendance <- dbGetQuery(con, query)
slice_head(chi_attendance, n = 6)

      date hometeam dayofweek attendance

- 1 20060402 CHA Sun 38802
- 2 20060404 CHA Tue 37591
- 3 20060405 CHA Wed 33586
- 4 20060407 CHN Fri 40869
- 5 20060408 CHN Sat 40182
- 6 20060409 CHN Sun 39839

dbGetQuery() 함수는 데이터베이스를 쿼리합니다. 인수는 이전에 설정된 연결 핸들(conn)과 유효한 SQL 문(query)으로 구성된 문자열입니다. SQL에 익숙한 독자라면 쿼리의 의미를 이해하는 데 문제가 없을 것입니다. SQL에 익숙하지 않은 독자들을 위해 여기서는 쿼리에 대한 간략한 설명을 제공하며, 언어에 대해 배우고 싶은 사람은 해당 주제를 다루는 수많은 리소스를 찾아볼 것을 권장합니다(https://dev.mysql.com/doc/refman/8.2/en/select.html 참조).

SQL 문의 첫 번째 행은 선택할 테이블의 열(이 경우 date, hometeam, dayofweek, attendance)을 나타냅니다. 두 번째 행은 이들을 어느 테이블에서 검색해야 하는지(gamelogs)를 명시합니다. 마지막으로 where 절은 검색할 행에 대한 조건을 지정합니다. 날짜(date)는 20060101보다 커야 하고 홈팀(hometeam) 값은 CHN이거나 CHA여야 합니다.

###### 11.5 R에서 데이터 쿼리하기

- 11.5.1 SQL에서 데이터 검색하기

모든 DBI 백엔드는 데이터베이스에서 SQL 쿼리의 결과를 검색하는 dbGetQuery()를 지원합니다. 특정 분석을 위해 R로 데이터를 선택적으로 가져오기 위해 데이터를 MySQL 데이터베이스에 저장하므로, 일반적으로 데이터베이스의 하나 이상의 테이블을 쿼리하여 R로 데이터를 선택적으로 읽어들입니다.

예를 들어, 2006 시즌 이후 두 시카고 팀의 요일별 관중 수를 비교하는 데 관심이 있다고 가정해 보겠습니다. 다음 코드는 원시 데이터를 R로 검색합니다.

query <- "
SELECT date, hometeam, dayofweek, attendance
FROM gamelogs
WHERE Date > 20060101
AND HomeTeam IN ('CHN', 'CHA');
"
chi_attendance <- dbGetQuery(con, query)
slice_head(chi_attendance, n = 6)

      date hometeam dayofweek attendance

- 1 20060402 CHA Sun 38802
- 2 20060404 CHA Tue 37591
- 3 20060405 CHA Wed 33586
- 4 20060407 CHN Fri 40869
- 5 20060408 CHN Sat 40182
- 6 20060409 CHN Sun 39839

dbGetQuery() 함수는 데이터베이스를 쿼리합니다. 인수는 이전에 설정된 연결 핸들(conn)과 유효한 SQL 문(query)으로 구성된 문자열입니다. SQL에 익숙한 독자라면 쿼리의 의미를 이해하는 데 문제가 없을 것입니다. SQL에 익숙하지 않은 독자들을 위해 여기서는 쿼리에 대한 간략한 설명을 제공하며, 언어에 대해 배우고 싶은 사람은 해당 주제를 다루는 수많은 리소스를 찾아볼 것을 권장합니다(https://dev.mysql.com/doc/refman/8.2/en/select.html 참조).

SQL 문의 첫 번째 행은 선택할 테이블의 열(이 경우 date, hometeam, dayofweek, attendance)을 나타냅니다. 두 번째 행은 이들을 어느 테이블에서 검색해야 하는지(gamelogs)를 명시합니다. 마지막으로 where 절은 검색할 행에 대한 조건을 지정합니다. 날짜(date)는 20060101보다 커야 하고 홈팀(hometeam) 값은 CHN이거나 CHA여야 합니다.

대안으로, 우리가 앞서 생성한 gamelogs 객체를 통해 dplyr 인터페이스를 사용하여 MySQL에 접근할 수도 있습니다.

gamelogs |>
filter(Date > 20060101, HomeTeam %in% c('CHN', 'CHA')) |>
select(Date, HomeTeam, DayOfWeek, Attendance) |>
head()

# Source: SQL [6 x 4]

# Database: mysql [abdwr@localhost:NA/abdwr]

      Date HomeTeam DayOfWeek Attendance
     <dbl> <chr>    <chr>          <dbl>

1 20060402 CHA Sun 38802
2 20060404 CHA Tue 37591
3 20060405 CHA Wed 33586
4 20060407 CHN Fri 40869
5 20060408 CHN Sat 40182
6 20060409 CHN Sun 39839

dplyr은 심지어 사용자의 dplyr 파이프라인을 유효한 SQL 쿼리로 변환해주는 show_query()라는 함수를 포함하고 있습니다. 우리가 앞서 작성한 SQL 코드와 아래 변환된 SQL 코드 사이의 유사점과 차이점에 주목하시기 바랍니다.

gamelogs |>
filter(Date > 20060101, HomeTeam %in% c('CHN', 'CHA')) |>
select(Date, HomeTeam, DayOfWeek, Attendance) |>
show_query()

<SQL>
SELECT `Date`, `HomeTeam`, `DayOfWeek`, `Attendance`
FROM `gamelogs`
WHERE (`Date` > 20060101.0) AND (`HomeTeam` IN ('CHN', 'CHA'))

11.5.2 데이터 정제

이러한 데이터를 도식화하기 전에 두 가지를 정제해야 합니다. 먼저 lubridate 패키지의 ymd() 함수를 사용하여 날짜를 인코딩하는 숫자를 R의 Date 필드로 변환합니다. 다음으로 na_if() 함수를 사용하여 관중 수가 0으로 보고된 게임의 관중 수를 NA로 설정합니다.5

5단일 입장 더블헤더(즉, 같은 날 두 경기가 치러지고 두 경기를 모두 관람하는 데 단일 티켓이 필요한 경우)의 경우 관중 수는 두 번째 경기에서만 보고되며 첫 번째 경기에서는 0으로 설정됩니다.

![image 83](images/imageFile83.png)

그림 11.1 컵스(CHN)와 화이트삭스(CHA)가 홈에서 치른 경기의 요일별 관중 수 비교.

chi_attendance <- chi_attendance |>
mutate(
the_date = ymd(date),
attendance = na_if(attendance, 0)
)

그림 11.1에서 우리는 두 시카고 구장의 관중 수를 그래픽으로 비교하여 보여줍니다. geom_smooth()가 작동하려면 가로축 변수가 숫자형이어야 합니다. 따라서 lubridate의 wday() 함수를 사용하여 날짜로부터 요일(숫자로)을 계산합니다. 축 레이블을 약어로 표시하려면 wday()를 다시 사용해야 하지만, label 인수를 TRUE로 설정해야 합니다.

ggplot(
chi_attendance,
aes(
x = wday(the_date), y = attendance, color = hometeam
)
) +
geom_jitter(height = 0, width = 0.2, alpha = 0.2) +
geom_smooth() +
scale_y_continuous("Attendance") +
scale_x_continuous(
"Day of the Week",
breaks = 1:7, labels = wday(1:7, label = TRUE)
) +
scale_color_manual(values = crc_fc)

우리는 주말에 양 팀 모두 더 많은 관중을 동원하는 가운데, 컵스가 일반적으로 화이트삭스보다 더 많은 팬을 동원한다는 점에 주목합니다.

그림 11.1 컵스(CHN)와 화이트삭스(CHA)가 홈에서 치른 경기의 요일별 관중 수 비교.

chi_attendance <- chi_attendance |>
mutate(
the_date = ymd(date),
attendance = na_if(attendance, 0)
)

그림 11.1에서 우리는 두 시카고 구장의 관중 수를 그래픽으로 비교하여 보여줍니다. geom_smooth()가 작동하려면 가로축 변수가 숫자형이어야 합니다. 따라서 lubridate의 wday() 함수를 사용하여 날짜로부터 요일(숫자로)을 계산합니다. 축 레이블을 약어로 표시하려면 wday()를 다시 사용해야 하지만, label 인수를 TRUE로 설정해야 합니다.

ggplot(
chi_attendance,
aes(
x = wday(the_date), y = attendance, color = hometeam
)
) +
geom_jitter(height = 0, width = 0.2, alpha = 0.2) +
geom_smooth() +
scale_y_continuous("Attendance") +

11.5.3 쿠어스 필드와 득점

여러 해에 걸친 데이터에 접근하는 예로써, 덴버에 위치한 콜로라도 로키스의 홈구장 쿠어스 필드가 수년 동안 득점에 미친 영향을 살펴봅니다. 쿠어스 필드는 해발 약 1마일 고도에 위치해 있기 때문에 독특한 구장입니다. 공기가 다른 경기장보다 희박해서 타구가 더 멀리 날아가고 커브볼이 덜 꺾입니다. Lopez, Matthews, and Baumer (2018)는 쿠어스 필드가 야구 전체를 통틀어 큰 홈구장 이점을 제공한다고 추정합니다.

우리는 SQL 쿼리와 dbGetQuery() 함수를 사용하여 1995년(쿠어스 필드로 이전한 해) 이후 로키스가 홈 또는 원정에서 치른 경기에 대한 데이터를 검색합니다.6

query <- "
SELECT date, parkid, visitingteam, hometeam,
visitorrunsscored AS awR, homerunsscore AS hmR
FROM gamelogs
WHERE (HomeTeam = 'COL' OR VisitingTeam = 'COL')
AND Date > 19950000;
"
rockies_games <- dbGetQuery(con, query)

경기 데이터는 rockies_games 데이터 프레임에 편리하게 저장됩니다. 홈팀과 원정팀이 득점한 점수를 더하여 각 경기에서 기록된 총 득점을 계산합니다. 또한 해당 경기가 쿠어스 필드에서 열렸는지 여부를 나타내는 coors라는 새 열을 추가합니다.7

rockies_games <- rockies_games |>
mutate(
runs = awR + hmR,

- 6SQL에서 AS 키워드는 열에 다른 이름을 할당하는 목적으로 사용됩니다. 따라서 visitorrunsscored AS awR은 쿼리가 반환하는 결과에서 visitorrunsscored 열이 awR로 명명될 것임을 SQL에 알려줍니다.
- 7쿠어스 필드의 레트로시트 코드는 DEN02입니다. 모든 구장 코드 목록은 https://www.retrosheet.org/parkcode.txt 에서 확인할 수 있습니다.

![image 84](images/imageFile84.png)

그림 11.2 로키스와 상대 팀이 쿠어스 필드와 다른 구장에서 기록한 득점 비교.

    coors = parkid == "DEN02"

)

그림 11.2에서 우리는 쿠어스와 다른 구장에서 로키스와 상대 팀이 만들어낸 공격적 성과를 그래픽으로 비교합니다.

ggplot(
rockies_games,
aes(x = year(ymd(date)), y = runs, color = coors)
) +
stat_summary(fun.data = "mean_cl_boot") +
xlab("Season") +
ylab("Runs per game (both teams combined)") +
scale_color_manual(
name = "Location",
values = crc_fc, labels = c("Other", "Coors Field")
)

우리는 x의 각 고유한 값에서 y 값을 요약하기 위해 stats_summary() 레이어를 사용합니다. fun.data 인수를 사용하면 사용자가 요약 함수를 지정할 수 있으며, 이 경우 mean_cl_boot()는 모집단 평균에 대한 신뢰 구간을 얻기 위한 비모수적 부트스트랩 절차를 구현합니다. 이 레이어에서 생성된 출력은 각 데이터 포인트를 나타내는 수직 막대입니다.

그림 11.2 로키스와 상대 팀이 쿠어스 필드와 다른 구장에서 기록한 득점 비교.

    coors = parkid == "DEN02"

)

그림 11.2에서 우리는 쿠어스와 다른 구장에서 로키스와 상대 팀이 만들어낸 공격적 성과를 그래픽으로 비교합니다.

ggplot(
rockies_games,
aes(x = year(ymd(date)), y = runs, color = coors)
) +
stat_summary(fun.data = "mean_cl_boot") +
xlab("Season") +
ylab("Runs per game (both teams combined)") +
scale_color_manual(
name = "Location",
values = crc_fc, labels = c("Other", "Coors Field")
)

우리는 x의 각 고유한 값에서 y 값을 요약하기 위해 stats_summary() 레이어를 사용합니다. fun.data 인수를 사용하면 사용자가 요약 함수를 지정할 수 있으며, 이 경우 mean_cl_boot()는 모집단 평균에 대한 신뢰 구간을 얻기 위한 비모수적 부트스트랩 절차를 구현합니다. 이 레이어에서 생성된 출력은 각 데이터 포인트를 나타내는 수직 막대입니다.

나만의 야구 데이터베이스 구축하기 271 scale_linetype_discrete() 레이어는 시리즈에 레이블을 지정하고(name 인수) 범례에 이름을 할당(labels)하는 데 사용됩니다.

그림 11.2에서 쿠어스 필드가 어떻게 타자 친화적인 구장으로서 한 시즌 동안 경기당 최대 6점까지 득점을 끌어올렸는지 알 수 있습니다. 그러나 콜로라도 구장의 효과는 2000년대 들어 다소 감소하여 2006-2011년 기간에는 더 작은 차이를 보입니다. 쿠어스가 덜 극단적인 구장이 된 한 가지 이유는 휴미더(습도 조절기)의 설치입니다. 2002 시즌부터 비정상적인 자연 대기 조건을 상쇄할 목적으로 각 경기 전에 야구공을 습도가 높은 방에 보관해 왔습니다.8

###### 11.6 나만의 야구 데이터베이스 구축하기

11.4.1절에서는 레트로시트 게임 로그 테이블을 생성하여 R 내에서 MySQL 데이터베이스를 채우는 방법을 설명했습니다. 야구 데이터로 데이터베이스를 생성하고 채우기 위한 여러 소위 SQL 덤프들이 온라인에 제공되고 있습니다. SQL 덤프는 SQL 테이블을 생성하고 채우기 위한 SQL 명령어를 포함하는 단순한 텍스트 파일(.sql 확장자)입니다.

- 11.6.1 라만(Lahman) 데이터베이스

션 라만(Sean Lahman)은 여러 형식으로 자신의 시즌 통계 과거 데이터베이스를 제공합니다. 이러한 데이터를 메모리에서 사용할 수 있게 해주는 R용 Lahman 패키지도 있습니다. 하지만 데이터베이스는 SQL 덤프 형태로도 제공되며, http://seanlahman.com/download-baseballdatabase/ 에서 다운로드할 수 있습니다 (2019 - MySQL 버전을 찾으세요). 아쉽게도 MySQL 덤프 파일은 더 이상 지원되지 않지만 이전 버전은 여전히 이용 가능합니다.

이 파일을 SQL로 가져오는 데는 여러 옵션이 있습니다. 이러한 모든 프로세스를 설명하는 것은 이 책의 범위를 벗어나므로, 여기서는 터미널에서 원하는 결과를 얻기 위한 명령어를 제공합니다. 다음 코드가 작동하려면 MySQL 서비스가 실행 중이어야 합니다9 (11.2절 참조).

mysql -u username -p lahmansbaseballdb < lahman-mysql-dump.sql

SQL 덤프 파일이 lahmansbaseballdb라는 새 데이터베이스를 생성한다는 점에 유의하시기 바랍니다.

8휴미더 효과에 대한 자세한 분석은 Nathan (2011)을 참조하세요. 9또한, .sql 파일이 포함된 디렉토리와 사용자 이름, 그리고

비밀번호를 적절히 변경하시기 바랍니다.

- 11.6.2 레트로시트 데이터베이스

부록 A에서 우리는 레트로시트 파일을 다운로드하고 이를 R에서 쉽게 읽을 수 있는 형식으로 변환하는 R 코드를 제공합니다. 11.4.2절에 제공된 코드를 약간 수정하면 이를 MySQL 데이터베이스에 추가할 수도 있습니다.

인터넷에는 레트로시트 파일을 처리하기 위한 많은 소프트웨어 패키지가 있습니다. 우리는 baseballr 패키지를 사용하여 데이터베이스를 구축했습니다.

- 11.6.3 스탯캐스트 데이터베이스

부록 C에서 설명한 바와 같이, baseballr 패키지는 베이스볼 서번트(Baseball Savant)에서 스탯캐스트 데이터를 다운로드할 수 있는 statcast_search() 함수를 제공합니다. 12장에서는 abdwr3edata 패키지가 이 기능을 활용하여 여러 시즌의 스탯캐스트 데이터를 어떻게 다운로드할 수 있는지 보여줍니다. 해당 장에서는 다양한 대안적 데이터 저장 옵션과 각각의 장단점도 논의합니다.

###### 11.7 기본적인 파크 팩터 계산하기

파크 팩터(보통 PF로 약칭)는 선수들의 가치를 평가할 때 구장의 효과를 상황에 맞게 해석하기 위한 도구로써 수십 년 동안 야구 분석가들에 의해 사용되어 왔습니다. 파크 팩터는 여러 가지 방법으로 계산되어 왔으며, 이 섹션에서는 그림 11.2에 나타난 바와 같이 쿠어스 필드에 있어 극단적인 시즌 중 하나였던 1996년에 초점을 맞추어 매우 기본적인 접근 방식을 설명합니다.

이어지는 설명에서 우리는 독자의 데이터베이스에 1990년대 레트로시트 데이터가 있다고 가정합니다.10 다음 코드는 baseballr 패키지를 사용하여 적합한 데이터베이스를 설정할 것입니다.

library(baseballr)
retro_data <- baseballr::retrosheet_data(
here::here("data_large/retrosheet"), 1990:1999
)
events <- retro_data |>
map(pluck, "events") |>
bind_rows() |>
as_tibble()

con |>

10데이터를 MySQL 데이터베이스로 가져오기 위해 필요한 단계를 수행하려면 11.6.2절을 참조하시기 바랍니다.

- 11.6.2 레트로시트 데이터베이스

부록 A에서 우리는 레트로시트 파일을 다운로드하고 이를 R에서 쉽게 읽을 수 있는 형식으로 변환하는 R 코드를 제공합니다. 11.4.2절에 제공된 코드를 약간 수정하면 이를 MySQL 데이터베이스에 추가할 수도 있습니다.

인터넷에는 레트로시트 파일을 처리하기 위한 많은 소프트웨어 패키지가 있습니다. 우리는 baseballr 패키지를 사용하여 데이터베이스를 구축했습니다.

- 11.6.3 스탯캐스트 데이터베이스

부록 C에서 설명한 바와 같이, baseballr 패키지는 베이스볼 서번트(Baseball Savant)에서 스탯캐스트 데이터를 다운로드할 수 있는 statcast_search() 함수를 제공합니다. 12장에서는 abdwr3edata 패키지가 이 기능을 활용하여 여러 시즌의 스탯캐스트 데이터를 어떻게 다운로드할 수 있는지 보여줍니다. 해당 장에서는 다양한 대안적 데이터 저장 옵션과 각각의 장단점도 논의합니다.

###### 11.7 기본적인 파크 팩터 계산하기

파크 팩터(보통 PF로 약칭)는 선수들의 가치를 평가할 때 구장의 효과를 상황에 맞게 해석하기 위한 도구로써 수십 년 동안 야구 분석가들에 의해 사용되어 왔습니다. 파크 팩터는 여러 가지 방법으로 계산되어 왔으며, 이 섹션에서는 그림 11.2에 나타난 바와 같이 쿠어스 필드에 있어 극단적인 시즌 중 하나였던 1996년에 초점을 맞추어 매우 기본적인 접근 방식을 설명합니다.

이어지는 설명에서 우리는 독자의 데이터베이스에 1990년대 레트로시트 데이터가 있다고 가정합니다.10 다음 코드는 baseballr 패키지를 사용하여 적합한 데이터베이스를 설정할 것입니다.

library(baseballr)
retro_data <- baseballr::retrosheet_data(
here::here("data_large/retrosheet"), 1990:1999
)
events <- retro_data |>
map(pluck, "events") |>
bind_rows() |>
as_tibble()

con |>

10데이터를 MySQL 데이터베이스로 가져오기 위해 필요한 단계를 수행하려면 11.6.2절을 참조하시기 바랍니다.

dbWriteTable(name = "events", value = events)
events_db <- con |>
tbl("events")

이 과정이 완료되면 데이터베이스에는 170만 개 이상의 행을 가진 events라는 테이블이 포함되어야 합니다. R에서 전체 테이블을 tibble 객체 events로 저장하면 거의 1GiB의 메모리를 차지하지만, 해당 데이터를 데이터베이스로 푸시하고 events_db 객체로서 데이터베이스에 대한 dplyr 인터페이스를 사용하면 메모리 공간을 거의 차지하지 않는다는 점에 유의하시기 바랍니다(물론 디스크 상에는 여전히 1GiB를 차지합니다).

- 11.7.1 R로 데이터 불러오기

첫 번째 단계로 MySQL 데이터베이스에 연결하여 원하는 데이터를 검색합니다. SQL 쿼리를 사용하여 events 테이블에서 홈팀 및 원정팀, 이벤트 코드가 포함된 열을 선택하고, 연도가 1996년이며 이벤트 코드가 타구를 나타내는 것에 해당하는 행만 유지합니다(부록 A 참조). 쿼리 결과는 R의 hr_PF 데이터 프레임에 저장됩니다.

query <- "
SELECT away_team_id, LEFT(game_id, 3) AS home_team_id, event_cd
FROM events
WHERE year = 1996
AND event_cd IN (2, 18, 19, 20, 21, 22, 23);
"
hr_PF <- dbGetQuery(con, query)
dim(hr_PF)

[1] 130437 3

- 11.7.2 홈런 파크 팩터

야구장은 다양한 선수의 성적 통계에 다른 영향을 미칠 수 있습니다. 예를 들어 보스턴 펜웨이 파크의 독특한 구조는 타구가 2루타가 될 가능성을 높이는데, 특히 그린 몬스터를 맞고 튕겨 나오는 좌측 플라이볼의 경우 더욱 그렇습니다. 반면 펜웨이 파크의 우측은 외야 펜스가 비정상적으로 깊기 때문에 홈런이 드물게 나옵니다.

이 예제에서는 홈런 파크 팩터를 계산하여 1996년 홈런에 미친 경기장 효과를 살펴봅니다. 먼저 hr_PF 데이터 프레임의 모든 행에 대해 홈런 발생 여부를 나타내는 새로운 열 was_hr을 생성합니다.

hr_PF <- hr_PF |>
mutate(was_hr = ifelse(event_cd == 23, 1, 0))

다음으로, 모든 MLB 팀에 대해 홈과 원정 모두에서 타구 당 홈런 빈도를 계산합니다. 이 작업을 위해 전체 데이터 프레임을 두 번 통과해야 한다는 점에 유의하시기 바랍니다.

ev_away <- hr_PF |>
group_by(team_id = away_team_id) |>
summarize(hr_event = mean(was_hr)) |>
mutate(type = "away")

ev_home <- hr_PF |>
group_by(team_id = home_team_id) |>
summarize(hr_event = mean(was_hr)) |>
mutate(type = "home")

그런 다음 두 결과 데이터 프레임을 결합하고 pivot_wider() 함수를 사용하여 홈 및 원정 홈런 빈도를 나란히 배치합니다.

ev_compare <- ev_away |>
bind_rows(ev_home) |>
pivot_wider(names_from = type, values_from = hr_event)

ev_compare

# A tibble: 28 x 3

team_id away home
<chr> <dbl> <dbl>

- 1 ATL 0.0323 0.0372
- 2 BAL 0.0488 0.0477
- 3 BOS 0.0385 0.0443
- 4 CAL 0.0387 0.0483
- 5 CHA 0.0424 0.0349
- 6 CHN 0.0374 0.0407
- 7 CIN 0.0403 0.0393
- 8 CLE 0.0440 0.0372
- 9 COL 0.0341 0.0538
- 10 DET 0.0457 0.0506

# i 18 more rows

파크 팩터는 일반적으로 100이라는 값이 중립적인 구장(해당 통계에 영향을 미치지 않는 구장)을 나타내는 반면, 100을 초과하는 값은 이벤트(이 경우 홈런)의 가능성을 높이는 경기장을 나타내고 100 미만의 값은 이벤트의 가능성을 낮추는 구장을 나타내도록 계산됩니다.

hr_PF <- hr_PF |>
mutate(was_hr = ifelse(event_cd == 23, 1, 0))

다음으로, 모든 MLB 팀에 대해 홈과 원정 모두에서 타구 당 홈런 빈도를 계산합니다. 이 작업을 위해 전체 데이터 프레임을 두 번 통과해야 한다는 점에 유의하시기 바랍니다.

ev_away <- hr_PF |>
group_by(team_id = away_team_id) |>
summarize(hr_event = mean(was_hr)) |>
mutate(type = "away")

ev_home <- hr_PF |>
group_by(team_id = home_team_id) |>
summarize(hr_event = mean(was_hr)) |>
mutate(type = "home")

그런 다음 두 결과 데이터 프레임을 결합하고 pivot_wider() 함수를 사용하여 홈 및 원정 홈런 빈도를 나란히 배치합니다.

ev_compare <- ev_away |>
bind_rows(ev_home) |>
pivot_wider(names_from = type, values_from = hr_event)

ev_compare

# A tibble: 28 x 3

team_id away home
<chr> <dbl> <dbl>

- 1 ATL 0.0323 0.0372
- 2 BAL 0.0488 0.0477
- 3 BOS 0.0385 0.0443
- 4 CAL 0.0387 0.0483
- 5 CHA 0.0424 0.0349
- 6 CHN 0.0374 0.0407
- 7 CIN 0.0403 0.0393
- 8 CLE 0.0440 0.0372
- 9 COL 0.0341 0.0538
- 10 DET 0.0457 0.0506

# i 18 more rows

파크 팩터는 일반적으로 100이라는 값이 중립적인 구장(해당 통계에 영향을 미치지 않는 구장)을 나타내는 반면, 100을 초과하는 값은 이벤트(이 경우 홈런)의 가능성을 높이는 경기장을 나타내고 100 미만의 값은 이벤트의 가능성을 낮추는 구장을 나타내도록 계산됩니다.

다음 코드를 사용하여 1996년 홈런 파크 팩터를 계산하고 arrange()를 사용하여 큰 파크 팩터와 작은 파크 팩터를 가진 구장들을 표시합니다.

ev_compare <- ev_compare |>
mutate(pf = 100 \* home / away)

ev_compare |>
arrange(desc(pf)) |>
slice_head(n = 6)

# A tibble: 6 x 4

team_id away home pf
<chr> <dbl> <dbl> <dbl>
1 COL 0.0341 0.0538 158.
2 CAL 0.0387 0.0483 125.
3 ATL 0.0323 0.0372 115.
4 BOS 0.0385 0.0443 115.
5 DET 0.0457 0.0506 111.
6 SDN 0.0294 0.0320 109.

쿠어스 필드는 158이라는 극단적인 값을 보이며 홈런 친화적인 구장 목록의 최상단에 위치해 있습니다. 이 구장은 1996년에 홈런 빈도를 50% 이상 끌어올렸습니다!

ev_compare |>
arrange(pf) |>
slice_head(n = 6)

# A tibble: 6 x 4

team_id away home pf
<chr> <dbl> <dbl> <dbl>

- 1 LAN 0.0360 0.0256 71.2
- 2 HOU 0.0344 0.0272 79.1
- 3 NYN 0.0363 0.0289 79.5
- 4 CHA 0.0424 0.0349 82.2
- 5 CLE 0.0440 0.0372 84.6
- 6 FLO 0.0316 0.0271 85.7

스펙트럼의 반대쪽 끝에는 71의 홈런 파크 팩터를 기록한 로스앤젤레스의 다저 스타디움이 있었는데, 이는 리그 평균 구장에 비해 홈런을 약 30% 억제했음을 의미합니다.

- 11.7.3 제안된 접근 방식의 가정

파크 팩터를 계산하기 위해 제안된 접근 방식은 몇 가지 단순화 가정을 전제로 합니다. 첫 번째 가정은 홈팀이 항상 동일한 홈구장에서 경기를 치른다는 것입니다. 이는 대부분의 시즌 대부분의 팀에게 해당되지만, 특정 경기를 위해 대체 구장이 사용된 적도 있습니다. 예를 들어,

1996 시즌 동안 오클랜드 애슬레틱스는 오클랜드-알라메다 카운티 콜리세움의 개보수가 완료되는 동안 첫 16번의 홈경기를 캐시먼 필드(네바다주 라스베이거스)에서 치렀습니다. 같은 해, MLB의 마케팅 전략의 일환으로 샌디에이고 파드리스는 멕시코의 에스타디오 데 베이스볼 몬테레이에서 뉴욕 메츠와 3연전을 치렀습니다.11

제안된 접근 방식의 또 다른 가정은 단일 파크 팩터가 구장이 일부 선수 범주에 다르게 영향을 미칠 수 있다는 점을 고려하지 않고 모든 선수에게 적절하다는 것입니다. 사실 비대칭적인 외야 구성은 경기장이 우타자와 좌타자에게 불평등한 영향을 미치게 합니다. 예를 들어, 좌측 외야에 위치한 보스턴의 앞서 언급한 그린 몬스터는 우타자가 타석에 있을 때 더 자주 경기에 영향을 미칩니다. 그리고 최근의 양키 스타디움에서는 좌타자들이 우측 외야 펜스의 짧은 거리를 활용하는 모습이 나타났습니다.

마지막으로, 제안된 파크 팩터(그리고 공개된 파크 팩터의 대부분의 버전들)는 본질적으로 각 이벤트에 관련된 선수들(이 경우 타자와 투수)을 무시합니다. 팀들이 플레이 바이 플레이 데이터 분석에 더 많이 의존함에 따라, 그들은 구장의 특성을 수용하기 위해 일반적으로 전략을 조정합니다. 예를 들어 그림 11.2에 나타난 바와 같이 득점에 미치는 쿠어스 필드의 영향이 감소한 것은 주로 휴미더에 기인하지만, 효과의 일부는 분명 팀들이 이 구장에서 경기할 때 다른 전략을 사용하기 때문이기도 합니다. 예를 들어, 팀들은 희박한 공기에 덜 영향을 받을 수 있는 땅볼을 많이 유도하는 투수들을 기용할 수 있습니다.

- 11.7.4 파크 팩터 적용하기

1996 시즌에 네 명의 로키스 선수가 30개 이상의 홈런을 기록했습니다. 안드레스 갈라라가가 47개로 팀을 이끌었고, 그 뒤를 비니 카스티야와 엘리스 버크스(40개로 동률), 단테 비솃(31개)이 이었습니다. 그 뒤로 래리 워커는 홈런이 18개에 불과했지만, 부상으로 인해 출전 시간이 매우 제한적이었습니다. 사실 워커의 타수 당 홈런 비율은 갈라라가 다음으로 두 번째였습니다. 이들의 공격력은 쿠어스 필드에서 81경기를 치름으로써 분명 크게 향상되었습니다. 이전에 계산된 파크 팩터를 사용하면, 갈라라가가 중립적인 구장 환경에서 쳤을 홈런 수를 추정할 수 있습니다.

MySQL 데이터베이스에서 타구로 끝난 갈라라가의 1996년 타석을 모두 검색하는 것으로 시작하며, 11.7.2절에서 했던 것처럼 해당 이벤트가 홈런이었는지를 나타내는 was_hr이라는 열을 정의합니다.

query <- "
SELECT away_team_id, LEFT(game_id, 3) AS home_team_id, event_cd

11대체 구장에서 치러진 경기 목록은 레트로시트 웹사이트의 url https://www.retrosheet.org/neutral.htm에 표시되어 있습니다.

1996 시즌 동안 오클랜드 애슬레틱스는 오클랜드-알라메다 카운티 콜리세움의 개보수가 완료되는 동안 첫 16번의 홈경기를 캐시먼 필드(네바다주 라스베이거스)에서 치렀습니다. 같은 해, MLB의 마케팅 전략의 일환으로 샌디에이고 파드리스는 멕시코의 에스타디오 데 베이스볼 몬테레이에서 뉴욕 메츠와 3연전을 치렀습니다.11

제안된 접근 방식의 또 다른 가정은 단일 파크 팩터가 구장이 일부 선수 범주에 다르게 영향을 미칠 수 있다는 점을 고려하지 않고 모든 선수에게 적절하다는 것입니다. 사실 비대칭적인 외야 구성은 경기장이 우타자와 좌타자에게 불평등한 영향을 미치게 합니다. 예를 들어, 좌측 외야에 위치한 보스턴의 앞서 언급한 그린 몬스터는 우타자가 타석에 있을 때 더 자주 경기에 영향을 미칩니다. 그리고 최근의 양키 스타디움에서는 좌타자들이 우측 외야 펜스의 짧은 거리를 활용하는 모습이 나타났습니다.

마지막으로, 제안된 파크 팩터(그리고 공개된 파크 팩터의 대부분의 버전들)는 본질적으로 각 이벤트에 관련된 선수들(이 경우 타자와 투수)을 무시합니다. 팀들이 플레이 바이 플레이 데이터 분석에 더 많이 의존함에 따라, 그들은 구장의 특성을 수용하기 위해 일반적으로 전략을 조정합니다. 예를 들어 그림 11.2에 나타난 바와 같이 득점에 미치는 쿠어스 필드의 영향이 감소한 것은 주로 휴미더에 기인하지만, 효과의 일부는 분명 팀들이 이 구장에서 경기할 때 다른 전략을 사용하기 때문이기도 합니다. 예를 들어, 팀들은 희박한 공기에 덜 영향을 받을 수 있는 땅볼을 많이 유도하는 투수들을 기용할 수 있습니다.

- 11.7.4 파크 팩터 적용하기

1996 시즌에 네 명의 로키스 선수가 30개 이상의 홈런을 기록했습니다. 안드레스 갈라라가가 47개로 팀을 이끌었고, 그 뒤를 비니 카스티야와 엘리스 버크스(40개로 동률), 단테 비솃(31개)이 이었습니다. 그 뒤로 래리 워커는 홈런이 18개에 불과했지만, 부상으로 인해 출전 시간이 매우 제한적이었습니다. 사실 워커의 타수 당 홈런 비율은 갈라라가 다음으로 두 번째였습니다. 이들의 공격력은 쿠어스 필드에서 81경기를 치름으로써 분명 크게 향상되었습니다. 이전에 계산된 파크 팩터를 사용하면, 갈라라가가 중립적인 구장 환경에서 쳤을 홈런 수를 추정할 수 있습니다.

MySQL 데이터베이스에서 타구로 끝난 갈라라가의 1996년 타석을 모두 검색하는 것으로 시작하며, 11.7.2절에서 했던 것처럼 해당 이벤트가 홈런이었는지를 나타내는 was_hr이라는 열을 정의합니다.

query <- "
SELECT away_team_id, LEFT(game_id, 3) AS home_team_id, event_cd

11대체 구장에서 치러진 경기 목록은 레트로시트 웹사이트의 url https://www.retrosheet.org/neutral.htm에 표시되어 있습니다.

추가 읽을거리 277

FROM events
WHERE year = 1996
AND event_cd IN (2, 18, 19, 20, 21, 22, 23)
AND bat_id = 'galaa001';
"
andres <- dbGetQuery(con, query) |>
mutate(was_hr = ifelse(event_cd == 23, 1, 0))

앞서 계산한 파크 팩터를 andres 데이터 프레임에 추가합니다. 이는 home_team_id 및 team_id 열을 키로 하여 inner_join() 함수를 사용하여 andres와 ev_compare 데이터 프레임을 병합함으로써 수행됩니다. 병합된 데이터 프레임 andres_pf에서 summarize()를 사용하여 갈라라가의 타석에 대한 평균 파크 팩터를 계산합니다.

andres_pf <- andres |>
inner_join(ev_compare, by = c("home_team_id" = "team_id")) |>
summarize(mean_pf = mean(pf))

andres_pf

mean_pf
1 129

갈라라가의 종합 파크 팩터는 홈에서 친 타구 252개와 원정에서 친 타구 225개(로스앤젤레스 다저 스타디움에서의 9개부터 휴스턴 애스트로돔에서의 23개까지 다양함)에서 파생되었으며, 이는 안드레스의 홈런 빈도가 중립 환경에 비해 약 29% 증가했음을 나타냅니다. 중립 환경에서의 홈런 추정치를 구하기 위해 갈라라가의 홈런을 100으로 나눈 그의 평균 홈런 파크 팩터로 나눕니다.

47 / (andres_pf / 100)

mean_pf
1 36.4

우리의 추정에 따르면, 갈라라가가 뛴 구장들(특히 그의 홈구장인 쿠어스 필드)에서 얻은 이점은 1996 시즌에 약 47−36 = 11 홈런에 달했습니다.

###### 11.8 추가 읽을거리

Adler (2006)의 2장에는 MySQL을 구하고 설치하는 방법과 레트로시트 데이터를 사용하여 역사적 야구 데이터베이스를 설정하는 방법에 대한 자세한 지침이 있습니다. Hack #56(같은 책의 5장)은 파크 팩터를 계산하고 적용하기 위한 SQL 코드를 제공합니다. Benjamin S. Baumer, Kaplan, and Horton (2021b)의 부록 F.2에는 11.2절에 제시된 것보다 더 완전한 단계별 지침이 포함되어 있습니다.

MySQL 참조 설명서는 MySQL 웹사이트의 http://dev.my sql.com/doc/ 에서 여러 형식으로 제공됩니다. 온라인 HTML 버전은 사용자가 특정 기능과 관련된 페이지를 신속하게 검색할 수 있도록 검색창을 제공합니다.

- 11.9 연습문제

- 1. 애스트로돔에서의 득점

- a. DBI 패키지의 dbGetQuery() 함수를 사용하여 애스트로돔이 애스트로스의 홈구장이었던 연도(즉, 1965년부터 1999년까지) 동안 애스트로스가 참가한(홈팀 또는 원정팀으로서) 경기를 선택하십시오.
- b. 애스트로돔과 다른 구장에서 열린 경기의 득점(양 팀 합산)을 연도별로 시각적으로 비교하는 그래프를 그리십시오.

- 2. 애스트로돔 홈런 파크 팩터

- a. 1965년부터 1999년 사이의 한 시즌 데이터를 선택하십시오. 원정팀 식별자, 홈팀 식별자 및 이벤트 코드를 나타내는 열과 인플레이 타구 이벤트를 식별하는 행을 유지하십시오. 홈런 발생 여부를 식별하는 새 열을 생성하십시오.
- b. 첫 번째 열에 팀 식별자를 포함하고, 두 번째 열에 팀이 원정 경기를 할 때 타구 당 홈런 빈도를, 세 번째 열에 팀이 홈 경기를 할 때 동일한 빈도를 포함하는 데이터 프레임을 준비하십시오.
- c. 모든 MLB 팀의 홈런 파크 팩터를 계산하고 휴스턴의 돔 구장이 홈런을 치는 데 어떤 영향을 미쳤는지 확인하십시오.

- 3. 숫자를 “조정”하기 위해 파크 팩터 적용하기

a. 이전 연습문제에서 선택한 것과 동일한 시즌을 사용하여 원하는 애스트로스 선수 한 명이 등장하는 타석(인플레이 타구로 끝나는) 데이터를 확보하십시오. 이 연습문제는 마운드에 있는 애스트로스 투수나 타석에 있는 애스트로스 타자가 등장하는 타석에 대해 수행할 수 있습니다. 예를 들어 선택한 시즌이 1988년인 경우, 베테랑 투수 놀란 라이언(Nolan Ryan)(레트로시트 id: ryann001)이 허용한 홈런 수나 신인 포수 크레이그 비지오(Craig Biggio)(id: biggc001)가 친 홈런 수에 애스트로돔이 어떤 영향을 미쳤는지 확인하는 데 관심이 있을 수 있습니다.

레트로시트 데이터를 사용하여 데이터베이스를 설정하는 방법에 대한 지침이 있습니다. Hack #56(같은 책의 5장)은 파크 팩터를 계산하고 적용하기 위한 SQL 코드를 제공합니다. Benjamin S. Baumer, Kaplan, and Horton (2021b)의 부록 F.2에는 11.2절에 제시된 것보다 더 완전한 단계별 지침이 포함되어 있습니다.

MySQL 참조 설명서는 MySQL 웹사이트의 http://dev.my sql.com/doc/ 에서 여러 형식으로 제공됩니다. 온라인 HTML 버전은 사용자가 특정 기능과 관련된 페이지를 신속하게 검색할 수 있도록 검색창을 제공합니다.

11.9 연습문제

- 1. 애스트로돔에서의 득점

- a. DBI 패키지의 dbGetQuery() 함수를 사용하여 애스트로돔이 애스트로스의 홈구장이었던 연도(즉, 1965년부터 1999년까지) 동안 애스트로스가 참가한(홈팀 또는 원정팀으로서) 경기를 선택하십시오.
- b. 애스트로돔과 다른 구장에서 열린 경기의 득점(양 팀 합산)을 연도별로 시각적으로 비교하는 그래프를 그리십시오.

- 2. 애스트로돔 홈런 파크 팩터

- a. 1965년부터 1999년 사이의 한 시즌 데이터를 선택하십시오. 원정팀 식별자, 홈팀 식별자 및 이벤트 코드를 나타내는 열과 인플레이 타구 이벤트를 식별하는 행을 유지하십시오. 홈런 발생 여부를 식별하는 새 열을 생성하십시오.
- b. 첫 번째 열에 팀 식별자를 포함하고, 두 번째 열에 팀이 원정 경기를 할 때 타구 당 홈런 빈도를, 세 번째 열에 팀이 홈 경기를 할 때 동일한 빈도를 포함하는 데이터 프레임을 준비하십시오.
- c. 모든 MLB 팀의 홈런 파크 팩터를 계산하고 휴스턴의 돔 구장이 홈런을 치는 데 어떤 영향을 미쳤는지 확인하십시오.

- 3. 숫자를 “조정”하기 위해 파크 팩터 적용하기

a. 이전 연습문제에서 선택한 것과 동일한 시즌을 사용하여 원하는 애스트로스 선수 한 명이 등장하는 타석(인플레이 타구로 끝나는) 데이터를 확보하십시오. 이 연습문제는 마운드에 있는 애스트로스 투수나 타석에 있는 애스트로스 타자가 등장하는 타석에 대해 수행할 수 있습니다. 예를 들어 선택한 시즌이 1988년인 경우, 베테랑 투수 놀란 라이언(Nolan Ryan)(레트로시트 id: ryann001)이 허용한 홈런 수나 신인 포수 크레이그 비지오(Craig Biggio)(id: biggc001)가 친 홈런 수에 애스트로돔이 어떤 영향을 미쳤는지 확인하는 데 관심이 있을 수 있습니다.

연습문제 279

b. 11.7.4절에서 설명한 대로 선택한 선수의 데이터와 이전에 계산된 파크 팩터를 결합하고 선수의 개별 파크 팩터(이는 선수가 다양한 구장에서 가진 각기 다른 출전 시간에 영향을 받습니다)를 계산한 다음, 이를 사용하여 친(또는 투수가 선택된 경우 허용한) “공정한” 홈런 수를 추정하십시오.

- 4. 다른 이벤트에 대한 파크 팩터

- a. 홈런 이외의 이벤트에 대해서도 파크 팩터를 추정할 수 있습니다. 예를 들어 SeamHeads.com 구장 데이터베이스에는 7가지 다른 이벤트에 대한 파크 팩터가 있으며, 타자의 손잡이에 따라 분할된 팩터도 제공합니다. 애스트로돔에 대한 페이지를 참조하십시오: http://www.se amheads.com/ballparks/ballpark.php?parkID=HOU02&tab=pf1.
- b. 이벤트(SeamHeads에 표시된 7가지와 다르더라도)를 선택하고 구장이 해당 빈도에 어떤 영향을 미치는지 계산하십시오. 제안하자면, 독자는 인조 잔디가 MLB 경기장의 40% 가까이에 설치되었던 80년대 시즌을 살펴보고, 콘크리트/합성 잔디 표면을 가진 구장이 아웃으로 전환된 타구(홈런 제외)의 빈도가 더 높은지 확인할 수 있습니다.12

- 5. 경기 시간

메이저리그 베이스볼은 경기 시간을 줄이기 위한 여러 조치 중 하나로 2023 시즌에 피치 클락을 도입했습니다. 레트로시트 게임 로그에는 각 경기의 길이(분)를 측정하는 Duration이라는 변수가 포함되어 있습니다. retrosheet_gamelog() 함수를 사용하여 2022년과 2023년의 경기 시간을 비교하십시오. 경기 시간의 분포를 설명하기 위해 상자 수염 그림(box plot)을 그리십시오.

- 6. 경기 시간 (계속)

이전 질문의 분석을 확장하여 2023년보다 평균 경기 시간이 더 짧았던 최근 연도(2023년 이전)를 찾기 위해 필요한 만큼 거슬러 올라가 보십시오.

12SeamHeads는 각 경기장 페이지에서 경기 표면에 대한 정보를 제공합니다. 예를 들어 앞서 언급한 애스트로돔 관련 페이지에서 특정 시즌의 구장 이름 위에 마우스를 올리면 구장 지붕 및 경기장 표면 모두에 대한 정보를 제공하는 팝업이 나타납니다. SeamHeads는 현재 구장 데이터베이스를 R에서 쉽게 읽을 수 있는 쉼표로 구분된 값(.csv) 파일이 포함된 zip 압축 파일로 제공하고 있습니다. 다운로드 링크는 구장 데이터베이스 섹션의 각 페이지 하단에서 찾을 수 있습니다.
