15

Shiny를 이용한 야구 애플리케이션

- 15.1 소개

R 생태계의 흥미로운 기능 중 하나는 shiny 패키지를 사용하여 R 작업의 웹 애플리케이션을 비교적 쉽게 구축할 수 있다는 점입니다. 이 장에서는 메이저 리그 야구 역사의 두 투수의 통산 궤적을 비교하고자 하는 야구 애플리케이션의 사용을 통해 Shiny 앱을 구축하는 방법을 설명합니다.

Shiny 앱 개발을 시작하기 좋은 방법은 앱에 표시하고자 하는 계산을 수행하는 함수를 작성하는 것입니다. 15.2절에서는 여러 R 함수를 사용하여 동시대 투수 그룹을 선택하고 그들의 통산 궤적 비교 그래프를 작성합니다. 15.3절에서는 사용자 인터페이스와 서버 구성 요소를 포함하여 Shiny 앱을 구축하고 실행하는 단계를 설명합니다. Shiny 앱이 완성된 후 15.4절에서는 다른 사람들이 앱을 사용해보도록 하는 여러 방법을 설명하고, 15.5절에서는 관심 있는 독자가 자신의 앱을 빠르게 실행하는 데 도움이 될 만한 몇 가지 팁을 제공하며 결론을 맺습니다.

- 15.2 두 투수의 궤적 비교

우리는 같은 야구 시대에 활약한 두 투수의 통산 궤적을 비교하는 데 초점을 맞춥니다. 특정 시즌 간격과 최소 투구 이닝이 주어졌을 때, 선택한 두 투수에 대해 시즌 또는 나이에 따른 성과 지표를 그래프로 표시하고자 합니다. 관련 데이터는 Lahman 패키지와 FIP 지표 계산에 필요한 값이 포함된 FanGraphs 테이블에 있습니다.

이러한 작업을 돕기 위해 두 개의 함수를 작성했습니다. 첫 번째 함수는 특정 기간 내에 경력의 중간을 보냈으며 최소 투구 이닝 기준을 달성한 모든 투수의 데이터 프레임을 반환하는 selectPlayers2()입니다. 야구 역사상 수많은 투수가 있어서 모두 나열하기는 어렵습니다. 그렇게 하면 앱을 사용하기가 번거로워집니다. 대신, selectPlayers2()는 투수 목록을 적절한 수로 좁히는 데 도움을 줍니다. 예를 들어 다음 코드는 최소 2000이닝을 투구하고 1959년에서 1966년 사이에 경력의 중간을 보낸 모든 투수를 반환합니다. 이 투수들은 앱에서 비교할 수 있는 자격이 있습니다.

```R
library(abdwr3edata) selectPlayers2(c(1959, 1966), 2000)

# A tibble: 26 x 2 playerID Name <chr> <chr>

- 1 bellga01 Gary Bell
- 2 buhlbo01 Bob Buhl
- 3 bunniji01 Jim Bunning
- 4 cardwdo01 Don Cardwell
- 5 chancde01 Dean Chance
- 6 drysddo01 Don Drysdale
- 7 ellswdi01 Dick Ellsworth
- 8 grantmu01 Mudcat Grant
- 9 jacksla01 Larry Jackson
- 10 klinero01 Ron Kline # i 16 more rows
```

함수 내부에서 우리는 먼저 중간 경력 연도와 투구 이닝 수를 계산합니다. 여기서 midYear는 투수의 첫 시즌과 마지막 시즌의 평균으로 정의됩니다. 앞서 언급한 값을 입력으로 받아 selectPlayers2()는 Lahman 패키지를 쿼리하고 기준을 충족하는 모든 투수의 선수 ID와 이름을 출력합니다. 함수의 전체 코드는 아래에 나와 있습니다.

```R
selectPlayers2

function(midYearRange, minIP) {

Lahman::Pitching |> mutate(IP = IPouts / 3) |> group_by(playerID) |> summarize(

minYear = min(yearID), maxYear = max(yearID), midYear = (minYear + maxYear) / 2, IP = sum(IP),

.groups = "drop" ) |> filter(

midYear <= max(midYearRange), midYear >= min(midYearRange), IP >= minIP

) |> select(playerID) |> inner_join(Lahman::People, by = "playerID") |> mutate(Name = paste(nameFirst, nameLast)) |> select(playerID, Name)

} <bytecode: 0x59bb9e639d20> <environment: namespace:abdwr3edata>
```

두 번째 도우미 함수 compare_plot()은 선택한 두 투수의 통산 궤적을 비교하는 그래프를 구성합니다. 이 함수는 두 투수의 선수 ID, 수직 축에 그래프로 나타낼 측정값(ERA, WHIP, FIP, SO 비율, BB 비율 중), 수평 축에 표시할 시간 변수(시즌 또는 나이)가 필요합니다.

compare_plot() 함수의 사용을 설명하기 위해, 위대한 다저스 투수 Sandy Koufax와 Don Drysdale의 나이에 따른 FIP(수비 무관 투구) 궤적을 비교하고자 한다고 가정해 보겠습니다. Lahman 패키지의 People 테이블에서 두 투수의 선수 ID를 수집합니다. fg 데이터 프레임에는 FanGraphs "guts" 테이블의 데이터가 포함되어 있습니다. 그런 다음 입력값 koufasa01, drysddo01, FIP, age와 함께 compare_plot() 함수를 적용합니다(그림 15.1 참조).

```R
compare_plot(

"koufasa01", "drysddo01", "FIP", "age" ) |>

pluck("plot1")
```

각 투수에 대해 이 함수는 나이에 대한 FIP 측정값의 산점도를 구성하고 평활화 곡선을 겹쳐 그립니다. geomtextpath 패키지의 geom_textsmooth() 함수는 각 평활화 곡선에 선수 레이블을 추가하는 데 사용됩니다. 이 함수의 전체 코드는 여기에 표시하기에 다소 길지만, abdwr3edata 패키지에서 코드에 액세스할 수 있습니다.

![image 106](images/imageFile106.png)

그림 15.1 Sandy Koufax와 Don Drysdale의 FIP 통산 궤적.

###### 15.3 Shiny 앱 만들기

- 15.3.1 기본 구조

Shiny 앱은 종종 app.R로 명명된 단일 R 스크립트 파일에 포함되어 있습니다. 이 파일에는 세 가지 기본 구성 요소가 포함되어 있습니다.

- 모든 입력 제어 요소를 포함하여 앱의 레이아웃을 설명하는 사용자 인터페이스 객체 ui
- 앱을 실행하는 데 필요한 지침을 설명하는 서버 함수 server()
- 사용자 인터페이스와 서버 정보가 주어졌을 때 앱을 생성하는 shinyApp() 함수에 대한 호출

다음 코드는 app.R 파일의 기본 구조를 표시합니다. 이 파일은 처음에는 두 함수 selectPlayers2()와 compare_plot()을 나열하고, 이어서 Shiny 구성 요소 ui와 Shiny 함수 server() 및 shinyApp()을 나열한다는 점에 유의하십시오.

```R
library(shiny) selectPlayers2 <- function(midYearRange, minIP) {

# ...code...

} compare_plot <- function(playerid_1, playerid_2, measure, xvar, fg) {

# ...code...

} ui <- fluidPage( # ...code...

) server <- function(input, output, session) {

# ...code...

} shinyApp(ui = ui, server = server)
```

![image 107](images/imageFile107.png)

그림 15.2 한 Shiny 앱의 레이아웃.

abdwr3edata 패키지의 compareTrajectories() 함수를 통해 앱의 전체 코드를 확인할 수 있습니다.

- 15.3.2 사용자 인터페이스 설계

이 특정 Shiny 앱의 레이아웃에서 그림 15.2에 표시된 것처럼 사용자 인터페이스 제어 요소는 앱의 왼쪽에 있고 출력은 오른쪽에 있습니다.

레이아웃은 ui 객체 내부의 fluidPage() 함수를 사용하여 정의됩니다. fluidRow() 함수는 사용자 인터페이스를 위해 4단위 너비를, 출력을 위해 8단위 너비를 갖는 Shiny 출력 창을 정의합니다.

```R
ui <- fluidPage( fluidRow( column(4, # user interface controls

), column(8,

# output functions )

) )
```

이 애플리케이션의 사용자 인터페이스 제어 요소는 슬라이더, 풀다운 메뉴 및 라디오 버튼으로 구성됩니다. shiny 패키지의 함수를 사용하여 앱에서 다양한 입력 유형을 구성합니다.

슬라이더 제어 요소는 중간 경력 및 최소 투구 이닝(IP) 값의 범위를 입력하는 데 사용됩니다. sliderInput() 함수는 첫 번째 슬라이더 입력 midyear를 정의하는 데 사용됩니다. 이 함수의 입력은 입력 레이블, 표시할 텍스트, 슬라이더 값 범위 및 현재 값입니다. value는 두 값의 벡터이므로 슬라이더에 값의 범위를 입력합니다.

```R
sliderInput( "midyear", label = "Select Range of Mid Career:", min = 1900, max = 2010, value = c(1975, 1985), sep = ""
```

)

selectInput() 함수는 풀다운 메뉴 입력 항목을 구성하는 데 사용됩니다. 아래에 player_name1 변수를 입력하기 위한 코드를 표시합니다. selectPlayers2() 함수가 특정 중간 경력 및 최소 PA 값을 가진 선수 이름 목록을 생성하는 데 사용된다는 점에 유의하십시오.

```R
selectInput( "player_name1", label = "Select First Pitcher:", choices = selectPlayers2(c(1975, 1985), 2000)$Name

)
```

라디오 버튼은 radioButtons() 함수를 사용하여 정의됩니다. 아래 코드는 type 변수에 대해 표시됩니다. 이 함수의 입력은 레이블, 표시되는 문자열 및 가능한 입력 값입니다.

```R
radioButtons( "type", label = "Select Measure:", choices = c("ERA", "WHIP", "FIP", "SO Rate", "BB Rate")

)
```

- 15.3.3 동적 사용자 입력 추가

이 특정 Shiny 앱의 특별한 기능 중 하나는 입력 제어 요소의 값이 다른 입력 제어 요소에 의해 수정될 수 있는 동적 UI의 사용입니다. 동적 UI는 server() 함수에서 observeEvent() 함수를 사용하여 달성됩니다. 다음 코드 스니펫의 observeEvent()에서 midyear 입력의 값이 변경될 때 player_name1 입력의 값이 수정됩니다. observeEvent() 함수는 midyear 또는 midpa의 값이 변경될 때마다 player_name1 및 player_name2의 값이 수정되도록 여러 번 사용됩니다.

```R
observeEvent( input$midyear, updateSelectInput(

inputId = "player_name1", choices = selectPlayers2(

input$midyear, input$minpa )$Name

) )
```

- 15.3.4 서버 구성 요소 완성

server() 함수에는 Shiny 앱을 위한 실제 작업도 포함되어 있습니다. 다음 스니펫은 출력 구성 요소 output$plot1이 어떻게 정의되는지 보여줍니다. 사용자 입력 input$midyear, input$midpa, input$player_name1 및 input$player_name2에서 selectPlayers2() 함수가 적용되어 선수 사용자 ID에 액세스합니다. 그런 다음 compare_plot() 함수를 이러한 입력과 함께 사용하여 플롯을 구성합니다. renderPlot() 함수는 이러한 입력이 변경될 때 앱에 그려지는 내용이 어떻게 변경되는지 제어합니다.

```R
output$plot1 <- renderPlot({ S <- selectPlayers2(input$midyear, input$minpa) id1 <- filter(S, Name == input$player_name1)$playerID id2 <- filter(S, Name == input$player_name2)$playerID compare_plot(id1, id2, input$type, input$xvar)$plot1

},

res = 96 )
```

![image 108](images/imageFile108.png)

![image 109](images/imageFile109.png)

![image 110](images/imageFile110.png)

그림 15.3 통산 궤적 Shiny 앱의 스냅샷.

15.3.5 앱 실행

일반적인 관행에서 Shiny 코드가 포함된 app.R 스크립트는 별도의 폴더에 배치됩니다. RStudio 콘솔 창에 다음을 입력하여 Shiny 앱을 실행합니다.

```R
runApp()
```

또는 화면 상단의 "Run App" 버튼을 누를 수 있습니다. 그림 15.3은 완성된 Shiny 앱의 스냅샷을 표시합니다. 이 특정 앱은 R 패키지의 일부이므로 다음을 입력하여 앱을 실행할 수 있습니다.

```R
compareTrajectories()
```

그림 15.3에서 중간 경력 간격을 1985–2000으로, 최소 PA 값을 2000으로 선택하고 명예의 전당 투수인 Greg Maddux와 Tom Glavine의 ERA 궤적을 비교하고 있습니다.

그들의 통산 궤적은 비슷했지만 Maddux가 전성기 동안 더 우수한 ERA를 기록했다는 점에 유의합니다.

- 15.4 앱 공유하기
  다른 사람과 Shiny 앱을 공유하는 방법에는 여러 가지가 있습니다.

- app.R 파일 공유. 앱이 단일 파일 app.R에 포함되어 있으므로 이 스크립트 파일을 다른 사람과 단순히 공유할 수 있습니다.
- 패키지에 넣기. 이것은 abdwr3edata 패키지의 compareTrajectories() 함수에 의해 설명된 방법입니다.
- Github을 통해 앱 공유. 앱을 공유하는 또 다른 방법은 Github 리포지토리를 생성하고 해당 리포지토리에 Shiny 앱을 저장하는 것입니다. 그러면 사용자는 runGitHub() 함수를 사용하여 리포지토리에서 Shiny 앱을 실행할 수 있습니다. 이 방법을 설명하기 위해 저자 중 한 명이 Github 리포지토리 testshinyapp을 생성한 다음 이 리포지토리에 통산 궤적 앱을 저장했습니다. runGitHub() 함수 덕분에 관심 있는 독자는 콘솔에 다음을 입력하여 이 앱을 실행할 수 있습니다.

```R
runGitHub( "testshinyapp", "bayesball")
```

- Shiny 서버에서 앱 호스팅. Posit은 현재 사용자가 웹 프로그램으로 앱을 볼 수 있도록 호스팅 서비스를 제공합니다. Posit 서비스를 사용하려면 https://www.shinyapps.io/에 계정을 설정해야 합니다. 그런 다음 Shiny 앱이 실행되면 앱 디스플레이에 서버에 앱을 업로드하는 Publish 버튼이 있습니다. 최근 저자 중 한 명이 통산 궤적 Shiny 앱에 대해 이 작업을 수행했으며 앱의 라이브 버전은 현재 다음 URL에서 사용할 수 있습니다.

https://bayesball.shinyapps.io/CareerTrajectoryPitching/

- 15.5 앱 제작 시작을 위한 팁

시작하는 쉬운 방법은 만들고자 하는 것과 유사한 사용자 인터페이스를 가진 Shiny 앱의 스크립트인 템플릿으로 시작하는 것입니다. 예를 들어 타자의 통산 궤적을 그리는 데 관심이 있다면 이 장에서 설명한 CareerTrajectoryPitching 앱을 수정할 수 있습니다.

Posit Shiny Gallery에는 다양한 유형의 Shiny 앱을 생성하기 위한 코드의 많은 그림이 있습니다. 샘플 app.R 스크립트로 시작하면 스크래치에서 프로그램을 구성할 때 범하기 쉬운 작은 코딩 오류를 피할 수 있습니다.

추가 읽을거리 343

- 15.6 추가 읽을거리

Posit은 Shiny R 사이트 https://shiny.posit.co/r/getstarted/shiny-basics/lesson1/index.html 에 Shiny 앱에 대한 많은 정보와 예제를 보유하고 있습니다. 또한 저자 중 한 명은 다양한 문제에 대한 야구 연구를 설명하기 위한 수많은 Shiny 앱이 포함된 R 패키지 ShinyBaseball(https://github.com/bayesball/ShinyBaseball 에 있음)을 만들었습니다. 이러한 앱은 https://baseballwithr.wordpress.com/ 의 "R로 야구 탐구하기" 블로그에 대한 R 작업을 설명하는 데 사용되었습니다.

- 15.7 연습문제

1. 인플레이 타구 위치 그리기

다음 함수 construct_zone_plot()은 선수의 인플레이 타구 영역 위치를 플롯팅하며, 플롯팅 점의 색상은 결과에 따라 달라집니다. 함수의 입력은 인플레이 타구의 Statcast 데이터 세트 sc_ip, 타자의 이름 p_name 및 결과 유형("Hit" 또는 "Home Run")입니다. 예를 들어 sc2023_ip가 2023 시즌의 인플레이 타구 데이터 프레임인 경우, 함수를 사용하여 타격별로 색칠된 Ronald Acu˜na의 모든 인플레이 타구 위치를 표시할 수 있습니다.

```R
construct_zone_plot(sc2023_ip, "Acu~a Jr., Ronald", "Hit")
```

![image 111](images/imageFile111.png)

선수 이름을 선택 목록을 통해 입력하고 결과 유형을 라디오 버튼을 사용하여 입력하는 이 함수를 사용하여 Shiny 앱을 구성하십시오.

```R
construct_zone_plot <- function(sc_ip, p_name, type) { require(dplyr) require(ggplot2) add_zone <- function() {

topKzone <- 3.5 botKzone <- 1.6 inKzone <- -0.85 outKzone <- 0.85 kZone <- data.frame(

x = c(inKzone, inKzone, outKzone, outKzone, inKzone), y = c(botKzone, topKzone, topKzone, botKzone, botKzone)

) geom_path(aes(.data$x, .data$y),

data = kZone, lwd = 1 )

} hits <- c("single", "double", "triple", "home_run") sc_player <- filter(sc_ip, player_name == p_name) |>

mutate( Hit = ifelse(events %in% hits, "YES", "NO"), Home_Run = ifelse(events == "home_run", "YES", "NO")

) ggplot() +

geom_point( data = sc_player, aes(plate_x, plate_z,
```

```R
color = .data[[type]]

) ) + add_zone() + coord_equal() + scale_colour_manual(values = c("tan", "red")) + labs(

title = paste( substr(sc_player$game_date[1], 1, 4), p_name

), subtitle = "Location of Balls in Play"

) + theme(

plot.title = element_text( color = "black", hjust = 0.5, size = 18

), plot.subtitle = element_text(

color = "black", hjust = 0.5, size = 14

) )

}
```

연습문제 345

- 2. 다른 결과를 사용하여 인플레이 타구 위치 그리기

연습문제 1에서 플롯팅 점의 색상은 "Hit" 또는 "Home Run" 결과에 의존할 수 있습니다. 유형 결과가 연속형 변수인 launch_angle, launch_speed 또는 estimated_ba_using_speedangle 중 하나일 수 있도록 construct_zone_plot() 함수를 수정하십시오. 사용자가 이 세 변수 중 하나를 입력할 수 있도록 Shiny 앱을 수정하십시오.

- 3. 브러싱을 사용하여 인플레이 타구 위치 그리기

Shiny를 사용하면 브러싱을 통해 그래프의 일부를 대화형으로 선택할 수 있습니다. 사용자 입력 섹션의 다음 코드는 브러시 옵션을 추가하여 plotOutput() 함수를 수정합니다.

```R
plotOutput("plot", brush = brushOpts("plot_brush", fill = "#0000ff"))
```

Shiny 앱의 서버 섹션의 새로운 output$data 구성 요소에서 다음 코드는 브러싱된 선택된 직사각형에 의해 정의된 sc_player 데이터 프레임의 부분 집합을 취합니다.

```R
sc1 <- brushedPoints(sc_player, input$plot_brush)
```

이 코드를 사용하여 연습문제 1의 Shiny 앱을 수정하여 산점도 브러싱을 허용하십시오. 디스플레이의 별도 영역에서 선택한 영역의 지점에 대한 인플레이 타구, 안타, 홈런 및 해당 안타 및 홈런 비율을 계산하십시오.

![image 112](images/imageFile112.png)

## 부록

![image 113](images/imageFile113.png)

### A Retrosheet 파일 참조

###### A.1 플레이 바이 플레이 파일 다운로드

- A.1.1 소개

1913년부터 2022년 사이의 모든 메이저 리그 시즌에 대한 플레이 바이 플레이 데이터 파일은 현재 Retrosheet 웹 페이지 https://www.retrosheet.org/game.htm 에서 사용할 수 있습니다. 단일 연도(1950)를 클릭하면 일련의 파일이 포함된 압축(.zip) 파일을 얻습니다. 파일 중 하나는 모든 팀의 홈 경기에 대한 플레이 정보가 포함된 파일 집합이고, 다른 하나는 각 팀의 선수 명단이 포함된 파일 집합입니다. 이 부록은 Retrosheet 파일로 작업하는 쉬운 방법을 설명합니다.

- A.1.2 Chadwick

Henry Chadwick은 박스 스코어, 타율 및 평균 자책점을 고안한 것으로 인정받는 스포츠 기자였습니다. Retrosheet 데이터를 처리하도록 설계된 특수 소프트웨어 도구는 그의 명예를 기려 이름이 지정되었습니다. 이러한 도구는 Ted Turocy가 유지 관리하며 https://github.com/chadwickbureau/chadwick 에서 사용할 수 있습니다. Chadwick 설치 지침을 따르십시오. 저장소에는 Windows 사용자에 적합한 바이너리가 포함되어 있으며 Linux 및 Mac 사용자는 소스 코드를 다운로드하고 컴파일하여 자신의 버전의 도구를 컴파일할 수 있습니다.

Retrosheet 플레이 바이 플레이 데이터를 생성하는 데 필요한 Chadwick의 특정 구성 요소를 cwevent라고 합니다. 이것은 명령줄에서 실행되는 프로그램입니다. 설치되어 제대로 작동하는 경우 명령줄에 단순히 cwevent를 입력하고 이와 같은 출력을 볼 수 있습니다.

```
cwevent

Chadwick expanded event descriptor, version 0.10.0

Type 'cwevent -h' for help. Copyright (c) 2002-2023 Dr T L Turocy, Chadwick Baseball Bureau (ted.turocy@gmail.com)

DOI: 10.1201/9781032668239-A 349

This is free software, subject to the terms of the GNU GPL license.
```

Chadwick을 설치했는데 cwevent를 실행할 때 오류가 발생하면 둘 중 하나의 문제일 수 있으며 둘 다 경로 환경 변수를 설정하는 것과 관련이 있습니다. 오류에 명령을 찾을 수 없다는 내용(또는 이와 유사한 내용)이 있으면 PATH 환경 변수에 cwevent 바이너리가 있는 디렉터리가 포함되어 있지 않기 때문에 운영 체제에서 cwevent 바이너리를 찾을 수 없는 것입니다. 이 Ubuntu 시스템1에서 다음을 입력하여 올바른 경로를 찾을 수 있습니다.

```
which cwevent

/usr/local/bin
```

echo를 사용하여 명령줄에서 현재 PATH 환경 변수의 값을 확인할 수 있습니다.

```
echo $PATH
```

export 지시문을 사용하여 cwevent 바이너리의 경로를 현재 PATH 환경 변수에 추가합니다.

```
export PATH=$PATH:/usr/local/bin
```

cwevent가 실행되지만 오류가 발생하는 경우 가능성 있는 문제는 cwevent가 Chadwick 공유 라이브러리를 찾을 수 없다는 것입니다. LD_LIBRARY_PATH 환경 변수를 설정하여 이 문제를 해결할 수 있습니다. 환경 변수는 시스템마다 다릅니다. 이 Ubuntu 시스템에서는 find 명령을 사용하여 Chadwick 공유 라이브러리를 찾을 수 있습니다.

```
find /usr/local -name "libchadwick*"

/usr/local/lib/libchadwick.la /usr/local/lib/libchadwick.a /usr/local/lib/libchadwick.so /usr/local/lib/libchadwick.so.0 /usr/local/lib/libchadwick.so.0.0.0
```

따라서 cwevent가 작동하려면 LD_LIBRARY_PATH 환경 변수에 /usr/local/lib가 포함되어야 합니다. 위에서 한 것처럼 export를 사용하여 LD_LIBRARY_PATH 환경 변수를 설정하거나, R 내부에서 환경 변수를 설정하려면 Sys.getenv() 및 Sys.setenv() 함수를 사용합니다.

1Windows에서 유사한 DOS 명령은 where입니다.

플레이 바이 플레이 파일 다운로드 351 함수. 아래에 제시하는 safe_add_ld_path() 함수는 baseballr 패키지의 chadwick_ld_library_path() 함수와 유사합니다.

```R
safe_add_ld_path <- function(path_new = "/usr/local/lib") { path_old <- Sys.getenv("LD_LIBRARY_PATH") path_old_parts <- path_old |>

str_split_1(":") |> unique()

if (!path_new %in% path_old_parts) { path_new_parts <- c(path_new, path_old_parts) Sys.setenv(

LD_LIBRARY_PATH = paste0(path_new_parts, collapse = ":") )

} Sys.getenv("LD_LIBRARY_PATH")

}

safe_add_ld_path()

[1] "/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:"
```

- A.1.3 하나 이상의 시즌에 대한 데이터 다운로드

Chadwick이 올바르게 설치되고 작동하면 baseballr 패키지의 retrosheet_data() 함수를 통해 Retrosheet 플레이 바이 플레이 데이터를 쉽게 얻을 수 있습니다.

retrosheet_data() 함수는 데이터를 관리하는 데 도움이 되는 세 가지 선택적 인수를 사용합니다. 이 함수는 cwevent를 호출하므로 부록 A.1.2와 같이 Chadwick을 올바르게 설정하지 않으면 작동하지 않습니다.

이 책에서 사용하는 Retrosheet 데이터를 다운로드하려면 다음을 입력합니다.

```R
retro_data <- baseballr::retrosheet_data( here::here("data_large/retrosheet"), c(1992, 1996, 1998, 2016)

)
```

이것은 4년 분량의 플레이 바이 플레이 데이터를 다운로드 및 처리하고 길이 4의 목록을 반환하며, 각 항목에는 길이 2의 목록이 포함됩니다. retro_data의 4개 항목은 지정한 4년에 해당합니다. 각 연도의 두 항목은 데이터 프레임입니다. 하나는 플레이 바이 플레이 데이터를 저장하는 events라는 파일이고, 다른 하나는 선수 명단을 저장하는 rosters라는 파일입니다.

단일 연도에 대한 플레이 바이 플레이 데이터를 분리하려면 pluck() 함수를 사용합니다.

```R
retro1992 <- retro_data |> pluck("1992") |> pluck("events")
```

- A.1.4 데이터 저장

사용할 때마다 이 모든 데이터를 다운로드하고 처리하기를 원하지는 않을 것입니다. R에 저장했으므로 나중에 사용할 수 있도록 저장하는 좋은 방법은 R의 내부 데이터 저장 형식을 사용하여 각 연도의 데이터 프레임을 디스크에 쓰는 것입니다. write_rds() 함수로 이 작업을 수행할 수 있습니다.

```R
retro1992 |>

write_rds( file = here::here("data/retro1992.rds"), compress = "xz"

)
```

compress 인수를 사용하십시오. 그러면 데이터 크기가 크게 줄어듭니다.

전체 Retrosheet 데이터 데이터베이스(다년도)를 구축하려는 경우 위에서 설명한 프로세스를 반복하고 11장에서 설명한 SQL 데이터베이스 구축 방법 및 12장에서 설명한 대규모 데이터 처리 논의와 결합하십시오.

- A.1.5 parse_retrosheet_pbp() 함수

이 책의 이전 버전에서는 Retrosheet 데이터를 다운로드하고 처리하는 데 사용할 수 있는 parse_retrosheet_pbp()라는 함수를 포함했습니다. 이 함수는 baseballr 패키지의 retrosheet_data() 함수로 대체되었으므로 더 이상 사용하지 않는 것이 좋습니다. 그러나 해당 논리를 살펴보는 데 관심이 있는 경우 abdwr3edata 패키지를 통해 코드를 사용할 수 있습니다.

```R
abdwr3edata::parse_retrosheet_pbp

function(season) { download_retrosheet(season) unzip_retrosheet(season) create_csv_file(season) create_csv_roster(season) cleanup()
```

```R
} <bytecode: 0x575fc1168c20> <environment: namespace:abdwr3edata>
```

추가 세부 정보를 위해 다양한 헬퍼 함수에 액세스할 수 있습니다. 예를 들면 다음과 같습니다.

```R
abdwr3edata::download_retrosheet

function(season) { # get zip file from retrosheet website utils::download.file(

url = paste0( ), destfile = file.path(

###### "http://www.retrosheet.org/events/", season, "eve.zip"

"retrosheet", "zipped", paste0(season, "eve.zip")

) )

} <bytecode: 0x575fc1330600> <environment: namespace:abdwr3edata>
```

- A.1.6 Chadwick의 대안

retrosheet 패키지(Douglas and Scriven 2024)는 getRetrosheet() 함수를 통해 Chadwick 소프트웨어에 대한 외부 종속성 없이 Retrosheet 데이터를 R로 가져오는 대체 방법을 제공합니다. 그러나 이 데이터는 데이터 프레임 목록이 아닌 목록의 목록으로 반환되므로 분석하기가 상당히 번거로울 수 있습니다.

###### A.2 Retrosheet 이벤트 파일: 짧은 참조

1장에서 언급했듯이 Retrosheet 이벤트 파일은 명시적으로 고안된 형식으로 제공되며 데이터 분석에 적합한 형식으로 변환하려면 일부 소프트웨어 도구를 사용해야 합니다. Retrosheet는 이러한 소프트웨어 도구 https://www.retrosheet.org/tools.htm 및 변환을 수행하기 위한 단계별 예제 https://www.retrosheet.org/stepex.txt 를 제공합니다.

Chadwick은 이 책에 사용된 플레이 바이 플레이 파일을 생성하는 데 사용된 Retrosheet 이벤트 파일을 구문 분석하기 위한 유사한 도구를 제공합니다(A.1.2절 참조).

Chadwick 도구는 Retrosheet 이벤트 파일의 각 플레이에 대해 97개의 "일반" 열(Retrosheet에서 제공하는 도구를 사용하여 얻은 것과 동일)과 63개의 "확장" 필드로 구성된 줄을 생성하여 Retrosheet 이벤트 파일에 포함된 모든 정보에 쉽게 액세스할 수 있도록 합니다. Chadwick 도구에서 생성한 150개 이상의 열을 각각 살펴보는 것은 이 책의 범위를 벗어나므로, 전체 목록은 Chadwick 웹사이트의 설명서를 가리킵니다.2 이 섹션에서는 이벤트와 해당 이벤트가 발생했을 때 게임의 상태를 설명하는 주요 필드를 제시합니다.

Chadwick 도구는 모든 변수 이름을 대문자로 반환하지만 retrosheet_data() 함수는 변수 이름에 스네이크 케이스를 사용합니다.

- A.2.1 게임 및 이벤트 식별자

게임은 Retrosheet 이벤트 파일에서 12자리 문자열(GAME_ID 열)로 식별됩니다. 처음 3자는 홈 팀을 식별하고, 다음 8자는 게임이 열린 날짜(YYYYMMDD 형식)를 나타내며, 마지막 문자는 더블헤더 게임을 구별하는 데 사용됩니다(따라서 "1"은 첫 번째 게임을, "2"는 두 번째 게임을 나타내고 "0"은 그 날 한 경기만 치렀음을 의미합니다).

이벤트는 각 게임에서 점진적으로 번호가 매겨지므로(EVENT_ID 열), Retrosheet 데이터베이스의 모든 단일 동작은 게임 식별자와 이벤트 식별자의 조합으로 고유하게 식별할 수 있습니다.

- A.2.2 게임 상태

특정 이벤트가 발생했을 때 게임 상태를 정의하는 데 몇 가지 필드가 도움이 됩니다. 이닝 및 공격 팀 변수는 각각 INN_CT 및 BAT_HOME_ID 필드에 저장됩니다. 후자 필드는

"0"(원정 팀 타격, 즉 이닝 초) 또는 "1"(홈 팀 타격, 이닝 말)의 값을 가정할 수 있습니다. 원정팀 점수 및 홈팀 점수 변수는 AWAY_SCORE_CT 및 HOME_SCORE_CT에 기록됩니다.

플레이 전의 아웃 횟수는 OUTS_CT 열에 표시되고, 주자 상황은 표 A.1에 표시된 대로 1에서 7까지의 숫자를 사용하여 START_BASES_CD 필드에 암호화됩니다.3

이벤트의 실제 설명은 플레이 결과(삼진, 단타 등)를 설명하는 문자열, 일부 추가 세부 정보(타구의 유형 및 위치) 및 주자의 진루로 구성된 EVENT_TX 열에 있습니다.

- 2모든 소프트웨어 도구에 대한 설명서는 https://chadwick.sourceforge.net/doc/cwtools.html에서 확인할 수 있습니다. 특히 이벤트 파일을 처리하는 도구(cwevent)는 https://chadwick.sourceforge.net/doc/cwevent.html#cwtools-cwevent에 문서화되어 있습니다.
- 3END_BASES_CD라는 유사한 열에는 플레이 종료 시 베이스 상태가 동일한 방식으로 암호화되어 포함됩니다.

- 표 A.1 주자 상황에 대한 Retrosheet 코딩.

코드 베이스 점유

- 0 비어 있음
- 1 1루만
- 2 2루만
- 3 1루 및 2루
- 4 3루만
- 5 1루 및 3루
- 6 2루 및 3루
- 7 만루

EVENT_TX 문자열을 디코딩하여 여러 열이 생성됩니다.

- EVENT_CD는 기본 이벤트를 반영하는 숫자 코드입니다. 표 A.2에는 이 열에 코딩된 가능한 플레이의 코드가 표시됩니다.
- BAT_EVENT_FL은 이벤트가 타격 이벤트인지 여부를 나타내는 플래그이며 이 경우 T로 레이블이 지정됩니다. 비타격 이벤트에는 예를 들어 도루, 폭투 및 일반적으로 타석의 끝을 표시하지 않는 모든 이벤트가 포함됩니다.
- H_CD는 기본 안타 유형을 나타내는 숫자 코드이며 단타의 경우 1에서 홈런의 경우 4까지입니다.
- BATTEDBALL_CD는 타구 유형을 나타내는 단일 문자 코드입니다. G(땅볼), L(라인 드라이브), F(뜬공), P(팝업) 중 하나의 값을 가정할 수 있습니다. Retrosheet 데이터베이스의 대부분의 시즌에 대해 타구 유형은 타자가 아웃으로 끝나는 타석에 대해서만 보고되는 반면 안타에서는 사용할 수 없다는 점에 유의하십시오.
- BATTEDBALL_LOC_TX는 https://www.retrosheet.org/location.htm에 표시된 다이어그램에 따라 암호화된 타구 위치를 나타내는 문자열입니다. 이 정보는 제한된 수의 시즌에만 사용할 수 있다는 점에 유의하십시오.
- FLD_CD는 타구에 처음 닿는 야수를 나타내는 숫자 코드이며 1(투수)에서 9(우익수)까지 기존 야구 수비 표기법으로 암호화됩니다.

투구의 시퀀스는 PITCH_SEQ_TX에 기록되며 6장에서 다루었으며 표 6.1에는 투구 결과 유형이 코딩되는 방법이 표시됩니다. 표 A.3에 표시된 대로 다양한 유형의 투구 결과의 빈도를 나타내는 이 열에서 여러 열이 생성됩니다.

- 표 A.2 이벤트 유형에 대한 Retrosheet 코딩.

코드 이벤트 유형

- 2 일반 아웃
- 3 삼진
- 4 도루
- 5 수비 무관심
- 6 도루 실패

- 8 견제구
- 9 폭투
- 10 패스트볼
- 11 보크
- 12 기타 진루
- 13 파울 오류
- 14 비고의적 볼넷
- 15 고의 4구
- 16 몸에 맞는 공
- 17 방해
- 18 실책
- 19 야수 선택
- 20 단타
- 21 2루타
- 22 3루타
- 23 홈런

- 표 A.3 다양한 투구 유형의 빈도를 보고하는 열.

열 이름 열 설명 PA BALL CT 타석에서의 볼 횟수 PA CALLED BALL CT 타석에서의 스트라이크 판정된 볼 횟수 PA INTENT BALL CT 타석에서의 고의 4구 볼 횟수 PA PITCHOUT BALL CT 타석에서의 피치아웃 횟수 PA HITBATTER BALL CT 타석에서의 몸에 맞는 공 투구 횟수

PA OTHER BALL CT 타석에서의 기타 볼 횟수 PA STRIKE CT 타석에서의 스트라이크 횟수 PA CALLED STRIKE CT 타석에서의 스트라이크 판정된 스트라이크 횟수 PA SWINGMISS STRIKE CT 타석에서의 헛스윙 스트라이크 횟수 PA FOUL STRIKE CT 타석에서의 파울 공 횟수 PA INPLAY STRIKE CT 타석에서의 인플레이 타구 횟수 PA OTHER STRIKE CT 타석에서의 기타 스트라이크 횟수

###### A.3 Retrosheet 투구 시퀀스 구문 분석

- A.3.1 소개

6장에서는 정규 표현식을 사용하여 타석이 1-0 또는 0-1 카운트를 거쳤는지 여부를 계산하는 방법을 보여주었습니다. 여기서는 모든 가능한 볼/스트라이크 카운트에 대해 동일한 정보를 검색하는 코드를 제공합니다.

- A.3.2 설정 먼저 2016 시즌의 Retrosheet 데이터를 로드합니다.

```R
retro2016 <- read_rds(here::here("data/retro2016.rds"))
```

그런 다음 타자에게 실제 투구를 나타내지 않는 모든 문자가 제거된 투구 시퀀스가 보고되는 새로운 열 시퀀스가 생성됩니다.4

```R
retro2016 <- retro2016 |> mutate(sequence = gsub("[.>123+*N]", "", pitch_seq_tx))
```

- A.3.3 모든 카운트 평가

모든 타석은 0-0 카운트로 시작합니다. 1-0 및 0-1 카운트 모두에 대한 코드는 6장에 설명되어 있습니다.

```R
retro2016 <- retro2016 |> mutate(

- c00 = TRUE, c10 = grepl("^[BIPV]", sequence),
- c01 = grepl("^[CFKLMOQRST]", sequence)

)
```

중괄호 안의 숫자는 일치시키기 위해 앞의 표현식이 문자열에서 반복되어야 하는 정확한 횟수를 나타냅니다. 다음 줄은 2-0, 3-0 및 0-2 카운트를 거치는 타석을 찾습니다.

```R
retro2016 <- retro2016 |> mutate(

c20 = grepl("^[BIPV]{2}", sequence), c30 = grepl("^[BIPV]{3}", sequence),
```

4참조로 6장의 표 6.1을 참조하십시오.

```R
c02 = grepl("^[CFKLMOQRST]{2}", sequence) )
```

| (수직 막대) 문자는 대안을 구분하는 데 사용됩니다. 다음 줄은 1-1, 2-1 및 3-1 카운트로 이어질 수 있는 다양한 시퀀스를 찾기 위해 시퀀스 문자열을 구문 분석합니다.

```R
b <- "[BIPV]" s <- "[CFKLMOQRST]" retro2016 <- retro2016 |>

mutate( c11 = grepl( paste0("^", s, b, "|^", b, s), sequence

), c21 = grepl(

paste0("^", s, b, b, "|^", b, s, b, "|^", b, b, s), sequence

), c31 = grepl(

paste0("^", s, b, b, b,

"|^", b, s, b, b, "|^", b, b, s, b, "|^", b, b, b, s), sequence

) )
```

2스트라이크 카운트에서 타자는 카운트에 영향을 미치지 않고 파울 공을 무한정 쳐낼 수 있습니다. 아래 줄에서 원하는 볼 수에 도달하기 전에 2스트라이크에 도달하는 시퀀스는 별표로 표시된 대로 0을 포함하여 횟수에 관계없이 발생하는 파울 공5을 나타내는 [FR]\* 표현식을 특징으로 합니다.

```R
retro2016 <- retro2016 |> mutate( c12 = grepl(

paste0("^", b, s, s, "|^", s, b, s, "|^", s, s, "[FR]*", b), sequence

), c22 = grepl(

paste0("^", b, b, s, s, "|^", b, s, b, s, "|^", b, s, s, "[FR]*", b, "|^", s, b, b, s,
```

5F는 파울 공을, R은 피치아웃 시 파울 공을 암호화합니다. 표 6.1을 참조하십시오.

```R
"|^", s, b, s, "[FR]*", b, "|^", s, s, "[FR]*", b, "[FR]*", b),

sequence

), c32 = grepl(

paste0("^", s, "*", b, s,

"*", b, s, "*", b), sequence ) & grepl(

paste0("^", b, "*", s, b, "*", s), sequence )

)
```

abdwr3edata 패키지의 retrosheet_add_counts()에는 이러한 횟수를 계산하는 데 필요한 모든 코드가 포함되어 있습니다.

```R
abdwr3edata::retrosheet_add_counts
```

### B

### PITCHf/x 데이터에 대한 역사적 참고 사항

B.1 소개

PITCHf/x는 축구의 퍼스트 다운 가상 선과 FoxTrax 하키 퍽과 같은 스포츠 방송 효과를 제작하는 회사인 Sportvision의 제품이었습니다. 각 MLB 구장에 설치된 두 대의 카메라는 투수 마운드와 홈 플레이트 사이의 야구공 비행을 기록했으며 고급 소프트웨어는 공의 위치, 속도 및 가속도를 계산하여 마운드에서 홈 플레이트까지의 공의 전체 궤적을 추정하기에 충분한 정보를 제공했습니다.

2006년부터 최근까지 메이저 리그 야구 고급 미디어(MLBAM)는 XML 형식으로 전달되는 실시간 데이터로 온라인 콘텐츠를 공급하는 공개적으로 액세스 가능한 Gameday 웹 서버를 유지 관리했습니다. 이 책의 이전 판에서는 이 피치 바이 피치 정보를 다운로드하기 위해 pitchRx 패키지를 사용했습니다. 불행히도 이 서버는 https://statsapi.mlb.com/에 위치한 API 기반 시스템으로 대체되었으므로 pitchRx 패키지는 더 이상 작동하지 않습니다. 이 새로운 시스템도 공개적으로 액세스 가능하고 실시간 데이터를 제공하지만 문서화가 잘 되어 있지 않으며 올바른 사용을 위해서는 개발자 등록이 필요합니다. 몇몇 개발자는 GitHub에 API 패키지를 게시했으며 눈에 띄는 것은 https://github.com/toddrob99/MLB-StatsAPI 에서 제공되는 Python용 MLB-StatsAPI 패키지이지만 우리가 아는 한 유사한 R 패키지는 널리 사용되지 않습니다.

이것이 나쁜 소식입니다.

좋은 소식은 기존 Gameday 서버에서 제공하던 유용한 데이터의 대부분(전부는 아니더라도)이 Baseball Savant를 통해 제공되는 Statcast 데이터를 통해 제공된다는 것입니다. 부록 C에서는 이 데이터의 검색 및 사용에 대해 자세히 설명합니다.

이 섹션의 나머지 부분에서는 PITCHf/x 데이터로 뒷받침된 세이버메트릭스 연구를 발전시키는 데 필수적이었던 몇 가지 리소스를 주로 후세를 위해 나열합니다.

DOI: 10.1201/9781032668239-B 360

온라인 리소스 361

B.2 온라인 리소스

다음 PITCHf/x 리소스는 월드 와이드 웹에서 사용할 수 있었습니다. 사이트 유지 관리자가 MLB 프런트 오피스에 고용되거나 PITCHf/x의 종료 또는 독점 라이선스 계약으로 인해 이러한 리소스는 제거되거나 이동될 수 있습니다.

- Mike Fast의 PITCHf/x 용어집(https://fastballs.wordpress.com/2007/08/ 02/glossary-of-the-gameday-pitch-fields/): PITCHf/x 필드에 대한 자세한 설명은 Mike Fast가 제공했습니다.
- Brooks Baseball(https://www.brooksbaseball.net/): Dan Brooks가 생성하고 유지 관리하며 주요 기능은 PITCHf/x 시스템이 설치된 구장에서 플레이한 적이 있는 모든 투수에 대한 테이블 및 차트로 구성된 플레이어 카드입니다. 테이블과 차트는 투구의 특성, 해당 투구의 사용(시퀀싱 포함) 및 생성되는 결과에 대한 정보를 보고합니다. 투구는 Pitch Info LLC에 의해 분류되므로 Brooks Baseball에서 사용되는 투구 분류는 MLBAM이 아닙니다. Brooks Baseball의 또 다른 유용한 리소스는 사이트 방문자가 한 경기의 한 투수를 선택하고 피치 바이 피치 테이블을 얻을 수 있는 PitchFX 도구입니다.
- Baseball Prospectus(https://www.baseballprospectus.com/): 통계 섹션에서 Baseball Prospectus는 PITCHf/x 타자 프로필, PITCHf/x 투수 프로필, PITCHf/x 리더보드 및 PITCHf/x 매치업을 제공합니다. 이러한 리소스의 구성 요소는 이전에 언급한 Brooks Baseball에서 비롯됩니다.
- FanGraphs(https://www.fangraphs.com/): FanGraphs에는 개별 선수에 대한 PITCHf/x 테이블 및 차트가 있습니다. 예를 들어 투수 James Shields의 PITCHf/x 페이지는 https://www.fangraphs.com/pitchfx.aspx?p layerid=7059&position=P에서 사용할 수 있습니다.
- F/X by Texas Leaguers(https://pitchfx.texasleaguers.com/): 기간을 설정하고 특정 선수에 대한 PITCHf/x 투구 또는 타격 데이터를 찾을 수 있습니다. 이 사이트에는 궤적 및 움직임에 대한 차트, 투구 특성에 대한 표, 결과 및 투수/타자 매치업이 포함되어 있습니다.
- Alan Nathan 교수의 야구 물리학(http://baseball.physics

.illinois.edu/): 야구 물리학에 대한 연구가 포함되어 있으며 http://baseball.physics.illinois.edu/pitchtracker.html에서 비디오 기술을 사용한 투구 추적에 전념하는 섹션이 있습니다.

- Katron의 MLB Gameday BIP 위치(https://katron.org/projects/baseball/hit-location/): 지정된 구장의 안타 위치 데이터를 선택한 다른 구장으로 옮길 수 있습니다. 타구 데이터에 대해 앞서 설명한 모든 주의 사항을 염두에 두고

362 PITCHf/x 데이터에 대한 역사적 참고 사항

새로운 팀으로의 이동이 선수의 타격에 미칠 수 있는 영향을 탐색하는 데 사용할 수 있습니다.

- Sportvision: Sportvision은 PITCHf/x 시스템을 고안한 회사였습니다. 그 이후로 SMT에 인수되었습니다.

- pos2 person id 포수의 Id

- pos3 person id 1루수의 Id

- pos4 person id 2루수의 Id

- pos5 person id 3루수의 Id

- pos6 person id 유격수의 Id

- pos7 person id 좌익수의 Id

- pos8 person id 중견수의 Id

- pos9 person id 우익수의 Id
  pitch number PA의 투구 번호

더 간단한 방법은 7.5절에서와 같이 baseballr 패키지의 chadwick_player_lu() 함수를 사용하는 것입니다. 이 파일을 다운로드하고 처리하는 데 1분이 걸리므로 write_rds() 함수를 사용하여 로컬 사본을 저장할 수 있습니다.

```R
master_id <- baseballr::chadwick_player_lu() |> write_rds(

here::here("data/chadwick_register.rds"), compress = "xz" )
```

- C.3 게임 상황 변수

많은 변수가 투구 당시의 게임 상황에 관한 것입니다(표 C.1 참조). 이러한 변수에는 날짜, 이닝 및 아웃 횟수가 포함됩니다.

표 C.1 Statcast의 게임 상황 변수.

이름 설명 game date 게임 날짜 batter 타자의 Id pitcher 투수의 Id stand 타자의 타석 위치 p throws 투수의 투구 손 home team 홈 팀 코드 away team 원정 팀 코드 balls 현재 볼 수 strikes 현재 스트라이크 수 on 3b 3루 주자의 Id on 2b 2루 주자의 Id on 1b 1루 주자의 Id outs when up 현재 아웃 횟수 inning 현재 이닝 inning topbot 이닝의 초 또는 말

- pos1 person id 투수의 Id
- pos2 person id 포수의 Id
- pos3 person id 1루수의 Id
- pos4 person id 2루수의 Id
- pos5 person id 3루수의 Id
- pos6 person id 유격수의 Id
- pos7 person id 좌익수의 Id
- pos8 person id 중견수의 Id
- pos9 person id 우익수의 Id
  pitch number PA의 투구 번호

투구 변수 365

- 표 C.2 Statcast의 투구 변수.

이름 설명 pitch type 투구 유형 코드 pitch name 투구 유형 설명 description 투구 결과에 대한 설명 release speed 방출 시 투구 속도(mph) effective speed 홈 플레이트를 통과할 때의 투구 속도(mph)

- release pos x 투구 방출 지점의 x 좌표

- release pos y 투구 방출 지점의 y 좌표

- release pos z 투구 방출 지점의 z 좌표
  zone 투구의 구역 위치 pfx x 투구의 수평 움직임 pfx z 투구의 수직 움직임 sz top 스트라이크 존 상단 수직 위치 sz bot 스트라이크 존 하단 수직 위치 plate x 투구의 수평 위치 plate z 투구의 수직 위치 vx0 투구 속도의 x 좌표 vy0 투구 속도의 y 좌표 vz0 투구 속도의 z 좌표

- ax 투구 가속도의 x 좌표
- ay 투구 가속도의 y 좌표
- az 투구 가속도의 z 좌표
  release spin rate 회전 속도(rpm) spin axis 회전축 spin direction 회전 방향

모든 현장 선수의 신원과 주자의 신원이 포함됩니다. 특정 타석과 관련하여 데이터 세트에는 투구 번호, 볼 및 스트라이크 수, 투수의 타격 측면 및 투구 손이 포함됩니다.

C.4 투구 변수

PITCHf/x 시스템과 유사하게 이 Statcast 데이터 세트에는 각 투구에 대한 정보가 포함되어 있습니다. 표 C.2의 변수에는 투구의 릴리스 포인트, 시간당 마일 단위의 속도, 수평 및 수직 방향의 움직임이 포함됩니다. 구역 내 투구의 위치가 기록되며 구역 변수를 사용하여 특정 영역으로 분류됩니다. 분류 방법을 사용하여 투구 유형이 기록됩니다. 약어 해독은 표 C.3을 참조하십시오.

- 표 C.3 Statcast에서 사용되는 투구 유형 및 투구 이름 변수.

pitch type pitch name CH Changeup CS Slow Curve CU Curveball EP Eephus FA Other FC Cutter FF 4-Seam Fastball FO Forkball FS Split-Finger KC Knuckle Curve KN Knuckleball PO Pitch Out SC Screwball SI Sinker SL Slider ST Sweeper SV Slurve NA NA

투구 변수에 대한 더 자세한 설명은 다음과 같습니다.

- release speed 및 effective speed: 릴리스 포인트와 공이 홈 플레이트 앞부분을 넘을 때의 시간당 마일 단위의 속도.

- sz top 및 sz bot: 현재 타석에 있는 타자의 스트라이크 존 상단과 하단의 수직 좌표입니다. 두 변수 모두 지면에서의 피트 단위로 표현되며 모든 타석 시작 시 수동으로 기록됩니다.

- pfx x 및 pfx z: 스핀으로 인한 움직임이 없는 동일한 속도의 이론적 투구와 비교한 투구의 수평 및 수직 움직임. 두 변수 모두 인치로 측정됩니다.

- plate x 및 plate z: 투구가 홈 플레이트 앞부분을 넘을 때 측정된 투구의 수평 및 수직 위치입니다. 좌표계는 홈 플레이트 중앙과 지면 높이에 중심을 두고 포수/심판의 시점에서 보므로 plate_x의 양수 값은 투구가 홈 플레이트 중앙의 오른쪽으로 지나감을 나타내고 음수 값은 왼쪽을 나타냅니다. plate_z의 음수 값은 투구가 홈 플레이트에 닿기 전에 바운드되었음을 나타냅니다. plate_x 및 plate_z 변수는 모두 피트로 측정됩니다.

- release pos x, release pos y, release pos z: 릴리스 포인트에서 공의 계산된 위치를 나타내는 좌표입니다. release_pos_y

투구 궤적 계산하기 367

매개변수는 홈 플레이트로부터의 거리를 나타내며 일반적으로 홈 플레이트에서 50피트로 설정됩니다. 연구원들은 55피트를 투구의 실제 릴리스 포인트를 더 잘 근사하는 거리로 발견했으며 따라서 C.5절에 설명된 대로 55피트 마크에서 좌표를 다시 계산하는 것이 좋습니다. release_pos_x, release_pos_y 및 release_pos_z는 plate_x 및 plate_z와 동일한 좌표계에서 릴리스 포인트의 왼쪽과 오른쪽 위치 및 높이입니다.

- vx0, vy0, 및 vz0: 초당 피트 단위로 릴리스에서 측정된 투구 속도의 세 가지 차원 구성 요소입니다.
- ax, ay, 및 az: 릴리스에서 ft/s^2로 측정된 투구 가속도의 세 가지 차원 구성 요소입니다.
- release spin rate: 분당 회전 수 단위의 공 회전 속도.

- spin axis: 공 회전의 방향으로, 0°는 완벽한 탑 스핀을 나타내고 180°는 완벽한 바텀 스핀을 나타냅니다.

- C.5 투구 궤적 계산하기

이전 섹션에서 볼 수 있듯이 Statcast는 투구의 위치, 속도 및 가속도에 대한 데이터를 추적합니다. 일정한 가속도에 대한 운동학 방정식을 사용하면 지정된 시간 t에서 공의 위치는 다음 방정식에 의해 결정될 수 있습니다.

- x = x0 + xv0t + 1/2 axt^2
- y = y0 + yv0t + 1/2 ayt^2
- z = z0 + zv0t + 1/2 azt^2

이전 방정식은 다음 함수 pitchloc()을 사용하여 R로 변환됩니다.1

```R
pitchloc <- function(t, x0, ax, vx0, y0, ay, vy0, z0, az, vz0) {

- x <- x0 + vx0 * t + 0.5 * ax * I(t ^ 2)
- y <- y0 + vy0 * t + 0.5 * ay * I(t ^ 2)
- z <- z0 + vz0 * t + 0.5 * az * I(t ^ 2)

```

1이 섹션의 코드는 https://code.google.com/p/rpitchfx/ 에서 약간 수정되었습니다.

```R
if(length(t) == 1) {

loc <- c(x, y, z) } else {

loc <- cbind(x, y, z)

} return(loc)

}
```

pitch_trajectory() 함수는 지정된 시간 간격(인수 interval의 기본 선택은 0.01초)으로 릴리스 포인트에서 홈 플레이트까지의 투구 궤적을 계산합니다.

```R
pitch_trajectory <- function(x0, ax, vx0, y0, ay, vy0, z0, az, vz0, interval = 0.01) {

cross_p <- (-1 * vy0 - sqrt(I(vy0 ^ 2) - 2 * y0 * ay)) / ay tracking <- t(

sapply( seq(0, cross_p, interval), pitchloc,

- x0 = x0, ax = ax, vx0 = vx0,
- y0 = y0, ay = ay, vy0 = vy0,
- z0 = z0, az = az, vz0 = vz0

)

) colnames(tracking) <- c("x", "y", "z") tracking <- data.frame(tracking) return(tracking)

}
```

- C.6 플레이 이벤트 변수

데이터 세트의 각 행은 투구를 나타내지만 표 C.4의 몇 가지 변수는 타석의 결과를 기록합니다. 유형 변수는 공이 스트라이크인지 볼인지 인플레이 타구인지 나타냅니다. events, des 및 description 변수는 타석 결과에 대한 설명을 제공합니다.

표 C.4 플레이 이벤트 변수.

이름 설명 type 볼 또는 스트라이크 또는 인플레이 타구 events 타석의 결과 des 타석 결과에 대한 자세한 설명

타구 변수 369

- 표 C.5 타구 변수.

이름 설명 hit distance sc 공이 착지한 곳까지 떨어진 거리(ft.)

- hc x 공이 착지한 위치의 x 위치

- hc y 공이 착지한 위치의 y 위치
  launch speed 타구에서 벗어날 때 공의 속도 launch angle 타석에서 벗어날 때 공의 수직 각도 barrel 비교 가능한 타구 유형이 최소 .500 타율 및 1.500 장타율을 이끈 타구 이벤트에 대한 분류

- C.7 타구 변수

Statcast 데이터 세트의 특별한 측면 중 하나는 표 C.5에 설명된 대로 인플레이 타구에 대한 변수가 포함되어 있다는 것입니다. 이러한 변수에는 방망이를 떠날 때의 탈출 속도 및 발사 각도, 타구 위치의 (x, y) 좌표 및 홈 플레이트에서의 예상 거리가 포함됩니다. 배럴은 이탈 속도와 발사 각도의 좋은 조합으로 잘 맞은 타구를 범주화하는 방법입니다.

타구 위치 변수 hc_x 및 hc_y는 다음 방정식으로 스프레이 각도 ϕ와 관련이 있습니다.

ϕ = atan((hc_x - 125.42) / (198.27 - hc_y))

이를 그림 C.1에 그래픽으로 표시합니다.

- C.8 파 파생 변수

타구 변수를 기반으로 Statcast는 표 C.6에 표시된 대로 특정 타구의 품질을 이해하는 데 도움이 되는 몇 가지 지표를 개발했습니다. 발사 속도 및 발사 각도를 기반으로 하나의 변수 estimated_ba_using_speedangle은 기본 안타의 예상 확률을 제공하고, 두 번째 변수 estimated_woba_using_speedangle은 이 타구에 대한 가중 출루율의 추정치를 제공합니다.

![image 114](images/imageFile114.png)

그림 C.1 Statcast 변수 hc_x 및 hc_y와 스프레이 각도 ϕ의 관계.

- 표 C.6 Statcast 파생 변수.

이름 설명 estimated ba using speedangle 예상 안타 확률 estimated woba using speedangle 예상 가중 출루율 값

- C.9 수비 변수

Statcast에는 표 C.7에 표시된 대로 팀의 수비 정렬에 대한 정보도 포함됩니다. if_fielding_alignment 변수는 수비 내야가 "표준", "내야 시프트"(2루의 같은 쪽에 3명 이상의 내야수) 또는 "전략적 위치"인지 여부를 나타냅니다. of_fielding_alignment는 "표준", "전략적" 또는 "4번째 외야수"일 수 있습니다. 현재 이러한 새로운 수비 정렬의 가치에 대한 논쟁이 있으며 이러한 변수를 포함하면 이러한 전략의 효과를 결정하는 데 도움이 될 수 있습니다.

Statcast 데이터 수집 371 표 C.7 Statcast 수비 정렬 변수.

이름 설명 if fielding alignment 내야 위치 지정 of fielding alignment 외야 위치 지정

C.10 Statcast 데이터 얻기

baseballr 패키지의 statcast_search() 함수를 사용하면 특정 기간 동안 또는 특정 선수에 대해 Baseball Savant에서 Statcast 데이터를 다운로드할 수 있습니다. 예를 들어 Andrew McCutchen, Freddie Freeman 및 Jose Altuve는 각각 2023년 6월 11일, 6월 25일 및 8월 19일에 개인 통산 2000번째 안타를 기록했습니다. McCutchen의 안타 전후 3일 동안의 데이터를 검색하려면 statcast_search() 함수를 사용할 수 있습니다. McCutchen의 MLB 선수 식별자를 찾는 방법에는 여러 가지가 있으며(C.2절 참조), 이 경우 457705입니다.

```R
library(baseballr) mccutchen <- statcast_search(

start_date = "2023-06-08", end_date = "2023-06-14", playerid = 457705

) mccutchen |>

filter(game_date == "2023-06-11", events == "single") |> select(pitch_type, release_speed, release_spin_rate)

# A tibble: 1 x 3

pitch_type release_speed release_spin_rate <chr> <dbl> <dbl>

1 SL 85.8 2502
```

McCutchen의 2000번째 안타는 분당 2500회 회전하는 86mph 슬라이더에서 나왔습니다.

1년 이상의 Statcast 데이터를 저장하는 방법에 대한 자세한 내용은 12.2절을 참조하십시오.

![image 115](images/imageFile115.png)
