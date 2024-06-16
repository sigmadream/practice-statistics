14

Quarto를 사용하여 과학적 프레젠테이션 만들기

Quarto는 R(또는 Python) 코드의 동적 계산을 포함하는 다양한 종류의 문서를 생성하기 위한 새로운 저작 시스템입니다. 여러 면에서 Quarto는 R Markdown의 자연스러운 후속작입니다. 이 책의 이전 판은 LATEX로 작성되었으며, knitr 패키지가 R 코드를 LATEX로 렌더링하는 메커니즘을 제공했고, 그 후 PDF로 컴파일되었습니다. 그러나 이 책은 Quarto로 작성되었습니다. 이러한 변화로 인해 웹사이트용 HTML과 인쇄용 PDF와 같이 책을 여러 형식으로 동시에 렌더링할 수 있습니다.

위에서 언급했듯이 Quarto 문서는 인기 있는 JavaScript 라이브러리인 reveal.js를 기반으로 하는 HTML 프레젠테이션을 포함하여 다양한 형식으로 렌더링할 수 있습니다. 이 장에서는 Quarto를 사용하여 전문적인 품질의 재현 가능한 과학적 프레젠테이션을 만드는 방법을 설명합니다.

14.1 Quarto 슬라이드 소개

모든 Markdown 문서와 마찬가지로 Quarto 문서는 YAML 헤더가 있고 그 뒤에 다양한 Markdown 코드 섹션이 옵니다. 아래 예시에서 YAML 헤더는 문서의 맨 위에 나타나며 첫 번째 줄의 ---와 네 번째 줄의 ---로 묶여 있습니다. YAML 섹션에서 다양한 옵션을 지정할 수 있습니다. 이 경우에는 저자, 제목, 그리고 (중요하게) 형식을 지정합니다. revealjs가 format 안에 중첩되어 있고, incremental이 revealjs 안에 중첩되어 있다는 점에 유의하시기 바랍니다. 이는 revealjs 형식 내에서 사용할 수 있는 incremental 속성을 true로 설정하겠다는 의미입니다. Quarto에서는 여러 형식에 대해 서로 다른 옵션을 동시에 지정하는 것이 가능합니다.

--title: "Home Run Hitting" author: "Jim Albert" format:

###### DOI: 10.1201/9781032668239-14 324

- 14


Quarto를 사용하여 과학적 프레젠테이션 만들기

Quarto는 R(또는 Python) 코드의 동적 계산을 포함하는 다양한 종류의 문서를 생성하기 위한 새로운 저작 시스템입니다. 여러 면에서 Quarto는 R Markdown의 자연스러운 후속작입니다. 이 책의 이전 판은 LATEX로 작성되었으며, knitr 패키지가 R 코드를 LATEX로 렌더링하는 메커니즘을 제공했고, 그 후 PDF로 컴파일되었습니다. 그러나 이 책은 Quarto로 작성되었습니다. 이러한 변화로 인해 웹사이트용 HTML과 인쇄용 PDF와 같이 책을 여러 형식으로 동시에 렌더링할 수 있습니다.

위에서 언급했듯이 Quarto 문서는 인기 있는 JavaScript 라이브러리인 reveal.js를 기반으로 하는 HTML 프레젠테이션을 포함하여 다양한 형식으로 렌더링할 수 있습니다. 이 장에서는 Quarto를 사용하여 전문적인 품질의 재현 가능한 과학적 프레젠테이션을 만드는 방법을 설명합니다.

14.1 Quarto 슬라이드 소개

모든 Markdown 문서와 마찬가지로 Quarto 문서는 YAML 헤더가 있고 그 뒤에 다양한 Markdown 코드 섹션이 옵니다. 아래 예시에서 YAML 헤더는 문서의 맨 위에 나타나며 첫 번째 줄의 ---와 네 번째 줄의 ---로 묶여 있습니다. YAML 섹션에서 다양한 옵션을 지정할 수 있습니다. 이 경우에는 저자, 제목, 그리고 (중요하게) 형식을 지정합니다. revealjs가 format 안에 중첩되어 있고, incremental이 revealjs 안에 중첩되어 있다는 점에 유의하시기 바랍니다. 이는 revealjs 형식 내에서 사용할 수 있는 incremental 속성을 true로 설정하겠다는 의미입니다. Quarto에서는 여러 형식에 대해 서로 다른 옵션을 동시에 지정하는 것이 가능합니다.

--title: "Home Run Hitting" author: "Jim Albert" format:

DOI: 10.1201/9781032668239-14 324

예시: 홈런 타격에 관한 Jim의 프레젠테이션 325

revealjs: incremental: true

---

다음으로, Markdown 섹션 헤더와 일반 Markdown 콘텐츠를 사용하여 슬라이드를 만듭니다. 예를 들어, 다음 Markdown 코드는 Mickey Mantle과 Aaron Judge 두 명의 선수 목록이 포함된 "Baseball"이라는 제목의 슬라이드를 생성합니다.

## Baseball

- - Mickey Mantle
- - Aaron Judge


14.2절에서는 이러한 간단한 원칙의 확장을 기반으로 복잡한 슬라이드 세트를 만듭니다.

14.2 예시: 홈런 타격에 관한 Jim의 프레젠테이션

이 절에서는 Quarto로 구성되고 reveal.js 프레임워크를 사용하여 HTML로 출력되는 과학적 프레젠테이션을 만드는 과정을 안내해 드립니다. Jim이 발표한 전체 프레젠테이션은 https://bayesball.github.io/homerun_talk/homeruns.html에서 볼 수 있습니다. 여기 제시된 내용과 일치하도록 약간 수정된 버전은 https://beanumber.github.io/abdwr3e/revealjs/hr_pres.html에서 볼 수 있습니다.


- 14.2.1 프레젠테이션의 섹션


먼저, 야구 역사상 홈런 타격의 일반적인 패턴을 소개하는 섹션, 2017년 MLB 홈런 위원회의 연구 결과를 설명하는 섹션(Albert 외 2018), 마지막으로 스탯캐스트 시대 동안 홈런 타격 패턴의 최근 변화를 설명하는 섹션으로 프레젠테이션을 세 부분으로 나누기로 결정합니다.

레벨 1 제목(#)을 사용하여 이러한 섹션 제목이 있는 별도의 슬라이드를 만듭니다.

# Introduction # What is Causing the Increase in Home Rate Rates? # Recent Exploration of Home Run Rates

326 Quarto를 사용하여 과학적 프레젠테이션 만들기

- 14.2.2 R 출력 포함하기


프레젠테이션을 만들 때 Quarto를 사용하는 매력적인 측면 중 하나는 R 코드와 출력을 문서에 통합할 수 있는 기능입니다. 예를 들어, 시즌별 팀당 홈런 수를 그래프로 표시하여 메이저 리그 야구 역사에 걸친 홈런 타격의 극적인 변화를 설명하는 데 관심이 있다고 가정해 봅니다. 이 데이터를 검색하기 위해 Lahman 데이터베이스를 사용합니다.

Quarto 문서에서 ```{r} 및 ``` 기호로 구분된 청크 내에 R 코드를 표시합니다. 컴파일된 문서에는 아래에 표시된 ggplot2 그래프가 포함됩니다. echo: true 옵션을 사용하면 컴파일된 문서에 R 코드가 포함됩니다. 프레젠테이션의 경우, 코드가 표시되지 않도록 일반적으로 echo: false 옵션을 사용합니다. 전체 슬라이드는 Jim의 프레젠테이션에서 볼 수 있습니다.

```{r} #| echo: true library(Lahman) br <- Batting |>

group_by(Year = yearID) |> summarize(HR = sum(HR)) ggplot(br, aes(Year, HR)) + geom_point(color = "black", size = 2) + geom_smooth(

formula = "y ~ x", color = "blue", se = FALSE, method = "loess", span = 0.20, linewidth = 1.5

) + labs(x = "Season", y = "Avg HR") + theme(text = element_text(size = 22))

```

![image 96](images/imageFile96.png)

326 Quarto를 사용하여 과학적 프레젠테이션 만들기

- 14.2.2 R 출력 포함하기


프레젠테이션을 만들 때 Quarto를 사용하는 매력적인 측면 중 하나는 R 코드와 출력을 문서에 통합할 수 있는 기능입니다. 예를 들어, 시즌별 팀당 홈런 수를 그래프로 표시하여 메이저 리그 야구 역사에 걸친 홈런 타격의 극적인 변화를 설명하는 데 관심이 있다고 가정해 봅니다. 이 데이터를 검색하기 위해 Lahman 데이터베이스를 사용합니다.

Quarto 문서에서 ```{r} 및 ``` 기호로 구분된 청크 내에 R 코드를 표시합니다. 컴파일된 문서에는 아래에 표시된 ggplot2 그래프가 포함됩니다. echo: true 옵션을 사용하면 컴파일된 문서에 R 코드가 포함됩니다. 프레젠테이션의 경우, 코드가 표시되지 않도록 일반적으로 echo: false 옵션을 사용합니다. 전체 슬라이드는 Jim의 프레젠테이션에서 볼 수 있습니다.

```{r} #| echo: true library(Lahman) br <- Batting |>

group_by(Year = yearID) |> summarize(HR = sum(HR)) ggplot(br, aes(Year, HR)) + geom_point(color = "black", size = 2) + geom_smooth(

formula = "y ~ x", color = "blue", se = FALSE, method = "loess", span = 0.20, linewidth = 1.5

) + labs(x = "Season", y = "Avg HR") + theme(text = element_text(size = 22))

```

예시: 홈런 타격에 관한 Jim의 프레젠테이션 327

- 14.2.3 여러 열 및 이미지 추가

프레젠테이션의 일부는 야구 역사상 유명한 홈런 타자 중 일부를 설명합니다. 아래 URL의 이미지 파일에는 Home Run Baker의 사진이 포함되어 있습니다. LATEX의 Beamer 프레젠테이션(14.3절 참조)과 달리 reveal.js 프레젠테이션의 이미지는 인터넷에서 직접 가져올 수 있다는 점에 유의하시기 바랍니다. url이 이미지가 포함된 폴더의 위치를 나타내는 ![](url) 구문을 사용하여 프레젠테이션에 이 이미지를 포함합니다.

Quarto 코드는 또한 왼쪽 열에 이미지가 포함되어 있고 오른쪽 열에 플레이어에 대한 간략한 설명이 있는 2열 형식의 사용을 보여줍니다. .column width 인수는 왼쪽 및 오른쪽 열의 너비에 대한 백분율을 설명합니다.

## Home Run Baker :::: columns ::: {.column width="40%"} ::: ::: {.column width="60%"}

- 데드볼 시대에 뛰었음

- - 1914년에 9개의 홈런으로 홈런 1위였음
- - 홈런은 경기의 큰 부분을 차지하지 않았음 ::: ::::


그림 14.1은 컴파일된 프레젠테이션의 완성된 슬라이드를 보여줍니다.

- 14.2.4 테이블 포함하기


###### ![](https://live.staticflickr.com/6/12033666_c111eb7fab_z.jpg)

표준 Markdown 방법을 사용하여 프레젠테이션에 테이블을 쉽게 포함할 수 있습니다. 여기에서는 일반적인 Markdown 파이프 형식을 사용하여 2015년에서 2022년까지의 총 홈런 수를 표시하는 테이블을 만듭니다. 결과 슬라이드는 그림 14.2에 나와 있습니다.

## Home Run Totals in the Statcast Era

| Season | HR Total | |--------|----------| | 2015 | 4909 | | 2016 | 5610 | | 2017 | 6105 | | 2018 | 5585 | | 2019 | 6776 |

![image 97](images/imageFile97.png)

![image 98](images/imageFile98.png)

![image 99](images/imageFile99.png)

- 그림 14.1 Home Run Baker를 보여주는 Jim의 프레젠테이션 슬라이드.

![image 100](images/imageFile100.png)

![image 101](images/imageFile101.png)

- 그림 14.2 홈런 수를 보여주는 테이블이 있는 Jim의 프레젠테이션 슬라이드.


| 2021 | 5944 | | 2022 | 5215 |

테이블을 표시하는 대안적인 방법으로 knitr 패키지의 kable() 함수를 사용합니다. 테이블 데이터는 tibble()

- 그림 14.1 Home Run Baker를 보여주는 Jim의 프레젠테이션 슬라이드.


- 그림 14.2 홈런 수를 보여주는 테이블이 있는 Jim의 프레젠테이션 슬라이드.


| 2021 | 5944 | | 2022 | 5215 |

테이블을 표시하는 대안적인 방법으로 knitr 패키지의 kable() 함수를 사용합니다. 테이블 데이터는 tibble()

예시: 홈런 타격에 관한 Jim의 프레젠테이션 329 표 14.1 kable()을 사용하여 렌더링된 홈런 수 테이블.

Season Home runs

2015 4909 2016 5610 2017 6105 2018 5585 2019 6776

- 2021 5944
- 2022 5215


함수를 사용하여 데이터 프레임에 배치됩니다. kable() 함수에서 "simple" 형식을 선택하고 양쪽 열을 가운데 정렬하며 열의 특별한 이름을 추가합니다. 아래는 이 R 작업을 구현하는 코드를 보여주고 표 14.1에 출력을 표시합니다.

```{r} #| echo: false df <- tibble(

Season = c(2015, 2016, 2017, 2018, 2019, 2021, 2022), HR_Total = c(4909, 5610, 6105, 5585, 6776, 5944, 5215)

) df |>

knitr::kable( "simple", align = "cc", col.names = c("Season", "Home Runs")

) ```

- 14.2.5 LATEX 포함하기 Quarto 문서의 또 다른 매력적인 기능은 수식을 표시하기 위한 전통적인 시스템인 LATEX를 통합할 수 있다는 점입니다.


LATEX 수식은 $$ 구분 기호 안에 배치할 수 있습니다. 여기에서는 인플레이 홈런 비율 공식을 표시하기 위해 LATEX를 사용합니다. Quarto 코드 아래 그림 14.3은 수학 식이 포함된 슬라이드의 스냅샷을 보여줍니다.

## In-Play Rates

- 홈런 비율을 모든 타구($AB - SO$) 중 $HR$의 비율로 정의합니다.

$$ HR \, Rate = \frac{HR}{AB - SO}

$$

- $HR$ 비율의 기록을 살펴봅니다.

![image 102](images/imageFile102.png)

그림 14.3 LaTeX로 설정된 수식을 보여주는 Jim의 프레젠테이션 슬라이드.

- 14.2.6 revealjs 형식을 사용한 옵션


revealjs 형식(reveal.js 프레젠테이션을 생성함)은 다양한 옵션을 허용합니다. 웹사이트 https://quarto.org/docs/presentations/revealjs/ 에서는 이 특정 프레젠테이션 형식의 모든 기능에 대한 개요를 제공합니다.

애니메이션 생성, 확대 및 축소, 슬라이드 번호 포함, PDF로 인쇄, 슬라이드에 그릴 수 있는 칠판 기능을 포함하여 다양한 고급 기능이 있습니다.

Reveal 프레젠테이션을 위한 11개의 내장 테마가 있으며, 지금까지 기본 테마의 스냅샷이 제공되었습니다. 다음 YAML 헤더를 사용하여 sky 형식을 사용할 수 있습니다. 그림 14.4에서는 이 테마를 사용한 Home Run Baker 슬라이드의 스냅샷이 표시됩니다.

--title: "Home Run Hitting" author: "Jim Albert" format:

revealjs: incremental: true theme: sky

---

$$

- $HR$ 비율의 기록을 살펴봅니다.

- 그림 14.3 LaTeX로 설정된 수식을 보여주는 Jim의 프레젠테이션 슬라이드.


- 14.2.6 revealjs 형식을 사용한 옵션


revealjs 형식(reveal.js 프레젠테이션을 생성함)은 다양한 옵션을 허용합니다. 웹사이트 https://quarto.org/docs/presentations/revealjs/ 에서는 이 특정 프레젠테이션 형식의 모든 기능에 대한 개요를 제공합니다.

애니메이션 생성, 확대 및 축소, 슬라이드 번호 포함, PDF로 인쇄, 슬라이드에 그릴 수 있는 칠판 기능을 포함하여 다양한 고급 기능이 있습니다.

Reveal 프레젠테이션을 위한 11개의 내장 테마가 있으며, 지금까지 기본 테마의 스냅샷이 제공되었습니다. 다음 YAML 헤더를 사용하여 sky 형식을 사용할 수 있습니다. 그림 14.4에서는 이 테마를 사용한 Home Run Baker 슬라이드의 스냅샷이 표시됩니다.

--title: "Home Run Hitting" author: "Jim Albert" format:

revealjs: incremental: true theme: sky

---

대안적인 출력 형식 331

![image 103](images/imageFile103.png)

![image 104](images/imageFile104.png)

![image 105](images/imageFile105.png)

그림 14.4 sky 테마를 사용하고 Home Run Baker를 보여주는 Jim의 프레젠테이션 슬라이드.

###### 14.3 대안적인 출력 형식

프레젠테이션 출력이 HTML 파일인 revealjs 출력 형식을 사용하는 데 중점을 두었습니다. 다른 인기 있는 형식으로는 Beamer(출력이 PDF 파일인 경우)와 PowerPoint(출력이 .pptx 파일인 경우)가 있습니다.

이러한 대안적인 형식은 YAML 헤더를 약간만 변경하면 됩니다. 예를 들어 Beamer LATEX 클래스를 사용하여 홈런 프레젠테이션의 PDF를 만들고자 한다고 가정합니다. YAML 헤더에서 beamer 클래스를 사용하고자 하는 format에 대한 옵션으로 지정합니다. beamer에 대한 옵션에서 점진적 목록을 원하고 seahorse 색상 테마와 함께 Boadilla beamer 테마를 사용한다고 나타냅니다.

--title: "Home Run Hitting" author: Jim Albert format:

beamer: incremental: true theme: Boadilla colortheme: seahorse

---

332 Quarto를 사용하여 과학적 프레젠테이션 만들기 대신에 PowerPoint 출력을 원한다고 가정합니다. 형식이 점진적 목록을 포함하는 pptx임을 아래에 나타냅니다. 프레젠테이션 내용에 대한 다른 변경은 필요하지 않습니다!

--title: "Home Run Hitting" author: Jim Albert format:

pptx: incremental: true

---

기본적으로 PowerPoint는 비교적 평범해 보이는 템플릿을 사용합니다. PowerPoint 템플릿을 만들고 reference-doc 옵션을 사용하여 이 템플릿을 적용함으로써 이를 수정할 수 있습니다.

- 14.4 추가 읽을거리

공식 Quarto 문서는 프레젠테이션을 만들기 위한 HTML, PowerPoint 및 Beamer 형식에 대한 소개를 제공합니다. reveal.js 사이트는 reveal.js 프레젠테이션 프레임워크의 사용에 대한 자세한 내용을 제공합니다.

- 14.5 연습문제


- 1. 최근 몇 년간의 홈런 타격

Lahman 패키지를 사용하여 2015-2022년에 친 총 홈런 수의 테이블을 생성하는 R 코드를 작성합니다. 14.2.4절에서 했던 것처럼 슬라이드에 테이블을 넣습니다.

- 2. 최근 몇 년간의 홈런 타격(계속)

Lahman 패키지를 사용하여 2000-2022년 시즌에 대한 인플레이 타구 홈런 비율(HR / (AB - SO))을 시즌의 함수로 표시하는 R 코드를 작성합니다. 14.2.2절에서 했던 것처럼 슬라이드에 그림을 넣습니다.

- 3. 그 선수가 명예의 전당에 들어가야 할까요?


야구 명예의 전당에 들어가야 한다고 생각하는 선수를 선택합니다. 선택한 선수와 현재 명예의 전당에 있는 같은 수비 위치의 다른 선수를 비교하는 프레젠테이션을 만듭니다. 프레젠테이션에는 두 선수의 그래프, 표 및 이미지가 포함되어야 합니다.

332 Quarto를 사용하여 과학적 프레젠테이션 만들기 대신에 PowerPoint 출력을 원한다고 가정합니다. 형식이 점진적 목록을 포함하는 pptx임을 아래에 나타냅니다. 프레젠테이션 내용에 대한 다른 변경은 필요하지 않습니다!

--title: "Home Run Hitting" author: Jim Albert format:

pptx: incremental: true

---

기본적으로 PowerPoint는 비교적 평범해 보이는 템플릿을 사용합니다. PowerPoint 템플릿을 만들고 reference-doc 옵션을 사용하여 이 템플릿을 적용함으로써 이를 수정할 수 있습니다.

- 14.4 추가 읽을거리

공식 Quarto 문서는 프레젠테이션을 만들기 위한 HTML, PowerPoint 및 Beamer 형식에 대한 소개를 제공합니다. reveal.js 사이트는 reveal.js 프레젠테이션 프레임워크의 사용에 대한 자세한 내용을 제공합니다.

- 14.5 연습문제


- 1. 최근 몇 년간의 홈런 타격

Lahman 패키지를 사용하여 2015-2022년에 친 총 홈런 수의 테이블을 생성하는 R 코드를 작성합니다. 14.2.4절에서 했던 것처럼 슬라이드에 테이블을 넣습니다.

- 2. 최근 몇 년간의 홈런 타격(계속)

Lahman 패키지를 사용하여 2000-2022년 시즌에 대한 인플레이 타구 홈런 비율(HR / (AB - SO))을 시즌의 함수로 표시하는 R 코드를 작성합니다. 14.2.2절에서 했던 것처럼 슬라이드에 그림을 넣습니다.

- 3. 그 선수가 명예의 전당에 들어가야 할까요?


야구 명예의 전당에 들어가야 한다고 생각하는 선수를 선택합니다. 선택한 선수와 현재 명예의 전당에 있는 같은 수비 위치의 다른 선수를 비교하는 프레젠테이션을 만듭니다. 프레젠테이션에는 두 선수의 그래프, 표 및 이미지가 포함되어야 합니다.

연습문제 333

4. 리더보드 프레젠테이션

일부 타격 또는 투구 지표와 관련하여 통산 상위 10명의 리더를 제시하는 프레젠테이션을 만듭니다. 이 프레젠테이션에는 10명의 선수에 대해 경력 중간 시즌과 통산 지표를 비교하여 표시하는 R 산점도가 포함되어야 합니다. 또한 선수에 대한 정보와 함께 리더보드에 있는 10명의 선수의 이미지를 포함합니다.

