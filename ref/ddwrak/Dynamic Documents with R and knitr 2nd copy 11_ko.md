### 12

###### 요령 및 해결 방법

이 장에서는 보고서를 보다 쉽고 빠르게 작성하고 컴파일하는 데 유용한 몇 가지 요령과 자주 묻는 질문에 대한 해결 방법을 소개합니다.

###### 12.1 청크 옵션

knitr에는 여러 가지 내장 청크 옵션이 있으며, 일반적으로 청크 헤더에서 값을 할당합니다. 하지만 이러한 고정 옵션을 사용자 정의하거나 옵션의 이름을 변경하는 것도 가능합니다.

###### 12.1.1 옵션 별칭

자주 사용하지만 입력하기에는 이름이 너무 긴 옵션들이 있을 수 있습니다. 이 경우 문서 시작 부분에서 `set_alias()` 함수를 사용하여 청크 옵션에 대한 별칭을 설정할 수 있습니다. 예를 들면 다음과 같습니다.

set_alias(w = "fig.width", h = "fig.height")

그러면 다음과 같이 그림 너비와 높이에 각각 `w`와 `h`를 사용할 수 있습니다. `<<fig-size, w=5, h=3>>= plot(1:10) @`

위의 청크는 다음과 동일합니다.

`<<fig-size, fig.width=5, fig.height=3>>= plot(1:10) @`

127

###### 12.1.2 옵션 템플릿

옵션 이름 외에도 자주 사용하는 옵션 값들을 템플릿으로 함께 묶을 수 있습니다. knitr의 `opts_template` 객체를 사용하여 이러한 템플릿을 만들 수 있습니다. 템플릿은 이름이 지정된 옵션 세트 모음입니다. 예를 들어, 그래픽 장치 크기를 7 × 5 인치로 설정하려는 플롯이 많고, 다른 플롯은 3.5 × 3 인치로 설정하려는 경우가 있습니다. 첫 번째 플롯 그룹에는 `fig.width = 7`, `fig.height = 5`를 입력하고 두 번째 그룹에는 `fig.width = 3.5`, `fig.height = 3`을 입력할 수 있지만, 옵션 별칭을 사용하더라도 이는 번거로울 수 있습니다. 이 경우 두 옵션 세트를 다음과 같이 템플릿에 넣으면 됩니다.

opts_template$set( fig.large = list(fig.width = 7, fig.height = 5), fig.small = list(fig.width = 3.5, fig.height = 3)

)

템플릿을 설정한 후에는 향후 청크 헤더에서 청크 옵션 `opts.label`을 사용하여 템플릿을 참조할 수 있습니다. 예를 들어, 아래 청크에서 큰 플롯에 대한 옵션을 원한다면 다음과 같이 작성합니다.

`<<fig-ex, opts.label= fig.large >>= plot(1:10) @`

이것은 다음과 동일합니다.

`<<fig-ex, fig.width=7, fig.height=7>>= plot(1:10) @`

###### 12.1.3 프로그래밍 가능한 청크 옵션

청크 옵션은 임의의 R 표현식을 취할 수 있으므로 숫자나 논리값 같은 고정된 값을 설정하는 것 외에도 청크 옵션을 프로그래밍할 수 있습니다. 아래에 gridExtra 패키지를 사용하여 표를 그리는 예를 보여드립니다. 먼저 `tableGrob()` 함수를 사용하여 표 Grob(그래픽 객체)을 만듭니다.

library(gridExtra)

###### g <- tableGrob(head(iris))

|     | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species |
| --- | ------------ | ----------- | ------------ | ----------- | ------- |
| 1   | 5.1          | 3.5         | 1.4          | 0.2         | setosa  |
| 2   | 4.9          | 3.0         | 1.4          | 0.2         | setosa  |
| 3   | 4.7          | 3.2         | 1.3          | 0.2         | setosa  |
| 4   | 4.6          | 3.1         | 1.5          | 0.2         | setosa  |
| 5   | 5.0          | 3.6         | 1.4          | 0.2         | setosa  |
| 6   | 5.4          | 3.9         | 1.7          | 0.4         | setosa  |

- 그림 12.1: gridExtra 패키지로 생성된 표: 표 Grob을 만들고 적절한 그래픽 장치에 그립니다.

다음으로, grid 패키지의 `grid.draw()`를 사용하여 객체를 플롯에 그립니다. 그 전에 그래픽 장치의 적절한 크기를 결정해야 합니다. 그렇지 않으면 플롯에 여분의 여백이 생길 수 있습니다. 실제로 grid 패키지의 `convertWidth()` 및 `convertHeight()` 함수는 미리 계산된 Grob의 너비와 높이를 인치로 변환할 수 있습니다. 따라서 일반적으로 사용하는 고정 숫자 대신 `fig.width`와 `fig.height` 청크 옵션에 두 개의 함수 호출을 전달합니다. 그림 12.1은 `grid.draw()`로 그린 붓꽃(iris) 데이터의 처음 4줄 표입니다.

`<<table, fig.width=convertWidth(grobWidth(g),  in , TRUE)>>= ## width and height in inches convertWidth(grobWidth(g), "in", value = TRUE)`

`## [1] 5.55 convertHeight(grobHeight(g), "in", value = TRUE) ## [1] 1.94 grid.draw(g) @`

프로그래밍 가능한 청크 옵션을 통해 여러 측면에서 보고서를 프로그래밍할 수 있습니다. 한 가지 잠재적인 응용으로, 각 절차를 하위 문서(섹션 9.3)에 포함하여 일반적인 진단 절차를 포함하는 선형 회귀 보고서를 작성할 수 있습니다. 그런 다음 특정 조건에 따라 특정 절차를 포함할지 여부를 결정할 수 있습니다. 예를 들어, 회귀 모델에서 이상치를 감지한 경우 이상치를 처리하는 모듈을 포함합니다. 아래 청크는 이 아이디어의 스케치를 보여줍니다.

`<<cooks-distance>>= cookd <- cooks.distance(fit) # include an outlier procedure if any distance is # greater than 1 <<outlier, child=if (any(cookd > 1))  outlier.Rnw >>= @`

###### 12.1.4 부록의 코드

때로는 보고서 본문에 코드 청크를 표시하고 싶지 않지만 코드 전체를 숨기고 싶지도 않은 경우가 있습니다. 이 경우 모든 코드 청크를 부록으로 이동할 수 있으며, 여기서 `ref.label` 청크 옵션이 유용하게 사용될 수 있습니다(섹션 9.1.2).

문서에 코드 청크가 소수만 있는 경우 다음과 같이 레이블을 수동으로 입력할 수 있습니다.

- `<<A, echo=FALSE>>=`

- 1+1

`<<B, echo=FALSE>>=`

- 2+2

- `<<C, echo=FALSE>>= rnorm(10) <<show-code, ref.label=c( A ,  B ,  C ), eval=FALSE>>= @`

여기서는 `echo = FALSE`를 통해 이전 청크의 코드를 숨기고 `ref.label`을 통해 마지막 청크에 모읍니다. 코드가 다시 평가되지 않도록 마지막 청크에 `eval = FALSE` 청크 옵션을 사용한 것에 유의하시기 바랍니다.

문서에 코드 청크가 많은 경우, knitr의 `all_labels()` 함수를 사용하여 문서의 모든 청크 레이블을 가져와서 `ref.label`에 전달할 수 있습니다. 예를 들면 다음과 같습니다.

`<<show-code, ref.label=all_labels()>>= @`

`opts_chunk$set()`으로 `echo = FALSE`를 전역적으로 설정하고, 마지막 청크에 `echo = TRUE`를 사용하여 거기에 코드를 표시할 수 있습니다. 물론 포함할 청크 레이블을 선택할 수도 있습니다. 예를 들어 `all_labels()[-1]`을 사용하여 첫 번째 청크를 제거할 수 있습니다.

###### 12.1.5 로컬 R 옵션

청크 옵션 `R.options`는 코드 청크를 위해 `options()`에 전달할 R 옵션 목록을 받을 수 있습니다. 이러한 옵션은 코드 청크에 적용되고 청크가 끝난 후 복원되므로 특정 코드 청크에 대해 일시적으로 R 옵션을 변경하려는 경우 유용할 수 있습니다.

예를 들어 다음 코드 청크에 로컬 옵션 `width = 30`(출력의 대략적인 너비) 및 `digits = 2`(출력 자릿수)를 사용합니다.

`<<R.options = list(width=30, digits=2)>>= seq(0, 10, length = 20)`

`## [1] 0.00 0.53 1.05 1.58 ## [5] 2.11 2.63 3.16 3.68 ## [9] 4.21 4.74 5.26 5.79 ## [13] 6.32 6.84 7.37 7.89`

- `## [17] 8.42 8.95 9.47 10.00 @`

###### 12.1.6 동적 코드

일반적으로 청크에 코드를 입력하거나 참조를 통해 다른 청크의 코드를 포함합니다(9장). `code`라는 청크 옵션을 사용하여 청크에 코드를 할당하는 또 다른 방법이 있습니다. 이렇게 하면 코드 청크를 동적으로 생성할 수 있습니다. 예를 들어 외부 스크립트에서 코드를 읽을 수 있습니다.

`<<code = readLines( foo.R )>>= @`

###### 12.2 패키지 옵션

이전에 구체적으로 언급하지는 않았지만, knitr에는 일부 패키지 수준 옵션을 제어하는 `opts_knit`이라는 개체가 있으며 그 사용법은 청크 옵션(`opts_chunk`)과 동일합니다.

기본적으로 knitr를 호출하면 진행률 표시줄이 나타나며, `opts_knit$set(progress = FALSE)`를 설정하여 표시하지 않을 수 있습니다. 진행률 표시줄은 `knit()`의 진행률을 보여주어 상대적으로 긴 시간이 걸리는 경우 현재 어떤 청크가 컴파일되고 있는지 알 수 있습니다. 소스 코드 등 청크에 대한 자세한 정보를 보려면 `opts_knit$set(verbose = TRUE)`로 장황한(verbose) 모드를 켤 수 있습니다.

패키지 옵션 `root.dir`을 사용하면 코드 청크를 평가할 때 루트 작업 디렉터리를 설정할 수 있습니다. 기본 작업 디렉터리는 입력 문서의 디렉터리이지만 이 옵션을 사용하여 변경할 수 있습니다. 예를 들어 다음과 같이 설정하면,

opts_knit$set(root.dir = "/home/foo/bar/")

전체 경로를 사용하지 않고도 해당 디렉터리의 데이터 파일을 읽을 수 있습니다. 하지만 일반적으로 데이터 세트와 소스 문서를 동일한 디렉터리에 두고 이 디렉터리를 작업 디렉터리로 사용하는 것을 권장합니다.

레이블이 없는 청크의 경우 `unnamed-chunk-i` 형태의 자동 레이블이 사용됩니다. 이는 패키지 옵션 `unnamed.chunk.label`을 통해 사용자 정의할 수 있습니다. 예를 들면 다음과 같습니다.

opts_knit$set(unnamed.chunk.label = "fig") 그러면 자동 청크 레이블이 fig-1, fig-2 등이 됩니다.

###### 12.3 조판

이 섹션에서는 보고서의 조판을 조정하는 몇 가지 해결 방법을 보여드립니다.

###### 12.3.1 출력 너비

LATEX에서 knitr를 사용할 때 흔히 발생하는 문제는 출력 너비가 페이지 여백을 초과할 수 있다는 것입니다. 너비에는 소스 코드, 텍스트 출력, 그래픽 출력의 세 가지 유형이 있습니다. 7.4절에서 그래픽 출력이 페이지 너비보다 넓어지지 않도록 보장하는 `\maxwidth`에 대해 언급했습니다.

소스 코드 및 텍스트 출력의 너비는 `options()`의 전역 옵션 `width`에 의해 제어됩니다(6.2.2절 참조). 이 옵션의 기본값은 75이며, 페이지 여백을 재설정하지 않은 경우(geometry 패키지 사용) LATEX 문서에는 다소 클 수 있습니다.

소스 코드나 텍스트 출력이 너무 넓다고 판단되면 더 작은 너비 옵션을 사용할 수 있습니다. 예를 들면 다음과 같습니다.

options(width = 55)

하지만 이 방법이 항상 작동하는 것은 아닙니다. 소스 코드의 경우 R이 소스 줄을 끊을 적절한 위치를 찾지 못할 수 있습니다. 텍스트 출력의 경우 원래 줄에 줄바꿈이 포함되지 않을 수 있습니다(verbatim 환경에 있기 때문에 LATEX가 자동으로 줄을 바꿈하지 않습니다). 아래 예시의 경우 너비 옵션이 아무리 작더라도 텍스트 줄바꿈이 수행되지 않습니다.

`# unable to wrap the source code x <- "thisistoolongandRisunabletofindaplacetoinsertthelinebreak" # unable to wrap the output line cat(x, "---") ## thisistoolongandRisunabletofindaplacetoinsertthelinebreak ---`

이것은 극단적인 예시입니다. 일반적으로 소스 코드는 여러 줄로 포맷할 수 있습니다. 소스 코드에 문자열이 너무 길면 `paste()`를 사용하여 여러 부분으로 수동으로 나누어 붙이는 것을 고려할 수 있습니다. 예를 들면 다음과 같습니다.

`x <- paste("this", "is", "too", "long", "and", "R", "is", "unable", "to", "find", "a", "place", "to", "insert", "the", "line", "break", sep = "")`

대안적인 접근 방식은 listings 스타일을 사용하는 것입니다(그림 5.2 및 `render_listings()` 함수 참조). LATEX 프리앰블에서 listings 패키지의 `breaklines` 옵션을 `true`로 설정할 수 있습니다. `\lstset{breaklines=true}`

LATEX에서 이 옵션을 사용하는 예는 그림 12.2를 참조하시기 바랍니다.

###### 12.3.2 메시지 색상

LATEX 출력의 경우 메시지, 경고 및 오류에 해당하는 세 가지 색상이 정의되어 있습니다.

\definecolor{messagecolor}{rgb}{0, 0, 0} \definecolor{warningcolor}{rgb}{1, 0, 1} \definecolor{errorcolor}{rgb}{1, 0, 0}

기본적으로 메시지는 검은색, 경고는 자홍색, 오류는 빨간색입니다. LATEX 프리앰블에서 `\definecolor{}` 명령을 사용하여 재정의할 수 있습니다.

| breaklines 옵션을 true로 설정하여 긴 줄을 넘길 수 있습니다.<br><br>print (” a s d l f j k s a d f l k j kljsd klwjr klwjre klwjer kljwre kljwer lkjrwee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf ”)<br><br> | [1] "asdlfjk sadflkj kljsd klwjr klwjre klwjer kljwre kljwer lkjrwee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf"<br><br> |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <br><br>비교를 위해 breaklines=false인 경우를 보여드립니다.<br><br>print (” a s d l f j k s a d f l k j kljsd klwjr klwjre klwjer kljwre kljwer lkjrw<br><br>                                                                                | [1] "asdlfjk sadflkj kljsd klwjr klwjre klwjer kljwre kljwer lkjrwee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf"<br><br> |
| ---                                                                                                                                                                                                                                          |

|
|---|

ee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf ”)

- 그림 12.2: listings로 긴 줄 나누기: R의 `render_listings()` 함수와 LATEX의 `\lstset{breaklines=true}`를 사용할 수 있습니다.

###### 12.3.3 상자 여백 (패딩)

6.2.3절에서 소개했듯이 knitr의 기본 LATEX 스타일은 framed 패키지를 기반으로 하며, 이 때문에 모든 코드 청크 아래에 음영 처리된 상자가 표시됩니다. 상자의 기본 여백이 너무 좁다고 생각되면 `\setlength`를 사용하여 `\fboxsep{}`의 길이를 재설정할 수 있습니다. 예를 들면 다음과 같습니다.

`\setlength\fboxsep{5mm}`

`## an intentional comment to to to to to to to to to to to to ## reach the page margin rpois(40, 5)`

`## [1] 6 4 6 4 9 5 2 4 2 4 4 10 6 3 1 8 8`

- `## [18] 2 7 4 10 6 5 2 7 4 6 4 2 5 8 7 2 3 ## [35] 2 7 7 3 3 3`

이제 5mm의 여백 공간과 함께 회색 상자가 더 커진 것을 볼 수 있습니다. HTML 출력의 경우 스타일을 디자인하기가 훨씬 쉽습니다. 예를 들어 CSS에서 클래스 청크를 다음과 같이 정의하여 여백을 5mm로 만들 수 있습니다.

div.chunk {

padding: 5mm; }

| \documentclass{beamer} \begin{document} \title{Using knitr in Beamer} \author{Yihui Xie}<br><br>\maketitle \begin{frame} \frametitle{Introduction} This is a normal slide. \end{frame} % need the option [fragile] for code output! \begin{frame}[fragile] \frametitle{Code chunks} <<test, out.width= .6\\linewidth , fig.align= center >>= par(mar = c(4, 4, .1, .1)) x = rnorm(100) hist(x, main= , col= lightblue , border= white ) rug(x) @ \end{frame} \end{document} |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

- FIGURE 7.4: 세 개의 표현식이 두 개의 플롯을 생성했습니다. 첫 번째 표현식은 플롯을 그리지 않고, 두 번째 표현식은 고수준 플롯을 그리며, 세 번째 표현식은 플롯에 저수준 변경(텍스트)을 추가합니다. 7.6절에서는 오른쪽 플롯에 LATEX 코드가 어떻게 렌더링되었는지 설명합니다.

par()는 플롯을 생성하지 않으므로 evaluate는 기본적으로 두 개의 플롯을 생성합니다(그림 7.4 참조).

par(mar = c(3, 3, 0.1, 0.1)) plot(1:10, ann = FALSE, las = 1) if (TRUE) {

text(5, 9, "mass $\\rightarrow$ energy\n$E=mc^2$") }

이것은 동적 문서를 위한 R의 기존 도구들과 큰 차이를 가져옵니다. 왜냐하면 저수준 플롯 변경 사항도 기록될 수 있는 반면, 기존 도구들(Sweave 등)은 이러한 변경 사항을 포착하지 못하기 때문입니다.

참고로 R에는 고수준 및 저수준 플로팅 명령이 있습니다. 고수준 플로팅 명령은 새롭고 완전한 플롯을 시작하고(plot(), hist(), boxplot()), 저수준 명령은 대개 기존 플롯에 추가 정보를 더합니다(text(), points(), segments()). 저수준 명령은 고수준 플롯이 생성된 후에 호출되어야 합니다. 자세한 정보는 Murrell(2011)을 참조하십시오.

일반적으로 저수준 플로팅 변경 사항을 별도의 플롯으로 캡처하는 것은 불가능하지는 않더라도 직관적이지 않습니다. evaluate 패키지는 이 작업을 쉽게 만들어 주었습니다.

그림 7.5는 두 개의 고수준 플롯을 생성하는 두 개의 표현식을 보여줍니다. knitr는 그래픽 출력을 자연스럽게 만들려고 노력한다는 것을 기억하십시오. 청크에 두 개의 플롯이 있으면 추가 작업 없이 둘 다 출력에 표시됩니다.

120

120

|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |

100

100

80

80

dist

60

60

40

40

20

20

0

0

5 10 15 20 25

dist

speed

- FIGURE 7.5: 모든 고수준 플롯이 캡처되어 나란히 정렬됩니다.

plot(cars) boxplot(cars$dist, xlab = "dist")

청크 옵션 fig.keep은 출력에 유지할 플롯을 제어합니다. fig.keep = 'all'은 저수준 변경 사항을 별도의 플롯으로 유지함을 의미합니다. 기본값은 fig.keep = 'high'이며, 이는 knitr가 저수준 플롯 변경 사항을 이전의 고수준 플롯에 병합함을 의미합니다. 이 기능은 R 그래픽을 단계별로 가르칠 때 유용할 수 있습니다. 그림 7.4가 그 예이며, 그림 7.6(두 개가 아닌 하나의 청크임에 유의)은 플롯이 생성된 위치에 배치되도록 fig.show = 'asis'와 함께 fig.keep = 'all'을 사용한 또 다른 예입니다.

하지만 다른 표현식 내부에 있는 저수준 플로팅 명령(반복문이 대표적인 경우)은 누적되어 기록되지 않지만, 고수준 플로팅 명령은 위치에 상관없이 항상 기록된다는 점에 유의하십시오. 예를 들어, 다음 청크는 2개의 완전한 표현식이 있으므로 21개의 플롯이 아닌 2개의 플롯만 생성합니다.

plot(0, 0, type = "n", ann = FALSE) for (i in seq(0, pi, length = 20)) points(cos(i), sin(i))

그러나 다음은 표현식이 하나뿐이더라도 plot()이 고수준 플로팅 명령이므로 예상대로 20개의 플롯을 생성합니다.

for (i in seq(0, pi, length = 20)) {

plot(cos(i), sin(i), xlim = c(-1, 1), ylim = c(-1, 1)) }

| plot(cars, pch = 19, col = "darkgray")<br><br>5 10 15 20 25<br><br>0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>120<br><br>speed<br><br>dist<br><br>lines(lowess(cars, f = 0.2), col = "red", lwd = 2)<br><br>5 10 15 20 25<br><br>0<br><br>20<br><br>40<br><br>60<br><br>80<br><br>100<br><br>120<br><br>speed<br><br>dist<br><br> |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

dist

- FIGURE 7.6: 코드 바로 아래에 플롯 표시: fig.show = 'asis' 옵션이 사용되었습니다.

fig.keep = 'last'를 사용하여 이전의 모든 플롯을 버리고 마지막 플롯만 유지하거나, fig.keep = 'first'를 사용하여 첫 번째 플롯만 유지하거나, fig.keep = 'none'을 사용하여 모든 플롯을 버릴 수 있습니다. 마지막 플롯을 유지하는 예제는 그림 7.7을 참조하시고, 코드는 아래에 있습니다.

library(ggplot2) pie <- ggplot(diamonds, aes(x = factor(1), fill = cut)) +

xlab("cut") + geom_bar(width = 1) pie + coord_polar(theta = "y") # a pie chart pie + coord_polar() # the bullseye chart

50000

40000

###### cut

30000

|     |
| --- |
|     |
|     |
|     |
|     |

Fair Good Very Good Premium Ideal

20000

10000

count

0

cut

- FIGURE 7.7: 이 청크에서 두 개의 플롯이 생성되었지만 마지막 플롯만 유지되었습니다. 이것은 많은 플롯을 실험해 보지만 마지막 결과만 원할 때 유용할 수 있습니다. (ggplot2 웹사이트에서 발췌)

플롯 기록에 대한 추가 참고 사항: knitr는 기록된 모든 플롯(R 객체)을 검사하고 순차적으로 비교합니다. 이전 플롯이 다음 플롯(= 이전 플롯 + 저수준 변경 사항)의 "부분 집합"인 경우 기본적으로(즉, fig.keep = 'high'일 때) 이전 플롯은 제거됩니다. 연속된 두 플롯이 동일한 경우 두 번째 플롯이 기본적으로 제거되므로, fig.keep 옵션을 변경하지 않으면 다음 청크가 하나의 플롯만 생성한다는 점이 놀라울 수 있습니다.

m <- matrix(1:100, ncol = 10) image(m) image(m \* 2) # exactly the same as previous plot

###### 7.3 Plot Rearrangement

청크 옵션 fig.show는 청크의 모든 플롯을 보류하여 청크 끝에 한꺼번에 "출력(flush)"할지(fig.show = 'hold', 예는 그림 7.4 및 7.5 참조), 아니면 단순히 플롯이 생성된 위치에 삽입할지(기본값은 fig.show = 'asis')를 결정합니다. 7.2절에서는 하나의 청크에 있는 두 개의 플롯에 대해 fig.show = 'asis'를 사용한 예를 보여주었습니다.

<<clock-animation, fig.show= animate , interval=1>>= par(mar = rep(3, 4)) for (i in seq(pi/2, -4/3 \* pi, length = 12)) {

plot(0, 0, pch = 20, ann = FALSE, axes = FALSE) arrows(0, 0, cos(i), sin(i)) axis(1, 0, "VI"); axis(2, 0, "IX") axis(3, 0, "XII"); axis(4, 0, "III"); box()

} @

- FIGURE 7.8: 시계 애니메이션. 이것은 Adobe Reader에서 보아야 합니다. 재생/일시 정지하려면 클릭하세요. 애니메이션 속도를 높이거나 늦추는 버튼도 있습니다(실제 애니메이션은 여기에 표시되지 않으므로 실제 애니메이션을 보려면 knitr 그래픽 매뉴얼을 참조하세요).

###### 7.3.1 Animation

'hold'와 'asis' 외에도 fig.show 옵션은 세 번째 값인 'animate'를 취할 수 있는데, 이는 출력 문서에 애니메이션을 삽입할 수 있게 해줍니다. LATEX에서는 animate 패키지를 사용하여 이미지 프레임을 하나의 애니메이션으로 엮습니다. 애니메이션이 작동하려면 청크에서 두 개 이상의 플롯이 생성되어야 합니다. 청크 옵션 interval은 애니메이션 프레임 간의 시간 간격을 제어하며, 기본값은 1초입니다. knitr는 이를 자동으로 추가하지 않으므로 LATEX 프리앰블(preamble)에 \usepackage{animate}를 추가해야 한다는 점에 유의하십시오. PDF 출력의 애니메이션은 Adobe Reader에서만 볼 수 있습니다. 패키지 웹사이트에 있는 knitr의 메인 매뉴얼과 그래픽 매뉴얼 모두에서 애니메이션 예제를 찾을 수 있습니다. 그림 7.8은 PDF 문서에서 애니메이션을 생성할 수 있는 청크의 소스 코드를 보여주지만, 종이에 인쇄할 때는 당연히 애니메이션이 작동하지 않으므로 여기에 출력을 표시하지는 않았습니다.

HTML 출력(마크다운 포함)의 경우에도 이 옵션이 작동하며, 세 가지 애니메이션 형식을 사용할 수 있습니다. 패키지 옵션 animation.fun을 사용하여 애니메이션을 생성하는 훅(hook) 함수를 설정할 수 있습니다. knitr 패키지에는 세 가지 기본 제공 훅 함수가 있습니다.

hook_ffmpeg_html FFmpeg를 호출하여 일련의 이미지 프레임을 비디오 파일로 변환합니다. 이 훅이 작동하려면 무료 소프트웨어 패키지인 FFmpeg가 설치되어 있어야 합니다.

hook_scianimator JavaScript 라이브러리 SciAnimator(https://github.com/brentertz/scianimator)를 사용하여 이미지 프레임을 하나씩 표시하여 애니메이션을 형성합니다. 이 훅을 사용하려면 HTML 출력의 헤더에 jQuery와 SciAnimator가 모두 포함되어 있어야 합니다. 예:

<head> <link rel="stylesheet" href="css/scianimator.css" /> <script src="js/jquery-1.4.4.min.js"></script> <script src="js/jquery.scianimator.pack.js"></script>

</head>

이러한 _.js 및 _.css 파일은 SciAnimator의 Github 저장소에서 다운로드할 수 있습니다. 보시다시피 이 훅 함수는 상당한 HTML 지식을 요구합니다.

hook_r2swf R2SWF 패키지(Qiu et al., 2015)를 사용하여 이미지를 Flash(SWF) 애니메이션으로 변환합니다. 이 훅은 R에 R2SWF 패키지만 설치되어 있으면 되며, 추가 소프트웨어 패키지나 구성이 필요하지 않으므로 사용하기 쉬울 수 있습니다.

다음은 이 패키지 옵션을 설정하는 방법입니다.

opts_knit$set(animation.fun = hook_scianimator) # or opts_knit$set(animation.fun = hook_r2swf)

###### 7.3.2 Alignment

청크 옵션 fig.align을 통해 그림 정렬을 지정할 수 있습니다(가능한 값은 'default', 'left', 'center', 'right'입니다). 이 책의 전역 옵션은 fig.align = 'center'이므로 대부분의 플롯은 가운데 정렬되어 있습니다. 그림 7.9는 아래 코드 청크에 의해 생성된 오른쪽 정렬 플롯의 예입니다.

stars(cbind(1:16, 10 \* (16:1)), draw.segments = TRUE)

LATEX의 경우, knitr는 플롯의 왼쪽 또는 오른쪽에 수평 채움(\hfill{})을 사용하여 플롯을 오른쪽 또는 왼쪽으로 정렬하며, 가운데 정렬에는 {\centering }이 사용됩니다. HTML 출력의 경우 CSS 클래스가 플롯에 연결되어 정렬됩니다. 예를 들어, 왼쪽 정렬 플롯의 경우 div 요소 <div class='rimage left'></div> 안에 배치되며, left 클래스의 CSS 정의는 float: left;입니다. 정렬 옵션은 마크다운에서는 무시됩니다.

- FIGURE 7.9: ?stars에서 가져온 오른쪽 정렬 플롯: 청크 옵션은 fig.align = 'right'입니다.

###### 7.4 Plot Size in Output

fig.width와 fig.height 옵션은 그래픽 장치에서 플롯의 크기를 지정하며, 출력 문서에서의 실제 크기는 다를 수 있습니다(out.width 및 out.height로 지정됨). 하나의 코드 청크에 여러 개의 플롯이 있는 경우, 여러 플롯을 나란히 배열할 수 있습니다. 예를 들어, LATEX에서는 out.width를 현재 줄 너비의 절반 미만으로 설정하기만 하면 됩니다(out.width = '.49\\linewidth' - 이는 이 장에서 플롯에 대한 일반적인 설정입니다). 그러면 다음과 같은 코드를 사용하여 LATEX 문서에 플롯이 삽입됩니다. \includegraphics[width=.49\linewidth]{plot-foo}

fig.width 및 fig.height는 일반적으로 숫자 값을 취하는 반면, out.width 및 out.height는 출력 형식에 따라 문자 값을 취합니다(HTML 출력의 그림의 경우 부모 컨테이너 너비의 50%를 나타내는 out.width = '50%' 또는 '480px'(480픽셀)).

LATEX 출력에 대한 out.width의 기본값은 \maxwidth입니다. 이는 표준 LATEX 길이는 아니며 다음과 같이 정의되었습니다.

% maxwidth is the original width if it s less than linewidth % otherwise use linewidth \makeatletter \def\maxwidth{ %

\ifdim\Gin@nat@width>\linewidth

\linewidth \else

\Gin@nat@width \fi

} \makeatother

이것은 합리적인 기본값입니다. 왜냐하면 플롯이 줄 너비보다 넓을 경우 줄 너비에 맞게 크기가 조정되고, 그렇지 않으면 원래 너비가 사용되기 때문입니다. 다시 말해 기본적으로 LATEX에서 플롯은 페이지 여백을 초과하지 않습니다.

###### X ~ bernoulli(p);

}
@

청크 옵션 engine = 'stan' 외에도 engine.opts = list(x = 'ex1')이라는 옵션을 지정했습니다. 여기서 x는 R 세션에 저장될 Stan 모델의 이름을 의미합니다. 이 코드 청크는 모델을 rstan의 stan_model() 함수에 전달하고 모델을 ex1이라는 객체에 저장할 것입니다. 그렇기 때문에 다음 청크에서 ex1 객체를 사용할 수 있습니다.

library(rstan)
fit <- sampling(ex1, data = list(X = rbinom(20, 1, 0.3)))

SAMPLING FOR MODEL 'anon_model' NOW (CHAIN 1).

Iteration: 1 / 2000 [ 0%] (Warmup)
Iteration: 200 / 2000 [ 10%] (Warmup)
Iteration: 400 / 2000 [ 20%] (Warmup)
Iteration: 600 / 2000 [ 30%] (Warmup)
Iteration: 800 / 2000 [ 40%] (Warmup)
Iteration: 1000 / 2000 [ 50%] (Warmup)
Iteration: 1001 / 2000 [ 50%] (Sampling)

....
print(fit)

Inference for Stan model: anon_model.
4 chains, each with iter=2000; warmup=1000; thin=1;
post-warmup draws per chain=1000, total post-warmup draws=4000.

mean se_mean sd 2.5% 25% 50% 75%
p 0.36 0.00 0.10 0.18 0.29 0.36 0.43
lp\_\_ -14.93 0.02 0.73 -16.99 -15.12 -14.65 -14.47

97.5% n_eff Rhat
p 0.57 1498 1
lp\_\_ -14.42 1703 1
....

p = 0.3인 베르누이 분포에서 20개의 무작위 데이터 포인트를 생성하고, 이를 베이지안 추론을 위한 샘플 데이터 Y로 사용했습니다. 샘플링 출력을 통해 p의 사후 평균이 0.3에 가깝다는 것을 확인할 수 있습니다.

###### 11.2.5 TikZ

7.6절에서 R 그래픽을 TikZ로 변환할 수 있게 해주는 tikzDevice 패키지를 소개했습니다(Tantau, 2008). 사실 엔진을 tikz로 설정하면 knitr에서 직접 원시 TikZ 코드를 작성할 수 있습니다.

tikz 엔진이 내부적으로 수행하는 작업은 LATEX 템플릿을 사용하여 코드 청크를 삽입하고 tex 문서를 PDF로 컴파일하는 것입니다. 기본적으로 knitr 내에 있는 템플릿(knitr 설치 디렉토리의 misc 디렉토리 아래에 있는 tikz2pdf.tex)을 사용합니다.

- f <- system.file("misc", "tikz2pdf.tex", package = "knitr")
  cat(readLines(f), sep = "\n")

\documentclass{article}
\include{preview}
\usepackage[pdftex,active,tightpage]{preview}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{matrix}
\begin{document}
\begin{preview}
%% TIKZ_CODE %%
\end{preview}
\end{document}

%% TIKZ_CODE %% 부분은 TikZ 코드 청크로 대체될 것입니다. 기본 템플릿이 만족스럽지 않다면 청크 옵션 engine.opts를 통해 템플릿을 제공할 수 있습니다. 예를 들면 engine.opts = list(template = 'path/to/tikz/template.tex')와 같이 할 수 있습니다. 그런 다음 이 TEX 파일은 R 함수 tools::texi2pdf()를 통해 PDF로 컴파일됩니다. 지정된 그림 파일 확장자(청크 옵션 fig.ext)가 pdf가 아닌 경우, ImageMagick(convert 유틸리티를 통해)이 호출되어 HTML 등 문서 형식에 맞게 PDF 파일을 PNG와 같은 다른 파일 형식으로 변환합니다.

그림 11.1은 아래의 원시 TikZ 코드로 그려진 다이어그램입니다.

\usetikzlibrary{arrows}
\begin{tikzpicture}[node distance=2cm, auto,>=latex', thick]
\node (P) {$P$};

f^ k

- P^

f

g

f

- g^ g

P B

A C

- 그림 11.1: TikZ로 그린 다이어그램. 소스 코드는 \*.tex 파일에 작성되고 LATEX에 의해 PDF로 컴파일됩니다.

- \node (B) [right of=P] {$B$};
  \node (A) [below of=P] {$A$};
- \node (C) [below of=B] {$C$};
  \node (P1) [node distance=1.4cm, left of=P, above of=P]

{$\hat{P}$};
\draw[->] (P) to node {$f$} (B);
\draw[->] (P) to node [swap] {$g$} (A);

- \draw[->] (A) to node [swap] {$f$} (C);
- \draw[->] (B) to node {$g$} (C);
  \draw[->, bend right] (P1) to node [swap] {$\hat{g}$} (A);
  \draw[->, bend left] (P1) to node {$\hat{f}$} (B);
  \draw[->, dashed] (P1) to node {$k$} (P);
  \end{tikzpicture}

TikZ 그래픽을 개발할 때는 qtikz나 ktikz 프로그램이 도움이 될 수 있습니다. 이 프로그램들은 그래픽 사용자 인터페이스(편집기)를 제공하여 결과를 미리 볼 수 있게 해줍니다.

###### 11.2.6 Graphviz

Graphviz(Ellson 등, 2002)는 널리 쓰이는 오픈 소스 그래프 시각화 소프트웨어 패키지입니다(http://www.graphviz.org). 이는 추상적인 그래프와 네트워크의 다이어그램을 그리는 데 강력합니다. Graphviz에는 방향성 그래프를 그리는 dot, 무방향성 그래프를 그리는 neato와 같은 몇 가지 "필터"가 포함되어 있습니다. engine = 'dot'인 경우 기본적으로 dot이 사용됩니다. 다른 필터를 사용하려면 예를 들어 engine.path = 'neato'로 설정할 수 있습니다.

그림 11.2는 Graphviz 문서에서 발췌한 예제입니다.

a

| b   |
| --- |

x y

hi

multi-line label

hello world

z

- 그림 11.2: Graphviz의 dot으로 그린 다이어그램(dot 매뉴얼에서 발췌).

여기서는 PDF 그래프 파일을 생성하기 위해 fig.ext = 'pdf'를 사용했으며, 이를 PNG와 같은 다른 파일 형식으로도 변경할 수 있습니다.
digraph test123 {
a -> b -> c;

- a -> {x y};
- b [shape=box];
- c [label="hello\nworld",color=blue,fontsize=24, fontname="Palatino-Italic",fontcolor=red,style=filled];

- a -> z [label="hi", weight=100];
  x -> z [label="multi-line\nlabel"];
  edge [style=dashed,color=red];
- b -> x;
  {rank=same; b x}

}

R 마크다운에서 생성된 HTML 문서에 다이어그램을 그리고 싶으시다면, 몇 가지 자바스크립트 라이브러리들을 래핑한 HTML 위젯 패키지인 DiagrammeR 패키지(https://github.com/rich-iannone/DiagrammeR)를 고려해 보시기 바랍니다(HTML 위젯에 대한 더 자세한 정보는 14.5.3절을 참고하십시오).

###### 11.2.7 Highlight

Highlight는 C, PHP, R 등을 포함한 다양한 언어에 대해 구문 강조(syntax highlighting)를 제공하는 Andre Simon(http://www.andre-simon.de)의 무료 오픈 소스 소프트웨어 패키지입니다. 이 소프트웨어는 출력을 LATEX 또는 HTML로 작성할 수 있습니다.

청크 옵션이 engine = 'highlight'일 때, 구문 강조된 코드 청크를 생성하기 위해 highlight 프로그램이 호출됩니다. 청크 옵션 engine.opts는 Highlight에 추가 인자를 전달하기 위한 문자열입니다. 예를 들어, -S를 통해 입력 구문을 지정하고, -O를 통해 출력 유형을 지정할 수 있습니다.

아래 청크는 앞서 보여드린 awk 예제에서 가져온 것입니다. 여기서는 입력 구문이 awk이고 출력 유형이 LATEX임을 Highlight에 알려주기 위해 청크 옵션 engine.opts = '-S awk -O latex'를 사용했습니다. 이를 통해 Highlight가 키워드에 대해 적절한 LATEX 명령어들을 생성할 수 있습니다. 이 책의 인쇄본에서는 색상을 구분하기 어려울 수 있으나, 최소한 첫 번째 줄이 이탤릭체(주석)라는 점은 확인할 수 있습니다.

# NEWS 파일에 빈 줄이 아닌 줄이 몇 개 있습니까?

NF {
i = i + 1

###### }

END { print i }

Highlight는 코드의 다양한 토큰들을 마크업하기 위해 \hlnum{}(숫자용) 및 \hlstr{}(문자열용)과 같은 명령어들을 생성한다는 점에 유의하시기 바랍니다. 이 명령어들은 knitr의 구문 강조 명령어들과 대부분 일치하지만 몇 가지 예외가 있습니다. 예를 들어, Highlight에 의해 생성된 \hlslc{}(주석용)은 knitr 명령어의 일부가 아니므로 LATEX 프리앰블(preamble)에서 이를 정의해 주어야 합니다. 마찬가지로 Highlight 출력이 HTML인 경우, hl slc 클래스에 대한 CSS 스타일을 정의해 주어야 합니다.

###### 11.2.8 기타 엔진

기본적으로 모든 언어를 위한 두 가지 엔진이 더 있는데, 바로 cat과 asis입니다. cat 엔진은 cat() 함수를 호출하여 코드 청크를 파일에 기록하며, 파일명은 청크 옵션 engine.opts = list(file = ?)에 제공될 수 있습니다. asis 엔진은 코드 청크를 그대로 출력물에 작성하는 것 외에는 아무것도 하지 않습니다. 그러나 이 엔진은 청크 옵션 eval과 echo를 따릅니다. 이 옵션들 중 하나라도 FALSE인 경우, 코드 청크는 출력물에서 숨겨집니다. 이는 출력물에서 특정 내용의 표시 여부를 동적으로 제어하고자 할 때 유용할 수 있습니다.

예를 들어, 다음과 같이 cat 엔진을 통해 아래 코드 청크를 styles.css라는 파일에 쓸 수 있습니다.

<<engine='cat', engine.opts = list(file = 'styles.css')>>=
p {
margin: 5px 2px 5px 2px;
}
@

만약 internal.only 변수가 TRUE라면(여러분의 그룹 내에서 내부적으로만 보여주고 싶은 보고서 내용의 일부가 있다고 가정해 보겠습니다) 다음 코드 청크가 최종 출력물에 포함될 것입니다.

<<engine='asis', echo = internal.only>>=
internal.only를 TRUE로 설정함에 따라 이 보고서의 공개용 버전에서는 숨겨져 있는 우리 분석에 대한 몇 가지 일급 비밀들을 소개합니다.
첫 번째 비밀: ...
@

###### 11.3 지속성 세션 (Persistent Sessions)

사실 앞서 소개된 인터프리터 언어용 엔진들에는 한 가지 주요한 결함이 있습니다. 바로 이 엔진의 매 코드 청크마다 새로운 엔진 세션이 설정된다는 것입니다. 이는 모든 코드 청크들이 메모리상 독립적이며, 이전 청크에서 생성된 변수들을 이후의 청크에서 사용할 수 없다는 것을 의미합니다. 유일한 예외는 R 코드 청크입니다. 모든 R 코드 청크들은 동일한 R 세션 안에서 평가됩니다. 이 문제를 해결하기 위해, 우리는 엔진에 대해 지속성 세션(persistent session)을 열고, 이 세션 안에서 코드 청크들을 계속 실행해야 합니다. 예를 들어, 파이썬 코드 청크에서 변수를 하나 생성한 후, 다음 파이썬 청크에서 이 변수를 계속 사용할 수 있어야 합니다.

runr 패키지(Xie, 2013)는 이 문제를 해결하기 위한 하나의 시도입니다. 현재 이 패키지는 소켓 연결에 기반하여 Bash 및 Julia 코드에 대한 실험적 지원을 제공하고 있습니다. 기본적인 아이디어는 다음과 같습니다(Julia 엔진을 예로 들어보겠습니다).

- 1. 소켓 서버를 시작하고 수신 대기 상태를 유지하는 백그라운드 Julia 프로세스를 엽니다(이 백그라운드 프로세스는 system('julia script.jl', wait = FALSE)를 통해 현재의 R 세션에서 분리됩니다).
- 2. R은 socketConnection(open = 'w')를 통해 Julia 소켓 서버에 연결하고, Julia 코드 청크를 서버로 전송합니다.
- 3. Julia는 코드를 수신하여 평가한 뒤, 표준 출력(일반 텍스트)을 소켓에 씁니다.
- 4. R은 socketConnection(open = 'r')를 통해 소켓으로부터 데이터를 읽어들이고, R 코드 청크 출력과 동일하게 Julia 출력을 보고서에 씁니다.
- 5. 다음 Julia 코드 청크가 들어오면 2~4단계를 반복합니다. R에서 quit() 코드를 전송하면 Julia는 종료됩니다.

이러한 방식으로, Julia 세션은 우리가 R에서 명시적으로 종료할 때까지 유지되며, 모든 Julia 코드 청크들은 동일한 Julia 세션에서 평가될 것입니다. runr 패키지는 아직 초기 단계에 있으며, 커뮤니티의 기여를 환영합니다.
