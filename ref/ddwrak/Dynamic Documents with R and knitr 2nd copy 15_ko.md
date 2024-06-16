---

###### 14.4 Shiny를 활용한 대화형 문서

Shiny(Chang 등, 2015)는 R을 사용하여 대화형 앱을 쉽게 만들 수 있는 웹 애플리케이션 프레임워크입니다. Shiny UI 함수를 사용하여 텍스트 입력 상자, 드롭다운 목록, 라디오 버튼, 슬라이더 등의 웹 사용자 인터페이스(UI)를 만들 수 있습니다. 버튼을 클릭한 후 R이 수행할 작업과 같이 R에서 서버 로직을 지정하면 이러한 UI 요소가 R과 상호 작용할 수 있습니다. Shiny에 익숙하지 않다면 http://shiny.rstudio.com 웹사이트를 방문하여 Shiny의 기본 사항을 알아보시기 바랍니다.

Shiny 앱은 기본적으로 HTML 페이지이며 R Markdown도 HTML로 렌더링될 수 있으므로 하나의 문서에 R Markdown과 Shiny를 결합할 수 있습니다. 이러한 문서에는 Shiny의 대화형 구성 요소가 포함되어 있으므로 "대화형 문서"라고 부릅니다. 그림 14.13은 대화형 문서의 간단한 예시를 보여줍니다. 소스 문서는 다음과 같습니다.

|![image 28](Dynamic Documents with R and knitr 2nd_images/imageFile28.png)|
|---|


###### 그림 14.12: Tufte 핸드아웃 스타일을 사용한 예시 페이지: 각주, 그림, 수식 등의 요소를 측면 여백에 배치할 수 있습니다.

![image 29](Dynamic Documents with R and knitr 2nd_images/imageFile29.png)

###### 그림 14.13: R Markdown과 Shiny를 사용한 간단한 대화형 문서: 슬라이더 값을 변경할 수 있으며, 히스토그램의 막대 개수가 자동으로 변경됩니다.

--title: "R Markdown v2 Demo" runtime: shiny output: html_document

--   {r} library(shiny) sliderInput("bins", "Number of bins:", min = 1, max = 50,

value = 30)

renderPlot({ x <- faithful[, 2] # Old Faithful Geyser data bins <- seq(min(x), max(x), length.out = input$bins + 1) # draw the histogram with the specified number of bins hist(x, breaks = bins, col =  darkgray , border =  white )

})

일반 R Markdown 문서를 대화형 문서로 바꾸려면 YAML 메타데이터에 runtime: shiny 옵션만 추가하면 됩니다. 그러면 shiny 패키지의 함수를 사용할 수 있습니다. 위 예시에서는 shiny의 UI 함수인 sliderInput()을 사용하여 HTML 페이지에 슬라이더를 만들었습니다. 슬라이더의 id는 bins입니다. 그런 다음 renderPlot() 함수를 사용하여 히스토그램을 렌더링했습니다. 이 코드 청크에서 중요한 부분은 id가 bins인 슬라이더와 연관된 변수 값인 input$bins입니다. 슬라이더 값을 업데이트하면 해당 값이 renderPlot()의 표현식으로 전달되고 그에 따라 플롯이 다시 그려집니다.

render() 대신 rmarkdown의 run() 함수로 대화형 문서를 컴파일해야 합니다. RStudio를 사용하는 경우 R Markdown 문서에 runtime: shiny를 추가한 후 도구 모음의 Knit 버튼 레이블이 Run Document로 바뀌는 것을 볼 수 있으며, 이 버튼을 클릭하여 문서를 실행할 수 있습니다.

모든 Shiny 앱이 그림 14.13처럼 간단한 것은 아닙니다. 여러 UI 요소가 있는 경우, 코드 청크에 선형적으로 작성하는 대신 별도의 앱으로 구성하고 싶을 수 있습니다. shiny의 shinyApp() 함수를 사용하면 하나의 함수에 모든 UI 요소와 서버 로직을 지정하여 전체 앱을 구축할 수 있습니다. 그런 다음 R Markdown에서 명시적으로 shinyApp()을 사용하여 전체 앱을 포함시키거나 shinyApp() 객체를 반환하는 고유한 함수를 작성하여 다른 사람들도 쉽게 앱을 사용할 수 있도록 할 수 있습니다.

정적 HTML 문서는 공유하려 할 때 임의의 웹사이트에 업로드하거나 이메일로 보낼 수 있습니다. 대화형 문서의 경우, 문서 이면에서 활성 R 세션이 실행되고 있어야 합니다. 대화형 문서를 공유하는 한 가지 방법은 RStudio에서 호스팅하는 http://shinyapps.io에 게시하는 것입니다. 이 웹사이트에 게시하고 싶지 않다면 자체 Shiny Server(http://www.rstudio.com/products/shiny/shiny-server/)를 설정할 수 있습니다.

###### 14.5 R Markdown v2 확장하기

출력 형식 함수 중 요구 사항을 충족하는 것이 없으면 확장하거나 완전히 새로운 형식을 작성할 수 있습니다. 그렇게 하기 전에 기존 출력 형식의 모든 가능성을 살펴보았는지 확인해 보시기 바랍니다. 새로운 것을 발명할 필요가 없는 경우도 있습니다. 예를 들어, 다른 LATEX 문서 클래스를 사용하려는 경우 원하는 문서 클래스로 새 템플릿을 작성할 수도 있지만 YAML 메타데이터에서 documentclass 옵션을 설정하는 편이 낫습니다. Tufte 핸드아웃을 예로 들어보겠습니다.

--title: "R Markdown v2 Demo" author: John Smith date: "2015/01/01" output: pdf_document documentclass: tufte-handout classoption: nohyper geometry: no

---

위 YAML 메타데이터는 기존 pdf_document() 형식을 활용합니다. 대안으로 다음과 같이 템플릿을 준비할 수도 있습니다. \documentclass{tufte-handout} $if(title)$ \title{$title$} $endif$ $if(author)$ \author{$for(author)$$author$$sep$ \and $endfor$} $endif$ $if(date)$ \date{$date$}

$endif$ \begin{document} $if(title)$ \maketitle $endif$ $body$ \end{document}

그런 다음 pdf_document에서 template 옵션을 사용합니다. 이와 같이 사용자 지정 템플릿을 작성할 경우 여러 단점이 있습니다.

- • Pandoc의 기본 LATEX(https://github.com/jgm/pandoc-templates)는 훨씬 유연하며 목차, 그림 목록, 초록 등도 처리할 수 있습니다.
- • 새 템플릿을 작성하는 것은 YAML의 기존 옵션을 사용하는 것보다 더 많은 작업이 필요합니다.
- • 템플릿을 작성한 후에는 템플릿을 손상시킬 수 있는 Pandoc의 향후 변경 사항에 주의해야 하며 유용한 새 기능을 놓칠 수도 있습니다. 반면 Pandoc의 템플릿을 사용하면 유지 관리할 필요가 없습니다.

그러면 왜 rmarkdown에 tufte_handout() 형식이 있는지 의문이 생길 수 있습니다. 사실 이 새로운 형식이 하는 일은 단순한 LATEX 템플릿 그 이상입니다. 전체 너비 그림(fig.fullwidth = TRUE) 및 여백 그림(fig.margin = TRUE)을 생성하기 위한 몇 가지 knitr 청크 옵션도 정의합니다. 기존 출력 형식에서는 이 두 가지 다른 그림 유형을 제공하지 않습니다.

###### 14.5.1 템플릿

rmarkdown 확장의 첫 번째 유형은 새로운 템플릿을 정의하는 것입니다. 위에서 Tufte 핸드아웃에 대한 예시를 보여주었고, 앞서 14.3.1절에서 HTML 문서 출력에 대한 예시도 보여주었습니다.

https://github.com/jgm/pandoc-templates 저장소에는 Pandoc에서 사용하는 모든 템플릿이 포함되어 있으며, https://github.com/rstudio/rmarkdown의 rmarkdown 소스 패키지에서도 사용자 지정 템플릿을 살펴볼 수 있습니다. 이해되지 않는 템플릿 변수가 있다면 http://johnmacfarlane.net/pandoc/ 문서를 확인해 보시기 바랍니다.

템플릿을 다른 사용자와 공유하는 쉬운 방법은 R 패키지의 inst/rmarkdown/templates/ 디렉터리에 넣는 것입니다. 예를 들어 my_template과 같은 새 디렉터리를 만들고 그 아래에 템플릿 파일을 넣을 수 있습니다. 템플릿에는 CSS/JavaScript 파일 또는 LATEX 패키지와 같은 특정 종속성이 필요할 수 있습니다. 이들은 my_template 아래의 skeleton/ 하위 디렉터리에 수집할 수 있습니다. skeleton/ 디렉터리에서 샘플 Rmd 파일 skeleton.Rmd를 제공할 수도 있습니다. 마지막으로 my_template 아래의 YAML 파일 template.yaml에서 다음 세 가지 YAML 필드로 템플릿을 설명할 수 있습니다.

name: 템플릿의 이름(“Journal of Statistical Software”)
description: 템플릿에 대한 간단한 설명(“This is a template for JSS articles”)
create_dir: yes 또는 no, 또는 true 또는 false(곧 설명 예정)

myPackage라는 R 패키지를 설치했다고 가정하면, draft() 함수를 사용하여 템플릿에서 새 초안을 만들 수 있습니다.

rmarkdown::draft("my_article.Rmd", template = "my_template", package = "myPackage")

이 함수는 myPackage에서 my_template 템플릿을 찾아 skeleton.Rmd를 my_article.Rmd라는 이름으로 현재 작업 디렉터리에 복사하고 종속성도 복사합니다. 위에서 언급한 YAML 옵션 create_dir은 초안 my_article.Rmd에 대한 새 디렉터리를 만들지 여부를 결정합니다.

RStudio는 이 과정을 더욱 쉽게 만들었습니다. 메뉴 File - New File - R Markdown에서 로컬에 설치된 모든 패키지의 모든 템플릿을 볼 수 있습니다(그림 14.14).

rticles 패키지(https://github.com/rstudio/rticles)는 여러 LATEX 문서 클래스를 위한 템플릿 모음입니다. 이 패키지의 템플릿을 사용하여 Journal of Statistical Software, The R Journal 등에 대한 논문을 R Markdown으로 작성할 수 있습니다.

###### 14.5.2 새 형식

rmarkdown 확장의 두 번째 유형은 새로운 출력 형식입니다. 새로운 형식은 기존 출력 형식을 기반으로 할 수도 있고 완전히 새로운 형식일 수도 있습니다. 전자의 경우, 기존 출력 형식 함수에서 특정 옵션을 수정한 출력 형식 객체를 반환하는 R 함수를 정의하기만 하면 되므로 쉽습니다. 간단한 예시로 아래에서 toc 인수의 기본값을 FALSE에서 TRUE로 변경하는 html_toc 함수를 만듭니다.

![image 30](Dynamic Documents with R and knitr 2nd_images/imageFile30.png)

- 그림 14.14: 템플릿에서 새 R Markdown 문서 만들기: 목록에서 템플릿을 선택할 수 있습니다.

html_toc <- function(toc = TRUE, ...) {

rmarkdown::html_document(toc = toc, ...) }

새로운 형식 함수는 R 패키지(이름이 myPackage라고 가정함)에 넣어야 하며, 그러면 YAML에서 사용할 수 있습니다. 다음은 두 가지 예시입니다.

--output: myPackage::html_toc

---

--output:

myPackage::html_toc: toc: no self_contained: no

---

![image 31](Dynamic Documents with R and knitr 2nd_images/imageFile31.png)

- 그림 14.15: R Markdown에서 전자책 만들기: 이 그림은 FBReader(무료 전자책 리더)에 표시된 EPUB 책의 제목 페이지를 보여줍니다.

두 번째 예시의 경우, 이 Rmd 파일을 렌더링할 때 호출되는 것은 다음과 같습니다.

rmarkdown::render("foo.Rmd", myPackage::html_doc(toc = FALSE,

self_contained = FALSE)) # which is essentially render( foo.Rmd , # html_document(toc = FALSE, self_contained = FALSE))

14.3.1절에서 설명했듯이 출력 형식은 knitr 옵션, Pandoc 옵션, rmarkdown 옵션 등 세 가지 유형의 옵션 목록입니다. 위의 간단한 예시에서는 Pandoc toc를 사용자 정의했지만, 출력 형식 함수에서 훨씬 더 많은 옵션을 사용자 정의할 수 있습니다. rmarkdown에는 출력 형식을 구성하는 데 사용할 수 있는 몇 가지 헬퍼 함수인 output_format(), knitr_options() 및 pandoc_options()가 있습니다. reveal.js(HTML5 프레젠테이션 형식)를 위한 새 형식을 만드는 방법에 대한 예시는 https://github.com/jjallaire/revealjs 저장소를 참고하시기 바랍니다. 아래에서는 EPUB(전자책 형식) 출력을 만드는 간단한 예시를 보여줍니다.

# @importFrom rmarkdown output_format # @importFrom rmarkdown knitr_options # @importFrom rmarkdown pandoc_options

epub_book <- function(to = c("epub", "epub3")) { to <- match.arg(to) optk <- knitr_options() optp <- pandoc_options(to, ext = ".epub") output_format(knitr = optk, pandoc = optp)

}

이 함수를 myPackage 패키지에 넣으면 R Markdown에서 전자책을 만들 수 있습니다. 다음은 간단한 R Markdown 예시입니다(그림 14.15).

--title: "R Markdown v2 Demo" author:

- - Li Lei
- - Han Meimei

date: "2015/01/01" output: myPackage::epub_book

- --# Start with a cool section

  {r} 1 + 1

형식 함수 epub_book()의 핵심은 pandoc_options()의 to 인수를 epub 또는 epub3로 지정하는 것이었습니다. Pandoc은 매우 다양한 문서 형식을 지원하지만 rmarkdown에는 그 중 일부만 포함되어 있습니다. 위에서 소개한 방법을 사용하여 고유한 형식 함수를 구축할 수 있습니다.

###### 14.5.3 HTML 위젯

14.3.1절에서 YAML 메타데이터의 includes 옵션을 설명했습니다. HTML 문서 출력에 JavaScript 라이브러리를 포함하려는 경우 includes 옵션을 사용할 수 있습니다. 이 방식에는 두 가지 단점이 있습니다.

- 1. 이동성이 떨어집니다. R Markdown 문서를 다른 사람과 공유할 때 includes 옵션에 지정된 종속성을 복사하는 것을 잊지 말아야 하며, 다른 사람이 사용자 종속성을 재사용하는 것도 편리하지 않습니다.
- 2. JavaScript 라이브러리를 호출하기 위해 R Markdown에 JavaScript 코드를 (때로는 아주 많이) 작성해야 하지만 모든 R 사용자가 JavaScript에 익숙한 것은 아니므로 R Markdown 문서에서 작업하지 못할 수도 있습니다.

HTML 위젯의 아이디어는 JavaScript 라이브러리에 대한 네이티브 R 인터페이스를 제공하는 것입니다. 이를 통해 JavaScript를 모르는 사용자도 기본 종속성이나 JavaScript 문법에 대해 걱정할 필요 없이 라이브러리를 사용할 수 있습니다. JavaScript 라이브러리를 사용하여 플롯을 그릴 때, 코드 청크에서 R 함수를 호출하기만 하면 됩니다.

htmlwidgets 패키지(Vaidyanathan 등, 2014)는 패키지 개발자가 JavaScript 라이브러리를 R로 쉽게 포팅할 수 있도록 설계되었습니다. http://www.htmlwidgets.org에 문서화가 잘 되어 있으며, 웹사이트에서 여러 예시 패키지도 볼 수 있습니다. 여기서는 기술적인 세부 사항은 설명하지 않고 HTML 위젯의 형태에 대한 간단한 예시만 보여드리겠습니다. 다음은 간단한 R Markdown 예시입니다(이 예시를 시도해 보기 전에 https://github.com/rstudio/DT에서 DT 패키지를 설치해야 합니다).

--title: "R Markdown v2 Demo" author:

- - Li Lei
- - Han Meimei

date: "2015/01/01" output: html_document

- --Here is a table generated by the DataTables library.

  {r} DT::datatable(iris)

그림 14.16은 그 출력을 보여줍니다. DT 패키지는 JavaScript 라이브러리 DataTables(http://datatables.net)에 대한 인터페이스입니다. 보시다시피 R Markdown 소스 문서는 매우 간단하며 JavaScript 파일이나 JavaScript 코드가 전혀 보이지 않습니다. 간단하게 datatable() 함수를 호출하면 DataTables를 통해 데이터 프레임이 표시됩니다. HTML 페이지에 데이터를 전달하고 구문 분석하여 렌더링하는 어려운 작업은 패키지 작성자가 수행하므로 사용자는 기본적인 기술적 세부 사항을 모두 이해할 필요가 없습니다.

###### 14.6 R Markdown v1에서 v2로의 변경 사항

v1 시절부터 R Markdown을 사용하기 시작했다면, v1에서 v2로 전환할 때 알아두어야 할 변경 사항 목록은 다음과 같습니다.

![image 32](Dynamic Documents with R and knitr 2nd_images/imageFile32.png)

###### 그림 14.16: R Markdown에서 DataTables 라이브러리가 생성한 표: 열을 정렬하고 표에서 검색할 수 있으며 전체 표를 여러 페이지에 나누어 표시할 수 있습니다.

- • v2에서는 더 이상 기본적으로 knitr 패키지가 로드(엄밀히 말하면 첨부)되지 않습니다. 즉, library(knitr) 명령 등을 통해 명시적으로 패키지를 로드하지 않으면 knitr 패키지의 함수와 객체를 사용할 수 없습니다. 그렇지 않으면 "object 'opts_chunk' not found"와 같은 오류가 발생할 수 있습니다.
- • 청크 옵션 fig.path(그림 경로) 및 cache.path(캐시 경로)는 Rmd 파일을 렌더링할 때 rmarkdown에서 수정됩니다. knitr에서는 각각 figure/ 및 cache/였습니다. 현재 rmarkdown에서는 각각 foo_files/figure-format/ 및 foo_files/cache-format/입니다. 여기서 foo는 파일 확장자가 없는 입력 Rmd 파일의 기본 파일 이름이고, format은 출력 형식(tex 또는 html)입니다.
- • 청크 옵션 error가 TRUE에서 FALSE로 변경되었습니다. 이는 R Markdown 출력 문서에 오류 메시지를 표시하는 대신 R이 기본적으로 실행을 중지한다는 의미입니다(6.2.4절 참고).
- • 출력 형식에 따라 청크 옵션 fig.width, fig.height 및 fig.retina가 다른 값을 가질 수 있습니다. 출력 형식 함수에 대한 rmarkdown 문서를 확인하거나 R Markdown 문서에서 str(knitr::opts_chunk$get())을 출력하여 청크 옵션의 값을 확인할 수 있습니다.

### 15

###### 활용

지금까지 단순성을 위해 짧은 예시를 사용하여 knitr의 사용법을 소개했습니다. 이 장에서는 구체적이고 완벽한 예시를 통해 knitr가 실제 애플리케이션에서 어떻게 작동하는지 보여줍니다. 애플리케이션의 모든 세부 사항을 다 설명하지는 않으며 중요한 부분만 지적합니다.

###### 15.1 과제

과제의 경우 단순성 때문에 R Markdown이 선호하는 문서 형식일 수 있으며, 일반적으로 과제는 출판을 목적으로 하지 않습니다. 앞서 언급했듯이 RPubs(http://rpubs.com)는 knitr를 통해 RStudio에서 생성된 (HTML) 보고서를 공유하는 플랫폼입니다. 과제 제출물도 많이 있습니다.

과제 보고서는 비교적 간단하므로 knitr 기능이 너무 많이 필요하지 않을 수 있습니다. 과제에서 사용되는 몇 가지 일반적인 기능으로는 플롯 크기 설정(fig.width 및 fig.height), 채점자가 읽고 싶어 하지 않을 수 있으므로 소스 코드 숨기기(echo = FALSE), 시간이 많이 걸리는 컴퓨팅 작업에 대한 캐시 활성화(cache = TRUE) 등이 있습니다. 기본적으로 제공되는 tidy = TRUE 및 highlight = TRUE와 같은 기타 기능은 코딩 스타일에 신경 쓰지 않는 사용자가 출력 문서에서 더 읽기 쉬운 코드를 생성하는 데 도움이 될 수 있습니다.

이제 깁스 샘플링(Gibbs sampling)의 예시를 보여드리겠습니다. 이변량 정규 분포에 대해

σX2 ρσXσY ρσXσY σY2

- X
- Y ∼ N

- µX
- µY

,

(15.1)

조건부 분포를 알고 있습니다.

σY σX

ρ(x − µX), (1 − ρ2)σY2

Y|X = x ∼ N µY +

- σX

- σY

ρ(y − µY), (1 − ρ2)σX2 (15.2)

X|Y = y ∼ N µX +

213

따라서 깁스 샘플링을 사용하여 결합 정규 분포에서 난수를 생성할 수 있습니다. 먼저 x(0)와 y(0)을 초기화한 다음 x(k) ∼ f(x|y(k−1)) 및 y(k) ∼ f(y|x(k))를 반복적으로 생성합니다. 아래의 R 코드는 15.2를 번역한 것입니다.

rbinormal <- function(n, mu1, mu2, sigma1, sigma2, rho) { # initialize

- x <- rnorm(1, mu1, sigma1)
- y <- rnorm(1, mu2, sigma2) xy <- matrix(nrow = n, ncol = 2, dimnames = list(NULL,

c("X", "Y"))) # sample from conditional distributions for (i in 1:n) {

- x <- rnorm(1, mu1 + sigma1/sigma2 _ rho _ (y - mu2),

- sqrt(1 - rho^2) \* sigma1)

y <- rnorm(1, mu2 + sigma2/sigma1 _ rho _ (x - mu1),

- sqrt(1 - rho^2) \* sigma2)

xy[i, ] <- c(x, y)

} xy

}

그림 15.1은 µX = 0, σX = 2, µY = 1, σY = 3, ρ = 0.7인 이변량 정규 분포에 대한 깁스 샘플링의 처음 20단계를 보여줍니다.
set.seed(123) n <- 20 z <- rbinormal(n, mu1 = 0, mu2 = 1, sigma1 = 2, sigma2 = 3,

rho = 0.7) plot(z, pch = 19) arrows(z[-n, 1], z[-n, 2], z[-1, 1], z[-1, 2], length = 0.15,

col = "gray40")

그리고 몇 가지 샘플을 추출할 수도 있습니다.

z <- rbinormal(5000, 0, 1, 2, 3, 0.7) smoothScatter(z, nbin = 64) points(0, 1, col = "white", pch = 19) # 이론적 평균

그림 15.2는 이 분포에서 추출한 5,000개의 샘플을 보여주며 샘플 평균, 표준 편차 및 상관 관계를 계산할 수 있는데, 이는 해당 이론적 값에 근접해야 합니다.

6

4

2

Y

0

- -6
- -4
- -2

-3 -2 -1 0 1 2 3

X

- 그림 15.1: 이변량 정규 분포에 대한 깁스 샘플링 추적: 화살표는 깁스 샘플링의 처음 20단계를 보여줍니다.

-6 -4 -2 0 2 4 6

-5

0

5

10

X

Y

- 그림 15.2: 깁스 샘플링의 5000개 지점: 평활화된 산점도는 2D 분포의 밀도를 보여줍니다.

apply(z, 2, mean) # 표본 평균

## X Y ## 0.001287 0.971010

apply(z, 2, sd) # 표본 표준 편차

## X Y ## 1.973 2.971

cor(z) # 표본 상관 관계

## X Y ## X 1.0000 0.6948 ## Y 0.6948 1.0000

이 소규모 애플리케이션에서는 캐시(비록 이 특정 예시가 너무 느리지는 않지만)와 TikZ 그래픽을 사용했습니다. 플롯 크기를 조정했습니다(그림 15.1의 경우 5 × 3, 그림 15.2의 경우 5 × 4). 서술과 코드 청크가 서로 얽혀 있어서 독자는 동일한 보고서에서 이론을 배우고, 컴퓨팅을 보고, 결과를 확인할 수 있습니다. 모든 것이 투명하므로 오류를 쉽게 찾아낼 수 있습니다. 때로는 우리가 작성한 컴퓨터 코드가 이론에서 말한 내용을 제대로 반영하지 못할 수 있으며, 컴퓨팅과 보고를 분리하면 그러한 오류를 찾아내기 어려울 것입니다.

데이터, 코드 및 소프트웨어 공유 측면에서 볼 때, 출판 자료를 공유하고 연구를 완전히 재현 가능하게 만드는 일은 아직 사람들의 선의나 자기 규율에 의존할 수 없습니다.

Huang과 Gottardo (2013) 생의학 데이터의 비교 가능성 및 재현성

재현 가능한 연구를 위해 데이터 분석 시 데이터, 코드, 소프트웨어를 공유하자는 제안이 계속해서 나오고 있습니다(Huang과 Gottardo(2013)). 우리는 교육 분야에서의 더 많은 노력이 중요한 단계가 되어야 한다고 믿으며, 재현 가능한 과제부터 시작할 수 있습니다.

###### 15.2 동적 문서 제공

servr 패키지(Xie, 2015c)는 httpuv 패키지를 기반으로 지정된 디렉터리 아래의 파일을 제공하는 몇 가지 간단한 HTTP 서버 함수를 제공합니다. 어느 정도는 이 패키지가 Python에 익숙한 경우 python -m SimpleHTTPServer 또는 python -m http.server와 유사합니다. 원래 디렉터리 아래의 정적 파일을 제공하도록 설계되었으며 주요 함수는 httd()였습니다.

servr::httd("./")

R 콘솔에서 위 함수를 실행하면 R이 웹 브라우저를 실행하여 현재 작업 디렉터리(./) 아래의 파일 목록을 표시하거나 해당 파일이 있으면 index.html을 표시합니다. 파일의 링크를 클릭하여 해당 내용을 볼 수 있습니다.

이후에 servr는 knitr 및 rmarkdown을 기반으로 확장되어 동적 R Markdown 문서도 제공할 수 있게 되었습니다. 이 패키지에는 (knitr 또는 rmarkdown을 통해) R Markdown 문서에서 생성된 HTML 파일을 제공하는 jekyll(), rmdv1() 및 rmdv2() 함수가 있습니다. HTML 출력 파일이 해당 소스 파일보다 오래된 경우 R Markdown 문서를 자동으로 다시 컴파일할 수 있으며, 그에 따라 웹 브라우저의 HTML 페이지를 자동으로 새로 고칠 수 있습니다. 따라서 R Markdown 문서를 작성하는 데 집중할 수 있으며 결과는 웹 브라우저에서 즉시 업데이트됩니다. 이를 통해 Knit HTML 버튼을 클릭하고 웹 브라우저를 새로 고치는 두 단계를 생략할 수 있습니다. 두 단계 모두 보고서를 작성할 때 주의를 분산시킬 수 있습니다. servr를 사용하면 서버를 시작한 후 R Markdown 문서를 작성하기만 하면 됩니다.

RStudio IDE에서 R Markdown 문서를 작성할 때 이는 훨씬 더 유용합니다. servr가 RStudio IDE를 감지할 때 웹 브라우저를 기본적으로 RStudio 뷰어로 설정하므로 그림 15.3의 레이아웃처럼 소스 문서와 해당 출력을 나란히 놓을 수 있기 때문입니다. RStudio를 사용하지 않아도 전혀 문제가 없습니다. 다른 편집기와 웹 브라우저를 사용하더라도 자동 컴파일 및 새로 고침 기능이 작동합니다.

rmdv1() 및 rmdv2() 함수는 각각 R Markdown v1 및 v2에 해당합니다. R 콘솔에서 servr::rmdv1() 또는 servr::rmdv2()를 호출한 후 소스 문서가 foo.Rmd인 경우 HTML 파일 foo.html을 클릭하여 HTML 출력을 볼 수 있습니다. 그런 다음 foo.Rmd를 편집하고 저장할 때마다 servr가 자동으로 이를 다시 컴파일하고 HTML 출력 페이지를 새로 고칩니다.

jekyll() 함수는 rmdv1() 및 rmdv2()와 유사하지만 Jekyll 웹사이트에 맞게 조정되었습니다. 13.4절에서 Jekyll을 간략하게 소개한 바 있습니다.

![image 33](Dynamic Documents with R and knitr 2nd_images/imageFile33.png)

- 그림 15.3: R Markdown 문서의 레이아웃(왼쪽 상단 패널)과 RStudio 뷰어의 출력(오른쪽 패널): R 콘솔(왼쪽 하단)에 servr 함수를 입력하면 RStudio 뷰어에 R Markdown의 출력이 표시됩니다. 이 그림은 설명 목적으로만 제공됩니다. 그림 안의 텍스트를 읽고 싶다면 https://github.com/yihui/servr에서 원본 이미지를 확인하십시오.

R Markdown 게시물이나 페이지를 Markdown으로 반복해서 컴파일하는 것은 지루한 일이며, 이것이 jekyll()이 유용한 이유입니다. Jekyll 웹사이트의 루트 디렉터리에서 servr::jekyll() 함수를 호출하면 웹 브라우저에서 웹사이트 미리보기를 얻을 수 있습니다. 또한 블로그 게시물을 편집하고 저장하면 웹 브라우저에서 페이지를 새로 고쳐 업데이트된 출력을 표시합니다. knitr-jekyll 저장소(https://github.com/yihui/knitr-jekyll)는 servr를 사용하여 Jekyll 웹사이트를 제공하는 예시입니다.

15.4절에서 패키지 비네트(vignette)를 소개할 예정이며, servr의 vign() 함수를 사용하여 R 패키지를 개발하는 동안 HTML 비네트를 제공할 수 있습니다. 비네트를 제공할 때 소스 패키지에 HTML 출력 파일을 보존하지 않아 소스 패키지를 깔끔하게 유지할 수 있다는 장점이 있습니다.

기술적인 세부 사항이 궁금한 분들을 위해 설명하자면 구현은 WebSockets를 기반으로 합니다. servr가 HTML 페이지를 표시할 때 약간의 JavaScript 코드도 삽입하여 주기적으로(1초 간격) R과 통신하기 위한 WebSocket 연결을 설정합니다. R이 WebSocket으로부터 요청을 받을 때마다 Rmd 파일과 해당 출력 HTML 파일의 타임스탬프를 비교합니다. Rmd 파일이 해당 HTML 출력보다 최신인 경우, servr는 knitr 또는 rmarkdown을 호출하여 Rmd 파일을 HTML로 다시 컴파일한 다음 WebSocket에 메시지를 다시 보냅니다.

all: example.html %.html: %.Rmd

Rscript -e "rmarkdown::render( $^ )"

- 그림 15.4: servr의 make() 함수에 대한 Makefile 예시: 생성할 HTML 파일이 대상 all에 지정되어 있으며, rmarkdown을 통해 Rmd 파일에서 HTML 파일을 생성하는 방법에 대한 규칙이 지정되어 있습니다.

WebSocket이 이 메시지를 받으면 JavaScript에서 location.reload()를 호출하여 페이지를 새로 고칩니다.

이 프로세스의 중요한 단계는 Rmd 파일을 다시 컴파일해야 하는지 여부를 확인하는 것입니다. 이는 GNU Make(http://www.gnu.org/software/make/)가 잘 수행하는 작업이므로, servr는 필요한 경우 Rmd 파일을 다시 빌드할 고유한 Makefile을 제공할 수 있도록 make() 함수도 제공했습니다. 그림 15.4는 make() 함수에 대한 Makefile 예시입니다.

기본적으로 서버 함수는 현재 R 세션을 차단하므로 동일한 R 세션에서 계속 작업하려는 경우 문제가 될 수 있습니다. 이 문제를 해결하려면 서버 함수에 daemon = TRUE 인수를 사용할 수 있습니다. 예: httd(daemon = TRUE) 또는 rmdv2(daemon = TRUE). 이는 servr에게 현재 R 세션을 차단하지 않는 데몬화된 서버를 시작하도록 지시합니다.

###### 15.3 웹사이트 및 블로그 작성

이 절에서는 knitr를 기반으로 구축된 몇 가지 웹사이트와 블로그를 소개하며, 웹 페이지는 R Markdown 또는 R HTML에서 생성됩니다.

###### 15.3.1 Vistat 및 Rcpp 갤러리

Vistat(http://vis.supstat.com)은 R Markdown과 Jekyll(13.4절)을 기반으로 하는 웹사이트입니다. 재현 가능한 통계 그래픽 갤러리를 제공하는 것을 목표로 합니다. 웹사이트 저장소는 Github(https://github.com/supstat/vistat)에서 공개적으로 사용할 수 있습니다.

이 저장소의 핵심은 일부 전역 청크 옵션을 설정하고 Rmd 문서를 Markdown 출력으로 컴파일하는 R 스크립트 ./\_bin/knit입니다. 수식은 MathJax에 의해 렌더링되고 애니메이션은 SciAnimator 라이브러리(7.3.1절)를 통해 지원되며, D3 라이브러리를 통해 웹 그래픽을 만들 수도 있습니다.

knitr가 Rmd 소스 파일을 Markdown 파일로 컴파일하고 나면, Jekyll이 Markdown을 HTML로 컴파일하여 웹사이트를 제공할 수 있습니다.

Rcpp 갤러리(http://gallery.rcpp.org)는 Rcpp(Eddelbuettel 등, 2015) 기사 및 예시를 위한 웹사이트이며 역시 R Markdown을 기반으로 구축되었습니다. 특히 knitr의 Rcpp 엔진(11.2.1절)을 사용합니다.

###### 15.3.2 UCLA R 튜토리얼

UCLA 통계 컨설팅 그룹(Statistical Consulting Group)은 수년간 여러 통계 패키지에 대한 소프트웨어 튜토리얼을 유지 관리해 왔으며 그중 하나는 R 전용(http://www.ats.ucla.edu/stat/r/)입니다. 2012년 이전에 이 웹사이트는 복사 및 붙여넣기 방식으로 구축되었습니다. 결과를 R에서 생성하여 HTML 페이지에 복사하는 방식이었습니다. 2012년 knitr가 출시된 후, 웹 관리자 중 한 명인 Joshua Wiley는 R HTML 형식을 사용하는 대신 knitr를 사용하여 R 튜토리얼 페이지를 다시 작성하기로 결정했습니다. 이제 웹 페이지를 유지 관리하기가 훨씬 쉬워졌고 R 출력의 재현성도 훨씬 향상되었습니다. R이 업데이트되거나 데이터세트가 변경된 후에는 모든 소스 문서를 다시 컴파일하여 전체 웹사이트를 자동으로 재빌드할 수 있습니다.

###### 15.3.3 cda 및 RHadoop Wiki

Github에는 각 저장소에 대한 통합 Wiki 시스템이 있습니다. Markdown, reStructuredText 등 다양한 형식으로 위키 페이지를 작성할 수 있습니다. 각 페이지는 본질적으로 파일이며 위키는 기본적으로 Git 저장소입니다. 따라서 Rmd 파일을 작성하고 Markdown 파일로 컴파일한 다음 Git을 통해 Github에 푸시할 수 있습니다.
porated.

Gentleman, R. (2005). Reproducible research: A bioinformatics case study. Statistical Applications in Genetics and Molecular Biology, 4(1):1034.

Gentleman, R. and Temple Lang, D. (2004). Statistical analyses and reproducible research. Bioconductor Project Working Papers. URL: http://biostats.bepress.com/bioconductor/paper2.

Gove, J. H. (2013). sampSurf: Sampling Surface Simulation for Areal Sampling Methods. R package version 0.6-8.

Gruber, J. (2004). The Markdown Project. URL: http://daringfireball. net/projects/markdown/.

Guo, J., Betancourt, M., Brubaker, M., Carpenter, B., Gao, Y., Goodrich, B., Hoffman, M., Lee, D., Li, P., Malecki, M., and Gelman, A. (2014). rstan: RStan: R interface to Stan. R package version 2.5.0.

Harrell, Jr., F. E. (2001). Regression Modeling Strategies: With Applications to Linear Models, Logistic Regression, and Survival Analysis. Springer New York.

Harrell, Jr., F. E. (2015). Hmisc: Harrell Miscellaneous. R package version 3.15-0.

Horner, J. (2011). brew: Templating Framework for Report Generation. R package version 1.0-6.

Horton, N., Aloisio, K., Zhang, R., and Loi, L. (2012). The statistical sleuth (2nd edition) in R. URL: http://www.math.smith.edu/ ~nhorton/sleuth/.

Huang, Y. and Gottardo, R. (2013). Comparability and reproducibility of biomedical data. Brieﬁngs in Bioinformatics, 14(4):391–401.

Ihaka, R. and Gentleman, R. (1996). R: A language for data analysis and graphics. Journal of Computational and Graphical Statistics, 5(3):299– 314.

Jockers, M. L. (2014). Text Analysis with R for Students of Literature. Springer.

- Knuth, D. E. (1983). The WEB system of structured documentation. Technical report, Department of Computer Science, Stanford University.
- Knuth, D. E. (1984). Literate programming. The Computer Journal, 27(2):97–111.

Lebanon, G. (2012). Probability: The Analysis of Data, volume 1. CreateSpace Independent Publishing Platform.

Lecoutre, E. (2014). R2HTML: HTML exportation for R objects. R package version 2.3.1.

Leisch, F. (2002). Sweave: Dynamic generation of statistical reports using literate data analysis. In COMPSTAT 2002 Proceedings in Computational Statistics, number 69, pages 575–580. Heidelberg: Physica Verlag.

Lenth, R. V. and Højsgaard, S. (2007). Sasweave: Literate programming using sas. Journal of Statistical Software, 19(8):1–20.

Murdoch, D. (2012). tables: Formula-driven table generation. R package version 0.7.

Murphy, D. (2012). Changes and additions to ggplot2 0.9.0. URL: https://github.com/djmurphy420/ggplot2-transition-guide. Murrell, P. (2011). R Graphics, Second Edition. Chapman & Hall/CRC. Murrell, P. and Ripley, B. (2006). Non-standard fonts in PostScript and

PDF graphics. R News, 6(2):41–47.

Oetiker, T., Partl, H., Hyna, I., and Schlegl, E. (1995). The not so short introduction to LATEX2ε. URL: http://www.ctan.org/tex-archive/ info/lshort/.

Peng, R. (2009). Reproducible research and biostatistics. Biostatistics, 10(3):405–408.

Peng, R. D. (2012). cacheSweave: Tools for caching Sweave computations. R package version 0.6-1.

Qiu, Y. and Xie, Y. (2015). highr: Syntax Highlighting for R Source Code. R package version 0.5.

Qiu, Y., Xie, Y., and Bracken, C. (2015). R2SWF: Convert R Graphics to Flash Animations. R package version 0.9.

- R Core Team (2014). R Language Deﬁnition. R Foundation for Statistical Computing, Vienna, Austria.
- R Core Team (2015). R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.

Ramsey, F. and Schafer, D. (2002). The Statistical Sleuth: A Course in Methods of Data Analysis, Second Edition. Duxbury Press.

Ramsey, N. (1994). Literate programming simpliﬁed. Software, IEEE, 11(5):97–105.

Rossini, A. (2002). Literate statistical analysis. In Proceedings of the 2nd International Workshop on Distributed Statistical Computing, pages 15– 17, Vienna, Austria.

Rossini, A., Heiberger, R., Sparapani, R., Maechler, M., and Hornik, K. (2004). Emacs speaks statistics: A multiplatform, multipackage development environment for statistical analysis. Journal of Computational and Graphical Statistics, 13(1):247–261.

Schulte, E., Davison, D., Dye, T., and Dominik, C. (2012). A multilanguage computing environment for literate programming and reproducible research. Journal of Statistical Software, 46(3):1–24.

Sharpsteen, C. and Bracken, C. (2015). tikzDevice: R Graphics Output in LaTeX Format. R package version 0.8.1.

Tantau, T. (2008). The TikZ and PGF Packages. URL: http:// sourceforge.net/projects/pgf/.

Tantau, T., Wright, J., and Miletic, V. (2012). User’s Guide to the Beamer Class. URL: http://bitbucket.org/rivanvx/beamer.

Temple Lang, D., Swayne, D., Wickham, H., and Lawrence, M. (2014). rggobi: Interface between R and GGobi. R package version 2.1.20.

Vaidyanathan, R. (2012). slidify: Generate reproducible html5 slides from R markdown. R package version 0.4.5.

Vaidyanathan, R., Cheng, J., Allaire, J., Xie, Y., and Russell, K. (2014). htmlwidgets: HTML Widgets for R. R package version 0.3.2.

van Heesch, D. (2008). Doxygen: Source code documentation generator tool. URL: http://www.doxygen.org/.

Venables, W. N. and Ripley, B. D. (2002). Modern Applied Statistics with S. Springer-Verlag, 4th edition.

Wei, T. (2013). corrplot: Visualization of a correlation matrix. R package version 0.73.

Wickham, H. (2015). evaluate: Parsing and Evaluation Tools that Provide More Details than the Default. R package version 0.7.

Wickham, H., Danenberg, P., and Eugster, M. (2015). roxygen2: In-Source Documentation for R. R package version 4.1.1.

- Xie, Y. (2013). runr: Run External Programs from R. R package version 0.0.6.
- Xie, Y. (2014). printr: Automatically Print R Objects According to knitr Output Format. R package version 0.0.3.
- Xie, Y. (2015a). formatR: Format R Code Automatically. R package version 1.2.

- Xie, Y. (2015b). knitr: A General-Purpose Package for Dynamic Report Generation in R. R package version 1.10.
- Xie, Y. (2015c). servr: A Simple HTTP Server to Serve Static Files or Dynamic Documents. R package version 0.2.

Yin, T., Cook, D., and Lawrence, M. (2012). ggbio: an R package for extending the grammar of graphics for genomic data. Genome Biology, 13(8):R77.

![image 35](Dynamic Documents with R and knitr 2nd_images/imageFile35.png)

통계학

초보자와 숙련자 모두에게 적합한 "R과 knitr를 활용한 동적 문서(Dynamic Documents with R and knitr)" 제2판은 컴퓨팅을 보고서 작성에 직접 통합함으로써 통계 보고서 작성을 더 쉽게 만들어 줍니다. 보고서의 범위는 과제, 프로젝트, 시험, 책, 블로그, 웹 페이지부터 통계 그래픽, 컴퓨팅, 데이터 분석과 관련된 거의 모든 문서에 이릅니다. 이 책은 초보자를 위한 기본 응용 프로그램을 다루는 동시에, 고급 사용자들이 knitr 패키지의 확장성을 이해할 수 있도록 안내합니다.

###### 제2판의 새로운 점

- • R Markdown v2를 소개하는 새로운 장 추가
- • knitr 패키지의 개선 사항을 반영한 변경
- • 테이블 생성, 코드 청크 내 객체의 사용자 정의 인쇄 메서드 정의, C/Fortran 엔진, Stan 엔진, 영구 세션에서의 엔진 실행, 동적 문서를 제공하기 위한 로컬 서버 시작에 대한 새로운 섹션 추가

높은 평가를 받았던 이전 판과 마찬가지로 이 책은 보고서 작성의 효율성을 높이는 방법을 보여줍니다. 이 책은 프로그램 출력에서부터 출판 품질의 보고서에 이르기까지 여러분의 보고서 모든 측면을 미세하게 조정할 수 있도록 돕습니다. 패키지에 대한 데모 및 기타 정보는 저자의 웹사이트에서 확인할 수 있습니다.

Yihui Xie는 RStudio의 소프트웨어 엔지니어입니다. 그는 Iowa 주립대학교 통계학과에서 박사 학위를 받았습니다. 그의 연구는 대화형 통계 그래픽 및 통계 컴퓨팅에 중점을 두고 있습니다. 그는 활발한 R 사용자이며 여러 차례 수상 경력이 있는 R 패키지의 저자이기도 합니다. 또한 중국의 대규모 온라인 통계 커뮤니티인 "Capital of Statistics"의 설립자입니다.

K25425

w w w . c r c p r e s s . c o m

제2판

DynamicDocumentswithRandknitr

Xie

## R 시리즈

# R과 knitr를 활용한 동적 문서

제2판

Yihui Xie

K25425_cover.indd 1 4/17/15 11:01 AM
