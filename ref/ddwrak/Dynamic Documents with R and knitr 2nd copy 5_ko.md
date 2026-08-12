### 5

###### 문서 형식

knitr 패키지의 설계는 이론상 어떠한 일반 텍스트 문서도 처리할 수 있을 만큼 유연합니다. 다음은 이 설계의 세 가지 핵심 구성 요소입니다.

- 1. 소스 구문 분석기 (parser)
- 2. 코드 평가기 (evaluator)
- 3. 출력 렌더러 (renderer)

구문 분석기는 소스 문서를 분석하여 문서 내의 인라인 코드뿐만 아니라 컴퓨터 코드 청크를 식별합니다. 평가기는 코드를 실행하고 결과를 반환합니다. 렌더러는 계산된 결과를 적절한 형식으로 구성하며, 이는 최종적으로 원본 문서와 결합됩니다.

코드 평가기는 문서 형식과 무관하지만, 구문 분석기와 렌더러는 문서 형식을 고려해야 합니다. 전자는 입력 구문에 해당하고 후자는 출력 구문과 관련이 있습니다.

###### 5.1 입력 구문

정규 표현식(Friedl, 2006, 또는 위키백과 참조)은 문서에서 인라인 코드와 같은 다른 요소 및 코드 블록(청크)을 식별하는 데 사용됩니다. 이러한 정규 표현식 패턴은 knitr의 all_patterns 객체에 저장됩니다. 예를 들어, Rnw 문서에서 코드 청크의 시작을 나타내는 패턴은 다음과 같습니다.

all_patterns$rnw$chunk.begin ## [1] "^\\s*<<(.*)>>=.\*$"

정규 표현식에서 ^는 문자열의 시작을 의미합니다. \s*는 임의의 개수(0개 포함)의 공백 문자와 일치합니다. .*는 임의의 문자와 임의의 개수로 일치합니다. 이 정규 표현식은 “줄 시작 부분의 임의의 공백 문자 + << + 임의의 문자 + >>=”를 의미하므로 아래의 줄들은 가능한 청크 헤더입니다.

<<>>= <<foo>>= <<bar, echo=TRUE>>=

<<a=1, b=2>>=

그리고 다음은 유효하지 않은 청크 헤더입니다(첫 번째는 줄 시작 부분에 < <가 나타나지 않으며, 두 번째는 >가 하나뿐이고, 세 번째는 =가 없습니다).

hi<<>>= <<foo>= <<bar>>

위 정규 표현식에 대한 두 가지 기술적 참고 사항은 다음과 같습니다.

- 1. 정규 표현식에서 \s는 공백을 나타내지만, R에서는 이중 백슬래시를 작성해야 합니다. 왜냐하면 R 문자열에서 \\는 실제로는 하나의 백슬래시를 의미하기 때문입니다(첫 번째 백슬래시는 이스케이프 역할을 하여 역시 백슬래시인 두 번째 문자를 처리합니다). 초보자에게 이스케이프 문자로 사용되는 백슬래시는 꽤 혼란스러울 수 있으며, 경험상 실제 백슬래시를 원할 때는 두 개의 백슬래시가 필요할 수 있습니다.
- 2. 정규 표현식의 괄호 ()는 문자열 그룹을 형성하여 후방 참조(back references)를 통해 이를 추출할 수 있도록 합니다. 예를 들어, abbbc에서 두 번째 문자 그룹을 추출합니다.

# [b]+ means to match b for one or more times gsub("(a)([b]+)(c)", "\\2", "abbbc")

## [1] "bbb"

우리는 청크 헤더에서 청크 옵션을 추출해야 하며, 이것이 바로 정규 표현식에서 ._를 ()로 감싸서 < <(._)> >=와 같이 작성한 이유입니다.

###### 5.1.1 청크 옵션

3장에서 언급했듯이, 청크 헤더에 청크 옵션을 작성할 수 있습니다. 청크 옵션의 구문은 R의 함수 인수 구문과 거의 동일합니다. 다음과 같은 형태를 취합니다.

option = value

R의 구문과 일관성이 있기 때문에 이 구문에 대해 특별히 기억할 것은 없습니다. 옵션 값이 유효한 R 코드이기만 하면 knitr에서도 유효합니다. echo = TRUE(논리값), out.width = '\\linewidth'(문자열) 또는 fig.height = 5(숫자)와 같은 상수 값 외에도, 청크 옵션에 대해 임의의 유효한 R 코드를 작성할 수 있어 소스 문서를 프로그래밍 가능하게 만듭니다. 다음은 간단한 예시입니다.

<<foo, eval=if (bar < 5) TRUE else FALSE>>=

이 청크 앞에 소스 문서에서 생성된 숫자 변수 bar가 있다고 가정해 보겠습니다. if (bar < 5) TRUE else FALSE 표현식을 eval 옵션에 전달할 수 있으며, 이로 인해 eval 옵션은 bar의 값에 의존하게 됩니다. 결과적으로 bar의 값에 따라 이 청크를 평가합니다(5보다 크면 청크가 평가되지 않음). 즉, 특정 청크를 선택적으로 평가할 수 있습니다. 이 예시는 청크 옵션에 아무리 복잡한 R 표현식도 작성할 수 있음을 보여줍니다. 사실, bar < 5 표현식이 일반적으로 TRUE 또는 FALSE를 반환하므로(bar가 NA가 아닌 경우) eval = bar < 5로 간소화할 수 있습니다.

###### 5.1.2 청크 레이블

유일한 예외는 청크 레이블로, 이는 구문 규칙을 따를 필요가 없습니다. 다시 말해, 유효하지 않은 R 코드여도 됩니다. 이는 역사적 이유(Sweave 관례)와 편의성(따옴표 입력 생략) 때문입니다. 엄밀히 말하자면, 청크 옵션의 일부인 청크 레이블은 문자 값을 가져야 하므로 따옴표로 묶어야 합니다. 하지만 대부분의 경우 knitr은 따옴표가 없는 레이블을 처리하고 내부적으로 따옴표를 추가할 수 있습니다. 레이블 표현식에 사용된 객체가 존재하지 않더라도 가능합니다. 다음은 청크 레이블을 작성하는 유효한 방법들입니다.

<<foo>>= <<foo-bar>>= <<foo_bar>>= <<"foo">>= << foo-bar >>= <<label="foo">>= <<echo=FALSE, label="foo-bar">>=

청크 레이블은 문서 내에서 고유한 식별자(ID) 역할을 해야 하며, 주로 이미지(7장) 및 캐시 파일(8장)과 같은 외부 파일을 생성하는 데 사용됩니다. 내용이 있는 두 청크가 동일한 레이블을 가지면 한 청크에서 생성된 파일이 다른 청크를 덮어쓸 위험이 있으므로 knitr은 작동을 멈추고 오류 메시지를 출력합니다. 청크 레이블을 비워두면 knitr은 unnamed-chunk-i 형태의 레이블을 자동으로 생성하며, 여기서 i는 1, 2, 3 등 순차적으로 증가하는 청크 번호입니다.

###### 5.1.3 전역 옵션

6장부터 11장까지 더 자세히 살펴보겠지만, 청크 옵션은 코드 청크의 모든 측면을 제어합니다. 대부분의 청크에서 공통적으로 사용되는 특정 옵션이 있다면, opts_chunk 객체를 사용하여 전역 청크 옵션으로 설정할 수 있습니다. 전역 옵션은 옵션이 설정된 위치 이후의 모든 다음 청크에서 공유되며, 청크 헤더의 지역 옵션이 전역 옵션을 덮어쓸 수 있습니다. 예를 들어, echo 옵션을 전역적으로 FALSE로 설정합니다.

opts_chunk$set(echo = FALSE)

그러면 아래 두 청크에 대해 echo는 각각 FALSE와 TRUE가 됩니다.

<<foo>>= 1+1 @ <<bar, echo=TRUE>>= rnorm(10) @

###### 5.1.4 청크 구문

문학적 프로그래밍(literate programming)의 원래 구문은 사실 다음과 같습니다. 컴퓨터 코드의 시작을 나타내는 마커 하나(< <> >=)와 문서의 시작을 나타내는 마커 하나(@)를 사용합니다. 이는 3장에서 소개한 내용과 약간 다릅니다. 문학적 프로그래밍 패러다임에서는 소스 문서가 다음과 같은 모습일 수 있습니다.

@ This is documentation. @

Another line of documentation. <<>>= 1 + 1 # some code <<>>= rnorm(10) # another code chunk @ More documentation.

knitr 구문에서는 코드 청크와 문서 청크를 각각 여는 대신 코드 청크를 열고 닫습니다. 전통적인 구문을 사용하지 않는 이유는 보고서에서 코드 청크가 일반 텍스트보다 덜 자주 나타나는 경향이 있으므로, 우리는 코드 청크의 구문에만 집중하기 때문입니다. 또한 코드를 보고서에 "포함(embedding)"하는 것이 더 직관적으로 보입니다. 새로운 구문을 기반으로 하면 다음 역시 knitr 소스 문서의 합당한 부분입니다.

Documentation here. <<>>= 1+1 <<>>= rnorm(10) @ More documentation.

###### 5.2 문서 형식

지금까지 예시로 Rnw 문서의 구문을 사용해왔습니다. 다음으로는 다른 문서 형식에서 R 코드를 작성하는 방법을 소개하겠습니다. 표 5.1은 구문의 요약입니다. 코드 청크는 모든 문서 형식에서 공백의 개수와 관계없이 들여쓰기가 가능합니다.

###### 5.2.1 마크다운

R 마크다운(Rmd) 문서의 경우, 우리는 `{r}과 ` 사이에 코드 청크를 작성하고 인라인 R 코드는 `r ` 안에 작성합니다. 청크 옵션은 청크 헤더의 닫는 중괄호 전에 작성됩니다. 인라인 R 코드는 백틱(`)을 포함할 수 없다는 점에 유의하세요. 예를 들어, `r pi*2`는 괜찮지만 `r `pi`*2`는 안 됩니다. `pi`\*2가 유효한 R 코드라 할지라도 구문 분석기는 첫 번째 백틱이 인라인 R 코드 표현식을 종료하기 위한 것이 아님을 알 수 없습니다.

표 5.1: 모든 문서 형식의 구문 요약: RLTX, R 마크다운, R HTML, R reStructuredText, RAE

| inline |            |               |     |     |     |               |     |     |
| ------ | ---------- | ------------- | --- | --- | --- | ------------- | --- | --- |
| end    |            |               |     |     |     |               |     |     |
| start  |            |               |     |     |     | //<br><br>### | .   |     |
| format | R<br><br>R | md<br><br>htm | l   |     |     |               |     |     |

in.rcode\*end.rcode--><!--rinlinex-->

.rcode\*%end.rcode\rinline{x}

@\Sexpr{x}

<%x%>

}....:r:`x`

}````rx`

n.rcode\*//end.rcode`rx`

gin.rcode\*###.end.rcode@rx@

endinline

AsciiDoc, R Textile, and brew.

마크다운을 사용하면 읽기 쉽고 쓰기 쉬운 일반 텍스트 형식으로 작성한 다음, 구조적으로 유효한 XHTML이나 HTML로 변환할 수 있습니다. 이메일을 작성하는 방법을 안다면 단 몇 분 만에 배울 수 있습니다(http://en.wikipedia.org/wiki/Markdown). 다음은 간단한 예시입니다.

# First level header ## Second level This is a paragraph. This is bold, and italic.

- - list item
- - list item

백틱은 `<code>` 태그를 생성합니다. This is [a link](url), and this is an ![image](url). A block of code ( <pre> tag):

1 + 1 rnorm(10)

### 세 번째 레벨 섹션 제목 순서가 있는 목록을 작성할 수 있습니다.

- 1. 항목 1
- 2. 항목 2

원래 마크다운 구문은 단순하게 설계되었으므로, 표 작성, LATEX 수학 수식 작성 또는 참고 문헌 등 저작 환경 측면에서 어느 정도 제약이 있는 것은 불가피합니다. 짧은 과제를 작성하는 것과 같은 일부 경우에는 복잡한 기능이 필요하지 않으므로 마크다운으로도 꽤 잘 작동할 것입니다.

마크다운의 한 가지 문제는 파생 버전입니다. Pandoc의 마크다운(http://johnmacfarlane.net/pandoc), Github Flavored 마크다운(http://github.com), kramdown(http://kramdown.rubyforge.org) 등과 같이 수많은 변형이 존재합니다. 이러한 변형들은 표와 같은 특정 요소를 작성하는 방법에 대해 고유한 정의를 가질 수 있습니다. CommonMark(http://commonmark.org)는 마크다운 구문을 모호하지 않게 정의하기 위한 노력이며, Pandoc의 마크다운은 CommonMark 표준과 호환됩니다. 또한, 현재로서는 Pandoc이 마크다운을 위한 포괄적인 도구일 것입니다. Pandoc은 원래 마크다운에 다음과 같은 유용한 확장 기능들을 많이 추가했습니다.

- 1. 세 개의 백틱 쌍으로 둘러싸인 펜스 코드 블록
- 2. 일반 LATEX(PDF 출력용) 또는 MathJax(http://mathjax.org, HTML 출력용)를 통한 LATEX 수학 수식 지원. 이를 통해 $math$ 또는 $$math$$와 같은 LATEX 구문을 사용하여 웹 페이지에 수학 방정식을 작성할 수 있습니다.
- 3. 문서에 대한 메타데이터(제목, 저자 및 날짜 정보)
- 4. 공백이나 파이프로 기둥(열)이 구분된 표
- 5. 정의 목록, 각주 및 인용구 등

다음은 일부 확장 기능이 어떻게 보이는지 보여줍니다.

--title: 내 보고서 제목 author: Yihui Xie

--평소처럼 4칸 들여쓰기하거나 아래에 코드를 작성합니다. r 1 + 1 rnorm(10)

인라인 수식: $\alpha + \beta$. 디스플레이 스타일: $$f(x) = x^{2} + 1$$ 인용에서 가져온 간단한 표 [@joe2014]:

| id | age | sex | |:----|----:|:---:|

- | a | 49 | M |
- | b | 32 | F |

더 중요한 것은, Pandoc이 마크다운을 PDF/LATEX, HTML, 워드(Microsoft Word 또는 OpenOffice) 및 프레젠테이션 슬라이드(LATEX beamer 또는 HTML5 슬라이드)를 포함한 여러 다른 문서 형식으로 변환할 수 있다는 점입니다. R 패키지 rmarkdown(Allaire 외, 2015a)은 knitr과 Pandoc을 기반으로 하며, 사용자가 기본적으로 꽤 아름다운 출력을 빠르게 만들 수 있도록 자주 사용되는 몇 가지 출력 형식을 포함하고 있습니다.

rmarkdown 패키지는 RStudio 개발자들이 도입했으므로, RStudio가 R 마크다운 문서 형식을 잘 지원한다는 것은 놀라운 일이 아닙니다. RStudio에서 Rmd 문서를 열거나 생성할 때(파일 새로 만들기 R Markdown), 원하는 출력 형식을 묻는 마법사가 나타납니다. 14장에서 R 마크다운을 자세히 다루겠습니다.

- 5.2.2 LATEX 마크다운은 주로 웹을 위해 설계되었으며, 더 복잡한 조판 목적을 위해서는 LATEX가 선호될 수 있습니다. 예를 들어, 이 책은 LATEX로 작성되었습니다. Oetiker 등(1995)은 초보자가 LATEX를 배우기에 좋은 고전적인 자습서입니다. 학습 곡선이 가파를 수 있지만 직접 조판하는 것에 관심이 있다면 그만한 가치가 있습니다.

LATEX 문서의 경우, 이전에 여러 번 보았듯이 R 코드 청크는 < <> >=와 @ 사이에 삽입되며 인라인 R 코드는 \Sexpr{} 안에 작성됩니다.

- 5.2.3 HTML

HTML(하이퍼텍스트 마크업 언어, Hyper-Text Markup Language)은 웹 페이지 이면에 있는 언어입니다. 보통 웹 브라우저가 이를 구문 분석하고 요소를 렌더링하기 때문에 HTML 코드를 직접 볼 일은 없습니다. 예를 들어 굵은 글씨를 볼 때 소스 코드는 <strong>굵은 글씨</strong>일 수 있습니다. 대부분의 웹 브라우저는 HTML 소스 코드를 표시할 수 있습니다. 예를 들어 파이어폭스와 구글 크롬에서는 Ctrl + U를 눌러 페이지 소스를 볼 수 있습니다.

HTML에는 페이지의 다양한 요소를 나타내는 크고(하지만 제한적인) 수많은 태그가 존재합니다. 태그와 명령어를 세심하게 구성하여 조판을 정밀하게 제어할 수 있다는 점에서 HTML은 LATEX와 비슷합니다. 단점은 입력해야 할 태그가 많아서 문서를 작성하는 데 오랜 시간이 걸릴 수 있다는 것입니다. 그래서 소규모 문서에는 마크다운이 더 나을 수 있습니다. 어쨌든 HTML이 가진 강력한 기능 때문에 가끔씩은 이를 사용해야 합니다. 다음은 HTML 문서의 예시입니다.

<html> <head>
<title>This is an HTML page</title> </head> <body>

<p>This is a <em>paragraph</em>.</p> <div>A <code>div</code> layer.</div> <!-- I m a comment; you cannot see me. -->

</body> </html>

HTML 문서에서 R 코드를 작성하려면 HTML의 주석 구문을 사용합니다. 예를 들어 다음과 같습니다.

<!--begin.rcode test-html, eval=TRUE 1 + 1 rnorm(10) end.rcode-->

<p>And here is the value of pi: <!--rinline pi -->.</p>

###### 5.2.4 reStructuredText

reStructuredText(reST) 문서(http://docutils.sourceforge.net/rst.html)에도 R 코드를 포함할 수 있습니다. 이는 마크다운과 비슷하지만 더 강력합니다(그만큼 더 복잡합니다). 다음은 R reST 문서에 R 코드가 포함된 예시입니다.

A reST document for knitr =========================

이것은 reStructuredText 문서(\*.Rrst)입니다. knitr을 위해 R 코드를 작성하는 방법은 다음과 같습니다.

.. {r test-rst, eval=TRUE} 1 + 1 rnorm(10)

.. .. pi의 값은 :r: pi 입니다.

Docutils 시스템(파이썬으로 작성됨)은 주로 reST 문서를 HTML로 변환하는 데 사용됩니다.

###### 5.2.5 AsciiDoc

AsciiDoc(http://en.wikipedia.org/wiki/AsciiDoc)은 소프트웨어 문서, 기사, 책 및 HTML 페이지와 같은 여러 유형의 출력으로 변환할 수 있는 일반 텍스트 문서 형식입니다. 다음은 책을 작성하기 위한 최소한의 R AsciiDoc 예시입니다.

= 책 제목 :author: 니터(Knitter)

== 첫 번째 장 Hello world! // begin.rcode test, eval=TRUE 1 + 1 rnorm(10) // end.rcode pi의 값은 r pi 입니다.

###### 5.2.6 Textile

Textile은 또 다른 경량 마크업 언어이며 보통 HTML로 변환됩니다. 위키백과 페이지 http://en.wikipedia.org/wiki/Textile_(markup_language)에서 더 많은 정보를 찾을 수 있습니다.

구문을 보여주는 R Textile 예시는 다음과 같습니다. h1. Textile 파일 니팅(Knitting) Hello world! ###. begin.rcode test, tidy=FALSE if (1 + 1 == 2) {

of course!

} ###. end.rcode

그리고 인라인 표현식 @r 2\*pi@ 입니다.

###### 5.2.7 사용자 정의

소스 문서를 구문 분석하기 위해 고유한 구문을 정의할 수 있습니다. 앞서 살펴보았듯이 구문 분석은 정규 표현식을 통해 이루어집니다. 내부적으로 knitr은 정규 표현식을 관리하기 위해 knit_patterns 객체를 사용합니다. 예를 들어, 이 책을 위한 세 가지 주요 패턴은 다음과 같습니다.

knit_patterns$get(

c("chunk.begin", "chunk.end", "inline.code") )

## $chunk.begin ## [1] "^\\s*<<(.*)>>=.*$" ## ## $chunk.end ## [1] "^\\s*@\\s*(%+.*|)$" ## ## $inline.code ## [1] "\\\\Sexpr\\{([^}]+)\\}"

우리만의 구문을 지정하려면 기본 구문을 덮어쓰는 knit_patterns$set()을 사용할 수 있습니다. 예를 들면 다음과 같습니다. knit_patterns$set(

chunk.begin = "^<<r(.\*)", chunk.end = "^r>>$", inline.code = "\\{\\{([^}]+)\\}\\}"

)

그러면 사용자 정의 구문을 사용하여 다음과 같이 문서를 구문 분석할 수 있습니다.

<<r test-syntax, eval=TRUE 1 + 1 x <- rnorm(10) r>>

x의 평균은 {{mean(x)}}입니다.

하지만 실제로는 이러한 사용자 정의가 불필요한 경우가 많습니다. 기본 구문을 따르는 것이 좋으며, 그렇지 않으면 소스 문서를 컴파일하기 위해 추가 지침이 필요합니다.

knitr에는 pat\_ 접두사가 붙은 일련의 함수들이 있는데, 이들은 구문 패턴을 설정하기 위한 편의 함수들입니다. 예를 들어 pat_rnw()는 knit_hooks$set()을 호출하여 Rnw 문서를 위한 패턴을 설정합니다. 모든 패턴 함수는 다음과 같습니다.

grep("^pat\_", ls("package:knitr"), value = TRUE) ## [1] "pat_asciidoc" "pat_brew" "pat_html"

- ## [4] "pat_md" "pat_rnw" "pat_rst" ## [7] "pat_tex" "pat_textile"

소스 문서를 구문 분석할 때 knitr은 파일 확장자에 따라 사용할 패턴 목록을 먼저 결정합니다. 예를 들어 \*.Rmd 문서는 R 마크다운 구문을 사용합니다. 파일 확장자를 알 수 없는 경우 knitr은 문서 내의 코드 청크를 추가로 감지하여 구문이 기존 패턴 목록과 일치하는지 확인합니다. 일치하는 경우 해당 패턴 목록이 사용됩니다. 예를 들어 foo.txt 파일의 경우 knitr은 txt 확장자를 인식하지 못하지만 이 파일에 ```{r}로 시작하는 코드 청크가 포함되어 있으면 knitr은 R 마크다운 구문을 자동으로 사용합니다.

###### 5.3 출력 렌더러

evaluate 패키지(Wickham, 2015)는 코드 청크를 실행하는 데 사용되며 기본 R의 eval() 함수는 인라인 R 코드를 실행하는 데 사용됩니다. 후자는 이해하기 쉽고 R의 "언어 기반 컴퓨팅(computing on the language)"(R 코어 팀, 2014)의 힘으로 가능해졌습니다. 문자열로 된 코드 조각 1+1이 있다고 가정하면, 이를 구문 분석하고 R 코드로 평가할 수 있습니다.

eval(parse(text = "1+1")) ## [1] 2

코드 청크의 경우는 더 복잡합니다. evaluate 패키지는 R 소스 코드를 가져와 평가하고 6가지 가능한 클래스(character(일반 텍스트 출력), source(소스 코드), warning(경고), message(메시지), error(오류), recordedplot(플롯))의 결과를 포함하는 목록을 반환합니다.

이러한 결과를 출력에 쓰려면 출력 형식을 고려해야 합니다. 예를 들어 소스 코드가 1+1이고 출력 형식이 TEX인 경우 verbatim 환경을 사용할 수 있지만 출력이 HTML이어야 하는 경우 출력에 <pre>1+1</pre>을 쓸 수 있습니다. 핵심적인 질문은 R의 원시(raw) 결과를 어떻게 포장(wrap up)할 것인가입니다. 이에 대한 답은 최종 출력을 구성하기 위한 출력 훅(hook) 함수 목록을 포함하는 knit_hooks 객체에 있습니다. 훅 함수는 종종 다음과 같은 형식으로 정의됩니다.

hook_fun <- function(x, options) {

# returns a character string with markup }

출력 훅에서 x는 대개 R의 원시 출력이며, options는 현재 청크 옵션 목록입니다. 출력 클래스에 해당하는 knit_hooks의 훅 이름은 표 5.2에 나열되어 있습니다.

message() 함수에서 방출되는 메시지 출력을 사용자 정의 LATEX 환경(Rmessage)에 넣고자 한다면, 메시지 훅을 다음과 같이 설정할 수 있습니다.

표 5.2: 출력 훅 함수 및 evaluate 패키지의 결과 객체 클래스

| 클래스       | 출력 훅  | 인수       |
| ------------ | -------- | ---------- |
| source       | source   | x, options |
| character    | output   | x, options |
| recordedplot | plot     | x, options |
| message      | message  | x, options |
| warning      | warning  | x, options |
| error        | error    | x, options |
|              | chunk    | x, options |
|              | inline   | x          |
|              | text     | x          |
|              | document | x          |

knit_hooks$set(message = function(x, options) {

paste0("\\begin{Rmessage}\n", x, "\\end{Rmessage}") })

물론 사전에 LATEX 서문(preamble)에 Rmessage 환경을 정의해야 합니다. 예를 들면 다음과 같습니다. \newenvironment{Rmessage}{

\rule[0.5ex]{1\columnwidth}{1pt} % 수평선 }{

\rule[0.5ex]{1\columnwidth}{1pt} }

그러면 출력에 메시지가 있을 때마다 위아래로 수평선을 볼 수 있습니다.

기본적으로 knitr은 각 출력 형식에 대해 일련의 기본 출력 훅을 설정하므로, 보통 우리가 직접 모든 훅을 설정할 필요는 없습니다. knitr의 render\_ 접두사가 붙은 일련의 함수를 사용하여 다양한 출력 형식에 대한 기본 출력 훅을 설정할 수 있습니다.

grep("^render\_", ls("package:knitr"), value = TRUE)

## [1] "render_asciidoc" "render_html" ## [3] "render_jekyll" "render_latex"

- ## [5] "render_listings" "render_markdown" ## [7] "render_rst" "render_sweave" ## [9] "render_textile"

이것은 여러분이 Sweave 스타일로 돌아가고자 할 때 해야 할 일의 전부입니다.
(생략된 내용 - Sweave 스타일 콘솔 출력 예시)

- 그림 5.1: knitr의 Sweave 스타일. Rnw 문서의 맨 처음에서 render_sweave()를 실행하면 Sweave 스타일을 볼 수 있습니다.

render_latex(), render_html(), render_markdown() 함수는 출력 형식이 각각 LATEX, HTML, 마크다운일 때 호출됩니다. render_sweave()와 render_listings()는 LATEX 출력의 두 가지 변형입니다. 전자는 Sweave.sty에 정의된 전통적인 Sweave 환경(Sinput 및 Soutput 등)을 사용하고, 후자는 출력을 꾸미기 위해 LATEX의 listings 패키지를 사용합니다. 두 스타일이 어떻게 보이는지는 그림 5.1 및 그림 5.2를 참조하세요.

출력 훅을 설정하려는 경우, 나머지 출력에 영향을 미칠 수 있도록 소스 문서의 맨 처음에 설정하는 것이 좋습니다. 예를 들어 아래 청크는 Rnw 문서의 첫 번째 청크가 될 수 있습니다(청크 옵션 include = FALSE는 독자에게 흥미롭지 않으므로 이 청크의 내용을 출력에 표시하지 않음을 의미합니다).

<<setup, include=FALSE>>= render_sweave() @

그러면 출력이 Sweave 스타일로 렌더링됩니다. 이 책은 구문 강조를 지원하는 기본 LATEX 스타일을 사용했으며, 코드 청크는 회색 음영 상자에 배치되었습니다.

1

표 5.2의 모든 출력 훅 중에서 추가 설명이 필요한 5개의 특수 훅이 있습니다.

이것은 여러분이 listings 패키지를 사용하고자 할 때 해야 할 일의 전부입니다.
(생략된 내용 - listings 패키지 스타일 콘솔 출력 예시)

- 그림 5.2: knitr의 listings 스타일. render_listings()는 이와 같은 스타일(색상이 있는 텍스트 및 회색 음영)을 생성합니다.

- • plot 훅은 파일 이름의 문자열(foo.pdf)을 입력 x로 받습니다. 아래는 LATEX 출력을 위한 plot 훅의 단순화된 버전입니다(실제 훅은 out.width 및 out.height 등 고려해야 할 청크 옵션이 많기 때문에 이보다 훨씬 더 복잡합니다).

knit_hooks$set(plot = function(x, options) {

paste("\\includegraphics{", x, "}", sep = "") })

- • chunk 훅은 전체 청크의 출력을 입력으로 사용하며, 이는 source, output, message 등과 같은 다른 훅에서 생성됩니다. 예를 들어 청크 출력을 HTML의 Rchunk 클래스를 가진 div 태그 안에 넣으려면 chunk 훅을 다음과 같이 정의할 수 있습니다.

knit_hooks$set(chunk = function(x, options) {

paste("<div class= Rchunk >", x, "</div>") })

그런 다음 우리는 이 HTML 문서의 CSS 스타일시트에서 Rchunk의 스타일을 정의해야 합니다.

- • inline 훅은 코드 청크와 관련이 없으며, 인라인 R 코드의 출력을 형식화하는 방법을 정의합니다. 예를 들어 우리는 다음과 같이 하길 원할 수 있습니다.

1

인라인 출력의 모든 숫자를 소수점 2자리로 반올림하기 위해 다음과 같이 inline 훅을 정의할 수 있습니다.

knit_hooks$set(inline = function(x) {

###### if (is.numeric(x))

x <- round(x, 2)

as.character(x) # x를 문자로 변환하고 반환합니다 })

knitr은 기본 inline 훅(6.1절)에서 반올림을 처리하므로, 사실 이 훅을 재설정할 필요는 없습니다.

- • text 훅은 텍스트 청크, 즉 서술 부분을 처리합니다. 예를 들어 텍스트 청크 주변의 공백을 자르는 훅을 설정할 수 있습니다.

knit_hooks$set(text = function(x) {

gsub("^\\s*|\\s*$", "", x) })

- • document 훅은 chunk 훅과 유사하며 전체 문서의 출력을 입력 x로 받습니다. 이 훅은 문서를 후처리하는 데 유용할 수 있습니다. 실제로 이 책은 모든 표 캡션 아래(tabular 환경 전)에 수직 간격 \medskip{}을 추가하기 위해 이 훅을 사용했습니다.

knit_hooks$set(document = function(x) {

gsub("\\begin{tabular}", "\\medskip{}\\begin{tabular}",

x, fixed = TRUE) })

###### 5.4 R 스크립트

knitr에는 특별한 소스 문서 형식이 있는데, 본질적으로 roxygen 주석이 있는 R 스크립트입니다(roxygen에 대한 자세한 내용은 Wickham 등(2015) 및 부록 A.1 참조). 일반적인 R 주석은 #으로 시작하고 roxygen 주석은 # 뒤에 아포스트로피가 붙습니다. 예를 들면 다음과 같습니다.

#' 이것은 roxygen 주석입니다
#' 저도요

때로는 R 코드와 일반 텍스트를 섞고 싶지 않고 대신 주석에 텍스트를 작성하여 전체 문서가 유효한 R 스크립트가 되도록 하고 싶을 때가 있습니다. knitr의 spin() 함수는 주석이 roxygen 구문을 사용하여 작성된 경우 이러한 R 스크립트를 처리할 수 있습니다. spin()의 기본 아이디어 역시 문학적 프로그래밍에서 영감을 받았습니다. 이 R 스크립트를 컴파일하면 #'가 제거되어 일반 텍스트가 "복원"되고 R 코드가 평가됩니다. roxygen 주석 뒤에 있지 않은 모든 것은 코드 청크로 취급됩니다. 청크 옵션을 작성하려면 또 다른 유형의 특수 주석인 #+ 또는 #- 뒤에 청크 옵션을 사용하면 됩니다. 다음은 간단한 예시입니다.

#' 여기에 방법을 소개하고 그 다음에 R 코드를 작성합니다.
1 + 1
x <- rnorm(10)

#' 다음과 같이 청크 옵션을 작성하는 것도 가능합니다.
#+ test-label, fig.height=4
plot(x)
#' 이제 문서가 완성되었습니다.

이 스크립트를 test.R이라는 파일에 저장하고 컴파일하여 보고서를 만들 수 있습니다.

library(knitr)
spin("test.R")

spin() 함수에는 출력 문서 형식을 지정하는 format 인수가 있습니다(기본값은 R 마크다운). 예를 들어 format = 'Rnw'인 경우 R 코드는 먼저 < <> >=와 @ 사이에 삽입된 다음, 컴파일되어 LATEX 출력을 생성합니다.

이것은 R 스크립트를 기반으로 보고서를 생성하는 3.3절의 stitch() 함수와 비슷해 보입니다. 하지만 spin()은 텍스트 청크를 작성할 수 있게 해주는 반면 stitch()는 사전 정의된 템플릿만 사용할 수 있어 자유도가 떨어집니다.
