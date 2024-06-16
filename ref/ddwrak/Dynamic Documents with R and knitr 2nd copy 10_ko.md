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
###### 10.2.1 Crop Plots (플롯 자르기)

일부 R 사용자는 R 플롯의 흰 여백(margin), 특히 기본 그래픽에서의 여백 때문에 불편을 겪었을 것입니다(ggplot2는 보통 이 부분에서 낫습니다). 그림 10.1에서 언급했듯이(또한 `?par` 참조) 기본 그래픽 옵션인 mar는 대략 c(5, 4, 4, 2)인데, 이는 종종 지나치게 큽니다. `par(mar)`를 끊임없이 조정하는 대신, 흰 여백을 자동으로 잘라주는 프로그램인 pdfcrop을 고려해 볼 수 있습니다(http://www.ctan.org/pkg/pdfcrop). knitr에서는 `hook_pdfcrop()` 훅이 가령 `crop`이라는 청크 옵션과 함께 작동하도록 설정할 수 있습니다.

-30-10-35-25-20-15

![image 7](Dynamic Documents with R and knitr 2nd_images/imageFile7.png)

![image 8](Dynamic Documents with R and knitr 2nd_images/imageFile8.png)

![image 9](Dynamic Documents with R and knitr 2nd_images/imageFile9.png)

lat

165 170 175 180 185

long

- FIGURE 10.3: The original plot produced in R, with a large white margin.

```r
knit_hooks$set(crop = hook_pdfcrop)
```

이제 아래의 동일한 코드 청크에 의해 생성된 두 개의 플롯을 비교해 보겠습니다. 첫 번째 플롯은 잘리지 않았습니다(그림 10.3). 그다음 플롯은 동일하게 생성되었지만, 자르기 훅을 호출하는 청크 옵션 `crop = TRUE`가 추가되었습니다(그림 10.4).

```r
par(mar = c(5, 4, 4, 2)) # large margin
plot(lat ~ long, data = quakes, pch = 20, col = rgb(0, 0, 0, 0.2))
```

보시다시피 흰 여백이 사라졌습니다(차이를 더 잘 보여드리기 위해 각 플롯 주위에 테두리를 추가했습니다). 만약 `par()`를 사용한다면, 여백이 지나치게 작지 않아 라벨이 잘리지 않으면서도 여백이 지나치게 크지 않은 적절한 크기를 알아내는 것은 어렵고 지루한 일일 수 있습니다.

-35-30-25-20-15-10

![image 10](Dynamic Documents with R and knitr 2nd_images/imageFile10.png)

![image 11](Dynamic Documents with R and knitr 2nd_images/imageFile11.png)

![image 12](Dynamic Documents with R and knitr 2nd_images/imageFile12.png)

lat

165 170 175 180 185

long

- FIGURE 10.4: The cropped plot; obviously the white margins on the top and right have been removed.

###### 10.2.2 rgl Plots (rgl 플롯)

`hook_rgl()` 훅을 사용하면 rgl 패키지에서 스냅샷을 쉽게 저장할 수 있습니다(Adler and Murdoch, 2014). rgl 훅은 훅 내에서 options 인자를 신중하게 사용하여 세부 사항을 처리하는 좋은 예입니다. 예를 들어, `rgl.snapshot()`이나 `rgl.postscript()`에서는 rgl 플롯의 너비와 높이를 직접 설정할 수 없으므로, options의 `fig.width`, `fig.height`, `dpi`를 활용하여 예상 창 크기를 계산한 다음, `par3d()`로 현재 창 크기를 조정하고 플롯을 저장합니다. 마지막으로 플롯을 출력에 삽입하기 위한 적절한 코드가 포함된 문자열을 반환합니다. 다음은 `hook_rgl()`의 빠르고 간단한 버전입니다.

```r
knit_hooks$set(rgl = function(before, options, envir) {
  library(rgl)
  if (before || rgl.cur() == 0)
    return() # return nothing before a chunk

  name <- paste(options$fig.path, options$label, sep = "")
```

![image 13](Dynamic Documents with R and knitr 2nd_images/imageFile13.png)

- FIGURE 10.5: An rgl plot captured by hook_rgl(): this hook function calls rgl.snapshot() in rgl to save the snapshot into a PNG image.

```r
  rgl.snapshot(paste(name, ".png", sep = ""), fmt = "png")
  paste("\\includegraphics{", name, "}\n", sep = "")
})
```

knitr에 있는 실제 훅 함수는 고려해야 할 세부 사항이 많기 때문에 이보다 훨씬 더 복잡합니다. 다음은 rgl 훅을 사용하여 rgl 플롯을 저장하는 방법의 예입니다. 먼저 `hook_rgl()` 함수를 위해 `rgl`이라는 이름의 훅을 정의합니다.

```r
knit_hooks$set(rgl = hook_rgl)
```

그런 다음 청크 옵션을 `rgl = TRUE`로 설정하기만 하면 캡처된 플롯이 그림 10.5와 같이 나타납니다.

```r
library(rgl)
demo("bivar", package = "rgl", echo = FALSE)
par3d(zoom = 0.7)
```

###### 10.2.3 Manually Save Plots (수동으로 플롯 저장하기)

7.2절에서 R 플롯이 어떻게 기록되는지 설명해 드렸습니다. rgl 플롯과 같이 `recordPlot()`으로 플롯을 캡처할 수 없는 경우가 있지만, 다른 함수를 사용하여 저장할 수 있습니다. 이러한 플롯을 출력에 삽입하려면 먼저 다음과 같이 훅을 설정해야 합니다(자세한 내용은 `?hook_plot_custom` 도움말 페이지를 참조하시기 바랍니다).

![image 14](Dynamic Documents with R and knitr 2nd_images/imageFile14.png)

- FIGURE 10.6: A plot created and exported by GGobi, and written into LATEX by the hook hook_plot_custom().

```r
knit_hooks$set(custom_plot = hook_plot_custom)
```

그 후 청크 옵션 `custom_plot = TRUE`로 설정하고 청크에서 직접 플롯 파일을 작성합니다. 다음은 rggobi 패키지의 `ggobi_display_save_picture()` 함수를 사용하여 GGobi 플롯을 캡처하는 예시입니다(Temple Lang et al., 2014).

```r
<<ggobi-plot, custom_plot=TRUE, fig.ext="png">>=
library(rggobi)
data("flea", package = "tourr")
ggobi(flea)
Sys.sleep(1) # wait for snapshot
ggobi_display_save_picture(path = fig_path(".png"))
@
```

그림 10.6은 GGobi에서 출력된 플롯입니다. 여기서 두 가지를 주의해야 합니다.

- 1. 플롯 파일 이름은 반드시 `fig_path()`를 통해 지정해야 합니다. 이 함수는 현재 청크에 대한 그림 경로(청크 라벨, `fig.path` 및 `fig.ext`의 조합)를 반환하는 편리한 함수입니다.
- 2. 그래픽 장치를 사용하지 않기 때문에 knitr가 자동으로 확장자를 파악할 수 없습니다. 따라서 청크 옵션 `fig.ext`(그림 파일 확장자)를 명시적으로 설정해야 합니다.

옵션 `fig.show = 'animate'`(7.3.1절 참조)를 사용하여 연속된 이미지를 저장해 애니메이션을 만들 수도 있습니다. 다음은 rgl을 사용하여 산점도를 확대하는 예시입니다(실제 애니메이션은 knitr의 메인 매뉴얼을 참조하시기 바랍니다).

```r
## use chunk options: custom_plot=TRUE, fig.ext="png",
## out.width="2.5in", fig.show="animate", fig.num=20
library(animation) # adapted from demo("rgl_animation")
data(pollen)
uM <- matrix(c(-0.37, -0.51, -0.77, 0, -0.73, 0.67, -0.1, 0, 0.57, 0.53, -0.63, 0, 0, 0, 0, 1), 4, 4)
library(rgl)
open3d(userMatrix = uM, windowRect = c(0, 0, 400, 400))
plot3d(pollen[, 1:3])
zm <- seq(1, 0.05, length = 20)
par3d(zoom = 1) # change the zoom factor gradually later
for (i in 1:length(zm)) {
  par3d(zoom = zm[i])
  Sys.sleep(0.05)
  rgl.snapshot(paste(fig_path(i), "png", sep = "."))
}
```

###### 10.2.4 Optimize PNG Plots (PNG 플롯 최적화)

무료 소프트웨어인 OptiPNG는 정보를 손실하지 않고 이미지 파일을 더 작은 크기로 다시 압축하는 PNG 최적화 도구입니다(http://optipng.sourceforge.net/). knitr에서 훅 함수인 `hook_optipng()`는 PNG 플롯을 압축하기 위해 OptiPNG를 감싸는 래퍼(wrapper) 역할을 하며, OptiPNG가 사전에 설치되어 있어야 합니다. Windows 사용자의 경우 실행 파일이 PATH 변수에 등록되어 있어야 합니다. 일반적인 방식대로 훅을 설정할 수 있습니다.

```r
knit_hooks$set(optipng = hook_optipng)
```

그런 다음 청크 옵션을 `optipng = TRUE`로 설정하여 청크에 대해 최적화를 활성화하거나, 이 옵션에 문자열을 전달하여 OptiPNG에서 추가 명령줄 인자로 사용하도록 할 수 있습니다. 예를 들어, 높은 수준의 최적화를 지정하려면 `optipng = '-o7'`을 사용할 수 있습니다. 가능한 모든 인자는 OptiPNG의 문서를 참조하시기 바랍니다.

![image 15](Dynamic Documents with R and knitr 2nd_images/imageFile15.png)

- FIGURE 10.7: Adding elements to an existing rgl plot: if we do not open a new device, latter elements will be added to the existing device.

###### 10.2.5 Close an rgl Device (rgl 장치 닫기)

기본 rgl 훅인 `hook_rgl()`은 새 플롯을 그리기 전에 rgl 장치를 닫지 않습니다. 이는 나중에 그려지는 플롯이 이전 장면에 겹쳐서 그려지기 때문에 문제가 될 수 있습니다. 예를 들어, 아래의 두 줄을 함께 실행하면 두 개의 구가 있는 하나의 플롯(그림 10.7)이 그려지지만, 첫 번째 플롯을 닫고 두 번째 줄을 실행하면 각각 하나의 구가 있는 두 개의 플롯이 생성됩니다.

```r
rgl.spheres(0, 0, 0)
rgl.spheres(0, 2, 0)
```

일반적으로 서로 다른 코드 청크는 다른 그래픽 장치를 사용하므로 나중 청크의 그래픽 요소가 이전 청크에 추가되지 않지만, rgl 플롯의 경우에는 예외입니다. 플롯을 그리기 전에 장치를 닫으려면 훅을 약간 수정해야 합니다. 예를 들어,

```r
knit_hooks$set(rgl = function(before, options, envir) {
  # if a device was opened before this chunk, close it
  if (before && rgl.cur() > 0)
    rgl.close()

  hook_rgl(before, options, envir)
})
```

`rgl.cur()` 함수는 현재 장치의 ID를 반환합니다. 이 값이 0보다 크다면 기존 장치가 있다는 의미이며, `rgl.close()`를 사용하여 이를 닫을 수 있습니다.

###### 10.2.6 WebGL

10.2.2절에서 정적인 rgl 플롯을 저장하는 방법에 대해 설명해 드렸습니다. 사실, `writeWebGL()` 함수를 사용하여 rgl 3D 플롯을 WebGL(http://en.wikipedia.org/wiki/WebGL) 형식으로 내보낼 수도 있습니다. 이를 통해 WebGL을 지원하는 웹 브라우저에서 플롯을 재현할 수 있습니다. 예를 들어, 플롯을 회전시키거나 확대/축소할 수 있습니다.

knitr에 있는 `hook_webgl()` 훅 함수는 rgl의 WebGL 함수를 위한 래퍼(wrapper)입니다. 이 훅을 사용하면 3D 장면을 HTML 출력으로 캡처할 수 있습니다.
