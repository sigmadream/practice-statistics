### 13

###### 보고서 출판

knitr를 통해 보고서를 컴파일한 후, 출력 문서가 바로 최종 결과물이 아닐 수 있습니다. 특히, Rnw 문서와 Rmd 문서의 출력 결과는 종종 추가적인 컴파일 과정이 필요합니다. Rnw의 직접적인 출력은 PDF로 컴파일할 수 있는 LATEX입니다. Rmd의 출력은 마크다운(Markdown)이며, 우리가 실제로 읽는 것은 마크다운에서 변환된 웹 페이지입니다.

LATEX에서 추가로 수행할 작업은 많지 않습니다. 도구 체인이 꽤 표준적이고 완성도가 높기 때문입니다(LATEX, PDFTEX, XeTEX, LuaTEX 등). Rnw 원본 문서를 바탕으로 보고서를 출판할 때, 우리는 단일 PDF 파일 하나만 출판하면 됩니다. 독자가 소스 코드를 읽는 데 관심이 없을 수 있으므로, 소스 코드를 숨기는 작업이 필요할 수 있습니다. 이러한 경우 청크(chunk) 옵션 echo를 전역적으로 FALSE로 설정할 수 있으며, 때로는 R에서 발생하는 메시지와 경고도 숨기고자 할 수 있습니다:

<<setup, include=FALSE>>= knitr::opts_chunk$set(

echo = FALSE, message = FALSE, warning = FALSE

) @

그러면 최종 보고서에는 결과만 표시됩니다. 본 장에서는 knitr의 결과를 최종 결과물로 변환하는 데 도움을 주는 몇 가지 도구와 프레젠테이션 도구를 소개합니다.

###### 13.1 RStudio

4.1절에서 소개한 바와 같이, RStudio는 knitr를 포괄적으로 지원합니다. RStudio가 획기적으로 쉽게 만든 것 중 하나는 R 마크다운에서 생성된 HTML 보고서의 출판입니다. Knit HTML 버튼을 클릭한 후, 미리보기 페이지의 도구 모음에서 Publish라는 이름의 버튼을 볼 수 있습니다. 이 버튼을 통해 클릭 한 번으로 웹사이트 http://rpubs.com 에 보고서를 출판할 수 있습니다. 보고서를 귀하의 계정으로 출판하려면 웹사이트에 미리 등록해야 합니다.

Knit HTML 버튼을 클릭할 때 내부적으로 일어나는 과정은 다음과 같습니다. RStudio는 knitr를 호출하여 Rmd를 마크다운으로 컴파일한 다음, Pandoc을 호출하여 마크다운을 HTML로 변환합니다. 두 번째 단계에서 Pandoc은 문서 내의 가능한 모든 이미지를 찾아 base64 문자열(12.4.2절)로 인코딩하여 HTML 파일이 자체 포함되도록 만듭니다. 웹사이트에 출판할 때 이미지 파일을 별도로 업로드할 필요가 없습니다. 대안으로 12.4.3절에서 소개한 imgur_upload()를 사용하여 이미지를 Imgur에 업로드할 수 있습니다.

이미지 인코딩 외에도 Pandoc은 문서 내의 LATEX 수식을 감지합니다. 수식이 존재하는 경우, 웹 페이지에서 수식이 올바르게 렌더링되도록 HTML 헤더에 JavaScript 라이브러리인 MathJax가 사용됩니다.

###### 13.2 Pandoc

Pandoc (http://johnmacfarlane.net/pandoc)은 범용 문서 변환기입니다. 특히 Pandoc은 마크다운을 LATEX, HTML, Rich Text Format (*.rtf), EBook (*.epub), Microsoft Word (*.docx), OpenDocument Text (*.odt) 등 다양한 문서 형식으로 변환할 수 있습니다. 이 절에서는 Pandoc의 내부 작동 방식을 설명하며, 본 절에서 소개하는 내용보다 훨씬 더 편리하게 사용할 수 있는 R 마크다운 v2에 대해서는 14장을 참고하시기 바랍니다.

Pandoc은 명령줄 도구입니다. Linux 및 Mac 사용자는 무리 없이 사용할 수 있으며, Windows 사용자의 경우 시작 메뉴의 실행에서 cmd를 입력하여 명령 창에 접근할 수 있습니다. 명령 창(또는 터미널)을 연 후에는 다음과 같은 명령을 입력하여 마크다운 파일(예: test.md)을 다른 형식으로 변환할 수 있습니다:

pandoc test.md -o test.html pandoc test.md -s --mathjax -o test.html pandoc test.md -o test.odt pandoc test.md -o test.rtf pandoc test.md -o test.docx pandoc test.md -o test.pdf pandoc test.md --latex-engine=xelatex -o test.html pandoc test.md -o test.epub

옵션 -o는 출력 파일명을 지정합니다. 그림 13.1은 OpenDocument Text 문서의 스크린샷을 보여주며, 외형 면에서 Microsoft Word와 유사해 보입니다.

knitr에는 R에서 Pandoc을 호출하는 pandoc() 함수가 있습니다. 이 함수는 Rmd 문서에 Pandoc 인수를 포함할 수 있게 해줍니다. 자세한 내용은 해당 문서를 참조하시기 바랍니다.

보편적으로 작동하는 문서 형식을 찾는 것은 항상 큰 과제입니다. 일부 사용자는 Word에 만족하지 않으며, 다른 사용자는 LATEX를 배우기 어려워합니다. 마크다운은 Pandoc이 다양한 문서 형식을 지원하므로 가능한 해결책 중 하나가 될 수 있습니다. 하지만 모든 문서 형식에서 조판의 세부 사항이 만족스럽지 않을 수 있으며, 나중에 변환된 문서를 수동으로 수정해야 할 가능성이 높습니다.

###### 13.3 HTML5 슬라이드

프레젠테이션을 제작하기 위해 12.3.4절에서 언급한 Beamer 클래스를 사용할 수 있습니다. 웹 기술의 발전에 따라, 평소처럼 슬라이드를 (PDF나 PPT) 파일로 다운로드할 필요 없이 웹 브라우저에서 볼 수 있는 HTML5 슬라이드를 웹 상에서 제작할 수도 있습니다. 또한 HTML5 슬라이드를 사용하면 비디오 클립이나 대화형 콘텐츠(예: JavaScript 애플리케이션)와 같은 리치 미디어를 슬라이드에 포함할 수 있습니다.

HTML5 슬라이드를 제작하는 방법은 여러 가지가 있습니다. 한 가지 방법은 Pandoc을 사용하여 마크다운에서 변환하는 것입니다. 그림 13.2는 knitr를 통해 마크다운으로 컴파일할 수 있는 Rmd 문서를 보여줍니다. 그런 다음 명령줄에서 Pandoc을 호출하여 이를 HTML5 슬라이드로 변환할 수 있습니다(파일명이 test.md라고 가정합니다):

pandoc -s -t dzslides test.md -o test.html

옵션 -s는 Pandoc이 (모든 CSS 정의가 문서에 작성된) 독립 실행형 문서를 생성하도록 지시합니다. 옵션 -t는 생성할 형식을 의미합니다. dzslides는 HTML5 슬라이드를 위한 가능한 값 중 하나일 뿐이며, 다른 형식에 대해서는 Pandoc의 온라인 문서를 참고하시기 바랍니다.

이제 웹 브라우저에서 HTML 파일을 열고 좌/우 화살표를 사용하여 슬라이드를 탐색할 수 있습니다.

명령줄 도구가 불편한 경우, slidify (Vaidyanathan, 2012) 및 rmarkdown (Allaire et al., 2015a)과 같은 일부 R 패키지를 사용하면 더 편리하게 작업할 수 있습니다. Rmd 파일에서 직접 HTML 슬라이드를 생성할 수 있으며, 이 패키지들과 함께 제공되는 유용한 템플릿과 테마도 있습니다.


그림 13.1: 마크다운에서 변환된 OpenDocument Text: 3.2.2절과 동일한 마크다운 문서를 사용했지만 청크 옵션을 제거했습니다.

|![image 19](Dynamic Documents with R and knitr 2nd_images/imageFile19.png)|
|---|


% Writing beautiful and reproducible slides quickly % Yihui Xie % 2012/12/05

# Introduction

- - knitr
- - pandoc # A code chunk


   {r computing} head(cars) cor(cars)

그림 13.2: HTML5 슬라이드 예제의 소스: knitr를 통해 이 문서를 컴파일한 다음, Pandoc을 통해 마크다운 출력을 DZSlides로 변환할 수 있습니다.

###### 13.4 Jekyll

Jekyll (http://jekyllrb.com)은 일반 텍스트 파일을 기반으로 하는 블로그 엔진입니다. 블로그 게시물은 마크다운으로 작성할 수 있으므로, knitr의 결과를 웹사이트에 출판하는 것이 가능합니다. 주의해야 할 점 중 하나는 코드 블록의 구문이 기존 마크다운(세 개의 백틱)과 다르다는 것입니다. Jekyll의 경우, 코드 블록을 Liquid 태그 안에 넣어야 합니다:

{% highlight lang %} # code here {% endhighlight %}

knitr에는 Jekyll용 렌더러인 render_jekyll()이 있으므로 이러한 기술적 세부 사항에 대해 걱정할 필요가 없습니다. 이 함수를 호출한 후에는 R 코드와 그 출력이 올바른 태그 안에 작성됩니다. 실제로 코드 블록의 구문은 Jekyll에 어떤 마크다운 렌더러를 사용하느냐에 따라 달라지기도 합니다. 기본 렌더러는 kramdown (http://kramdown.gettalong.org)이며 세 개의 백틱을 지원하지 않지만, redcarpet (https://github.com/vmg/redcarpet)과 같은 다른 렌더러는 이 구문을 지원할 수 있습니다. 5.2.1절에서 언급했듯이, 마크다운의 큰 문제는 렌더러마다 구문이 다르다는 것입니다.

참고로 knitr 웹사이트 (http://yihui.name/knitr)는 Jekyll로 구축되었으며 Github에서 호스팅됩니다.

###### 13.5 WordPress

WordPress는 PHP와 MySQL을 기반으로 하는 무료 오픈 소스의 인기 있는 블로깅 시스템입니다. 이 시스템은 타사 클라이언트에서 블로그 게시물을 출판할 수 있도록 허용하는 API를 가지고 있습니다. RWordPress 패키지는 WordPress 사이트와 통신하는 R 함수를 제공합니다. knitr의 knit2wp() 래퍼 함수를 사용하면 Rmd 문서를 컴파일하여 WordPress로 직접 전송할 수 있습니다. 로그인 이름 및 비밀번호와 같은 구성의 세부 사항은 http://yihui.name/knitr/demo/wordpress/ 를 참고하시기 바랍니다.

### 14

###### R 마크다운

본 책의 초판 이후 R 마크다운 개발에 많은 진전이 있었습니다. 명확히 하자면, R 마크다운에는 두 가지 버전이 있습니다. markdown 패키지(Allaire et al., 2015b)에 구현된 버전을 "R 마크다운 v1"(https://github.com/rstudio/markdown)이라 부르고, rmarkdown 패키지(Allaire et al., 2015a)에 구현된 버전을 "R 마크다운 v2"(http://rmarkdown.rstudio.com)라 부릅니다. 별도로 명시하지 않는 한, 본 장에서 사용하는 "R 마크다운"이라는 용어는 R 마크다운 v2를 지칭합니다.


R 마크다운 v1은 C 라이브러리인 sundown을 기반으로 하며, 주요 초점은 HTML 출력입니다. 인용이나 각주를 지원하지 않는 등 그 기능이 제한적입니다. R 마크다운 v2는 Pandoc을 기반으로 하여 마크다운을 완전히 새로운 수준으로 끌어올렸습니다. 개선된 부분은 두 가지 측면이 있습니다. Pandoc 마크다운 구문이 더 풍부해져서 더 많은 유형의 요소를 작성할 수 있게 되었으며, 출력 형식이 더 이상 HTML에 국한되지 않고 마크다운을 LATEX/PDF, Word, HTML5 슬라이드 등으로 내보낼 수 있게 되었습니다. 본 장에서는 rmarkdown의 설계 철학, 기능, 그리고 이를 사용자 정의하거나 확장하는 방법을 소개합니다.

###### 14.1 개요

knitr가 다양한 문서 형식을 지원하지만(5장), R 마크다운은 아마도 가장 인기 있는 형식일 것입니다. 마크다운은 기능 면에서는 제한적이지만, 초보자에게는 훌륭한 문서 언어입니다. 한편으로 저자는 기능 자체를 원하지 않을 수도 있습니다. 마크다운이 LATEX 사용자에게는 제한적으로 보일 수 있지만, 모든 사람이 조판 세부 사항에 그렇게 신경 쓸 필요는 없습니다.


마크다운의 한계는 Pandoc을 통해 크게 해소될 수 있지만, 문제는 Pandoc이 명령줄 도구라는 점입니다. 고급 사용자는 이를 실제 문제로 여기지 않을 수 있지만, 수많은 명령줄 인수는 초보자를 압도할 수 있습니다.


rmarkdown과 R 마크다운 v2의 목표는 R 마크다운 파일을 합리적으로 아름다운 템플릿을 사용하여 다른 문서 형식으로 빠르게 변환하는 기능을 제공하는 것입니다. 이 목표를 달성하는 방법은 자주 사용되는 명령줄 인수를 rmarkdown의 R 함수로 래핑하는 것입니다. R 마크다운 문서를 다른 문서 형식으로 렌더링하기 위한 rmarkdown의 주요 함수는 render()입니다. 첫 번째 인수는 Rmd 파일명이고, 두 번째 인수는 출력 형식이며 이에 대해서는 본 장 후반부에서 자세히 소개합니다. 예를 들어 R 마크다운 문서 foo.Rmd를 Word로 변환하려면 단 한 줄의 코드만 실행하면 됩니다:

rmarkdown::render("foo.Rmd", "word_document")

물론 더 어려운 방식을 택할 수도 있습니다. 먼저 knitr에서 knit()를 호출하여 foo.Rmd를 foo.md로 컴파일한 다음, 13.2절에서 소개한 바와 같이 터미널을 열거나 R 함수 system()을 사용하여 다음과 같은 명령을 실행할 수 있습니다:


pandoc foo.md --output foo.docx \

--from markdown+tex_math_single_backslash \

--highlight-style tango

현재 rmarkdown에는 PDF, HTML, Word, 마크다운, ioslides, Slidy, Beamer 등 일곱 가지 출력 형식 함수가 있습니다. 처음 네 가지는 문서 형식이고, 나머지 세 가지는 프레젠테이션 형식입니다. 이들은 knitr와 Pandoc 모두를 위한 래퍼 함수이므로 수많은 knitr 옵션과 Pandoc 인수를 기억할 필요가 없습니다. knitr 청크 옵션과 Pandoc 명령줄 인수는 rmarkdown 함수 인수로 변환됩니다. 예를 들어, Pandoc 인수 --toc 또는 --table-of-contents는 rmarkdown의 함수 인수 toc = TRUE에 대응됩니다.

또한 rmarkdown은 기본적으로 시각적으로 돋보이도록 고안된 자체 템플릿을 제공합니다. 예를 들어 HTML 출력의 경우 Twitter Bootstrap 스타일과 테마를 사용합니다. 프로그램 코드에 대한 구문 강조 표시도 기본적으로 활성화되어 있습니다.

rmarkdown 패키지는 RStudio IDE에서 잘 지원됩니다. render() 함수를 수동으로 호출할 필요가 없으며, 도구 모음의 Knit 버튼만 클릭하면 됩니다. 도구 모음의 톱니바퀴 버튼을 통해 나타나는 작은 GUI 창에서 출력 형식과 해당 옵션을 설정할 수도 있습니다. RStudio 외부에서 rmarkdown을 실행하려면 후반부에 rmarkdown의 작동 방식에 대한 자세한 내용을 학습하셔야 합니다.

참고로 RStudio에는 Pandoc이 내장되어 있으므로 RStudio를 사용하는 경우 Pandoc을 별도로 설치할 필요가 없습니다. 그렇지 않은 경우에는 Pandoc을 직접 설치해야 합니다. Pandoc이 별도로 설치되어 있는 경우, 해당 버전이 RStudio의 Pandoc 버전보다 높을 때만 RStudio에서 이를 사용합니다.

###### 14.2 Pandoc의 마크다운 확장

먼저 Pandoc 마크다운의 구문을 소개합니다. R 마크다운 v1에 익숙하다면 Pandoc에서도 해당 구문을 계속 사용할 수 있으며, 유일한 주요 변경 사항은 수식 요소가 아닌 위첨자를 작성하는 방법입니다. v1에서는 x^2와 같이 단일 캐럿(caret)을 사용합니다. Pandoc 마크다운에서는 위첨자를 ^로 둘러싸야 합니다(예: x^2^). 수식 표현의 경우 여전히 $x^2$와 같이 하나의 캐럿을 사용합니다.

###### 14.2.1 기본 구문

Pandoc 마크다운에서 다른 요소의 구문은 거의 동일하게 유지됩니다. 예를 들어, 첫 번째 수준 섹션 헤더를 작성할 때 하나의 # 기호를 사용하고, 두 번째 수준 헤더에는 두 개의 # 기호를 사용합니다. 마크다운의 기본 요소 구문에 대해서는 5.2.1절을 검토하시기 바랍니다. 아래는 유용할 수 있는 몇 가지 새로운 요소이며(전체 문서는 http://johnmacfarlane.net/pandoc/ 참조), 글머리 기호 아래에 이러한 요소의 간단한 예제를 보여줍니다:

- • 정의 목록 및 예제 목록
A Special Term : Describe/explain the term here.

(@) This is a numbered example. (@) Another numbered example.

(@cool-example) This example is labeled. This is a normal paragraph, and we can reference the example (@cool-example) here.

- • ^[...]를 사용한 각주 및 [@id]를 사용한 인용


We write a nice description of X here^[Not to be confused with Y], and X is useful.

Actually you should read the reference [@joe2014] to know more about X. Here  joe2014  is a key in the bibliography database.

- • 그림/표 캡션

Pandoc에는 기본적으로 활성화되어 있는 implicit_figures라는 마크다운 확장이 있습니다. 다음과 같은 이미지는
![A figure caption.](path/to/image.png)
LaTeX에서 다음과 유사하게 렌더링됩니다:
\begin{figure}

\includegraphics{path/to/image.png} \caption{A figure caption.}

\end{figure}
마찬가지로 표 캡션도 추가할 수 있습니다. 예: Table: This is a table caption.

--- ---- ---A B C

--- ---- ---a 10 bc d 25 ef --- ---- ----

- • 원시 TEX/HTML 내용


때로는 마크다운이 제한적이라고 느껴져서 LaTeX를 사용하고 싶은 유혹을 받을 수 있습니다. 그것은 괜찮습니다. 마크다운 안에 원시 LaTeX 코드를 작성할 수 있습니다.

Markdown version: ![A long caption.](foo.png)

LaTeX version: \begin{figure}

\includegraphics[width=.8\textwidth]{foo.png} \caption[A short caption]{A long caption.}

\end{figure} 
Pandoc은 이 문서를 LaTeX/PDF로 변환할 때 원시 TeX 내용을 보존할 수 있습니다.

인용을 사용할 때는 참고문헌 데이터베이스를 지정해야 합니다. LATEX에 익숙하다면 BibTEX도 알 가능성이 높습니다. 참고문헌 데이터베이스는 YAML 메타데이터의 bibliography 필드에 지정된 .bib 파일일 수 있습니다(다음 절 참조). BibTEX를 모르는 경우, (bibliography 대신) references 필드를 사용하여 YAML 메타데이터에 참고문헌 항목을 포함할 수 있습니다. 예를 들면 다음과 같습니다.

--references:

- - id: joe2014 title: A Nice Paper author:
그림 16.1: IPython의 스크린샷: 입력은 In[n ]으로 표시되고, 출력은 Out[n ]으로 표시됩니다.

###### 16.4 더 많은 도구

R 및 Python 패키지 외에도 다른 프로그램에 동적 문서를 위한 도구들이 있습니다. 이 장에서 동적 문서용 도구를 모두 나열하는 것은 불가능합니다. Schulte et al. (2012)는 Javadoc, cweb, noweb, Sweave, SASweave 등 리터럴 프로그래밍과 재현 가능한 연구를 위한 기존 도구들의 목록을 제공한 바 있습니다.

###### 16.4.1 Org-mode

Org-mode는 Emacs 텍스트 편집기(Schulte et al., 2012)에서 구현된 일반 텍스트 마크업 언어입니다. 리터럴 프로그래밍과 재현 가능한 연구(동적 문서의 의미에서)를 모두 지원합니다. 이는 다소 WEB 및 noweb과 같은 초기 리터럴 프로그래밍 구현의 구문을 따르고 있습니다. 즉, 코드 청크와 텍스트 청크("산문(prose)"이라고도 함)의 개념을 가지고 있습니다. Org-mode의 코드 청크는 다음과 같이 보입니다:

#+name: c-chunk
#+begin_src C
int main(){
return 0; }
#+end_src

이에 비해 knitr에서는 동일한 청크를 다음과 같이 작성합니다:

<<c-chunk, engine='c'>>=
int main(){
return 0;
}
@

메타데이터는 청크 헤더에 저장됩니다. Org-mode는 임의의 입력 언어를 지원하며, LATEX 또는 HTML을 출력 형식으로 사용할 수 있습니다.

Schulte et al. (2012)는 기존 도구들의 리터럴 프로그래밍 기능(예: Sweave는 이를 지원하지 않음)을 언급했지만, 보고서 작성자에게 크게 흥미롭게 들리지 않을 수 있어 본 서적에서는 강조하지 않았습니다. 사실 knitr에도 코드 청크를 재구성하는 기능이 있습니다 (9장을 참조하시기 바랍니다). 아래는 나중에 정의된 청크 B를 앞선 청크 A에 포함하는 간단한 예시입니다:

- <<A>>=
df <- data.frame(x = 1:10, y = rnorm(10))
- <<B>>
coef(fit)
@


<<B>>=
fit <- lm(y ~ x, data = df)
@

강력한 기능을 제공하지만, Org-mode의 Emacs 기반 특성은 초보자에게 진입 장벽이 될 수 있습니다.

###### 16.4.2 SASweave

SASweave(http://homepage.cs.uiowa.edu/~rlenth/SASweave)는 SAS와 R을 사용한 리터럴 프로그래밍 구현체로, gawk로 작성되었습니다. 기본적인 아이디어는 Sweave 및 knitr와 동일합니다. 자세한 내용은 Lenth와 Højsgaard(2007)를 참조하시기 바랍니다. SASweave와 비교했을 때 knitr 패키지는 R에 대해 더 포괄적인 지원을 제공하지만, SAS에 대한 지원은 상대적으로 부족합니다.

###### 16.4.3 Office 도구

우리가 동적 문서를 위해 반드시 일반 텍스트 형식을 선택해야 하는 것은 아닙니다. 본 서적에서 소개한 거의 모든 것이 일반 텍스트를 기반으로 하지만, OpenOffice(또는 OpenDocument Text)나 Microsoft Office 제품(간단히 Office 문서라고 부름) 기반의 도구들도 존재하며, 첫눈에 매력적으로 보일 수 있습니다. 핵심적으로 Office 문서는 대개 압축되어 있을 수 있는 XML 파일이므로 코드 청크를 그 안에 포함시키는 것이 가능합니다. 우리는 코드 청크를 구문 분석하고, 실행한 후 그 결과를 다시 삽입할 수 있습니다.

우리가 파악한 주요 문제점은 XML 형식이 너무 복잡하고 표준이 과도하게 많아, 수정된 문서가 여전히 유효한 Office 문서인지 확인하는 것이 결코 쉬운 일이 아니라는 것입니다. 한 가지 예로, StatWeave 패키지(http://homepage.stat.uiowa.edu/~rlenth/StatWeave/)는 "OpenOffice가 수정된 문서를 손상된 문서로 표시"하기 때문에 더 이상 OpenOffice(3.2 이상 버전)에서 작동하지 않습니다.

이에 비해 일반 텍스트 파일은 다루기가 훨씬 수월하며, ECMA-376과 같이 고려해야 할 복잡한 표준이 존재하지 않습니다. 만약 Office 문서를 굳이 원한다면, 최소한 Markdown에서 변환하는 방법들이 있습니다. 1장에서 인용했던 문구를 다시 한번 떠올려 보시기 바랍니다:

소스 코드가 진짜입니다.

### A

###### 내부 구조

본 부록에서는 knitr 패키지의 일부 내부 구조를 설명합니다. 이는 다른 개발자들이 이 패키지를 더 잘 이해하고 필요시 코드를 기여하는 데 도움이 될 수 있습니다. 일반 사용자는 이 부록을 읽으실 필요가 없습니다. 문서화(documentation), 클로저(closure)의 적용, 그리고 일부 기능의 구현이라는 세 가지 측면에서 내부 구조를 보여드리겠습니다.

###### A.1 문서화

knitr의 문서화에는 R 문서(Rd), PDF 매뉴얼, 그리고 웹사이트라는 세 가지 유형이 있습니다.

R 문서는 roxygen2(Wickham et al., 2015)를 기반으로 합니다. 이를 사용하면 태그와 함께 roxygen 주석(#')에 Rd를 작성할 수 있으며, 이 주석들은 실제 Rd로 변환됩니다. 아래는 roxygen 주석의 예시입니다:

#' @author Yihui Xie

이 주석은 다음과 같이 Rd로 변환됩니다:
\author{Yihui Xie}

roxygen에는 @usage, @param, @return, @examples와 같은 일련의 태그들이 있으며, 이들은 각각 Rd의 \usage{}, \arguments{\item{}}, \value{}, \examples{}에 대응합니다. 공식 Rd 대신 roxygen 주석을 작성하는 것의 이점은 문서와 소스 코드를 동일한 파일에 보관할 수 있다는 점입니다. 이와 대조적으로, R 패키지를 작성하는 공식적인 접근 방식은 R/ 디렉토리 하위에 R 소스를 작성하고, man/ 하위에 *.Rd 파일로 매뉴얼 페이지를 작성하는 것입니다. 이 방식은 두 파일 사이를 오가야 하고, R 소스를 업데이트하면서 문서 업데이트를 잊어버리기 쉽기 때문에 불편합니다. Roxygen 주석은 소스의 R 함수 바로 위에 위치하므로, 소스와 문서 모두를 유지 관리하기가 훨씬 쉽습니다.

247

아래는 roxygen 주석으로 문서화된 함수의 완전한 예시입니다:

#' Repeat a character string
#'
#' Repeat a string n times and make one string.
#' @param x a character string
#' @param n an integer
#' @return A character string.
#' @examples
#' f("hi", n = 5)
f <- function(x, n = 10) {
  paste(rep(x, n), collapse = "")
}

roxygen2의 roxygenize() 함수를 사용하여 roxygen 주석을 공식 Rd 파일로 변환할 수 있습니다. knitr의 모든 객체는 이 방식으로 문서화되어 있습니다. 게다가 roxygen2는 NAMESPACE와 DESCRIPTION의 Collate 필드도 자동으로 처리해 주므로, 개발자는 R 소스 파일 작업에 집중할 수 있습니다.

PDF 매뉴얼의 소스 문서는 examples 디렉토리(소스 패키지의 inst/examples/ 참조)에 위치하며, 주요 매뉴얼은 knitr-manual.Rnw입니다. Rnw 파일은 LYX 파일에서 내보낸 것이므로(4.2절 참조), PDF 매뉴얼을 편집하거나 컴파일하려면 LYX 파일을 열어 작업하는 것을 권장합니다. PDF 매뉴얼이 소스 패키지에 포함되지 않는 이유는 (1) 바이너리 파일을 버전 관리 시스템에 넣고 싶지 않으며(특히 소스 파일의 부산물인 경우), (2) 패키지 웹사이트에 호스팅되어 있기 때문입니다.

패키지 웹사이트는 13.4절에서 소개된 바와 같이 Jekyll을 기반으로 구축되었습니다. 구체적으로 모든 페이지는 Markdown으로 작성되며, Git 저장소의 gh-pages 브랜치에 저장됩니다(패키지 자체는 master 브랜치에 있습니다). Git을 통해 gh-pages 브랜치에 변경 사항을 푸시하면 Github이 자동으로 웹사이트를 다시 빌드합니다. 웹사이트에 기여하고 싶다면 gh-pages 브랜치로 전환하여 Markdown 파일을 업데이트하시면 됩니다.

###### A.2 클로저 (Closures)

클로저는 knitr에서 핵심적인 역할을 합니다. opts_chunk(5.1.1절) 및 knit_engines(11장)과 같은 일부 일반적인 객체들은 클로저를 기반으로 구축되었습니다.

클로저는 본질적으로 함수이며, 비지역(non-local) 변수에 접근할 수 있습니다. 아래는 간단한 예시입니다:

f <- function() {
  x <- 1
  function(y) x + y
}
g <- f()

- g(5) # x에 5를 더합니다
## [1] 6
ls(environment(g)) # g는 x를 볼 수 있습니다
## [1] "x"


함수 g()는 f()에서 생성되었으며(f()가 함수를 반환함을 유의하세요), g()는 f() 내부에 생성된 x 객체를 사용하고 이 x는 f() 내에만 존재합니다. g()가 어디서 호출되든 상관없이 항상 이 x에 접근할 수 있습니다.

사실 클로저를 통해 비지역 변수를 수정하는 것도 가능합니다. 아래는 청크 옵션 관리자인 opts_chunk가 어떻게 작동하는지 보여주는 최소한의 예시입니다:

new_list <- function(default = list()) {
  list(get = function() default,
       set = function(...) {
         x <- list(...)
         if (length(x)) default[names(x)] <<- x
       })
}

new_list() 함수는 함수의 목록(setter와 getter)을 반환합니다. default 객체는 이 두 함수에 바인딩됩니다. 이를 청크 옵션의 기본 목록이라고 생각할 수 있습니다. 다음은 청크 옵션을 가져오고 설정하는 방법을 보여줍니다.

opts <- new_list(list(eval = TRUE))
str(opts$get())

- ## List of 1
## $ eval: logi TRUE

opts$set(eval = FALSE) # eval을 FALSE로 변경합니다
opts$set(results = "markup") # 청크 옵션을 추가합니다
str(opts$get())

- ## List of 2
## $ eval : logi FALSE
## $ results: chr "markup"

opts$set(results = "hide") # results 옵션을 변경합니다

$set() 함수 내에서 <<-를 사용하여 인수를 default 객체에 할당했으며, 이것이 부모 환경에서 해당 객체를 수정할 수 있는 이유입니다 (만약 일반적인 <-를 사용했다면 부모 환경의 default는 수정되지 않고 지역 복사본이 생성되었을 것입니다).

클로저를 사용함으로써 knitr는 동일한 구문을 사용하여 객체들을 자체 환경에서 관리할 수 있습니다. knitr의 내부 함수 new_defaults()는 이와 같은 클로저 목록을 생성하는 데 사용됩니다.

opts_chunk(청크 옵션 관리용)와 knit_engines(언어 엔진 관리용) 객체 외에도 이와 유사한 객체들이 몇 가지 더 존재합니다:

- opts_knit : 패키지 옵션 (12.2절)
- opts_current : 현재 청크의 청크 옵션
- opts_template : 청크 옵션 템플릿 (12.1.2절)
- knit_hooks : 후크 함수 (출력 후크 및 청크 후크 모두)
- knit_patterns : 파서를 위한 구문 패턴 (5.1절)

###### A.3 구현 (Implementation)

이 절에서는 이 패키지의 몇 가지 구현 세부 사항을 설명합니다. 먼저 한 가지 사소한 점을 말씀드리면, 필자는 할당 연산자로 <- 대신 =를 사용하므로 소스 코드 전반에서 =를 보시게 될 것입니다. 이는 개인적인 취향의 문제이며 별다른 단점이 없다고 판단하지만, 이 패키지에 코드를 기여하실 때는 =를 사용해주시기 바랍니다. 본 서적에서는 필자가 등호를 입력했지만 formatR에 의해 자동으로 <-로 대체되었기 때문에 <-를 보게 되실 것입니다.

###### A.3.1 파서 (Parser)

문서 파서(5.1절)는 다음과 같이 작동합니다: 구문 패턴 객체의 하위 요소인 chunk.begin과 chunk.end를 사용하여 문서를 분할하고(코드 청크와 텍스트 청크), 코드 청크의 경우 청크 옵션(즉, 첫 번째 줄에서 추출된 텍스트)이 R 코드로 구문 분석됩니다. 이것이 청크 옵션이 R 구문을 따라야 하는 이유입니다. 다음은 knitr가 텍스트 조각에서 청크 옵션을 가져오는 방법을 설명하는 예시입니다:

## 이것이 청크 옵션 텍스트라고 가정합니다
txt <- "label, eval=TRUE, echo=1:3, foo=if(TRUE) 2 else 5"
opc <- eval(parse(text = paste("alist(", txt, ")")))
names(opc) # 청크 레이블에는 이름이 지정되지 않습니다

## [1] "" "eval" "echo" "foo"
str(opc) # 일부는 평가되지 않은 표현식입니다

## List of 4
## $ : symbol label
## $ eval: logi TRUE
## $ echo: language 1:3
## $ foo : language if (TRUE) 2 else 5

먼저 텍스트 주변에 alist() 함수를 추가했습니다. 이 함수는 인수들을 마치 함수 인수를 설명하는 것처럼 취급하므로 이 시점에서는 어떤 "인수"도 평가되지 않습니다. 그러나 구문은 최소한 유효해야 합니다. 한 가지 예외는 청크 레이블입니다: 청크 레이블은 문자열이어야 하므로 필요한 경우 자동으로 따옴표로 묶입니다. 청크 옵션을 구문 분석하는 데에는 내부 함수 parse_params()가 사용됩니다:

p <- knitr:::parse_params
str(p("chunk-label, eval=TRUE, foo=5"))

- ## List of 3
## $ label: chr "chunk-label"
## $ eval : logi TRUE
## $ foo : num 5


# 2a는 R에서 유효한 기호가 아니지만, knitr는 구문 분석이 원활하도록
# 자동으로 이를 따옴표로 묶습니다
parse(text = "alist(2a)")
## Error: <text>:1:8: unexpected symbol
## 1: alist(2a
## ^
str(p("2a, eval=FALSE"))

## List of 2
## $ label: chr "2a"
## $ eval : logi FALSE

str(p("'2a', eval=FALSE")) # 또는 직접 따옴표로 묶을 수 있습니다

## List of 2
## $ label: chr "2a"
## $ eval : logi FALSE

청크 옵션은 청크가 실행되기 직전까지 평가되지 않으므로, 파싱 시점의 문서 내에서 알 수 없는 값을 가진 객체를 청크 옵션으로 사용할 수 있습니다. 예를 들어, 위의 echo와 foo 옵션은 아직 평가되지 않은 표현식이며, 이는 나중에 명시적으로 평가될 것입니다:

eval(opc$echo)
## [1] 1 2 3
eval(opc$foo)
## [1] 2

모든 코드 청크는 내부 객체 knit_code에 이름이 지정된 리스트 형태로 저장됩니다. 이름은 청크 레이블이며 내용은 코드입니다. 이 객체 역시 클로저 목록으로 생성되므로 get() 및 set() 메서드를 가지고 있지만, 예상치 못한 결과를 초래할 수 있으므로 이 객체를 직접 수정하는 것은 권장하지 않습니다. 필요한 경우 knitr:::knit_code$get('chunk-label')을 통해 코드 청크에 접근할 수 있습니다.

###### A.3.2 청크 후크 (Chunk Hooks)

knit_hooks에는 다음과 같이 출력 후크(5.3절)에 해당하는 다수의 기본 후크가 존재합니다:

names(knit_hooks$get(default = TRUE))

## [1] "source" "output" "warning" "message"
## [5] "error" "plot" "inline" "chunk"
## [9] "text" "document"

이 객체 내의 다른 모든 후크는 청크 후크(10장)로 취급됩니다. 코드 청크가 실행되기 전후에 모든 추가 후크가 호출됩니다. 다음은 의사 코드(pseudo code)입니다:

hook(before = TRUE, ...)
evaluate(code)
hook(before = FALSE, ...)

기억해야 할 한 가지 문제는 후크의 실행 순서입니다: 만약 knit_hooks에 정의된 A와 B라는 두 개의 후크가 있다면, 어떤 순서로 호출될까요? 이 순서는 청크 옵션에서 얻어집니다: 해당 두 후크에 대응하는 A와 B라는 두 개의 청크 옵션이 있어야 하며, 청크 옵션의 순서가 후크를 실행할 순서를 결정합니다. 예를 들어 A가 B보다 앞에 있다면 후크 A가 B보다 먼저 호출됩니다. 그러나 코드 청크가 평가된 후에는 그 순서가 역전됩니다. 이는 후크가 반환하는 결과가 짝을 이루도록 하기 위함입니다. 예를 들어, 후크 A가 청크 전에 \begin{Aenvir}를 반환하고 청크 후에 \end{Aenvir}를 반환한다고 가정해 봅시다. 마찬가지로 B는 Benvir를 반환합니다. 그렇다면 출력에서 우리가 원하는 것은 다음과 같습니다:

\begin{Aenvir}
\begin{Benvir}
% 청크의 결과
\end{Benvir}
\end{Aenvir}

\end{Benvir}가 \end{Aenvir}보다 앞에 온다는 점에 유의하시기 바랍니다. 이러한 이유로 후크 A와 B가 정의되었을 때 다음 두 청크는 서로 다른 결과를 반환합니다:

- <<A=TRUE, B=TRUE>>=
- <<B=TRUE, A=TRUE>>=

###### A.3.3 옵션 별칭 (Option Aliases)

리스트 내의 특정 요소를 치환하는 간단한 연산이기 때문에, 청크 옵션 별칭(12.1.1절)을 구현하는 데에는 단 몇 줄의 코드만 필요합니다. 아래는 이러한 아이디어를 보여주는 짧은 함수입니다:

apply_aliases <- function(x, list) {
  ## names는 x의 별칭입니다
  list[x] <- list[names(x)]
  list
}
al <- c(w = "fig.width", h = "fig.height", a = "fig.align")
op <- list(w = 7, h = 7, echo = TRUE, a = "center")
str(op) # 사용자의 옵션

- ## List of 4
## $ w : num 7
## $ h : num 7
## $ echo: logi TRUE
## $ a : chr "center"

str(apply_aliases(al, op)) # 수정된 옵션
