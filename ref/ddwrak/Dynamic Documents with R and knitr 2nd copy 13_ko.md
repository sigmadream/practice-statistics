### 13

###### 보고서 출판

knitr를 통해 보고서를 컴파일한 후 출력된 문서가 바로 최종 결과물이 아닐 수도 있습니다. 특히 Rnw 문서와 Rmd 문서의 출력은 종종 추가적인 컴파일을 필요로 합니다. Rnw의 직접적인 출력은 PDF로 컴파일할 수 있는 LATEX입니다. Rmd의 출력은 마크다운(Markdown)이며, 우리가 실제로 읽는 것은 마크다운에서 변환된 웹 페이지입니다.

LATEX의 경우 남은 작업이 많지 않습니다. 도구 체인이 꽤 표준적이고 성숙해 있기 때문입니다(LATEX, PDFTEX, XeTEX, LuaTEX 등). Rnw 원본 문서를 기반으로 보고서를 출판할 때 우리는 단지 하나의 PDF 파일만 출판하면 됩니다. 한 가지 필요할 수 있는 작업은 소스 코드를 숨기는 것인데, 독자가 이를 읽는 데 관심이 없을 수 있기 때문입니다. 이 경우 청크 옵션 echo를 전역적으로 FALSE로 설정할 수 있으며, 때로는 R에서 나오는 메시지와 경고도 숨기고 싶을 수 있습니다.

<<setup, include=FALSE>>= knitr::opts_chunk$set(

echo = FALSE, message = FALSE, warning = FALSE

) @

그러면 최종 보고서에는 결과만 표시됩니다. 이 장에서는 knitr의 결과를 최종 결과물로 변환하는 데 도움을 줄 수 있는 몇 가지 도구와 프레젠테이션 도구를 소개합니다.

###### 13.1 RStudio

4.1절에서 소개한 바와 같이, RStudio는 knitr를 포괄적으로 지원합니다. RStudio가 정말 쉽게 만든 것 중 하나는 R 마크다운에서 생성된 HTML 보고서의 출판입니다. Knit HTML 버튼을 클릭하면 미리보기 페이지의 툴바에서 Publish라는 버튼을 볼 수 있습니다. 이 버튼을 통해 단 한 번의 클릭으로 보고서를 http://rpubs.com 웹사이트에 출판할 수 있습니다. 보고서를 귀하의 계정에 출판하려면 사전에 웹사이트에 가입해야 합니다.

Knit HTML 버튼을 클릭할 때 이면에서 일어나는 과정은, RStudio가 knitr를 호출하여 Rmd를 마크다운으로 컴파일한 다음, Pandoc을 호출하여 마크다운을 HTML로 변환하는 것입니다. 두 번째 단계에서 Pandoc은 문서 내의 모든 가능한 이미지를 찾아 base64 문자열(12.4.2절)로 인코딩하여 HTML 파일이 자체 포함되도록 합니다. 이를 웹사이트에 출판할 때는 이미지 파일을 별도로 업로드할 필요가 없습니다. 대안으로 12.4.3절에서 소개한 imgur_upload()를 사용하여 이미지를 Imgur에 업로드할 수도 있습니다.

이미지 인코딩 외에도 Pandoc은 문서 내의 LATEX 수식 표현을 감지합니다. 수식이 있는 경우 HTML 헤더에 MathJax 자바스크립트 라이브러리가 사용되어 수식이 웹 페이지에서 올바르게 렌더링되도록 합니다.

###### 13.2 Pandoc

Pandoc(http://johnmacfarlane.net/pandoc)은 범용 문서 변환기입니다. 특히 Pandoc은 마크다운을 LATEX, HTML, 서식 있는 텍스트 형식(_.rtf), EBook(_.epub), 마이크로소프트 워드(_.docx), 개방형 문서 텍스트(_.odt) 등 다양한 문서 형식으로 변환할 수 있습니다. 이 절에서는 Pandoc이 이면에서 어떻게 작동하는지 설명하며, 이 절에서 소개하는 것보다 훨씬 편리하게 작업할 수 있는 R 마크다운 v2에 대해서는 14장을 참조하시기 바랍니다.

Pandoc은 명령줄 도구입니다. 리눅스 및 맥 사용자는 쉽게 사용할 수 있으며, 윈도우 사용자의 경우 시작 메뉴를 통해 명령 창에 접속한 다음 Run cmd를 실행하면 됩니다. 명령 창(또는 터미널)을 열고 나면 다음과 같은 명령을 입력하여 마크다운 파일(test.md)을 다른 형식으로 변환할 수 있습니다.

pandoc test.md -o test.html pandoc test.md -s --mathjax -o test.html pandoc test.md -o test.odt pandoc test.md -o test.rtf pandoc test.md -o test.docx pandoc test.md -o test.pdf pandoc test.md --latex-engine=xelatex -o test.html pandoc test.md -o test.epub

-o 옵션은 출력 파일 이름을 지정합니다. 그림 13.1은 개방형 문서 텍스트(OpenDocument Text) 문서의 스크린샷을 보여주는데, 외관상 마이크로소프트 워드와 매우 비슷해 보입니다.

knitr에는 R에서 Pandoc을 호출하는 pandoc() 함수가 있습니다. 이 함수는 Pandoc 인수를 Rmd 문서에 삽입할 수 있게 해주며, 자세한 내용은 해당 문서를 참조하십시오.

범용적으로 작동하는 문서 형식을 찾는 것은 항상 큰 과제입니다. 워드에 만족하지 못하는 사용자도 있고 LATEX를 배우기 어려워하는 사용자도 있습니다. 마크다운은 Pandoc이 매우 다양한 문서 형식을 지원하므로 잠재적인 해결책이 될 수 있습니다. 하지만 조판의 세부 사항은 모든 문서 형식에서 만족스럽지 않을 수 있으며, 나중에 변환된 문서를 수동으로 다듬어야 할 가능성이 매우 높습니다.

###### 13.3 HTML5 슬라이드

프레젠테이션을 만들기 위해 12.3.4절에서 언급한 Beamer 클래스를 사용할 수 있습니다. 웹 기술이 발전함에 따라 웹 브라우저에서 볼 수 있는 HTML 슬라이드를 웹상에 만들 수도 있는데, 이는 평소처럼 슬라이드를 (PDF 또는 PPT) 파일로 다운로드할 필요가 없습니다. HTML5 슬라이드는 비디오 클립 및 대화형 콘텐츠(자바스크립트 애플리케이션)와 같은 리치 미디어를 슬라이드에 삽입할 수 있게 해줍니다.

HTML5 슬라이드를 만드는 방법에는 여러 가지가 있습니다. 한 가지 방법은 Pandoc을 통해 마크다운에서 시작하는 것입니다. 그림 13.2는 knitr를 통해 마크다운으로 컴파일할 수 있는 Rmd 문서를 보여줍니다. 그런 다음 명령줄에서 Pandoc을 호출하여 이를 HTML5 슬라이드로 변환할 수 있습니다(파일 이름이 test.md라고 가정):

pandoc -s -t dzslides test.md -o test.html

-s 옵션은 Pandoc에 단독 문서(모든 CSS 정의가 이 문서 안에 작성됨)를 생성하도록 지시하며, -t 옵션은 생성할 형식을 의미합니다. dzslides는 HTML5 슬라이드를 위한 하나의 가능한 값일 뿐이며, 다른 형식에 대해서는 Pandoc 온라인 문서를 참조하십시오.

이제 웹 브라우저에서 HTML 파일을 열고 왼쪽/오른쪽 화살표를 사용하여 슬라이드를 탐색할 수 있습니다.

명령줄 도구가 불편하다면 slidify(Vaidyanathan, 2012) 및 rmarkdown(Allaire 등, 2015a)과 같이 작업을 더 쉽게 만들어 주는 몇 가지 R 패키지가 있습니다. Rmd 파일에서 직접 HTML 슬라이드를 만들 수 있으며 이 패키지들과 함께 제공되는 훌륭한 템플릿과 테마도 있습니다.

- 그림 13.1: 마크다운에서 변환된 개방형 문서 텍스트(OpenDocument Text): 3.2.2절에서와 동일한 마크다운 문서를 사용했지만 청크 옵션은 제거했습니다.

fig.align='center'

% 빠르고 재현 가능한 아름다운 슬라이드 작성하기 % Yihui Xie % 2012/12/05

# 서론

- knitr
- pandoc # 코드 청크

  {r computing} head(cars) cor(cars)

- 그림 13.2: HTML5 슬라이드 예제의 소스: 이 문서를 knitr를 통해 컴파일한 다음 Pandoc을 통해 마크다운 출력을 DZSlides로 변환할 수 있습니다.

###### 13.4 Jekyll

Jekyll(http://jekyllrb.com)은 일반 텍스트 파일을 기반으로 하는 블로그 엔진입니다. 블로그 글을 마크다운으로 작성할 수 있으므로 knitr의 결과를 웹사이트에 출판하는 것이 가능합니다. 한 가지 주의해야 할 점은 코드 블록의 구문이 기존 마크다운(백틱 3개)과 다르다는 것입니다. Jekyll의 경우 코드 블록을 Liquid 태그 안에 넣어야 합니다.

{% highlight lang %} # 코드 입력 {% endhighlight %}

knitr에 Jekyll용 렌더러인 render_jekyll()이 있으므로 이 기술적인 세부 사항에 대해 걱정할 필요는 없습니다. 이 함수를 호출하고 나면 R 코드와 그 출력이 올바른 태그 안에 기록됩니다. 사실 코드 블록의 구문은 Jekyll에 어떤 마크다운 렌더러를 사용하느냐에 따라 달라지기도 합니다. 기본 렌더러인 kramdown(http://kramdown.gettalong.org)은 백틱 3개를 지원하지 않지만, redcarpet(https://github.com/vmg/redcarpet)과 같은 다른 일부 렌더러는 이 구문을 지원할 수도 있습니다. 5.2.1절에서 언급했듯이 마크다운의 큰 문제점은 렌더러마다 구문이 다르다는 점입니다.

실제로 knitr 웹사이트(http://yihui.name/knitr)는 Jekyll로 구축되었으며 Github에서 호스팅됩니다.

###### 13.5 WordPress

워드프레스(WordPress)는 PHP와 MySQL을 기반으로 하는 무료 오픈 소스의 인기 있는 블로깅 시스템입니다. 제3자 클라이언트에서 블로그 게시물을 발행할 수 있는 API를 갖추고 있습니다. RWordPress 패키지는 워드프레스 사이트와 통신하기 위한 R 함수를 제공합니다. knitr에는 Rmd 문서를 컴파일하여 워드프레스에 직접 전송할 수 있게 해주는 래퍼 함수인 knit2wp()가 있습니다. 로그인 이름과 비밀번호와 같은 구성에 대한 자세한 내용은 http://yihui.name/knitr/demo/wordpress/ 를 참조하십시오.

### 14

###### R 마크다운

이 책의 초판 이후로 R 마크다운 개발에는 많은 진전이 있었습니다. 명확히 하자면, R 마크다운에는 두 가지 버전이 있습니다. markdown 패키지(Allaire 등, 2015b)의 구현체를 "R 마크다운 v1"(https://github.com/rstudio/markdown)이라 부르고, rmarkdown 패키지(Allaire 등, 2015a)의 구현체를 "R 마크다운 v2"(http://rmarkdown.rstudio.com)라고 부릅니다. 특별한 언급이 없는 한 이 장에서 "R 마크다운"이라는 용어의 사용은 R 마크다운 v2를 의미합니다.

R 마크다운 v1은 C 라이브러리인 sundown에 기반을 두고 있으며, 주요 초점은 HTML 출력입니다. 인용이나 각주를 지원하지 않는 등 그 기능이 매우 제한적입니다. R 마크다운 v2는 Pandoc을 기반으로 하여 마크다운을 완전히 새로운 수준으로 끌어올렸습니다. 개선 사항에는 두 가지 측면이 있습니다. Pandoc 마크다운 구문이 더 풍부해져서 더 많은 유형의 요소를 작성할 수 있으며, 출력 형식이 더 이상 HTML에 국한되지 않고 마크다운을 LATEX/PDF, 워드, HTML5 슬라이드 등으로도 내보낼 수 있습니다. 이 장에서는 rmarkdown의 설계 철학, 기능, 그리고 이를 사용자 정의하거나 확장하는 방법을 소개합니다.

###### 14.1 개요

knitr가 다양한 문서 형식(5장)을 지원하지만 R 마크다운이 아마도 가장 인기 있는 문서 형식일 것입니다. 마크다운은 기능 면에서 제한적이기는 하지만 초보자에게는 좋은 문서 언어입니다. 다른 한편으로 작성자는 애초에 많은 기능을 원하지 않을 수도 있습니다. LATEX 사용자의 눈에는 마크다운이 제한적일 수 있지만 모든 사람이 조판 세부 사항에 그렇게 신경 써야 하는 것은 아닙니다.

마크다운의 한계는 Pandoc에 의해 대부분 제거될 수 있지만 문제는 Pandoc이 명령줄 도구라는 것입니다. 파워 유저는 이것을 진짜 문제라고 생각하지 않을 수 있지만, 수많은 명령줄 인수는 초보자를 압도할 수 있습니다.

rmarkdown과 R 마크다운 v2의 목표는 합리적이고 아름다운 템플릿을 사용하여 R 마크다운 파일을 다른 문서 형식으로 빠르게 변환하는 것입니다. 이 목표를 달성하는 방법은 자주 사용되는 명령줄 인수를 rmarkdown의 R 함수로 래핑하는 것입니다. R 마크다운 문서를 다른 문서 형식으로 렌더링하는 rmarkdown의 주요 함수는 render()입니다. 첫 번째 인수는 Rmd 파일 이름이고, 두 번째 인수는 출력 형식이며 이에 대해서는 이 장의 뒷부분에서 자세히 소개할 것입니다. 예를 들어 R 마크다운 문서 foo.Rmd를 워드로 변환하려면 코드 한 줄만 실행하면 됩니다.

rmarkdown::render("foo.Rmd", "word_document")

물론 더 어려운 방법으로 할 수도 있습니다. 먼저 knitr의 knit()를 호출하여 foo.Rmd를 foo.md로 컴파일한 다음, 13.2절에서 소개한 것처럼 터미널을 열거나 R 함수 system()을 사용하여 다음과 같은 명령을 실행할 수 있습니다.

pandoc foo.md --output foo.docx \
--from markdown+tex_math_single_backslash \
--highlight-style tango

현재 rmarkdown에는 PDF, HTML, Word, Markdown, ioslides, Slidy, Beamer 등 7개의 출력 형식 함수가 있습니다. 처음 4개는 문서 형식이고 뒤의 3개는 프레젠테이션 형식입니다. 이들은 knitr와 Pandoc 모두를 위한 래퍼 함수이므로 수많은 knitr 옵션과 Pandoc 인수를 기억할 필요가 없습니다. knitr 청크 옵션과 Pandoc 명령줄 인수는 rmarkdown 함수 인수로 변환됩니다. 예를 들어 Pandoc 인수 --toc 또는 --table-of-contents는 rmarkdown의 함수 인수 toc = TRUE에 해당합니다.

또한 rmarkdown은 기본적으로 시각적으로 즐거움을 주는 것을 목표로 하는 자체 템플릿을 제공합니다. 예를 들어 HTML 출력의 경우 트위터 부트스트랩 스타일과 테마를 사용합니다. 프로그램 코드의 구문 강조도 기본적으로 활성화되어 있습니다.

rmarkdown 패키지는 RStudio IDE에서 잘 지원됩니다. render() 함수를 수동으로 호출할 필요가 없으며 툴바의 Knit 버튼을 클릭하기만 하면 됩니다. 또한 툴바의 톱니바퀴 버튼을 통해 팝업되는 작은 GUI에서 출력 형식과 그 옵션을 설정할 수 있습니다. RStudio 외부에서 rmarkdown을 실행하고 싶다면 나중에 rmarkdown이 작동하는 방식에 대한 더 자세한 내용을 알아야 할 것입니다.

참고로 RStudio에는 Pandoc이 내장되어 있으므로 RStudio를 사용하는 경우 Pandoc을 별도로 설치할 필요가 없으며, 그렇지 않은 경우에는 사용자가 직접 Pandoc을 설치해야 합니다. Pandoc을 별도로 설치한 경우, 해당 버전이 RStudio의 Pandoc 버전보다 높을 때만 RStudio에서 별도로 설치한 버전을 사용합니다.

###### 14.2 Pandoc의 마크다운 확장

먼저 Pandoc 마크다운의 구문을 소개합니다. R 마크다운 v1에 익숙하다면 Pandoc에서도 그 구문을 여전히 사용할 수 있으며, 유일한 중요한 변경 사항은 수식 요소가 아닌 윗첨자를 작성하는 방법입니다. v1에서는 단일 캐럿을 사용합니다(x^2). Pandoc의 마크다운에서는 윗첨자를 ^로 묶어야 합니다(x^2^). 수식 표현의 경우 여전히 캐럿 하나를 사용합니다($x^2$).

###### 14.2.1 기본 구문

다른 요소의 구문은 Pandoc의 마크다운에서도 다소 동일하게 유지됩니다. 예를 들어 1단계 섹션 헤더에는 # 기호 1개를 사용하고 2단계 헤더에는 # 기호 2개를 사용합니다. 마크다운의 기본 요소 구문은 5.2.1절을 참조하십시오. 다음은 유용할 수 있는 몇 가지 새로운 요소이며(전체 문서는 http://johnmacfarlane.net/pandoc/ 참조), 글머리 기호 아래에 이러한 요소의 짧은 예시를 보여줍니다.

- 정의 목록 및 예제 목록
  특별 용어 : 용어에 대한 설명/해설을 입력합니다.

(@) 번호가 매겨진 예제입니다. (@) 번호가 매겨진 또 다른 예제입니다.

(@cool-example) 이 예제에는 레이블이 있습니다. 이것은 일반 단락이며 여기에서 예제 (@cool-example)를 참조할 수 있습니다.

- ^[...]을 사용한 각주 및 [@id]를 사용한 인용

여기에 X에 대한 멋진 설명을 씁니다^[Y와 혼동하지 마십시오]. X는 유용합니다.

실제로 X에 대해 자세히 알아보려면 참고 문헌 [@joe2014]를 읽어야 합니다. 여기서 joe2014는 참고 문헌 데이터베이스의 키입니다.

- 그림/표 캡션

Pandoc에는 기본적으로 활성화되어 있는 implicit_figures라는 마크다운 확장이 있습니다. 다음과 같은 이미지는
![그림 캡션.](path/to/image.png)
LATEX에서 다음과 같이 렌더링됩니다.
\begin{figure}
\includegraphics{path/to/image.png} \caption{그림 캡션.}
\end{figure}
마찬가지로 다음과 같이 표 캡션을 추가할 수 있습니다. 표: 표 캡션입니다.

---

A B C

---

a 10 b
c d 25
e f

---

- 원시 TEX/HTML 콘텐츠

때때로 마크다운이 여전히 제한적이라고 느껴지고 LATEX를 사용하고 싶은 유혹을 크게 느낄 수 있습니다. 괜찮습니다. 마크다운에 원시 \TeX{} 코드를 작성할 수 있습니다.

마크다운 버전: ![긴 캡션.](foo.png)

LATEX 버전: \begin{figure}

\includegraphics[width=.8\textwidth]{foo.png} \caption[짧은 캡션]{긴 캡션.}

\end{figure} Pandoc은 이 문서를 LATEX/PDF로 변환할 때 원시 TeX 콘텐츠를 유지할 수 있습니다.

인용을 사용할 때는 참고 문헌 데이터베이스를 지정해야 합니다. LATEX에 익숙하다면 BibTEX도 알 가능성이 높습니다. 참고 문헌 데이터베이스는 YAML 메타데이터의 bibliography 필드에 지정된 .bib 파일일 수 있습니다(다음 절 참조). BibTEX를 모르는 경우 references 필드(bibliography 대신)를 사용하여 YAML 메타데이터에 참고 문헌 항목을 포함할 수 있습니다(예:

--references:

- id: joe2014
  title: A Nice Paper
  author:

- family: Smith
  given: Joe
  issued:
  year: 2014
  container-title: The Journal of Awesome Research
  type: article-journal

- id: john1980
  title: A Great Book
  author:

- family: Brown
  given: John
  issued:
  year: 1980
  publisher: An Excellent Publisher
  type: book

---

원시 TEX/HTML 코드를 제외한 모든 요소는 모든 문서 형식에서 이식이 가능합니다. 예를 들어 각주 ^[foo bar]는 출력 형식이 LATEX일 때 \footnote{foo}로 변환되고, 출력 형식이 HTML일 때는 페이지 하단의 각주 항목인 footnote-1을 링크 대상으로 하는 <a href="#footnote-1"><sup>1</sup></a>과 같은 형태가 됩니다.

마크다운의 원시 TEX가 워드로 완벽하게 변환되거나 원시 HTML이 Beamer로 완벽하게 변환될 것으로 기대해서는 안 됩니다. 원시 TEX와 HTML 콘텐츠는 꽤 복잡할 수 있으며 완벽한 변환은 거의 불가능하기 때문입니다.

###### 14.2.2 YAML 메타데이터

Pandoc 마크다운의 또 다른 중요한 확장은 YAML 메타데이터입니다. YAML은 "YAML은 마크업 언어가 아니다(YAML Ain't Markup Language)" 또는 "또 다른 마크업 언어(Yet Another Markup Language)"의 약자이며, 기본적으로 중첩된 목록 구조를 갖습니다. Pandoc은 YAML을 사용하여 제목, 저자, 날짜 정보와 같은 문서의 메타데이터를 작성합니다. 메타데이터는 대개 문서의 시작 부분에 나타나며 ---와 같은 대시 3개로 이루어진 두 줄 사이에 묶입니다. 전형적인 YAML 메타데이터는 다음과 같습니다.

--title: "A Nice Report"
author: "John Smith"
date: 2014/12/31
output:
html_document:
toc: yes
number_sections: yes
word_document: default

-- R 마크다운 문서의 본문.

rmarkdown에서 YAML 메타데이터의 가장 중요한 필드는 output 필드입니다. 이곳은 원하는 출력 형식을 지정하는 위치입니다. 이것이 누락된 경우 rmarkdown은 출력 형식을 HTML 문서로 간주합니다. 여러 형식이 지정된 경우, 사용자가 render()의 두 번째 인수를 명시적으로 지정하지 않는 한 render() 함수는 기본적으로 첫 번째 형식을 사용합니다. render('foo.Rmd', 'all')을 사용하여 output 필드에 정의된 모든 형식을 렌더링할 수도 있습니다.

###### 14.3 출력 형식

rmarkdown에는 html_document(), pdf_document(), beamer_presentation() 등과 같이 \_document 및 \_presentation 접미사가 있는 일련의 형식 함수가 있습니다. 이 함수들은 render()의 두 번째 인수로 사용할 수 있습니다(예:

library(rmarkdown)
render("foo.Rmd")
render("foo.Rmd", pdf_document())
render("foo.Rmd", word_document())
render("foo.Rmd", beamer_presentation())
render("foo.Rmd", ioslides_presentation())

각 출력 형식 함수에는 고유한 인수가 있습니다. 예를 들어 HTML 문서에서 목차를 활성화하려면 다음과 같이 호출할 수 있습니다.

library(rmarkdown)
render("foo.Rmd", html_document(toc = TRUE))

이것은 YAML 메타데이터를 다음과 같이 제공하는 것과 동일합니다.

--output:
html_document:
toc: yes

---

YAML에서 yes와 true는 모두 논리값 TRUE를 의미합니다. YAML 메타데이터를 사용하고 두 번째 인수 없이 render()를 호출할 수도 있고, YAML 메타데이터를 생략하거나 무시하고 render()에 두 번째 인수를 명시적으로 제공할 수도 있습니다. YAML 접근 방식이 더 편리하고 일반적입니다. 출력 정보가 원본 문서에 포함되기 때문입니다. 두 번째 접근 방식은 YAML에 정의된 출력 형식을 오버라이드하려는 경우 유용할 수 있습니다. 가능한 옵션이 무엇인지 알아보려면 각 출력 형식 함수의 도움말 페이지를 참조하십시오. 예를 들어 PDF 출력 옵션을 보려면 R 콘솔에 ?rmarkdown::pdf_document를 입력하십시오.

출력 형식 함수는 knitr 패키지/청크 옵션, Pandoc 인수, 기타 rmarkdown용 보조 옵션 등을 포함하는 옵션 목록을 반환합니다. html_document()를 예로 들어 설명하겠습니다.

###### 14.3.1 HTML 문서

html_document()가 실제로 반환하는 것을 확인하려면 이를 실행하고 반환된 객체의 구조를 인쇄해 볼 수 있습니다.

library(rmarkdown)
str(html_document(), width = 55, strict.width = "wrap")

이 목록에는 Pandoc 옵션도 포함되어 있습니다. pandoc$to 요소에서 볼 수 있듯이 출력 형식은 html이며, --smart 및 --self-contained와 같은 몇 가지 Pandoc 인수도 이 목록에 포함되어 있습니다.

rmarkdown을 위한 몇 가지 보조 옵션도 존재합니다. 예를 들어 clean_supporting은 HTML 파일이 렌더링된 후 중간 출력 파일을 정리할지 여부를 의미합니다. 중간 파일에는 그림 파일이 포함될 수 있습니다. HTML 파일을 자체 포함형으로 만들고 싶은 경우, Pandoc은 외부 리소스(이미지)를 해당 파일 안에 모두 삽입하므로 더 이상 이러한 외부 파일이 필요하지 않게 됩니다. 이런 상황이라면 render()는 HTML 파일을 렌더링한 후 외부 파일들을 삭제할 것입니다.

출력 형식 함수의 내부 구조를 알게 되면, 다른 knitr/Pandoc 옵션들을 사용하여 고유한 형식 함수를 만들 수 있습니다. 사용자 지정 형식을 구현하는 방법은 이 장의 후반부에서 소개하도록 하겠습니다.

이제 Rmd-v2.Rmd라는 이름의 R 마크다운 v2 문서에 대한 전체 예시를 보여드리겠습니다. 내용이 다소 길지만 Pandoc과 rmarkdown의 대부분 기능을 살펴볼 수 있습니다.

--title: "R 마크다운 v2 데모" author:

- 리 레이
- 한 메이메이 date: "2015/01/01" output:
  html*document:
  fig_caption: yes pdf_document:
  template: null
  word_document: default bibliography: Rmd-v2.bib
  --# 멋진 섹션으로 시작 약간의 *소개\_입니다. [링크](http://yihui.name/knitr)와 코드와 같은 기존의 **마크다운** 구문을 사용할 수 있습니다. # 또 다른 섹션 계속 물론 다음과 같은 목록을 작성할 수 있습니다.

- 사과
- 배
- 바나나
  또는 순서가 지정된 목록:

1. 항목이
1. 순서대로
1. 나열될
1. 것입니다

- 중첩된
- 항목들 # 더 많은 섹션 ## 안녕 안녕 안녕 ## 안녕하세요 안녕하세요 안녕하세요 ## 하우디 하우디 하우디 # 자, 일부 R 코드 {r linear-model} fit = lm(dist ~ speed, data = cars) b = coef(fit) # 계수 summary(fit)

코드는 모든 출력 형식에서 강조 표시됩니다. # 그리고 일부 그림 {r lm-vis, fig.cap= '회귀 진단' } par(mfrow = c(2, 2), pch = 20, mar = c(4, 4, 2, .1),

bg = 'white' ) plot(fit)

# 약간의 수학 우리의 회귀 방정식은 $Y= r b[1] + r b[2] x$이며 모델은 다음과 같습니다.

$$ Y = \beta_0 + \beta_1 x + \epsilon$$ # Pandoc 확장: 정의 목록 프로그래머 : 커피를 코드로 바꾸는 사람입니다. LATEX : 몇 개의 백슬래시가 있는 간단한 언어입니다. # Pandoc 확장: 예제 몇 가지 예제가 있습니다. (@) 0.3 + 0.4 - 0.7이 무엇인지 생각해 보세요. 영(0)입니다. 쉽죠. (@weird) 이제 0.3 - 0.7 + 0.4가 무엇인지 생각해 보세요. 여전히 영일까요? 사람들은 (@weird)에 종종 놀랍니다. # Pandoc 확장: 표 여기에 표가 있습니다. 표: 간단한 표 구문의 예. {r echo=FALSE} knitr::kable(head(iris))

# Pandoc 확장: 각주 각주[^1]를 쓸 수도 있습니다. [^1]: 안녕, 나는 각주야 아니면 인라인 각주^[여기서 볼 수 있듯이]를 쓸 수도 있습니다. # Pandoc 확장: 인용 R [@R-base]의 **knitr** [@R-knitr]를 통해 R 마크다운 파일을 마크다운으로 컴파일합니다. @R-knitr에 대한 자세한 내용은 <http://yihui.name/knitr>를 참조하십시오.

- 그림 14.1: RStudio 창에서 확인하는 R 마크다운 v2 HTML 출력 문서 미리보기.

# 참고 문헌 {r include=FALSE} knitr::write_bib(c('base', 'knitr'), 'Rmd-v2.bib')

kable() 또는 write_bib()가 어떻게 작동하는지 확실하지 않다면 6.3절과 12.4.1절을 복습해야 할 수도 있습니다.

그림 14.1은 RStudio에서 이 예제를 렌더링한 후의 HTML 출력 문서 미리보기입니다. 제목, 저자, 날짜 및 문서의 처음 몇 섹션을 보여줍니다. 이것이 rmarkdown의 기본 트위터 부트스트랩 스타일입니다. 그림 14.2는 마지막 몇 섹션의 미리보기입니다. 각주와 인용이 HTML의 네이티브 요소는 아니지만(LATEX 사용자에게는 자연스러울 수 있음) 어쨌든 Pandoc은 이를 HTML로 생성해 냈습니다.

HTML 출력에 맞게 조정할 수 있는 많은 옵션이 있습니다. 전체 목록은 도움말 페이지 ?rmarkdown::html_document를 참조하십시오.

- 그림 14.2: 표, 각주, 인용 미리보기: 표는 kable()에 의해 생성되었고 서지 데이터베이스는 knitr의 write_bib()에 의해 생성되었습니다.

예를 들어 YAML에서 theme 필드를 사용하여 CSS 테마를 변경하고, toc 필드를 사용하여 목차를 추가하며, number_sections 필드를 사용하여 섹션 제목에 번호를 매길 수 있습니다(그림 14.3):

--output:
html_document: fig_caption: yes number_sections: yes theme: readable toc: yes

---

현재 이러한 CSS 테마는 rmarkdown에서 사용할 수 있습니다(http://bootswatch.com 에서 미리 볼 수 있음):

## [1] "default" "cerulean" "journal" "flatly" ## [5] "readable" "spacelab" "united" "cosmo"

출력 모양을 더 미세하게 조정해야 하는 경우 css 필드를 사용하여 고유한 CSS 파일을 적용할 수 있습니다. 예:

--output:
html_document: css: my_own.css

---

단지 자신만의 CSS를 사용하고 rmarkdown이 제공하는 테마(구문 강조 테마 포함)를 원하지 않는 경우, theme 및 highlight를 null로 지정하여 이를 완전히 제거할 수 있습니다.

--output:
html_document: css: my_own.css theme: null highlight: null

---

HTML 페이지는 CSS, 자바스크립트, 이미지 파일과 같은 외부 종속성이 있는 경우가 많기 때문에 다른 사람과 HTML 파일을 공유할 때 불편할 수 있습니다. 파일을 보낼 때 이러한 종속성도 함께 포함되었는지 확인해야 하기 때문입니다.

- 그림 14.3: 목차와 번호가 매겨진 섹션이 있는 "readable" 테마의 미리보기(그림 14.1과 글꼴이 다른 것을 알 수 있습니다).

Pandoc에는 모든 외부 종속성을 HTML 파일에 임베딩하여 HTML 파일을 자체 포함형으로 만드는 옵션이 있습니다. 예를 들어, 자바스크립트 파일은 HTML 파일 안으로 읽혀 들어가고, 이미지는 base64로 인코딩됩니다. 마치 PDF 파일처럼 자체 포함형 HTML 파일을 공유할 수 있으며, 필요한 모든 것이 하나의 파일에 통합되어 있습니다. rmarkdown에서 이는 self_contained 옵션에 의해 제어됩니다. rmarkdown을 통해 렌더링해야 할 Rmd 파일이 여러 개일 경우 자체 포함형 모드를 끄는 것이 좋을 수 있습니다. 그렇지 않으면 일부 외부 종속성이 개별 HTML 출력 파일마다 모두 삽입되어 중복이 많이 발생할 것입니다. 자체 포함형 모드가 꺼져 있으면 공유 종속성을 lib_dir 옵션으로 지정한 공통 디렉터리에 넣을 수 있습니다. 예:

--output:
html_document: self_contained: no lib_dir: assets

---

때때로 문서의 본문 이전이나 이후, 또는 HTML 헤더에 추가 콘텐츠를 넣고 싶을 수 있습니다. 이러한 경우 rmarkdown에는 추가 콘텐츠 파일의 이름을 지정할 수 있는 includes 옵션이 있습니다. HTML 출력에서 자바스크립트 라이브러리 D3(http://d3js.org)를 사용하고 싶다면, doc_header.html 파일에 다음과 같이 작성할 수 있습니다.

<script src="http://d3js.org/d3.v3.min.js" charset="utf-8"> </script>

또한 doc_before.html 및 doc_after.html이라는 두 개의 파일도 있는데, 이는 각각 본문 이전과 이후에 삽입될 콘텐츠를 담습니다. 예를 들어, doc_before.html에는 탐색 메뉴를, doc_after.html에는 일부 저작권 정보를 작성할 수 있습니다. 다음 설정을 통해 이 세 개의 파일을 HTML 출력 파일에 포함할 수 있습니다.

--output:
html_document:
includes: in_header: doc_header.html before_body: doc_before.html after_body: doc_after.html

모든 출력 형식에서 Pandoc은 출력 파일을 생성하기 위한 템플릿이 필요합니다. 템플릿에서 사용할 수 있는 몇 가지 Pandoc 변수가 있으며 이러한 변수를 사용하여 나만의 템플릿을 정의할 수 있습니다. 예를 들어 다음은 최소한의 HTML 템플릿이 될 수 있습니다.

<html> <head>
<title>$title$</title> </head> <body> $body$ </body>
</html>

이 템플릿에서는 두 개의 변수 $title$과 $body$만 사용했습니다. 첫 번째 변수는 YAML 메타데이터의 title 필드에 지정된 문서 제목을 포함합니다. 두 번째 변수는 마크다운 문서가 HTML로 변환된 후의 문서 본문입니다. rmarkdown 소스 패키지(https://github.com/rstudio/rmarkdown) 또는 Pandoc의 기본 템플릿(https://github.com/jgm/pandoc-templates)에서 더 많은 사용 가능 변수를 알아볼 수 있습니다.

사용자 지정 템플릿을 사용하려면 YAML의 template 필드를 사용할 수 있습니다. 예:

--output:
html_document: template: my_template.html

---

마지막으로 pandoc_args 필드를 사용하여 Pandoc에 전달할 명령줄 인수를 사용자 지정할 수 있습니다. 실제로는 html_document()에 있는 R 인수가 최종적으로 Pandoc 인수로 변환됩니다. 예를 들어 R 인수 self_contained = TRUE(또는 YAML의 self_contained: yes)는 Pandoc 인수 --self-contained와 동일하며 YAML의 다음 설정과도 같습니다.

--output:
html_document: pandoc_args: "--self-contained"

지금까지 우리는 Pandoc의 마크다운 측면에서 출력을 사용자 지정할 수 있는 대부분의 옵션을 다루었습니다. YAML에서 knitr 청크 옵션을 사용자 지정하는 것도 가능합니다. 현재 YAML에서 설정할 수 있는 청크 옵션은 다음 네 가지입니다.

fig_width, fig_height 그림의 기본 크기

fig_retina 레티나 디스플레이를 위한 배율 지정; rmarkdown의 기본값은 2입니다. 즉, 크기가 m × n인 그림의 실제 크기는 2m × 2n이지만 출력에서는 실제 크기의 절반으로 축소됩니다(이를 통해 레티나 디스플레이에서 이미지 품질을 향상시킬 수 있습니다).

fig_caption 그림 캡션을 렌더링하고 표시할지 여부(출력 형식이 LATEX일 때 기본적으로 \caption{}이 포함된 figure 환경을 의미함); FALSE인 경우 캡션은 보이지 않는 <img> 태그의 alt 속성에 배치되므로 HTML 출력에서 그림 캡션이 보이지 않습니다.

분명히 fig_retina 옵션은 이미지 품질을 향상하는 대가로 이미지의 파일 크기를 커지게 만들 것입니다. fig_retina = TRUE와 FALSE를 번갈아 시도하여, 사용 중인 기기에서 어떤 차이를 느낄 수 있는지 확인해 볼 수 있습니다.

###### 14.3.2 LATEX/PDF 문서

HTML 문서 형식에 친숙해지면 다른 출력 형식도 마스터하기 쉬워질 것입니다. 많은 옵션이 이러한 형식들 전반에 걸쳐 공통적이기 때문입니다. 예를 들어 pdf_document()에서도 fig_width, fig_height, toc, number_sections, highlight와 같은 옵션을 사용할 수 있습니다. 이 절에서는 PDF 문서 출력에 특화된 옵션에만 중점을 둡니다.

그림 14.4는 앞 절에서 사용한 동일한 예제로부터 생성된 PDF 출력 중 한 페이지의 미리보기입니다. 이는 그림 14.2와 크게 다르지 않아 보입니다. 동일한 R 마크다운 문서에 대해 HTML 출력에서 작동했던 모든 요소들(섹션 제목, 표, 각주, 인용 등)이 LATEX/PDF에서도 여전히 잘 작동합니다.

마찬가지로 HTML 출력에서와 같이 목차를 추가하고 섹션 번호를 지정할 수 있습니다(그림 14.5).

--output:
pdf_document: number_sections: yes toc: yes

- 그림 14.4: R 마크다운 v2 예제에서 생성된 PDF 출력 문서의 네 번째 페이지 미리보기.

- 그림 14.5: 목차와 번호가 매겨진 섹션이 있는 PDF 출력 문서의 미리보기.

Pandoc에는 YAML 메타데이터에서 사용할 수 있는 LATEX 전용 옵션이 몇 가지 있으며 전체 설명서는 Pandoc 웹사이트에서 확인할 수 있습니다. 여기에 그 중 몇 가지만 나열합니다.

fontsize 문서의 글꼴 크기(10pt, 11pt, 12pt)
documentclass 문서 클래스(article, book, report)
classoption 문서 클래스에 대한 옵션(a4paper, twocolumn)
geometry geometry 패키지의 옵션(tmargin=2cm, bmargin=2cm, lmargin=3cm, rmargin=3cm)

이들은 YAML의 최상위 레벨 옵션이며 pdf_document 필드 아래에 두어서는 안 됩니다.

기본 LATEX 엔진은 pdflatex이며 pdf_document() 내의 latex_engine 옵션을 통해 변경할 수 있습니다. 현재 사용 가능한 엔진은 pdflatex, xelatex, lualatex입니다. 또한 디버깅 및 기타 목적을 위해 유용할 수 있는 중간 LATEX 출력 파일을 keep_tex 옵션을 사용하여 유지할 수도 있습니다.

다음은 book 클래스, 11pt의 글꼴 크기, 2단 레이아웃(twocolumn), 사용자 지정 여백 설정, XeLATEX 엔진을 사용하고 LATEX 파일을 보존하도록 하는 YAML 메타데이터의 예시입니다.

--documentclass: book classoption: twocolumn fontsize: 11pt geometry:

- tmargin=2cm
- bmargin=2cm
- lmargin=3cm
- rmargin=3cm

output:
pdf_document: latex_engine: xelatex keep_tex: yes

---

앞 절에서 includes와 template 옵션을 소개했는데, 이는 LATEX 출력에 더 유용할 수 있습니다. 왜냐하면 LATEX 사용자는 문서 서문(preamble)에서 특정 LATEX 패키지를 사용하여 출력을 커스터마이징하는 것이 매우 일반적이기 때문입니다. 이러한 내용을 외부 파일에 넣고 includes 옵션 아래의 in_header 옵션을 통해 서문에 포함시킬 수 있습니다. 기본 LATEX 템플릿에 만족하지 못하는 경우 직접 작성할 수도 있습니다. 실제로 템플릿을 작성하기 전에, YAML 옵션을 통해 원하는 결과를 얻을 수 있는지 확인하기 위해 Pandoc 문서를 주의 깊게 살펴보시기 바랍니다. 새로운 LATEX 템플릿을 작성하는 것은 비교적 쉽지만, 향후 Pandoc의 변경 가능성을 인지하고 있어야 하므로 그 템플릿을 관리하는 것은 간단하지 않을 수 있습니다.

###### 14.3.3 워드 문서

워드 문서를 위해 사용자 지정할 수 있는 옵션은 많지 않습니다. 그래도 여전히 그림 크기, 구문 강조 테마 등을 설정할 수 있습니다. 그림 14.6은 마이크로소프트 워드 2013에서 열린 예제의 워드 출력을 보여줍니다.

워드 문서에서 가장 중요하고 유용한 기능은 아마도 템플릿일 것입니다. 다른 문서 형식의 경우 일반 텍스트 템플릿을 제공할 수 있지만 워드의 경우 그렇게 하기 쉽지 않습니다. 워드 문서는 상대적으로 복잡한 이진(binary) 파일이기 때문입니다. 하지만 Pandoc은 워드 문서를 "참조 문서"로 제공할 수 있게 해 주며, 이는 본질적으로 스타일 템플릿 역할을 합니다. 이 참조 문서는 Pandoc의 워드 출력 문서 중 하나를 기반으로 해야 하며, 거기서 여러 요소들의 스타일을 업데이트합니다. 참고로 이 문서에 정의된 스타일만 사용되며, 콘텐츠는 대부분 무시됩니다.

워드 문서에서 스타일을 정의하는 방법을 보여주기 위해 짧은 비디오를 준비했습니다(https://vimeo.com/110804387). 그림 14.7 및 14.8도 함께 확인할 수 있습니다. 기본 단계는 다음과 같습니다.

1. YAML 메타데이터의 출력 옵션으로 word_document를 사용하여 Pandoc으로 임의의 워드 문서를 생성합니다.
2. 워드 문서를 열고 그림 14.7에 표시된 "스타일" 패널을 찾습니다.
3. 스타일을 수정하려는 요소에 커서를 놓으면 스타일 패널에서 하나의 항목이 강조 표시되어야 합니다.
4. 오른쪽에 있는 ¶ 기호를 클릭하여 항목을 열면 그림 14.8과 같은 창이 나타납니다. 이곳에서 스타일을 수정할 수 있습니다. 예를 들어, 제목 요소의 글꼴 계열(font family)을 Bookman Old Style로 변경할 수 있습니다.

이 워드 문서의 스타일을 업데이트한 후 (예를 들어 Rmd 파일과 동일한 디렉터리에 template.docx로) 저장하고 이를 참조 문서로 사용할 수 있습니다.

--output:
word_document:

- 그림 14.6: R 마크다운 v2로 생성한 마이크로소프트 워드(2013) 문서의 미리보기.

- 그림 14.7: 워드에서 스타일 패널 열기: 도구 모음에서 "스타일"이라는 창을 찾아 플로팅 패널로 확장합니다.

## renference_docx: template.docx

요소의 스타일 외에도 Pandoc 버전을 1.13 이상으로 사용하는 경우 레이아웃 스타일도 반영될 수 있습니다. 예를 들어 참조 문서의 여백, 페이지 크기, 페이지 방향, 머리글 및 바닥글이 새로운 워드 문서에도 그대로 적용됩니다.

###### 14.3.4 마크다운 문서

R 마크다운 문서는 Pandoc 마크다운, 원본(순수) 마크다운, Github Flavored 마크다운, MultiMarkdown 및 PHP 마크다운 엑스트라 등 다양한 종류의 마크다운 문서로 변환될 수 있습니다. YAML에서 output: md_document를 사용하거나 render()를 위한 함수로 md_document()를 사용할 수 있습니다. md_document의 주요 옵션은 원하는 마크다운 종류를 지정하는 variant입니다.

- 그림 14.8: 워드에서 요소의 스타일 수정하기: 글꼴 계열, 글꼴 크기, 글꼴 스타일, 색상 등을 변경할 수 있습니다.

###### 14.3.5 ioslides 프레젠테이션

R 마크다운은 프레젠테이션용 슬라이드를 만드는 데 사용할 수 있습니다. 웹 기술의 발전에 따라 최근에는 HTML5 슬라이드가 인기를 얻고 있습니다. 웹 브라우저에서 슬라이드를 발표할 수 있습니다. 슬라이드를 표시하기 위해 특수한 소프트웨어 패키지가 필요하지 않으며 거의 모든 곳에서 웹 브라우저를 찾을 수 있기 때문에 편리합니다. 마이크로소프트 파워포인트나 Mac의 키노트 같은 독점 소프트웨어의 경우에는 해당하지 않는 이야기입니다.

rmarkdown에는 내장된 두 가지 HTML5 프레젠테이션 형식인 ioslides와 Slidy가 있습니다. rmarkdown을 확장하여 여러분이 선호하는 자신만의 HTML5 프레젠테이션 라이브러리를 사용할 수도 있습니다.

ioslides의 경우 각 최상위(1단계) 섹션 제목은 기본적으로 짙은 배경을 지닌 별도의 슬라이드를 생성합니다. 각 2단계 섹션 제목은 해당 섹션의 내용을 포함하는 새 슬라이드를 만듭니다. 섹션 제목을 원하지 않는다면 세 개의 대시(---)를 사용하여 새 슬라이드를 만들 수 있습니다.

- 그림 14.9는 이전 섹션들과 YAML 메타데이터가 동일한 예제를 사용하여 생성한 ioslides를 RStudio 미리보기 창에서 본 스크린샷입니다(이 예제를 실제로 사용해 보려면 첫 번째 레벨 헤딩과 두 번째 레벨 헤딩 사이의 내용을 지우고 싶을 수 있습니다).

--output:
ioslides_presentation: default

---

프레젠테이션을 할 때 전체 화면 모드를 사용하고 싶을 수 있는데, 키보드 단축키 f(문자 F 키 누르기)를 통해 이 모드를 켤 수 있습니다. W 키를 누르면 와이드스크린 모드가 켜고 꺼집니다. 슬라이드 크기가 너무 크거나 작으면 화면을 확대/축소할 수 있습니다. 일반적으로 Ctrl(또는 Command) 키를 누른 상태에서 플러스(+) 또는 마이너스(-)를 누르면 됩니다.

슬라이드의 모양을 조절하기 위해 ioslides_presentation 형식에서 사용할 수 있는 몇 가지 옵션은 다음과 같습니다.

incremental (yes/no) 글머리 기호를 순차적으로 표시할지 여부
logo 슬라이드 로고로 사용할 이미지(각 슬라이드의 바닥글에 표시됨)
css 사용자 정의 CSS 파일

각 슬라이드를 개별적으로 사용자 지정할 수도 있습니다. 예를 들어 2단계 섹션 헤딩 뒤에 {.build}라는 토큰을 넣으면 프레젠테이션 진행 상황에 따라 이 페이지의 요소들이 점진적으로(순차적으로) 화면에 표시됩니다. 예:

## 새로운 슬라이드 {.build} 먼저 이것을 보여주세요. 그런 다음 저것을 보여주세요. 마지막으로 재미있는 GIF 애니메이션을 보여주세요. ![](foo.png)

HTML5 슬라이드는 일반적으로 인쇄 목적이 아니라 프레젠테이션용입니다. 하지만 웹 브라우저에서 슬라이드를 PDF로 인쇄할 수도 있습니다. 현재 슬라이드를 인쇄하고 싶다면 Google Chrome을 사용하는 것을 권장합니다. 인쇄된 슬라이드의 모양은 화면에 표시되는 슬라이드와 다를 수 있다는 점을 감안해야 합니다.

###### 14.3.6 Slidy 프레젠테이션

Slidy를 위한 슬라이드 작성 규칙은 ioslides와 같습니다. rmarkdown에서 Slidy 프레젠테이션 출력을 위한 함수는 slidy_presentation()입니다.

- 그림 14.10은 R 마크다운 예제에서 생성된 Slidy 프레젠테이션의 한 슬라이드를 보여줍니다.

목차를 보기 위한 C, 글꼴을 작게 만드는 S, 글꼴을 크게 만드는 B 등 몇 가지 키보드 단축키를 사용할 수 있습니다.

- 그림 14.10: R 마크다운 예제에서 생성된 Slidy 프레젠테이션의 한 슬라이드. 하단의 "Contents"를 클릭하면 목차를 볼 수도 있습니다.

이전에 언급한 incremental 및 css 옵션 외에도 Slidy에는 유용할 수 있는 몇 가지 추가 기능이 있습니다. 이러한 옵션은 다음과 같습니다.

duration 지정한 시간을 알 수 있도록 바닥글에 카운트다운 타이머를 설정합니다(50분 동안 발표를 해야 하는 경우 YAML에 duration: 50을 설정할 수 있음).

footer 바닥글에 사용자 지정 메시지를 표시합니다(기관명이나 저작권 정보를 표시할 수 있음).

Slidy 슬라이드를 인쇄하려면 Google Chrome을 사용할 수도 있습니다.

###### 14.3.7 Beamer 프레젠테이션

12.3.4절에서 소개된 Beamer는 LATEX 애플리케이션이므로 12.3.4절에서 살펴본 것처럼 코드 청크가 포함된 Rnw 파일을 LATEX 문서로 빌드하여 직접 PDF 형식으로 컴파일할 수 있습니다. 숙련된 LATEX 사용자를 제외하면 마크다운이 더 간단하고 빠르기 때문에 beamer_presentation 형식을 사용하는 것을 권장합니다. 보다 고급 형태의 Beamer 또는 LATEX 기능이 필요하다면 Pandoc이 마크다운 내에서 LATEX 코드를 지원하므로 마크다운 내에 추가할 수 있습니다.
그림 14.11은 이전 R 마크다운 예제에서 생성된 Beamer 프레젠테이션의 슬라이드 두 장을 보여줍니다. 우리가 한 일이라고는 YAML 메타데이터를 다음과 같이 변경한 것뿐입니다.

--title: "R 마크다운 v2 데모" author:

- 리 레이

- 한 메이메이 date: "2015/01/01" output:

beamer_presentation: theme: AnnArbor bibliography: Rmd-v2.bib

---

슬라이드를 원시 LATEX로 작성한다면 소스 문서는 다음과 같을 것입니다.

###### \documentclass{beamer} \usetheme{AnnArbor}

\title{R 마크다운 v2 데모} \author{리 레이 \and 한 메이메이} \date{2015/01/01}

###### \begin{document} \frame{\titlepage}

\begin{frame}{멋진 섹션으로 시작} 약간의 \emph{소개}입니다. \href{http://yihui.name/knitr}{링크} 및 \texttt{코드}와 같은 기존의 \textbf{마크다운} 구문을 사용할 수 있습니다. \end{frame} \begin{frame}{또 다른 섹션 계속}

- 그림 14.11: R 마크다운으로 만든 Beamer 프레젠테이션의 슬라이드 두 장: 제목 슬라이드와 예제 환경의 Pandoc 확장을 보여주는 슬라이드.

물론 다음과 같은 목록을 작성할 수 있습니다.

\begin{itemize} \item

사과 \item

###### 배 \item

바나나

\end{itemize}

.... \end{document}

이를 14.3.1절의 R 마크다운 소스 코드와 비교해 보면 원시 LATEX로 작성할 때 마크다운으로 작성할 때보다 얼마나 더 많은 코드를 입력해야 하는지 알 수 있을 것입니다.

각각의 새로운 슬라이드는 마크다운의 새로운 섹션이 되며, 섹션의 레벨은 슬라이드 콘텐츠 바로 앞에 오는 문서 계층 구조의 가장 높은 레벨에 의해 결정됩니다. 다음 예에서 각 첫 번째 레벨 헤딩(#)은 새로운 슬라이드가 됩니다.

--output: beamer_presentation

--# 한 섹션

- 내용
- 내용 # 또 다른 섹션 ![](foo.png)

이 예에서는 각 하위 섹션(##)이 새로운 슬라이드입니다.

--output: beamer_presentation

--# 한 섹션

## 하나의 하위 섹션

- 내용 - 내용

# 또 다른 섹션 ## 또 다른 하위 섹션 ![](foo.png)

목록 항목을 점진적(순차적)으로 표시하려면 ioslides 및 Slidy 프레젠테이션에서 할 수 있는 것처럼 incremental 옵션을 사용할 수 있습니다. toc, highlight, fig_width, fig_height, fig_caption, includes, template과 같은 다른 옵션은 이전 절에서 설명했습니다.

Beamer에는 (글꼴 테마 및 색상 테마를 포함하여) 많은 테마가 있습니다. theme, fonttheme, colortheme 옵션을 통해 이를 사용할 수 있습니다. 그림 14.11에서는 AnnArbor 테마와 기본 글꼴/색상 테마를 사용했습니다. RStudio를 사용하는 경우 GUI에서 이러한 테마를 선택할 수 있으므로 많은 테마 이름을 기억할 필요가 없습니다.

###### 14.3.8 기타 형식

문서 및 프레젠테이션 형식 외에도 rmarkdown에는 두 가지 특수한 출력 형식이 있습니다. HTML 패키지 비네트를 위한 html_vignette()(15.4절)와 Tufte 핸드아웃(Tufte handout, 여기서 Tufte는 Edward R. Tufte를 뜻함)을 위한 tufte_handout()입니다.

html_vignette() 형식은 특별한 CSS 테마를 가진 html_document()의 래퍼입니다. 기본적으로 트위터 부트스트랩 리소스, jQuery 라이브러리 및 highlight.js를 포함하기 때문에 html_document()에서 생성한 HTML 비네트의 파일 크기는 너무 큽니다. html_vignette() 형식은 이러한 구성 요소를 모두 제거하고 가벼운 단일 CSS 파일을 사용합니다. 또한 이미지 파일 크기를 더 줄이기 위해 fig_retina 옵션을 1로 설정했습니다. 이 형식 함수는 기존 형식 함수를 기반으로 자신만의 형식을 빌드하는 방법을 보여주는 좋은 예이며, 그 소스 코드는 매우 간단합니다.

html_vignette <- function(fig_width = 3, fig_height = 3, dev = "png", css = NULL,

...) { if (is.null(css)) {

css <- system.file("rmarkdown", "templates",

"html_vignette", "resources", "vignette.css", package = "rmarkdown")

} html_document(fig_width = fig_width,

fig_height = fig_height, dev = dev, fig_retina = FALSE, css = css, theme = NULL, highlight = "pygments", ...)

}

tufte_handout() 형식은 LATEX 문서 클래스 tufte-handout.cls의 래퍼입니다. Tufte 핸드아웃 스타일의 가장 두드러진 특징은 아마도 여백 노트(sidenotes)를 사용하고 조판이 잘 설계되었다는 점일 것입니다. 그림 14.12에서 예제 페이지를 참조하십시오. 그 YAML 메타데이터는 다음과 같습니다.

--title: "Tufte Handout" author: "John Smith" date: "2014년 8월 13일" output: rmarkdown::tufte_handout

---

###### 14.4 Shiny를 사용한 대화형 문서

Shiny(Chang 등, 2015)는 R을 사용하여 대화형 앱을 쉽게 만들 수 있게 해주는 웹 애플리케이션 프레임워크입니다. 텍스트 입력 상자, 드롭다운 목록, 라디오 버튼, 슬라이더 등과 같은 Shiny UI 함수를 사용하여 웹 사용자 인터페이스(UI)를 만들 수 있습니다. 이러한 UI 요소들은 버튼을 클릭한 후 R에서 기대하는 동작 등 R 내의 서버 로직을 지정한 뒤에 R과 상호 작용할 수 있습니다. Shiny에 익숙하지 않은 경우 웹사이트 http://shiny.rstudio.com 에서 Shiny에 대한 기본 사항을 알아보시기 바랍니다.

Shiny 앱은 기본적으로 HTML 페이지이고 R 마크다운 역시 HTML로 렌더링될 수 있기 때문에 R 마크다운과 Shiny를 하나의 문서에 결합하는 것이 가능합니다. 이러한 문서를 "대화형 문서(interactive documents)"라고 부르며 여기에는 Shiny의 대화형 구성 요소가 포함되어 있습니다. 그림 14.13은 대화형 문서의 최소 예제를 보여줍니다. 소스 문서는 다음과 같습니다.

- 그림 14.12: Tufte 핸드아웃 스타일을 사용한 예제 페이지: 각주, 그림, 수식 등과 같은 요소들을 측면 여백에 정렬할 수 있습니다.

- 그림 14.13: R 마크다운과 Shiny를 사용한 간단한 대화형 문서: 슬라이더 값을 변경하면 히스토그램의 막대(bins) 개수가 자동으로 변경됩니다.

--title: "R 마크다운 v2 데모" runtime: shiny output: html_document

-- {r} library(shiny) sliderInput("bins", "막대 개수:", min = 1, max = 50,

value = 30)

renderPlot({ x <- faithful[, 2] # Old Faithful Geyser 데이터 bins <- seq(min(x), max(x), length.out = input$bins + 1) # 지정된 막대 수로 히스토그램 그리기 hist(x, breaks = bins, col = 'darkgray', border = 'white')

})

일반적인 R 마크다운 문서를 대화형 문서로 전환하려면 YAML 메타데이터에 runtime: shiny 옵션만 추가하면 됩니다. 그러면 shiny 패키지의 함수를 사용할 수 있습니다. 위 예시에서는 shiny의 UI 함수인 sliderInput()을 사용하여 HTML 페이지에 슬라이더를 만들었습니다. 슬라이더의 id는 bins입니다. 그런 다음 renderPlot() 함수를 사용하여 히스토그램을 렌더링했습니다. 이 코드 청크에서 가장 중요한 부분은 input$bins로 이는 id가 bins인 슬라이더와 연관된 변수 값입니다. 슬라이더의 값을 업데이트하면 그 값이 renderPlot()의 표현식으로 전달되고 그에 따라 플롯이 다시 그려집니다.

대화형 문서는 render() 대신 rmarkdown의 run() 함수로 컴파일해야 합니다. RStudio를 사용하는 경우 R 마크다운 문서에 runtime: shiny를 추가하면 도구 모음의 Knit 버튼 레이블이 Run Document로 바뀌는 것을 볼 수 있으며 이 버튼을 클릭하여 문서를 실행할 수 있습니다.

모든 Shiny 앱이 그림 14.13처럼 간단할 수는 없습니다. 여러 UI 요소가 있는 경우, 코드 청크에 요소들을 선형으로 입력하는 대신 별도의 앱으로 구성하고 싶을 수 있습니다. shiny의 shinyApp() 함수를 사용하면 모든 UI 요소와 서버 로직을 하나의 함수에 지정하여 전체 앱을 빌드할 수 있습니다. 그런 다음 R 마크다운에 shinyApp()을 명시적으로 사용하여 전체 앱을 임베딩하거나, shinyApp() 객체를 반환하는 사용자 정의 함수를 작성하여 다른 사람도 앱을 쉽게 사용할 수 있도록 할 수 있습니다.

정적 HTML 문서는 공유하고 싶을 때 모든 웹사이트에 업로드하거나 이메일로 전송할 수 있습니다. 대화형 문서의 경우, 문서 이면에서 백그라운드로 실행되는 활성 R 세션이 있어야 합니다. 대화형 문서를 공유할 수 있는 한 가지 방법은 RStudio가 호스팅하는 http://shinyapps.io 에 게시하는 것입니다. 이 웹사이트에 게시하고 싶지 않다면 자체 Shiny 서버(http://www.rstudio.com/products/shiny/shiny-server/)를 설정할 수 있습니다.

###### 14.5 R 마크다운 v2 확장하기

출력 형식 함수 중에 사용자의 요구를 충족하는 것이 없다면, 사용자는 이를 확장하거나 완전히 새로운 형식을 작성할 수 있습니다. 그렇게 하기 전에 먼저 기존 출력 형식에 있는 모든 가능한 옵션들을 살펴보시기 바랍니다. 때로는 완전히 새로운 것을 발명할 필요가 없습니다. 예를 들어 원하는 것이 단순히 다른 LATEX 문서 클래스를 사용하는 것이라면, (원하는 문서 클래스로 새로운 템플릿을 작성할 수도 있겠지만) YAML 메타데이터의 documentclass 옵션을 설정하는 것이 낫습니다. Tufte 핸드아웃을 예로 들어 보겠습니다.

--title: "R 마크다운 v2 데모" author: John Smith date: "2015/01/01" output: pdf_document documentclass: tufte-handout classoption: nohyper geometry: no

---

위의 YAML 메타데이터는 기존의 pdf_document() 형식을 활용합니다. 대안으로 다음과 같은 템플릿을 준비할 수도 있습니다. \documentclass{tufte-handout} $if(title)$ \title{$title$} $endif$ $if(author)$ \author{$for(author)$$author$$sep$ \and $endfor$} $endif$ $if(date)$ \date{$date$}

$endif$ \begin{document} $if(title)$ \maketitle $endif$ $body$ \end{document}

그런 다음 pdf_document의 template 옵션을 사용합니다. 이처럼 사용자 지정 템플릿을 작성할 때 몇 가지 단점이 있습니다.

- Pandoc의 기본 LATEX 템플릿(https://github.com/jgm/pandoc-templates)이 훨씬 유연하여, 목차, 그림 목록 및 초록 등도 처리할 수 있습니다.
- 새로운 템플릿을 작성하는 것은 YAML의 기존 옵션을 사용하는 것보다 더 많은 수고가 듭니다.
- 템플릿을 작성한 후에는 향후 Pandoc의 변경 사항으로 인해 템플릿이 손상되거나 유용한 새 기능을 놓칠 수 있으므로 주의해야 합니다. 반면, Pandoc의 템플릿을 사용하면 유지 관리할 필요가 없습니다.

그렇다면 rmarkdown에는 왜 굳이 tufte_handout() 형식이 있는지 물을 수 있습니다. 사실 이 새로운 형식은 LATEX 템플릿 그 이상의 역할을 합니다. 즉, 전체 폭 그림(fig.fullwidth = TRUE) 및 여백 그림(fig.margin = TRUE)을 생성하기 위한 몇 가지 knitr 청크 옵션도 정의합니다. 기존 출력 형식들은 이 두 가지 다른 종류의 그림을 지원하지 않습니다.

###### 14.5.1 템플릿

rmarkdown 확장의 첫 번째 유형은 새로운 템플릿을 정의하는 것입니다. 우리는 방금 Tufte 핸드아웃의 예를 살펴봤고 앞의 14.3.1절에서 HTML 문서 출력에 대한 예도 보여주었습니다.

https://github.com/jgm/pandoc-templates 저장소에는 Pandoc에서 사용하는 모든 템플릿이 포함되어 있으며, rmarkdown 소스 패키지(https://github.com/rstudio/rmarkdown)에서 사용자 지정 템플릿을 살펴볼 수도 있습니다. 템플릿 변수 중에 이해가 가지 않는 것이 있으면 http://johnmacfarlane.net/pandoc/ 에 있는 문서를 확인해 보시기 바랍니다.

템플릿을 다른 사용자와 공유하는 가장 쉬운 방법은 이를 R 패키지의 inst/rmarkdown/templates/ 디렉터리에 넣는 것입니다. 새로운 디렉터리(my_template)를 만들고 그 안에 템플릿 파일을 넣을 수 있습니다. 템플릿에는 CSS/JavaScript 파일이나 LATEX 패키지와 같은 특정 종속성이 필요할 수 있습니다. 이러한 파일들은 my_template 아래의 하위 디렉터리인 skeleton/에 수집할 수 있습니다. skeleton/ 디렉터리에는 샘플 Rmd 파일 skeleton.Rmd도 제공할 수 있습니다. 마지막으로 my_template 아래의 YAML 파일인 template.yaml에 3개의 YAML 필드를 사용하여 템플릿을 설명할 수 있습니다.

name 템플릿의 이름("Journal of Statistical Software")
description 템플릿에 대한 간단한 설명("JSS 논문을 위한 템플릿입니다")
create_dir yes 또는 no, 혹은 true 또는 false(곧 설명함)

이런 형태의 R 패키지(예를 들어 이름이 myPackage)를 설치했다면 draft() 함수를 사용하여 템플릿에서 새 초안을 만들 수 있습니다.

rmarkdown::draft("my_article.Rmd", template = "my_template", package = "myPackage")

이 함수는 myPackage에서 my_template 템플릿을 찾고, skeleton.Rmd를 현재 작업 디렉터리에 my_article.Rmd로 복사하며 종속성도 함께 복사합니다. 위에서 언급한 YAML 옵션 create_dir은 초안 my_article.Rmd를 위해 새 디렉터리를 만들지 여부를 결정합니다.

RStudio는 이 과정을 훨씬 더 쉽게 만들었습니다. 메뉴 File New File R Markdown에서 로컬에 설치된 모든 패키지의 모든 템플릿을 볼 수 있습니다(그림 14.14).

rticles 패키지(https://github.com/rstudio/rticles)는 여러 LATEX 문서 클래스에 대한 템플릿 모음입니다. 이 템플릿을 사용하여 Journal of Statistical Software, The R Journal 등에 제출할 논문을 R 마크다운으로 작성할 수 있습니다.

###### 14.5.2 새 형식

rmarkdown 확장의 두 번째 유형은 새로운 출력 형식입니다. 새로운 형식은 기존 출력 형식을 기반으로 할 수도 있고 완전히 새로운 형식일 수도 있습니다. 전자는 매우 쉽습니다. 기존 출력 형식 함수에서 일부 옵션을 수정하여 출력 형식 객체를 반환하는 R 함수를 정의하기만 하면 됩니다. 최소한의 예시로, 아래와 같이 toc 인수의 기본값을 FALSE에서 TRUE로 바꾸는 html_toc 함수를 생성해 봅니다.

- 그림 14.14: 템플릿에서 새 R 마크다운 문서 만들기: 목록에서 템플릿을 선택할 수 있습니다.

html_toc <- function(toc = TRUE, ...) {

rmarkdown::html_document(toc = toc, ...) }

새 형식 함수는 (여전히 패키지 이름이 myPackage라고 가정할 때) R 패키지에 넣어야 하며, 그런 다음 YAML에서 사용할 수 있습니다. 다음은 두 가지 예입니다.

--output: myPackage::html_toc

---

--output:

myPackage::html_toc: toc: no self_contained: no

---

- 그림 14.15: R 마크다운에서 E-book 만들기: 이 그림은 FBReader(무료 E-book 리더)에 표시된 EPUB 책의 제목 페이지를 보여줍니다.

두 번째 예의 경우, 이 Rmd 파일을 렌더링할 때 다음이 호출됩니다. rmarkdown::render("foo.Rmd", myPackage::html_doc(toc = FALSE,

self_contained = FALSE)) # 즉 본질적으로는 다음을 의미합니다 render('foo.Rmd', # html_document(toc = FALSE, self_contained = FALSE))

14.3.1절에서 설명했듯이 출력 형식은 knitr 옵션, Pandoc 옵션, rmarkdown 옵션 등 세 가지 옵션으로 이루어진 목록입니다. 위의 최소 예제에서는 Pandoc toc를 사용자 지정했지만 출력 형식 함수에서 더 많은 옵션을 원하는 대로 사용자 지정할 수 있습니다. rmarkdown에는 출력 형식을 구성할 때 사용할 수 있는 output_format(), knitr_options() 및 pandoc_options() 등 몇 가지 도움 함수가 있습니다. reveal.js(HTML5 프레젠테이션 형식)용 새 형식을 만드는 방법에 대한 예시는 https://github.com/jjallaire/revealjs 저장소를 참조하십시오. 아래에서는 EPUB(E-book 형식)용 출력을 만드는 방법에 대한 최소한의 예를 보여줍니다.

# @importFrom rmarkdown output_format # @importFrom rmarkdown knitr_options # @importFrom rmarkdown pandoc_options

epub_book <- function(to = c("epub", "epub3")) { to <- match.arg(to) optk <- knitr_options() optp <- pandoc_options(to, ext = ".epub") output_format(knitr = optk, pandoc = optp)

}

이 함수를 myPackage 패키지에 넣으면 R 마크다운에서 E-book을 만들 수 있습니다. 다음은 최소한의 R 마크다운 예제입니다(그림 14.15):

--title: "R 마크다운 v2 데모" author:

- 리 레이
- 한 메이메이

date: "2015/01/01" output: myPackage::epub_book

- --# 멋진 섹션으로 시작

  {r} 1 + 1

형식 함수 epub_book()의 핵심은 pandoc_options()의 to 인수를 epub 또는 epub3로 지정하는 것이었습니다. Pandoc은 매우 다양한 문서 형식을 지원하며 rmarkdown은 그중 일부만 포함하고 있습니다. 여러분은 위에서 소개한 접근 방식을 사용하여 자신만의 형식 함수를 만들 수 있습니다.

###### 14.5.3 HTML 위젯

14.3.1절에서 YAML 메타데이터의 includes 옵션을 설명했습니다. HTML 문서 출력에 자바스크립트 라이브러리를 포함하려는 경우 includes 옵션을 사용할 수 있습니다. 이 접근 방식에는 두 가지 단점이 있습니다.

1. 휴대성(이동성)이 부족합니다. 다른 사람과 R 마크다운 문서를 공유할 때 includes 옵션에 지정된 종속성을 항상 함께 복사해야 하기 때문입니다. 이 방식은 다른 사람들이 여러분의 종속성을 재사용하기에도 불편합니다.
2. 자바스크립트 라이브러리를 호출하려면 R 마크다운에 자바스크립트 코드(때로는 매우 많은 양의 코드)를 작성해야 합니다. 하지만 모든 R 사용자가 자바스크립트에 익숙한 것은 아니기 때문에 R 마크다운 문서를 온전히 다루지 못할 수 있습니다.

HTML 위젯의 핵심 아이디어는 자바스크립트 라이브러리에 대한 네이티브 R 인터페이스를 제공하는 것입니다. 이를 통해 자바스크립트를 이해하지 못하는 사람이라도 기본 종속성이나 자바스크립트 구문에 대해 걱정할 필요 없이 라이브러리를 사용할 수 있습니다. 자바스크립트 라이브러리를 사용하여 플롯을 그릴 때는, 코드 청크 내에서 R 함수를 호출하기만 하면 됩니다.

htmlwidgets 패키지(Vaidyanathan 등, 2014)는 패키지 개발자가 자바스크립트 라이브러리를 R로 쉽게 이식할 수 있도록 설계되었습니다. http://www.htmlwidgets.org 에 설명서가 잘 정리되어 있으며, 웹사이트에서 여러 예제 패키지도 볼 수 있습니다. 여기서는 기술적인 세부 사항은 설명하지 않고 HTML 위젯의 작동 방식을 보여주는 짧은 예제만 소개하겠습니다. 다음은 R 마크다운 최소 예제입니다(이 예제를 시도하기 전에 https://github.com/rstudio/DT 에서 DT 패키지를 설치해야 합니다):

--title: "R 마크다운 v2 데모" author:

- 리 레이
- 한 메이메이

date: "2015/01/01" output: html_document

- --이 표는 DataTables 라이브러리를 통해 생성되었습니다.

  {r} DT::datatable(iris)

그림 14.16은 그 출력을 보여줍니다. DT 패키지는 자바스크립트 라이브러리 DataTables(http://datatables.net)에 대한 인터페이스입니다. 보시다시피 R 마크다운 소스 문서는 매우 단순하며 자바스크립트 파일이나 자바스크립트 코드가 전혀 보이지 않습니다. datatable() 함수를 호출하기만 하면 데이터 프레임이 DataTables를 통해 표시됩니다. 데이터를 HTML 페이지로 전달하고 이를 구문 분석 및 렌더링하는 어려운 작업은 패키지 작성자에 의해 이미 완료되었으므로 사용자는 모든 기본 기술 세부 사항을 이해할 필요가 없습니다.

###### 14.6 R 마크다운 v1에서 v2로 변경된 사항

v1 시절에 R 마크다운을 사용하기 시작했다면, v1에서 v2로 전환할 때 다음과 같은 변경 사항을 알아두어야 합니다.

- 그림 14.16: R 마크다운의 DataTables 라이브러리에 의해 생성된 표: 열을 정렬하고, 표에서 검색할 수 있으며, 전체 표를 여러 페이지에 걸쳐 표시할 수 있습니다.

- v2에서는 기본적으로 knitr 패키지가 로드되지 않습니다(엄밀히 말하면 attach되지 않습니다). 즉, library(knitr) 명령 등을 통해 명시적으로 패키지를 로드하지 않는 한 knitr 패키지의 함수와 객체를 사용할 수 없습니다. 그렇지 않으면 "'opts_chunk' 객체를 찾을 수 없습니다(object 'opts_chunk' not found)"와 같은 오류가 발생할 수 있습니다.
- Rmd 파일을 렌더링할 때 청크 옵션 fig.path(그림 경로)와 cache.path(캐시 경로)가 rmarkdown에서 수정됩니다. knitr에서는 각각 figure/와 cache/입니다. 현재 rmarkdown에서는 각각 foo_files/figure-format/과 foo_files/cache-format/이며, 여기서 foo는 파일 확장자가 없는 입력 Rmd 파일의 기본 이름이고 format은 출력 형식(tex 또는 html)입니다.
- 청크 옵션 error가 TRUE에서 FALSE로 변경되었습니다. 이는 R 마크다운 출력 문서에 오류 메시지를 표시하는 대신 기본적으로 R이 중지됨을 의미합니다(6.2.4절 참조).
- 출력 형식에 따라 청크 옵션 fig.width, fig.height, fig.retina가 각기 다른 값을 가질 수 있습니다. 출력 형식 함수에 대한 rmarkdown 설명서를 확인하거나, R 마크다운 문서에 str(knitr::opts_chunk$get())을 인쇄하여 청크 옵션의 값을 확인할 수 있습니다.

### 15

###### 응용

지금까지는 단순성을 위해 짧은 예제들로 knitr의 사용법을 소개해 왔습니다. 이 장에서는 구체적이고 완전한 예제 몇 가지를 사용하여 knitr가 실제 응용 프로그램과 어떻게 작동하는지 보여줍니다. 이러한 애플리케이션의 모든 세부 사항을 일일이 설명하지는 않으며, 그중 중요한 부분만을 짚고 넘어가겠습니다.

###### 15.1 과제

과제를 작성할 때 R 마크다운은 그 단순성 때문에 선호되는 문서 형식일 수 있으며, 일반적으로 과제는 출판을 목적으로 하지 않습니다. 앞서 언급했듯이 RPubs(http://rpubs.com)는 knitr에 의해 RStudio에서 생성된 (HTML) 보고서를 공유하기 위한 플랫폼입니다. 이곳에 수많은 과제들도 제출되어 있습니다.

과제 보고서는 비교적 단순하기 때문에 너무 많은 knitr 기능이 필요하지 않을 수 있습니다. 과제에 사용되는 몇 가지 공통 기능은 다음과 같습니다. 플롯 크기 설정(fig.width 및 fig.height), 채점자가 읽고 싶어하지 않을 수 있으므로 소스 코드 숨기기(echo = FALSE), 시간이 많이 소요되는 계산 작업에 캐시 활성화(cache = TRUE) 등입니다. 기본적으로 제공되는 tidy = TRUE 및 highlight = TRUE와 같은 다른 기능은 코딩 스타일을 신경 쓰지 않는 사용자가 출력 문서에서 더 읽기 쉬운 코드를 작성하는 데 도움이 될 수 있습니다.

이제 깁스 샘플링(Gibbs sampling)의 예를 보여드리겠습니다. 이변량 정규 분포에 대해

$\begin{pmatrix} X \\ Y \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} \mu_X \\ \mu_Y \end{pmatrix}, \begin{pmatrix} \sigma_X^2 & \rho\sigma_X\sigma_Y \\ \rho\sigma_X\sigma_Y & \sigma_Y^2 \end{pmatrix} \right),$

(15.1)

조건부 분포는 다음과 같습니다.

$X|Y = y \sim \mathcal{N} \left( \mu_X + \frac{\sigma_X}{\sigma_Y}\rho(y - \mu_Y), (1 - \rho^2)\sigma_X^2 \right)$
$Y|X = x \sim \mathcal{N} \left( \mu_Y + \frac{\sigma_Y}{\sigma_X}\rho(x - \mu_X), (1 - \rho^2)\sigma_Y^2 \right)$

(15.2)

따라서 우리는 깁스 샘플링을 사용하여 결합 정규 분포(joint Normal distribution)에서 난수를 생성할 수 있습니다. 먼저 x(0)과 y(0)을 초기화한 다음 $x^{(k)} \sim f(x|y^{(k-1)})$와 $y^{(k)} \sim f(y|x^{(k)})$를 반복적으로 생성합니다. 아래의 R 코드는 식 15.2를 변환한 것입니다.

rbinormal <- function(n, mu1, mu2, sigma1, sigma2, rho) { # 초기화

- x <- rnorm(1, mu1, sigma1)
- y <- rnorm(1, mu2, sigma2) xy <- matrix(nrow = n, ncol = 2, dimnames = list(NULL,

c("X", "Y"))) # 조건부 분포로부터의 샘플링 for (i in 1:n) {

- x <- rnorm(1, mu1 + sigma1/sigma2 _ rho _ (y - mu2),

- sqrt(1 - rho^2) \* sigma1)

y <- rnorm(1, mu2 + sigma2/sigma1 _ rho _ (x - mu1),

- sqrt(1 - rho^2) \* sigma2)

xy[i, ] <- c(x, y)

} xy

}

그림 15.1은 $\mu_X = 0, \sigma_X = 2, \mu_Y = 1, \sigma_Y = 3, \rho = 0.7$인 이변량 정규 분포에 대한 깁스 샘플링의 처음 20단계를 보여줍니다.

set.seed(123) n <- 20 z <- rbinormal(n, mu1 = 0, mu2 = 1, sigma1 = 2, sigma2 = 3,

rho = 0.7) plot(z, pch = 19) arrows(z[-n, 1], z[-n, 2], z[-1, 1], z[-1, 2], length = 0.15,

col = "gray40")

그리고 우리는 일부 샘플을 추출할 수도 있습니다.

z <- rbinormal(5000, 0, 1, 2, 3, 0.7) smoothScatter(z, nbin = 64) points(0, 1, col = "white", pch = 19) # 이론적 평균

그림 15.2는 이 분포의 샘플 5,000개를 보여주며 우리는 해당 이론값에 가까워야 하는 표본 평균, 표준 편차 및 상관 관계를 계산할 수 있습니다.

- 그림 15.1: 이변량 정규 분포에 대한 깁스 샘플링 흔적: 화살표는 깁스 샘플링의 처음 20단계를 보여줍니다.

- 그림 15.2: 깁스 샘플링의 5,000개 포인트: 평활화된 산점도는 2D 분포의 밀도를 보여줍니다.

apply(z, 2, mean) # 표본 평균

## X Y ## 0.001287 0.971010

apply(z, 2, sd) # 표본 편차

## X Y ## 1.973 2.971

cor(z) # 표본 상관 관계

## X Y ## X 1.0000 0.6948 ## Y 0.6948 1.0000

이 소규모 애플리케이션에서는 캐시(이 특정 예제는 너무 느리지는 않지만)와 TikZ 그래픽을 사용했습니다. 또한 플롯 크기를 조절했습니다(그림 15.1의 경우 5 × 3, 그림 15.2의 경우 5 × 4). 이야기 서술과 코드 청크가 직조되어 있어 독자가 동일한 보고서에서 이론을 배우고 컴퓨터 연산을 확인하며 결과를 검증할 수 있습니다. 모든 것이 투명하고 오류를 쉽게 찾을 수 있을 것입니다. 때때로 우리가 작성하는 컴퓨터 코드가 우리가 언급한 이론을 실제로 반영하지 못할 수 있으며, 연산과 보고서를 분리해 놓을 경우 이러한 오류를 찾기가 어려워집니다.

데이터, 코드 및 소프트웨어 공유의 측면에서, 출판물을 공유하고 연구를 완전히 재현 가능하게 만드는 것에 있어 순전히 선의와 자기 통제에만 의존할 수는 없습니다.

Huang 및 Gottardo (2013) 생체의학 데이터의 비교 가능성과 재현 가능성

재현 가능한 연구를 위해 사람들은 데이터 분석에서 데이터, 코드 및 소프트웨어를 공유할 것을 제안해 왔습니다(Huang 및 Gottardo (2013)). 우리는 교육 분야에서 더 많은 노력이 필요하다고 믿으며 재현 가능한 과제부터 그 시작을 도모할 수 있습니다.

###### 15.2 동적 문서 제공하기(서빙)

servr 패키지(Xie, 2015c)는 httpuv 패키지를 기반으로 주어진 디렉터리 아래의 파일을 제공하는 몇 가지 간단한 HTTP 서버 기능을 갖추고 있습니다. 파이썬에 익숙하다면 어느 정도 python -m SimpleHTTPServer 또는 python -m http.server와 비슷하게 느껴질 것입니다. 원래는 디렉터리 아래의 정적 파일을 제공하도록 설계되었으며 기본 기능은 httd()였습니다.

servr::httd("./")

R 콘솔에서 위 함수를 실행하면 R이 웹 브라우저를 띄워 현재 작업 디렉터리(./) 아래의 파일 목록을 표시하거나, index.html 파일이 있으면 이를 표시합니다. 파일의 링크를 클릭하여 해당 내용을 볼 수 있습니다.

나중에 servr는 knitr 및 rmarkdown을 기반으로 확장되어 동적 R 마크다운 문서도 제공할 수 있게 되었습니다. 이 패키지에는 (knitr 또는 rmarkdown을 통해) R 마크다운 문서에서 생성된 HTML 파일을 제공하는 jekyll(), rmdv1() 및 rmdv2() 함수가 있습니다. HTML 출력 파일이 해당 소스 파일보다 이전 버전인 경우 R 마크다운 문서가 자동으로 다시 컴파일될 수 있으며 웹 브라우저의 HTML 페이지가 그에 따라 자동으로 새로고침될 수 있습니다. 따라서 사용자는 R 마크다운 문서 작성에만 집중할 수 있고 결과는 웹 브라우저에서 실시간으로 업데이트됩니다. 이로써 'Knit HTML 버튼 클릭하기'와 '웹 브라우저 새로고침하기'라는 두 단계를 절약할 수 있습니다. 보고서를 작성할 때 이 두 가지 단계는 주의를 산만하게 할 수 있습니다. servr를 사용하면 서버를 시작한 후 R 마크다운 문서를 작성하기만 하면 됩니다.

RStudio IDE에서 R 마크다운 문서를 작성할 때 이 기능은 훨씬 더 유용합니다. servr는 RStudio IDE를 감지하면 기본적으로 RStudio 뷰어를 웹 브라우저로 설정하기 때문에 그림 15.3의 레이아웃처럼 소스 문서와 그 출력을 나란히 배치할 수 있기 때문입니다. RStudio를 사용하지 않아도 전혀 문제가 없습니다. 다른 편집기나 웹 브라우저를 사용할 때도 자동 컴파일과 새로고침 기능이 동일하게 작동합니다.

rmdv1() 및 rmdv2() 함수는 각각 R 마크다운 v1 및 v2에 해당합니다. R 콘솔에서 servr::rmdv1() 또는 servr::rmdv2()를 호출한 후 foo.html 파일의 소스 문서 foo.Rmd가 있다면 HTML 파일 foo.html을 클릭하여 HTML 출력을 볼 수 있습니다. 그 다음 foo.Rmd를 편집하고 저장할 때마다 servr가 이를 자동으로 다시 컴파일하고 HTML 출력 페이지를 새로고침합니다.

jekyll() 함수는 rmdv1() 및 rmdv2()와 유사하지만 Jekyll 웹사이트에 맞춤화되어 있습니다. Jekyll에 대해서는 13.4절에서 간략하게 소개했습니다.

- 그림 15.3: R 마크다운 문서(왼쪽 위 패널)와 RStudio 뷰어의 해당 출력(오른쪽 패널)의 레이아웃: R 콘솔(왼쪽 아래)에 servr 함수를 입력했고 RStudio 뷰어에 R 마크다운의 출력이 표시되었습니다. 이 그림은 단지 설명을 위한 목적입니다. 포함된 텍스트를 읽고 싶다면 https://github.com/yihui/servr 에서 원본 이미지를 확인하십시오.

R 마크다운 게시물이나 페이지를 마크다운으로 반복해서 계속 컴파일하는 것은 지루한 작업이므로 이 부분에서 jekyll()이 유용할 수 있습니다. Jekyll 웹사이트의 루트 디렉터리에서 servr::jekyll() 함수를 호출하면 웹 브라우저에서 웹사이트의 미리보기를 얻을 수 있습니다. 게다가 블로그 게시물을 편집하고 저장하면 웹 브라우저가 페이지를 새로고침하여 업데이트된 출력을 표시합니다. knitr-jekyll 저장소(https://github.com/yihui/knitr-jekyll)는 servr를 사용하여 Jekyll 웹사이트를 서빙하는 예시를 담고 있습니다.

나중에 15.4절에서 패키지 비네트를 소개할 텐데 R 패키지를 개발할 때 servr의 vign() 함수를 사용하여 HTML 비네트를 제공할 수 있습니다. 이것의 장점은 비네트를 서빙할 때 HTML 출력 파일을 소스 패키지에 보존하지 않아 소스 패키지를 깔끔하게 유지한다는 것입니다.

기술적 세부 사항이 궁금한 분들을 위해 덧붙이자면, 이 구현은 WebSockets를 기반으로 합니다. servr는 HTML 페이지를 표시할 때 그 안에 자바스크립트 코드 조각도 함께 주입하여 주기적으로(1초 간격으로) R과 통신하는 WebSocket 연결을 설정합니다. R은 WebSocket에서 요청을 받을 때마다 Rmd 파일의 타임스탬프를 HTML 출력 파일의 타임스탬프와 비교합니다. Rmd 파일이 HTML 출력보다 최신 파일인 경우 servr는 knitr 또는 rmarkdown을 호출하여 Rmd 파일을 HTML로 다시 컴파일한 다음 WebSocket에 메시지를 다시 보냅니다.

all: example.html %.html: %.Rmd

Rscript -e "rmarkdown::render('$^')"

- 그림 15.4: servr의 make() 함수를 위한 Makefile 예제: 생성할 HTML 파일은 target all에 지정되고 rmarkdown을 통해 Rmd 파일에서 HTML 파일을 생성하는 규칙이 명시되어 있습니다.

WebSocket이 이 메시지를 받으면 자바스크립트에서 location.reload()를 호출하여 페이지를 새로고침합니다.

이 과정에서 중요한 단계는 Rmd 파일을 다시 컴파일해야 하는지 여부를 확인하는 것입니다. 이는 GNU Make(http://www.gnu.org/software/make/)가 잘 수행하는 작업이므로 servr는 필요할 때 사용자가 고유한 Makefile을 제공하여 Rmd 파일을 재빌드할 수 있도록 make() 함수도 제공했습니다. 그림 15.4는 make() 함수의 예제 Makefile입니다.

기본적으로 서버 기능은 현재 R 세션을 차단하며, 이는 동일한 R 세션에서 계속 작업하려는 경우 문제가 될 수 있습니다. 이 문제를 해결하려면 서버 함수에 대해 daemon = TRUE 인수(httd(daemon = TRUE) 또는 rmdv2(daemon = TRUE))를 사용할 수 있습니다. 이는 servr에 현재 R 세션을 차단하지 않는 데몬화된 서버를 실행하도록 지시합니다.

###### 15.3 웹사이트 및 블로깅

이 절에서는 knitr를 기반으로 구축된 몇몇 웹사이트와 블로그를 소개합니다. 그리고 웹 페이지는 R 마크다운이나 R HTML로 작성되었습니다.

###### 15.3.1 Vistat 및 Rcpp Gallery

Vistat(http://vis.supstat.com)은 R 마크다운과 Jekyll(13.4절) 기반의 웹사이트입니다. 이 웹사이트의 목표는 재현 가능한 통계 그래픽의 갤러리를 제공하는 것입니다. 웹사이트용 저장소는 Github(https://github.com/supstat/vistat)에서 공개적으로 사용할 수 있습니다.

이 저장소의 핵심은 일부 전역 청크 옵션을 설정하고 Rmd 문서를 마크다운 출력으로 컴파일하는 R 스크립트 ./\_bin/knit입니다. 수학 수식은 MathJax에 의해 렌더링되며 애니메이션은 SciAnimator 라이브러리(7.3.1절)를 통해 지원되고 D3 라이브러리를 통해 웹 그래픽을 만들 수도 있습니다.

knitr가 Rmd 소스 파일을 마크다운 파일로 컴파일하고 나면, Jekyll이 마크다운을 HTML로 컴파일하여 웹사이트를 제공할 수 있습니다.

Rcpp Gallery(http://gallery.rcpp.org)는 Rcpp(Eddelbuettel 등, 2015) 기사와 예제를 위한 웹사이트이며 역시 R 마크다운을 기반으로 구축되었습니다. 특히 knitr의 Rcpp 엔진(11.2.1절)을 사용합니다.

###### 15.3.2 UCLA R 자습서

UCLA 통계 컨설팅 그룹(UCLA Statistical Consulting Group)은 수년간 여러 통계 패키지에 대한 소프트웨어 튜토리얼을 유지해 왔으며 그중 하나는 R 전용(http://www.ats.ucla.edu/stat/r/)입니다. 2012년 이전에 이 웹사이트는 복사 및 붙여넣기로 만들어졌습니다. 즉, 결과물을 R에서 생성한 후 HTML 페이지에 복사하는 식이었습니다. 2012년 knitr가 출시된 후 웹 관리자 중 한 명인 Joshua Wiley는 R HTML 형식 대신 knitr를 사용하여 R 자습서 페이지를 다시 작성하기로 결정했습니다. 이제 웹 페이지를 유지 관리하기가 훨씬 쉬워졌고 R 출력은 훨씬 더 나은 재현성을 확보했습니다. R이 업데이트되거나 데이터 세트가 변경된 후에 모든 소스 문서를 다시 컴파일하면 전체 웹사이트를 자동으로 다시 빌드할 수 있습니다.

###### 15.3.3 cda 및 RHadoop Wiki

Github에는 각 저장소에 대한 통합 Wiki 시스템이 있습니다. 마크다운 및 reStructuredText 등과 같은 다양한 형식으로 위키 페이지를 작성할 수 있습니다. 각 페이지는 본질적으로 파일이며 위키는 본질적으로 Git 저장소입니다. 따라서 Rmd 파일을 작성하여 마크다운 파일로 컴파일한 다음, Git을 통해 Github에 푸시할 수 있습니다.

cda 패키지(Auguie, 2013)는 위와 같은 방식을 사용하여 Github에 위키 사이트(https://github.com/baptiste/cda/wiki)를 구축했습니다. 패키지의 wiki 디렉터리 아래에서 Rmd 소스 파일을 찾을 수 있습니다.

RHadoop 프로젝트도 https://github.com/RevolutionAnalytics/RHadoop/wiki 에 이와 유사한 위키를 가지고 있습니다.

###### 15.3.4 ggbio 패키지

ggbio 패키지(Yin 등, 2012)는 ggplot2 패키지를 기반으로 유전체 데이터에 대한 그래픽 문법(Grammar of Graphics)을 확장하기 위한 R 구현체입니다. http://tengfei.github.com/ggbio/ 라는 자체 웹사이트가 있으며 여기에서 관련 문서를 확인할 수 있습니다. knit_rd() 함수(12.4.8절)를 사용하여 R 설명서 페이지를 HTML로 컴파일함으로써 예제의 출력을 직접 볼 수 있게 했습니다. 이 패키지가 한 번 설치되면 단 한 줄의 코드로 HTML 페이지를 얻을 수 있습니다.

knitr::knit_rd("ggbio")

그런 다음 HTML 파일을 Github에 게시할 수 있습니다. 이미지들은 HTML 파일 내에 base64로 인코딩되어 있으므로 별도의 이미지 파일을 다룰 필요가 없습니다.

참고로 ggbio 패키지에는 knitr로 작성된 PDF 비네트도 있습니다. 이는 웹사이트에서 확인하거나 다음 명령어로 찾아볼 수 있습니다. vignette("ggbio", package = "ggbio")

###### 15.3.5 Geospatial Data in R and Beyond

Barry Rowlingson은 useR! 2012 컨퍼런스에서 R을 사용한 지리공간(Geospatial) 데이터 분석에 대한 튜토리얼 워크숍을 진행했습니다. 해당 웹사이트는 http://www.maths.lancs.ac.uk/~rowlings/Teaching/UseR2012/ 입니다. 이 웹사이트는 R HTML 파일로 만들어졌으며 트위터 부트스트랩(유명한 CSS 프레임워크)의 멋진 스타일을 차용했습니다. R HTML을 R 마크다운보다 사용할 때의 장점은 스타일을 완전히 제어할 수 있다는 것입니다. 이 웹사이트는 사용자 지정 CSS 스타일을 갖춘 div 요소들 안에 R 코드 청크와 출력을 정렬해 놓은 좋은 예입니다.

###### 15.4 패키지 비네트

Gentleman과 Temple Lang (2004)이 논의했듯이 R 패키지는 컴퓨팅 루틴을 제공하는 명백한 기능을 너머, 재현 가능한 보고서를 작성하고 배포할 수 있는 커다란 잠재력을 가지고 있습니다. 특히 R 패키지 비네트는 기능, 단위 테스트 및 데이터 세트와 같이 인프라를 제공하는 패키지의 다른 구성 요소와 함께 재현 가능한 보고서를 작성하는 데 있어 이상적인 형식이 될 수 있습니다. R 패키지 비네트는 논문과 같으며, 출력 파일은 패키지 빌드 과정(즉 R CMD build) 중에 소스 문서에서 동적으로 컴파일됩니다.

3.0.0 버전 미만의 R에서는 Sweave를 사용하여 패키지 비네트를 빌드했습니다. Sweave의 한계(16.1절)와 LATEX의 장벽 때문에 R 패키지 비네트는 R 3.0.0 이전에는 널리 사용되지 않았습니다. 하지만 BioConductor는 예외였는데, 이 시스템에서는 비네트 작성이 의무적이었기 때문입니다.

R 3.0.0 이후로 패키지 비네트의 컴파일은 훨씬 자연스럽고 쉬워졌습니다. 이는 Henrik Bengtsson, Duncan Murdoch, 그리고 R 핵심 개발자들의 노고 덕분입니다. 현재 CRAN의 약 300개 패키지에는 knitr로 컴파일된 500개 이상의 패키지 비네트가 등록되어 있습니다(https://gist.github.com/yihui/7698648). 다음 절에서는 knitr 비네트 엔진에 대해 소개하고 몇 가지 예시를 살펴보겠습니다. 15.4.3절과 15.4.4절은 구버전 R에 관심 있는 독자만을 위한 것이므로, 그곳에서 언급된 요령들은 더 이상 사용할 것을 권장하지 않습니다.

###### 15.4.1 비네트 메타데이터와 엔진

knitr를 사용하여 비네트를 빌드하려면 다음과 같은 간단한 단계를 따르기만 하면 됩니다.

- 비네트 소스 문서(Rnw 또는 Rmd 파일)에 %\VignetteEngine{knitr::knitr}와 같은 비네트 엔진을 지정합니다.
- 패키지 DESCRIPTION 파일에 VignetteBuilder: knitr 필드를 추가합니다.
- DESCRIPTION의 Suggests 필드에 knitr를 추가합니다.

그런 다음 knitr 구문(코드 청크의 경우 < <> >= 또는 {r})을 사용하여 비네트를 작성할 수 있습니다. 비네트는 패키지 루트의 vignettes/ 디렉터리에 위치해야 함을 명심하십시오.

R 매뉴얼인 "Writing R Extensions"에 따라 \VignetteIndexEntry{}에 비네트 제목도 적어야 합니다. \VignetteKeyword{}와 같은 다른 몇 가지 선택적인 메타데이터 사양도 존재합니다. knitr의 R 마크다운 v2 비네트에 대한 비네트 메타데이터(제목 및 비네트 엔진) 예제는 그림 15.5를 참조하십시오. 패키지를 빌드하고 나면 해당 비네트는 HTML 인덱스 페이지에 나열됩니다.

knitr 패키지에는 이런 식으로 컴파일된 여러 PDF 및 HTML 비네트가 있으며, 다음 명령을 실행하여 볼 수 있습니다. browseVignettes(package = "knitr") # 파일 이름을 아는 경우 특정 비네트 보기 vignette("knitr-intro", package = "knitr") vignette("knitr-refcard", package = "knitr")

비네트 엔진 knitr::knitr는 knitr에서 사용 가능한 여러 엔진 중 하나일 뿐입니다. 모든 엔진을 확인하려면 tools 패키지의 vignetteEngine() 함수를 사용할 수 있습니다.

--title: "knitr 소개가 아닙니다" author: "Yihui Xie" date: "`r Sys.Date()`" bibliography:

- ../inst/examples/knitr-packages.bib

- ../inst/examples/knitr-manual.bib

vignette: > %\VignetteEngine{knitr::rmarkdown} %\VignetteIndexEntry{knitr 소개가 아님}

output: knitr:::html_vignette

---

- 그림 15.5: knitr 비네트의 메타데이터: 이것은 knitr 비네트에서 추출한 것이며 system.file('doc', 'knitr-intro.Rmd', package='knitr')에서 찾을 수 있습니다.

library(knitr) sort(names(tools::vignetteEngine(package = "knitr")))

## [1] "knitr::docco_classic" ## [2] "knitr::docco_classic_notangle" ## [3] "knitr::docco_linear" ## [4] "knitr::docco_linear_notangle" ## [5] "knitr::knitr" ## [6] "knitr::knitr_notangle" ## [7] "knitr::rmarkdown" ## [8] "knitr::rmarkdown_notangle"

접미사 \_notangle이 있는 엔진은 접미사가 없는 엔진과 같은 weave 기능을 가지고 있지만 tangle 기능이 비활성화되어 있습니다. 즉 R CMD build 또는 R CMD check 중에 비네트에서 R 스크립트가 생성되지 않음을 의미합니다. 코드가 위빙(weave)에서 이미 실행되었기 때문에 R CMD check에서 동일한 코드를 다시 실행하는 것이 중복되기도 하고, 현재 인라인 R 코드 표현식이 탱글(tangle) 출력에 포함되지 않아 문제를 일으킬 수도 있기 때문에 가끔 우리는 비네트에서 R 스크립트를 탱글하고 싶지 않을 수 있습니다.

비네트 엔진 내에서 :: 연산자는 특별한 의미가 없다는 점에 유의하십시오. base R에서 패키지의 내보낸 객체를 가져오는 연산자가 ::(stats::lm)이므로 혼동을 줄 수 있습니다. 하지만 비네트 엔진 표기법에서 ::는 패키지 이름과 엔진 이름을 구분하는 구분 기호에 불과합니다. 따라서 knitr::rmarkdown은 rmarkdown이 knitr 내의 함수라는 의미가 아니라 knitr의 비네트 엔진 중 하나일 뿐임을 의미합니다.

rmarkdown 비네트 엔진을 사용할 때는 현재 R이 이 두 가지 유형의 비네트 출력만 인식하기 때문에, 파일 확장자가 .html 또는 .pdf이기만 하다면 어떠한 출력 형식을 선택하든 자유롭습니다. 출력 형식이 HTML인 경우 HTML 문서이거나 HTML5 프레젠테이션(ioslides 또는 Slidy) 중 하나일 수 있습니다. PDF인 경우에는 PDF 문서나 Beamer 슬라이드가 될 수 있습니다.

###### 15.4.2 비네트 예제

https://gist.github.com/yihui/7698648 에 knitr 비네트 엔진을 사용하는 현재 CRAN 패키지들의 비네트 목록을 정리해 두었으니 이 예제들을 통해 참고해 볼 수 있습니다.

Murphy (2012)의 ggplot2 transition guide(전환 가이드)는 ggplot2 패키지와 함께 제공되지는 않지만, R 패키지 비네트의 훌륭한 예입니다. 이 가이드는 새로운 기능을 알리고 이전 버전의 사용자에게 영향을 미칠 수 있는 ggplot2 0.9.0의 변경 사항을 설명하기 위해 작성되었습니다.

이 가이드의 좋은 기능 중 하나는 Rnw 문서를 컬러나 흑백 버전으로 컴파일할 수 있다는 점입니다. 이는 전역 변수 bw_version에 의해 제어되는데, 값이 TRUE이면 흑백 버전이 생성됩니다. 이는 흑백 플롯을 생성하는 청크(ggplot2의 경우 theme_bw() 및 scale_fill_gray()와 같은 회색조)에 대해 청크 옵션 eval = bw_version 및 echo = bw_version을 설정하여 달성할 수 있습니다. bw_version이 FALSE인 경우 이러한 청크는 출력에서 숨겨집니다(소스 코드는 평가되지 않고 메아리치지 않습니다). 마찬가지로 옵션 eval = !bw_version 및 echo = !bw_version을 가진 다른 청크들이 있으며 이 청크들은 컬러 플롯을 생성합니다. 요약하자면 단일 변수를 통해 PDF 출력을 컬러 또는 흑백으로 설정할지 제어할 수 있으며 이는 매우 편리합니다(5.1.1절 참고). 그림 15.6은 컬러 버전 전환 가이드의 샘플 페이지입니다.

corrplot 패키지(Wei, 2013)에는 HTML 비네트의 예제가 있습니다. Github의 https://github.com/taiyun/corrplot/tree/master/vignettes 에서 그 비네트 소스 문서를 찾을 수 있습니다. 분명히 그것은 Rmd 문서입니다(5.2.1절). 여기서 R 마크다운 v1을 사용한다는 점에 유의하십시오. 텍스트 편집기(RStudio)로 이 문서를 열면 그 안에 있는 R 코드 청크를 볼 수 있습니다. 다음을 실행하여 컴파일된 HTML 비네트를 웹 브라우저에서 볼 수 있습니다.

help(package = "corrplot", help_type = "html")

그러면 corrplot 설명서의 HTML 인덱스 페이지가 표시되며,

- 그림 15.6: ggplot2 전환 가이드의 샘플 페이지: ggplot2 0.9.0에 추가된 새로운 geom인 geom_violin()을 소개합니다.

PDFS= foo.pdf bar.pdf all: $(PDFS) clean:

rm -f _.tex _.bbl _.blg _.aux _.out _.log %.pdf: %.Rnw

$(R_HOME)/bin/Rscript -e "knitr::knit2pdf('$\*.Rnw')"

- 그림 15.7: knitr를 사용하여 PDF 비네트를 컴파일하기 위한 Makefile: Rnw 문서를 PDF로 컴파일하기 위해 knit2pdf()를 사용합니다.

"Overview of user guides and package vignettes" 비네트 링크를 볼 수 있습니다. corrplot은 상관 행렬을 시각화하기 위한 패키지이므로 그래픽 예제가 많으며 이러한 예제는 HTML 비네트에 표시됩니다.

knitr의 소스 패키지에는 PDF와 HTML 비네트가 모두 포함되어 있으며 이 모든 것은 이 패키지의 HTML 도움말 페이지에 나열되어 있습니다.

sampSurf 패키지(Gove, 2013)도 http://sampsurf.r-forge.r-project.org 에 멋진 HTML 비네트를 가지고 있습니다. 이는 R HTML 소스 문서에서 만들어졌으며 심지어 rgl 패키지에서 생성된 일부 3D 플롯도 포함하고 있습니다.

###### 15.4.3 PDF 비네트

R 3.0.0 이하 버전에 맞추어 knitr로 비네트를 빌드하려면 약간의 요령이 필요합니다. 한 가지 방법은 Makefile(http://www.gnu.org/software/make/)을 사용하는 것입니다. 이 Makefile은 R CMD build가 비네트를 빌드할 때 사용하게 됩니다. 이 Makefile에서 knitr와 같은 커스텀 도구를 사용하여 PDF 파일을 생성하는 규칙을 지정할 수 있습니다.

이 Makefile은 소스 패키지의 vignettes/ 디렉터리에 있습니다. R이 비네트를 컴파일할 때 먼저 Sweave()를 호출합니다. 만약 Makefile이 존재한다면, 거기에 make 명령이 실행됩니다. 이 Makefile 내에서도 R에 접근할 수 있으므로, 명령줄을 통해 knitr를 호출하여 비네트를 컴파일하는 것이 가능합니다. 그림 15.7은 knitr를 사용해 비네트를 컴파일할 때 쓰이는 Makefile의 예시를 보여줍니다. 여기서 핵심은 Rnw 파일에 knitr::knit2pdf()를 실행하는 것입니다. 생성될 모든 PDF 파일들을 PDFS 변수에 명시해둡니다.

물론, 이 방식의 단점은 이후의 과정으로 넘어가기 전에 먼저 모든 Rnw 문서를 Sweave로 컴파일해야 한다는 점입니다.
HTMLS= foo.html bar.html all: $(HTMLS) clean:

rm -rf figure/ \*.md %.html: %.Rmd

$(R_HOME)/bin/Rscript -e "knitr::knit2html( '$\*.Rmd' )"

- 그림 15.8: HTML 비네트 컴파일을 위한 Makefile: knit2html()을 사용하여 Rmd 문서를 HTML로 컴파일합니다.

또한 R 3.0.0 이상에서의 새로운 접근 방식은 make 유틸리티를 설치할 필요가 없습니다.

###### 15.4.4 HTML 비네트

마찬가지로 R 마크다운 문서에서 HTML 형식의 패키지 비네트를 만들 수도 있습니다. 이 역시 R 3.0.0 이전에는 HTML 비네트를 Makefile로 컴파일해야 했습니다. 그림 15.8은 HTML 비네트를 빌드하기 위한 샘플 Makefile의 소스를 보여주며, 여기서 knit2html() 함수가 호출되었습니다. 여기서 make clean이 figure/ 디렉터리를 제거한다는 점에 유의하십시오. knitr가 생성한 이미지는 HTML 출력에 base64로 인코딩되므로 이미지 파일이 더 이상 필요하지 않기 때문입니다.

###### 15.5 책

knitr로 책을 쓸 수도 있습니다. 이 책을 쓸 당시 최소한 한 권의 책(Lebanon, 2012)이 출판되었고 Regression Modeling Strategies(Harrell, 2001)라는 책은 knitr를 기반으로 하는 새로운 버전으로 개정 중이었습니다.

###### 15.5.1 이 책

"자신의 개 사료 먹기(eating one’s own dog food)"(이 말이 불분명하다면 위키백과 참조)라는 정신에 따라 이 책은 LYX(4.2절 참조)에서 knitr로 작성되었습니다. 개별 파일로 장을 나누는 것이 완전히 가능하긴 하지만 전체 책이 하나의 LYX 파일에 있습니다.

문서의 맨 처음에 cache = TRUE(속도용), dev = 'tikz'(그래픽 스타일용), fig.align = 'center'(플롯 정렬용) 등 몇 가지 청크 옵션이 전역적으로 설정되었습니다. 또한 options(formatR.arrow = TRUE)를 설정했습니다(formatR 패키지 참조). 저자는 할당 연산자로 <- 대신 =를 선호하지만 R 사용자는 보통 <-를 더 많이 사용하기 때문입니다. 이 옵션을 사용하면 제가 실제로 입력한 것은 등호뿐일지라도 가능한 모든 곳에서 등호를 왼쪽 화살표로 자동으로 바꿀 수 있습니다.

이 책에는 다양한 목적으로 사용하는 몇 가지 청크 후크(10장)가 있습니다. 예를 들어 그래픽 매개변수를 다음과 같이 설정하는 par 후크가 있습니다.

par(mar = c(4, 4, 0.1, 0.1), cex.lab = 0.95, cex.axis = 0.9,
mgp = c(2, 0.7, 0), tcl = -0.3, las = 1)

따라서 이 매개변수 세트를 사용하고 싶을 때는 이를 반복해서 입력할 필요 없이 청크 옵션 par = TRUE를 추가하기만 하면 됩니다.

이 책에서는 코드 청크와 플롯이 분리되어 있는 것을 볼 수 있지만 소스 문서에서는 그렇지 않습니다. 코드 청크는 실제로 figure 환경 내부에 있지만, 문서 후크 hook_movecode()를 사용하여 결국 figure 환경 밖으로 코드 청크를 이동시켰습니다.

교육적 목적으로 가끔 청크 헤더를 표시해야 하기 때문에 청크 출력에 < <> >=와 @를 추가하는 append라는 청크 후크가 있습니다.

knit_hooks$get("append")

## function(before, options, envir) { ## txt = options$append[[ifelse(before, 1, 2)]] ## txt = c("\\begin{alltt}", txt, "\\end{alltt}") ## paste(txt, collapse = "\n") ## }

기본적으로 이 후크를 사용하면 청크 앞 및/또는 뒤에 추가 문자열을 쓸 수 있습니다. 예를 들어 청크 옵션 append = list('< <A> >=', '@')를 사용하여 청크 출력에 구문 정보를 추가할 수 있습니다. 청크 헤더를 소스 문서에 직접 쓸 수 없기 때문에 이 후크를 사용해야 합니다. 직접 쓰게 되면 이것이 구문 분석되어 최종 출력에서 사라집니다.

플롯에 프레임 상자를 추가하여 기본 플롯 후크 함수를 수정하는 출력 후크가 있으며 그림 10.3과 그림 10.4에서 사용되었습니다.

모든 R 패키지의 참고 문헌 데이터베이스는 12.4.1절에서 소개한 바와 같이 write_bib() 함수에 의해 동적으로 기록되므로 버전 정보가 최신 상태임을 보장할 수 있습니다(최소한 원고를 출판사에 제출하기 전까지는).

###### 15.5.2 The Analysis of Data

또 다른 주목할 만한 예는 Lebanon (2012)의 책 The Analysis of Data입니다. 이 책의 가장 주목할 만한 특징은 이중 PDF/HTML 버전을 갖추고 있다는 것입니다. HTML 버전은 http://theanalysisofdata.com 에서 무료로 이용할 수 있습니다. 두 버전 모두 기본적으로 동일한 소스 문서 세트에서 생성됩니다. HTML 버전의 경우 추가 설정이 있습니다. 예를 들어 수학 수식의 조판은 MathJax 라이브러리에 의해 수행되므로 HTML 소스의 head 섹션에 포함되어야 합니다.

###### 15.5.3 The Statistical Sleuth in R

The Statistical Sleuth(Ramsey 및 Schafer, 2002)는 통계학의 훌륭한 교재이며 이 책의 한 가지 특징은 데이터 세트가 많다는 것입니다. 책 자체는 knitr로 작성되지 않았지만 일부 다른 저자(Horton 등, 2012)가 웹사이트(http://www.math.smith.edu/~nhorton/sleuth/)를 만들어 책의 많은 데이터 분석 예제를 R로 다시 수행했습니다. 해당 웹사이트에서 PDF 문서와 Rnw 소스 파일을 모두 확인할 수 있습니다.

###### 15.5.4 문학도를 위한 R을 이용한 텍스트 분석

Jockers (2014)의 저서 'Text Analysis with R for Students of Literature(문학도를 위한 R을 이용한 텍스트 분석)'는 LATEX와 knitr를 사용하여 작성되었습니다. 이 책에 관한 가장 놀라운 사실은 아마도 저자가 LATEX로 이 책을 종합하기 전에 혼자 LATEX를 독학했고 단 몇 달 만에 책 초안을 완성했다는 점일 것입니다. 이 책은 컴퓨터 기반 텍스트 분석에 대한 입문서이며 많은 짧은 예제가 있습니다. 만약 저자가 각 예제를 실행하고 그 출력을 일일이 손으로 LATEX 원고에 복사해야 했다면 이 작업은 지루하기 짝이 없었을 것입니다.

###### 15.6 R 패키지를 위한 문학적 프로그래밍(Literate Programming)

이 책의 서두에서 문학적 프로그래밍(Literate Programming, LP)을 소개했지만, 우리는 실제 프로그래밍 목적으로 knitr 패키지를 사용하지는 않습니다. 대부분의 경우 데이터 분석 및 보고 목적으로 knitr를 사용합니다. 본래 LP 패러다임은 위빙(weaving)과 탱글링(tangling)에 관한 것입니다. 즉, 소프트웨어 설명서로 소스 문서를 위빙하거나, 실행하기 위해 프로그램 코드를 탱글링할 수 있습니다. knitr를 사용할 때는 위빙 과정에서 코드 실행이 바로 이루어지기 때문에 굳이 실행을 목적으로 프로그램 코드를 엮을(tangle) 필요가 없습니다.

흥미롭게도 Knuth의 원래 LP 패러다임이 가장 흔하게 응용되는 분야는 패키지 작성자를 위한 "프로그래밍"이 아니라 사용자를 위해 (특수한 형태의 주석을 사용하여) 소프트웨어를 문서화하는 것으로 보입니다. 즉, 소스 코드를 문서화하는 대신 소프트웨어의 사용법을 문서화하는 데 LP를 사용합니다. Doxygen(van Heesch, 2008), Javadoc(http://en.wikipedia.org/wiki/Javadoc), roxygen2(Wickham 등, 2015)의 예를 참조하십시오. 하지만 LATEX 세계에는 한 가지 예외가 존재합니다. 일부 LATEX 패키지 작성자는 LATEX 코드와 설명서를 모두 하나의 문서로 작성하고, 소스 코드와 설명서가 모두 포함된 PDF 문서로 엮습니다(weave). TEX와 Pascal을 사용한 Knuth의 원래 LP 구현을 고려하면 이는 완전히 놀라운 일은 아닙니다. Terry Therneau의 survival 및 coxme 패키지와 같이 LP를 사용하는 소수의 R 패키지도 있습니다.

LP는 널리 쓰이는 프로그래밍 접근 방식은 아닌 듯하지만 여전히 흥미로운 아이디어이며, 자신이 선호하는 언어에 적용될 때 특히 유용할 수 있습니다. LATEX 소스 코드를 읽는 것은 일부 사람들에게는 지루할 수 있지만 R 소스 코드를 읽는 것은 더 즐거울 수 있습니다. 객관적인 의견을 떠나 우리는 LP에 적어도 두 가지 장점이 있다고 믿습니다.

1. 평소에 주석으로 남기는 것보다 훨씬 광범위하고 풍부한 설명서를 작성할 수 있습니다. 일반적으로 코드의 주석은 간결하고(혹은 간결해야 하며) 일반 텍스트에 한정됩니다. 보통은 몇 줄의 코드를 설명하기 위해 다섯 문단의 주석을 작성하지 않을 것이며, 읽을 수 있는 수학 표현식을 쓰거나 비디오를 주석 안에 삽입할 수도 없습니다.
2. 코드 청크에 레이블을 지정하고 레이블을 사용하여 이를 참조/재사용할 수 있으므로 여러 코드 청크를 결합하여 유연하게 프로그램을 구성할 수 있습니다. 예를 들어, 나중에 문서에서 코드 청크를 정의하고 설명하더라도 해당 레이블을 사용하여 이전의 코드 청크에 이를 삽입할 수 있습니다. 이 기능은 Knuth에 의해 강조되었지만 어떤 이유에서인지 널리 채택되지는 않았습니다. 아마도 대부분의 사람들이 코드 청크 대신 함수와 같은 더 작은 단위로 큰 프로그램을 디자인하는 것에 더 익숙해져 있을 수 있으며, 이것은 실제로 좋은 아이디어입니다.

실제로 R 패키지 개발에 LP를 적용할 수 있습니다. 목표를 달성하는 방법은 여러 가지가 있지만 여기서는 다음 도구들을 사용하여 한 가지 방법만 소개합니다.

1. 소스 문서에서 프로그램 코드를 추출할 수 있게 해 주는 knitr의 purl() 함수
2. 프로그램 코드와 설명서를 모두 포함할 수 있는 패키지 비네트
3. 소스 파일에서 출력 파일을 생성하는 시기와 방법을 정의할 수 있게 해 주는 GNU Make

rlp 패키지(https://github.com/yihui/rlp)는 LP 기법을 사용하여 R 패키지를 작성하는 예입니다. 이 저장소에서 세부 사항을 확인할 수 있으며 그 구현의 기본 아이디어는 다음과 같습니다.

1. 패키지의 R/ 디렉터리 아래에 R 소스 코드를 작성하는 대신, vignettes/ 디렉터리 아래의 패키지 비네트(R 마크다운)에 코드를 작성할 수 있습니다.
2. Makefile을 사용하여 비네트 vignettes/_.Rmd에서 R 스크립트 R/_.R을 생성하는 방법을 정의합니다.
3. make를 실행하여 R/에 R 스크립트를 생성하고 R CMD build를 실행하여 패키지를 빌드합니다.

이러한 단계는 RStudio IDE를 사용하면 쉽게 만들 수 있으며, 버튼 하나만 클릭하면 이 작업을 모두 완료할 수 있습니다. 이에 대한 구현 세부 사항은 이 책에서 다루기에는 너무 기술적이고 구체적이므로 해당 패키지의 문서를 살펴보는 것은 독자의 몫으로 남겨두겠습니다.

### 16

###### 기타 도구

knitr 외에도 동적 문서를 위한 많은 다른 도구가 있습니다. 일부는 R 패키지이고 다른 일부는 파이썬 및 awk와 같은 다른 언어의 도구입니다. 이 장에서는 이러한 도구에 대해 간략히 개요를 살펴보고 knitr와 비교하며 특히 Sweave 사용자를 위해 Sweave와 knitr의 차이점을 설명합니다.

###### 16.1 Sweave

knitr 패키지는 R에서 오랫동안 유명했던 동적 문서 작성 도구이자 base R의 일부인(utils 패키지의 Sweave() 함수) Sweave(Leisch, 2002)에서 큰 영감을 받았습니다. Sweave는 주로 Rnw 문서를 다루지만, 다른 문서 형식으로 확장할 수 있는 모듈식 디자인도 갖추고 있습니다. CRAN에는 Sweave를 기반으로 하는 수많은 확장이 존재하며 다음 섹션에서 소개할 것입니다.

Sweave를 실행하는 방법에는 두 가지가 있습니다. 대화형 R 세션에서 호출할 수 있습니다(utils 패키지를 로드할 필요가 없습니다).

Sweave("your_file.Rnw") # your_file.tex가 생성됩니다

또한 다음과 같이 명령줄을 사용할 수도 있습니다. R CMD Sweave your_file.Rnw

Sweave는 base R의 일부이기 때문에 그 개발이 최근 몇 년 동안 거의 정체되었습니다. 또 다른 주요 문제는 그것의 모듈식 디자인이 충분히 모듈화되지 않아서 base R에서 Sweave가 업데이트될 때마다 확장이 호환되지 않을 수 있다는 것입니다. 저희가 아는 한, Sweave를 기반으로 하는 몇몇 R 패키지는 Sweave에서 많은 양의 핵심 코드를 복사했으며 더 이상 Sweave 개발 방향과 맞지 않게 되었습니다.

knitr의 청크 옵션 중 eval, echo, results 등 많은 것들이 Sweave에서 차용되었지만 그 디자인이 다르기 때문에 이 둘 사이에는 몇 가지 차이점이 있습니다. 1.0 버전 이전의 knitr는 Sweave와 호환되려고 노력했습니다. 내부적으로 몇 가지 함수들이 자동으로 차이점을 고쳐주어 knitr가 Sweave 문서를 컴파일할 수 있었습니다. 하지만 v1.0 이후로는 호환성이 더 이상 유지되지 않으며, 대신 Sweave 문서를 수동으로 knitr 문서로 변환하는 Sweave2knitr() 함수를 제공합니다. 아래는 utils 패키지의 Rnw 문서를 변환하고 그 전후의 차이점(<는 원본 문서, >는 변환된 문서)을 보여주는 예제입니다.

testfile <- system.file("Sweave", "Sweave-test-1.Rnw",
package = "utils") outfile <- tempfile(fileext = ".Rnw") Sweave2knitr(testfile, output = outfile) # true/false를 대문자 TRUE/FALSE로 변경: # _ fig=true # 불필요한 옵션 fig=TRUE 제거: # _ fig=TRUE # _ fig=TRUE # results 옵션을 따옴표로 묶음: # _ results='hide' # 'print', 'term', 'prefix' 옵션 제거: # _ print=TRUE # _ echo=TRUE,print=TRUE # true/false를 대문자 TRUE/FALSE로 변경: # _ echo=true # \SweaveOpts{}를 opts_chunk$set()으로 변경: # _ \SweaveOpts{echo=FALSE} # _ \SweaveOpts{echo=true} # 잉여 줄 제거(#n은 줄 번호를 보여줌): # _ (#69) @ cat(system(sprintf("diff %s %s", shQuote(testfile),
shQuote(outfile)), intern = TRUE), sep = "\n")

# 7c7,14 # < \SweaveOpts{echo=FALSE} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=FALSE

# > ) # > @ # > # 15c22 # < <<print=TRUE>>= # --# > <<>>= # 17c24 # < <<results=hide>>= # --# > <<results='hide'>>= # 22c29 # < <<echo=TRUE,print=TRUE>>= # --# > <<echo=TRUE>>= # 43c50,57 # < \SweaveOpts{echo=true} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=TRUE # > ) # > @ # > # 53c67 # < <<fig=TRUE>>= # --# > <<>>= # 63c77 # < <<fig=true>>= # --# > <<>>= # 69d82 # < @

###### 16.1.1 구문

기본적으로 knitr는 R 함수 인수와 유사한, 청크 옵션을 분석하기 위한 새로운 유형의 구문을 사용합니다. 이는 기존의 Sweave 구문보다 훨씬 더 강력한 힘을 제공합니다. 우리는 청크 옵션에서 임의의 객체를 사용할 수 있고 R의 모든 강력한 기능을 활용할 수 있습니다.

Sweave는 청크 옵션을 문자열로 처리하고 옵션들을 쉼표로 분할하여 구문 분석을 진행하는 반면, knitr는 R 구문을 사용합니다. 옵션이 문자값을 사용하는 경우 R에서 하는 것처럼 따옴표로 묶어야 합니다(results = 'hide' (Sweave에서는 results = hide라고 씁니다)). 청크 옵션 내에서 직접 연산을 수행하는 예제는 12.1.3절을 참조하십시오. 아래는 이 새로운 구문이 얼마나 유연한지 보여주는 또 다른 예입니다(그림 캡션을 동적으로 만들 수 있습니다).

<<cap, fig.cap=paste('P값은', t.test(x)$p.value, '입니다')>>= x <- rnorm(100) boxplot(x) @

구문의 다른 사소한 차이점은 knitr의 경우 그 앞에 청크 헤더가 없는 한 @를 텍스트 청크의 시작으로 인식하지 않는다는 것입니다. 예를 들어 knitr는 아래 예시에서 첫 번째 @를 유지하지만 Sweave는 이를 제거합니다.

text @ <<A>>= 1 + 1 @

Sweave2knitr()는 이 문제를 자동으로 해결할 수 있습니다.

###### 16.1.2 옵션

Sweave의 일부 옵션이 knitr에서 삭제되었으며 일부는 변경되었습니다. 여기에는 다음이 포함됩니다.

concordance는 주로 RStudio를 지원하기 위해 변경되었습니다. 패키지 옵션 opts_knit$get('concordance')가 TRUE이면 출력 줄 번호가 입력 줄 번호에 매핑된 input-concordance.tex라는 파일이 작성됩니다. 그 구현이 Sweave보다 덜 정확하다는 점을 유의하십시오.

keep.source는 더 유연한 옵션인 tidy로 병합되었습니다.

print가 삭제되었습니다. R 표현식이 인쇄될지 여부는 R 사용 경험과 일치합니다(x <- 1은 인쇄되지 않지만 1:10은 인쇄됩니다. 단순히 R 콘솔에서 명령을 입력한다고 상상해 보십시오). 표현식의 출력이 진정으로 보이지 않게 하려면 invisible() 함수를 사용할 수 있습니다.

term이 삭제되었습니다(term = TRUE라고 생각하십시오).
prefix가 삭제되었습니다(prefix = TRUE라고 생각하십시오).
prefix.string의 이름이 fig.path로 변경되었으며 항상 그림 파일 이름에 사용됩니다.

eps, pdf 및 그래픽 장치에 대한 모든 논리형 옵션이 삭제되었습니다. 대신 Sweave의 grdevice와 유사하지만 20개 이상의 미리 정의된 그래픽 장치가 있는 새 옵션 dev를 사용하십시오(7장 참조).

fig가 삭제되었습니다. 이제 fig.keep을 사용하십시오. knitr에서 fig.keep = 'high'는 fig = TRUE와 같고, fig.keep = 'none'은 Sweave의 fig = FALSE와 같습니다.

width, height는 각각 fig.width 및 fig.height로 이름이 변경되었습니다. 한편 \SweaveOpts{} 및 \SweaveInput{}은 더 이상 사용되지 않습니다. 각각 opts_chunk$set() 및 청크 옵션 child를 사용하여 전역 청크 옵션을 설정하고 자식 문서를 포함하십시오.

논리형 옵션의 경우 TRUE/FALSE/T/F만 지원되며(처음 두 가지를 권장함) true/false는 작동하지 않습니다. 예를 들어 eval = FALSE는 괜찮지만 eval = false는 그렇지 않습니다(논리값 FALSE를 갖는 false라는 R 객체가 존재하는 경우 제외). < <label> > 구문을 사용한 청크 참조는 계속 사용할 수 있으며 새 옵션 ref.label을 사용하는 등 청크를 재사용하는 다른 방법이 있습니다. 청크 참조는 9장에 소개된 바와 같이 재귀적일 수 있습니다.

###### 16.1.3 문제점

Sweave에서 알려진 몇 가지 문제점과 자주 묻는 질문들이 knitr에서는 해결되었습니다.

- 빈 그림 청크는 Sweave에서 LATEX 오류를 유발하지만, knitr에서는 그림이 전혀 생성되지 않기 때문에 오류가 발생하지 않습니다. knitr는 청크에 플롯이 있을 때만 그림을 LATEX에 작성합니다.
- lattice(및 ggplot2) 그래픽은 명시적으로 print()를 실행하지 않으면 Sweave에서 작동하지 않지만, knitr에서는 R 콘솔에서와 마찬가지로 작동합니다(이러한 플롯 객체가 최상위 환경에 나타나면 인쇄할 필요가 없습니다).
- 출력되는 그림의 너비는 기본적으로 Sweave에서 LATEX 스타일 Sweave.sty에 정의된 \setkeys{Gin}{width=.8\textwidth}를 통해 .8\textwidth로 설정됩니다. 이는 Sweave에서 생성되었는지 여부와 관계없이 문서의 모든 그림에 영향을 미치며, 그림에 개별 너비를 설정하는 간단한 방법이 없습니다. 이 문제는 knitr에서 out.width 옵션으로 해결되었습니다.
- 기본적으로 Sweave에서는 하나의 그림 청크에서 여러 그림을 지원하지 않으므로 이 경우 직접 LATEX 코드를 작성해야 합니다. knitr의 경우 하나의 청크에 아무리 많은 플롯이 있더라도 차이가 없습니다.
- 출력 후크를 사용하여 knitr에서 출력 형식을 변경하는 것이 가능하므로 Sweave의 Sinput/Soutput과 같이 하드 코딩된 LATEX 환경을 사용할 필요가 없습니다. 실제로 render_sweave()를 호출하여 knitr에서 Sweave 스타일을 렌더링할 수 있습니다.
- knitr(R HTML 또는 R 마크다운 사용)를 사용하면 HTML 출력을 쉽게 생성할 수 있으며 Sweave는 HTML만 처리하는 R2HTML과 같은 확장이 필요합니다.

때때로 Sweave를 실행한 후 Rplots.pdf 파일이 덩그러니 남는 것을 볼 수 있습니다. 이는 대화형이 아닌 R 세션에 대한 R의 기본 그래픽 장치가 pdf()이기 때문이며, 이로 인해 Rplots.pdf가 생성됩니다. knitr에서는 방황하는 PDF 파일이 생성되지 않도록 기본 장치가 널(null) 장치(pdf(file = NULL))로 설정되어 있습니다.

###### 16.2 기타 R 패키지

Sweave 및 아래에 소개되는 R 패키지(R2HTML 제외)의 대부분의 기능은 knitr에서 다루어지므로 이 절은 주로 역사적 흥미를 위한 것입니다.

highlight 패키지(Francois, 2013)는 Rnw 문서에서 R 코드에 대한 구문 강조 표시를 제공합니다. 아래의 pgfSweave, cacheSweave, R2HTML과 마찬가지로 highlight도 Sweave를 기반으로 확장되었습니다. 초기 버전(v0.6 이전)에서는 knitr가 구문 강조 표시를 하기 위해 highlight에 의존했지만, 유지 관리 문제와 (Rcpp 및 parser 패키지라는) 추가 종속성이 있다는 사실 때문에 이 의존성은 나중에 제거되었습니다. 이제 knitr는 자체 구문 강조 함수를 사용합니다. 이는 R 3.0.0 이전에는 정규 표현식 기반이었고, R 3.0.0 이후에는 base R의 utils 패키지에 있는 getParseData() 함수에 의존합니다. highlight와 유사한 기능을 얻으려면 knitr에서 청크 옵션 highlight = TRUE를 사용하기만 하면 됩니다.

cacheSweave 패키지(Peng, 2012)는 캐시 시스템이라는 중요한 기능을 Sweave에 추가했습니다. weaver 패키지(Falcon, 2013)는 다른 구현을 통해 비슷한 작업을 수행했습니다. 청크 옵션 cache 및 dependson이 추가되었으며 이는 knitr에서와 동일한 의미를 갖습니다(8장 참조).

pgfSweave 패키지(Bracken 및 Sharpsteen, 2012)는 highlight 및 cacheSweave의 기능을 결합하고 그래픽에 대한 추가 지원을 제공했습니다. 특히 글꼴 스타일 일관성을 위해 플롯도 캐시할 수 있으며 tikzDevice 패키지를 통한 TikZ 그래픽도 지원됩니다. 이 책의 저자는 pgfSweave가 출시되었을 때 Sweave에서 교체하였고, formatR 지원(tidy 옵션)에 기여했지만, 시간이 지나면서 Sweave의 변화를 따라잡는 것이 점점 더 어려워졌습니다. 이 패키지는 CRAN 저장소에서 삭제되었습니다. 어찌되었든, knitr의 디자인은 저자가 pgfSweave를 다룬 경험에서 많은 이점을 얻었습니다.

brew 패키지(Horner, 2011)는 가벼운 템플릿 프레임워크이며 그 구문은 PHP(<?php ?>)와 유사합니다. 기본적으로 템플릿 태그 <% %> 내부의 R 코드를 구문 분석하고 실행합니다. 이것을 Sweave와 knitr의 인라인 R 코드라고 생각할 수 있습니다. 캐시 시스템이 있지만 직접적인 그래픽 지원은 없습니다. knitr 패키지에도 brew 구문에 대한 부분적 지원이 있는데 이것은 5장에서는 언급하지 않았습니다. 아래는 knitr를 통해 컴파일할 수 있는 예제입니다.

pi 값은 <% pi %>이고, 2 곱하기 pi는 <% 2\*pi %>입니다.

입력 파일의 확장자가 \*.brew인 경우 knitr는 brew 구문을 자동으로 사용합니다. 실제로 brew는 여러 인라인 식에서 불완전한 코드 조각을 지원하므로 PHP와 정말 유사합니다. 다음은 brew에서 가져온 예시이지만 knitr는 이를 컴파일할 수 없습니다.

<% for (i in c('1+1', '1+pi', '1+pi', 'sin(pi/2)')) { -%> > <%=i%> <% print(eval(parse(text=i))) %> <% } -%>

R2HTML 패키지(Lecoutre, 2014)에는 R 객체를 HTML로 내보내는 많은 함수가 포함되어 있습니다. 주 기능은 S3 제네릭 함수 HTML()로, 데이터 프레임, 표, (lm()에 의해 반환된) lm 객체 등과 같은 다양한 R 객체에 적용될 수 있습니다. 아래는 iris 데이터의 하위 집합을 HTML 표로 변환한 것입니다.

library(R2HTML) HTML(head(iris[, -5], 1), "", caption = NULL)

<p align='center'> <table cellspacing=0 border=1><tr><td>
<table border=0 class=dataframe> <tbody> <tr class='firstline'>
<th>&nbsp; </th> <th>Sepal.Length </th> <th>Sepal.Width </th> <th>Petal.Length </th> <th>Petal.Width</th>
</tr>
<tr> <td class=firstcolumn>1 </td> <td class=cellinside>5.1 </td> <td class=cellinside>3.5 </td> <td class=cellinside>1.4 </td> <td class=cellinside>0.2 </td></tr>
</tbody> </table>
</td></table>

결과를 raw HTML 코드로 출력하기 위해, R HTML 문서에 대한 knitr 내에서 청크 옵션 results = 'asis'와 함께 R2HTML을 사용할 수 있습니다.

R2HTML의 또 다른 주요 기여는 Sweave 확장으로 이를 통해 사용자는 Sweave 기반의 HTML 보고서를 작성할 수 있습니다.

CRAN에는 재현 가능한 연구에 대한 태스크 뷰인 http://cran.r-project.org/web/views/ReproducibleResearch.html 이 있으며, 여기에서 이 주제에 관한 더 많은 패키지를 찾을 수 있습니다.

###### 16.3 파이썬 패키지

이 절에서는 파이썬(Python) 기반의 동적 문서 패키지 세 가지, 즉 Dexy, PythonTEX, 그리고 IPython을 소개합니다.

###### 16.3.1 Dexy

Dexy(http://www.dexy.it)는 범용적인 설계가 돋보이는 무료 파이썬 패키지입니다. 웹사이트에 따르면:

Dexy는 코드가 통합된 어떠한 형태의 기술 문서든 작성할 수 있게 해 주는 자유 형식의 문학적 문서 작성 도구입니다. Dexy는 코드가 변경되더라도 오랜 기간 올바른 문서를 쉽게 작성하고 유지 관리할 수 있도록 도와줍니다.

네 가지 주요 특징은 다음과 같습니다.

1. 모든 언어 (소스 코드)
2. 모든 마크업 (출력)
3. 모든 템플릿
4. 모든 API (프로그래밍)

다국어 지원 등 Dexy와 knitr 사이에는 분명 유사점이 있습니다. Dexy의 중요한 개념은 "필터(filter)"입니다. 필터는 입력 파일을 받아서 출력 파일로 변환하는데, 이는 쉘 스크립트의 파이프 |와 비슷합니다. Dexy의 필터는 사실 knitr에 있는 개념들의 조합입니다. 필터는 출력을 렌더링하거나(마크다운에서 HTML로), 프로그래밍 언어를 실행하거나(knitr의 언어 엔진처럼), 또는 knitr의 청크 후크와 같이 추가 작업을 수행할 수 있습니다.

일반적으로 Dexy는 컴퓨터 코드와 템플릿을 분리하는데, 이는 장단점이 될 수 있습니다. 좋은 점은 소스 스크립트를 재사용할 수 있다는 것이고, 나쁜 점은 보고서 환경과 소스 코드 사이를 앞뒤로 왔다 갔다 해야 한다는 점입니다. 기본적으로 knitr는 보고서에 코드 청크를 직접 삽입하지만 9장에 소개된 것처럼 코드 청크를 외부화할 수도 있습니다.

###### 16.3.2 PythonTEX

PythonTEX(https://github.com/gpoore/pythontex)는 LATEX 내에서 파이썬 코드를 실행할 수 있는 기능을 갖춘 LATEX 패키지입니다. 그 문서에 따르면:

PythonTEX는 LATEX 내에서 빠르고 사용자 친화적인 파이썬 접근 권한을 제공합니다. LATEX 문서 안에 입력된 파이썬 코드를 실행하고, 그 결과를 원래 문서에 포함할 수 있게 해 줍니다. 또한 Pygments 패키지를 통해 LATEX 문서 내 코드에 대한 구문 강조 기능을 제공합니다.

\pyb{} 명령을 사용하여 인라인 파이썬 코드를 삽입하거나, pyconsole 환경을 사용하여 LATEX에서 파이썬 세션을 에뮬레이션할 수 있습니다. 예:

\begin{pyconsole}[][frame=single]

- x = 123
- y = 345
- z = x + y z def f(expr):

return(expr\*\*4)

- f(x) print('콘솔에서 파이썬이 인사합니다!') \end{pyconsole}

이 문서를 컴파일하면 파이썬 코드가 평가되고 결과가 출력에 삽입됩니다.

파이썬 기반이기 때문에 SymPy(기호 조작) 및 matplotlib(플롯)과 같은 다른 파이썬 패키지와도 잘 통합됩니다.

###### 16.3.3 IPython

IPython(http://ipython.org)은 코드, 텍스트, 수식 표현, 인라인 플롯 등 리치 미디어에 대한 지원, 병렬 컴퓨팅을 위한 고성능 도구 등을 지원하는 웹 기반 노트북을 갖춘 대화형 쉘입니다.

그림 16.1은 Ubuntu 환경의 GNOME 터미널에서 구동되는 IPython의 스크린샷입니다. 쉘에 x.spl<TAB>을 입력하면 아래에 자동 완성 항목이 표시되는 등 쉘의 기본적인 명령 자동 완성 기능을 갖추고 있음을 알 수 있습니다.

보고서 생성과 관련하여 가장 주목할 만한 기능은 웹 기반 노트북입니다. 웹 브라우저에서 파이썬 명령으로 작업하고 결과(수치 및 그래픽 결과 모두 포함)를 실시간으로 확인할 수 있으며 노트북에 콘텐츠를 추가 입력함에 따라 지속적으로 노트북이 업데이트될 수 있습니다. 이는 knitr에서 코드 청크를 작성하는 것과 매우 비슷합니다.

IPython 노트북은 확장자가 \*.ipynb인 JSON 파일로 저장할 수 있으며 다른 사람과 공유할 수 있습니다. 노트북에는 출력이 포함될 수도 있고 포함되지 않을 수도 있습니다. 출력이 없는 노트북은 knitr용 소스 문서(Rnw 및 Rmd 문서)와 유사합니다.

IPython에서 영감을 받아 knitr도 유사한 웹 노트북(하지만 기능이 더 적은)을 얻었으며 이에 대해서는 3.2.2절에서 언급했습니다.

- 그림 16.1: IPython의 스크린샷: 입력은 In[n]으로 표시되고 출력은 Out[n]으로 표시됩니다.

###### 16.4 더 많은 도구

R 및 파이썬 패키지 외에도 다른 프로그램에 여러 도구가 있습니다. 이 장에서 동적 문서를 위한 모든 도구를 나열하는 것은 불가능합니다. Schulte 등 (2012)은 Javadoc, cweb, noweb, Sweave, SASweave 등과 같이 문학적 프로그래밍과 재현 가능한 연구를 위해 기존 도구 목록을 제공했습니다.

###### 16.4.1 Org-mode

Org-mode는 일반 텍스트 마크업 언어이며 Emacs 텍스트 편집기에 구현되어 있습니다(Schulte 등, 2012). 문학적 프로그래밍과 재현 가능한 연구(동적 문서의 관점에서)를 모두 지원합니다. 이는 다소간 WEB 및 noweb과 같은 초기 문학적 프로그래밍 구현의 구문을 따릅니다. 즉, 코드 청크와 텍스트 청크(텍스트 청크는 종종 "산문(prose)"이라고 불림)의 개념이 있습니다. Org-mode의 코드 청크는 다음과 같습니다.

#+name: c-chunk #+begin_src C

int main(){

return 0; }

#+end_src

비교를 위해, 동일한 청크를 knitr에서는 다음과 같이 씁니다. <<c-chunk, engine='c'>>= int main(){

return 0;

} @

메타데이터는 청크 헤더에 저장됩니다. Org-mode는 출력을 LATEX 또는 HTML 형식으로 하여 어떠한 입력 언어도 지원할 수 있습니다.

Schulte 등 (2012)은 기존 도구들의 문학적 프로그래밍 기능(Sweave에는 없는 기능)을 언급했는데, 저희는 이 책에서 보고서 작성자에게 별로 흥미로울 것 같지 않아 강조하지 않았습니다. 사실 knitr에도 코드 청크를 재구성하는 기능이 있습니다(9장 참조). 아래는 청크 B를 나중에 정의하지만 이전 청크 A에 삽입하는 간단한 예입니다.

- <<A>>= df <- data.frame(x = 1:10, y = rnorm(10))
- <<B>> coef(fit) @

<<B>>= fit <- lm(y ~ x, data = df) @

아무리 강력하더라도 Org-mode의 Emacs적 특성은 초보자에게 장애물이 될 수 있습니다.

###### 16.4.2 SASweave

SASweave(http://homepage.cs.uiowa.edu/~rlenth/SASweave)는 SAS와 R을 사용한 문학적 프로그래밍 구현입니다. 이는 gawk로 작성되었습니다. 기본적인 아이디어는 Sweave 및 knitr와 같습니다. 자세한 정보는 Lenth와 Højsgaard (2007)를 참조하십시오. knitr 패키지는 SASweave에 비해 R에 대한 더 포괄적인 지원을 제공하지만 SAS에 대한 지원은 적습니다.

###### 16.4.3 Office

동적 문서를 위해 꼭 일반 텍스트 형식을 선택해야만 하는 것은 아니지만, 이 책에서 소개한 거의 모든 것은 일반 텍스트 기반이었습니다. OpenOffice(또는 OpenDocument Text)나 Microsoft Office 제품(줄여서 Office 문서) 기반 도구가 있으며 언뜻 보기에는 매력적으로 보일 수 있습니다. 핵심적으로 볼 때 Office 문서는 대개 (압축될 수 있는) XML 파일이므로 코드 청크를 그 안에 포함시키는 것이 가능합니다. 우리는 코드 청크를 구문 분석하고 실행한 후 결과를 다시 삽입할 수 있습니다.

저희가 보는 주요 문제는 XML 형식이 너무 복잡하고 기준이 너무 많아서 수정된 문서가 여전히 유효한 Office 문서인지 확인하는 것이 간단하지 않다는 것입니다. 한 가지 예로, StatWeave 패키지(http://homepage.stat.uiowa.edu/~rlenth/StatWeave/)는 "OpenOffice가 수정된 문서를 손상된 것으로 플래그 처리하기" 때문에 더 이상 OpenOffice(3.2 이상)에서 작동하지 않습니다.

이에 반해 일반 텍스트 파일은 다루기가 훨씬 더 쉽습니다. 신경 써야 할 ECMA-376과 같은 복잡한 기준이 없습니다. 그래도 Office 문서를 원한다면 최소한 마크다운에서 변환할 가능성은 열려 있습니다. 1장에서 인용한 것을 기억하십시오:

소스 코드는 진짜다(The source code is real).

### A

###### 내부

이 부록에서는 knitr 패키지의 몇몇 내부 구조를 설명합니다. 이는 다른 개발자가 이 패키지를 더 잘 이해하고 필요할 때 코드를 기여하는 데 도움이 될 수 있습니다. 일반 사용자는 이 부록을 읽을 필요가 없습니다. 문서화(documentation), 클로저(closures) 적용 및 일부 기능의 구현 등 세 가지 측면에서 내부 구조를 설명합니다.

###### A.1 문서화

knitr에는 R 설명서(Rd), PDF 매뉴얼, 웹사이트 등 세 가지 유형의 문서화 방식이 있습니다.

R 설명서는 roxygen2(Wickham 등, 2015)를 기반으로 하며, roxygen 주석(#') 안에 태그를 달아서 Rd를 작성할 수 있습니다. 이 주석들은 실제 Rd로 변환됩니다. 다음은 roxygen 주석의 예입니다.

# @author Yihui Xie

이것은 Rd로 변환되면 다음과 같습니다. \author{Yihui Xie}

roxygen에는 @usage, @param, @return 및 @examples와 같은 일련의 태그가 있으며 이는 Rd의 \usage{}, \arguments{\item{}}, \value{} 및 \examples{}에 해당합니다. 공식 Rd에 비해 roxygen 주석을 사용하는 것의 장점은 설명서와 소스 코드를 같은 파일에 유지할 수 있다는 것입니다. 이에 반해 R 패키지를 작성하는 공식적인 접근 방식은 R 소스를 R/ 디렉터리 아래에 작성하고 매뉴얼 페이지는 man/ 아래에 \*.Rd 파일로 작성하는 것입니다. 이 방식은 두 파일 사이를 왔다 갔다 해야 하므로 불편하며, R 소스를 업데이트하고 설명서를 업데이트하는 것을 잊어버리기 쉽습니다. roxygen 주석은 소스의 R 함수 바로 위에 표시되므로 소스와 설명서를 모두 유지 관리하기가 훨씬 쉽습니다.
하지만 왜 모든 문서에 Rnw 구문을 사용하지 않는 것일까요? 그 결정은 저술 형식에 따라 더 자연스러운 구문을 사용하고 싶었기 때문입니다. < <> >=는 어떤 문서 형식에서도 유효한 마크업이 아닙니다(LATEX 명령도 아니고 HTML 태그도 아닙니다). 사실 Sweave에는 LATEX와 유사한 다른 구문 세트가 있습니다. 예:

\begin{Scode}{fig = TRUE, echo = FALSE} library("graphics") boxplot(Ozone ~ Month, data = airquality) \end{Scode}

저는 청크 옵션을 지정할 때 {}보다 []를 더 선호하며, 이는 LATEX에서 더 자연스러운 선택일 것입니다. 어찌되었든, < <> >=는 인기가 많아 knitr에 그대로 남게 되었습니다.

역사적인 이유로 인한 Rnw 문서를 제외하고, 다른 형식들은 R 코드가 실행되기 전에도 knitr 소스 문서가 여전히 유효한 문서가 되도록 합니다. 예를 들어, R HTML 문서의 R 코드는 HTML 주석(<!-- -->) 안에 배치됩니다.

###### 참고 문헌

Adler, D. 및 Murdoch, D. (2014). rgl: 3D visualization device system (OpenGL). R package version 0.95.1201.

Allaire, J., Cheng, J., Xie, Y., McPherson, J., Chang, W., Allen, J., Wickham, H., 및 Hyndman, R. (2015a). rmarkdown: Dynamic Documents for R. R package version 0.5.1.

Allaire, J., Horner, J., Marti, V., 및 Porte, N. (2015b). markdown: Markdown Rendering for R. R package version 0.7.7.

Auguie, B. (2013). cda: Coupled dipole approximation in electromagnetic scattering. R package version 1.3.3.

Baggerly, K. A., Morris, J. S., 및 Coombes, K. R. (2004). Reproducibility of seldi-tof protein patterns in serum: comparing datasets from different experiments. Bioinformatics, 20(5):777–785.

Bracken, C. 및 Sharpsteen, C. (2012). pgfSweave: Quality speedy graphics compilation and caching with Sweave. R package version 1.3.0.

Buckheit, J. 및 Donoho, D. (1995). Wavelab and reproducible research. Wavelets and Statistics, 103:55.

Chang, W., Cheng, J., Allaire, J., Xie, Y., 및 McPherson, J. (2015). shiny: Web Application Framework for R. R package version 0.11.1.

Dahl, D. B. (2014). xtable: Export tables to LaTeX or HTML. R package version 1.7-4.

Eddelbuettel, D., Francois, R., Allaire, J., Ushey, K., Bates, D., 및 Chambers, J. (2015). Rcpp: Seamless R and C++ Integration. R package version 0.11.5.

Ellson, J., Gansner, E., Koutsofios, L., North, S., 및 Woodhull, G. (2002). Graphviz — open source graph drawing tools. In Graph Drawing, pages 483–484. Springer-Verlag.

Falcon, S. (2013). weaver: Tools and extensions for processing Sweave documents. R package version 1.26.0.

Fomel, S. 및 Claerbout, J. (2009). Guest editors’ introduction: Reproducible research. Computing in Science & Engineering, 11(1):5–7.

Francois, R. (2013). highlight: Syntax highlighter. R package version 0.4.3.

Friedl, J. (2006). Mastering Regular Expressions. O’Reilly Media, Incorporated.

Gentleman, R. (2005). Reproducible research: A bioinformatics case study. Statistical Applications in Genetics and Molecular Biology, 4(1):1034.

Gentleman, R. 및 Temple Lang, D. (2004). Statistical analyses and reproducible research. Bioconductor Project Working Papers. URL: http://biostats.bepress.com/bioconductor/paper2.

Gove, J. H. (2013). sampSurf: Sampling Surface Simulation for Areal Sampling Methods. R package version 0.6-8.

Gruber, J. (2004). The Markdown Project. URL: http://daringfireball.net/projects/markdown/.

Guo, J., Betancourt, M., Brubaker, M., Carpenter, B., Gao, Y., Goodrich, B., Hoffman, M., Lee, D., Li, P., Malecki, M., 및 Gelman, A. (2014). rstan: RStan: R interface to Stan. R package version 2.5.0.

Harrell, Jr., F. E. (2001). Regression Modeling Strategies: With Applications to Linear Models, Logistic Regression, and Survival Analysis. Springer New York.

Harrell, Jr., F. E. (2015). Hmisc: Harrell Miscellaneous. R package version 3.15-0.

Horner, J. (2011). brew: Templating Framework for Report Generation. R package version 1.0-6.

Horton, N., Aloisio, K., Zhang, R., 및 Loi, L. (2012). The statistical sleuth (2nd edition) in R. URL: http://www.math.smith.edu/~nhorton/sleuth/.

Huang, Y. 및 Gottardo, R. (2013). Comparability and reproducibility of biomedical data. Briefings in Bioinformatics, 14(4):391–401.

Ihaka, R. 및 Gentleman, R. (1996). R: A language for data analysis and graphics. Journal of Computational and Graphical Statistics, 5(3):299–314.

Jockers, M. L. (2014). Text Analysis with R for Students of Literature. Springer.

- Knuth, D. E. (1983). The WEB system of structured documentation. Technical report, Department of Computer Science, Stanford University.
- Knuth, D. E. (1984). Literate programming. The Computer Journal, 27(2):97–111.

Lebanon, G. (2012). Probability: The Analysis of Data, volume 1. CreateSpace Independent Publishing Platform.

Lecoutre, E. (2014). R2HTML: HTML exportation for R objects. R package version 2.3.1.

Leisch, F. (2002). Sweave: Dynamic generation of statistical reports using literate data analysis. In COMPSTAT 2002 Proceedings in Computational Statistics, number 69, pages 575–580. Heidelberg: Physica Verlag.

Lenth, R. V. 및 Højsgaard, S. (2007). Sasweave: Literate programming using sas. Journal of Statistical Software, 19(8):1–20.

Murdoch, D. (2012). tables: Formula-driven table generation. R package version 0.7.

Murphy, D. (2012). Changes and additions to ggplot2 0.9.0. URL: https://github.com/djmurphy420/ggplot2-transition-guide.

Murrell, P. (2011). R Graphics, Second Edition. Chapman & Hall/CRC.

Murrell, P. 및 Ripley, B. (2006). Non-standard fonts in PostScript and PDF graphics. R News, 6(2):41–47.

Oetiker, T., Partl, H., Hyna, I., 및 Schlegl, E. (1995). The not so short introduction to LATEX2ε. URL: http://www.ctan.org/tex-archive/info/lshort/.

Peng, R. (2009). Reproducible research and biostatistics. Biostatistics, 10(3):405–408.

Peng, R. D. (2012). cacheSweave: Tools for caching Sweave computations. R package version 0.6-1.

Qiu, Y. 및 Xie, Y. (2015). highr: Syntax Highlighting for R Source Code. R package version 0.5.

Qiu, Y., Xie, Y., 및 Bracken, C. (2015). R2SWF: Convert R Graphics to Flash Animations. R package version 0.9.

- R Core Team (2014). R Language Definition. R Foundation for Statistical Computing, Vienna, Austria.
- R Core Team (2015). R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.

Ramsey, F. 및 Schafer, D. (2002). The Statistical Sleuth: A Course in Methods of Data Analysis, Second Edition. Duxbury Press.

Ramsey, N. (1994). Literate programming simplified. Software, IEEE, 11(5):97–105.

Rossini, A. (2002). Literate statistical analysis. In Proceedings of the 2nd International Workshop on Distributed Statistical Computing, pages 15– 17, Vienna, Austria.

Rossini, A., Heiberger, R., Sparapani, R., Maechler, M., 및 Hornik, K. (2004). Emacs speaks statistics: A multiplatform, multipackage development environment for statistical analysis. Journal of Computational and Graphical Statistics, 13(1):247–261.

Schulte, E., Davison, D., Dye, T., 및 Dominik, C. (2012). A multilanguage computing environment for literate programming and reproducible research. Journal of Statistical Software, 46(3):1–24.

Sharpsteen, C. 및 Bracken, C. (2015). tikzDevice: R Graphics Output in LaTeX Format. R package version 0.8.1.

Tantau, T. (2008). The TikZ and PGF Packages. URL: http://sourceforge.net/projects/pgf/.

Tantau, T., Wright, J., 및 Miletic, V. (2012). User’s Guide to the Beamer Class. URL: http://bitbucket.org/rivanvx/beamer.

Temple Lang, D., Swayne, D., Wickham, H., 및 Lawrence, M. (2014). rggobi: Interface between R and GGobi. R package version 2.1.20.

Vaidyanathan, R. (2012). slidify: Generate reproducible html5 slides from R markdown. R package version 0.4.5.

Vaidyanathan, R., Cheng, J., Allaire, J., Xie, Y., 및 Russell, K. (2014). htmlwidgets: HTML Widgets for R. R package version 0.3.2.

van Heesch, D. (2008). Doxygen: Source code documentation generator tool. URL: http://www.doxygen.org/.

Venables, W. N. 및 Ripley, B. D. (2002). Modern Applied Statistics with S. Springer-Verlag, 4th edition.

Wei, T. (2013). corrplot: Visualization of a correlation matrix. R package version 0.73.

Wickham, H. (2015). evaluate: Parsing and Evaluation Tools that Provide More Details than the Default. R package version 0.7.

Wickham, H., Danenberg, P., 및 Eugster, M. (2015). roxygen2: In-Source Documentation for R. R package version 4.1.1.

- Xie, Y. (2013). runr: Run External Programs from R. R package version 0.0.6.
- Xie, Y. (2014). printr: Automatically Print R Objects According to knitr Output Format. R package version 0.0.3.
- Xie, Y. (2015a). formatR: Format R Code Automatically. R package version 1.2.

- Xie, Y. (2015b). knitr: A General-Purpose Package for Dynamic Report Generation in R. R package version 1.10.
- Xie, Y. (2015c). servr: A Simple HTTP Server to Serve Static Files or Dynamic Documents. R package version 0.2.

Yin, T., Cook, D., 및 Lawrence, M. (2012). ggbio: an R package for extending the grammar of graphics for genomic data. Genome Biology, 13(8):R77.

- 그림 15.17: 책 표지 이미지

통계학

초보자와 숙련된 사용자 모두에게 적합한 『R과 knitr를 이용한 동적 문서(Dynamic Documents with R and knitr) 제2판』은 연산과 보고를 직접 통합하여 통계 보고서 작성을 더욱 쉽게 만들어 줍니다. 보고서의 범위는 과제, 프로젝트, 시험, 책, 블로그, 웹 페이지부터 통계 그래픽, 연산 및 데이터 분석과 관련된 거의 모든 문서에 이릅니다. 이 책은 초보자를 위한 기본적인 활용법을 다루는 한편, 고급 사용자가 knitr 패키지의 확장성을 이해할 수 있도록 안내합니다.

###### 제2판의 새로운 점

- R 마크다운 v2를 소개하는 새로운 장
- knitr 패키지의 개선 사항을 반영한 변경 사항
- 표 생성, 코드 청크 내 객체에 대한 사용자 정의 출력 방식 정의, C/Fortran 엔진, Stan 엔진, 지속적인 세션에서 엔진 실행, 그리고 동적 문서를 제공하기 위한 로컬 서버 시작에 대한 새로운 절

호평을 받은 이전 판과 마찬가지로, 이 개정판은 보고서 작성 시 효율성을 높이는 방법을 보여줍니다. 이 책은 프로그램 출력부터 출판 품질의 보고서 작성까지 안내하여 보고서의 모든 측면을 미세 조정할 수 있도록 도와줍니다. 패키지에 대한 데모 및 기타 정보는 저자의 웹사이트에서 확인할 수 있습니다.

Yihui Xie는 RStudio의 소프트웨어 엔지니어입니다. 아이오와 주립대학교 통계학과에서 박사 학위를 받았습니다. 대화형 통계 그래픽과 통계 연산을 중점적으로 연구하고 있습니다. 활발한 R 사용자이며 여러 차례 수상 경력이 있는 R 패키지의 저자이기도 합니다. 또한 중국 내 대규모 온라인 통계 커뮤니티인 "Capital of Statistics"의 설립자입니다.

K25425

w w w . c r c p r e s s . c o m

제2판

Dynamic Documents with R and knitr

Xie

## The R Series

# Dynamic Documents with R and knitr

제2판

Yihui Xie

K25425_cover.indd 1 4/17/15 11:01 AM
