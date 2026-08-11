### 16

###### 기타 도구

knitr 외에도 동적 문서를 위한 다수의 다른 도구들이 존재합니다. 일부는 R 패키지이며, 다른 일부는 Python 및 awk와 같은 타 언어 기반의 도구입니다. 본 장에서는 이러한 도구들에 대한 간략한 개요와 함께 knitr와의 비교를 제공하며, 특히 Sweave 사용자를 위해 Sweave와 knitr의 차이점을 중점적으로 설명합니다.

###### 16.1 Sweave

knitr 패키지는 R 환경에서 동적 문서를 위한 주요 도구로 오랫동안 자리 잡아 온 Sweave (Leisch, 2002)에 큰 영향을 받았습니다. Sweave는 기본 R에 포함되어 있습니다 (utils 패키지의 Sweave() 함수 형태). Sweave는 주로 Rnw 문서를 처리하지만, 다른 문서 형식으로 확장할 수 있는 모듈식 설계를 갖추고 있습니다. CRAN에는 Sweave를 기반으로 한 다수의 확장 프로그램이 존재하며, 다음 절에서 이를 소개하겠습니다.

Sweave를 실행하는 방법은 두 가지가 있습니다. 첫째, 대화형 R 세션에서 호출할 수 있습니다 (utils 패키지를 로드할 필요는 없습니다):

Sweave("your_file.Rnw") # your_file.tex를 생성합니다

또한, 명령줄을 사용할 수도 있습니다. R CMD Sweave your_file.Rnw

Sweave는 기본 R의 일부이기 때문에 최근 몇 년간 그 발전이 거의 정체된 상태입니다. 또 다른 주요 문제는 모듈식 설계가 충분히 모듈화되어 있지 않아, 기본 R에서 Sweave가 업데이트될 때 확장 프로그램들이 호환되지 않을 수 있다는 점입니다. 알려진 바에 따르면, Sweave를 기반으로 하는 일부 R 패키지들은 Sweave의 핵심 코드 상당 부분을 복사하여 사용하고 있으며, 이는 더 이상 Sweave의 개발과 동기화되지 않습니다.

knitr의 청크 옵션 중 다수는 Sweave에서 차용되었습니다 (eval, echo, results 등). 하지만 설계가 다르기 때문에 이들 사이에는 몇 가지 차이점이 존재합니다. 1.0 버전 이전의 knitr는 Sweave와의 호환성을 유지하기 위해 노력했습니다. 내부 함수들을 통해 차이점을 자동으로 수정함으로써 knitr가 Sweave 문서를 컴파일할 수 있었습니다. 하지만 v1.0부터 이러한 호환성 지원은 중단되었으며, 대신 Sweave 문서를 knitr로 수동 변환할 수 있는 Sweave2knitr() 변환 함수가 제공됩니다. 아래는 utils 패키지의 Rnw 문서를 변환하고 그 결과를 보여주는 예시입니다 (< 는 원본 문서를, > 는 변환된 파일을 나타냅니다):

testfile <- system.file("Sweave", "Sweave-test-1.Rnw",

package = "utils") outfile <- tempfile(fileext = ".Rnw") Sweave2knitr(testfile, output = outfile) # true/false를 대문자 TRUE/FALSE로 변경합니다. # _ fig=true # 불필요한 fig=TRUE 옵션을 제거합니다. # _ fig=TRUE # _ fig=TRUE # results 옵션에 따옴표를 추가합니다. # _ results=hide # 'print', 'term', 'prefix' 옵션을 제거합니다. # _ print=TRUE # _ echo=TRUE,print=TRUE # true/false를 대문자 TRUE/FALSE로 변경합니다. # _ echo=true # \SweaveOpts{}를 opts_chunk$set()으로 변경합니다. # _ \SweaveOpts{echo=FALSE} # _ \SweaveOpts{echo=true} # 추가된 줄을 제거합니다 (#n은 줄 번호를 나타냅니다): # _ (#69) @ cat(system(sprintf("diff %s %s", shQuote(testfile),

shQuote(outfile)), intern = TRUE), sep = "\n")

# 7c7,14 # < \SweaveOpts{echo=FALSE} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=FALSE

# > ) # > @ # > # 15c22 # < <<print=TRUE>>= # --# > <<>>= # 17c24 # < <<results=hide>>= # --# > <<results= hide >>= # 22c29 # < <<echo=TRUE,print=TRUE>>= # --# > <<echo=TRUE>>= # 43c50,57 # < \SweaveOpts{echo=true} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=TRUE # > ) # > @ # > # 53c67 # < <<fig=TRUE>>= # --# > <<>>= # 63c77 # < <<fig=true>>= # --# > <<>>= # 69d82 # < @

###### 16.1.1 구문

기본적으로 knitr는 청크 옵션을 구문 분석하기 위해 R 함수 인수와 유사한 새로운 유형의 구문을 사용합니다. 이는 기존의 Sweave 구문보다 훨씬 더 강력한 기능을 제공합니다. 청크 옵션에 임의의 객체를 사용할 수 있으며, R의 모든 기능을 활용할 수 있습니다.

Sweave는 청크 옵션을 문자열로 취급하고 쉼표로 분할하여 구문 분석하는 반면, knitr는 R 구문을 사용합니다. 만약 옵션이 문자값을 취한다면, R에서 하듯 따옴표로 묶어야 합니다 (results = 'hide' - Sweave에서는 results = hide로 작성합니다). 청크 옵션에서 직접 연산을 수행하는 예시는 12.1.3절을 참조하시기 바랍니다. 아래는 새로운 구문이 얼마나 유연한지 보여주는 또 다른 예시입니다 (그림 캡션을 동적으로 생성할 수 있습니다):

<<cap, fig.cap=paste('The P-value is', t.test(x)$p.value)>>= x <- rnorm(100) boxplot(x) @

구문상의 또 다른 사소한 차이점은, knitr의 경우 청크 헤더가 앞에 있지 않는 한 @를 텍스트 청크의 시작으로 인식하지 않는다는 것입니다. 예를 들어, 아래 예시에서 knitr는 첫 번째 @를 유지하지만, Sweave는 이를 제거합니다.

text @ <<A>>= 1 + 1 @

Sweave2knitr()는 이 문제를 자동으로 해결할 수 있습니다.

###### 16.1.2 옵션

Sweave의 일부 옵션은 knitr에서 제외되거나 다음과 같이 변경되었습니다.

concordance는 주로 RStudio를 지원하기 위해 변경되었습니다. 패키지 옵션 opts_knit$get('concordance')가 TRUE이면, 출력 줄 번호가 입력 줄 번호에 매핑된 inputconcordance.tex라는 파일이 생성됩니다. 이 구현은 Sweave보다 정밀도가 다소 떨어집니다.

keep.source는 더 유연한 옵션인 tidy로 통합되었습니다.

print는 제외되었습니다. R 표현식의 출력 여부는 R 사용 경험과 일치하게 설계되었습니다 (예를 들어, x <- 1은 출력되지 않지만 1:10은 출력됩니다. R 콘솔에 명령어를 입력하는 상황을 가정해 보십시오). 표현식의 출력을 숨기고 싶으시다면 invisible() 함수를 사용할 수 있습니다.

term은 제외되었습니다 (term = TRUE로 간주합니다). prefix는 제외되었습니다 (prefix = TRUE로 간주합니다). prefix.string은 fig.path로 이름이 변경되었으며 항상 그림 파일 이름에 사용됩니다.

eps, pdf 및 그래픽 장치에 대한 모든 논리적 옵션이 제외되었습니다. 대신 새로운 옵션인 dev를 사용합니다. 이는 Sweave의 grdevice와 유사하지만 20개 이상의 사전 정의된 그래픽 장치를 지원합니다 (7장을 참조하시기 바랍니다).

fig는 제외되었습니다. 이제 fig.keep을 사용합니다. knitr의 fig.keep = 'high'는 Sweave의 fig = TRUE와 동일하며, fig.keep = 'none'은 fig = FALSE와 동일합니다.

width와 height는 각각 fig.width와 fig.height로 이름이 변경되었습니다. 한편, \SweaveOpts{}와 \SweaveInput{}는 더 이상 권장되지 않습니다. 전역 청크 옵션을 설정하고 하위 문서를 포함하려면 각각 opts_chunk$set()와 청크 옵션 child를 사용하시기 바랍니다.

논리적 옵션의 경우 TRUE/FALSE/T/F만 지원되며(앞의 두 가지를 권장합니다), true/false는 작동하지 않습니다. 예를 들어, eval = FALSE는 가능하지만 eval = false는 작동하지 않습니다 (우연히 논리값 FALSE를 갖는 false라는 이름의 R 객체가 존재하지 않는 한 작동하지 않습니다). <<label>> 구문을 사용한 청크 참조는 여전히 사용 가능하며, ref.label과 같은 새로운 옵션을 사용하는 등 청크를 재사용하는 다른 방법들도 있습니다. 9장에서 소개한 바와 같이 청크 참조는 재귀적으로 수행될 수 있습니다.

###### 16.1.3 문제점

Sweave에서 알려진 일부 문제들과 자주 묻는 질문들이 knitr에서는 다음과 같이 해결되었습니다.

- • 빈 그림 청크는 Sweave에서 LATEX 오류를 발생시키지만 knitr에서는 발생시키지 않습니다. 이는 그림이 아예 생성되지 않기 때문입니다. knitr는 청크 내에 플롯이 있을 때만 그림을 LATEX에 작성합니다.

- • lattice (및 ggplot2) 그래픽은 명시적으로 print()를 호출하지 않으면 Sweave에서 작동하지 않습니다. 하지만 knitr에서는 R 콘솔에서와 마찬가지로 작동합니다 (이러한 플롯 객체가 최상위 환경에 나타나면 출력할 필요가 없습니다).
- • Sweave에서는 LATEX 스타일 Sweave.sty에 정의된 \setkeys{Gin}{width=.8\textwidth}를 통해 출력되는 그림의 너비가 기본적으로 .8\textwidth로 설정됩니다. 이는 Sweave에 의해 생성되었는지 여부와 관계없이 문서의 모든 그림에 영향을 미치며, 개별 그림의 너비를 설정하는 직관적인 방법이 없습니다. 이 문제는 knitr의 out.width 옵션으로 해결되었습니다.

- • Sweave에서는 기본적으로 하나의 그림 청크에서 여러 그림을 생성하는 것이 작동하지 않으므로, 이 경우 사용자가 직접 LATEX 코드를 작성해야 합니다. knitr의 경우 하나의 청크에 플롯이 몇 개가 있든 관계없이 정상적으로 작동합니다.

- • knitr에서는 출력 후크를 사용하여 출력 형식을 변경하는 것이 가능하므로, Sweave의 Sinput/Soutput과 같이 하드코딩된 LATEX 환경을 사용할 필요가 없습니다. 실제로 render_sweave()를 호출하여 knitr에서 Sweave 스타일을 렌더링할 수 있습니다.

- • knitr를 사용하면 HTML 출력을 쉽게 생성할 수 있습니다 (R HTML 또는 R Markdown 사용). 반면 Sweave는 R2HTML과 같은 HTML 전용 확장이 필요합니다.

Sweave를 실행한 후 종종 불필요한 Rplots.pdf 파일을 발견하게 되는데, 이는 비대화형 R 세션에 대한 R의 기본 그래픽 장치가 pdf()로 설정되어 있어 Rplots.pdf를 생성하기 때문입니다. knitr에서는 기본 장치가 null 장치(pdf(file = NULL))로 설정되어 있어 불필요한 PDF 파일이 생성되지 않습니다.

###### 16.2 기타 R 패키지

아래 소개되는 R 패키지(R2HTML 제외)와 Sweave의 대부분의 기능은 knitr에서 지원되므로, 이 섹션은 주로 역사적 참고를 위해 제공됩니다.

highlight 패키지(Francois, 2013)는 Rnw 문서의 R 코드에 대한 구문 강조 기능을 제공합니다. 다음의 pgfSweave, cacheSweave, R2HTML과 마찬가지로 highlight는 Sweave를 기반으로 확장되었습니다. 초기 버전(v0.6 이전)의 knitr는 구문 강조를 위해 highlight에 의존했지만, 유지 관리 문제와 추가 종속성(Rcpp 및 parser 패키지) 문제로 인해 이후 이 종속성은 제거되었습니다. 현재 knitr는 자체 구문 강조 함수를 사용하며, 이는 R 3.0.0 이전에는 정규 표현식을 기반으로 했고, R 3.0.0 이후에는 기본 R의 utils 패키지에 있는 getParseData() 함수에 의존합니다. highlight와 유사한 기능을 얻으려면 knitr에서 청크 옵션 highlight = TRUE를 사용하면 됩니다.

cacheSweave 패키지(Peng, 2012)는 Sweave에 캐시 시스템이라는 중요한 기능을 추가했습니다. weaver 패키지(Falcon, 2013) 역시 다른 구현 방식을 통해 유사한 기능을 수행했습니다. 청크 옵션 cache 및 dependson이 추가되었으며, 이는 knitr에서와 동일한 의미를 갖습니다 (8장을 참조하시기 바랍니다).

pgfSweave 패키지(Bracken and Sharpsteen, 2012)는 highlight와 cacheSweave의 기능을 결합하고, 추가적인 그래픽 지원을 제공했습니다. 특히 플롯도 캐시할 수 있으며, 폰트 스타일의 일관성을 위해 tikzDevice 패키지를 통한 TikZ 그래픽도 지원합니다. 본 서적의 저자는 pgfSweave가 출시되었을 때 Sweave에서 이를 사용하기 시작했고, formatR 지원(tidy 옵션)을 기여하기도 했습니다. 그러나 시간이 지남에 따라 Sweave의 변경 사항을 따라가는 것이 점점 더 어려워졌습니다. 이 패키지는 현재 CRAN 저장소에서 삭제되었습니다. 그럼에도 불구하고, knitr의 설계는 저자가 pgfSweave를 사용하며 얻은 경험으로부터 큰 이점을 얻었습니다.

brew 패키지(Horner, 2011)는 가벼운 템플릿 프레임워크로, 그 구문은 PHP (<?php ?>)와 유사합니다. 기본적으로 템플릿 태그 <% %> 내의 R 코드를 구문 분석하고 실행합니다. 이는 Sweave 및 knitr의 인라인 R 코드와 유사하다고 생각하시면 됩니다. 캐시 시스템이 있지만 그래픽에 대한 직접적인 지원은 제공하지 않습니다. knitr 패키지 역시 brew 구문을 부분적으로 지원합니다(5장에서는 언급하지 않았습니다). 아래는 knitr를 통해 컴파일할 수 있는 예시입니다.

The value of pi is <% pi %>, and 2 times pi is <% 2\*pi %>.

입력 파일의 확장자가 \*.brew인 경우, knitr는 자동으로 brew 구문을 사용합니다. brew는 실제로 여러 인라인 표현식에서 불완전한 코드 조각을 지원하며, 이 점이 PHP와 매우 유사하게 만듭니다. 다음은 brew에서 가져온 예시이지만, knitr는 이를 컴파일할 수 없습니다.

<% for (i in c( 1+1 , 1+pi , 1+pi , sin(pi/2) )) { -%> > <%=i%> <% print(eval(parse(text=i))) %> <% } -%>

R2HTML 패키지(Lecoutre, 2014)는 R 객체를 HTML로 내보내는 수많은 함수를 포함하고 있습니다. 주요 함수는 S3 제네릭 함수 HTML()로, 데이터 프레임, 테이블, lm 객체(lm()에서 반환됨) 등과 같은 다양한 R 객체에 적용할 수 있습니다. 아래는 iris 데이터의 일부를 HTML 표로 변환한 예시입니다.

library(R2HTML) HTML(head(iris[, -5], 1), "", caption = NULL)

<p align= center > <table cellspacing=0 border=1><tr><td>

<table border=0 class=dataframe> <tbody> <tr class= firstline >

<th>&nbsp; </th> <th>Sepal.Length </th> <th>Sepal.Width </th> <th>Petal.Length </th> <th>Petal.Width</th>

</tr>

<tr> <td class=firstcolumn>1 </td> <td class=cellinside>5.1 </td> <td class=cellinside>3.5 </td> <td class=cellinside>1.4 </td> <td class=cellinside>0.2 </td></tr>

</tbody> </table>

</td></table>

R HTML 문서를 위해 knitr 내부에서 R2HTML을 활용할 수 있으며, 원시 HTML 코드를 출력에 쓰기 위해 청크 옵션 results = 'asis'를 설정할 수 있습니다.

R2HTML의 또 다른 주요 기여는 Sweave 확장으로, 사용자가 Sweave를 기반으로 HTML 보고서를 작성할 수 있게 해줍니다.

CRAN에는 재현 가능한 연구(reproducible research)에 대한 작업 뷰(http://cran.r-project.org/web/views/ReproducibleResearch.html)가 존재하며, 이 주제에 대한 더 많은 패키지들을 찾을 수 있습니다.

###### 16.3 Python 패키지

본 섹션에서는 동적 문서를 위한 Python 기반의 세 가지 패키지, 즉 Dexy, PythonTEX, IPython을 소개합니다.

###### 16.3.1 Dexy

Dexy(http://www.dexy.it)는 매우 범용적인 설계를 특징으로 하는 무료 Python 패키지입니다. 공식 웹사이트에 따르면:

Dexy는 코드가 포함된 모든 종류의 기술 문서를 작성하기 위한 자유 형식의 리터럴 문서화 도구입니다. Dexy는 올바른 문서를 작성하고, 코드가 변경됨에 따라 시간이 지나도 쉽게 유지 관리할 수 있도록 지원합니다.

네 가지 주요 특징은 다음과 같습니다.

- 1. 모든 언어 지원 (소스 코드)
- 2. 모든 마크업 지원 (출력)
- 3. 모든 템플릿 지원
- 4. 모든 API 지원 (프로그래밍)

Dexy와 knitr 사이에는 다국어 지원과 같이 분명한 유사점들이 존재합니다. Dexy의 중요한 개념 중 하나는 "필터(filter)"입니다. 필터는 입력 파일을 가져와 출력 파일로 변환하며, 이는 쉘 스크립트의 파이프 | 와 유사합니다. Dexy의 필터는 실제로 knitr 개념들의 조합입니다. 필터는 출력을 렌더링하거나(Markdown을 HTML로), 프로그래밍 언어를 실행하거나(knitr의 언어 엔진과 유사), knitr의 청크 후크와 같은 추가 작업을 수행할 수 있습니다.

일반적으로 Dexy는 컴퓨터 코드와 템플릿을 분리하는데, 이는 장단점이 있습니다. 장점은 소스 스크립트를 재사용할 수 있다는 점이고, 단점은 보고서 환경과 소스 코드 사이를 계속 오가야 한다는 점입니다. 기본적으로 knitr는 코드 청크를 보고서에 직접 포함하지만, 9장에서 소개된 바와 같이 코드 청크를 외부화하는 것도 가능합니다.

###### 16.3.2 PythonTEX

PythonTEX(https://github.com/gpoore/pythontex)는 LATEX 패키지로서, LATEX 내에서 Python 코드를 실행할 수 있는 기능을 제공합니다. 해당 문서에 따르면:

PythonTEX는 LATEX 내에서 Python에 빠르고 사용자 친화적인 접근을 제공합니다. LATEX 문서 내에 입력된 Python 코드를 실행하고, 그 결과를 원본 문서에 포함시킬 수 있습니다. 또한 Pygments 패키지를 통해 LATEX 문서 내 코드에 대한 구문 강조 기능을 제공합니다.

\pyb{} 명령을 사용하여 인라인 Python 코드를 삽입하거나, 다음과 같이 pyconsole 환경을 사용하여 LATEX에서 Python 세션을 에뮬레이션할 수 있습니다.

\begin{pyconsole}[][frame=single]

- x = 123
- y = 345
- z = x + y z def f(expr):

return(expr\*\*4)

- f(x) print( Python says hi from the console! ) \end{pyconsole}

이 문서를 컴파일하면 Python 코드가 평가되고 그 결과가 출력에 삽입됩니다.

Python을 기반으로 하므로 SymPy(기호 조작) 및 matplotlib(플롯)와 같은 다른 Python 패키지들과 연동도 지원합니다.

###### 16.3.3 IPython

IPython(http://ipython.org)은 코드, 텍스트, 수학 수식, 인라인 플롯 및 기타 리치 미디어를 지원하는 웹 기반 노트북과 병렬 컴퓨팅을 위한 고성능 도구 등을 특징으로 하는 Python용 대화형 쉘입니다.

그림 16.1은 Ubuntu 환경의 GNOME 터미널에서 실행되는 IPython의 스크린샷입니다. 터미널에 x.spl<TAB>을 입력하면 아래와 같은 자동 완성을 볼 수 있는 등, 쉘의 기본적인 기능(명령어 자동 완성 등)을 갖추고 있음을 확인할 수 있습니다.

보고서 생성과 관련하여 가장 눈에 띄는 기능은 웹 기반 노트북입니다. 웹 브라우저에서 Python 명령어로 작업하고, 결과(수치 및 그래픽 결과 모두)를 실시간으로 확인할 수 있으며, 노트북에 내용을 추가함에 따라 지속적으로 업데이트할 수 있습니다. 이는 knitr에서 코드 청크를 작성하는 것과 매우 유사합니다.

IPython 노트북은 \*.ipynb 확장자를 가진 JSON 파일로 저장할 수 있어 다른 사람들과 공유할 수 있습니다. 노트북에는 출력이 포함될 수도 있고 포함되지 않을 수도 있습니다. 출력이 없는 노트북은 knitr의 소스 문서(Rnw 및 Rmd 문서)와 유사합니다.

IPython에서 영감을 받아, knitr 역시 유사한 웹 노트북(일부 기능이 간소화됨)을 제공하며, 이는 3.2.2절에서 이미 언급한 바 있습니다.

![image 34](Dynamic Documents with R and knitr 2nd_images/imageFile34.png)
그림 16.1: IPython의 스크린샷: 입력은 In[n ]으로 표시되고, 출력은 Out[n ]으로 표시됩니다.

###### 16.4 더 많은 도구

R 및 Python 패키지 외에도 다른 프로그램에 동적 문서를 위한 도구들이 있습니다. 이 장에서 동적 문서용 도구를 모두 나열하는 것은 불가능합니다. Schulte et al. (2012)는 Javadoc, cweb, noweb, Sweave, SASweave 등 리터럴 프로그래밍과 재현 가능한 연구를 위한 기존 도구들의 목록을 제공한 바 있습니다.

###### 16.4.1 Org-mode

Org-mode는 Emacs 텍스트 편집기(Schulte et al., 2012)에서 구현된 일반 텍스트 마크업 언어입니다. 리터럴 프로그래밍과 재현 가능한 연구(동적 문서의 의미에서)를 모두 지원합니다. 이는 다소 WEB 및 noweb과 같은 초기 리터럴 프로그래밍 구현의 구문을 따르고 있습니다. 즉, 코드 청크와 텍스트 청크("산문(prose)"이라고도 함)의 개념을 가지고 있습니다. Org-mode의 코드 청크는 다음과 같이 보입니다.

#+name: c-chunk
#+begin_src C
int main(){
return 0; }
#+end_src

이에 비해 knitr에서는 동일한 청크를 다음과 같이 작성합니다.

<<c-chunk, engine='c'>>=
int main(){
return 0;
}
@

메타데이터는 청크 헤더에 저장됩니다. Org-mode는 임의의 입력 언어를 지원하며, LATEX 또는 HTML을 출력 형식으로 사용할 수 있습니다.

Schulte et al. (2012)는 기존 도구들의 리터럴 프로그래밍 기능(Sweave는 이를 지원하지 않음)을 언급했지만, 보고서 작성자에게 크게 흥미롭게 들리지 않을 수 있어 본 서적에서는 강조하지 않았습니다. 사실 knitr에도 코드 청크를 재구성하는 기능이 있습니다 (9장을 참조하시기 바랍니다). 아래는 나중에 정의된 청크 B를 앞선 청크 A에 포함하는 간단한 예시입니다.

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

이에 비해 일반 텍스트 파일은 다루기가 훨씬 수월하며, ECMA-376과 같이 고려해야 할 복잡한 표준이 존재하지 않습니다. 만약 Office 문서를 굳이 원한다면, 최소한 Markdown에서 변환하는 방법들이 있습니다. 1장에서 인용했던 문구를 다시 한번 떠올려 보시기 바랍니다.

소스 코드가 진짜입니다.

### A

###### 내부 구조

본 부록에서는 knitr 패키지의 일부 내부 구조를 설명합니다. 이는 다른 개발자들이 이 패키지를 더 잘 이해하고 필요시 코드를 기여하는 데 도움이 될 수 있습니다. 일반 사용자는 이 부록을 읽으실 필요가 없습니다. 문서화(documentation), 클로저(closure)의 적용, 그리고 일부 기능의 구현이라는 세 가지 측면에서 내부 구조를 보여드리겠습니다.

###### A.1 문서화

knitr의 문서화에는 R 문서(Rd), PDF 매뉴얼, 그리고 웹사이트라는 세 가지 유형이 있습니다.

R 문서는 roxygen2(Wickham et al., 2015)를 기반으로 합니다. 이를 사용하면 태그와 함께 roxygen 주석(#')에 Rd를 작성할 수 있으며, 이 주석들은 실제 Rd로 변환됩니다. 아래는 roxygen 주석의 예시입니다.

#' @author Yihui Xie

이 주석은 다음과 같이 Rd로 변환됩니다.
\author{Yihui Xie}

roxygen에는 @usage, @param, @return, @examples와 같은 일련의 태그들이 있으며, 이들은 각각 Rd의 \usage{}, \arguments{\item{}}, \value{}, \examples{}에 대응합니다. 공식 Rd 대신 roxygen 주석을 작성하는 것의 이점은 문서와 소스 코드를 동일한 파일에 보관할 수 있다는 점입니다. 이와 대조적으로, R 패키지를 작성하는 공식적인 접근 방식은 R/ 디렉토리 하위에 R 소스를 작성하고, man/ 하위에 \*.Rd 파일로 매뉴얼 페이지를 작성하는 것입니다. 이 방식은 두 파일 사이를 오가야 하고, R 소스를 업데이트하면서 문서 업데이트를 잊어버리기 쉽기 때문에 불편합니다. Roxygen 주석은 소스의 R 함수 바로 위에 위치하므로, 소스와 문서 모두를 유지 관리하기가 훨씬 쉽습니다.

247

아래는 roxygen 주석으로 문서화된 함수의 완전한 예시입니다.

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

클로저는 본질적으로 함수이며, 비지역(non-local) 변수에 접근할 수 있습니다. 아래는 간단한 예시입니다.

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

사실 클로저를 통해 비지역 변수를 수정하는 것도 가능합니다. 아래는 청크 옵션 관리자인 opts_chunk가 어떻게 작동하는지 보여주는 최소한의 예시입니다.

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

opts_chunk(청크 옵션 관리용)와 knit_engines(언어 엔진 관리용) 객체 외에도 이와 유사한 객체들이 몇 가지 더 존재합니다.

- opts_knit : 패키지 옵션 (12.2절)
- opts_current : 현재 청크의 청크 옵션
- opts_template : 청크 옵션 템플릿 (12.1.2절)
- knit_hooks : 후크 함수 (출력 후크 및 청크 후크 모두)
- knit_patterns : 파서를 위한 구문 패턴 (5.1절)

###### A.3 구현 (Implementation)

이 절에서는 이 패키지의 몇 가지 구현 세부 사항을 설명합니다. 먼저 한 가지 사소한 점을 말씀드리면, 필자는 할당 연산자로 <- 대신 =를 사용하므로 소스 코드 전반에서 =를 보시게 될 것입니다. 이는 개인적인 취향의 문제이며 별다른 단점이 없다고 판단하지만, 이 패키지에 코드를 기여하실 때는 =를 사용해주시기 바랍니다. 본 서적에서는 필자가 등호를 입력했지만 formatR에 의해 자동으로 <-로 대체되었기 때문에 <-를 보게 되실 것입니다.

###### A.3.1 파서 (Parser)

문서 파서(5.1절)는 다음과 같이 작동합니다. 구문 패턴 객체의 하위 요소인 chunk.begin과 chunk.end를 사용하여 문서를 분할하고(코드 청크와 텍스트 청크), 코드 청크의 경우 청크 옵션(즉, 첫 번째 줄에서 추출된 텍스트)이 R 코드로 구문 분석됩니다. 이것이 청크 옵션이 R 구문을 따라야 하는 이유입니다. 다음은 knitr가 텍스트 조각에서 청크 옵션을 가져오는 방법을 설명하는 예시입니다.

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

먼저 텍스트 주변에 alist() 함수를 추가했습니다. 이 함수는 인수들을 마치 함수 인수를 설명하는 것처럼 취급하므로 이 시점에서는 어떤 "인수"도 평가되지 않습니다. 그러나 구문은 최소한 유효해야 합니다. 한 가지 예외는 청크 레이블입니다. 청크 레이블은 문자열이어야 하므로 필요한 경우 자동으로 따옴표로 묶입니다. 청크 옵션을 구문 분석하는 데에는 내부 함수 parse_params()가 사용됩니다.

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

청크 옵션은 청크가 실행되기 직전까지 평가되지 않으므로, 파싱 시점의 문서 내에서 알 수 없는 값을 가진 객체를 청크 옵션으로 사용할 수 있습니다. 예를 들어, 위의 echo와 foo 옵션은 아직 평가되지 않은 표현식이며, 이는 나중에 명시적으로 평가될 것입니다.

eval(opc$echo)

## [1] 1 2 3

eval(opc$foo)

## [1] 2

모든 코드 청크는 내부 객체 knit_code에 이름이 지정된 리스트 형태로 저장됩니다. 이름은 청크 레이블이며 내용은 코드입니다. 이 객체 역시 클로저 목록으로 생성되므로 get() 및 set() 메서드를 가지고 있지만, 예상치 못한 결과를 초래할 수 있으므로 이 객체를 직접 수정하는 것은 권장하지 않습니다. 필요한 경우 knitr:::knit_code$get('chunk-label')을 통해 코드 청크에 접근할 수 있습니다.

###### A.3.2 청크 후크 (Chunk Hooks)

knit_hooks에는 다음과 같이 출력 후크(5.3절)에 해당하는 다수의 기본 후크가 존재합니다.

names(knit_hooks$get(default = TRUE))

## [1] "source" "output" "warning" "message"

## [5] "error" "plot" "inline" "chunk"

## [9] "text" "document"

이 객체 내의 다른 모든 후크는 청크 후크(10장)로 취급됩니다. 코드 청크가 실행되기 전후에 모든 추가 후크가 호출됩니다. 다음은 의사 코드(pseudo code)입니다.

hook(before = TRUE, ...)
evaluate(code)
hook(before = FALSE, ...)

기억해야 할 한 가지 문제는 후크의 실행 순서입니다. 만약 knit_hooks에 정의된 A와 B라는 두 개의 후크가 있다면, 어떤 순서로 호출될까요? 이 순서는 청크 옵션에서 얻어집니다. 해당 두 후크에 대응하는 A와 B라는 두 개의 청크 옵션이 있어야 하며, 청크 옵션의 순서가 후크를 실행할 순서를 결정합니다. 예를 들어 A가 B보다 앞에 있다면 후크 A가 B보다 먼저 호출됩니다. 그러나 코드 청크가 평가된 후에는 그 순서가 역전됩니다. 이는 후크가 반환하는 결과가 짝을 이루도록 하기 위함입니다. 예를 들어, 후크 A가 청크 전에 \begin{Aenvir}를 반환하고 청크 후에 \end{Aenvir}를 반환한다고 가정해 봅시다. 마찬가지로 B는 Benvir를 반환합니다. 그렇다면 출력에서 우리가 원하는 것은 다음과 같습니다.

\begin{Aenvir}
\begin{Benvir}
% 청크의 결과
\end{Benvir}
\end{Aenvir}

\end{Benvir}가 \end{Aenvir}보다 앞에 온다는 점에 유의하시기 바랍니다. 이러한 이유로 후크 A와 B가 정의되었을 때 다음 두 청크는 서로 다른 결과를 반환합니다.

- <<A=TRUE, B=TRUE>>=
- <<B=TRUE, A=TRUE>>=

###### A.3.3 옵션 별칭 (Option Aliases)

리스트 내의 특정 요소를 치환하는 간단한 연산이기 때문에, 청크 옵션 별칭(12.1.1절)을 구현하는 데에는 단 몇 줄의 코드만 필요합니다. 아래는 이러한 아이디어를 보여주는 짧은 함수입니다.

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

- 2. Word 문서를 열고 그림 14.7에 표시된 "스타일" 패널을 찾습니다.
- 3. 스타일을 수정하려는 요소에 커서를 놓으면 스타일 패널에서 해당 항목이 강조 표시됩니다.
- 4. 오른쪽의 ¶ 기호를 클릭하여 항목을 열면 그림 14.8과 같은 창이 나타납니다. 여기서 스타일을 수정할 수 있습니다. 예를 들어 title 요소의 글꼴 계열을 Bookman Old Style로 변경할 수 있습니다.

이 Word 문서의 스타일을 업데이트한 후 Rmd 파일과 같은 디렉터리에 template.docx로 저장하고 이를 참조 문서로 사용할 수 있습니다.

--output:

word_document:

![image 23](Dynamic Documents with R and knitr 2nd_images/imageFile23.png)

###### 그림 14.6: R Markdown v2의 Microsoft Word(2013) 문서 미리보기.

![image 24](Dynamic Documents with R and knitr 2nd_images/imageFile24.png)

- 그림 14.7: Word에서 스타일 패널 열기: 도구 모음에서 "스타일"이라는 창을 찾아 부동 패널로 확장합니다.

reference_docx: template.docx

---

요소의 스타일 외에도 Pandoc 1.13 이상 버전을 사용하면 레이아웃의 스타일도 유지할 수 있습니다. 예를 들어, 참조 문서의 여백, 페이지 크기, 페이지 방향, 머리글 및 바닥글이 새 Word 문서로 그대로 적용됩니다.

###### 14.3.4 Markdown 문서

R Markdown 문서는 Pandoc Markdown, 원본(엄격한) Markdown, Github Flavored Markdown, MultiMarkdown, PHP Markdown Extra 등 다양한 종류의 Markdown 문서로 변환할 수 있습니다. render() 함수에 md_document()를 사용하거나 YAML에서 output: md_document를 지정할 수 있습니다. md_document의 주요 옵션은 variant이며, 이를 통해 원하는 Markdown 종류를 지정합니다.

![image 25](Dynamic Documents with R and knitr 2nd_images/imageFile25.png)

- 그림 14.8: Word에서 요소의 스타일 수정: 글꼴 계열, 글꼴 크기, 글꼴 스타일, 색상 등을 변경할 수 있습니다.

###### 14.3.5 ioslides 프레젠테이션

R Markdown을 사용하여 프레젠테이션용 슬라이드를 만들 수 있습니다. 웹 기술의 발전에 따라 최근에는 HTML5 슬라이드가 인기를 끌고 있습니다. 웹 브라우저에서 슬라이드를 발표할 수 있는데, 이는 슬라이드를 표시하기 위해 별도의 소프트웨어 패키지가 필요하지 않으며 웹 브라우저는 거의 어디서나 사용할 수 있으므로 편리합니다. Microsoft PowerPoint나 Mac용 Keynote와 같은 독점 소프트웨어에서는 그렇지 않습니다.

rmarkdown에는 ioslides와 Slidy라는 두 가지 내장 HTML5 프레젠테이션 형식이 있습니다. 선호하는 HTML5 프레젠테이션 라이브러리를 사용하도록 rmarkdown을 확장할 수 있습니다.

ioslides의 경우 기본적으로 각 첫 번째 수준 섹션 헤더가 어두운 배경의 개별 슬라이드를 생성합니다. 두 번째 수준 헤더는 해당 섹션의 내용을 포함하는 새 슬라이드를 만듭니다. 섹션 제목을 원하지 않으면 세 개의 대시 ---로 새 슬라이드를 만들 수 있습니다.

![image 26](Dynamic Documents with R and knitr 2nd_images/imageFile26.png)

- 그림 14.9: ioslides 프레젠테이션의 제목 슬라이드. RStudio의 목차를 사용하여 슬라이드를 탐색할 수도 있습니다.

- 그림 14.9는 이전 섹션과 동일한 예시 및 YAML 메타데이터를 사용하여 생성된 RStudio 미리보기 창의 ioslides 스크린샷입니다(이 예시를 직접 실행해 보려면 첫 번째 수준 헤더와 두 번째 수준 헤더 사이의 내용을 제거하는 것이 좋습니다).

--output:

ioslides_presentation: default

---

프레젠테이션을 할 때 키보드 단축키 f(F 키를 누름)를 통해 전체 화면 모드를 켤 수 있습니다. W 키는 와이드스크린 모드를 전환합니다. 슬라이드 크기가 너무 크거나 작으면 페이지를 확대/축소할 수 있습니다. 일반적으로 Ctrl(또는 Command) 키를 누른 상태에서 더하기(+) 또는 빼기(-) 키 단축키를 눌러 수행합니다.

ioslides_presentation 형식에는 슬라이드의 모양을 조정할 수 있는 몇 가지 옵션이 있습니다.
incremental (yes/no): 글머리 기호를 점진적으로 표시할지 여부
logo: 슬라이드에서 로고로 사용할 이미지(각 슬라이드의 바닥글에 표시됨)
css: 사용자 지정 CSS 파일

각 슬라이드를 개별적으로 사용자 정의할 수도 있습니다. 예를 들어, 두 번째 수준 섹션 헤더 뒤에 {.build} 토큰을 넣으면 프레젠테이션을 진행할 때 이 페이지의 요소가 점진적으로 표시됩니다. 예:

## A new slide {.build} First show this. Then show that. Finally show a funny GIF animation. ![](foo.png)

HTML5 슬라이드는 대개 인쇄용이 아니라 프레젠테이션용입니다. 하지만 웹 브라우저에서 슬라이드를 PDF로 인쇄할 수도 있습니다. 현재로서는 슬라이드를 인쇄하려는 경우 Google Chrome을 사용하는 것을 권장합니다. 인쇄된 슬라이드의 모양은 화면에 표시된 슬라이드와 다를 수 있습니다.

###### 14.3.6 Slidy 프레젠테이션

Slidy용 슬라이드 작성 규칙은 ioslides와 동일합니다. rmarkdown에서 Slidy 프레젠테이션 출력에 사용되는 함수는 slidy_presentation()입니다.

- 그림 14.10은 R Markdown 예시에서 생성된 Slidy 프레젠테이션의 슬라이드 하나를 보여줍니다.

C를 눌러 목차 보기, S를 눌러 글꼴 작게 하기, B를 눌러 글꼴 크게 하기 등 몇 가지 키보드 단축키를 사용할 수 있습니다.

![image 27](Dynamic Documents with R and knitr 2nd_images/imageFile27.png)

- 그림 14.10: R Markdown 예시에서 생성된 Slidy 프레젠테이션의 한 슬라이드. 하단의 "Contents"를 클릭하여 목차를 표시할 수도 있습니다.

앞서 언급한 incremental 및 css 옵션 외에도, Slidy에는 유용하게 사용할 수 있는 몇 가지 추가 기능이 있으며 다음 옵션이 포함됩니다.
duration: 바닥글에 남은 시간을 알려주는 카운트다운 타이머를 설정합니다. 예를 들어 50분 발표인 경우 YAML에 duration: 50으로 설정할 수 있습니다.
footer: 바닥글에 사용자 지정 메시지를 지정합니다. 예를 들어 소속 기관 이름이나 저작권 정보를 표시할 수 있습니다. Slidy 슬라이드를 인쇄할 때도 Google Chrome을 사용할 수 있습니다.

###### 14.3.7 Beamer 프레젠테이션

12.3.4절에서 소개한 Beamer는 LATEX 애플리케이션이므로, 12.3.4절에 표시된 코드 청크가 포함된 LATEX 문서로 Rnw 파일을 작성하고 PDF 형식으로 직접 컴파일할 수 있습니다. 숙련된 LATEX 사용자를 제외한 모든 사람에게는 Markdown이 더 간단하고 빠르기 때문에 beamer_presentation 형식을 사용하여 작성해 보는 것을 권장합니다. 고급 Beamer 또는 LATEX 기능이 필요한 경우, Pandoc은 Markdown 내에서 LATEX 코드를 지원하므로 Markdown 내에 이를 추가할 수 있습니다.

그림 14.11은 이전 R Markdown 예시로 생성된 Beamer 프레젠테이션의 두 슬라이드를 보여줍니다. 우리가 한 일은 YAML 메타데이터를 다음과 같이 변경한 것뿐입니다.

--title: "R Markdown v2 Demo" author:

- Li Lei

- Han Meimei date: "2015/01/01" output:

beamer_presentation: theme: AnnArbor bibliography: Rmd-v2.bib

---

슬라이드를 원본 LATEX로 작성한다면 소스 문서는 다음과 같을 것입니다.

###### \documentclass{beamer} \usetheme{AnnArbor}

\title{R Markdown v2 Demo} \author{Li Lei \and Han Meimei} \date{2015/01/01}

###### \begin{document} \frame{\titlepage}

\begin{frame}{Start with a cool section} A bit \emph{introduction} here. You can use traditional \textbf{Markdown} syntax, such as \href{http://yihui.name/knitr}{links} and \texttt{code}. \end{frame} \begin{frame}{Followed by another section}

| R Markdown v2 Demo<br><br>Li Lei Han Meimei<br><br>2015/01/01<br><br>Li Lei, Han Meimei R Markdown v2 Demo 2015/01/01 1 / 13<br><br> |
| ------------------------------------------------------------------------------------------------------------------------------------ |

| Pandoc extension: examples<br><br>We have some examples.<br><br>1 Think what is 0.3 + 0.4 - 0.7. Zero. Easy.<br>2 Now think what is 0.3 - 0.7 + 0.4. Still zero?<br><br><br>People are often surprised by (2).<br><br>Li Lei, Han Meimei R Markdown v2 Demo 2015/01/01 9 / 13<br><br> |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

###### 그림 14.11: RMarkdown으로 생성된 Beamer 프레젠테이션의 두 슬라이드: 제목 슬라이드 및 예시 환경의 Pandoc 확장 기능을 보여주는 슬라이드.

물론 목록도 작성할 수 있습니다.

\begin{itemize} \item

apple \item

###### pear \item

banana

\end{itemize}

.... \end{document}

이를 14.3.1절의 R Markdown 소스 코드와 비교해 보면, Markdown으로 작성할 때보다 원본 LATEX로 작성할 때 얼마나 많은 코드를 더 입력해야 하는지 확인할 수 있을 것입니다.

각 새 슬라이드는 Markdown의 새로운 섹션이며, 섹션의 수준은 슬라이드 콘텐츠 바로 앞에 오는 문서 계층 구조의 가장 높은 수준에 의해 결정됩니다. 다음 예시에서는 각 첫 번째 수준 섹션(#)이 새 슬라이드입니다.

--output: beamer_presentation

--# One Section

- - content
- - content # Another Section ![](foo.png)

그리고 이 예시에서는 각 하위 섹션(##)이 새 슬라이드입니다.

- --output: beamer_presentation

--# One Section

## One Sub-section

- content - content

# Another Section ## Another Sub-section ![](foo.png)

목록 항목을 점진적으로 표시하려면 ioslides 및 Slidy 프레젠테이션에서 했던 것처럼 incremental 옵션을 사용할 수 있습니다. toc, highlight, fig_width, fig_height, fig_caption, includes 및 template과 같은 다른 옵션은 이전 절에서 설명했습니다.

Beamer에는 많은 테마(글꼴 테마 및 색상 테마 포함)가 있습니다. theme, fonttheme 및 colortheme 옵션을 통해 이를 사용할 수 있습니다. 그림 14.11에서는 AnnArbor 테마와 기본 글꼴/색상 테마를 사용했습니다. RStudio를 사용하는 경우 GUI에서 이러한 테마를 선택할 수 있으므로 많은 테마 이름을 기억할 필요가 없습니다.

###### 14.3.8 기타 형식

문서 및 프레젠테이션 형식 외에도 rmarkdown에는 두 가지 특수 출력 형식이 있습니다. HTML 패키지 비네트용 html_vignette()(15.4절)와 Tufte 핸드아웃용 tufte_handout()(여기서 Tufte는 Edward R. Tufte를 의미함)입니다.

html_vignette() 형식은 특수 CSS 테마가 있는 html_document()의 래퍼입니다. 기본적으로 Twitter Bootstrap 자산, jQuery 라이브러리 및 highlight.js가 포함되어 있으므로 html_document()로 생성된 HTML 비네트의 파일 크기는 너무 큽니다. html_vignette() 형식은 이러한 모든 구성 요소를 제거하고 가벼운 단일 CSS 파일을 사용합니다. 이미지 파일 크기를 더 줄이기 위해 fig_retina 옵션이 1로 설정되었습니다. 이 형식 함수는 기존 형식 함수를 기반으로 고유한 형식을 구축하는 방법의 좋은 예시이며, 소스 코드는 매우 간단합니다.

html_vignette <- function(fig_width = 3, fig_height = 3, dev = "png", css = NULL,

...) { if (is.null(css)) {

css <- system.file("rmarkdown", "templates",

"html_vignette", "resources", "vignette.css", package = "rmarkdown")

} html_document(fig_width = fig_width,

fig_height = fig_height, dev = dev, fig_retina = FALSE, css = css, theme = NULL, highlight = "pygments", ...)

}

tufte_handout() 형식은 LATEX 문서 클래스 tufte-handout.cls의 래퍼입니다. Tufte 핸드아웃 스타일의 가장 눈에 띄는 특징은 측면 주석(sidenote)의 사용과 잘 설계된 타이포그래피일 것입니다. 예시 페이지는 그림 14.12를 참조하십시오. 이 문서의 YAML 메타데이터는 다음과 같습니다.

--title: "Tufte Handout" author: "John Smith" date: "August 13th, 2014" output: rmarkdown::tufte_handout
