### 14

###### R Markdown

이 책의 초판 이후 R Markdown 개발에 많은 진전이 있었습니다. 명확히 하자면, R Markdown에는 두 가지 버전이 있습니다. markdown 패키지(Allaire 등, 2015b)의 구현체를 "R Markdown v1"(https://github.com/rstudio/markdown)이라고 부르고, rmarkdown 패키지(Allaire 등, 2015a)의 구현체를 "R Markdown v2"(http://rmarkdown.rstudio.com)라고 부릅니다. 달리 명시되지 않는 한, 이 장에서 "R Markdown"이라는 용어는 R Markdown v2를 의미합니다.

R Markdown v1은 C 라이브러리인 sundown을 기반으로 하며 HTML 출력에 주로 초점을 맞춥니다. 인용이나 각주를 지원하지 않는 등 기능이 매우 제한적입니다. R Markdown v2는 Pandoc을 기반으로 하며 Markdown을 완전히 새로운 수준으로 끌어올렸습니다. 여기에는 두 가지 측면의 개선 사항이 있습니다. 우선 Pandoc Markdown 구문이 더 풍부해져서 더 다양한 유형의 요소를 작성할 수 있습니다. 또한 출력 형식이 더 이상 HTML에 국한되지 않으며 Markdown을 LATEX/PDF, Word, HTML5 슬라이드 등으로 내보낼 수도 있습니다. 이 장에서는 rmarkdown의 설계 철학, 기능, 그리고 이를 사용자 정의하거나 확장하는 방법을 소개합니다.

###### 14.1 개요(Overview)

knitr가 다양한 문서 형식을 지원하지만(5장), R Markdown이 아마도 가장 인기 있는 형식일 것입니다. 기능 측면에서 제한적이기는 하지만 Markdown은 초보자에게 훌륭한 문서 언어입니다. 다른 한편으로 작성자는 많은 기능을 원하지 않을 수도 있습니다. LATEX 사용자에게는 Markdown이 제한적으로 보일 수 있지만 모든 사람이 조판 세부 사항에 그렇게 신경 쓰는 것은 아닙니다.

Markdown의 한계는 Pandoc을 통해 크게 해소될 수 있지만, 문제는 Pandoc이 명령줄 도구라는 점입니다. 파워 사용자에게는 이것이 큰 문제가 아닐 수 있지만, 초보자에게는 방대한 명령줄 인수가 부담스러울 수 있습니다.

rmarkdown 및 R Markdown v2의 목표는 적절하고 아름다운 템플릿을 사용하여 R Markdown 파일을 다른 문서 형식으로 빠르게 변환하는 것입니다. 이 목표를 달성하는 방법은 자주 사용되는 명령줄 인수를 rmarkdown의 R 함수로 래핑하는 것입니다. R Markdown 문서를 다른 문서 형식으로 렌더링하기 위한 rmarkdown의 기본 함수는 render()입니다. 첫 번째 인수는 Rmd 파일 이름이고, 두 번째 인수는 출력 형식이며 이에 대해서는 이 장의 뒷부분에서 자세히 소개할 것입니다. 예를 들어 R Markdown 문서 foo.Rmd를 Word로 변환하려면 코드 한 줄만 실행하면 됩니다.

rmarkdown::render("foo.Rmd", "word_document")

물론 더 어려운 방법으로 할 수도 있습니다. 먼저 knitr에서 knit()를 호출하여 foo.Rmd를 foo.md로 컴파일합니다. 그런 다음 터미널을 열거나 R 함수 system()을 사용하여 13.2절에서 소개한 것처럼 다음과 같은 명령을 실행합니다.

pandoc foo.md --output foo.docx \

--from markdown+tex_math_single_backslash \

--highlight-style tango

현재 rmarkdown에는 PDF, HTML, Word, Markdown, ioslides, Slidy, Beamer의 7가지 출력 형식 함수가 있습니다. 처음 네 개는 문서 형식이고 나머지 세 개는 프레젠테이션 형식입니다. 이들은 knitr 및 Pandoc 모두에 대한 래퍼 함수이므로 수많은 knitr 옵션과 Pandoc 인수를 기억할 필요가 없습니다. knitr 청크 옵션과 Pandoc 명령줄 인수는 rmarkdown 함수 인수로 변환됩니다. 예를 들어 Pandoc 인수 --toc 또는 --table-of-contents는 rmarkdown의 함수 인수 toc = TRUE에 해당합니다.

또한 rmarkdown은 기본적으로 시각적으로 즐거움을 주는 자체 템플릿을 제공합니다. 예를 들어 HTML 출력의 경우 Twitter Bootstrap 스타일과 테마를 사용합니다. 프로그램 코드에 대한 구문 강조 기능도 기본적으로 활성화되어 있습니다.

rmarkdown 패키지는 RStudio IDE에서 잘 지원됩니다. render() 함수를 수동으로 호출할 필요 없이 도구 모음에서 Knit 버튼을 클릭하기만 하면 됩니다. 도구 모음의 톱니바퀴 버튼을 통해 나타나는 작은 GUI에서 출력 형식과 해당 옵션을 설정할 수도 있습니다. RStudio 외부에서 rmarkdown을 실행하려면 이 장의 뒷부분에서 rmarkdown의 작동 방식에 대한 자세한 내용을 배우는 것이 좋습니다.

참고로 RStudio에는 Pandoc이 내장되어 있으므로 RStudio를 사용하는 경우 Pandoc을 별도로 설치할 필요가 없습니다. 그렇지 않은 경우에는 Pandoc을 직접 설치해야 합니다. Pandoc을 별도로 설치한 경우 RStudio는 해당 버전이 RStudio의 Pandoc 버전보다 높을 때만 이를 사용합니다.

###### 14.2 Pandoc Markdown 확장

먼저 Pandoc Markdown의 구문을 소개합니다. R Markdown v1에 익숙하다면 Pandoc에서도 해당 구문을 계속 사용할 수 있으며, 유일한 중요한 변경 사항은 수식 요소가 아닌 위첨자를 작성하는 방법입니다. v1에서는 x^2와 같이 단일 캐럿(^)을 사용합니다. Pandoc Markdown에서는 위첨자를 ^로 묶어야 합니다(예: x^2^). 수식의 경우 여전히 캐럿 하나를 사용합니다(예: $x^2$).

###### 14.2.1 기본 구문

다른 요소에 대한 구문은 Pandoc Markdown에서도 거의 동일하게 유지됩니다. 예를 들어 첫 번째 수준의 섹션 헤더를 작성하려면 # 기호를 하나 사용하고, 두 번째 수준 헤더에는 # 기호를 두 개 사용합니다. Markdown의 기본 요소 구문은 5.2.1절을 참조하십시오. 다음은 유용할 수 있는 몇 가지 새로운 요소이며(전체 문서는 http://johnmacfarlane.net/pandoc/ 참조), 글머리 기호 아래에 이러한 요소의 짧은 예시를 보여줍니다.

- • 정의 목록 및 예시 목록 특수 용어: 여기에 용어를 설명/해설합니다.

(@) 번호가 매겨진 예시입니다. (@) 또 다른 번호가 매겨진 예시입니다.

(@cool-example) 이 예시에는 레이블이 있습니다. 이것은 일반적인 단락이며, 여기에서 예시(@cool-example)를 참조할 수 있습니다.

- • ^[...]를 사용한 각주 및 [@id]를 사용한 인용


여기에 X에 대한 멋진 설명을 작성합니다^[Y와 혼동하지 마십시오]. X는 유용합니다.

사실 X에 대해 자세히 알아보려면 참고 문헌[@joe2014]을 읽어야 합니다. 여기서 joe2014는 참고 문헌 데이터베이스의 키입니다.

- • 그림/표 캡션

Pandoc에는 기본적으로 활성화되어 있는 implicit_figures라는 Markdown 확장이 있습니다. 다음과 같은 이미지는

![A figure caption.](path/to/image.png) LaTeX에서 이와 유사한 형식으로 렌더링됩니다. \begin{figure}

\includegraphics{path/to/image.png} \caption{A figure caption.}

\end{figure} 마찬가지로 표 캡션을 추가할 수 있습니다. 예: 표: 이것은 표 캡션입니다.

--- ---- ---A B C

--- ---- ---a 10 bc d 25 ef --- ---- ----

- • 원시 TEX/HTML 내용


가끔 Markdown이 제한적이라고 느껴서 LaTeX를 사용하고 싶은 유혹을 강하게 느낄 수 있습니다. 괜찮습니다. Markdown에 원시 \TeX{} 코드를 작성할 수 있습니다.

Markdown 버전: ![A long caption.](foo.png)

LaTeX 버전: \begin{figure}

\includegraphics[width=.8\textwidth]{foo.png} \caption[A short caption]{A long caption.}

\end{figure} Pandoc은 이 문서를 LaTeX/PDF로 변환할 때 원시 TeX 내용을 보존할 수 있습니다.

인용을 사용할 때는 참고 문헌 데이터베이스를 지정해야 합니다. LATEX에 익숙하다면 BibTEX도 알 가능성이 높습니다. 참고 문헌 데이터베이스는 YAML 메타데이터의 bibliography 필드에 지정된 .bib 파일일 수 있습니다(다음 절 참조). BibTEX를 모르는 경우 bibliography 대신 references 필드를 사용하여 YAML 메타데이터에 참고 문헌 항목을 포함시킬 수 있습니다. 예:

--references:

- - id: joe2014 title: A Nice Paper author:

- family: Smith

given: Joe issued:

year: 2014 container-title: The Journal of Awesome Research type: article-journal

- - id: john1980 title: A Great Book author:

- family: Brown given: John issued:

year: 1980 publisher: An Excellent Publisher type: book

- ---


원시 TEX/HTML 코드를 제외한 모든 다른 요소는 모든 문서 형식에서 호환됩니다. 예를 들어 각주 ^[foo bar]는 출력 형식이 LATEX일 때 \footnote{foo}로 변환되고, 출력 형식이 HTML일 때는 페이지 하단의 각주 항목인 링크 대상 footnote-1과 함께 <a href=”#footnote-1”><sup>1</sup></a> 같은 형식으로 변환됩니다. 원시 TEX 및 HTML 내용은 꽤 복잡할 수 있고 완벽한 변환이 거의 불가능하기 때문에, Markdown의 원시 TEX가 Word로 완벽하게 변환되거나 원시 HTML이 Beamer로 완벽하게 변환될 것이라고 기대해서는 안 됩니다.

###### 14.2.2 YAML 메타데이터

Pandoc Markdown의 또 다른 중요한 확장은 YAML 메타데이터입니다. YAML은 "YAML Ain't Markup Language" 또는 "Yet Another Markup Language"의 약자이며 기본적으로 중첩된 목록 구조입니다. Pandoc은 YAML을 사용하여 제목, 작성자, 날짜 정보와 같은 문서의 메타데이터를 작성합니다. 메타데이터는 일반적으로 문서의 시작 부분에 나타나며 세 개의 대시로 이루어진 두 줄 --- 사이에 위치합니다. 일반적인 YAML 메타데이터는 다음과 같습니다.

--title: "A Nice Report" author: "John Smith" date: 2014/12/31 output:

html_document: toc: yes number_sections: yes

word_document: default

--R Markdown 문서의 본문.

rmarkdown의 YAML 메타데이터에서 가장 중요한 필드는 output 필드입니다. 여기서 원하는 출력 형식을 지정합니다. 이 필드가 누락된 경우 rmarkdown은 출력 형식을 HTML 문서로 간주합니다. 여러 형식이 지정된 경우, render()의 두 번째 인수를 명시적으로 지정하지 않는 한 render() 함수는 기본적으로 첫 번째 형식을 사용합니다. render('foo.Rmd', 'all')을 사용하여 output 필드에 정의된 모든 형식을 렌더링할 수도 있습니다.

###### 14.3 출력 형식

rmarkdown에는 _document 및 _presentation 접미사가 있는 일련의 형식 함수가 있습니다(예: html_document(), pdf_document(), beamer_presentation() 등). 이러한 함수는 render()의 두 번째 인수로 사용할 수 있습니다. 예:

library(rmarkdown) render("foo.Rmd") render("foo.Rmd", pdf_document()) render("foo.Rmd", word_document()) render("foo.Rmd", beamer_presentation()) render("foo.Rmd", ioslides_presentation())

각 출력 형식 함수에는 고유한 인수가 있습니다. 예를 들어 HTML 문서에 목차를 활성화하려면 다음을 호출할 수 있습니다.

library(rmarkdown) render("foo.Rmd", html_document(toc = TRUE))

이것은 YAML 메타데이터를 다음과 같이 제공하는 것과 동일합니다.

--output:

html_document: toc: yes

---

YAML에서 yes와 true는 모두 논리값 TRUE를 의미합니다. YAML 메타데이터를 사용하고 두 번째 인수 없이 render()를 호출하거나, YAML 메타데이터를 생략/무시하고 render()에 두 번째 인수를 명시적으로 제공할 수 있습니다. YAML 접근 방식이 더 편리하고 일반적입니다. 출력 정보가 소스 문서에 포함되기 때문입니다. 두 번째 접근 방식은 YAML에 정의된 출력 형식을 재정의할 때 유용할 수 있습니다. 가능한 옵션이 무엇인지 보려면 각 출력 형식 함수의 도움말 페이지를 참조하십시오. 예를 들어 R 콘솔에 ?rmarkdown::pdf_document를 입력하면 PDF 출력에 대한 옵션을 볼 수 있습니다.

출력 형식 함수는 knitr 패키지/청크 옵션, Pandoc 인수, 기타 rmarkdown용 보조 옵션을 포함한 옵션 목록을 반환합니다. html_document()를 예시로 들어 이들을 설명할 것입니다.

###### 14.3.1 HTML 문서

html_document()가 실제로 무엇을 반환하는지 확인하기 위해 이를 실행하고 반환된 객체의 구조를 인쇄해 볼 수 있습니다.

library(rmarkdown) str(html_document(), width = 55, strict.width = "wrap")

## List of 6 ## $ knitr :List of 3 ## ..$ opts_knit : NULL ## ..$ opts_chunk:List of 5 ## .. ..$ dev : chr "png" ## .. ..$ dpi : num 96 ## .. ..$ fig.width : num 7 ## .. ..$ fig.height: num 5 ## .. ..$ fig.retina: num 2 ## ..$ knit_hooks: NULL ## $ pandoc :List of 5 ## ..$ to : chr "html" ## ..$ from : chr ## "markdown+autolink_bare_uris+ascii_identifiers+te".. ## ..$ args : chr [1:8] "--smart" "--email-obfuscation" ## "none" "--self-contained" ... ## ..$ keep_tex: logi FALSE ## ..$ ext : NULL ## $ keep_md : logi FALSE ## $ clean_supporting: logi TRUE ## $ pre_processor :function (...) ## $ post_processor :function (metadata, input_file, ## output_file, clean, ## verbose) ## - attr(*, "class")= chr "rmarkdown_output_format"

보시다시피 html_document()는 fig.height(knitr의 기본값은 7) 및 fig.retina(원래 기본값은 1)와 같은 knitr 기본 청크 옵션 중 일부를 수정했습니다. 이러한 변경은 미학적인 이유 때문이지만, 어떤 종류의 옵션 값이 더 보기 좋은 결과를 제공하는지 결정하는 것은 다소 주관적입니다.

목록에는 Pandoc 옵션도 포함되어 있습니다. pandoc$to 요소에서 볼 수 있듯이 출력 형식은 html입니다. --smart 및 --self-contained와 같은 몇 가지 Pandoc 인수도 목록에 포함되어 있습니다.

rmarkdown을 위한 몇 가지 보조 옵션도 있습니다. 예를 들어 clean_supporting은 HTML 파일이 렌더링된 후 중간 출력 파일을 정리할지 여부를 의미합니다. 중간 파일에는 그림 파일이 포함될 수 있습니다. HTML 파일을 독립형으로 만들려는 경우 Pandoc은 모든 외부 리소스(예: 이미지)를 그 안에 포함시키므로 이러한 외부 파일이 더 이상 필요하지 않습니다. 이 경우 render()는 HTML 파일을 렌더링한 후 외부 파일을 삭제합니다.

출력 형식 함수의 내부 구조를 알면 서로 다른 knitr/Pandoc 옵션을 사용하여 자체 형식 함수를 작성할 수 있습니다. 사용자 정의 형식을 구현하는 방법은 이 장의 뒷부분에서 소개하겠습니다.

이제 이름이 Rmd-v2.Rmd인 R Markdown v2 문서의 전체 예시를 보여드리겠습니다. 조금 길기는 하지만 Pandoc 및 rmarkdown의 대부분의 기능을 보여줍니다.

--title: "R Markdown v2 Demo" author:

- Li Lei

- Han Meimei date: "2015/01/01" output:

html_document:

fig_caption: yes pdf_document:

template: null

word_document: default bibliography: Rmd-v2.bib

--# Start with a cool section A bit _introduction_ here. You can use traditional **Markdown** syntax, such as [links](http://yihui.name/knitr) and  code . # Followed by another section Of course you can write lists:

- - apple
- - pear
- - banana Or ordered lists:


1. items

1. will

1. be

1. ordered

- - nested
- - items # More sections ## Hi hi hi ## Hello hello hello ## Howdy howdy howdy # Okay, some R code    {r linear-model} fit = lm(dist ~ speed, data = cars) b = coef(fit) # coefficients summary(fit)


코드는 모든 출력 형식에서 강조 표시됩니다. # And some pictures    {r lm-vis, fig.cap= Regression diagnostics } par(mfrow = c(2, 2), pch = 20, mar = c(4, 4, 2, .1),

bg =  white ) plot(fit)

# A little bit math Our regression equation is $Y= r b[1] + r b[2] x$, and the model is:

$$ Y = \beta_0 + \beta_1 x + \epsilon$$ # Pandoc extension: definition lists Programmer : A programmer is the one who turns coffee into code. LaTeX : A simple language with a couple of backslashes. # Pandoc extension: examples We have some examples. (@) Think what is  0.3 + 0.4 - 0.7 . Zero. Easy. (@weird) Now think what is  0.3 - 0.7 + 0.4 . Still zero? People are often surprised by (@weird). # Pandoc extension: tables A table here. Table: Demonstration of simple table syntax.    {r echo=FALSE} knitr::kable(head(iris))

# Pandoc extension: footnotes We can also write footnotes[^1]. [^1]: hi, I m a footnote Or write some inline footnotes^[as you can see here]. # Pandoc extension: citations We compile the R Markdown file to Markdown through **knitr** [@R-knitr] in R [@R-base]. For more about @R-knitr, see <http://yihui.name/knitr>.

![image 20](Dynamic Documents with R and knitr 2nd_images/imageFile20.png)

- 그림 14.1: RStudio 창에서 R Markdown v2로 만든 HTML 출력 문서의 미리보기.


# References    {r include=FALSE} knitr::write_bib(c( base ,  knitr ),  Rmd-v2.bib )

kable() 또는 write_bib()가 어떻게 작동하는지 확실하지 않은 경우 6.3절 및 12.4.1절을 다시 검토해야 할 수 있습니다.

그림 14.1은 RStudio에서 이 예시를 렌더링한 후의 HTML 출력 문서 미리보기입니다. 제목, 작성자, 날짜 및 문서의 처음 몇 개 섹션이 표시됩니다. 이는 rmarkdown의 기본 Twitter Bootstrap 스타일입니다. 그림 14.2는 마지막 몇 개 섹션의 미리보기입니다. 각주와 인용은 HTML의 기본 요소가 아니지만(LATEX 사용자에게는 자연스러울 수 있음), 그럼에도 불구하고 Pandoc은 이를 HTML로 생성해 냈습니다.

HTML 출력에 맞게 미세 조정할 수 있는 수많은 옵션이 있습니다. 전체 목록은 도움말 페이지 ?rmarkdown::html_document를 참조하십시오.

|![image 21](Dynamic Documents with R and knitr 2nd_images/imageFile21.png)|
|---|


###### 그림 14.2: 표, 각주 및 인용의 미리보기: 표는 kable()에 의해 생성되었고 참고 문헌 데이터베이스는 knitr의 write_bib()에서 생성되었습니다.

예를 들어 theme 필드를 사용하여 CSS 테마를 변경하고, toc 필드를 사용하여 목차를 추가하고, YAML의 number_sections 필드를 사용하여 섹션 제목에 번호를 매길 수 있습니다(그림 14.3).

--output:

html_document: fig_caption: yes number_sections: yes theme: readable toc: yes
HTMLS= foo.html bar.html all: $(HTMLS) clean:

rm -rf figure/ *.md %.html: %.Rmd

$(R_HOME)/bin/Rscript -e "knitr::knit2html( $*.Rmd )"

- 그림 15.8: HTML 비네트를 컴파일하기 위한 Makefile: knit2html()을 사용하여 Rmd 문서를 HTML로 컴파일합니다.


게다가 R 3.0.0 이상 버전의 새로운 방식에서는 make 유틸리티를 설치할 필요가 없습니다.

###### 15.4.4 HTML 비네트

유사하게 R Markdown 문서에서 HTML 형식으로 패키지 비네트를 만들 수 있습니다. 다시 말하지만 R 3.0.0 이전에는 Makefile로 HTML 비네트를 컴파일해야 했습니다. 그림 15.8은 knit2html() 함수가 호출된 HTML 비네트 빌드용 Makefile 샘플 소스를 보여줍니다. make clean 명령은 figure/ 디렉터리를 제거하는데, 이는 knitr로 생성된 이미지가 HTML 출력에 base64로 인코딩되어 이미지 파일이 더 이상 필요하지 않기 때문입니다.

###### 15.5 도서

knitr로 책을 쓸 수도 있습니다. 이 책을 쓸 당시 최소 한 권의 책(Lebanon, 2012)이 출판되었으며 Regression Modeling Strategies(Harrell, 2001)라는 책은 knitr를 기반으로 하는 새 버전을 위해 개정 중이었습니다.

###### 15.5.1 이 책에 대하여

"자신의 개밥을 먹는다(eating one's own dog food)"(명확하지 않은 경우 위키백과 참조)라는 정신에 입각하여 이 책은 LYX에서 knitr를 사용하여 작성되었습니다(4.2절 참고). 장을 개별 파일로 분할하는 것도 충분히 가능하지만 책 전체가 하나의 LYX 파일로 되어 있습니다.

속도를 위한 cache = TRUE, 그래픽 스타일을 위한 dev = 'tikz', 플롯 정렬을 위한 fig.align = 'center'와 같은 몇 가지 청크 옵션이 문서 맨 처음에 전역으로 설정되었습니다. 또한 저자는 할당 연산자로 <- 대신 =를 선호하지만 R 사용자는 <-를 더 자주 사용하기 때문에 options(formatR.arrow = TRUE)도 설정했습니다(formatR 패키 참조). 이 옵션을 사용하면 제가 입력한 것이 실제로는 모두 등호지만 적용 가능한 모든 곳에서 등호를 왼쪽 화살표로 자동으로 바꿀 수 있습니다.

이 책에는 다양한 목적을 위한 몇 가지 청크 훅(10장)이 있습니다. 예를 들어 다음과 같이 그래픽 매개변수를 설정하는 par 훅이 있습니다.

par(mar = c(4, 4, 0.1, 0.1), cex.lab = 0.95, cex.axis = 0.9,

mgp = c(2, 0.7, 0), tcl = -0.3, las = 1)

따라서 이 매개변수 세트를 사용하고 싶을 때 매번 다시 입력할 필요 없이 청크 옵션 par = TRUE만 추가하면 됩니다.

이 책에서는 코드 청크와 플롯이 분리되어 있는 것처럼 보이지만 소스 문서에서는 그렇지 않습니다. 코드 청크는 실제로 그림(figure) 환경 내부에 있지만, 문서 훅 hook_movecode()를 사용하여 결국 코드 청크를 그림 환경 밖으로 옮겼습니다.

교육 목적으로 가끔 청크 헤더를 보여주어야 하므로, 청크 출력에 <<>>= 및 @를 추가하기 위해 append라는 청크 훅을 사용했습니다.

knit_hooks$get("append")

## function(before, options, envir) { ## txt = options$append[[ifelse(before, 1, 2)]] ## txt = c("\\begin{alltt}", txt, "\\end{alltt}") ## paste(txt, collapse = "") ## }

기본적으로 이 훅을 사용하면 청크 앞이나 뒤에 추가 문자열을 쓸 수 있습니다. 예를 들어 청크 옵션 append = list('<<A>>=', '@')를 사용하여 청크 출력에 구문 정보를 추가할 수 있습니다. 청크 헤더를 소스 문서에 직접 쓸 수 없기 때문에 이 훅을 사용해야 합니다. 그렇지 않으면 구문 분석되어 최종 출력에서 사라집니다.

플롯에 프레임 상자를 추가하여 기본 플롯 훅 함수를 수정하는 출력 훅이 있으며, 이는 그림 10.3과 그림 10.4에서 사용되었습니다.

모든 R 패키지의 참고 문헌 데이터베이스는 12.4.1절에 소개된 write_bib() 함수에 의해 동적으로 작성되므로 버전 정보가 최신 상태로 유지됨이 보장됩니다(적어도 원고가 출판사에 제출되기 전까지는).

###### 15.5.2 The Analysis of Data

또 다른 주목할 만한 예는 Lebanon(2012)의 책 The Analysis of Data입니다. 이 책의 가장 큰 특징은 PDF 및 HTML 버전을 모두 제공한다는 점입니다. HTML 버전은 http://theanalysisofdata.com에서 무료로 제공됩니다. 두 버전 모두 기본적으로 동일한 소스 문서 세트에서 생성됩니다. HTML 버전의 경우 추가 설정이 있는데, 예를 들어 수식 조판은 MathJax 라이브러리에서 수행하므로 HTML 소스의 head 섹션에 포함시켜야 합니다.

###### 15.5.3 R 기반 The Statistical Sleuth

The Statistical Sleuth(Ramsey 및 Schafer, 2002)는 통계학의 훌륭한 교재이며 이 책의 한 가지 특징은 다수의 데이터세트가 있다는 것입니다. 책 자체는 knitr로 작성되지 않았지만, 다른 저자들(Horton 등, 2012)이 책의 수많은 데이터 분석 예시를 R에서 다시 구현한 웹사이트(http://www.math.smith.edu/~nhorton/sleuth/)를 만들었습니다. 웹사이트에서 PDF 문서와 Rnw 소스 파일을 모두 확인할 수 있습니다.

###### 15.5.4 문학도를 위한 R 기반 텍스트 분석(Text Analysis with R for Students of Literature)

Jockers(2014)의 책 Text Analysis with R for Students of Literature는 LATEX와 knitr를 사용하여 작성되었습니다. 이 책에 대해 가장 놀라운 사실은 아마도 저자가 이 책을 LATEX로 엮기 시작하기 전에 LATEX를 독학했고 단 몇 달 만에 책 초안을 완성했다는 점일 것입니다. 이 책은 컴퓨터 기반 텍스트 분석에 대한 소개서이며 짧은 예시를 많이 포함하고 있습니다. 저자가 각 예시를 실행하고 그 출력을 수동으로 LATEX 원고에 복사해야 했다면 무척이나 지루했을 것입니다.

###### 15.6 R 패키지를 위한 문학적 프로그래밍(Literate Programming)

이 책의 첫머리에 문학적 프로그래밍(LP)을 소개하기는 했지만, 실제로는 프로그래밍 목적으로 knitr 패키지를 사용하지 않습니다. 대신 대부분 데이터 분석 및 보고 목적으로 knitr를 사용합니다. 원래의 LP 패러다임은 위빙(weaving)과 탱글링(tangling)에 모두 관련되어 있습니다. 소스 문서를 엮어(weave) 소프트웨어 문서로 만들거나, 프로그램 코드를 추출(tangle)하여 실행할 수 있습니다. 보아하니 knitr를 사용할 때는 코드 실행이 위빙 과정에서 바로 발생하기 때문에 실행 목적으로 프로그램 코드를 추출(tangle)할 필요가 전혀 없습니다.

흥미롭게도 Knuth의 원래 LP 패러다임이 가장 흔히 적용되는 분야는 패키지 작성자를 위한 "프로그래밍"이라기보다는 사용자를 위해 (특수한 형태의 주석을 사용하여) 소프트웨어를 문서화하는 작업인 것 같습니다. 다시 말해 LP를 사용하여 소스 코드를 문서화하는 대신 소프트웨어 사용법을 문서화하는 것입니다. Doxygen(van Heesch, 2008), Javadoc(http://en.wikipedia.org/wiki/Javadoc), roxygen2(Wickham 등, 2015)를 예로 들 수 있습니다. 하지만 LATEX 세계에는 한 가지 예외가 존재합니다. 일부 LATEX 패키지 작성자는 단일 문서에 LATEX 코드와 문서를 모두 작성한 다음 소스 코드와 문서를 모두 포함하는 PDF 문서로 엮습니다(weave). TEX와 Pascal을 사용한 Knuth의 원래 LP 구현을 생각해 보면 완전히 놀라운 일은 아닙니다. Terry Therneau의 survival 및 coxme 패키지와 같이 LP를 사용하는 R 패키지도 소수 존재합니다.

LP는 프로그래밍에 널리 쓰이는 접근 방식은 아닌 것 같지만 여전히 흥미로운 아이디어이며 자신이 선호하는 언어에 적용할 때 특히 유용할 수 있습니다. LATEX 소스 코드를 읽는 것이 지루한 사람들도 있겠지만, R 소스 코드를 읽는 것은 더 즐거울 수 있습니다. 객관적인 견해는 차치하고라도 우리는 LP가 최소한 두 가지 장점이 있다고 믿습니다.

- 1. 주석으로 하는 일반적인 경우보다 훨씬 더 광범위하고 풍부한 문서를 작성할 수 있습니다. 일반적으로 코드의 주석은 간결하고 일반 텍스트로 제한됩니다(또는 그래야 합니다). 보통 코드 몇 줄을 설명하기 위해 주석을 다섯 단락씩 쓰지는 않으며 주석 안에 읽기 쉬운 수식을 쓰거나 동영상을 삽입할 수도 없습니다.
- 2. 코드 청크에 레이블을 지정하고 레이블을 사용하여 참조하거나 재사용할 수 있으므로 여러 코드 청크 조각을 유연하게 조합하여 프로그램을 구성할 수 있습니다. 예를 들어 문서 뒷부분에서 코드 청크를 정의하고 설명하더라도 레이블을 사용하여 이전 코드 청크에 삽입할 수 있습니다. 이 기능은 Knuth가 강조했지만, 무슨 이유에서인지 널리 채택되지는 않았습니다. 아마도 대부분의 사람들이 코드 청크 대신 함수와 같은 더 작은 단위로 큰 프로그램을 설계하는 것을 더 편안하게 느끼기 때문일 텐데, 이는 사실 좋은 생각입니다.


사실 LP를 R 패키지 개발에 적용할 수 있습니다. 목표를 달성하는 방법은 여러 가지가 있으며, 여기서는 다음 도구를 사용하는 한 가지 방법만 소개합니다.

- 1. 소스 문서에서 프로그램 코드를 추출할 수 있게 해주는 knitr의 purl() 함수
- 2. 프로그램 코드와 문서를 모두 포함할 수 있는 패키지 비네트
- 3. 소스 파일에서 출력 파일을 생성하는 시기와 방법을 정의할 수 있게 해주는 GNU Make


rlp 패키지(https://github.com/yihui/rlp)는 LP 기법을 사용하여 R 패키지를 작성하는 예시입니다. 자세한 내용은 이 저장소에서 확인할 수 있으며, 구현의 기본 아이디어는 다음과 같습니다.

- 1. 패키지의 R/ 디렉터리 아래에 R 소스 코드를 작성하는 대신, vignettes/ 디렉터리 아래의 패키지 비네트(R Markdown)에 코드를 작성할 수 있습니다.
- 2. Makefile을 사용하여 비네트(vignettes/*.Rmd)에서 R 스크립트(R/*.R)를 생성하는 방법을 정의합니다.
- 3. make를 실행하여 R/에 R 스크립트를 생성하고 R CMD build를 실행하여 패키지를 빌드합니다.


RStudio IDE를 사용하면 이러한 단계를 쉽게 수행할 수 있으며, 실제로 버튼 클릭만으로 수행할 수도 있습니다. 구현 세부 사항은 이 책에서 다루기에는 너무 기술적이고 구체적이므로 패키지 문서를 통해 확인하는 것은 독자들의 몫으로 남겨 두겠습니다.

### 16

###### 기타 도구

knitr 외에도 동적 문서를 위한 다수의 다른 도구들이 있습니다. R 패키지도 있고 Python이나 awk 같은 다른 언어의 도구도 있습니다. 이 장에서는 이러한 도구에 대해 간략히 개괄하고 knitr와 비교하며, 특히 Sweave 사용자를 위해 Sweave와 knitr의 차이점을 설명합니다.

###### 16.1 Sweave

knitr 패키지는 R에서 오랫동안 동적 문서를 위한 탁월한 도구로 자리 잡았으며 기본 R의 일부(utils 패키지의 Sweave() 함수)이기도 한 Sweave(Leisch, 2002)에서 큰 동기를 얻었습니다. Sweave는 다른 문서 형식으로 확장할 수 있는 모듈식 디자인도 갖추고 있지만 주로 Rnw 문서를 처리합니다. CRAN에는 Sweave를 기반으로 하는 수많은 확장이 존재하며, 다음 절에서 이를 소개할 것입니다.

Sweave를 실행하는 방법에는 두 가지가 있습니다. 대화형 R 세션에서 호출할 수 있습니다(utils 패키지를 로드할 필요 없음).

Sweave("your_file.Rnw") # gives you your_file.tex

In addition, we can use the command line, too: R CMD Sweave your_file.Rnw

Sweave는 기본 R의 일부이므로 최근 몇 년 동안 거의 개발이 정체 상태에 이른 상황입니다. 또 다른 주요 문제는 모듈식 설계가 충분히 모듈화되지 않아 기본 R에서 Sweave가 업데이트됨에 따라 확장이 호환되지 않을 수 있다는 것입니다. 저희가 아는 한 Sweave를 기반으로 하는 몇몇 R 패키지는 Sweave에서 대량의 핵심 코드를 복사했으며 더 이상 Sweave 개발과 동기화되지 않습니다.

eval, echo, results 등 knitr의 많은 청크 옵션이 Sweave에서 차용되었지만 설계가 다르기 때문에 이들 사이에는 몇 가지 차이점이 있습니다. 버전 1.0 이전의 knitr는 Sweave와 호환되도록 시도했습니다. 자동적으로 차이점을 수정하는 몇 가지 내부 함수 덕분에 knitr는 Sweave 문서를 컴파일할 수 있었습니다. v1.0부터는 호환성이 중단되었으며 Sweave 문서를 knitr용으로 수동 변환하는 Sweave2knitr() 변환 함수가 제공됩니다. 아래는 utils 패키지의 Rnw 문서를 변환하고 변환 후의 차이점을 보여주는 예시입니다(<는 원본 문서를 나타내고 >는 변환된 파일을 나타냅니다).

testfile <- system.file("Sweave", "Sweave-test-1.Rnw",

package = "utils") outfile <- tempfile(fileext = ".Rnw") Sweave2knitr(testfile, output = outfile) # capitalizing true/false to TRUE/FALSE: # * fig=true # removing the unnecessary option fig=TRUE: # * fig=TRUE # * fig=TRUE # quoting the results option: # * results=hide # removing options ’print’, ’term’, ’prefix’: # * print=TRUE # * echo=TRUE,print=TRUE # capitalizing true/false to TRUE/FALSE: # * echo=true # changing \SweaveOpts{} to opts_chunk$set(): # * \SweaveOpts{echo=FALSE} # * \SweaveOpts{echo=true} # removing extra lines (#n shows line numbers): # * (#69) @ cat(system(sprintf("diff %s %s", shQuote(testfile),

shQuote(outfile)), intern = TRUE), sep = "\n")

# 7c7,14 # < \SweaveOpts{echo=FALSE} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=FALSE

# > ) # > @ # > # 15c22 # < <<print=TRUE>>= # --# > <<>>= # 17c24 # < <<results=hide>>= # --# > <<results= hide >>= # 22c29 # < <<echo=TRUE,print=TRUE>>= # --# > <<echo=TRUE>>= # 43c50,57 # < \SweaveOpts{echo=true} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=TRUE # > ) # > @ # > # 53c67 # < <<fig=TRUE>>= # --# > <<>>= # 63c77 # < <<fig=true>>= # --# > <<>>= # 69d82 # < @

###### 16.1.1 구문(Syntax)

기본적으로 knitr는 청크 옵션을 구문 분석할 때 R 함수 인수와 유사한 새로운 유형의 구문을 사용합니다. 이를 통해 기존의 Sweave 구문보다 훨씬 강력한 기능을 활용할 수 있습니다. 청크 옵션에서 임의의 객체를 사용하여 R의 강력한 기능을 최대한 활용할 수 있습니다.

Sweave는 청크 옵션을 문자열로 취급하고 옵션을 쉼표로 분할하여 구문 분석하는 반면 knitr는 R 구문을 사용합니다. 즉, 옵션이 문자 값을 취하는 경우 R에서와 마찬가지로 따옴표로 묶어야 합니다(예: results = 'hide' (Sweave에서는 results = hide로 작성)). 청크 옵션에서 직접 컴퓨팅을 수행하는 예시는 12.1.3절을 참조하십시오. 아래는 새로운 구문이 얼마나 유연한지를 보여주는 또 다른 예시입니다(동적으로 그림 캡션을 만들 수 있습니다).

<<cap, fig.cap=paste( The P-value is , t.test(x)$p.value)>>= x <- rnorm(100) boxplot(x) @

The other minor difference in syntax is that knitr does not recognize @ as the beginning of text chunks unless there is a chunk header before it. For example, knitr will keep the ﬁrst @ in the example below but Sweave will remove it:

text @ <<A>>= 1 + 1 @

Sweave2knitr()는 이 문제를 자동으로 해결할 수 있습니다.

###### 16.1.2 옵션

Sweave의 일부 옵션은 knitr에서 제외되었으며 일부는 다음과 같이 변경되었습니다.

concordance는 주로 RStudio를 지원하기 위해 변경되었습니다. 패키지 옵션 opts_knit$get('concordance')가 TRUE이면 출력 줄 번호가 입력 줄 번호에 매핑된 inputconcordance.tex라는 파일이 작성됩니다. 참고로 이 구현은 Sweave보다 덜 정확합니다.

keep.source는 더 유연한 옵션인 tidy에 병합되었습니다.

print가 제외되었습니다. R 표현식이 출력되는지 여부는 사용자의 R 사용 경험과 일치합니다(예: x <- 1은 출력되지 않는 반면 1:10은 출력됩니다. R 콘솔에 명령을 입력한다고 상상해 보십시오). 표현식의 출력을 보이지 않게 하려면 invisible() 함수를 사용할 수 있습니다.

term이 제외되었습니다(term = TRUE로 간주). prefix가 제외되었습니다(prefix = TRUE로 간주). prefix.string의 이름이 fig.path로 바뀌었으며 항상 그림 파일 이름에 사용됩니다.

eps, pdf 및 그래픽 장치에 대한 모든 논리 옵션이 제외되었습니다. 대신 Sweave의 grdevice와 유사하지만 20개 이상의 미리 정의된 그래픽 장치가 있는 새로운 옵션 dev를 사용하십시오(7장 참조).

fig가 제외되었습니다. 이제 fig.keep을 사용합니다. knitr의 fig.keep = 'high'는 fig = TRUE와 동일하고 fig.keep = 'none'은 Sweave의 fig = FALSE와 동일합니다.

width, height의 이름이 각각 fig.width 및 fig.height로 변경되었습니다. 한편, \SweaveOpts{} 및 \SweaveInput{}은 사용 중단(deprecated)되었습니다. opts_chunk$set() 및 청크 옵션 child를 사용하여 각각 전역 청크 옵션을 설정하고 하위 문서를 포함하십시오.

논리 옵션의 경우 TRUE/FALSE/T/F만 지원되며(처음 두 개가 권장됨), true/false는 작동하지 않습니다. 예를 들어 eval = FALSE는 괜찮지만 eval = false는 안 됩니다(false라는 이름의 R 객체가 논리값 FALSE를 갖는 경우가 아니라면). <<label>> 구문을 사용하는 청크 참조는 여전히 사용할 수 있으며 청크를 재사용하기 위한 다른 접근 방식도 있습니다. 예를 들어 새 옵션인 ref.label을 사용할 수 있으며 청크 참조는 9장에 소개된 것처럼 재귀적일 수 있습니다.

###### 16.1.3 문제점(Problems)

Sweave의 일부 알려진 문제와 자주 묻는 질문이 knitr에서 해결되었습니다.

- • 빈 그림 청크는 Sweave에서 LATEX 오류를 발생시키지만 knitr에서는 그림이 아예 생성되지 않으므로 오류가 발생하지 않습니다. knitr는 청크에 플롯이 있을 때만 LATEX에 그림을 작성합니다.
- • 명시적으로 print()를 실행하지 않으면 Sweave에서는 lattice(및 ggplot2) 그래픽이 작동하지 않지만 knitr에서는 R 콘솔에서와 마찬가지로 작동합니다(이러한 플롯 객체가 최상위 환경에 나타나면 출력할 필요가 없습니다).
- • 출력 그림의 너비는 LATEX 스타일 Sweave.sty에 정의된 \setkeys{Gin}{width=.8\textwidth}를 통해 Sweave의 경우 기본적으로 .8\textwidth로 설정됩니다. 이는 Sweave에서 생성되었는지 여부와 관계없이 문서의 모든 그림에 영향을 미치며, 그림의 개별 너비를 설정하는 직관적인 방법이 없습니다. 이 문제는 knitr의 out.width 옵션으로 해결되었습니다.
- • 기본적으로 Sweave에서는 하나의 그림 청크에서 여러 그림이 작동하지 않으므로 사용자가 직접 LATEX 코드를 작성해야 합니다. knitr의 경우 한 청크에 플롯이 몇 개 있든 차이가 없습니다.
- • 출력 훅을 사용하여 knitr에서 출력의 형식을 변경할 수 있으며 Sweave에서처럼 Sinput/Soutput과 같이 하드 코딩된 LATEX 환경을 사용할 필요가 없습니다. 사실 render_sweave()를 호출하여 knitr에서 Sweave 스타일을 렌더링할 수 있습니다.
- • knitr(R HTML 또는 R Markdown 사용)를 사용하면 HTML 출력을 쉽게 생성할 수 있으며 Sweave는 HTML만 처리하는 R2HTML과 같은 확장이 필요합니다.


Sweave를 실행한 후 종종 쓸데없는 Rplots.pdf 파일이 보일 때가 있는데, 이는 비대화형 R 세션에 대한 R의 기본 그래픽 장치가 pdf()이기 때문이며 이것이 Rplots.pdf를 생성합니다. knitr에서는 불필요한 PDF 파일이 생성되지 않도록 기본 장치가 널 장치(pdf(file = NULL))로 설정되어 있습니다.

###### 16.2 기타 R 패키지

Sweave 및 아래에 소개된 R 패키지(R2HTML 제외)의 대부분의 기능은 knitr가 포괄하므로 이 절은 주로 역사적인 관심사를 다룹니다.

highlight 패키지(Francois, 2013)는 Rnw 문서의 R 코드에 대한 구문 강조 기능을 제공합니다. 아래의 pgfSweave, cacheSweave 및 R2HTML과 마찬가지로 highlight는 Sweave를 기반으로 확장되었습니다. 초기 버전(v0.6 이전)에서 knitr는 구문 강조를 위해 highlight에 의존했지만 유지 관리 문제와 추가 종속성(Rcpp 및 parser 패키지)이 있다는 사실 때문에 이 의존성은 나중에 제거되었습니다. 이제 knitr는 자체 구문 강조 기능을 사용하는데, 이는 R 3.0.0 이전에는 정규 표현식을 기반으로 했으며 R 3.0.0 이후에는 기본 R의 utils 패키지에 있는 getParseData() 함수에 의존합니다. highlight와 유사한 기능을 얻으려면 knitr에서 청크 옵션 highlight = TRUE를 사용하기만 하면 됩니다.

cacheSweave 패키지(Peng, 2012)는 Sweave에 캐시 시스템이라는 중요한 기능을 추가했습니다. weaver 패키지(Falcon, 2013)는 다른 구현으로 유사한 기능을 제공했습니다. 청크 옵션 cache 및 dependson이 추가되었으며 knitr(8장 참조)와 동일한 의미를 갖습니다.
pgfSweave 패키지(Bracken 및 Sharpsteen, 2012)는 highlight 및 cacheSweave의 기능을 결합하고 그래픽에 대한 추가 지원을 제공했습니다. 구체적으로 플롯도 캐시될 수 있으며 글꼴 스타일의 일관성을 위해 tikzDevice 패키지를 통한 TikZ 그래픽도 지원됩니다. 이 책의 저자는 pgfSweave가 출시되었을 때 Sweave에서 pgfSweave로 전환했고 이에 대한 formatR 지원(tidy 옵션)을 기여했지만, 시간이 지남에 따라 Sweave의 변경 사항을 따라가기가 점점 더 어려워졌습니다. 이 패키지는 CRAN 저장소에서 제거되었습니다. 어쨌든 knitr의 디자인은 저자의 pgfSweave 경험에서 많은 혜택을 받았습니다.

brew 패키지(Horner, 2011)는 가벼운 템플릿 프레임워크이며 해당 구문은 PHP(<?php ?>)와 유사합니다. 기본적으로 템플릿 태그 <% %> 내의 R 코드를 구문 분석하고 실행합니다. 이를 Sweave 및 knitr의 인라인 R 코드라고 생각할 수 있습니다. 캐시 시스템이 있지만 그래픽을 직접 지원하지는 않습니다. knitr 패키지는 5장에서 언급하지 않은 brew 구문도 부분적으로 지원합니다. 아래는 knitr를 통해 컴파일할 수 있는 예시입니다.

pi 값은 <% pi %>이고, pi의 2배는 <% 2*pi %>입니다.

입력 파일의 확장자가 *.brew인 경우 knitr는 brew 구문을 자동으로 사용합니다. brew는 실제로 여러 인라인 표현식에서 불완전한 코드 조각을 지원하므로 PHP와 매우 유사하다는 점에 유의하십시오. 다음은 brew에서 가져온 예시이지만 knitr는 이를 컴파일할 수 없습니다.

<% for (i in c( 1+1 , 1+pi , 1+pi , sin(pi/2) )) { -%> > <%=i%> <% print(eval(parse(text=i))) %> <% } -%>

R2HTML 패키지(Lecoutre, 2014)에는 R 객체를 HTML로 내보내는 다수의 함수가 포함되어 있습니다. 기본 함수는 S3 제네릭 함수인 HTML()이며, 이는 데이터 프레임, 테이블, lm 객체(lm()에 의해 반환됨) 등과 같은 다양한 R 객체에 적용될 수 있습니다. 아래는 HTML 표로 변환된 iris 데이터의 일부입니다.

library(R2HTML) HTML(head(iris[, -5], 1), "", caption = NULL)

<p align= center > <table cellspacing=0 border=1><tr><td>

<table border=0 class=dataframe> <tbody> <tr class= firstline >

<th>&nbsp; </th> <th>Sepal.Length </th> <th>Sepal.Width </th> <th>Petal.Length </th> <th>Petal.Width</th>

</tr>

<tr> <td class=firstcolumn>1 </td> <td class=cellinside>5.1 </td> <td class=cellinside>3.5 </td> <td class=cellinside>1.4 </td> <td class=cellinside>0.2 </td></tr>

</tbody> </table>

</td></table>

R HTML 문서에 대해 knitr 내에서 R2HTML을 사용할 수 있으며, 출력에 원시 HTML 코드를 작성하기 위해 청크 옵션 results = 'asis'를 사용합니다.

R2HTML의 또 다른 주요 기여는 Sweave 확장으로, Sweave를 기반으로 HTML 보고서를 작성할 수 있게 해줍니다.

CRAN에는 재현 가능한 연구에 대한 작업 뷰(http://cran.r-project.org/web/views/ReproducibleResearch.html)가 있으며, 여기서 이 주제에 대한 더 많은 패키지를 찾을 수 있습니다.

###### 16.3 Python 패키지

이 절에서는 동적 문서를 위한 Python 기반의 세 가지 패키지, 즉 Dexy, PythonTEX 및 IPython을 소개합니다.

###### 16.3.1 Dexy

Dexy(http://www.dexy.it)는 매우 일반적인 설계를 특징으로 하는 무료 Python 패키지입니다. 해당 웹사이트에 따르면 다음과 같습니다.

Dexy는 코드를 통합하는 모든 종류의 기술 문서를 작성하기 위한 자유로운 형식의 문학적 문서화 도구입니다. Dexy는 올바른 문서를 작성하고 코드가 변경됨에 따라 시간이 지나도 쉽게 유지 관리할 수 있도록 돕습니다.

네 가지 주요 특징은 다음과 같습니다.

- 1. 모든 언어(소스 코드)
- 2. 모든 마크업(출력)
- 3. 모든 템플릿
- 4. 모든 API(프로그래밍)


다중 언어 지원 등 Dexy와 knitr 사이에는 분명한 유사점이 있습니다. Dexy의 중요한 개념은 "필터(filter)"입니다. 필터는 입력 파일을 가져와 출력 파일로 변환하며 쉘 스크립의 파이프 |와 유사합니다. Dexy의 필터는 사실 knitr에 있는 개념들의 조합입니다. 필터는 (예: Markdown에서 HTML로) 출력을 렌더링하거나 (knitr의 언어 엔진처럼) 프로그래밍 언어를 실행하거나 knitr의 청크 훅처럼 추가 작업을 수행할 수 있습니다.

일반적으로 Dexy는 컴퓨터 코드와 템플릿을 분리하는데, 이는 좋을 수도 있고 나쁠 수도 있습니다. 좋은 점은 소스 스크립트를 재사용할 수 있다는 것이고, 나쁜 점은 보고서 환경과 소스 코드 사이를 이리저리 오가야 한다는 것입니다. 기본적으로 knitr는 보고서에 코드 청크를 직접 삽입하지만, 9장에 소개된 대로 코드 청크를 외부화할 수도 있습니다.

###### 16.3.2 PythonTEX

PythonTEX(https://github.com/gpoore/pythontex)는 LATEX 내에서 Python 코드를 실행할 수 있는 LATEX 패키지입니다. 해당 문서에 따르면 다음과 같습니다.

PythonTEX는 LATEX 내에서 Python에 빠르고 사용자 친화적으로 접근할 수 있게 해줍니다. LATEX 문서 내에 입력된 Python 코드를 실행하고 그 결과를 원본 문서에 포함할 수 있게 합니다. 또한 Pygments 패키지를 통해 LATEX 문서 내 코드에 대한 구문 강조 기능을 제공합니다.

\pyb{} 명령을 사용하여 인라인 Python 코드를 삽입하거나 pyconsole 환경을 사용하여 LATEX에서 Python 세션을 에뮬레이트할 수 있습니다. 예:

\begin{pyconsole}[][frame=single]

- x = 123
- y = 345
- z = x + y z def f(expr):


return(expr**4)

- f(x) print( Python says hi from the console! ) \end{pyconsole}


이 문서를 컴파일하면 Python 코드가 평가되고 그 결과가 출력에 삽입됩니다.

Python에서 유래했기 때문에 SymPy(기호 조작) 및 matplotlib(플롯)와 같은 다른 Python 패키지와의 통합 기능도 있습니다.

###### 16.3.3 IPython

IPython(http://ipython.org)은 코드, 텍스트, 수식, 인라인 플롯 및 기타 리치 미디어, 병렬 컴퓨팅을 위한 고성능 도구 등을 지원하는 웹 기반 노트북을 특징으로 하는 Python용 대화형 쉘입니다.

그림 16.1은 우분투(Ubuntu)의 그놈(GNOME) 터미널에 있는 IPython의 스크린샷입니다. 쉘에 x.spl<TAB>을 입력하면 아래와 같이 자동 완성 기능이 나타나는 등 쉘의 기본 기능이 있음을 알 수 있습니다.

보고서 생성과 관련된 가장 주목할 만한 기능은 웹 기반 노트북입니다. Python 명령을 사용하여 웹 브라우저에서 작업하고 수치 및 그래픽 결과를 포함한 결과를 즉시 확인할 수 있으며 노트북에 더 많은 내용을 입력함에 따라 노트북을 지속적으로 업데이트할 수 있습니다. 이는 knitr에서 코드 청크를 작성하는 것과 매우 비슷합니다.

IPython 노트북은 다른 사람과 공유할 수 있는 확장자 *.ipynb를 가진 JSON 파일로 저장할 수 있습니다. 노트북에는 출력이 포함될 수도 있고 포함되지 않을 수도 있습니다. 출력이 없는 노트북은 knitr의 소스 문서(예: Rnw 및 Rmd 문서)와 유사합니다.

IPython에서 영감을 받아 knitr에는 (기능은 적지만) 유사한 웹 노트북이 있으며, 3.2.2절에서 언급한 바 있습니다.

![image 34](Dynamic Documents with R and knitr 2nd_images/imageFile34.png)

그림 16.1: IPython 스크린샷: 입력은 In[n]으로 표시되고 출력은 Out[n]으로 표시됩니다.

###### 16.4 기타 추가 도구

R 및 Python 패키지 외에도 다른 프로그램의 도구들이 있습니다. 이 장에서 동적 문서를 위한 모든 도구를 열거하는 것은 불가능합니다. Schulte 등(2012)은 Javadoc, cweb, noweb, Sweave, SASweave 등과 같이 문학적 프로그래밍과 재현 가능한 연구를 위한 기존 도구 목록을 제공했습니다.

###### 16.4.1 Org-mode

Org-mode는 일반 텍스트 마크업 언어이며 Emacs 텍스트 편집기에 구현되어 있습니다(Schulte 등, 2012). 문학적 프로그래밍과 (동적 문서의 의미에서) 재현 가능한 연구를 모두 지원합니다. 다소 차이는 있지만 WEB 및 noweb과 같은 문학적 프로그래밍의 초기 구현 구문을 따릅니다. 즉, 코드 청크와 텍스트 청크(때로는 텍스트 청크를 "prose(산문)"라고도 부름)의 개념을 가지고 있습니다. Org-mode의 코드 청크는 다음과 같습니다.

#+name: c-chunk #+begin_src C

int main(){

return 0; }

#+end_src

By comparison, the same chunk is written like this in knitr: <<c-chunk, engine= c >>= int main(){

return 0;

} @

메타데이터는 청크 헤더에 저장됩니다. Org-mode는 입력 언어를 자유롭게 지원하며 출력 형식으로 LATEX 또는 HTML을 사용합니다.

Schulte 등(2012)은 기존 도구들의 문학적 프로그래밍 기능(예: Sweave에는 이 기능이 없음)을 언급했지만, 보고서 작성자에게 흥미롭지 않은 내용이라 판단하여 이 책에서는 강조하지 않았습니다. 사실 knitr에도 코드 청크를 재구성하는 기능이 있습니다(9장 참조). 다음은 나중에 청크 B를 정의하되 이전 청크 A에 삽입하는 간단한 예시입니다.

- <<A>>= df <- data.frame(x = 1:10, y = rnorm(10))
- <<B>> coef(fit) @


<<B>>= fit <- lm(y ~ x, data = df) @

강력하기는 하지만 Org-mode의 Emacs적인 특성은 초보자에게 장애물이 될 수 있습니다.

###### 16.4.2 SASweave

SASweave(http://homepage.cs.uiowa.edu/~rlenth/SASweave)는 SAS 및 R을 사용한 문학적 프로그래밍의 구현체입니다. gawk로 작성되었습니다. 기본 아이디어는 Sweave 및 knitr와 동일합니다. 자세한 내용은 Lenth와 Højsgaard(2007)를 참조하십시오. knitr 패키지는 SASweave에 비해 R을 더 포괄적으로 지원하지만 SAS에 대한 지원은 덜합니다.

###### 16.4.3 오피스(Office)

동적 문서를 위해 일반 텍스트 형식을 고집할 필요는 없지만, 이 책에서 소개한 거의 모든 내용은 일반 텍스트를 기반으로 합니다. OpenOffice(또는 OpenDocument Text)나 Microsoft Office 제품(줄여서 오피스 문서라고 함)을 기반으로 하는 도구가 있으며 처음에는 매력적으로 보일 수 있습니다. 핵심적으로 오피스 문서는 대개 XML 파일(압축될 수 있음)이므로 여기에 코드 청크를 포함할 수 있습니다. 코드 청크를 구문 분석하고 실행한 다음 결과를 다시 삽입할 수 있습니다.

여기서 볼 수 있는 주요 문제점은 XML 형식이 너무 복잡하고 표준이 너무 많아서 수정된 문서가 여전히 유효한 오피스 문서인지 확인하는 것이 쉽지 않다는 것입니다. 한 예로 StatWeave 패키지(http://homepage.stat.uiowa.edu/~rlenth/StatWeave/)는 "OpenOffice에서 수정된 문서를 손상된 것으로 표시"하기 때문에 OpenOffice(3.2 이상)에서 더 이상 작동하지 않습니다.

이에 비해 일반 텍스트 파일은 다루기가 훨씬 쉽습니다. ECMA-376과 같은 복잡한 표준을 신경 쓸 필요가 없습니다. 그래도 굳이 오피스 문서를 원한다면 적어도 Markdown에서 변환할 수 있는 가능성은 있습니다. 1장에서 인용한 내용을 떠올려 보십시오.

소스 코드는 진짜다(The source code is real).

### A

###### 내부 구조(Internals)

이 부록에서는 knitr 패키지의 일부 내부 구조를 설명합니다. 이는 다른 개발자가 패키지를 더 잘 이해하고 필요할 때 코드를 기여하는 데 도움이 될 수 있습니다. 일반 사용자는 이 부록을 읽을 필요가 없습니다. 내부 구조를 세 가지 측면, 즉 문서화, 클로저(closures) 적용 및 일부 기능의 구현 측면에서 설명합니다.

###### A.1 문서화

knitr에는 R 문서(Rd), PDF 매뉴얼, 웹사이트의 세 가지 문서화 유형이 있습니다.

R 문서는 roxygen2(Wickham 등, 2015)를 기반으로 하며 이를 통해 사용자는 태그와 함께 roxygen 주석(#') 안에 Rd를 작성할 수 있고, 이 주석은 실제 Rd로 변환됩니다. 다음은 roxygen 주석의 예시입니다.

#  @author Yihui Xie

이것은 다음처럼 Rd로 변환됩니다. \author{Yihui Xie}

roxygen에는 @usage, @param, @return 및 @examples와 같은 일련의 태그가 있으며 이는 Rd의 \usage{}, \arguments{\item{}}, \value{} 및 \examples{}에 각각 해당합니다. 공식 Rd에 roxygen 주석을 작성할 때의 장점은 문서와 소스 코드를 동일한 파일에 보관할 수 있다는 것입니다. 이에 반해 R 패키지를 작성하는 공식적인 방법은 R/ 디렉터리 아래에 R 소스를 작성하고 man/ 아래에 매뉴얼 페이지를 *.Rd 파일로 작성하는 것입니다. 이 경우 두 파일 사이를 번갈아 가야 하므로 불편하고, R 소스만 업데이트하고 문서 업데이트는 잊어버리기 쉽습니다. Roxygen 주석은 소스의 R 함수 바로 위에 나타나므로 소스와 문서를 모두 유지 관리하기가 훨씬 쉽습니다.

다음은 roxygen 주석으로 문서화된 함수의 완전한 예시입니다.

#  Repeat a character string #  #  Repeat a string n times and make one string. #  @param x a character string #  @param n an integer #  @return A character string. #  @examples f( hi , n = 5) f <- function(x, n = 10) {

paste(rep(x, n), collapse = "") }

roxygen2의 roxygenize() 함수를 사용하여 roxygen 주석을 공식 Rd 파일로 변환할 수 있습니다. knitr의 모든 객체는 이러한 방식으로 문서화되어 있습니다. 또한 roxygen2는 NAMESPACE 및 DESCRIPTION의 Collate 필드도 자동으로 처리하므로 R 소스 파일 작업에 오롯이 집중할 수 있습니다.

PDF 매뉴얼의 소스 문서는 examples 디렉터리(소스 패키지의 inst/examples/ 참조)에 있습니다. 예로써 기본 매뉴얼은 knitr-manual.Rnw입니다. Rnw 파일은 LYX 파일(4.2절)에서 내보낸 것이므로 LYX 파일을 열어 PDF 매뉴얼을 편집하거나 컴파일하는 것이 좋습니다. PDF 매뉴얼은 소스 패키지와 함께 제공되지 않는데, 이는 (1) 버전 제어에 바이너리 파일(특히 소스 파일의 부산물인 경우)을 넣고 싶지 않고 (2) 패키지 웹사이트에 호스팅되기 때문입니다.

패키지 웹사이트는 13.4절에서 소개된 바와 같이 Jekyll로 구축되었습니다. 특히 모든 페이지는 Markdown으로 작성되었으며 Git 저장소의 gh-pages 브랜치에 배치되었습니다(패키지 자체는 master 브랜치에 있음). Git을 통해 변경 사항이 푸시되면 Github가 자동으로 웹사이트를 재빌드합니다. 웹사이트에 기여하고 싶다면 gh-pages 브랜치로 전환하여 Markdown 파일을 업데이트하기만 하면 됩니다.

###### A.2 클로저(Closures)

클로저는 knitr에서 중심적인 역할을 합니다. opts_chunk(5.1.1절) 및 knit_engines(11장)와 같은 몇 가지 공통 객체는 클로저를 기반으로 빌드됩니다.

클로저는 기본적으로 함수이며 비지역 변수에도 접근할 수 있습니다. 다음은 간단한 예시입니다.

f <- function() { x <- 1 function(y) x + y

} g <- f()

- g(5) # add 5 to x ## [1] 6 ls(environment(g)) # g can see x ## [1] "x"


g() 함수는 f()에서 생성되었으며(f()가 함수를 반환함에 유의), g()는 f() 내부에서 생성된 객체 x를 사용하고 x는 f()에만 존재합니다. g()가 호출되는 위치와 관계없이 항상 이 x에 접근할 수 있습니다.

사실 클로저를 통해 비지역 변수를 수정할 수도 있습니다. 아래는 청크 옵션 관리자인 opts_chunk의 작동 방식을 보여주는 최소한의 예시입니다.

new_list <- function(default = list()) {

list(get = function() default, set = function(...) { x <- list(...) if (length(x)) default[names(x)] <<- x

}) }

new_list() 함수는 함수 목록(세터(setter) 및 게터(getter))을 반환합니다. 객체 default는 이 두 함수에 바인딩되어 있습니다. 이를 청크 옵션의 기본 목록이라고 생각할 수 있습니다. 다음으로 청크 옵션을 가져오고 설정하는 방법을 보여드리겠습니다.
opts <- new_list(list(eval = TRUE)) str(opts$get())

- ## List of 1 ## $ eval: logi TRUE

opts$set(eval = FALSE) # change eval to FALSE opts$set(results = "markup") # add a chunk option str(opts$get())

- ## List of 2 ## $ eval : logi FALSE ## $ results: chr "markup"


opts$set(results = "hide") # change the results option

$set() 함수에서 인수들을 default 객체에 할당하기 위해 <<-를 사용했습니다. 이것이 상위 환경에서 이 객체를 수정할 수 있는 이유입니다(일반적인 <-를 사용했다면 상위 환경의 default는 수정되지 않고 대신 지역 복사본이 생성되었을 것입니다).

클로저를 사용함으로써 knitr는 동일한 구문을 사용하여 객체를 각자의 환경에서 관리할 수 있습니다. knitr의 내부 함수 new_defaults()는 이러한 클로저 목록을 만드는 데 사용됩니다.

객체 opts_chunk(청크 옵션 관리용) 및 knit_engines(언어 엔진 관리용) 외에도 다음과 같은 몇 가지 유사한 객체가 있습니다.

opts_knit: 패키지 옵션 (12.2절)
opts_current: 현재 청크에 대한 청크 옵션
opts_template: 청크 옵션 템플릿 (12.1.2절)
knit_hooks: 훅 함수(출력 훅 및 청크 훅 모두 해당)
knit_patterns: 파서를 위한 구문 패턴 (5.1절)

###### A.3 구현(Implementation)

이 절에서는 패키지에 대한 몇 가지 구현 세부 사항을 설명합니다. 먼저 짚고 넘어갈 사소한 점은 제가 할당 연산자로 <- 대신 =를 사용한다는 것입니다. 소스 코드 곳곳에서 =를 볼 수 있을 것입니다. 이는 개인적인 취향 문제이며 실제 단점은 없다고 생각하지만, 이 패키지에 코드를 기여할 때는 =를 따를 것으로 기대합니다. 이 책에서는 <-가 보일 텐데, 이는 제가 등호를 입력했지만 formatR에 의해 자동으로 바뀌었기 때문입니다.

###### A.3.1 파서(Parser)

문서 파서(5.1절)는 다음과 같이 작동합니다. 구문 패턴 객체의 하위 요소인 chunk.begin과 chunk.end를 사용하여 문서를 조각(코드 청크 및 텍스트 청크)으로 분할합니다. 코드 청크의 경우 청크 옵션(즉, 첫 번째 줄에서 추출한 텍스트)이 R 코드로 구문 분석되며, 이것이 청크 옵션이 R 구문을 따라야 하는 이유입니다. 다음은 knitr가 텍스트 조각에서 청크 옵션을 가져오는 방법을 설명하는 예시입니다.

## suppose this is the chunk options text txt <- "label, eval=TRUE, echo=1:3, foo=if(TRUE) 2 else 5" opc <- eval(parse(text = paste("alist(", txt, ")"))) names(opc) # the chunk label is not named

## [1] "" "eval" "echo" "foo" str(opc) # some are unevaluated expressions

## List of 4 ## $ : symbol label ## $ eval: logi TRUE ## $ echo: language 1:3 ## $ foo : language if (TRUE) 2 else 5

먼저 텍스트 주위에 alist() 함수를 추가했습니다. 이 함수는 인수를 마치 함수 인수를 설명하는 것처럼 취급하므로 이 시점에서는 "인수"가 평가되지 않습니다. 그러나 구문은 적어도 유효해야 합니다. 한 가지 예외는 청크 레이블인데, 문자열이어야 하므로 필요한 경우 자동으로 따옴표로 묶입니다. 내부 함수 parse_params()는 청크 옵션을 구문 분석하는 데 사용됩니다.

p <- knitr:::parse_params str(p("chunk-label, eval=TRUE, foo=5"))

- ## List of 3 ## $ label: chr "chunk-label" ## $ eval : logi TRUE ## $ foo : num 5


# 2a is not a valid symbol in R, but knitr will quote it # automatically so parsing is OK parse(text = "alist(2a)") ## Error: <text>:1:8: unexpected symbol ## 1: alist(2a ## ^ str(p("2a, eval=FALSE"))

## List of 2 ## $ label: chr "2a" ## $ eval : logi FALSE

str(p(" 2a , eval=FALSE")) # or you can quote it manually

## List of 2 ## $ label: chr "2a" ## $ eval : logi FALSE

청크 옵션은 청크가 실행되기 전까지 평가되지 않으므로 구문 분석 시간에 문서에서 값을 알 수 없는 객체를 사용할 수 있습니다. 예를 들어 위의 echo 및 foo 옵션은 평가되지 않은 표현식이며 나중에 명시적으로 평가할 것입니다.

eval(opc$echo) ## [1] 1 2 3 eval(opc$foo) ## [1] 2

모든 코드 청크는 내부 객체 knit_code에 명명된 목록으로 저장됩니다. 이름은 청크 레이블이고 내용은 코드입니다. 이 객체도 클로저 목록으로 생성되므로 get() 및 set() 메서드가 있지만, 예상치 못한 결과가 발생할 수 있으므로 수정하지 않는 것이 좋습니다. 필요한 경우 knitr:::knit_code$get('chunk-label')을 통해 코드 청크에 접근할 수 있습니다.

###### A.3.2 청크 훅(Chunk Hooks)

knit_hooks에는 여러 개의 기본 훅이 있으며 이들은 출력 훅(5.3절)입니다.

names(knit_hooks$get(default = TRUE))

## [1] "source" "output" "warning" "message" ## [5] "error" "plot" "inline" "chunk" ## [9] "text" "document"

이 객체의 다른 모든 훅은 청크 훅으로 취급됩니다(10장). 코드 청크가 실행되기 전과 후에 모든 추가 훅이 호출됩니다. 다음은 의사 코드(pseudo code)입니다.

hook(before = TRUE, ...) evaluate(code) hook(before = FALSE, ...)

명심해야 할 한 가지 문제는 훅이 실행되는 순서입니다. knit_hooks에 두 개의 훅 A와 B가 정의되어 있다면 어떤 순서로 호출될까요? 이 순서는 청크 옵션에서 얻습니다. 이 두 훅에 해당하는 두 개의 청크 옵션 A와 B가 있어야 하며 청크 옵션의 순서가 훅을 실행할 순서를 결정합니다. 예를 들어 A가 B 앞에 있으면 훅 A가 B보다 먼저 호출됩니다. 그러나 코드 청크가 평가된 후에는 순서가 반대로 되는데, 그 이유는 훅이 반환하는 결과가 짝을 이루어 그룹을 형성하도록 하기 위해서입니다. 예를 들어 훅 A가 청크 앞에 \begin{Aenvir}를 반환하고 청크 뒤에 \end{Aenvir}를 반환하며, B도 유사하게 Benvir를 반환한다고 가정해 봅시다. 그러면 출력에서 원하는 결과는 다음과 같습니다.

\begin{Aenvir} \begin{Benvir} % results from the chunk \end{Benvir} \end{Aenvir}

참고로 \end{Benvir}는 \end{Aenvir} 앞에 옵니다. 이러한 이유로 훅 A와 B가 정의될 때 다음 두 청크는 다른 결과를 반환합니다.

- <<A=TRUE, B=TRUE>>=
- <<B=TRUE, A=TRUE>>=


###### A.3.3 옵션 별칭(Option Aliases)

청크 옵션 별칭(12.1.1절)을 구현하는 데는 몇 줄밖에 걸리지 않습니다. 목록 내의 특정 요소를 대치하는 간단한 작업이기 때문입니다. 다음은 이 아이디어를 설명하는 짧은 함수입니다.

apply_aliases <- function(x, list) {

## names are aliases of x list[x] <- list[names(x)] list

} al <- c(w = "fig.width", h = "fig.height", a = "fig.align") op <- list(w = 7, h = 7, echo = TRUE, a = "center") str(op) # user s options

- ## List of 4 ## $ w : num 7


## $ h : num 7 ## $ echo: logi TRUE ## $ a : chr "center"

str(apply_aliases(al, op)) # corrected options

## List of 7 ## $ w : num 7 ## $ h : num 7 ## $ echo : logi TRUE ## $ a : chr "center" ## $ fig.width : num 7 ## $ fig.height: num 7 ## $ fig.align : chr "center"

별칭은 명명된 문자 벡터에 설정되며 이름은 벡터 내 요소의 별칭입니다. 위 예시에서 apply_aliases()는 사용자가 지정한 w와 h 값에 따라 각각 fig.width 및 fig.height 요소를 op 목록에 추가했지만 knitr는 내부적으로 여전히 fig.width 및 fig.height를 사용합니다.

###### A.3.4 캐시(Cache)

knitr의 캐시 또한 클로저로 구성된 객체에 의해 관리되지만 더 복잡합니다(내부 함수 new_cache() 참조). 클로저는 캐시 파일을 저장, 로드 및 삭제하는 데 사용되며 여기서는 캐시의 한 가지 측면, 즉 출력의 부작용이 어떻게 캐시되는지에 대해서만 설명합니다(8.4절).

5.3절에서 언급했듯이 코드 청크는 evaluate 패키지에 의해 평가됩니다. 사실 출력 결과는 문자열로 반환되며 전체 청크의 출력 또한 (출력 렌더러에 의해 포맷된) 문자열입니다. 이 문자열은 변수에 할당되며, 변수 이름은 MD5 해시 및 청크 레이블로 구성됩니다. 이 변수는 청크에서 생성된 다른 모든 변수와 함께 캐시 데이터베이스에 저장됩니다. 다음에 청크를 평가할 때 knitr는 청크를 업데이트해야 하는지 확인합니다. 업데이트할 필요가 없다면 출력 결과(사실 이 청크의 모든 것)를 포함하고 있는 청크 출력 객체 등 모든 객체가 직접 로드됩니다. 즉, 청크를 다시 평가하는 대신 이 객체가 출력에 직접 쓰입니다.

###### A.3.5 Sweave와의 호환성

knitr는 Sweave와 몇 가지 다른 청크 옵션을 사용하기 때문에 부적절한 옵션 및 해당 값을 수정하기 위한 Sweave2knitr() 함수가 있습니다. 예를 들어 results = tex는 자동으로 results = 'markup'으로 변경됩니다('tex'는 results 옵션이 실제로 수행하는 작업을 반영하는 적절한 값이 아니기 때문입니다).

구현은 주로 정규 표현식을 기반으로 하며 다음은 간단한 예시입니다.
op <- "<<eval=TRUE, results=tex>>=" gsub("(results)\\s*=\\s*tex", "\\1= markup ", op) ## [1] "<<eval=TRUE, results= markup >>="

Sweave2knitr()는 부적절한 청크 옵션을 비롯하여 \SweaveOpts{} 및 \SweaveInput{}과 관련된 수많은 사례를 처리합니다. 예시는 16.1절을 참조하십시오.

###### A.3.6 일치(Concordance)

일치(concordance) 개념은 Rnw/LATEX에 고유합니다. 해결해야 할 문제는 TEX 출력과 Rnw 소스 간의 줄 번호 매핑입니다. LATEX에서 오류가 발생하면 (오류 로그를 구문 분석하여) 문제 줄의 줄 번호를 알 수는 있지만 두 문서의 줄 번호가 일치하지 않을 수 있으므로 Rnw 소스 문서의 해당하는 줄 번호는 알지 못합니다. Rnw 문서에 있는 5줄짜리 청크 하나가 출력에서는 10줄이나 3줄의 LATEX 코드를 생성할 수 있습니다.

Sweave는 knitr보다 일치를 더 잘 구현합니다. 매핑은 Sweave에서 더 정확합니다. knitr에서는 다음과 같은 방식으로 달성된 근삿값일 뿐입니다. 소스 문서를 구문 분석할 때 코드 청크와 텍스트 청크의 줄 수가 기록되고, 이러한 청크가 평가된 후 해당 출력 청크의 줄 수가 다시 계산됩니다. 소스 청크 하나가 5줄이라고 가정할 때 다음과 같이 작동합니다.
- • 출력도 5줄인 경우 소스의 i번째 줄이 출력의 i번째 줄에 매핑됩니다.
- • 출력이 3줄인 경우 소스의 처음 3줄이 출력의 3줄에 매핑됩니다.
- • 출력이 10줄인 경우 소스의 5줄이 출력의 처음 5줄에 매핑됩니다.

분명히 좋은 근삿값이 아닐 수는 있지만 오류를 탐색하는 데 충분히 도움이 될 수 있습니다. 적어도 LATEX의 오류 번호가 문제 소스의 대략적인 영역을 가리킬 수는 있습니다.

일치의 또 다른 용도는 PDF와 Rnw 파일 간 탐색입니다. SyncTEX는 이러한 종류의 탐색을 지원합니다. PDF 문서에서 한 줄을 클릭하여 소스 파일로 돌아가거나 소스에서 한 줄을 클릭하여 PDF로 점프할 수 있습니다. 일치 정보가 없으면 Rnw와 PDF 사이를 탐색할 수 없습니다(TEX↔PDF만 가능).

현재로서는 RStudio만 knitr에 의해 생성된 일치 정보를 사용합니다. 일치를 활성화하려면(기본적으로 비활성화되어 있음) 패키지 옵션을 설정할 수 있습니다(RStudio는 이 작업을 자동으로 수행함).

opts_knit$set(concordance = TRUE)

일치가 활성화되어 있을 때 Rnw 파일 이름이 input.Rnw인 경우 input-concordance.tex 파일이 생성됩니다. 이 파일에는 압축된 매핑 정보가 포함되어 있습니다.

###### A.4 구문(Syntax)

사용자들은 왜 knitr가 다른 문서 형식에 다른 입력 구문을 사용하는지 궁금해할 수 있습니다(5.1절). 예를 들어 Rnw는 <<>>=를 사용하고 Rmd는 ```{r}을 사용합니다. 사실 구문은 문서 형식에 얽매이지 않으며 분명하게 Rmd 문서에 Rnw 구문을 사용할 수도 있습니다.

# This is a markdown document Here is a **code chunk**: <<test>>= 1 + 1 rnorm(5) @ And an inline value \Sexpr{pi}.

위의 예시 문서(이름이 test.Rmd라고 가정)의 경우 다음과 같이 컴파일할 수 있습니다.

library(knitr) pat_rnw() # input is Rnw syntax render_markdown() # output is markdown knit("test.Rmd")

pat_rnw() 함수는 구문을 Rnw로 설정하고, render_markdown() 함수는 출력 렌더러를 Markdown 훅으로 설정합니다.

그렇다면 왜 모든 문서에 Rnw 구문을 사용하지 않을까요? 이 결정은 제작 형식에 따라 더 자연스러운 구문을 원했기 때문에 내려졌습니다. <<>>=는 어떤 문서 형식에서도 유효한 마크업이 아닙니다. 예를 들어 LATEX 명령도 아니고 HTML 태그도 아닙니다. 사실 Sweave에는 LATEX와 유사한 또 다른 구문 세트가 있습니다. 예:

\begin{Scode}{fig = TRUE, echo = FALSE} library("graphics") boxplot(Ozone ~ Month, data = airquality) \end{Scode}

저는 청크 옵션에 {}보다 []를 선호하는데, 이는 LATEX에서 더 자연스러운 선택이 될 것입니다. 어쨌든 <<>>=는 인기가 높아서 knitr에 남았습니다.

Rnw 문서(역사적인 이유로 인해)를 제외한 다른 형식은 R 코드가 실행되기 전에도 knitr 소스 문서를 여전히 유효한 문서로 만듭니다. 예를 들어 R HTML 문서의 R 코드는 HTML 주석(<!-- -->) 안에 들어갑니다.

###### 참고 문헌(Bibliography)

Adler, D. and Murdoch, D. (2014). rgl: 3D visualization device system (OpenGL). R package version 0.95.1201.

Allaire, J., Cheng, J., Xie, Y., McPherson, J., Chang, W., Allen, J., Wickham, H., and Hyndman, R. (2015a). rmarkdown: Dynamic Documents for R. R package version 0.5.1.

Allaire, J., Horner, J., Marti, V., and Porte, N. (2015b). markdown: Markdown Rendering for R. R package version 0.7.7.

Auguie, B. (2013). cda: Coupled dipole approximation in electromagnetic scattering. R package version 1.3.3.

Baggerly, K. A., Morris, J. S., and Coombes, K. R. (2004). Reproducibility of seldi-tof protein patterns in serum: comparing datasets from different experiments. Bioinformatics, 20(5):777–785.

Bracken, C. and Sharpsteen, C. (2012). pgfSweave: Quality speedy graphics compilation and caching with Sweave. R package version 1.3.0.

Buckheit, J. and Donoho, D. (1995). Wavelab and reproducible research. Wavelets and Statistics, 103:55.

Chang, W., Cheng, J., Allaire, J., Xie, Y., and McPherson, J. (2015). shiny: Web Application Framework for R. R package version 0.11.1.

Dahl, D. B. (2014). xtable: Export tables to LaTeX or HTML. R package version 1.7-4.

Eddelbuettel, D., Francois, R., Allaire, J., Ushey, K., Bates, D., and Chambers, J. (2015). Rcpp: Seamless R and C++ Integration. R package version 0.11.5.

Ellson, J., Gansner, E., Koutsoﬁos, L., North, S., and Woodhull, G.

(2002). Graphviz — open source graph drawing tools. In Graph Drawing, pages 483–484. Springer-Verlag.

Falcon, S. (2013). weaver: Tools and extensions for processing Sweave documents. R package version 1.26.0.

259

Fomel, S. and Claerbout, J. (2009). Guest editors’ introduction: Reproducible research. Computing in Science & Engineering, 11(1):5–7.

Francois, R. (2013). highlight: Syntax highlighter. R package version 0.4.3. Friedl, J. (2006). Mastering Regular Expressions. O’Reilly Media, Incor-

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

통계학(Statistics)

초보자와 고급 사용자 모두에게 적합한 Dynamic Documents with R and knitr(2판)는 컴퓨팅을 보고서 작성과 직접 통합하여 통계 보고서 작성을 더 쉽게 만들어 줍니다. 보고서는 과제, 프로젝트, 시험, 책, 블로그, 웹페이지부터 통계 그래픽, 컴퓨팅 및 데이터 분석과 관련된 사실상 모든 문서에 이르기까지 다양합니다. 이 책은 초보자를 위한 기본 응용 프로그램을 다루는 동시에 파워 사용자가 knitr 패키지의 확장성을 이해하도록 안내합니다.

###### 2판의 새로운 점

- • R Markdown v2를 소개하는 새로운 장
- • knitr 패키지의 향상된 기능을 반영한 변경 사항
- • 표 생성, 코드 청크 내 객체의 사용자 정의 인쇄 메서드 정의, C/Fortran 엔진, Stan 엔진, 영구 세션에서 엔진 실행, 동적 문서를 제공하는 로컬 서버 시작에 대한 새로운 절


호평을 받았던 이전 판과 마찬가지로, 이번 판은 보고서 작성의 효율성을 높이는 방법을 보여줍니다. 이 책은 프로그램 출력에서 출판 품질의 보고서 작성까지 모든 과정을 다루며, 보고서의 모든 측면을 미세하게 조정할 수 있도록 돕습니다. 패키지에 대한 데모 및 기타 정보는 저자의 웹사이트에서 볼 수 있습니다.

Yihui Xie는 RStudio의 소프트웨어 엔지니어입니다. 그는 아이오와 주립대학교(Iowa State University) 통계학과에서 박사 학위를 취득했습니다. 대화형 통계 그래픽 및 통계 컴퓨팅을 주로 연구합니다. 활발한 R 사용자이자 여러 수상 경력에 빛나는 R 패키지의 저자입니다. 또한 중국의 대규모 온라인 통계 커뮤니티인 "Capital of Statistics"의 설립자이기도 합니다.

K25425

www.crcpress.com

Second Edition

DynamicDocumentswithRandknitr

Xie

## TheRSeries

# Dynamic Documents with R and knitr

Second Edition

Yihui Xie

K25425_cover.indd 1 4/17/15 11:01 AM
```r
bg = "lightgreen")
```

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |


(플롯 데이터 생략)

- 그림 7.10: 회전 각도가 다른 두 개의 플롯 회전: 첫 번째 플롯은 -30도 회전하고 두 번째 플롯은 90도 회전합니다.


###### 7.6 tikz() 장치

PDF, PNG 및 기타 전통적인 R 그래픽 장치 외에도 knitr는 pgfSweave 패키지의 기능과 유사한 tikzDevice 패키지(Sharpsteen 및 Bracken, 2015)를 통해 TikZ 그래픽(Tantau, 2008)을 특별히 지원합니다. 청크 옵션 dev = 'tikz'를 설정하면 tikzDevice의 tikz() 장치가 플롯을 생성하는 데 사용됩니다. knitr가 파일 이름 확장자 `*.tikz`를 사용하지만 tikz() 장치에서 만든 플롯 파일은 기본적으로 LATEX 파일입니다.

sanitize(`\` 및 `%`와 같이 플롯에서 특수 TEX 문자를 이스케이프하기 위한) 및 external 옵션은 tikz() 장치와 관련이 있습니다. 자세한 내용은 tikz() 설명서를 참조하십시오. knitr에서 external = TRUE는 tikz()에서 standAlone = TRUE를 의미하며, TikZ 그래픽 출력은 생성 직후 PDF로 컴파일되므로 "외부화"는 LATEX의 tikz 패키지에 있는 공식적이지만 복잡한 외부화 명령에 의존하지 않습니다(PGF 및 TikZ 설명서 참조). 외부화의 장점은 주 LATEX 문서가 컴파일될 때 TikZ 그래픽을 PDF로 컴파일하는 시간을 절약한다는 것입니다.

(글꼴) 스타일의 일관성을 유지하기 위해 knitr는 입력 문서의 프리앰블을 읽어 tikz() 장치에 전달하므로

|p(θ | x) ∝ π(θ)f(x | θ)|
|---|


- 그림 7.11: 플롯에서 수학 식을 작성하는 전통적인 접근 방식: 식 객체를 신중하게 구성해야 합니다.


p(θ|x) ∝ π(θ)f(x|θ)

- 그림 7.12: tikz() 장치를 사용하여 기본 LATEX로 수학 작성: 모


든 것이 자연스러운 LATEX 코드입니다. `paste()` 함수는 이 책을 조판하기 위한 목적으로만 사용되었습니다(같은 문자열에 쓸 수 있었던 긴 문자열을 두 줄로 나눔).

플롯의 글꼴 스타일이 전체 LATEX 문서의 스타일과 동일해집니다.

글꼴 스타일의 일관성 외에도 tikz() 장치를 사용하면 임의의 LATEX 식을 R 플롯에 쓸 수 있습니다. 일반적인 용도는 수학 식을 작성하는 것입니다. R의 전통적인 접근 방식은 플롯에 수학 기호를 쓰기 위해 `expression()` 객체를 사용하는 것이지만, tikz() 장치의 경우 일반 LATEX 코드만 작성하면 됩니다. 아래는 두 가지 접근 방식을 각각 사용하는 수학 식 p(θ|x) ∝ π(θ)f(x|θ)의 예입니다. 다음은 그림 7.11(전통적인 접근 방식)에 대한 코드 청크입니다.

```r
plot(0, type = "n", ann = FALSE) 
text(0, expression(p(theta ~ "|" ~ bold(x)) %prop%

pi(theta) * f(bold(x) ~ "|" ~ theta)), cex = 2)
```

tikz() 장치를 사용하면 간단하고(LATEX에 익숙하다면) 더 아름답습니다(그림 7.12).

```r
plot(0, type = "n", ann = FALSE) 
text(0, paste("$p(\\theta|\\mathbf{x})", "\\propto",

"\\pi(\\theta)f(\\mathbf{x}|\\theta)$"), cex = 2)
```

전통적인 접근 방식에서 글꼴을 개선하는 것이 불가능한 것은 아닙니다. 자세한 내용은 Murrell 및 Ripley(2006)를 참조하십시오.

tikz() 장치의 한 가지 단점은 LATEX가 대용량 tikz 파일을 처리하지 못할 수 있다는 것입니다(LATEX에 메모리 부족이 발생할 수 있음). 예를 들어 수만 개의 그래픽 요소가 있는 R 플롯을 

tikz() 장치를 사용하는 경우 LATEX에서 컴파일하지 못할 수 있습니다. 이러한 경우 PDF 또는 PNG 장치로 전환하거나 플롯 유형에 대한 결정을 재고할 수 있습니다. 예를 들어 수백만 개의 점이 있는 산점도는 일반적으로 읽기 어렵고 

2D 밀도를 보여주는 등고선 플롯이나 육각형 플롯이 더 나은 대안이 될 수 있습니다(크기가 더 작음).

문서를 컴파일할 때 PDFTEX 대신 XeTEX 또는 LuaTEX를 사용하는 경우 모든 플롯 청크 이전(가급적 첫 번째 청크)에 tikzDefaultEngine 옵션을 설정해야 합니다.

```r
options(tikzDefaultEngine = "xetex") # or 'luatex'
```

이것은 다중 바이트 문자가 포함된 tikz 플롯을 컴파일할 때 유용하고 종종 필요합니다.

###### 7.7 그림 환경

LATEX 출력의 플롯에 대해 knitr는 자동으로 figure 환경을 생성할 수 있습니다. 이것은 fig.cap 옵션을 그림 캡션의 문자열로 설정할 때 발생합니다. 그림 환경은 다음과 같습니다.

```latex
\begin{figure}[position] 
% e.g., \includegraphics{foo} here 
\caption[short caption]{full caption.} 
\label{label}

\end{figure}
```

fig.cap 옵션은 전체 캡션을 지정합니다. 다른 관련 청크 옵션은 다음과 같습니다(괄호 안의 기본값). 
fig.env ('figure') 사용할 환경 이름, 예를 들어 우리는

기본 figure 환경 대신 marginfigure 또는 sidewaysfigure 환경을 사용할 수 있습니다.

fig.pos ('') 그림의 위치 정렬, 예: 'tbp' 
fig.scap (NULL) 짧은 캡션. NULL인 경우 fig.cap의 `.` 또는 `;` 또는 `:`

앞의 모든 단어가 짧은 캡션으로 사용됩니다. NA인 경우 무시됩니다.

fig.lp ('fig:') 레이블 접두사. 각 청크에 대해 그림 레이블은 청크 레이블에서 파생되며, fig.lp를 접두사로 사용합니다. 예를 들어 청크 레이블이 foo이면 기본적으로 그림 레이블은 fig:foo가 됩니다. 그림 레이블을 사용하면 LATEX 명령 `\ref{}`를 사용하여 그림을 상호 참조할 수 있습니다.

청크에서 생성된 플롯이 여러 개인 경우 이에 따라 여러 그림 환경을 만들 수 있습니다. 이 경우 fig.cap은 그림 캡션 벡터여야 하며 길이는 플롯 수와 같습니다.

(플롯 데이터 생략)

(a) 이것은 하나의 플롯입니다.

(b) 이것은 또 다른 플롯입니다.

- 그림 7.13: 하위 그림이 있는 그림 환경: fig.subcap 및 fig.cap 옵션을 사용하여 만들 수 있습니다.


한편, 청크 옵션 fig.show는 'asis'여야 합니다(그렇지 않으면 하나의 그림 환경만 생성됨).

청크당 플롯이 여러 개인 경우 플롯을 정렬하는 또 다른 방법은 하위 그림을 사용하는 것인데, 이를 위해 LATEX 프리앰블에 subfig 패키지가 필요합니다. 모든 플롯을 하위 그림 환경에 넣으려면 fig.subcap 옵션을 통해 플롯에 하위 캡션을 할당해야 합니다. 예를 들어 `fig.subcap = c('sub caption 1', 'sub caption 2')`이고 `fig.cap = 'full main caption.'`이면 다음과 같이 그 안에 하위 부동소수점(`\subfloat{}`)이 있는 그림 환경을 생성합니다.

```latex
\begin{figure}

- \subfloat[sub caption 1\label{foo1}]{\includegraphics{foo1}}
- \subfloat[sub caption 2\label{foo2}]{\includegraphics{foo2}} 
\caption[short main caption]{full main caption.} 
\label{foo} 
\end{figure}
```

그림 7.13은 하나의 그림 환경에 있는 두 개의 플롯을 보여줍니다. 나란히 놓일 수 있도록 플롯의 출력 너비가 `.49\linewidth`로 설정되었습니다.

외관상 그림 환경은 LATEX에만 적용되지만 fig.cap은 HTML의 플롯에도 사용할 수 있으며 이 경우 캡션은 title 및 alt 속성으로 `<img />` 태그에 쓰입니다. 다음은 LATEX에서 그림 환경을 만드는 예입니다.

```r
<<waiting, fig.cap='Waiting time: Old Faithful geyser.'>>= 
hist(faithful$waiting, main = "") 
@
```

LATEX 출력은 다음과 같습니다.

```latex
\begin{figure}[] \includegraphics{figure/waiting} 
\caption[Waiting time]{Waiting time:

Old Faithful geyser.} 
\label{fig:waiting} 
\end{figure}
```

만약 HTML의 코드 청크였다면 다음을 생성했을 것입니다. 
```html
<img src = "figure/waiting.png"

title = "Waiting time: Old Faithful geyser." 
alt = "Waiting time: Old Faithful geyser." />
```

###### 7.8 그림 경로

그래픽 장치를 소개했지만 플롯이 어떻게 실제로 파일로 저장되는지는 설명하지 않았습니다. 각 플롯은 파일로 저장되며 파일 유형은 그래픽 장치에 따라 다릅니다. 파일 이름은 청크 레이블, fig.path 및 fig.ext의 세 가지 청크 옵션에 의해 결정됩니다. fig.path 옵션은 그림의 경로를 지정하고(기본적으로 상대 디렉터리 figure/임) fig.ext는 플롯 파일의 파일 이름 확장자를 지정합니다(기본적으로 dev 옵션에서 자동으로 파생됨. 예: Cairo_pdf 장치에 해당하는 확장자는 pdf임). 엄밀히 말하면 fig.path는 경로 접두사입니다. 예를 들어 `fig.path = 'figure/mcmc-'`로 설정하면 모든 플롯 파일의 figure/ 디렉터리 아래 접두사가 mcmc-가 됩니다.

청크의 모든 플롯 파일은 foo-1, foo-2, ..., foo-n 순서로 순차적으로 이름이 지정됩니다. 여기서 foo는 청크 레이블이고 n은 청크의 총 플롯 수입니다. 청크에 플롯이 하나만 있더라도 파일 이름에는 여전히 -1 접미사가 있습니다.

fig.path에 존재하지 않는 디렉터리가 포함되어 있으면 knitr는 디렉터리를 자동으로 만들려고 시도합니다. LATEX 출력의 경우 그림 경로 및 파일 이름에 영숫자, 하이픈(-) 및 밑줄(_)만 허용되며 다른 모든 문자는 밑줄로 바뀝니다. 이는 LATEX에서 이러한 문자(예: 공백 및 점)에 문제가 있을 수 있기 때문입니다.

대부분의 경우 fig.ext를 지정할 필요가 없지만

사용자 지정 장치를 사용하여 그래픽을 저장할 때 knitr는 적절한 파일 이름 확장자를 알 수 없으므로 이 옵션을 명시적으로 문자열로 설정해야 합니다.

5.1절에서 청크 레이블의 고유성을 강조했는데, 이것이 고유해야 하는 한 가지 이유입니다. 청크 레이블은 플롯의 파일 이름에 사용됩니다. 동일한 레이블을 공유하는 두 개의 청크가 있는 경우 뒤의 청크가 이전 청크에서 생성된 플롯을 덮어씁니다. 다음 장의 캐시 파일도 마찬가지입니다.

### 8

###### 캐시

동적 문서의 한 가지 문제는 일부 코드 청크를 실행하는 데 오랜 시간이 걸릴 수 있고 이러한 청크는 자주 수정되거나 업데이트되지 않을 수 있다는 것입니다. 이 경우 캐싱이 매우 유용할 수 있습니다. 기본 아이디어는 마지막 실행 이후 수정되지 않는 한 청크가 다시 실행되지 않고 이전 결과가 바로 로드된다는 것입니다.

###### 8.1 구현

캐시는 새로운 아이디어가 아닙니다. cacheSweave와 weaver 패키지 모두 Sweave를 기반으로 캐시를 구현했으며, 전자는 filehash를 사용하고 후자는 `*.RData` 이미지를 사용합니다. cacheSweave는 filehash를 기반으로 한 객체의 지연 로딩도 지원합니다. knitr 패키지는 (tools:::makeLazyLoadDB())를 저장하고 객체를 지연 로딩(lazyLoad())하기 위해 내부 기본 R 함수를 직접 사용합니다.

cacheSweave 비네트는 지연 로딩의 개념을 명확하게 설명했습니다. 대략적으로 지연 로딩은 객체가 실제로 어딘가에 사용될 때까지 메모리에 로드되지 않음을 의미합니다. 대신 "약속"만 생성되며 이는 일반적으로 메모리 소비 측면에서 빠르고 저렴합니다. 계산에 이 약속이 사용될 때 실제 객체가 하드 디스크에서 로드됩니다. 이것은 캐시에 매우 유용합니다. 때로는 대규모 객체를 읽고 캐시한 다음 분석을 위해 하위 집합을 취하고 이 하위 집합도 캐시합니다. 나중에 계산이 하위 집합의 객체에만 기반을 둔다면 초기 대규모 객체는 R에 로드되지 않습니다. R의 약속에 대한 자세한 내용은 `?promise`를 참조하십시오.

캐싱을 켜려면 청크 옵션 cache를 TRUE(기본값은 FALSE)로 설정할 수 있습니다. 다음은 캐시의 효과를 빠르게 보여주는 코드 청크입니다.

```r
- x <- 1 
Sys.sleep(10)
- x <- 2
```

81

우리는 R이 10초 동안 잠자도록 `Sys.sleep()`을 사용했습니다. 이 청크가 처음 컴파일될 때 일시 중지되는 것을 볼 수 있지만, 코드 평가가 실제로 완전히 건너뛰어지기 때문에 다시 컴파일하면 일시 중지되지 않습니다. 이 청크에는 객체 x가 생성되어 있으며 다음 번에 지연 로드됩니다. knitr는 청크에서 새로 생성된 모든 객체를 파악하고 이를 지연 로드 데이터베이스(`*.rdb` 및 `*.rdx` 파일)에 저장합니다. 이제 x의 값을 확인할 수 있습니다.

```r
x # value from previous chunk 
## [1] 2
```

###### 8.2 캐시 쓰기

캐시 파일의 경로는 청크 옵션 cache.path에 의해 결정됩니다. 기본적으로 모든 캐시 파일은 현재 작업 디렉터리를 기준으로 cache/ 디렉터리 아래에 만들어집니다. 옵션 값에 디렉터리가 포함된 경우(예: `cache.path = 'cache/abc-'`), 캐시 파일은 해당 디렉터리 아래에 저장됩니다. 그림 경로와 유사하게 캐시 디렉터리가 없으면 자동으로 생성되며 cache.path도 실제 경로 대신 캐시 파일의 접두사일 수 있습니다.

캐시는 R 코드와 청크 옵션을 포함하여 코드 청크에 대한 변경 사항이 있을 경우 무효화되고 제거됩니다. 즉, 이 청크의 이전 캐시 파일이 제거되고 새 캐시 파일로 대체됩니다. 캐시 파일 이름은 청크 레이블을 접두사로 식별되며(청크 레이블은 문서에서 고유해야 함을 기억하십시오) 캐시 파일 이름의 접미사는 R 코드, 청크 옵션 및 `getOption('width')` 값을 포함하는 목록인 R 객체의 MD5 해시 문자열입니다. MD5 해시는 digest 패키지에 의해 계산되며 knitr에서 캐시 파일 이름 생성을 에뮬레이트하는 아래 예제를 통해 작동 방식을 분명히 알 수 있습니다.

```r
d <- digest::digest 
## imagine x$code is the code chunk; x$options are chunk ## options

- x <- list(code = "1+1", options = list(results = "asis", fig.height = 3), width = getOption("width"))


d(x) 
## [1] "667308d70fc72f26eb7454dde04af9a0"

x$code <- "1 + 1" # add spaces to code 
d(x)

## [1] "e903b616477cfa3e2314a3da65062dfb" 
x$options$eval <- FALSE # add option eval as FALSE 
d(x) 
## [1] "8decb2a180f7f49b47de54bd5ec8fb34" 
x$width <- 40 
d(x) 
## [1] "7e1d77987b195b14d9b563b9a8f0ca6c"
```

위의 길이가 32인 문자열은 MD5 해시입니다. MD5 해시는 콘텐츠의 변경에 민감하다는 것을 알 수 있습니다. 변경이 단지 공백이더라도 변경 시 새로운 해시 문자열이 생성됩니다. 캐시 파일 이름의 형식은 label_hash.rdb입니다. 매번 knitr는 현재 청크의 해시를 캐시 파일 이름과 비교합니다. 일치하지 않으면 청크에 변경 사항이 있음을 의미하며 이전 캐시를 제거해야 합니다.

한 가지 예외는 include 옵션입니다. include = TRUE / FALSE는 코드 평가에 영향을 미치지 않기 때문에 캐시되지 않으므로 캐시에 영향을 미치지 않고 이 청크 옵션을 변경할 수 있습니다.

`getOption('width')`가 캐시에 영향을 미치는 이유는 
인쇄된 텍스트 출력의 너비에 영향을 미칠 수 있기 때문입니다.

###### 8.3 캐시를 업데이트해야 할 때

위에서 설명한 세 가지 구성 요소를 고려하는 것이 합리적으로 보이지만 특정 상황에서는 캐시를 언제 업데이트해야 하는지 명확하지 않을 수 있습니다. 다음과 같은 두 가지 경우를 고려해 보겠습니다.

- 1. R은 여전히 몇 달마다 업데이트되며 새 버전마다 버그가 수정되고 새로운 기능이 도입됩니다. R을 새 버전으로 업그레이드한 후 캐시를 업데이트해야 합니까?(R 패키지에도 유사한 문제가 적용됨)
- 2. 소스 문서에서 외부 데이터 파일을 읽었는데 해당 파일이 수정된 경우 어떻게 해야 합니까? 어떻게 하면 knitr에게 알려주어


모든 캐시된 결과를 업데이트해야 합니까(소스 문서가 변경되지 않은 경우에도)?

이러한 경우 해시를 계산하기 위해 객체에 더 많은 구성 요소를 넣어야 합니다. 코드 청크는 임의의 옵션(이 책에서 소개된 옵션뿐만 아니라)을 허용할 수 있고 모든 청크 옵션이 해시에 반영되므로 추가 청크 옵션을 사용하여 캐시에 영향을 미칠 수 있습니다.

첫 번째 질문에 답하기 위해 다음과 같이 R의 버전을 값으로 취하는 청크 옵션(예: version)을 문서에 추가할 수 있습니다.

```r
<<cache-rversion, cache=TRUE, version=R.version.string>>= 
# code which may be affected by R version 
R.version.string 
## [1] "R version 3.2.0 (2015-04-16)" 
@
```

그러면 R이 업그레이드된 경우 이 청크가 다시 실행됩니다. 두 번째 문제를 해결하려면 외부 파일의 변경 사항을 knitr에 알려야 합니다. 한 가지 자연스러운 지표는 파일의 수정 시간이며, 이는 `file.info()` 함수를 통해 얻을 수 있습니다. 데이터 파일 이름이 iris.csv라고 가정하고 해당 파일의 수정 시간을 iris_time 청크 옵션에 넣을 수 있습니다. 예:

```r
<<itime, cache=TRUE, iris_time=file.info('iris.csv')$mtime>>= 
# data will be re-read if iris.csv becomes newer 
iris <- read.csv("iris.csv") 
@
```

캐시를 언제 업데이트할지 또는 업데이트할지 여부에 대한 고정된 규칙은 없습니다. 이는 특정 응용 프로그램에 따라 다릅니다. 예를 들어 R을 업그레이드한 후 캐시를 비울 필요는 없습니다. 어쨌든 결과가 항상 최신 상태인지 확인하기 위해 청크 옵션을 주의해서 설정해야 합니다.

###### 8.4 부작용

컴퓨터 과학에서 부작용은 반환된 값이 아닌 함수 외부에서 발생하는 상태 변경을 의미합니다. 일반적인 부작용에는 플롯(창 또는 파일) 생성, 파일 쓰기, 콘솔에 결과 인쇄 등이 포함됩니다. 부작용은 캐시하기가 간단하지 않습니다. R 객체를 캐시 데이터베이스에 쉽게 저장할 수 있지만,

플롯 창은 함수에서 반환되는 값이 아니기 때문에 플롯 창을 저장하는 방법은 명확하지 않습니다. 이러한 이유로 weaver 및 cacheSweave와 같은 패키지는 부작용을 캐시하지 않지만, knitr는 다음과 같은 일부 부작용을 유지하려고 시도합니다.

- 1. 인쇄된 결과: 코드 청크의 모든 출력이 캐시된 청크에 대해 실제로 평가되지 않더라도 출력 문서에 로드됨을 의미합니다. 그 이유는 knitr가 청크의 출력을 문자열로도 캐시하기 때문입니다. 이는 그래픽 출력이 출력의 일부이므로 이 역시 캐시된다는 의미이기도 합니다.
- 2. 로드된 패키지: 캐시된 각 청크를 평가한 후 현재 R 세션에서 사용된 패키지 목록이 캐시 경로 아래의 `__packages` 접미사가 있는 파일에 기록됩니다. 다음에 캐시된 청크를 다시 작성해야 하는 경우 이러한 패키지가 먼저 로드됩니다. 패키지 이름을 캐시하는 이유는 다음과 같습니다. 일부 패키지를 로드하는 속도가 느릴 수 있으며, 후자만 다시 작성해야 할 때 다음 캐시된 청크에서는 사용할 수 없는 이전 캐시된 청크에 패키지가 로드될 수 있습니다. 이는 캐시된 청크에만 적용되며, 캐시되지 않은 청크의 경우 패키지를 명시적으로 로드하려면 항상 `library()`를 사용해야 합니다.
- 3. 난수 시드: 청크에서 난수 시드(정수 벡터)를 생성한 경우 다음 번에 시드를 저장하고 로드하여 무작위 시뮬레이션의 재현성을 향상시킵니다(12.4.7절 참조).


knitr가 일부 부작용을 유지하려고 시도하지만 `par()` 또는 `options()` 설정과 같이 캐시되지 않는 다른 유형의 부작용이 여전히 있습니다. 사용자는 이러한 특수한 경우를 알고 있어야 하며, 캐시되지 않아야 하는 코드를 캐시되지 않은 청크로 명확하게 분리해야 합니다. 예를 들어 문서의 첫 번째 청크에서 모든 전역 옵션을 설정하고 해당 청크를 캐시하지 않습니다. 일반적으로 이 청크를 문서의 첫 번째 청크로 둡니다.

```r
<<setup, cache=FALSE, include=FALSE>>= 
# set up some global options for the document 
options(width = 60, show.signif.stars = FALSE) 
# also set up global chunk options 
library(knitr) 
opts_chunk$set(fig.width = 5, fig.height = 4, tidy = FALSE) 
@
```

위 청크에서는 cache = FALSE가 기본값이므로 보통 불필요합니다. 보수적이고 이 청크가 실제로 캐시되지 않도록 하려면 거기에 둘 수 있습니다.

###### 8.5 청크 종속성

때로는 캐시된 청크가 다른 캐시된 청크의 객체를 사용해야 할 수 있으며 이는 심각한 문제를 가져올 수 있습니다. 이전 

- 청크의 객체가 변경된 경우 다른 청크에서 이러한 변경 사항을 감지하는 방법이 없는 한 이 청크는 변경 사항을 알지 못하고 여전히 캐시된 이전 결과를 사용합니다. 따라서 캐시된 청크에 종속성을 도입해야 합니다.


###### 8.5.1 수동 종속성

knitr에는 dependson이라는 청크 옵션이 있습니다(아이디어는 cacheSweave에서 가져옴). 이는 `dependson = c('chunkA', 'chunkB')`와 같이 청크 레이블 벡터를 설정하여 이 청크가 의존하는 다른 청크를 지정합니다. 그러면 캐시된 청크 chunkA 또는 chunkB 중 하나가 다시 작성될 때마다 이 청크는 캐시를 잃고 같이 다시 작성됩니다.

청크 종속성은 사슬을 형성할 수 있습니다. 다음 예제에서 chunkC는 chunkB에 의존하고, chunkB는 차례로 chunkA에 의존합니다. 
```r
<<chunkA>>=

- x <- 1

- <<chunkB, dependson='chunkA'>>=

y <- x + 2

- <<chunkC, dependson='chunkB'>>=




- y + 5 
@
```

chunkC는 chunkB에 생성된 객체 y를 사용하고 chunkB는 chunkA에 생성된 x 값이 필요하기 때문에 종속성이 필요합니다. 첫 번째 청크의 x가 변경되면 후자의 두 청크도 그에 따라 업데이트되어야 합니다.

dependson 옵션은 청크 인덱스의 정수 벡터를 취할 수도 있습니다. 예를 들어 dependson = 1은 이 청크가 문서의 첫 번째 청크에 의존함을 의미하고, dependson = c(3, 5)는 세 번째 및 다섯 번째 청크에 대한 종속성을 나타냅니다. 인덱스가 음수이면 이 청크에서 뒤로 계산함을 의미합니다. 예를 들어 dependson = -1은

이 청크가 이전 청크에 의존함을 의미하고 -c(1, 2, 3)은 이전의 세 청크를 의미합니다. dependson이 정수 값을 취할 때 나중 청크에 의존하게 할 수 없습니다(이전 청크만 가능한 후보임). dependson의 문자 값에는 이 제한이 없습니다.

###### 8.5.2 자동 종속성

청크 간의 종속성을 지정하는 또 다른 방법은 autodep 청크 옵션과 `dep_auto()` 함수를 사용하는 것입니다. 이것은 weaver에서 차용한 실험적인 기능으로, 청크 종속성을 수동으로 설정할 필요가 없습니다. 기본 아이디어는 나중 청크가 이전 청크에서 생성된 객체를 사용하는 경우 나중 청크가 이전 청크에 의존한다고 말합니다.

codetools 패키지의 `findGlobals()` 함수는 다음을 찾는 데 사용됩니다.

- 청크의 모든 전역 객체를 찾아내고, 해당 설명서에 따르면 결과는 근사치입니다. 전역 객체는 대략 로컬에서 생성되지 않은 객체를 의미합니다. 예를 들어 `function() {y <- x}` 식에서 x는 이 함수의 본문에서 생성되는 것을 볼 수 없기 때문에 외부의 기존 전역 객체(실제 객체가 무엇이든 상관없이)여야 하는 반면 y는 로컬입니다. 한편, 캐시된 각 청크에서 생성된 객체 목록도 저장해야 나중에 청크의 전역 객체와 비교할 수 있습니다. 예를 들어 청크 A에서 객체 x를 생성하고 청크 B에서 이 객체를 사용하는 경우 청크 B는 A에 종속되어야 합니다. 즉, A가 변경될 때마다 B도 업데이트되어야 합니다.


autodep = TRUE일 때 knitr는 캐시된 청크에 생성된 객체의 이름과 전역 객체의 이름을 각각 `__objects` 및 `__globals`라는 두 파일에 기록합니다. 나중에 `dep_auto()` 함수를 사용하여 객체 이름을 분석하여 자동으로 종속성을 파악할 수 있습니다. 일반적인 사용법은 다음과 같습니다.

```r
<<setup, cache=FALSE, include=FALSE>>= 
opts_chunk$set(autodep = TRUE) # set autodep globally 
dep_auto() # figure out dependencies 
@
```

종속성을 지정하는 또 다른 방법은 `dep_prev()`입니다. 이것은 캐시된 청크가 모든 이전 청크에 의존하도록 종속성을 설정하는 보수적인 접근 방식입니다. 즉, 이전 청크가 업데이트될 때마다 이후의 모든 청크가 그에 따라 업데이트됩니다.

어떤 경우든 knitr는 캐시된 청크의 변경 사항만 확인하기 때문에 캐시되지 않은 청크에 대한 종속성은 knitr에 의미가 없습니다. knitr는 캐시되지 않은 청크에 대한 종속성을 볼 때 경고를 표시합니다. 

어쨌든 캐시되지 않은 청크에 종속되어야 한다면 8.3절에 소개된 요령을 사용할 수 있습니다. 즉, 캐시되지 않은 객체를 캐시된 청크의 청크 옵션에 넣는 것입니다. 다음은 그 예입니다.

```r
- <<A, cache=FALSE>>=

- x <- 1 
@

<<B, cache=TRUE, foo=x>>=

- y <- x + 2 
@
```

캐시되지 않은 청크 A에 객체 x를 생성하고 캐시된 청크 B에서 사용했습니다. 두 청크 간에 종속성이 없으면 A가 업데이트될 때 B가 업데이트되지 않지만 청크 B에 foo = x 옵션을 설정한 경우 x 값이 변경되면 B가 자동으로 업데이트되어 B의 청크 옵션이 변경됩니다.

###### 8.6 수동으로 캐시 로드

일반적으로 캐시 데이터베이스는 캐시된 청크에 대해 자동으로 로드되며 실제로 수동으로 로드할 수 있습니다. 이것은 유용한 응용 프로그램이 있습니다. 나중에 문서의 청크에서 x 값을 계산했지만 문서 앞부분에서 이 값을 사용하고 싶다고 가정해 보겠습니다. knitr는 문서를 선형 방식으로 컴파일하고 미래에 생성된 객체를 사용할 수 없기 때문에 불가능합니다. 그러나 해당 청크에 대해 캐시를 켰다면 캐시 데이터베이스를 일찍 로드하기만 하면 됩니다.

knitr의 `load_cache()` 함수는 이러한 목적을 위해 설계되었습니다. 청크 레이블을 사용하여 캐시 데이터베이스를 찾으며 선택적으로 이 함수가 캐시에서 반환하도록 할 객체를 지정할 수 있습니다.

```r
load_cache(label, object, notfound = "NOT AVAILABLE", 
path = opts_chunk$get("cache.path"), lazy = TRUE)
```

이제 문서 뒷부분에 객체 x를 만드는 foo라는 캐시된 청크가 있다고 가정해 보겠습니다. 해당 청크에서 x 값을 가져오려면 `load_cache('foo', 'x')`를 사용할 수 있습니다. 물론 문서를 처음 컴파일할 때 x는 사용할 수 없으며 이것이 notfound 인수가 있는 목적입니다. 인라인 R 식에서 x를 사용하면 출력에 NOT AVAILABLE이 표시되며, 청크 foo가 캐시되었으므로 문서를 다시 컴파일하면 이 텍스트가 x의 값으로 바뀝니다.

###### 8.7 기타 옵션

지연 로딩은 유용하지만 아직 저희에게 명확하지 않은 이유로 일부 경우에 작동하지 않을 수 있습니다. 어쨌든 청크 옵션 cache.lazy = FALSE를 사용하여 지연 로딩을 끌 수 있습니다. 이 경우 knitr는 `save()`를 사용하여 객체를 저장하고 `load()`를 사용하여 로드하며 항상 잘 작동할 것입니다.

때로는 코드의 다른 부분을 실제로 변경하지 않고 코드의 주석만 조정하는 경우가 있는데, 코드 주석만 업데이트했다고 해서 캐시 데이터베이스를 업데이트하고 싶지는 않을 것입니다. 이 경우 청크 옵션 cache.comments = FALSE를 사용할 수 있습니다. 그러면 MD5 해시를 계산할 때 주석이 제외되므로 주석 변경 사항이 캐시에 영향을 미치지 않습니다.

### 9

###### 상호 참조

knitr에서 코드 청크와 하위 문서를 상호 참조할 수 있습니다. 이를 통해 소스 문서를 더 잘 구성할 수 있습니다. 다음은 실제적인 예입니다. 사용자 지정 ggplot2 테마가 있고 문서의 몇 가지 플롯에 이 테마를 적용하려고 합니다.

```r
<<my-theme, eval=FALSE>>= 
theme(legend.text = element_text(size = 12, angle = 45)) +

theme(legend.position = "bottom") 
@
```

이 코드 조각을 한 번만 사용한다면 코드 청크에 복사하여 붙여넣을 수 있지만 여러 청크에 붙여넣는 것은 관리하기가 매우 어렵기 때문에 절대 좋은 생각이 아닙니다. 청크 레이블을 사용하여 이 테마를 간단히 참조할 수 있습니다. 예:

```r
qplot(carat, price, data = diamonds, color = cut) +

<<my-theme>>
```

그러면 knitr는 이 청크를 평가하기 전에 `<<my-theme>>`을 실제 소스 코드로 확장합니다. 이 참조를 여러 곳에서 사용할 수 있지만 소스의 복사본은 하나만 유지합니다.

###### 9.1 청크 참조

청크 참조를 사용하면 코드 청크를 다시 입력하지 않고 쉽게 재사용할 수 있습니다. 정의된 청크를 다른 청크에 포함시키거나 전체 청크를 새 청크로 재사용할 수 있습니다.

###### 9.1.1 코드 청크 포함

하나의 청크를 다른 청크의 일부로 사용할 수 있으며 구문은 `<<label>>`입니다(앞에 공백이 허용됨. label은 청크

91

레이블을 의미함). 청크 헤더처럼 `>>` 뒤에 `=`가 없다는 점에 유의하십시오. 예를 들어 청크 A를 B에 포함시킵니다.

```r
- <<A>>= 
x <- rnorm(1) 
@
- <<B>>= 
x 
<<A>> 
x 
@
```

이 경우 청크 B는 본질적으로 다음과 같습니다(`<<A>>`는 청크 A의 코드로 바뀌지만 eval을 포함하여 A의 모든 청크 옵션은 무시됨).

```r
x 
x <- rnorm(1) 
x
```

재귀가 유한한 한 청크는 서로 반복해서 중첩될 수 있습니다. 예를 들어 A를 B에 포함시키고 B를 C에 포함시키지만 C를 A에 다시 포함시켜서는 안 됩니다. 그렇지 않으면 무한 재귀가 발생합니다.

###### 9.1.2 전체 청크 재사용

전체 청크를 재사용하는 방법에는 두 가지가 있습니다. 첫 번째는 동일한 레이블을 사용하되 청크를 비워두는 것입니다. 이 접근 방식의 한 가지 문제점은 MD5 해시가 다를 것이기 때문에 두 청크의 청크 옵션이 다르면 두 청크 모두 캐시할 수 없다는 것입니다. knitr는 레이블당 한 세트의 캐시 파일만 허용합니다. 다음은 한 가지 예입니다.

```r
<<chunkA, eval=FALSE>>= 
x <- 1 + 1 
@ 
<<chunkA, eval=TRUE>>= 
@
```

두 번째 방법은 원본 청크의 청크 레이블 벡터를 취하는 ref.label 옵션을 사용하는 것입니다. 대상 청크에 새 레이블을 사용할 수 있습니다. 다음 예제에서 청크 C는 A와 B 모두의 코드를 사용합니다.

```r
- <<A>>=

- x <- rnorm(1) 
@

<<B>>=

- y <- x + 2 
@




- <<C, ref.label=c('A', 'B')>>= 
@
```

청크 C의 코드는 본질적으로 다음과 같습니다.

```r
- x <- rnorm(1)
- y <- x + 2
```

###### 9.2 코드 외부화

R 코드 청크를 소스 문서에 섞어 넣는 것보다 별도의 R 스크립트에 쓰는 것이 더 편리할 수 있습니다. 예를 들어 다른 텍스트를 건너뛸 필요 없이 순수한 R 스크립트에서 이 청크 저 청크의 R 코드를 계속해서 실행할 수 있습니다.

또 다른 이유는 LYX와 같은 일부 편집기가 대화형으로 R 코드를 실행하도록 지원하지 않아 단일 청크의 결과만 알고 싶더라도 매번 전체 문서를 다시 컴파일해야 하기 때문입니다.

따라서 knitr는 코드 외부화 기능을 도입했습니다. 코드 청크는 `read_chunk()`를 통해 외부 R 스크립트에서 읽을 수 있습니다. R 스크립트는 두 가지 형태로 작성될 수 있습니다. 스크립트에서 레이블을 사용하여 코드 청크를 구분하거나 줄 번호를 기준으로 청크를 지정할 수 있습니다.

###### 9.2.1 레이블 지정된 청크

설정은 다음과 같습니다. R 스크립트도 청크 레이블을 사용합니다(`## ---- chunk-label` 형식으로 표시됨). 소스 문서의 코드 청크가 비어 있으면 knitr는 이 레이블을 R 스크립트의 레이블과 연결하여 외부 R 코드를 입력합니다.

예를 들어 소스 문서와 동일한 디렉터리에 있는 shared.R이라는 R 스크립트에서 Q1로 레이블이 지정된 코드 청크가 다음과 같다고 가정해 보겠습니다.

```r
## ---- Q1 ----
gcd <- function(m, n) {

while ((r <- m%%n) != 0) {

- m <- n
- n <- r


} 
n

}
```

소스 문서에서 먼저 `read_chunk()` 함수를 사용하여 스크립트를 읽을 수 있습니다.

```r
read_chunk("shared.R")
```

이 작업은 일반적으로 문서의 첫 번째 청크와 같은 초기 청크에서 수행되며 나중에 소스 문서에서 Q1 청크를 사용할 수 있습니다.

```r
<<Q1>>= 
@
```

###### 9.2.2 줄 기반 청크

기본적으로 `read_chunk()`는 R 스크립트에 레이블이 지정되어 있다고 가정하며(`## ----`가 구분 기호임) 코드를 지정하는 대안적 방법이 있습니다. 세 가지 인수 labels, from, to를 통해 동일한 길이의 벡터를 취하는 방식입니다. 코드 청크의 시작 및 끝 줄 번호는 각각 from 및 to를 통해 설정할 수 있으며 labels는 청크 레이블 벡터입니다.

예를 들어 foo.R R 스크립트의 1~5, 7~9 및 15~21줄이 A, B, C 레이블을 가진 세 개의 청크를 형성하도록 하려면 다음과 같이 `read_chunk()` 함수를 호출할 수 있습니다.

```r
read_chunk("foo.R", labels = c("A", "B", "C"), from = c(1,

7, 15), to = c(5, 9, 21))
```

그런 다음 소스 문서에 A, B, C라는 레이블이 지정된 세 개의 빈 청크를 작성할 수 있습니다. 또는 from과 to를 시작 줄과 끝 줄에 대한 정규 표현식으로 만들 수 있습니다.

서로 다른 문서가 동일한 R 스크립트를 읽을 수 있으므로 R 코드를 서로 다른 입력 문서에 재사용할 수 있습니다.

###### 9.3 하위 문서

하위 문서의 개념은 주 문서가 클 때 LATEX 사용자에게 친숙할 것입니다. 문서를 더 작은 부분으로 분할하고 `\input{foo.tex}`를 사용하여 주 문서에 입력할 수 있습니다. 예를 들어 책을 장으로 나누고 각 장을 한 파일에 넣을 수 있습니다.

###### 9.3.1 하위 문서 입력

마찬가지로 knitr 소스 문서를 하위 문서 모음으로 관리할 수 있습니다. 청크 옵션 child는 하위 문서에 대한 참조를 제공합니다. book.Rnw라는 주 문서와 같은 디렉터리 아래에 chap1.Rnw라는 하위 문서가 있다고 가정해 보겠습니다. 주 문서에는 다음이 있습니다.

주 문서의 한 청크입니다.

```r
- <<A, eval=TRUE>>=

- x <- rnorm(12) 
@ 
```
변수 x를 사용하는 하위 문서를 포함합니다.
```r
<<B, child='chapt1.Rnw'>>= 
@
```

자유도가 12인 카이제곱 확률 변수의 한 실현값은 `\Sexpr{y}`입니다.

청크 B에서 하위 문서를 참조했습니다. 주 문서가 컴파일될 때 knitr는 하위 문서를 찾아 그에 따라 컴파일합니다. 이 시점까지 주 문서 환경의 모든 것(예: 변수 x)을 하위 문서에서 사용할 수 있습니다. 하위 문서는 다음과 같습니다.

이것은 하위 문서입니다.
```r
<<B1>>=

- y <- sum(x^2) 
@
```

우리는 하위 문서에서 새 객체 y를 만들었습니다. 하위 문서가 컴파일된 후 이 객체는

주 문서의 이후 청크에서도 사용할 수 있습니다. 이것이 `\Sexpr{y}`가 작동하는 이유입니다. 참고로 n개의 i.i.d(독립 동일 분포) 표준 정규 확률 변수의 합은 χ2n 분포(n 자유도)를 따르므로 y는 χ212에서 생성된 난수 하나입니다.

청크 참조와 마찬가지로 하위 문서는 중첩 수준에 제한이 없습니다. 하나의 하위 문서가 추가 하위 문서를 가질 수 있으며 하나의 청크에 둘 이상의 하위 문서를 포함할 수 있습니다.

###### 9.3.2 템플릿으로서의 하위 문서

다른 데이터 입력을 사용하여 동일한 템플릿으로 동일한 분석을 수행하는 것이 일반적이며 하위 문서도 이러한 작업에 도움이 될 수 있습니다. 간단한 예로, 주 문서에서 카이제곱 분포의 또 다른 난수를 계속해서 생성해 보겠습니다.

```latex
% second part of book.Rnw 
Continue the above example. Now we change the degrees of freedom to 8.
```
```r
- <<C, eval=TRUE>>= 
x <- rnorm(8) 
@ 
```
그리고 하위 문서를 다시 포함합니다.
```r
- <<D, child='chapt1.Rnw'>>= 
@
```

자유도가 8인 카이제곱 확률 변수의 한 실현값은 `\Sexpr{y}`입니다.

여기서 하위 문서가 하는 일은 x에 대한 제곱합을 계산하여 그 결과를 y에 할당하는 것뿐입니다. 우리가 일반적으로 보는 "순수한 소스 코드"는 아닐지라도 서브루틴과 매우 유사합니다.

청크 참조 및 하위 문서를 사용하면 프로그래밍과 같은 방식으로 분석을 모듈화할 수 있습니다.

###### 9.3.3 독립 실행형 모드

이 절은 LATEX에만 적용됩니다. Rnw 하위 문서는 종종 LATEX 프리앰블(`\documentclass`에서 `\begin{document}`까지의 행)이 없다는 점에서 불완전하므로 직접 컴파일하면 LATEX 오류가 발생하게 됩니다.

하위 문서는 상위 문서와 관련이 있어야 하지만 일부 경우에는 반드시 그런 것은 아닙니다. 때로는 거대한 문서를 구성하기 위한 목적으로만 하위 문서가 존재하며 하위 문서에서의 계산은 상위 문서와 완전히 무관할 수 있습니다. 이 경우 필요한 것은 상위 문서의 프리앰블을 차용하여 결과를 컴파일할 때 하위 문서에 추가하는 것뿐입니다.

`set_parent()` 함수는 하위 문서의 상위 문서를 knitr에 알립니다. 이 함수가 호출되면 knitr는 상위 문서의 프리앰블을 읽고 Rnw 문서를 TEX로 컴파일할 때 이를 하위 문서에 씁니다. 예를 들어 chapt1.Rnw에서 이 작업을 수행할 수 있습니다.

```r
<<parent, include=FALSE>>= 
set_parent("book.Rnw") 
@
```

그러면 book.Rnw의 프리앰블에 정의된 LATEX 스타일이 chapt1.Rnw의 내용이 book.Rnw에 있는 것처럼 chapt1.tex에서 사용 가능하게 됩니다.

### 10

###### 훅(Hooks)

훅은 knitr를 확장하는 중요한 구성 요소입니다. 훅은 knitr의 기본 기능을 넘어서는 작업을 수행하기 위해 사용자가 정의한 R 함수입니다. 훅에는 청크 훅(chunk hooks)과 출력 훅(output hooks) 두 가지 유형이 있습니다. 5.3절에서 이미 몇 가지 기본 제공 출력 훅과 청크 및 인라인 R 출력을 모두 사용자 지정하는 방법을 소개했습니다. 이 장에서는 청크 훅에 초점을 맞춥니다.

###### 10.1 청크 훅

청크 훅은 knit_hooks에 저장되고 사용자 지정 청크 옵션에 의해 트리거되는 함수입니다. 모든 청크 훅에는 before, options 및 envir(나중에 설명됨)라는 세 가지 인수가 있습니다.

###### 10.1.1 청크 훅 생성

청크 훅은 knit_hooks의 기존 훅과 충돌하지 않는 한 임의로 이름을 지정할 수 있습니다. 모든 기본 제공 훅의 이름은 다음과 같습니다.

```r
names(knit_hooks$get(default = TRUE))

## [1] "source" "output" "warning" "message" 
## [5] "error" "plot" "inline" "chunk" 
## [9] "text" "document"
```

예를 들어 margin이라는 이름은 위 이름에 없으므로 청크 훅 이름을 margin으로 지정할 수 있습니다.

```r
knit_hooks$set(margin = function(before, options, envir) {

if (before)

par(mar = c(4, 4, 0.1, 0.1)) 
else NULL 
})
```

99

(플롯 데이터 생략)

- 그림 10.1: 기본 여백(즉, `par(mar = c(5.1, 4.1, 4.1, 2.1))`)이 있는 플롯.


이 훅은 R 기본 그래픽의 `par()`를 사용하여 margin 매개변수를 설정하는 데 사용됩니다(기본 여백이 너무 큰 경우가 많기 때문입니다).

###### 10.1.2 청크 훅 트리거

훅을 정의한 후에는 훅 함수를 실행하기 위해 이름이 같은 청크 옵션을 NULL이 아닌 값으로 설정해야 합니다. 기본적으로 정의되지 않은 모든 청크 옵션은 NULL이므로 아래 청크는 margin = NULL 옵션이 있는 청크와 같으며 청크를 컴파일할 때 방금 정의한 훅을 호출하지 않습니다(그림 10.1).

```r
<<mar-normal>>= 
par(bg = "gray") 
plot(1:10) 
@
```

그러나 margin = TRUE로 설정하면 TRUE는 NULL이 아니기 때문에 청크가 평가되기 전에 훅이 호출됩니다(그림 10.2).
```r
<<mar-small, margin=TRUE>>= 
par(bg = "gray") 
plot(1:10) 
@
```

여백을 더 명확하게 보여주기 위해 플롯 배경을 회색으로 설정했습니다.

(플롯 데이터 생략)

- 그림 10.2: margin 훅(`par(mar = c(4, 4, .1, .1))`)을 사용하여 여백이 더 작은 플롯.


###### 10.1.3 훅 인수

이제 청크 훅의 네 가지 인수에 대해 설명합니다. 참고로 네 가지 인수 모두 선택 사항입니다.

before 논리값. 청크 이전에 훅이 호출되면 TRUE이고

청크 이후에 훅이 호출되면 FALSE입니다.

options 현재 청크 옵션의 목록. 예를 들어 options$label은 현

재 청크 레이블입니다.

envir 현재 코드 청크가 평가되는 환경.

예를 들어 envir$x는 현재 청크의 객체 x입니다(존재하는 경우).
name 현재 훅 함수의 이름.

청크 훅은 청크에 대해 청크 이전과 청크 이후에 두 번 호출됩니다. 위의 margin 훅에서는 청크가 평가되기 전에 `par()`가 호출되었으므로 플롯은 `par()`로 설정된 매개변수를 사용합니다. 청크 이후에 `par()`를 설정하면 플롯이 이미 그려졌으므로 너무 늦습니다(따라서 쓸모가 없습니다).

###### 10.1.4 훅 및 청크 옵션

청크 훅은 해당 청크 옵션이 NULL이 아닌 한 호출되므로, 문서의 모든 청크에 청크 훅을 적용하려면 전역적으로 이러한 청크 옵션을 설정할 수 있습니다. 예:

```r
opts_chunk$set(margin = TRUE)
```

NULL이 아니라는 것이 반드시 TRUE를 의미하는 것은 아닙니다. 위의 예에서 margin = 1 또는 margin = 'hello' 등으로 설정할 수도 있습니다. 이러한 값 역시 NULL이 아니기 때문입니다.

knitr는 임의의 청크 옵션을 허용하므로 청크 훅의 options 인수는 매우 유연할 수 있습니다. 이전 예제에서는 훅에서 이 옵션이 기본적으로 무시되었기 때문에 margin 청크 옵션을 실제로 잘 활용하지 못했습니다. 이제 margin이 `par(mar = ...)`에 전달될 벡터가 되도록 훅을 조금 확장해 보겠습니다.

```r
knit_hooks$set(margin = function(before, options, envir) {

if (before) { 
m <- options$margin 
if (is.numeric(m) && length(m) == 4L) {

par(mar = m) 
}

###### } else NULL 
})
```

여백 매개변수에 `c(4, 4, .1, .1)`이라는 고정 값을 사용하는 대신 이제 길이 4의 임의의 숫자 벡터를 사용할 수 있습니다. 예: 
```r
<<mar-numeric, margin=c(2, 3, 1, .1)>>= 
plot(1:10) 
@
```

그러면 이 청크가 평가되기 전에 `par(mar = c(2, 3, 1, .1))`이 먼저 호출됩니다.

###### 10.1.5 출력 쓰기

청크 훅은 함수이므로 반환된 값도 있습니다. 반환된 값이 문자이면 출력에 쓰입니다. 이전 훅들은 문자 값을 반환하지 않았기 때문에 출력에 아무것도 쓰지 않았습니다(`par()`는 목록을 반환함).

다음은 문자 값을 반환하는 훅입니다. 청크 앞의 아래 중괄호와 청크 뒤의 위 중괄호입니다.

```r
knit_hooks$set(brace = function(before, options, envir) {

if (before) {

"\\noindent\\downbracefill{}\n\n"

###### } else {

"\n\n\\noindent\\upbracefill{}\n" 
}

})
```

이 brace 훅을 다음 청크에 적용합니다.

```r
<<test, brace=TRUE>>= 
1 + 1

## [1] 2 
rnorm(10)

## [1] -0.1738 1.1675 0.8677 -0.8149 -1.6213 0.8553 
## [7] -1.8358 -0.7550 -1.6286 -0.6447

@
```

문자 값을 반환하는 청크 훅을 사용하면 우리가 원하는 무엇이든 청크 출력에 쓸 수 있습니다. 한 가지 중요한 용도는 청크의 R 코드를 통해 만든 이미지를 출력에 쓰는 것입니다. 문자 값은 `\includegraphics{...}`(LATEX), `<img src='...' />`(HTML) 또는 `![](...)`(Markdown) 등과 같을 수 있습니다. 이것은 rgl 및 GGobi 플롯을 저장하는 것과 같이 다음 몇 절에서 사용할 요령입니다.

###### 10.2 예제

이 절에서는 청크 훅의 몇 가지 예제를 제공하며, 그중 대부분은 knitr에 미리 정의되어 있습니다. 즉, knitr를 로드한 직후에 직접 사용할 수 있습니다.

###### 10.2.1 플롯 자르기

일부 R 사용자는 특히 기본 그래픽(ggplot2가 일반적으로 이 측면에서 더 나음)에서 R 플롯의 추가적인 흰색 여백으로 인해 어려움을 겪었을 것입니다. 그림 10.1(또한 `?par` 참조)에서 언급했듯이

기본 그래픽 옵션 mar은 약 `c(5, 4, 4, 2)`이며 이는 보통 너무 큽니다.

(플롯 데이터 생략)

- 그림 10.3: R에서 생성된 여백이 큰 원본 플롯.


끝없이 `par(mar)`를 조정하는 대신 흰색 여백을 자동으로 자를 수 있는 프로그램인 pdfcrop(http://www.ctan.org/pkg/pdfcrop)을 고려할 수 있습니다. knitr에서 청크 옵션(예: crop)과 함께 작동하도록 훅 `hook_pdfcrop()`을 설정할 수 있습니다.

```r
knit_hooks$set(crop = hook_pdfcrop)
```

이제 아래의 동일한 코드 청크에 의해 생성된 두 개의 플롯을 비교합니다. 첫 번째는 잘리지 않았습니다(그림 10.3). 그리고 동일한 플롯이 생성되지만 청크 옵션 crop = TRUE를 사용하면 자르기 훅을 호출합니다(그림 10.4).

```r
par(mar = c(5, 4, 4, 2)) # large margin 
plot(lat ~ long, data = quakes, pch = 20, col = rgb(0, 0,

0, 0.2))
```

보시다시피 흰색 여백이 사라졌습니다(차이를 더 잘 확인하기 위해 각 플롯 주위에 프레임 상자를 넣었습니다). `par()`를 사용하면

(플롯 데이터 생략)

- 그림 10.4: 잘린 플롯. 상단 및 오른쪽의 흰색 여백이 확실히 제거되었습니다.


너무 작은 여백으로 인해 레이블이 잘리지 않으면서 너무 크지 않은 적절한 양의 여백을 찾는 것이 어렵고 지루할 수 있습니다.

###### 10.2.2 rgl 플롯

훅 `hook_rgl()`을 사용하여 rgl 패키지(Adler 및 Murdoch, 2014)에서 스냅샷을 쉽게 저장할 수 있습니다. rgl 훅은 훅에서 options 인수를 신중하게 사용하여 세부 사항을 처리하는 좋은 예입니다. 예를 들어 `rgl.snapshot()` 또는 `rgl.postscript()`에서 rgl 플롯의 너비와 높이를 직접 설정할 수 없으므로 options fig.width, fig.height 및 dpi를 사용하여 창의 예상 크기를 계산한 다음 `par3d()`로 현재 창의 크기를 조정하고 플롯을 저장한 다음 마지막으로 플롯을 출력에 삽입하기 위한 적절한 코드가 포함된 문자열을 반환합니다. 다음은 `hook_rgl()`의 간단하고 빠른 버전입니다.

```r
knit_hooks$set(rgl = function(before, options, envir) { 
library(rgl) 
if (before || rgl.cur() == 0)

return() # return nothing before a chunk

name <- paste(options$fig.path, options$label, sep = "")
```

(플롯 데이터 생략)

- 그림 10.5: `hook_rgl()`로 캡처한 rgl 플롯: 이 훅 함수는 rgl의 `rgl.snapshot()`을 호출하여 스냅샷을 PNG 이미지에 저장합니다.
- `width`, `height`는 각각 `fig.width`와 `fig.height`로 이름이 변경되었습니다. 한편, `\SweaveOpts{}`와 `\SweaveInput{}`은 사용되지 않습니다(deprecated). 전역 청크 옵션을 설정하려면 `opts_chunk$set()`을, 자식 문서를 포함하려면 청크 옵션 `child`를 사용하세요.

논리(logical) 옵션의 경우 `TRUE/FALSE/T/F`만 지원되며(처음 두 개를 권장함), `true/false`는 작동하지 않습니다. 예를 들어 `eval = FALSE`는 괜찮지만 `eval = false`는 (공교롭게도 논리값 `FALSE`를 취하는 `false`라는 R 객체가 없는 한) 작동하지 않습니다. `<<label>>` 구문을 사용한 청크 참조는 여전히 사용할 수 있으며, 청크를 재사용하는 다른 방법(예: 새 옵션 `ref.label` 사용)도 있습니다. 9장에서 소개한 것처럼 청크 참조는 재귀적일 수 있습니다.

###### 16.1.3 문제점

Sweave에서 알려진 몇 가지 문제점과 자주 묻는 질문은 knitr에서 해결되었습니다.

- • Sweave에서는 빈 그림 청크가 LATEX 오류를 발생시키지만 knitr에서는 오류를 발생시키지 않습니다. 왜냐하면 그림이 아예 생성되지 않기 때문입니다. knitr는 청크에 플롯이 있을 때만 그림을 LATEX에 씁니다.
- • lattice(및 ggplot2) 그래픽은 명시적으로 `print()`하지 않으면 Sweave에서 작동하지 않지만, knitr에서는 R 콘솔과 똑같이 작동합니다(이러한 플롯 개체가 최상위 환경에 나타나면 인쇄할 필요가 없습니다).
- • Sweave에서는 LATEX 스타일 `Sweave.sty`에 정의된 `\setkeys{Gin}{width=.8\textwidth}`를 통해 출력되는 그림의 너비가 기본적으로 `.8\textwidth`로 설정됩니다. 이는 Sweave에서 생성되었는지 여부와 상관없이 문서의 모든 그림에 영향을 미치며, 그림의 개별 너비를 설정하는 간단한 방법이 없습니다. 이 문제는 knitr의 `out.width` 옵션으로 해결되었습니다.
- • 기본적으로 Sweave에서는 하나의 그림 청크에서 여러 그림을 생성하는 것이 작동하지 않으며 이 경우 LATEX 코드를 직접 작성해야 합니다. 반면 knitr의 경우 한 청크에 몇 개의 플롯이 있든 상관없이 잘 작동합니다.
- • 출력 후크를 사용하여 knitr에서 출력 형식을 변경할 수 있으며 Sweave에서처럼 하드 코딩된 LATEX 환경(예: Sinput/Soutput)을 사용할 필요가 없습니다. 실제로 `render_sweave()`를 호출하여 knitr에서 Sweave 스타일을 렌더링할 수 있습니다.
- • knitr(R HTML 또는 R Markdown)를 사용하면 HTML 출력을 쉽게 생성할 수 있지만, Sweave는 HTML만 처리하는 R2HTML과 같은 확장이 필요합니다.


간혹 Sweave를 실행한 후 생성되지 않아야 할 `Rplots.pdf` 파일이 보이는 경우가 있는데, 비대화형 R 세션에 대한 R의 기본 그래픽 장치가 `pdf()`여서 `Rplots.pdf`를 생성하기 때문입니다. knitr에서는 잘못된 PDF 파일이 생성되지 않도록 기본 장치가 널(null) 장치(`pdf(file = NULL)`)로 설정되어 있습니다.

###### 16.2 기타 R 패키지

Sweave 및 아래에 소개된 R 패키지(R2HTML 제외)의 대부분의 기능은 knitr에서 다루고 있으므로 이 절은 주로 역사적인 관심사를 위한 것입니다.

highlight 패키지(Francois, 2013)는 Rnw 문서에서 R 코드에 대한 구문 강조(syntax highlighting)를 제공합니다. 아래의 pgfSweave, cacheSweave, R2HTML과 마찬가지로 highlight는 Sweave를 기반으로 확장되었습니다. 초기 버전(v0.6 이전)에서 knitr는 구문 강조를 수행하기 위해 highlight에 의존했지만 이 종속성은 유지 관리 문제와 추가 종속성(Rcpp 및 parser 패키지)이 있다는 사실 때문에 나중에 제거되었습니다. 이제 knitr는 자체 구문 강조 함수를 사용하며, 이 함수는 R 3.0.0 이전에는 정규 표현식을 기반으로 하고 R 3.0.0 이후에는 기본 R의 utils 패키지에 있는 `getParseData()` 함수에 의존합니다. highlight와 유사한 기능을 얻으려면 knitr에서 `highlight = TRUE` 청크 옵션을 사용하기만 하면 됩니다.

cacheSweave 패키지(Peng, 2012)는 Sweave에 중요한 기능인 캐시 시스템을 추가했습니다. weaver 패키지(Falcon, 2013)도 다른 구현으로 유사한 작업을 수행했습니다. `cache` 및 `dependson` 청크 옵션이 추가되었으며, 이는 knitr에서와 동일한 의미를 갖습니다(8장 참조).

pgfSweave 패키지(Bracken and Sharpsteen, 2012)는 highlight 및 cacheSweave의 기능을 결합하고 그래픽에 대한 추가 지원을 제공했습니다. 특히 플롯도 캐시할 수 있으며 글꼴 스타일의 일관성을 위해 tikzDevice 패키지를 통한 TikZ 그래픽도 지원됩니다. 이 책의 저자는 pgfSweave가 나왔을 때 Sweave에서 전환했고 여기에 formatR 지원(`tidy` 옵션)을 기여했지만, 시간이 지남에 따라 Sweave의 변화를 따라가기가 점점 더 어려워졌습니다. 이 패키지는 CRAN 리포지토리에서 제거되었습니다. 어쨌든 knitr의 디자인은 저자가 pgfSweave를 사용한 경험에서 많은 이점을 얻었습니다.

brew 패키지(Horner, 2011)는 가벼운 템플릿 프레임워크이며 구문은 PHP(`<?php ?>`)와 유사합니다. 기본적으로 템플릿 태그 `<% %>` 내부의 R 코드를 분석하고 실행합니다. 이것을 Sweave 및 knitr의 인라인 R 코드라고 생각할 수 있습니다. 캐시 시스템은 있지만 직접적인 그래픽 지원은 없습니다. knitr 패키지는 또한 brew 구문을 부분적으로 지원하는데, 5장에서는 언급하지 않았습니다. 아래는 knitr를 통해 컴파일할 수 있는 예제입니다.

`The value of pi is <% pi %>, and 2 times pi is <% 2*pi %>.`

입력 파일의 확장자가 `*.brew`인 경우 knitr는 자동으로 brew 구문을 사용합니다. brew는 실제로 여러 인라인 표현식에서 불완전한 코드 조각을 지원하므로 PHP와 정말 유사합니다. 다음은 brew에서 가져온 예이지만 knitr에서는 이 코드를 컴파일할 수 없습니다.

`<% for (i in c('1+1', '1+pi', '1+pi', 'sin(pi/2)')) { -%> > <%=i%> <% print(eval(parse(text=i))) %> <% } -%>`

R2HTML 패키지(Lecoutre, 2014)에는 R 객체를 HTML로 내보내는 수많은 함수가 포함되어 있습니다. 기본 함수는 S3 일반 함수 `HTML()`로, 데이터 프레임, 테이블, `lm` 객체(`lm()`에 의해 반환됨) 등과 같은 다양한 R 객체에 적용할 수 있습니다. 아래는 iris 데이터의 부분 집합을 HTML 테이블로 변환한 것입니다.

`library(R2HTML) HTML(head(iris[, -5], 1), "", caption = NULL)`

`<p align= center > <table cellspacing=0 border=1><tr><td>`

`<table border=0 class=dataframe> <tbody> <tr class= firstline >`

`<th>&nbsp; </th> <th>Sepal.Length </th> <th>Sepal.Width </th> <th>Petal.Length </th> <th>Petal.Width</th>`

`</tr>`

`<tr> <td class=firstcolumn>1 </td> <td class=cellinside>5.1 </td> <td class=cellinside>3.5 </td> <td class=cellinside>1.4 </td> <td class=cellinside>0.2 </td></tr>`

`</tbody> </table>`

`</td></table>`

청크 옵션 `results = 'asis'`를 사용하면 R HTML 문서의 knitr 내부에서 R2HTML을 활용하여 원시 HTML 코드를 출력에 쓸 수 있습니다.

R2HTML의 또 다른 주요 기여는 Sweave 확장으로, Sweave를 기반으로 HTML 보고서를 작성할 수 있게 해줍니다.

CRAN에는 재현 가능한 연구(reproducible research)에 대한 작업 보기(task view)가 있습니다: http://cran.r-project.org/web/views/ReproducibleResearch.html. 여기에서 이 주제에 대한 더 많은 패키지를 찾을 수 있습니다.

###### 16.3 Python 패키지

이 섹션에서는 동적 문서를 위한 Python 기반 패키지 3개(Dexy, PythonTEX, IPython)를 소개합니다.

###### 16.3.1 Dexy

Dexy(http://www.dexy.it) 는 매우 범용적인 디자인을 특징으로 하는 무료 Python 패키지입니다. 웹사이트에 따르면:

Dexy는 코드가 포함된 모든 종류의 기술 문서를 작성하기 위한 자유로운 형식의 문학적 문서화(literate documentation) 도구입니다. Dexy는 올바른 문서를 작성하고, 시간이 지나면서 코드가 변경되어도 쉽게 유지 관리할 수 있도록 도와줍니다.

네 가지 주요 기능은 다음과 같습니다.

- 1. 모든 언어 (소스 코드)
- 2. 모든 마크업 (출력)
- 3. 모든 템플릿
- 4. 모든 API (프로그래밍)


다중 언어 지원과 같이 Dexy와 knitr 사이에는 분명 유사점이 있습니다. Dexy의 중요한 개념은 "필터(filter)"입니다. 필터는 셸 스크립트의 파이프 `|`와 유사하게 입력 파일을 받아서 출력 파일로 변환합니다. Dexy의 필터는 사실 knitr 개념들의 조합입니다. 필터는 (예: Markdown에서 HTML로) 출력을 렌더링하거나, (knitr의 언어 엔진과 같이) 프로그래밍 언어를 실행하거나, knitr의 청크 후크와 같은 추가 작업을 수행할 수 있습니다.

일반적으로 Dexy는 컴퓨터 코드를 템플릿과 분리하는데, 이는 좋을 수도 있고 나쁠 수도 있습니다. 좋은 점은 소스 스크립트를 재사용할 수 있다는 것이고, 나쁜 점은 보고서 환경과 소스 코드 사이를 왔다 갔다 해야 한다는 것입니다. 기본적으로 knitr는 코드 청크를 보고서에 직접 포함하지만, 9장에 소개된 것처럼 코드 청크를 외부화할 수도 있습니다.

###### 16.3.2 PythonTEX

PythonTEX(https://github.com/gpoore/pythontex) 는 LATEX 패키지이며 LATEX 내에서 Python 코드를 실행하는 기능을 제공합니다. 설명서에 따르면:

PythonTEX는 LATEX 내에서 빠르고 사용자 친화적인 Python 액세스를 제공합니다. LATEX 문서에 입력된 Python 코드를 실행하고 그 결과를 원본 문서에 포함할 수 있습니다. 또한 Pygments 패키지를 통해 LATEX 문서 내 코드에 대한 구문 강조 기능을 제공합니다.

`\pyb{}` 명령을 사용하여 인라인 Python 코드를 삽입하거나 `pyconsole` 환경을 사용하여 LATEX에서 Python 세션을 에뮬레이션할 수 있습니다. 예를 들어,

`\begin{pyconsole}[][frame=single]`

`- x = 123`
`- y = 345`
`- z = x + y z def f(expr):`

`return(expr**4)`

`- f(x) print('Python says hi from the console!') \end{pyconsole}`

이 문서를 컴파일하면 Python 코드가 평가되고 그 결과가 출력에 삽입됩니다.

Python에서 기원했기 때문에 SymPy(기호 조작) 및 matplotlib(플롯)와 같은 다른 Python 패키지와의 통합 기능도 있습니다.

###### 16.3.3 IPython

IPython(http://ipython.org) 은 코드, 텍스트, 수식, 인라인 플롯 및 기타 리치 미디어를 지원하는 웹 기반 노트북, 병렬 컴퓨팅을 위한 고성능 도구 등을 특징으로 하는 Python용 대화형 셸입니다.

그림 16.1은 Ubuntu의 GNOME 터미널에 있는 IPython의 스크린샷입니다. 명령 자동 완성 기능과 같은 셸의 기본 기능이 있음을 알 수 있습니다. 셸에 `x.spl<TAB>`을 입력하면 아래와 같은 자동 완성 기능이 표시됩니다.

보고서 생성과 관련된 가장 눈에 띄는 기능은 웹 기반 노트북입니다. 웹 브라우저에서 Python 명령으로 작업하고, 즉석에서 결과(숫자 및 그래픽 결과 모두 포함)를 볼 수 있으며, 노트북에 내용을 추가로 입력함에 따라 노트북을 지속적으로 업데이트할 수 있습니다. 이는 knitr에서 코드 청크를 작성하는 것과 매우 비슷합니다.

IPython 노트북은 확장이 `*.ipynb`인 JSON 파일로 저장하여 다른 사람과 공유할 수 있습니다. 노트북에는 출력이 포함될 수도 있고 포함되지 않을 수도 있습니다. 출력이 없는 노트북은 knitr용 소스 문서(예: Rnw 및 Rmd 문서)와 유사합니다.

IPython에서 영감을 받아 knitr도 유사한 웹 노트북(기능은 더 적음)을 가지게 되었으며, 3.2.2절에서 언급한 바 있습니다.

![image 34](Dynamic Documents with R and knitr 2nd_images/imageFile34.png)

그림 16.1: IPython의 스크린샷: 입력은 `In[n]`으로 표시되고 출력은 `Out[n]`으로 표시됩니다.

###### 16.4 추가 도구

R 및 Python 패키지 외에도 다른 프로그램의 도구가 있습니다. 이 장에서 동적 문서를 위한 모든 도구를 열거하는 것은 불가능합니다. Schulte 등(2012)은 Javadoc, cweb, noweb, Sweave, SASweave 등과 같이 문학적 프로그래밍과 재현 가능한 연구를 위한 기존 도구 목록을 제공했습니다.

###### 16.4.1 Org-mode

Org-mode는 일반 텍스트 마크업 언어이며 Emacs 텍스트 편집기에 구현되어 있습니다(Schulte 등, 2012). 문학적 프로그래밍과 재현 가능한 연구(동적 문서의 관점에서)를 모두 지원합니다. 다소 차이는 있지만 WEB 및 noweb과 같은 문학적 프로그래밍의 초기 구현 구문을 어느 정도 따릅니다. 즉, 코드 청크와 텍스트 청크(텍스트 청크는 종종 "산문(prose)"이라고 함)의 개념이 있습니다. Org-mode의 코드 청크는 다음과 같습니다.

`#+name: c-chunk #+begin_src C`

`int main(){`

`return 0; }`

`#+end_src`

비교를 위해 같은 청크를 knitr에서는 다음과 같이 작성합니다. 
`<<c-chunk, engine='c'>>= int main(){`

`return 0;`

`}`
`@`

메타데이터는 청크 헤더에 저장됩니다. Org-mode는 LATEX 또는 HTML을 출력 형식으로 사용하여 모든 입력 언어를 지원합니다.

Schulte 등(2012)은 기존 도구의 문학적 프로그래밍 기능(예: Sweave에는 이 기능이 없음)을 언급했지만, 보고서 작성자에게 흥미롭게 들리지 않기 때문에 이 책에서는 강조하지 않았습니다. 사실 knitr에도 코드 청크를 재구성하는 기능이 있습니다(9장 참조). 다음은 나중에 청크 B를 정의하되 이전 청크 A에 포함시키는 간단한 예입니다.

`- <<A>>= df <- data.frame(x = 1:10, y = rnorm(10))`
`- <<B>> coef(fit) @`


`<<B>>= fit <- lm(y ~ x, data = df) @`

Org-mode가 강력하긴 하지만, Emacs의 특성이 초보자에게는 장애물이 될 수 있습니다.

###### 16.4.2 SASweave

SASweave(http://homepage.cs.uiowa.edu/~rlenth/SASweave) 는 SAS 및 R을 사용한 문학적 프로그래밍의 구현체입니다. gawk로 작성되었습니다. 기본적인 아이디어는 Sweave 및 knitr와 같습니다. 자세한 내용은 Lenth와 Højsgaard(2007)를 참조하십시오. knitr 패키지는 SASweave에 비해 R에 대해서는 더 포괄적으로 지원하지만 SAS에 대해서는 덜 지원합니다.

###### 16.4.3 Office

우리가 이 책에서 소개한 거의 모든 것이 일반 텍스트를 기반으로 하지만, 동적 문서에 대해 반드시 일반 텍스트 형식을 선택할 필요는 없습니다. OpenOffice(또는 OpenDocument Text)나 Microsoft Office 제품(줄여서 Office 문서라고 함)을 기반으로 하는 도구들이 있으며, 이들은 언뜻 보기에 매력적일 수 있습니다. 핵심적으로 Office 문서는 대개 (압축될 수 있는) XML 파일이므로 코드 청크를 포함하는 것이 가능합니다. 코드 청크를 분석하고 실행한 다음 결과를 다시 삽입할 수 있습니다.

우리가 보는 주요 문제는 XML 형식이 너무 복잡하고 표준이 너무 많아서 수정된 문서가 여전히 유효한 Office 문서인지 확인하는 것이 간단하지 않다는 것입니다. 한 가지 예로, StatWeave 패키지(http://homepage.stat.uiowa.edu/~rlenth/StatWeave/) 는 "OpenOffice가 수정된 문서를 손상된 것으로 표시"하기 때문에 더 이상 OpenOffice(3.2 이상)에서 작동하지 않습니다.

이에 비해 일반 텍스트 파일은 다루기가 훨씬 쉽습니다. ECMA-376과 같이 신경 써야 할 복잡한 표준이 없습니다. 굳이 Office 문서를 원한다면 적어도 Markdown에서 변환할 수 있는 방법은 있습니다. 1장에서 인용한 내용을 떠올려보세요.

소스 코드는 진짜다(The source code is real).

### A

###### 내부(Internals)

이 부록에서는 knitr 패키지의 일부 내부 구조를 설명합니다. 이는 다른 개발자가 이 패키지를 더 잘 이해하고 필요한 경우 코드에 기여하는 데 도움이 될 수 있습니다. 일반 사용자는 이 부록을 읽을 필요가 없습니다. 내부 구조를 세 가지 측면, 즉 문서, 클로저(closures)의 적용, 그리고 일부 기능의 구현으로 나누어 설명합니다.

###### A.1 문서

knitr에는 R 설명서(Rd), PDF 매뉴얼, 웹사이트의 세 가지 유형의 문서가 있습니다.

R 설명서는 roxygen2(Wickham 등, 2015)를 기반으로 합니다. 이 도구를 사용하면 태그가 있는 roxygen 주석(`# '`)으로 Rd를 작성할 수 있으며 이 주석은 실제 Rd로 변환됩니다. 다음은 roxygen 주석의 예입니다.

`#' @author Yihui Xie`
이 주석은 `\author{Yihui Xie}`와 같이 Rd로 변환됩니다.

roxygen에는 `@usage`, `@param`, `@return`, `@examples`와 같은 일련의 태그가 있으며, 이는 Rd에서 각각 `\usage{}`, `\arguments{\item{}}`, `\value{}`, `\examples{}`에 해당합니다. 공식 Rd보다 roxygen 주석을 작성하는 것의 장점은 문서와 소스 코드를 동일한 파일에 보관할 수 있다는 것입니다. 반면, R 패키지를 작성하는 공식적인 방법은 R 소스를 `R/` 디렉터리 아래에 작성하고 매뉴얼 페이지를 `man/` 아래에 `*.Rd` 파일로 작성하는 것입니다. 두 파일 사이를 번갈아 가며 작업해야 하고 R 소스를 업데이트하면서 문서를 업데이트하는 것을 잊어버리기 쉽기 때문에 이 방법은 불편합니다. roxygen 주석은 소스의 R 함수 바로 위에 나타나므로 소스와 문서를 모두 유지 관리하기가 훨씬 쉽습니다.

아래는 roxygen 주석으로 문서화된 함수의 완전한 예입니다.

`#' Repeat a character string #' #' Repeat a string n times and make one string. #' @param x a character string #' @param n an integer #' @return A character string. #' @examples f('hi', n = 5) f <- function(x, n = 10) {`

`paste(rep(x, n), collapse = "") }`

roxygen2 패키지의 `roxygenize()` 함수를 사용하여 roxygen 주석을 공식 Rd 파일로 변환할 수 있습니다. knitr의 모든 객체는 이 방식으로 문서화됩니다. 게다가 roxygen2는 `DESCRIPTION`의 `NAMESPACE`와 `Collate` 필드도 자동으로 처리하므로 R 소스 파일 작업에 집중할 수 있습니다.

PDF 매뉴얼의 소스 문서는 `examples` 디렉터리(소스 패키지의 `inst/examples/` 참조) 아래에 있습니다. 예를 들어 기본 매뉴얼은 `knitr-manual.Rnw`입니다. Rnw 파일은 LYX 파일에서 내보내기 되므로(4.2절) PDF 매뉴얼을 편집하거나 컴파일하려면 LYX 파일을 여는 것이 좋습니다. PDF 매뉴얼은 소스 패키지와 함께 제공되지 않는데, 그 이유는 (1) 이진 파일을 버전 제어 하에 두고 싶지 않고(특히 소스 파일의 부산물일 때), (2) 패키지 웹사이트에 호스팅되기 때문입니다.

패키지 웹사이트는 13.4절에서 소개한 대로 Jekyll을 기반으로 구축되었습니다. 구체적으로 모든 페이지는 Markdown으로 작성되며 Git 저장소의 `gh-pages` 브랜치에 배치됩니다(패키지 자체는 `master` 브랜치에 있습니다). Git을 통해 변경 사항이 푸시되면 Github에서 웹사이트를 자동으로 다시 빌드합니다. 웹사이트에 기여하고 싶다면 `gh-pages` 브랜치로 전환하고 Markdown 파일을 업데이트하기만 하면 됩니다.

###### A.2 클로저(Closures)

클로저는 knitr에서 중심적인 역할을 합니다. `opts_chunk`(5.1.1절) 및 `knit_engines`(11장)와 같은 몇 가지 공통 객체는 클로저를 기반으로 구축됩니다.

클로저는 본질적으로 함수이며 비지역(non-local) 변수에도 접근할 수 있습니다. 아래는 간단한 예입니다.

`f <- function() { x <- 1 function(y) x + y`

`}`
`g <- f()`

`g(5) # add 5 to x ## [1] 6 ls(environment(g)) # g can see x ## [1] "x"`

함수 `g()`는 `f()`에서 생성되었고(`f()`는 함수를 반환함을 유의하세요), `g()`는 `f()` 내부에 생성된 객체 `x`를 사용하며, `x`는 `f()`에만 존재합니다. `g()`가 호출되는 위치에 상관없이 항상 이 `x`에 접근할 수 있습니다.

사실 클로저를 통해 비지역 변수를 수정할 수도 있습니다. 아래는 청크 옵션 관리자인 `opts_chunk`가 어떻게 작동하는지 보여주는 최소한의 예입니다.

`new_list <- function(default = list()) {`

`list(get = function() default, set = function(...) { x <- list(...) if (length(x)) default[names(x)] <<- x`

`}) }`

`new_list()` 함수는 함수 목록(세터(setter) 및 게터(getter))을 반환합니다. `default` 객체는 이 두 함수에 바인딩됩니다. 이를 청크 옵션의 기본 목록이라고 생각할 수 있습니다. 다음으로 청크 옵션을 가져오고 설정하는 방법을 보여드리겠습니다.

`opts <- new_list(list(eval = TRUE)) str(opts$get())`

`## List of 1 ## $ eval: logi TRUE`

`opts$set(eval = FALSE) # change eval to FALSE opts$set(results = "markup") # add a chunk option str(opts$get())`

`## List of 2 ## $ eval : logi FALSE ## $ results: chr "markup"`

`opts$set(results = "hide") # change the results option`

`$set()` 함수에서는 `<<-`를 사용하여 인수를 `default` 객체에 할당했습니다. 이것이 상위(parent) 환경에서 이 객체를 수정할 수 있는 이유입니다(일반적인 `<-`를 사용했다면 상위 환경의 `default`는 수정되지 않고 대신 지역(local) 사본이 생성되었을 것입니다).

클로저를 사용함으로써 knitr는 동일한 구문으로 고유한 환경에서 객체를 관리할 수 있습니다. knitr의 내부 함수 `new_defaults()`는 이러한 클로저 목록을 만드는 데 사용됩니다.

(`opts_chunk`(청크 옵션 관리용) 및 `knit_engines`(언어 엔진 관리용) 외에도 다음과 같은 몇 가지 유사한 객체가 있습니다.

- `opts_knit`: 패키지 옵션(12.2절)
- `opts_current`: 현재 청크에 대한 청크 옵션
- `opts_template`: 청크 옵션 템플릿(12.1.2절)
- `knit_hooks`: 후크 함수(출력 후크 및 청크 후크 모두)
- `knit_patterns`: 파서에 대한 구문 패턴(5.1절)


###### A.3 구현

이 섹션에서는 이 패키지의 몇 가지 구현 세부 사항을 설명합니다. 먼저 언급할 한 가지 사소한 점은 제가 할당 연산자로 `<-` 대신 `=`를 사용한다는 것이며, 소스 코드 곳곳에서 `=`를 보게 될 것입니다. 이는 개인적인 취향 문제이며 딱히 단점은 없다고 생각하지만, 이 패키지에 코드를 기여할 때는 `=`를 따르셔야 합니다. 이 책에서는 제가 등호를 입력했지만 formatR이 자동으로 교체했기 때문에 `<-`가 표시됩니다.

###### A.3.1 파서(Parser)

문서 파서(5.1절)는 다음과 같이 작동합니다. 구문 패턴 객체의 자식 요소인 `chunk.begin`과 `chunk.end`를 사용하여 문서를 조각(코드 청크 및 텍스트 청크)으로 분할합니다. 코드 청크의 경우 청크 옵션(즉, 첫 번째 줄에서 추출한 텍스트)이 R 코드로 구문 분석되며, 이것이 청크 옵션이 R 구문을 따라야 하는 이유입니다. 다음은 knitr가 텍스트 조각에서 청크 옵션을 가져오는 방법을 설명하는 예입니다.

`## suppose this is the chunk options text txt <- "label, eval=TRUE, echo=1:3, foo=if(TRUE) 2 else 5" opc <- eval(parse(text = paste("alist(", txt, ")"))) names(opc) # the chunk label is not named`

`## [1] "" "eval" "echo" "foo" str(opc) # some are unevaluated expressions`

`## List of 4 ## $ : symbol label ## $ eval: logi TRUE ## $ echo: language 1:3 ## $ foo : language if (TRUE) 2 else 5`

먼저 텍스트 주위에 `alist()` 함수를 추가했는데, 이 함수는 인수가 마치 함수 인수를 설명하는 것처럼 취급하므로 이 시점에서는 어떤 "인수"도 평가되지 않습니다. 그러나 최소한 구문은 유효해야 합니다. 한 가지 예외는 청크 레이블입니다. 이는 문자열로 간주되므로 필요한 경우 자동으로 따옴표로 묶입니다. 내부 함수 `parse_params()`는 청크 옵션을 구문 분석하는 데 사용됩니다.

`p <- knitr:::parse_params str(p("chunk-label, eval=TRUE, foo=5"))`

`## List of 3 ## $ label: chr "chunk-label" ## $ eval : logi TRUE ## $ foo : num 5`

`# 2a is not a valid symbol in R, but knitr will quote it # automatically so parsing is OK parse(text = "alist(2a)") ## Error: <text>:1:8: unexpected symbol ## 1: alist(2a ## ^ str(p("2a, eval=FALSE"))`

`## List of 2 ## $ label: chr "2a" ## $ eval : logi FALSE`

`str(p("'2a', eval=FALSE")) # or you can quote it manually`

`## List of 2 ## $ label: chr "2a" ## $ eval : logi FALSE`

청크 옵션은 청크가 실행되기 전까지 평가되지 않으므로 파싱 시 문서 내에서 값을 알 수 없는 객체를 청크 옵션에 사용할 수 있습니다. 예를 들어 위의 옵션 `echo`와 `foo`는 평가되지 않은 표현식이며, 나중에 명시적으로 평가할 것입니다.

`eval(opc$echo) ## [1] 1 2 3 eval(opc$foo) ## [1] 2`

모든 코드 청크는 내부 객체 `knit_code`에 명명된 목록(named list)으로 저장됩니다. 이름은 청크 레이블이고 내용은 코드입니다. 이 객체 역시 클로저 목록으로 생성되므로 `get()` 및 `set()` 메서드가 있지만 예상치 못한 결과가 발생할 수 있으므로 이 객체를 수정하는 것은 권장되지 않습니다. 필요한 경우 `knitr:::knit_code$get('chunk-label')`을 통해 코드 청크에 접근할 수 있습니다.

###### A.3.2 청크 후크

`knit_hooks`에는 출력 후크인(5.3절) 여러 기본 후크가 있습니다.

`names(knit_hooks$get(default = TRUE))`

`## [1] "source" "output" "warning" "message" ## [5] "error" "plot" "inline" "chunk" ## [9] "text" "document"`

이 객체의 다른 모든 후크는 청크 후크(10장)로 취급됩니다. 코드 청크가 실행되기 전과 후에 모든 추가 후크가 호출됩니다. 다음은 의사(pseudo) 코드입니다.

`hook(before = TRUE, ...) evaluate(code) hook(before = FALSE, ...)`

명심해야 할 한 가지 문제는 후크가 실행되는 순서입니다. `knit_hooks`에 두 개의 후크 A와 B가 정의된 경우 어떤 순서로 호출될까요? 이 순서는 청크 옵션에서 가져옵니다. 이 두 후크에 해당하는 두 개의 청크 옵션 A와 B가 있어야 하며 청크 옵션의 순서가 후크를 실행하는 순서를 결정합니다. 예를 들어 A가 B 앞에 있으면 후크 A가 B보다 먼저 호출됩니다. 그러나 코드 청크가 평가된 후에는 이 순서가 역전되며 그 이유는 후크가 반환하는 결과가 쌍(pair)을 이루어 그룹화되도록 하기 위함입니다. 예를 들어 후크 A가 청크 앞에서는 `\begin{Aenvir}`를 반환하고 청크 뒤에서는 `\end{Aenvir}`를 반환한다고 가정해 보겠습니다. 이와 비슷하게 B는 Benvir를 반환합니다. 그러면 우리가 출력에서 원하는 것은 다음과 같습니다.

`\begin{Aenvir} \begin{Benvir} % results from the chunk \end{Benvir} \end{Aenvir}`

`\end{Benvir}`가 `\end{Aenvir}`보다 먼저 온다는 점에 유의하세요. 이러한 이유로 후크 A와 B가 정의된 경우 다음 두 청크는 서로 다른 결과를 반환합니다.

`- <<A=TRUE, B=TRUE>>=`
`- <<B=TRUE, A=TRUE>>=`


###### A.3.3 옵션 별칭(Option Aliases)

청크 옵션 별칭(12.1.1절)을 구현하는 데는 몇 줄밖에 걸리지 않습니다. 이는 목록의 특정 요소를 대체하는 간단한 작업이기 때문입니다. 다음은 아이디어를 보여주는 짧은 함수입니다.

`apply_aliases <- function(x, list) {`

`## names are aliases of x list[x] <- list[names(x)] list`

`} al <- c(w = "fig.width", h = "fig.height", a = "fig.align") op <- list(w = 7, h = 7, echo = TRUE, a = "center") str(op) # user's options`

`## List of 4 ## $ w : num 7`

`## $ h : num 7 ## $ echo: logi TRUE ## $ a : chr "center"`

`str(apply_aliases(al, op)) # corrected options`

`## List of 7 ## $ w : num 7 ## $ h : num 7 ## $ echo : logi TRUE ## $ a : chr "center" ## $ fig.width : num 7 ## $ fig.height: num 7 ## $ fig.align : chr "center"`

별칭은 명명된 문자형 벡터(named character vector)에 설정되며 이름은 벡터에 있는 요소의 별칭입니다. 위 예제에서 `apply_aliases()`는 사용자가 지정한 `w` 및 `h` 값에 따라 각각 `fig.width` 및 `fig.height` 요소를 목록 `op`에 추가했지만 내부적으로 knitr는 여전히 `fig.width` 및 `fig.height`를 사용합니다.

###### A.3.4 캐시

knitr의 캐시 또한 클로저로 구성된 객체에 의해 관리되지만 좀 더 복잡합니다(내부 함수 `new_cache()` 참조). 클로저는 캐시 파일을 저장, 로드 및 삭제하는 데 사용되며 여기서는 캐시의 한 가지 측면만 설명합니다. 바로 출력(printing)의 부작용이 캐시되는 방식입니다(8.4절).

5.3절에서 언급했듯이 코드 청크는 evaluate 패키지에 의해 평가됩니다. 사실 출력 결과는 문자열로 반환되며 전체 청크의 출력도 (출력 렌더러에 의해 포맷된) 문자열입니다. 이 문자열은 변수에 할당되며 변수 이름은 MD5 해시와 청크 레이블로 구성됩니다. 이 변수는 청크에서 생성된 모든 다른 변수와 함께 캐시 데이터베이스에 저장됩니다. 다음에 청크를 평가해야 할 때 knitr는 청크를 업데이트해야 하는지 확인합니다. 그렇지 않으면 인쇄된 결과(사실 이 청크의 모든 것)도 포함된 청크 출력 객체를 포함하여 모든 객체가 직접 로드됩니다. 청크를 다시 평가하는 대신 이 객체가 출력에 직접 기록됩니다.

###### A.3.5 Sweave와의 호환성

knitr는 Sweave와 몇 가지 다른 청크 옵션을 사용하므로 부적절한 옵션과 그 값을 수정하는 함수 `Sweave2knitr()`가 있습니다. 예를 들어, `results = tex`는 자동으로 `results = 'markup'`으로 변경됩니다('tex'는 `results` 옵션이 실제로 수행하는 작업을 반영하는 적절한 값이 아니기 때문입니다).

구현은 주로 정규 표현식을 기반으로 하며 간단한 예는 다음과 같습니다. 
`op <- "<<eval=TRUE, results=tex>>=" gsub("(results)\\s*=\\s*tex", "\\1='markup'", op) ## [1] "<<eval=TRUE, results='markup'>>="`

`Sweave2knitr()`는 부적절한 청크 옵션은 물론 `\SweaveOpts{}` 및 `\SweaveInput{}`의 수많은 사례를 처리합니다. 예제는 16.1절을 참조하세요.

###### A.3.6 콩코던스(Concordance)

콩코던스의 개념은 Rnw/LATEX에만 적용됩니다. 해결해야 할 문제는 TEX 출력과 Rnw 소스 간의 줄 번호 매핑입니다. LATEX에서 오류가 발생하면 문제가 발생한 줄의 줄 번호(오류 로그 분석)를 알 수 있지만, 두 문서의 줄 번호가 일치하지 않을 수 있으므로 Rnw 소스 문서의 해당 줄 번호는 알 수 없습니다. Rnw 문서의

- 5줄로 된 청크 하나가 출력에서는 LATEX 코드로 10줄 또는 3줄을 생성할 수 있습니다.


Sweave에는 knitr보다 더 나은 콩코던스 구현이 있습니다. 매핑은 Sweave에서 더 정확합니다. knitr에서는 다음과 같은 방식으로 대략적인 매핑만 수행합니다. 소스 문서를 분석할 때 코드 청크 및 텍스트 청크의 줄 수가 기록됩니다. 이러한 청크가 평가된 후 해당 출력 청크의 줄 수가 다시 계산됩니다. 한 소스 청크에 5개의 줄이 있다고 가정하면

- • 출력도 5줄인 경우 소스의 i번째 줄은 출력의 i번째 줄에 매핑됩니다.
- • 출력이 3줄인 경우 소스의 처음 3줄이 출력의 3줄에 매핑됩니다.
- • 출력이 10줄인 경우 소스의 5줄이 출력의 처음 5줄에 매핑됩니다.


분명히 이것은 좋은 근사치가 아닐 수 있지만 오류 탐색에는 충분히 도움이 될 것입니다. 적어도 LATEX의 오류 번호는 문제가 있는 소스의 대략적인 영역을 가리킬 수 있습니다.

콩코던스의 또 다른 용도는 PDF와 Rnw 파일 간의 탐색입니다. SyncTEX는 이러한 종류의 탐색을 지원합니다. PDF 문서에서 한 줄을 클릭하면 소스 파일로 돌아가거나, 소스에서 한 줄을 클릭하면 PDF로 이동할 수 있습니다. 콩코던스 정보가 없으면 Rnw와 PDF 사이를 이동할 수 없습니다(TEX↔PDF만 가능).

현재로서는 RStudio만 knitr에서 생성된 콩코던스 정보를 사용합니다. 콩코던스를 활성화하려면(기본적으로 비활성화되어 있음) 패키지 옵션을 다음과 같이 설정할 수 있습니다(RStudio는 이 작업을 자동으로 수행합니다).

`opts_knit$set(concordance = TRUE)`

콩코던스가 활성화된 경우 Rnw 파일 이름이 `input.Rnw`라면 `input-concordance.tex` 파일이 생성됩니다. 이 파일에는 압축된 매핑 정보가 포함되어 있습니다.

###### A.4 구문(Syntax)

사용자는 knitr가 여러 문서 형식(5.1절)에 대해 서로 다른 입력 구문을 사용하는 이유를 궁금해할 수 있습니다. 예를 들어, Rnw는 `<<>>=`를 사용하고 Rmd는 `{r}`을 사용합니다. 사실 구문은 문서 형식에 묶여 있지 않습니다. Rmd 문서에 대해 Rnw 구문을 사용할 수도 있습니다.

`# This is a markdown document Here is a **code chunk**: <<test>>= 1 + 1 rnorm(5) @ And an inline value \Sexpr{pi}.`

위의 예제 문서(이름이 `test.Rmd`라고 가정)의 경우 다음 명령을 통해 컴파일할 수 있습니다.

`library(knitr) pat_rnw() # input is Rnw syntax render_markdown() # output is markdown knit("test.Rmd")`

`pat_rnw()` 함수는 구문을 Rnw로 설정하고 `render_markdown()` 함수는 출력 렌더러를 Markdown 후크로 설정합니다.

하지만 왜 모든 문서에 Rnw 구문을 사용하지 않을까요? 작성 형식(authoring format)에 따라 좀 더 자연스러운 구문을 원했고, `<<>>=`는 그 어떤 문서 형식에서도 유효한 마크업이 아니기(예를 들어 LATEX 명령도 아니고 HTML 태그도 아님) 때문입니다. 사실 Sweave에는 다음과 같은 또 다른 LATEX 계열 구문 세트가 있습니다.

`\begin{Scode}{fig = TRUE, echo = FALSE} library("graphics") boxplot(Ozone ~ Month, data = airquality) \end{Scode}`

저는 청크 옵션에 `{}`보다는 `[]`를 선호하는데, 이는 LATEX에서

더 자연스러운 선택일 것입니다. 어쨌든 `<<>>=`는 널리 사용되기 때문에 knitr에 남았습니다.

(역사적인 이유로 인한) Rnw 문서를 제외하고, 다른 형식들은 R 코드가 실행되기 전에도 knitr 소스 문서를 여전히 유효한 문서로 만듭니다. 예를 들어, R HTML 문서의 R 코드는 HTML 주석(`<!-- -->`)에 포함됩니다.
###### 서지(Bibliography)

Adler, D. and Murdoch, D. (2014). rgl: 3D visualization device system (OpenGL). R package version 0.95.1201.

Allaire, J., Cheng, J., Xie, Y., McPherson, J., Chang, W., Allen, J., Wickham, H., and Hyndman, R. (2015a). rmarkdown: Dynamic Documents for R. R package version 0.5.1.

Allaire, J., Horner, J., Marti, V., and Porte, N. (2015b). markdown: Markdown Rendering for R. R package version 0.7.7.

Auguie, B. (2013). cda: Coupled dipole approximation in electromagnetic scattering. R package version 1.3.3.

Baggerly, K. A., Morris, J. S., and Coombes, K. R. (2004). Reproducibility of seldi-tof protein patterns in serum: comparing datasets from different experiments. Bioinformatics, 20(5):777–785.

Bracken, C. and Sharpsteen, C. (2012). pgfSweave: Quality speedy graphics compilation and caching with Sweave. R package version 1.3.0.

Buckheit, J. and Donoho, D. (1995). Wavelab and reproducible research. Wavelets and Statistics, 103:55.

Chang, W., Cheng, J., Allaire, J., Xie, Y., and McPherson, J. (2015). shiny: Web Application Framework for R. R package version 0.11.1.

Dahl, D. B. (2014). xtable: Export tables to LaTeX or HTML. R package version 1.7-4.

Eddelbuettel, D., Francois, R., Allaire, J., Ushey, K., Bates, D., and Chambers, J. (2015). Rcpp: Seamless R and C++ Integration. R package version 0.11.5.

Ellson, J., Gansner, E., Koutsofios, L., North, S., and Woodhull, G.

(2002). Graphviz — open source graph drawing tools. In Graph Drawing, pages 483–484. Springer-Verlag.

Falcon, S. (2013). weaver: Tools and extensions for processing Sweave documents. R package version 1.26.0.

Fomel, S. and Claerbout, J. (2009). Guest editors’ introduction: Reproducible research. Computing in Science & Engineering, 11(1):5–7.

Francois, R. (2013). highlight: Syntax highlighter. R package version 0.4.3. Friedl, J. (2006). Mastering Regular Expressions. O’Reilly Media, Incor-

porated.

Gentleman, R. (2005). Reproducible research: A bioinformatics case study. Statistical Applications in Genetics and Molecular Biology, 4(1):1034.

Gentleman, R. and Temple Lang, D. (2004). Statistical analyses and reproducible research. Bioconductor Project Working Papers. URL: http://biostats.bepress.com/bioconductor/paper2.

Gove, J. H. (2013). sampSurf: Sampling Surface Simulation for Areal Sampling Methods. R package version 0.6-8.

Gruber, J. (2004). The Markdown Project. URL: http://daringfireball.net/projects/markdown/.

Guo, J., Betancourt, M., Brubaker, M., Carpenter, B., Gao, Y., Goodrich, B., Hoffman, M., Lee, D., Li, P., Malecki, M., and Gelman, A. (2014). rstan: RStan: R interface to Stan. R package version 2.5.0.

Harrell, Jr., F. E. (2001). Regression Modeling Strategies: With Applications to Linear Models, Logistic Regression, and Survival Analysis. Springer New York.

Harrell, Jr., F. E. (2015). Hmisc: Harrell Miscellaneous. R package version 3.15-0.

Horner, J. (2011). brew: Templating Framework for Report Generation. R package version 1.0-6.

Horton, N., Aloisio, K., Zhang, R., and Loi, L. (2012). The statistical sleuth (2nd edition) in R. URL: http://www.math.smith.edu/~nhorton/sleuth/.

Huang, Y. and Gottardo, R. (2013). Comparability and reproducibility of biomedical data. Briefings in Bioinformatics, 14(4):391–401.

Ihaka, R. and Gentleman, R. (1996). R: A language for data analysis and graphics. Journal of Computational and Graphical Statistics, 5(3):299–314.

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

Oetiker, T., Partl, H., Hyna, I., and Schlegl, E. (1995). The not so short introduction to LATEX2ε. URL: http://www.ctan.org/tex-archive/info/lshort/.

Peng, R. (2009). Reproducible research and biostatistics. Biostatistics, 10(3):405–408.

Peng, R. D. (2012). cacheSweave: Tools for caching Sweave computations. R package version 0.6-1.

Qiu, Y. and Xie, Y. (2015). highr: Syntax Highlighting for R Source Code. R package version 0.5.

Qiu, Y., Xie, Y., and Bracken, C. (2015). R2SWF: Convert R Graphics to Flash Animations. R package version 0.9.

- R Core Team (2014). R Language Definition. R Foundation for Statistical Computing, Vienna, Austria.
- R Core Team (2015). R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.


Ramsey, F. and Schafer, D. (2002). The Statistical Sleuth: A Course in Methods of Data Analysis, Second Edition. Duxbury Press.

Ramsey, N. (1994). Literate programming simplified. Software, IEEE, 11(5):97–105.

Rossini, A. (2002). Literate statistical analysis. In Proceedings of the 2nd International Workshop on Distributed Statistical Computing, pages 15–17, Vienna, Austria.

Rossini, A., Heiberger, R., Sparapani, R., Maechler, M., and Hornik, K. (2004). Emacs speaks statistics: A multiplatform, multipackage development environment for statistical analysis. Journal of Computational and Graphical Statistics, 13(1):247–261.

Schulte, E., Davison, D., Dye, T., and Dominik, C. (2012). A multilanguage computing environment for literate programming and reproducible research. Journal of Statistical Software, 46(3):1–24.

Sharpsteen, C. and Bracken, C. (2015). tikzDevice: R Graphics Output in LaTeX Format. R package version 0.8.1.

Tantau, T. (2008). The TikZ and PGF Packages. URL: http://sourceforge.net/projects/pgf/.

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

통계

초보자와 숙련된 사용자 모두에게 적합한 Dynamic Documents with R and knitr, Second Edition은 계산과 보고서를 직접 통합하여 통계 보고서를 더 쉽게 작성할 수 있도록 합니다. 보고서는 숙제, 프로젝트, 시험, 책, 블로그, 웹 페이지에서 통계 그래픽, 계산, 데이터 분석과 관련된 거의 모든 문서에 이르기까지 다양합니다. 이 책은 초보자를 위한 기본 응용 프로그램을 다루는 동시에 파워 사용자가 knitr 패키지의 확장성을 이해할 수 있도록 안내합니다.

###### 제2판의 새로운 기능

- • R Markdown v2를 소개하는 새로운 장
- • knitr 패키지의 개선 사항을 반영하는 변경 사항
- • 테이블 생성, 코드 청크 내 객체의 사용자 정의 인쇄 메서드 정의, C/Fortran 엔진, Stan 엔진, 영구 세션(persistent session)에서 엔진 실행, 동적 문서를 제공하기 위한 로컬 서버 시작에 대한 새로운 섹션


높은 평가를 받은 이전 버전과 마찬가지로, 이 에디션에서는 보고서 작성 시 효율성을 높이는 방법을 보여줍니다. 이 책은 프로그램 출력에서 출판 수준의 보고서에 이르기까지 보고서의 모든 측면을 세밀하게 조정할 수 있도록 도와줍니다. 패키지에 대한 데모 및 기타 정보는 저자의 웹사이트에서 확인할 수 있습니다.

Yihui Xie는 RStudio의 소프트웨어 엔지니어입니다. Iowa State University의 통계학과에서 박사 학위를 취득했습니다. 그의 연구는 대화형 통계 그래픽 및 통계 컴퓨팅에 중점을 두고 있습니다. 그는 활발한 R 사용자이자 여러 수상 경력이 있는 R 패키지의 저자입니다. 또한 중국의 대규모 온라인 통계 커뮤니티인 "Capital of Statistics"의 설립자이기도 합니다.

K25425

w w w . c r c p r e s s . c o m

Second Edition

DynamicDocumentswithRandknitr

Xie

## TheRSeries

# Dynamic Documents with R and knitr

Second Edition

Yihui Xie

K25425_cover.indd 1 4/17/15 11:01 AM
