# 28장. Quarto

# 소개

Quarto는 코드, 그 결과, 그리고 산문(prose)을 결합하여 데이터 과학을 위한 통일된 저작 프레임워크(unified authoring framework)를 제공합니다. Quarto 문서는 완전히 재현 가능하며 PDF, Word 파일, 프레젠테이션 등과 같은 수십 가지 출력 형식을 지원합니다.

Quarto 파일은 세 가지 방식으로 사용하도록 설계되었습니다:

- 분석 이면의 코드가 아니라 결론에 초점을 맞추려는 의사 결정권자와 소통하기 위해

- 결론과 결론에 도달한 방법(즉, 코드) 모두에 관심이 있는 다른 데이터 과학자(미래의 자신 포함!)와 협력하기 위해

- 데이터 과학을 *수행*하는 환경으로서, 수행한 작업뿐만 아니라 생각했던 내용도 기록할 수 있는 현대적인 랩 노트(lab notebook)로서

Quarto는 R 패키지가 아니라 명령줄 인터페이스 도구입니다. 즉, 대체로 `?`를 통해 도움말을 이용할 수 없습니다. 대신, 이 장을 진행하고 향후 Quarto를 사용할 때 [Quarto 문서](https://oreil.ly/_6LNH)를 참조해야 합니다.

R Markdown 사용자라면 "Quarto는 R Markdown과 많이 비슷하게 들리네"라고 생각할 수 있습니다. 틀린 말이 아닙니다! Quarto는 R Markdown 생태계(rmarkdown, bookdown, distill, xaringan 등)의 많은 패키지 기능을 일관된 단일 시스템으로 통합할 뿐만 아니라 R 외에 Python 및 Julia와 같은 여러 프로그래밍 언어에 대한 기본 지원으로 기능을 확장합니다. 어떤 면에서 Quarto는 10년 동안 R Markdown 생태계를 확장하고 지원하면서 배운 모든 것을 반영합니다.

## 사전 준비

Quarto 명령줄 인터페이스(Quarto CLI)가 필요하지만, RStudio가 필요할 때 자동으로 둘 다 수행하므로 명시적으로 설치하거나 로드할 필요가 없습니다.

# Quarto 기본 (Quarto Basics)

이것은 확장자가 `.qmd`인 일반 텍스트 파일인 Quarto 파일입니다:

````
---
title: "Diamond sizes"
date: 2022-09-12
format: html
---

```{r}
#| label: setup
#| include: false

library(tidyverse)

smaller <- diamonds |> 
  filter(carat <= 2.5)
```

We have data about `r nrow(diamonds)` diamonds. Only 
`r nrow(diamonds) - nrow(smaller)` are larger than
2.5 carats. The distribution of the remainder is shown
below:

```{r}
#| label: plot-smaller-diamonds
#| echo: false

smaller |> 
  ggplot(aes(x = carat)) + 
  geom_freqpoly(binwidth = 0.01)
```
````

여기에는 세 가지 중요한 유형의 콘텐츠가 포함되어 있습니다:

- `---`로 둘러싸인 (선택적) *YAML 헤더*
- ```` ``` ````로 둘러싸인 R 코드 *청크(Chunks)*
- `# heading` 및 `_italics_`와 같은 간단한 텍스트 서식과 혼합된 텍스트

<a href="#fig-diamond-sizes-notebook" data-type="xref">그림 28-1</a>은 코드와 출력이 인터리브(interleave)된 노트북 인터페이스가 있는 RStudio의 `.qmd` 문서를 보여줍니다. 실행(Run) 아이콘(청크 상단의 재생 버튼 모양)을 클릭하거나 Cmd/Ctrl+Shift+Enter를 눌러 각 코드 청크를 실행할 수 있습니다. RStudio는 코드를 실행하고 코드와 함께(inline) 결과를 표시합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2801.png" alt="RStudio window with a Quarto document titled &quot;diamond-sizes.qmd&quot; on the left and a blank Viewer window on the right. The Quarto document has a code chunk that creates a frequency plot of diamonds that weigh less than 2.5 carats. The plot shows that the frequency decreases as the weight increases." />
<h6 id="figure-28-1.-a-quarto-document-in-rstudio.-code-and-output-are-interleaved-in-the-document-with-the-plot-output-appearing-right-underneath-the-code.">그림 28-1. RStudio의 Quarto 문서. 코드와 출력이 문서에 인터리브되어 있으며, 플롯 출력이 코드 바로 아래에 나타납니다.</h6>
</figure>

문서에서 플롯과 출력을 보는 것이 싫고 RStudio의 콘솔 및 플롯 창을 활용하고 싶다면, <a href="#fig-diamond-sizes-console-output" data-type="xref">그림 28-2</a>에 표시된 것처럼 렌더링(Render) 옆의 톱니바퀴 아이콘을 클릭하고 Chunk Output in Console로 전환할 수 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2802.png" alt="RStudio window with a Quarto document titled &quot;diamond-sizes.qmd&quot; on the left and the Plot pane on the bottom right. The Quarto document has a code chunk that creates a frequency plot of diamonds that weigh less than 2.5 carats. The plot is displayed in the Plot pane and shows that the frequency decreases as the weight increases. The RStudio option to show Chunk Output in Console is also highlighted." />
<h6 id="figure-28-2.-a-quarto-document-in-rstudio-with-the-plot-output-in-the-plots-pane.">그림 28-2. 플롯 출력이 플롯 창에 있는 RStudio의 Quarto 문서.</h6>
</figure>

모든 텍스트, 코드 및 결과를 포함하는 완전한 보고서를 생성하려면 Render를 클릭하거나 Cmd/Ctrl+Shift+K를 누릅니다. `quarto::quarto_render("diamond-sizes.qmd")`를 사용하여 프로그래밍 방식으로 이 작업을 수행할 수도 있습니다. 이렇게 하면 <a href="#fig-diamond-sizes-report" data-type="xref">그림 28-3</a>과 같이 뷰어 창에 보고서가 표시되고 HTML 파일이 생성됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2803.png" alt="RStudio window with a Quarto document titled &quot;diamond-sizes.qmd&quot; on the left and the Plot pane on the bottom right. The rendered document does not show any of the code, but the code is visible in the source document." />
<h6 id="figure-28-3.-a-quarto-document-in-rstudio-with-the-rendered-document-in-the-viewer-pane.">그림 28-3. 렌더링된 문서가 뷰어 창에 있는 RStudio의 Quarto 문서.</h6>
</figure>

문서를 렌더링할 때 Quarto는 `.qmd` 파일을 <a href="https://oreil.ly/HvFDz" class="uri">knitr</a>로 보내 모든 코드 청크를 실행하고 코드와 그 출력을 포함하는 새로운 마크다운(`.md`) 문서를 만듭니다. knitr에서 생성된 마크다운 파일은 그런 다음 최종 파일을 만드는 역할을 하는 <a href="https://oreil.ly/QxUsn" class="uri">pandoc</a>에서 처리합니다. <a href="#fig-quarto-flow" data-type="xref">그림 28-4</a>는 이 프로세스를 보여줍니다. 이 2단계 워크플로우의 장점은 <a href="ch29.html#chp-quarto-formats" data-type="xref">29장</a>에서 배우게 될 것처럼 매우 다양한 출력 형식을 만들 수 있다는 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2804.png" alt="Workflow diagram starting with a qmd file, then knitr, then md, then pandoc, then PDF, MS Word, or HTML. " />
<h6 id="figure-28-4.-diagram-of-quarto-workflow-from-qmd-to-knitr-to-md-to-pandoc-to-output-in-pdf-ms-word-or-html-formats.">그림 28-4. qmd에서 knitr, md, pandoc을 거쳐 PDF, MS Word 또는 HTML 형식으로 출력되는 Quarto 워크플로우 다이어그램.</h6>
</figure>

고유한 `.qmd` 파일을 시작하려면 메뉴 표시줄에서 File \> New File \> Quarto Document…를 선택합니다. RStudio는 Quarto의 주요 기능이 어떻게 작동하는지 상기시켜 주는 유용한 콘텐츠로 파일을 미리 채우는 데 사용할 수 있는 마법사를 시작합니다.

다음 섹션에서는 Quarto 문서의 세 가지 구성 요소인 마크다운 텍스트, 코드 청크, YAML 헤더에 대해 자세히 살펴봅니다.

## 연습문제 (Exercises)

1.  File \> New File \> Quarto Document를 선택하여 새 Quarto 문서를 만듭니다. 지침을 읽어보세요. 청크를 개별적으로 실행하는 연습을 합니다. 그런 다음 적절한 버튼을 클릭하고 적절한 키보드 단축키를 사용하여 문서를 렌더링합니다. 코드를 수정하고, 다시 실행하고, 수정된 출력을 볼 수 있는지 확인합니다.

2.  내장된 세 가지 형식인 HTML, PDF, Word에 대해 각각 하나씩 새 Quarto 문서를 만듭니다. 세 문서를 각각 렌더링합니다. 출력은 어떻게 다릅니까? 입력은 어떻게 다릅니까? (PDF 출력을 빌드하려면 LaTeX를 설치해야 할 수 있습니다. 필요한 경우 RStudio에서 메시지를 표시합니다.)

# 비주얼 에디터 (Visual Editor)

RStudio의 비주얼 에디터는 Quarto 문서를 작성하기 위한 [WYSIWYM 인터페이스](https://oreil.ly/nEiGf)를 제공합니다. 내부적으로 Quarto 문서(`.qmd` 파일)의 산문은 일반 텍스트 파일의 서식을 지정하기 위한 가벼운 규칙 집합인 마크다운(Markdown)으로 작성됩니다. 실제로 Quarto는 표, 인용, 상호 참조(cross-references), 각주, div/span, 정의 목록, 속성(attributes), 원시 HTML/TeX 등과 함께 코드 셀 실행 및 결과의 인라인 보기를 지원하는 Pandoc 마크다운(Quarto가 이해하는 마크다운의 약간 확장된 버전)을 사용합니다. 마크다운은 읽고 쓰기 쉽도록 설계되었지만, <a href="#sec-source-editor" data-type="xref">“소스 에디터”</a>에서 보게 되겠지만 여전히 새로운 구문을 배워야 합니다. 따라서 `.qmd` 파일과 같은 계산 문서(computational documents)를 처음 접하지만 Google Docs 또는 MS Word와 같은 도구를 사용한 경험이 있는 경우 RStudio에서 Quarto를 시작하는 가장 쉬운 방법은 비주얼 에디터입니다.

비주얼 에디터에서는 메뉴 모음의 버튼을 사용하여 이미지, 표, 상호 참조 등을 삽입하거나 만능 Cmd/Ctrl+/ 단축키를 사용하여 거의 모든 것을 삽입할 수 있습니다. <a href="#fig-visual-editor" data-type="xref">그림 28-5</a>에 표시된 것처럼 줄의 시작 부분에 있는 경우 /만 입력하여 단축키를 호출할 수도 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2805.png" alt="A Quarto document displaying various features of the visual editor such as text formatting (italic, bold, underline, small caps, code, superscript, and subscript), first through third level headings, bulleted and numbered lists, links, linked phrases, and images (along with a pop-up window for customizing image size, adding a caption and alt text, etc.), tables with a header row, and the insert anything tool with options to insert an R code chunk, a Python code chunk, a div, a bullet list, a numbered list, or a first level heading (the top few choices in the tool)." />
<h6 id="figure-28-5.-quarto-visual-editor.">그림 28-5. Quarto 비주얼 에디터.</h6>
</figure>

이미지를 삽입하고 표시되는 방식을 사용자 지정하는 것도 비주얼 에디터에서 원활하게 이루어집니다. 클립보드의 이미지를 비주얼 에디터에 직접 붙여넣거나(그러면 RStudio는 프로젝트 디렉터리에 해당 이미지의 복사본을 배치하고 연결합니다) 비주얼 에디터의 Insert \> Figure/Image 메뉴를 사용하여 삽입하려는 이미지를 찾아보거나 해당 URL을 붙여넣을 수 있습니다. 또한 동일한 메뉴를 사용하여 크기를 조정하고 캡션, 대체 텍스트(alternative text) 및 링크를 추가할 수 있습니다.

비주얼 에디터에는 여기에 열거하지 않은 훨씬 더 많은 기능이 있으며 이를 사용하여 저작 경험을 쌓을 때 유용할 수 있습니다.

가장 중요한 것은 비주얼 에디터가 서식이 지정된 콘텐츠를 표시하는 동안, 내부적으로는 콘텐츠를 일반 마크다운으로 저장하며, 어떤 도구를 사용하든 비주얼 에디터와 소스 에디터 간에 앞뒤로 전환하여 콘텐츠를 보고 편집할 수 있다는 것입니다.

## 연습문제 (Exercises)

1.  비주얼 에디터를 사용하여 <a href="#fig-visual-editor" data-type="xref">그림 28-5</a>의 문서를 다시 만드세요.
2.  비주얼 에디터에서 Insert 메뉴를 사용한 다음 insert anything 도구를 사용하여 코드 청크를 삽입하세요.
3.  비주얼 에디터를 사용하여 다음을 수행하는 방법을 알아내세요:
    1.  각주 추가하기.
    2.  수평선 추가하기.
    3.  인용구(block quote) 추가하기.
4.  비주얼 에디터에서 Insert \> Citation을 선택하고 디지털 객체 식별자(DOI)인 [10.21105/joss.01686](https://oreil.ly/H_Xn-)을 사용하여 ["Welcome to the Tidyverse"](https://oreil.ly/I9_I7)라는 제목의 논문에 대한 인용을 삽입합니다. 문서를 렌더링하고 문서에 참조가 어떻게 나타나는지 관찰합니다. 문서의 YAML에서 어떤 변화를 관찰할 수 있습니까?

# 소스 에디터 (Source Editor)

비주얼 에디터의 도움 없이 RStudio의 소스 에디터를 사용하여 Quarto 문서를 편집할 수도 있습니다. 비주얼 에디터는 Google Docs와 같은 도구에서 작성한 경험이 있는 사람들에게 친숙하게 느껴지는 반면, 소스 에디터는 R 스크립트 또는 R 마크다운 문서를 작성한 경험이 있는 사람들에게 친숙하게 느껴질 것입니다. 소스 에디터는 일반 텍스트에서 포착하기가 종종 더 쉽기 때문에 Quarto 구문 오류를 디버깅하는 데도 유용할 수 있습니다.

다음 가이드는 소스 에디터에서 Quarto 문서를 작성하기 위해 Pandoc의 마크다운을 사용하는 방법을 보여줍니다:

``
## Text formatting

*italic*   **bold**   ~~strikeout~~  `code`

superscript^2^   subscript~2~

[underline]{.underline}   [small caps]{.smallcaps}

## Headings

# 1st Level Header

## 2nd Level Header

### 3rd Level Header

## Lists

-   Bulleted list item 1

-   Item 2

    - Item 2a

    - Item 2b

1.  Numbered list item 1

1.  Item 2. The numbers are incremented automatically in the output.

## Links and images

<http://example.com>

[linked phrase](http://example.com)

![optional caption text](quarto.png){ fig-alt="Quarto logo and the word quarto spelled in small case letters"}

## Tables

| First Header  | Second Header |
|---------------|---------------|
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
``

이러한 사항을 배우는 가장 좋은 방법은 단순히 시도해 보는 것입니다. 며칠이 걸리겠지만 곧 제2의 천성(second nature)이 되어 생각할 필요가 없어질 것입니다. 잊어버린 경우 Help \> Markdown Quick Reference를 사용하여 편리한 참조 시트를 얻을 수 있습니다.

## 연습문제 (Exercises)

1.  짧은 이력서를 만들어 배운 내용을 연습하세요. 제목은 귀하의 이름이어야 하며 (적어도) 교육 또는 고용에 대한 제목을 포함해야 합니다. 각 섹션에는 직업/학위의 글머리 기호 목록이 포함되어야 합니다. 연도를 굵게 강조 표시합니다.

2.  소스 에디터와 마크다운 빠른 참조를 사용하여 다음을 수행하는 방법을 알아보세요:

    1.  각주 추가하기.
    2.  수평선 추가하기.
    3.  인용구(block quote) 추가하기.

3.  [`diamond-sizes.qmd`](https://oreil.ly/Auuh2)의 내용을 복사하여 로컬 R Quarto 문서에 붙여넣습니다. 실행할 수 있는지 확인한 다음, 가장 눈에 띄는 특징을 설명하는 텍스트를 빈도 다각형(frequency polygon) 뒤에 추가합니다.

4.  제목, 하이퍼링크, 서식 있는 텍스트 등과 같은 일부 콘텐츠가 포함된 문서를 Google Docs 또는 MS Word로 만듭니다(또는 이전에 만든 문서를 찾습니다). 이 문서의 내용을 복사하여 비주얼 에디터의 Quarto 문서에 붙여넣습니다. 그런 다음 소스 에디터로 전환하여 소스 코드를 검사합니다.

# 코드 청크 (Code Chunks)

Quarto 문서 내에서 코드를 실행하려면 청크를 삽입해야 합니다. 이를 수행하는 세 가지 방법이 있습니다:

- 키보드 단축키 Cmd+Option+I / Ctrl+Alt+I 누르기

- 에디터 도구 모음에서 삽입(insert) 버튼 아이콘 클릭하기

- 청크 구분 기호(delimiters) ```` ```{r} ```` 및 ```` ``` ```` 수동으로 입력하기

키보드 단축키를 배우는 것이 좋습니다. 장기적으로 많은 시간을 절약할 수 있습니다!

(우리가 희망하건대!) 이제는 여러분이 알고 사랑하는 키보드 단축키인 Cmd/Ctrl+Enter를 사용하여 코드를 계속 실행할 수 있습니다. 그러나 청크에는 청크의 모든 코드를 실행하는 새로운 키보드 단축키인 Cmd/Ctrl+Shift+Enter가 있습니다. 청크를 함수처럼 생각하십시오. 청크는 비교적 자급자족(self-contained)해야 하며 단일 작업에 집중해야 합니다.

다음 섹션에서는 ```` ```{r} ````과 그 뒤에 선택적 청크 레이블 및 기타 다양한 청크 옵션으로 구성된 청크 헤더(각각 `#|`로 표시된 별도의 줄에 있음)에 대해 설명합니다.

## 청크 레이블 (Chunk Label)

청크에 선택적 레이블을 지정할 수 있습니다:

```r
#| label: simple-addition
1 + 1
```
`#> [1] 2`

이것은 세 가지 장점이 있습니다:

- 스크립트 에디터의 왼쪽 하단에 있는 드롭다운 코드 내비게이터를 사용하여 특정 청크로 더 쉽게 이동할 수 있습니다.

  <figure>
  <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_28in01.png" alt="Snippet of RStudio IDE showing only the drop-down code navigator which shows three chunks. Chunk 1 is setup. Chunk 2 is cars and it is in a section called Quarto. Chunk 3 is pressure and it is in a section called Including plots." />
  </figure>

- 청크에 의해 생성된 그래픽은 다른 곳에서 사용하기 더 쉽게 만드는 유용한 이름을 갖게 됩니다. 자세한 내용은 <a href="#sec-figures" data-type="xref">“그림(Figures)”</a>을 참조하세요.

- 매번 실행될 때마다 값비싼 계산을 다시 수행하지 않도록 캐시된 청크 네트워크를 설정할 수 있습니다. 자세한 내용은 <a href="#sec-caching" data-type="xref">“캐싱(Caching)”</a>을 참조하세요.

청크 레이블은 짧지만 연상되어야(evocative) 하며 공백을 포함해서는 안 됩니다. 청크 레이블에서는 밑줄(`_`) 대신 대시(`-`)를 사용하여 단어를 분리하고 다른 특수 문자를 피하는 것이 좋습니다.

일반적으로 원하는 대로 청크 레이블을 지정할 수 있지만 특별한 동작을 부여하는 하나의 청크 이름이 있습니다: 바로 `setup`입니다. 노트북 모드에 있을 때 `setup`이라는 이름의 청크는 다른 코드가 실행되기 전에 자동으로 한 번 실행됩니다.

또한 청크 레이블은 중복될 수 없습니다. 각 청크 레이블은 고유해야 합니다.

## 청크 옵션 (Chunk Options)

청크 출력은 청크 헤더에 제공되는 필드인 *옵션(options)*으로 사용자 정의할 수 있습니다. Knitr는 코드 청크를 사용자 정의하는 데 사용할 수 있는 거의 60개의 옵션을 제공합니다. 여기서는 자주 사용하게 될 가장 중요한 청크 옵션을 다룰 것입니다. 전체 목록은 <a href="https://oreil.ly/38bld" class="uri">여기</a>에서 볼 수 있습니다.

가장 중요한 옵션 세트는 코드 블록이 실행되는지 여부와 완료된 보고서에 삽입될 결과를 제어합니다:

`eval: false`  
코드가 평가되는 것을 방지합니다. (코드가 실행되지 않으면 결과가 생성되지 않는 것은 당연합니다.) 예제 코드를 표시하거나 각 줄을 주석 처리하지 않고 큰 코드 블록을 비활성화하는 데 유용합니다.

`include: false`  
코드를 실행하지만 최종 문서에 코드나 결과를 표시하지 않습니다. 보고서를 어지럽히지 않으려는 설정(setup) 코드에 사용합니다.

`echo: false`  
코드가 완성된 파일에 나타나는 것을 방지하지만 결과는 나타나게 합니다. 기본 R 코드를 보고 싶어하지 않는 사람들을 대상으로 한 보고서를 작성할 때 사용합니다.

`message: false` 또는 `warning: false`  
메시지 또는 경고가 완성된 파일에 나타나는 것을 방지합니다.

`results: hide`  
인쇄된 출력을 숨깁니다.

`fig-show: hide`  
플롯을 숨깁니다.

`error: true`  
코드가 오류를 반환하더라도 렌더링이 계속되도록 합니다. 이것은 보고서의 최종 버전에 포함하고자 하는 경우는 드물지만 `.qmd` 내부에서 정확히 무슨 일이 일어나고 있는지 디버그해야 하는 경우 유용할 수 있습니다. 또한 R을 가르치고 있고 고의로 오류를 포함하고 싶은 경우에도 유용합니다. 기본값인 `error: false`는 문서에 단일 오류라도 있으면 렌더링이 실패하도록 합니다.

이러한 각 청크 옵션은 `#|`에 이어 청크 헤더에 추가됩니다. 예를 들어, 다음 청크에서는 `eval`이 false로 설정되어 있으므로 결과가 인쇄되지 않습니다:

```r
#| label: simple-multiplication
#| eval: false
2 * 2
```

다음 표는 각 옵션이 어떤 유형의 출력을 억제하는지 요약합니다:

| Option           | Run Code | Show Code | Output | Plots | Messages | Warnings |
|------------------|:--------:|:---------:|:------:|:-----:|:--------:|:--------:|
| `eval: false`    |    X     |           |   X    |   X   |    X     |    X     |
| `include: false` |          |     X     |   X    |   X   |    X     |    X     |
| `echo: false`    |          |     X     |        |       |          |          |
| `results: hide`  |          |           |   X    |       |          |          |
| `fig-show: hide` |          |           |        |   X   |          |          |
| `message: false` |          |           |        |       |    X     |          |
| `warning: false` |          |           |        |       |          |    X     |

## 전역 옵션 (Global Options)

knitr를 더 많이 사용하다 보면 일부 기본 청크 옵션이 요구 사항에 맞지 않아 변경하고 싶어지는 경우가 있을 것입니다.

문서 YAML의 `execute` 아래에 원하는 옵션을 추가하여 이를 수행할 수 있습니다. 예를 들어, 코드는 볼 필요 없이 결과와 설명만 보면 되는 대상을 위한 보고서를 준비하는 경우 문서 수준에서 `echo: false`를 설정할 수 있습니다. 그렇게 하면 기본적으로 코드가 숨겨지고 ( `echo: true`를 사용하여) 의도적으로 표시하도록 선택한 청크만 표시됩니다. `message: false` 및 `warning: false` 설정을 고려할 수도 있지만, 최종 문서에 어떤 메시지도 표시되지 않기 때문에 문제를 디버그하기가 더 어려워집니다.

```yaml
title: "My report"
execute: 
  echo: false
```

Quarto는 다국어(multilingual)로 설계되었으므로(R뿐만 아니라 Python, Julia 등과 같은 다른 언어에서도 작동함) 일부 knitr 옵션은 문서 실행 수준에서 사용할 수 없습니다. 일부 옵션은 knitr에서만 작동하고 Quarto가 다른 언어(예: Jupyter)로 코드를 실행하는 데 사용하는 다른 엔진에서는 작동하지 않기 때문입니다. 그러나 `opts_chunk` 아래의 `knitr` 필드에서 이를 문서에 대한 전역 옵션으로 계속 설정할 수 있습니다. 예를 들어 책이나 튜토리얼을 작성할 때 다음과 같이 설정합니다:

```yaml
title: "Tutorial"
knitr: 
  opts_chunk: 
    comment: "#>"
    collapse: true
```

이것은 선호하는 주석(comment) 형식을 사용하고 코드와 출력이 밀접하게 얽혀 있도록 보장합니다.

## 인라인 코드 (Inline Code)

Quarto 문서에 R 코드를 삽입하는 또 다른 방법이 있습니다. `` `r ` ``를 사용하여 텍스트에 직접 삽입하는 것입니다. 이는 텍스트에서 데이터의 속성을 언급하는 경우 유용할 수 있습니다. 예를 들어, 장 시작 부분에서 사용된 예제 문서는 다음과 같습니다:

> We have data about `` `r nrow(diamonds)` `` diamonds. Only `` `r nrow(diamonds) - nrow(smaller)` `` are larger than 2.5 carats. The distribution of the remainder is shown below:

보고서가 렌더링될 때 이러한 계산 결과가 텍스트에 삽입됩니다:

> We have data about 53940 diamonds. Only 126 are larger than 2.5 carats. The distribution of the remainder is shown below:

텍스트에 숫자를 삽입할 때 <a href="https://rdrr.io/r/base/format.html" class="orm:hideurl"><code>format()</code></a>은 훌륭한 친구입니다. 숫자를 우스꽝스러울 정도로 정확하게 인쇄하지 않도록 `digits`의 수를 설정할 수 있으며, `big.mark`를 사용하여 숫자를 읽기 쉽게 만들 수 있습니다. 이러한 것들을 헬퍼 함수로 결합할 수 있습니다:

```r
comma <- function(x) format(x, digits = 2, big.mark = ",")
comma(3452345)
#> [1] "3,452,345"
comma(.12358124331)
#> [1] "0.12"
```

## 연습문제 (Exercises)

1.  컷, 색상 및 투명도에 따라 다이아몬드 크기가 어떻게 변하는지 탐색하는 섹션을 추가합니다. R을 모르는 사람을 위해 보고서를 작성한다고 가정하고, 각 청크에 `echo: false`를 설정하는 대신 전역 옵션을 설정합니다.

2.  [`diamond-sizes.qmd`](https://oreil.ly/Auuh2)를 다운로드합니다. 가장 중요한 속성을 표시하는 표를 포함하여 가장 큰 다이아몬드 20개를 설명하는 섹션을 추가합니다.

3.  보기 좋게 서식이 지정된 출력을 생성하도록 `label_comma()`를 사용하도록 `diamonds-sizes.qmd`를 수정합니다. 또한 2.5캐럿보다 큰 다이아몬드의 비율도 포함합니다.

# 그림 (Figures)

Quarto 문서의 그림은 포함되거나(예: PNG 또는 JPEG 파일) 코드 청크의 결과로 생성될 수 있습니다.

외부 파일의 이미지를 포함하려면 비주얼 에디터 RStudio의 Insert 메뉴를 사용하여 Figure/Image를 선택할 수 있습니다. 그러면 삽입하려는 이미지를 찾아볼 수 있을 뿐만 아니라 대체 텍스트(alternative text)나 캡션을 추가하고 크기를 조정할 수 있는 메뉴가 팝업으로 나타납니다. 비주얼 에디터에서는 클립보드의 이미지를 문서에 간단히 붙여넣을 수도 있으며 RStudio는 해당 이미지의 복사본을 프로젝트 폴더에 저장합니다.
그림을 생성하는 코드 청크(예: `ggplot()` 호출 포함)를 포함하면 결과 그림이 Quarto 문서에 자동으로 포함됩니다.

## 그림 크기 조정 (Figure Sizing)

Quarto에서 그래픽의 가장 큰 과제는 그림의 올바른 크기와 모양을 얻는 것입니다. 그림 크기를 제어하는 다섯 가지 주요 옵션이 있습니다: `fig-width`, `fig-height`, `fig-asp`, `out-width` 및 `out-height`. 이미지 크기 조정이 까다로운 이유는 크기가 두 가지(R에 의해 생성된 그림의 크기와 출력 문서에 삽입될 때의 크기)이고 크기를 지정하는 방법이 여러 가지(즉, 높이, 너비 및 가로 세로 비율: 세 개 중 두 개 선택)이기 때문입니다.

다섯 가지 옵션 중 세 가지를 권장합니다:

- 플롯은 너비가 일관될 때 시각적으로 더 만족스러운 경향이 있습니다. 이를 적용하려면 기본값으로 `fig-width: 6` (6인치) 및 `fig-asp: 0.618` (황금비)을 설정합니다. 그런 다음 개별 청크에서 `fig-asp`만 조정합니다.

- `out-width`를 사용하여 출력 크기를 제어하고 출력 문서 본문 너비의 백분율로 설정합니다. `out-width: "70%"`와 `fig-align: center`를 권장합니다. 그러면 너무 많은 공간을 차지하지 않으면서 플롯이 숨쉴 수 있는 공간이 생깁니다.

- 한 행에 여러 개의 플롯을 배치하려면 플롯이 두 개인 경우 `layout-ncol`을 2로 설정하고, 세 개인 경우 3으로 설정합니다. 설명하려는 내용(예: 데이터 표시 또는 플롯 변형 표시)에 따라 다음에서 설명하는 대로 `fig-width`를 수정할 수도 있습니다.

플롯의 텍스트를 읽기 위해 눈을 가늘게 뜨고 봐야 한다면 `fig-width`를 미세 조정해야 합니다. `fig-width`가 그림이 최종 문서에 렌더링되는 크기보다 크면 텍스트가 너무 작아집니다. `fig-width`가 더 작으면 텍스트가 너무 큽니다. 종종 `fig-width`와 문서의 최종 너비 사이의 올바른 비율을 파악하기 위해 약간의 실험을 해야 합니다. 원리를 설명하기 위해 다음 세 가지 플롯은 각각 `fig-width`가 4, 6, 8입니다:

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_28in02.png" alt="Scatterplot of highway mileage vs. displacement of cars, where the points are normally sized and the axis text and labels are in similar font size to the surrounding text." />
</figure>

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_28in03.png" alt="Scatterplot of highway mileage vs. displacement of cars, where the points are smaller than in the previous plot and the axis text and labels are smaller than the surrounding text." />
</figure>

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_28in04.png" alt="Scatterplot of highway mileage vs. displacement of cars, where the points are even smaller than in the previous plot and the axis text and labels are even smaller than the surrounding text." />
</figure>

모든 그림에서 글꼴 크기가 일관되게 유지되도록 하려면 `out-width`를 설정할 때마다 기본 `out-width`와 동일한 비율을 유지하도록 `fig-width`도 조정해야 합니다. 예를 들어, 기본 `fig-width`가 6이고 `out-width`가 "70%"인 경우 `out-width: "50%"`로 설정할 때 `fig-width`를 4.3(6 * 0.5 / 0.7)으로 설정해야 합니다.

그림 크기 및 비율 조정(scaling)은 예술이자 과학이며, 제대로 하려면 반복적인 시행착오 접근 방식이 필요할 수 있습니다. 그림 크기 조정에 대한 자세한 내용은 ["플롯 스케일링 제어하기(Taking Control of Plot Scaling)" 블로그 게시물](https://oreil.ly/EfKFq)에서 알아볼 수 있습니다.

## 기타 중요한 옵션 (Other Important Options)

이 책에서와 같이 코드와 텍스트를 혼합할 때 코드 뒤에 플롯이 표시되도록 `fig-show: hold`를 설정할 수 있습니다. 이는 큰 코드 블록을 설명과 함께 나누도록 강제하는 즐거운 부작용이 있습니다.

플롯에 캡션을 추가하려면 `fig-cap`을 사용합니다. Quarto에서는 이렇게 하면 그림이 인라인에서 "떠다니는(floating)" 것으로 바뀝니다.

PDF 출력을 생성하는 경우 기본 그래픽 유형은 PDF입니다. PDF는 고품질 벡터 그래픽이기 때문에 좋은 기본값입니다. 그러나 수천 개의 점을 표시하는 경우 크고 느린 플롯이 생성될 수 있습니다. 이 경우 PNG를 강제로 사용하도록 `fig-format: "png"`를 설정합니다. 품질은 약간 떨어지지만 훨씬 더 작을(compact) 것입니다.

일반적으로 다른 청크에 레이블을 지정하지 않더라도 그림을 생성하는 코드 청크의 이름을 지정하는 것이 좋습니다. 청크 레이블은 디스크에 저장된 그래픽의 파일 이름을 생성하는 데 사용되므로 청크의 이름을 지정하면 다른 상황에서 플롯을 훨씬 더 쉽게 선택하고 재사용할 수 있습니다(예: 단일 플롯을 이메일에 빠르게 추가하려는 경우).

## 연습문제 (Exercises)

1.  비주얼 에디터에서 `diamond-sizes.qmd`를 열고, 다이아몬드 이미지를 찾아 복사한 후 문서에 붙여넣습니다. 이미지를 두 번 클릭하고 캡션을 추가합니다. 이미지 크기를 조정하고 문서를 렌더링합니다. 현재 작업 디렉터리에 이미지가 어떻게 저장되는지 관찰합니다.
2.  플롯을 생성하는 `diamond-sizes.qmd`의 코드 청크 레이블을 편집하여 `fig-` 접두사로 시작하게 하고 청크 옵션 `fig-cap`을 사용하여 그림에 캡션을 추가합니다. 그런 다음 코드 청크 위의 텍스트를 편집하여 Insert \> Cross Reference를 사용하여 그림에 대한 상호 참조를 추가합니다.
3.  다음 청크 옵션을 한 번에 하나씩 사용하여 그림의 크기를 변경하고, 문서를 렌더링한 다음, 그림이 어떻게 변하는지 설명합니다.
    1.  `fig-width: 10`

    2.  `fig-height: 3`

    3.  `out-width: "100%"`

    4.  `out-width: "20%"`

# 표 (Tables)

그림과 마찬가지로 Quarto 문서에 두 가지 유형의 표를 포함할 수 있습니다. Quarto 문서에서 (Insert Table 메뉴를 사용하여) 직접 생성한 마크다운 표이거나 코드 청크의 결과로 생성된 표일 수 있습니다. 이 섹션에서는 계산을 통해 생성된 후자에 초점을 맞출 것입니다.

기본적으로 Quarto는 콘솔에서 볼 수 있는 것처럼 데이터 프레임과 행렬을 인쇄합니다:

```r
mtcars[1:5, ]
#>                    mpg cyl disp  hp drat    wt  qsec vs am gear carb
#> Mazda RX4         21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
#> Mazda RX4 Wag     21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
#> Datsun 710        22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
#> Hornet 4 Drive    21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
#> Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
```

데이터가 추가 서식과 함께 표시되는 것을 선호하는 경우 <a href="https://rdrr.io/pkg/knitr/man/kable.html" class="orm:hideurl"><code>knitr::kable()</code></a> 함수를 사용할 수 있습니다. 다음 코드는 <a href="#tbl-kable" data-type="xref">표 28-1</a>을 생성합니다:

```r
knitr::kable(mtcars[1:5, ])
```

|                   |  mpg | cyl | disp |  hp | drat |    wt |  qsec |  vs |  am | gear | carb |
|:------------------|-----:|----:|-----:|----:|-----:|------:|------:|----:|----:|-----:|-----:|
| Mazda RX4         | 21.0 |   6 |  160 | 110 | 3.90 | 2.620 | 16.46 |   0 |   1 |    4 |    4 |
| Mazda RX4 Wag     | 21.0 |   6 |  160 | 110 | 3.90 | 2.875 | 17.02 |   0 |   1 |    4 |    4 |
| Datsun 710        | 22.8 |   4 |  108 |  93 | 3.85 | 2.320 | 18.61 |   1 |   1 |    4 |    1 |
| Hornet 4 Drive    | 21.4 |   6 |  258 | 110 | 3.08 | 3.215 | 19.44 |   1 |   0 |    3 |    1 |
| Hornet Sportabout | 18.7 |   8 |  360 | 175 | 3.15 | 3.440 | 17.02 |   0 |   0 |    3 |    2 |

표 28-1. knitr kable {#tbl-kable .table .table-sm .table-striped}

표를 사용자 지정할 수 있는 다른 방법을 보려면 <a href="https://rdrr.io/pkg/knitr/man/kable.html" class="orm:hideurl"><code>?knitr::kable</code></a> 문서를 읽어보세요. 훨씬 더 깊은 사용자 지정을 위해 gt, huxtable, reactable, kableExtra, xtable, stargazer, pander, tables 및 ascii 패키지를 고려해 보세요. 각각은 R 코드에서 형식이 지정된 표를 반환하기 위한 도구 세트를 제공합니다.

## 연습문제 (Exercises)

1.  비주얼 에디터에서 `diamond-sizes.qmd`를 열고 코드 청크를 삽입한 다음 `diamonds` 데이터 프레임의 처음 5개 행을 표시하는 <a href="https://rdrr.io/pkg/knitr/man/kable.html" class="orm:hideurl"><code>knitr::kable()</code></a>이 포함된 표를 추가합니다.
2.  대신 <a href="https://gt.rstudio.com/reference/gt.html" class="orm:hideurl"><code>gt::gt()</code></a>를 사용하여 동일한 표를 표시합니다.
3.  접두사 `tbl-`로 시작하는 청크 레이블을 추가하고 청크 옵션 `tbl-cap`을 사용하여 표에 캡션을 추가합니다. 그런 다음 코드 청크 위의 텍스트를 편집하여 Insert \> Cross Reference를 사용하여 표에 대한 상호 참조를 추가합니다.

# 캐싱 (Caching)

일반적으로 문서의 렌더링은 매번 완전히 깨끗한 상태(clean slate)에서 시작됩니다. 이는 코드의 모든 중요한 계산을 포착했는지 확인해주기 때문에 재현성에 아주 좋습니다. 그러나 시간이 오래 걸리는 계산이 있는 경우 고통스러울 수 있습니다. 해결책은 `cache: true`입니다.

표준 YAML 옵션을 사용하여 문서 내의 모든 계산 결과를 캐시하기 위해 문서 수준에서 knitr 캐시를 활성화할 수 있습니다:

```yaml
---
title: "My Document"
execute: 
  cache: true
---
```

특정 청크에서 계산 결과를 캐싱하기 위해 청크 수준에서 캐싱을 활성화할 수도 있습니다:

```r
#| cache: true
# code for lengthy computation...
```

설정된 경우 이렇게 하면 청크의 출력이 디스크의 특수 이름이 지정된 파일에 저장됩니다. 후속 실행 시 knitr는 코드가 변경되었는지 확인하고 변경되지 않았으면 캐시된 결과를 재사용합니다.

캐싱 시스템은 주의해서 사용해야 합니다. 왜냐하면 기본적으로 종속성이 아니라 코드에만 기반하기 때문입니다. 예를 들어 여기에서 `processed_data` 청크는 `raw-data` 청크에 종속됩니다:

````
```{r}
#| label: raw-data
#| cache: true
rawdata <- readr::read_csv("a_very_large_file.csv")
```

```{r}
#| label: processed_data
#| cache: true
processed_data <- rawdata |> 
  filter(!is.na(import_var)) |> 
  mutate(new_variable = complicated_transformation(x, y, z))
```
````

`processed_data` 청크를 캐싱하면 dplyr 파이프라인이 변경될 경우 다시 실행되지만, `read_csv()` 호출이 변경되면 다시 실행되지 않는다는 의미입니다. `dependson` 청크 옵션을 사용하여 해당 문제를 방지할 수 있습니다:

````
```{r}
#| label: processed-data
#| cache: true
#| dependson: "raw-data"
processed_data <- rawdata |> 
  filter(!is.na(import_var)) |> 
  mutate(new_variable = complicated_transformation(x, y, z))
```
````

`dependson`은 캐시된 청크가 의존하는 *모든* 청크의 문자형 벡터를 포함해야 합니다. Knitr는 종속성 중 하나가 변경되었음을 감지할 때마다 캐시된 청크에 대한 결과를 업데이트합니다.

knitr 캐싱은 `.qmd` 파일 내의 변경 사항만 추적하기 때문에 `a_very_large_file.csv`가 변경되어도 청크는 업데이트되지 않습니다. 해당 파일의 변경 사항도 추적하려면 `cache.extra` 옵션을 사용할 수 있습니다. 이것은 변경될 때마다 캐시를 무효화하는 임의의 R 표현식입니다. 사용하기 좋은 함수는 <a href="https://rdrr.io/r/base/file.info.html" class="orm:hideurl"><code>file.mtime()</code></a>입니다. 이 함수는 마지막으로 수정된 시간을 반환합니다. 그런 다음 다음과 같이 작성할 수 있습니다:

````
```{r}
#| label: raw-data
#| cache: true
#| cache.extra: !expr file.mtime("a_very_large_file.csv")
rawdata <- readr::read_csv("a_very_large_file.csv")
```
````

이러한 청크의 이름을 지정하라는 [David Robinson의 조언](https://oreil.ly/yvPFt)을 따랐습니다. 각 청크의 이름은 생성되는 기본(primary) 객체의 이름을 따서 지정됩니다. 이렇게 하면 `dependson` 사양을 더 쉽게 이해할 수 있습니다.

캐싱 전략이 점차 복잡해짐에 따라 <a href="https://rdrr.io/pkg/knitr/man/clean_cache.html" class="orm:hideurl"><code>knitr::clean_cache()</code></a>를 사용하여 정기적으로 모든 캐시를 지우는 것이 좋습니다.

## 연습문제 (Exercises)

1.  `d`는 `c`와 `b`에 종속되고 `b`와 `c`는 모두 `a`에 종속되는 청크 네트워크를 설정합니다. 각 청크가 <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>lubridate::now()</code></a>를 인쇄하도록 하고 `cache: true`를 설정한 다음 캐싱에 대한 이해를 확인합니다.

# 문제 해결 (Troubleshooting)

Quarto 문서 문제 해결은 더 이상 대화형 R 환경에 있지 않고 몇 가지 새로운 기술을 배워야 하기 때문에 어려울 수 있습니다. 또한 오류는 Quarto 문서 자체의 문제로 인한 것이거나 Quarto 문서의 R 코드로 인한 것일 수 있습니다.

코드 청크가 있는 문서에서 흔히 발생하는 오류 중 하나는 중복된 청크 레이블로, 워크플로우에 코드 청크 복사 및 붙여넣기가 포함된 경우 특히 만연합니다. 이 문제를 해결하려면 중복된 레이블 중 하나를 변경하기만 하면 됩니다.

오류가 문서의 R 코드로 인한 경우 가장 먼저 시도해야 할 것은 대화형 세션에서 문제를 재현하는 것입니다. R을 다시 시작한 다음 Code 메뉴, Run 영역 아래에서 또는 키보드 단축키 Ctrl+Alt+R을 눌러 "Run all chunks"를 선택합니다. 운이 좋으면 문제가 재현되고 대화형으로 진행 중인 상황을 파악할 수 있습니다.

그것이 도움이 되지 않는다면 대화형 환경과 Quarto 환경 사이에 뭔가 다른 것이 있어야 합니다. 옵션을 체계적으로 탐색해야 합니다. 가장 흔한 차이점은 작업 디렉터리입니다. Quarto의 작업 디렉터리는 그것이 위치한 디렉터리입니다. 청크에 <a href="https://rdrr.io/r/base/getwd.html" class="orm:hideurl"><code>getwd()</code></a>를 포함하여 작업 디렉터리가 예상한 대로인지 확인합니다.

다음으로 버그를 유발할 수 있는 모든 사항을 브레인스토밍합니다. R 세션과 Quarto 세션에서 동일한지 체계적으로 확인해야 합니다. 가장 쉬운 방법은 문제를 일으키는 청크에 `error: true`를 설정한 다음 <a href="https://rdrr.io/r/base/print.html" class="orm:hideurl"><code>print()</code></a> 및 <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>을 사용하여 설정이 예상대로인지 확인하는 것입니다.

# YAML 헤더 (YAML Header)

YAML 헤더의 매개변수(parameters)를 조정하여 다른 많은 "전체 문서(whole document)" 설정을 제어할 수 있습니다. YAML이 무엇을 의미하는지 궁금할 수 있습니다. 이는 "YAML Ain't Markup Language(YAML은 마크업 언어가 아니다)"로 인간이 읽고 쓰기 쉬운 방식으로 계층적 데이터를 표현하도록 설계되었습니다. Quarto는 이를 사용하여 출력의 많은 세부 사항을 제어합니다. 여기서는 독립 실행형(self-contained) 문서, 문서 매개변수, 참고 문헌(bibliographies)의 세 가지를 논의합니다.

## 독립 실행형 (Self-Contained)

HTML 문서는 일반적으로 여러 외부 종속성(예: 이미지, CSS 스타일시트, JavaScript 등)을 가지며 기본적으로 Quarto는 이러한 종속성을 `.qmd` 파일과 동일한 디렉터리의 `_files` 폴더에 배치합니다. 호스팅 플랫폼(예: [QuartoPub](https://oreil.ly/SF3Pm))에 HTML 파일을 게시하는 경우 이 디렉터리의 종속성이 문서와 함께 게시되므로 게시된 보고서에서 사용할 수 있습니다. 그러나 보고서를 동료에게 이메일로 보내려는 경우 모든 종속성이 포함된 단일 독립 실행형 HTML 문서를 선호할 수 있습니다. `embed-resources` 옵션을 지정하여 이를 수행할 수 있습니다.

```yaml
format: 
  html:
    embed-resources: true
```

결과 파일은 독립 실행형이 되며 브라우저에서 제대로 표시하기 위해 외부 파일이나 인터넷 액세스가 필요하지 않습니다.

## 매개변수 (Parameters)

Quarto 문서에는 보고서를 렌더링할 때 값을 설정할 수 있는 하나 이상의 매개변수(parameters)가 포함될 수 있습니다. 매개변수는 다양한 주요 입력에 대해 서로 다른 값으로 동일한 보고서를 다시 렌더링하려는 경우에 유용합니다. 예를 들어 지점별 판매 보고서, 학생별 시험 결과 또는 국가별 인구 통계 요약을 생성할 수 있습니다. 하나 이상의 매개변수를 선언하려면 `params` 필드를 사용합니다.

이 예제는 `my_class` 매개변수를 사용하여 표시할 자동차 등급을 결정합니다:

````
---
format: html
params:
  my_class: "suv"
---

```{r}
#| label: setup
#| include: false
library(tidyverse)
class <- mpg |> filter(class == params$my_class)
```

# Fuel economy for `r params$my_class`s

```{r}
#| message: false
ggplot(class, aes(x = displ, y = hwy)) + 
  geom_point() + 
  geom_smooth(se = FALSE)
```
````

보시다시피 매개변수는 코드 청크 내에서 `params`라는 읽기 전용 리스트로 사용할 수 있습니다.

원자형 벡터를 YAML 헤더에 직접 작성할 수 있습니다. 매개변수 값 앞에 `!expr`을 붙여서 임의의 R 표현식을 실행할 수도 있습니다. 이것은 날짜/시간 매개변수를 지정하는 좋은 방법입니다.

```yaml
params:
  start: !expr lubridate::ymd("2015-01-01")
  snapshot: !expr lubridate::ymd_hms("2015-01-01 12:30:00")
```

## 참고 문헌 및 인용 (Bibliographies and Citations)

Quarto는 여러 가지 스타일로 인용(citations)과 참고 문헌(bibliography)을 자동으로 생성할 수 있습니다. Quarto 문서에 인용 및 참고 문헌을 추가하는 가장 간단한 방법은 RStudio의 비주얼 에디터를 사용하는 것입니다.

비주얼 에디터를 사용하여 인용을 추가하려면 Insert \> Citation을 선택합니다. 인용은 다양한 소스에서 삽입할 수 있습니다:

- [DOI](https://oreil.ly/sxxlC) 참조

- [Zotero](https://oreil.ly/BDpHv) 개인 또는 그룹 라이브러리

- [Crossref](https://oreil.ly/BpPdW), [DataCite](https://oreil.ly/vSwdK) 또는 [PubMed](https://oreil.ly/Hd2Ey) 검색

- 문서 참고 문헌(문서 디렉터리의 `.bib` 파일)

내부적으로 비주얼 모드는 인용을 위해 표준 Pandoc 마크다운 표현(예: `[@citation]`)을 사용합니다.

처음 세 가지 방법 중 하나를 사용하여 인용을 추가하면 비주얼 에디터가 자동으로 `bibliography.bib` 파일을 만들고 여기에 참조를 추가합니다. 또한 문서 YAML에 `bibliography` 필드를 추가합니다. 참조를 더 추가하면 이 파일이 인용으로 채워집니다. BibLaTeX, BibTeX, EndNote 및 Medline을 포함한 여러 일반적인 참고 문헌 형식을 사용하여 이 파일을 직접 편집할 수도 있습니다.

소스 에디터의 `.qmd` 파일 내에서 인용을 만들려면 @ 와 참고 문헌 파일의 인용 식별자(identifier)로 구성된 키를 사용합니다. 그런 다음 인용을 대괄호 안에 넣습니다. 다음은 몇 가지 예입니다:

여러 인용을 `` `;` ``로 구분합니다: 어쩌구 저쩌구 \[@smith04; `@doe99`\]. 대괄호 안에 임의의 주석을 추가할 수 있습니다: 어쩌구 저쩌구 \[see `@doe99`, pp. 33-35; also `@smith04`, ch. 1\]. 텍스트 내 인용(in-text citation)을 만들려면 대괄호를 제거합니다: `@smith04`는 어쩌구라고 말하거나 `@smith04` \[p. 33\]은 어쩌구라고 말합니다. 저자의 이름을 표시하지 않으려면 인용 앞에 `` `-` ``를 추가합니다: Smith는 어쩌구라고 말합니다 \[-@smith04\].

Quarto가 파일을 렌더링하면 참고 문헌을 작성하여 문서 끝에 추가합니다. 참고 문헌에는 참고 문헌 파일의 인용된 각 참조가 포함되지만 섹션 제목은 포함되지 않습니다. 결과적으로 `# References` 또는 `# Bibliography`와 같이 참고 문헌에 대한 섹션 헤더로 파일을 끝내는 것이 일반적인 관행입니다.

`csl` 필드에서 인용 스타일 언어(CSL, citation style language) 파일을 참조하여 인용 및 참고 문헌 스타일을 변경할 수 있습니다:

```yaml
bibliography: rmarkdown.bib
csl: apa.csl
```

참고 문헌 필드와 마찬가지로 CSL 파일에는 파일 경로가 포함되어야 합니다. 여기서는 CSL 파일이 `.qmd` 파일과 동일한 디렉터리에 있다고 가정합니다. 일반적인 참고 문헌 스타일에 대한 CSL 스타일 파일을 찾기에 좋은 곳은 [인용 스타일에 대한 공식 저장소](https://oreil.ly/bYJez)입니다.

# 워크플로우 (Workflow)

앞서 *콘솔(console)*에서 대화형으로 작업한 다음 작동하는 것을 *스크립트 에디터(script editor)*에서 캡처하는 R 코드를 캡처하기 위한 기본 워크플로우를 논의했습니다. Quarto는 콘솔과 스크립트 에디터를 하나로 모아 대화형 탐색과 장기적인 코드 캡처 간의 경계를 모호하게 합니다. Cmd/Ctrl+Shift+Enter를 사용하여 청크 내에서 빠르게 반복하고 편집하고 다시 실행할 수 있습니다. 만족스러우면 이동하여 새 청크를 시작합니다.

Quarto가 중요한 또 다른 이유는 산문과 코드를 매우 긴밀하게 통합하기 때문입니다. 코드를 개발하고 생각을 기록할 수 있게 해주기 때문에 훌륭한 *분석 노트북(analysis notebook)*이 됩니다. 분석 노트북은 물리학의 고전적인 랩 노트와 많은 동일한 목표를 공유합니다:

- 무엇을 했는지, 왜 했는지 기록합니다. 기억력이 아무리 좋더라도 하는 일을 기록하지 않으면 중요한 세부 사항을 잊어버릴 때가 올 것입니다. 잊어버리지 않도록 적어두세요!

- 엄격한 사고를 지원합니다. 진행하면서 생각을 기록하고 계속 성찰한다면 강력한 분석을 도출할 가능성이 더 큽니다. 이는 또한 결국 다른 사람과 공유할 분석을 작성할 때 시간을 절약해 줍니다.

- 다른 사람들이 여러분의 작업을 이해하도록 돕습니다. 혼자 데이터 분석을 하는 경우는 드물며 팀의 일원으로 일하는 경우가 많습니다. 랩 노트는 동료나 실험실 동료와 자신이 한 일뿐만 아니라 왜 했는지 공유하는 데 도움이 됩니다.

랩 노트를 효과적으로 사용하는 것에 대한 좋은 조언의 대부분은 분석 노트북으로 변환될 수도 있습니다. 우리는 우리 자신의 경험과 [랩 노트](https://oreil.ly/n1pLD)에 대한 Colin Purrington의 조언을 바탕으로 다음 팁을 도출했습니다:

- 각 노트북에는 설명이 포함된 제목, 연상되는 파일 이름, 분석의 목표를 간략하게 설명하는 첫 번째 단락이 있어야 합니다.

- YAML 헤더 날짜 필드를 사용하여 노트북 작업을 시작한 날짜를 기록하세요:

  ```yaml
  date: 2016-08-23
  ```

  모호함이 없도록 ISO8601 YYYY-MM-DD 형식을 사용하세요. 평소에 그렇게 날짜를 쓰지 않더라도 사용하세요!

- 분석 아이디어에 많은 시간을 할애했지만 막다른 골목으로 판명되었다면 삭제하지 마세요! 실패한 이유에 대해 간단한 메모를 작성하고 노트북에 남겨두세요. 그렇게 하면 미래에 분석으로 다시 돌아올 때 같은 막다른 골목으로 가는 것을 피하는 데 도움이 됩니다.

- 일반적으로 R 외부에서 데이터 입력을 수행하는 것이 좋습니다. 하지만 데이터의 작은 스니펫을 기록해야 하는 경우 <a href="https://tibble.tidyverse.org/reference/tribble.html" class="orm:hideurl"><code>tibble::tribble()</code></a>을 사용하여 명확하게 배치하세요.

- 데이터 파일에서 오류를 발견하면 직접 수정하지 말고 코드를 작성하여 값을 수정하세요. 수정 이유를 설명하세요.

- 그날의 작업을 마치기 전에 노트북을 렌더링할 수 있는지 확인하세요. 캐싱을 사용하는 경우 캐시를 지우세요. 그렇게 하면 코드가 기억에 아직 생생할 때 문제를 고칠 수 있습니다.

- 코드가 장기적으로 재현 가능하기를 원한다면(즉, 다음 달이나 내년에 코드를 실행하기 위해 돌아올 수 있도록) 코드가 사용하는 패키지의 버전을 추적해야 합니다. 엄격한 접근 방식은 프로젝트 디렉터리에 패키지를 저장하는 [*renv*](https://oreil.ly/_I4xb)를 사용하는 것입니다. 빠르고 대충하는 방법(quick and dirty hack)은 <a href="https://rdrr.io/r/utils/sessionInfo.html" class="orm:hideurl"><code>sessionInfo()</code></a>를 실행하는 청크를 포함하는 것입니다. 그렇게 하면 지금 있는 패키지를 쉽게 다시 만들 수는 없지만 적어도 어떤 패키지가 있었는지는 알 수 있습니다.

- 여러분은 경력을 쌓는 동안 수없이 많은 분석 노트북을 만들게 될 것입니다. 나중에 다시 찾을 수 있도록 어떻게 구성하시겠습니까? 개별 프로젝트에 저장하고 좋은 명명 체계를 고안하는 것이 좋습니다.

# 요약 (Summary)

이 장에서는 코드와 산문을 한곳에 포함하는 재현 가능한 계산 문서를 작성하고 게시하기 위한 Quarto를 소개했습니다. 비주얼 또는 소스 에디터를 사용하여 RStudio에서 Quarto 문서를 작성하는 방법, 코드 청크가 작동하는 방식 및 옵션을 사용자 지정하는 방법, Quarto 문서에 그림과 표를 포함하는 방법, 계산 캐싱 옵션에 대해 배웠습니다. 또한, 독립 실행형 문서나 매개변수화된(parameterized) 문서를 생성하고 인용 및 참고 문헌을 포함하기 위해 YAML 헤더 옵션을 조정하는 방법에 대해 배웠습니다. 또한 문제 해결 및 워크플로우 팁도 제공했습니다.

이 소개가 Quarto를 시작하는 데 충분하지만 여전히 배울 것이 훨씬 더 많습니다. Quarto는 여전히 상대적으로 젊고 빠르게 성장하고 있습니다. 최신 혁신 정보를 얻을 수 있는 가장 좋은 곳은 공식 <a href="https://oreil.ly/_6LNH" class="uri">Quarto 웹사이트</a>입니다.

여기서 다루지 않은 두 가지 중요한 주제가 있습니다: 공동 작업(collaboration)과 아이디어를 다른 사람에게 정확하게 전달하는(communicating) 세부 사항입니다. 공동 작업은 현대 데이터 과학의 중요한 부분이며, Git 및 GitHub와 같은 버전 제어 도구를 사용하면 훨씬 더 편안하게 작업할 수 있습니다. Jenny Bryan이 작성한 R 사용자를 위한 Git 및 GitHub의 친숙한 소개인 *Happy Git with R*을 추천합니다. 이 책은 [온라인](https://oreil.ly/bzjrw)에서 무료로 이용할 수 있습니다.

우리는 분석 결과를 명확하게 전달하기 위해 실제로 무엇을 써야 하는지에 대해서도 다루지 않았습니다. 작문을 개선하기 위해 Joseph M. Williams 및 Joseph Bizup의 *Style: Lessons in Clarity and Grace* (Pearson) 또는 George Gopen의 *The Sense of Structure: Writing from the Reader’s Perspective* (Pearson)를 읽는 것을 강력히 추천합니다. 두 책 모두 문장과 단락의 구조를 이해하는 데 도움이 되고 글을 더 명확하게 만드는 도구를 제공합니다. (이 책들은 새 책으로 구입하면 다소 비싸지만 많은 영어 수업에서 사용되므로 저렴한 중고책이 많이 있습니다.) George Gopen은 여러 편의 [작문에 대한 짧은 기사(short articles on writing)](https://oreil.ly/qS7tS)도 가지고 있습니다. 변호사를 대상으로 하지만 거의 모든 내용이 데이터 과학자에게도 적용됩니다.
