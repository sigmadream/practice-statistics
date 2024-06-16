# 29장. Quarto 형식 (Quarto Formats)

# 소개

지금까지 Quarto를 사용하여 HTML 문서를 생성하는 방법을 살펴보았습니다. 이 장에서는 Quarto로 생성할 수 있는 다른 여러 출력 유형에 대한 간략한 개요를 제공합니다.

문서의 출력을 설정하는 방법에는 두 가지가 있습니다.

- YAML 헤더를 수정하여 영구적으로:

  ```yaml
  title: "Diamond sizes"
  format: html
  ```

- 수동으로 `quarto::quarto_render()`를 호출하여 일시적으로:

  ```r
  quarto::quarto_render("diamond-sizes.qmd", output_format = "docx")
  ```

  이것은 `output_format` 인자가 값들의 리스트를 취할 수도 있기 때문에 프로그래밍 방식으로 여러 유형의 출력을 생성하려는 경우에 유용합니다.

  ```r
  quarto::quarto_render(
    "diamond-sizes.qmd", output_format = c("docx", "pdf")
  )
  ```

# 출력 옵션 (Output Options)

Quarto는 광범위한 출력 형식을 제공합니다. 전체 목록은 [모든 형식에 대한 Quarto 문서](https://oreil.ly/mhYNQ)에서 확인할 수 있습니다. 많은 형식이 일부 출력 옵션(목차 포함을 위한 `toc: true`)을 공유하지만 다른 형식에는 형식별 옵션이 있습니다(`code-fold: true`는 HTML 출력의 경우 사용자가 필요에 따라 표시할 수 있도록 코드 청크를 `<details>` 태그로 축소하며 PDF 또는 Word 문서에는 적용되지 않습니다).

기본 옵션을 재정의하려면 확장된 `format` 필드를 사용해야 합니다. 예를 들어 떠다니는 목차가 있는 HTML 문서를 렌더링하려면 다음을 사용합니다.

```yaml
format:
  html:
    toc: true
    toc_float: true
```

형식 리스트를 제공하여 여러 출력으로 렌더링할 수도 있습니다.

```yaml
format:
  html:
    toc: true
    toc_float: true
  pdf: default
  docx: default
```

기본 옵션을 재정의하지 않으려면 특수 구문(`pdf: default`)에 유의하세요.

문서의 YAML에 지정된 모든 형식으로 렌더링하려면 `output_format = "all"`을 사용할 수 있습니다.

```r
quarto::quarto_render("diamond-sizes.qmd", output_format = "all")
```

# 문서 (Documents)

이전 장에서는 기본 `html` 출력에 중점을 두었습니다. 해당 테마에는 여러 가지 기본 변형이 있어 다양한 유형의 문서를 생성합니다. 예를 들어:

- `pdf`는 설치해야 하는 오픈 소스 문서 레이아웃 시스템인 LaTeX를 사용하여 PDF를 만듭니다. 아직 없는 경우 RStudio에서 메시지를 표시합니다.

- Microsoft Word(`.docx`) 문서용 `docx`.

- OpenDocument 텍스트(`.odt`) 문서용 `odt`.

- 서식 있는 텍스트 형식(Rich Text Format, `.rtf`) 문서용 `rtf`.

- GitHub Flavored Markdown(`.md`) 문서용 `gfm`.

- Jupyter 노트북(`.ipynb`)용 `ipynb`.

의사 결정자와 공유할 문서를 생성할 때 문서 YAML에서 전역 옵션을 설정하여 코드의 기본 표시를 끌 수 있음을 기억하세요.

```yaml
execute:
  echo: false
```

HTML 문서의 경우 또 다른 옵션은 코드 청크를 기본적으로 숨기지만 클릭하면 표시되도록 만드는 것입니다.

```yaml
format:
  html:
    code: true
```

# 프레젠테이션 (Presentations)

Quarto를 사용하여 프레젠테이션을 생성할 수도 있습니다. Keynote나 PowerPoint 같은 도구를 사용할 때보다 시각적 제어는 덜하지만, R 코드 결과를 프레젠테이션에 자동으로 삽입하면 엄청난 시간을 절약할 수 있습니다. 프레젠테이션은 콘텐츠를 슬라이드로 나누어 작동하며, 두 번째 수준 헤더(`##`)마다 새 슬라이드가 시작됩니다. 또한 첫 번째 수준 헤더(`#`)는 기본적으로 중앙에 정렬되는 섹션 제목 슬라이드와 함께 새 섹션의 시작을 나타냅니다.

Quarto는 다음을 포함하여 다양한 프레젠테이션 형식을 지원합니다.

`revealjs`  
revealjs를 이용한 HTML 프레젠테이션

`pptx`  
PowerPoint 프레젠테이션

`beamer`  
LaTeX Beamer를 이용한 PDF 프레젠테이션

[Quarto](https://oreil.ly/Jg7T9)로 프레젠테이션을 만드는 방법에 대해 자세히 알아볼 수 있습니다.

# 상호 작용 (Interactivity)

다른 HTML 문서와 마찬가지로 Quarto로 만든 HTML 문서에도 대화형(interactive) 구성 요소를 포함할 수 있습니다. 여기서는 Quarto 문서에 상호 작용을 포함하기 위한 두 가지 옵션인 htmlwidgets와 Shiny를 소개합니다.

## htmlwidgets

HTML은 대화형 형식이므로 대화형 HTML 시각화를 생성하는 R 함수인 *htmlwidgets*를 통해 해당 상호 작용을 활용할 수 있습니다. 예를 들어 다음에 표시된 _leaflet_ 지도를 살펴보세요. 웹에서 이 페이지를 보고 있다면 지도를 이리저리 끌거나(drag) 확대 및 축소할 수 있습니다. 책에서는 분명히 그렇게 할 수 없으므로 Quarto가 자동으로 정적 스크린샷을 삽입해 줍니다.

```r
library(leaflet)
leaflet() |>
  setView(174.764, -36.877, zoom = 16) |>
  addTiles() |>
  addMarkers(174.764, -36.877, popup = "Maungawhau")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_29in01.png" alt="Leaflet map of Maungawhau/Mount Eden." />
</figure>

htmlwidgets의 좋은 점은 이것을 사용하기 위해 HTML이나 JavaScript에 대해 아무것도 몰라도 된다는 것입니다. 모든 세부 사항이 패키지 내부에 래핑되어 있으므로 걱정할 필요가 없습니다.

다음과 같이 htmlwidgets를 제공하는 많은 패키지가 있습니다.

- 대화형 시계열 시각화를 위한 [dygraphs](https://oreil.ly/SE3qV)

- 대화형 표를 위한 [DT](https://oreil.ly/l3tFl)

- 대화형 3D 플롯을 위한 [threejs](https://oreil.ly/LQZud)

- 다이어그램(순서도 및 간단한 노드-링크 다이어그램 등)을 위한 [DiagrammeR](https://oreil.ly/gQork)

htmlwidgets에 대해 자세히 알아보고 이를 제공하는 전체 패키지 목록을 보려면 [_https://oreil.ly/lmdha_](https://oreil.ly/lmdha)를 방문하세요.

## Shiny

htmlwidgets는 _클라이언트 측_ 상호 작용을 제공합니다. 모든 상호 작용은 R과 독립적으로 브라우저에서 발생합니다. R에 연결하지 않고도 HTML 파일을 배포할 수 있기 때문에 좋습니다. 그러나 기본적으로 HTML 및 JavaScript로 구현된 것으로 수행할 수 있는 작업이 제한됩니다. 또 다른 접근 방식은 JavaScript가 아닌 R 코드를 사용하여 상호 작용을 만들 수 있는 패키지인 shiny를 사용하는 것입니다.

Quarto 문서에서 Shiny 코드를 호출하려면 YAML 헤더에 `server: shiny`를 추가합니다.

```yaml
title: "Shiny Web App"
format: html
server: shiny
```

그런 다음 "input" 함수를 사용하여 문서에 대화형 구성 요소를 추가할 수 있습니다.

```r
library(shiny)

textInput("name", "What is your name?")
numericInput("age", "How old are you?", NA, min = 0, max = 150)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_29in02.png" alt="Two input boxes on top of each other. Top one says, &quot;What is your name?&quot;, the bottom, &quot;How old are you?&quot;." />
</figure>

또한 Shiny 서버에서 실행해야 하는 코드를 포함하는 청크 옵션 `context: server`가 있는 코드 청크도 필요합니다.

그런 다음 `input$name` 및 `input$age`로 값을 참조할 수 있으며, 이들을 사용하는 코드는 값이 변경될 때마다 자동으로 다시 실행됩니다.

Shiny 상호 작용이 *서버 측*에서 발생하기 때문에 여기서는 라이브 Shiny 앱을 보여줄 수 없습니다. 즉, JavaScript를 몰라도 대화형 앱을 작성할 수 있지만 앱을 실행할 서버가 필요합니다. 이로 인해 물류상의 문제가 발생합니다. Shiny 앱을 온라인에서 실행하려면 Shiny 서버가 필요합니다. 자체 컴퓨터에서 Shiny 앱을 실행할 때 Shiny는 자동으로 Shiny 서버를 설정하지만, 이런 종류의 상호 작용을 온라인에 게시하려면 공개용(public-facing) Shiny 서버가 필요합니다. 그것이 Shiny의 근본적인 트레이드오프입니다. R에서 할 수 있는 모든 것을 Shiny 문서에서 할 수 있지만 누군가가 R을 실행하고 있어야 합니다.

Shiny에 대해 더 자세히 알아보려면 해들리 위컴(Hadley Wickham)의 [_Mastering Shiny_](https://oreil.ly/4Id6V)를 읽어보는 것을 권장합니다.

# 웹사이트 및 책 (Websites and Books)

약간의 추가 인프라를 통해 Quarto를 사용하여 완전한 웹사이트나 책을 생성할 수 있습니다.

- `.qmd` 파일을 단일 디렉터리에 넣습니다. `index.qmd`가 홈페이지가 됩니다.

- 사이트 탐색(navigation)을 제공하는 `_quarto.yml`이라는 YAML 파일을 추가합니다. 이 파일에서 `project` 유형을 `book` 또는 `website`로 설정합니다. 예:

  ```yaml
  project:
    type: book
  ```

예를 들어, 다음 `_quarto.yml` 파일은 `index.qmd`(홈페이지), `viridis-colors.qmd`, `terrain-colors.qmd`의 세 가지 소스 파일에서 웹사이트를 만듭니다.

```yaml
project:
  type: website

website:
  title: "A website on color scales"
  navbar:
    left:
      - href: index.qmd
        text: Home
      - href: viridis-colors.qmd
        text: Viridis colors
      - href: terrain-colors.qmd
        text: Terrain colors
```

책에 필요한 `_quarto.yml` 파일도 비슷하게 구성되어 있습니다. 다음 예제는 세 가지 다른 출력(`html`, `pdf` 및 `epub`)으로 렌더링되는 4개의 장으로 구성된 책을 만드는 방법을 보여줍니다. 다시 한 번 말하지만, 소스 파일은 `.qmd` 파일입니다.

```yaml
project:
  type: book

book:
  title: "A book on color scales"
  author: "Jane Coloriste"
  chapters:
    - index.qmd
    - intro.qmd
    - viridis-colors.qmd
    - terrain-colors.qmd

format:
  html:
    theme: cosmo
  pdf: default
  epub: default
```

웹사이트 및 책에 RStudio 프로젝트를 사용하는 것이 좋습니다. `_quarto.yml` 파일을 기반으로 RStudio는 작업 중인 프로젝트 유형을 인식하고 IDE에 빌드(Build) 탭을 추가하여 웹사이트와 책을 렌더링하고 미리 보는 데 사용할 수 있습니다. 웹사이트와 책은 모두 `quarto::render()`를 사용하여 렌더링할 수도 있습니다.

[Quarto 웹사이트](https://oreil.ly/P-n37) 및 [책](https://oreil.ly/fiB1h)에 대해 자세히 알아보세요.

# 기타 형식 (Other Formats)

Quarto는 훨씬 더 많은 출력 형식을 제공합니다.

- [Quarto Journal Templates](https://oreil.ly/ovWgb)를 사용하여 학술지 논문을 작성할 수 있습니다.

- [`format: ipynb`](https://oreil.ly/q-E7l)을 사용하여 Quarto 문서를 Jupyter Notebook으로 출력할 수 있습니다.

더 많은 형식의 목록은 [Quarto 형식 문서](https://oreil.ly/-iGxF)를 참조하세요.

# 요약 (Summary)

이 장에서는 정적 및 대화형 문서에서 프레젠테이션, 웹사이트, 책에 이르기까지 Quarto와 결과를 소통하기 위한 다양한 옵션을 제시했습니다.

이러한 다양한 형식에서의 효과적인 소통에 대해 더 자세히 알아보려면 다음 리소스를 권장합니다.

- 프레젠테이션 기술을 향상시키려면 닐 포드(Neal Ford), 매튜 매컬러(Matthew McCollough), 나다니엘 슈타(Nathaniel Schutta)의 [_Presentation Patterns_](https://oreil.ly/JnOwJ)를 시도해 보세요. 프레젠테이션을 개선하기 위해 적용할 수 있는 효과적인 패턴(낮은 수준과 높은 수준 모두) 세트를 제공합니다.

- 학술적인 강연을 한다면 ["강연을 위한 Leek 그룹 가이드"](https://oreil.ly/ST4yc)가 마음에 들 것입니다.

- 우리가 직접 수강하지는 않았지만 맷 맥가리티(Matt McGarrity)의 [대중 연설](https://oreil.ly/lXY9u)에 대한 온라인 과정에 대해 좋은 이야기를 들었습니다.

- 대시보드를 많이 만드는 경우 Stephen Few의 _Information Dashboard Design: The Effective Visual Communication of Data_(O'Reilly)를 꼭 읽어보세요. 보기 좋은 대시보드가 아니라 진정으로 유용한 대시보드를 만드는 데 도움이 될 것입니다.

- 아이디어를 효과적으로 전달하는 것은 종종 그래픽 디자인에 대한 약간의 지식에서 이점을 얻습니다. Robin Williams의 _The Non-Designer’s Design Book_(Peachpit Press)이 시작하기에 좋은 곳입니다.
