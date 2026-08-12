# 제24장. 웹 스크래핑

# 소개

이 장에서는 [rvest](https://oreil.ly/lUNa6)를 사용한 웹 스크래핑의 기본 사항을 소개합니다. 웹 스크래핑은 웹 페이지에서 데이터를 추출하는 데 유용한 도구입니다. 일부 웹사이트는 데이터를 JSON으로 반환하는 구조화된 HTTP 요청 집합인 API를 제공하며, 이는 <a href="ch23.html#chp-rectangling" data-type="xref">23장</a>의 기술을 사용하여 처리합니다. 가능한 한 API를 사용해야 하는데,<sup><a href="ch24.html#idm44771274112496" id="idm44771274112496-marker" data-type="noteref">1</a></sup> 일반적으로 더 신뢰할 수 있는 데이터를 제공하기 때문입니다. 하지만 안타깝게도 웹 API 프로그래밍은 이 책의 범위를 벗어납니다. 대신 우리는 사이트가 API를 제공하는지 여부에 관계없이 작동하는 기술인 스크래핑을 가르칩니다.

이 장에서는 HTML의 기본 사항으로 들어가기 전에 먼저 스크래핑의 윤리성과 합법성에 대해 논의할 것입니다. 그런 다음 페이지에서 특정 요소를 찾는 CSS 선택자(selector)의 기본 사항과, rvest 함수를 사용하여 HTML에서 텍스트 및 속성(attribute)의 데이터를 R로 가져오는 방법을 배울 것입니다. 이어서 몇 가지 사례 연구와 동적 웹사이트에 대한 간략한 논의로 마무리하기 전에, 스크래핑하는 페이지에 필요한 CSS 선택자가 무엇인지 파악하는 몇 가지 기술에 대해 논의할 것입니다.

## 사전 준비

이 장에서는 rvest가 제공하는 도구에 초점을 맞출 것입니다. rvest는 tidyverse의 멤버이지만 핵심 멤버는 아니므로 명시적으로 로드해야 합니다. 또한 스크래핑한 데이터로 작업할 때 일반적으로 유용하게 쓰일 것이므로 전체 tidyverse도 로드할 것입니다.

```
library(tidyverse)
library(rvest)
```

# 스크래핑의 윤리성과 합법성

웹 스크래핑을 수행하는 데 필요한 코드를 논의하기 전에, 이 작업을 수행하는 것이 합법적이고 윤리적인지에 대해 이야기해야 합니다. 전반적으로 이 두 가지 모두와 관련하여 상황이 복잡합니다.

합법성은 여러분이 거주하는 곳에 따라 크게 달라집니다. 하지만 일반적인 원칙으로, 데이터가 공개적(public)이고, 비개인적(nonpersonal)이며, 사실(factual)이라면 괜찮을 가능성이 높습니다.<sup><a href="ch24.html#idm44771274043040" id="idm44771274043040-marker" data-type="noteref">2</a></sup> 우리가 논의하겠지만, 이 세 가지 요소가 사이트의 이용 약관(terms and conditions), 개인 식별 정보(personally identifiable information), 저작권(copyright)과 연결되어 있기 때문에 중요합니다.

데이터가 공개적이지 않거나, 비개인적이지 않거나, 사실이 아니거나, 또는 데이터를 스크래핑하여 구체적으로 돈을 벌려고 한다면 변호사와 상의해야 합니다. 어떤 경우든 스크래핑하려는 페이지를 호스팅하는 서버의 자원을 존중해야 합니다. 중요한 것은 많은 페이지를 스크래핑하는 경우 각 요청 사이에 잠시 기다려야 한다는 것입니다. 이를 쉽게 하는 한 가지 방법은 Dmytro Perepolkin의 [polite 패키지](https://oreil.ly/rlujg)를 사용하는 것입니다. 이것은 요청 사이를 자동으로 일시 중지하고 결과를 캐시하므로 같은 페이지를 두 번 묻지 않게 해줍니다.

## 서비스 약관

자세히 살펴보면 많은 웹사이트가 페이지 어딘가에 "이용 약관(terms and conditions)" 또는 "서비스 약관(terms of service)" 링크를 포함하고 있으며, 해당 페이지를 주의 깊게 읽어보면 사이트가 웹 스크래핑을 구체적으로 금지한다는 사실을 종종 발견할 수 있습니다. 이러한 페이지는 기업이 매우 광범위한 주장을 펴는 법적인 땅따먹기(land grab)인 경향이 있습니다. 가능한 한 이러한 서비스 약관을 존중하는 것이 예의지만, 어떤 주장이든 비판적으로 받아들여야 합니다(take any claims with a grain of salt).

미국 법원은 웹사이트 바닥글(footer)에 서비스 약관을 넣는 것만으로는 여러분이 그 약관에 구속되기에 충분하지 않다고 일반적으로 판결했습니다([_HiQ Labs v. LinkedIn_](https://oreil.ly/mDAin)). 일반적으로 서비스 약관에 구속되려면 계정을 만들거나 확인란(check box)을 선택하는 등의 명시적인 조치를 취했어야 합니다. 이것이 데이터가 *공개적(public)*인지 여부가 중요한 이유입니다. 데이터에 접근하기 위해 계정이 필요하지 않다면 서비스 약관에 구속될 가능성이 낮습니다. 하지만 명시적으로 동의하지 않았더라도 서비스 약관이 집행 가능하다고 법원이 판결한 유럽에서는 상황이 다소 다르다는 점에 유의하세요.

## 개인 식별 정보

데이터가 공개적이라 하더라도 이름, 이메일 주소, 전화번호, 생년월일 등 개인을 식별할 수 있는 정보를 스크래핑할 때는 극도로 주의해야 합니다. 유럽은 이러한 데이터의 수집 및 저장에 관해 특히 엄격한 법률([GDPR](https://oreil.ly/nzJwO))을 가지고 있으며, 어디에 거주하든 윤리적 수렁에 빠질 가능성이 높습니다. 예를 들어 2016년에 한 연구자 그룹이 데이팅 사이트 OkCupid에 있는 70,000명의 공개 프로필 정보(사용자 이름, 나이, 성별, 위치 등)를 스크래핑하고 익명화 시도 없이 이 데이터를 공개적으로 배포했습니다. 연구자들은 데이터가 이미 공개되어 있기 때문에 이것에 아무런 문제가 없다고 느꼈지만, 이 연구는 데이터 세트에 정보가 공개된 사용자의 식별 가능성을 둘러싼 윤리적 우려로 인해 광범위한 비난을 받았습니다. 만약 여러분의 작업에 개인 식별 정보를 스크래핑하는 것이 포함되어 있다면, OkCupid 연구<sup><a href="ch24.html#idm44771274018592" id="idm44771274018592-marker" data-type="noteref">3</a></sup>와 개인 식별 정보의 획득 및 배포가 포함된 연구 윤리에 의문이 제기되는 유사한 연구들에 대해 읽어보시기를 강력히 권장합니다.

## 저작권

마지막으로 저작권법에 대해서도 걱정해야 합니다. 저작권법은 복잡하지만, 무엇이 보호되는지 정확히 설명하는 [미국 법](https://oreil.ly/OqUgO)을 살펴볼 가치가 있습니다. "[...] 어떤 유형의 표현 매체에 고정된 독창적인 저작물, [...]". 그런 다음 문학 작품, 음악 작품, 영화 등 이것이 적용되는 구체적인 범주를 설명합니다. 주목할 점은 저작권 보호에서 데이터가 제외된다는 것입니다. 즉, 스크래핑을 사실(facts)로만 제한하는 한 저작권 보호가 적용되지 않습니다. (하지만 유럽에는 데이터베이스를 보호하는 별도의 [“독자적(sui generis)” 권리](https://oreil.ly/0ewJe)가 있다는 점에 유의하세요.)

간단한 예로, 미국에서는 재료 목록과 설명서는 저작권이 없으므로 레시피를 보호하는 데 저작권을 사용할 수 없습니다. 하지만 해당 레시피 목록에 실질적으로 새로운 문학적 내용이 동반된다면 그것은 저작권이 있습니다. 인터넷에서 레시피를 찾을 때 항상 요리법 전에 그렇게 많은 내용이 있는 이유가 바로 이것입니다.

(텍스트나 이미지와 같은) 원본 콘텐츠를 스크래핑해야 하는 경우, [공정 이용(fair use)의 원칙](https://oreil.ly/oFh0-)에 따라 보호받을 수도 있습니다. 공정 이용은 엄격하고 빠른 규칙이 아니라 여러 가지 요소를 저울질하는 것입니다. 연구나 비상업적 목적으로 데이터를 수집하고 스크래핑 대상을 필요한 것만으로 제한하는 경우 적용될 가능성이 더 높습니다.

# HTML 기초

웹 페이지를 스크래핑하려면 먼저 웹 페이지를 설명하는 언어인 *HTML*에 대해 조금 이해해야 합니다. HTML은 HyperText Markup Language의 약자이며 대략 다음과 같이 생겼습니다.

```
<html>
<head>
  <title>Page title</title>
</head>
<body>
  <h1 id='first'>A heading</h1>
  <p>Some text &amp; <b>some bold text.</b></p>
  <img src='myimg.png' width='100' height='100'>
</body>
```

HTML은 *요소(element)*로 구성된 계층 구조를 가지고 있으며, 요소는 시작 태그(start tag, 예: `<tag>`), 선택적 _속성(attributes)_(`id='first'`), 종료 태그(end tag)<sup><a href="ch24.html#idm44771273952560" id="idm44771273952560-marker" data-type="noteref">4</a></sup>(`</tag>`와 같은 형태), 그리고 _내용물(contents)_ (시작 태그와 종료 태그 사이의 모든 것)로 구성됩니다.

`<`와 `>`는 시작 및 종료 태그에 사용되므로 직접 작성할 수 없습니다. 대신 HTML _이스케이프(escape)_ 문자열인 `&gt;`(크다)와 `&lt;`(작다)를 사용해야 합니다. 그리고 이러한 이스케이프가 `&`를 사용하기 때문에 글자 그대로의 앰퍼샌드(&)를 원한다면 `&amp;`로 이스케이프해야 합니다. 가능한 HTML 이스케이프의 종류는 매우 다양하지만 rvest가 이를 자동으로 처리해 주므로 너무 걱정할 필요는 없습니다.

웹 스크래핑이 가능한 이유는 스크래핑하려는 데이터가 포함된 대부분의 페이지가 일반적으로 일관된 구조를 가지고 있기 때문입니다.

## 요소

100개 이상의 HTML 요소가 있습니다. 중요한 것 중 일부는 다음과 같습니다.

- 모든 HTML 페이지는 `<html>` 요소 안에 있어야 하며, 두 개의 자식을 가져야 합니다. 페이지 제목과 같은 문서 메타데이터가 포함된 `<head>`, 브라우저에서 보는 콘텐츠가 포함된 `<body>`입니다.

- `<h1>`(제목 1), `<section>`(섹션), `<p>`(문단), `<ol>`(순서 있는 목록)과 같은 블록(Block) 태그는 페이지의 전체 구조를 형성합니다.

- `<b>`(굵게), `<i>`(기울임꼴), `<a>`(링크)와 같은 인라인(Inline) 태그는 블록 태그 내부의 텍스트 형식을 지정합니다.

이전에 본 적 없는 태그를 발견하면 구글 검색을 통해 그 역할을 알아낼 수 있습니다. 또 다른 좋은 출발점은 웹 프로그래밍의 거의 모든 측면을 설명하는 [MDN Web Docs](https://oreil.ly/qIgHp)입니다.

대부분의 요소는 시작 태그와 종료 태그 사이에 내용물을 가질 수 있습니다. 이 내용물은 텍스트일 수도 있고 더 많은 요소일 수도 있습니다. 예를 들어 다음 HTML에는 텍스트로 된 문단이 포함되어 있으며 한 단어가 굵게 표시되어 있습니다.

```
<p> Hi! My <b>name</b> is Hadley. </p>
```

*자식(children)*은 그것이 포함하는 요소를 말합니다. 따라서 이전의 `<p>` 요소에는 하나의 자식인 `<b>` 요소가 있습니다. `<b>` 요소에는 자식이 없지만 내용물(텍스트 "name")은 있습니다.

## 속성

태그는 `name1='value1' name2='value2'`처럼 보이는 이름이 지정된 *속성(attributes)*을 가질 수 있습니다. 중요한 두 가지 속성은 `id`와 `class`이며, 이는 페이지의 시각적 모양을 제어하기 위해 캐스케이딩 스타일 시트(Cascading Style Sheets, CSS)와 함께 사용됩니다. 이들은 종종 페이지에서 데이터를 스크래핑할 때 유용합니다. 속성은 링크의 목적지(`<a>` 요소의 `href` 속성)와 이미지의 출처(`<img>` 요소의 `src` 속성)를 기록하는 데도 사용됩니다.

# 데이터 추출하기

스크래핑을 시작하려면 스크래핑할 페이지의 URL이 필요한데, 일반적으로 웹 브라우저에서 복사할 수 있습니다. 그런 다음 <a href="http://xml2.r-lib.org/reference/read_xml.html" class="orm:hideurl"><code>read_html()</code></a>을 사용하여 해당 페이지의 HTML을 R로 읽어 들여야 합니다. 그러면 `xml_document`<sup><a href="ch24.html#idm44771273871552" id="idm44771273871552-marker" data-type="noteref">5</a></sup> 객체가 반환되며, 이를 rvest 함수를 사용하여 조작하게 됩니다.

```
html <- read_html("http://rvest.tidyverse.org/")
html
#> {html_document}
#> <html lang="en">
#> [1] <head>\n<meta http-equiv="Content-Type" content="text/html; charset=UT ...
#> [2] <body>\n  <a href="#container" class="visually-hidden-focusable">Ski ...
```

rvest에는 인라인(inline)으로 HTML을 작성할 수 있는 함수도 포함되어 있습니다. 간단한 예제를 통해 다양한 rvest 함수가 어떻게 작동하는지 가르치면서 이 장에서 이것을 많이 사용할 것입니다.

```
html <- minimal_html("
  <p>This is a paragraph</p>
  <ul>
    <li>This is a bulleted list</li>
  </ul>
")
html
#> {html_document}
#> <html>
#> [1] <head>\n<meta http-equiv="Content-Type" content="text/html; charset=UT ...
#> [2] <body>\n<p>This is a paragraph</p>\n<p>\n  </p>\n<ul>\n<li>This is a b ...
```

이제 R에 HTML이 있으므로 관심 있는 데이터를 추출할 차례입니다. 먼저 관심 요소를 식별할 수 있게 해주는 CSS 선택자(selector)와 해당 요소에서 데이터를 추출하는 데 사용할 수 있는 rvest 함수에 대해 배울 것입니다. 그런 다음 몇 가지 특수 도구가 있는 HTML 테이블에 대해 간략히 다루겠습니다.

## 요소 찾기

CSS는 HTML 문서의 시각적 스타일링을 정의하기 위한 도구입니다. CSS에는 *CSS 선택자(CSS selectors)*라는 페이지의 요소를 선택하기 위한 미니 언어가 포함되어 있습니다. CSS 선택자는 HTML 요소를 찾기 위한 패턴을 정의하며, 추출하려는 요소를 설명하는 간결한 방법을 제공하기 때문에 스크래핑에 유용합니다.

<a href="#sec-css-selectors" data-type="xref">“올바른 선택자 찾기”</a>에서 CSS 선택자에 대해 더 자세히 다루겠지만, 다행히 다음 세 가지만으로도 많은 것을 할 수 있습니다.

`p`  
모든 `<p>` 요소를 선택합니다.

`.title`  
`class`가 "title"인 모든 요소를 선택합니다.

`#title`  
`id` 속성이 "title"인 요소를 선택합니다. `id` 속성은 문서 내에서 고유해야 하므로 항상 단 하나의 요소만 선택합니다.

간단한 예제에서 이 선택자들을 사용해 보겠습니다.

```
html <- minimal_html("
  <h1>This is a heading</h1>
  <p id='first'>This is a paragraph</p>
  <p class='important'>This is an important paragraph</p>
")
```

선택자와 일치하는 모든 요소를 찾으려면 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>를 사용하세요.

```
html |> html_elements("p")
#> {xml_nodeset (2)}
#> [1] <p id="first">This is a paragraph</p>
#> [2] <p class="important">This is an important paragraph</p>

html |> html_elements(".important")
#> {xml_nodeset (1)}
#> [1] <p class="important">This is an important paragraph</p>

html |> html_elements("#first")
#> {xml_nodeset (1)}
#> [1] <p id="first">This is a paragraph</p>
```

또 다른 중요한 함수는 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>이며, 항상 입력과 같은 수의 출력을 반환합니다. 전체 문서에 적용하면 첫 번째 일치 항목을 제공합니다.

```
html |> html_element("p")
#> {html_node}
#> <p id="first">
```

어떤 요소와도 일치하지 않는 선택자를 사용할 때 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>와 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a> 사이에는 중요한 차이가 있습니다. <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>는 길이가 0인 벡터를 반환하는 반면 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>는 결측값을 반환합니다. 이것은 잠시 후에 중요해질 것입니다.

```
html |> html_elements("b")
#> {xml_nodeset (0)}
html |> html_element("b")
#> {xml_missing}
#> <NA>
```

## 중첩 선택

대부분의 경우 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>와 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>를 함께 사용하게 되며, 일반적으로 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>를 사용하여 관측값(observation)이 될 요소를 식별한 다음 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>를 사용하여 변수가 될 요소를 찾습니다. 간단한 예제를 통해 이것이 어떻게 작동하는지 보겠습니다. 여기 각 리스트 항목(`<li>`)이 *스타워즈(Star Wars)*의 네 캐릭터에 대한 약간의 정보를 포함하는 순서 없는 목록(`<ul>`)이 있습니다.

```
html <- minimal_html("
  <ul>
    <li><b>C-3PO</b> is a <i>droid</i> that weighs <span class='weight'>167 kg</span></li>
    <li><b>R4-P17</b> is a <i>droid</i></li>
    <li><b>R2-D2</b> is a <i>droid</i> that weighs <span class='weight'>96 kg</span></li>
    <li><b>Yoda</b> weighs <span class='weight'>66 kg</span></li>
  </ul>
  ")
```

<a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>를 사용하여 각 요소가 다른 캐릭터에 해당하는 벡터를 만들 수 있습니다.

```
characters <- html |> html_elements("li")
characters
#> {xml_nodeset (4)}
#> [1] <li>\n<b>C-3PO</b> is a <i>droid</i> that weighs <span class="weight"> ...
#> [2] <li>\n<b>R4-P17</b> is a <i>droid</i>\n</li>
#> [3] <li>\n<b>R2-D2</b> is a <i>droid</i> that weighs <span class="weight"> ...
#> [4] <li>\n<b>Yoda</b> weighs <span class="weight">66 kg</span>\n</li>
```

각 캐릭터의 이름을 추출하려면 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>를 사용합니다. <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>의 출력에 적용될 때 요소당 하나의 응답을 반환하도록 보장되기 때문입니다.

```
characters |> html_element("b")
#> {xml_nodeset (4)}
#> [1] <b>C-3PO</b>
#> [2] <b>R4-P17</b>
#> [3] <b>R2-D2</b>
#> [4] <b>Yoda</b>
```

이름의 경우 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>와 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>의 차이가 중요하지 않지만, 몸무게의 경우는 중요합니다. 몸무게를 뜻하는 `<span>`이 없는 경우에도 각 캐릭터에 대해 하나의 몸무게 값을 얻고 싶습니다. 그것이 바로 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>가 하는 일입니다.

```
characters |> html_element(".weight")
#> {xml_nodeset (4)}
#> [1] <span class="weight">167 kg</span>
#> [2] <NA>
#> [3] <span class="weight">96 kg</span>
#> [4] <span class="weight">66 kg</span>
```

<a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>는 `characters`의 자식인 모든 몸무게 `<span>`을 찾습니다. 세 개밖에 없으므로 이름과 몸무게 간의 연결이 끊어집니다.

```
characters |> html_elements(".weight")
#> {xml_nodeset (3)}
#> [1] <span class="weight">167 kg</span>
#> [2] <span class="weight">96 kg</span>
#> [3] <span class="weight">66 kg</span>
```

관심 요소를 선택했으니 이제 텍스트 내용물이나 일부 속성에서 데이터를 추출해야 합니다.

## 텍스트와 속성

<a href="https://rvest.tidyverse.org/reference/html_text.html" class="orm:hideurl"><code>html_text2()</code></a><sup><a href="ch24.html#idm44771273417888" id="idm44771273417888-marker" data-type="noteref">6</a></sup>는 HTML 요소의 일반 텍스트 내용물을 추출합니다.

```
characters |>
  html_element("b") |>
  html_text2()
#> [1] "C-3PO"  "R4-P17" "R2-D2"  "Yoda"

characters |>
  html_element(".weight") |>
  html_text2()
#> [1] "167 kg" NA       "96 kg"  "66 kg"
```

모든 이스케이프는 자동으로 처리된다는 점에 유의하세요. HTML 이스케이프는 소스 HTML에서만 볼 수 있고 rvest가 반환한 데이터에서는 절대 볼 수 없습니다.

<a href="https://rvest.tidyverse.org/reference/html_attr.html" class="orm:hideurl"><code>html_attr()</code></a>은 속성에서 데이터를 추출합니다.

```
html <- minimal_html("
  <p><a href='https://en.wikipedia.org/wiki/Cat'>cats</a></p>
  <p><a href='https://en.wikipedia.org/wiki/Dog'>dogs</a></p>
")

html |>
  html_elements("p") |>
  html_element("a") |>
  html_attr("href")
#> [1] "https://en.wikipedia.org/wiki/Cat" "https://en.wikipedia.org/wiki/Dog"
```

<a href="https://rvest.tidyverse.org/reference/html_attr.html" class="orm:hideurl"><code>html_attr()</code></a>은 항상 문자열을 반환하므로 숫자나 날짜를 추출하는 경우에는 일부 사후 처리가 필요합니다.

## 테이블

운이 좋다면 데이터가 이미 HTML 테이블에 저장되어 있을 것이며, 그저 테이블에서 읽어오기만 하면 됩니다. 일반적으로 브라우저에서 테이블을 인식하는 것은 간단합니다. 행과 열의 직사각형 구조를 가지며 Excel과 같은 도구에 복사하여 붙여넣을 수 있습니다.

HTML 테이블은 `<table>`, `<tr>`(테이블 행), `<th>`(테이블 헤딩), `<td>`(테이블 데이터)라는 4가지 주요 요소로 구성됩니다. 다음은 열 2개와 행 3개가 있는 간단한 HTML 테이블입니다.

```
html <- minimal_html("
  <table class='mytable'>
    <tr><th>x</th>   <th>y</th></tr>
    <tr><td>1.5</td> <td>2.7</td></tr>
    <tr><td>4.9</td> <td>1.3</td></tr>
    <tr><td>7.2</td> <td>8.1</td></tr>
  </table>
  ")
```

rvest는 이런 종류의 데이터를 읽는 방법을 아는 <a href="https://rvest.tidyverse.org/reference/html_table.html" class="orm:hideurl"><code>html_table()</code></a>이라는 함수를 제공합니다. 이 함수는 페이지에서 발견된 각 테이블마다 하나의 티블을 포함하는 리스트를 반환합니다. 추출하려는 테이블을 식별하려면 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_element()</code></a>를 사용하세요.

```
html |>
  html_element(".mytable") |>
  html_table()
#> # A tibble: 3 × 2
#>       x     y
#>   <dbl> <dbl>
#> 1   1.5   2.7
#> 2   4.9   1.3
#> 3   7.2   8.1
```

`x`와 `y`가 자동으로 숫자로 변환되었다는 점에 유의하세요. 이러한 자동 변환이 항상 작동하는 것은 아니므로, 더 복잡한 시나리오에서는 `convert = FALSE`로 이를 끄고 직접 변환을 수행하고 싶을 수 있습니다.

# 올바른 선택자 찾기

데이터에 필요한 선택자를 파악하는 것은 일반적으로 문제에서 어려운 부분입니다. 구체적이면서(즉, 신경 쓰지 않는 것은 선택하지 않음) 민감한(즉, 신경 쓰는 모든 것을 선택함) 선택자를 찾기 위해 종종 약간의 실험을 해야 합니다. 많은 시행착오가 프로세스의 정상적인 부분입니다! 이 프로세스를 돕기 위해 두 가지 주요 도구를 사용할 수 있습니다. SelectorGadget과 브라우저의 개발자 도구입니다.

[SelectorGadget](https://oreil.ly/qui0z)은 여러분이 제공하는 긍정 및 부정 예제를 기반으로 CSS 선택자를 자동으로 생성하는 JavaScript 북마크릿(bookmarklet)입니다. 항상 작동하는 것은 아니지만, 작동할 때는 마법 같습니다! [vignette](https://oreil.ly/qui0z)를 읽거나 [Mine의 비디오](https://oreil.ly/qNv6l)를 시청하여 SelectorGadget을 설치하고 사용하는 방법을 배울 수 있습니다.

모든 최신 브라우저에는 개발자를 위한 몇 가지 툴킷이 포함되어 있지만, 기본 브라우저가 아니더라도 Chrome을 권장합니다. Chrome의 웹 개발자 도구는 최고 수준이며 즉시 사용할 수 있습니다. 페이지의 요소를 마우스 오른쪽 버튼으로 클릭하고 "검사(Inspect)"를 클릭하세요. 그러면 방금 클릭한 요소를 중심으로 확장 가능한 전체 HTML 페이지 보기가 열립니다. 이를 사용하여 페이지를 탐색하고 어떤 선택자가 작동할지 파악할 수 있습니다. `class`와 `id` 속성은 종종 페이지의 시각적 구조를 형성하는 데 사용되므로 찾고 있는 데이터를 추출하기에 좋은 도구가 되기 때문에 특별히 주의를 기울이세요.

"요소(Elements)" 보기 내부에서 요소를 마우스 오른쪽 버튼으로 클릭하고 "복사(Copy)" -> "선택자 복사(Copy selector)"를 선택하여 관심 요소를 고유하게 식별할 선택자를 생성할 수도 있습니다.

SelectorGadget이나 Chrome 개발자 도구가 이해할 수 없는 CSS 선택자를 생성했다면, CSS 선택자를 쉬운 영어로 번역해 주는 [Selectors Explained](https://oreil.ly/eD6eC)를 사용해 보세요. 이런 작업을 많이 하고 있다는 것을 알게 되면 일반적으로 CSS 선택자에 대해 더 배우고 싶을 수 있습니다. 재미있는 [CSS dinner](https://oreil.ly/McJtu) 튜토리얼로 시작한 다음 [MDN 웹 문서](https://oreil.ly/mpfMF)를 참조하는 것을 추천합니다.

# 모든 것을 하나로 모으기

이 모든 것을 종합하여 일부 웹사이트를 스크래핑해 보겠습니다. 여러분이 실행할 때쯤에는 이 예제들이 더 이상 작동하지 않을 위험이 약간 있습니다. 이것이 웹 스크래핑의 근본적인 과제입니다. 사이트의 구조가 변경되면 스크래핑 코드를 변경해야 합니다.

## 스타워즈

rvest에는 <a href="https://rvest.tidyverse.org/articles/starwars.html" class="orm:hideurl"><code>vignette("starwars")</code></a>에 매우 간단한 예제가 포함되어 있습니다. 최소한의 HTML로 구성된 간단한 페이지이므로 좋은 출발점입니다. 지금 해당 페이지로 이동하여 요소 검사(Inspect Element)를 사용해 _스타워즈_ 영화의 제목인 제목들 중 하나를 검사해 보시길 권장합니다. 키보드나 마우스를 사용하여 HTML의 계층 구조를 탐색하고 각 영화에 사용된 공통 구조를 파악할 수 있는지 확인해 보세요.

각 영화에 다음과 같은 공통 구조가 있음을 알 수 있을 것입니다.

```
<section>
  <h2 data-id="1">The Phantom Menace</h2>
  <p>Released: 1999-05-19</p>
  <p>Director: <span class="director">George Lucas</span></p>
  <div class="crawl">
    <p>...</p>
    <p>...</p>
    <p>...</p>
  </div>
</section>
```

우리의 목표는 이 데이터를 `title`, `year`, `director`, `intro` 변수가 있는 7행의 데이터 프레임으로 바꾸는 것입니다. HTML을 읽고 모든 `<section>` 요소를 추출하는 것부터 시작하겠습니다.

```
url <- "https://rvest.tidyverse.org/articles/starwars.html"
html <- read_html(url)

section <- html |> html_elements("section")
section
#> {xml_nodeset (7)}
#> [1] <section><h2 data-id="1">\nThe Phantom Menace\n</h2>\n<p>\nReleased: 1 ...
#> [2] <section><h2 data-id="2">\nAttack of the Clones\n</h2>\n<p>\nReleased: ...
#> [3] <section><h2 data-id="3">\nRevenge of the Sith\n</h2>\n<p>\nReleased: ...
#> [4] <section><h2 data-id="4">\nA New Hope\n</h2>\n<p>\nReleased: 1977-05-2 ...
#> [5] <section><h2 data-id="5">\nThe Empire Strikes Back\n</h2>\n<p>\nReleas ...
#> [6] <section><h2 data-id="6">\nReturn of the Jedi\n</h2>\n<p>\nReleased: 1 ...
#> [7] <section><h2 data-id="7">\nThe Force Awakens\n</h2>\n<p>\nReleased: 20 ...
```

이 코드는 해당 페이지에 있는 7개의 영화와 일치하는 7개의 요소를 검색하며, 이는 `section`을 선택자로 사용하는 것이 좋다는 것을 시사합니다. 데이터가 항상 텍스트에서 발견되기 때문에 개별 요소를 추출하는 것은 간단합니다. 올바른 선택자를 찾는 문제일 뿐입니다.

```
section |> html_element("h2") |> html_text2()
#> [1] "The Phantom Menace"      "Attack of the Clones"
#> [3] "Revenge of the Sith"     "A New Hope"
#> [5] "The Empire Strikes Back" "Return of the Jedi"
#> [7] "The Force Awakens"
section |> html_element(".director") |> html_text2()
#> [1] "George Lucas"     "George Lucas"     "George Lucas"
#> [4] "George Lucas"     "Irvin Kershner"   "Richard Marquand"
#> [7] "J. J. Abrams"
```

각 구성 요소에 대해 이 작업을 수행하고 나면 모든 결과를 티블로 감쌀 수 있습니다.

```
tibble(
  title = section |>
    html_element("h2") |>
    html_text2(),
  released = section |>
    html_element("p") |>
    html_text2() |>
    str_remove("Released: ") |>
    parse_date(),
  director = section |>
    html_element(".director") |>
    html_text2(),
  intro = section |>
    html_element(".crawl") |>
    html_text2()
)
#> # A tibble: 7 × 4
#>   title                   released   director         intro
#>   <chr>                   <date>     <chr>            <chr>
#> 1 The Phantom Menace      1999-05-19 George Lucas     "Turmoil has engulfed …
#> 2 Attack of the Clones    2002-05-16 George Lucas     "There is unrest in th…
#> 3 Revenge of the Sith     2005-05-19 George Lucas     "War! The Republic is …
#> 4 A New Hope              1977-05-25 George Lucas     "It is a period of civ…
#> 5 The Empire Strikes Back 1980-05-17 Irvin Kershner   "It is a dark time for…
#> 6 Return of the Jedi      1983-05-25 Richard Marquand "Luke Skywalker has re…
#> # … with 1 more row
```

나중에 분석에서 사용하기 쉬운 변수를 얻기 위해 `released`를 약간 더 처리했습니다.

## IMDb Top Films

다음 작업으로 조금 더 까다로운 문제를 다루어 보겠습니다. IMDb에서 상위 250개 영화를 추출하는 것입니다. 우리가 이 장을 작성할 당시의 페이지는 <a href="#fig-scraping-imdb" data-type="xref">그림 24-1</a>과 같았습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2401.png" alt="The screenshot shows a table with columns &quot;Rank and Title&quot;, &quot;IMDb Rating&quot;, and &quot;Your Rating&quot;. 9 movies out of the top 250 are shown. The top 5 are the Shawshank Redemption, The Godfather, The Dark Knight, The Godfather: Part II, and 12 Angry Men." />
<h6 id="figure-24-1.-imdb-top-movies-web-page-taken-on-2022-12-05.">그림 24-1. 2022-12-05에 캡처한 IMDb 최고 영화 웹 페이지.</h6>
</figure>

이 데이터는 명확한 표 구조를 가지고 있으므로 <a href="https://rvest.tidyverse.org/reference/html_table.html" class="orm:hideurl"><code>html_table()</code></a>로 시작할 가치가 있습니다.

```
url <- "https://www.imdb.com/chart/top"
html <- read_html(url)

table <- html |>
  html_element("table") |>
  html_table()
table
#> # A tibble: 250 × 5
#>   ``    `Rank & Title`           `IMDb Rating` `Your Rating` ``
#>   <lgl> <chr>                            <dbl> <chr>         <lgl>
#> 1 NA    "1.\n      The Shawshank Redempt…           9.2 "12345678910\n… NA
#> 2 NA    "2.\n      The Godfather\n      …           9.2 "12345678910\n… NA
#> 3 NA    "3.\n      The Dark Knight\n    …           9   "12345678910\n… NA
#> 4 NA    "4.\n      The Godfather Part II…           9   "12345678910\n… NA
#> 5 NA    "5.\n      12 Angry Men\n       …           9   "12345678910\n… NA
#> 6 NA    "6.\n      Schindler's List\n   …           8.9 "12345678910\n… NA
#> # … with 244 more rows
```

이것은 빈 열을 몇 개 포함하고 있지만 전반적으로 테이블에서 정보를 잘 캡처합니다. 하지만 사용하기 쉽게 만들려면 약간의 처리가 더 필요합니다. 먼저 열 이름을 다루기 쉽게 바꾸고 순위와 제목에서 불필요한 공백을 제거하겠습니다. 우리는 (<a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename()</code></a> 대신) <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>를 사용하여 이 두 열만 선택하고 이름을 바꾸는 작업을 한 번에 수행할 것입니다. 그런 다음 줄 바꿈과 여분의 공백을 제거하고, <a href="ch15.html#sec-extract-variables" data-type="xref">“변수 추출하기”</a>의 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>를 적용하여 제목, 연도, 순위를 각각의 변수로 뽑아낼 것입니다.

```
ratings <- table |>
  select(
    rank_title_year = `Rank & Title`,
    rating = `IMDb Rating`
  ) |>
  mutate(
    rank_title_year = str_replace_all(rank_title_year, "\n +", " ")
  ) |>
  separate_wider_regex(
    rank_title_year,
    patterns = c(
      rank = "\\d+", "\\. ",
      title = ".+", " +\\(",
      year = "\\d+", "\\)"
    )
  )
ratings
#> # A tibble: 250 × 4
#>   rank  title                    year  rating
#>   <chr> <chr>                    <chr>  <dbl>
#> 1 1     The Shawshank Redemption 1994     9.2
#> 2 2     The Godfather            1972     9.2
#> 3 3     The Dark Knight          2008     9
#> 4 4     The Godfather Part II    1974     9
#> 5 5     12 Angry Men             1957     9
#> 6 6     Schindler's List         1993     8.9
#> # … with 244 more rows
```

대부분의 데이터가 테이블 셀에서 나오는 이 경우에도 원시 HTML을 살펴보는 것은 여전히 가치가 있습니다. 그렇게 하면 속성 중 하나를 사용하여 약간의 추가 데이터를 더할 수 있다는 것을 알게 될 것입니다. 이것이 페이지 소스를 파고드는 데 약간의 시간을 할애할 가치가 있는 이유 중 하나입니다. 추가 데이터를 찾거나 파싱 경로가 약간 더 쉬운 것을 발견할 수 있습니다.

```
html |>
  html_elements("td strong") |>
  head() |>
  html_attr("title")
#> [1] "9.2 based on 2,712,990 user ratings"
#> [2] "9.2 based on 1,884,423 user ratings"
#> [3] "9.0 based on 2,685,826 user ratings"
#> [4] "9.0 based on 1,286,204 user ratings"
#> [5] "9.0 based on 801,579 user ratings"
#> [6] "8.9 based on 1,370,458 user ratings"
```

이것을 표 데이터와 결합하고 다시 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>를 적용하여 우리가 관심 있는 데이터 부분을 추출할 수 있습니다.

```
ratings |>
  mutate(
    rating_n = html |> html_elements("td strong") |> html_attr("title")
  ) |>
  separate_wider_regex(
    rating_n,
    patterns = c(
      "[0-9.]+ based on ",
      number = "[0-9,]+",
      " user ratings"
    )
  ) |>
  mutate(
    number = parse_number(number)
  )
#> # A tibble: 250 × 5
#>   rank  title                    year  rating  number
#>   <chr> <chr>                    <chr>  <dbl>   <dbl>
#> 1 1     The Shawshank Redemption 1994     9.2 2712990
#> 2 2     The Godfather            1972     9.2 1884423
#> 3 3     The Dark Knight          2008     9   2685826
#> 4 4     The Godfather Part II    1974     9   1286204
#> 5 5     12 Angry Men             1957     9    801579
#> 6 6     Schindler's List         1993     8.9 1370458
#> # … with 244 more rows
```

# 동적 사이트

지금까지는 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a>가 브라우저에서 보는 것을 반환하는 웹사이트에 초점을 맞추었고, 반환된 것을 파싱하는 방법과 깔끔한 데이터 프레임으로 그 정보를 어떻게 구성하는지 논의했습니다. 하지만 때로는 <a href="https://rvest.tidyverse.org/reference/html_element.html" class="orm:hideurl"><code>html_elements()</code></a> 및 친구들이 브라우저에서 보는 것과 같은 내용을 반환하지 않는 사이트에 부딪힐 때가 있습니다. 많은 경우, 그 이유는 JavaScript를 사용하여 페이지의 콘텐츠를 동적으로 생성하는 웹사이트를 스크래핑하려고 시도하기 때문입니다. rvest는 원시 HTML을 다운로드하고 JavaScript는 실행하지 않기 때문에 현재는 이 방식이 rvest에서 작동하지 않습니다.

이러한 유형의 사이트를 스크래핑하는 것은 여전히 가능하지만, rvest는 더 비싼(expensive) 프로세스를 사용해야 합니다. 모든 JavaScript 실행을 포함하여 웹 브라우저를 완전히 시뮬레이션하는 것입니다. 이 기능은 글을 쓰는 시점에는 사용할 수 없지만 우리가 적극적으로 작업하고 있는 부분이며 여러분이 이 글을 읽을 때쯤에는 사용할 수 있을지도 모릅니다. 이것은 백그라운드에서 실제로 Chrome 브라우저를 실행하는 [chromote 패키지](https://oreil.ly/xaHTf)를 사용하며, 인간이 텍스트를 입력하고 버튼을 클릭하는 것처럼 사이트와 상호 작용할 수 있는 추가 도구를 제공합니다. 자세한 내용은 [rvest 웹사이트](https://oreil.ly/YoxV7)를 확인하세요.

# 요약

이 장에서는 웹 페이지에서 데이터를 스크래핑하는 이유, 하지 말아야 할 이유, 그 방법에 대해 배웠습니다. 먼저 HTML의 기본 사항과 특정 요소를 참조하기 위해 CSS 선택자를 사용하는 것에 대해 배웠고, 그다음 rvest 패키지를 사용하여 HTML에서 데이터를 빼내어 R로 가져오는 방법에 대해 배웠습니다. 그리고 두 가지 사례 연구를 통해 웹 스크래핑을 시연했습니다. rvest 패키지 웹사이트에서 _스타워즈_ 영화에 대한 데이터를 스크래핑하는 더 간단한 시나리오와 IMDb에서 상위 250개 영화를 스크래핑하는 더 복잡한 시나리오입니다.

웹에서 데이터를 스크래핑하는 기술적 세부 사항은 특히 사이트를 다룰 때 복잡할 수 있습니다. 하지만 법적, 윤리적 고려 사항은 훨씬 더 복잡할 수 있습니다. 데이터를 스크래핑하기 전에 이 두 가지 모두에 대해 스스로 교육하는 것이 중요합니다.

이로써 데이터가 있는 곳(스프레드시트, 데이터베이스, JSON 파일, 웹사이트)에서 데이터를 가져와 R에서 깔끔한 형태(tidy form)로 만드는 기술을 배운 이 책의 가져오기(import) 파트가 끝났습니다. 이제 프로그래밍 언어로서 R을 최대한 활용하는 새로운 주제로 시선을 돌릴 때입니다.

<sup>[1](ch24.html#idm44771274112496-marker)</sup> 널리 사용되는 많은 API는 이미 이를 감싸고 있는(wrap) CRAN 패키지를 가지고 있으므로 먼저 약간의 조사를 통해 시작하세요!

<sup>[2](ch24.html#idm44771274043040-marker)</sup> 당연히 우리는 변호사가 아니며, 이것은 법적 조언이 아닙니다. 하지만 이 주제에 대해 많은 것을 읽은 후 우리가 제공할 수 있는 최선의 요약입니다.
<sup>[3](ch24.html#idm44771274018592-marker)</sup> OkCupid 연구에 관한 기사의 한 예가 [Wired](https://oreil.ly/rzd7z)에 게재되었습니다.

<sup>[4](ch24.html#idm44771273952560-marker)</sup> 다수의 태그(`<p>`와 `<li>` 포함)는 종료 태그를 요구하지 않지만, HTML의 구조를 보기가 조금 더 쉬워지므로 포함하는 것이 좋다고 생각합니다.

<sup>[5](ch24.html#idm44771273871552-marker)</sup> 이 클래스는 [xml2 패키지](https://oreil.ly/lQNBa)에서 온 것입니다. xml2는 rvest가 그 위에서 구축되는 저수준(low-level) 패키지입니다.

<sup>[6](ch24.html#idm44771273417888-marker)</sup> rvest는 <a href="https://rvest.tidyverse.org/reference/html_text.html" class="orm:hideurl"><code>html_text()</code></a>도 제공하지만 중첩된 HTML을 텍스트로 변환하는 작업을 더 잘 수행하는 <a href="https://rvest.tidyverse.org/reference/html_text.html" class="orm:hideurl"><code>html_text2()</code></a>를 거의 항상 사용해야 합니다.
