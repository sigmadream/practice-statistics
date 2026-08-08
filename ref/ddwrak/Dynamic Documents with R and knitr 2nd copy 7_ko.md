### 7

###### Graphics

그래픽은 보고서의 중요한 부분이며, knitr에서는 그래픽 출력이 자연스럽고 유연하게 이루어지도록 많은 노력이 기울여졌습니다. 예를 들어, knitr는 R 콘솔의 동작을 모방하려고 시도하며, grid 그래픽(Murrell, 2011)은 동일한 코드가 R 콘솔에서 플롯을 생성할 수 있는 한 명시적으로 출력할 필요가 없습니다(하지만 반복문 내부 등 일부 경우에는 R 콘솔에서도 그렇게 해야 하므로 명시적으로 출력해야 합니다). 아래는 R 콘솔과 knitr 모두에서 플롯을 생성하는 코드 청크입니다(그림 7.1 참조).

library(ggplot2) p <- qplot(carat, price, data = diamonds) + geom_hex() p # no need to print(p)

count

15000

5000

4000

price

10000

3000

2000

5000

1000

0

0 1 2 3 4 5

carat

- FIGURE 7.1: ggplot2에서 생성된 플롯으로 명시적으로 출력할 필요가 없습니다(비교를 위해 Sweave에서는 print(p)를 해야 하며 이는 매우 혼란스럽습니다. 16.1절 참조).


59

###### 7.1 Graphical Devices

knitr에서는 청크 옵션 dev를 통해 20개 이상의 그래픽 장치를 지원합니다. 예를 들어, dev = 'png'는 기본 R의 grDevices 패키지에 있는 png() 장치를 사용하고, dev = 'CairoJPEG'는 추가 패키지 Cairo에 있는 CairoJPEG() 장치를 사용합니다(물론 먼저 설치되어 있어야 합니다). dev에 사용할 수 있는 값은 다음과 같습니다.

[1] "bmp" "postscript" "pdf" [4] "png" "svg" "jpeg" [7] "pictex" "tiff" "win.metafile"

[10] "cairo_pdf" "cairo_ps" "quartz_pdf" [13] "quartz_png" "quartz_jpeg" "quartz_tiff" [16] "quartz_gif" "quartz_psd" "quartz_bmp" [19] "CairoJPEG" "CairoPNG" "CairoPS" [22] "CairoPDF" "CairoSVG" "CairoTIFF" [25] "Cairo_pdf" "Cairo_png" "Cairo_ps" [28] "Cairo_svg" "tikz"

###### 7.1.1 Custom Device

이러한 장치 중 만족스러운 것이 없다면, 사용자가 정의한 장치 함수의 이름을 제공할 수 있습니다. 이 함수는 사용하기 전에 다음과 같은 형태로 정의되어야 합니다.

custom_dev <- function(file, width, height, ...) { # open the device here, e.g., pdf(file, width, height, # ...)

}

그런 다음 청크 옵션을 dev = 'custom_dev'로 설정할 수 있습니다(장치 이름은 위에서 정의한 함수 이름입니다).

###### 7.1.2 Choose a Device

Rnw 문서의 기본 장치는 PDF(grDevices의 pdf())이고, Rmd/Rhtml/Rrst 문서의 경우 PNG(grDevices의 png())입니다. 왜냐하면 일반적으로 PDF는 HTML 출력에서 작동하지 않기 때문입니다. Cairo 시리즈 장치는 PNG나 JPEG와 같은 고품질 래스터 이미지를 원할 때 유용할 수 있으며, 파일 크기는 grDevices의 png() 또는 jpeg()로 생성된 플롯 파일의 크기보다 종종 더 큽니다. CairoXXX 장치는 Cairo 패키지에서, Cairo_xxx 장치는 cairoDevice 패키지에서 제공됩니다. quartz_xxx 장치는 Mac OS 전용입니다.

HTML 출력의 경우 보통 래스터 이미지를 사용하지만, 오늘날 대부분의 웹 브라우저는 벡터 그래픽 형식인 SVG도 지원합니다. 래스터 그래픽에 비해 벡터 그래픽이 가지는 명백한 장점 중 하나는 고품질이라는 점입니다. 예를 들어, SVG 이미지는 품질 저하 없이 확대하거나 축소할 수 있습니다. dev = 'svg'를 사용하여 마크다운이나 HTML용 SVG 플롯을 생성할 수 있습니다. 하지만 고품질을 얻기 위해서는 파일 크기가 커지는 것을 감수해야 합니다(이는 R 플롯 전반에 적용됩니다. 다만 SVG 플롯이 항상 래스터 이미지보다 커야 하는 것은 아닙니다).

모든 장치를 모든 출력 형식에 사용할 수 있는 것은 아닙니다. 앞서 언급했듯이 현재 웹 브라우저에서는 PDF가 자동으로 작동하지 않으며, 마찬가지로 win.metafile(Windows Metafile) 장치는 LATEX와 함께 작동하지 않습니다.

###### 7.1.3 Device Size

청크 옵션 fig.width와 fig.height는 그래픽 장치에 전달되어 플롯의 너비와 높이를 설정하며(단위는 인치, 두 옵션 모두 기본값은 7), 출력에서 다른 옵션을 사용하여 플롯의 크기를 재조정할 수 있습니다(7.4절). png()와 같은 비트맵 장치의 경우 R의 기본 단위는 인치가 아니라 픽셀이지만, knitr는 모든 장치에 대해 단위를 통일했습니다. 픽셀을 인치로 변환하기 위해 청크 옵션 dpi(인치당 도트 수)가 사용됩니다. 기본값은 72이며, 이는 1인치가 72픽셀과 같음을 의미하므로 fig.width = 7은 PNG 이미지의 경우 504픽셀을 의미합니다.

###### 7.1.4 More Device Options

플롯 파일의 크기를 설정하는 옵션 외에도 dev.args 옵션을 리스트 형태로 사용하여 더 많은 인수를 장치에 전달할 수 있습니다. 이는 특정 그래픽 장치의 사용 가능한 인수에 따라 결정됩니다. 예를 들어, png 장치에 dev.args = list(pointsize = 10)을 전달하여 포인트 크기를 변경하거나, pdf 장치에 dev.args = list(family = 'Bookman')을 전달하여 글꼴 패밀리를 변경할 수 있습니다. 그림 7.2는 Bookman 글꼴 패밀리를 사용하여 생성되었습니다. 비록 아래 코드에서는 해당 설정을 볼 수 없지만(소스 문서에 포함되어 있습니다), 다음과 같습니다.

plot(rep(0:1, 10), pch = 1:20, col = 2, xlab = "xlab font",

ylab = "ylab font")

mtext("Bookman in the PDF device", side = 3, cex = 1.2) text(6, 0.5, "Aa Bb Cc\nRr Ss Tt\nXx Yy Zz", cex = 1.5) text(16, 0.5, "g", cex = 6, col = 3)

Bookman in the PDF device

0.00.20.40.60.81.0

| | |
|---|---|
| | |


g

Aa Bb Cc Rr Ss Tt Xx Yy Zz

ylabfont

5 10 15 20

xlab font

- FIGURE 7.2: Bookman 글꼴 패밀리를 사용한 플롯: 이 플롯을 위한 청크 옵션은 dev.args = list(family = 'Bookman')입니다(dev = 'pdf'와 함께 사용됨).

그림 7.2의 글꼴 패밀리를 pdf 장치의 기본 글꼴 패밀리(Helvetica)를 사용한 그림 7.1과 비교해 보면 두 글꼴 스타일이 확연히 다르다는 것을 알 수 있습니다.

###### 7.1.5 Encoding

pdf 장치의 경우 pdf.options()를 통해 전역으로 옵션을 설정할 수 있습니다. 즉, 이 함수에서 설정한 옵션은 현재 R 세션의 모든 pdf 장치에 영향을 미칩니다. 이 함수의 중요한 응용 중 하나는 플롯에 다바이트 문자가 포함된 경우 pdf 장치의 인코딩을 설정하는 것입니다. 예를 들어, 유로 기호나 어큐트 악센트가 있는 문자 A를 작성하려면 인코딩을 CP1250(라틴 문자를 사용하는 중부 및 동부 유럽 언어의 텍스트를 나타내기 위해)으로 설정해야 할 수 있습니다(http://en.wikipedia.org/wiki/Windows-1250 참조).

pdf.options(encoding = "CP1250")

사용 가능한 전체 인코딩 목록을 보려면 다음을 참조하십시오.

list.files(system.file("enc", package = "grDevices"))

## [1] "AdobeStd.enc" "AdobeSym.enc" "CP1250.enc" ## [4] "CP1251.enc" "CP1253.enc" "CP1257.enc" ## [7] "Cyrillic.enc" "Greek.enc" "ISOLatin1.enc" ## [10] "ISOLatin2.enc" "ISOLatin7.enc" "ISOLatin9.enc" ## [13] "KOI8-R.enc" "KOI8-U.enc" "MacRoman.enc" ## [16] "PDFDoc.enc" "TeXtext.enc" "WinAnsi.enc"

그림 7.3은 Windows-1250 코드 페이지의 문자 표를 보여주며, 이는 아래 코드에서 생성되었습니다.

x <- c("\U20AC", "\U201A", "\U201E", "\U2026", "\U2020",

"\U2021", "\U2030", "\U0160", "\U2039", "\U015A", "\U0164", "\U017D", "\U0179", "\U2018", "\U2019", "\U201C", "\U201D", "\U2022", "\U2013", "\U2014", "\U2122", "\U0161", "\U203A", "\U015B", "\U0165", "\U017E", "\U017A", "\U02C7", "\U02D8", "\U0141",

- "\U00A4", "\U0104", "\U00A6", "\U00A7", "\U00A8", "\U00A9", "\U015E", "\U00AB", "\U00AC", "\U00AE", "\U017B", "\U00B0", "\U00B1", "\U02DB", "\U0142",
- "\U00B4", "\U00B5", "\U00B6", "\U00B7", "\U00B8", "\U0105", "\U015F", "\U00BB", "\U013D", "\U02DD", "\U013E", "\U017C", "\U0154", "\U00C1", "\U00C2", "\U0102", "\U00C4", "\U0139", "\U0106", "\U00C7", "\U010C")


plot(c(1, 11), c(1, 6), type = "n", ann = F, axes = F) box() text(rep(1:11, 6), rep(1:6, each = 11), x)

적절한 인코딩을 설정하지 않으면 아래와 같은 경고가 나타나고 문자가 "..."로 대체될 수 있습니다(아래의 문자 \U20AC는 유로 기호 €입니다).

plot(1, main = "\U20AC")

## Warning: conversion failure on ’€’ in ’mbcsToSbcs’: dot substituted for <e2> ## Warning: conversion failure on ’€’ in ’mbcsToSbcs’: dot substituted for <82> ## Warning: conversion failure on ’€’ in ’mbcsToSbcs’: dot substituted for <ac>

|‚ „ … † ‡ ‰ Š ‹<br><br>Ž ‘ ’ “ ” • – — ™ š<br><br>› ž ˇ ˘ Ł ¤ ¦<br><br>§ ¨ © « ¬ ® ° ± ˛<br><br>ł ´<br><br>µ ¶ · ¸ » ˝<br><br>Á Â Ä Ç|
|---|


- FIGURE 7.3: Windows-1250 코드 페이지 표: 이 표는 유로 기호나 어큐트 악센트가 있는 문자 A와 같이 코드 페이지의 일부 문자만 보여줍니다.


###### 7.1.6 The Dingbats Font

pdf()의 문서에 따르면, useDingbats 인수는 작은 원이 포함된 PDF의 파일 크기를 줄일 수 있습니다. RStudio에서 knitr를 사용하는 경우 이 옵션은 기본적으로 비활성화되어 있습니다. 산점도 크기가 큰 경우 소스 문서에 pdf.options(useDingbats = TRUE)를 입력하여 이를 활성화하면 PDF 플롯 파일이 작아집니다. 다른 편집기를 사용하는 사용자는 이 옵션을 FALSE로 설정하려는 경우가 아니라면 굳이 신경 쓸 필요가 없습니다.

###### 7.2 Plot Recording

코드 청크의 모든 플롯은 먼저 evaluate 패키지에 의해 R 객체로 기록된 다음 그래픽 장치 내에서 다시 실행되어(재생되어) 플롯 파일을 생성합니다. 플롯의 소스는 두 가지입니다. 첫째, plot.new() 또는 grid.newpage()가 호출될 때마다(이는 R 기본 및 grid 플롯이 생성되기 전에 발생합니다), evaluate는 현재 플롯이 존재한다면 스냅샷을 저장하려고 시도합니다. 둘째, 완전한 표현식이 평가된 후에도 스냅샷이 저장됩니다. 기술적인 세부 사항은 ?setHook 및 ?recordPlot을 참조하십시오(둘 다 기본 R의 함수입니다).

기록 속도를 높이기 위해 빈 그래픽 장치인 pdf(file = NULL)가 사용됩니다. 아래는 플롯이 어떻게 기록되는지 보여주는 간단한 예입니다.

pdf(file = NULL) # open a pdf device to record plots ## enable recording for the current 장치 dev.control("enable") plot(rnorm(100)) # draw a plot x <- recordPlot() dev.off()

## pdf ## 2

str(x, 1) # an R object of class recordedplot

## List of 3 ## $ :Dotted pair list of 8 ## $ : raw [1:35992] 00 00 00 00 ... ## $ : NULL ## - attr(*, "pid")= int 31856 ## - attr(*, "class")= chr "recordedplot"

print(x) # redraw the plot object

빈 장치는 대부분의 경우에 잘 작동합니다. 잘 작동하지 않을 수 있는 한 가지 경우는 플롯에 다바이트 문자가 포함되어 있고 글꼴 처리가 복잡한 경우입니다(Murrell and Ripley, 2006). options()에서 device 옵션을 설정하여 기록 장치를 변경할 수 있습니다. 예를 들어, cairo_pdf() 장치는 비표준 글꼴 처리에 더 유리하므로 대신 이 장치를 지정하여 그래픽을 기록할 수 있습니다.

options(device = function(width = 7, height = 7, ...) {

cairo_pdf(tempfile(), width, height, ...) })

그런 다음 청크 옵션을 dev = 'cairo_pdf'로 설정하여 플롯을 PDF 파일로 저장할 수도 있습니다.

evaluate 패키지는 표현식 단위로 플롯을 기록합니다. 즉, 소스 코드는 개별적인 완전한 표현식으로 분할되며 evaluate는 단일 표현식이 평가된 후 스냅샷에서 플롯의 변경 가능성을 검사합니다. R 표현식이 반드시 코드 한 줄과 같지는 않다는 점에 유의하십시오. 예를 들어, 아래 코드는 3개의 표현식으로 구성되며 그 중 2개는 플롯 그리기와 관련이 있습니다(첫 번째 줄은

10

10

mass → energy E = mc2

8

8

6

6

4

4

2

2

2 4 6 8 10

2 4 6 8 10
- - banana 또는 순서가 있는 목록:


1. items

1. will

1. be

1. ordered

- - 중첩된
- - 항목 # 더 많은 섹션 ## Hi hi hi ## Hello hello hello ## Howdy howdy howdy # 이제, 약간의 R 코드    {r linear-model} fit = lm(dist ~ speed, data = cars) b = coef(fit) # 계수 summary(fit)


코드는 모든 출력 형식에서 구문 강조 표시가 적용됩니다. # 그리고 약간의 그림    {r lm-vis, fig.cap= Regression diagnostics } par(mfrow = c(2, 2), pch = 20, mar = c(4, 4, 2, .1),

bg =  white ) plot(fit)

# 약간의 수학 회귀 방정식은 $Y= r b[1] + r b[2] x$이며, 모델은 다음과 같습니다.

$$ Y = \beta_0 + \beta_1 x + \epsilon$$ # Pandoc 확장: 정의 목록 프로그래머 : 커피를 코드로 바꾸는 사람입니다. LaTeX : 몇 개의 백슬래시를 사용하는 간단한 언어입니다. # Pandoc 확장: 예시 몇 가지 예시가 있습니다. (@) 0.3 + 0.4 - 0.7이 무엇인지 생각해 보십시오. 0입니다. 간단합니다. (@weird) 이제 0.3 - 0.7 + 0.4가 무엇인지 생각해 보십시오. 여전히 0일까요? 사람들은 종종 (@weird)에 놀라곤 합니다. # Pandoc 확장: 표 여기에 표가 있습니다. Table: 간단한 표 문법 시연.    {r echo=FALSE} knitr::kable(head(iris))

# Pandoc 확장: 각주 각주[^1]를 작성할 수도 있습니다. [^1]: 안녕하세요, 각주입니다. 또는 인라인 각주^[여기에 보이는 것처럼]를 작성할 수도 있습니다. # Pandoc 확장: 인용 R Markdown 파일을 R[@R-base]의 knitr[@R-knitr]를 통해 Markdown으로 컴파일합니다. @R-knitr에 대한 자세한 내용은 <http://yihui.name/knitr>를 참고하십시오.

![image 20](Dynamic Documents with R and knitr 2nd_images/imageFile20.png)

- 그림 14.1: RStudio 창에서 R Markdown v2의 HTML 출력 문서 미리보기.


# 참고 문헌    {r include=FALSE} knitr::write_bib(c( base ,  knitr ),  Rmd-v2.bib )

kable()이나 write_bib()가 어떻게 작동하는지 확실하지 않다면 6.3절 및 12.4.1절을 다시 확인해 보시기 바랍니다.

그림 14.1은 RStudio에서 이 예시를 렌더링한 후의 HTML 출력 문서 미리보기입니다. 문서의 제목, 작성자, 날짜 및 처음 몇 개의 섹션이 표시됩니다. 이는 rmarkdown의 기본 Twitter Bootstrap 스타일입니다. 그림 14.2는 마지막 몇 개의 섹션에 대한 미리보기입니다. 각주와 인용이 HTML의 기본 요소는 아니지만(LATEX 사용자에게는 자연스러울 수 있음) Pandoc은 어쨌든 HTML에서 이들을 생성해 냈습니다.

HTML 출력에 대해 조정할 수 있는 옵션이 다양하게 있습니다. 전체 목록은 도움말 페이지 ?rmarkdown::html_document를 참고하시기 바랍니다.

|![image 21](Dynamic Documents with R and knitr 2nd_images/imageFile21.png)|
|---|


###### 그림 14.2: 표, 각주, 인용 미리보기: 표는 kable()로 생성되었고, 참고 문헌 데이터베이스는 knitr의 write_bib()로 생성되었습니다.

예를 들어, YAML에서 theme 필드를 사용하여 CSS 테마를 변경하고, toc 필드를 사용하여 목차를 추가하며, number_sections 필드를 사용하여 섹션 제목에 번호를 매길 수 있습니다(그림 14.3).

--output:

html_document: fig_caption: yes number_sections: yes theme: readable toc: yes

---

현재 rmarkdown에서 사용할 수 있는 CSS 테마는 다음과 같습니다(http://bootswatch.com에서 미리보기를 확인할 수 있습니다).

## [1] "default" "cerulean" "journal" "flatly" ## [5] "readable" "spacelab" "united" "cosmo"

출력의 모양을 추가로 조정해야 하는 경우, css 필드를 사용하여 자체 CSS 파일을 적용할 수 있습니다. 예:

--output:

html_document: css: my_own.css

---

rmarkdown의 테마(구문 강조 표시 테마 포함)를 원하지 않고 고유한 CSS만 사용하려는 경우, theme 및 highlight를 null로 지정하여 완전히 제거할 수 있습니다.

--output:

html_document: css: my_own.css theme: null highlight: null

---

HTML 페이지에는 종종 CSS, JavaScript, 이미지 파일 등 외부 종속성이 있으므로 다른 사람과 HTML 파일을 공유할 때 불편할 수 있습니다. HTML 파일을 보낼 때 이러한 종속성도 포함되어 있는지 확인해야 하기 때문입니다.

|![image 22](Dynamic Documents with R and knitr 2nd_images/imageFile22.png)|
|---|


###### 그림 14.3: 목차 및 번호가 매겨진 섹션이 포함된 readable 테마 미리보기(그림 14.1과 글꼴이 다른 것을 확인할 수 있습니다).

Pandoc에는 모든 외부 종속성을 HTML 파일에 포함시켜 HTML 파일을 독립적으로 만드는 옵션이 있습니다. 예를 들어, JavaScript 파일은 HTML 파일로 읽어오고 이미지는 base64로 인코딩됩니다. 독립적인 HTML 파일은 PDF 파일처럼 공유할 수 있으며, 필요한 모든 것이 단일 파일에 포함되어 있습니다. rmarkdown에서는 self_contained 옵션으로 이를 제어합니다. rmarkdown으로 렌더링할 Rmd 파일이 여러 개인 경우 독립 실행 모드를 끄는 것이 좋습니다. 그렇지 않으면 일부 외부 종속성이 모든 개별 HTML 출력 파일에 포함되어 중복이 많이 발생할 수 있습니다. 독립 실행 모드가 꺼져 있을 때는 lib_dir 옵션을 통해 지정된 공통 디렉터리에 공유 종속성을 넣을 수 있습니다. 예:

--output:

html_document: self_contained: no lib_dir: assets

---

때로는 HTML 헤더, 본문 앞 또는 문서의 본문 뒤에 추가 콘텐츠를 포함하고 싶을 수 있습니다. 이러한 경우 rmarkdown에는 추가 콘텐츠의 파일 이름을 지정할 수 있는 includes 옵션이 있습니다. HTML 출력에서 JavaScript 라이브러리 D3(http://d3js.org)를 사용한다고 가정해 보겠습니다. 이 경우 doc_header.html 파일에 다음 내용을 작성할 수 있습니다.

<script src="http://d3js.org/d3.v3.min.js" charset="utf-8"> </script>

본문 앞뒤에 삽입할 콘텐츠인 doc_before.html 및 doc_after.html 두 파일도 지정할 수 있습니다. 예를 들어 doc_before.html에는 탐색 메뉴를, doc_after.html에는 저작권 정보를 작성할 수 있습니다. 이 세 파일은 다음과 같이 HTML 출력 파일에 포함될 수 있습니다.

--output:

html_document:

includes: in_header: doc_header.html before_body: doc_before.html after_body: doc_after.html

모든 출력 형식에 대해 Pandoc은 출력 파일을 생성하기 위한 템플릿을 필요로 합니다. 템플릿에는 사용할 수 있는 몇 가지 Pandoc 변수가 있으며, 이 변수들을 사용하여 자신만의 템플릿을 정의할 수 있습니다. 예를 들어, 다음과 같은 최소 HTML 템플릿이 가능합니다.

<html> <head>

<title>$title$</title> </head> <body> $body$ </body>

</html>

이 템플릿에서는 $title$ 및 $body$ 두 변수만 사용했습니다. 첫 번째 변수는 YAML 메타데이터의 title 필드에 지정된 문서 제목을 포함합니다. 두 번째 변수는 Markdown 문서가 HTML로 변환된 후의 본문입니다. rmarkdown 소스 패키지(https://github.com/rstudio/rmarkdown) 또는 Pandoc의 기본 템플릿(https://github.com/jgm/pandoc-templates)에서 사용 가능한 다른 변수들을 확인할 수 있습니다.

사용자 지정 템플릿을 사용하려면 YAML의 template 필드를 사용할 수 있습니다. 예:

--output:

html_document: template: my_template.html

---

마지막으로 pandoc_args 필드를 통해 Pandoc에 전달할 명령줄 인수를 사용자 정의할 수 있습니다. 사실, html_document()의 R 인수는 결과적으로 Pandoc 인수로 변환됩니다. 예를 들어, R 인수 self_contained = TRUE(또는 YAML의 self_contained: yes)는 Pandoc 인수 --self-contained와 동일하며, 다음 YAML과도 동일합니다.

--output:

html_document: pandoc_args: "--self-contained"

지금까지 Pandoc의 Markdown 측면에서 출력을 사용자 정의할 수 있는 대부분의 방법을 다루었습니다. YAML에서 knitr 청크 옵션을 사용자 정의하는 것도 가능합니다. 현재 YAML에서 설정할 수 있는 청크 옵션은 4가지가 있습니다.

fig_width, fig_height: 그림의 기본 크기

fig_retina: Retina 디스플레이의 비율 조정. rmarkdown의 기본값은 2입니다. 즉, 크기가 m × n인 그림의 실제 크기는 2m × 2n이지만 출력에서는 실제 크기의 절반으로 축소됩니다(이를 통해 Retina 디스플레이에서 이미지 품질을 향상시킬 수 있습니다).

fig_caption: 그림 캡션을 렌더링하고 표시할지 여부(출력 형식이 LATEX일 때 기본적으로 \caption{}이 포함된 figure 환경을 의미함). FALSE일 경우 캡션이 보이지 않는 <img> 태그의 alt 속성에 배치되므로 HTML 출력에서 그림 캡션을 볼 수 없습니다.

fig_retina 옵션은 이미지 품질을 높이는 대신 이미지 파일 크기를 더 크게 만듭니다. fig_retina = TRUE와 FALSE를 각각 시도해 보고 기기에서 차이점이 눈에 띄는지 확인해 볼 수 있습니다.

###### 14.3.2 LATEX/PDF 문서

HTML 문서 형식에 익숙해지면 다른 출력 형식도 쉽게 익힐 수 있습니다. 많은 옵션이 여러 형식에 공통적으로 사용되기 때문입니다. 예를 들어 pdf_document()에서도 fig_width, fig_height, toc, number_sections 및 highlight와 같은 옵션을 사용할 수 있습니다. 이 절에서는 PDF 문서 출력에 관련된 옵션에만 중점을 둡니다.

그림 14.4는 이전 절에서 사용한 것과 동일한 예시의 PDF 출력 페이지 미리보기입니다. 그림 14.2와 크게 다르지 않아 보입니다. 동일한 R Markdown 문서의 경우 섹션 제목, 표, 각주, 인용 등 HTML 출력에서 작동했던 모든 요소가 LATEX/PDF에서도 여전히 작동합니다.

마찬가지로 HTML 출력에서 했던 것처럼 목차를 추가하고 섹션에 번호를 매길 수 있습니다(그림 14.5).

--output:

pdf_document: number_sections: yes toc: yes

|Pandoc extension: tables<br><br>A table here.<br><br>Table 1: Demonstration of simple table syntax.<br><br>Sepal.Length Sepal.Width Petal.Length Petal.Width Species<br><br>5.1 3.5 1.4 0.2 setosa 4.9 3.0 1.4 0.2 setosa<br><br>4.7 3.2 1.3 0.2 setosa<br><br>4.6 3.1 1.5 0.2 setosa<br>5.0 3.6 1.4 0.2 setosa<br><br><br>5.4 3.9 1.7 0.4 setosa<br><br><br>Pandoc extension: footnotes<br><br>We can also write footnotes1. Or write some inline footnotes2.<br><br>Pandoc extension: citations<br><br>We compile the R Markdown ﬁle to Markdown through knitr (Xie 2014) in R (R Core Team 2014). For more about Xie (2014), see http://yihui.name/knitr.<br><br>References<br><br>R Core Team. 2014. R: A Language and Environment for Statistical Computing. Vienna, Austria: R Foundation for Statistical Computing. http://www.R-project. org/.<br><br>Xie, Yihui. 2014. Knitr: A General-Purpose Package for Dynamic Report Generation in R. http://yihui.name/knitr/.<br><br>1hi, I’m a footnote 2as you can see here|
|---|


###### 그림 14.4: R Markdown v2 예시의 PDF 출력 문서 4페이지 미리보기.

|R Markdown v2 Demo<br><br>Li Lei Han Meimei<br><br>2015/01/01<br><br>Contents<br><br>1 Start with a cool section 2<br>2 Followed by another section 2<br>3 More sections 2<br><br>3.1 Hi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2<br>3.2 Hello . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2<br>3.3 Howdy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2<br><br><br>4 Okay, some R code 3<br>5 And some pictures 3<br>6 A little bit math 4<br>7 Pandoc extension: deﬁnition lists 4<br>8 Pandoc extension: examples 4<br>9 Pandoc extension: tables 5<br>10 Pandoc extension: footnotes 5<br>11 Pandoc extension: citations 5<br><br><br>References 5|
|---|


1

- 그림 14.5: 목차와 번호가 매겨진 섹션이 포함된 PDF 출력 문서 미리보기.


Pandoc에는 YAML 메타데이터에서 사용할 수 있는 몇 가지 LATEX 전용 옵션이 있으며, Pandoc 웹사이트에서 전체 문서를 찾을 수 있습니다. 여기서는 그 중 몇 가지만 나열합니다.

fontsize: 문서의 글꼴 크기(예: 10pt, 11pt, 12pt)
documentclass: 문서 클래스(예: article, book, report)
classoption: 문서 클래스에 대한 옵션(예: a4paper, twocolumn)
geometry: geometry 패키지에 대한 옵션(예: tmargin=2cm, bmargin=2cm, lmargin=3cm, rmargin=3cm)

참고: 이는 YAML의 최상위 옵션이므로 pdf_document 필드 아래에 두어서는 안 됩니다.

기본 LATEX 엔진은 pdflatex이며 pdf_document()의 latex_engine 옵션을 통해 변경할 수 있습니다. 현재 사용 가능한 엔진은 pdflatex, xelatex, lualatex입니다. keep_tex 옵션을 통해 중간 LATEX 출력 파일을 보존할 수도 있으며, 이는 디버깅 및 기타 목적에 유용하게 사용할 수 있습니다.

다음은 book 클래스, 11pt 글꼴 크기, 두 열 레이아웃, 사용자 지정 여백 설정, XeLATEX 엔진을 사용하고 LATEX 파일도 보존하는 문서에 대한 YAML 메타데이터의 예입니다.

--documentclass: book classoption: twocolumn fontsize: 11pt geometry:

- - tmargin=2cm
- - bmargin=2cm
- - lmargin=3cm
- - rmargin=3cm

output:

pdf_document: latex_engine: xelatex keep_tex: yes

- ---


이전 절에서 includes 및 template 옵션을 소개했는데, 이는 LATEX 출력에 더 유용할 수 있습니다. LATEX 사용자가 프리앰블(preamble)에서 특정 LATEX 패키지를 사용하여 출력을 사용자 정의하는 것이 매우 일반적이기 때문입니다. 이러한 콘텐츠를 외부 파일에 넣고 includes 옵션 아래의 in_header 옵션을 통해 프리앰블에 포함시킬 수 있습니다. 기본 LATEX 템플릿에 만족하지 않는 경우 직접 작성할 수 있습니다. 실제로 작성하기 전에 Pandoc 문서를 주의 깊게 확인하여 YAML 옵션을 통해 원하는 결과를 얻을 수 있는지 확인해 보시기 바랍니다. 새로운 LATEX 템플릿을 작성하는 것은 비교적 쉽지만, Pandoc의 향후 변경 가능성을 인지하고 있어야 하므로 나중에 템플릿을 유지 관리하는 것이 간단하지 않을 수 있습니다.

###### 14.3.3 Word 문서

Word 문서에 대해 사용자 정의할 수 있는 옵션은 많지 않습니다. 그래도 그림 크기와 구문 강조 표시 테마 등을 설정할 수는 있습니다. 그림 14.6은 Microsoft Word 2013에서 해당 예시의 Word 출력을 보여줍니다.

Word 문서에서 가장 중요하고 유용한 기능은 템플릿일 것입니다. 다른 문서 형식의 경우 일반 텍스트 템플릿을 제공할 수 있지만, Word 문서는 비교적 복잡한 바이너리 파일이므로 쉽게 텍스트 템플릿을 적용할 수 없습니다. 하지만 Pandoc에서는 Word 문서를 스타일 템플릿 역할을 하는 "참조 문서"로 제공할 수 있습니다. 이 참조 문서는 Pandoc의 Word 출력 문서 중 하나를 기반으로 해야 하며, 해당 문서에서 다양한 요소의 스타일을 업데이트합니다. 문서에 정의된 스타일만 사용되며 콘텐츠는 대부분 무시된다는 점에 유의하십시오.

Word 문서에서 스타일을 정의하는 방법을 보여주기 위해 https://vimeo.com/110804387에 짧은 동영상을 준비했습니다. 그림 14.7 및 14.8도 확인해 보실 수 있습니다. 기본 단계는 다음과 같습니다.

- 1. Pandoc을 사용하여 임의의 Word 문서를 만듭니다. 예: YAML 메타데이터의 output 옵션으로 word_document를 사용합니다.
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

#' 여기에 방법을 소개하고 그 다음에 R 코드를 작성합니다:
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
title = "Waiting time: Old Faithful geyser." alt = "Waiting time: Old Faithful geyser." />

###### 7.8 Figure Path

그래픽 장치에 대해 소개했지만, 플롯이 실제로 어떻게 파일로 저장되는지는 설명하지 않았습니다. 각 플롯은 파일로 저장되며 파일 형식은 그래픽 장치에 따라 다릅니다. 파일명은 세 가지 청크 옵션(청크 라벨, fig.path, fig.ext)에 의해 결정됩니다. fig.path 옵션은 그림의 경로(기본적으로 상대 디렉토리인 figure/)를 지정하고, fig.ext는 플롯 파일의 파일명 확장자(기본적으로 dev 옵션에서 자동으로 파생됨, 예: Cairo_pdf 장치에 해당하는 확장자는 pdf)를 지정합니다. 엄밀히 말해 fig.path는 경로 접두사입니다. 예를 들어 fig.path = 'figure/mcmc-'는 모든 플롯 파일이 figure/ 디렉토리 아래에서 mcmc-라는 접두사를 가지게 합니다.

청크 내의 모든 플롯 파일은 foo-1, foo-2, ..., foo-n과 같이 순차적으로 이름이 지정됩니다. 여기서 foo는 청크 라벨이고 n은 청크 내 플롯의 총 개수입니다. 청크에 플롯이 하나만 있더라도 파일명에는 여전히 접미사 -1이 붙습니다.

fig.path에 존재하지 않는 디렉토리가 포함된 경우 knitr는 자동으로 디렉토리를 생성하려고 시도합니다. LATEX 출력의 경우 그림 경로와 파일명에는 영숫자, 하이픈(-), 밑줄(_)만 허용되며, 다른 모든 문자는 밑줄로 대체됩니다. 이는 LATEX가 공백이나 점과 같은 문자를 처리하는 데 문제가 있을 수 있기 때문입니다.

대부분의 경우 fig.ext를 지정할 필요는 없지만, 그래픽을 저장하기 위해 사용자 정의 장치를 사용하는 경우 knitr는 적절한 파일명 확장자를 알 수 없으므로 문자열로 이 옵션을 명시적으로 설정해야 합니다.

5.1절에서 청크 라벨의 고유성을 강조했는데, 이것이 고유해야 하는 이유 중 하나입니다. 즉 청크 라벨은 플롯의 파일명에 사용되므로, 만약 두 청크가 같은 라벨을 공유한다면 뒤의 청크가 앞의 청크에서 생성된 플롯을 덮어쓰게 됩니다. 다음 장에서 다룰 캐시 파일의 경우에도 마찬가지입니다.
