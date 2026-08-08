### 3

###### 첫 번째 살펴보기

knitr 패키지는 범용 리터러트 프로그래밍 엔진입니다. 이 패키지는 LATEX, HTML, Markdown과 같은 문서 형식(5장 참조)과 R, Python, awk, C++, 셸 스크립트 등과 같은 프로그래밍 언어(11장 참조)를 지원합니다. 시작하기 전에 R에 knitr를 설치해야 합니다. 그런 다음 간단한 예제를 통해 기본 개념을 소개하겠습니다. 마지막으로 순수한 R 스크립트에서 보고서를 빠르게 생성하는 방법을 보여드리겠습니다. 이는 동적 문서에 대해 알지 못하는 초보자에게 유용할 수 있습니다.

###### 3.1 설정

knitr는 R 패키지이므로 R에서 일반적인 방법으로 CRAN을 통해 설치할 수 있습니다.

install.packages("knitr", dependencies = TRUE)

여기서 dependencies = TRUE는 선택 사항이며, 반드시 필요하지는 않지만 유용한 기능으로 패키지를 향상시킬 수 있는 모든 패키지를 설치합니다. 개발 버전은 Github(https://github.com/yihui/knitr)에서 호스팅되며, 항상 최신 개발 버전을 확인할 수 있습니다. 최신 개발 버전은 안정적이지 않을 수 있지만 최신 버그 수정 및 새로운 기능을 포함합니다. knitr에 문제가 발생하면 가장 먼저 버전을 확인해야 합니다.

packageVersion("knitr") # if not the latest version, run update.packages()

조판 도구로 LATEX를 선택하는 경우 MiKTEX(Windows, http://miktex.org/), MacTEX(Mac OS, http://tug.org/mactex/) 또는 TEXLive(Linux, http://tug.org/texlive/)를 설치해야 할 수 있습니다. HTML 또는 Markdown으로 작업하는 경우에는

11

출력 결과가 웹 브라우저로 볼 수 있는 웹 페이지이므로 다른 것을 설치할 필요가 없습니다.

knitr가 설치되면 knit() 함수를 사용하여 소스 문서를 컴파일할 수 있습니다. 예를 들면 다음과 같습니다.

library(knitr) knit("your-file.Rnw")

*.Rnw 파일은 일반적으로 R 코드가 포함된 LATEX 문서입니다. 이는 다음 섹션과 더 많은 유형의 문서를 소개하는 5장에서 확인할 수 있습니다.

###### 3.2 간단한 예제

동적 문서의 구조를 설명하기 위해 각각 LATEX와 Markdown으로 작성된 두 가지 간단한 예제를 사용합니다. 지금은

LATEX나 Markdown의 구문에 대해서는 논의하지 않습니다(대신 5장 참조). 단순화를 위해 R의 기본 cars 데이터 세트를 사용하여 단순 선형 회귀 모델을 구축합니다. 자세한 문서를 보려면 R에서 ?cars를 입력하세요. 기본적으로 이 데이터 세트에는 속도와 거리라는 두 가지 변수가 있습니다.

str(cars)

##  data.frame : 50 obs. of 2 variables: ## $ speed: num 4 4 7 7 8 9 10 10 10 11 ... ## $ dist : num 2 10 4 22 16 10 18 26 34 17 ...

###### 3.2.1 LATEX 예제

그림 3.1은 LATEX에 포함된 R 코드의 전체 예제입니다. 파일 이름 확장자가 관례상 Rnw이므로 앞으로 이러한 종류의 문서를 Rnw 문서라고 부릅니다. 이를 minimal.Rnw 파일로 저장하고 앞서 설명한 대로 R에서 knit('minimal.Rnw')를 실행하면, knitr는 minimal.tex라는 LATEX 출력 문서를 생성합니다. LATEX에 익숙한 경우 pdflatex를 통해 이 문서를 PDF로 컴파일할 수 있습니다. 그림

- 3.2는 Rnw 문서에서 컴파일된 PDF 문서입니다.


여기서 핵심은 R 코드를 LATEX에 포함하는 방법입니다. Rnw 문서에서 < <> >=는 코드 청크의 시작을 나타내고 @는 코드 청크를 종료합니다(이 설명은 엄밀하지 않지만 이해하기 쉬운 경우가 많습니다).

|\documentclass{article} \begin{document} \title{간단한 예제} \author{Yihui Xie} \maketitle<br><br>단순 선형 회귀 모델을 사용하여 속도와 정지 거리 사이의 관계를 조사합니다: $Y = \beta_0 + \beta_1 x + \epsilon$.<br><br><<model, fig.width=4, fig.height=3, fig.align= center >>= par(mar = c(4, 4, 1, 1), mgp = c(2, 1, 0), cex = 0.8) plot(cars, pch = 20, col =  darkgray ) fit <- lm(dist ~ speed, data = cars) abline(fit, lwd = 2) @<br><br>단순 선형 회귀의 기울기는 다음과 같습니다. \Sexpr{coef(fit)[2]}. \end{document}|
|---|


- 그림 3.1: 간단한 Rnw 문서의 소스입니다. 그림 3.2의 출력을 참조하세요.


이 예제에서는 두 마커 사이에 산점도를 그리고 선형 모델을 적합하며 산점도에 회귀선을 추가하는 네 줄의 R 코드가 있습니다. \Sexpr{} 명령은 인라인 R 코드를 포함하는 데 사용됩니다(예: 이 예제의 coef(fit)[2]). < < 및 > >= 사이에 코드 청크에 대한 청크 옵션을 작성할 수 있습니다. 이 예제의 청크 옵션은 플롯 크기를 4 x 3인치(fig.width 및 fig.height)로 지정하고 플롯을 가운데 정렬(fig.align)하도록 지정했습니다.

이 간단한 예제에는 보고서의 기본적인 요소가 대부분 포함되어 있습니다.

- 1. 제목, 저자 및 날짜
- 2. 모델 설명
- 3. 데이터 및 계산
- 4. 그래픽
- 5. 수치 결과


모든 출력은 R에서 동적으로 생성됩니다. 데이터가 변경되더라도

|간단한 예제<br><br>Yihui Xie 2015년 4월 11일<br><br>단순 선형 회귀 모델을 사용하여 속도와 정지 거리 사이의 관계를 조사합니다: Y = β0 + β1x + .<br><br>par(mar = c(4, 4, 1, 1), mgp = c(2, 1, 0), cex = 0.8) plot(cars, pch = 20, col = "darkgray") fit <- lm(dist ~ speed, data = cars) abline(fit, lwd = 2)<br><br>5 10 15 20 25<br><br>020406080100<br><br>speed<br><br>dist<br><br>단순 선형 회귀의 기울기는 3.9324088입니다.|
|---|


- 그림 3.2: R 코드 청크, 플롯, 수치 출력(회귀 계수)이 포함된 LATEX의 간단한 예제입니다.


1

|--title: 간단한 예제<br><br>--단순 선형 회귀 모델을 사용하여 속도와 정지 거리 사이의 관계를 조사합니다: $Y = \beta_0 + \beta_1 x + \epsilon$.    {r fig.width=4, fig.height=3, fig.align= center } par(mar = c(4, 4, 1, 1), mgp = c(2, 1, 0), cex = 0.8) plot(cars, pch = 20, col =  darkgray ) fit <- lm(dist ~ speed, data = cars) abline(fit, lwd = 2)<br><br>단순 선형 회귀의 기울기는 다음과 같습니다.  r coef(fit)[2] .|
|---|


- 그림 3.3: 간단한 Rmd 문서의 소스입니다. 그림 3.4의 출력을 참조하세요.


보고서를 처음부터 다시 작성할 필요가 없으며, 데이터를 업데이트하고 보고서를 다시 컴파일하면 그에 따라 출력이 업데이트됩니다.

###### 3.2.2 Markdown 예제

LATEX는 많은 명령어 때문에 초보자에게 복잡하게 보일 수 있습니다. 이에 비해 Markdown(Gruber, 2004)은 더 단순한 형식입니다. 그림 3.3은 이전 예제와 동일한 분석을 수행하는 Markdown 예제입니다.

Markdown의 이상적인 출력은 그림 3.4(Mozilla Firefox)에 표시된 것처럼 HTML 웹 페이지입니다. 마찬가지로 Markdown 문서에서 R 코드의 구문을 볼 수 있습니다. {r}은 코드 청크를 열고, 청크를 종료하며, 인라인 R 코드는 r 안에 넣을 수 있습니다. 여기서 백틱을 사용합니다.

knitr에서 약간 더 긴 예제는 Markdown을 기반으로 하는 notebook이라는 데모입니다. 이는 Markdown의 잠재력뿐만 아니라 knitr로 웹 애플리케이션을 구축할 수 있는 가능성도 보여줍니다. 데모를 보려면 아래 코드를 실행하세요.

![image 2](Dynamic Documents with R and knitr 2nd_images/imageFile2.png)

- 그림 3.4: 그림 3.2와 동일한 분석을 수행하는 Markdown의 간단한 예제입니다. 단, 현재 출력은 PDF 대신 HTML입니다.


if (!require("shiny")) install.packages("shiny") demo("notebook", package = "knitr")

기본 웹 브라우저가 실행되어 웹 노트북을 보여줍니다. 소스 코드는 왼쪽 패널에 있고, 실시간 결과는 오른쪽 패널에 있습니다. 소스 코드를 자유롭게 실험해 보고 노트북을 다시 컴파일해 볼 수 있습니다.

###### 3.3 빠른 보고서 작성

사용자가 R에 대한 기본 지식만 있고 knitr에 대해서는 아무것도 모르거나, R 스크립트 외에 다른 것을 작성하고 싶지 않은 경우에도 stitch() 함수를 사용하여 이 R 스크립트에서 빠른 보고서를 생성할 수 있습니다.

stitch()의 기본 개념은 knitr가 기본 설정이 포함된 소스 문서의 템플릿을 제공하여 사용자가 이 템플릿에 R 스크립트(하나의 코드 청크)만 입력하면 된다는 것입니다. 그러면 knitr가 템플릿을 보고서로 컴파일합니다. 현재 LATEX, HTML 및 Markdown을 위한 내장 템플릿이 있습니다. 사용법은 다음과 같습니다.

library(knitr) stitch("your-script.R")

###### 3.4 R 코드 추출

리터러트 프로그래밍 문서의 경우 보고서로 컴파일(코드 실행)하거나 그 안의 프로그램 코드를 추출할 수 있습니다. 이를 각각 "위빙(weaving)"과 "탱글링(tangling)"이라고 합니다. 명백하게 knit() 함수는 위빙을 위한 것이며, 해당하는 탱글링 함수는 knitr의 purl()입니다. 예를 들면 다음과 같습니다.

library(knitr) purl("your-file.Rnw") purl("your-file.Rmd")

탱글링의 결과는 R 스크립트입니다. 위 예제에서 기본 출력은 your-file.R이며, 소스 문서의 모든 코드 청크로 구성됩니다.

지금까지 knitr의 명령줄 사용법을 소개했는데, 명령어를 반복해서 입력하는 것은 번거로운 경우가 많습니다. 다음 장에서는 좋은 편집기가 마우스 클릭 한 번이나 키보드 단축키로 소스 문서를 편집하고 컴파일하는 데 어떻게 도움이 되는지 보여줍니다.
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

그런 다음 청크 옵션을 `optipng = TRUE`로 설정하여 청크에 대해 최적화를 활성화하거나, 이 옵션에 문자열을 전달하여 OptiPNG에서 추가 명령줄 인자로 사용하도록 할 수 있습니다. 예를 들어, 가장 높은 수준의 최적화를 지정하려면 `optipng = '-o7'`을 사용할 수 있습니다. 가능한 모든 인자는 OptiPNG의 문서를 참조하시기 바랍니다.

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
