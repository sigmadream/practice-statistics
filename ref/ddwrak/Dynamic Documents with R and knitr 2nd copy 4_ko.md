### 4

###### 편집기 (Editors)

knitr를 위한 문서는 일반 텍스트 파일이므로 어떠한 텍스트 편집기로도 작성할 수 있습니다. 예를 들어, 윈도우의 메모장(Notepad)이나 리눅스의 Gedit과 같은 가벼운 편집기들도 작동합니다. 특별한 텍스트 편집기가 필요한 주요 이유는 다음과 같습니다.

- 1. R 코드 청크를 더 쉽게 입력하기 위함입니다. 예를 들어, 매번 문자를 입력하는 대신 키보드 단축키를 사용하여 < <> >= 및 @를 입력할 수 있습니다.
- 2. R을 열고 knitr::knit() 명령어를 입력하는 대신 편집기 내에서 R과 knitr를 호출하여 소스 문서를 PDF/HTML로 컴파일하고, 더 나아가 편집기 내에서 R 코드 청크를 R로 직접 전송하기 위함입니다.

LATEX, HTML, Markdown 문서를 위한 완성도 높고 훌륭한 편집기들이 다수 존재하며, 일부 편집기에는 knitr가 내장되어 있습니다. 이에 대해서는 다음 섹션에서 설명하겠습니다.

###### 4.1 RStudio

RStudio는 R에 특별히 맞춰진 비교적 새로운 편집기입니다. Sweave와 knitr를 포괄적으로 지원하기 때문에 초보자가 시작하기에 적합한 편집기일 수 있습니다. RStudio는 크로스 플랫폼을 지원하는 무료 오픈소스 소프트웨어이며, http://www.rstudio.com 에서 이용할 수 있습니다. R 프로그래밍에 대한 탁월한 지원 외에도, 다른 여러 편집기에는 없는 주목할 만한 기능이 있습니다. 데스크톱 버전과 동일하게 보이는 서버 버전이 제공되며, 리눅스 서버에 서버 버전을 설치한 후 웹 브라우저에서 R을 사용할 수 있습니다.

전체 문서는 웹사이트에서 찾을 수 있습니다. 여기서는 동적 문서와 관련된 기능만을 간략히 소개합니다. RStudio에서 knitr를 사용하여 Rnw 문서(LATEX)를 작성하려는 경우, 먼저 해야 할 일은 도구(Tools) 메뉴에서 옵션을 변경하는 것입니다.

19

그림 4.1: RStudio에서 Rnw 문서 편집. 청크 헤더 내에 자동 완성 기능이 있습니다. (“”를 입력하면

모든 후보 항목이 나타납니다.) 코드 청크는 메뉴나 키보드 단축키를 통해 삽입할 수 있습니다. Compile PDF 버튼은

![image 3](Dynamic Documents with R and knitr 2nd_images/imageFile3.png)

Rnw에서 PDF를 원클릭으로 생성하도록 지원합니다.

옵션(Options) 메뉴에서 엮기(weaving, 즉 컴파일) Rnw 문서의 기본 옵션은 Sweave이며, R에 knitr를 설치했다면 이를 knitr로 전환할 수 있습니다. knitr와 Sweave의 비교에 대한 더 자세한 논의는 16.1절을 참조하시기 바랍니다. R Markdown과 같은 다른 유형의 문서로 작업할 계획이라면 어떠한 옵션도 구성할 필요가 없으며, RStudio는 필요한 패키지가 누락된 경우 설치하라는 팁을 제공할 것입니다.

RStudio에서 지원하는 모든 문서 형식은 파일(File) 메뉴의 새로 만들기(New) 아래에서 찾을 수 있습니다. 현재 지원되는 형식에는 R Sweave, R Markdown, R HTML이 있습니다. 모든 문서 형식에 대해 원클릭 컴파일이 지원됩니다. 즉, 버튼 하나를 클릭하여 소스 문서를 해당 출력 형식(LATEX는 PDF로, Markdown은 HTML로 등)으로 컴파일할 수 있습니다. Ctrl + Alt + I를 사용하여 R 코드 청크를 입력할 수 있으며, 청크 헤더의 청크 옵션에 대한 자동 완성 기능이 있습니다. 예를 들어, Rnw 문서에서 < < 와 > >= 사이에 “fig.”를 입력하면 fig.width, fig.height 등과 같은 가능한 후보 항목을 볼 수 있습니다. 일반적인 R 스크립트에서 하는 것과 마찬가지로 Ctrl + Enter를 통해 청크 안의 R 코드를 R 콘솔로 전송할 수 있습니다. 이런 방식으로, 전체 문서를 컴파일하기 전에 특정 R 코드 청크를 대화형으로 실행해 볼 수 있습니다. 그림 4.1은 RStudio에서 Rnw 문서가 어떻게 보이는지 보여주는 스크린샷입니다.

Rnw 문서의 경우, 최종 출력 형식은 일반적으로 (LATEX를 통한) PDF입니다.

RStudio는 PDF 문서와 소스 문서 간의 동기화를 제공하며, 이는 다음과 같은 기능을 의미합니다.

- 1. 순방향 검색(forward search): 소스 문서의 한 줄에서 소스 줄에 해당하는 PDF 문서의 적절한 위치로 이동할 수 있습니다.
- 2. 역방향 검색(inverse search): PDF 문서를 클릭하면 RStudio가 Rnw 소스의 해당 줄로 돌아오게 할 수도 있습니다.
- 3. 오류 탐색(error navigation): R이나 LATEX에서 오류가 발생했을 때, RStudio는

오류의 원인이 되는 소스 문서의 위치로 이동시켜 줍니다. 이는 R 또는 LATEX 코드의 문제를 더 빠르게 해결하는 데 도움이 될 수 있습니다.

R Markdown 문서의 경우, RStudio는 HTML을 포함한 다양한 형식으로의 원클릭 컴파일을 제공합니다. 또한, HTML 출력에서 이미지를 base64로 인코딩하고 (MathJax 라이브러리를 통해) LATEX 수학 표현식을 렌더링할 수도 있습니다. 전자의 기능은 생성된 HTML 페이지가 페이지에 임베드되어 외부 이미지에 의존하지 않도록, 즉 독립적으로 작동하도록 보장하기 위한 것입니다. 후자의 기능은 통계학자들이 웹 페이지에 수식을 작성하고자 할 때 유용합니다.

R Markdown(Rmd) 형식은 상당히 단순하여 5분 만에 쉽게

그림 4.2: RStudio에서 Rmd 문서 편집. 청크 옵션 값에 대한 자동 완성 기능도 있습니다. Knit 버튼은

![image 4](Dynamic Documents with R and knitr 2nd_images/imageFile4.png)

Rmd에서 HTML 페이지를 원클릭으로 생성하도록 지원합니다.

숙달할 수 있습니다. 그 단순성 덕분에 이 형식으로 작성된 많은 보고서들이 RPubs에 게시되어 있습니다. RPubs는 사용자의 knitr 보고서를 호스팅하기 위해 RStudio에서 제공하는 무료 플랫폼입니다. 더 많은 예시는 http://rpubs.com 을 참조하시기 바랍니다. 그림 4.2는 RStudio의 샘플 Rmd 문서를 보여줍니다.

3.3절에서 빠른 보고(quick reporting)에 대해 언급했는데, 이 역시 RStudio에서 지원됩니다. RStudio의 R 스크립트에 대해 도구 모음의 버튼을 클릭하여 스크립트 기반의 “R 노트북”(순수하게 R 스크립트에 기반한 보고서)을 만들 수 있습니다.

###### 4.2 LYX

LYX는 문서 작성을 돕는 훌륭한 GUI를 갖춘, 본질적으로 LATEX의 프런트엔드입니다. 화면상으로는 여러 워드 프로세서처럼 보이지만,

- 핵심은 LATEX입니다. 순수 LATEX 편집기와 LYX의 한 가지 주요 차이점은 순수 LATEX에서는 \alpha + \beta 만 볼 수 있는 반면, LYX에서는 화면 이면에 본질적으로 \alpha + \beta 인

α + β를 본다는 것입니다. LYX의 모든 것은 LATEX이지만 백슬래시로 가득 찬 화면으로 시야가 방해받지 않습니다.

버전 2.0.3부터 LYX는 knitr를 공식 모듈로 지원하기 시작했습니다. 자세한 내용은 http://yihui.name/knitr/demo/lyx/ 에서 확인할 수 있습니다. 이 모듈은 다음과 같은 방식으로 작동합니다.

∗.tex LaTeX−→ ∗.pdf (weave) ∗.R (tangle)

∗.lyx −→LyX ∗.Rnw R+−→knitr

현재 LYX에서 사용할 수 있는 유일한 형식은 Rnw입니다. R 코드와 LYX를 섞어 쓰는 것처럼 보이지만, LYX는 래퍼(wrapper)일 뿐이므로 실제로는 Rnw 문서에 R 코드를 임베드하고 있는 것입니다.

리눅스와 Mac OS 사용자의 경우, 이 모듈의 사용법은 다음과 같습니다.

- 1. 새로운 LYX 문서를 만듭니다.
- 2. 문서(Document) 설정(Settings) 모듈(Modules)로 이동하여 Rnw (knitr)라는 모듈을 삽입합니다.
- 3. TEX 코드 삽입(Insert TEX Code)으로 문서에 R 코드 청크를 삽입한 다음 평소처럼 <<>>= 와 @를 입력하기 시작합니다.

도구 모음의 보기(View) 버튼을 클릭하거나 Ctrl + R을 눌러 문서를 PDF로 컴파일하고 결과를 확인합니다. 파일(File) 내보내기(Export) R/S 코드 메뉴를 통해 LYX 문서에서 R 코드를 추출할 수도 있습니다. R 코드가 포함된 LYX의 스크린샷이 그림 4.3에 나와 있습니다.

그림 4.3: LYX에서 knitr 사용하기. Rnw 구문을 사용하여 빨간색 상자에 R 코드가 삽입됩니다. View 버튼을 클릭하면

![image 5](Dynamic Documents with R and knitr 2nd_images/imageFile5.png)

LYX와 knitr를 통해 컴파일된 PDF 문서를 볼 수 있습니다.

윈도우 환경에서 knitr 모듈을 사용하기 전에 한 가지 더 해야 할 단계가 있습니다. 도구(Tools) 기본 설정(Preferences) 경로(Paths) PATH 접두사(PATH prefix)로 이동하여 그곳에 R의 bin 경로를 추가합니다. 이는 보통 C:\Program Files\R\R-x.x.x\bin 과 같으며 R에서 다음을 통해 찾을 수 있습니다.

R.home("bin")

이 변경을 수행한 후에는 도구(Tools) 재구성(Reconfigure)을 통해 LYX를 재구성해야 합니다. 이는 LYX가 R이 설치된 위치를 인지하여 R과 knitr를 호출해 Rnw 문서를 컴파일할 수 있도록 하기 위함입니다. 구체적으로는 Rscript.exe 가 어디에 있는지 알아야 합니다. PATH에 이것이 존재하지 않으면 knitr 모듈을 사용할 수 없습니다. 리눅스 및 Mac OS에서는 이러한 시스템들이 기본적으로 PATH에 R 실행 파일을 포함시키기 때문에 이 단계가 필요하지 않은 경우가 많습니다.

그래픽 인터페이스가 사용하기 꽤 쉬워 보이지만, LYX를 시도하기 전에 LATEX를 숙달할 것을 여전히 강력히 권장합니다. 그렇지 않으면 오류가 발생했을 때 LATEX 문제를 진단하기 어려울 수 있습니다. 어쨌든 LYX는 워드(Word)가 아니기 때문입니다.

###### 4.3 Emacs/ESS

ESS(Emacs Speaks Statistics)는 텍스트 편집기 Emacs를 위한 애드온 패키지입니다 (Rossini et al., 2004). 이는 R, S-Plus, SAS, JAGS 등과 같은 통계 소프트웨어 패키지를 지원합니다. knitr에 대한 지원은 버전 12.09 이후에 추가되었습니다. 그 이전에는 Sweave만 지원되었습니다.

ESS 역시 무료 오픈소스 소프트웨어이며, http://ess.r-project.org 에서 이용할 수 있습니다. Emacs와 함께 설치된 후에는 Emacs에서 knitr를 호출하기가 상당히 쉽습니다. Rnw 문서의 기본 옵션은 Sweave이며, 다음 명령어를 통해 이를 knitr로 변경할 수 있습니다 (Emacs 키 표기법에서 M은 Meta 키를 나타내며 대부분의 키보드에서 Alt 키에 해당하고, M-x는 Meta를 누른 상태에서 x를 누르는 것을 의미합니다).

M-x customize-group ess-R

ess-swv-processor 옵션을 찾아 knitr로 변경합니다. 그런 다음 새로운 Rnw 문서를 만들고, M-n s를 눌러 Rnw를 TEX로 컴파일하며, M-n P를 눌러 TEX를 PDF로 컴파일할 수 있습니다.

ESS에서 Rmd 문서 및 기타 문서 형식에 대한 지원은 아직 개발 중입니다. 개발자들에 따르면 이 기능은

ESS 13.03에 포함될 수 있으며, 독자들은 향후 공식 발표에 주목해 주시기 바랍니다.

###### 4.4 기타 편집기

사용자 정의 명령어를 정의하여 문서를 컴파일할 수 있도록 허용하는 편집기라면, 다른 편집기에서도 지원을 추가하는 것은 어렵지 않습니다. 일반적으로 사용자 정의 명령어는 다음과 같습니다.

Rscript -e "library(knitr); knit( input.ext )"

이 명령어는 R을 호출하여 knitr 패키지를 로드하고, knit() 함수를 사용하여 input.ext 라는 이름의 입력 문서를 컴파일합니다.

WinEdt (상용 소프트웨어)에는 knitr를 지원하는 R-Sweave라는 모드가 있으며, Tinn-R (무료)에는 기본적으로 지원 기능이 내장되어 있습니다. Texmaker, Eclipse, TextMate, TEXShop, Vim과 같은 다른 텍스트 편집기를 구성하여 그 안에서 편리하게 보고서를 컴파일하도록 설정하는 것도 가능합니다. 구성 지침은 http://yihui.name/knitr/demo/editors/ 에 수집되어 있습니다.
