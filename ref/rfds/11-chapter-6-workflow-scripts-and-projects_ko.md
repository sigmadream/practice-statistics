# 6장. 워크플로우: 스크립트와 프로젝트(Workflow: Scripts and Projects)

이 장에서는 코드를 구성하는 데 필수적인 두 가지 도구인 스크립트(scripts)와 프로젝트(projects)를 소개합니다.

# 스크립트(Scripts)

지금까지 여러분은 코드를 실행하기 위해 콘솔을 사용했습니다. 콘솔은 시작하기에 좋은 곳이지만, 더 복잡한 ggplot2 그래픽과 더 긴 dplyr 파이프라인을 생성함에 따라 곧 공간이 비좁아진다는 것을 알게 될 것입니다. 작업할 공간을 더 확보하려면 스크립트 편집기(script editor)를 사용하세요. 파일(File) 메뉴를 클릭하고 새 파일(New File)을 선택한 다음 R 스크립트(R script)를 선택하거나 키보드 단축키 Cmd/Ctrl+Shift+N을 사용하여 스크립트 편집기를 엽니다. 이제 <a href="#fig-rstudio-script" data-type="xref">그림 6-1</a>과 같이 네 개의 창이 나타납니다. 스크립트 편집기는 코드를 실험하기에 아주 좋은 장소입니다. 무언가를 변경하고 싶을 때 전체를 다시 입력할 필요가 없습니다. 스크립트를 편집하고 다시 실행하기만 하면 됩니다. 또한 잘 작동하고 원하는 결과를 내는 코드를 작성했다면, 나중에 쉽게 돌아갈 수 있도록 스크립트 파일로 저장할 수 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0601.png" alt="편집기, 콘솔 및 출력이 강조 표시된 RStudio IDE." />
<h6 id="figure-6-1.-opening-the-script-editor-adds-a-new-pane-at-the-top-left-of-the-ide.">그림 6-1. 스크립트 편집기를 열면 IDE의 왼쪽 상단에 새 창이 추가됩니다.</h6>
</figure>

## 코드 실행(Running Code)

스크립트 편집기는 복잡한 ggplot2 플롯이나 긴 dplyr 조작 순서를 구축하기에 훌륭한 장소입니다. 스크립트 편집기를 효과적으로 사용하는 핵심은 가장 중요한 키보드 단축키 중 하나인 Cmd/Ctrl+Enter를 기억하는 것입니다. 이것은 콘솔에서 현재의 R 표현식을 실행합니다. 예를 들어, 다음 코드를 살펴보겠습니다:

`library``(``dplyr``)` `library``(``nycflights13``)` `not_cancelled` `<-` `flights` `|>` `filter``(``!``is.na``(``dep_delay``)`█`,` `!``is.na``(``arr_delay``))` `not_cancelled` `|>` `group_by``(``year``,` `month``,` `day``)` `|>` `summarize``(``mean` `=` `mean``(``dep_delay``))`

커서가 █ 위치에 있을 때 Cmd/Ctrl+Enter를 누르면 `not_cancelled`를 생성하는 전체 명령이 실행됩니다. 또한 커서가 다음 명령문(`not_cancelled |>`로 시작하는 줄)으로 이동합니다. 이렇게 하면 Cmd/Ctrl+Enter를 반복해서 눌러 전체 스크립트를 쉽게 단계별로 살펴볼 수 있습니다.

코드를 표현식 단위로 실행하는 대신, Cmd/Ctrl+Shift+S를 사용하여 전체 스크립트를 한 번에 실행할 수도 있습니다. 이 작업을 정기적으로 수행하는 것은 코드의 모든 중요한 부분이 스크립트에 잘 캡처되었는지 확인하는 좋은 방법입니다.

항상 필요한 패키지로 스크립트를 시작할 것을 권장합니다. 그렇게 하면 다른 사람과 코드를 공유할 때 그들이 어떤 패키지를 설치해야 하는지 쉽게 알 수 있습니다. 하지만 주의할 점은 공유하는 스크립트에 절대 <a href="https://rdrr.io/r/utils/install.packages.html" class="orm:hideurl"><code>install.packages()</code></a>를 포함해서는 안 된다는 것입니다. 상대방이 주의하지 않을 경우 그들의 컴퓨터에 무언가를 설치하게 만드는 스크립트를 건네는 것은 배려 없는 행동입니다!

이후의 장들을 공부할 때는 스크립트 편집기에서 시작하여 키보드 단축키를 연습하는 것을 강력히 권장합니다. 시간이 지나면 이렇게 코드를 콘솔로 보내는 방식이 너무 자연스러워져서 생각조차 하지 않게 될 것입니다.

## RStudio 진단(RStudio Diagnostics)

스크립트 편집기에서 RStudio는 사이드바의 빨간색 물결선과 십자가로 구문 오류(syntax error)를 강조 표시합니다:

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_06in01.png" alt="x y &lt;- 10 스크립트가 있는 스크립트 편집기. 빨간색 X는 구문 오류가 있음을 나타냅니다." />
</figure>

문제가 무엇인지 확인하려면 십자가 위에 마우스를 올려보세요:

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_06in02.png" alt="스크립트 편집기의 구문 오류 표시 위에 마우스를 올려서 툴팁을 확인하는 모습." />
</figure>

RStudio는 잠재적인 문제에 대해서도 알려줍니다:

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_06in03.png" alt="3 == NA 스크립트가 있는 스크립트 편집기. 노란색 느낌표는 잠재적 문제가 있음을 나타냅니다." />
</figure>

## 저장 및 이름 짓기(Saving and Naming)

RStudio는 당신이 종료할 때 스크립트 편집기의 내용을 자동으로 저장하고 다시 열 때 자동으로 불러옵니다. 그럼에도 불구하고, Untitled1, Untitled2, Untitled3 등과 같은 이름을 피하고 대신 정보가 담긴 이름으로 스크립트를 저장하는 것이 좋습니다.

파일 이름을 `code.R`이나 `myscript.R`로 짓고 싶은 유혹이 들 수도 있지만, 파일 이름을 선택하기 전에 좀 더 신중하게 생각해야 합니다. 파일 이름 지정을 위한 세 가지 중요한 원칙은 다음과 같습니다:

1.  파일 이름은 기계가 읽을 수 있어야(machine readable) 합니다: 공백, 기호, 특수 문자를 피하세요. 대소문자 구분에 의존하여 파일을 구별하지 마세요.
2.  파일 이름은 사람이 읽을 수 있어야(human readable) 합니다: 파일 이름을 사용하여 파일의 내용을 설명하세요.
3.  파일 이름은 기본 정렬(default ordering)과 잘 어울려야 합니다: 알파벳순 정렬이 사용 순서대로 나타나도록 파일 이름을 숫자로 시작하세요.

예를 들어, 프로젝트 폴더에 다음과 같은 파일들이 있다고 가정해 봅시다:

`alternative model.R code for exploratory analysis.r finalreport.qmd FinalReport.qmd fig 1.png Figure_02.png model_first_try.R run-first.r temp.txt`

여기에는 여러 가지 문제가 있습니다. 어떤 파일을 먼저 실행해야 할지 찾기 어렵고, 파일 이름에 공백이 포함되어 있으며, 이름은 같지만 대소문자가 다른 두 파일이 존재하고(`finalreport` 대 `FinalReport`<sup><a href="ch06.html#idm44771313233616" id="idm44771313233616-marker" data-type="noteref">1</a></sup>), 일부 이름은 그 내용을 전혀 설명하지 못합니다(`run-first` 및 `temp`).

다음은 동일한 파일 세트의 이름을 짓고 구성하는 더 나은 방법입니다:

`01-load-data.R 02-exploratory-analysis.R 03-model-approach-1.R 04-model-approach-2.R fig-01.png fig-02.png report-2022-03-20.qmd report-2022-04-02.qmd report-draft-notes.txt`

주요 스크립트에 번호를 매기면 실행 순서가 명확해지고, 일관된 명명 체계(naming scheme)를 사용하면 무엇이 다른지 쉽게 파악할 수 있습니다. 추가로, 그림 파일들도 비슷하게 라벨링되었고, 보고서 파일들은 이름에 날짜를 포함하여 구별되었으며, `temp`는 그 내용을 더 잘 설명하도록 `report-draft-notes`로 변경되었습니다. 디렉터리에 파일이 많은 경우, 정리를 한 단계 더 나아가서 서로 다른 종류의 파일들(스크립트, 그림 등)을 서로 다른 디렉터리에 배치하는 것이 권장됩니다.

# 프로젝트(Projects)

언젠가 당신은 R을 종료하고 다른 일을 하다가 나중에 분석으로 돌아와야 할 때가 있을 것입니다. 언젠가 여러 분석을 동시에 수행하면서 이들을 별도로 분리하고 싶을 때도 있을 것입니다. 언젠가 외부 세계에서 R로 데이터를 가져오거나 R에서 외부 세계로 숫자 결과 및 그림을 보내야 할 때도 있을 것입니다.

이러한 실제 상황을 처리하려면 두 가지 결정을 내려야 합니다:

- 진실의 원천(source of truth)은 무엇입니까? 일어난 일의 지속적인 기록으로 무엇을 저장하시겠습니까?
- 분석은 어디에 위치합니까(where does your analysis live)?

## 진실의 원천은 무엇인가? (What Is the Source of Truth?)

초보자일 때는 분석 과정 전반에 걸쳐 생성한 모든 객체를 유지하기 위해 현재 환경(environment)에 의존해도 괜찮습니다. 하지만 더 큰 프로젝트를 진행하거나 다른 사람들과 협업하는 것을 더 쉽게 만들려면, 여러분의 진실의 원천(source of truth)은 바로 R 스크립트여야 합니다. R 스크립트(그리고 데이터 파일)를 가지고 있으면 환경을 재창조할 수 있습니다. 환경만 가지고 R 스크립트를 재창조하는 것은 훨씬 더 어렵습니다. 기억을 더듬어 많은 코드를 다시 입력해야 하거나(이 과정에서 필연적으로 실수가 발생합니다) R 히스토리(history)를 신중하게 뒤져야 합니다.

분석을 위한 진실의 원천으로 R 스크립트를 유지하는 데 도움이 되도록, 세션 간에 작업 공간(workspace)을 보존하지 않도록 RStudio를 설정하는 것을 강력히 권장합니다. <a href="https://usethis.r-lib.org/reference/use_blank_slate.html" class="orm:hideurl"><code>usethis::use_blank_slate()</code></a><sup><a href="ch06.html#idm44771313209872" id="idm44771313209872-marker" data-type="noteref">2</a></sup>를 실행하거나 <a href="#fig-blank-slate" data-type="xref">그림 6-2</a>에 표시된 옵션을 모방하여 이 작업을 수행할 수 있습니다. 이는 RStudio를 다시 시작할 때 지난번에 실행한 코드를 더 이상 기억하지 못하고, 생성한 객체나 읽어온 데이터 세트를 사용할 수 없게 되므로 단기적으로 약간의 고통을 유발할 것입니다. 하지만 이 단기적인 고통은 모든 중요한 절차를 코드에 캡처하도록 강제하기 때문에 장기적인 고통을 덜어줍니다. 3개월이 지난 후에야 중요한 계산 결과만 환경에 저장하고 계산 자체는 코드에 저장하지 않았다는 것을 발견하는 것보다 더 나쁜 것은 없습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0602.png" alt="RStudio 글로벌 옵션 창." />
<h6 id="figure-6-2.-copy-these-selections-in-your-rstudio-options-to-always-start-your-rstudio-session-with-a-clean-slate.">그림 6-2. RStudio 세션을 항상 깨끗한 상태(clean slate)로 시작하려면 RStudio 옵션에서 이 선택 항목을 복사하세요.</h6>
</figure>

코드의 중요한 부분을 편집기에 잘 캡처했는지 확인하기 위해 함께 작동하는 훌륭한 키보드 단축키 쌍이 있습니다:

1.  R을 다시 시작하려면 Cmd/Ctrl+Shift+0/F10을 누릅니다.
2.  현재 스크립트를 다시 실행하려면 Cmd/Ctrl+Shift+S를 누릅니다.

저희는 일주일에 이 패턴을 수백 번씩 사용합니다.

키보드 단축키를 사용하지 않는 경우, 대안으로 Session > Restart R을 선택한 다음 현재 스크립트를 강조 표시하고 다시 실행할 수 있습니다.

# RStudio Server

RStudio Server를 사용하는 경우 기본적으로 R 세션이 다시 시작되지 않습니다. RStudio Server 탭을 닫으면 R을 닫는 것처럼 느껴질 수 있지만, 서버는 실제로는 그것을 백그라운드에서 계속 실행합니다. 다음에 돌아올 때는 떠났던 바로 그 위치에 있게 될 것입니다. 이 때문에 정기적으로 R을 다시 시작하여 깨끗한 상태(clean slate)에서 시작하는 것이 훨씬 더 중요해집니다.

## 당신의 분석은 어디에 위치합니까? (Where Does Your Analysis Live?)

R에는 *작업 디렉터리(working directory)*라는 강력한 개념이 있습니다. 이는 R이 불러오도록 요청하는 파일을 찾는 곳이자, 저장하도록 요청하는 모든 파일을 배치할 곳입니다. RStudio는 콘솔 맨 위에 현재 작업 디렉터리를 표시합니다:

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_06in04.png" alt="현재 작업 디렉터리를 표시하는 콘솔 탭." />
</figure>

<a href="https://rdrr.io/r/base/getwd.html" class="orm:hideurl"><code>getwd()</code></a>를 실행하여 R 코드에서 이를 출력할 수 있습니다:

`getwd``()` `#> [1] "/Users/hadley/Documents/r4ds"`

이 R 세션에서 현재 작업 디렉터리(이것을 "홈"이라고 생각하세요)는 Hadley의 *Documents* 폴더 안, *r4ds*라는 하위 폴더에 있습니다. 당신의 컴퓨터는 Hadley와 다른 디렉터리 구조를 가지고 있기 때문에, 이 코드를 실행하면 다른 결과를 반환할 것입니다!

R 초보자일 때는 작업 디렉터리를 홈 디렉터리, 문서 디렉터리 또는 컴퓨터의 다른 이상한 디렉터리로 두어도 괜찮습니다. 하지만 당신은 이미 이 책의 7개 장을 살펴보고 있으며 더 이상 초보자가 아닙니다. 곧 프로젝트를 디렉터리로 구성하고 프로젝트에서 작업할 때 R의 작업 디렉터리를 해당 디렉터리로 설정하도록 발전해야 합니다.

R 내에서 작업 디렉터리를 설정할 수 있지만 *권장하지는 않습니다*:

`setwd``(``"/path/to/my/CoolProject"``)`

더 나은 방법이 있습니다. 당신을 R 작업을 전문가처럼 관리하는 길로 인도해 줄 방법입니다. 그 방법이 바로 *RStudio 프로젝트(RStudio project)*입니다.

## RStudio 프로젝트 (RStudio Projects)

주어진 프로젝트와 관련된 모든 파일(입력 데이터, R 스크립트, 분석 결과 및 그림)을 한 디렉터리에 보관하는 것은 매우 현명하고 일반적인 관행이므로, RStudio는 *프로젝트(projects)*를 통해 이를 기본적으로 지원합니다. 이 책의 나머지 부분을 작업하는 동안 사용할 프로젝트를 만들어 보겠습니다. File > New Project를 선택한 다음 <a href="#fig-new-project" data-type="xref">그림 6-3</a>에 표시된 단계를 따르세요.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0603.png" alt="새 프로젝트 메뉴의 세 스크린샷." />
<h6 id="figure-6-3.-to-create-new-project-top-first-click-new-directory-then-middle-click-new-project-then-bottom-fill-in-the-directory-project-name-choose-a-good-subdirectory-for-its-home-and-click-create-project.">그림 6-3. 새 프로젝트를 생성하려면: (위) 먼저 New Directory를 클릭하고, (가운데) New Project를 클릭한 다음, (아래) 디렉터리(프로젝트) 이름을 입력하고 홈으로 적절한 하위 디렉터리를 선택한 후 Create Project를 클릭합니다.</h6>
</figure>

프로젝트 이름을 `r4ds`로 지정하고 어느 하위 디렉터리에 프로젝트를 배치할지 신중하게 생각해 보세요. 분별 있는 곳에 저장하지 않으면 나중에 찾기 힘들 것입니다!

이 과정이 완료되면, 이 책을 위한 새 RStudio 프로젝트가 생깁니다. 프로젝트의 "홈(home)"이 현재 작업 디렉터리인지 확인하세요:

`getwd``()` `#> [1] /Users/hadley/Documents/r4ds`

이제 스크립트 편집기에 다음 명령을 입력하고 `diamonds.R`이라는 이름으로 파일을 저장합니다. 그런 다음 `data`라는 새 폴더를 만드세요. RStudio의 Files 창(pane)에 있는 New Folder 버튼을 클릭하면 이를 할 수 있습니다. 마지막으로, 전체 스크립트를 실행하면 PNG와 CSV 파일이 프로젝트 디렉터리에 저장됩니다. 세부 사항에 대해서는 걱정하지 마세요. 책의 뒷부분에서 배우게 될 것입니다.

`library``(``tidyverse``)` `ggplot``(``diamonds``,` `aes``(``x` `=` `carat``,` `y` `=` `price``))` `+` `geom_hex``()` `ggsave``(``"diamonds.png"``)` `write_csv``(``diamonds``,` `"data/diamonds.csv"``)`

RStudio를 종료하세요. 당신의 프로젝트와 관련된 폴더를 살펴보고 `.Rproj` 파일을 확인해 보세요. 해당 파일을 두 번 클릭하여 프로젝트를 다시 엽니다. 이전에 하던 지점으로 돌아간 것을 확인할 수 있습니다. 동일한 작업 디렉터리와 명령 기록(command history)이 있으며, 작업하던 파일들이 여전히 열려 있습니다. 하지만 우리의 지시를 따랐기 때문에, 당신은 완전히 새로운 환경을 갖게 되어 깨끗한 상태에서 시작한다는 것을 보장받게 됩니다.

가장 선호하는 OS별 방식으로 컴퓨터에서 `diamonds.png`를 검색해 보세요. 그러면 PNG(당연히)를 찾을 수 있을 뿐만 아니라 *그것을 만든 스크립트*(`diamonds.R`)도 찾을 수 있습니다. 이것은 엄청난 이점입니다! 언젠가 그림을 다시 만들거나 단지 그것이 어디서 왔는지 이해하고 싶어질 것입니다. 그림을 마우스나 클립보드가 아닌 *R 코드와 함께* 파일로 엄격하게 저장한다면, 옛 작업물을 쉽게 재현할 수 있을 것입니다!

## 상대 경로와 절대 경로 (Relative and Absolute Paths)

프로젝트 내부에서는 절대 절대 경로(absolute paths)가 아닌 상대 경로(relative paths)만 사용해야 합니다. 차이점이 무엇일까요? 상대 경로는 작업 디렉터리, 즉 프로젝트의 홈에 대한 상대적인 경로입니다. 앞서 Hadley가 `data/diamonds.csv`라고 썼을 때, 그것은 `/Users/hadley/Documents/r4ds/data/diamonds.csv`에 대한 단축어였습니다. 그러나 중요한 것은, 만약 Mine이 자신의 컴퓨터에서 이 코드를 실행했다면 그것은 `/Users/Mine/Documents/r4ds/data/diamonds.csv`를 가리킬 것이라는 점입니다. 이것이 상대 경로가 중요한 이유입니다: 상대 경로는 R 프로젝트 폴더가 결국 어디에 위치하든 상관없이 제대로 작동할 것입니다.

절대 경로는 작업 디렉터리에 관계없이 같은 위치를 가리킵니다. 이것은 운영 체제에 따라 조금 다르게 생겼습니다. Windows에서는 드라이브 문자(예: `C:`) 또는 두 개의 백슬래시(예: `\\servername`)로 시작하고, Mac/Linux에서는 슬래시 / (예: `/users/hadley`)로 시작합니다. 당신과 정확히 똑같은 디렉터리 구성을 가진 다른 사람은 아무도 없기 때문에 공유를 방해하므로 스크립트에서 *절대* 절대 경로를 사용해서는 안 됩니다.

운영 체제 간에는 또 다른 중요한 차이점이 있습니다: 경로의 구성 요소를 구분하는 방법입니다. Mac과 Linux는 슬래시(예: `data/diamonds.csv`)를 사용하고 Windows는 백슬래시(예: `data\diamonds.csv`)를 사용합니다. R은 (현재 사용 중인 플랫폼이 무엇이든) 두 유형 모두에서 작동할 수 있지만, 안타깝게도 백슬래시는 R에서 특별한 의미를 가지며, 경로에 단일 백슬래시를 사용하려면 두 개의 백슬래시를 입력해야 합니다! 이는 삶을 답답하게 만들므로 우리는 항상 슬래시가 있는 Linux/Mac 스타일을 사용할 것을 권장합니다.

# 연습문제

1.  [RStudio Tips 트위터 계정](https://twitter.com/rstudiotips)으로 가서 흥미로워 보이는 팁을 하나 찾으세요. 사용해 보는 연습을 해보세요!

2.  RStudio 진단(diagnostics)은 또 어떤 일반적인 실수들을 보고할까요? 알아보려면 [코드 진단에 관한 이 문서](https://oreil.ly/coili)를 읽어보세요.

# 요약 (Summary)

이 장에서는 R 코드를 스크립트(파일)와 프로젝트(디렉터리)로 구성하는 방법을 배웠습니다. 코드 스타일과 마찬가지로 처음에는 귀찮은 작업처럼 느껴질 수 있습니다. 하지만 여러 프로젝트에 걸쳐 코드가 쌓이면서 초기의 약간의 정리가 나중에 얼마나 많은 시간을 절약해 주는지 그 가치를 깨닫게 될 것입니다.

요약하자면, 스크립트와 프로젝트는 미래에 큰 도움이 될 견고한 워크플로우를 제공합니다:

- 각 데이터 분석 프로젝트마다 하나의 RStudio 프로젝트를 만듭니다.
- 유익한 이름이 지정된 스크립트를 프로젝트에 저장하고, 편집하고, 일부분씩 또는 전체적으로 실행합니다. R을 자주 다시 시작하여 스크립트에 모든 것을 담았는지 확인하세요.
- 절대 절대 경로가 아닌 항상 상대 경로만 사용하세요.

그러면 필요한 모든 것이 한 곳에 있고 작업 중인 다른 모든 프로젝트와 깔끔하게 분리됩니다.

지금까지 우리는 R 패키지에 번들로 포함된 데이터셋을 가지고 작업했습니다. 이렇게 하면 미리 준비된 데이터에 대해 쉽게 연습할 수 있지만 분명히 실제 여러분의 데이터는 이런 식으로 제공되지 않을 것입니다. 따라서 다음 장에서는 readr 패키지를 사용하여 디스크에서 R 세션으로 데이터를 로드하는 방법을 배울 것입니다.

<sup>[1](ch06.html#idm44771313233616-marker)</sup> 이름에 "final(최종)"을 사용하여 운명을 시험하고 있다는 점은 말할 것도 없습니다. 만화 'Piled Higher and Deeper'에 [이에 대한 재미있는 연재물](https://oreil.ly/L9ip0)이 있습니다.

<sup>[2](ch06.html#idm44771313209872-marker)</sup> 설치되어 있지 않은 경우 `install.packages("usethis")`를 사용하여 설치할 수 있습니다.
