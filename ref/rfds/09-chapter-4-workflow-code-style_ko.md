# 4장. 워크플로: 코드 스타일

좋은 코딩 스타일은 올바른 구두점과 같습니다. 구두점이 없어도 어떻게든 할 수는 있지만,구두점이있으면읽기가훨씬수월해집니다. 이제 막 프로그래밍을 시작한 사람이라 할지라도 코드 스타일에 신경 쓰는 것은 좋은 생각입니다. 일관된 스타일을 사용하면 다른 사람들(미래의 자신을 포함하여!)이 여러분의 작업을 더 쉽게 읽을 수 있으며, 특히 다른 사람의 도움이 필요할 때 중요합니다. 이 장에서는 이 책 전체에서 사용되는 [tidyverse 스타일 가이드](https://oreil.ly/LykON)의 가장 중요한 점들을 소개합니다.

처음에는 코드의 스타일을 지정하는 것이 약간 지루하게 느껴질 수 있지만, 연습하다 보면 곧 제2의 천성(second nature)이 될 것입니다. 게다가 Lorenz Walthert가 만든 [styler](https://oreil.ly/8_Z1c) 패키지와 같이 기존 코드의 스타일을 빠르게 다시 지정해주는 훌륭한 도구들도 있습니다. `install.packages("styler")`로 설치한 후 RStudio의 명령 팔레트(command palette)를 통해 사용하는 것이 쉬운 방법입니다. 명령 팔레트를 사용하면 모든 내장 RStudio 명령과 패키지에서 제공하는 많은 애드인(addin)을 사용할 수 있습니다. Cmd/Ctrl+Shift+P를 눌러 팔레트를 연 다음 *styler*를 입력하여 styler가 제공하는 모든 단축키를 확인해 보세요. <a href="#fig-styler" data-type="xref">그림 4-1</a>에 결과가 나와 있습니다.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0401.png" alt="A screenshot showing the command palette after typing &quot;styler&quot;, showing the four styling tool provided by the package." /></p>
<h6 id="figure-4-1.-rstudios-command-palette-makes-it-easy-to-access-every-rstudio-command-using-only-the-keyboard.">그림 4-1. RStudio의 명령 팔레트를 사용하면 키보드만 사용하여 모든 RStudio 명령에 쉽게 액세스할 수 있습니다.</h6>
</figure>

이 장의 코드 예제에는 tidyverse와 nycflights13 패키지를 사용할 것입니다.

`library``(``tidyverse``)` `library``(``nycflights13``)`

# 이름

<a href="ch02.html#sec-whats-in-a-name" data-type="xref">"이름에 담긴 의미"</a>에서 이름에 대해 간략히 논의했습니다. 변수 이름(`<-`로 생성된 것과 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>로 생성된 것)에는 소문자, 숫자, `_`만 사용해야 한다는 것을 기억하세요. 이름 안에서 단어를 구분할 때는 `_`를 사용하세요.

`# Strive for:` `short_flights` `<-` `flights` `|>` `filter``(``air_time` `<` `60``)` `# Avoid:` `SHORTFLIGHTS` `<-` `flights` `|>` `filter``(``air_time` `<` `60``)`

일반적인 경험 법칙으로, 타이핑하기 빠른 간결한 이름보다는 이해하기 쉬운 길고 의미가 잘 드러나는 이름을 선호하는 것이 좋습니다. 짧은 이름은 코드를 작성할 때 상대적으로 시간을 덜 절약해 주지만(특히 자동 완성이 타이핑을 끝내는 데 도움을 주기 때문입니다), 나중에 예전 코드로 돌아와서 암호 같은 약어를 억지로 해석해야 할 때는 많은 시간이 걸릴 수 있습니다.

관련된 항목들에 대한 여러 이름이 있다면 최선을 다해 일관성을 유지하세요. 이전의 규칙을 잊어버렸을 때 불일치가 발생하기 쉬우므로, 돌아가서 이름들을 바꿔야 하더라도 기분 나빠하지 마세요. 일반적으로 하나의 주제에 대한 변형인 변수가 여러 개 있다면, 자동 완성 기능이 변수의 시작 부분에서 가장 잘 작동하므로 공통 접미사(suffix)보다는 공통 접두사(prefix)를 부여하는 것이 더 낫습니다.

# 공백

`^`를 제외한 수학 연산자(즉, `+`, `-`, `==`, `<`, …)의 양쪽과 할당 연산자(`<-`) 주위에는 공백을 두세요.

`# Strive for` `z` `<-` `(``a` `+` `b``)``^2` `/` `d` `# Avoid` `z``<-``(` `a` `+` `b` `)` `^` `2``/``d`

일반적인 함수 호출의 경우 괄호 안팎에 공백을 두지 마세요. 표준 영어에서처럼 쉼표 뒤에는 항상 공백을 두세요.

`# Strive for` `mean``(``x``,` `na.rm` `=` `TRUE``)` `# Avoid` `mean ``(``x` `,``na.rm``=``TRUE``)`

정렬(alignment)을 개선하기 위해 여분의 공백을 추가하는 것은 괜찮습니다. 예를 들어, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에서 여러 변수를 생성하는 경우 모든 `=`가 줄을 맞추도록 공백을 추가하고 싶을 수 있습니다.<sup><a href="ch04.html#idm44771324908448" id="idm44771324908448-marker" data-type="noteref">1</a></sup> 이렇게 하면 코드를 더 쉽게 훑어볼 수 있습니다.

`flights` `|>` `mutate``(` `speed` `=` `distance` `/` `air_time``,` `dep_hour` `=` `dep_time` `%/%` `100``,` `dep_minute` `=` `dep_time` `%%` `100` `)`

# 파이프

`|>`는 항상 그 앞에 공백이 있어야 하며, 일반적으로 줄의 맨 마지막에 있어야 합니다. 이렇게 하면 새로운 단계를 추가하거나, 기존 단계를 재배열하거나, 단계 내의 요소를 수정하거나, 왼쪽의 동사들을 훑어봄으로써 전체적인 관점(10,000-foot view)을 파악하기가 더 쉬워집니다.

`# Strive for ` `flights` `|>` `filter``(``!``is.na``(``arr_delay``),` `!``is.na``(``tailnum``))` `|>` `count``(``dest``)` `# Avoid` `flights``|>``filter``(``!``is.na``(``arr_delay``),` `!``is.na``(``tailnum``))``|>``count``(``dest``)`

파이프로 연결되는 함수에 이름이 지정된 인자가 있는 경우(<a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>나 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>처럼) 각 인자를 새 줄에 두세요. 함수에 이름이 지정된 인자가 없는 경우(<a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>나 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>처럼) 모두 한 줄에 유지하되, 줄에 다 들어가지 않는 경우에는 각 인자를 고유한 줄에 두어야 합니다.

`# Strive for` `flights` `|>` `group_by``(``tailnum``)` `|>` `summarize``(` `delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``()` `)` `# Avoid` `flights` `|>` `group_by``(` `tailnum` `)` `|>` `summarize``(``delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``())`

파이프라인의 첫 번째 단계 이후에는 각 줄을 두 칸씩 들여쓰기하세요. RStudio는 `|>` 다음에 줄 바꿈을 할 때 자동으로 공백을 넣어줍니다. 각 인자를 별도의 줄에 넣는 경우 추가로 두 칸 더 들여쓰기하세요. `)`는 자체 줄에 두고 함수 이름의 수평 위치와 일치하도록 들여쓰기를 해제하세요.

`# Strive for ` `flights` `|>` `group_by``(``tailnum``)` `|>` `summarize``(` `delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``()` `)` `# Avoid` `flights``|>` `group_by``(``tailnum``)` `|>` `summarize``(` `delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``()` `)` `# Avoid` `flights``|>` `group_by``(``tailnum``)` `|>` `summarize``(` `delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``()` `)`

파이프라인이 한 줄에 쉽게 들어간다면 이러한 규칙 중 일부를 피하는 것도 괜찮습니다. 하지만 우리의 집단적 경험에 비추어 볼 때, 짧은 코드 조각이 점점 길어지는 경우가 흔하므로 처음부터 필요한 모든 수직 공간(vertical space)을 사용하여 시작하는 것이 장기적으로 볼 때 일반적으로 시간을 절약합니다.

`# This fits compactly on one line` `df` `|>` `mutate``(``y` `=` `x` `+` `1``)` `# While this takes up 4x as many lines, it's easily extended to ` `# more variables and more steps in the future` `df` `|>` `mutate``(` `y` `=` `x` `+` `1` `)`

마지막으로 10~15줄보다 길어지는 아주 긴 파이프를 작성하는 것은 주의하세요. 더 작은 하위 작업(subtasks)으로 나누고 각 작업에 정보 제공용 이름을 부여해 보세요. 이름은 독자에게 무슨 일이 일어나고 있는지 단서를 제공하는 데 도움이 되며 중간 결과가 예상대로인지 확인하기 더 쉽게 만들어 줍니다. 데이터의 구조를 근본적으로 변경할 때(예: 피벗팅이나 요약 후)와 같이 무언가에 정보 제공용 이름을 부여할 수 있을 때는 항상 그렇게 해야 합니다. 처음부터 제대로 할 것이라고 기대하지 마세요! 이는 좋은 이름을 얻을 수 있는 중간 상태가 있다면 긴 파이프라인을 쪼개는 것을 의미합니다.

# ggplot2

파이프에 적용되는 것과 동일한 기본 규칙이 ggplot2에도 적용됩니다. `+`를 `|>`와 동일하게 다루면 됩니다:

`flights` `|>` `group_by``(``month``)` `|>` `summarize``(` `delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``)` `)` `|>` `ggplot``(``aes``(``x` `=` `month``,` `y` `=` `delay``))` `+` `geom_point``()` `+` `geom_line``()`

마찬가지로 함수의 모든 인자를 한 줄에 담을 수 없다면 각 인자를 고유한 줄에 두세요:

`flights` `|>` `group_by``(``dest``)` `|>` `summarize``(` `distance` `=` `mean``(``distance``),` `speed` `=` `mean``(``distance` `/` `air_time``,` `na.rm` `=` `TRUE``)` `)` `|>` `ggplot``(``aes``(``x` `=` `distance``,` `y` `=` `speed``))` `+` `geom_smooth``(` `method` `=` `"loess"``,` `span` `=` `0.5``,` `se` `=` `FALSE``,` `color` `=` `"white"``,` `linewidth` `=` `4` `)` `+` `geom_point``()`

`|>`에서 `+`로의 전환에 유의하세요. 이 전환이 필요하지 않았다면 좋았겠지만 불행히도 ggplot2는 파이프가 발견되기 전에 작성되었습니다.

# 섹션 주석

스크립트가 길어지면 *섹션 주석(sectioning comments)*을 사용하여 파일을 관리하기 쉬운 조각으로 나눌 수 있습니다:

`# Load data --------------------------------------` `# Plot data --------------------------------------`

RStudio는 이러한 헤더를 생성하는 키보드 단축키(Cmd/Ctrl+Shift+R)를 제공하며, <a href="#fig-rstudio-sections" data-type="xref">그림 4-2</a>와 같이 편집기의 왼쪽 하단에 있는 코드 탐색 드롭다운(code navigation drop-down)에 이를 표시합니다.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0402.png" /></p>
<h6 id="figure-4-2.-after-adding-sectioning-comments-to-your-script-you-can-easily-navigate-to-them-using-the-code-navigation-tool-in-the-bottom-left-of-the-script-editor.">그림 4-2. 스크립트에 섹션 주석을 추가한 후, 스크립트 편집기의 왼쪽 하단에 있는 코드 탐색 도구를 사용하여 해당 위치로 쉽게 이동할 수 있습니다.</h6>
</figure>

# 연습 문제

1.  이전 가이드라인에 따라 다음 파이프라인들의 스타일을 다시 지정하세요:

    `flights``|>``filter``(``dest``==``"IAH"``)``|>``group_by``(``year``,``month``,``day``)``|>``summarize``(``n``=``n``(),` `delay``=``mean``(``arr_delay``,``na.rm``=``TRUE``))``|>``filter``(``n``>``10``)` `flights``|>``filter``(``carrier``==``"UA"``,``dest``%in%``c``(``"IAH"``,``"HOU"``),``sched_dep_time``>` `0900``,``sched_arr_time``<``2000``)``|>``group_by``(``flight``)``|>``summarize``(``delay``=``mean``(` `arr_delay``,``na.rm``=``TRUE``),``cancelled``=``sum``(``is.na``(``arr_delay``)),``n``=``n``())``|>``filter``(``n``>``10``)`

# 요약

이 장에서는 코드 스타일의 가장 중요한 원칙들을 배웠습니다. 처음에는 이것들이 자의적인 규칙들의 집합처럼 느껴질 수 있지만(실제로 그러니까요!), 시간이 지나면서 코드를 더 많이 작성하고 더 많은 사람과 코드를 공유하게 되면 일관된 스타일이 얼마나 중요한지 알게 될 것입니다. 그리고 styler 패키지를 잊지 마세요: 스타일이 엉망인 코드의 품질을 빠르게 향상시키는 훌륭한 방법입니다.

다음 장에서는 타이디 데이터(tidy data)에 대해 배우며 데이터 과학 도구로 다시 전환합니다. 타이디 데이터는 데이터 프레임을 구성하는 일관된 방법으로 tidyverse 전반에 걸쳐 사용됩니다. 일단 타이디 데이터를 얻게 되면 대부분의 tidyverse 함수들과 바로 호환되어 여러분의 삶을 더 쉽게 만들어주기 때문에 이러한 일관성은 중요합니다. 물론 삶이 항상 쉽지만은 않으며 실전에서 만나는 대부분의 데이터셋은 이미 타이디한 상태가 아닐 것입니다. 그래서 우리는 여러분의 지저분한 데이터를 타이디하게 만들기 위해 tidyr 패키지를 사용하는 방법도 가르쳐 드릴 것입니다.

<sup>[1](ch04.html#idm44771324908448-marker)</sup> `dep_time`이 `HMM` 또는 `HHMM` 형식이므로 정수 나눗셈(`%/%`)을 사용하여 시간을 구하고 나머지(모듈로 연산, `%%`)를 사용하여 분을 구합니다.
