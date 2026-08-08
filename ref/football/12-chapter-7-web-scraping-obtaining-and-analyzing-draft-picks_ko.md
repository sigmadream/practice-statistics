# 7장. 웹 스크래핑: 드래프트 픽 수집 및 분석 (Chapter 7. Web Scraping: Obtaining and Analyzing Draft Picks)

미식축구 대중 분석에서 가장 큰 성과 중 하나는 `nflscrapR`과 그 이후의 `nflfastR`입니다. 이러한 패키지를 사용하면 우리가 모두 사랑하는 게임을 쉽게 분석할 수 있습니다. 컴퓨팅 공간에 데이터를 포함하는 것은 종종 Python이나 R에서 패키지를 다운로드하는 것만큼 간단하며, 바로 시작할 수 있습니다.

하지만 때로는 그렇게 쉽지 않을 수도 있습니다. 종종 여러분이 직접 웹에서 데이터를 *스크래핑(scrape)*해야(컴퓨터 프로그램을 사용하여 데이터 다운로드) 할 필요가 있습니다. Python과 R의 웹 스크래핑에 대한 모든 것을 알려주는 것은 이 책의 범위를 벗어나지만, 몇 가지 매우 쉬운 명령을 통해 분석할 수 있는 상당량의 데이터를 얻을 수 있습니다.

이 장에서는 [프로 풋볼 레퍼런스(Pro Football Reference)](https://www.pro-football-reference.com)에서 NFL 드래프트(NFL Draft) 및 NFL 스카우팅 콤바인(NFL Scouting Combine) 데이터를 스크래핑할 것입니다. 이 사이트는 펜실베이니아주 필라델피아에서 제공하는 훌륭한 리소스입니다. 상상할 수 있는 모든 스포츠에 대한 무료 데이터를 제공하는 Sports Reference가 소유하고 있습니다. 이 웹사이트를 사용하여 NFL 드래프트 및 NFL 스카우팅 콤바인에 대한 데이터를 얻을 것입니다.

*NFL 드래프트(NFL Draft)*는 매년 전국의 여러 도시에서 열리는 행사입니다. 드래프트에서 각 팀은 고등학교 졸업 후 최소 3년이 지난 선수 풀에서 선수를 선발합니다. 예전에는 라운드가 더 많았지만 현재 NFL 드래프트는 7라운드로 구성되어 있습니다. 각 라운드의 드래프트 순서는 각 팀이 전년도에 얼마나 잘했는지에 따라 결정됩니다. 약팀이 강팀보다 드래프트에서 더 높은 순위로 선발합니다. 팀들은 드래프트 픽을 다른 드래프트 픽이나 선수와 교환할 수 있습니다.

*NFL 스카우팅 콤바인(NFL Scouting Combine)*은 인디애나주 인디애나폴리스에서 매년 열리는 행사입니다. 콤바인에서는 NFL 드래프트 자격이 있는 운동선수 풀이 NFL 팀의 평가자들과 만나 다양한 신체적, 심리적 테스트를 수행합니다. 또한 이것은 일반적으로 팀과 에이전트 간의 거래가 시작되고 때로는 마무리되는 NFL의 연례 회의로 간주됩니다.

이 두 데이터 세트의 조합은 두 가지 이유로 풋볼 분석 초보자에게 훌륭한 리소스입니다. 첫째, 이 데이터는 1년에 한 번 며칠에 걸쳐 수집되며 그 이후에는 변경되지 않습니다. 일부 선수가 나중에 신체 검사를 다시 받을 수도 있고 선수들이 여러 가지 이유로 자신을 지명한 팀을 떠나는 경우도 종종 발생하지만, 드래프트 팀은 바뀔 수 없습니다. 따라서 한 번 데이터를 얻고 나면 일반적으로 1년 내내 유용하게 사용할 수 있으며, 그 이후에는 다음 해에 데이터를 얻었을 때 단순히 새 데이터를 추가하기만 하면 됩니다.

2022년의 모든 NFL 스카우팅 콤바인과 NFL 드래프트 데이터를 스크래핑하는 것으로 시작하여 이후 분석을 위해 다음 연도 데이터를 추가할 것입니다.

###### 팁 (Tip)

웹 스크래핑에는, 특히 처음 시작할 때 많은 시행착오가 따릅니다. 일반적으로 제대로 작동하는 예제를 찾은 다음 필요한 결과를 얻을 때까지 한 번에 하나씩 부분을 변경합니다.

# Python으로 웹 스크래핑하기 (Web Scraping with Python)

###### 팁 (Tip)

웹 스크래핑을 시작하기 전에 먼저 웹 페이지로 이동하여 다운로드하려는 내용을 확인하세요.

다음 코드를 사용하면 `for` 루프를 사용하여 Python으로 스크래핑할 수 있습니다. 장을 건너뛰었거나 복습이 필요한 경우, <a href="ch06.html#sec-chp-6-ipmm" data-type="xref">"개별 선수 시장 및 모델링(Individual Player Markets and Modeling)"</a>에서 `for` 루프에 대한 소개를 제공합니다. *URL(Uniform Resource Locator)* 또는 웹 주소를 `url` 객체에 저장합니다. 이 경우 URL은 2022년 NFL 드래프트의 URL일 뿐입니다.

다음으로, `pandas` 패키지의 `read_html()`을 사용하여 주어진 URL에서 테이블을 읽어오기만 하면 됩니다. Python은 0부터 계산을 시작한다는 것을 기억하세요. 따라서 `read_html()`에서 가져온 `draft_py` 데이터프레임의 0번째 요소는 단순히 웹 페이지의 첫 번째 테이블입니다. 또한 `NA` 초안 대략적인 값(draft approximate values)을 `0`으로 변경해야 합니다.

```
## Python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

url = "https://www.pro-football-reference.com/years/2022/draft.htm"

draft_py = pd.read_html(url, header=1)[0]
draft_py.loc[draft_py["DrAV"].isnull(), "DrAV"] = 0
```

`print()`를 사용하여 데이터를 살짝 엿볼 수 있습니다.

```
## Python
print(draft_py)
```

결과는 다음과 같습니다.

Rnd Pick Tm Sk College/Univ Unnamed: 28 0 1 1 JAX 3.5 Georgia College Stats 1 1 2 DET 9.5 Michigan College Stats 2 1 3 HOU 1.0 LSU College Stats 3 1 4 NYJ NaN Cincinnati College Stats 4 1 5 NYG 4.0 Oregon College Stats .. .. ... ... ... ... ... 263 7 258 GNB NaN Nebraska College Stats 264 7 259 KAN NaN Marshall College Stats 265 7 260 LAC NaN Purdue College Stats 266 7 261 LAR NaN Michigan St. College Stats 267 7 262 SFO NaN Iowa St. College Stats \[5871 rows x 31 columns\]

###### 경고 (Warning)

웹 스크래핑을 할 때 웹사이트를 너무 많이 *조회(hit)*하거나 웹사이트에서 데이터를 너무 많이 가져오지 않도록 주의하세요. 웹사이트에서 차단당할 수도 있습니다. 이런 경우 다시 시도할 때까지 잠시 기다려야 합니다. 또한 많은 웹사이트에는 웹페이지를 스크래핑할 수 있는지 여부와 스크래핑 방법에 대한 지침을 제공하는 규정(보다 공식적으로는 *이용 약관(Terms & Conditions)*으로 알려짐)이 있습니다.

다소 조잡하지만 이 웹 스크래핑 프로세스는 실행 가능합니다! 여러 연도(예: 2000~2022년)를 스크래핑하려면 간단한 `for` 루프를 사용할 수 있으며, 이는 종종 데이터의 체계적인 변경 덕분에 가능합니다. 실험이 핵심입니다.

###### 팁 (Tip)

웹 페이지에서 여러 번 데이터를 가져오는 것을 피하기 위해 책을 집필할 때 파일을 캐시해두고 필요할 때만 다운로드했습니다. 예를 들어 이 장의 앞부분에서 이 코드를 사용했습니다(그리고 독자인 여러분에게는 숨겼습니다).

```
## Python
import pandas as pd
import os.path

file_name = "draft_demo_py.csv"

if not os.path.isfile(file_name):
    ## Python
    url = \
        "https://www.pro-football-reference.com/" + \
        "years/2022/draft.htm"
    draft_py = pd.read_html(url, header=1)[0]

    conditions = [
        (draft_py.Tm == "SDG"),
        (draft_py.Tm == "OAK"),
        (draft_py.Tm == "STL"),
    ]
    choices = ["LAC", "LV", "LAR"]

    draft_py["Tm"] = \
        np.select(conditions, choices,
                  default = draft_py.Tm)
    draft_py.loc[draft_py["DrAV"].isnull(), "DrAV"] = 0
    draft_py.to_csv(file_name)
else:
    draft_py = pd.read_csv(file_name)
    draft_py.loc[draft_py["DrAV"].isnull(), "DrAV"] = 0
```

또한 자체적으로 `for` 루프를 생성할 때 종종 간단한 인덱스 값(예: `i = 1`로 설정)으로 시작한 다음 코드가 작동하도록 만듭니다. 코드가 제대로 작동하게 한 후에 코드를 여러 값에 대해 실행하도록 `for` 줄을 추가합니다.

###### 경고 (Warning)

`for` 루프를 작성하는 동안 인덱스 값을 `1`과 같은 값으로 설정할 때 코드에서 자리 표시자 인덱스(예: `i = 1`)를 제거해야 합니다. 그렇지 않으면 루프는 동일한 함수나 데이터에 대해 여러 번 실행될 뿐입니다. 저희도 인정하고 싶지 않지만 코딩할 때 이런 실수를 저지른 적이 있습니다.

이제 Python에서 더 많은 데이터를 다운로드해 보겠습니다. 이 프로세스의 일부로 데이터를 정리해야 합니다. 여기에는 `pandas`에 어떤 행이 헤더인지 알려주는 것이 포함됩니다. 이 경우 두 번째 행 또는 `header=1`입니다. Python은 0부터 계산하므로 1은 두 번째 항목에 해당한다는 점을 기억하세요. 마찬가지로 루프의 일부로 `season`을 자체 열에 저장해야 합니다. 또한 추가 헤딩 정보가 포함된 행을 제거합니다(이상하게도 데이터 세트의 일부 행은 데이터 헤더의 중복입니다). 이렇게 하려면 값이 열 이름과 같지 않은 행만 저장합니다(예: `tm != "Tm"` 사용).

```
## Python
draft_py = pd.DataFrame()
for i in range(2000, 2022 + 1):
    url = "https://www.pro-football-reference.com/years/" + \
           str(i) + \
           "/draft.htm"
    web_data = pd.read_html(url, header=1)[0]
    web_data["Season"] = i
    web_data = web_data.query('Tm != "Tm"')
    draft_py = pd.concat([draft_py, web_data])

draft_py.reset_index(drop=True, inplace=True)
```

일부 팀은 지난 10년 동안 연고지를 이동했으므로 새 위치를 반영하도록 팀 이름(`Tm`)을 변경해야 합니다. 이를 위해 `np.select()` 함수를 사용할 수 있으며, 여기에는 이전 이름을 가진 `conditions`과 새 이름을 가진 `choices`가 사용됩니다. `np.select()`의 기본값도 변경하여 연고지를 옮기지 않은 팀은 동일하게 유지되도록 해야 합니다.

```
## Python
# 차저스(Chargers)는 샌디에이고에서 로스앤젤레스로 연고지를 옮겼습니다
# 레이더스(Raiders)는 오클랜드에서 라스베이거스로 연고지를 옮겼습니다
# 램스(Rams)는 세인트루이스에서 로스앤젤레스로 연고지를 옮겼습니다
conditions = [
    (draft_py.Tm == "SDG"),
    (draft_py.Tm == "OAK"),
    (draft_py.Tm == "STL"),
]
choices = ["LAC", "LVR", "LAR"]

draft_py["Tm"] = \
    np.select(conditions, choices, default = draft_py.Tm)
```

마지막으로 인덱스를 재설정하고 파일을 저장하기 전에 누락된 초안 대략적 가치(DrAV)를 0으로 대체합니다.

```
## Python
draft_py.loc[draft_py["DrAV"].isnull(), "DrAV"] = 0
draft_py.to_csv("data_py.csv", index=False)
```

이제 데이터를 살펴볼 수 있습니다.

```
## Python
print(draft_py.head())
```

결과는 다음과 같습니다.

Unnamed: 0 Rnd Pick Tm ... Sk College/Univ Unnamed: 28 Season 0 0 1 1 CLE ... 19.0 Penn St. College Stats 2000 1 1 1 2 WAS ... 23.5 Penn St. College Stats 2000 2 2 1 3 WAS ... NaN Alabama College Stats 2000 3 3 1 4 CIN ... NaN Florida St. College Stats 2000 4 4 1 5 BAL ... NaN Tennessee College Stats 2000 \[5 rows x 31 columns\]

사용할 수 있는 다른 열을 살펴보겠습니다.

```
## Python
print(draft_py.columns)
```

결과는 다음과 같습니다.

Index(\['Unnamed: 0', 'Rnd', 'Pick', 'Tm', 'Player', 'Pos', 'Age', 'To', 'AP1', 'PB', 'St', 'wAV', 'DrAV', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Solo', 'Int.1', 'Sk', 'College/Univ', 'Unnamed: 28', 'Season'\], dtype='object')

###### 팁 (Tip)

R과 Python의 경우 보통 컴퓨터에 업데이트 내용을 저장하도록 지시해야 합니다. 따라서 `draft_py = draft_py.drop(labels = 0, axis = 0)`처럼 동일한 이름 위에 객체를 저장하는 경우가 많습니다. 일반적으로 원하거나 필요한 데이터를 나중에 삭제하지 않도록 이 복사 동작을 이해하세요.

마지막으로, 분석 목적으로 관리할 데이터에 대한 메타데이터 또는 *데이터 사전(data dictionary)* (데이터에 관한 데이터)은 다음과 같습니다.

- 선수가 드래프트된 시즌 (`Season`)

- 몇 번째로 지명되었는지 (`Pick`)

- 선수를 지명한 팀 (`Tm`)

- 선수 이름 (`Player`)

- 선수 포지션 (`Pos`)

- 선수의 전체 통산 대략적 가치 (`wAv`)

- 지명 팀에 대한 선수의 대략적 가치 (`DrAV`)

마지막으로 특정 열의 순서를 변경하고 선택하고 싶을 수 있습니다. 예를 들어 6개의 열만 원하고 그 순서를 변경하고 싶을 수 있습니다.

```
## Python
draft_py_use = \
    draft_py[["Season", "Pick", "Tm", "Player", "Pos", "wAV", "DrAV"]]

print(draft_py_use)
```

결과는 다음과 같습니다.

Season Pick Tm Player Pos wAV DrAV 0 2000 1 CLE Courtney Brown DE 27.0 21.0 1 2000 2 WAS LaVar Arrington LB 46.0 45.0 2 2000 3 WAS Chris Samuels T 63.0 63.0 3 2000 4 CIN Peter Warrick WR 27.0 25.0 4 2000 5 BAL Jamal Lewis RB 69.0 53.0 ... ... ... ... ... ... ... ... 5866 2022 258 GNB Samori Toure WR 1.0 1.0 5867 2022 259 KAN Nazeeh Johnson SAF 1.0 1.0 5868 2022 260 LAC Zander Horvath RB 0.0 0.0 5869 2022 261 LAR AJ Arcuri OT 1.0 1.0 5870 2022 262 SFO Brock Purdy QB 6.0 6.0 \[5871 rows x 7 columns\]

나중에 사용하기 위해 이 데이터를 로컬에 저장하는 것이 좋습니다. 사용할 때마다 데이터를 다운로드하고 정리하고 싶지는 않을 것입니다.

###### 경고 (Warning)

웹 스크래핑을 통해 얻은 데이터는 당사의 예제 데이터와 다를 수 있습니다. 예를 들어 기술 검토자는 *draft pick* 열을 연속적인 정수나 숫자가 아닌 이산적인 문자(discrete character)로 취급했습니다. 데이터가 이상해 보인다면(예: 플롯이 맞지 않는 경우) <a href="ch02.html#sec-EDA-stable" data-type="xref">2장</a>과 다른 장의 도구를 사용하여 데이터를 검사하여 데이터 타입을 확인하세요.

# R로 웹 스크래핑하기 (Web Scraping in R)

###### 경고 (Warning)

일부 Python 및 R 패키지에는 특히 macOS 및 Linux에서 외부 종속성이 필요합니다. 패키지를 설치하려고 할 때 오류 메시지가 나타나면 오류 메시지를 읽어보세요. 보통 오류 메시지가 난해한 경우가 많기 때문에 디버깅을 위해 검색 엔진을 사용하는 경우가 많습니다.

`rvest` 패키지를 사용하여 R에서 유사한 루프를 만들 수 있습니다. 먼저 패키지를 로드하고 빈 `tibble`을 생성합니다.

```
## R
library(janitor)
library(tidyverse)
library(rvest)
library(htmlTable)
library(zoo)

draft_r <- tibble()
```

그런 다음 2000년부터 2022년까지 루프를 돌립니다. `2000:2022`처럼 콜론을 사용하여 범위를 지정할 수 있습니다. 그러나 우리는 더 강력하기 때문에 `seq()` 명령을 명시적으로 사용하는 것을 선호합니다.

Python 코드와 비교하여 R 코드의 주요 차이점은 파이프(`|>`)를 사용하여 `html_nodes` 명령을 호출한다는 것입니다. 또한 원시 `web_data`에서 웹 데이터프레임(`web_df`)을 추출해야 합니다. `row_to_names()` 함수는 빈 헤더 행을 정리하고 데이터의 헤더를 첫 번째 행으로 바꿉니다. 일부 열에 중복된 이름이 있기 때문에 `janitor::clean_names()` 함수가 열 이름을 정리합니다. `mutate()` 함수는 루프의 시즌을 저장합니다.

다음으로, 데이터에 `filter()`를 사용하여 중복 헤더가 추가 행으로 포함된 행을 제거합니다.

```
## R

for (i in seq(from = 2000, to = 2022)) {
    url <- paste0(
        "https://www.pro-football-reference.com/years/",
        i,
        "/draft.htm"
    )
    web_data <-
        read_html(url) |>
        html_nodes(xpath = '//*[@id="drafts"]') |>
        html_table()
    web_df <-
        web_data[[1]]
    web_df_clean <-
        web_df |>
        janitor::row_to_names(row_number = 1) |>
        janitor::clean_names(case = "none") |>
        mutate(Season = i) |> # 시즌 추가
        filter(Tm != "Tm") # 추가 열 헤더 모두 제거

    draft_r <-
        bind_rows(
            draft_r,
            web_df_clean
        )
}
```

출력을 저장하기 전에 `case_when()`을 사용하여 이동한 팀을 반영하도록 팀(`Tm`)의 이름을 변경합니다. 그러면 다시 다운로드할 필요가 없습니다. 또한 이렇게 하면 R을 위해 데이터가 더 정리되기 때문에 데이터를 저장하고 다시 로드합니다.

```
## R
# 차저스(chargers)는 샌디에이고에서 로스앤젤레스로 연고지를 옮겼습니다
# 레이더스(Raiders)는 오클랜드에서 라스베이거스로 연고지를 옮겼습니다
# 램스(Rams)는 세인트루이스에서 로스앤젤레스로 연고지를 옮겼습니다
draft_r <-
    draft_r |>
    mutate(Tm = case_when(Tm == "SDG" ~ "LAC",
                          Tm == "OAK" ~ "LVR",
                          Tm == "STL" ~ "LAR",
                          TRUE ~ Tm),
            DrAV = ifelse(is.na(DrAV), 0, DrAV))
write_csv(draft_r, "draft_data_r.csv")
draft_r <- read_csv( "draft_data_r.csv")
```

이제 데이터가 준비되었으므로 `select()`를 사용하여 나중에 분석에 필요한 데이터를 가져옵니다.

```
## R
draft_r_use <-
    draft_r |>
    select(Season, Pick, Tm, Player, Pos, wAV, DrAV)

print(draft_r_use)
```

결과는 다음과 같습니다.

\# A tibble: 5,871 × 7 Season Pick Tm Player Pos wAV DrAV \<dbl\> \<dbl\> \<chr\> \<chr\> \<chr\> \<dbl\> \<dbl\> 1 2000 1 CLE Courtney Brown DE 27 21 2 2000 2 WAS LaVar Arrington LB 46 45 3 2000 3 WAS Chris Samuels T 63 63 4 2000 4 CIN Peter Warrick WR 27 25 5 2000 5 BAL Jamal Lewis RB 69 53 6 2000 6 PHI Corey Simon DT 45 41 7 2000 7 ARI Thomas Jones RB 62 7 8 2000 8 PIT Plaxico Burress WR 70 34 9 2000 9 CHI Brian Urlacher HOF LB 119 119 10 2000 10 BAL Travis Taylor WR 30 23 \# ℹ 5,861 more rows

###### 참고 (Note)

두 웹 스크래핑 방법을 비교해 보세요. Python 함수는 더 자립적인(self-contained) 경향이 있으며 객체에 속하는 경향이 있습니다(Python에서 객체에 속하는 함수를 *메서드(methods)*라고 합니다). 반대로 R은 동일한 객체에 여러 함수를 사용하는 경향이 있습니다. 이것은 언어의 설계 특성입니다. Python은 객체 지향 언어(object-oriented language)에 가까운 반면, R은 함수형 언어(functional language)에 가깝습니다. 당신은 어떤 스타일이 더 마음에 드십니까?

여기서 이 장의 분석에는 영향을 주지 않지만 이 데이터로 계속 작업을 진행하려는 경우 영향을 줄 수 있는 한 가지 사항을 발견할 수 있습니다. 2000년 NFL 드래프트의 9번째 지명 이름은 *Brian Urlacher HOF*입니다. *HOF* 부분은 그가 결국 NFL 명예의 전당(Hall of Fame)에 입성했음을 나타냅니다. 이 데이터를 사용하여 다른 데이터 세트와 병합하려는 경우 그러한 세부 정보가 제외되도록 이름을 변경해야 합니다.

# NFL 드래프트 분석하기 (Analyzing the NFL Draft)

NFL 드래프트는 매년 열리며 팀이 자격을 갖춘 선수를 선발할 수 있게 해줍니다. 이 행사는 모든 팀이 경쟁력을 유지하고 재능 있는 선수를 확보할 수 있도록 하기 위한 방법으로 1936년에 시작되었습니다. 드래프트가 진행되는 동안 각 팀은 라운드당 한 장의 지명권(pick)을 얻습니다. 각 라운드의 순서는 팀의 기록을 기반으로 하며, 기록이 같은 팀에 대한 타이브레이크 규칙(tie-breaking rules)이 적용됩니다. 따라서 슈퍼볼에서 우승한 팀이 가장 늦게 선발하고 슈퍼볼에서 패배한 팀이 끝에서 두 번째로 선발합니다. 하지만 팀들은 종종 선수를 트레이드하고 거래의 일부로 드래프트 픽을 포함하기도 합니다. 따라서 픽에는 사람들이 정량화하고 이해하고 싶어 하는 추가적인 가치가 있을 수 있습니다.

드래프트에 대해 많은 흥미로운 질문을 던질 수 있습니다. 특히 (판타지 팀이나 실제 팀을 위해) 선수를 드래프트하는 경우라면 더욱 그렇습니다. 각 드래프트 픽의 가치는 얼마(그리고 어떤 단위)입니까? 어떤 팀은 다른 팀보다 선수 드래프트에 더 능숙합니까? 드래프트에서 특정 포지션이 다른 포지션보다 더 나은 도박(gambles)입니까?

언뜻 보기에 이 데이터로 대답하기 가장 간단한 질문은 첫 번째 질문, 즉 각 드래프트 픽에 가치를 할당하는 "각 드래프트 픽의 가치는 얼마(그리고 어떤 단위)인가?"입니다. 이것이 중요한 이유는 팀들이 드래프트 픽의 효용성(utility)을 팀의 현재 요구 사항과 일치시키기 위해 종종 서로 픽을 트레이드하기 때문입니다.

예를 들어 2018년 드래프트에서 전체 6순위 지명권을 보유하고 있던 뉴욕 제츠(New York Jets)는 2018년 3순위 픽을 얻기 위해 2018년 37순위, 49순위 픽, 2019년 2라운드 픽과 함께 해당 픽을 인디애나폴리스 콜츠(Indianapolis Colts)로 트레이드했습니다. 최고 드래프트 픽은 일반적으로 쿼터백을 위해 예약되어 있으며 제츠는 FA 시장에서 커크 커즌스(Kirk Cousins) 영입전에서 패배한 후 쿼터백이 필요했습니다. 콜츠가 2018년 전체 3순위 지명권을 "얻은" 주된 이유는 2017년에 프랜차이즈 쿼터백인 앤드류 럭(Andrew Luck)이 어깨 부상에서 회복 중이었기 때문에 팀이 어려움을 겪었기 때문입니다. 즉, 전체 3순위 지명권은 콜츠에게는 제츠에게 갖는 효용성보다 낮았기 때문에 두 팀이 트레이드를 성사시킨 것입니다.

팀들은 이러한 픽의 "공정한(fair)" 시장 가치가 무엇인지 어떻게 결정할까요? 1989년으로 돌아가서, 아칸소주 석유 백만장자인 제리 존스(Jerry Jones)가 매입한 후 댈러스 카우보이스(Dallas Cowboys)는 오랫동안 명예의 전당에 오른 헤드 코치 톰 랜드리(Tom Landry)를 해임하고 대학 코치인 지미 존슨(Jimmy Johnson)으로 교체했습니다. 카우보이스는 1988년에 3승 시즌을 보냈고 로스터에는 차이를 만들어낼(difference-making) 선수가 상대적으로 부족했습니다. 전설에 따르면 존슨은 조깅을 하던 중 스타 러닝백(이자 미래의 미국 상원의원 후보)인 허셜 워커(Herschel Walker)를 결국 3개의 1라운드 픽, 3개의 2라운드 픽, 1개의 3라운드 픽, 1개의 6라운드 픽, 그리고 몇 명의 선수로 이루어진 패키지와 트레이드하기로 결정했다고 합니다.

카우보이스는 1989년을 1승 15패라는 NFL 최악의 성적으로 마감했고, 이로 인해 남아있던 본래 드래프트 픽의 순위가 상승했습니다(팀은 1990년 1라운드 픽을 보충 드래프트에서 쿼터백을 위해 일찍 사용했습니다). 존슨은 임기 초반에 마이크 맥코이(Mike McCoy)가 만든 가치 차트(현재는 각 라운드의 각 픽 가치를 일치시키도록 설계된 *지미 존슨 차트(Jimmy Johnson chart)*로 알려져 있음)를 사용하여 게임 내의 다른 어떤 코치나 경영진보다 많은 드래프트 픽 트레이드를 성사시켰습니다. 이 차트는 전체 1순위 지명권에 3,000점을 할당하며, 이후 각 지명권의 가치는 기하급수적으로(exponentially) 떨어집니다.

지미 존슨 차트는 여전히 많은 NFL 팀들이 선호하는 차트이지만, 이후의 픽에 비해 드래프트 최상위 픽을 과대평가하는 것으로 나타났습니다. 실제로 노벨상을 수상한 경제학자이자 스포츠 분석의 전설인 케이드 매시(Cade Massey)는 ["The Loser's Curse: Decision Making and Market Efficiency in the National Football League Draft(패자의 저주: NFL 드래프트에서의 의사 결정 및 시장 효율성)"](https://oreil.ly/vFGt6)에서 각 픽과 계약하는 비용을 고려할 때 첫 번째 픽의 잉여 가치(surplus value), 즉 선발된 선수가 받는 급여에 대한 해당 픽의 경기장 내(on-field) 가치는 첫 번째 픽에 의해 극대화되는 것이 아니라 1라운드 중간이나 끝, 또는 2라운드 초반 픽에서 극대화된다는 것을 보여주었습니다.

이 주제에 대한 많은 논평이 존재하며, 그 중 일부는 유용하고 일부는 코드를 통한 대화형(interactive)입니다. 에릭의 PFF 시절 전 동료인 티모 리스케(Timo Riske)는 "드래프트 픽의 잉여 가치(the surplus value of draft picks)"에 대한 [기사](https://oreil.ly/7J9HO)를 썼습니다. 또한 `nflfastR`을 만드는 데 도움을 주고 훨씬 더 얕은 드래프트 곡선을 만든 바로 그 벤 볼드윈(Ben Baldwin)과 [마이클 로페즈(Michael Lopez)](https://oreil.ly/e0mMH), [벤 볼드윈(Ben Baldwin)](https://oreil.ly/eR4Ra)을 포함한 다른 사람들도 이 주제에 대한 연구를 수행했습니다.

이 섹션에서는 각 선수가 자신을 드래프트한 팀을 위해 창출한 대략적 가치, 즉 *드래프트 대략적 가치(draft approximate value, DrAV)*를 사용하여 그 연구를 재현하는 것을 목표로 합니다. 이를 플롯해 보면 팀들이 드래프트 초반에 더 나은 픽을 얻기 때문에 미래의 선수를 드래프트하는 팀들이 시장 효율성(market efficiency)을 가지고 있음을 분명히 알 수 있습니다(여기에는 팀이 높은 드래프트 픽을 더 많이 출전시킨다는 약간의 편향이 존재하지만, 드래프트 슬롯에 따라 플레이당 효율성도 떨어진다는 것을 보여줄 수 있습니다).

먼저 2019년 이전의 연도를 선택합니다. 2019년 이후의 연도를 걸러내는 이유는 그 선수들이 아직 루키 계약 기간 동안 뛰고 있으며 다음 팀과 자유롭게 계약할 기회가 없었기 때문입니다.

그런 다음 각 픽을 평균 드래프트 가치와 비교하기 위해 데이터를 플롯합니다. Python에서는 이 코드를 사용하여 <a href="#fig-py-pick-drav-2" data-type="xref">그림 7-1</a>을 만듭니다.

```
## Python
# 장(chapter)에 맞게 테마 변경
sns.set_theme(style="whitegrid", palette="colorblind")

draft_py_use_pre2019 =     draft_py_use    .query("Season <= 2019")

## 열을 숫자형 또는 정수형으로 형식화
draft_py_use_pre2019 =     draft_py_use_pre2019    .astype({"Pick": int, "DrAV": float})

sns.regplot(data=draft_py_use_pre2019,
            x="Pick",
            y="DrAV",
            line_kws={"color": "red"},
            scatter_kws={'alpha':0.2});
plt.show();
```

<figure>
<img src="D:\sd\Practicesny2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0701.png" />
<h6 id="figure-7-1.-scatterplot-with-a-linear-trendline-for-draft-pick-number-against-draft-approximate-value-plotted-with-seaborn">그림 7-1. <code>seaborn</code>으로 플롯한, 드래프트 대략적 가치 대비 드래프트 픽 번호에 대한 선형 추세선을 포함한 산점도</h6>
</figure>

R에서는 이 코드를 사용하여 <a href="#fig-r-pick-drav" data-type="xref">그림 7-2</a>를 만듭니다.

```
## R
draft_r_use_pre2019 <-
    draft_r_use |>
    mutate(DrAV = as.numeric(DrAV),
           wAV = as.numeric(wAV),
           Pick = as.integer(Pick)) |>
    filter(Season <= 2019)

ggplot(draft_r_use_pre2019, aes(Pick, DrAV)) +
    geom_point(alpha = 0.2) +
    stat_smooth() +
    theme_bw()
```

<figure>
<img src="D:\sd\Practicesny2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0702.png" />
<h6 id="figure-7-2.-scatterplot-with-a-smoothed-spline-trendline-for-draft-pick-number-against-draft-approximate-value-plotted-with-ggplot2">그림 7-2. <code>ggplot2</code>로 플롯한, 드래프트 대략적 가치 대비 드래프트 픽 번호에 대한 평활 스플라인 추세선을 포함한 산점도</h6>
</figure>

그림 <a href="#fig-py-pick-drav-2" data-type="xref" data-xrefstyle="select:labelnumber">7-1</a> 및 <a href="#fig-r-pick-drav" data-type="xref" data-xrefstyle="select:labelnumber">7-2</a>는 픽 번호가 커질수록 픽의 가치가 감소함을 보여줍니다.

이제 이 곡선을 도출하려고 할 때 제기되는 진짜 질문은 팀들이 드래프트할 때 무엇을 찾고 있는가입니다. 그들은 해당 픽에서 산출되는 평균(average) 가치를 찾고 있습니까? 그 픽에서 산출되는 중앙값(median value)을 찾고 있습니까? 아니면 다른 백분위수(percentile)입니까?

후반 픽에 대한 중앙값은 0을 산출할 가능성이 높으며(팀들이 항상 후반 픽을 트레이드하기 때문에 이는 분명 사실이 아님), 일부 팀에 의한 드래프트 후반의 대박(hits) 때문에 평균(mean) 역시 후반 픽을 과대평가할 수 있습니다. 통계적으로 픽의 50% 이상이 0의 가치를 갖는다면 중앙값은 0입니다. 하지만 평균은 드래프트 후반에 뽑힌 훌륭한 소수의 선수들에 의해 영향을 받을 수 있습니다. 예를 들어 패트리어츠(Patriots)는 2000년 드래프트에서 199번째 픽으로 톰 브래디(Tom Brady)를 선택했고, 그는 역대 최고의 미식축구 선수 중 한 명이 되었습니다. 코크 게인즈(Cork Gaines)의 [비즈니스 인사이더(Business Insider) 기사](https://oreil.ly/wqcHe)는 이 픽의 뒷이야기를 전해줍니다.

지금은 평균(mean)을 사용하지만, 연습문제에서는 중앙값(median)을 사용하여 변경되는 것이 있는지 확인하게 됩니다. <a href="ch09.html#sec-quant-reg" data-type="xref">"분위수 회귀(Quantile Regression)"</a>는 다른 모델 유형을 탐구하고 싶은 경우 살펴볼 가치가 있는 또 다른 유형의 회귀인 분위수 회귀에 대한 간략한 개요를 제공합니다.

###### 참고 (Note)

시계열(series)이 포함된 데이터 세트(예: 일일 온도 또는 드래프트 픽)에는 종종 패턴과 노이즈가 모두 포함되어 있습니다. 이 노이즈를 평활화하는(smooth out) 한 가지 방법은 여러 순차적인 관측값에 대한 평균을 계산하는 것입니다. 이 평균을 흔히 *이동 평균(rolling average, moving mean, running average)*이라고 부르며 평균을 나타내는 다른 유사한 용어들이 사용되기도 합니다. 이동 평균의 주요 입력값에는 *창(window)* (사용할 입력 개수), 방법(예: 평균 또는 중앙값), 시계열의 시작과 끝을 어떻게 처리할지(예: 전체 창이 없는 첫 번째 항목을 삭제해야 하는지 아니면 다른 규칙을 사용해야 하는지)가 포함됩니다.

각 픽에 대한 가치를 평활화(smooth out)하려면 먼저 각 픽에 대한 평균 가치를 계산하세요. 일부 낮은 픽에는 `NaN` 값이 있으므로 이러한 값을 `0`으로 바꿉니다. 그런 다음 각 픽의 평균 가치를 둘러싼 6픽 이동 평균(moving mean) `DrAV`를 계산합니다(즉, `13`개의 창에 대해 각각의 `DrAV` 값과 이전 `6`개 및 이후 `6`개를 사용). 또한 `rolling()` 함수에 `min_periods=1`을 사용하고 평균을 중앙에 배치(`center`)하도록 지시합니다(이동 평균이 현재 `DrAV`를 중심으로 함). 마지막으로 `pick`을 기준으로 `groupby()`를 수행한 다음 각 픽 위치에 대한 평균 `DrAV`를 계산합니다. Python에서는 이 코드를 사용합니다.

```
## Python
draft_chart_py =     draft_py_use_pre2019    .groupby(["Pick"])    .agg({"DrAV": ["mean"]})

draft_chart_py.columns =     list(map("_".join, draft_chart_py.columns))

draft_chart_py.loc[draft_chart_py.DrAV_mean.isnull()] = 0

draft_chart_py["roll_DrAV"] = (
    draft_chart_py["DrAV_mean"]
    .rolling(window=13, min_periods=1, center=True)
    .mean()
)
```

###### 팁 (Tip)

Python의 연습문제에서는 `rolling()`의 `mean()`을 `agg()`의 일부인 `groupby()`의 `mean()`이 아닌 `median()`과 같은 다른 함수로 변경하려고 할 것입니다.

그런 다음 결과를 플롯하여 <a href="#fig-py-pick-drav" data-type="xref">그림 7-3</a>을 만듭니다.

```
## Python
sns.scatterplot(draft_chart_py, x="Pick", y="roll_DrAV")
plt.show()
```

<figure>
<img src="D:\sd\Practicesny2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0703.png" />
<h6 id="figure-7-3.-scatterplot-for-draft-pick-number-against-draft-approximate-value-seaborn">그림 7-3. 드래프트 대략적 가치 대비 드래프트 픽 번호에 대한 산점도 (<code>seaborn</code>)</h6>
</figure>

R에서는 `group_by()`를 통해 픽을 그룹화하고 `mean()`과 함께 `summarize()`를 수행합니다. `NA` 값을 `0`으로 바꿉니다. 그런 다음 `zoo` 패키지의 `rollapply()` 함수를 사용합니다(이 패키지는 책에서 처음 사용하는 것이므로 패키지를 설치하고 `library(zoo)`를 실행했는지 확인하세요). `rollapply()`와 함께 `width = 13`, 그리고 `mean()` 함수(`FUN = mean`)를 사용합니다.

`mean()` 함수에 `na.rm = TRUE`를 통해 `NA` 값을 무시하도록 지시합니다. 누락된 값을 `NA`로 채우고, 평균을 중앙에 배치(`center`)(이동 평균이 현재 `DrAV`를 중심으로 하도록)하며, 관측값이 13개 미만인 경우(데이터프레임의 시작과 끝 등)에 평균을 계산합니다.

```
## R
draft_chart_r <-
    draft_r_use_pre2019 |>
    group_by(Pick) |>
    summarize(mean_DrAV = mean(DrAV, na.rm = TRUE)) |>
    mutate(mean_DrAV = ifelse(is.na(mean_DrAV),
                              0, mean_DrAV
    )) |>
    mutate(
        roll_DrAV =
            rollapply(mean_DrAV,
                width = 13,
                FUN = mean,
                na.rm = TRUE,
                fill = "extend",
                partial = TRUE
            )
    )
```

###### 팁 (Tip)

R의 연습문제에서는 `summarize()`의 일부인 `group_by()`의 `mean()`이 아닌 `rollapply()`의 `mean()`을 `median()`과 같은 다른 함수로 변경하려고 할 것입니다.

그런 다음 플롯하여 <a href="#fig-r-pick-drav-2" data-type="xref">그림 7-4</a>를 만듭니다.

```
## R
ggplot(draft_chart_r, aes(Pick, roll_DrAV)) +
    geom_point() +
    geom_smooth() +
    theme_bw() +
    ylab("Rolling average (± 6) DrAV") +
    xlab("Draft pick")
```

여기서부터는 결과를 정량화하는 데 도움을 주기 위해 단순히 데이터에 모델을 맞출 수 있습니다. 이 모델을 사용하면 단순히 그림만 조사하는 것이 아니라 숫자를 사용할 수 있습니다. 다양한 모델을 사용할 수 있으며, 일부 모델(예: LOESS 곡선 또는 GAM)은 이 책의 범위를 벗어납니다. 우리는 y절편을 고정한 채 데이터의 로그에 단순 선형 모델을 맞추고, 지수 함수(exponential function)를 사용하여 다시 역변환하게 할 것입니다.

<figure>
<img src="D:\sd\Practicesny2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0704.png" />
<h6 id="figure-7-4.-scatterplot-with-smoothed-trendline-for-draft-pick-number-against-draft-approximate-value-ggplot2">그림 7-4. 드래프트 대략적 가치 대비 드래프트 픽 번호에 대한 평활 추세선을 포함한 산점도 (<code>ggplot2</code>)</h6>
</figure>

###### 경고 (Warning)

`log(0)` 함수는 수학적으로 정의되지 않았기 때문에 사람들은 이 변환이 일어나도록 종종 값을 더하곤 합니다. `1` 또는 `0.1`과 같은 작은 숫자가 자주 사용됩니다. 주의: 이 변환은 때때로 모델 결과를 변경할 수 있으므로, 더할 숫자에 다른 값을 시도해보는 것이 좋습니다.

Python에서는 먼저 (모델을 통해 `Pick`에 접근할 수 있도록) 인덱스를 삭제한 다음 플롯합니다.

```
## Python
draft_chart_py.reset_index(inplace=True)

draft_chart_py["roll_DrAV_log"] =    np.log(draft_chart_py["roll_DrAV"] + 1)

DrAV_pick_fit_py =     smf.ols(formula="roll_DrAV_log ~ Pick",
            data=draft_chart_py)    .fit()

print(DrAV_pick_fit_py.summary())
```

결과는 다음과 같습니다.

OLS Regression Results ============================================================================== Dep. Variable: roll_DrAV_log R-squared: 0.970 Model: OLS Adj. R-squared: 0.970 Method: Least Squares F-statistic: 8497. Date: Sun, 04 Jun 2023 Prob (F-statistic): 1.38e-200 Time: 09:42:13 Log-Likelihood: 177.05 No. Observations: 262 AIC: -350.1 Df Residuals: 260 BIC: -343.0 Df Model: 1 Covariance Type: nonrobust ============================================================================== coef std err t P\>\|t\| \[0.025 0.975\] ------------------------------------------------------------------------------ Intercept 3.4871 0.015 227.712 0.000 3.457 3.517 Pick -0.0093 0.000 -92.180 0.000 -0.010 -0.009 ============================================================================== Omnibus: 3.670 Durbin-Watson: 0.101 Prob(Omnibus): 0.160 Jarque-Bera (JB): 3.748 Skew: 0.274 Prob(JB): 0.154 Kurtosis: 2.794 Cond. No. 304. ============================================================================== Notes: \[1\] Standard Errors assume that the covariance matrix of the errors is correctly specified.

그런 다음 다시 `draft_chart_py`로 병합하고 데이터의 상단을 살펴봅니다.

```
## Python
draft_chart_py["fitted_DrAV"] =     np.exp(DrAV_pick_fit_py.predict()) - 1

draft_chart_py    .head()
```

결과는 다음과 같습니다.

Pick DrAV_mean roll_DrAV roll_DrAV_log fitted_DrAV 0 1 47.60 38.950000 3.687629 31.386918 1 2 39.85 37.575000 3.652604 31.086948 2 3 44.45 37.883333 3.660566 30.789757 3 4 31.15 36.990000 3.637323 30.495318 4 5 43.65 37.627273 3.653959 30.203606

R에서는 다음을 사용합니다.

```
## R
DrAV_pick_fit_r <-
    draft_chart_r |>
    lm(formula = log(roll_DrAV + 1) ~ Pick)

summary(DrAV_pick_fit_r)
```

결과는 다음과 같습니다.

Call: lm(formula = log(roll_DrAV + 1) ~ Pick, data = draft_chart_r) Residuals: Min 1Q Median 3Q Max -0.32443 -0.07818 -0.02338 0.08797 0.34123 Coefficients: Estimate Std. Error t value Pr(\>\|t\|) (Intercept) 3.4870598 0.0153134 227.71 \<2e-16 \*\*\* Pick -0.0093052 0.0001009 -92.18 \<2e-16 \*\*\* --- Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1 Residual standard error: 0.1236 on 260 degrees of freedom Multiple R-squared: 0.9703, Adjusted R-squared: 0.9702 F-statistic: 8497 on 1 and 260 DF, p-value: \< 2.2e-16

그런 다음 다시 `draft_chart_r`로 병합하고 데이터의 상단을 살펴봅니다.

```
## R
draft_chart_r <-
    draft_chart_r |>
    mutate(
        fitted_DrAV =
            pmax(
                 0,
                 exp(predict(DrAV_pick_fit_r)) - 1
            )
    )
draft_chart_r |>
    head()
```

결과는 다음과 같습니다.

\# A tibble: 6 × 4 Pick mean_DrAV roll_DrAV fitted_DrAV \<int\> \<dbl\> \<dbl\> \<dbl\> 1 1 47.6 39.0 31.4 2 2 39.8 37.6 31.1 3 3 44.4 37.9 30.8 4 4 31.2 37.0 30.5 5 5 43.6 37.6 30.2 6 6 34.7 37.4 29.9

요약하자면, 이 섹션에서는 각 픽에 대한 추정 가치를 계산했습니다. 이 추정은 지수 회귀의 2-파라미터(two-parameter) 특성 때문에 드래프트 맨 처음 픽의 가치를 과소평가할 가능성이 높다는 점에 유의하세요. 이는 이전에 언급한 예와 같은 다른 모델 유형으로 개선될 수 있습니다. 단점에도 불구하고 이 추정치를 통해 다음 섹션에서 살펴볼 드래프트 상황을 탐색할 수 있습니다.

# 2018년 제츠/콜츠 트레이드 평가 (The Jets/Colts 2018 Trade Evaluated)

이제 각 드래프트 픽의 가치에 대한 추정치가 있으므로 <a href="#tbl-jets-colts" data-type="xref">표 7-1</a>에서 이 모델이 제츠와 콜츠 간의 트레이드에 대해 어떻게 이야기했을지 살펴보겠습니다.

```
## R
library(kableExtra)
future_pick <-
    tibble(
        Pick = "Future 2nd round",
        Value = "14.8 (discounted at rate of 25%)"
    )

team <- tibble("Receiving team" = c("Jets", rep("Colts", 4)))

tbl_1 <-
    draft_chart_r |>
    filter(Pick %in% c(3, 6, 37, 49)) |>
    select(Pick, fitted_DrAV) |>
    rename(Value = fitted_DrAV) |>
    mutate(
        Pick = as.character(Pick),
        Value = as.character(round(Value, 1))
    ) |>
    bind_rows(future_pick)

team |>
    bind_cols(tbl_1) |>
    kbl(format = "pipe") |>
    kable_styling()
```

| Receiving team | Pick             | Value                            |
|----------------|------------------|----------------------------------|
| Jets           | 3                | 30.8                             |
| Colts          | 6                | 29.9                             |
| Colts          | 37               | 22.2                             |
| Colts          | 49               | 19.7                             |
| Colts          | Future 2nd round | 14.8 (discounted at rate of 25%) |

표 7-1. 제츠와 콜츠 간의 트레이드 {#tbl-jets-colts}

보시다시피 미래의 픽은 25% 할인되는데, 루키 계약이 4년이고 1년을 기다리는 것은 현재 연도 계약의 1/4, 즉 25%이기 때문입니다.

<a href="#tbl-jets-colts" data-type="xref">표 7-1</a>의 값을 합산해 보면 제츠는 이 트레이드에서 예상 55.6 DrAV를 잃으며 털린(fleeced) 것처럼 보입니다. 이는 전체 1순위 지명권의 가치보다 큽니다! 이 값들은 이전에 드래프트를 기반으로 이 장에서 개발한 모델을 사용하여 해당 포지션의 일반적인 드래프트 픽에서 통계적으로 "예상된(expected)" 가치일 뿐입니다. 이제 이 책을 집필하는 시점에서 예측했던 바가 실현되었습니다. 새로운 데이터를 사용하여 선수의 실제 DrAV를 확인하기 위해 <a href="#tbl-jets-colts-trade" data-type="xref">표 7-2</a>를 만들 수 있습니다.

```
## R
library(kableExtra)
```

future_pick <-
    tibble(
        Pick = "Future 2nd round",
        Value = "14.8 (discounted at rate of 25)"
    )

results_trade <-
    tibble(
        Team = c("Jets", rep("Colts", 5)),
        Pick = c(
            3, 6, 37,
            "49-traded for 52",
            "49-traded for 169",
            "52 in 2019"
        ),
        Player = c(
            "Sam Darnold",
            "Quenton Nelson",
            "Braden Smith",
            "Kemoko Turay",
            "Jordan Wilkins",
            "Rock Ya-Sin"
        ),
        "DrAV" = c(25, 55, 32, 5, 8, 11)
    )


results_trade |>
    kbl(format = "pipe") |>
    kable_styling()
```

| Team  | Pick              | Player         | DrAV |
|-------|-------------------|----------------|------|
| Jets  | 3                 | Sam Darnold    | 25   |
| Colts | 6                 | Quenton Nelson | 55   |
| Colts | 37                | Braden Smith   | 32   |
| Colts | 49—traded for 52  | Kemoko Turay   | 5    |
| Colts | 49—traded for 169 | Jordan Wilkins | 8    |
| Colts | 52 in 2019        | Rock Ya-Sin    | 11   |

표 7-2. 제츠와 콜츠 간의 트레이드 결과 {#tbl-jets-colts-trade}

따라서 최종 집계는 제츠 25 DrAV, 콜츠 111로 86 DrAV의 손실이며 이는 전체 1순위 지명권 가치의 거의 세 배에 달합니다!

팀이 쿼터백을 드래프트하기 위해 트레이드업(trade up)을 할 때 항상 이런 식인 것은 아닙니다. 예를 들어, 칩스(Chiefs)는 2017년에 두 개의 1라운드 픽과 한 개의 3라운드 픽을 사용하여 패트릭 마홈스(Patrick Mahomes)를 위해 트레이드업을 단행하여 텍사스 공대(Texas Tech) 출신의 이 야전 사령관(signal caller)을 선택했고, 이는 85 DrAV의 성과를 거두었으며 (2022년 기준) 두 번의 슈퍼볼 우승을 차지했습니다.

드래프트 픽의 가격을 책정하는 더 확실한 방법은 <a href="#sec-chp7-atnd" data-type="xref">"NFL 드래프트 분석하기(Analyzing the NFL Draft)"</a>에서 언급된 출처에서 찾을 수 있습니다. 일반적으로 선수의 루키 계약 이후 첫 번째 계약의 규모와 같은 시장 기반 데이터(market-based data)를 사용하는 것이 업계 표준입니다. 프로 풋볼 레퍼런스(Pro Football Reference)의 DrAV 값은 괜찮은 프록시(proxy)이지만 몇 가지 문제가 있습니다. 즉, 포지션 가치를 제대로 고려하지 못합니다(팀이 다른 포지션에 비해 쿼터백 포지션을 드래프트할 경우 훨씬 더 가치가 높습니다). 드래프트 곡선에 대한 자세한 내용은 에릭의 PFF 시절 전 동료인 브래드 스필버거(Brad Spielberger)와 Over The Cap의 설립자인 제이슨 피츠제럴드(Jason Fitzgerald)가 저술한 *The Drafting Stage: Creating a Marketplace for NFL Draft Picks* (자가 출판, 2020)에서 시작하는 것이 좋습니다.

# 어떤 팀이 다른 팀보다 선수 드래프트를 더 잘합니까? (Are Some Teams Better at Drafting Players Than Others?)

어떤 팀이 다른 팀보다 드래프트를 더 잘하는지에 대한 질문은 드래프트 픽이 팀에 할당되는 방식 때문에 대답하기 어렵습니다. 최고의 팀이 각 라운드의 마지막에 선택하며, 앞서 살펴보았듯이 더 나은 선수가 약한 선수보다 먼저 선택됩니다. 따라서 최악의 팀이 드래프트를 가장 잘하는 팀이라고 잘못 가정할 수 있으며 그 반대의 경우도 마찬가지입니다. 이를 고려하려면 이전에 만든 모델을 사용하여 각 픽에 대한 기대치를 조정해야 합니다. 그렇게 하고 `DrAV`와 `fitted_DrAV` 간의 차이에 대한 평균과 표준 편차를 취하여 2000-2019년 드래프트에 걸쳐 집계하면 Python을 사용하여 다음 순위에 도달합니다.

```
## Python
draft_py_use_pre2019 =     draft_py_use_pre2019    .merge(draft_chart_py[["Pick", "fitted_DrAV"]],
           on="Pick")

draft_py_use_pre2019["OE"] = (
    draft_py_use_pre2019["DrAV"] -
    draft_py_use_pre2019["fitted_DrAV"]
)

draft_py_use_pre2019    .groupby("Tm")    .agg({"OE": ["count", "mean", "std"]})    .reset_index()    .sort_values([("OE", "mean")], ascending=False)
```

결과는 다음과 같습니다.

Tm OE count mean std 26 PIT 161 3.523873 18.878551 11 GNB 180 3.371433 20.063320 8 DAL 160 2.461129 16.620351 1 ATL 148 2.291654 16.124529 21 NOR 131 2.263655 18.036746 22 NWE 176 2.162438 20.822443 13 IND 162 1.852253 15.757658 4 CAR 148 1.842573 16.510813 2 BAL 170 1.721930 16.893993 27 SEA 181 1.480825 16.950089 16 LAC 144 1.393089 14.608528 5 CHI 149 0.672094 16.052031 20 MIN 167 0.544533 13.986365 15 KAN 154 0.501463 15.019527 25 PHI 162 0.472632 15.351785 6 CIN 176 0.466203 15.812953 14 JAX 158 0.182685 13.111672 30 TEN 172 0.128566 12.662670 12 HOU 145 -0.075827 12.978999 28 SFO 184 -0.092089 13.449491 31 WAS 150 -0.450485 9.951758 24 NYJ 137 -0.534640 13.317478 0 ARI 149 -0.601563 14.295335 23 NYG 145 -0.879900 12.471611 29 TAM 153 -0.922181 11.409698 3 BUF 161 -0.985761 12.458855 17 LAR 175 -1.439527 11.985219 19 MIA 151 -1.486282 10.470145 9 DEN 159 -1.491545 12.594449 10 DET 155 -1.765868 12.061696 18 LVR 162 -2.587423 10.217426 7 CLE 170 -3.557266 10.336729

###### 참고 (Note)

Python의 `.reset_index()` 함수는 `pandas`의 데이터프레임에 값을 추가할 때 혼동을 줄 수 있는 행 이름(*index*)이 있기 때문에 우리에게 도움이 됩니다.

또는 R에서는 다음과 같습니다.

```
## R
draft_r_use_pre2019 <-
    draft_r_use_pre2019 |>
    left_join(draft_chart_r |> select(Pick, fitted_DrAV),
        by = "Pick"
    )

draft_r_use_pre2019 |>
    group_by(Tm) |>
    summarize(
        total_picks = n(),
        DrAV_OE = mean(DrAV - fitted_DrAV, na.rm = TRUE),
        DrAV_sigma = sd(DrAV - fitted_DrAV, na.rm = TRUE)
    ) |>
    arrange(-DrAV_OE) |>
    print(n = Inf)
```

결과는 다음과 같습니다.

\# A tibble: 32 × 4 Tm total_picks DrAV_OE DrAV_sigma \<chr\> \<int\> \<dbl\> \<dbl\> 1 PIT 161 3.52 18.9 2 GNB 180 3.37 20.1 3 DAL 160 2.46 16.6 4 ATL 148 2.29 16.1 5 NOR 131 2.26 18.0 6 NWE 176 2.16 20.8 7 IND 162 1.85 15.8 8 CAR 148 1.84 16.5 9 BAL 170 1.72 16.9 10 SEA 181 1.48 17.0 11 LAC 144 1.39 14.6 12 CHI 149 0.672 16.1 13 MIN 167 0.545 14.0 14 KAN 154 0.501 15.0 15 PHI 162 0.473 15.4 16 CIN 176 0.466 15.8 17 JAX 158 0.183 13.1 18 TEN 172 0.129 12.7 19 HOU 145 -0.0758 13.0 20 SFO 184 -0.0921 13.4 21 WAS 150 -0.450 9.95 22 NYJ 137 -0.535 13.3 23 ARI 149 -0.602 14.3 24 NYG 145 -0.880 12.5 25 TAM 153 -0.922 11.4 26 BUF 161 -0.986 12.5 27 LAR 175 -1.44 12.0 28 MIA 151 -1.49 10.5 29 DEN 159 -1.49 12.6 30 DET 155 -1.77 12.1 31 LVR 162 -2.59 10.2 32 CLE 170 -3.56 10.3

놀랍지 않게도 NFL의 유서 깊은 프랜차이즈인 피츠버그 스틸러스(Pittsburgh Steelers), 그린베이 패커스(Green Bay Packers), 댈러스 카우보이스(Dallas Cowboys) 등 일부 팀은 2000년 이후 드래프트 곡선 위에서 가장 좋은 드래프트를 기록했습니다.

또한 이 목록의 마지막 세 팀인 클리블랜드 브라운스(Cleveland Browns), 오클랜드/라스베이거스 레이더스(Oakland/Las Vegas Raiders), 디트로이트 라이언스(Detroit Lions)가 이 글을 쓰는 시점에서 모두 팀의 성공이라는 측면에서 엄청난 가뭄을 겪고 있다는 사실도 누구에게나 놀라운 일이 아닙니다. 레이더스는 2002년 마지막으로 슈퍼볼에 진출한 이후 디비전 우승을 차지하지 못했고, 라이언스는 1993년 NFC 중부 지구(NFC Central)라고 불리던 시절 이후 우승하지 못했으며, 클리블랜드 브라운스는 리그를 떠나 볼티모어 레이븐스가 되었다가 1989년 마지막 디비전 우승 이후 다시 돌아왔습니다.

문제는 이러한 성공과 헛수고(futility)가 통계적으로 유의미한가 하는 것입니다. <a href="app02.html#sec-ssdw-pass" data-type="xref">부록 B</a>에서는 표준 오차와 신뢰 구간에 대해 이야기합니다. 이 표에 표준 편차를 추가한 이유 중 하나는 각 팀의 표준 오차를 쉽게 계산할 수 있도록 하기 위해서입니다. 이는 Python에서 수행할 수 있습니다.

```
## Python
draft_py_use_pre2019 =     draft_py_use_pre2019    .merge(draft_chart_py[["Pick", "fitted_DrAV"]],
           on="Pick")

draft_py_use_pre2019_tm = (
    draft_py_use_pre2019.groupby("Tm")
    .agg({"OE": ["count", "mean", "std"]})
    .reset_index()
    .sort_values([("OE", "mean")], ascending=False)
)

draft_py_use_pre2019_tm.columns =     list(map("_".join, draft_py_use_pre2019_tm.columns))

draft_py_use_pre2019_tm.reset_index(inplace=True)

draft_py_use_pre2019_tm["se"] = (
    draft_py_use_pre2019_tm["OE_std"] /
    np.sqrt(draft_py_use_pre2019_tm["OE_count"])
)

draft_py_use_pre2019_tm["lower_bound"] = (
    draft_py_use_pre2019_tm["OE_mean"] - 1.96 * draft_py_use_pre2019_tm["se"]
)

draft_py_use_pre2019_tm["upper_bound"] = (
    draft_py_use_pre2019_tm["OE_mean"] + 1.96 * draft_py_use_pre2019_tm["se"]
)

print(draft_py_use_pre2019_tm)
```

결과는 다음과 같습니다.

index Tm\_ OE_count ... se lower_bound upper_bound 0 26 PIT 161 ... 1.487838 0.607710 6.440036 1 11 GNB 180 ... 1.495432 0.440387 6.302479 2 8 DAL 160 ... 1.313954 -0.114221 5.036479 3 1 ATL 148 ... 1.325428 -0.306186 4.889493 4 21 NOR 131 ... 1.575878 -0.825066 5.352375 5 22 NWE 176 ... 1.569551 -0.913882 5.238757 6 13 IND 162 ... 1.238039 -0.574302 4.278809 7 4 CAR 148 ... 1.357180 -0.817501 4.502647 8 2 BAL 170 ... 1.295710 -0.817661 4.261522 9 27 SEA 181 ... 1.259890 -0.988560 3.950210 10 16 LAC 144 ... 1.217377 -0.992970 3.779149 11 5 CHI 149 ... 1.315034 -1.905372 3.249560 12 20 MIN 167 ... 1.082297 -1.576770 2.665836 13 15 KAN 154 ... 1.210308 -1.870740 2.873667 14 25 PHI 162 ... 1.206150 -1.891423 2.836686 15 6 CIN 176 ... 1.191946 -1.870012 2.802417 16 14 JAX 158 ... 1.043109 -1.861808 2.227178 17 30 TEN 172 ... 0.965520 -1.763852 2.020984 18 12 HOU 145 ... 1.077847 -2.188407 2.036754 19 28 SFO 184 ... 0.991510 -2.035448 1.851270 20 31 WAS 150 ... 0.812558 -2.043098 1.142128 21 24 NYJ 137 ... 1.137789 -2.764706 1.695427 22 0 ARI 149 ... 1.171119 -2.896957 1.693831 23 23 NYG 145 ... 1.035711 -2.909893 1.150093 24 29 TAM 153 ... 0.922419 -2.730123 0.885761 25 3 BUF 161 ... 0.981895 -2.910275 0.938754 26 17 LAR 175 ... 0.905997 -3.215282 0.336228 27 19 MIA 151 ... 0.852048 -3.156297 0.183732 28 9 DEN 159 ... 0.998805 -3.449202 0.466113 29 10 DET 155 ... 0.968819 -3.664752 0.133017 30 18 LVR 162 ... 0.802757 -4.160827 -1.014020 31 7 CLE 170 ... 0.792791 -5.111136 -2.003396 \[32 rows x 8 columns\]

또는 R에서는 다음과 같습니다.

```
## R
draft_r_use_pre2019 |>
    group_by(Tm) |>
    summarize(
        total_picks = n(),
        DrAV_OE = mean(DrAV - fitted_DrAV,
            na.rm = TRUE
        ),
        DrAV_sigma = sd(DrAV - fitted_DrAV,
            na.rm = TRUE
        )
    ) |>
    mutate(
        se = DrAV_sigma / sqrt(total_picks),
        lower_bound = DrAV_OE - 1.96 * se,
        upper_bound = DrAV_OE + 1.96 * se
    ) |>
    arrange(-DrAV_OE) |>
    print(n = Inf)
```

결과는 다음과 같습니다.

\# A tibble: 32 × 7 Tm total_picks DrAV_OE DrAV_sigma se lower_bound upper_bound \<chr\> \<int\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 PIT 161 3.52 18.9 1.49 0.608 6.44 2 GNB 180 3.37 20.1 1.50 0.440 6.30 3 DAL 160 2.46 16.6 1.31 -0.114 5.04 4 ATL 148 2.29 16.1 1.33 -0.306 4.89 5 NOR 131 2.26 18.0 1.58 -0.825 5.35 6 NWE 176 2.16 20.8 1.57 -0.914 5.24 7 IND 162 1.85 15.8 1.24 -0.574 4.28 8 CAR 148 1.84 16.5 1.36 -0.818 4.50 9 BAL 170 1.72 16.9 1.30 -0.818 4.26 10 SEA 181 1.48 17.0 1.26 -0.989 3.95 11 LAC 144 1.39 14.6 1.22 -0.993 3.78 12 CHI 149 0.672 16.1 1.32 -1.91 3.25 13 MIN 167 0.545 14.0 1.08 -1.58 2.67 14 KAN 154 0.501 15.0 1.21 -1.87 2.87 15 PHI 162 0.473 15.4 1.21 -1.89 2.84 16 CIN 176 0.466 15.8 1.19 -1.87 2.80 17 JAX 158 0.183 13.1 1.04 -1.86 2.23 18 TEN 172 0.129 12.7 0.966 -1.76 2.02 19 HOU 145 -0.0758 13.0 1.08 -2.19 2.04 20 SFO 184 -0.0921 13.4 0.992 -2.04 1.85 21 WAS 150 -0.450 9.95 0.813 -2.04 1.14 22 NYJ 137 -0.535 13.3 1.14 -2.76 1.70 23 ARI 149 -0.602 14.3 1.17 -2.90 1.69 24 NYG 145 -0.880 12.5 1.04 -2.91 1.15 25 TAM 153 -0.922 11.4 0.922 -2.73 0.886 26 BUF 161 -0.986 12.5 0.982 -2.91 0.939 27 LAR 175 -1.44 12.0 0.906 -3.22 0.336 28 MIA 151 -1.49 10.5 0.852 -3.16 0.184 29 DEN 159 -1.49 12.6 0.999 -3.45 0.466 30 DET 155 -1.77 12.1 0.969 -3.66 0.133 31 LVR 162 -2.59 10.2 0.803 -4.16 -1.01 32 CLE 170 -3.56 10.3 0.793 -5.11 -2.00

###### 팁 (Tip)

이 긴 코드 출력을 볼 때 95% 신뢰 구간(CI)은 어느 팀의 `DrAV_OE`가 0과 다른지 확인하는 데 도움이 될 수 있습니다. Python과 R 출력 모두에서 95% 신뢰 구간은 `lower_bound` 및 `upper_bound`입니다. 이 구간에 값이 포함되지 않으면 통계적으로 0과 다르다고 간주할 수 있습니다. `DrAV_OE`가 이 구간보다 크면 팀이 통계적으로 평균보다 우수한 성과를 거둔 것입니다. `DrAV_OE`가 이 구간보다 작으면 팀이 통계적으로 평균보다 나쁜 성과를 거둔 것입니다.

따라서 95% 신뢰 구간을 사용하여 볼 때 두 팀(스틸러스와 패커스)은 픽 단위로 다른 팀에 비해 선수 드래프트를 통계적으로 유의미하게 더 잘하는 반면, 두 팀(레이더스와 브라운스)은 다른 팀에 비해 선수 드래프트를 통계적으로 유의미하게 더 못하는 것으로 보입니다. 이는 적절한 시간 간격(예: 단장이나 코치의 평균 경력 길이)에 걸쳐서 드래프트 재능을 파악하기가 매우 어렵다는 것을 시사하는 이 주제에 대한 연구와 일치합니다.

NFL 드래프트에서 "승리(win)"하는 방법은 콜츠가 제츠를 상대로 했던 것처럼 드래프트 픽 트레이드를 성사시켜 말하자면 사과를 더 많이 베어 물 수 있는(more bites at the apple) 기회를 스스로에게 주는 것입니다. 티모 리스케(Timo Riske)는 PFF 기사 ["A New Look at Historical Draft Success for all 32 NFL Teams(32개 NFL 팀 전체의 역사적 드래프트 성공에 대한 새로운 시각)"](https://oreil.ly/-KdlX)에서 이에 대해 더 자세히 논의합니다.

그러나 유명하게 실패한 사례 중 하나는 2002년 이후 역사적으로 드래프트에 서툴렀던 두 팀 중 하나인 오클랜드/라스베이거스 레이더스입니다. 2018년 레이더스는 그들의 최고 선수인 에지 플레이어(edge player) 칼릴 맥(Khalil Mack)을 시카고 베어스(Chicago Bears)로 트레이드하고 2개의 1라운드 픽과 후순위 픽을 교환했습니다. 레이더스는 맥과 계약 연장을 체결하지 못했고 베어스는 나중에 NFL 수비수 역사상 가장 높은 금액의 계약을 그와 체결했습니다. 세계에서 가장 유명한 스포츠 분석 전문가 모임인 [슬론 스포츠 분석 컨퍼런스(Sloan Sports Analytics Conference)](https://oreil.ly/Ppf5a)는 레이더스의 트레이드를 극찬하며 2019년 컨퍼런스에서 해당 팀에 최고 거래상(best transaction award)을 수여했습니다.

일반적으로 한 선수를 여러 선수와 트레이드하는 것은 트레이드 대상이 드래프트 픽일지라도 다수의 선수를 영입하는 팀에게 유리하게 작용할 것입니다. 그러나 통계적으로 픽을 망치는 것으로 악명 높은 레이더스는 그 픽들로 많은 것을 해내지 못했으며 그중 최고의 픽은 러닝백 조시 제이콥스(Josh Jacobs)였습니다. 제이콥스는 2022년에 NFL 러싱 야드 1위를 차지했지만, 그 이전까지는 NFL에서 두각을 나타내지 못했고 루키 계약에서 5년 차 자격을 얻지 못했습니다. 트레이드에서 얻은 또 다른 1라운드 지명 선수인 데이먼 아네트(Damon Arnette)는 팀에서 2년도 채 뛰지 못한 반면, 맥이 로스터에 포함된 베어스는 맥이 2018년 구단에 합류한 첫해에 디비전 우승을 차지하고 플레이오프에 두 번 진출했습니다.

이제 웹 스크래핑의 기본 사항을 살펴보았습니다. 이 데이터로 무엇을 할지는 전적으로 여러분에게 달려 있습니다! 거의 모든 것이 그렇듯 웹 스크래핑을 많이 할수록 더 능숙해질 것입니다.

###### 팁 (Tip)

URL을 찾기 위한 한 가지 제안은 웹 브라우저(예: Google Chrome, Microsoft Edge, Firefox)의 검사(inspection) 도구를 사용하는 것입니다. 이는 방문 중인 웹 페이지의 HTML 코드를 보여줍니다. 이 코드를 사용하면 HTML 및 CSS 선택자를 기반으로 원하는 테이블의 경로를 찾는 데 도움이 될 수 있습니다.

# 이 장에서 사용된 데이터 과학 도구 (Data Science Tools Used in This Chapter)

이 장에서는 다음 주제를 다루었습니다.

- Python 및 R에서 [프로 풋볼 레퍼런스(Pro Football Reference)](https://www.pro-football-reference.com)의 NFL 드래프트 및 NFL 스카우팅 콤바인 데이터 웹 스크래핑하기

- Python 및 R에서 `for` 루프 사용하기

- Python에서는 `rolling()`, R에서는 `rollapply()`를 사용하여 이동 평균 계산하기

- 이전 장에서 배운 데이터 랭글링 도구 재적용하기

# 연습문제 (Exercises)

1. 웹 스크래핑 예제를 NFL 드래프트의 다른 연도 범위로 변경해 보세요. 오류가 발생합니까? 이유는 무엇입니까?

2. 일반 URL <a href="https://www.pro-football-reference.com/draft/YEAR-combine.htm" class="bare"><em>https://www.pro-football-reference.com/draft/YEAR-combine.htm</em></a>을 사용하여 NFL 스카우팅 콤바인 데이터를 스크래핑하려면 이 장에 제시된 프로세스를 사용하세요(*YEAR*를 변경해야 함). 이는 이 데이터를 더 깊이 파고들 <a href="ch08.html#sec-pca-clus" data-type="xref">8장</a>의 미리보기입니다.

3. NFL 스카우팅 콤바인 데이터를 사용하여 각 선수의 40야드 대시(dash) 기록을 플롯하고, 점 색상은 선수가 플레이하는 포지션에 따라 결정되도록 합니다. 가장 빠른 포지션은 무엇입니까? 가장 느린 포지션은요? 다른 이벤트에 대해서도 동일한 작업을 수행합니다. 어떤 패턴이 보이나요?

4. NFL 스카우팅 콤바인 데이터를 이 장에서 스크래핑한 NFL 드래프트 데이터와 함께 사용합니다. NFL 드래프트에서 선수가 선택된 위치와 40야드 대시 시간 간의 관계는 무엇입니까? 일부 포지션에서 이 관계가 더 두드러집니까? 질문 3에 대한 여러분의 답변이 이 질문에 대한 접근 방식에 어떤 영향을 미칩니까?

5. 드래프트 곡선 연습문제의 경우 6-픽 이동 평균(moving average)을 6-픽 이동 중앙값(moving median)으로 변경합니다. 어떻게 됩니까? 이것이 더 낫습니까?

# 추천 도서 (Suggested Readings)

웹 스크래핑에 관한 책과 리소스는 많이 있습니다. R의 `rvest`와 `pandas`의 `read_html()`에 대한 패키지 설명서 외에, 시작하기에 좋은 두 권의 책은 다음과 같습니다.

- *R Web Scraping Quick Start Guide* (Olgun Aydin 저, Packt Publishing, 2018)

- <a href="https://learning.oreilly.com/library/view/web-scraping-with/9781491985564/" class="orm:hideurl"><em>Web Scraping with Python,</em> 2판,</a> (Ryan Mitchell 저, O’Reilly, 2018); 2024년에 3판 출간 예정

이 장의 앞부분에서 언급한 *The Drafting Stage: Creating a Marketplace for NFL Draft Picks*는 많은 훌륭한 세부 정보와 함께 NFL 드래프트에 대한 개요를 제공합니다.
