# 8장. 주성분 분석과 클러스터링: 선수 속성 (Chapter 8. Principal Component Analysis and Clustering: Player Attributes)

빅 데이터 시대에 어떤 사람들은 단순히 데이터에 "모든 것을 쏟아붓고(throw the kitchen sink)" 패턴을 찾고 가치를 이끌어내려는 강한 충동을 느낍니다. 풋볼 분석에서도 이러한 충동은 강합니다. 이 스포츠는 역동적이고 표본이 적은 특성을 가지고 있기 때문에 일반적으로 이러한 접근 방식은 주의해서 사용해야 합니다. 하지만 주의 깊게 다룬다면, 풋볼 분석가로서 유용한 통찰력을 얻을 수 있는 _비지도 학습(unsupervised learning)_ 과정(몇 단락 후에 정의되는 _지도 학습(supervised learning)_ 과 대조됨)이 될 수 있습니다.

이 장에서는 프로 풋볼 레퍼런스(Pro Football Reference)를 통해 얻을 2000년부터 2023년까지의 NFL 스카우팅 콤바인(Scouting Combine) 데이터를 사용할 것입니다. <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a>에서 언급했듯이, NFL 스카우팅 콤바인은 매년 열리는 행사로, 보통 인디애나주 인디애나폴리스(Indianapolis, Indiana)에서 개최되며 NFL 선수들이 NFL 드래프트를 준비하기 위해 일련의 신체적(및 기타) 테스트를 거칩니다. 풋볼 산업 전체가 본질적으로 연례 컨퍼런스를 위해 모이지만, 많은 최고의 선수들이 더 이상 그곳에서 테스트를 받지 않기로 하면서(대신 그들의 대학 캠퍼스에서 열리는 더 우호적인 _프로 데이(Pro Days)_ 에서 테스트받기를 선택함), 많은 사람들의 눈에 현장 이벤트의 중요성은 줄어들고 있습니다. 또한 애널리스트로서 우리의 삶에 트래킹 데이터가 추가되면서 선수들의 운동 능력을 더 정확하고 시기적절하게 추정할 수 있게 되었고, 이는 그 자체로 빠르게 증가하는 문제들의 집합입니다.

최근의 여러 기사에서 NFL 스카우팅 콤바인에 대한 논의를 제공합니다. 슬론 스포츠 분석 컨퍼런스(Sloan Sports Analytics Conference)에서 에릭(Eric)과 다른 사람들이 작성한 ["Using Tracking and Charting Data to Better Evaluate NFL Players: A Review(선수 트래킹 및 차트 데이터를 사용하여 NFL 선수를 더 잘 평가하기: 리뷰)"](https://oreil.ly/cJd5R)는 선수 데이터를 측정하는 다른 방법에 대해 논의합니다. *Washington Post*의 샘 포티에(Sam Fortier)가 작성한 ["Beyond the 40-Yard Dash: How Player Tracking Could Modernize the NFL Combine(40야드 대시 그 이상: 선수 트래킹이 NFL 콤바인을 현대화하는 방법)"](https://oreil.ly/M67dm)은 선수 트래킹 데이터가 어떻게 사용될 수 있는지 설명합니다. *Athletic*의 조르단 로드리그(Jourdan Rodrigue)가 작성한 ["Inside the Rams’ Major Changes to their Draft Process, and Why They Won’t Go Back to ‘Normal’(램스(Rams)의 드래프트 프로세스 주요 변화 내부, 그리고 그들이 '정상'으로 돌아가지 않는 이유)"](https://oreil.ly/Us5oh)는 램스의 드래프트 과정에 대한 기사를 제공합니다.

NFL 스카우팅 콤바인의 궁극적으로 시대에 뒤떨어진 특성(적어도 현장 테스트 부분)에도 불구하고, 이 데이터는 주성분 분석(PCA)과 클러스터링(clustering)을 모두 연구하기에 좋은 수단을 제공합니다. _주성분 분석(Principal component analysis, PCA)_ 은 어떤 형태의 다중공선성(collinearity)을 지닌 특징(feature)들의 집합(이 문맥에서 _다중공선성_ 이란 예측 변수들이 개념적으로 중복되는 정보를 포함하고 수치적으로 상관관계가 있음을 의미함)을 취하여, 각각이 서로 (선형적으로) 독립적인 더 작은 특징들의 하위 집합으로 "뭉개는(mushing)" 과정입니다.

운동 능력(Athleticism) 데이터는 이런 종류의 것들로 가득 차 있습니다. 예를 들어, 선수가 40야드 대시(dash)를 얼마나 빨리 달리는지는 그들의 체중(비록 완벽하지는 않지만)에 매우 많이 의존하는 반면, 얼마나 높이 뛰는지는 얼마나 멀리 뛸 수 있는지와 상관관계가 있습니다. 각자 나름대로 매우 중요하지만 서로 완전히 독립적이지는 않은 특징들의 집합을 취하여, 열(column) 방향으로 더 작은 데이터 집합을 생성할 수 있는 능력은 데이터 과학 및 관련 분야에서 종종 _차원 축소(dimensionality reduction)_ 라고 불리는 과정입니다.

_클러스터링(Clustering)_ 은 일련의 특징(feature)을 기반으로 데이터 포인트를 유사한 그룹(_클러스터(clusters)_)으로 나누는 과정입니다. 다양한 방식으로 데이터를 클러스터링할 수 있지만, 그룹이 사전에(a priori) 알려지지 않은 경우 이 과정은 비지도 학습 범주에 속합니다. _지도 학습(Supervised learning)_ 알고리즘은 훈련을 위해 미리 정의된 반응 변수(response variable)가 필요합니다. 대조적으로, _비지도 학습(unsupervised learning)_ 은 본질적으로 데이터가 스스로 무(thin air)에서 반응 변수—클러스터링의 경우 클러스터—를 생성하도록 허용합니다.

팀 스포츠에서는 선수들이 공식적으로든 비공식적으로든 종종 포지션 그룹으로 묶이기 때문에 클러스터링은 팀 및 개인 스포츠에서 매우 효과적인 접근 방식입니다. 때때로 이러한 포지션 그룹의 특성은 시간이 지남에 따라 변하며, 데이터는 팀이 포지션에 대한 아이디어에 맞는 선수들로 로스터(roster)를 구축하는 과정을 조정하는 데 도움을 주기 위해 그러한 변화를 감지하도록 도울 수 있습니다.

게다가, 팀 및 개인 스포츠 모두에서 선수는 데이터에서 발견할 수 있는 _스타일(styles)_ 을 가지고 있습니다. 선수의 스타일을 묘사하는 일련의 특징(종종 일종의 플롯을 통해 표시됨)은 수학적 성향을 가진 사람들에게는 도움이 될 수 있지만, 전통주의자(traditionalists)들은 종종 선수 유형을 그룹별로 설명하기를 원합니다. 따라서 여기서 클러스터링의 가치가 있습니다.

클러스터링 이전에 데이터에 PCA를 실행하는 과정이 필수적이라고 상상할 수 있을 것입니다. 선수가 40야드 대시를 얼마나 빨리 달리는지가 수직 점프 테스트 시 얼마나 높이 뛰는지와 거의 동일한 신호를 전달한다면, 이 둘을 모두 하나의 변수로 취급하는 것(전통적인 의미에서 "이중 계산(double counting)"은 아닐지라도)은 다른 특성들을 희생시키면서 일부 특성을 과도하게 계산하는 것이 될 것입니다. 따라서 에릭의 [DataCamp 과정(DataCamp course)](https://oreil.ly/RRNpW)에서 채택한 데이터에 PCA를 실행하는 과정부터 시작하겠습니다.

###### 팁 (Tip)

이 장에서는 다변량 통계(multivariate statistics)를 위한 기본적인 입문 방법을 제시합니다. 고급 방법들은 정기적으로 등장합니다. 예를 들어, 균일 매니폴드 근사 및 투영(uniform manifold approximation and projection, UMAP)은 이 글을 쓰는 시점에서 대중적인 새로운 거리 기반 도구 중 하나로 떠오르고 있으며, 위상 데이터 분석(topological data analysis, 기하학적 속성을 사용하는 거리 대신)은 또 다른 방법입니다. 우리가 제시하는 기본 방법을 이해한다면, 이러한 새로운 도구를 배우는 것이 더 쉬울 것이며 이러한 새로운 방법을 비교할 수 있는 벤치마크를 갖게 될 것입니다.

# 웹 스크래핑 및 NFL 스카우팅 콤바인 데이터 시각화하기 (Web Scraping and Visualizing NFL Scouting Combine Data)

데이터를 얻기 위해 <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a>에서와 유사한 웹 스크래핑 도구를 사용할 것입니다. URL 코드에서 `draft`가 `combine`으로 변경된 점에 유의하세요. 또한 데이터에 약간의 정리가 필요합니다. 때때로 데이터에 추가 헤딩(heading)이 포함되어 있습니다. 이를 제거하려면 값이 헤딩과 같은 행(`Ht != "Ht"`)을 제거합니다. 두 언어 모두에서 키(`Ht`)는 피트-인치(foot-inch)에서 인치(inches)로 변환해야 합니다.

Python을 사용하면 다음 코드를 사용하여 데이터를 다운로드하고 저장합니다.

```
## Python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

combine_py = pd.DataFrame()
for i in range(2000, 2023 + 1):
    url = (
            "https://www.pro-football-reference.com/draft/" +
            str(i) +
            "-combine.htm"
    )
    web_data = pd.read_html(url)[0]
    web_data["Season"] = i
    web_data = web_data.query('Ht != "Ht"')
    combine_py = pd.concat([combine_py, web_data])

combine_py.reset_index(drop=True, inplace=True)
combine_py.to_csv("combine_data_py.csv", index=False)

combine_py[["Ht-ft", "Ht-in"]] = \
    combine_py["Ht"].str.split("-", expand=True)

combine_py = \
    combine_py\
    .astype({
        "Wt": float,
        "40yd": float,
        "Vertical": float,
        "Bench": float,
        "Broad Jump": float,
        "3Cone": float,
        "Shuttle": float,
        "Ht-ft": float,
        "Ht-in": float
        })

combine_py["Ht"] = (
    combine_py["Ht-ft"] * 12.0 +
    combine_py["Ht-in"]
)

combine_py\
    .drop(["Ht-ft", "Ht-in"], axis=1, inplace=True)

combine_py.describe()
```

Ht Wt ... Shuttle Season count 7970.000000 7975.000000 ... 4993.000000 7999.000000 mean 73.801255 242.550094 ... 4.400925 2011.698087 std 2.646040 45.296794 ... 0.266781 6.950760 min 64.000000 144.000000 ... 3.730000 2000.000000 25% 72.000000 205.000000 ... 4.200000 2006.000000 50% 74.000000 232.000000 ... 4.360000 2012.000000 75% 76.000000 279.500000 ... 4.560000 2018.000000 max 82.000000 384.000000 ... 5.560000 2023.000000 \[8 rows x 9 columns\]

R에서는 다음 코드를 사용합니다(R에서는 데이터를 저장한 후 다시 R로 읽어 들이는데, 이는 R이 열 유형을 인식하도록 돕는 빠른 방법이기 때문입니다).

```
## R
library(tidyverse)
library(rvest)
library(htmlTable)
library(multiUS)
library(ggthemes)

combine_r <- tibble()
for (i in seq(from = 2000, to = 2023)) {
    url <- paste0("https://www.pro-football-reference.com/draft/",
                  i,
                  "-combine.htm")
    web_data <-
        read_html(url) |>
        html_table()
    web_data_clean <-
        web_data[[1]] |>
        mutate(Season = i) |>
        filter(Ht != "Ht")
    combine_r <-
        bind_rows(combine_r,
                  web_data_clean)
}
write_csv(combine_r, "combine_data_r.csv")
combine_r <- read_csv("combine_data_r.csv")

combine_r <-
    combine_r |>
    mutate(ht_ft = as.numeric(str_sub(Ht, 1, 1)),
           ht_in = str_sub(Ht, 2, 4),
           ht_in = as.numeric(str_remove(ht_in, "-")),
           Ht = ht_ft * 12 + ht_in) |>
    select(-Ht_ft, -Ht_in)
summary(combine_r)
```

Player Pos School College Length:7999 Length:7999 Length:7999 Length:7999 Class :character Class :character Class :character Class :character Mode :character Mode :character Mode :character Mode :character Ht Wt 40yd Vertical Bench Min. :64.0 Min. :144.0 Min. :4.220 Min. :17.50 Min. : 2.00 1st Qu.:72.0 1st Qu.:205.0 1st Qu.:4.530 1st Qu.:30.00 1st Qu.:16.00 Median :74.0 Median :232.0 Median :4.690 Median :33.00 Median :21.00 Mean :73.8 Mean :242.6 Mean :4.774 Mean :32.93 Mean :20.74 3rd Qu.:76.0 3rd Qu.:279.5 3rd Qu.:4.970 3rd Qu.:36.00 3rd Qu.:25.00 Max. :82.0 Max. :384.0 Max. :6.050 Max. :46.50 Max. :49.00 NA's :29 NA's :24 NA's :583 NA's :1837 NA's :2802 Broad Jump 3Cone Shuttle Drafted (tm/rnd/yr) Min. : 74.0 Min. :6.280 Min. :3.730 Length:7999 1st Qu.:109.0 1st Qu.:6.980 1st Qu.:4.200 Class :character Median :116.0 Median :7.190 Median :4.360 Mode :character Mean :114.8 Mean :7.285 Mean :4.401 3rd Qu.:121.0 3rd Qu.:7.530 3rd Qu.:4.560 Max. :147.0 Max. :9.120 Max. :5.560 NA's :1913 NA's :3126 NA's :3006 Season Min. :2000 1st Qu.:2006 Median :2012 Mean :2012 3rd Qu.:2018 Max. :2023

여기서 R 테이블의 `NA`로 입증되듯이 아주 많은 선수들이 전체 데이터 세트를 가지고 있지 않다는 점에 유의하세요. 많은 선수들이 콤바인에서 그저 자신의 키와 체중 정보만 제공하기 때문에 해당 항목의 `NA` 수가 적은 것입니다. 잠시 후에 이 문제를 해결하겠지만, 애초에 PCA가 필요한 이유인 이벤트 간 쌍별(pair-wise) 상관관계에 대해 먼저 살펴보겠습니다.

먼저, 키 대비 체중을 살펴보겠습니다. Python에서 <a href="#fig-py-ht-wt" data-type="xref">그림 8-1</a>을 만듭니다.

```
# Python
sns.set_theme(style="whitegrid", palette="colorblind")

sns.regplot(data=combine_py, x="Ht", y="Wt");
plt.show();
```

<figure>
<img src="assets/fapr_0801.png" />
<h6 id="figure-8-1.-scatterplot-with-trendline-for-player-height-plotted-against-player-weight-plotted-with-seaborn">그림 8-1. 선수의 체중(Wt)에 따른 키(Ht)를 보여주는 추세선이 있는 산점도(<code>seaborn</code>으로 플로팅됨)</h6>
</figure>

또는 R에서 <a href="#fig-r-ht-wt" data-type="xref">그림 8-2</a>를 만듭니다.

```
# R
ggplot(combine_r, aes(x = Ht, y = Wt)) +
    geom_point() +
    theme_bw() +
    xlab("Player Height (inches)") +
    ylab("Player Weight (pounds)") +
    geom_smooth(method = "lm", formula = y ~ x)
```

<figure>
<img src="assets/fapr_0802.png" />
<h6 id="figure-8-2.-scatterplot-with-trendline-for-player-height-plotted-against-player-weight-plotted-with-ggplot2">그림 8-2. 선수의 체중(Wt)에 따른 키(Ht)를 보여주는 추세선이 있는 산점도(<code>ggplot2</code>로 플로팅됨)</h6>
</figure>

이것은 일리가 있습니다. 키가 클수록 몸무게가 더 많이 나갑니다. 따라서 여기에 완전히 독립적인 두 가지 정보가 있는 것은 아닙니다. 체중 대비 40야드 대시를 살펴보겠습니다. Python에서 <a href="#fig-py-wt-40" data-type="xref">그림 8-3</a>을 만듭니다.

```
# Python
sns.regplot(data=combine_py,
            x="Wt",
            y="40yd",
            line_kws={"color": "red"});
plt.show();
```

<figure>
<img src="assets/fapr_0803.png" />
<h6 id="figure-8-3.-scatterplot-with-trendline-for-player-weight-plotted-against-40-yard-dash-time-seaborn">그림 8-3. 40야드 대시 기록에 대한 선수 체중을 보여주는 추세선이 있는 산점도(<code>seaborn</code>)</h6>
</figure>

또는 R에서 유사한 그림(<a href="#fig-r-wt-40" data-type="xref">그림 8-4</a>)을 만듭니다.

```
# R
ggplot(combine_r, aes(x = Wt, y = `40yd`)) +
    geom_point() +
    theme_bw() +
    xlab("Player Weight (pounds)") +
    ylab("Player 40-yard dash (seconds)") +
    geom_smooth(method = "lm", formula = y ~ x)
```

<figure>
<img src="assets/fapr_0804.png" />
<h6 id="figure-8-4.-scatterplot-with-trendline-for-player-weight-plotted-against-40-yard-dash-time-ggplot2">그림 8-4. 40야드 대시 기록에 대한 선수 체중을 보여주는 추세선이 있는 산점도(<code>ggplot2</code>)</h6>
</figure>

###### 경고 (Warning)

대부분의 컴퓨터 언어에서 `40yd`처럼 객체 이름을 숫자로 시작하는 것은 나쁩니다. 컴퓨터는 산술 연산 같은 것이 일어날 것이라고 생각하는데 문자가 대신 입력되어 무엇을 해야 할지 모르기 때문입니다. R은 <a href="#fig-py-wt-40" data-type="xref">그림 8-3</a>을 만들 때 했던 것처럼 백틱(\`)으로 감싸면 부적절한 이름을 사용할 수 있게 해줍니다.

여기에서도 양의 상관관계가 있습니다. 하지만 여기서도 데이터에 이미 두 개의 클러스터가 나타나고 있음을 알 수 있습니다. 아주 무거운 선수들(300파운드 이상)이 많이 있고, 아주 가벼운 선수들(225파운드 근처)이 많이 있습니다. 이 두 그룹은 양봉 분포(bimodal distribution)의 예시가 됩니다. 즉, 정규 분포처럼 하나의 중심을 가지는 것이 아니라 두 개의 그룹이 존재합니다. 이제 40야드 대시와 수직 점프를 살펴보겠습니다. Python에서 <a href="#fig-py-40-vert" data-type="xref">그림 8-5</a>를 만듭니다.

```
# Python
sns.regplot(data=combine_py,
            x="40yd",
            y="Vertical",
            line_kws={"color": "red"});
plt.show();
```

<figure>
<img src="assets/fapr_0805.png" />
<h6 id="figure-8-5.-scatterplot-with-trendline-for-player-40-yard-dash-time-plotted-against-vertical-jump-seaborn">그림 8-5. 수직 점프에 대한 선수 40야드 대시 기록을 보여주는 추세선이 있는 산점도(<code>seaborn</code>)</h6>
</figure>

또는 R에서 <a href="#fig-r-40-vert" data-type="xref">그림 8-6</a>을 만듭니다.

```
# R
ggplot(combine_r, aes(x = `40yd`, y = Vertical)) +
    geom_point() +
    theme_bw() +
    xlab("Player 40-yard dash (seconds)") +
    ylab("Player vertical jump (inches)") +
    geom_smooth(method = "lm", formula = y ~ x)
```

결과는 다음과 같습니다.

geom_smooth: na.rm = FALSE, orientation = NA, se = TRUE stat_smooth: na.rm = FALSE, orientation = NA, se = TRUE, method = lm, formula = y ~ x position_identity

<figure>
<img src="assets/fapr_0806.png" />
<h6 id="figure-8-6.-scatterplot-with-trendline-for-player-40-yard-dash-time-plotted-against-vertical-jump-ggplot2">그림 8-6. 수직 점프에 대한 선수 40야드 대시 기록을 보여주는 추세선이 있는 산점도(<code>ggplot2</code>)</h6>
</figure>

여기에는 음의 관계가 있습니다. 선수가 빠를수록(40야드 대시의 초가 낮을수록) 수직 점프(인치 단위)가 더 높습니다. 민첩성(3콘(three-cone) 드릴로 측정한 값, <a href="#fig-three-cone-drill" data-type="xref">그림 8-7</a> 참조)도 40야드 대시를 따라갈까요?

<figure>
<img src="assets/fapr_0807.png" />
<h6 id="figure-8-7.-in-the-three-cone-drill-players-run-around-the-cones-following-the-path-and-their-time-is-recorded">그림 8-7. 3콘 드릴에서 선수는 경로를 따라 콘 주위를 달리며 기록을 잰다</h6>
</figure>

Python에서 <a href="#fig-py-40-3cone" data-type="xref">그림 8-8</a>을 만듭니다.

```
# Python
sns.regplot(data=combine_py,
            x="40yd",
            y="3Cone",
            line_kws={"color": "red"});
plt.show();
```

<figure>
<img src="assets/fapr_0808.png" />
<h6 id="figure-8-8.-scatterplot-with-trendline-for-player-40-yard-dash-time-plotted-against-their-three-cone-drill-seaborn">그림 8-8. 3콘 드릴에 대한 선수 40야드 대시 기록을 보여주는 추세선이 있는 산점도(<code>seaborn</code>)</h6>
</figure>

또는 R에서 <a href="#fig-r-40-3cone" data-type="xref">그림 8-9</a>를 만듭니다.

```
# R
ggplot(combine_r, aes(x = `40yd`, y = `3Cone`)) +
    geom_point() +
    theme_bw() +
    xlab("Player 40-yard dash (seconds)") +
    ylab("Player 3 cone drill (inches)") +
    geom_smooth(method = "lm", formula = y ~ x)
```

<figure>
<img src="assets/fapr_0809.png" />
<h6 id="figure-8-9.-scatterplot-with-trend-line-for-player-40-yard-dash-time-plotted-against-three-cone-drill-time-ggplot2">그림 8-9. 3콘 드릴 기록에 대한 선수 40야드 대시 기록을 보여주는 추세선이 있는 산점도(<code>ggplot2</code>)</h6>
</figure>

여기에도 양의 관계가 나타납니다. 따라서 운동 능력은 다양한 방식으로 측정되며, 그중 어느 것도 다른 것들과 독립적이지 않다고 가정하는 것이 합리적임이 분명합니다.

PCA 과정을 시작하려면, 먼저 결측 데이터(missing data)를 "채워 넣어야(fill in)" 합니다. 우리는 이 책의 범위를 벗어나는 K-최근접 이웃(k-nearest neighbors) 알고리즘을 사용합니다. 데이터 구조에 대해 더 많이 알게 되면 자신만의 조사를 통해 결측값을 대체할 다른 방법을 찾고 싶어질 수 있습니다. 지금은 코드를 실행하기만 하세요.

###### 참고 (Note)

우리는 이 책이 자체적으로 완결성을 가지며 여러분 스스로 모든 데이터를 재생성할 수 있도록 웹 스크래핑 단계(다시 한번)와 대체(imputation) 단계를 포함시켰습니다.

이 방법을 사용하면 콤바인에서 키와 체중이 기록된 선수들의 데이터만 _대체되었습니다(imputed)_ (즉, 통계적 방법을 사용하여 결측값을 추정했습니다). 파일이 현재 디렉터리에 다운로드되어 저장되어 있지 않은 경우에만 코드가 실행되도록 `if`-`else` 문을 포함시켰습니다.

Python에서 다음을 실행합니다.

```
## Python
import numpy as np
import os
from sklearn.impute import KNNImputer

combine_knn_py_file = "combine_knn_py.csv"
col_impute = ["Ht", "Wt", "40yd", "Vertical",
              "Bench", "Broad Jump", "3Cone",
              "Shuttle"]

if not os.path.isfile(combine_knn_py_file):
    combine_knn_py = combine_py.drop(col_impute, axis=1)
    imputer = KNNImputer(n_neighbors=10)
    knn_out_py = imputer.fit_transform(combine_py[col_impute])
    knn_out_py = pd.DataFrame(knn_out_py)
    knn_out_py.columns = col_impute
    combine_knn_py = pd.concat([combine_knn_py, knn_out_py], axis=1)
    combine_knn_py.to_csv(combine_knn_py_file)

else:
    combine_knn_py = pd.read_csv(combine_knn_py_file)

combine_knn_py.describe()
```

결과는 다음과 같습니다.

Unnamed: 0 Season ... 3Cone Shuttle count 7999.000000 7999.000000 ... 7999.000000 7999.000000 mean 3999.000000 2011.698087 ... 7.239512 4.373727 std 2309.256735 6.950760 ... 0.374693 0.240294 min 0.000000 2000.000000 ... 6.280000 3.730000 25% 1999.500000 2006.000000 ... 6.978000 4.210000 50% 3999.000000 2012.000000 ... 7.122000 4.310000 75% 5998.500000 2018.000000 ... 7.450000 4.510000 max 7998.000000 2023.000000 ... 9.120000 5.560000 \[8 rows x 10 columns\]

R에서는 다음을 실행합니다.

```
## R
combine_knn_r_file <- "combine_knn_r.csv"

if (!file.exists(combine_knn_r_file)) {
    imput_input <-
        combine_r |>
        select(Ht:Shuttle) |>
        as.data.frame()

    knn_out_r <-
        KNNimp(imput_input, k = 10,
               scale = TRUE,
               meth = "median") |>
        as_tibble()

    combine_knn_r <-
        combine_r |>
        select(Player:College, Season) |>
        bind_cols(knn_out_r)
    write_csv(x = combine_knn_r,
              file = combine_knn_r_file)
} else {
    combine_knn_r <- read_csv(combine_knn_r_file)
}
```

```
combine_knn_r |>
    summary()
```

결과는 다음과 같습니다.

Player Pos School College Length:7999 Length:7999 Length:7999 Length:7999 Class :character Class :character Class :character Class :character Mode :character Mode :character Mode :character Mode :character Season Ht Wt 40yd Vertical Min. :2000 Min. :64.0 Min. :144.0 Min. :4.22 Min. :17.50 1st Qu.:2006 1st Qu.:72.0 1st Qu.:205.0 1st Qu.:4.53 1st Qu.:30.50 Median :2012 Median :74.0 Median :232.0 Median :4.68 Median :33.50 Mean :2012 Mean :73.8 Mean :242.5 Mean :4.77 Mean :32.93 3rd Qu.:2018 3rd Qu.:76.0 3rd Qu.:279.0 3rd Qu.:4.97 3rd Qu.:35.50 Max. :2023 Max. :82.0 Max. :384.0 Max. :6.05 Max. :46.50 Bench Broad Jump 3Cone Shuttle Min. : 2.00 Min. : 74.0 Min. :6.280 Min. :3.730 1st Qu.:16.00 1st Qu.:109.0 1st Qu.:6.975 1st Qu.:4.210 Median :19.50 Median :116.0 Median :7.140 Median :4.330 Mean :20.04 Mean :114.7 Mean :7.252 Mean :4.383 3rd Qu.:24.00 3rd Qu.:121.0 3rd Qu.:7.470 3rd Qu.:4.525 Max. :49.00 Max. :147.0 Max. :9.120 Max. :5.560

주목할 점은 더 이상 결측 데이터가 없다는 것입니다. 하지만 이 책의 범위를 벗어나며 이 [Stack Overflow 게시물(Stack Overflow post)](https://oreil.ly/-SoJi)에서 설명한 것과 유사한 이유로, 두 방법은 약간 다른 가정과 방법을 사용하기 때문에 유사하지만 약간 다른 결과를 산출합니다.

# 주성분 분석(PCA) 소개 (Introduction to PCA)

이후 분석에 사용할 PCA를 피팅(fitting)하기 전에, 잠시 쉬어 가며 PCA가 어떻게 작동하는지 살펴보겠습니다. 개념적으로 PCA는 가능한 한 적은 수를 사용하기 위해 데이터의 차원(dimensions) 수를 줄입니다. 그래픽으로 볼 때 *차원*은 데이터를 설명하는 데 필요한 축(axis)의 수를 의미합니다. 표 형식(tabularly)에서는 데이터를 설명하는 데 필요한 열(column)의 수를 의미합니다. 대수학(algebraically)에서는 데이터를 설명하는 데 필요한 독립 변수의 수를 의미합니다.

<a href="#fig-py-ht-wt" data-type="xref" data-xrefstyle="select:labelnumber">그림 8-1</a>과 <a href="#fig-r-ht-wt" data-type="xref" data-xrefstyle="select:labelnumber">8-2</a>에 표시된 길이-체중(length-weight) 관계를 다시 살펴보겠습니다. 이 데이터를 설명하기 위해 x축과 y축이 모두 필요할까요? 아마 아닐 것입니다. PCA를 피팅해 보고, 그다음 결과를 살펴보겠습니다. 결측값을 제거한 후 원시 데이터(raw data)를 사용할 것입니다. Python에서는 `scikit-learn` 패키지의 `PCA()` 함수를 사용합니다.

```
## Python
from sklearn.decomposition import PCA

pca_wt_ht = PCA(svd_solver="full")
wt_ht_py =     combine_py[["Wt", "Ht"]]    .query("Wt.notnull() & Ht.notnull()")    .copy()

pca_fit_wt_ht_py =     pca_wt_ht.fit_transform(wt_ht_py)
```

또는 R에서는 핵심(core) R 패키지 세트의 일부인 `stats` 패키지의 `prcomp()` 함수를 사용합니다.

```
## r
wt_ht_r <-
    combine_r |>
    select(Wt, Ht) |>
    filter(!is.na(Wt) & !is.na(Ht))

pca_fit_wt_ht_r <-
    prcomp(wt_ht_r)
```

이제 모델 세부 정보를 살펴보겠습니다. Python에서는 이 코드를 사용하여 데이터의 새로운 축이 되는 각각의 새로운 _주성분(principal components, PCs)_ 이 설명하는 분산을 살펴봅니다.

```
## Python
print(pca_wt_ht.explained_variance_ratio_)
```

결과는 다음과 같습니다.

\[0.99829949 0.00170051\]

R에서는 피팅된 모델의 `summary()`를 살펴봅니다.

```
## R
summary(pca_fit_wt_ht_r)
```

결과는 다음과 같습니다.

Importance of components: PC1 PC2 Standard deviation 45.3195 1.8704 Proportion of Variance 0.9983 0.0017 Cumulative Proportion 0.9983 1.0000

두 출력 모두 `PC1`(또는 데이터의 축)이 데이터 변동성의 99.8%를 포함하고 있음을 보여줍니다. 이는 첫 번째 PC만이 데이터에 중요하다는 것을 알려줍니다.

데이터의 새로운 표현을 확인하려면 새 데이터를 플로팅하세요. Python에서는 `matplotlib`의 `plot()`을 사용하여 출력의 간단한 산점도를 만들어서 <a href="#fig-py-ht-wt-2" data-type="xref">그림 8-10</a>을 만듭니다.

```
## Python
plt.plot(pca_fit_wt_ht_py[:, 0], pca_fit_wt_ht_py[:, 1], "o");
plt.show();
```

<figure>
<img src="assets/fapr_0810.png" />
<h6 id="figure-8-10.-scatterplot-of-the-pca-rotation-for-weight-and-height-with-plot-from-matplotlot">그림 8-10. 체중과 키에 대한 PCA 회전 산점도(<code>matplotlot</code>의 <code>plot()</code> 활용)</h6>
</figure>

R에서는 `ggplot`을 사용하여 <a href="#fig-r-ht-wt-2" data-type="xref">그림 8-11</a>을 만듭니다.

```
## Python
pca_fit_wt_ht_r$x |>
    as_tibble() |>
    ggplot(aes(x = PC1, y = PC2)) +
    geom_point() +
    theme_bw()
```

이 그림들에는 많은 일들이 벌어지고 있습니다. 먼저, <a href="#fig-py-ht-wt" data-type="xref">그림 8-1</a>을 <a href="#fig-py-ht-wt-2" data-type="xref">그림 8-10</a>과 비교하거나, <a href="#fig-r-ht-wt" data-type="xref">그림 8-2</a>를 <a href="#fig-r-ht-wt-2" data-type="xref">그림 8-11</a>과 비교해 보세요. 공간 패턴 인식에 능숙하다면, 그림이 회전하기만 했을 뿐 동일한 데이터를 가지고 있다는 것을 알 수 있을 것입니다(주요 무리에서 벗어난 이상치(outliers)들을 살펴보면 이 데이터 포인트들이 어떻게 이동했는지 볼 수 있습니다). 이는 PCA가 데이터를 회전시켜 데이터의 차원을 더 적게 만들기 때문입니다. 이 예제에는 두 가지 차원만 있기 때문에 데이터에서 패턴을 볼 수 있습니다.

<figure>
<img src="assets/fapr_0811.png" />
<h6 id="figure-8-11.-scatterplot-of-the-pca-rotation-for-weight-and-height-with-ggplot2">그림 8-11. 체중과 키에 대한 PCA 회전 산점도(<code>ggplot2</code> 활용)</h6>
</figure>

Python에서는 `components_`를 보거나 R에서는 피팅된 PCA를 출력하여 이러한 회전 값을 얻을 수 있습니다. 이 책에서는 이 값들을 더 이상 사용하지 않지만, 기계 학습(machine learning)에서 사람들이 PCA 공간 안팎으로 데이터를 변환해야 할 때 유용하게 쓰입니다. Python에서는 다음 코드를 사용하여 회전 값을 추출합니다.

```
## Python
pca_wt_ht.components_
```

결과는 다음과 같습니다.

array(\[\[ 0.9991454 , 0.04133366\], \[-0.04133366, 0.9991454 \]\])

R에서는 다음 코드를 사용하여 회전 값을 추출합니다.

```
## R
print(pca_fit_wt_ht_r)
```

결과는 다음과 같습니다.

Standard deviations (1, .., p=2): \[1\] 45.319497 1.870442 Rotation (n x k) = (2 x 2): PC1 PC2 Wt -0.99914540 -0.04133366 Ht -0.04133366 0.99914540

여러분의 값은 저희의 값과 부호가 다를 수 있지만(예를 들어 저희는 –0.999를 얻고, 여러분은 0.999를 얻을 수 있음) 이는 괜찮으며 무작위로 다를 수 있습니다. 예를 들어 Python의 PCA는 책의 숫자들과 동일한 세 개의 양수를 가지지만, R의 PCA는 동일한 숫자를 가지지만 음수입니다. 이 숫자들은 선수의 체중에 0.999를 곱하고 선수의 키에 0.041을 곱한 값을 더하여 첫 번째 PC를 만든다는 것을 의미합니다.

이 값들이 왜 그렇게 다를까요? <a href="#fig-py-ht-wt-2" data-type="xref" data-xrefstyle="select:labelnumber">그림 8-10</a>과 <a href="#fig-r-ht-wt-2" data-type="xref" data-xrefstyle="select:labelnumber">8-11</a>을 다시 살펴보세요. x축과 y축의 척도(scale)가 크게 다르다는 점에 유의하세요. 이는 또한 다른 입력 특징(features)들이 다른 수준의 영향을 미치게 할 수 있습니다. 또한, 같지 않은 숫자는 때때로 계산 문제를 일으킵니다.

다음 섹션에서는 입력을 스케일링하여 모든 특징을 동일한 단위 수준에 맞출 것입니다. _스케일링(Scaling)_ 은 특징을 변환하는 것을 의미하며, 보통 평균을 0, 표준편차를 1로 맞춥니다. 따라서 스케일링 후에는 특징의 다른 단위와 크기가 더 이상 중요하지 않습니다.

# 모든 데이터에 대한 PCA (PCA on All Data)

이제 R과 Python의 내장 알고리즘을 적용하여 모든 데이터에 대해 PCA 분석을 수행하세요. 이는 서로 독립적인 새롭고 더 적은 수의 예측 변수를 만드는 데 도움이 될 것입니다. 먼저 데이터를 스케일링한 다음 PCA를 실행합니다. Python에서는 다음과 같습니다.

```
## Python
from sklearn.decomposition import PCA

scaled_combine_knn_py = (
    combine_knn_py[col_impute] -
    combine_knn_py[col_impute].mean()) /     combine_knn_py[col_impute].std()

pca = PCA(svd_solver="full")
pca_fit_py =     pca.fit_transform(scaled_combine_knn_py)
```

또는 R에서는 다음과 같습니다.

```
## R
scaled_combine_knn_r <-
    scale(combine_knn_r |> select(Ht:Shuttle))

pca_fit_r <-
    prcomp(scaled_combine_knn_r)
```

`pca_fit` 객체는 데이터 객체라기보다는 모델 객체에 가깝습니다. 여기에는 흥미로운 정보들이 있습니다. 한 가지 예로, 각 PC의 가중치를 볼 수 있습니다. Python에서는 다음과 같습니다.

```
## Python
rotation = pd.DataFrame(pca.components_, index=col_impute)
print(rotation)
```

결과는 다음과 같습니다.

0 1 2 ... 5 6 7 Ht 0.280591 0.393341 0.390341 ... -0.367237 0.381342 0.377843 Wt 0.506953 0.273279 -0.063500 ... 0.359464 -0.110215 -0.130109 40yd -0.709435 -0.001356 -0.082813 ... -0.096306 0.068683 -0.019891 Vertical -0.203781 0.033044 0.012393 ... 0.296674 0.523851 0.509379 Bench -0.142324 0.161150 0.593645 ... -0.369323 -0.035026 -0.428910 Broad Jump 0.206559 -0.080594 -0.613440 ... -0.641948 0.277464 -0.094715 3Cone -0.005106 -0.044482 0.027751 ... -0.298070 -0.677284 0.620678 Shuttle -0.237684 0.857359 -0.327257 ... 0.035926 -0.162377 -0.047622 \[8 rows x 8 columns\]

또는 R에서는 다음과 같습니다.

```
## R
print(pca_fit_r$rotation)
```

결과는 다음과 같습니다.

PC1 PC2 PC3 PC4 PC5 Ht -0.2797884 0.4656585 0.747620897 0.21562254 -0.06128240 Wt -0.3906321 0.2803488 0.002635803 -0.04180851 0.14466721 40yd -0.3937993 -0.0994878 0.045495814 -0.01403113 0.48636319 Vertical 0.3456004 0.4186011 0.002756780 -0.53609337 0.54959455 Bench -0.2668254 0.6109690 -0.642865102 0.18678232 -0.16950424 Broad Jump 0.3674448 0.3388903 0.139855673 -0.31774247 -0.43822503 3Cone -0.3823998 -0.1115644 -0.060381468 -0.51566448 0.04602633 Shuttle -0.3770497 -0.1373520 0.049975069 -0.51225797 -0.46240812 PC6 PC7 PC8 Ht 0.1394355 -0.164300591 -0.22194154 Wt -0.1553924 0.089682873 0.84494838 40yd -0.4083595 0.543622221 -0.36595878 Vertical 0.3348374 0.043881290 -0.04282478 Bench 0.0604662 -0.009981142 -0.27362455 Broad Jump -0.6246519 0.216613922 -0.02156424 3Cone -0.2978993 -0.675688165 -0.15604839 Shuttle 0.4415283 0.404889985 -0.03684955

###### 경고 (Warning)

PCA 구성요소(components)는 _고유값(eigenvalues)_ 및 _고유벡터(eigenvectors)_ 라고 불리는 행렬의 수학적 속성에 기반합니다. 이것들은 스칼라이며, 여러분의 PC들은 우리의 예제와 반대 부호를 가질 수 있습니다(예를 들어, 우리의 `PC1`에 대한 `Ht`가 음수라면, 여러분의 것은 양수일 수 있습니다). 걱정하지 마세요. 그것이 부호가 다른 이유라는 점만 알아두세요. 또한 이 현상은 우리가 이 책을 쓸 때 R과 Python에서 발생했던 것으로 보입니다.

첫 번째 PC는 40야드 대시와 체중에 거의 동일한 가중치(요인 가중치 –0.39)를 부여한다는 점에 주목하세요. 작을수록 더 좋은 크기가 아닌 지표들(40야드 대시, 민첩성 훈련)의 대부분은 음의 가중치를 갖는 반면, 클수록 더 좋은 지표들(수직 점프 및 넓이뛰기)은 양의 가중치를 갖는다는 점에 유의하세요.

###### 참고 (Note)

우리의 Python과 R 예제는 두 언어 간의 대체(imputation) 방법과 PCA의 차이 때문에 약간씩 갈라지기 시작합니다. 그러나 정성적인 결과(qualitative results)는 동일합니다.

이것은 여러분이 뭔가 의미 있는 것을 발견하고 있다는 좋은 신호입니다. 각 PC가 분산의 어느 정도 비율을 설명하는지 볼 수 있습니다. Python에서는 다음과 같습니다.

```
## Python
print(pca.explained_variance_)
```

결과는 다음과 같습니다.

\[5.60561713 0.83096684 0.62448842 0.37527929 0.21709371 0.13913206 0.12108346 0.08633909\]

또는 R에서는 표준 편차 제곱을 봅니다.

```
## R
print(pca_fit_r$sdev^2)
```

결과는 다음과 같습니다.

\[1\] 5.67454385 0.84556662 0.61894619 0.35175168 0.19651463 0.11815058 0.11166060 \[8\] 0.08286586

여기서 예상대로 첫 번째 PC가 데이터 변동성의 상당 부분을 처리하지만, 후속 PC들 역시 변동성에 어느 정도 영향을 미친다는 것을 알 수 있습니다. R에서 이러한 표준 편차들을 가져와 제곱하여 분산으로 변환한 다음(PCA1<sup>2</sup>), 모든 분산의 합으로 나누면 각 축이 설명하는 분산 백분율을 볼 수 있습니다. Python의 PCA는 추가적인 계산 없이 이를 포함합니다.

Python에서는 다음과 같습니다.

```
## Python
pca_percent_py =     pca.explained_variance_ratio_.round(4) * 100
print(pca_percent_py)
```

결과는 다음과 같습니다.

\[70.07 10.39 7.81 4.69 2.71 1.74 1.51 1.08\]

또는 R에서는 다음과 같습니다.

```
## R
pca_var_r <- pca_fit_r$sdev^2
pca_percent_r <-
    round(pca_var_r / sum(pca_var_r) * 100, 2)
print(pca_percent_r)
```

결과는 다음과 같습니다.

\[1\] 70.93 10.57 7.74 4.40 2.46 1.48 1.40 1.04

이제부터 먼저 해야 할 일은 처음 몇 개의 PC를 그래프로 그려서 자연스러운 클러스터가 나타나는지 확인하는 것입니다. Python에서 <a href="#fig-py-pca-axes" data-type="xref">그림 8-12</a>를 만듭니다.

```
## Python
sns.scatterplot(data=combine_knn_py,
                x="PC1",
                y="PC2");
plt.show();
```

<figure>
<img src="assets/fapr_0812.png" />
<h6 id="figure-8-12.-plot-of-first-two-pca-components-seaborn">그림 8-12. 처음 두 개의 PCA 구성 요소 플롯(<code>seaborn</code>)</h6>
</figure>

R에서 <a href="#fig-r-pca-axes" data-type="xref">그림 8-13</a>을 만듭니다.

```
## R
ggplot(combine_knn_r, aes(x = PC1, y = PC2)) +
    geom_point() +
    theme_bw() +
    xlab(paste0("PC1 = ", pca_percent_r[1], "%")) +
    ylab(paste0("PC2 = ", pca_percent_r[2], "%"))
```

<figure>
<img src="assets/fapr_0813.png" />
<h6 id="figure-8-13.-plot-of-first-two-pca-components-ggplot2">그림 8-13. 처음 두 개의 PCA 구성 요소 플롯(<code>ggplot2</code>)</h6>
</figure>

###### 참고 (Note)

PC는 고유값을 기반으로 하므로 여러분의 그림이 우리의 예제 중 하나와 뒤집혀 있을 수 있습니다. 예를 들어, 우리의 R과 Python 그림은 서로의 거울 이미지입니다.

이미 두 개의 클러스터가 보입니다! 더 많은 데이터를 사용하여 다른 가능성을 밝혀보면 어떨까요? 세 번째 PC 값에 따라 각 포인트의 색상을 지정해 보겠습니다. Python에서 <a href="#fig-py-pca-axes-col" data-type="xref">그림 8-14</a>를 만듭니다.

```
## Python
sns.scatterplot(data=combine_knn_py,
                x="PC1",
                y="PC2",
                hue="PC3");
plt.show();
```

<figure>
<img src="assets/fapr_0814.png" />
<h6 id="figure-8-14.-plot-of-the-first-two-pca-components-with-the-third-pca-component-as-the-point-color-seaborn">그림 8-14. 세 번째 PCA 구성 요소를 점 색상으로 사용한 처음 두 개의 PCA 구성 요소 플롯(<code>seaborn</code>)</h6>
</figure>

R에서 <a href="#fig-r-pca-axes-col" data-type="xref">그림 8-15</a>를 만듭니다.

```
## R
ggplot(combine_knn_r,
       aes(x = PC1, y = PC2, color = PC3)) +
    geom_point() +
    theme_bw() +
    xlab(paste0("PC1 = ", pca_percent_r[1], "%")) +
    ylab(paste0("PC2 = ", pca_percent_r[2], "%")) +
    scale_color_continuous(
        paste0("PC3 = ", pca_percent_r[3], "%"),
        low="skyblue", high="navyblue")
```

<figure>
<img src="assets/fapr_0815.png" />
<h6 id="figure-8-15.-plot-of-the-first-two-pca-components-with-the-third-pca-component-as-the-point-color-ggplot2">그림 8-15. 세 번째 PCA 구성 요소를 점 색상으로 사용한 처음 두 개의 PCA 구성 요소 플롯(<code>ggplot2</code>)</h6>
</figure>

흥미롭네요. 플롯 가장자리에 있는 선수들은 `PC3` 값이 낮아 더 어두운 색조에 해당하는 것 같습니다. 포지션별로 색상을 다르게 하면 어떨까요?

Python에서 <a href="#fig-py-pca-axes-pos-2" data-type="xref">그림 8-16</a>을 만듭니다.

```
## Python
sns.scatterplot(data=combine_knn_py,
                x="PC1",
                y="PC2",
                hue="Pos");
plt.show();
```

<figure>
<img src="assets/fapr_0816.png" />
<h6 id="figure-8-16.-plot-of-the-first-two-pca-components-with-the-point-player-position-as-the-color-seaborn">그림 8-16. 점 색상으로 선수의 포지션을 사용한 처음 두 개의 PCA 구성 요소 플롯(<code>seaborn</code>)</h6>
</figure>

R에서는 색맹(colorblind) 친화적인 팔레트를 사용하여 <a href="#fig-r-pca-axes-pos-2" data-type="xref">그림 8-17</a>을 만듭니다.

```
## R
library(RColorBrewer)
color_count <- length(unique(combine_knn_r$Pos))
get_palette <- colorRampPalette(brewer.pal(9, "Set1"))

ggplot(combine_knn_r,
       aes(x = PC1, y = PC2, color = Pos)) +
    geom_point(alpha = 0.75) +
    theme_bw() +
    xlab(paste0("PC1 = ", pca_percent_r[1], "%")) +
    ylab(paste0("PC2 = ", pca_percent_r[2], "%")) +
    scale_color_manual("Player position",
                       values = get_palette(color_count))
```

<figure>
<img src="assets/fapr_0817.png" />
<h6 id="figure-8-17.-plot-of-the-first-two-pca-components-with-the-point-player-position-as-the-color-ggplot2">그림 8-17. 점 색상으로 선수의 포지션을 사용한 처음 두 개의 PCA 구성 요소 플롯(<code>ggplot2</code>)</h6>
</figure>

좋습니다. 재미있네요. 실제로 포지션이 데이터에서 명확한 그룹을 생성하는 것으로 보입니다. 언뜻 보기에 이 데이터를 대략 5~7개의 클러스터로 나눌 수 있을 것 같습니다.

###### 팁 (Tip)

미국 남성의 약 8%와 여성의 약 0.5%가 _색각 이상(color vision deficiency)_ (흔히 _색맹(colorblindness)_ 으로 알려짐)을 가지고 있습니다. 이는 흑백만 볼 수 있는 것부터, 더 흔하게는 모든 색상을 구별하지 못하는 것까지 다양합니다. 예를 들어, 리차드는 빨간색과 초록색 구별에 어려움을 겪으며, 이로 인해 보라색도 보기 어렵습니다. 따라서 더 많은 사람들이 사용할 수 있는 색상을 선택하도록 노력하세요. [Color Oracle](https://www.colororacle.org)이나 [Sim Daltonism](https://michelf.ca/projects/sim-daltonism)과 같은 도구를 사용하면 수치를 테스트하고 색맹인 사람의 시각으로 볼 수 있습니다.

# 콤바인 데이터 클러스터링하기 (Clustering Combine Data)

여기서 사용할 클러스터링 알고리즘은 k-평균 클러스터링(k-means clustering)으로, 이는 데이터 세트를 각각의 관찰(observation)이 가까운 평균(_클러스터 중심(cluster centers)_, 또는 _클러스터 도심(cluster centroid)_)을 갖는 클러스터에 속하도록 *k*개의 클러스터로 나누는 것을 목표로 하며, 이는 클러스터의 원형(prototype) 역할을 합니다.

수치적 방법론이 다르기 때문에 이 섹션을 두 단위로 나눕니다. 이는 두 방법 중 하나가 다른 것보다 "틀렸다"거나 "낫다"는 의미가 아닙니다. 대신, 단순히 방법이 다를 수 있다는 것을 보여주기 위함입니다. 또한, 다변량 통계 방법론(특히 비지도 방법)을 이해하고 해석하는 것은 주관적일 수 있습니다. 리차드의 텍사스 공대(Texas Tech) 시절 교수 중 한 명인 스티븐 콕스(Stephen Cox)는 이 과정을 찻잎(tea leaves)을 읽는 것과 비슷하다고 묘사하곤 했는데, 주의하지 않으면 찾고 있는 패턴을 종종 발견할 수 있기 때문입니다.

###### 경고 (Warning)

다변량 통계가 찻잎 읽기와 비슷하다는 비교에도 불구하고, 이 방법들은 강력하고 유용하기 때문에 (정당하게) 널리 사용됩니다. 그러나 여러분이 사용자와 모델러로서 이 방법을 사용하려면 이러한 한계를 이해해야 합니다.

## Python에서 콤바인 데이터 클러스터링하기 (Clustering Combine Data in Python)

###### 경고 (Warning)

만약 여러분이 (바라건대) 코드를 따라 하고 있다면, 클러스터링 결과가 우리와 다를 가능성이 높습니다. 컴퓨터에 나타난 그룹 중 어떤 클러스터 번호가 해당하는지 결과를 확인해야 합니다.

Python으로 시작하려면, `scipy` 패키지의 `kmeans`를 사용하고 6개의 중심에 대해 피팅합니다(이 책을 위한 코드를 실행할 때마다 동일한 결과를 얻을 수 있도록 `seed`를 `1234`로 설정했습니다).

```
## Python
from scipy.cluster.vq import vq, kmeans

k_means_fit_py =     kmeans(combine_knn_py[["PC1", "PC2"]], 6, seed = 1234)
```

다음으로 이 클러스터를 데이터 세트에 첨부합니다.

```
## Python
combine_knn_py["cluster"] =     vq(combine_knn_py[["PC1", "PC2"]], k_means_fit_py[0])[0]

combine_knn_py.head()
```

결과는 다음과 같습니다.

Unnamed: 0 Player Pos ... PC7 PC8 cluster 0 0 John Abraham OLB ... -0.146522 0.292073 3 1 1 Shaun Alexander RB ... -0.073008 0.060237 1 2 2 Darnell Alford OT ... -0.491523 -0.068370 0 3 3 Kyle Allamon TE ... 0.328718 -0.059768 2 4 4 Rashard Anderson CB ... -0.674786 -0.276374 1 \[5 rows x 24 columns\]

데이터의 헤드 부분에서는 이렇다 할 만한 것을 얻을 수 없습니다. 그러나 할 수 있는 한 가지는 클러스터가 비슷한 포지션과 선수 유형을 함께 묶는지 확인하는 것입니다. 클러스터 1을 살펴보겠습니다.

```nb
print(
    combine_knn_py.query("cluster == 1")
    .groupby("Pos")
    .agg({"Ht": ["count", "mean"], "Wt": ["count", "mean"]})
)
```

결과는 다음과 같습니다.

Ht Wt count mean count mean Pos CB 219 72.442922 219 197.506849 DB 27 72.074074 27 201.074074 DE 13 75.384615 13 250.307692 EDGE 5 74.800000 5 247.400000 FB 8 72.500000 8 235.000000 ILB 20 73.700000 20 237.700000 K 9 73.777778 9 207.666667 LB 45 73.673333 45 234.486667 OLB 78 73.987179 78 236.397436 P 24 74.416667 24 206.041667 QB 40 74.450000 40 217.000000 RB 163 71.225767 163 216.932515 S 221 72.800905 221 209.330317 TE 20 75.450000 20 244.100000 WR 409 73.795844 409 207.871638

여기에는 조금씩 다 섞여 있지만, 주로 공에서 멀리 떨어진 선수들, 즉 코너백, 세이프티, 와이드 리시버가 있고 약간의 러닝백이 섞여 있습니다. 이 포지션 그룹 중에서는 좀 더 체중이 많이 나가는 선수들입니다.

컴퓨터의 난수 생성기에 따라 다른 클러스터는 다른 번호를 가질 수 있으므로 클러스터링 체제(regimes) 간에 비교할 때 주의해야 합니다. 이제 플롯을 사용하여 전체 클러스터 요약을 살펴보겠습니다. Python에서 <a href="#fig-py-cluster_results" data-type="xref">그림 8-18</a>을 만듭니다.

```
## Python
combine_knn_py_cluster =     combine_knn_py    .groupby(["cluster", "Pos"])    .agg({"Ht": ["count", "mean"],
          "Wt": ["mean"]}
)

combine_knn_py_cluster.columns =     list(map("_".join, combine_knn_py_cluster.columns))

combine_knn_py_cluster.reset_index(inplace=True)

combine_knn_py_cluster    .rename(columns={"Ht_count": "n",
                     "Ht_mean": "Ht",
                     "Wt_mean": "Wt"},
                     inplace=True)

combine_knn_py_cluster.cluster =     combine_knn_py_cluster.cluster.astype(str)

sns.catplot(combine_knn_py_cluster, x="n", y="Pos",
            col="cluster", col_wrap=3, kind="bar");
plt.show();
```

<figure>
<img src="assets/fapr_0818.png" />
<h6 id="figure-8-18.-plot-of-positions-by-cluster-seaborn">그림 8-18. 클러스터별 포지션 플롯(<code>seaborn</code>)</h6>
</figure>

여기서 클러스터 0은 주로 공격 라인맨과 내부 수비 라인맨처럼 덩치가 큰 선수들입니다. 클러스터 2는 클러스터 0과 비슷한 포지션을 포함하면서도 수비 엔드와 타이트 엔드가 추가되었고, 이들은 아웃사이드 라인배커와 함께 클러스터 3에서도 큰 비중을 차지합니다. 클러스터 1에 대해서는 앞에서 논의했습니다. 클러스터 4는 클러스터 1과 동일한 포지션이 많지만 디펜시브 백과 와이드 리시버가 더 많습니다(크기는 다음에 살펴보겠습니다). 클러스터 5에는 많은 수의 쿼터백과 상당수의 다른 스킬 포지션(_스킬 포지션(skill positions)_ 이란 풋볼에서 일반적으로 공을 잡고 득점을 책임지는 포지션)이 포함되어 있습니다.

체중과 키를 비교하기 위해 클러스터별 요약을 살펴보겠습니다.

```
## Python
combine_knn_py_cluster    .groupby("cluster")    .agg({"Ht": ["mean"], "Wt": ["mean"]})
```

결과는 다음과 같습니다.

Ht Wt mean mean cluster 0 75.866972 293.708339 1 73.631939 223.254368 2 74.966517 272.490225 3 75.230958 250.847219 4 71.099940 205.290840 5 73.098379 229.605847

가설을 세운 대로, 클러스터 1과 클러스터 4는 대체로 동일한 포지션이 포함되어 있지만, 클러스터 1에는 키와 몸무게 면에서 훨씬 덩치가 큰 선수들이 포함되어 있습니다. 클러스터 0과 2도 유사한 결과를 보여줍니다.

## R에서 콤바인 데이터 클러스터링하기 (Clustering Combine Data in R)

R로 시작하려면, 핵심 패키지와 함께 제공되는 `stats` 패키지의 `kmeans()` 함수를 사용하세요. 여기서 `iter.max`는 클러스터의 수 안에서 클러스터와 중심을 찾기 위해 허용되는 최대 반복 횟수입니다. 알고리즘이 모델을 피팅하기 위해 여러 번의 시도, 즉 _반복(iterations)_ 이 필요하기 때문에 이 설정이 필요합니다. 이는 다음 스크립트를 사용하여 수행됩니다(일관된 결과를 얻기 위해 `set.seed(123)`으로 R의 난수 시드를 설정합니다).

```
## R
set.seed(123)
k_means_fit_r <-
    kmeans(combine_knn_r |> select(PC1, PC2),
           centers = 6, iter.max = 10)
```

다음으로 이 클러스터를 데이터 세트에 첨부합니다.

```
## R
combine_knn_r <-
    combine_knn_r |>
    mutate(cluster = k_means_fit_r$cluster)

combine_knn_r |>
    select(Pos, Ht:Shuttle, cluster) |>
    head()
```

결과는 다음과 같습니다.

\# A tibble: 6 × 10 Pos Ht Wt \`40yd\` Vertical Bench \`Broad Jump\` \`3Cone\` Shuttle cluster \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<int\> 1 OLB 76 252 4.55 38.5 23.5 124 6.94 4.22 6 2 RB 72 218 4.58 35.5 19 120 7.07 4.24 4 3 OT 76 334 5.56 25 23 94 8.48 4.98 3 4 TE 74 253 4.97 29 21 104 7.29 4.49 1 5 CB 74 206 4.55 34 15 123 7.18 4.15 4 6 K 70 202 4.55 36 16 120. 6.94 4.17 4

Python 예제와 마찬가지로 R에서는 인덱스가 0이 아닌 1부터 시작한다는 사실 외에는 여기서도 데이터의 헤드 부분에서 많은 것을 얻을 수는 없습니다. 첫 번째 클러스터를 살펴보세요.

```
## R
combine_knn_r |>
    filter(cluster == 1) |>
    group_by(Pos) |>
    summarize(n = n(), Ht = mean(Ht), Wt = mean(Wt)) |>
    arrange(-n) |>
    print(n = Inf)
```

결과는 다음과 같습니다.

\# A tibble: 21 × 4 Pos n Ht Wt \<chr\> \<int\> \<dbl\> \<dbl\> 1 QB 236 74.8 223. 2 TE 200 76.2 255. 3 DE 193 75.3 266. 4 ILB 127 73.0 242. 5 OLB 116 73.5 242. 6 FB 65 72.3 247. 7 P 60 74.8 221. 8 RB 49 71.1 226. 9 LB 43 72.9 235. 10 DT 38 73.9 288. 11 WR 29 74.9 219. 12 LS 28 73.9 241. 13 EDGE 27 75.3 255. 14 K 23 73.2 213. 15 DL 22 75.5 267. 16 S 11 72.6 220. 17 C 3 74.7 283 18 OG 2 75.5 300 19 CB 1 73 214 20 DB 1 72 197 21 OL 1 72 238

이 클러스터에서는 쿼터백, 타이트 엔드, 수비 엔드의 비율이 높습니다. 이는 메릴랜드 대학교에서 쿼터백으로 뛰다가 프로 타이트 엔드가 된 전 바이킹스 타이트 엔드(이자 수석 코치) 마이크 타이스(Mike Tice)처럼 타이트 엔드 중에는 쿼터백에서 전향한 선수들이 많기 때문에 풋볼 관점에서 어느 정도 일리가 있습니다.

컴퓨터의 난수 생성기에 따라 각 클러스터의 번호는 다를 수 있습니다. R에서 <a href="#fig-r-cluster_results" data-type="xref">그림 8-19</a>를 만듭니다.

```
## R
combine_knn_r_cluster <-
    combine_knn_r |>
    group_by(cluster, Pos) |>
    summarize(n = n(), Ht = mean(Ht), Wt = mean(Wt),
              .groups="drop")

combine_knn_r_cluster |>
    ggplot(aes(x = n, y = Pos)) +
    geom_col(position='dodge') +
    theme_bw() +
    facet_wrap(vars(cluster)) +
    theme(strip.background = element_blank()) +
    ylab("Position") +
    xlab("Count")
```

<figure>
<img src="assets/fapr_0819.png" />
<h6 id="figure-8-19.-plot-of-positions-by-cluster-ggplot2">그림 8-19. 클러스터별 포지션 플롯(<code>ggplot2</code>)</h6>
</figure>

여기서는 이전에 Python에서 했던 것과 유사한 결과를 얻을 수 있습니다. 공격 라인맨과 수비 라인맨이 일부 클러스터에 함께 그룹화되는 반면, 스킬 포지션 선수들은 다른 클러스터에서 더 자주 나타납니다. 체중과 키를 비교하기 위해 클러스터별 요약을 살펴보겠습니다. R에서는 다음을 사용합니다.

```
## R
combine_knn_r_cluster |>
    group_by(cluster) |>
    summarize(ave_ht = mean(Ht),
              ave_wt = mean(Wt))
```

결과는 다음과 같습니다.

\# A tibble: 6 × 3 cluster ave_ht ave_wt \<int\> \<dbl\> \<dbl\> 1 1 73.8 242. 2 2 72.4 214. 3 3 75.6 291. 4 4 71.7 211. 5 5 75.7 281. 6 6 75.0 246.

클러스터 2와 4에는 공에서 더 멀리 떨어져 뛰어서 체구가 작은 선수들이 포함되어 있는 반면, 클러스터 3과 5에는 스크리미지 라인(line of scrimmage)을 따라 뛰는 더 큰 선수들이 있습니다. 클러스터 1(앞서 설명함)과 6에는 쿼터백, 타이트 엔드, 아웃사이드 라인배커, 그리고 때로는 수비 엔드와 같은 포지션에서 뛰는 "트위너(tweener)" 선수들이 더 많습니다(_트위너_ 선수는 여러 포지션을 잘 소화할 수 있지만 어떤 포지션에서든 특출나게 최고가 아닐 수도 있는 선수를 말합니다).

## 클러스터링에 대한 마무리 생각 (Closing Thoughts on Clustering)

이러한 초기 분석만으로도 이 접근법의 위력을 알 수 있습니다. 데이터를 약간 조정하는 것만으로 사전에 클러스터를 정의하지 않고도 선수들의 합리적인 그룹화를 만들 수 있기 때문입니다. 초기 데이터가 많지 않은 선수의 경우, 이를 통해 비교 대상, 적합성 및 기타 사항에 대한 대화를 시작할 수 있습니다. 또한 특정 코치의 시스템에 맞지 않는 선수를 솎아낼 수도 있습니다.

초기 그룹화가 완료되면 더 깊이 파고들 수 있습니다. 와이드 리시버만 놓고 볼 때, 이 와이드 리시버가 경합 상황에서 캐치로 승리하는 키가 크고 느린 유형입니까? 아니면 분리(separation)로 승리하는 키가 작고 날렵한 선수입니까? 그는 랜디 모스(Randy Moss)나 캘빈 존슨(Calvin Johnson) 같은 유니콘(unicorn)인가요? 그는 이미 로스터에 있는 기존 선수(팀이 연봉을 줄이고 싶어 하는 나이 든 선수)와 포지션이 겹치나요? 아니면 FA나 트레이드를 통해 떠난 선수를 대체하여 다른 선수를 보완해 주나요? *The Athletic*의 아리프 하산(Arif Hasan)은 [“Vikings Combine Trends: What Might They Look For in Their Offensive Draftees?”](https://oreil.ly/7zaxH)에서 특정 코치를 위한 이러한 특성들을 예로 들어 논의합니다.

클러스터링은 리시버의 패스 루트와 같은 것들을 그룹화하는 더 복잡한 문제에 사용되어 왔습니다. 이 문제에서는 선수의 실제 (_x_, _y_) 궤적에 대해 모델 기반 곡선 클러스터링(model-based curve clustering)을 사용하여 PFF와 같은 회사가 그동안 눈으로 해왔던 작업(분석을 위해 각 플레이를 차트로 작성)을 수학적으로 수행합니다. 앞서 언급했듯이 대부분의 구식 코치와 프런트 오피스 관계자들은 그룹화를 선호하므로 이러한 방법은 미식축구에서 항상 매력을 가질 것입니다. 다니 추(Dani Chu)와 그의 동료들은 [“Route Identification in the National Football League”](https://oreil.ly/BPi2e) (오픈 액세스 [프리프린트(preprint)](https://oreil.ly/OLwup)로도 제공됨)에서 루트 식별과 같은 접근법을 설명합니다.

k-평균 클러스터링이 가지는 구체적인 단점 중 일부는 초기 조건과 선택한 클러스터 수에 매우 민감하다는 것입니다. 이러한 문제를 줄이기 위해 (이전 예제에서 수행한 것처럼) 난수 시드 설정을 비롯한 조치를 취할 수 있지만, 현명한 분석가나 통계학자는 자신의 방법론이 가진 핵심 가정을 이해해야 합니다. 모든 것이 그렇듯 새로운 데이터가 들어올 때마다 지속적인 관리가 필요하며, 그래야만 해가 지남에 따라 결과가 너무 급격하게 변경되지 않습니다. 경기의 진화(evolution in the game)로 인해 어떤 해에는 클러스터를 추가하거나 삭제해야 할 수도 있지만, 이러한 결정은 그렇게 함으로써 발생하는 파급 효과에 대한 철저하고 신중한 분석 후에 내려져야 합니다.

# 이 장에서 사용된 데이터 과학 도구 (Data Science Tools Used in This Chapter)

이 장에서는 다음 주제들을 다루었습니다.

- <a href="ch07.html#sec-webscrap-draft" data-type="xref">7장</a>의 웹 스크래핑 도구를 약간 다른 웹페이지에 적용하기

- 차원의 수를 줄이기 위해 PCA 사용하기

- 그룹에 대한 데이터를 조사하기 위해 클러스터 분석 사용하기

# 연습 문제 (Exercises)

1. 모든 `NAs`가 포함된 원본 데이터에 대해 PCA를 수행하면 어떻게 될까요? 이것이 여러분의 향후 풋볼 분석 워크플로에 어떤 영향을 미칠까요?

2. 처음 세 개의 PC를 사용하여 이 장의 k-평균 클러스터링을 수행해 보세요. 차이점이 보이나요? 처음 네 개는 어떤가요?

3. 5개 및 7개의 클러스터를 사용하여 이 장의 k-평균 클러스터링을 수행해 보세요. 이것이 결과를 어떻게 바꾸나요?

4. 클러스터링 접근법을 사용하여 향상할 수 있는 이 책의 다른 문제는 무엇이 있을까요?

# 권장 도서 (Suggested Readings)

다양한 기초 통계학 서적에서 이 장에 제시된 방법들을 다루고 있습니다. 다음은 그 중 두 권입니다.

- 토머스 닐드(Thomas Nield)의 <a href="https://learning.oreilly.com/library/view/essential-math-for/9781098102920" class="orm:hideurl"><em>Essential Math for Data Science</em></a> (O’Reilly, 2022)는 응용 데이터 과학자를 위한 수학과 통계학에 대한 부드러운 소개를 제공합니다.

- 해들리 위컴(Hadley Wickham) 외 공저 <a href="https://learning.oreilly.com/library/view/r-for-data/9781492097396/" class="orm:hideurl"><em>R for Data Science</em></a> 제2판 (O’Reilly Media, 2023)은 응용 데이터 과학자를 위한 많은 도구와 방법론에 대한 소개를 제공합니다.
