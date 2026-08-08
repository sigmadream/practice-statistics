# 4장. 다중 회귀: 기대 대비 러싱 야드 (Multiple Regression: Rushing Yards Over Expected)

이전 장에서는 퍼스트 다운이나 터치다운을 얻기 위해 전진해야 하는 야드 수를 통제함으로써 다른 렌즈를 통해 러싱 야드를 살펴보았습니다. 퍼스트 다운이나 터치다운을 위해 얻어야 하는 야드가 많을수록 그라운드에서 야드를 얻기 더 쉽다는 상식이 통한다는 것을 알게 되었습니다. 이는 그러한 것들을 조정하는 것이 러닝백의 플레이를 이해하는 데 중요한 부분이 될 것임을 말해줍니다.

단순 선형 회귀의 한계는 러닝 게임에서 조정해야 할 중요한 변수가 하나 이상이라는 점입니다. 퍼스트 다운이나 터치다운까지의 거리가 매우 중요하지만, 어쩌면 다운(down)도 중요할 수 있습니다. 팀은 서드 다운에 10야드가 남았을 때보다 퍼스트 다운에 10야드가 남았을 때 공을 들고 뛸 가능성이 더 높으며, 따라서 수비팀은 후자보다는 전자에서 런을 막기 위해 장비를 갖출 가능성이 더 높습니다.

또 다른 예로 점수 차이(point differential)가 있습니다. 상대팀과 접전을 벌이고 있는 팀만큼 스크리미지 라인(line of scrimmage)에 많이 모여들지 않을 것이기 때문에 경기 점수는 여러 방식으로 기대치에 영향을 미칩니다. 일반적으로 풋볼 플레이를 평가할 때는 수많은 변수들을 *통제해야(controlled for)* 합니다. 이를 수행하는 방법이 바로 다중 선형 회귀입니다.

# 다중 선형 회귀의 정의

우리는 공을 들고 뛰는 것이 단지 한 가지 요인에 의해서만 영향을 받는 것이 아니라는 것을 알고 있으므로 러싱 야드를 예측하는 모델을 구축하되, 예측에 영향을 미칠 수 있는 다른 요인들을 설명하기 위해 더 많은 특성을 포함해야 합니다. 따라서 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>을 기반으로 다중 회귀가 필요합니다.

*다중 회귀(Multiple regression)* 는 예측 변수들의 선형 결합(또는 *회귀(regression)*)을 사용하여 단일 반응에 대한 여러(*다중(multiple)*) 예측 변수의 영향을 추정합니다. <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에서는 다중 회귀의 특수한 경우인 *단순 선형 회귀(simple linear regression)* 를 제시했습니다. 단순 선형 회귀에는 두 개의 매개변수, 즉 *절편(intercept)* (또는 평균값)과 *기울기(slope)* 가 존재합니다. 이것들은 연속형 예측 변수가 반응에 미치는 영향을 모델링합니다. <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에서 여러분의 단순 선형 회귀에 대한 Python/R 공식은 절편과 남은 야드(yards to go)에 의해 예측된 러싱 야드, 즉 `rushing_yards ~ 1 + ydstogo`를 가지고 있었습니다.

하지만 여러분은 여러 예측 변수에 관심을 가질 수 있습니다. 예를 들어 예상 러싱 야드를 추정할 때 다운에 대해 *수정(corrects)* 하는 다중 회귀를 고려해 보겠습니다. 남은 야드와 다운을 기반으로 러싱 야드를 예측하는 방정식(또는 공식)인 `rushing_yards ~ ydstogo + down`을 사용할 것입니다. 남은 야드는 *연속형(continuous)* 예측 변수의 한 예입니다. 비공식적으로, 연속형 예측 변수를 선수의 몸무게(예: 135, 302, 274 등)와 같은 숫자로 생각하세요. 사람들은 때때로 연속형 예측 변수에 *기울기*라는 용어를 사용합니다. 다운은 *이산형(discrete)* 예측 변수의 한 예입니다. 비공식적으로 이산형 예측 변수를 포지션(예: 러닝백, 쿼터백 등)과 같은 그룹이나 범주로 생각하세요. 기본적으로 `statsmodels`와 기본 R의 `formula` 옵션은 이산형 예측 변수를 *대조(contrasts)* 로 처리합니다.

예시 공식인 `rushing_yards ~ ydstogo + down`을 자세히 살펴보겠습니다. R은 가장 낮은(또는 알파벳 순으로 첫 번째) 다운에 대한 절편을 추정한 다음 나머지 다운에 대해서는 *대조*, 즉 차이를 추정합니다. 예를 들어, 4개의 예측 변수가 추정됩니다. 남은 야드에 대한 평균 `rushing_yards`인 절편 1개, 퍼스트 다운과 비교한 세컨드 다운에 대한 대조 1개, 퍼스트 다운과 비교한 서드 다운에 대한 대조 1개, 퍼스트 다운과 비교한 포스 다운에 대한 대조 1개입니다. 이를 확인하기 위해 R에서 *설계 행렬(design matrix)* 또는 *모델 행렬(model matrix)* 을 살펴보세요(Python에는 여기에는 표시되지 않은 유사한 기능이 있습니다). 먼저 데모 데이터 세트를 만듭니다.

```
## R
library(tidyverse)

demo_data_r <- tibble(down = c("first", "second"),
                      ydstogo = c(10, 5))
```

그런 다음, 공식의 우변과 `model.matrix()` 함수를 사용하여 모델 행렬을 만듭니다.

```
## R
model.matrix(~ ydstogo + down,
             data = demo_data_r)
```

결과는 다음과 같습니다.

(Intercept) ydstogo downsecond 1 1 10 0 2 1 5 1 attr(,"assign") \[1\] 0 1 2 attr(,"contrasts") attr(,"contrasts")\$down \[1\] "contr.treatment"

출력에 세 개의 열이 있다는 점에 유의하세요. 절편, `ydstogo`에 대한 기울기, 서드 다운에 대한 대조(`downsecond`)입니다.

하지만 `rushing_yards ~ ydstogo + down -1`과 같이 `-1`을 사용하여 각 다운에 대한 절편을 추정할 수도 있습니다. 이렇게 하면 퍼스트 다운에 대한 절편 1개, 세컨드 다운에 대한 절편 1개, 서드 다운에 대한 절편 1개, 포스 다운에 대한 절편 1개 등 총 4개의 예측 변수가 추정됩니다. R을 사용하여 예제 모델 행렬을 살펴보세요.

```
## R
model.matrix(~ ydstogo + down - 1,
             data = demo_data_r)
```

결과는 다음과 같습니다.

ydstogo downfirst downsecond 1 10 1 0 2 5 0 1 attr(,"assign") \[1\] 1 2 2 attr(,"contrasts") attr(,"contrasts")\$down \[1\] "contr.treatment"

이전과 열의 수는 동일하지만 각 다운에는 고유한 열이 있다는 점에 유의하세요.

###### 경고 (Warning)

Python 및 R과 같은 컴퓨터 언어는 다운(down)과 같은 일부 그룹에 혼동을 겪습니다. 컴퓨터는 이러한 예측 변수를 퍼스트 다운, 세컨드 다운, 서드 다운, 포스 다운이 아닌 숫자 1, 2, 3, 4와 같은 연속형으로 취급하려고 시도합니다. `pandas`에서는 `down`을 문자열(`str`)로 변경하고, R에서는 `down`을 문자(character)로 변경하게 됩니다.

다중 회귀의 공식은 많은 이산형 및 연속형 예측 변수를 허용합니다. `down + team`과 같이 여러 이산형 예측 변수가 존재하는 경우, 첫 번째 변수(이 경우 `down`)는 절편 또는 대조 매개변수로 추정될 수 있습니다. 다른 모든 이산형 예측 변수는 대조로 추정되며, 첫 번째 그룹화는 절편의 일부로 처리됩니다. 기울기와 절편에 얽매이기보다는 *계수(coefficients)* 라는 용어를 사용하여 다중 회귀에 대해 추정된 예측 변수를 설명할 수 있습니다.

# 탐색적 데이터 분석 (Exploratory Data Analysis)

러싱 야드의 경우, 다중 선형 회귀 모델에서 다운(`down`), 거리(`ydstogo`), 엔드존까지 남은 야드(`yardline_100`), 런 위치(`run_location`), 점수 차이(`score_differential`) 변수들을 *특성(features)* 으로 사용할 것입니다. 물론 다른 변수도 사용할 수 있지만, 지금은 이 변수들이 모두 어떤 방식으로든 러싱 야드에 영향을 미치기 때문에 주로 이 변수들을 사용합니다.

먼저, 사용할 데이터와 패키지를 로드하고, (<a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에서 했던 것처럼) 런(run) 데이터에 대해서만 필터링한 후 정규 다운이 아닌 플레이를 제거합니다. Python으로 다음을 수행하세요.

```
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

seasons = range(2016, 2022 + 1)
pbp_py = nfl.import_pbp_data(seasons)

pbp_py_run = \
    pbp_py\
    .query('play_type == "run" & rusher_id.notnull() &' +
           "down.notnull() & run_location.notnull()")\
    .reset_index()

pbp_py_run\
    .loc[pbp_py_run.rushing_yards.isnull(), "rushing_yards"] = 0
```

또는 R로 다음과 같이 수행합니다.

```
library(tidyverse)
library(nflfastR)

pbp_r <- load_pbp(2016:2022)

pbp_r_run <-
    pbp_r |>
    filter(play_type == "run" & !is.na(rusher_id) &
        !is.na(down) & !is.na(run_location)) |>
    mutate(rushing_yards = ifelse(is.na(rushing_yards),
        0,
        rushing_yards
    ))
```

다음으로 Python을 사용하여 다운 및 얻은 러싱 야드에 대한 히스토그램인 <a href="#fig-py-ry-box" data-type="xref">그림 4-1</a>을 만들어 보겠습니다.

```
## Python
# Change theme for chapter
sns.set_theme(style="whitegrid", palette="colorblind")

# Change down to be an integer
pbp_py_run.down =\
    pbp_py_run.down.astype(str)

# Plot rushing yards by down
g = \
    sns.FacetGrid(data=pbp_py_run,
                  col="down", col_wrap=2);
g.map_dataframe(sns.histplot, x="rushing_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0401.png" />
<h6 id="figure-4-1.-histogram-of-rushing-yards-by-downs-with-seaborn">그림 4-1. seaborn을 이용한 다운별 러싱 야드 히스토그램</h6>
</figure>

또는 R로 <a href="#fig-r-ry-box" data-type="xref">그림 4-2</a>를 만듭니다.

```
## R
# Change down to be an integer
pbp_r_run <-
    pbp_r_run |>
    mutate(down = as.character(down))

# Plot rushing yards by down
ggplot(pbp_r_run, aes(x = rushing_yards)) +
    geom_histogram(binwidth = 1) +
    facet_wrap(vars(down), ncol = 2,
               labeller = label_both) +
    theme_bw() +
    theme(strip.background = element_blank())
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0402.png" />
<h6 id="figure-4-2.-histogram-of-rushing-yards-by-downs-with-ggplot2">그림 4-2. ggplot2를 이용한 다운별 러싱 야드 히스토그램</h6>
</figure>

다운이 증가함에 따라 러싱 야드가 감소하는 것처럼 보이기 때문에 이는 흥미롭습니다. 하지만 데이터에는 교란 요인(confounder)이 존재하는데, 그것은 바로 런(runs)이 종종 거리가 짧게 남은 후반 다운(late downs)에서 발생한다는 것입니다. `ydstogo == 10`인 상황만 살펴봅시다. Python에서 <a href="#fig-py-ry-down-box" data-type="xref">그림 4-3</a>을 생성합니다.

```
## Python
sns.boxplot(data=pbp_py_run.query("ydstogo == 10"),
            x="down",
            y="rushing_yards");
plt.show()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0403.png" />
<h6 id="figure-4-3.-boxplot-of-rushing-yards-by-downs-for-plays-with-10-yards-to-go-seaborn">그림 4-3. 10야드가 남은 플레이에 대한 다운별 러싱 야드의 박스플롯 (seaborn)</h6>
</figure>

또는 R을 사용하여 <a href="#fig-r-ry-down-box" data-type="xref">그림 4-4</a>를 생성합니다.

```
## R
pbp_r_run |>
    filter(ydstogo == 10) |>
    ggplot(aes(x = down, y = rushing_yards)) +
    geom_boxplot() +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0404.png" />
<h6 id="figure-4-4.-boxplot-of-rushing-yards-by-downs-for-plays-with-10-yards-to-go-ggplot2">그림 4-4. 10야드가 남은 플레이에 대한 다운별 러싱 야드의 박스플롯 (ggplot2)</h6>
</figure>

좋습니다. 이제 예상했던 내용을 보게 되었습니다. 이것이 *심슨의 역설(Simpson’s paradox)* 의 한 예입니다. 추가적인 세 번째 그룹화 변수를 포함하면 두 다른 변수 사이의 관계가 바뀝니다. 그럼에도 불구하고, 다운이 플레이에서의 러싱 야드에 영향을 미친다는 것은 분명하며 이 점을 고려해야 합니다. 마찬가지로 <a href="#fig-py-ry-y100-lm" data-type="xref">그림 4-5</a>를 통해 `seaborn`에서 엔드존까지 남은 야드를 살펴보겠습니다(`scatter_kws={'alpha':0.25}`로 투명도를 변경하고 `line_kws={'color': 'red'}`로 회귀선의 색상을 변경합니다).

```
## Python
sns.regplot(
    data=pbp_py_run,
    x="yardline_100",
    y="rushing_yards",
    scatter_kws={"alpha": 0.25},
    line_kws={"color": "red"}
);
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0405.png" />
<h6 id="figure-4-5.-scatterplot-with-linear-trendline-for-ball-position-yards-to-go-to-the-endzone-and-rushing-yards-from-a-play-seaborn">그림 4-5. 볼 위치(엔드존까지 남은 야드)와 플레이에서 획득한 러싱 야드에 대한 선형 추세선이 포함된 산점도 (seaborn)</h6>
</figure>

또는 R에서 <a href="#fig-r-ry-y100-lm" data-type="xref">그림 4-6</a>을 생성합니다(겹쳐진 점들을 쉽게 볼 수 있도록 `alpha = 0.25`로 투명도를 변경합니다).

```
## R
ggplot(pbp_r_run, aes(x = yardline_100, y = rushing_yards)) +
    geom_point(alpha = 0.25) +
    stat_smooth(method = "lm") +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0406.png" />
<h6 id="figure-4-6.-scatterplot-with-linear-trendline-for-ball-position-yards-to-go-to-the-endzone-and-rushing-yards-from-a-play-ggplot2">그림 4-6. 볼 위치(엔드존까지 남은 야드)와 플레이에서 획득한 러싱 야드에 대한 선형 추세선이 포함된 산점도 (ggplot2)</h6>
</figure>

이것만으로는 별 의미가 없어 보이지만 Python을 사용하여 구간화(bin)하고 평균을 낸 후 어떤 일이 일어나는지 살펴봅시다.

```
## Python
pbp_py_run_y100 =\
    pbp_py_run\
    .groupby("yardline_100")\
    .agg({"rushing_yards": ["mean"]})

pbp_py_run_y100.columns =\
    list(map("_".join, pbp_py_run_y100.columns))

pbp_py_run_y100.reset_index(inplace=True)
```

이제 이 결과들을 활용하여 <a href="#fig-py-ry-y100-lm_bin" data-type="xref">그림 4-7</a>을 만들어 보세요.

```
sns.regplot(
    data=pbp_py_run_y100,
    x="yardline_100",
    y="rushing_yards_mean",
    scatter_kws={"alpha": 0.25},
    line_kws={"color": "red"}
);
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0407.png" />
<h6 id="figure-4-7.-scatterplot-with-linear-trendline-for-ball-position-and-rushing-yards-for-data-binned-by-yard-seaborn">그림 4-7. 1야드 단위로 구간화된 데이터의 볼 위치 및 러싱 야드에 대한 선형 추세선이 포함된 산점도 (seaborn)</h6>
</figure>

또는 R에서 <a href="#fig-r-ry-y100-lm_bin" data-type="xref">그림 4-8</a>을 생성합니다.

```
## R
pbp_r_run |>
    group_by(yardline_100) |>
    summarize(rushing_yards_mean = mean(rushing_yards)) |>
    ggplot(aes(x = yardline_100, y = rushing_yards_mean)) +
    geom_point() +
    stat_smooth(method = "lm") +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0408.png" />
<h6 id="figure-4-8.-scatterplot-with-linear-trendline-for-ball-position-and-rushing-yards-for-data-binned-by-yard-ggplot2">그림 4-8. 1야드 단위로 구간화된 데이터의 볼 위치 및 러싱 야드에 대한 선형 추세선이 포함된 산점도 (ggplot2)</h6>
</figure>

그림 <a href="#fig-py-ry-y100-lm_bin" data-type="xref" data-xrefstyle="select:labelnumber">4-7</a>과 <a href="#fig-r-ry-y100-lm_bin" data-type="xref" data-xrefstyle="select:labelnumber">4-8</a>은 미식축구에 대한 몇 가지 통찰을 보여줍니다. 남은 거리가 15야드 미만인 러닝 플레이는 엔드존까지 남은 거리가 제한적이고 레드존(red-zone) 수비가 더 빡빡하기 때문에 거리의 제한을 받습니다. 마찬가지로, 남은 거리가 90야드 이상인 플레이는 팀을 자기 진영 엔드존에서 빠져나오게 합니다. 따라서 수비팀은 세이프티(safety)를 강제하려고 노력할 것이고, 공격팀은 펀트(punt)를 차거나 세이프티를 허용하지 않기 위해 보수적인 플레이를 할 가능성이 높습니다.

여기서 여러분은 평균 러싱 야드와 엔드존까지 남은 야드 사이의 명확한 양의(하지만 비선형적인) 관계를 확인할 수 있으므로, 모델에 이 특성을 포함하는 것이 유익합니다. <a href="#sec-check-lin" data-type="xref">"선형성 가정(Assumption of Linearity)"</a>에서는 15야드 미만 또는 90야드 이상인 값을 제거할 경우 모델에 어떤 일이 일어나는지 확인할 수 있습니다. 실제로는 더 복잡한 모델들이 이러한 비선형성을 효과적으로 처리할 수 있지만, 그 내용은 다른 책에서 다루도록 남겨두겠습니다. 이제 Python을 사용해 <a href="#fig-py-ry-run_loc" data-type="xref">그림 4-9</a>에서 런 위치를 살펴봅시다.

```
## Python
sns.boxplot(data=pbp_py_run,
            x="run_location",
            y="rushing_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0409.png" />
<h6 id="figure-4-9.-boxplot-of-rushing-yards-by-run-location-seaborn">그림 4-9. 런 위치별 러싱 야드의 박스플롯 (seaborn)</h6>
</figure>

또는 R에서 <a href="#fig-r-ry-run_loc" data-type="xref">그림 4-10</a>을 생성합니다.

```
## R
ggplot(pbp_r_run, aes(run_location, rushing_yards)) +
    geom_boxplot() +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0410.png" />
<h6 id="figure-4-10.-boxplot-of-rushing-yards-by-run-location-ggplot2">그림 4-10. 런 위치별 러싱 야드의 박스플롯 (ggplot2)</h6>
</figure>

여기서 평균/중앙값이 약간 다를 뿐만 아니라 분산/사분위 범위(variances/interquartile ranges)도 다양하게 나타나는 것 같으므로 이를 모델에 유지하세요. 런 위치에 대한 또 다른 주석은 각 경우의 75번째 백분위수(75th percentile)가 정말 낮다는 것입니다. 선수가 왼쪽, 오른쪽, 중앙 어느 쪽으로 가든 4분의 3의 시간 동안 10야드 이하로 이동합니다. 긴 러시(long rush)를 볼 수 있는 경우는 극히 드뭅니다.

마지막으로 Python에서 엔드존까지 남은 야드에 대해 사용했던 구간화(binning) 및 집계를 사용하여 점수 차이를 살펴봅시다.

```
## Python
pbp_py_run_sd = \
    pbp_py_run\
    .groupby("score_differential")\
    .agg({"rushing_yards": ["mean"]}
)

pbp_py_run_sd.columns =\
     list(map("_".join, pbp_py_run_sd.columns))

pbp_py_run_sd.reset_index(inplace=True)
```

이제 이 결과들을 사용하여 <a href="#fig-py-ry-sd" data-type="xref">그림 4-11</a>을 만들어 보세요.

```
## Python
sns.regplot(
    data=pbp_py_run_sd,
    x="score_differential",
    y="rushing_yards_mean",
    scatter_kws={"alpha": 0.25},
    line_kws={"color": "red"}
);
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0411.png" />
<h6 id="figure-4-11.-scatterplot-with-linear-trendline-for-score-differential-and-rushing-yards-for-data-binned-by-score-differential-seaborn">그림 4-11. 점수 차이 단위로 구간화된 데이터의 점수 차이 및 러싱 야드에 대한 선형 추세선이 포함된 산점도 (seaborn)</h6>
</figure>

또는 R에서 <a href="#fig-r-ry-sd" data-type="xref">그림 4-12</a>를 생성합니다.

```
## R
pbp_r_run |>
    group_by(score_differential) |>
    summarize(rushing_yards_mean = mean(rushing_yards)) |>
    ggplot(aes(score_differential, rushing_yards_mean)) +
    geom_point() +
    stat_smooth(method = "lm") +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0412.png" />
<h6 id="figure-4-12.-scatterplot-with-linear-trendline-for-score-differential-and-rushing-yards-for-data-binned-by-score-differential-ggplot2">그림 4-12. 점수 차이 단위로 구간화된 데이터의 점수 차이 및 러싱 야드에 대한 선형 추세선이 포함된 산점도 (ggplot2)</h6>
</figure>

앞서 가설을 세웠던 것처럼 분명한 음의 관계를 볼 수 있습니다. 따라서 모델에 점수 차이를 남겨두게 될 것입니다.

###### 팁 (Tip)

이 책에서 플롯에 대한 코드를 볼 때 플롯을 어떻게 개선할지 생각해 보세요. 또한 코드를 살펴보고 이해가 안 되는 인수가 있다면 검색하여 플롯팅 코드를 변경하는 방법을 알아보세요. 더 나은 데이터 플로터(data plotter)가 되는 가장 좋은 방법은 직접 플롯을 탐색하고 만들어 보는 것입니다.

# 다중 선형 회귀 적용하기

이제 다중 선형 회귀를 적용하여 기대 대비 러싱 야드(RYOE)를 다시 도출합니다. 이 섹션에서는 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에서 다룬 단계들을 간략히 설명합니다.

먼저, 모델을 맞춥니다. 그런 다음 계산된 잔차를 RYOE로 저장합니다. 잔차는 모델이 예측한 값과 데이터에서 관측된 값의 차이라는 점을 상기하세요. 이는 관측된 러싱 야드에서 모델에서 예측된 러싱 야드를 빼서 직접 계산할 수 있습니다. 하지만 잔차는 통계에서 흔히 사용되며 Python과 R 모두 잔차를 모델 적합성의 일부로 포함합니다. 이 파생 방식(derivation)은 더 복잡한 모델을 만들었기 때문에 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>의 방법과 다릅니다.

모델은 다음을 생성하여 `rushing_yards`를 예측합니다.

- 절편 (`1`)

- 세컨드, 서드, 포스 다운을 퍼스트 다운과 대조하는 항 (`down`)

- `ydstogo`에 대한 계수

- 각 `down`에 대한 `ydstogo` 대조를 추정하는 `ydstogo`와 `down` 간의 *상호 작용(interaction)* (`ydstogo:down`)

- 엔드존까지 남은 야드에 대한 계수 (`yardline_100`)

- 필드 내 러닝 플레이 위치 (`run_location`)

- 각 팀 간의 점수 차이 (`score_differential`)

###### 참고 (Note)

공식을 사용하여 상호 작용을 나타내는 여러 가지 접근 방식이 있습니다. 이 예제에 대해 가장 길지만 가장 직관적인 접근 방식은 `down + ydstogo + as.factor(down):ydstogo`일 것입니다. 이는 `down * ydstogo`로 줄여서 쓸 수 있습니다. 따라서 예제 공식인 `rushing_yards ~ 1 + down + ydstogo + down:ydstogo + yardline_100 + run_location + score_differential`는 `rushing_yards ~ down * ydstogo + yardline_100 + run_location + score_differential`로 작성될 수 있으며 3개의 항을 쓰는 수고를 덜어줍니다.

Python의 경우 `statsmodels` 패키지를 사용하여 모델을 맞춥니다. 그런 다음 잔차를 `RYOE`로 저장합니다.

```
## Python
pbp_py_run.down =\
    pbp_py_run.down.astype(str)

expected_yards_py =\
    smf.ols(
        data=pbp_py_run,
        formula="rushing_yards ~ 1 + down + ydstogo + " +
        "down:ydstogo + yardline_100 + " +
        "run_location + score_differential")\
        .fit()

pbp_py_run["ryoe"] =\
    expected_yards_py.resid
```

###### 참고 (Note)

이 책의 페이지에 맞도록 코드에 줄 바꿈을 포함했습니다. 예를 들어 Python에서는 `"rushing_yards ~ 1 + down + ydstogo + "` 문자열 다음에 `+`를 쓰고 줄 바꿈을 합니다. 그런 다음 `"down:ydstogo + yardline_100 + "`과 `"run_location + score_differential"` 문자열 각각에 줄 바꿈을 한 다음, `+` 문자를 사용하여 문자열들을 함께 *추가(add)* 합니다. 이러한 줄 바꿈은 필수는 아니지만 사람의 눈에 코드가 더 잘 보이게 하고 이상적으로는 가독성을 높이는 데 도움이 됩니다.

마찬가지로 R에서 모델을 맞추고 잔차를 `RYOE`로 저장합니다.

```
## R
pbp_r_run <-
    pbp_r_run |>
    mutate(down = as.character(down))

expected_yards_r <-
    lm(rushing_yards ~ 1 + down + ydstogo + down:ydstogo +
       yardline_100 + run_location + score_differential,
       data = pbp_r_run
    )

pbp_r_run <-
    pbp_r_run |>
    mutate(ryoe = resid(expected_yards_r))
```

이제 Python에서 모델의 요약(summary)을 살펴보세요.

```
## Python
print(expected_yards_py.summary())
```

결과는 다음과 같습니다.

OLS Regression Results ============================================================================== Dep. Variable: rushing_yards R-squared: 0.016 Model: OLS Adj. R-squared: 0.016 Method: Least Squares F-statistic: 136.6 Date: Sun, 04 Jun 2023 Prob (F-statistic): 3.43e-313 Time: 09:36:43 Log-Likelihood: -2.9764e+05 No. Observations: 91442 AIC: 5.953e+05 Df Residuals: 91430 BIC: 5.954e+05 Df Model: 11 Covariance Type: nonrobust =============================================================================== coef std err t P\>\|t\| \[0.025 0.975\] ------------------------------------------------------------------------------ Intercept 1.6085 0.136 11.849 0.000 1.342 1.875 down\[T.2.0\] 1.6153 0.153 10.577 0.000 1.316 1.915 down\[T.3.0\] 1.2846 0.161 7.990 0.000 0.969 1.600 down\[T.4.0\] 0.2844 0.249 1.142 0.254 -0.204 0.773 run_location\[T.middle\] -0.5634 0.053 -10.718 0.000 -0.666 -0.460 run_location\[T.right\] -0.0382 0.049 -0.784 0.433 -0.134 0.057 ydstogo 0.2024 0.014 14.439 0.000 0.175 0.230 down\[T.2.0\]:ydstogo -0.1466 0.016 -8.957 0.000 -0.179 -0.115 down\[T.3.0\]:ydstogo -0.0437 0.019 -2.323 0.020 -0.081 -0.007 down\[T.4.0\]:ydstogo 0.2302 0.090 2.567 0.010 0.054 0.406 yardline_100 0.0186 0.001 21.230 0.000 0.017 0.020 score_differential -0.0040 0.002 -2.023 0.043 -0.008 -0.000 ============================================================================== Omnibus: 80510.527 Durbin-Watson: 1.979 Prob(Omnibus): 0.000 Jarque-Bera (JB): 3941200.520 Skew: 4.082 Prob(JB): 0.00 Kurtosis: 34.109 Cond. No. 838. ============================================================================== Notes: \[1\] Standard Errors assume that the covariance matrix of the errors is correctly specified.

또는 R에서 모델의 요약을 검토하세요.

```
## R
print(summary(expected_yards_r))
```

결과는 다음과 같습니다.

Call: lm(formula = rushing_yards ~ 1 + down + ydstogo + down:ydstogo + yardline_100 + run_location + score_differential, data = pbp_r_run) Residuals: Min 1Q Median 3Q Max -32.233 -3.130 -1.173 1.410 94.112 Coefficients: Estimate Std. Error t value Pr(\>\|t\|) (Intercept) 1.608471 0.135753 11.849 \< 2e-16 \*\*\* down2 1.615277 0.152721 10.577 \< 2e-16 \*\*\* down3 1.284560 0.160775 7.990 1.37e-15 \*\*\* down4 0.284433 0.249106 1.142 0.2535 ydstogo 0.202377 0.014016 14.439 \< 2e-16 \*\*\* yardline_100 0.018576 0.000875 21.230 \< 2e-16 \*\*\* run_locationmiddle -0.563369 0.052565 -10.718 \< 2e-16 \*\*\* run_locationright -0.038176 0.048684 -0.784 0.4329 score_differential -0.004028 0.001991 -2.023 0.0431 \* down2:ydstogo -0.146602 0.016367 -8.957 \< 2e-16 \*\*\* down3:ydstogo -0.043703 0.018814 -2.323 0.0202 \* down4:ydstogo 0.230179 0.089682 2.567 0.0103 \* --- Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1 Residual standard error: 6.272 on 91430 degrees of freedom Multiple R-squared: 0.01617, Adjusted R-squared: 0.01605 F-statistic: 136.6 on 11 and 91430 DF, p-value: \< 2.2e-16

각각의 추정된 계수는 플레이 중 러싱 야드에 대한 이야기를 전달하는 데 도움을 줍니다.

- 세컨드 다운(Python의 `down[T.2.0]` 또는 R의 `down2`)의 러닝 플레이는 다른 모든 조건이 동일할 때 퍼스트 다운에 비해 캐리당 기대 야드가 더 많으며, 이 경우 약 1.6야드입니다.

- 서드 다운(Python의 `down[T.3.0]` 또는 R의 `down3`)의 러닝 플레이는 다른 모든 조건이 동일할 때 퍼스트 다운에 비해 캐리당 기대 야드가 더 많으며, 이 경우 약 1.3야드입니다.

- 상호 작용 항은 퍼스트 다운을 얻기 위해 남은 야드가 적을수록 이 점이 특히 더 사실임을 알려줍니다(상호 작용 항은 모두 음수임). 풋볼 관점에서 볼 때 이것은 퍼스트 다운에 10야드가 남은 상황보다 세컨드 다운, 서드 다운이거나 퍼스트 다운 또는 터치다운까지 남은 거리가 짧을 때 공을 들고 뛰는 공격에 더 유리하다는 의미일 뿐입니다.

- 반대로, 포스 다운의 러닝 플레이는 다른 모든 조건이 동일할 때 퍼스트 다운(Python의 `down[T.4.0]` 또는 R의 `down4`)과 비교하여 얻은 야드 수가 약간 더 많지만, 세컨드 다운이나 서드 다운만큼은 아닙니다.

- 다른 모든 조건이 동일할 때, 남은 야드 수(`ydstogo`)가 증가함에 따라 러싱 야드도 증가하며 남은 1야드는 약 5분의 1야드의 가치가 있습니다. 이는 `ydstogo` 추정치가 양수이기 때문입니다.

- 공이 엔드존에서 멀어질수록 러싱 플레이는 플레이당 야드를 약간 더 많이 생산합니다(엔드존까지 남은 야드당 약 0.02, `yardline_100`). 예를 들어, 팀에게 100야드가 남았더라도 0.02의 계수는 플레이에서 단 2야드의 추가 러싱이 이루어진다는 의미인데, 이는 다른 계수와 비교할 때 큰 영향을 미치지 않습니다.

- 필드 중앙과 필드 좌측의 대조 추정치(Python의 `run_location[T.middle]` 또는 R의 `run_locationmiddle`)를 바탕으로 볼 때, 필드 중앙에서의 러싱 플레이는 필드 좌측으로 향하는 플레이보다 약 0.5야드를 덜 얻습니다.

- 음수의 `score_differential` 계수는 통계적으로 0과 다릅니다. 따라서 팀이 앞서고 있을 때(양수의 점수 차이를 가질 때), 그들은 평균 러닝 플레이에서 플레이당 획득하는 야드가 줄어듭니다. 그러나 이 효과는 너무 작고(0.004) 다른 계수에 비해 그다지 중요하지 않기 때문에 무시할 수 있습니다 (예를 들어 50점 차이로 이기고 있으면 캐리당 야드 수가 0.2야드만 감소함).

모든 다른 요인이 동일할 때, 필드 중앙으로 뛰는 것이 바깥쪽으로 뛰는 것보다 더 어렵다는 것을 계수에서 확인하세요. 실제로 퍼스트 다운 마커까지의 거리나 남은 야드, 그리고 엔드존까지의 거리가 모두 러싱 야드에 긍정적인 영향을 미친다는 것을 알 수 있습니다. 즉 공격 선수가 골대에서 멀리 떨어져 있을수록 수비팀은 평균적으로 해당 공격 선수에게 더 많은 거리를 내주게 됩니다.

###### 팁 (Tip)

R의 `kableExtra` 패키지는 R Markdown 및 Quarto 문서뿐만 아니라 화면상에서도 형식이 잘 갖춰진 표를 생성하는 데 도움이 됩니다. 아직 설치하지 않았다면 패키지를 설치해야 합니다.

표는 회귀 계수를 나타내는 또 다른 방법을 제공합니다. 예를 들어, `broom` 패키지를 사용하면 R에서 <a href="#tbl-reg-out" data-type="xref">표 4-1</a>과 같이 `kableExtra` 패키지를 사용하여 형식을 지정할 수 있는 깔끔한(tidy) 표를 만들 수 있습니다. 특히 이 코드를 사용하여 모델 적합성인 `expected_yards_r`을 취한 다음, 모델을 파이프하여 `tidy(conf.int = TRUE)`를 사용하여 *깔끔한(tidy)* 모델 출력(95% CI 포함)을 추출하세요. 그런 다음 `kbl(format = "pipe", digits = 2)`를 사용하여 표를 `kable` 표로 변환하고 두 자릿수를 표시합니다. 마지막으로 `kable_styling()`을 사용하여 `kableExtra` 패키지의 스타일링을 적용합니다.

```
## R
library(broom)
library(kableExtra)
expected_yards_r |>
    tidy(conf.int = TRUE) |>
    kbl(format = "pipe", digits = 2) |>
    kable_styling()
```

| Term | `estimate` | `std.error` | `statistic` | `p.value` | `conf.low` | `conf.high` |
|----|----|----|----|----|----|----|
| (Intercept) | 1.61 | 0.14 | 11.85 | 0.00 | 1.34 | 1.87 |
| `down2` | 1.62 | 0.15 | 10.58 | 0.00 | 1.32 | 1.91 |
| `down3` | 1.28 | 0.16 | 7.99 | 0.00 | 0.97 | 1.60 |
| `down4` | 0.28 | 0.25 | 1.14 | 0.25 | -0.20 | 0.77 |
| `ydstogo` | 0.20 | 0.01 | 14.44 | 0.00 | 0.17 | 0.23 |
| `yardline_100` | 0.02 | 0.00 | 21.23 | 0.00 | 0.02 | 0.02 |
| `run_locationmiddle` | -0.56 | 0.05 | -10.72 | 0.00 | -0.67 | -0.46 |
| `run_locationright` | -0.04 | 0.05 | -0.78 | 0.43 | -0.13 | 0.06 |
| `score_differential` | 0.00 | 0.00 | -2.02 | 0.04 | -0.01 | 0.00 |
| `down2:ydstogo` | -0.15 | 0.02 | -8.96 | 0.00 | -0.18 | -0.11 |
| `down3:ydstogo` | -0.04 | 0.02 | -2.32 | 0.02 | -0.08 | -0.01 |
| `down4:ydstogo` | 0.23 | 0.09 | 2.57 | 0.01 | 0.05 | 0.41 |

표 4-1. 회귀 계수 표의 예. `term`은 회귀 계수이고, `estimate`는 계수의 추정값이고, `std.error`는 표준 오차이며, `statistic`은 *t*-점수이고, `p.value`는 *p*-값이며, `conf.low`는 95% CI의 하단이고, `conf.high`는 95% CI의 상단입니다. {#tbl-reg-out style="width: 100%"}

###### 팁 (Tip)

회귀에 대해 글을 쓰는 것은 어려울 수 있으며 청중과 그들의 배경지식을 파악하는 것이 중요합니다. 예를 들어 "계수에서 확인하세요..."라는 단락은 캐주얼한 블로그에는 적절하겠지만 동료 평가를 받는 스포츠 저널(peer-reviewed sports journal)에는 적합하지 않을 것입니다. 마찬가지로 <a href="#tbl-reg-out" data-type="xref">표 4-1</a>과 같은 표는 학술지 기사에 포함될 수 있지만 기술 보고서나 학술지 기사의 보충 자료의 일부로 포함될 가능성이 더 높습니다. 개별 계수에 대해 글머리 기호가 있는 목록 형식으로 설명하는 방법은 항목별 설명을 원하는 고객에 대한 보고서나 풋볼 분석에 관한 블로그나 책과 같은 교육 자료에 적합할 수 있습니다.

# RYOE 분석

<a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에 있는 RYOE의 첫 번째 버전과 마찬가지로 이제 러셔를 위한 메트릭의 새 버전을 분석할 것입니다. 먼저 Python에서 총 RYOE, 평균, 캐리당 야드에 대한 요약 표를 생성합니다. 다음으로 캐리 횟수가 50회 이상인 선수들의 데이터만 저장합니다. 또한 열의 이름을 바꾸고 총 RYOE를 기준으로 정렬합니다.

```
## Python
ryoe_py =\
    pbp_py_run\
    .groupby(["season", "rusher_id", "rusher"])\
    .agg({
        "ryoe": ["count", "sum", "mean"],
        "rushing_yards": ["mean"]})

ryoe_py.columns =\
    list(map("_".join, ryoe_py.columns))
ryoe_py.reset_index(inplace=True)

ryoe_py =\
    ryoe_py\
    .rename(columns={
        "ryoe_count": "n",
        "ryoe_sum": "ryoe_total",
        "ryoe_mean": "ryoe_per",
        "rushing_yards_mean": "yards_per_carry"
    })\
    .query("n > 50")

print(ryoe_py\
    .sort_values("ryoe_total", ascending=False)
    )
```

결과는 다음과 같습니다.

season rusher_id rusher ... ryoe_total ryoe_per yards_per_carry 1870 2021 00-0036223 J.Taylor ... 471.232840 1.419376 5.454819 1350 2020 00-0032764 D.Henry ... 345.948778 0.875820 5.232911 1183 2019 00-0034796 L.Jackson ... 328.524757 2.607339 6.880952 1069 2019 00-0032764 D.Henry ... 311.641243 0.807361 5.145078 1383 2020 00-0033293 A.Jones ... 301.778866 1.365515 5.565611 ... ... ... ... ... ... ... ... 627 2018 00-0027029 L.McCoy ... -208.392834 -1.294365 3.192547 51 2016 00-0027155 R.Jennings ... -228.084591 -1.226261 3.344086 629 2018 00-0027325 L.Blount ... -235.865233 -1.531592 2.714286 991 2019 00-0030496 L.Bell ... -338.432836 -1.381359 3.220408 246 2016 00-0032241 T.Gurley ... -344.314622 -1.238542 3.183453 \[533 rows x 7 columns\]

다음으로 Python에서 캐리당 평균 RYOE를 기준으로 정렬합니다.

```
## Python
print(
    ryoe_py\
    .sort_values("ryoe_per", ascending=False)
    )
```

결과는 다음과 같습니다.

season rusher_id rusher ... ryoe_total ryoe_per yards_per_carry 2103 2022 00-0034796 L.Jackson ... 280.752317 3.899338 7.930556 1183 2019 00-0034796 L.Jackson ... 328.524757 2.607339 6.880952 1210 2019 00-0035228 K.Murray ... 137.636412 2.596913 6.867925 2239 2022 00-0036945 J.Fields ... 177.409631 2.304021 6.506494 1467 2020 00-0034796 L.Jackson ... 258.059489 2.186945 6.415254 ... ... ... ... ... ... ... ... 1901 2021 00-0036414 C.Akers ... -129.834294 -1.803254 2.430556 533 2017 00-0032940 D.Washington ... -105.377929 -1.848736 2.684211 1858 2021 00-0035860 T.Jones ... -100.987077 -1.870131 2.629630 60 2016 00-0027791 J.Starks ... -129.298259 -2.052353 2.301587 1184 2019 00-0034799 K.Ballage ... -191.983153 -2.594367 1.824324 \[533 rows x 7 columns\]

동일한 표를 R에서도 만들 수 있습니다.

```


## R
ryoe_r <-
    pbp_r_run |>
    group_by(season, rusher_id, rusher) |>
    summarize(
        n = n(), ryoe_total = sum(ryoe), ryoe_per = mean(ryoe),
        yards_per_carry = mean(rushing_yards)
    ) |>
    filter(n > 50)

ryoe_r |>
    arrange(-ryoe_total) |>
    print()
```

결과는 다음과 같습니다.

\# A tibble: 533 × 7 \# Groups: season, rusher_id \[533\] season rusher_id rusher n ryoe_total ryoe_per yards_per_carry \<dbl\> \<chr\> \<chr\> \<int\> \<dbl\> \<dbl\> \<dbl\> 1 2021 00-0036223 J.Taylor 332 471. 1.42 5.45 2 2020 00-0032764 D.Henry 395 346. 0.876 5.23 3 2019 00-0034796 L.Jackson 126 329. 2.61 6.88 4 2019 00-0032764 D.Henry 386 312. 0.807 5.15 5 2020 00-0033293 A.Jones 221 302. 1.37 5.57 6 2022 00-0034796 L.Jackson 72 281. 3.90 7.93 7 2019 00-0031687 R.Mostert 190 274. 1.44 5.83 8 2016 00-0033045 E.Elliott 342 274. 0.800 5.14 9 2020 00-0034796 L.Jackson 118 258. 2.19 6.42 10 2021 00-0034791 N.Chubb 228 248. 1.09 5.52 \# ℹ 523 more rows

그런 다음 R에서 캐리당 평균 RYOE로 정렬합니다.

```
## R
ryoe_r |>
    filter(n > 50) |>
    arrange(-ryoe_per) |>
    print()
```

결과는 다음과 같습니다.

\# A tibble: 533 × 7 \# Groups: season, rusher_id \[533\] season rusher_id rusher n ryoe_total ryoe_per yards_per_carry \<dbl\> \<chr\> \<chr\> \<int\> \<dbl\> \<dbl\> \<dbl\> 1 2022 00-0034796 L.Jackson 72 281. 3.90 7.93 2 2019 00-0034796 L.Jackson 126 329. 2.61 6.88 3 2019 00-0035228 K.Murray 53 138. 2.60 6.87 4 2022 00-0036945 J.Fields 77 177. 2.30 6.51 5 2020 00-0034796 L.Jackson 118 258. 2.19 6.42 6 2017 00-0027939 C.Newton 92 191. 2.08 6.17 7 2020 00-0035228 K.Murray 70 144. 2.06 6.06 8 2021 00-0034750 R.Penny 119 242. 2.03 6.29 9 2019 00-0034400 J.Wilkins 51 97.8 1.92 6.02 10 2022 00-0033357 T.Hill 95 171. 1.80 6.05 \# ℹ 523 more rows

위의 결과는 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>의 결과와 유사하지만 RYOE를 추정할 때 추가적인 특징(features)을 *보정(corrects for)*한 모델에서 나온 것입니다.

총 RYOE에 관해서는 조나단 테일러(Jonathan Taylor)의 2021 시즌이 여전히 1위를 차지하고 있으며, 이제 2위 선수보다 100야드 이상 더 많은 기대 야드를 기록하고 있습니다. 데릭 헨리(Derrick Henry)도 몇 번 다시 등장합니다. 닉 처브(Nick Chubb)는 2018년 2라운드에 지명된 이후 리그 전체에서 최고의 러너 중 한 명이었으며, 2019년 NFC 챔피언인 샌프란시스코 포티나이너스(San Francisco 49ers)의 라힘 모스터트(Raheem Mostert)는 드래프트에 지명되지도 않았지만 명단에 올랐습니다. 카우보이스(Cowboys)의 에제키엘 엘리엇(Ezekiel Elliott)과 피츠버그 스틸러스(Pittsburgh Steelers), 뉴욕 제츠(New York Jets), 치프스(Chiefs), 레이븐스(Ravens)의 르비온 벨(Le’Veon Bell)도 여러 곳에 등장하는데, 이에 대해서는 나중에 이야기하겠습니다.

캐리당 RYOE(한 시즌 캐리 횟수가 50회 이상인 선수)에 관해서는 래샤드 페니(Rashaad Penny)의 빛나는 2021 시즌이 다시 돋보입니다. 하지만 이 명단의 대부분은 2019년 리그 MVP인 라마 잭슨(Lamar Jackson), 2011년과 2019년 전체 1순위 지명자인 캠 뉴튼(Cam Newton)과 카일러 머레이(Kyler Murray), 뉴올리언스 세인츠(New Orleans Saints)의 테이섬 힐(Taysom Hill)과 같은 쿼터백입니다. 2022년에 쿼터백 포지션 역사상 최고의 러싱 시즌 중 하나를 보냈던 저스틴 필즈(Justin Fields)도 등장합니다.

캐리당 야드 대비 이 메트릭의 안정성과 관련하여 이전 장에서 안정성이 약간 상승했지만 러셔를 평가할 때 캐리당 RYOE가 캐리당 야드보다 확실히 우수한 메트릭이라고 말하기에는 충분하지 않았음을 상기해 보세요. 이 분석을 다시 해봅시다.

```
## Python
#  필요한 열만 유지합니다
cols_keep =\
    ["season", "rusher_id", "rusher",
     "ryoe_per", "yards_per_carry"]

# 현재 데이터프레임을 생성합니다
ryoe_now_py =\
    ryoe_py[cols_keep].copy()

# 작년의 데이터프레임을 생성합니다
ryoe_last_py =\
    ryoe_py[cols_keep].copy()

# 열 이름을 변경합니다
ryoe_last_py\
    .rename(columns = {'ryoe_per': 'ryoe_per_last',
                       'yards_per_carry': 'yards_per_carry_last'},
                       inplace=True)

# season에 1을 더합니다
ryoe_last_py["season"] += 1

# 함께 병합합니다
ryoe_lag_py =\
    ryoe_now_py\
    .merge(ryoe_last_py,
           how='inner',
           on=['rusher_id', 'rusher',
               'season'])
```

그런 다음 캐리당 야드에 대한 상관관계를 조사해 보세요.

```
## Python
ryoe_lag_py[["yards_per_carry_last", "yards_per_carry"]]\
    .corr()
```

결과는 다음과 같습니다.

yards_per_carry_last yards_per_carry yards_per_carry_last 1.000000 0.347267 yards_per_carry 0.347267 1.000000

RYOE로 반복합니다.

```
## Python
ryoe_lag_py[["ryoe_per_last", "ryoe_per"]]\
    .corr()
```

결과는 다음과 같습니다.

ryoe_per_last ryoe_per ryoe_per_last 1.000000 0.373582 ryoe_per 0.373582 1.000000

이러한 계산은 R을 사용하여 수행할 수도 있습니다.

```
## R
# 현재 데이터프레임을 생성합니다
ryoe_now_r <-
    ryoe_r |>
    select(-n, -ryoe_total)

# 작년의 데이터프레임을 생성하고
# season에 1을 더합니다
ryoe_last_r <-
    ryoe_r |>
    select(-n, -ryoe_total) |>
    mutate(season = season + 1) |>
    rename(ryoe_per_last = ryoe_per,
           yards_per_carry_last = yards_per_carry)

# 함께 병합합니다
ryoe_lag_r <-
    ryoe_now_r |>
    inner_join(ryoe_last_r,
               by = c("rusher_id", "rusher", "season")) |>
    ungroup()
```

그런 다음 두 개의 캐리당 야드 열을 선택하고 상관관계를 살펴보세요.

```
## R
ryoe_lag_r |>
    select(yards_per_carry, yards_per_carry_last) |>
    cor(use = "complete.obs")
```

결과는 다음과 같습니다.

yards_per_carry yards_per_carry_last yards_per_carry 1.000000 0.347267 yards_per_carry_last 0.347267 1.000000

RYOE 열로 반복합니다.

```
## R
ryoe_lag_r |>
    select(ryoe_per, ryoe_per_last) |>
    cor(use = "complete.obs")
```

결과는 다음과 같습니다.

ryoe_per ryoe_per_last ryoe_per 1.0000000 0.3735821 ryoe_per_last 0.3735821 1.0000000

흥미로운 결과네요! 새 모델을 사용하면 RYOE에 대한 연도별 안정성이 약간 향상되는데, 이는 상황에서 더 많은 문맥을 제거한 후 기대치를 뛰어넘는 러닝백의 능력에서 더 많은 신호를 추출해 냈음을 의미합니다. 문제는 여전히 이 수치가 *r* 값 차이 0.03 미만의 미미한 개선이라는 점입니다. 더 나은 데이터(추적 데이터 등)와 더 나은 모델(트리 기반 모델 등)을 사용하여 이 문제를 좀 더 심층적으로 조사하는 추가 작업이 필요합니다.

###### 팁 (Tip)

이 장과 <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>의 코드가 반복적이고 비슷하다는 것을 눈치채셨을 것입니다. 이처럼 정기적으로 코드를 반복해야 한다면 우리만의 일련의 함수를 작성하고 패키지에 넣어 코드를 쉽게 재사용할 수 있도록 할 것입니다. <a href="ch09.html#sec-package" data-type="xref">"패키지 (Packages)"</a>에서 이 주제에 대한 개요를 제공합니다.

# 그렇다면, 러닝백이 중요할까요?

이 질문은 언뜻 보기에 어리석어 보입니다. 당연히 1년에 50번에서 250번이나 공을 나르는 러닝백은 중요합니다. 짐 브라운(Jim Brown), 프랑코 해리스(Franco Harris), 배리 샌더스(Barry Sanders), 마셜 포크(Marshall Faulk), 에이드리언 피터슨(Adrian Peterson), 데릭 헨리(Derrick Henry), 짐 테일러(Jim Taylor) 등 NFL의 역사는 위대한 러너들의 업적을 빼놓고는 쓸 수 없습니다.

###### 팁 (Tip)

벤 볼드윈(Ben Baldwin)은 이 장에서 우리가 선택한 단어를 설명하는 데 도움이 되는 [너드에서 일반인 번역기(Nerd-to-Human Translator)](https://oreil.ly/L2YJ6)를 호스팅하고 있습니다. 그는 "너드(nerd)들이 하는 말: 러닝백은 중요하지 않다."에 대해 "너드들의 진짜 의미: 러닝 플레이의 결과는 주로 누가 공을 나르느냐가 아니라 런 블로킹(run blocking)과 박스(box) 내 수비수들에 의해 결정된다. 러닝백은 서로 대체가 가능하며, (드래프트나 자유계약 시장에서) 많은 자원을 투자하는 것은 합리적이지 않다."라고 적고 있습니다. 그런 다음 그는 뒷받침하는 증거를 제공합니다. 이 증거와 기타 유용한 팁을 보려면 그의 페이지를 확인하세요.

그러나 몇 가지 유의해야 할 사항이 있습니다. 첫째, 지난 수십 년 동안 미식축구에서 패싱은 점점 더 쉬워졌습니다. ["스포츠에서 가장 중요한 일은 그 어느 때보다 쉬워졌다(The Most Important Job in Sports Is Easier Than Ever Before)"](https://oreil.ly/RVXv3)에서 케빈 클라크(Kevin Clark)는 "스키마의 변화, 규칙의 변화, 운동선수의 변화"가 어떻게 쿼터백이 패스를 통해 더 많은 야드를 획득하는 데 도움을 주는지 언급합니다. 케빈의 목록에 기술(예: 선수들이 착용하는 장갑)과 대학 오펜스 시스템의 채택 및 창의적인 풋볼 패스 방법들을 추가할 수도 있습니다.

즉 "패스를 하면 세 가지(완료, 미완료, 가로채기) 상황만 발생할 수 있으며 이 중 두 가지가 나쁜 상황이다"라는 개념은 첫 번째 상황(완료)이 일어나는 증가하는 비율을 고려하지 않은 것입니다. 패싱은 오랫동안 러닝보다 더 효율적이었습니다(<a href="#sec-4-exe" data-type="xref">"연습문제 (Exercises)"</a> 참고). 하지만 이제 패싱의 변동성이 충분히 줄어들어 대부분의 경우 러닝의 낮은 변동성(및 낮은 효율성)보다 선호됩니다.

플레이당 야드 또는 기대 득점 기여도(EPA)와 같은 것으로 측정하더라도 공을 들고 뛰는 것이 공을 패스하는 것보다 덜 효율적이라는 사실 외에도 공을 들고 뛰는 선수가 러닝 플레이의 결과에 미치는 영향력이 이전에 생각했던 것보다 명백히 적습니다.

우리가 `nflfastR` 데이터를 사용하여 설명할 수 없는 다른 요인들도 러닝백이 자신의 생산성에 영향을 미치는 것에 반하는 결과를 낳습니다. 에릭(Eric)은 [PFF에 있는 동안](https://oreil.ly/4i_pH) 특정 플레이에서 오펜시브 라인맨들의 PFF 선수 등급을 사용하여 러싱 플레이에 있어 오펜시브 라인의 플레이가 큰 영향을 미친다는 것을 보여주었습니다.

에릭은 또한 [2021 시즌 동안](https://oreil.ly/tqFn7) 완벽하게 방어된 런(런 블로커가 수비수에게 밀리는 등의 부정적인 플레이를 하지 않은 런닝 플레이)이라는 개념이 러닝 플레이의 결과를 대략 *기대 점수의 절반(half of an expected point)* 정도 바꿨다는 것을 보여주었습니다. 러닝백이 블로커를 세우는 등의 방법으로 완벽하게 방어된 플레이를 만드는 데 도움을 줄 수는 있지만 개인 러닝백의 영향력 크기가 이 수치에 근접할 가능성은 낮습니다.

오랫동안 NFL은 이러한 현상에 대체로 동의해 왔으며, 파이브서티에이트(FiveThirtyEight)의 벤저민 모리스(Benjamin Morris)가 쓴 ["러닝백이 마침내 그들의 가치만큼 연봉을 받고 있다(Running Backs Are Finally Getting Paid What They’re Worth)"](https://oreil.ly/7j9Vw) 기사에서 보여주듯 지난 10년이 넘는 기간 동안 러닝백에 대한 샐러리 캡(salary cap) 비율을 계속해서 줄여 왔습니다. 그럼에도 불구하고 이 포지션에 대해서는 여전히 약간의 논쟁이 있으며, "틀을 깨는(break the mold)" 선수들이 대형 계약을 맺기도 하지만 대부분은 소속 팀에 실망을 안겨주곤 합니다.

스타 러닝백에게 과다하게 지불하고 아마도 후회했을 팀의 예로는 2019년 가을 달라스 카우보이스가 에제키엘 엘리엇과 5천만 달러 보장에 6년간 9천만 달러 계약을 맺은 사례를 들 수 있습니다. 엘리엇은 새로운 계약을 위해 버티고 있었는데, 카우보이스 구단주인 제리 존스(Jerry Jones)는 의심할 여지 없이 NFL 역대 최다 러싱 기록 보유자인 에밋 스미스(Emmitt Smith)가 디펜딩 챔피언 카우보이스의 첫 두 경기 출전을 거부했던 1993년의 상황이 떠올랐을 것입니다. 카우보이스는 스미스 없이 두 경기를 모두 패한 후 스타 선수의 요구에 굴복했고 스미스는 정규 시즌 동안 NFL MVP 상을 수상하며 이에 보답했습니다. 그는 슈퍼볼 MVP까지 거머쥐며 시즌을 마감했고 달라스가 90년대 중반 3번의 챔피언십 우승 중 2번째 우승을 차지하도록 도왔습니다.

엘리엇도 커리어 초반은 스미스와 비슷한 출발을 보였는데 계약에 반발하기 전 처음 3시즌 동안 경기당 평균 러싱 야드에서 리그 선두를 차지했으며 시즌 전체를 소화했던 두 시즌(2017년 시즌은 징계를 받아 일부 경기 결장함) 동안에는 전체 러싱 야드에서도 리그 선두를 차지했습니다. 카우보이스는 2016년과 2018년에 지구 우승을 차지했고 2018년에는 1996년 이후 두 번째로 플레이오프 경기에서 승리했습니다.

분석 커뮤니티의 많은 사람들이 예측했듯이, 지크(Zeke)는 경기당 러싱 야드가 2019년 84.8야드에서 2020년 65.3야드로 떨어졌고 2021년과 2022년에는 각각 58.9야드와 58.4야드로 평탄해지며 계약에 부응하지 못했습니다. 2022년 엘리엇의 캐리당 야드는 4.0 미만인 3.8야드로 떨어졌고 결국 2023년 오프시즌에 자신의 선발 자리와 직업 모두 백업인 토니 폴라드(Tony Pollard)에게 뺏기고 말았습니다.

엘리엇은 자신의 계약 조건을 충족하지 못했을 뿐만 아니라 그의 계약은 카우보이스가 샐러리 캡을 낮추기 위해 리시버 아마리 쿠퍼(Amari Cooper)와 같이 생산성 있는 선수를 방출해야 할 만큼 부담스러웠는데 이는 단순히 러닝백의 잉여 플레이가 낳은 부정적 영향보다 팀을 훨씬 더 약화시키는 도미노 효과였습니다.

2018년에는 팀이 입장을 굽히지 않아 안도의 한숨을 내쉬었을 법한 사례가 발생했습니다. <a href="#sec-ryoe_mr" data-type="xref">"RYOE 분석 (Analyzing RYOE)"</a>에 등장했던 르비온 벨(Le’Veon Bell)은 계약 분쟁으로 인해 트레이닝 캠프와 피츠버그 스틸러스 정규 시즌 출전을 거부하며 *프랜차이즈 태그(franchise tag)*를 적용받아 뛰는 것을 거절했습니다. 프랜차이즈 태그는 팀이 해당 포지션 상위 5명 평균 연봉으로 1시즌 동안 선수를 잔류시키기 위해 사용하는 제도입니다. 보통 장기 계약을 원하는 선수들은 이런 형태의 계약에 묶여 뛰는 것을 꺼리는 경우가 많으며, 스틸러스에서의 첫 4년 동안 러싱 야드 5,000야드 이상을 기록하고 2017년 터치(캐리 및 리셉션) 부문에서 리그 1위를 차지했던 벨도 그런 선수 중 한 명이었습니다.

벨의 문제는 피츠버그에서 쉽게 대체되었다는 것이었습니다. 피츠버그 대학교 출신으로 암을 극복하고 2017년 NFL 드래프트 3라운드에 지명된 제임스 코너(James Conner)는 4.5야드의 평균 러싱(벨의 스틸러스 커리어 평균은 4.3야드)으로 950야드 이상을 달렸고, 13개의 총 터치다운을 기록하며 프로볼(Pro Bowl) 출전권을 따냈습니다.

벨은 다음 해에 하위 팀인 제츠로 떠날 수 있게 되었지만 그곳에서 17경기 동안 캐리당 평균 3.3야드만을 기록하고 4번의 득점만을 올렸습니다. 그는 2020 시즌 도중 방출된 후 캔자스시티 치프스에 합류했습니다. 그 팀은 슈퍼볼에 진출했지만 벨은 그 경기에 뛰지 않았고 치프스는 탬파베이에 31-9로 패했습니다. 2021 시즌이 끝난 후 그는 풋볼계를 떠났습니다.

엘리엇과 벨의 이야기는 아마도 선발 러닝백으로서 가장 극적으로 몰락한 사례일 수 있지만 그들만이 유일한 것은 아닙니다. 그렇기 때문에 연봉 측면에서 잘못된 투자를 하지 않으려면 선수와 팀의 시스템/나머지 로스터 양쪽에 적절한 비율로 생산성의 공을 올바르게 돌리는 것이 중요합니다.

# 선형성 가정 (Assumption of Linearity)

그림 <a href="#fig-py-ry-y100-lm_bin" data-type="xref" data-xrefstyle="select:labelnumber">4-7</a>과 <a href="#fig-r-ry-y100-lm_bin" data-type="xref" data-xrefstyle="select:labelnumber">4-8</a>은 분명히 비선형 관계를 보여줍니다. 기술적으로 선형 회귀는 잔차가 정규 분포를 따르며 관측된 관계가 선형적이라고 가정합니다. 그러나 이 둘은 대개 함께 진행됩니다.

기본(Base) R에는 선형 모델의 결과를 살펴보기 위한 유용한 진단 도구가 포함되어 있습니다. 이 도구는 기본 R의 `plot()` 함수를 사용합니다(이는 우리가 `plot()`을 선호하는 몇 안 되는 경우 중 하나입니다. 이 함수는 다른 사람들과 공유하기 위해서가 아니라 오직 내부 진단용으로만 사용하는 간단한 함수를 만듭니다). 먼저 `par(mfrow=c(2,2))`를 사용하여 4개의 하위 플롯(subplot)을 만듭니다. 그런 다음 이전에 `expected_yards_r`로 저장하고 맞춘 다중 회귀 모델에 `plot()`을 사용하여 <a href="#fig-r-lm-diag_1" data-type="xref">그림 4-13</a>을 생성합니다.

```
## R
par(mfrow = c(2, 2))
plot(expected_yards_r)
```

<a href="#fig-r-lm-diag_1" data-type="xref">그림 4-13</a>에는 4개의 하위 플롯이 포함되어 있습니다. 왼쪽 상단은 예측값 대 적합값(또는 잔차)의 차이와 비교된 모델의 추정값(또는 적합값)을 보여줍니다. 오른쪽 상단은 이론적 값에 대한 모델 적합값의 누적 분포를 보여줍니다. 왼쪽 하단은 표준화 잔차의 제곱근 대 적합값을 보여줍니다. 오른쪽 하단은 표준화 잔차 대비 매개변수가 모델 적합도에 미치는 영향(influence)을 나타냅니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0413.png" />
<h6 id="figure-4-13.-four-diagnostic-subplots-for-a-regression-model">그림 4-13. 회귀 모델에 대한 4가지 진단 하위 플롯</h6>
</figure>

"잔차 대 적합값(Residuals vs. Fitted)" 하위 플롯은 그다지 *병적으로* 보이지 않습니다. 이 하위 플롯은 단순히 데이터 포인트의 상당수가 모델에 잘 맞지 않으며 데이터에 비대칭(skew)이 존재함을 보여줍니다. "정규 Q-Q(Normal Q-Q)" 하위 플롯은 많은 데이터 포인트가 예상 모델에서 벗어나기 시작한다는 것을 보여줍니다. 따라서 일부 경우에 모델이 데이터에 잘 맞지 않습니다. "척도-위치(Scale-Location)" 하위 플롯은 "잔차 대 적합값"과 비슷한 패턴을 보여줍니다. 또한 이 플롯은 데이터의 정수(integer) 특성(예: 0, 1, 2, 3)으로 인해 0 근처에서 <i>W</i>자 모양의 선이 나타나는 독특한 패턴을 가지고 있습니다. 마지막으로 "잔차 대 지렛대(Residuals vs. Leverage)"는 일부 데이터 관측치가 모델 추정치에 상당한 '지렛대(leverage)' 효과를 미치고 있지만, 이들이 쿡의 거리(Cook’s distance)를 기반으로 볼 때 예상 범위 내에 있음을 보여줍니다. 비공식적으로 *쿡의 거리*는 모델 적합성에 미치는 관측치의 예상 영향력을 뜻합니다. 기본적으로 값이 클수록 관찰 결과가 모델의 추정치에 더 큰 영향을 미친다는 것을 의미합니다.

15야드 미만 또는 90야드 초과의 플레이를 제거하여 <a href="#fig-r-lm-diag_2" data-type="xref">그림 4-14</a>를 만들면 어떻게 되는지 살펴보세요.

```
## R
expected_yards_filter_r <-
    pbp_r_run |>
    filter(rushing_yards > 15 & rushing_yards < 90) |>
    lm(formula = rushing_yards ~ 1 + down + ydstogo + down:ydstogo +
                 yardline_100 + run_location + score_differential)

par(mfrow = c(2, 2))
plot(expected_yards_filter_r)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0414.png" />
<h6 id="figure-4-14.-this-figure-is-a-re-creation-of-figure-4-13-with-the-model-including-only-rushing-plays-with-more-than-15-yards-and-less-than-95-yards">그림 4-14. 이 그림은 15야드를 초과하고 95야드 미만인 러싱 플레이만을 포함하는 모델을 사용하여 <a href="#fig-r-lm-diag_1" data-type="xref">그림 4-13</a>을 다시 만든 것입니다.</h6>
</figure>

<a href="#fig-r-lm-diag_2" data-type="xref">그림 4-14</a>는 새로운 선형 모델인 `expected_yards_filter_r`이 더 잘 맞음을 보여줍니다. "잔차 대 적합값" 하위 플롯에는 일그러진 직선이 있지만(데이터가 필터링되었음을 나타냄), 다른 하위 플롯은 더 잘 보입니다. 가장 많이 개선된 하위 플롯은 "정규 Q-Q"입니다. <a href="#fig-r-lm-diag_1" data-type="xref">그림 4-13</a>의 범위는 -5에서 15 사이인 반면, 이 플롯의 범위는 현재 -1에서 5 사이입니다.

마지막 점검으로 모델의 요약을 살펴보고 향상된 모델 적합성을 확인하세요. $`R^{2}`$ 값은 약 0.01에서 0.05로 향상되었습니다.

```
## R
summary(expected_yards_filter_r)
```

결과는 다음과 같습니다.

Call: lm(formula = rushing_yards ~ 1 + down + ydstogo + down:ydstogo + yardline_100 + run_location + score_differential, data = filter(pbp_r_run, rushing_yards \> 15 & rushing_yards \< 95)) Residuals: Min 1Q Median 3Q Max -17.158 -7.795 -3.766 3.111 63.471 Coefficients: Estimate Std. Error t value Pr(\>\|t\|) (Intercept) 21.950963 2.157834 10.173 \<2e-16 \*\*\* down2 -2.853904 2.214676 -1.289 0.1976 down3 -0.696781 2.248905 -0.310 0.7567 down4 0.418564 3.195993 0.131 0.8958 ydstogo -0.420525 0.204504 -2.056 0.0398 \* yardline_100 0.130255 0.009975 13.058 \<2e-16 \*\*\* run_locationmiddle 0.680770 0.562407 1.210 0.2262 run_locationright 0.635015 0.443208 1.433 0.1520 score_differential 0.048017 0.019098 2.514 0.0120 \* down2:ydstogo 0.207071 0.224956 0.920 0.3574 down3:ydstogo 0.165576 0.234271 0.707 0.4798 down4:ydstogo 0.860361 0.602634 1.428 0.1535 --- Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1 Residual standard error: 12.32 on 3781 degrees of freedom Multiple R-squared: 0.05074, Adjusted R-squared: 0.04798 F-statistic: 18.37 on 11 and 3781 DF, p-value: \< 2.2e-16

요약하면 모델의 잔차를 확인함으로써 이 모델이 15야드보다 짧거나 95야드보다 긴 플레이에서는 성능이 떨어진다는 것을 알 수 있었습니다. 이러한 한계를 파악하고 계량화하는 것은 최소한 자신의 모델이 파악하지 못하는 부분을 이해하는 데 도움을 줍니다.

# 이 장에서 사용된 데이터 과학 도구 (Data Science Tools Used in This Chapter)

이 장에서는 다음의 주제들을 다루었습니다.

- `OLS()`를 사용한 Python 또는 `lm()`을 사용한 R에서의 다중 회귀 모델 적합시키기

- 다중 회귀에서 나오는 계수 이해하고 읽기

- 이전 장들에서 배운 데이터 처리(data-wrangling) 도구들 재적용하기

- 회귀 분석의 적합성(regression’s fit) 평가하기

# 연습문제 (Exercises)

1. 캐리 기준을 50회 캐리에서 100회 캐리로 변경해 보세요. 여전히 이 장에서 확인한 안정성 차이를 확인할 수 있나요?

2. 전체 `nflfastR` 데이터를 사용하여 캐리당 야드 및 캐리당 EPA를 바탕으로 러싱이 패싱보다 효율성이 낮음을 보여주세요. 또한 이 두 플레이 유형의 변동성도 조사하세요.

3. 일부 상황(예: 상대 팀의 엔드존 근처)에서는 러싱이 패싱보다 가치가 더 클까요?

4. 제임스 코너(James Conner) 커리어의 RYOE 값을 벨(Bell)의 것과 비교하여 조사하세요. 두 러닝백의 메트릭에서 어떤 점을 발견했나요?

5. 이 장에서 러셔를 대상으로 거쳤던 과정을 패싱 게임의 리시버를 대상으로 반복하세요. 이렇게 하려면 `play_type == "pass"` 그리고 `receiver_id`가 `NA` 또는 `NULL`이 아닌 데이터만 필터링해야 합니다. 특징(features)을 찾아내는 것은 어렵겠지만, 이 장의 절차를 지침으로 참고하세요. 예를 들어, `down`과 `distance`를 사용하고, 모델에 `air_yards` 같은 것도 추가하여 예상치를 도출해 보세요.

# 추천 도서 (Suggested Readings)

<a href="ch03.html#sec-chp3-fr" data-type="xref">"추천 도서 (Suggested Readings)"</a> 목록에 있는 책들은 이 장에도 적용됩니다. 이 목록을 바탕으로 도움이 될 만한 다른 자료들을 추가로 소개합니다.

- 제인 E. 밀러(Jane E. Miller)의 *The Chicago Guide to Writing about Numbers*, 2판 (Chicago Press, 2015)은 다양한 형태의 글쓰기에서 수치를 기술하는 훌륭한 예시를 제공합니다.

- 제인 E. 밀러(Jane E. Miller)의 *The Chicago Guide to Writing about Multivariate Analysis*, 2판 (University of Chicago Press, 2013)은 다중 회귀(multiple regression)를 설명하는 수많은 예시를 제공합니다. 저자가 *다변량 회귀(multivariate regression)*를 *다중 회귀*와 동의어로 사용하는 데는 동의하지 않지만 이 책은 회귀 분석 결과를 설명하는 훌륭한 사례들을 제공합니다.

- 마이크 X 코헨(Mike X Cohen)의 <a href="https://learning.oreilly.com/library/view/practical-linear-algebra/9781098120603/" class="orm:hideurl"><em>Practical Linear Algebra for Data Science</em></a> (O’Reilly Media, 2022)는 회귀 분석을 더 잘 이해하는 데 도움이 되는 선형대수학 지식을 제공합니다. 선형대수학은 다중 회귀 분석을 포함한 거의 모든 통계적 방법의 기초를 형성합니다.

- [FiveThirtyEight](https://fivethirtyeight.com)에는 데이터 저널리즘이 상당히 많이 포함되어 있으며, 네이트 실버(Nate Silver)가 설립하여 2023년 ABC/Disney 소유의 해당 사이트에서 떠날 때까지 운영했습니다. 게시물들을 살펴보면서 사이트에서 회귀 모델이 어떻게 활용되는지 확인해 보세요.

- 앤드루 겔먼(Andrew Gelman)이 만들고 여러 작가가 참여한 [*Statistical Modeling, Causal Inference, and Social Science*](https://statmodeling.stat.columbia.edu)는 회귀 모델링에 대해 자주 다루는 블로그입니다. 겔먼은 네이트 실버의 정치학자 버전이라 할 수 있으며, 학구적인 성향이 짙습니다.

- 프랭크 해럴(Frank Harrell)의 [*Statistical Thinking*](https://www.fharrell.com)은 회귀 분석에 대해 자주 다루는 또 다른 블로그입니다. 해럴은 네이트 실버의 통계학자 버전이라 할 수 있으며 통계에 더 초점을 둡니다. 그러나 그의 많은 게시물들은 어떠한 유형의 회귀 분석을 하는 사람들에게든 종종 관련이 있습니다.


