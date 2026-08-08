# 5장. 일반화 선형 모델: 기대 대비 패스 성공률 (Chapter 5. Generalized Linear Models: Completion Percentage over Expected)

<a href="ch03.html#sec-lm-ryoa" data-type="xref" data-xrefstyle="select:labelnumber">3장</a>과 <a href="ch04.html#sec-mr-ryoe2" data-type="xref" data-xrefstyle="select:labelnumber">4장</a>에서는 단순 및 다중 회귀를 사용하여 플레이의 *상황(context)*에 맞게 플레이별 데이터를 조정했습니다. 공을 운반하는 선수(ball carrier)의 경우, 플레이 수준과 이후에는 시즌 수준에서 개별 선수의 통계를 조정하기 위해 상황(다운, 남은 거리, 목표까지 남은 거리 등)을 보정했습니다. 이 접근법은 명백하게 패싱 게임, 특히 쿼터백에게 적용될 수 있습니다. <a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에서 논의했듯이 미네소타의 쿼터백 샘 브래드퍼드(Sam Bradford)는 2016년에 놀라운 수치인 71.6%의 패스를 성공시키며 시즌 패스 성공률에 대한 NFL 신기록을 세웠습니다.

그러나 패스 시도당 야드 수, 패스 시도당 기대 점수 또는 터치다운 패스 등으로 측정되는 효율성 면에서 브래드퍼드는 중간 정도 수준의 쿼터백에 불과했습니다. 바이킹스(Vikings)는 그해 그가 선발 출장한 15경기 중 7경기만 승리했습니다. 브래드퍼드의 패스 성공률이 그렇게 높았던 이유는 타깃까지의 평균 거리가 단 6.6야드(PFF 기준 NFL 37위)에 불과했기 때문입니다. 일반적으로 먼 거리로 던져지는 패스는 성공률이 더 낮습니다.

이를 확인하기 위해 Python에서 <a href="#fig-py-ay_prc" data-type="xref">그림 5-1</a>을 만들거나 R에서 <a href="#fig-r-ay_prc" data-type="xref">그림 5-2</a>를 만들 것입니다. 먼저 데이터를 로드합니다. 그런 다음 패서(Python의 경우 `passer_id.notnull()`, R의 경우 `!is.na(passer_id)`)와 패스 깊이(Python의 경우 `air_yards.notnull()`, R의 경우 `!is.na(air_yards)`)가 있는 패스 플레이(`play_type == "pass"`)를 필터링합니다. Python에서는 이 코드를 사용합니다.

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

seasons = range(2016, 2022 + 1)
pbp_py = nfl.import_pbp_data(seasons)

pbp_py_pass = \
  pbp_py\
  .query('play_type == "pass" & passer_id.notnull() &' +
         'air_yards.notnull()')\
  .reset_index()
```

또는 R에서는 이 코드를 사용합니다.

```
## R
library(tidyverse)
library(nflfastR)
library(broom)

pbp_r <- load_pbp(2016:2022)
pbp_r_pass <-
  pbp_r |>
  filter(play_type == "pass" & !is.na(passer_id) &
         !is.na(air_yards))
```

다음으로 충분히 큰 표본 크기를 확보하기 위해 에어 야드(air yards)를 `0` 야드 초과 `20` 야드 이하로 제한합니다. 데이터를 요약하여 패스 성공률인 `comp_pct`를 계산합니다. 그런 다음 결과를 플롯하여 <a href="#fig-py-ay_prc" data-type="xref">그림 5-1</a>을 만듭니다.

```
## Python
# 해당 장의 테마를 변경합니다
sns.set_theme(style="whitegrid", palette="colorblind")

# 포맷팅 후 플롯을 생성합니다
pass_pct_py = \
  pbp_py_pass\
  .query('0 < air_yards <= 20')\
  .groupby('air_yards')\
  .agg({"complete_pass": ["mean"]})

pass_pct_py.columns = \
  list(map('_'.join, pass_pct_py.columns))

pass_pct_py\
  .reset_index(inplace=True)
pass_pct_py\
  .rename(columns={'complete_pass_mean': 'comp_pct'},
                   inplace=True)

sns.regplot(data=pass_pct_py, x='air_yards', y='comp_pct',
            line_kws={'color': 'red'});
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0501.png" />
<h6 id="figure-5-1.-scatterplot-with-linear-trendline-for-air-yards-and-completion-percentage-plotted-with-seaborn">그림 5-1. 에어 야드 및 패스 성공률에 대한 선형 추세선이 포함된 산점도, <code>seaborn</code>으로 플롯함</h6>
</figure>

또는 R을 사용하여 <a href="#fig-r-ay_prc" data-type="xref">그림 5-2</a>를 만듭니다.

```
## R
pass_pct_r <-
  pbp_r_pass |>
  filter(0 < air_yards & air_yards <= 20) |>
  group_by(air_yards) |>
  summarize(comp_pct = mean(complete_pass),
            .groups = 'drop')

pass_pct_r |>
  ggplot(aes(x = air_yards, y=comp_pct)) +
  geom_point() +
  stat_smooth(method='lm') +
  theme_bw() +
  ylab("Percent completion") +
  xlab("Air yards")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0502.png" />
<h6 id="figure-5-2.-scatterplot-with-linear-trendline-for-air-yards-and-completion-percentage-plotted-with-ggplot2">그림 5-2. 에어 야드 및 패스 성공률에 대한 선형 추세선이 포함된 산점도, <code>ggplot2</code>로 플롯함</h6>
</figure>

그림 <a href="#fig-py-ay_prc" data-type="xref" data-xrefstyle="select:labelnumber">5-1</a>과 <a href="#fig-r-ay_prc" data-type="xref" data-xrefstyle="select:labelnumber">5-2</a>는 예상대로 추세를 명확하게 보여줍니다. 따라서 패스 성공률로 측정되는 쿼터백의 정확도에 대한 논의에는 플레이 스타일에 대한 약간의 조정(adjustment)이 수반되어야 합니다.

풋볼 분석 세계에서 *CPOE*로 불리는 *기대 대비 패스 성공률(Completion percentage over expected)*은 (1장에서 소개됨) 주류로 자리 잡은 조정된 메트릭 중 하나입니다. 개방된(open) 풋볼 데이터에 대한 훌륭한 참조 자료인 벤 볼드윈(Ben Baldwin)의 [웹사이트](https://rbsdm.com)에서는 CPOE를 주요 메트릭 중 하나로 표시하고 있는데, 이는 CPOE가 패스 플레이당 EPA와 함께 쿼터백 플레이를 해마다 예측하는 데 있어 가장 예측력이 뛰어난 공개 메트릭으로 입증되었기 때문입니다. [NFL의 Next Generation Stats (NGS) 그룹](https://nextgenstats.nfl.com)은 웹사이트 전면에 표시되는 자체적인 CPOE 버전을 가지고 있으며, 여기에는 가장 가까운 수비수로부터의 리시버 분리와 같은 추적(tracking) 데이터 엔지니어링 특징이 포함되어 있습니다. ESPN은 방송에서 지속적으로 이 메트릭을 사용합니다.

이러한 방식으로 쿼터백의 성과를 측정하는 데는 몇 가지 문제가 있으며, 장의 끝에서 다루겠지만 CPOE는 계속 사용될 것입니다. 우리는 일반화 선형 모델(generalized linear models)을 사용하여 이 메트릭의 개발 과정을 안내하는 것부터 시작하겠습니다.

# 일반화 선형 모델 (Generalized Linear Models)

<a href="ch03.html#sec-lm-ryoa" data-type="xref">3장</a>에서는 단순 선형 회귀의 몇 가지 주요 가정을 정의하고 설명했습니다. 이러한 가정은 다음과 같습니다.

- 예측 변수(predictor)는 단일 종속 변수 즉, 특징(feature)과 선형적으로 관련되어 있습니다.

- 하나의 예측 변수(단순 선형 회귀) 또는 두 개 이상의 예측 변수(다중 회귀)가 종속 변수를 설명합니다.

다중 회귀의 또 다른 주요 가정은 잔차 분포가 정규 분포 또는 종 모양 곡선(bell-curve) 분포를 따른다는 것입니다. 거의 모든 데이터 세트가 이 마지막 가정을 위반하지만 대개 이 가정은 "충분히 잘" 작동합니다. 하지만 어떤 데이터 구조는 다중 회귀 분석을 실패하게 하거나 무의미한 결과를 산출하게 만듭니다. 예를 들어, 패스는 불완전(`pass_complete = 0`)하거나 성공(`pass_complete = 1`)하기 때문에 패스 성공률은 0과 1 사이로 제한된(bounded) 값입니다(*제한된(bounded)*이라는 것은 값이 0보다 작거나 1보다 클 수 없음을 의미합니다). 따라서 반응 변수에 경계가 없다고 가정하는 선형 회귀는 부적절한 경우가 많습니다.

마찬가지로 다른 데이터도 흔히 이 가정을 위반합니다. 예를 들어, 카운트(count) 데이터(예: 경기당 색(sack))는 0이 너무 많아서 정규 분포를 이룰 수 없으며 음수도 될 수 없습니다. 마찬가지로, 두 가지 결과(예: 승리/패배 또는 패스 실패/성공)가 있는 이진 데이터 및 이산 결과(discrete outcomes, 예: 패스 위치가 우측, 좌측 또는 중앙)도 반응 데이터로서 다중 회귀 분석과 함께 사용되지 않습니다.

이러한 유형의 결과를 모델링하기 위해 존재하는 회귀 모델의 범주가 *일반화 선형 모델(generalized linear models)*(*GLM*)입니다. GLM은 *선형 모델*을 *일반화(generalize)*하거나 확장하여 정규 분포가 아닌 분포에서 가져온 것으로 가정되는 반응 변수(이진 응답 또는 카운트 등)를 허용합니다. 특정 유형의 반응 분포를 *패밀리(family)*라고 부릅니다. 특수한 유형의 GLM 중 하나가 이진 데이터를 모델링하는 데 사용될 수 있으며 이 장에서 다룹니다. <a href="ch06.html#sec-pos-td" data-type="xref">6장</a>에서는 카운트 데이터와 함께 푸아송 회귀(Poisson regression)라는 다른 유형의 GLM을 사용하는 방법을 다룹니다.

GLM을 통해 다른 유형의 데이터를 분석할 수 있습니다. 예를 들어, 순서형 회귀(ordinal regression) (순서 분류(ordinal classification)라고도 함)를 사용하여 이산 결과를 분석할 수 있지만 이 책의 범위를 벗어납니다. 마지막으로 선형 모델(*선형 회귀(linear regression)* 및 *일반 최소 제곱(ordinary least squares)*이라고도 함)은 GLM의 특별한 유형으로, 구체적으로 정규 또는 가우스(Gaussian) 패밀리를 갖는 GLM입니다.

GLM이 작동하는 방식 이면의 기본 이론을 이해하기 위해, 1(성공) 또는 0(실패)이 될 수 있는 완료된 패스를 살펴보겠습니다. 두 가지 결과가 가능하기 때문에 이는 *이진(binary)* 응답이며, *이항(binomial)* 분포가 데이터를 설명하는 데 "충분히 좋은" 역할을 한다고 가정할 수 있습니다. 정규 분포는 두 가지 매개변수를 가정하는데, 하나는 종 모양 곡선의 중심(평균)이고 두 번째는 종 모양 곡선의 너비(표준 편차)를 나타냅니다. 반면 이항 분포는 하나의 매개변수인 성공 확률만 필요로 합니다. 패스의 예를 들면, 패스를 성공시킬 확률이 될 것입니다. 그러나 확률을 통계적으로 모델링하는 것은 0과 1에 의해 제한되기 때문에 어렵습니다. 따라서 *연결 함수(link function)*가 확률(0과 1 사이의 값)을 $`- \infty`$에서 $`\infty`$ 범위의 값으로 변환(또는 *연결*)합니다. 가장 일반적인 연결 함수는 *로짓(logit)*으로, GLM의 가장 일반적인 유형 중 하나인 *로지스틱 회귀(logistic regression)*에 이름을 부여합니다.

# GLM 구축하기 (Building a GLM)

GLM, 특히 로지스틱 회귀를 적용하기 위해 간단한 예부터 살펴보겠습니다. 완료된 패스를 예측하기 위한 하나의 특징으로 `air_yards`를 조사해 봅시다. 그림 <a href="#fig-py-ay_prc" data-type="xref" data-xrefstyle="select:labelnumber">5-1</a>과 <a href="#fig-r-ay_prc" data-type="xref" data-xrefstyle="select:labelnumber">5-2</a>에서 시사하듯, 거리가 긴 패스일수록 성공할 확률이 낮아집니다. 이제 모델을 사용하여 이러한 관계를 계량화할 것입니다.

Python의 경우 GLM을 맞추기 위해 플레이별(play-by-play) 데이터와 함께 `statsmodels.formula.api`(가명 `smf`로 임포트됨)의 `glm()` 함수 및 `statsmodels.api`(가명 `sm`으로 임포트됨)의 `binomial` 패밀리를 사용한 후 모델 적합성 요약(model fit’s summary)을 살펴보세요.

```
## Python
complete_ay_py  = \
  smf.glm(formula='complete_pass ~ air_yards',
          data=pbp_py_pass,
          family=sm.families.Binomial())\
          .fit();

complete_ay_py.summary()
```

결과는 다음과 같습니다.

\<class 'statsmodels.iolib.summary.Summary'\> """ Generalized Linear Model Regression Results ============================================================================== Dep. Variable: complete_pass No. Observations: 131606 Model: GLM Df Residuals: 131604 Model Family: Binomial Df Model: 1 Link Function: Logit Scale: 1.0000 Method: IRLS Log-Likelihood: -81073. Date: Sun, 04 Jun 2023 Deviance: 1.6215e+05 Time: 09:37:33 Pearson chi2: 1.32e+05 No. Iterations: 5 Pseudo R-squ. (CS): 0.07013 Covariance Type: nonrobust ============================================================================== coef std err z P\>\|z\| \[0.025 0.975\] ------------------------------------------------------------------------------ Intercept 1.0720 0.008 133.306 0.000 1.056 1.088 air_yards -0.0573 0.001 -91.806 0.000 -0.059 -0.056 ============================================================================== """

마찬가지로 R에서는 핵심 R 패키지에 포함된 `glm()` 함수를 사용하고 `binomial` 패밀리를 포함시킨 다음 요약을 살펴봅니다.

```
## R
complete_ay_r <-
  glm(complete_pass ~ air_yards,
      data = pbp_r_pass,
      family = "binomial")

summary(complete_ay_r)
```

결과는 다음과 같습니다.

Call: glm(formula = complete_pass ~ air_yards, family = "binomial", data = pbp_r_pass) Coefficients: Estimate Std. Error z value Pr(\>\|z\|) (Intercept) 1.0719692 0.0080414 133.31 \<2e-16 \*\*\* air_yards -0.0573223 0.0006244 -91.81 \<2e-16 \*\*\* --- Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1 (Dispersion parameter for binomial family taken to be 1) Null deviance: 171714 on 131605 degrees of freedom Residual deviance: 162145 on 131604 degrees of freedom AIC: 162149 Number of Fisher Scoring iterations: 4

###### 팁 (Tip)

Python에서 `OLS` 결과물을 다루거나 R에서 `lm` 결과물을 다루기 위해 존재하는 `summary()`와 같은 많은 도구와 함수는 `glm` 결과물에서도 작동합니다.

Python과 R의 출력이 모두 <a href="ch03.html#sec-lm-ryoa" data-type="xref" data-xrefstyle="select:labelnumber">3장</a> 및 <a href="ch04.html#sec-mr-ryoe2" data-type="xref" data-xrefstyle="select:labelnumber">4장</a>의 출력과 비슷하다는 것을 눈치챘을 것입니다. 이 두 모델 모두 `air_yards`가 증가할수록 성공 확률은 감소합니다. 계수가 통계적으로 중요한지 확인하기 위해 계수가 0과 다른지 여부에 주의를 기울여야 합니다.

###### 경고 (Warning)

그림 <a href="#fig-py-pass-comp" data-type="xref" data-xrefstyle="select:labelnumber">5-3</a> 및 <a href="#fig-r-pass-comp" data-type="xref" data-xrefstyle="select:labelnumber">5-4</a>와 같은 이 책의 일부 플롯은 완료하는 데 시간이 꽤(몇 분 이상) 걸릴 수 있습니다. 데이터로 작업할 때 정기적인 플롯 생성 시간 때문에 작업이 지연된다고 느껴진다면 원시 데이터 대신 데이터 요약을 표시하는 것을 고려해 보세요. 예를 들어, <a href="ch03.html#sec-eda-bin" data-type="xref">"탐색적 데이터 분석 (Exploratory Data Analysis)"</a>에서 사용한 구간화(binning)가 그 중 한 방법입니다. 이 책에서 다루지 않은 다른 도구로는 `hexbin` [R 패키지](https://oreil.ly/_QQDZ) 또는 [`matplotlib`의 `hexbin()` 플롯 함수](https://oreil.ly/CdSQF)에서 만든 것과 같은 *hexbin* 플롯이 있습니다.

로지스틱 회귀 결과를 볼 수 있도록 돕기 위해 Python과 R은 모두 플롯 생성 도구를 제공합니다. Python의 경우 `seaborn`의 `regplot()`을 사용하되 `logistic` 옵션을 `True`로 설정하여 <a href="#fig-py-pass-comp" data-type="xref">그림 5-3</a>을 생성하세요 (이 모델에 선형 회귀 분석을 적용하는 것이 좋지 않은 아이디어인 이유를 확인하려면 기본 옵션인 `False`를 사용하여 선이 데이터의 위아래로 어떻게 나타나는지 확인해 보세요).

```
## Python
sns.regplot(data=pbp_py_pass, x='air_yards', y='complete_pass',
            logistic=True,
            line_kws={'color': 'red'},
            scatter_kws={'alpha':0.05});
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0503.png" />
<h6 id="figure-5-3.-pass-completion-as-a-function-of-air-yards-plotted-with-a-logistic-curve-in-seaborn">그림 5-3. <code>seaborn</code>에서 로지스틱 곡선으로 표시한 에어 야드에 따른 패스 성공률 함수</h6>
</figure>

이 플롯에서 구부러진 곡선이 로지스틱 함수입니다. 반투명한 점들은 패스 성공에 대한 이진 결과를 보여줍니다. 겹치는 점이 많으므로 데이터에서 어떠한 추세를 파악하려면 로지스틱 곡선이 필요합니다.

마찬가지로 겹치는 지점을 쉽게 볼 수 있도록 y축에 지터링(jittering)을 주고 R의 `ggplot2`를 사용하여 비슷한 플롯을 만들 수 있습니다(<a href="#fig-r-pass-comp" data-type="xref">그림 5-4</a> 참고).

```
## R
ggplot(data=pbp_r_pass,
       aes(x=air_yards, y=complete_pass)) +
  geom_jitter(height = 0.05, width = 0,
              alpha = 0.05) +
  stat_smooth(method = 'glm',
              method.args=list(family="binomial")) +
  theme_bw() +
  ylab("Completed pass (1 = yes, 0 = no)") +
  xlab("air yards")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0504.png" />
<h6 id="figure-5-4.-pass-completion-as-a-function-of-air-yards-plotted-with-a-logistic-curve-in-ggplot2">그림 5-4. <code>ggplot2</code>에서 로지스틱 곡선으로 표시한 에어 야드에 따른 패스 성공률 함수</h6>
</figure>

이 플롯에서 구부러진 곡선이 로지스틱 함수입니다. 반투명한 점들은 패스 성공에 대한 이진 결과를 보여줍니다. 점이 겹치지 않도록 점들을 y축을 따라 지터링(jittered) 처리합니다. 겹치는 점이 많으므로 데이터에서 어떠한 추세를 파악하려면 로지스틱 곡선이 필요합니다.

# 성공률 분석에 GLM 적용하기 (GLM Application to Completion Percentage)

<a href="#sec-build-glm" data-type="xref">"GLM 구축하기 (Building a GLM)"</a>의 결과를 사용하여 플레이별(play-by-play) 패스 데이터프레임에 잔차를 추가함으로써 기대 성공률을 추출하세요. 선형 모델(또는 선형 회귀)의 경우 잔차는 한 가지 유형만 존재하기 때문에 CPOE는 단순히 잔차가 됩니다. 하지만 GLM에는 다양한 유형의 잔차가 존재하므로, (적합성에서 추출하는 대신) 수동으로 잔차를 계산하여 현재 사용 중인 잔차 유형이 무엇인지 확실히 파악해야 합니다.

Python에서는 `predict()`를 사용하여 이전에 맞춘 모델에서 예측값을 추출한 다음, 관측값에서 이 예측값을 빼서 CPOE를 구하는 방식으로 이 작업을 수행합니다.

```
## Python
pbp_py_pass["exp_completion"] = \
  complete_ay_py.predict()

pbp_py_pass["cpoe"] = \
  pbp_py_pass["complete_pass"] - \
  pbp_py_pass["exp_completion"]
```

###### 경고 (Warning)

GLM 모델은 관측 데이터와 다른 척도(scale)에서 나타나기 때문에 잔차와 예측값을 계산하기 위한 다양한 방법이 존재합니다. 예를 들어 R의 `predict()` 함수 도움말 파일에는 여러 유형의 예측이 존재한다고 명시되어 있습니다. 기본값은 선형 예측 변수 척도이고 대안 응답은 반응 변수 척도에 있습니다. 따라서 기본 이항 모델의 경우 기본 예측은 로그 오즈(log-odds, 로짓 척도의 확률)가 되고 `type = "response"`는 예상 확률을 제공합니다.

R의 경우, `pbp_r_pass` 데이터프레임을 가져온 다음 `mutate()`를 사용하여 새 열을 만듭니다. 새 열의 이름은 `exp_completion`이며, 이전에 맞춘 모델에서 `type = "resp"`와 함께 `predict()` 함수를 통해 예측된 모델 적합성을 추출하여 값을 가져옵니다. 그런 다음 이 값을 `complete_pass`에서 빼서 CPOE를 계산하세요.

```
## R
pbp_r_pass <-
  pbp_r_pass |>
  mutate(exp_completion = predict(complete_ay_r, type = "resp"),
         cpoe = complete_pass - exp_completion)
```

###### 팁 (Tip)

이 장의 코드는 <a href="ch03.html#sec-lm-ryoa" data-type="xref" data-xrefstyle="select:labelnumber">3장</a> 및 <a href="ch04.html#sec-mr-ryoe2" data-type="xref" data-xrefstyle="select:labelnumber">4장</a>의 코드와 비슷합니다. 자세한 내용이 필요하면 해당 장을 참고하세요. 그러나 GLM은 그 출력 단위(output units)와 구조 면에서 선형 모델과 다르다는 점에 유의하세요.

먼저, 2016년 이후 CPOE 1위 선수와 실제 패스 성공률 1위 선수를 살펴보세요. `air_yards` 수치가 `NA`가 아닌 패스만 살펴보고 있다는 점을 상기하세요. 또한 100회 이상의 패스 시도를 한 쿼터백만 포함시킵니다. `NA` 데이터를 걸러내면 무의미한 플레이를 제거하는 데 도움이 됩니다. 시도 횟수가 100번 이상인 쿼터백만 포함되도록 필터링하면 플레이 횟수가 적어서 이상치(outlier)가 될 가능성이 높은 쿼터백을 피할 수 있습니다.

Python에서는 평균 CPOE와 평균 패스 성공률을 계산한 다음 `compl`로 정렬합니다.

```
## Python
cpoe_py = \
  pbp_py_pass\
  .groupby(["season", "passer_id", "passer"])\
  .agg({"cpoe": ["count", "mean"],
        "complete_pass": ["mean"]})

cpoe_py.columns = \
  list(map('_'.join, cpoe_py.columns))
cpoe_py.reset_index(inplace=True)

cpoe_py = \
  cpoe_py\
   .rename(columns = {"cpoe_count": "n",
                      "cpoe_mean": "cpoe",
                      "complete_pass_mean": "compl"})\
    .query("n > 100")

print(
  cpoe_py\
    .sort_values("cpoe", ascending=False)
  )
```

결과는 다음과 같습니다.

season passer_id passer n cpoe compl 299 2019 00-0020531 D.Brees 406 0.094099 0.756158 193 2018 00-0020531 D.Brees 566 0.086476 0.738516 467 2020 00-0033537 D.Watson 542 0.073453 0.704797 465 2020 00-0033357 T.Hill 121 0.072505 0.727273 22 2016 00-0026143 M.Ryan 631 0.068933 0.702060 .. ... ... ... ... ... ... 91 2016 00-0033106 J.Goff 204 -0.108739 0.549020 526 2021 00-0027939 C.Newton 126 -0.109908 0.547619 112 2017 00-0025430 D.Stanton 159 -0.110229 0.496855 730 2022 00-0037327 S.Thompson 150 -0.116812 0.520000 163 2017 00-0031568 B.Petty 112 -0.151855 0.491071 \[300 rows x 6 columns\]

R에서는 평균 CPOE와 평균 패스 성공률(`compl`)을 계산한 다음, `compl`을 기준으로 정렬합니다.

```
## R
pbp_r_pass |>
  group_by(season, passer_id, passer) |>
  summarize(n = n(),
            cpoe = mean(cpoe, na.rm = TRUE),
            compl = mean(complete_pass, na.rm = TRUE),
            .groups = "drop") |>
  filter(n >= 100) |>
  arrange(-cpoe) |>
  print(n = 20)
```

결과는 다음과 같습니다.

\# A tibble: 300 × 6 season passer_id passer n cpoe compl \<dbl\> \<chr\> \<chr\> \<int\> \<dbl\> \<dbl\> 1 2019 00-0020531 D.Brees 406 0.0941 0.756 2 2018 00-0020531 D.Brees 566 0.0865 0.739 3 2020 00-0033537 D.Watson 542 0.0735 0.705 4 2020 00-0033357 T.Hill 121 0.0725 0.727 5 2016 00-0026143 M.Ryan 631 0.0689 0.702 6 2019 00-0029701 R.Tannehill 343 0.0689 0.691 7 2020 00-0023459 A.Rodgers 607 0.0618 0.705 8 2017 00-0020531 D.Brees 606 0.0593 0.716 9 2018 00-0026143 M.Ryan 607 0.0590 0.695 10 2021 00-0036442 J.Burrow 659 0.0564 0.703 11 2016 00-0020531 D.Brees 664 0.0548 0.708 12 2018 00-0032950 C.Wentz 399 0.0546 0.699 13 2018 00-0023682 R.Fitzpatrick 246 0.0541 0.667 14 2022 00-0030565 G.Smith 605 0.0539 0.701 15 2016 00-0027854 S.Bradford 551 0.0529 0.717 16 2018 00-0029604 K.Cousins 603 0.0525 0.705 17 2017 00-0031345 J.Garoppolo 176 0.0493 0.682 18 2022 00-0031503 J.Winston 113 0.0488 0.646 19 2021 00-0023459 A.Rodgers 556 0.0482 0.694 20 2020 00-0034857 J.Allen 692 0.0478 0.684 \# ℹ 280 more rows

미래의 명예의 전당 헌액자인 드류 브리스(Drew Brees)는 NFL 역사상 가장 정확한 시즌을 보냈을 뿐만 아니라 패스 깊이를 조정한 후에도 NFL 역사상 가장 정확한 시즌 중 몇몇 기록을 보유하고 있습니다. 결과에서 브리스는 상위 11위 안에 총 4번 이름을 올렸습니다. 클리블랜드 브라운스의 쿼터백 데숀 왓슨(Deshaun Watson)은 2022년에 당시 NFL 역사상 가장 높은 전액 보장 계약을 체결했는데 2020년 CPOE에서 놀라울 정도로 좋은 점수를 기록했습니다. 반면 2016년 맷 라이언(Matt Ryan)은 리그 MVP를 수상했을 뿐만 아니라 패스 시도당 CPOE를 6.9%를 달성하면서 애틀랜타 팰컨스를 슈퍼볼로 이끌었습니다. 라이언의 2018년 시즌 기록도 선두권에 포함되어 있습니다.
2020년 애런 로저스(Aaron Rodgers)는 리그 MVP를 차지했습니다. 2021년 조 버로우(Joe Burrow)는 시도당 패싱 야드 및 패스 성공률 부문에서 리그 1위를 기록하며 벵골스를 1988년 이후 처음으로 슈퍼볼에 진출시켰습니다. 샘 브래드퍼드(Sam Bradford)의 2016년 시즌은 여전히 역대 패스 성공률 측면에서 상위 5위 안에 드는 시즌이며, 이후 드류 브리스와 동료 세인츠 쿼터백인 테이섬 힐([4장](ch04.html#sec-mr-ryoe2)에도 등장함)이 몇 번 넘어섰지만, 앞서 논의한 바와 같이 브래드퍼드는 역사적으로 CPOE에서 최고의 선수로 나타나지는 않습니다.

패스 깊이가 패스 성공률 측면에서 중요한 유일한 변수는 아닙니다. 모델에 몇 가지 특징(feature)을 추가해 보겠습니다. 다운(`down`), 퍼스트 다운을 위해 남은 거리(`ydstogo`), 엔드존까지 남은 거리(`yardline_100`), 패스 위치(`pass_location`) 및 쿼터백이 타격을 입었는지 여부(`qb_hit`; 이에 대해서는 나중에 자세히 설명합니다)입니다. 공식에는 `down`과 `ydstogo` 간의 교호 작용(interaction)도 포함될 것입니다.

먼저 Python에서 변수를 팩터(factor)로 변경하고, 사용할 열을 선택한 다음, `NA` 값을 삭제합니다.

```
## Python
# 누락된 데이터 제거 및 데이터 포맷팅
pbp_py_pass['down'] = pbp_py_pass['down'].astype(str)
pbp_py_pass['qb_hit'] = pbp_py_pass['qb_hit'].astype(str)

pbp_py_pass_no_miss = \
  pbp_py_pass[["passer", "passer_id", "season",
              "down", "qb_hit", "complete_pass",
              "ydstogo", "yardline_100",
              "air_yards",
              "pass_location"]]\
              .dropna(axis = 0)
```

그런 다음 Python에서 모델을 구축하고 맞춥니다.

```
## Python
complete_more_py = \
  smf.glm(formula='complete_pass ~ down * ydstogo + ' +
                  'yardline_100 + air_yards + ' +
                  'pass_location + qb_hit',
          data=pbp_py_pass_no_miss,
          family=sm.families.Binomial())\
          .fit()
```

다음으로 결과물을 추출하고 CPOE를 계산합니다.

```
## Python
pbp_py_pass_no_miss["exp_completion"] = \
  complete_more_py.predict()

pbp_py_pass_no_miss["cpoe"] = \
  pbp_py_pass_no_miss["complete_pass"] - \
  pbp_py_pass_no_miss["exp_completion"]
```

이제 결과물을 요약하고 열 이름과 형식을 바꿉니다.

```
## Python
cpoe_py_more = \
  pbp_py_pass_no_miss\
  .groupby(["season", "passer_id", "passer"])\
  .agg({"cpoe": ["count", "mean"],
        "complete_pass": ["mean"],
        "exp_completion": ["mean"]})

cpoe_py_more.columns = \
  list(map('_'.join, cpoe_py_more.columns))
cpoe_py_more.reset_index(inplace=True)

cpoe_py_more = \
  cpoe_py_more\
  .rename(columns = {"cpoe_count": "n",
                     "cpoe_mean": "cpoe",
                     "complete_pass_mean": "compl",
                     "exp_completion_mean": "exp_completion"})\
  .query("n > 100")
```

마지막으로 상위 20개 항목을 출력합니다(지면을 절약하기 위해 제한된 수의 행만 출력하지만 더 많이 출력해 볼 것을 권장합니다).

```
## Python
print(
  cpoe_py_more\
  .sort_values("cpoe", ascending=False)
  )
```

결과는 다음과 같습니다.

season passer_id passer n cpoe compl exp_completion 193 2018 00-0020531 D.Brees 566 0.088924 0.738516 0.649592 299 2019 00-0020531 D.Brees 406 0.087894 0.756158 0.668264 465 2020 00-0033357 T.Hill 121 0.082978 0.727273 0.644295 22 2016 00-0026143 M.Ryan 631 0.077565 0.702060 0.624495 467 2020 00-0033537 D.Watson 542 0.072763 0.704797 0.632034 .. ... ... ... ... ... ... ... 390 2019 00-0035040 D.Blough 174 -0.100327 0.540230 0.640557 506 2020 00-0036312 J.Luton 110 -0.107358 0.545455 0.652812 91 2016 00-0033106 J.Goff 204 -0.112375 0.549020 0.661395 526 2021 00-0027939 C.Newton 126 -0.123251 0.547619 0.670870 163 2017 00-0031568 B.Petty 112 -0.166726 0.491071 0.657798 \[300 rows x 7 columns\]

마찬가지로 R에서는 누락된 데이터를 제거하고 데이터 포맷을 지정합니다.

```
## R
pbp_r_pass_no_miss <-
  pbp_r_pass |>
  mutate(down = factor(down),
         qb_hit = factor(qb_hit)) |>
  filter(complete.cases(down, qb_hit, complete_pass,
                        ydstogo,yardline_100, air_yards,
                        pass_location, qb_hit))
```

그런 다음 R에서 모델을 실행하고 결과물을 저장합니다.

```
## R
complete_more_r <-
  pbp_r_pass_no_miss  |>
  glm(formula = complete_pass ~ down * ydstogo + yardline_100 +
                air_yards + pass_location + qb_hit,
      family = "binomial")
```

다음으로 CPOE를 계산합니다.

```
## R
pbp_r_pass_no_miss <-
  pbp_r_pass_no_miss |>
  mutate(exp_completion = predict(complete_more_r, type = "resp"),
         cpoe = complete_pass - exp_completion)
```

데이터를 요약합니다.

```
## R
cpoe_more_r <-
  pbp_r_pass_no_miss |>
  group_by(season, passer_id, passer) |>
  summarize(n = n(),
            cpoe = mean(cpoe , na.rm = TRUE),
            compl = mean(complete_pass),
            exp_completion = mean(exp_completion),
            .groups = "drop") |>
  filter(n > 100)
```

마지막으로 상위 20개 항목을 출력합니다(지면을 절약하기 위해 제한된 수의 행만 출력하지만 더 많이 출력해 볼 것을 권장합니다).

```
## R
cpoe_more_r |>
  arrange(-cpoe) |>
  print(n = 20)
```

결과는 다음과 같습니다.

\# A tibble: 300 × 7 season passer_id passer n cpoe compl exp_completion \<dbl\> \<chr\> \<chr\> \<int\> \<dbl\> \<dbl\> \<dbl\> 1 2018 00-0020531 D.Brees 566 0.0889 0.739 0.650 2 2019 00-0020531 D.Brees 406 0.0879 0.756 0.668 3 2020 00-0033357 T.Hill 121 0.0830 0.727 0.644 4 2016 00-0026143 M.Ryan 631 0.0776 0.702 0.624 5 2020 00-0033537 D.Watson 542 0.0728 0.705 0.632 6 2019 00-0029701 R.Tannehill 343 0.0667 0.691 0.624 7 2016 00-0027854 S.Bradford 551 0.0615 0.717 0.655 8 2018 00-0023682 R.Fitzpatrick 246 0.0613 0.667 0.605 9 2020 00-0023459 A.Rodgers 607 0.0612 0.705 0.644 10 2018 00-0026143 M.Ryan 607 0.0597 0.695 0.636 11 2018 00-0032950 C.Wentz 399 0.0582 0.699 0.641 12 2017 00-0020531 D.Brees 606 0.0574 0.716 0.659 13 2021 00-0036442 J.Burrow 659 0.0559 0.703 0.647 14 2016 00-0025708 M.Moore 122 0.0556 0.689 0.633 15 2022 00-0030565 G.Smith 605 0.0551 0.701 0.646 16 2021 00-0023459 A.Rodgers 556 0.0549 0.694 0.639 17 2017 00-0031345 J.Garoppolo 176 0.0541 0.682 0.628 18 2018 00-0033537 D.Watson 548 0.0539 0.682 0.629 19 2019 00-0029263 R.Wilson 573 0.0538 0.663 0.609 20 2018 00-0029604 K.Cousins 603 0.0533 0.705 0.652 \# ℹ 280 more rows

브리스의 최고 시즌 순위가 뒤바뀌어 이제 2018년 시즌이 CPOE 측면에서 가장 좋았고 2019년이 그 뒤를 이었습니다. 이러한 순위 변동이 발생하는 이유는 모델의 특징(features)이 다르기 때문에 모델이 선수에 대해 약간 다른 추정치를 산출하기 때문입니다. 맷 라이언의 2016년 MVP 시즌은 왓슨의 2020년 기록을 뛰어넘는 한편 게임 조건을 함께 고려하면 샘 브래드퍼드도 다시 선두 경쟁에 뛰어듭니다. 2018년 제이미스 윈스턴(Jameis Winston)과 출전 시간을 나눠 가졌음에도 패스 시도당 야드 부문에서 NFL 1위를 차지했던 저니맨 라이언 피츠패트릭(Ryan Fitzpatrick)도 상위 그룹에 합류합니다.

# CPOE는 패스 성공률보다 더 안정적일까? (Is CPOE More Stable Than Completion Percentage?)

러닝백에 대해 했던 것과 마찬가지로 CPOE가 단순 패스 성공률보다 더 안정적인지 확인하는 것이 중요합니다. 만약 그렇다면 주변 상황보다 선수의 퍼포먼스를 분리해 내고 있다고 확신할 수 있습니다. 이를 위해 코드에 대해 깊이 파헤쳐 보겠습니다.

먼저 Python을 사용하여 현재 CPOE와 작년 CPOE 간의 시차(lag)를 계산합니다.

```
## Python
# 필요한 열만 유지합니다
cols_keep =\
    ["season", "passer_id", "passer",
     "cpoe", "compl", "exp_completion"]

# 현재 데이터프레임을 생성합니다
cpoe_now_py =\
    cpoe_py_more[cols_keep].copy()

# 작년 데이터프레임을 생성합니다
cpoe_last_py =\
    cpoe_now_py[cols_keep].copy()

# 열 이름을 바꿉니다
cpoe_last_py\
    .rename(columns = {'cpoe': 'cpoe_last',
                       'compl': 'compl_last',
                       'exp_completion': 'exp_completion_last'},
                       inplace=True)

# season에 1을 더합니다
cpoe_last_py["season"] += 1

# 병합합니다
cpoe_lag_py =\
    cpoe_now_py\
    .merge(cpoe_last_py,
           how='inner',
           on=['passer_id', 'passer',
               'season'])
```

그런 다음 패스 성공의 상관관계를 살펴봅니다.

```
## Python
cpoe_lag_py[['compl_last', 'compl']].corr()
```

결과는 다음과 같습니다.

compl_last compl compl_last 1.000000 0.445465 compl 0.445465 1.000000

다음은 CPOE입니다.

```
## Python
cpoe_lag_py[['cpoe_last', 'cpoe']].corr()
```

결과는 다음과 같습니다.

cpoe_last cpoe cpoe_last 1.000000 0.464974 cpoe 0.464974 1.000000

R에서도 이 계산을 수행할 수 있습니다.

```
## R
# 현재 데이터프레임을 생성합니다
cpoe_now_r <-
    cpoe_more_r |>
    select(-n)

# 작년 데이터프레임을 생성하고
# season에 1을 더합니다
cpoe_last_r <-
    cpoe_more_r |>
    select(-n) |>
    mutate(season = season + 1) |>
    rename(cpoe_last = cpoe,
           compl_last = compl,
           exp_completion_last = exp_completion
           )

# 병합합니다
cpoe_lag_r <-
    cpoe_now_r |>
    inner_join(cpoe_last_r,
               by = c("passer_id", "passer", "season")) |>
    ungroup()
```

그런 다음 두 패스 성공(passing completion) 열을 선택하고 상관관계를 조사합니다.

```
## R
cpoe_lag_r |>
  select(compl_last, compl) |>
  cor(use="complete.obs")
```

결과는 다음과 같습니다.

compl_last compl compl_last 1.0000000 0.4454646 compl 0.4454646 1.0000000

CPOE 열을 가지고 반복합니다.

```
## R
cpoe_lag_r |>
  select(cpoe_last, cpoe) |>
  cor(use="complete.obs")
```

결과는 다음과 같습니다.

cpoe_last cpoe cpoe_last 1.0000000 0.4649739 cpoe 0.4649739 1.0000000

CPOE가 패스 성공률보다 약간 더 안정적인 것으로 나타났습니다! 따라서 일관성 측면에서 CPOE를 구축함으로써 상황을 약간 더 낫게 만들고 있습니다.

첫 번째로, 그리고 가장 중요하게: 패스 성공 기대치에 포함된 특성들은 쿼터백에게 근본적인 것일 수 있습니다. 드류 브리스와 같은 일부 쿼터백은 그저 특징적으로 더 짧은 패스를 던집니다. 더 많은 태클을 당하는 선수도 있습니다. 사실 [에릭을 포함하여](https://oreil.ly/Y1psK) 많은 사람들은 태클을 당하는 것이 적어도 부분적으로는 쿼터백의 잘못이라고 주장했습니다. 일부 팀은 경험적으로 완료하기 더 쉬운 초반 다운에서 많이 던지는 반면, 다른 팀은 후반 다운에서만 던집니다. 쿼터백은 팀을 자주 바꾸지 않으므로, 그 상황이 반드시 쿼터백 본인에게 내재되어 있지 않더라도 그들이 플레이하는 체계(scheme)는 안정적일 수 있습니다.

마지막으로 예상 성공률의 안정성을 살펴봅니다.

```
## R
cpoe_lag_r |>
  select(exp_completion_last, exp_completion) |>
  cor(use="complete.obs")
```

결과는 다음과 같습니다.

exp_completion_last exp_completion exp_completion_last 1.000000 0.473959 exp_completion 0.473959 1.000000

이 장에서 가장 안정적인 메트릭은 실제로 쿼터백의 평균 *기대 성공률(expected completion percentage)*입니다.

# 잔차 메트릭에 관한 의문점 (A Question About Residual Metrics)

이 장의 결과는 전반적으로 풋볼과 스포츠 모델링에서 발생할 수 있는 문제를 명확히 보여줍니다. 메트릭에서 상황적 맥락(context)을 제거하려는 시도에는 실수할 여지가 많습니다. 주어진 플레이에서 선수가 자신이 속한 상황을 좌우하지 않는다는 가정은 반복적으로 위반될 가능성이 높습니다.

예를 들어, NFL NGS 버전의 CPOE에는 리시버 분리(receiver separation)가 포함되는데, 얼핏 보기에는 쿼터백에게 외적인 요소인 것처럼 보입니다. 리시버가 오픈 상태가 되는 것은 쿼터백의 몫이 아니니까요. 하지만 쿼터백은 몇 가지 방식으로 이에 기여합니다. 첫째, 쿼터백이 패스하기로 결정한 선수 즉, 쿼터백이 선택한 분리(separation)는 그들의 선택입니다. 둘째, 쿼터백은 시선으로 수비수를 움직일 수 있으며 이로 인해 분리 프로필(separation profile)을 바꿀 수 있습니다. 많은 사람들이 제56회 슈퍼볼에서 매튜 스태퍼드(Matthew Stafford)가 보여준 노룩 패스를 기억할 것입니다.

마지막으로, 쿼터백이 실제로 패스를 하는지 여부에는 어떤 신호(signal)가 있습니다. 이미 언급했듯이 조 버로우는 2021년에 패스 시도당 야드와 성공률 부문에서 리그 1위를 차지했습니다. 또한 피색(sacks taken) 부문에서도 51개로 NFL 1위를 차지했습니다. 다른 쿼터백들은 압박을 피하고, 어떤 쿼터백들은 공을 들고 뛰는 반면, 또 어떤 쿼터백들은 공을 밖으로 던져버립니다(throw it away). 이러한 요소들은 (적어도 부분적으로) 쿼터백에 의해 주도되는 이유로 기대치를 변경합니다.

그렇다면 누군가는 어떻게 해야 할까요? 이에 대한 대답은 여러 질문에 대한 대답과 마찬가지로, 상황에 따라 다릅니다(it depends). NFL에서 가장 정확한 패서를 가려내려는 경우 단일 숫자로는 충분하지 않을 수 있습니다(단 하나의 메트릭이 이 문제나 풋볼과 관련된 질문에 명확하게 답하기에 충분할 가능성은 낮습니다).

선수 영입, 판타지 풋볼 또는 스포츠 베팅을 목적으로 선수의 퍼포먼스를 예측하려는 경우 기대치에서 맥락(context)을 제거한 다음, "기대치 이상(over expected)" 분석을 잘 조정된 모델에 적용해 보는 것도 괜찮을 것입니다. 예를 들어, 시즌 중 패트릭 마홈스(Patrick Mahomes)의 패스 성공률을 예측하려는 경우 그의 상황이 주어졌을 때의 기대 패스 성공률과 그의 CPOE를 더해야 합니다. 후자는 많은 사람들이 거의 전적으로 마홈스에게 기인한다고 여기지만, 전자에도 역시 마홈스의 게임 능력이 어느 정도 포함되어 있습니다. 이 둘이 완전히 독립적이라고 가정하는 것은 오류를 초래할 가능성이 높습니다.

더 많고 더 좋은 데이터에 접근할 수 있게 되면, 이러한 모델링 오류 중 일부를 줄일 수 있는 가능성도 얻게 됩니다. 그러나 이러한 감소를 위해서는 모델링에서 부단한 노력이 요구됩니다. 그것이 이 과정을 재미있게 만드는 요소입니다.

###### 팁 (Tip)

"더 좋은" 데이터를 구매하기 전에 데이터 기술을 연마하는 것을 권장합니다. 무료 데이터의 한계에 도달하면 왜 더 좋은 데이터가 필요한지 깨닫게 될 것입니다. 그리고 나서야 실제로 그 값비싼 데이터를 활용할 수 있게 될 것입니다.

# 오즈비에 대한 간략한 입문서 (A Brief Primer on Odds Ratios)

로지스틱 회귀를 사용하면 계수를 로그 오즈(log-odds) 용어로도 이해할 수 있습니다. 일상 생활에서 오즈(odds)가 흔히 사용되지 않기 때문에 대부분의 사람들은 로그 오즈를 이해하지 못합니다. 또한 오즈비(odds ratio)의 *오즈(odds)*는 베팅 오즈(betting odds)와는 다릅니다.

###### 경고 (Warning)

오즈비는 때때로 로지스틱 회귀를 이해하는 데 도움을 줄 수 있습니다. 반면, 사람을 심하게 오도할 수도 있습니다.

예를 들어, 오즈비에서 패스 실패 2번당 패스 성공 3번을 예상한다면 오즈비는 3대 2가 될 것입니다. 3대 2의 오즈비는 1.5대 1로 암시되며 소수점 형식으로 오즈비 1.5로 쓸 수 있습니다($`\frac{3}{2} = 1\!.\! 5`$이므로).

오즈비는 로지스틱 회귀 계수의 지수(많은 계산기에서 $`e^{x}`$ 또는 `exp()` 함수)를 취하여 계산할 수 있습니다. 예를 들어 `broom` 패키지에는 오즈비를 쉽게 계산하고 표시할 수 있는 `tidy()` 함수가 있습니다.

```
## R
complete_ay_r |>
  tidy(exponentiate = TRUE, conf.int = TRUE)
```

결과는 다음과 같습니다.

\# A tibble: 2 × 7 term estimate std.error statistic p.value conf.low conf.high \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 2.92 0.00804 133. 0 2.88 2.97 2 air_yards 0.944 0.000624 -91.8 0 0.943 0.945

오즈비 척도에서는 값이 1과 다른지에 주의를 기울여야 하는데, 1:1의 오즈비는 예측 변수(predictor)가 사건의 결과를 바꾸지 않거나 계수가 모델의 예측에 영향을 미치지 않음을 의미하기 때문입니다. 이제 이 절편은 남은 야드가 0일 때 패스 성공 오즈비가 `2.92`임을 알려줍니다. 그러나 남은 거리가 1야드 증가할 때마다 오즈비는 `0.94`만큼 감소합니다.

1야드가 추가될 때마다 오즈비가 얼마나 감소하는지 확인하려면 절편과 `air_yards` 계수를 곱합니다. 먼저 추가로 필요한 매 야드마다 `air_yards` 계수를 자체적으로 곱하거나(예: `air_yards` × `air_yards`) 더 일반적으로 `air_yards`를 남은 거리만큼 거듭제곱합니다(예: 2야드인 경우 `air_yards`<sup>2</sup>, 9야드인 경우 `air_yards`<sup>9</sup>). 예를 들어 그림 <a href="#fig-py-pass-comp" data-type="xref" data-xrefstyle="select:labelnumber">5-3</a> 또는 <a href="#fig-r-pass-comp" data-type="xref" data-xrefstyle="select:labelnumber">5-4</a>를 보면 에어 야드가 약 20야드 이상인 패스는 성공 확률이 50% 미만이라는 것을 알 수 있습니다. 2.92에 0.94의 20제곱을 곱하면(2.92 × 20<sup>20</sup>) 남은 거리가 20야드일 때 패스 성공 확률이 0.85임을 알 수 있습니다. 이는 50%보다 약간 적으며 그림 <a href="#fig-py-pass-comp" data-type="xref" data-xrefstyle="select:labelnumber">5-3</a> 및 <a href="#fig-r-pass-comp" data-type="xref" data-xrefstyle="select:labelnumber">5-4</a>의 내용과 일치합니다.

오즈비를 더 잘 이해할 수 있도록 R에서 오즈비를 계산하는 방법을 보여 드리겠습니다. R 언어에 `glm()` 결과물을 다루기 위한 더 나은 도구가 있고 계산을 직접 할 수 있는 것보다 계산 과정을 따라가는 것이 더 중요하기 때문에 R을 사용합니다. 먼저 플레이별 패스 데이터세트의 모든 데이터에 대한 패스 성공률을 계산합니다. 다음으로 패스 성공률을 1에서 패스 성공률을 뺀 값으로 나누어 오즈를 계산합니다. 그런 다음 자연로그를 취하여 로그 오즈를 계산합니다.

```
## R
pbp_r_pass |>
  summarize(comp_pct = mean(complete_pass)) |>
  mutate(odds = comp_pct / (1 - comp_pct),
         log_odds = log(odds))
```

결과는 다음과 같습니다.

\# A tibble: 1 × 3 comp_pct odds log_odds \<dbl\> \<dbl\> \<dbl\> 1 0.642 1.79 0.583

다음으로 전역 절편(global intercept, 즉 모든 관측치에 대한 평균)만 있는 로지스틱 회귀에 대해 이 결과물을 로지스틱 회귀 결과물과 비교합니다. 먼저 전역 절편이 포함된 모델(`complete_pass ~ 1`)을 구축합니다. 원시(raw) 출력 및 지수화된 출력에 대한 `tidy()` 계수를 확인합니다.

```
## R
complete_global_r <-
  glm(complete_pass ~ 1,
      data = pbp_r_pass,
      family = "binomial")

complete_global_r |>
  tidy()
```

결과는 다음과 같습니다.

\# A tibble: 1 × 5 term estimate std.error statistic p.value \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 0.583 0.00575 101. 0

```
complete_global_r |>
  tidy(exponentiate = TRUE)
```

결과는 다음과 같습니다.

\# A tibble: 1 × 5 term estimate std.error statistic p.value \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> 1 (Intercept) 1.79 0.00575 101. 0

결과물을 백분율을 사용하여 이전에 계산한 숫자와 비교해 보세요. 이제 다시는 오즈비를 손으로 계산할 필요가 없기를 바랍니다!

# 이 장에서 사용된 데이터 과학 도구 (Data Science Tools Used in This Chapter)

이 장에서는 다음 주제를 다루었습니다.

- `glm()`을 사용하여 Python과 R에서 로지스틱 회귀 맞추기

- 오즈비를 포함하여 로지스틱 회귀에서 도출된 계수를 읽고 이해하기

- 이전 장에서 배운 데이터 랭글링(data-wrangling) 도구를 다시 적용하기

# 연습문제 (Exercises)

1.  `qb_hit`를 특징 중 하나로 삼지 않고 이 분석을 반복해 보세요. 순위표(leaderboard)는 어떻게 변하나요? 이를 통해 무엇을 알 수 있나요?

2.  로지스틱 회귀에 추가할 수 있는 다른 특징으로는 어떤 것들이 있을까요? 이 장의 안정성 결과에는 어떤 영향을 미치나요?

3.  리시버(receivers)에 대해 이 분석을 시도해 보세요. 흥미로운 사실이 나타나나요?

4.  수비 포지션(defensive positions)에 대해 이 분석을 시도해 보세요. 흥미로운 사실이 나타나나요?

# 추천 도서 (Suggested Readings)

[76페이지](ch03.html#sec-chp3-fr) 및 [112페이지](ch04.html#sec-chp4-sr)의 "추천 도서(Suggested Readings)"에서 추천하는 자료는 일반화 선형 모델을 이해하는 데 도움이 될 것입니다. 다른 참고 문헌은 다음과 같습니다.

- 쿼터백에 관한 에릭의 PFF 기사, ["통제 하의 쿼터백: 누가 압박률을 통제하는지에 대한 PFF 데이터 연구(Quarterbacks in Control: A PFF Data Study of Who Controls Pressure Rates)"](https://oreil.ly/lhIJu).

- Paul Roback과 Julie Legler가 저술한 *Beyond Multiple Linear Regression: Applied Generalized Linear Models and Multilevel Models in R* (CRC Press, 2021). 제목에서 알 수 있듯이 이 책은 선형 회귀 분석을 넘어 모델의 중요한 가정을 포함하여 일반화 선형 모델을 가르치는 훌륭한 교재입니다.

- Andrew Gelman 외 공저, *Bayesian Data Analysis* 3판 (CRC Press, 2013). 이 책은 고급 모델링 기술을 위한 고전적인 서적이지만 선형 대수에 대한 탄탄한 이해를 필요로 합니다.
