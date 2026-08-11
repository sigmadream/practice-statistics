### 8 경력 궤적

- 8.1 소개

R 시스템은 데이터에 통계 모델을 맞추는 데 적합합니다. 세이버메트릭스에서 인기 있는 주제 중 하나는 선수의 메이저리그 데뷔부터 은퇴까지 시즌 타격, 수비 또는 투구 통계의 상승과 하락입니다. 일반적으로 대부분의 선수들은 20대 후반에 기량이 정점에 달한다고 여겨지지만, 일부 선수들은 더 늦은 나이에 정점에 달하는 경향이 있습니다. 선수의 궤적을 모델링하는 간단한 방법은 이차 곡선(quadratic curve) 또는 포물선(parabolic curve)을 이용하는 것입니다. R의 lm() (선형 모델) 함수를 사용하면 선수의 나이와 OPS 통계를 이용하여 이 모델을 쉽게 맞출 수 있습니다.

8.2절에서는 유명한 경력 궤적을 살펴보며 시작합니다. 미키 맨틀(Mickey Mantle)은 19세에 뉴욕 양키스에 즉각적인 영향을 미쳤으며 빠르게 야구계 최고의 타자 중 한 명으로 성장했습니다. 하지만 부상은 맨틀의 성과에 큰 타격을 입혔고, 36세에 은퇴할 때까지 그의 타격은 감소했습니다. 우리는 맨틀을 통해 이차 모델을 소개합니다. 이 모델을 사용하면 그의 전성기 나이, 최대 성과, 그리고 성과 향상 및 감소율을 정의할 수 있습니다.

비슷한 선수들의 경력 성과를 비교하려면 그들의 궤적을 대조하는 것이 도움이 되며, 8.3절은 많은 적합된 궤적의 계산을 보여줍니다. 빌 제임스(Bill James)의 유사도 점수 개념을 사용하여, 주어진 타자와 가장 유사한 선수를 찾는 함수를 작성합니다. 그런 다음 이러한 유사한 선수들의 OPS 궤적을 그래픽으로 비교합니다. 이 그래프들을 통해 가능한 궤적 형태에 대한 전반적인 이해를 얻습니다.

일반적인 문제는 선수의 전성기 나이에 초점을 맞춥니다. 8.4절에서는 통산 2000타수 이상을 기록한 모든 타자들의 적합된 궤적을 살펴봅니다. 시대별 및 통산 타수 함수로서의 전성기 나이 패턴을 탐구합니다. 또한, 같은 포지션을 뛰는 선수들을 비교하는 것이 일반적이므로, 8.5절에서는 1985-1995년 기간에 초점을 맞추고 다른 수비 포지션을 뛰는 선수들의 전성기 나이를 대조합니다.

DOI: 10.1201/9781032668239-8 181

- 8.2 미키 맨틀의 타격 궤적

경력 궤적을 살펴보기 위해 위대한 강타자 미키 맨틀의 타격 데이터를 고려합니다. 그의 시즌별 타격 통계를 얻기 위해 People과 Batting 데이터 프레임이 포함된 Lahman 패키지를 불러옵니다. 추가로 tidyverse 패키지를 불러옵니다.

library(tidyverse) library(Lahman)

먼저 People 데이터 프레임에서 맨틀의 playerID를 추출합니다. filter() 함수를 사용하여 People 데이터 파일에서 nameFirst가 "Mickey"이고 nameLast가 "Mantle"인 행을 찾습니다. 그의 선수 ID는 mantle_id 벡터에 저장됩니다.

mantle_id <- People |> filter(nameFirst == "Mickey", nameLast == "Mantle") |> pull(playerID)

한 가지 작은 문제점은 희생 플라이(SF)나 몸에 맞는 공(HBP)과 같은 특정 통계가 오래된 시즌에는 기록되지 않았으며 현재 NA로 코딩되어 있다는 것입니다. tidyr 패키지의 replace_na() 함수를 사용하면 이러한 결측값을 0으로 다시 코딩하는 편리한 방법입니다.

batting <- Batting |> replace_na(list(SF = 0, HBP = 0))

각 시즌별 맨틀의 나이를 계산하려면 People 데이터 프레임에서 제공되는 그의 출생 연도를 알아야 합니다. 메이저리그 야구는 선수의 나이를 해당 시즌 6월 30일 기준 나이로 정의합니다.

사용자 정의 함수 get_stats()를 통해 맨틀의 타격 통계를 얻습니다. 입력은 선수 ID이고 출력은 선수의 타격 통계를 포함하는 데이터 프레임입니다. 이 함수는 모든 시즌에 대한 선수의 나이(Age 변수)를 계산하고, 또한 모든 시즌에 대한 선수의 장타율(SLG), 출루율(OBP) 및 OPS를 계산합니다.

get_stats <- function(player_id) {

batting |> filter(playerID == player_id) |> inner_join(People, by = "playerID") |> mutate(

8.2 미키 맨틀의 타격 궤적

경력 궤적을 살펴보기 위해 위대한 강타자 미키 맨틀의 타격 데이터를 고려합니다. 그의 시즌별 타격 통계를 얻기 위해 People과 Batting 데이터 프레임이 포함된 Lahman 패키지를 불러옵니다. 추가로 tidyverse 패키지를 불러옵니다.

library(tidyverse) library(Lahman)

먼저 People 데이터 프레임에서 맨틀의 playerID를 추출합니다. filter() 함수를 사용하여 People 데이터 파일에서 nameFirst가 "Mickey"이고 nameLast가 "Mantle"인 행을 찾습니다. 그의 선수 ID는 mantle_id 벡터에 저장됩니다.

mantle_id <- People |> filter(nameFirst == "Mickey", nameLast == "Mantle") |> pull(playerID)

한 가지 작은 문제점은 희생 플라이(SF)나 몸에 맞는 공(HBP)과 같은 특정 통계가 오래된 시즌에는 기록되지 않았으며 현재 NA로 코딩되어 있다는 것입니다. tidyr 패키지의 replace_na() 함수를 사용하면 이러한 결측값을 0으로 다시 코딩하는 편리한 방법입니다.

batting <- Batting |> replace_na(list(SF = 0, HBP = 0))

각 시즌별 맨틀의 나이를 계산하려면 People 데이터 프레임에서 제공되는 그의 출생 연도를 알아야 합니다. 메이저리그 야구는 선수의 나이를 해당 시즌 6월 30일 기준 나이로 정의합니다.

사용자 정의 함수 get_stats()를 통해 맨틀의 타격 통계를 얻습니다. 입력은 선수 ID이고 출력은 선수의 타격 통계를 포함하는 데이터 프레임입니다. 이 함수는 모든 시즌에 대한 선수의 나이(Age 변수)를 계산하고, 또한 모든 시즌에 대한 선수의 장타율(SLG), 출루율(OBP) 및 OPS를 계산합니다.

get_stats <- function(player_id) {

batting |> filter(playerID == player_id) |> inner_join(People, by = "playerID") |> mutate(

![image 67](images/imageFile67.png)

그림 8.1 미키 맨틀의 나이에 따른 OPS 산점도.

}

birthyear = if_else( birthMonth >= 7, birthYear + 1, birthYear

), Age = yearID - birthyear, SLG = (H - X2B - X3B - HR + 2 _ X2B + 3 _ X3B + 4 \* HR) / AB, OBP = (H + BB + HBP) / (AB + BB + HBP + SF), OPS = SLG + OBP

) |> select(Age, SLG, OBP, OPS)

함수 get_stats()를 R로 읽어들인 후, 입력값 mantle_id와 함께 이 함수를 적용하여 맨틀의 통계를 얻습니다. 결과 타격 통계 데이터 프레임은 Mantle에 저장됩니다.

Mantle <- get_stats(mantle_id)

타격 성과를 측정하는 좋은 지표는 선수의 장타율과 출루율의 합인 OPS입니다. 맨틀의 OPS 시즌 값은 그의 나이에 따라 어떻게 변할까요? 이 질문에 답하기 위해 ggplot2를 사용하여 나이에 따른 OPS 산점도를 구성합니다 (그림 8.1 참고).

ggplot(Mantle, aes(Age, OPS)) + geom_point()

그림 8.1에서 맨틀의 OPS 값은 19세부터 20대 후반까지 증가하는 경향이 있으며, 이후 36세에 은퇴할 때까지 전반적으로 감소한다는 것이 분명합니다. 이 오르내리는 관계는 부드러운 곡선을 사용하여 모델링할 수 있습니다. 이 곡선은 맨틀의 경력 타격 궤적을 이해하고 요약하는 데 도움이 되며, 맨틀의 궤적을 비슷한 타격 성과를 가진 다른 선수들과 쉽게 비교할 수 있도록 해줍니다.

부드러운 곡선의 편리한 선택은 다음과 같은 형태의 이차 함수입니다.
A + B(Age − 30) + C(Age − 30)2,

여기서 상수 A, B, C는 곡선이 산점도의 점들과 "가장 잘" 일치하도록 선택됩니다. 이 이차 곡선은 사용하기 쉽게 만드는 다음과 같은 좋은 특성들을 가지고 있습니다.

- 1. 상수 A는 선수가 30세일 때 예상되는 OPS 값입니다.
- 2. 이 함수는 다음에서 가장 큰 값에 도달합니다.

PEAK AGE = 30 −

B 2C

.

이것은 선수가 경력 동안 최고 타격 성과를 낼 것으로 예상되는 나이입니다.

- 3. 곡선의 최대값은 다음과 같습니다.

MAX = A −

B2 4C

.

이것은 선수의 경력 동안 예상되는 가장 큰 OPS입니다.

- 4. 일반적으로 음수 값을 갖는 계수 C는 이차 함수의 곡률 정도에 대해 알려줍니다. 만약 선수가 "큰" C 값을 가진다면, 이것은 그가 더 빠르게 최고 수준에 도달하고 은퇴할 때까지 능력이 더 빠르게 감소한다는 것을 나타냅니다. 한 가지 간단한 해석은 C가 전성기 나이부터 1년 후까지의 OPS 변화를 나타낸다는 것입니다.

우리는 이 이차 곡선을 선수의 타격 데이터에 맞추기 위해 새로운 함수 fit_model()을 작성합니다. 이 함수의 입력은 Age와 OPS 변수를 포함하는 선수의 타격 통계를 포함하는 데이터 프레임 d입니다. 이차 곡선을 맞추는 데 lm() 함수가 사용됩니다. 공식

OPS ∼ I(Age – 30) + I((Age – 30)2)

은 OPS가 반응 변수이고 (Age - 30)과 (Age - 30)^2가 예측 변수임을 나타냅니다. 추정된 계수 A, B, C는 coef() 함수를 사용하여 벡터 b에 저장됩니다. 전성기 나이와 최대값은 Age_max와 Max 변수에 저장됩니다.

그림 8.1에서 맨틀의 OPS 값은 19세부터 20대 후반까지 증가하는 경향이 있으며, 이후 36세에 은퇴할 때까지 전반적으로 감소한다는 것이 분명합니다. 이 오르내리는 관계는 부드러운 곡선을 사용하여 모델링할 수 있습니다. 이 곡선은 맨틀의 경력 타격 궤적을 이해하고 요약하는 데 도움이 되며, 맨틀의 궤적을 비슷한 타격 성과를 가진 다른 선수들과 쉽게 비교할 수 있도록 해줍니다.

부드러운 곡선의 편리한 선택은 다음과 같은 형태의 이차 함수입니다.
A + B(Age − 30) + C(Age − 30)2,

여기서 상수 A, B, C는 곡선이 산점도의 점들과 "가장 잘" 일치하도록 선택됩니다. 이 이차 곡선은 사용하기 쉽게 만드는 다음과 같은 좋은 특성들을 가지고 있습니다.

- 1. 상수 A는 선수가 30세일 때 예상되는 OPS 값입니다.
- 2. 이 함수는 다음에서 가장 큰 값에 도달합니다.

PEAK AGE = 30 −

B 2C

.

이것은 선수가 경력 동안 최고 타격 성과를 낼 것으로 예상되는 나이입니다.

- 3. 곡선의 최대값은 다음과 같습니다.

MAX = A −

B2 4C

.

이것은 선수의 경력 동안 예상되는 가장 큰 OPS입니다.

- 4. 일반적으로 음수 값을 갖는 계수 C는 이차 함수의 곡률 정도에 대해 알려줍니다. 만약 선수가 "큰" C 값을 가진다면, 이것은 그가 더 빠르게 최고 수준에 도달하고 은퇴할 때까지 능력이 더 빠르게 감소한다는 것을 나타냅니다. 한 가지 간단한 해석은 C가 전성기 나이부터 1년 후까지의 OPS 변화를 나타낸다는 것입니다.

우리는 이 이차 곡선을 선수의 타격 데이터에 맞추기 위해 새로운 함수 fit_model()을 작성합니다. 이 함수의 입력은 Age와 OPS 변수를 포함하는 선수의 타격 통계를 포함하는 데이터 프레임 d입니다. 이차 곡선을 맞추는 데 lm() 함수가 사용됩니다. 공식

OPS ∼ I(Age – 30) + I((Age – 30)2)

은 OPS가 반응 변수이고 (Age - 30)과 (Age - 30)^2가 예측 변수임을 나타냅니다. 추정된 계수 A, B, C는 coef() 함수를 사용하여 벡터 b에 저장됩니다. 전성기 나이와 최대값은 Age_max와 Max 변수에 저장됩니다.

fit_model <- function(d) { fit <- lm(OPS ~ I(Age - 30) + I((Age - 30)^2), data = d) b <- coef(fit) Age_max <- 30 - b[2] / b[3] / 2 Max <- b[1] - b[2] ^ 2 / b[3] / 4 list(fit = fit, Age_max = Age_max, Max = Max)

}

그런 다음 맨틀의 데이터 프레임에 fit_model() 함수를 적용합니다. 이 함수 F2의 출력에는 이차 적합의 모든 계산 결과를 저장하는 객체가 포함됩니다. 추가로 이 함수는 다음 코드에 표시된 전성기 나이와 최대값을 출력합니다.

F2 <- fit_model(Mantle) F2 |>

pluck("fit") |> coef()

(Intercept) I(Age - 30) I((Age - 30)^2) 1.04313 -0.02288 -0.00387

c(F2$Age_max, F2$Max)

I(Age - 30) (Intercept) 27.04 1.08

가장 잘 맞는 곡선은 1.04313 − 0.02288(Age − 30) − 0.00387(Age − 30)2 로 주어집니다.

이 모델을 사용하면 맨틀은 27세에 정점에 도달했으며 곡선에 대한 그의 최대 OPS는 1.08로 추정됩니다. 곡률 매개변수의 추정값은 -0.00387입니다. 따라서 맨틀의 전성기 나이와 1년 후의 OPS 감소량은 0.00387입니다.

이 가장 잘 맞는 이차 곡선을 산점도 위에 배치합니다. geom_smooth() 함수는 나이 값의 순서에 대한 곡선에서 맨틀의 OPS를 추정하고 이 값들을 현재 플롯 위에 선으로 겹쳐 그리는 데 사용됩니다. geom_vline()과 geom_hline()을 적용하여 각각 전성기 나이와 최대값의 위치를 표시하고, annotate() 함수를 사용하여 이 값들에 레이블을 지정합니다. 결과 그래프는 그림 8.2에 표시됩니다.

ggplot(Mantle, aes(Age, OPS)) + geom_point() +

geom_smooth( method = "lm", se = FALSE, linewidth = 1.5, formula = y ~ poly(x, 2, raw = TRUE)

|                                                                                                  |     |     |     |     |          |     |     |     |     |
| ------------------------------------------------------------------------------------------------ | --- | --- | --- | --- | -------- | --- | --- | --- | --- |
|                                                                                                  |     |     |     |     |          |     |     |     |
| Max                                                                                              |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     |          |     |     |     |
|                                                                                                  |     |     |     |     | Peak age |     |     |
| <br><br>0.7<br><br>0.8<br><br>0.9<br><br>1.0<br><br>1.1<br><br>20 25 30 35<br><br>Age<br><br>OPS |

OPS

그림 8.2 이차 평활 곡선에서 식별된 전성기 나이와 최대 OPS를 포함하는 미키 맨틀의 OPS 척도 경력 궤적.

) + geom_vline(

xintercept = F2$Age_max, linetype = "dashed", color = "red"

) + geom_hline(

yintercept = F2$Max, linetype = "dashed", color = "red"

) + annotate(

geom = "text", x = c(29, 20), y = c(0.72, 1.1), label = c("Peak age", "Max"), size = 5, color = "red"

)

가장 잘 맞는 이차 곡선에 초점을 맞추었지만, 적합 과정에 대한 더 자세한 내용은 lm()의 출력과 F2 변수에 저장됩니다. 적합의 summary()를 찾아 출력의 일부를 표시합니다. 여기서 우리는 목록에서 항목을 검색하기 위해 pluck() 함수의 사용을 설명합니다.

F2 |> pluck("fit") |> summary()

Call:

1.1

1.0

OPS

0.9

0.8

Max

0.7

Peak age

20 25 30 35

Age

그림 8.2 이차 평활 곡선에서 식별된 전성기 나이와 최대 OPS를 포함하는 미키 맨틀의 OPS 척도 경력 궤적.

) + geom_vline(

xintercept = F2$Age_max, linetype = "dashed", color = "red"

) + geom_hline(

yintercept = F2$Max, linetype = "dashed", color = "red"

) + annotate(

geom = "text", x = c(29, 20), y = c(0.72, 1.1), label = c("Peak age", "Max"), size = 5, color = "red"

)

가장 잘 맞는 이차 곡선에 초점을 맞추었지만, 적합 과정에 대한 더 자세한 내용은 lm()의 출력과 F2 변수에 저장됩니다. 적합의 summary()를 찾아 출력의 일부를 표시합니다. 여기서 우리는 목록에서 항목을 검색하기 위해 pluck() 함수의 사용을 설명합니다.

F2 |> pluck("fit") |> summary()

Call:

lm(formula = OPS ~ I(Age - 30) + I((Age - 30)^2), data = d) Residuals:

Min 1Q Median 3Q Max

-0.1728 -0.0401 0.0220 0.0451 0.1282 Coefficients:

Estimate Std. Error t value Pr(>|t|) (Intercept) 1.043134 0.027901 37.39 3.2e-16 **\* I(Age - 30) -0.022883 0.005638 -4.06 1e-03 ** I((Age - 30)^2) -0.003869 0.000828 -4.67 3e-04 \*\*\*

--Signif. codes: 0 '**\*' 0.001 '**' 0.01 '\*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.0842 on 15 degrees of freedom Multiple R-squared: 0.602, Adjusted R-squared: 0.549 F-statistic: 11.3 on 2 and 15 DF, p-value: 0.001

R2의 값은 0.602입니다. 이것은 맨틀의 OPS 값 변동성의 약 60%가 이차 곡선으로 설명될 수 있음을 의미합니다. 잔차 표준 오차는 0.084와 같습니다. 곡선으로부터의 수직 편차(잔차)의 약 2/3는 잔차 표준 오차의 플러스와 마이너스 사이에 위치합니다. 이 경우 잔차의 약 2/3가 -0.084와 0.084 사이에 위치한다고 해석됩니다.

### 8.3 궤적 비교

- 8.3.1 몇 가지 예비 작업

선수들의 타격 궤적을 생각할 때 관련 있는 변수 중 하나는 선수의 수비 포지션인 것 같습니다. 중요한 수비 포지션인 포수의 타격 기대치는 1루수의 타격 기대치와 다릅니다. 동일한 포지션 선수들의 궤적을 비교하려면 수비 포지션을 데이터베이스에 기록해야 합니다. 앞서 생성한 batting 데이터 프레임을 기억하십시오. 수비 데이터는 Lahman 패키지의 Fielding 데이터 프레임에 저장되어 있습니다.

야구 역사의 많은 선수들은 짧은 경력을 가졌으며 궤적 연구에서 최소 타수 이상을 기록한 선수로 분석을 제한하는 것이 합리적입니다. 2000타수 이상인 선수만 고려합니다. 이것은 투수 및 기타 짧은 경력을 가진 선수들의 타격 데이터를 제거합니다. batting 데이터 프레임의 이 하위 집합을 취하기 위해 dplyr 패키지의 group_by() 및 summarize() 함수를 사용하여 모든 선수의 통산 타수를 계산합니다. 새로운 변수는 AB_career라고 합니다. inner_join() 함수를 사용하여 이 새로운 변수를 batting 데이터 프레임에 추가합니다. 마지막으로 filter() 함수를 사용하여 "최소 2000타수" 타자로만 구성된 새로운 데이터 프레임 batting_2000을 생성합니다.

batting_2000 <- batting |> group_by(playerID) |> summarize(AB_career = sum(AB, na.rm = TRUE)) |> inner_join(batting, by = "playerID") |> filter(AB_career >= 2000)

데이터 프레임에 수비 정보를 추가하려면 주어진 선수의 주 수비 포지션을 찾아야 합니다. 각 가능한 포지션에서 플레이한 경기 수를 집계하고, Positions 데이터 프레임은 각 선수에 대해 가장 많은 경기를 플레이한 포지션을 반환합니다.1

Positions <- Fielding |> group_by(playerID, POS) |> summarize(Games = sum(G)) |> arrange(playerID, desc(Games)) |> filter(POS == first(POS))

그런 다음 inner_join() 함수를 사용하여 이 새로운 수비 정보를 batting_2000 데이터 프레임과 결합합니다.

batting_2000 <- batting_2000 |> inner_join(Positions, by = "playerID")

- 8.3.2 경력 통계 계산

경력 통계를 바탕으로 유사한 타자 그룹을 찾습니다. 이 목표를 향해 batting_2000 데이터 프레임의 각 선수에 대한 통산 출장 경기, 타수, 득점, 안타 등을 계산해야 합니다. 이것은 group_by() 및 summarize() 함수를 사용하여 편리하게 수행됩니다. R 코드에서는 각 타자에 대해 vars 벡터에 정의된 다양한 타격 통계 모음의 합을 찾기 위해 across() 함수를 사용합니다. 선수 ID 변수 playerID와 새로운 경력 변수를 가진 새로운 데이터 프레임 C_totals를 생성합니다.

my_vars <- c("G", "AB", "R", "H", "X2B", "X3B", "HR", "RBI", "BB", "SO", "SB")

1드물게 가장 많은 경기를 뛰는 포지션이 두 개 이상인 경우, first() 함수는 첫 번째 포지션을 취합니다.

모든 선수들에 대하여 새로운 변수는 AB_career라고 합니다. inner_join() 함수를 사용하여 이 새로운 변수를 batting 데이터 프레임에 추가합니다. 마지막으로 filter() 함수를 사용하여 "최소 2000타수" 타자로만 구성된 새로운 데이터 프레임 batting_2000을 생성합니다.

batting_2000 <- batting |> group_by(playerID) |> summarize(AB_career = sum(AB, na.rm = TRUE)) |> inner_join(batting, by = "playerID") |> filter(AB_career >= 2000)

데이터 프레임에 수비 정보를 추가하려면 주어진 선수의 주 수비 포지션을 찾아야 합니다. 각 가능한 포지션에서 플레이한 경기 수를 집계하고, Positions 데이터 프레임은 각 선수에 대해 가장 많은 경기를 플레이한 포지션을 반환합니다.1

Positions <- Fielding |> group_by(playerID, POS) |> summarize(Games = sum(G)) |> arrange(playerID, desc(Games)) |> filter(POS == first(POS))

그런 다음 inner_join() 함수를 사용하여 이 새로운 수비 정보를 batting_2000 데이터 프레임과 결합합니다.

batting_2000 <- batting_2000 |> inner_join(Positions, by = "playerID")

- 8.3.2 경력 통계 계산

경력 통계를 바탕으로 유사한 타자 그룹을 찾습니다. 이 목표를 향해 batting_2000 데이터 프레임의 각 선수에 대한 통산 출장 경기, 타수, 득점, 안타 등을 계산해야 합니다. 이것은 group_by() 및 summarize() 함수를 사용하여 편리하게 수행됩니다. R 코드에서는 각 타자에 대해 vars 벡터에 정의된 다양한 타격 통계 모음의 합을 찾기 위해 across() 함수를 사용합니다. 선수 ID 변수 playerID와 새로운 경력 변수를 가진 새로운 데이터 프레임 C_totals를 생성합니다.

my_vars <- c("G", "AB", "R", "H", "X2B", "X3B", "HR", "RBI", "BB", "SO", "SB")

1드물게 가장 많은 경기를 뛰는 포지션이 두 개 이상인 경우, first() 함수는 첫 번째 포지션을 취합니다.

C_totals <- batting |> group_by(playerID) |> summarize(across(all_of(my_vars), ~sum(.x, na.rm = TRUE)))

새로운 데이터 프레임에서 mutate()를 사용하여 각 선수의 경력 타율 AVG와 경력 장타율 SLG를 계산합니다.

C_totals <- C_totals |>

mutate( AVG = H / AB, SLG = (H - X2B - X3B - HR + 2 _ X2B + 3 _ X3B + 4 \* HR) / AB

)

그런 다음 경력 통계 데이터 프레임 C_totals를 수비 데이터 프레임 Positions와 병합합니다. 각 수비 포지션에는 연관된 값이 있으며, case_when() 함수를 사용하여 각 포지션 POS에 대한 값 Value_POS를 정의합니다. 이러한 값은 빌 제임스(Bill James)가 James (1994)에서 소개했으며 Baseball-Reference의 유사도 점수(Similarity Scores) 페이지에 표시됩니다.

C_totals <- C_totals |> inner_join(Positions, by = "playerID") |> mutate(

Value_POS = case_when( POS == "C" ~ 240, POS == "SS" ~ 168, POS == "2B" ~ 132, POS == "3B" ~ 84, POS == "OF" ~ 48, POS == "1B" ~ 12, TRUE ~ 0

) )

8.3.3 유사도 점수 계산

빌 제임스는 경력 통계를 바탕으로 선수들의 비교를 용이하게 하기 위해 유사도 점수 개념을 도입했습니다. 두 타자를 비교하기 위해 1000점에서 시작하여 서로 다른 통계 범주의 차이에 따라 점수를 뺍니다. 다음 차이마다 1점이 차감됩니다. (1) 출장 경기 수 20경기, (2) 타수 75타수, (3) 득점 10득점, (4) 안타 15개, (5) 2루타 5개, (6) 3루타 4개, (7) 홈런 2개, (8) 타점 10타점, (9) 볼넷 25개, (10) 삼진 150개, (11) 도루 20개, (12) 타율 0.001, (13) 장타율 0.002. 또한, 두 선수의 수비 포지션 값의 차이의 절댓값을 뺍니다.

similar() 함수는 경력 통계와 수비 포지션에 대한 유사도 점수를 사용하여 주어진 선수와 가장 유사한 선수를 찾습니다. 특정 선수의 ID와 찾을 유사한 선수 수(주어진 선수 포함)를 입력합니다. 출력은 유사도 점수 내림차순으로 정렬된 선수 통계 데이터 프레임입니다.

similar <- function(p, number = 10) { P <- C_totals |>

filter(playerID == p) C_totals |> mutate(

sim_score = 1000 -

- floor(abs(G - P$G) / 20) floor(abs(AB - P$AB) / 75) floor(abs(R - P$R) / 10) -
- floor(abs(H - P$H) / 15) floor(abs(X2B - P$X2B) / 5) floor(abs(X3B - P$X3B) / 4) floor(abs(HR - P$HR) / 2) floor(abs(RBI - P$RBI) / 10) floor(abs(BB - P$BB) / 25) floor(abs(SO - P$SO) / 150) floor(abs(SB - P$SB) / 20) floor(abs(AVG - P$AVG) / 0.001) floor(abs(SLG - P$SLG) / 0.002) abs(Value_POS - P$Value_POS)

) |> arrange(desc(sim_score)) |> slice_head(n = number)

}

이 함수의 사용법을 설명하기 위해, 미키 맨틀과 가장 유사한 다섯 명의 선수를 찾는 데 관심이 있다고 가정해 보겠습니다. 맨틀의 선수 ID는 mantle_id 벡터에 저장되어 있음을 기억하십시오. mantle_id와 6을 입력으로 사용하여 similar() 함수를 사용합니다.

similar(mantle_id, 6)

# A tibble: 6 x 18

playerID G AB R H X2B X3B HR RBI BB <chr> <int> <int> <int> <int> <int> <int> <int> <int> <int>

- 1 mantlmi~ 2401 8102 1677 2415 344 72 536 1509 1733
- 2 thomafr~ 2322 8199 1494 2468 495 12 521 1704 1667
- 3 matheed~ 2391 8537 1509 2315 354 72 512 1453 1444
- 4 schmimi~ 2404 8352 1506 2234 408 59 548 1595 1507
- 5 sheffga~ 2576 9217 1636 2689 467 27 509 1676 1475

similar() 함수는 경력 통계와 수비 포지션에 대한 유사도 점수를 사용하여 주어진 선수와 가장 유사한 선수를 찾습니다. 특정 선수의 ID와 찾을 유사한 선수 수(주어진 선수 포함)를 입력합니다. 출력은 유사도 점수 내림차순으로 정렬된 선수 통계 데이터 프레임입니다.

similar <- function(p, number = 10) { P <- C_totals |>

filter(playerID == p) C_totals |> mutate(

sim_score = 1000 -

- floor(abs(G - P$G) / 20) floor(abs(AB - P$AB) / 75) floor(abs(R - P$R) / 10) -
- floor(abs(H - P$H) / 15) floor(abs(X2B - P$X2B) / 5) floor(abs(X3B - P$X3B) / 4) floor(abs(HR - P$HR) / 2) floor(abs(RBI - P$RBI) / 10) floor(abs(BB - P$BB) / 25) floor(abs(SO - P$SO) / 150) floor(abs(SB - P$SB) / 20) floor(abs(AVG - P$AVG) / 0.001) floor(abs(SLG - P$SLG) / 0.002) abs(Value_POS - P$Value_POS)

) |> arrange(desc(sim_score)) |> slice_head(n = number)

}

이 함수의 사용법을 설명하기 위해, 미키 맨틀과 가장 유사한 다섯 명의 선수를 찾는 데 관심이 있다고 가정해 보겠습니다. 맨틀의 선수 ID는 mantle_id 벡터에 저장되어 있음을 기억하십시오. mantle_id와 6을 입력으로 사용하여 similar() 함수를 사용합니다.

similar(mantle_id, 6)

# A tibble: 6 x 18

playerID G AB R H X2B X3B HR RBI BB <chr> <int> <int> <int> <int> <int> <int> <int> <int> <int>

1 mantlmi~ 2401 8102 1677 2415 344 72 536 1509 1733 2 thomafr~ 2322 8199 1494 2468 495 12 521 1704 1667 3 matheed~ 2391 8537 1509 2315 354 72 512 1453 1444 4 schmimi~ 2404 8352 1506 2234 408 59 548 1595 1507 5 sheffga~ 2576 9217 1636 2689 467 27 509 1676 1475

6 sosasa01 2354 8813 1475 2408 379 45 609 1667 929 # i 8 more variables: SO <int>, SB <int>, AVG <dbl>, SLG <dbl>, # POS <chr>, Games <int>, Value_POS <dbl>, sim_score <dbl>

선수 ID를 읽어보면 경력 타격 통계와 포지션 측면에서 유사한 5명의 선수, 즉 프랭크 토마스(Frank Thomas), 에디 매슈스(Eddie Mathews), 마이크 슈미트(Mike Schmidt), 게리 셰필드(Gary Sheffield), 새미 소사(Sammy Sosa)를 볼 수 있습니다.

- 8.3.4 나이, OBP, SLG 및 OPS 변수 정의

유사한 타자 그룹에 대한 타격 궤적을 맞추고 그래프로 그리려면, 각 선수에 대한 모든 시즌의 나이 및 OPS 통계가 필요합니다. Lahman Batting 테이블 작업 시 한 가지 복잡한 점은 한 시즌 동안 여러 팀에서 뛴 타자의 경우 별도의 타격 줄이 사용된다는 것입니다. 여러 팀에서 뛴 선수의 경우 다른 값(1, 2, ...)을 제공하는 stint 변수가 있습니다. 다음 코드는 group_by() 및 summarize() 함수를 사용하여 여러 줄을 각 선수의 각 연도에 대한 단일 행으로 결합합니다. 또한 타격 지표 SLG, OBP 및 OPS를 계산합니다. (앞서 HBP 및 SF의 결측값을 0으로 대체했으므로 OBP 및 OPS 변수 계산 시 결측값이 없습니다.)

batting_2000 <- batting_2000 |> group_by(playerID, yearID) |> summarize(

- G = sum(G), AB = sum(AB), R = sum(R),
- H = sum(H), X2B = sum(X2B), X3B = sum(X3B), HR = sum(HR), RBI = sum(RBI), SB = sum(SB), CS = sum(CS), BB = sum(BB), SH = sum(SH), SF = sum(SF), HBP = sum(HBP), AB_career = first(AB_career), POS = first(POS)

) |> mutate(

SLG = (H - X2B - X3B - HR + 2 _ X2B + 3 _ X3B + 4 \* HR) / AB, OBP = (H + BB + HBP) / (AB + BB + HBP + SF), OPS = SLG + OBP

)

따라서 시즌 동안 한 선수의 타격 통계가 한 줄에 기록된 batting_2000 데이터 프레임의 새 버전을 생성합니다.

다음 작업은 모든 시즌에 대한 모든 선수의 나이를 얻는 것입니다. 3.7절에서 특정 선수의 MLB 출생 연도를 계산하기 위해 유사한 기법을 사용했음을 기억하십시오. 여기서 우리는 모든 선수의 출생 연도를 계산하고 inner_join() 함수를 사용하여 이 출생 연도 정보를 타격 데이터와 병합합니다. 이제 모든 선수의 출생 연도가 있으므로 새 변수 Age를 시즌 연도와 출생 연도의 차이로 정의할 수 있습니다.

batting_2000 <- batting_2000 |> inner_join(People, by = "playerID") |> mutate(

Birthyear = if_else( birthMonth >= 7, birthYear + 1, birthYear

), Age = yearID - Birthyear

)

작은 문제점은 19세기의 몇몇 야구 선수들에 대해서는 출생 연도가 기록되지 않아서 이 변수들에 대한 나이 변수가 누락되어 있다는 것입니다. 이에 따라 drop_na() 함수를 사용하여 누락된 나이 기록을 생략하고, 업데이트된 데이터 프레임 batting_2000에는 Age 변수를 사용할 수 있는 선수들만 포함됩니다.

batting_2000 |> drop_na(Age) -> batting_2000

- 8.3.5 궤적 적합 및 도식화

유사한 선수 그룹이 주어졌을 때, 각 선수에게 이차 곡선을 맞추고 비교를 용이하게 하는 방식으로 궤적을 그래프로 그리는 plot_trajectories() 함수를 작성합니다. 이 함수는 선수의 이름과 성, 비교할 선수 수(관심 있는 선수 포함), 그리고 다중 패널 플롯의 열 수를 입력으로 사용합니다.

plot_trajectories() 함수는 먼저 People 데이터 프레임을 사용하여 해당 선수의 선수 ID를 찾습니다. 그런 다음 similar() 함수를 사용하여 선수 ID의 벡터인 player_list를 찾습니다. Batting_new 데이터 프레임은 선수 목록에 있는 선수들만의 시즌 타격 통계로 구성됩니다. 그래프 그리기는 ggplot2 패키지를 사용하여 수행됩니다. 공식 인수 y ∼ x + I(x2)와 함께 geom_smooth()를 사용하여 모든 선수에 대한 나이 및 적합 궤적 곡선을 구성합니다. ncol 인수가 있는 facet_wrap() 함수는 이러한 궤적을 별도의 패널에 배치하며, 여기서 다중 패널 표시의 열 수는 함수의 인수에 지정된 값입니다.

plot_trajectories <- function(player, n_similar = 5, ncol) { flnames <- unlist(str_split(player, " ")) player <- People |>

filter(nameFirst == flnames[1], nameLast == flnames[2]) |> select(playerID)

player_list <- player |> pull(playerID) |>

batting_2000 <- batting_2000 |> inner_join(People, by = "playerID") |> mutate(

Birthyear = if_else( birthMonth >= 7, birthYear + 1, birthYear

), Age = yearID - Birthyear

)

작은 문제점은 19세기의 몇몇 야구 선수들에 대해서는 출생 연도가 기록되지 않아서 이 변수들에 대한 나이 변수가 누락되어 있다는 것입니다. 이에 따라 drop_na() 함수를 사용하여 누락된 나이 기록을 생략하고, 업데이트된 데이터 프레임 batting_2000에는 Age 변수를 사용할 수 있는 선수들만 포함됩니다.

batting_2000 |> drop_na(Age) -> batting_2000

- 8.3.5 궤적 적합 및 도식화

유사한 선수 그룹이 주어졌을 때, 각 선수에게 이차 곡선을 맞추고 비교를 용이하게 하는 방식으로 궤적을 그래프로 그리는 plot_trajectories() 함수를 작성합니다. 이 함수는 선수의 이름과 성, 비교할 선수 수(관심 있는 선수 포함), 그리고 다중 패널 플롯의 열 수를 입력으로 사용합니다.

plot_trajectories() 함수는 먼저 People 데이터 프레임을 사용하여 해당 선수의 선수 ID를 찾습니다. 그런 다음 similar() 함수를 사용하여 선수 ID의 벡터인 player_list를 찾습니다. Batting_new 데이터 프레임은 선수 목록에 있는 선수들만의 시즌 타격 통계로 구성됩니다. 그래프 그리기는 ggplot2 패키지를 사용하여 수행됩니다. 공식 인수 y ∼ x + I(x2)와 함께 geom_smooth()를 사용하여 모든 선수에 대한 나이 및 적합 궤적 곡선을 구성합니다. ncol 인수가 있는 facet_wrap() 함수는 이러한 궤적을 별도의 패널에 배치하며, 여기서 다중 패널 표시의 열 수는 함수의 인수에 지정된 값입니다.

plot_trajectories <- function(player, n_similar = 5, ncol) { flnames <- unlist(str_split(player, " ")) player <- People |>

filter(nameFirst == flnames[1], nameLast == flnames[2]) |> select(playerID)

player_list <- player |> pull(playerID) |>

![image 68](images/imageFile68.png)

- 그림 8.3 미키 맨틀과 5명의 유사한 타자들에 대한 추정된 경력 궤적.

similar(n_similar) |> pull(playerID)

Batting_new <- batting_2000 |> filter(playerID %in% player_list) |> mutate(Name = paste(nameFirst, nameLast))

}

ggplot(Batting_new, aes(Age, OPS)) +

geom_smooth( method = "lm", formula = y ~ x + I(x^2), linewidth = 1.5

) + facet_wrap(vars(Name), ncol = ncol) + theme_bw()

여기 plot_trajectories() 사용의 몇 가지 예시가 있습니다. 그림 8.3에서 우리는 미키 맨틀의 궤적을 그와 가장 유사한 5명의 타자들의 궤적과 비교합니다.

plot_trajectories("Mickey Mantle", 6, 2)

그림 8.4에서 우리는 데릭 지터(Derek Jeter)의 OPS 궤적을 8명의 유사한 선수들과 비교합니다. 이 경우 ggplot2 객체가 변수에 저장된다는 점에 유의하십시오.

![image 69](images/imageFile69.png)

- 그림 8.4 데릭 지터와 8명의 유사한 타자들에 대한 추정된 경력 궤적.

dj_plot. (이 객체에서 관련 데이터를 추출할 수 있다는 것을 곧 알게 될 것입니다.)

dj_plot <- plot_trajectories("Derek Jeter", 9, 3) dj_plot

그림 8.3과 8.4를 보면, 우리는 이러한 궤적들에서 눈에 띄는 차이점들을 볼 수 있습니다.

- • 에디 매슈스, 프랭크 토마스, 미키 맨틀, 로베르토 알로마(Roberto Alomar)와 같이 경력 초기에 정점에 도달한 것으로 보이는 선수들이 있습니다.
- • 반면에, 마이크 슈미트, 크레이그 비지오(Craig Biggio), 훌리오 프랑코(Julio Franco)와 같은 선수들은 30대에 정점에 도달했습니다.
- • 선수들은 또한 궤적의 형태에서 차이를 보여줍니다. 예를 들어 폴 몰리터(Paul Molitor)는 비교적 평탄한 궤적을 가졌던 반면 로베르토 알로마는 곡률이 높은 궤적을 가졌습니다.

이러한 궤적들은 전성기 나이, 최대값, 그리고 곡률로 요약할 수 있습니다. 먼저, dj_plot$data 구성요소에는 지터와 유사한 선수 그룹의 타격 데이터가 포함되어 있습니다. 데이터를 각 선수에 대한 하나의 요소가 있는 목록으로 분할하기 위해 group_split() 함수를 사용합니다. 그런 다음 map()을 사용하여 각 선수에게 이차 모델을 맞춥니다. broom 패키지의 tidy() 함수는 계수를 깔끔하게 복구하는 데 도움을 줍니다.

출력 데이터 프레임 regressions에는 각 선수에 대한 회귀 추정치가 포함되어 있습니다.

library(broom) data_grouped <- dj_plot$data |>

group_by(Name)

player_names <- data_grouped |> group_keys() |> pull(Name)

regressions <- data_grouped |> group_split() |> map(~lm(OPS ~ I(Age - 30) + I((Age - 30) ^ 2), data = .)) |> map(tidy) |> set_names(player_names) |> bind_rows(.id = "Name")

regressions |> slice_head(n = 6)

그림 8.4 데릭 지터와 8명의 유사한 타자들에 대한 추정된 경력 궤적.

dj_plot. (이 객체에서 관련 데이터를 추출할 수 있다는 것을 곧 알게 될 것입니다.)

dj_plot <- plot_trajectories("Derek Jeter", 9, 3) dj_plot

그림 8.3과 8.4를 보면, 우리는 이러한 궤적들에서 눈에 띄는 차이점들을 볼 수 있습니다.

- • 에디 매슈스, 프랭크 토마스, 미키 맨틀, 로베르토 알로마(Roberto Alomar)와 같이 경력 초기에 정점에 도달한 것으로 보이는 선수들이 있습니다.
- • 반면에, 마이크 슈미트, 크레이그 비지오(Craig Biggio), 훌리오 프랑코(Julio Franco)와 같은 선수들은 30대에 정점에 도달했습니다.
- • 선수들은 또한 궤적의 형태에서 차이를 보여줍니다. 예를 들어 폴 몰리터(Paul Molitor)는 비교적 평탄한 궤적을 가졌던 반면 로베르토 알로마는 곡률이 높은 궤적을 가졌습니다.

이러한 궤적들은 전성기 나이, 최대값, 그리고 곡률로 요약할 수 있습니다. 먼저, dj_plot$data 구성요소에는 지터와 유사한 선수 그룹의 타격 데이터가 포함되어 있습니다. 데이터를 각 선수에 대한 하나의 요소가 있는 목록으로 분할하기 위해 group_split() 함수를 사용합니다. 그런 다음 map()을 사용하여 각 선수에게 이차 모델을 맞춥니다. broom 패키지의 tidy() 함수는 계수를 깔끔하게 복구하는 데 도움을 줍니다.

출력 데이터 프레임 regressions에는 각 선수에 대한 회귀 추정치가 포함되어 있습니다.

# A tibble: 6 x 6

Name term estimate std.error statistic p.value <chr> <chr> <dbl> <dbl> <dbl> <dbl>

- 1 Cal Ripken (Inte~ 0.820 0.0436 18.8 2.74e-13
- 2 Cal Ripken I(Age~ 0.00273 0.00479 0.570 5.76e- 1
- 3 Cal Ripken I((Ag~ -0.00148 0.000887 -1.67 1.12e- 1
- 4 Charlie Gehringer (Inte~ 0.932 0.0415 22.4 1.60e-13
- 5 Charlie Gehringer I(Age~ 0.00507 0.00504 1.00 3.30e- 1
- 6 Charlie Gehringer I((Ag~ -0.00285 0.00103 -2.76 1.40e- 2

다음으로, regressions와 함께 summarize() 함수를 사용하여 전성기 나이, 최대값, 곡률을 포함한 모든 선수의 요약 통계를 찾습니다. 이 계산은 지터와 8명의 유사한 선수에 대해 설명되었음을 기억하십시오.

S <- regressions |> group_by(Name) |> summarize(

- b1 = estimate[1],
- b2 = estimate[2], Curvature = estimate[3], Age_max = round(30 - b2 / Curvature / 2, 1), Max = round(b1 - b2 ^ 2 / Curvature / 4, 3)

)

9명의 선수 궤적 간의 차이를 이해하는 데 도움을 주기 위해, ggplot() 함수를 사용하여 전성기 나이와 곡률 통계의 산점도를 구성합니다. geom_label_repel() 함수를 사용하여 선수 레이블을 추가합니다.

![image 70](images/imageFile70.png)

그림 8.5 데릭 지터와 그와 가장 유사한 8명의 선수에 대한 추정된 전성기 나이 및 곡률 통계.

library(ggrepel) ggplot(S, aes(Age_max, Curvature, label = Name)) +

geom_point() + geom_label_repel()

그림 8.5는 알로마가 이른 나이에 정점에 도달했고, 프랑코와 몰리터는 늦은 나이에 정점에 도달했으며, 알로마와 게링거(Gehringer)는 곡률이 가장 크게 나타나 정점 이후 성과가 빠르게 감소했음을 분명히 나타냅니다.

### 8.4 전성기 나이의 일반적인 패턴

8.4.1 모든 적합된 궤적 계산

우리는 유사한 선수 그룹의 타격 경력 궤적을 탐구했습니다. 야구 역사를 통틀어 경력 궤적은 어떻게 변해왔을까요? 우리는 선수의 전성기 나이에 초점을 맞추고 이것이 시간에 따라 어떻게 변했는지 탐구할 것입니다. 또한 전성기 나이와 통산 타수 간의 관계를 탐구할 것입니다.

우리는 더 이상 활동하지 않는 선수들에게 초점을 맞추고 싶으므로, People 데이터 프레임의 finalgame 변수를 사용하여 2021년 11월 1일 이전에 마지막 경기를 치른 선수들로 관심을 제한합니다.

not_current_playerID <- People |> filter(finalGame < "2021-11-01") |> pull(playerID)

batting_2000 <- batting_2000 |> filter(playerID %in% not_current_playerID)

그림 8.5 데릭 지터와 그와 가장 유사한 8명의 선수에 대한 추정된 전성기 나이 및 곡률 통계.

library(ggrepel) ggplot(S, aes(Age_max, Curvature, label = Name)) +

geom_point() + geom_label_repel()

그림 8.5는 알로마가 이른 나이에 정점에 도달했고, 프랑코와 몰리터는 늦은 나이에 정점에 도달했으며, 알로마와 게링거(Gehringer)는 곡률이 가장 크게 나타나 정점 이후 성과가 빠르게 감소했음을 분명히 나타냅니다.

### 8.4 전성기 나이의 일반적인 패턴

- 8.4.1 모든 적합된 궤적 계산

우리는 유사한 선수 그룹의 타격 경력 궤적을 탐구했습니다. 야구 역사를 통틀어 경력 궤적은 어떻게 변해왔을까요? 우리는 선수의 전성기 나이에 초점을 맞추고 이것이 시간에 따라 어떻게 변했는지 탐구할 것입니다. 또한 전성기 나이와 통산 타수 간의 관계를 탐구할 것입니다.

우리는 더 이상 활동하지 않는 선수들에게 초점을 맞추고 싶으므로, People 데이터 프레임의 finalgame 변수를 사용하여 2021년 11월 1일 이전에 마지막 경기를 치른 선수들로 관심을 제한합니다.

각 선수에 대해, yearID 변수에는 플레이한 시즌이 포함됩니다. 새로운 변수 Midyear를 선수의 첫 시즌과 마지막 시즌의 평균으로 정의합니다. group_by() 및 summarize() 함수를 사용하여 모든 선수의 Midyear를 계산하고, inner_join() 함수를 사용하여 이 새로운 변수를 batting_2000 데이터 프레임에 추가합니다.

midcareers <- batting_2000 |> group_by(playerID) |> summarize(

Midyear = (min(yearID) + max(yearID)) / 2, AB_total = first(AB_career)

) batting_2000 <- batting_2000 |> inner_join(midcareers, by = "playerID")

purrr 패키지의 map() 함수를 또 한 번 적용하여 모든 경력 궤적에 대한 이차 곡선을 맞춥니다. 먼저 group_split()을 적용하는데, 여기서 playerID는 그룹화 변수이고 모델 적합은 각 선수의 데이터에 개별적으로 맞추어집니다. 출력 models는 모든 선수의 계수를 포함하는 데이터 프레임이며, 한 행은 특정 선수에 해당합니다.

batting_2000_grouped <- batting_2000 |> group_by(playerID)

ids <- batting_2000_grouped |> group_keys() |> pull(playerID)

models <- batting_2000_grouped |> group_split() |> map(~lm(OPS ~ I(Age - 30) + I((Age - 30)^2), data = .)) |> map(tidy) |> set_names(ids) |> bind_rows(.id = "playerID")

공식 Peak age = 30 − B/(2C)를 사용하여 모든 선수에 대한 추정된 전성기 나이를 계산합니다. 새로운 변수 Peak_age를 beta_coefs 데이터 프레임에 추가합니다.

![image 71](images/imageFile71.png)

그림 8.6

최소 2000타수 이상을 기록한 모든 선수들의 전성기 나이와 중간 경력의 산점도. 평활 곡선을 사용하여 산점도에서 일반적인 패턴을 봅니다.

beta_coefs <- models |> group_by(playerID) |> summarize(

A = estimate[1], B = estimate[2], C = estimate[3]

) |> mutate(Peak_age = 30 - B / 2 / C) |> inner_join(midcareers, by = "playerID")

8.4.2 시간에 따른 전성기 나이 패턴

야구 역사상 전성기 나이가 어떻게 변하는지 조사하기 위해, ggplot() 함수를 사용하여 Midyear에 대한 Peak_age의 산점도를 구성합니다. 산점도만 봐서는 일반적인 패턴을 파악하기 어려우므로 geom_smooth() 함수를 사용하여 평활 곡선을 맞추고 이를 플롯에 추가합니다(그림 8.6 참조).

age_plot <- ggplot(beta_coefs, aes(Midyear, Peak_age)) + geom_point(alpha = 0.5) + geom_smooth(color = "red", method = "loess") + ylim(20, 40) +

그림 8.6

최소 2000타수 이상을 기록한 모든 선수들의 전성기 나이와 중간 경력의 산점도. 평활 곡선을 사용하여 산점도에서 일반적인 패턴을 봅니다.

![image 72](images/imageFile72.png)

그림 8.7

최소 2000타수 이상을 기록한 모든 선수들의 통산 타수 로그와 전성기 나이의 산점도. 평활 곡선을 사용하여 산점도에서 일반적인 패턴을 봅니다.

beta_coefs <- models |> group_by(playerID) |> summarize(

A = estimate[1], B = estimate[2], C = estimate[3]

) |> mutate(Peak_age = 30 - B / 2 / C) |> inner_join(midcareers, by = "playerID")

- 8.4.2 시간에 따른 전성기 나이 패턴

야구 역사상 전성기 나이가 어떻게 변하는지 조사하기 위해, ggplot() 함수를 사용하여 Midyear에 대한 Peak_age의 산점도를 구성합니다. 산점도만 봐서는 일반적인 패턴을 파악하기 어려우므로 geom_smooth() 함수를 사용하여 평활 곡선을 맞추고 이를 플롯에 추가합니다(그림 8.6 참조).

age_plot <- ggplot(beta_coefs, aes(Midyear, Peak_age)) + geom_point(alpha = 0.5) + geom_smooth(color = "red", method = "loess") + ylim(20, 40) +

xlab("Mid Career") + ylab("Peak Age") age_plot

그림 8.6에서, 시간에 따른 전성기 나이의 점진적인 증가를 봅니다. 평균 선수의 전성기 나이는 1880년에 약 27세였으며 이 평균은 1880년에서 2016년까지 28세로 점진적으로 증가했습니다.

8.4.3 전성기 나이와 통산 타수

선수의 전성기 나이와 통산 타수 사이에 어떤 관계가 있을까요? ggplot2를 사용하여, 통산 타수 변수 AB_career의 로그(밑이 2)에 대한 Peak_age의 그래프를 구성합니다. 타수를 로그 척도로 플로팅하여 모든 가능한 값에 대해 점들이 더 고르게 퍼지도록 합니다. 다시 LOESS 평활 곡선을 겹쳐 그려 그림 8.7의 패턴을 살펴봅니다.

age_plot +

aes(x = log2(AB_total)) + xlab("Log2 of Career AB")

여기서 명확한 관계를 봅니다. 경력이 비교적 짧고 2000 통산 타수인 선수들은 약 27세에 정점에 도달하는 경향이 있습니다. 대조적으로 긴 경력(예를 들어 9000타수 이상)을 가진 선수들은 30세에 더 가까운 나이에 정점에 도달하는 경향이 있습니다.

8.5 궤적 및 수비 포지션

선수들을 비교할 때, 우리는 일반적으로 동일한 수비 포지션에 있는 선수들을 비교하고자 합니다. 주 수비 포지션 POS는 이미 정의되어 있으며 우리는 이 변수를 사용하여 포지션별로 분류된 선수들의 전성기 나이를 비교합니다.

중간 경력이 1985년에서 1995년 사이인 선수들을 고려한다고 가정해 보겠습니다. filter() 함수를 사용하여 이러한 선수들로만 구성된 새로운 데이터 프레임 Batting_2000a를 생성합니다.

batting_2000a <- batting_2000 |> filter(Midyear >= 1985, Midyear <= 1995)

map() 함수의 또 다른 적용은 batting_2000a 데이터 프레임에 있는 선수들의 궤적 데이터에 이차 곡선을 맞추는 데 사용되며 이차 적합은 객체 models에 저장됩니다. summarize() 및 mutate() 함수를 사용하여 회귀 적합을 요약합니다. 출력은 추정된 계수 A, B, C, 선수의 추정된 전성기 나이 Peak_age 및 수비 포지션 Position입니다. 이 정보는 데이터 프레임 beta_estimates에 저장됩니다.

batting_2000a_grouped <- batting_2000a |> group_by(playerID)

ids <- batting_2000a_grouped |> group_keys() |> pull(playerID)

models <- batting_2000a_grouped |> group_split() |> map(~lm(OPS ~ I(Age - 30) + I((Age - 30)^2), data = .)) |> map(tidy) |> set_names(ids) |> bind_rows(.id = "playerID")

beta_estimates <- models |> group_by(playerID) |> summarize(

- A = estimate[1],
- B = estimate[2],
- C = estimate[3]

) |> mutate(Peak_age = 30 - B / 2 / C) |> inner_join(midcareers) |> inner_join(Positions) |> rename(Position = POS)

- 8.5 궤적 및 수비 포지션

선수들을 비교할 때, 우리는 일반적으로 동일한 수비 포지션에 있는 선수들을 비교하고자 합니다. 주 수비 포지션 POS는 이미 정의되어 있으며 우리는 이 변수를 사용하여 포지션별로 분류된 선수들의 전성기 나이를 비교합니다.

중간 경력이 1985년에서 1995년 사이인 선수들을 고려한다고 가정해 보겠습니다. filter() 함수를 사용하여 이러한 선수들로만 구성된 새로운 데이터 프레임 Batting_2000a를 생성합니다.

batting_2000a <- batting_2000 |> filter(Midyear >= 1985, Midyear <= 1995)

map() 함수의 또 다른 적용은 batting_2000a 데이터 프레임에 있는 선수들의 궤적 데이터에 이차 곡선을 맞추는 데 사용되며 이차 적합은 객체 models에 저장됩니다. summarize() 및 mutate() 함수를 사용하여 회귀 적합을 요약합니다. 출력은 추정된 계수 A, B, C, 선수의 추정된 전성기 나이 Peak_age 및 수비 포지션 Position입니다. 이 정보는 데이터 프레임 beta_estimates에 저장됩니다.

batting_2000a_grouped <- batting_2000a |> group_by(playerID)

ids <- batting_2000a_grouped |> group_keys() |> pull(playerID)

models <- batting_2000a_grouped |> group_split() |> map(~lm(OPS ~ I(Age - 30) + I((Age - 30)^2), data = .)) |> map(tidy) |> set_names(ids) |> bind_rows(.id = "playerID")

beta_estimates <- models |> group_by(playerID) |> summarize(

A = estimate[1], B = estimate[2], C = estimate[3]

) |> mutate(Peak_age = 30 - B / 2 / C) |> inner_join(midcareers) |> inner_join(Positions) |> rename(Position = POS)

궤적 및 수비 포지션 201

![image 73](images/imageFile73.png)

그림 8.8 주 수비 포지션에 대한 전성기 나이 산점도.

우리는 투수와 지명타자를 제외한 주 수비 포지션에 초점을 맞춥니다. filter() 함수는 이러한 다른 포지션들을 제거합니다. inner_join() 함수를 사용하여 궤적 및 수비 정보를 People 정보와 결합하고 결합된 정보를 데이터 프레임 beta_fielders에 저장합니다.

beta_fielders <- beta_estimates |> filter(

Position %in% c("1B", "2B", "3B", "SS", "C", "OF") ) |> inner_join(People)

수비 포지션에 대한 선수의 전성기 나이를 그래프로 그리기 위해 스트립차트(stripchart)를 사용합니다(그림 8.8 참조). 전성기 나이 추정치 중 일부는 합리적인 값이 아니기 때문에 수평축의 한계는 20과 40으로 설정됩니다.

ggplot(beta_fielders, aes(Position, Peak_age)) + geom_jitter(width = 0.2) + ylim(20, 40) + geom_label_repel(

data = filter(beta_fielders, Peak_age > 37), aes(Position, Peak_age, label = nameLast)

)

일반적으로 모든 수비 포지션에 대해 이 1990년 선수들의 전성기 나이는 27세에서 32세 사이인 경향이 있습니다. 전성기 나이 추정치의 변동성은

타자들의 경력 궤적 형태가 다르다는 사실을 반영합니다. 높은 전성기 나이 추정치를 보여 눈에 띄는 외야수는 세 명이며 포수는 없습니다. 37세 이후에 정점에 도달한 강조된 여섯 명의 선수는 안드레스 갈라라가(Andre´s Galarraga), 랜디 레디(Randy Ready), 에릭 데이비스(Eric Davis), 토니 필립스(Tony Phillips), 짐 아이젠라이크(Jim Eisenreich), 알바로 에스피노자(Alvaro Espinoza)입니다. 독자들은 이러한 "독특한" 선수들의 궤적을 탐구하여 그들이 실제로 독특한 경력 성과 패턴을 가지고 있는지 확인해보시기 바랍니다.

- 8.6 추가 읽을거리

제임스(James) (1982)는 "전성기 찾기(Looking for the Prime)"라는 에세이를 썼습니다. 통계 연구를 바탕으로 그는 타자들이 27세에 정점에 도달하는 경향이 있다는 결론에 도달했습니다. 베리(Berry), 리스(Reese), 라키(Larkey) (1999)는 하키, 야구, 골프 선수의 경력 궤적에 대한 일반적인 논의를 제공합니다. 앨버트(Albert)와 베넷(Bennett) (2003)의 11장에서는 아홉 명의 위대한 역사적 강타자들의 홈런 비율 경력 궤적을 고려합니다. 앨버트(2002)와 앨버트(2009)는 야구 역사상 타자와 투수의 궤적의 일반적인 패턴을 논의하며, 페어(Fair) (2008)는 이차 모델을 기반으로 야구 경력 궤적에 대한 광범위한 분석을 수행합니다. 앨버트와 리조(Rizzo) (2012)의 7장에서는 R을 사용한 회귀 모델링의 예시를 제공합니다.

- 8.7 연습문제

- 1. 윌리 메이스(Willie Mays)의 경력 궤적

- a. gets_stats() 함수를 사용하여 윌리 메이스의 경력 전체 시즌에 대한 타격 데이터를 추출하십시오.
- b. 윌리 메이스의 나이에 대한 시즌 OPS 값의 산점도를 구성하십시오.
- c. 윌리 메이스의 경력 궤적에 이차 함수를 맞추십시오. 이 모델을 기반으로 메이스의 전성기 나이와 적합을 기반으로 추정된 그의 가장 큰 OPS 값을 추정하십시오.

- 2. 궤적 비교

- a. 제임스의 유사도 점수 지표(similar() 함수)를 사용하여 타격 통계가 윌리 메이스와 가장 유사한 다섯 명의 타자를 찾으십시오.
- b. 메이스와 다섯 명의 유사한 타자에 대한 (Age, OPS) 데이터에 이차 함수를 맞추십시오. 단일 패널에 여섯 개의 적합된 궤적을 표시하십시오.
- c. 그래프를 바탕으로 여섯 명의 선수 궤적 간의 차이점을 설명하십시오. 어느 선수가 가장 이른 전성기 나이를 가졌습니까?

타자들의 경력 궤적 형태가 다르다는 사실을 반영합니다. 높은 전성기 나이 추정치를 보여 눈에 띄는 외야수는 세 명이며 포수는 없습니다. 37세 이후에 정점에 도달한 강조된 여섯 명의 선수는 안드레스 갈라라가(Andre´s Galarraga), 랜디 레디(Randy Ready), 에릭 데이비스(Eric Davis), 토니 필립스(Tony Phillips), 짐 아이젠라이크(Jim Eisenreich), 알바로 에스피노자(Alvaro Espinoza)입니다. 독자들은 이러한 "독특한" 선수들의 궤적을 탐구하여 그들이 실제로 독특한 경력 성과 패턴을 가지고 있는지 확인해보시기 바랍니다.

- 8.6 추가 읽을거리

제임스(James) (1982)는 "전성기 찾기(Looking for the Prime)"라는 에세이를 썼습니다. 통계 연구를 바탕으로 그는 타자들이 27세에 정점에 도달하는 경향이 있다는 결론에 도달했습니다. 베리(Berry), 리스(Reese), 라키(Larkey) (1999)는 하키, 야구, 골프 선수의 경력 궤적에 대한 일반적인 논의를 제공합니다. 앨버트(Albert)와 베넷(Bennett) (2003)의 11장에서는 아홉 명의 위대한 역사적 강타자들의 홈런 비율 경력 궤적을 고려합니다. 앨버트(2002)와 앨버트(2009)는 야구 역사상 타자와 투수의 궤적의 일반적인 패턴을 논의하며, 페어(Fair) (2008)는 이차 모델을 기반으로 야구 경력 궤적에 대한 광범위한 분석을 수행합니다. 앨버트와 리조(Rizzo) (2012)의 7장에서는 R을 사용한 회귀 모델링의 예시를 제공합니다.

- 8.7 연습문제

- 1. 윌리 메이스(Willie Mays)의 경력 궤적

- a. gets_stats() 함수를 사용하여 윌리 메이스의 경력 전체 시즌에 대한 타격 데이터를 추출하십시오.
- b. 윌리 메이스의 나이에 대한 시즌 OPS 값의 산점도를 구성하십시오.
- c. 윌리 메이스의 경력 궤적에 이차 함수를 맞추십시오. 이 모델을 기반으로 메이스의 전성기 나이와 적합을 기반으로 추정된 그의 가장 큰 OPS 값을 추정하십시오.

- 2. 궤적 비교

- a. 제임스의 유사도 점수 지표(similar() 함수)를 사용하여 타격 통계가 윌리 메이스와 가장 유사한 다섯 명의 타자를 찾으십시오.
- b. 메이스와 다섯 명의 유사한 타자에 대한 (Age, OPS) 데이터에 이차 함수를 맞추십시오. 단일 패널에 여섯 개의 적합된 궤적을 표시하십시오.
- c. 그래프를 바탕으로 여섯 명의 선수 궤적 간의 차이점을 설명하십시오. 어느 선수가 가장 이른 전성기 나이를 가졌습니까?

연습문제 203

- 3. 통산 최다 안타 타자들의 궤적 비교

- a. 통산 3200안타 이상을 기록한 타자들을 찾으십시오.
- b. AVG가 타율인 이 타자 그룹의 (Age, AVG) 데이터에 이차 함수를 맞추십시오. 단일 패널에 적합된 궤적을 표시하십시오.
- c. 수행한 작업을 바탕으로 어느 선수가 평균적으로 가장 일관된 타자였습니까? 적합된 궤적을 바탕으로 일관성을 어떻게 측정했는지 설명하십시오.

- 4. 홈런 타자들의 궤적 비교

- a. 야구 역사상 가장 많은 통산 홈런을 기록한 열 명의 선수를 찾으십시오.
- b. HRrate = HR/AB인 이 열 명의 선수의 홈런 비율에 이차 함수를 맞추십시오. 단일 패널에 적합된 궤적을 표시하십시오.
- c. 수행한 작업을 바탕으로 전성기에 가장 높은 추정 홈런 비율을 보인 선수는 누구입니까? 열 명 중 가장 낮은 전성기 홈런 비율을 보인 선수는 누구입니까?
- d. 특이한 경력 궤적 형태를 가진 선수가 있습니까? 이러한 특이한 형태에 대한 가능한 설명이 있습니까?

- 5. 야구 역사의 전성기 나이

- a. 1940년과 1945년 사이에 야구계에 입문하여 최소 2000통산 타수를 기록한 모든 선수를 찾으십시오.
- b. 1970년과 1975년 사이에 야구계에 입문하여 최소 2000통산 타수를 기록한 모든 선수를 찾으십시오.
- c. (Age, OPS) 데이터에 이차 함수를 맞춤으로써 (a)와 (b)의 모든 선수의 전성기 나이를 추정하십시오.
- d. 1940년대 선수들의 전성기 나이와 1970년대 선수들의 전성기 나이를 비교함으로써, 이 30년 기간 동안 전성기 나이가 어떻게 변했는지에 대한 결론을 내릴 수 있습니까?

### 9 시뮬레이션

9.1 소개

야구 시즌은 팀 간의 경기들의 모음으로 구성되며, 각 경기는 9이닝으로 구성되고, 반 이닝은 일련의 타석 등장으로 구성됩니다. 이러한 깔끔한 구조 덕분에, 야구라는 스포츠는 비교적 단순한 확률 모델로 나타낼 수 있습니다. 이 모델들의 시뮬레이션은 경기의 다양한 특성을 이해하는 데 도움이 됩니다.

R 시스템의 한 가지 매력적인 측면은 다양한 확률 분포에서 시뮬레이션할 수 있는 능력입니다. 이 장에서는 많은 수의 타석 등장으로 구성된 경기를 시뮬레이션하기 위한 R 함수의 사용을 설명합니다. 또한, 전체 시즌 동안 팀 간의 경기 대 경기의 경쟁을 시뮬레이션하기 위해 R을 사용합니다.

- 9.2절은 마르코프 체인(Markov chain)이라는 특수한 확률 모델을 사용하여 야구 반 이닝의 사건들을 시뮬레이션하는 데 초점을 맞춥니다. 베이스에 있는 주자와 아웃 수는 상태(state)를 정의하며 이 확률 모델은 3아웃에 도달할 때까지 상태 간의 이동을 설명합니다. 이동 또는 전이 확률은 2016 시즌의 실제 데이터를 사용하여 구합니다. 이 모델을 사용하여 많은 반 이닝을 시뮬레이션함으로써, 득점 패턴의 기본을 이해할 수 있습니다.
- 9.3절은 브래들리-테리(Bradley-Terry) 확률 모델을 사용한 전체 야구 시즌의 시뮬레이션을 설명합니다. 팀들은 종 모양(정규) 분포에서 재능을 할당받고 야구 경기의 시즌은 재능에 기반한 승리 확률을 사용하여 진행됩니다. 많은 시즌을 시뮬레이션함으로써, 162경기 시즌에서 팀의 재능과 성과 사이의 관계를 배웁니다. 포스트시즌 시리즈를 시뮬레이션하는 방법을 설명하고 "가장 뛰어난" 팀, 즉 최고의 능력을 가진 팀이 실제로 월드 시리즈에서 우승할 확률을 평가합니다.

DOI: 10.1201/9781032668239-9 204
