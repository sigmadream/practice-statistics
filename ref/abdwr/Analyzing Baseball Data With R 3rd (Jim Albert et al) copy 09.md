### 9 시뮬레이션

9.1 서론

야구 시즌은 팀 간의 경기들로 구성되며, 각 경기는 9이닝으로 이루어지고, 반 이닝은 타석의 연속으로 구성됩니다. 이러한 명확한 구조 때문에 야구라는 스포츠는 비교적 간단한 확률 모델로 표현할 수 있습니다. 이 모델들을 활용한 시뮬레이션은 경기의 다양한 특징을 이해하는 데 도움이 됩니다.

R 시스템의 매력적인 측면 중 하나는 다양한 확률 분포에서 시뮬레이션할 수 있는 기능입니다. 이 장에서는 많은 수의 타석으로 구성된 경기를 시뮬레이션하기 위해 R 함수를 사용하는 방법을 설명합니다. 또한, 시즌 전체 동안 팀들의 경기 대 경기의 경쟁을 시뮬레이션하기 위해 R을 사용합니다.

- 9.2절은 마르코프 체인(Markov chain)이라는 특수한 확률 모델을 사용하여 야구의 반 이닝 내 이벤트를 시뮬레이션하는 데 중점을 둡니다. 누상의 주자와 아웃 카운트가 상태(state)를 정의하며, 이 확률 모델은 3아웃에 도달할 때까지 상태 간의 이동을 설명합니다. 2016년 시즌의 실제 데이터를 사용하여 이동 또는 전이 확률(transition probabilities)을 구합니다. 이 모델을 사용하여 여러 번의 반 이닝을 시뮬레이션함으로써 득점 패턴에 대한 기본적인 이해를 얻을 수 있습니다.
- 9.3절은 BradleyTerry 확률 모델을 사용하여 전체 야구 시즌을 시뮬레이션하는 방법을 설명합니다. 종 모양(정규) 분포에서 팀의 기량이 할당되며, 이 기량에 기반한 승률을 사용하여 야구 경기 시즌이 진행됩니다. 많은 시즌을 시뮬레이션함으로써 162경기 시즌에서 팀의 기량과 성적 간의 관계에 대해 알 수 있습니다. 포스트시즌 시리즈 시뮬레이션을 설명하고, 최고의 팀, 즉 뛰어난 기량을 가진 팀이 실제로 월드 시리즈에서 우승할 확률을 평가합니다.

DOI: 10.1201/9781032668239-9 204

반 이닝 시뮬레이션 205

9.2 반 이닝 시뮬레이션

- 9.2.1 마르코프 체인

마르코프 체인은 상태(states)라고 불리는 위치 간의 이동을 설명하는 데 유용한 특별한 유형의 확률 모델입니다. 야구의 맥락에서 상태는 누상의 주자와 이닝의 아웃 카운트를 설명하는 것으로 간주됩니다. 세 개의 베이스 각각에 주자가 있을 수도 있고 없을 수도 있으므로, 2 × 2 × 2 = 8가지의 가능한 주자 상황이 있습니다. 아웃 카운트는 세 가지(0, 1, 2)가 가능하므로 8 × 3 = 24가지의 가능한 주자와 아웃 상태가 존재합니다. 3아웃 상태를 포함하면 야구의 반 이닝 동안 총 25가지의 가능한 상태가 있습니다(5.1절 참조).

마르코프 체인에서는 전이 확률 행렬을 사용하여 서로 다른 상태 간에 어떻게 이동하는지 설명합니다. 예를 들어, 현재 1아웃에 주자가 1, 2루에 있다고 가정해 보겠습니다. 타석의 결과에 따라 상태가 바뀔 수 있습니다. 예를 들어, 타자가 단타를 쳐서 2루 주자가 득점하고 1루 주자가 3루로 이동할 수 있습니다. 이 경우 새로운 상태는 1아웃에 주자 1, 3루입니다. 또는 타자가 삼진을 당하면 새로운 상태는 2아웃에 주자 1, 2루가 됩니다. 전이 확률 행렬의 특정 행을 살펴보면 1아웃 1, 3루로 이동할 확률이나 2아웃 1, 2루로 이동할 확률, 혹은 다른 모든 가능한 상태로 이동할 확률을 알 수 있습니다.

마르코프 체인에는 전이 상태(transition states)와 흡수 상태(absorbing states)라는 두 가지 유형의 상태가 있습니다. 일단 흡수 상태로 이동하면 그곳에 머무르며 다른 전이 상태로 돌아갈 수 없습니다. 야구의 반 이닝에서는 3아웃이 되면 이닝이 끝나기 때문에 이 3아웃 상태가 흡수 상태 역할을 합니다.

마르코프 체인 모델에는 몇 가지 특별한 가정이 있습니다. 우리는 새로운 상태로 이동할 확률이 현재 상태에만 의존한다고 가정합니다. 따라서 현재 주자와 아웃 상황 이전에 발생한 모든 야구 이벤트는 확률을 구하는 데 관련이 없습니다1. 즉, 이 모델은 이닝 동안 타격을 이어가는 데 있어 모멘텀 효과가 없다고 가정합니다. 또한 이러한 이동 확률이 모든 팀, 모든 투수, 경기 중 모든 이닝에 대해 동일하다고 가정합니다. 분명히 모든 팀이 평균적이라는 이러한 가정은 현실적이지 않지만, 이 문제는 이 장의 다른 절에서 다룰 것입니다.

야구의 반 이닝을 모델링하기 위해 마르코프 체인을 사용하는 데에는 몇 가지 매력적인 측면이 있습니다. 첫째, 5장의 계산을 통해 2016년 시즌 데이터를 사용하여 전이 확률 행렬을 쉽게 구성할 수 있습니다. 이 모델을 사용하여 야구의 많은 반 이닝을 플레이해 볼 수 있으며, 발견된 득점 패턴은 실제 MLB 야구의 실제 득점과 유사합니다.

1이것은 종종 무기억성(memoryless property)으로 알려져 있습니다.

마지막으로, 마르코프 체인에는 이닝 동안 타석에 들어서는 타자의 수와 같은 흥미로운 계산을 단순화하는 특별한 특성이 있습니다.

- 9.2.2 기대 득점(run expectancy) 연구 복습

마르코프 체인의 전이 행렬을 구성하려면 서로 다른 주자/아웃 상태에서 다른 가능한 주자/아웃 상태로 이동하는 빈도를 알아야 합니다. 특정 시즌의 Retrosheet 플레이-바이-플레이 데이터를 사용하여 이러한 빈도를 얻을 수 있습니다. 여기서 5장의 작업을 복습합니다.

2016년 시즌의 플레이-바이-플레이 데이터를 읽어들여 retro2016 데이터 프레임을 생성하는 것으로 시작합니다.

library(tidyverse) retro2016 <- read_rds(here::here("data/retro2016.rds"))

먼저 5장에서 작성하고 abdwr3edata 패키지에 저장한 retrosheet_add_states() 함수를 사용합니다. 이 함수는 retro2016에 여러 유용한 새 변수를 추가합니다. 특히, 이제 state라는 새 변수(각 플레이가 시작될 때 주자 위치와 아웃 카운트를 제공)와 new_state라는 또 다른 새 변수(플레이가 끝날 때 동일한 정보를 포함)가 생겼음을 기억하십시오.

library(abdwr3edata) retro2016 <- retro2016 |>

retrosheet_add_states()

다음으로 각 야구 경기의 각 반 이닝에 대한 고유 식별자로 half_inning_id 변수를 생성합니다. 새 변수 runs는 각 플레이에서 기록된 득점 수를 나타냅니다. 새로운 데이터 프레임 half_innings는 2016년에 치러진 야구의 각 반 이닝에 대해 집계된 데이터를 포함합니다.

half_innings <- retro2016 |>

mutate( runs = away_score_ct + home_score_ct, half_inning_id = paste(game_id, inn_ct, bat_home_id)

) |> group_by(half_inning_id) |> summarize(

outs_inning = sum(event_outs_ct), runs_inning = sum(runs_scored), runs_start = first(runs), max_runs = runs_inning + runs_start

야구. 마지막으로, 마르코프 체인에는 이닝 동안 타석에 들어서는 타자의 수와 같은 흥미로운 계산을 단순화하는 특별한 특성이 있습니다.

9.2.2 기대 득점(run expectancy) 연구 복습

마르코프 체인의 전이 행렬을 구성하려면 서로 다른 주자/아웃 상태에서 다른 가능한 주자/아웃 상태로 이동하는 빈도를 알아야 합니다. 특정 시즌의 Retrosheet 플레이-바이-플레이 데이터를 사용하여 이러한 빈도를 얻을 수 있습니다. 여기서 5장의 작업을 복습합니다.

2016년 시즌의 플레이-바이-플레이 데이터를 읽어들여 retro2016 데이터 프레임을 생성하는 것으로 시작합니다.

library(tidyverse) retro2016 <- read_rds(here::here("data/retro2016.rds"))

먼저 5장에서 작성하고 abdwr3edata 패키지에 저장한 retrosheet_add_states() 함수를 사용합니다. 이 함수는 retro2016에 여러 유용한 새 변수를 추가합니다. 특히, 이제 state라는 새 변수(각 플레이가 시작될 때 주자 위치와 아웃 카운트를 제공)와 new_state라는 또 다른 새 변수(플레이가 끝날 때 동일한 정보를 포함)가 생겼음을 기억하십시오.

library(abdwr3edata) retro2016 <- retro2016 |>

retrosheet_add_states()

다음으로 각 야구 경기의 각 반 이닝에 대한 고유 식별자로 half_inning_id 변수를 생성합니다. 새 변수 runs는 각 플레이에서 기록된 득점 수를 나타냅니다. 새로운 데이터 프레임 half_innings는 2016년에 치러진 야구의 각 반 이닝에 대해 집계된 데이터를 포함합니다.

half_innings <- retro2016 |>

mutate( runs = away_score_ct + home_score_ct, half_inning_id = paste(game_id, inn_ct, bat_home_id)

) |> group_by(half_inning_id) |> summarize(

outs_inning = sum(event_outs_ct), runs_inning = sum(runs_scored), runs_start = first(runs), max_runs = runs_inning + runs_start

)

dplyr 패키지의 filter() 함수를 사용하여 상태의 변화가 있거나 득점이 발생한 플레이에 초점을 맞춥니다. filter()를 또 한 번 적용하여 3아웃이 존재하고 타석 이벤트가 있는 완전한 이닝으로 관심을 제한하며, 새 데이터셋의 이름은 retro2016_complete입니다. 여기서는 도루, 도루자, 폭투, 포일과 같은 비타격 플레이는 무시됩니다. 득점 생산이라는 관점에서 이러한 비타격 플레이를 제거하는 데에는 명백한 결과가 따르며, 이 문제는 이 장의 후반부에서 논의합니다.

retro2016_complete <- retro2016 |> mutate(

half_inning_id = paste(game_id, inn_ct, bat_home_id) ) |> inner_join(half_innings, join_by(half_inning_id)) |> filter(state != new_state | runs_scored > 0) |> filter(outs_inning == 3, bat_event_fl)

new_state 변수의 정의에서 우리는 3아웃일 때의 주자 위치를 기록했습니다. 주자 위치는 중요하지 않으므로 아웃 카운트가 3일 때 new_state가 항상 3의 값을 갖도록 재코딩합니다. str_replace() 함수는 세 문자의 이진 문자열 뒤에 공백과 3이 오는 정규 표현식 [0-1]{3} 3을 3으로 대체합니다.

retro2016_complete <- retro2016_complete |> mutate(new_state = str_replace(new_state, "[0-1]{3} 3", "3"))

9.2.3 전이 확률 계산

이제 state와 new_state 변수가 정의되었으므로 table() 함수를 사용하여 상태 간의 모든 가능한 전이 빈도를 계산할 수 있습니다. 카운트 행렬은 T_matrix입니다. 시작 상태 state에는 24개의 가능한 값이 있고, 3아웃 상태를 포함하여 최종 상태 new_state에는 25개의 값이 있습니다.

T_matrix <- retro2016_complete |> select(state, new_state) |> table()

dim(T_matrix)

[1] 24 25

이 행렬은 prop.table() 함수를 사용하여 확률 행렬로 변환할 수 있습니다. 결과 행렬은 P_matrix로 나타냅니다.

P_matrix <- prop.table(T_matrix, 1) dim(P_matrix)

- [1] 24 25

마지막으로 이 전이 확률 행렬에 3아웃 상태로부터의 전이에 해당하는 행을 추가합니다. 이닝이 3아웃에 도달하면 3아웃에 머무르므로 이 상태에 머무를 확률은 1입니다.

P_matrix <- P_matrix |> rbind("3" = c(rep(0, 24), 1))

P_matrix 행렬은 이제 마르코프 체인에서 상태 간의 전이를 모델링할 수 있게 해주는 두 가지 중요한 특성을 갖습니다. 1) 정방 행렬이며; 2) 각 행의 항목 합이 1입니다.

dim(P_matrix)

- [1] 25 25

P_matrix |> apply(MARGIN = 1, FUN = sum)

000 0 000 1 000 2 001 0 001 1 001 2 010 0 010 1 010 2 011 0

1 1 1 1 1 1 1 1 1 1 011 1 011 2 100 0 100 1 100 2 101 0 101 1 101 2 110 0 110 1

1 1 1 1 1 1 1 1 1 1 110 2 111 0 111 1 111 2 3

1 1 1 1 1

이 전이 행렬을 더 잘 이해하기 위해 주자가 없고 아웃이 없는 "000 0" 상태에서 시작하는 전이 확률을 아래에 표시합니다. (양수 확률만 표시되며, 확률을 수직으로 표시하기 위해 as_tibble() 및 pivot_longer() 함수가 사용됩니다.) 가능성이 높은 전이는 확률 0.676인 "주자 없음, 1아웃" 상태와 확률 0.235인 "1루 주자, 무사" 상태입니다. "000 0" 상태에서 "000 0" 상태로 이동할 확률은 0.033입니다. 즉, 주자가 없는 무사 상황에서 홈런이 나올 확률은 0.033입니다.

P_matrix |> as_tibble(rownames = "state") |> filter(state == "000 0") |>

이 행렬은 prop.table() 함수를 사용하여 확률 행렬로 변환할 수 있습니다. 결과 행렬은 P_matrix로 나타냅니다.

P_matrix <- prop.table(T_matrix, 1) dim(P_matrix)

- [1] 24 25

마지막으로 이 전이 확률 행렬에 3아웃 상태로부터의 전이에 해당하는 행을 추가합니다. 이닝이 3아웃에 도달하면 3아웃에 머무르므로 이 상태에 머무를 확률은 1입니다.

P_matrix <- P_matrix |> rbind("3" = c(rep(0, 24), 1))

P_matrix 행렬은 이제 마르코프 체인에서 상태 간의 전이를 모델링할 수 있게 해주는 두 가지 중요한 특성을 갖습니다. 1) 정방 행렬이며; 2) 각 행의 항목 합이 1입니다.
dim(P_matrix)

- [1] 25 25

P_matrix |> apply(MARGIN = 1, FUN = sum)

- 000 0 000 1 000 2 001 0 001 1 001 2 010 0 010 1 010 2 011 0 1 1 1 1 1 1 1 1 1 1

011 1 011 2 100 0 100 1 100 2 101 0 101 1 101 2 110 0 110 1

1 1 1 1 1 1 1 1 1 1 110 2 111 0 111 1 111 2 3

1 1 1 1 1

이 전이 행렬을 더 잘 이해하기 위해, 주자가 없고 아웃이 없는 "000 0" 상태에서 시작하는 전이 확률을 아래에 표시합니다. (양의 확률만 표시하며 as_tibble() 및 pivot_longer() 함수를 사용하여 확률을 세로로 표시합니다.) 가능성이 높은 전이는 확률 0.676인 "주자 없음, 1아웃" 상태와 확률 0.235인 "1루 주자, 아웃 없음" 상태입니다. "000 0" 상태에서 "000 0" 상태로 이동할 확률은 0.033입니다. 즉, 주자가 없고 아웃이 없는 상태에서 홈런이 나올 확률은 0.033입니다.

P_matrix |> as_tibble(rownames = "state") |> filter(state == "000 0") |>

pivot_longer( cols = -state, names_to = "new_state", values_to = "Prob"

) |> filter(Prob > 0)

# A tibble: 5 x 3

state new_state Prob <chr> <chr> <dbl>

- 1 000 0 000 0 0.0334
- 2 000 0 000 1 0.676
- 3 000 0 001 0 0.00563
- 4 000 0 010 0 0.0503
- 5 000 0 100 0 0.235

이를 2루에 주자가 있고 2아웃인 "010 2" 상태에서 시작하는 가능한 전이와 비교해 보겠습니다. 가능성이 높은 전이는 "3아웃"(확률 0.650), "1루와 2루 주자, 2아웃"(확률 0.156), "1루 주자, 2아웃"(확률 0.074)입니다.

P_matrix |> as_tibble(rownames = "state") |> filter(state == "010 2") |> pivot_longer(

cols = -state, names_to = "new_state", values_to = "Prob"

) |> filter(Prob > 0)

# A tibble: 8 x 3

state new_state Prob <chr> <chr> <dbl>

- 1 010 2 000 2 0.0233
- 2 010 2 001 2 0.00587
- 3 010 2 010 2 0.0576
- 4 010 2 011 2 0.000451
- 5 010 2 100 2 0.0745
- 6 010 2 101 2 0.0325
- 7 010 2 110 2 0.156
- 8 010 2 3 0.650

  9.2.4 마르코프 연쇄 시뮬레이션

이 마르코프 연쇄 모델을 여러 번 시뮬레이션하여 2016년 야구의 반 이닝 동안 득점한 점수 분포를 얻을 수 있습니다. 첫 번째 단계는 상태 간의 가능한 모든 전이에서 득점한 점수를 보여주는 행렬을 구성하는 것입니다. Nrunners를 한 상태의 주자 수로, O를 아웃 수로 나타내겠습니다. 이닝에서 이미 타석에 선 모든 선수는 출루했거나 아웃되었거나 득점했으므로 타격 플레이의 경우 득점 수는 다음과 같습니다.

runs = (Nrunners(b) + O(b) + 1) − (Nrunners(a) + O(a)).

즉, 득점 수는 플레이 이전(b)의 주자와 아웃 수의 합에서 플레이 이후(a)의 주자와 아웃 수의 합을 빼고 1을 더한 값입니다. 예를 들어, 1아웃에 1루와 2루에 주자가 있고 플레이 후에 2아웃에 2루에 주자가 있다고 가정해 보겠습니다. 득점 수는 다음과 같습니다.

runs = (2 + 1 + 1) − (1 + 2) = 1.

상태를 입력으로 받아 주자와 아웃 수의 합을 반환하는 새로운 함수 num_havent_scored()를 정의합니다. 그런 다음 이 함수를 가능한 모든 상태에 적용하고(map_int() 함수 사용) 해당 합계를 runners_out 벡터에 저장합니다.

num_havent_scored <- function(s) {

s |> str_split("") |> pluck(1) |> as.numeric() |> sum(na.rm = TRUE)

} runners_out <- T_matrix |>

row.names() |> set_names() |> map_int(num_havent_scored)

—(빼기) 연산이 있는 outer() 함수는 가능한 모든 상태 쌍에 대해 득점 계산을 수행하고 결과 행렬은 R_runs 행렬에 저장됩니다. R_runs 행렬을 살펴보면 일부 음수 값과 이상하게 큰 양수 값을 볼 수 있습니다. 하지만 한 번의 타격 플레이에서 "000 0" 상태와 "000 2" 상태 사이의 이동과 같은 해당 전이는 불가능하므로 이는 문제가 되지 않습니다. 행렬을 정사각 행렬로 만들기 위해 cbind() 함수를 사용하여 이 런 행렬에 0으로 이루어진 추가 열을 더합니다.

- 9.2.4 마르코프 연쇄 시뮬레이션

이 마르코프 연쇄 모델을 여러 번 시뮬레이션하여 2016년 야구의 반 이닝 동안 득점한 점수 분포를 얻을 수 있습니다. 첫 번째 단계는 상태 간의 가능한 모든 전이에서 득점한 점수를 보여주는 행렬을 구성하는 것입니다. Nrunners를 한 상태의 주자 수로, O를 아웃 수로 나타내겠습니다. 이닝에서 이미 타석에 선 모든 선수는 출루했거나 아웃되었거나 득점했으므로 타격 플레이의 경우 득점 수는 다음과 같습니다.

runs = (Nrunners(b) + O(b) + 1) − (Nrunners(a) + O(a)).

즉, 득점 수는 플레이 이전(b)의 주자와 아웃 수의 합에서 플레이 이후(a)의 주자와 아웃 수의 합을 빼고 1을 더한 값입니다. 예를 들어, 1아웃에 1루와 2루에 주자가 있고 플레이 후에 2아웃에 2루에 주자가 있다고 가정해 보겠습니다. 득점 수는 다음과 같습니다.

runs = (2 + 1 + 1) − (1 + 2) = 1.

상태를 입력으로 받아 주자와 아웃 수의 합을 반환하는 새로운 함수 num_havent_scored()를 정의합니다. 그런 다음 이 함수를 가능한 모든 상태에 적용하고(map_int() 함수 사용) 해당 합계를 runners_out 벡터에 저장합니다.

num_havent_scored <- function(s) {

s |> str_split("") |> pluck(1) |> as.numeric() |> sum(na.rm = TRUE)

} runners_out <- T_matrix |>

row.names() |> set_names() |> map_int(num_havent_scored)

—(빼기) 연산이 있는 outer() 함수는 가능한 모든 상태 쌍에 대해 득점 계산을 수행하고 결과 행렬은 R_runs 행렬에 저장됩니다. R_runs 행렬을 살펴보면 일부 음수 값과 이상하게 큰 양수 값을 볼 수 있습니다. 하지만 한 번의 타격 플레이에서 "000 0" 상태와 "000 2" 상태 사이의 이동과 같은 해당 전이는 불가능하므로 이는 문제가 되지 않습니다. 행렬을 정사각 행렬로 만들기 위해 cbind() 함수를 사용하여 이 런 행렬에 0으로 이루어진 추가 열을 더합니다.

R_runs <- outer( runners_out + 1, runners_out, FUN = "-"

) |> cbind("3" = rep(0, 24))

이제 새로운 함수 simulate_half_inning()을 사용하여 야구의 반 이닝을 시뮬레이션할 준비가 되었습니다. 입력은 확률 전이 행렬 P, 런 행렬 R, 시작 상태 s(1에서 24 사이의 정수)입니다. 출력은 반 이닝 동안 득점한 수입니다.

simulate_half_inning <- function(P, R, start = 1) { s <- start path <- NULL runs <- 0 while (s < 25) {

s_new <- sample(1:25, size = 1, prob = P[s, ]) path <- c(path, s_new) runs <- runs + R[s, s_new] s <- s_new

} runs

}

이 시뮬레이션에는 두 가지 핵심 문이 있습니다. 현재 상태가 s인 경우, sample() 함수는 전이 행렬 P의 s 행을 사용하여 새로운 상태를 시뮬레이션합니다. 새로운 상태는 s_new로 표시됩니다. 이닝에서 득점한 총 점수는 s 행과 런 행렬 R의 s_new 열에 있는 값을 사용하여 업데이트됩니다.

map_int() 함수를 사용하면 야구의 많은 반 이닝을 시뮬레이션할 수 있습니다. 아래 코드에서는 주자와 아웃이 없는 상태(상태 1)에서 시작하여 10,000번의 반 이닝을 시뮬레이션하고, 득점한 점수를 simulated_runs 벡터에 수집합니다. set.seed() 함수는 난수 시드를 설정하여 독자가 이 코드를 실행하여 이 특정 시뮬레이션 결과를 재현할 수 있도록 합니다.

set.seed(111653) simulated_runs <- 1:10000 |>

map_int(~simulate_half_inning(T_matrix, R_runs))

반 이닝 동안 득점 가능한 점수를 찾기 위해 table() 함수를 사용하여 simulated_runs의 값을 표로 만듭니다.

table(simulated_runs)

simulated_runs

0 1 2 3 4 5 6 7 8 9 7364 1437 651 324 126 50 34 10 2 2

10,000번의 시뮬레이션에서 50 + 34 + 10 + 2 + 2 = 98번의 반 이닝에서 5점 이상 득점했으므로, 5점 이상 득점할 확률은 98 / 10,000 = 0.0098이 됩니다. 이 계산은 sum() 함수를 사용하여 확인할 수 있습니다.

sum(simulated_runs >= 5) / 10000

[1] 0.0098

mean() 함수를 simulated_runs에 적용하여 득점한 점수의 평균을 계산합니다.

mean(simulated_runs)

[1] 0.477 10,000번의 반 이닝 동안 평균 0.477점이 득점되었습니다.

다양한 주자와 아웃 상황의 득점 가능성을 이해하기 위해 다른 시작 상태에 대해 이 시뮬레이션 절차를 반복할 수 있습니다. 상태 j에서 시작하여 득점한 평균 점수를 계산하는 함수 runs_j()를 작성합니다. map_int() 함수를 사용하여 가능한 모든 시작 상태 1부터 24까지 함수 runs_j()를 적용합니다. 출력은 mean_run_value 열에 저장된 득점한 평균 점수의 벡터입니다. 이 값은 시뮬레이션된 예상 런 행렬로 아래에 표시됩니다(섹션 5.1 참조).

runs_j <- function(j) {

1:10000 |> map_int(~simulate_half_inning(T_matrix, R_runs, j)) |> mean()
} erm_2016_mc <- tibble(

state = row.names(T_matrix), mean_run_value = map_dbl(1:24, runs_j)

) |>

mutate( bases = str_sub(state, 1, 3), outs_ct = as.numeric(str_sub(state, 5, 5))

) |> select(-state)

table(simulated_runs)

simulated_runs

0 1 2 3 4 5 6 7 8 9 7364 1437 651 324 126 50 34 10 2 2

10,000번의 시뮬레이션에서 50 + 34 + 10 + 2 + 2 = 98번의 이닝 절반 동안 5점 이상 득점했습니다. 따라서 5점 이상 득점할 확률은 98 /

- 10,000 = 0.0098이 됩니다. 이 계산은 sum() 함수를 사용하여 확인할 수 있습니다.

sum(simulated_runs >= 5) / 10000

[1] 0.0098

mean() 함수를 simulated_runs에 적용하여 평균 득점 수를 계산합니다.

mean(simulated_runs)

[1] 0.477 10,000번의 이닝 절반 동안 평균 0.477점이 득점되었습니다.

다양한 주자 및 아웃 상황의 득점 잠재력을 이해하기 위해 다른 시작 상태에 대해 이 시뮬레이션 절차를 반복할 수 있습니다. 상태 j에서 시작하여 평균 득점 수를 계산하는 runs_j() 함수를 작성합니다. map_int() 함수를 사용하여 1부터 24까지의 모든 가능한 시작 상태에 runs_j() 함수를 적용합니다. 출력은 mean_run_value 열에 저장된 평균 득점 벡터입니다. 이 값들은 시뮬레이션된 기대 득점 행렬(섹션 5.1 참조)로 아래에 표시됩니다.

runs_j <- function(j) {

1:10000 |> map_int(~simulate_half_inning(T_matrix, R_runs, j)) |> mean()

} erm_2016_mc <- tibble(

state = row.names(T_matrix), mean_run_value = map_dbl(1:24, runs_j)

) |>

mutate( bases = str_sub(state, 1, 3), outs_ct = as.numeric(str_sub(state, 5, 5))

) |> select(-state)

erm_2016_mc |> pivot_wider(names_from = outs_ct, values_from = mean_run_value)

# A tibble: 8 x 4

bases `0` `1` `2` <chr> <dbl> <dbl> <dbl>

- 1 000 0.481 0.255 0.103
- 2 001 1.32 0.925 0.338
- 3 010 1.14 0.640 0.295
- 4 011 1.93 1.30 0.474
- 5 100 0.855 0.500 0.211
- 6 101 1.71 1.14 0.425
- 7 110 1.39 0.875 0.406
- 8 111 2.19 1.46 0.667

시뮬레이션 모델이 타격 플레이만을 기반으로 한다는 것을 기억하십시오. 타격 이외의 플레이(도루, 도루 실패, 폭투 등)가 득점에 미치는 영향을 이해하기 위해, 이 기대 득점 행렬을 모든 타격 및 비타격 플레이를 사용한 5장에서 찾은 행렬과 비교합니다. 이들의 차이는 평균 득점 수에 대한 비타격 플레이의 기여도입니다.

erm_2016 <- read_rds(here::here("data/erm2016.rds")) erm_2016 |>

inner_join(erm_2016_mc, join_by(bases, outs_ct)) |> mutate(

run_value_diff = round(mean_run_value.x - mean_run_value.y, 2) ) |> select(bases, outs_ct, run_value_diff) |> pivot_wider(names_from = outs_ct, values_from = run_value_diff)

# A tibble: 8 x 4 # Groups: bases [8]

bases `0` `1` `2` <chr> <dbl> <dbl> <dbl>

- 1 000 0.02 0.01 0
- 2 001 0.03 0.01 0.03
- 3 010 0 0.03 0.02
- 4 011 0 0.06 0.07
- 5 100 0 0.01 0.01
- 6 101 0.02 0.06 0.05
- 7 110 0.06 0.05 0.01
- 8 111 -0.08 0.07 0.03

차이 값의 대부분이 양수이며, 이는 비타격 플레이가 일반적으로 득점을 창출함을 나타냅니다. 큰 값은 폭투나 패스트볼로 득점할 수 있는 3루 주자가 있는 상황에서 발생하는 경향이 있습니다.

9.2.5 기대 득점 그 이상

마르코프 체인의 특성을 사용하면 전환 행렬을 사용하여 주자/아웃 상태를 통과하는 움직임에 대해 더 많이 배우는 것이 간단합니다.

확률 행렬 P_matrix를 세 번 곱하면, 세 번의 타석 후 이닝 상태의 가능성에 대해 알 수 있습니다. R에서 행렬 곱셈은 %\*% 기호로 표시됩니다. 결과는 P_matrix_3 행렬에 저장됩니다.

P*matrix_3 <- P_matrix %*% P*matrix %*% P_matrix

P_matrix_3의 첫 번째 행은 "000 0" 상태에서 시작하여 세 명의 타자 이후 25개 상태 각각에 있을 확률을 제공합니다. 이 값들을 소수점 이하 셋째 자리에서 반올림하고, 큰 것부터 작은 것 순으로 정렬하여 큰 값을 표시합니다.

P_sorted <- P_matrix_3 |> as_tibble(rownames = "state") |> filter(state == "000 0") |> pivot_longer(

cols = -state, names_to = "new_state", values_to = "Prob" ) |> arrange(desc(Prob))

P_sorted |> slice_head(n = 6)

# A tibble: 6 x 3

state new_state Prob <chr> <chr> <dbl>

- 1 000 0 3 0.372
- 2 000 0 100 2 0.241
- 3 000 0 110 1 0.0815
- 4 000 0 010 2 0.0739
- 5 000 0 000 2 0.0529
- 6 000 0 001 2 0.0286

세 번의 타석 이후 가능성 있는 결과는 3아웃(확률 0.372), 2아웃에 주자 1루(확률 0.241), 1아웃에 주자 1, 2루(확률 0.081)입니다.

차이 값의 대부분이 양수이며, 이는 비타격 플레이가 일반적으로 득점을 창출함을 나타냅니다. 큰 값은 폭투나 패스트볼로 득점할 수 있는 3루 주자가 있는 상황에서 발생하는 경향이 있습니다.

- 9.2.5 기대 득점 그 이상

마르코프 체인의 특성을 사용하면 전환 행렬을 사용하여 주자/아웃 상태를 통과하는 움직임에 대해 더 많이 배우는 것이 간단합니다.

확률 행렬 P_matrix를 세 번 곱하면, 세 번의 타석 후 이닝 상태의 가능성에 대해 알 수 있습니다. R에서 행렬 곱셈은 %\*% 기호로 표시됩니다. 결과는 P_matrix_3 행렬에 저장됩니다.

P*matrix_3 <- P_matrix %*% P*matrix %*% P_matrix

P_matrix_3의 첫 번째 행은 "000 0" 상태에서 시작하여 세 명의 타자 이후 25개 상태 각각에 있을 확률을 제공합니다. 이 값들을 소수점 이하 셋째 자리에서 반올림하고, 큰 것부터 작은 것 순으로 정렬하여 큰 값을 표시합니다.

P_sorted <- P_matrix_3 |> as_tibble(rownames = "state") |> filter(state == "000 0") |> pivot_longer(

cols = -state, names_to = "new_state", values_to = "Prob" ) |> arrange(desc(Prob))

P_sorted |> slice_head(n = 6)

# A tibble: 6 x 3

state new_state Prob <chr> <chr> <dbl>

- 1 000 0 3 0.372
- 2 000 0 100 2 0.241
- 3 000 0 110 1 0.0815
- 4 000 0 010 2 0.0739
- 5 000 0 000 2 0.0529
- 6 000 0 001 2 0.0286

세 번의 타석 이후 가능성 있는 결과는 3아웃(확률 0.372), 2아웃에 주자 1루(확률 0.241), 1아웃에 주자 1, 2루(확률 0.081)입니다.
모든 주자-아웃 상태를 방문하는 횟수를 알아보는 것도 쉽습니다. 행렬 Q를 전이 행렬에서 마지막 행과 열(3아웃 상태)을 제거하여 구한 24x24 부분 행렬로 정의합니다. 단위 행렬에서 행렬 Q를 빼고 그 결과의 역행렬을 취하면 흡수 마르코프 체인의 기본 행렬 N을 얻을 수 있습니다. (diag() 함수는 단위 행렬을 구성하는 데 사용되며 solve() 함수는 행렬의 역행렬을 계산합니다.)

Q <- P_matrix[-25, -25] N <- solve(diag(rep(1, 24)) - Q)

기본 행렬을 이해하기 위해 행렬의 첫 번째 행의 시작 항목을 표시합니다.

N_0000 <- round(N["000 0", ], 2) head(N_0000, n = 6)

000 0 000 1 000 2 001 0 001 1 001 2 1.05 0.75 0.60 0.01 0.03 0.05

이닝의 시작("000 0" 상태)에서 출발할 때 이닝이 "000 0" 상태에 있는 평균 횟수는 1.05번, "000 1" 상태에 있는 평균 횟수는 0.75번, "000 2" 상태에 있는 평균 횟수는 0.6번 등입니다. sum() 함수를 사용하여 방문하는 평균 상태 수를 찾습니다.

sum(N_0000)

[1] 4.27

다시 말해, 3아웃 전 반 이닝에서 타석에 들어서는 평균 횟수는 4.27번입니다.

기본 행렬 N에 1로 구성된 열 벡터를 곱하여 모든 시작 상태에서 3아웃까지의 평균 타격 플레이 횟수를 계산할 수 있습니다. 평균 플레이 횟수의 벡터는 avg_num_plays 변수에 저장되며 이 벡터의 8개 값이 표시됩니다.

avg_num_plays <- N %\*% rep(1, 24) |> t() |> round(2)

avg_num_plays[,1:8]

000 0 000 1 000 2 001 0 001 1 001 2 010 0 010 1 4.27 2.87 1.46 4.33 2.99 1.53 4.34 2.93

이는 가능한 각 상태에서 시작하여 남은 이닝의 평균 길이를 알려줍니다. 예를 들어 주자 없는 1아웃 상태에서 시작하면 평균적으로 2.87명의 타자가 더 나올 것으로 예상합니다. 반면 3루에 주자가 있고 2아웃인 경우 1.53명의 타자가 더 나올 것으로 예상합니다.

9.2.6 개별 팀에 대한 전이 확률

전이 확률 행렬은 평균적인 팀에 대한 상태 간의 이동을 설명합니다. 타격 능력이 다른 팀의 경우 이 확률이 달라지며, 투수 능력이 다른 팀을 상대로 할 때도 확률이 달라질 것입니다. 우리는 여러 타격 팀에 초점을 맞추고 모든 팀에 대해 전이 확률의 좋은 추정치를 얻는 방법을 논의합니다.

관련 데이터를 얻으려면 각 반 이닝에서 타격 팀을 알려주는 새로운 변수 batting_team을 정의해야 합니다. str_sub() 함수를 사용하여 홈 팀 변수 home_team_id를 정의하고, if_else() 함수를 사용하여 타격 팀을 정의합니다.

retro2016_complete <- retro2016_complete |>

mutate( home_team_id = str_sub(game_id, 1, 3), batting_team = if_else(

bat_home_id == 0, away_team_id, home_team_id

) )

group_by()와 count() 함수를 사용하여 현재 상태에서 새로운 상태로의 전이에서 각 팀의 횟수를 알려주는 데이터 프레임 T_team을 구성합니다.

T_team <- retro2016_complete |> group_by(batting_team, state, new_state) |> count()

예를 들어 batting_team을 ANA로 필터링하면 2016년 시즌 애너하임의 전이 횟수가 나타납니다.

T_team |> filter(batting_team == "ANA") |> slice_head(n = 6)

# A tibble: 192 x 4 # Groups: batting_team, state, new_state [192]

batting_team state new_state n

이는 가능한 각 상태에서 시작하여 남은 이닝의 평균 길이를 알려줍니다. 예를 들어 주자 없는 1아웃 상태에서 시작하면 평균적으로 2.87명의 타자가 더 나올 것으로 예상합니다. 반면 3루에 주자가 있고 2아웃인 경우 1.53명의 타자가 더 나올 것으로 예상합니다.

- 9.2.6 개별 팀에 대한 전이 확률

전이 확률 행렬은 평균적인 팀에 대한 상태 간의 이동을 설명합니다. 타격 능력이 다른 팀의 경우 이 확률이 달라지며, 투수 능력이 다른 팀을 상대로 할 때도 확률이 달라질 것입니다. 우리는 여러 타격 팀에 초점을 맞추고 모든 팀에 대해 전이 확률의 좋은 추정치를 얻는 방법을 논의합니다.

관련 데이터를 얻으려면 각 반 이닝에서 타격 팀을 알려주는 새로운 변수 batting_team을 정의해야 합니다. str_sub() 함수를 사용하여 홈 팀 변수 home_team_id를 정의하고, if_else() 함수를 사용하여 타격 팀을 정의합니다.

retro2016_complete <- retro2016_complete |>

mutate( home_team_id = str_sub(game_id, 1, 3), batting_team = if_else(

bat_home_id == 0, away_team_id, home_team_id

) )

group_by()와 count() 함수를 사용하여 현재 상태에서 새로운 상태로의 전이에서 각 팀의 횟수를 알려주는 데이터 프레임 T_team을 구성합니다.

T_team <- retro2016_complete |> group_by(batting_team, state, new_state) |> count()

예를 들어 batting_team을 ANA로 필터링하면 2016년 시즌 애너하임의 전이 횟수가 나타납니다.

T_team |> filter(batting_team == "ANA") |> slice_head(n = 6)

# A tibble: 192 x 4 # Groups: batting_team, state, new_state [192]

batting_team state new_state n

<chr> <chr> <chr> <int>

- 1 ANA 000 0 000 0 40
- 2 ANA 000 0 000 1 1007
- 3 ANA 000 0 001 0 9
- 4 ANA 000 0 010 0 75
- 5 ANA 000 0 100 0 359
- 6 ANA 000 1 000 1 31
- 7 ANA 000 1 000 2 720
- 8 ANA 000 1 001 1 3
- 9 ANA 000 1 010 1 54
- 10 ANA 000 1 100 1 261 # i 182 more rows

다른 타격 팀에 대한 득점 생산성을 비교하는 데 관심이 있다면, 성능의 현실적인 예측을 얻기 위해 팀 전이 확률 행렬에 일부 조정을 가할 필요가 있습니다. 문제를 설명하기 위해 "100 2" 상태에서의 전이에 초점을 맞춥니다. tally() 함수를 사용하여 전이 횟수를 T_team_S 데이터 프레임에 저장하고 6개 팀에 대해 이 표의 몇 행을 아래에 표시합니다.

T_team_S <- retro2016_complete |> filter(state == "100 2") |> group_by(batting_team, state, new_state) |> tally()

T_team_S |> ungroup() |> sample_n(size = 6)

# A tibble: 6 x 4

batting_team state new_state n <chr> <chr> <chr> <int>

- 1 MIN 100 2 010 2 15
- 2 CIN 100 2 101 2 19
- 3 NYN 100 2 101 2 14
- 4 SEA 100 2 010 2 6
- 5 DET 100 2 011 2 9
- 6 DET 100 2 101 2 15

덜 일반적인 일부 전이의 경우 팀 간 횟수의 변동성이 크며 이로 인해 해당 팀 전이 확률을 신뢰하기 어렵게 됩니다. pTEAM이 특정 팀에 대한 팀 전이 확률이고 pALL이 평균 전이 확률인 경우 팀 확률에 대한 더 나은 추정치는 다음과 같은 형태를 갖습니다.

n n + K

K n + K

pEST =

pTEAM +

pALL,

여기서 n은 팀의 전이 횟수이고 K는 평활화 횟수입니다. 방법론에 대한 설명은 이 책의 범위를 벗어나지만, 이 경우 K = 1274의 평활화 횟수는 팀 전이 확률에 대한 좋은 추정치를 제공합니다. (K의 선택은 시작 상태에 따라 다릅니다.)

이 방법은 "100 2" 상태에서 시작하는 워싱턴의 전이 횟수에 대해 설명됩니다. 데이터 프레임 T_WAS에서 전이 횟수는 변수 n에 저장되고 해당 비율은 p에 저장됩니다. 마찬가지로 모든 팀에 대해 n과 p는 데이터 프레임 T_all의 횟수 및 비율입니다.

T_WAS <- T_team_S |> filter(batting_team == "WAS") |> mutate(p = n / sum(n))

T_all <- retro2016_complete |> filter(state == "100 2") |> group_by(new_state) |> tally() |> mutate(p = n / sum(n))

공식을 사용하여 워싱턴의 전이 비율에 대한 개선된 추정치를 계산하고 그 결과를 p_EST에 저장합니다. 세 가지 비율 세트(워싱턴, 전체 및 개선된 값)가 데이터 프레임에 표시됩니다.

T_WAS |> inner_join(T_all, by = "new_state") |> mutate(

p*EST = (n.x / (1274 + n.x)) * p.x + (1274 / (1274 + n.x)) \_ p.y ) |> select(batting_team, new_state, p.x, p.y, p_EST)

# A tibble: 8 x 6 # Groups: batting_team, state [1]

state batting_team new_state p.x p.y p_EST <chr> <chr> <chr> <dbl> <dbl> <dbl>

- 1 100 2 WAS 000 2 0.0319 0.0291 0.0291
- 2 100 2 WAS 001 2 0.00532 0.00577 0.00577
- 3 100 2 WAS 010 2 0.0213 0.0220 0.0219
- 4 100 2 WAS 011 2 0.0213 0.0220 0.0219
- 5 100 2 WAS 100 2 0.00266 0.000775 0.000776
- 6 100 2 WAS 101 2 0.0452 0.0435 0.0435
- 7 100 2 WAS 110 2 0.184 0.195 0.194
- 8 100 2 WAS 3 0.689 0.682 0.683

개선된 전환 비율은 팀의 비율과 전체 값 사이의 절충안이라는 점에 유의하시기 바랍니다. 예를 들어,

여기서 n은 해당 팀의 전환 횟수이고 K는 평활화(smoothing) 횟수입니다. 이 방법론에 대한 설명은 이 책의 범위를 벗어나지만, 이 경우 K = 1274인 평활화 횟수는 팀의 전환 확률에 대한 좋은 추정치로 이어집니다. (K의 선택은 시작 상태에 따라 달라집니다.)

이 방법은 “100 2” 상태에서 시작하는 워싱턴의 전환 횟수에 대해 설명되어 있습니다. 데이터 프레임 T_WAS에서 전환 횟수는 변수 n에 저장되고 해당 비율은 p에 저장됩니다. 마찬가지로 모든 팀의 경우 n과 p는 데이터 프레임 T_all의 횟수 및 비율입니다.

T_WAS <- T_team_S |> filter(batting_team == "WAS") |> mutate(p = n / sum(n))

T_all <- retro2016_complete |> filter(state == "100 2") |> group_by(new_state) |> tally() |> mutate(p = n / sum(n))

수식을 사용하여 워싱턴의 전환 비율에 대한 개선된 추정치를 계산하고 그 결과를 p_EST에 저장합니다. 세 가지 비율(워싱턴, 전체, 개선된 비율) 세트가 데이터 프레임에 표시됩니다.

T_WAS |> inner_join(T_all, by = "new_state") |> mutate(

p*EST = (n.x / (1274 + n.x)) * p.x + (1274 / (1274 + n.x)) \_ p.y ) |> select(batting_team, new_state, p.x, p.y, p_EST)

# A tibble: 8 x 6 # Groups: batting_team, state [1]

state batting_team new_state p.x p.y p_EST <chr> <chr> <chr> <dbl> <dbl> <dbl>

- 1 100 2 WAS 000 2 0.0319 0.0291 0.0291
- 2 100 2 WAS 001 2 0.00532 0.00577 0.00577
- 3 100 2 WAS 010 2 0.0213 0.0220 0.0219
- 4 100 2 WAS 011 2 0.0213 0.0220 0.0219
- 5 100 2 WAS 100 2 0.00266 0.000775 0.000776
- 6 100 2 WAS 101 2 0.0452 0.0435 0.0435
- 7 100 2 WAS 110 2 0.184 0.195 0.194
- 8 100 2 WAS 3 0.689 0.682 0.683

개선된 전환 비율은 팀의 비율과 전체 값 사이의 절충안이라는 점에 유의하시기 바랍니다. 예를 들어,

상태 “100 2”에서 “010 2”로의 전환의 경우, 워싱턴 값은 0.0213이고 전체 값은 0.0220이며 개선된 값 0.0219는 워싱턴 값과 전체 값 사이에 있습니다. 이 방법은 이번 시즌 한 팀에서는 발생하지 않을 수 있지만 향후 이러한 전환이 발생할 확실한 가능성이 있다는 것을 알고 있는 “100 2”에서 “100 2”로의 전환과 같은 특정 전환에 유용합니다.

이 평활화 방법은 모든 팀과 전환 행렬의 모든 행에 적용하여 팀의 확률 전환 행렬에 대한 향상된 추정치를 얻을 수 있습니다. 이렇게 계산된 팀 전환 행렬을 사용하면 개별 타격 팀의 득점 행동을 탐색할 수 있습니다.

###### 9.3 야구 시즌 시뮬레이션

- 9.3.1 Bradley-Terry 모델

야구 경기와 같은 쌍별 비교 데이터를 모델링하는 매력적인 방법은 Bradley-Terry 모델입니다. 정규 시즌과 플레이오프 시스템이 비교적 단순한 구조를 가졌던 1968년 메이저리그 야구 시즌에 대한 시뮬레이션을 통해 이 모델링 기법을 설명합니다. 더 복잡한 일정과 플레이오프 시스템을 갖춘 현재의 야구 시즌에 이 방법론을 적용하는 것은 간단합니다.

1968년에는 내셔널리그 10팀, 아메리칸리그 10팀 등 총 20개 팀이 있었습니다. 각 팀이 경기에서 이길 수 있는 재능이나 능력이 있다고 가정합니다. 20개 팀의 재능은 값 T1,...,T20으로 나타냅니다. 재능은 평균 0과 표준 편차 sT를 갖는 정규 곡선 모델에서 분포한다고 가정합니다. 평균적인 능력을 가진 팀의 재능 값은 0에 가까우며,

좋은 팀은 양의 재능을 갖고, 나쁜 팀은 음의 재능을 갖게 됩니다. A 팀이 단일 경기에서 B 팀과 경기를 한다고 가정해 보겠습니다. Bradley-Terry 모델에 따르면, A 팀이 경기에서 승리할 확률은 다음과 같은 로지스틱 함수로 주어집니다.

exp(TA) exp(TA) + exp(TB)

P(Awins) =

.

이 모델은 1980년대 Bill James가 야구 요약서(Baseball Abstract)에서 개발한 log5 방법론과 밀접한 관련이 있습니다(James (1982) 참조). PA와 PB가 A팀과 B팀의 승률인 경우 James의 공식은 다음과 같이 주어집니다.

PA/(1 − PA) PA/(1 − PA) + PB/(1 − PB)

P(Awins) =

.

두 공식을 비교해 보면 log5 방식은 팀의 재능 T가 승리할 로그 오즈

log(P/(1 − P))와 같게 설정되는 Bradley-Terry 모델의 특별한 경우임을 알 수 있습니다. 재능 T = 0인 팀은 (장기적으로) 경기의 절반(P = 0.5)을 이깁니다. 반면, 재능 T = 0.2인 팀은 (log 5 값을 사용하면) 경기의 약 55%를 승리하고 재능 T = -0.2인 팀은 경기의 45%를 승리합니다.

이 모델을 사용하면 다음과 같이 야구 시즌을 시뮬레이션할 수 있습니다.

- 1. 1968년 야구 일정을 구성합니다. 이번 시즌에는 각 리그의 10개 팀이 같은 리그의 다른 팀과 각각 18경기를 치르며, 각 팀의 홈구장에서 9경기가 치러집니다. (1968년에는 인터리그 경기가 없었습니다.)
- 2. 평균 0과 표준편차

sT를 갖는 정규 분포에서 20개의 재능을 시뮬레이션합니다. sT의 값은 이 모델에서 시뮬레이션된 시즌 승률이 이 시즌의 실제 승률과 유사하도록 선택됩니다.

- 3. 확률 공식과 재능 값을 사용하여 홈팀이 모든 경기에서 승리할 확률을 계산합니다. 이 확률을 가진 일련의 동전 던지기를 통해 모든 경기의 승자를 결정합니다.
- 4. 각 리그의 승자를 결정하고 (동점은 어떤 무작위 메커니즘으로 해결해야 합니다) Bradley-Terry 모델과 두 개의 재능 숫자를 사용하여 계산된 승리 확률을 사용하여 7전 4선승제 월드 시리즈를 플레이합니다.

- 9.3.2 일정 짜기

시뮬레이션의 첫 번째 단계는 경기 일정을 구성하는 것입니다. 이 작업을 돕기 위해 짧은 함수 make_schedule()을 작성했습니다. 입력은 팀 이름 벡터 teams와 첫 번째 팀의 홈 구장에서 두 팀 간에 플레이할 경기 수 k입니다. 출력은 각 행이 경기에 해당하고 Home과 Visitor가 홈팀과 방문팀의 이름을 제공하는 데이터 프레임입니다. 벡터의 반복 복사본을 생성하는 rep() 함수가 이 함수에서 여러 번 사용됩니다.

make_schedule <- function(teams, k) { num_teams <- length(teams) Home <- rep(rep(teams, each = num_teams), k) Visitor <- rep(rep(teams, num_teams), k) tibble(Home = Home, Visitor = Visitor) |>

filter(Home != Visitor) }

이 함수는 1968년 시즌의 일정을 구성하는 데 사용됩니다. 내셔널리그 및 아메리칸리그 팀에 대한 약어가 포함된 두 개의 벡터 NL 및 AL이 구성됩니다. 한 팀이 다른 팀과 9번의 경기를 주최하므로 k = 9를 사용하여 각 리그에 한 번씩 make_schedule() 함수를 두 번 적용합니다. list_rbind() 함수를 사용하여 NL 및 AL

승리할 로그 오즈 log(P/(1 − P))와 같게 설정되는 Bradley-Terry 모델의 특별한 경우임을 알 수 있습니다. 재능 T = 0인 팀은 (장기적으로) 경기의 절반(P = 0.5)을 이깁니다. 반면, 재능 T = 0.2인 팀은 (log 5 값을 사용하면) 경기의 약 55%를 승리하고 재능 T = -0.2인 팀은 경기의 45%를 승리합니다.

이 모델을 사용하면 다음과 같이 야구 시즌을 시뮬레이션할 수 있습니다.

- 1. 1968년 야구 일정을 구성합니다. 이번 시즌에는 각 리그의 10개 팀이 같은 리그의 다른 팀과 각각 18경기를 치르며, 각 팀의 홈구장에서 9경기가 치러집니다. (1968년에는 인터리그 경기가 없었습니다.)
- 2. 평균 0과 표준편차 sT를 갖는 정규 분포에서 20개의 재능을 시뮬레이션합니다. sT의 값은 이 모델에서 시뮬레이션된 시즌 승률이 이 시즌의 실제 승률과 유사하도록 선택됩니다.
- 3. 확률 공식과 재능 값을 사용하여 홈팀이 모든 경기에서 승리할 확률을 계산합니다. 이 확률을 가진 일련의 동전 던지기를 통해 모든 경기의 승자를 결정합니다.
- 4. 각 리그의 승자를 결정하고 (동점은 어떤 무작위 메커니즘으로 해결해야 합니다) Bradley-Terry 모델과 두 개의 재능 숫자를 사용하여 계산된 승리 확률을 사용하여 7전 4선승제 월드 시리즈를 플레이합니다.

  9.3.2 일정 짜기

시뮬레이션의 첫 번째 단계는 경기 일정을 구성하는 것입니다. 이 작업을 돕기 위해 짧은 함수 make_schedule()을 작성했습니다. 입력은 팀 이름 벡터 teams와 첫 번째 팀의 홈 구장에서 두 팀 간에 플레이할 경기 수 k입니다. 출력은 각 행이 경기에 해당하고 Home과 Visitor가 홈팀과 방문팀의 이름을 제공하는 데이터 프레임입니다. 벡터의 반복 복사본을 생성하는 rep() 함수가 이 함수에서 여러 번 사용됩니다.

make_schedule <- function(teams, k) { num_teams <- length(teams) Home <- rep(rep(teams, each = num_teams), k) Visitor <- rep(rep(teams, num_teams), k) tibble(Home = Home, Visitor = Visitor) |>

filter(Home != Visitor) }

이 함수는 1968년 시즌의 일정을 구성하는 데 사용됩니다. 내셔널리그 및 아메리칸리그 팀에 대한 약어가 포함된 두 개의 벡터 NL 및 AL이 구성됩니다. 한 팀이 다른 팀과 9번의 경기를 주최하므로 k = 9를 사용하여 각 리그에 한 번씩 make_schedule() 함수를 두 번 적용합니다. list_rbind() 함수를 사용하여 NL 및 AL

일정을 붙여넣어 데이터 프레임 schedule을 생성합니다.

library(Lahman) teams_68 <- Teams |>

filter(yearID == 1968) |> select(teamID, lgID) |> mutate(teamID = as.character(teamID)) |> group_by(lgID)

schedule <- teams_68 |> group_split() |> set_names(pull(group_keys(teams_68), "lgID")) |> map(~make_schedule(teams = .x$teamID, k = 9)) |> list_rbind(names_to = "lgID")

dim(schedule)

[1] 1620 3
schedule에는 1622·20개의 행이 있다는 점에 유의하시기 바랍니다. 각 경기에는 두 팀이 참여하기 때문입니다.

- 9.3.3 재능 시뮬레이션 및 승리 확률 계산

다음 단계는 시즌 일정에 있는 모든 경기에 대한 승리 확률을 계산하는 것입니다. 팀의 재능은 평균 0과 표준 편차 s_talent를 갖는 정규 분포에서 나온다고 가정하고 s_talent = 0.20을 할당합니다. (모델에서 생성된 시즌 팀 승률이 실제 팀 승률과 유사하도록 표준 편차의 값을 선택한다는 점을 기억하시기 바랍니다.) 재능을 20개 팀에 할당하는 rnorm() 함수를 사용하여 재능을 시뮬레이션합니다. inner_join() 함수를 두 번 적용하여 schedule 데이터 프레임에 팀 재능을 추가합니다. 새 데이터 프레임의 이름은 schedule_talent입니다.

s_talent <- 0.20 teams_68 <- teams_68 |>

mutate(talent = rnorm(10, 0, s_talent))

schedule_talent <- schedule |> inner_join(teams_68, join_by(lgID, Home == teamID)) |> rename(talent_home = talent) |> inner_join(teams_68, join_by(lgID, Visitor == teamID)) |> rename(talent_visitor = talent)

마지막으로, 모든 경기에 대해 홈팀과 원정팀의 재능을 확보하면 Bradley-Terry 모델을 적용하여 모든 경기에 대한 홈팀 승리 확률을 계산합니다. 이러한 확률은 변수 prob_home에 저장됩니다.

schedule_talent <- schedule_talent |> mutate( prob_home = exp(talent_home) /

(exp(talent_home) + exp(talent_visitor)) )

데이터 프레임 schedule_talent의 처음 6개 행이 아래에 표시되어 있으며 예정된 경기, 홈팀과 원정팀의 재능, 홈팀이 대결에서 승리할 확률을 볼 수 있습니다.

slice_head(schedule_talent, n = 6)

# A tibble: 6 x 6

lgID Home Visitor talent_home talent_visitor prob_home <chr> <chr> <chr> <dbl> <dbl> <dbl>

- 1 AL BAL BOS 0.197 0.269 0.482
- 2 AL BAL CAL 0.197 -0.230 0.605
- 3 AL BAL CHA 0.197 -0.00924 0.551
- 4 AL BAL CLE 0.197 -0.185 0.594
- 5 AL BAL DET 0.197 0.409 0.447
- 6 AL BAL MIN 0.197 -0.208 0.600

- 9.3.4 정규 시즌 시뮬레이션

전체 시즌 경기를 시뮬레이션하기 위해 일련의 동전 던지기를 수행하며, 홈팀이 승리할 확률은 승리 확률에 따라 달라집니다. rbinom() 함수는 예정된 1620 경기에 대해 동전 던지기를 수행하고, 그 결과는 0과 1의 연속입니다. if_else() 함수를 사용하여 결과가 1이면 승자 변수를 홈팀으로, 그렇지 않으면 원정팀으로 정의합니다.

schedule_talent <- schedule_talent |>

mutate( outcome = rbinom(nrow(schedule_talent), 1, prob_home), winner = if_else(outcome == 1, Home, Visitor)

)

첫 여섯 경기의 팀, 홈팀 승리 확률, 결과는 아래에 표시되어 있습니다.

schedule_talent |> select(Visitor, Home, prob_home, outcome, winner) |> slice_head(n = 6)

schedule_talent <- schedule_talent |> mutate( prob_home = exp(talent_home) /

(exp(talent_home) + exp(talent_visitor)) )

데이터 프레임 schedule_talent의 첫 여섯 행은 아래에 표시되어 있으며, 여기서 예정된 경기, 홈팀과 원정팀의 재능, 그리고 매치업에서 홈팀이 승리할 확률을 볼 수 있습니다.

slice_head(schedule_talent, n = 6)

# A tibble: 6 x 6

lgID Home Visitor talent_home talent_visitor prob_home <chr> <chr> <chr> <dbl> <dbl> <dbl>

- 1 AL BAL BOS 0.197 0.269 0.482
- 2 AL BAL CAL 0.197 -0.230 0.605
- 3 AL BAL CHA 0.197 -0.00924 0.551
- 4 AL BAL CLE 0.197 -0.185 0.594
- 5 AL BAL DET 0.197 0.409 0.447
- 6 AL BAL MIN 0.197 -0.208 0.600

  9.3.4 정규 시즌 시뮬레이션

전체 시즌 경기를 시뮬레이션하기 위해 일련의 동전 던지기를 수행하며, 홈팀이 승리할 확률은 승리 확률에 따라 달라집니다. rbinom() 함수는 예정된 1620 경기에 대해 동전 던지기를 수행하고, 그 결과는 0과 1의 연속입니다. if_else() 함수를 사용하여 결과가 1이면 승자 변수를 홈팀으로, 그렇지 않으면 원정팀으로 정의합니다.

schedule_talent <- schedule_talent |>

mutate( outcome = rbinom(nrow(schedule_talent), 1, prob_home), winner = if_else(outcome == 1, Home, Visitor)

)

첫 여섯 경기의 팀, 홈팀 승리 확률, 결과는 아래에 표시되어 있습니다.

schedule_talent |> select(Visitor, Home, prob_home, outcome, winner) |> slice_head(n = 6)

# A tibble: 6 x 5 Visitor Home prob_home outcome winner <chr> <chr> <dbl> <int> <chr>

- 1 BOS BAL 0.482 0 BOS
- 2 CAL BAL 0.605 0 CAL
- 3 CHA BAL 0.551 1 BAL
- 4 CLE BAL 0.594 1 BAL
- 5 DET BAL 0.447 0 DET
- 6 MIN BAL 0.600 0 MIN

이 특정 시뮬레이션 시즌 동안 팀들은 어떤 성과를 거두었을까요? group_by() 및 summarize() 함수를 사용하여 모든 팀의 승수를 찾습니다. 이 정보를 팀 이름과 함께 WIN 데이터 프레임에 수집하고, inner_join() 함수를 사용하여 시즌 결과와 팀 재능을 결합하여 results 데이터 프레임을 생성합니다.

results <- schedule_talent |> group_by(winner) |> summarize(Wins = n()) |> inner_join(teams_68, by = c("winner" = "teamID"))

9.3.5 포스트시즌 시뮬레이션

정규 시즌 이후에 포스트시즌 시리즈를 시뮬레이션할 수 있습니다. 리그 챔피언십을 시뮬레이션하는 win_league() 함수를 작성합니다. 입력값은 팀과 총 승수의 데이터 프레임 res입니다. min_rank() 함수를 사용하여 각 리그에서 많은 승수를 기록한 팀을 식별합니다. 한 팀이 최대 승수를 가진 경우, 해당 팀에 대해 1이 되는 지시 변수 is_winner_lg가 생성됩니다. 두 개 이상의 팀이 총 승수에서 동률을 이루는 것을 피하기 위해, runif() 함수를 사용하여 모든 팀의 총 승수에 무작위 타이브레이커 수치(1 미만)를 임의로 더합니다.

win_league <- function(res) {

res |> group_by(lgID) |> mutate(

tiebreaker = runif(n = length(talent)), wins_total = Wins + tiebreaker, rank = min_rank(desc(wins_total)), is_winner_lg = wins_total == max(wins_total)

) }

포스트시즌을 시뮬레이션하기 위해 새로운 변수 is_winner_ws를 채웁니다. 이것은 월드 시리즈 우승팀을 나타내는 지시표입니다. win_league()를 적용하여,

각 리그의 우승팀을 찾습니다. 동전을 일곱 번 던져(rmultinom()) 월드 시리즈를 시뮬레이션하며, 승리 확률은 exp(talent)에 비례합니다. is_winner_ws는 과반수의 경기를 승리한 팀을 나타냅니다.

sim_one <- win_league(results) ws_winner <- sim_one |>

filter(is_winner_lg) |> ungroup() |> mutate(

outcome = as.numeric(rmultinom(1, 7, exp(talent))), is_winner_ws = outcome > 3

) |> filter(is_winner_ws) |> select(winner, is_winner_ws)

sim_one |> left_join(ws_winner, by = c("winner")) |> replace_na(list(is_winner_ws = 0))

# A tibble: 20 x 9 # Groups: lgID [2]

winner Wins lgID talent tiebreaker wins_total rank <chr> <int> <fct> <dbl> <dbl> <dbl> <int>

- 1 ATL 83 NL -0.215 0.867 83.9 5
- 2 BAL 86 AL 0.197 0.0260 86.0 3
- 3 BOS 99 AL 0.269 0.354 99.4 2
- 4 CAL 62 AL -0.230 0.936 62.9 10
- 5 CHA 85 AL -0.00924 0.246 85.2 4
- 6 CHN 85 NL -0.107 0.829 85.8 4
- 7 CIN 81 NL -0.0612 0.264 81.3 6
- 8 CLE 71 AL -0.185 0.290 71.3 9
- 9 DET 100 AL 0.409 0.0841 100. 1
- 10 HOU 90 NL -0.0871 0.845 90.8 2
- 11 LAN 63 NL -0.326 0.729 63.7 10
- 12 MIN 74 AL -0.208 0.667 74.7 7
- 13 NYA 78 AL 0.0424 0.422 78.4 6
- 14 NYN 80 NL 0.100 0.569 80.6 7
- 15 OAK 82 AL 0.287 0.927 82.9 5
- 16 PHI 93 NL 0.265 0.819 93.8 1
- 17 PIT 71 NL -0.146 0.0460 71.0 9
- 18 SFN 88 NL 0.249 0.119 88.1 3
- 19 SLN 76 NL -0.348 0.425 76.4 8
- 20 WS2 73 AL 0.0842 0.730 73.7 8 # i 2 more variables: is_winner_lg <lgl>, is_winner_ws <lgl>

각 리그의 우승팀을 찾습니다. 동전을 일곱 번 던져(rmultinom()) 월드 시리즈를 시뮬레이션하며, 승리 확률은 exp(talent)에 비례합니다. is_winner_ws는 과반수의 경기를 승리한 팀을 나타냅니다.

sim_one <- win_league(results) ws_winner <- sim_one |>

filter(is_winner_lg) |> ungroup() |> mutate(

outcome = as.numeric(rmultinom(1, 7, exp(talent))), is_winner_ws = outcome > 3

) |> filter(is_winner_ws) |> select(winner, is_winner_ws)

sim_one |> left_join(ws_winner, by = c("winner")) |> replace_na(list(is_winner_ws = 0))

# A tibble: 20 x 9 # Groups: lgID [2]

winner Wins lgID talent tiebreaker wins_total rank <chr> <int> <fct> <dbl> <dbl> <dbl> <int>

- 1 ATL 83 NL -0.215 0.867 83.9 5
- 2 BAL 86 AL 0.197 0.0260 86.0 3
- 3 BOS 99 AL 0.269 0.354 99.4 2
- 4 CAL 62 AL -0.230 0.936 62.9 10
- 5 CHA 85 AL -0.00924 0.246 85.2 4
- 6 CHN 85 NL -0.107 0.829 85.8 4
- 7 CIN 81 NL -0.0612 0.264 81.3 6
- 8 CLE 71 AL -0.185 0.290 71.3 9
- 9 DET 100 AL 0.409 0.0841 100. 1
- 10 HOU 90 NL -0.0871 0.845 90.8 2
- 11 LAN 63 NL -0.326 0.729 63.7 10
- 12 MIN 74 AL -0.208 0.667 74.7 7
- 13 NYA 78 AL 0.0424 0.422 78.4 6
- 14 NYN 80 NL 0.100 0.569 80.6 7
- 15 OAK 82 AL 0.287 0.927 82.9 5
- 16 PHI 93 NL 0.265 0.819 93.8 1
- 17 PIT 71 NL -0.146 0.0460 71.0 9
- 18 SFN 88 NL 0.249 0.119 88.1 3
- 19 SLN 76 NL -0.348 0.425 76.4 8
- 20 WS2 73 AL 0.0842 0.730 73.7 8 # i 2 more variables: is_winner_lg <lgl>, is_winner_ws <lgl>

  9.3.6 한 시즌 시뮬레이션 함수

make_schedule() 함수와 win_league() 함수를 포함한 이 모든 명령어를 abdwr3edata 패키지에서 찾을 수 있는 하나의 one_simulation_68() 함수 안에 배치하는 것이 편리합니다. 유일한 입력값은 정규 재능 분포의 퍼짐 정도를 설명하는 표준편차 s_talent입니다. 출력값은 팀, 재능, 시즌 승수, 그리고 포스트시즌 성공 여부를 포함하는 데이터 프레임입니다. 한 시즌을 시뮬레이션하는 것을 보여드리고, 반환되는 데이터 프레임인 results_1을 표시하겠습니다.

library(abdwr3edata) set.seed(111653) results_1 <- one_simulation_68(0.20) results_1

# A tibble: 20 x 6

Team Wins League Talent Winner.Lg Winner.WS <chr> <int> <dbl> <dbl> <dbl> <dbl>

- 1 SFN 93 1 -0.0591 1 0
- 2 PHI 93 1 -0.00979 0 0
- 3 LAN 87 1 0.00406 0 0
- 4 HOU 84 1 -0.117 0 0
- 5 SLN 80 1 -0.128 0 0
- 6 ATL 79 1 -0.100 0 0
- 7 CIN 79 1 -0.235 0 0
- 8 NYN 76 1 -0.269 0 0
- 9 CHN 76 1 -0.0199 0 0
- 10 PIT 63 1 -0.313 0 0
- 11 NYA 100 2 0.284 1 1
- 12 DET 93 2 0.379 0 0
- 13 CHA 87 2 0.139 0 0
- 14 BOS 86 2 -0.102 0 0
- 15 WS2 84 2 0.0915 0 0
- 16 OAK 82 2 -0.0622 0 0
- 17 CAL 78 2 -0.129 0 0
- 18 BAL 74 2 -0.0728 0 0
- 19 MIN 65 2 -0.207 0 0
- 20 CLE 61 2 -0.292 0 0

순위표 형식으로 시즌 승수를 더 익숙하게 나타내기 위해 display_standings()라는 새로운 함수를 작성합니다. 이 함수의 입력값은 results_1 데이터 프레임과 리그 표시자입니다.

display_standings <- function(data, league) {

data |> filter(League == league) |> select(Team, Wins) |> mutate(Losses = 162 - Wins) |> arrange(desc(Wins))

}

그런 다음 map()을 사용하여 각 리그에 대해 한 번씩 이 함수를 두 번 적용하고, bind_cols() 함수를 사용하여 두 순위표를 하나의 데이터 프레임으로 결합합니다. 리그 우승팀과 월드시리즈 우승팀도 아래에 표시됩니다.

map(1:2, display_standings, data = results_1) |> bind_cols()

# A tibble: 10 x 6

Team...1 Wins...2 Losses...3 Team...4 Wins...5 Losses...6 <chr> <int> <dbl> <chr> <int> <dbl>

1 SFN 93 69 NYA 100 62 2 PHI 93 69 DET 93 69 3 LAN 87 75 CHA 87 75 4 HOU 84 78 BOS 86 76 5 SLN 80 82 WS2 84 78 6 ATL 79 83 OAK 82 80 7 CIN 79 83 CAL 78 84 8 NYN 76 86 BAL 74 88 9 CHN 76 86 MIN 65 97

10 PIT 63 99 CLE 61 101

results_1 |> filter(Winner.Lg == 1) |> select(Team, Winner.WS)

# A tibble: 2 x 2 Team Winner.WS <chr> <dbl>

1 SFN 0 2 NYA 1

이 특정 시뮬레이션 시즌에서는 필라델피아 필리스(PHI)와 샌프란시스코 자이언츠(SFN)가 93승으로 내셔널 리그 우승을 공동으로 차지했고, 뉴욕 양키스(NYA)가 100승으로 아메리칸 리그에서 우승했습니다. 월드 시리즈에서는 양키스가 자이언츠를 꺾었습니다. 이 시즌에서 뛰어난 재능을 가진 팀은 디트로이트(재능 값 0.379)였으나 그들은 ALCS에서 패배했습니다. 즉 "야구에서 최고의 팀"이 이 시뮬레이션 시즌 동안 성공적이지는 않았습니다. 최고의 팀이 일반적으로 월드 시리즈에서 우승하는지는 곧 살펴볼 것입니다.

display_standings <- function(data, league) {

data |> filter(League == league) |> select(Team, Wins) |> mutate(Losses = 162 - Wins) |> arrange(desc(Wins))

}

그런 다음 map()을 사용하여 각 리그에 대해 한 번씩 이 함수를 두 번 적용하고, bind_cols() 함수를 사용하여 두 순위표를 하나의 데이터 프레임으로 결합합니다. 리그 우승팀과 월드시리즈 우승팀도 아래에 표시됩니다.

map(1:2, display_standings, data = results_1) |> bind_cols()

# A tibble: 10 x 6

Team...1 Wins...2 Losses...3 Team...4 Wins...5 Losses...6 <chr> <int> <dbl> <chr> <int> <dbl>

- 1 SFN 93 69 NYA 100 62
- 2 PHI 93 69 DET 93 69
- 3 LAN 87 75 CHA 87 75
- 4 HOU 84 78 BOS 86 76
- 5 SLN 80 82 WS2 84 78
- 6 ATL 79 83 OAK 82 80
- 7 CIN 79 83 CAL 78 84
- 8 NYN 76 86 BAL 74 88
- 9 CHN 76 86 MIN 65 97
- 10 PIT 63 99 CLE 61 101

results_1 |> filter(Winner.Lg == 1) |> select(Team, Winner.WS)

# A tibble: 2 x 2 Team Winner.WS <chr> <dbl>

- 1 SFN 0
- 2 NYA 1

이 특정 시뮬레이션 시즌에서는 필라델피아 필리스(PHI)와 샌프란시스코 자이언츠(SFN)가 93승으로 내셔널 리그 우승을 공동으로 차지했고, 뉴욕 양키스(NYA)가 100승으로 아메리칸 리그에서 우승했습니다. 월드 시리즈에서는 양키스가 자이언츠를 꺾었습니다. 이 시즌에서 뛰어난 재능을 가진 팀은 디트로이트(재능 값 0.379)였으나 그들은

![image 74](images/imageFile74.png)

그림 9.1 여러 시즌 시뮬레이션에 대한 팀의 재능과 승수의 산점도입니다.

ALCS에서 패배했습니다. 즉 "야구에서 최고의 팀"이 이 시뮬레이션 시즌 동안 성공적이지는 않았습니다. 최고의 팀이 일반적으로 월드 시리즈에서 우승하는지는 곧 살펴볼 것입니다.

9.3.7 많은 시즌 시뮬레이션

많은 야구 시즌을 시뮬레이션하면 팀의 능력과 시즌 성적 사이의 관계에 대해 배울 수 있습니다. 1000개의 시즌을 시뮬레이션하기 위해, rep() 함수를 사용하여 길이 1000의 벡터를 생성하고, map()을 사용하여 이 벡터에 one_simulation_68() 함수를 반복적으로 적용한 다음 그 결과를 many_results에 저장합니다.

set.seed(111653) many_results <- rep(0.20, 1000) |>

map(one_simulation_68) |> list_rbind()

many_results 데이터 프레임은 1000 × 20 = 20,000개 팀의 재능 수치와 승수를 포함합니다. alpha = 0.05 인수를 사용한 geom_point() 함수를 활용하여 그림 9.1과 같이 재능과 승수의 "부드러운" 산점도를 작성합니다.

ggplot(many_results, aes(Talent, Wins)) + geom_point(alpha = 0.05)

![image 75](images/imageFile75.png)

그림 9.2 시뮬레이션에서 평균적인 재능을 가진 팀들의 승수 히스토그램입니다.

예상대로 그래프에는 양의 추세가 있으며, 이는 더 나은 팀이 더 많은 경기를 이기는 경향이 있음을 나타냅니다. 그러나 산점도에는 수직 방향의 퍼짐이 큰데, 이는 재능과 승수 간의 관계가 강하지 않음을 의미합니다.

마지막 요점을 강조하기 위해, 재능 수치가 -0.05에서 0.05 사이인 "평균적인" 팀에 초점을 맞춰보겠습니다. filter() 함수를 사용하여 이 평균적인 팀들의 재능과 승수 데이터를 분리합니다. 이 팀들의 시즌 승수 히스토그램이 그림 9.2에 표시되어 있습니다.

many_results |> filter(Talent > -0.05, Talent < 0.05) |> ggplot(aes(Wins)) + geom_histogram(color = crcblue, fill = "white")

이 평균적인 팀들은 약 81경기를 이길 것으로 예상됩니다. 하지만 놀라운 것은 승수 합계의 변동성입니다. 평균적인 팀들이 70승에서 90승 사이의 승수 합계를 기록하는 경우가 규칙적으로 발생하며, 100승에 가까운 승수 합계를 가지는 것도 가능합니다(비록 가능성은 낮지만).

팀의 재능과 포스트시즌 성공 사이에는 어떤 관계가 있을까요? 먼저 팀의 재능(변수 Talent)과 리그 우승(변수 Winner.Lg) 사이의 관계를 살펴보겠습니다. Winner.Lg는 이진(0 또는 1) 변수이므로, 이 관계를 나타내기 위한 일반적인 접근 방식은 로지스틱 모델입니다. 이는 반응 변수가 연속형이 아닌 이진형인 일반적인 회귀 모델의 일반화입니다. family 인수를 binomial로 설정한 glm() 함수를 사용하여 로지스틱 모델을 적합시킵니다. 출력 결과는

그림 9.2 시뮬레이션에서 평균적인 재능을 가진 팀들의 승수 히스토그램입니다.

예상대로 그래프에는 양의 추세가 있으며, 이는 더 나은 팀이 더 많은 경기를 이기는 경향이 있음을 나타냅니다. 그러나 산점도에는 수직 방향의 퍼짐이 큰데, 이는 재능과 승수 간의 관계가 강하지 않음을 의미합니다.

마지막 요점을 강조하기 위해, 재능 수치가 -0.05에서 0.05 사이인 "평균적인" 팀에 초점을 맞춰보겠습니다. filter() 함수를 사용하여 이 평균적인 팀들의 재능과 승수 데이터를 분리합니다. 이 팀들의 시즌 승수 히스토그램이 그림 9.2에 표시되어 있습니다.

many_results |> filter(Talent > -0.05, Talent < 0.05) |> ggplot(aes(Wins)) + geom_histogram(color = crcblue, fill = "white")

이 평균적인 팀들은 약 81경기를 이길 것으로 예상됩니다. 하지만 놀라운 것은 승수 합계의 변동성입니다. 평균적인 팀들이 70승에서 90승 사이의 승수 합계를 기록하는 경우가 규칙적으로 발생하며, 100승에 가까운 승수 합계를 가지는 것도 가능합니다(비록 가능성은 낮지만).

팀의 재능과 포스트시즌 성공 사이에는 어떤 관계가 있을까요? 먼저 팀의 재능(변수 Talent)과 리그 우승(변수 Winner.Lg) 사이의 관계를 살펴보겠습니다. Winner.Lg는 이진(0 또는 1) 변수이므로, 이 관계를 나타내기 위한 일반적인 접근 방식은 로지스틱 모델입니다. 이는 반응 변수가 연속형이 아닌 이진형인 일반적인 회귀 모델의 일반화입니다. family 인수를 binomial로 설정한 glm() 함수를 사용하여 로지스틱 모델을 적합시킵니다. 출력 결과는
fit1 변수에 저장됩니다. 비슷한 방식으로 로지스틱 모델을 사용하여 월드시리즈 우승(변수 Winner.WS)과 재능 간의 관계를 모델링합니다. 출력 결과는 fit2 변수에 저장됩니다.

```r
fit1 <- glm( Winner.Lg ~ Talent, data = many_results, family = binomial

) fit2 <- glm(

Winner.WS ~ Talent, data = many_results, family = binomial

)
```

로지스틱 모델은 다음과 같은 형태를 가집니다.

p =

exp(a + bT) 1 + exp(a + bT)

,

여기서 T는 팀의 재능, (a,b)는 회귀 계수, p는 사건의 확률입니다.

다음 코드에서는 팀 재능의 그럴듯한 값들로 이루어진 벡터를 생성하고 이를 talent_values 벡터에 저장합니다. 그런 다음 predict() 함수를 사용하여 페넌트레이스 우승 확률과 월드시리즈 우승 확률의 예측값을 계산합니다. (type = "response" 인수는 a + bT 값을 확률 척도로 매핑합니다.) 그리고 geom_line() 함수를 사용하여 재능과 확률의 그래프를 구성하며, 여기서 선의 색상은 성취 유형에 해당합니다. 완성된 그래프는 그림 9.3에 표시되어 있습니다.

```r
tdf <- tibble( Talent = seq(-0.4, 0.4, length.out = 100)

) tdf |>

mutate( Pennant = predict(fit1, newdata = tdf, type = "response"), `World Series` = predict(fit2, newdata = tdf, type = "response")

) |> pivot_longer(

cols = -Talent, names_to = "Outcome", values_to = "Probability"

) |> ggplot(aes(Talent, Probability, color = Outcome)) + geom_line() + ylim(0, 1) + scale_color_manual(values = crc_fc)
```

![image 76](images/imageFile76.png)

그림 9.3 다양한 재능을 가진 팀들의 페넌트레이스 우승 및 월드시리즈 우승 확률.

예상대로, 팀이 페넌트레이스에서 우승할 확률(실선)은 재능에 비례하여 증가합니다. T = 0인 평균적인 팀은 페넌트레이스에서 우승할 확률이 적습니다. 재능이 0.4에 가까운 우수한 팀은 페넌트레이스에서 우승할 확률이 약 60%입니다. 월드시리즈 우승 확률(점선으로 표시)은 페넌트레이스 우승 확률보다 실질적으로 더 낮습니다. 예를 들어, 이 우수한(T = 0.4) 팀이 월드시리즈에서 우승할 확률은 약 35%에 불과합니다. 사실, 월드시리즈 우승 팀이 뛰어난 재능(큰 T 값)을 가진 팀이 아닐 가능성이 높다는 것을 증명할 수 있습니다.

9.4 추가 참고 문헌

마르코프 연쇄 확률 모델에 대한 일반적인 설명은 Kemeny와 Snell (1960)에 포함되어 있습니다. Pankin (1987)과 Bukiet, Harold, Palacios (1997)는 야구를 모델링하기 위해 마르코프 연쇄를 사용하는 것을 설명합니다. Albert (2017)의 9장은 마르코프 연쇄에 대한 입문적인 설명을 제공하고 1987년 시즌 데이터를 사용하여 전이 행렬의 구성 및 사용을 설명합니다. Bradley-Terry 모델(Bradley 및 Terry 1952)은 쌍체 비교를 위한 인기 있는 통계 모델입니다. Albert와 Bennett (2003)의 9장은 야구 팀 경쟁을 위한 Bradley-Terry 모델의 적용을 설명합니다.

시뮬레이션에서 R의 사용은 Albert와 Rizzo (2012)의 11장에 소개되어 있습니다. Lopez, Matthews, Baumer (2018)는 최고의 팀이 얼마나 자주 이기는지에 대한 질문을 해결하기 위해 Bradley-Terry 상태 공간 모델을 사용합니다.

그림 9.3 다양한 재능을 가진 팀들의 페넌트레이스 우승 및 월드시리즈 우승 확률.

예상대로, 팀이 페넌트레이스에서 우승할 확률(실선)은 재능에 비례하여 증가합니다. T = 0인 평균적인 팀은 페넌트레이스에서 우승할 확률이 적습니다. 재능이 0.4에 가까운 우수한 팀은 페넌트레이스에서 우승할 확률이 약 60%입니다. 월드시리즈 우승 확률(점선으로 표시)은 페넌트레이스 우승 확률보다 실질적으로 더 낮습니다. 예를 들어, 이 우수한(T = 0.4) 팀이 월드시리즈에서 우승할 확률은 약 35%에 불과합니다. 사실, 월드시리즈 우승 팀이 뛰어난 재능(큰 T 값)을 가진 팀이 아닐 가능성이 높다는 것을 증명할 수 있습니다.

- 9.4 추가 참고 문헌

마르코프 연쇄 확률 모델에 대한 일반적인 설명은 Kemeny와 Snell (1960)에 포함되어 있습니다. Pankin (1987)과 Bukiet, Harold, Palacios (1997)는 야구를 모델링하기 위해 마르코프 연쇄를 사용하는 것을 설명합니다. Albert (2017)의 9장은 마르코프 연쇄에 대한 입문적인 설명을 제공하고 1987년 시즌 데이터를 사용하여 전이 행렬의 구성 및 사용을 설명합니다. Bradley-Terry 모델(Bradley 및 Terry 1952)은 쌍체 비교를 위한 인기 있는 통계 모델입니다. Albert와 Bennett (2003)의 9장은 야구 팀 경쟁을 위한 Bradley-Terry 모델의 적용을 설명합니다.

9.5 연습 문제

- 1. 단순한 마르코프 연쇄

이닝의 아웃 카운트에만 관심이 있다고 가정해 보겠습니다. 한 이닝에는 네 가지 가능한 상태(무사, 1사, 2사, 3사)가 있으며 각 타석마다 이러한 상태 사이를 이동합니다. 각 타석에서 아웃 카운트가 증가하지 않을 확률은 0.3이고, 아웃 카운트가 하나 증가할 확률은 0.7이라고 가정합니다. 다음 R 코드는 이 마르코프 연쇄의 전이 확률을 행렬 P에 넣습니다.

```r
P <- matrix(c(.3, .7, 0, 0, 0, .3, .7, 0, 0, 0, .3, .7, 0, 0, 0, 1), 4, 4, byrow = TRUE)
```

a. 행렬 P를 자기 자신 P와 곱하여 행렬 P2를 얻는다면 다음과 같습니다.

```r
P2 <- P %*% P
```

P2의 첫 번째 행은 두 타석 후 무사에서 4가지 상태 각각으로 이동할 확률을 제공합니다. P2를 계산하세요. 이 계산을 바탕으로 두 타석 후 무사에서 1사로 이동할 확률을 구하세요.

b. 기본 행렬 N은 다음과 같이 계산됩니다.

```r
N <- solve(diag(c(1, 1, 1)) - P[-4, -4])
```

첫 번째 행은 한 이닝에서 무사, 1사, 2사에서의 평균 타석 수를 제공합니다. N을 계산하고 이 모델에서 한 이닝의 평균 타석 수를 구하세요.

- 2. 단순한 마르코프 연쇄, 계속

다음 simulate_half_inning() 함수는 연습 문제 1에서 설명한 마르코프 연쇄 모델의 단일 이닝(초 또는 말) 타석 수를 시뮬레이션하며, 여기서 입력 P는 전이 확률 행렬입니다.

```r
simulate_half_inning <- function(P) { s <- 1 path <- NULL while(s < 4){

s_new <- sample(1:4, 1, prob = P[s, ]) path <- c(path, s_new) s <- s_new

} length(path)

}
```

- a. map() 함수를 사용하여 이 마르코프 연쇄의 이닝(초/말)을 1000번 시뮬레이션하고 이러한 시뮬레이션된 이닝의 길이를 벡터 lengths에 저장하세요.
- b. 이 시뮬레이션 결과를 사용하여 이닝(초/말)이 정확히 4개의 타석을 포함할 확률을 구하세요.
- c. 시뮬레이션 결과를 사용하여 이닝(초/말)의 평균 타석 수를 구하세요. 당신의 답을 연습 문제 1의 (b) 부분의 정확한 답과 비교하세요.

- 3. 이닝(초/말) 시뮬레이션

  9.2.4절에서는 2016년 시즌 데이터를 사용하여 24가지 가능한 주자-아웃 상황 각각에 대해 계산된 기대 득점을 보여줍니다. 이러한 값이 시즌에 따라 어떻게 달라질 수 있는지 알아보기 위해 Retrosheet에서 1968년 시즌의 플레이 바이 플레이 데이터를 다운로드하고, 확률 전이 행렬을 구성하며, 24가지 상황 각각에서 10,000번의 이닝(초/말)을 시뮬레이션하고, 득점 기대 행렬을 계산하세요. 이 1968년 득점 기대 행렬을 2016년 데이터를 사용하여 계산된 행렬과 비교하세요.

- 4. 1950년 시즌 시뮬레이션

내셔널 리그의 1950년 정규 시즌을 시뮬레이션하는 데 관심이 있다고 가정해 보겠습니다. 이 시즌에서 팀 약어는 "PHI", "BRO", "NYG", "BSN", "STL", "CIN", "CHC", "PIT"였으며 각 팀은 다른 모든 팀과 22경기(각 구장에서 11경기)를 치렀습니다.

- a. make_schedule() 함수를 사용하여 이 NL 시즌의 경기 일정을 구성하세요.
- b. 팀 재능이 평균이 0이고 표준 편차가 0.25인 정규 분포를 따른다고 가정합니다. Bradley-Terry 모델을 사용하여 일정의 모든 경기에 대한 홈팀 승리 확률을 할당하세요.
- c. rbinom() 함수를 사용하여 NL 1950년 시즌의 전체 616경기 결과를 시뮬레이션하세요.
- d. 시뮬레이션에서 모든 팀의 시즌 승수를 계산하세요.

```r
simulate_half_inning <- function(P) { s <- 1 path <- NULL while(s < 4){

s_new <- sample(1:4, 1, prob = P[s, ]) path <- c(path, s_new) s <- s_new

} length(path)

}
```

- a. map() 함수를 사용하여 이 마르코프 연쇄의 이닝(초/말)을 1000번 시뮬레이션하고 이러한 시뮬레이션된 이닝의 길이를 벡터 lengths에 저장하세요.
- b. 이 시뮬레이션 결과를 사용하여 이닝(초/말)이 정확히 4개의 타석을 포함할 확률을 구하세요.
- c. 시뮬레이션 결과를 사용하여 이닝(초/말)의 평균 타석 수를 구하세요. 당신의 답을 연습 문제 1의 (b) 부분의 정확한 답과 비교하세요.

- 3. 이닝(초/말) 시뮬레이션

  9.2.4절에서는 2016년 시즌 데이터를 사용하여 24가지 가능한 주자-아웃 상황 각각에 대해 계산된 기대 득점을 보여줍니다. 이러한 값이 시즌에 따라 어떻게 달라질 수 있는지 알아보기 위해 Retrosheet에서 1968년 시즌의 플레이 바이 플레이 데이터를 다운로드하고, 확률 전이 행렬을 구성하며, 24가지 상황 각각에서 10,000번의 이닝(초/말)을 시뮬레이션하고, 득점 기대 행렬을 계산하세요. 이 1968년 득점 기대 행렬을 2016년 데이터를 사용하여 계산된 행렬과 비교하세요.

- 4. 1950년 시즌 시뮬레이션

내셔널 리그의 1950년 정규 시즌을 시뮬레이션하는 데 관심이 있다고 가정해 보겠습니다. 이 시즌에서 팀 약어는 "PHI", "BRO", "NYG", "BSN", "STL", "CIN", "CHC", "PIT"였으며 각 팀은 다른 모든 팀과 22경기(각 구장에서 11경기)를 치렀습니다.

- a. make_schedule() 함수를 사용하여 이 NL 시즌의 경기 일정을 구성하세요.
- b. 팀 재능이 평균이 0이고 표준 편차가 0.25인 정규 분포를 따른다고 가정합니다. Bradley-Terry 모델을 사용하여 일정의 모든 경기에 대한 홈팀 승리 확률을 할당하세요.
- c. rbinom() 함수를 사용하여 NL 1950년 시즌의 전체 616경기 결과를 시뮬레이션하세요.
- d. 시뮬레이션에서 모든 팀의 시즌 승수를 계산하세요.

- 5. 1950년 시즌 시뮬레이션, 계속

- a. 연습 문제 4에서 설명한 시뮬레이션 체계를 수행하는 함수를 작성하세요. 이 함수는 뛰어난 재능을 가진 팀과 많은 승리를 거둔 팀을 반환하도록 합니다. (리그 페넌트레이스에서 동률이 발생할 경우, 함수가 무작위로 최고의 팀 중 하나를 반환하도록 하세요.)
- b. 이 시뮬션을 1000 시즌 동안 반복하여 모든 시즌에 대해 재능 있는 팀과 성공적인 팀을 수집하세요.
- c. 시뮬레이션을 바탕으로 재능 있는 팀이 페넌트레이스에서 우승할 확률은 얼마입니까?

- 6. 월드시리즈 시뮬레이션

- a. 월드시리즈를 시뮬레이션하는 함수를 작성하세요. 입력은 단일 경기에서 AL 팀이 NL 팀을 이길 확률 p입니다.
- b. 재능이 0.40인 AL 팀이 재능이 0.25인 NL 팀과 경기한다고 가정합니다. Bradley-Terry 모델을 사용하여 AL이 경기에서 승리할 확률 p를 결정하세요.
- c. (b) 부분에서 결정된 p 값을 사용하여 1000번의 월드시리즈를 시뮬레이션하고 AL 팀이 월드시리즈에서 우승할 확률을 구하세요.
- d. 동일한 재능을 가진 AL 팀과 NL 팀에 대해 (b) 및 (c) 부분을 반복하세요.

### 10 연쇄적인 성과 탐구

10.1 서론

야구에서 흥미로운 현상 중 하나는 타자와 투수의 연쇄적인 또는 뜨겁거나 차가운 성과입니다. 시즌의 특정 기간 동안 특정 선수는 높은 타율을 기록할 것이며, 다른 기간에는 선수가 "슬럼프"에 빠져 모든 타구가 아웃으로 처리되는 것처럼 보일 것입니다. 이 장에서는 R을 사용하여 연쇄적인 타격 성과를 탐구할 것입니다.

야구 역사상 위대한 타격 성취 중 하나는 조 디마지오의 56경기 연속 안타이며, 10.2절에서는 1941년 시즌에 대한 디마지오의 경기별 타격을 탐구합니다. R 함수를 사용하여 디마지오의 모든 연속 안타를 찾고, 짧은 시간 간격 동안 디마지오의 타율을 탐구하기 위해 이동 평균 함수를 사용합니다. Retrosheet 플레이 바이 플레이 데이터는 모든 타석에서의 타자 성과를 기록하며, 우리는 10.3절에서 이 데이터를 사용하여 개별 타석에서의 연속 안타를 탐구합니다. 타자가 "20타수 무안타" 슬럼프를 겪고 있다고 가정해 보겠습니다. 우리가 놀라야 할까요? 이 질문에 대답하는 한 가지 방법은 특정 야구 시즌의 모든 타자에 대해 긴 타격 슬럼프를 찾는 것입니다. 이 타격 슬럼프의 규모를 이해하는 두 번째 방법은 이 타격을 무작위 모델에서의 슬럼프 패턴과 대조하는 것입니다. 우리는 안타와 아웃의 무작위 패턴을 시뮬레이션하는 방법을 설명하고 이 방법을 사용하여 특정 선수가 타격 시퀀스에서 우연에 의해 기대되는 것보다 더 많은 연쇄성을 나타내는지 평가합니다.

연쇄성에 대한 이러한 논의는 안타와 아웃의 패턴에 중점을 두며, 확실히 타석의 질은 단순히 안타를 치는 것 이상에 달려 있습니다. 10.4절에서는 타구의 발사 속도를 사용한 연쇄성 패턴에 대해 논의합니다. 시즌 동안 5경기 그룹에 걸친 선수들의 평균 발사 속도를 살펴봅니다. 연쇄적인 타격 행동을 설명하는 한 가지 방법은 5경기 평균 발사 속도 값의 변동성을 살펴보는 것입니다. 이 연쇄성 척도를 사용하여 2016년 시즌 동안 연쇄적인 타자를 식별합니다.

DOI: 10.1201/9781032668239-10 234
