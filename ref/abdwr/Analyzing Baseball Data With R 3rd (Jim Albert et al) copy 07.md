### 7. 포수 프레이밍

- 7.1 소개

이 장에서는 2022 시즌의 스탯캐스트 데이터를 사용하여 포수 프레이밍 능력에 대한 개념을 탐구합니다.

세이버메트릭스에서 포수 프레이밍 능력에 대한 이야기는 흥미롭습니다. 역사적으로 스카우트와 코치들은 특정 포수들에게 심판을 위한 투구를 "프레이밍"하는 능력이 있다고 주장했습니다. 글러브를 비교적 가만히 유지함으로써 기술적으로 스트라이크 존을 벗어났더라도 심판이 투구를 스트라이크로 판정하도록 속일 수 있다는 생각이었습니다(시각적인 설명은 Lindbergh (2013)를 참조하십시오). 세이버메트릭스 학자들은 일반적으로 이 기술의 존재와 영향 모두에 대해 회의적이었습니다. 포수 수비의 영향을 연구한 대부분의 사람들은 이것이 스카우트나 코치들이 믿었던 것만큼 가치 있지 않다는 결론을 내렸습니다.

문제의 일부는 2000년대 중반까지 투구 수준의 데이터를 구하기 어려웠다는 것입니다. PITCHf/x의 등장으로 이러한 더 세분화된 데이터에서 더 정교한 모델링 기법을 사용할 수 있게 되었습니다. 포수 프레이밍의 영향을 추정한 새로운 연구들은 지속적인 능력의 존재(즉, 프레이밍 수치가 좋은 포수들이 시간이 지나도 계속 좋은 성적을 유지함)와 그 효과의 크기(즉, 좋은 프레이머가 실제로 가치가 높음)를 모두 입증했습니다 (Turkenkopf 2008; Fast 2011; Brooks and Pavlidis 2014; Brooks, Pavilidis, and Judge 2015; Deshpande and Wyner 2017; Judge 2018).

이러한 새로운 발견은 야구 산업에 변화를 가져왔습니다. 타격 실력으로는 정당화되지 않는 호세 몰리나(Jose Molina)와 같은 수비형 포수들이 다년 계약을 맺기 시작했습니다. 마이너리그 훈련은 프레이밍 기술을 향상시키는 데 더 중점을 두었습니다. 물론, 메이저리그가 로봇에게 볼과 스트라이크 판정을 맡기기로 결정하는 즉시 이 포수 프레이밍 능력은 순식간에 사라질 것입니다.

이 문제는 세이버메트릭스적 사고가 데이터의 가용성과 모델링 기법의 정교함에 따라 어떻게 변할 수 있는지(그리고 어떻게 변하는지), 그리고 세이버메트릭스의 통찰력으로 인해 경기장의 게임이 어떻게 변할 수 있는지를 보여준다는 점에서 좋은 사례입니다.

DOI: 10.1201/9781032668239-7 164

투구 수준 데이터 획득하기 165

- 7.2 투구 수준 데이터 획득하기

볼이나 스트라이크로 판정된 "지켜본(taken)" 투구에 대한 스탯캐스트 데이터만 수집하고자 합니다. 여기서는 평가되지 않은 다음 코드를 사용하여 이 과정을 설명합니다. 2022 시즌의 스탯캐스트 데이터를 읽어오는 것으로 시작합니다. 1년 치 스탯캐스트 데이터를 얻는 방법에 대한 설명은 12.2절을 참조하십시오. mutate() 및 case_match() 함수를 사용하여 설명 변수를 "볼", "헛스윙" 및 "스트라이크 판정"의 세 가지 범주로 다시 코딩하는 Outcome 변수를 정의합니다. 또한 홈팀이 타격 중인지 여부를 나타내는 Home 변수와 볼 및 스트라이크 카운트를 제공하는 Count 변수도 정의합니다.

sc2022 <- here::here("data_large/statcast_rds/statcast_2022.rds") |> read_rds() sc2022 <- sc2022 |> mutate(

Outcome = case_match( description, c("ball", "blocked_ball", "pitchout",

"hit_by_pitch") ~ "ball",

c("swinging_strike", "swinging_strike_blocked", "foul", "foul_bunt", "foul_tip", "hit_into_play", "missed_bunt" ) ~ "swing",

"called_strike" ~ "called_strike"), Home = ifelse(inning_topbot == "Bot", 1, 0), Count = paste(balls, strikes, sep = "-")

)

filter() 함수를 사용하여, taken 데이터 프레임은 타자의 스윙이 없었던 투구들로 구성되므로 볼과 스트라이크 판정만 포함됩니다. select() 함수를 사용하여 이 데이터 세트에서 관심 있는 변수를 선택하고 write_rds() 함수는 taken 데이터 프레임을 압축된 형식으로 sc_taken_2022.rds 파일에 저장합니다.

taken <- sc2022 |> filter(Outcome != "swing")

taken_select <- select( taken, pitch_type, release_speed, description, stand, p_throws, Outcome, plate_x, plate_z, fielder_2_1, pitcher, batter, Count, Home, zone

) write_rds(

taken_select,

here::here("data/sc_taken_2022.rds"), compress = "xz"

)

이 데이터가 저장되면 read_rds() 함수를 사용하여 이 데이터를 R로 읽어올 수 있습니다. sample_n() 함수를 사용하여 추출한 이 데이터 세트의 50,000개 행의 무작위 샘플을 사용하는 데 중점을 둡니다.

sc_taken <- read_rds(here::here("data/sc_taken_2022.rds")) set.seed(12345) taken <- sample_n(sc_taken, 50000)

- 7.3 스트라이크 존은 어디인가요?

포수 프레이밍의 영향을 이해하기 위해서는 어떤 투구가 스트라이크로 판정될 확률을 특성화할 방법이 필요합니다. 스탯캐스트 데이터에서 각 투구는 Outcome 변수를 가지며, 스트라이크 판정의 경우 called_strike, 볼의 경우 ball입니다. 그림 7.1에 이러한 결과를 플로팅합니다.

![image 62](images/imageFile62.png)

그림 7.1 2022 시즌, 2000개 투구의 볼과 스트라이크 판정 위치의 산점도.

here::here("data/sc_taken_2022.rds"), compress = "xz"

)

이 데이터가 저장되면 read_rds() 함수를 사용하여 이 데이터를 R로 읽어올 수 있습니다. sample_n() 함수를 사용하여 추출한 이 데이터 세트의 50,000개 행의 무작위 샘플을 사용하는 데 중점을 둡니다.

sc_taken <- read_rds(here::here("data/sc_taken_2022.rds")) set.seed(12345) taken <- sample_n(sc_taken, 50000)

- 7.3 스트라이크 존은 어디인가요?

포수 프레이밍의 영향을 이해하기 위해서는 어떤 투구가 스트라이크로 판정될 확률을 특성화할 방법이 필요합니다. 스탯캐스트 데이터에서 각 투구는 Outcome 변수를 가지며, 스트라이크 판정의 경우 called_strike, 볼의 경우 ball입니다. 그림 7.1에 이러한 결과를 플로팅합니다.

그림 7.1 2022 시즌, 2000개 투구의 볼과 스트라이크 판정 위치의 산점도.

스트라이크 존은 어디인가요? 167 스트라이크 존에 던져진 투구는 스트라이크로 판정되는 경향이 있음에 유의하십시오. 또한 기술적으로 스트라이크 존을 벗어났더라도 많은 투구가 스트라이크로 판정됨을 알 수 있습니다.

plate_width <- 17 + 2 * (9/pi) k_zone_plot <- ggplot(

NULL, aes(x = plate_x, y = plate_z) ) +

geom_rect( xmin = -(plate_width/2)/12, xmax = (plate_width/2)/12, ymin = 1.5, ymax = 3.6, color = crcblue, alpha = 0

) + coord_equal() +

- scale_x_continuous( "수평 위치 (ft.)", limits = c(-2, 2)

) +

- scale_y_continuous( "수직 위치 (ft.)", limits = c(0, 5)


)

스트라이크 존이 어디인지 어떻게 알 수 있습니까? 규정에 따르면, 공의 일부만 홈 플레이트를 통과하면 투구가 스트라이크로 판정될 수 있습니다. 홈 플레이트의 너비는 17인치이고 공의 둘레는 9인치이므로, 우리 관점에서 스트라이크 존의 바깥쪽 가장자리는 약 ± 0.947피트입니다. 스트라이크 존의 위아래는 타자마다 다르지만 여기서는 비교적 관심이 적습니다. k_zone_plot 객체는 그림 7.1의 스탯캐스트 데이터 중 무작위로 추출한 2000개의 행을 플로팅하는 빈 ggplot2 객체입니다.

k_zone_plot %+% sample_n(taken, size = 2000) + aes(color = Outcome) + geom_point(alpha = 0.2) + scale_color_manual(values = crc_fc)

스트라이크 존에 대해 생각하는 또 다른 방법은 스탯캐스트에서 미리 정의한 영역을 기준으로 하는 것입니다. 스트라이크 존 자체는 3 × 3 격자로 나뉘며, 스트라이크 존 밖에 추가로 4개의 영역이 정의되어 있습니다. 먼저 해당 각 영역의 경계뿐만 아니라 스트라이크 판정이 나올 관찰 확률을 계산합니다. 이상치의 영향을 줄이기 위해 quantile() 함수를 사용합니다.

zones <- taken |> group_by(zone) |> summarize(

N = n(), right_edge = min(1.5, max(plate_x)), left_edge = max(-1.5, min(plate_x)), top_edge = min(5, quantile(plate_z, 0.95, na.rm = TRUE)), bottom_edge = max(0, quantile(plate_z, 0.05, na.rm = TRUE)), strike_pct = sum(Outcome == "called_strike") / n(), plate_x = mean(plate_x), plate_z = mean(plate_z)

)

그림 7.2에서는 해당 영역에서 투구된 공이 스트라이크로 판정될 확률과 함께 각 영역을 플로팅합니다. 미리 정의된 이러한 영역은 경계에 있는 투구("on the black")를 제외합니다.

library(ggrepel) k_zone_plot %+% zones +

geom_rect( aes(

xmax = right_edge, xmin = left_edge, ymax = top_edge, ymin = bottom_edge, fill = strike_pct, alpha = strike_pct

), color = "lightgray"

) + geom_text_repel(

size = 3, aes(

label = round(strike_pct, 2), color = strike_pct < 0.5

) ) + scale_fill_gradient(low = "gray70", high = crcblue) + scale_color_manual(values = crc_fc) + guides(color = FALSE, alpha = FALSE)

7.4 스트라이크 판정 확률 모델링

그림 7.2의 영역 기반 스트라이크 확률은 이산적인 성격 때문에 한계가 있습니다. 진정으로 원하는 것은 수평 및 수직 위치를 기반으로 모든 투구에 대해 예상 스트라이크 확률을 제공하는 모델입니다. 다음과 같이

zones <- taken |> group_by(zone) |> summarize(

N = n(), right_edge = min(1.5, max(plate_x)), left_edge = max(-1.5, min(plate_x)), top_edge = min(5, quantile(plate_z, 0.95, na.rm = TRUE)), bottom_edge = max(0, quantile(plate_z, 0.05, na.rm = TRUE)), strike_pct = sum(Outcome == "called_strike") / n(), plate_x = mean(plate_x), plate_z = mean(plate_z)

)

그림 7.2에서는 해당 영역에서 투구된 공이 스트라이크로 판정될 확률과 함께 각 영역을 플로팅합니다. 미리 정의된 이러한 영역은 경계에 있는 투구("on the black")를 제외합니다.

library(ggrepel) k_zone_plot %+% zones +

geom_rect( aes(

xmax = right_edge, xmin = left_edge, ymax = top_edge, ymin = bottom_edge, fill = strike_pct, alpha = strike_pct

), color = "lightgray"

) + geom_text_repel(

size = 3, aes(

label = round(strike_pct, 2), color = strike_pct < 0.5

) ) + scale_fill_gradient(low = "gray70", high = crcblue) + scale_color_manual(values = crc_fc) + guides(color = FALSE, alpha = FALSE)

![image 63](images/imageFile63.png)

- 그림 7.2 미리 정의된 스트라이크 존 영역에 투구된 공에 대한 스트라이크 확률.

이를 위해 일반화 가법 모델(generalized additive model, GAM)을 적합합니다. 이 모델은 위치에 대한 두 개의 설명 변수만 포함하면서 전체 영역에 걸쳐 부드러운 표면을 적합합니다. mgcv 패키지의 s() 함수는 어떤 변수에 대해 평활화(smoothing)가 발생할지 나타냅니다(plate_x 및 plate_z). 이항(binomial)으로 family 인수를 설정하여, Outcome == "called_strike"라는 부울 표현식으로 정의된 이진 응답 변수를 모델링하는 데 적절한 링크 함수(이 경우 로지스틱 함수)가 사용되도록 합니다.

library(mgcv) strike_mod <- gam(

Outcome == "called_strike" ~ s(plate_x, plate_z), family = binomial, data = taken

)

###### 7.4 스트라이크 판정 확률 모델링

그림 7.2의 영역 기반 스트라이크 확률은 이산적인 성격 때문에 한계가 있습니다. 진정으로 원하는 것은 수평 및 수직 위치를 기반으로 모든 투구에 대해 예상 스트라이크 확률을 제공하는 모델입니다. 다음과 같이

- 7.4.1 추정치 시각화하기

모델에서 생성된 추정치를 시각화하는 쉬운 방법은 적합된 값을 플로팅하는 것입니다. 여기서 broom 패키지의 augment() 함수를 사용하여 이러한 적합된 값을 계산하고 데이터 프레임에 추가합니다. type.predict

![image 64](images/imageFile64.png)

- 그림 7.3 일반화 가법 모델을 사용한 투구의 예상 스트라이크 확률.

인수는 확률 척도(즉, 응답 변수의)에서 추정치를 계산하도록 R에 지시합니다.

library(broom) hats <- strike_mod |>

augment(type.predict = "response")

다음으로, 이 새로운 데이터 프레임으로 k_zone_plot 객체를 업데이트하고 점을 추가한 다음(geom_point()), 색상 미적 요소를 방금 계산한 적합된 값(.fitted)에 매핑하기만 하면 됩니다. 그림 7.3은 이러한 데이터에서 GAM이 볼과 스트라이크 패턴을 효과적으로 매핑했음을 보여줍니다.

k_zone_plot %+% sample_n(hats, 10000) + geom_point(aes(color = .fitted), alpha = 0.1) + scale_color_gradient(low = "gray70", high = crcblue)

- 7.4.2 예상 표면 시각화하기

물론 우리가 만든 GAM은 연속적인 표면입니다. 처음부터 이러한 모델을 적합하는 이점 중 하나는 우리가 위치 좌표를 알고 있는 모든 투구에 대한 스트라이크 판정 확률을 추정할 수 있다는 것입니다. 이는 훈련 데이터 세트에 있는 것뿐만이 아닙니다.

그림 7.3 일반화 가법 모델을 사용한 투구의 예상 스트라이크 확률.

인수는 확률 척도(즉, 응답 변수의)에서 추정치를 계산하도록 R에 지시합니다.

library(broom) hats <- strike_mod |>

augment(type.predict = "response")

우리가 위치 좌표를 알고 있는 모든 투구에 대한 스트라이크 판정 확률 - 훈련 데이터 세트에 존재하는 것만이 아님.

가로 및 세로 좌표 쌍의 미세한 격자에서 예상 확률을 플로팅하여 모델을 표면으로 시각화할 수 있습니다. modelr 패키지에는 데이터와 관련된 값의 격자를 만드는 데 도움이 되는 data_grid() 및 seq_range()를 포함한 여러 함수가 있습니다.

library(modelr) grid <- taken |>

data_grid( plate_x = seq_range(plate_x, n = 100), plate_z = seq_range(plate_z, n = 100)

)

다음으로 이전과 같이 augment() 함수를 사용하되, 이번에는 newdata 인수를 방금 만든 격자점의 데이터 프레임으로 지정합니다. 그러면 각 좌표 쌍에 대한 예상 스트라이크 판정 확률이 포함된 10000행 데이터 프레임이 생성됩니다.

grid_hats <- strike_mod |> augment(type.predict = "response", newdata = grid)

다시 한 번 이 새로운 데이터로 k_zone_plot을 업데이트합니다. 그림 7.4의 geom_tile() 함수는 geom_contour()에 대한 좋은 대안을 제공합니다.

tile_plot <- k_zone_plot %+% grid_hats + geom_tile(aes(fill = .fitted), alpha = 0.7) + scale_fill_gradient(low = "gray92", high = crcblue)

tile_plot

다음으로, 이 새로운 데이터 프레임으로 k_zone_plot 객체를 업데이트하고 점을 추가한 다음(geom_point()), 색상 미적 요소를 방금 계산한 적합된 값(.fitted)에 매핑하기만 하면 됩니다. 그림 7.3은 이러한 데이터에서 GAM이 볼과 스트라이크 패턴을 효과적으로 매핑했음을 보여줍니다.

k_zone_plot %+% sample_n(hats, 10000) + geom_point(aes(color = .fitted), alpha = 0.1) + scale_color_gradient(low = "gray70", high = crcblue)

7.4.2 예상 표면 시각화하기

물론 우리가 만든 GAM은 연속적인 표면입니다. 처음부터 이러한 모델을 적합하는 이점 중 하나는 다음과 같이 추정할 수 있다는 것입니다.

7.4.3 투구하는 손 제어하기

규정집에 명시된 내용과 달리, 실제 스트라이크 존은 투수가 어느 손으로 던지는지, 타자가 타석 어느 쪽에 서는지에 따라 달라질 수 있다는 것은 이치에 맞습니다.

결과 데이터 프레임에는 plate_x 및 plate_z로 인코딩된 위치 데이터 외에도 p_throws 및 stand에 대한 변수가 있습니다. 이제 이 4개 변수에 걸쳐 다른 GAM을 적합할 수 있습니다. 이진 변수 p_throws 및 stand는 평활화되지 않으므로 모델 사양 공식의 s() 함수 외부에 있습니다.

![image 65](images/imageFile65.png)

그림 7.4 일반화 가법 모델을 사용한 투구의 격자에 대한 예상 스트라이크 확률.

hand_mod <- gam( Outcome == "called_strike" ~

p_throws + stand + s(plate_x, plate_z), family = binomial, data = taken

)

이제 두 개의 추가 이진 변수가 포함되도록 값의 격자를 다시 계산해야 합니다.

hand_grid <- taken |>

data_grid( plate_x = seq_range(plate_x, n = 100), plate_z = seq_range(plate_z, n = 100), p_throws, stand

) hand_grid_hats <- hand_mod |> augment(type.predict = "response", newdata = hand_grid)

그림 7.4 일반화 가법 모델을 사용한 투구의 격자에 대한 예상 스트라이크 확률.

![image 66](images/imageFile66.png)

그림 7.5 투수-타자 좌우 조합 4가지 모두에 걸친 예상 스트라이크 판정 확률의 표준 편차.

hand_mod <- gam( Outcome == "called_strike" ~

p_throws + stand + s(plate_x, plate_z), family = binomial, data = taken

)

이제 두 개의 추가 이진 변수가 포함되도록 값의 격자를 다시 계산해야 합니다.

hand_grid <- taken |>

data_grid( plate_x = seq_range(plate_x, n = 100), plate_z = seq_range(plate_z, n = 100), p_throws, stand

) hand_grid_hats <- hand_mod |> augment(type.predict = "response", newdata = hand_grid)

다음 코드는 타자와 투수의 좌우 조합 4가지에 걸쳐 면 분할된(faceted) 플롯을 생성합니다. 그러나 이 4개의 측면에서 뚜렷한 차이를 인지하기 어렵기 때문에 여기서는 플롯을 생략합니다.

tile_plot %+% hand_grid_hats + facet_grid(p_throws ~ stand)

대신 그림 7.5에서 4가지 좌우 조합에 걸친 표준 편차를 플로팅합니다. 스트라이크 존의 중심에서는 손에 따른 차이가 보이지 않습니다. 그러나 스트라이크 존 주변의 일부 영역에서는 스트라이크 판정 확률의 표준 편차가 최대 2% 포인트나 됩니다.

diffs <- hand_grid_hats |> group_by(plate_x, plate_z) |> summarize(

N = n(),

.fitted = sd(.fitted),

.groups = "drop" )

tile_plot %+% diffs

7.5 포수 프레이밍 모델링

포수의 프레이밍 능력을 추정하기 위해서는 매 투구마다 포수가 누구인지 알아야 합니다.

모델링을 위해 이러한 데이터를 준비하려면 각 투구에 대한 스트라이크 판정 확률에 대해 GAM을 평가합니다. 이를 통해 각 투구의 위치를 제어할 수 있습니다.

taken <- taken |>

filter( is.na(plate_x) == FALSE, is.na(plate_z) == FALSE

) |> mutate(

strike_prob = predict( strike_mod, type = "response"

) )

다음으로 Brooks, Pavilidis, and Judge (2015)를 따라 일반화 선형 혼합 모델(generalized linear mixed model)을 적합합니다. 응답 변수는 투구가 스트라이크로 판정되었는지 또는 볼로 판정되었는지 여부입니다. pj를 j번째로 불린 투구가 스트라이크일 확률이라고 합시다. 첫 번째 혼합 모델은 스트라이크 확률 pj의 로짓(logit)을 합계로 씁니다.

pj 1 − pj

log

= β0 + β1 · strike probj + αc(j).

이 모델에서 strike_prob_j는 이전 모델에서 계산된 위치를 기반으로 j번째 투구의 예상 스트라이크 판정 확률에 대한 "고정 효과(fixed effect)"입니다. 따라서 이 모델에서는 근본적으로 투구 위치를 통제하고 있습니다. 또한 αc(j)는 포수 c(j)로 인한 효과를 나타냅니다. 개별 포수는 평균이 0이고 표준 편차가 sc인 α1,...,αC라는 "무작위(random)" 매개변수를 갖는다고 가정합니다.

이 모델은 lme4 패키지의 glmer() 함수를 사용하여 적합시킬 수 있습니다. 코드는 응답 변수가 Outcome == "called_strike"이고, strike_prob이 고정 효과이며, fielder_2_1(포수 ID)이 무작위 효과를 나타냄을 지정합니다.

library(lme4) mod_a <- glmer(

Outcome == "called_strike" ~

strike_prob + (1|fielder_2_1), data = taken, family = binomial

)

- 7.5 포수 프레이밍 모델링

포수의 프레이밍 능력을 추정하기 위해서는 매 투구마다 포수가 누구인지 알아야 합니다.

모델링을 위해 이러한 데이터를 준비하려면 각 투구에 대한 스트라이크 판정 확률에 대해 GAM을 평가합니다. 이를 통해 각 투구의 위치를 제어할 수 있습니다.

taken <- taken |>

filter( is.na(plate_x) == FALSE, is.na(plate_z) == FALSE

) |> mutate(

strike_prob = predict( strike_mod, type = "response"

) )

다음으로 Brooks, Pavilidis, and Judge (2015)를 따라 일반화 선형 혼합 모델을 적합시킵니다. 응답 변수는 투구가 스트라이크로 판정되었는지 또는 볼로 판정되었는지 여부입니다. pj를 j번째로 불린 투구가 스트라이크일 확률이라고 합시다. 첫 번째 혼합 모델은 스트라이크 확률 pj의 로짓을 합계로 씁니다.

pj 1 − pj

log

= β0 + β1 · strike probj + αc(j).

이 모델에서 strike_prob_j는 이전 모델에서 계산된 위치를 기반으로 j번째 투구의 예상 스트라이크 판정 확률에 대한 "고정 효과"입니다. 따라서 이 모델에서는 근본적으로 투구 위치를 통제하고 있습니다. 또한 αc(j)는 포수 c(j)로 인한 효과를 나타냅니다. 개별 포수는 평균이 0이고 표준 편차가 sc인 α1,...,αC라는 "무작위" 매개변수를 갖는다고 가정합니다.

이 모델은 lme4 패키지의 glmer() 함수를 사용하여 적합시킬 수 있습니다. 코드는 응답 변수가 Outcome == "called_strike"이고, strike_prob이 고정 효과이며, fielder_2_1(포수 ID)이 무작위 효과를 나타냄을 지정합니다.

library(lme4) mod_a <- glmer(

Outcome == "called_strike" ~

strike_prob + (1|fielder_2_1), data = taken, family = binomial

)

fixed.effects() 함수를 사용하여 고정 효과에 대한 정보를 복구합니다.

fixed.effects(mod_a)

(Intercept) strike_prob -4.00 7.67

확실히 서로 다른 포수들은 스트라이크 판정 확률에 각기 다른 영향을 미칠 것입니다. 이러한 영향의 변동성은 VarCorr() 함수로 표시하는 이러한 무작위 포수 효과 sc의 표준 편차로 측정됩니다.

VarCorr(mod_a)

Groups Name Std.Dev. fielder_2_1 (Intercept) 0.218

이 모델은 또한 ranef() 함수로 추출하는 포수 무작위 효과 αk의 추정치를 제공합니다. 추정치를 c_effects 데이터 프레임의 포수 ID와 함께 넣습니다.

c_effects <- mod_a |> ranef() |> as_tibble() |> transmute(

id = as.numeric(levels(grp)), effect = condval

)

포수 이름이 누락되었지만 baseballr 패키지의 chadwick_player_lu() 함수를 사용하여 이러한 ID와 이름에 대한 테이블을 구성합니다.

master_id <- baseballr::chadwick_player_lu() |>

mutate( mlb_name = paste(name_first, name_last), mlb_id = key_mlbam

) |> select(mlb_id, mlb_name) |> filter(!is.na(mlb_id))

이름 정보를 c_effects 데이터 프레임과 병합하고 가장 크고 가장 작은 무작위 효과 추정치를 가진 포수 이름을 아래에 표시합니다.

c_effects <- c_effects |>

left_join( select(master_id, mlb_id, mlb_name), join_by(id == mlb_id)

) |> arrange(desc(effect))

c_effects |> slice_head(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 664848 0.358 Donny Sands
- 2 669004 0.294 MJ Melendez
- 3 642020 0.287 Chuckie Robinson
- 4 672832 0.275 Israel Pineda
- 5 571912 0.260 Luke Maile
- 6 575929 0.243 Willson Contreras c_effects |> slice_tail(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 664731 -0.293 P. J. Higgins
- 2 455139 -0.304 Robinson Chirinos
- 3 661388 -0.336 William Contreras
- 4 608360 -0.357 Chris Okey
- 5 435559 -0.357 Kurt Suzuki
- 6 595956 -0.390 Cam Gallagher

이 출력에서 Donny Sands가 스트라이크 판정을 받는 데 가장 효과적이었고 Cam Gallagher가 가장 덜 효과적이었음을 알 수 있습니다.

이 첫 번째 모델에 대한 한 가지 비판은 투수나 타자를 고려하지 않았다는 것이며, 두 사람 모두 스트라이크 판정 확률에 영향을 미친다고 여겨집니다. 위의 모델을 투수와 타자 모두에 대한 무작위 효과를 포함하도록 확장할 수 있습니다. 이 모델은 다음과 같이 작성됩니다.

pj 1 − pj

= β0 + β1strike probj + αc(j) + γp(j) + δb(j).

log

여기서 개별 투수는 표준 편차가 sp인 분포에서 무작위라고 가정하는 매개변수 γ1,...,γP를 할당받습니다. 또한 개별 타자는 표준 편차가 sb인 분포에서 나온 매개변수 δ1,...,δB를 할당받습니다.

c_effects <- c_effects |>

left_join( select(master_id, mlb_id, mlb_name), join_by(id == mlb_id)

) |> arrange(desc(effect))

c_effects |> slice_head(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 664848 0.358 Donny Sands
- 2 669004 0.294 MJ Melendez
- 3 642020 0.287 Chuckie Robinson
- 4 672832 0.275 Israel Pineda
- 5 571912 0.260 Luke Maile
- 6 575929 0.243 Willson Contreras c_effects |> slice_tail(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 664731 -0.293 P. J. Higgins
- 2 455139 -0.304 Robinson Chirinos
- 3 661388 -0.336 William Contreras
- 4 608360 -0.357 Chris Okey
- 5 435559 -0.357 Kurt Suzuki
- 6 595956 -0.390 Cam Gallagher

이 출력에서 Donny Sands가 스트라이크 판정을 받는 데 가장 효과적이었고 Cam Gallagher가 가장 덜 효과적이었음을 알 수 있습니다.

이 첫 번째 모델에 대한 한 가지 비판은 투수나 타자를 고려하지 않았다는 것이며, 두 사람 모두 스트라이크 판정 확률에 영향을 미친다고 여겨집니다. 위의 모델을 투수와 타자 모두에 대한 무작위 효과를 포함하도록 확장할 수 있습니다. 이 모델은 다음과 같이 작성됩니다.

pj 1 − pj

= β0 + β1strike probj + αc(j) + γp(j) + δb(j).

log

여기서 개별 투수는 표준 편차가 sp인 분포에서 무작위라고 가정하는 매개변수 γ1,...,γP를 할당받습니다. 또한 개별 타자는 표준 편차가 sb인 분포에서 나온 매개변수 δ1,...,δB를 할당받습니다.

이 더 큰 모델은 회귀 표현식에 타자와 투수를 입력으로 추가하여 glmer() 함수를 두 번째 적용하여 적합시킵니다.

mod_b <- glmer(

Outcome == "called_strike" ~ strike_prob + (1|fielder_2_1) + (1|batter) + (1|pitcher),

data = taken, family = binomial

)

VarCorr() 함수를 사용하여 세 개의 표준 편차 sc, sp, sb의 추정치를 표시합니다. sc의 값이 이전 모델과 약간 다르다는 점에 유의하십시오.

VarCorr(mod_b)

Groups Name Std.Dev. pitcher (Intercept) 0.267 batter (Intercept) 0.251 fielder_2_1 (Intercept) 0.209

이 표는 스트라이크 판정의 전체 변동성에 가장 큰 기여를 하는 구성 요소를 식별하는 데 유용합니다. 가장 큰 표준 편차는 sp = 0.267 및 sb = 0.251로, 스트라이크 판정이 투수와 타자의 정체성에 가장 큰 영향을 받고 포수의 정체성이 그 뒤를 잇는다는 것을 나타냅니다.

이전과 마찬가지로 ranef() 함수를 통해 포수 효과 추정치를 추출하고 모든 포수의 ID, 이름 및 추정치로 데이터 프레임을 만든 다음 프레이밍과 관련하여 가장 훌륭한 포수와 가장 못한 포수를 표시합니다. 이 목록은 더 단순한 무작위 효과 모델로 준비된 목록과 유사하지 않으며, 이는 이러한 포수들이 스트라이크 판정에 영향을 미친 서로 다른 투수 및 타자와 협력했음을 시사합니다.

c_effects <- mod_b |> ranef() |> as_tibble() |> filter(grpvar == "fielder_2_1") |> transmute(

id = as.numeric(as.character(grp)), effect = condval

) c_effects <- c_effects |>

left_join( select(master_id, mlb_id, mlb_name), join_by(id == mlb_id)

) |> arrange(desc(effect))

c_effects |> slice_head(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 624431 0.313 Jose Trevino
- 2 669221 0.277 Sean Murphy
- 3 425877 0.263 Yadier Molina
- 4 664874 0.253 Seby Zavala
- 5 543309 0.229 Kyle Higashioka
- 6 608700 0.221 Kevin Plawecki c_effects |> slice_tail(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 596117 -0.277 Garrett Stubbs
- 2 435559 -0.281 Kurt Suzuki
- 3 521692 -0.291 Salvador Perez
- 4 553869 -0.327 Elias D´az
- 5 455139 -0.336 Robinson Chirinos
- 6 669004 -0.347 MJ Melendez

이것은 작은 데이터 세트만 사용하고 스트라이크 판정 확률에 영향을 미칠 수 있는 심판과 같은 다른 효과를 포함하지 않았기 때문에 분명히 철저한 분석은 아닙니다. 그러나 고정 효과와 무작위 효과를 포함하는 이러한 혼합 모델은 다른 관련 입력을 조정하여 선수 능력의 추정치를 얻는 데 유용합니다.

7.6 추가 읽을거리

PITCHf/x를 사용한 포수 프레이밍에 대한 첫 번째 연구는 Turkenkopf(2008)였습니다. 후속 기사는 Fast(2011)를 참조하십시오. Lindbergh(2013)는 포수 프레이밍에 대한 생각의 진화에 대해 읽기 쉬운 일반적인 개요를 제공합니다. 포수 프레이밍에 대한 더 정교한 모델에는 Brooks and Pavlidis(2014), Brooks, Pavilidis, and Judge(2015), Judge(2018), Deshpande and Wyner(2017)가 있습니다.

) |> arrange(desc(effect))

c_effects |> slice_head(n = 6)

# A tibble: 6 x 3

id effect mlb_name <dbl> <dbl> <chr>

- 1 624431 0.313 Jose Trevino
- 2 669221 0.277 Sean Murphy
- 3 425877 0.263 Yadier Molina
- 4 664874 0.253 Seby Zavala
- 5 543309 0.229 Kyle Higashioka
- 6 608700 0.221 Kevin Plawecki c_effects |> slice_tail(n = 6)

# A tibble: 6 x 3 id effect mlb_name

<dbl> <dbl> <chr> 1 596117 -0.277 Garrett Stubbs 2 435559 -0.281 Kurt Suzuki 3 521692 -0.291 Salvador Perez 4 553869 -0.327 Elias D´az 5 455139 -0.336 Robinson Chirinos 6 669004 -0.347 MJ Melendez

이것은 작은 데이터 세트만 사용하고 스트라이크 판정 확률에 영향을 미칠 수 있는 심판과 같은 다른 효과를 포함하지 않았기 때문에 분명히 철저한 분석은 아닙니다. 그러나 고정 효과와 무작위 효과를 포함하는 이러한 혼합 모델은 다른 관련 입력을 조정하여 선수 능력의 추정치를 얻는 데 유용합니다.

- 7.6 추가 읽을거리

PITCHf/x를 사용한 포수 프레이밍에 대한 첫 번째 연구는 Turkenkopf(2008)였습니다. 후속 기사는 Fast(2011)를 참조하십시오. Lindbergh(2013)는 포수 프레이밍에 대한 생각의 진화에 대해 읽기 쉬운 일반적인 개요를 제공합니다. 포수 프레이밍에 대한 더 정교한 모델에는 Brooks and Pavlidis(2014), Brooks, Pavilidis, and Judge(2015), Judge(2018), Deshpande and Wyner(2017)가 있습니다.

연습 문제 179

7.7 연습 문제

- 1. 격자의 스트라이크 확률

- a. 다음 코드를 사용하여 구역 영역을 빈(bin)으로 나눕니다.

seq_x <- seq(-1.4, 1.4, by = 0.4) seq_z <- seq(1.1, 3.9, by = 0.4) taken <- taken |>

mutate( plate_x = cut(plate_x, seq_x), plate_z = cut(plate_z, seq_z)

)

- b. group_by() 및 summarize() 함수를 사용하여 각 빈에서 투구된 공 중 스트라이크와 볼의 수를 찾습니다.
- c. 각 빈의 스트라이크 비율을 찾습니다. 여러 빈에 걸쳐 스트라이크 비율에서 흥미로운 패턴이 있는지 의견을 제시하십시오.

- 2. 스트라이크 확률 타자 효과

첫 번째 연습 문제에서는 다른 영역에 대한 스트라이크 확률 비율을 찾았습니다. 여러 빈과 stand 변수에 대해 볼과 스트라이크를 표로 만들어 타자가 서 있는 쪽에 따라 스트라이크 확률이 어떻게 다른지 살펴보십시오.

- 3. 스트라이크 확률 투수 효과

첫 번째 연습 문제에서는 다른 영역에 대한 스트라이크 확률 비율을 찾았습니다. 여러 빈과 p_throws 변수에 대해 볼과 스트라이크를 표로 만들어 투수가 던지는 팔에 따라 스트라이크 확률이 어떻게 다른지 살펴보십시오.

- 4. 카운트 효과

카운트가 스트라이크 확률에 미치는 영향을 탐구하는 한 가지 방법은 glm() 함수를 사용하여 로지스틱 모델을 적합하는 것입니다.

fit <- glm( Outcome == "called_strike" ~ Count, data = taken, family = binomial

)

이 표현식에서 Count는 taken 데이터 프레임의 볼과 스트라이크 변수에서 파생된 새로운 변수입니다. 이 적합의 출력으로부터 스트라이크 확률이 카운트에 따라 어떻게 달라지는지 해석하십시오.

- 5. 홈/원정 효과

홈 구장이 스트라이크 확률에 미치는 영향을 탐구하는 한 가지 방법은 glm() 함수를 사용하여 로지스틱 모델을 적합하는 것입니다.

fit <- glm( Outcome == "called_strike" ~ Home, data = taken, family = binomial

)

이 표현식에서 Home은 타자가 홈팀 소속이면 1이고 그렇지 않으면 0인 새로운 변수입니다. 이 적합의 출력으로부터 홈 타자와 원정 타자 사이에서 스트라이크가 어떻게 다른지 해석하십시오.

5. 홈/원정 효과

홈 구장이 스트라이크 확률에 미치는 영향을 탐구하는 한 가지 방법은 glm() 함수를 사용하여 로지스틱 모델을 적합하는 것입니다.

fit <- glm( Outcome == "called_strike" ~ Home, data = taken, family = binomial

)

이 표현식에서 Home은 타자가 홈팀 소속이면 1이고 그렇지 않으면 0인 새로운 변수입니다. 이 적합의 출력으로부터 홈 타자와 원정 타자 사이에서 스트라이크가 어떻게 다른지 해석하십시오.

