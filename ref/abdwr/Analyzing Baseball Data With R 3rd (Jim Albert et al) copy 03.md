### 3 그래픽스

- 3.1 서론

ggplot2 패키지(Wickham 2016b)를 사용하여 R에서 그래프를 생성하는 방법을 설명하기 위해, 현재 명예의 전당 회원의 모든 통산 타격 통계를 고려합니다. abdwr3edata 패키지의 데이터 프레임 hof_batting에는 이 그룹의 통산 타격 통계가 포함되어 있습니다. 이 데이터를 hof라는 이름의 데이터 프레임으로 복사합니다.

library(tidyverse) library(abdwr3edata) hof <- hof_batting

데이터 세트에서 투수의 타격 통계를 제거하면 167명의 비투수 통계가 남습니다. 사용하는 그래프의 종류는 변수의 측정 척도에 따라 다릅니다. 두 가지 기본적인 데이터 유형인 측정과 범주가 있으며, 이는 R에서 숫자(numeric)와 문자(character) 변수로 표시됩니다. 처음에 단일 문자 변수와 단일 숫자 변수에 대한 그래프를 설명한 다음, 변수 간의 관계를 이해하는 데 유용한 그래픽 디스플레이를 설명합니다. ggplot2 시스템을 사용하면 레이블을 추가하고 플로팅 기호와 선의 스타일을 변경하여 그래프의 속성을 쉽게 수정할 수 있습니다. 그래픽 방법을 설명한 후 두 가지 홈런 이야기에 대한 그래프를 만드는 과정을 설명합니다. 3.7절에서는 야구 역사상 위대한 슬러거 4명의 홈런 통산 진행 상황을 비교하고, 3.8절에서는 1998년 시즌 동안 마크 맥과이어와 새미 소사의 유명한 홈런 레이스를 설명합니다.

- 3.2 문자 변수 3.2.1 막대 그래프


명예의 전당 입성자들은 여러 야구 시대에 활동했습니다. 시대의 일반적인 분류 중 하나는 19세기(1900년 시즌까지), 데드볼(1901년부터 1919년까지), 라이블리볼(1920년부터 1941년까지), 통합(1942년

DOI: 10.1201/9781032668239-3 66

### 3 그래픽스

###### 3.1 서론

ggplot2 패키지(Wickham 2016b)를 사용하여 R에서 그래프를 생성하는 방법을 설명하기 위해, 현재 명예의 전당 회원의 모든 통산 타격 통계를 고려합니다. abdwr3edata 패키지의 데이터 프레임 hof_batting에는 이 그룹의 통산 타격 통계가 포함되어 있습니다. 이 데이터를 hof라는 이름의 데이터 프레임으로 복사합니다.

library(tidyverse) library(abdwr3edata) hof <- hof_batting

데이터 세트에서 투수의 타격 통계를 제거하면 167명의 비투수 통계가 남습니다. 사용하는 그래프의 종류는 변수의 측정 척도에 따라 다릅니다. 두 가지 기본적인 데이터 유형인 측정과 범주가 있으며, 이는 R에서 숫자(numeric)와 문자(character) 변수로 표시됩니다. 처음에 단일 문자 변수와 단일 숫자 변수에 대한 그래프를 설명한 다음, 변수 간의 관계를 이해하는 데 유용한 그래픽 디스플레이를 설명합니다. ggplot2 시스템을 사용하면 레이블을 추가하고 플로팅 기호와 선의 스타일을 변경하여 그래프의 속성을 쉽게 수정할 수 있습니다. 그래픽 방법을 설명한 후 두 가지 홈런 이야기에 대한 그래프를 만드는 과정을 설명합니다. 3.7절에서는 야구 역사상 위대한 슬러거 4명의 홈런 통산 진행 상황을 비교하고, 3.8절에서는 1998년 시즌 동안 마크 맥과이어와 새미 소사의 유명한 홈런 레이스를 설명합니다.

###### 3.2 문자 변수

- 3.2.1 막대 그래프


명예의 전당 입성자들은 여러 야구 시대에 활동했습니다. 시대의 일반적인 분류 중 하나는 19세기(1900년 시즌까지), 데드볼(1901년부터 1919년까지), 라이블리볼(1920년부터 1941년까지), 통합(1942년

DOI: 10.1201/9781032668239-3 66

문자 변수 67 부터 1960년까지), 확장(1961년부터 1976년까지), 자유계약(1977년부터 1993년까지), 롱볼(1993년 이후)입니다. 각 선수의 시대를 나타내는 새로운 문자 변수 Era를 생성하고자 합니다. 먼저, 선수의 통산 중반기(변수 MidCareer)를 야구에서의 첫 시즌과 마지막 시즌의 평균으로 정의합니다. 그런 다음 mutate() 및 cut() 함수를 사용하여 새로운 팩터(factor) 변수 Era를 만듭니다. 함수의 인수는 이산화할 숫자 변수, 절단점(cut points)의 벡터, 그리고 팩터 변수의 범주에 대한 레이블 벡터입니다.

hof <- hof |>

mutate( MidCareer = (From + To) / 2, Era = cut(

MidCareer, breaks = c(1800, 1900, 1919, 1941, 1960, 1976, 1993, 2050), labels = c(

"19th Century", "Dead Ball", "Lively Ball", "Integration", "Expansion", "Free Agency", "Long Ball"

) )

)

변수 Era의 빈도표는 n() 함수와 함께 summarize() 함수를 사용하여 구성할 수 있습니다. 아래에서 해당 출력을 데이터 프레임 hof_eras에 저장합니다.

hof_eras <- hof |> group_by(Era) |> summarize(N = n())

hof_eras

# A tibble: 7 x 2

Era N <fct> <int>

- 1 19th Century 18
- 2 Dead Ball 19
- 3 Lively Ball 46
- 4 Integration 24
- 5 Expansion 23
- 6 Free Agency 22
- 7 Long Ball 15


이 데이터에서 ggplot2의 geom_bar() 함수를 사용하여 막대 그래프를 구성합니다.

aes() 함수는 미적 요소(aesthetics)를 정의합니다. 플롯의 시각적 요소와 데이터 프레임의 변수 간에 매핑이 있습니다. 여기서는 문자 벡터 Era를 수평 위치를 정의하는 x 미적 요소에 매핑합니다. 그림 3.1

![image 30](images/imageFile30.png)

그림 3.1 명예의 전당 비투수의 시대 막대 그래프입니다. 생성된 그래프를 보여줍니다. 명예의 전당 선수 중 다수가 라이블리볼 시대에 뛰었음을 알 수 있습니다.

ggplot(hof, aes(x = Era)) + geom_bar()

- 3.2.2 축 레이블 및 제목 추가

좋은 방법으로, 그래프에는 디스플레이의 주요 메시지를 설명하기 위한 설명적인 축 레이블과 제목이 있어야 합니다. ggplot2 패키지에서 xlab() 및 ylab() 함수는 수평 및 수직 축 레이블을 추가하고 ggtitle() 함수는 제목을 추가합니다. 막대 그래프를 구성하기 위한 다음 코드에서 야구 시대와 빈도라는 레이블을 추가하고 비투수 명예의 전당 회원의 시대라는 제목을 추가합니다. 향상된 플롯은 그림 3.2에 나와 있습니다.

ggplot(hof, aes(Era)) + geom_bar() + xlab("Baseball Era") + ylab("Frequency") + ggtitle("Era of the Nonpitching Hall of Famers")

- 3.2.3 문자 변수의 다른 그래프


문자 변수의 빈도표에 대한 대안적인 그래픽 디스플레이가 있습니다. 시대 빈도의 데이터 프레임에 대해 geom_point() 함수를 사용하여 표시된 클리블랜드 스타일(Cleveland 1985) 점 플롯을 구성합니다.

- 그림 3.1 명예의 전당 비투수의 시대 막대 그래프입니다.


생성된 그래프를 보여줍니다. 명예의 전당 선수 중 다수가 라이블리볼 시대에 뛰었음을 알 수 있습니다.

ggplot(hof, aes(x = Era)) + geom_bar()

- 3.2.2 축 레이블 및 제목 추가

좋은 방법으로, 그래프에는 디스플레이의 주요 메시지를 설명하기 위한 설명적인 축 레이블과 제목이 있어야 합니다. ggplot2 패키지에서 xlab() 및 ylab() 함수는 수평 및 수직 축 레이블을 추가하고 ggtitle() 함수는 제목을 추가합니다. 막대 그래프를 구성하기 위한 다음 코드에서 야구 시대와 빈도라는 레이블을 추가하고 비투수 명예의 전당 회원의 시대라는 제목을 추가합니다. 향상된 플롯은 그림 3.2에 나와 있습니다.

ggplot(hof, aes(Era)) + geom_bar() + xlab("Baseball Era") + ylab("Frequency") + ggtitle("Era of the Nonpitching Hall of Famers")

- 3.2.3 문자 변수의 다른 그래프


문자 변수의 빈도표에 대한 대안적인 그래픽 디스플레이가 있습니다. 시대 빈도의 데이터 프레임에 대해 geom_point() 함수를 사용하여 표시된 클리블랜드 스타일(Cleveland 1985) 점 플롯을 구성합니다.

문자 변수 69

![image 31](images/imageFile31.png)

- 그림 3.2 비투수 명예의 전당 회원의 시대.

![image 32](images/imageFile32.png)

- 그림 3.3 명예의 전당 비투수의 시대 점 플롯.


그림 3.3에 나와 있습니다. 점 플롯은 문자 벡터의 범주 수가 많을 때 유용합니다. 점은 geom_plot()의 color = "red" 인수에 의해 빨간색으로 지정됩니다.

ggplot(hof_eras, aes(Era, N)) + geom_point(color = "red") + xlab("Baseball Era") + ylab("Frequency") + ggtitle("Era of the Nonpitching Hall of Famers") + coord_flip()

3.3 그래프 저장

R에서 그래프가 생성된 후에는 일반적인 그래픽 형식 중 하나로 내보내어 문서, 블로그 또는 웹 사이트에서 사용할 수 있도록 하는 것이 간단합니다. RStudio 인터페이스에서 그래프를 저장하는 단계를 설명합니다.

RStudio의 Plots 창에 그래프가 나타나면 Export 메뉴에서 이미지를 다른 이름으로 저장, PDF로 저장, 또는 클립보드에 플롯 복사를 선택할 수 있습니다. 이미지를 다른 이름으로 저장 옵션을 선택하면 드롭다운 메뉴에서 옵션을 선택하여 그래프를 PNG, JPEG, TIFF, BMP, 메타파일, 클립보드, SVG 또는 EPS 형식으로 저장할 수 있습니다. PNG 형식은 웹 페이지에 업로드하는 데 편리하며, EPS 및 PDF 형식은 LATEX 문서에서 사용하기에 적합합니다. 메타파일 및 클립보드 옵션은 Microsoft Word 문서에 그래프를 삽입하는 데 유용합니다.

또는 콘솔 창에 입력한 R 함수를 사용하여 플롯을 저장할 수 있습니다. 예를 들어 그림 3.2에 표시된 막대 그래프를 PNG 형식의 그래픽 파일로 저장하고자 한다고 가정합니다. 먼저 그래프를 생성하기 위한 R 명령을 입력합니다. 그런 다음 특별한 ggsave() 함수를 사용하는데, 이때 인수는 저장된 그래픽 파일의 이름입니다. 파일 이름의 확장자가 png이므로 그래프는 PNG 형식으로 저장됩니다.

ggplot(hof, aes(Era)) + geom_bar() + xlab("Baseball Era") + ylab("Frequency") + ggtitle("Era of the Nonpitching Hall of Famers")

ggsave("bargraph.png")

현재 디렉토리를 보면 PNG 형식의 이미지를 포함하는 새로운 bargraph.png 파일을 볼 수 있습니다. 다른 확장자를 사용하여 대안적인 그래픽 형식으로 그래프를 저장할 수 있습니다. 예를 들어, 그래프를 PDF 형식으로 저장하려면 ggsave()에 대한 인수는 pdf가 되고, JPEG 형식으로 저장하려면 jpeg가 됩니다.

ggplot(hof_eras, aes(Era, N)) + geom_point(color = "red") + xlab("Baseball Era") + ylab("Frequency") + ggtitle("Era of the Nonpitching Hall of Famers") + coord_flip()

- 3.3 그래프 저장


R에서 그래프가 생성된 후에는 일반적인 그래픽 형식 중 하나로 내보내어 문서, 블로그 또는 웹 사이트에서 사용할 수 있도록 하는 것이 간단합니다. RStudio 인터페이스에서 그래프를 저장하는 단계를 설명합니다.

RStudio의 Plots 창에 그래프가 나타나면 Export 메뉴에서 이미지를 다른 이름으로 저장, PDF로 저장, 또는 클립보드에 플롯 복사를 선택할 수 있습니다. 이미지를 다른 이름으로 저장 옵션을 선택하면 드롭다운 메뉴에서 옵션을 선택하여 그래프를 PNG, JPEG, TIFF, BMP, 메타파일, 클립보드, SVG 또는 EPS 형식으로 저장할 수 있습니다. PNG 형식은 웹 페이지에 업로드하는 데 편리하며, EPS 및 PDF 형식은 LATEX 문서에서 사용하기에 적합합니다. 메타파일 및 클립보드 옵션은 Microsoft Word 문서에 그래프를 삽입하는 데 유용합니다.

또는 콘솔 창에 입력한 R 함수를 사용하여 플롯을 저장할 수 있습니다. 예를 들어 그림 3.2에 표시된 막대 그래프를 PNG 형식의 그래픽 파일로 저장하고자 한다고 가정합니다. 먼저 그래프를 생성하기 위한 R 명령을 입력합니다. 그런 다음 특별한 ggsave() 함수를 사용하는데, 이때 인수는 저장된 그래픽 파일의 이름입니다. 파일 이름의 확장자가 png이므로 그래프는 PNG 형식으로 저장됩니다.

ggplot(hof, aes(Era)) + geom_bar() + xlab("Baseball Era") + ylab("Frequency") + ggtitle("Era of the Nonpitching Hall of Famers")

ggsave("bargraph.png")

현재 디렉토리를 보면 PNG 형식의 이미지를 포함하는 새로운 bargraph.png 파일을 볼 수 있습니다. 다른 확장자를 사용하여 대안적인 그래픽 형식으로 그래프를 저장할 수 있습니다. 예를 들어, 그래프를 PDF 형식으로 저장하려면 ggsave()에 대한 인수는 pdf가 되고, JPEG 형식으로 저장하려면 jpeg가 됩니다.

여러 그래프를 단일 파일로 저장하고자 할 경우 그래프를 저장하는 다른 방법이 유용합니다. 예를 들어 patchwork 라이브러리를 사용하여 둘 이상의 ggplot을 단일 ggplot 객체로 결합할 수 있습니다. 그런 다음 앞서 언급한 ggsave() 명령을 사용하여 이 복합 플롯을 저장할 수 있습니다. 예를 들어 다음과 같이 입력하면:

library(patchwork) p1 <- ggplot(hof, aes(Era)) + geom_bar() p2 <- ggplot(hof_eras, aes(Era, N)) + geom_point() p1 + p2 ggsave("graphs.pdf")

막대 그래프와 점 플롯 그래프가 함께 PDF 파일 graphs.pdf에 저장됩니다.

3.4 숫자 변수: 1차원 산점도 및 히스토그램

선수 그룹에서 타율, 출루율(OBP) 또는 OPS와 같은 숫자 변수를 수집할 때 일반적으로 그 분포에 대해 알고 싶어 합니다. 예를 들어, 비투수 명예의 전당 입성자들의 OPS 값을 조사하면 OPS 값의 전반적인 모양에 대해 관심을 갖게 됩니다. 예를 들어 OPS 값의 분포가 대칭입니까, 아니면 오른쪽이나 왼쪽으로 치우쳐 있습니까? 또한 명예의 전당 OPS 값의 전형적이거나 대표적인 값과 OPS 값이 어떻게 퍼져 있는지에 대해서도 알고 싶어 합니다. 그래픽 디스플레이는 야구 통계의 분포를 시각적으로 빠르게 연구할 수 있는 방법을 제공합니다.

단일 숫자 변수의 경우, 분포를 시각화하는 데 유용한 두 가지 디스플레이는 1차원 산점도와 히스토그램입니다. 1차원 산점도는 기본적으로 숫자 선 그래프이며, 통계 값은 변수의 가능한 모든 값에 걸쳐 있는 숫자 선 위에 표시됩니다. ggplot2에서 geom_jitter() 함수를 사용하여 명예의 전당 입성자의 OPS 값 그래프를 구성합니다. 데이터 프레임 hof에서 OPS는 x 미적 요소에 매핑되고 더미 변수 y는 상수 값으로 설정됩니다. 테마 요소는 y축에서 눈금 표시, 텍스트 및 제목을 제거하도록 선택됩니다.

ggplot(hof, aes(x = OPS, y = 1)) + geom_jitter(height = 0.2) + ylim(0, 2) + theme(

![image 33](images/imageFile33.png)

그림 3.4 명예의 전당 선수들의 OPS 값에 대한 1차원 산점도.

axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()

) + coord_fixed(ratio = 0.03)

생성된 그래프는 그림 3.4에 나와 있습니다. 대부분의 OPS 값이 0.700에서 1.000 사이에 있지만, 더 조사해 볼 가치가 있는 비정상적으로 높은 값이 몇 개 있음을 알 수 있습니다.

숫자 변수에 대한 두 번째 그래픽 디스플레이는 값을 동일한 너비의 빈(bin)으로 그룹화하고 빈 주파수를 겹치지 않는 막대로 표시하는 히스토그램입니다. OPS 값의 히스토그램은 ggplot2 시스템에서 geom_histogram() 함수를 사용하여 구성됩니다. 유일한 미적 요소 매핑은 변수 OPS에 대한 것입니다(그림 3.5 참조).

ggplot(hof, aes(x = OPS)) + geom_histogram()

히스토그램 구성 시 한 가지 문제는 빈 선택이며, geom_histogram() 함수는 데이터 분포를 잘 표시하기 위해 일반적으로 합리적인 선택을 합니다. breaks 인수를 사용하여 geom_histogram()에서 자체 빈을 선택할 수 있습니다. 예를 들어 대체 빈 끝점 0.4, 0.5, ..., 1.2를 선택하고자 하는 경우 다음 코드로 히스토그램을 구성할 수 있습니다(그림 3.6 참조). color 및 fill 인수를 사용하면 막대의 선이 흰색으로 칠해지고 막대는 주황색으로 채워집니다.

ggplot(hof, aes(x = OPS)) +

geom_histogram( breaks = seq(0.4, 1.2, by = 0.1), color = "white", fill = "orange"

)

- 그림 3.4 명예의 전당 선수들의 OPS 값에 대한 1차원 산점도.


axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank()

) + coord_fixed(ratio = 0.03)

생성된 그래프는 그림 3.4에 나와 있습니다. 대부분의 OPS 값이 0.700에서 1.000 사이에 있지만, 더 조사해 볼 가치가 있는 비정상적으로 높은 값이 몇 개 있음을 알 수 있습니다.

숫자 변수에 대한 두 번째 그래픽 디스플레이는 값을 동일한 너비의 빈(bin)으로 그룹화하고 빈 주파수를 겹치지 않는 막대로 표시하는 히스토그램입니다. OPS 값의 히스토그램은 ggplot2 시스템에서 geom_histogram() 함수를 사용하여 구성됩니다. 유일한 미적 요소 매핑은 변수 OPS에 대한 것입니다(그림 3.5 참조).

ggplot(hof, aes(x = OPS)) + geom_histogram()

히스토그램 구성 시 한 가지 문제는 빈 선택이며, geom_histogram() 함수는 데이터 분포를 잘 표시하기 위해 일반적으로 합리적인 선택을 합니다. breaks 인수를 사용하여 geom_histogram()에서 자체 빈을 선택할 수 있습니다. 예를 들어 대체 빈 끝점 0.4, 0.5, ..., 1.2를 선택하고자 하는 경우 다음 코드로 히스토그램을 구성할 수 있습니다(그림 3.6 참조). color 및 fill 인수를 사용하면 막대의 선이 흰색으로 칠해지고 막대는 주황색으로 채워집니다.

ggplot(hof, aes(x = OPS)) +

geom_histogram( breaks = seq(0.4, 1.2, by = 0.1), color = "white", fill = "orange"

)

![image 34](images/imageFile34.png)

- 그림 3.5 명예의 전당 선수들의 OPS 값 히스토그램.

![image 35](images/imageFile35.png)

- 그림 3.6 다른 빈(bin) 및 다른 색상/채우기 옵션을 사용한 명예의 전당 선수들의 OPS 값 히스토그램.


###### 3.5 두 숫자 변수

3.5.1 산점도

많은 선수를 대상으로 두 개의 숫자 변수를 수집하면 이들의 관계를 탐구하는 데 관심이 생깁니다. 산점도는 두 개의 숫자 변수를 그래프로 나타내는 표준 방법이며, ggplot2 시스템에서 x 및 y 미적 요소와 geom_point() 함수를 사용하여 산점도를 생성할 수 있습니다.

이전 절에서 OPS 통계의 분포를 살펴보았습니다. 선수의 OPS와 야구 시대 사이에 어떤 관계가 있습니까? 명예의 전당 OPS 값이 비정상적으로 높거나 낮았던 특정 시즌이 있었습니까?

geom_point()를 사용하여 변수 MidCareer 및 OPS를 각각 x 및 y 미적 요소에 매핑한 산점도를 작성하여 이러한 질문에 답할 수 있습니다. 산점도 패턴을 시각적으로 감지하기 어려울 수 있으므로 일반적인 연관성을 보여주기 위해 geom_smooth() 함수를 사용하여 평활화 곡선을 추가하는 것이 도움이 됩니다. 이 함수는 기본적으로 대중적인 LOESS 평활화 방법(Cleveland 1979)을 구현합니다.

ggplot(hof, aes(MidCareer, OPS)) + geom_point() + geom_smooth()

그림 3.7의 산점도를 보면 비정상적으로 큰 통산 OPS 값이 3개 있다는 것을 알 수 있으며, 이러한 극단적인 값을 가진 선수들을 식별하고자 합니다.

![image 36](images/imageFile36.png)

- 그림 3.7 명예의 전당 선수들의 OPS 및 통산 중반기 값의 산점도.


###### 3.5 두 숫자 변수

- 3.5.1 산점도


많은 선수를 대상으로 두 개의 숫자 변수를 수집하면 이들의 관계를 탐구하는 데 관심이 생깁니다. 산점도는 두 개의 숫자 변수를 그래프로 나타내는 표준 방법이며, ggplot2 시스템에서 x 및 y 미적 요소와 geom_point() 함수를 사용하여 산점도를 생성할 수 있습니다.

이전 절에서 OPS 통계의 분포를 살펴보았습니다. 선수의 OPS와 야구 시대 사이에 어떤 관계가 있습니까? 명예의 전당 OPS 값이 비정상적으로 높거나 낮았던 특정 시즌이 있었습니까?

geom_point()를 사용하여 변수 MidCareer 및 OPS를 각각 x 및 y 미적 요소에 매핑한 산점도를 작성하여 이러한 질문에 답할 수 있습니다. 산점도 패턴을 시각적으로 감지하기 어려울 수 있으므로 일반적인 연관성을 보여주기 위해 geom_smooth() 함수를 사용하여 평활화 곡선을 추가하는 것이 도움이 됩니다. 이 함수는 기본적으로 대중적인 LOESS 평활화 방법(Cleveland 1979)을 구현합니다.

ggplot(hof, aes(MidCareer, OPS)) + geom_point() + geom_smooth()

그림 3.7의 산점도를 보면 비정상적으로 큰 통산 OPS 값이 3개 있다는 것을 알 수 있으며, 이러한 극단적인 값을 가진 선수들을 식별하고자 합니다.

![image 37](images/imageFile37.png)

- 그림 3.8 포인트가 식별된 명예의 전당 선수들의 OPS 및 통산 중반기 값의 산점도.


그림 3.8은 포인트가 식별된 산점도를 보여줍니다. ggrepel 패키지의 geom_text_repel() 함수를 사용하여 플롯에 텍스트 레이블을 추가하여 이를 수행합니다. filter()를 사용하여 데이터의 작은 부분 집합만 이 함수로 보냅니다. 또한 레이블은 geom_text_repel()에 대한 color = "red" 인수를 사용하여 빨간색으로 지정됩니다.

library(ggrepel) ggplot(hof, aes(MidCareer, OPS)) +

geom_point() + geom_smooth() + geom_text_repel(

data = filter(hof, OPS > 1.05 | OPS < .5), aes(MidCareer, OPS, label = Player), color = "red"

)

그림 3.7과 3.8에서 무엇을 알 수 있습니까? 명예의 전당 선수의 전형적인 OPS는 수년 동안 상당히 일정하게 유지되었습니다. 하지만 베이브 루스와 루 게릭이 전성기를 누렸던 1930년대에는 OPS가 증가했습니다. 흥미롭게도 최근 시즌에는 이 선수들 간의 OPS 값 변동성이 작아진 것으로 보입니다.

그림 3.7 명예의 전당 선수들의 OPS 및 통산 중반기 값의 산점도.

3.5.2 단계별로 그래프 구축하기

일반적으로 그래프 구축은 반복적인 과정입니다. 관심 있는 변수와 산점도와 같은 특정 그래픽 방법을 선택하는 것으로 시작합니다.

![image 38](images/imageFile38.png)

- 그림 3.9 명예의 전당 선수들의 OPS 및 SLG 값의 산점도.


생성된 디스플레이를 검사하면 일반적으로 그래프를 개선할 수 있는 방법을 찾을 수 있습니다. 몇 가지 선택적 인수를 사용하여 더 명확하고 정보가 많은 디스플레이가 되도록 그래프를 변경할 수 있습니다. 두 변수 간의 관계를 탐구하는 상황에서 이 그래프 구성 과정을 설명합니다.

타격에는 출루율(OBP)로 측정되는 출루 능력과 장타율(SLG)로 측정되는 누상에 있는 주자를 진루시키는 능력이라는 두 가지 측면이 있습니다. 이 두 측정값의 산점도를 작성하여 선수들의 타격 성과를 더 잘 이해할 수 있습니다. geom_plot() 함수를 사용하여 OBP 및 SLG 산점도를 구성합니다(그림 3.9 참조).

(p <- ggplot(hof, aes(OBP, SLG)) + geom_point())

그림 3.9를 보면 이 디스플레이에 몇 가지 문제가 있음을 알 수 있습니다. 특히 두 축에 대해 더 설명적인 레이블을 사용하면 그래프를 읽기 더 쉬울 것입니다. 이러한 새로운 아이디어를 통합하기 위해 새 그림을 그립니다. xlab() 및 ylab() 함수를 사용하여 OBP 및 SLG를 각각

출루율(On-Base Percentage) 및 장타율(Slugging Percentage)로 바꿉니다. 업데이트된 디스플레이는 그림 3.10에 나와 있습니다.

(p <- p + xlab("On Base Percentage") +

그림 3.9 명예의 전당 선수들의 OPS 및 SLG 값의 산점도.

![image 39](images/imageFile39.png)

- 그림 3.10 두 축에 대한 설명적인 레이블이 있는 명예의 전당 선수들의 OPS 및 SLG 값의 산점도.

생성된 디스플레이를 검사하면 일반적으로 그래프를 개선할 수 있는 방법을 찾을 수 있습니다. 몇 가지 선택적 인수를 사용하여 더 명확하고 정보가 많은 디스플레이가 되도록 그래프를 변경할 수 있습니다. 두 변수 간의 관계를 탐구하는 상황에서 이 그래프 구성 과정을 설명합니다.

타격에는 출루율(OBP)로 측정되는 출루 능력과 장타율(SLG)로 측정되는 누상에 있는 주자를 진루시키는 능력이라는 두 가지 측면이 있습니다. 이 두 측정값의 산점도를 작성하여 선수들의 타격 성과를 더 잘 이해할 수 있습니다. geom_plot() 함수를 사용하여 OBP 및 SLG 산점도를 구성합니다(그림 3.9 참조).

(p <- ggplot(hof, aes(OBP, SLG)) + geom_point())

그림 3.9를 보면 이 디스플레이에 몇 가지 문제가 있음을 알 수 있습니다. 특히 두 축에 대해 더 설명적인 레이블을 사용하면 그래프를 읽기 더 쉬울 것입니다. 이러한 새로운 아이디어를 통합하기 위해 새 그림을 그립니다. xlab() 및 ylab() 함수를 사용하여 OBP 및 SLG를 각각

출루율(On-Base Percentage) 및 장타율(Slugging Percentage)로 바꿉니다. 업데이트된 디스플레이는 그림 3.10에 나와 있습니다.

(p <- p + xlab("On Base Percentage") +

ylab("Slugging Percentage"))

마찬가지로 scale_x_continuous() 및 scale_y_continuous() 함수를 직접 호출하여 한계와 레이블을 변경할 수 있습니다.

타격 성과에 대한 좋은 지표는 OPS = OBP + SLG로 정의되는 OPS 통계입니다. 그래프의 타자들을 OPS 기준으로 평가하기 위해 그래프에 OPS의 일정한 값을 그리는 것이 도움이 됩니다. OBP와 SLG를 x와 y로 나타낸다면, OPS = 0.7이거나 x + y = 0.7인 선을 그리고자 한다고 가정해 보겠습니다. 즉, 그래프에 함수 y = 0.7 − x를 그리기를 원합니다. 이는 ggplot2 시스템에서 geom_abline() 함수를 통해 수행되며, 함수의 인수는 slope = -1 및 intercept = 0.7로 주어집니다. 이와 유사하게 geom_abline() 함수를 세 번 더 적용하여 그래프에 OPS가 0.8, 0.9 및 1.0 값을 가지는 선을 그립니다. 결과 디스플레이는 그림 3.11에 나와 있습니다.

(p <- p +

geom_abline( slope = -1, intercept = seq(0.7, 1, by = 0.1), color = "red"

) )

마지막 반복 단계에서는 상수 OPS 값을 보여주는 선에 레이블을 추가하고 평생 OPS가

![image 40](images/imageFile40.png)

- 그림 3.11 기준선이 있는 명예의 전당 선수들의 OPS 및 SLG 값의 산점도.


1을 초과하는 선수들에 해당하는 점에 레이블을 지정합니다. 각 선 레이블은 annotate() 함수를 사용하여 수행됩니다. 세 가지 인수는 텍스트를 그릴 x 위치 및 y 위치이며, label은 표시할 텍스트 문자열의 벡터입니다(그림 3.12 참조).

p + annotate( "text", angle = -13,

- x = rep(0.31, 4) ,
- y = seq(0.4, 0.7, by = 0.1) + 0.02, label = paste("OPS = ", seq(0.7, 1, by = 0.1)), color = "red"


)

이러한 레이블을 수동으로 입력하는 대신 좌표와 레이블이 포함된 데이터 프레임을 만든 다음 geom_text() 함수를 사용하여 플롯에 레이블을 추가할 수 있습니다.

ops_labels <- tibble( OBP = rep(0.3, 4), SLG = seq(0.4, 0.7, by = 0.1) + 0.02, label = paste("OPS =", OBP + SLG), angle = -13

)

그림 3.11 기준선이 있는 명예의 전당 선수들의 OPS 및 SLG 값의 산점도.

![image 41](images/imageFile41.png)

- 그림 3.12 기준선 및 레이블이 있는 명예의 전당 선수들의 OPS 및 SLG 값의 산점도.


1을 초과하는 선수들에 해당하는 점에 레이블을 지정합니다. 각 선 레이블은 annotate() 함수를 사용하여 수행됩니다. 세 가지 인수는 텍스트를 그릴 x 위치 및 y 위치이며, label은 표시할 텍스트 문자열의 벡터입니다(그림 3.12 참조).

p + annotate( "text", angle = -13,

- x = rep(0.31, 4) ,
- y = seq(0.4, 0.7, by = 0.1) + 0.02, label = paste("OPS = ", seq(0.7, 1, by = 0.1)), color = "red"


)

이러한 레이블을 수동으로 입력하는 대신 좌표와 레이블이 포함된 데이터 프레임을 만든 다음 geom_text() 함수를 사용하여 플롯에 레이블을 추가할 수 있습니다.

p +

geom_text( data = ops_labels, hjust = "left", aes(label = label, angle = angle), color = "red"

)

이 최종 그래프는 이 명예의 전당 선수들의 타격 성과에 대해 많은 정보를 제공합니다. 다수의 타자들이 0.8과 0.9 사이의 통산 OPS 값을 가지며, 오직 6명의 선수(행크 그린버그, 로저스 혼스비, 지미 폭스, 테드 윌리엄스, 루 게릭, 베이브 루스)만이 통산 OPS 값이 1.0을 초과했음을 볼 수 있습니다. 주요 점 구름 오른쪽의 점들은 출루 능력은 강하지만 주자를 홈으로 진루시키는 능력은 상대적으로 약한 선수에 해당합니다. 반면 주요 점 구름 왼쪽의 점들은 출루보다는 장타에 더 뛰어난 타자들에 해당합니다.

ops_labels <- tibble( OBP = rep(0.3, 4), SLG = seq(0.4, 0.7, by = 0.1) + 0.02, label = paste("OPS =", OBP + SLG), angle = -13

)

3.6 숫자 변수 및 팩터 변수

OPS와 같은 숫자 변수와 시대와 같은 팩터를 수집할 때 팩터의 다른 값에 걸친 숫자 변수의 분포를 비교하는 데 보통 관심이 생깁니다. ggplot2 시스템에서 geom_jitter()

![image 42](images/imageFile42.png)

- 그림 3.13 시대별 홈런 비율의 1차원 산점도.


함수는 팩터 값에 대한 평행 스트립차트(stripcharts) 또는 숫자 선 그래프를 구성하는 데 사용할 수 있으며, geom_boxplot() 함수는 팩터 전체에서 평행 상자 그림(숫자 변수 요약 그래프)을 구성합니다.

홈런 타격은 야구 역사상 극적인 변화를 겪었으며, 야구 시대에 걸친 이러한 변화를 탐구하는 데 관심이 있다고 가정해 봅시다. 명예의 전당 선수들에 대해 HR/AB로 정의되는 홈런 비율에 초점을 맞춘다고 가정해 보겠습니다. 데이터 프레임 hof에 새로운 변수 hr_rate를 추가합니다.

hof <- hof |> mutate(hr_rate = HR / AB)

3.6.1 평행 스트립차트

geom_jitter() 함수를 사용하여 Era별 hr_rate의 평행 스트립차트를 구성합니다. x 및 y 미적 요소는 각각 hr_rate와 Era에 매핑됩니다. 점의 수직 흔들림 양을 줄이기 위해 height = 0.1 인수를 사용합니다.

ggplot(hof, aes(hr_rate, Era)) + geom_jitter(height = 0.1)

그림 3.13은 여러 시대에 걸쳐 홈런 타격 비율이 어떻게 변화했는지 보여줍니다. 홈런은 19세기 및 데드볼 시대에 드물었습니다. 라이블리볼 시대에는 홈런 타격이 여전히 비교적 낮았지만,

그림 3.13 시대별 홈런 비율의 1차원 산점도.

![image 43](images/imageFile43.png)

그림 3.14 시대별 홈런 비율의 평행 상자 그림.

함수는 팩터 값에 대한 평행 스트립차트 또는 숫자 선 그래프를 구성하는 데 사용할 수 있으며, geom_boxplot() 함수는 팩터 전체에서 평행 상자 그림(숫자 변수 요약 그래프)을 구성합니다.

홈런 타격은 야구 역사상 극적인 변화를 겪었으며, 야구 시대에 걸친 이러한 변화를 탐구하는 데 관심이 있다고 가정해 봅시다. 명예의 전당 선수들에 대해 HR/AB로 정의되는 홈런 비율에 초점을 맞춘다고 가정해 보겠습니다. 데이터 프레임 hof에 새로운 변수 hr_rate를 추가합니다.

hof <- hof |> mutate(hr_rate = HR / AB)

- 3.6.1 평행 스트립차트


geom_jitter() 함수를 사용하여 Era별 hr_rate의 평행 스트립차트를 구성합니다. x 및 y 미적 요소는 각각 hr_rate와 Era에 매핑됩니다. 점의 수직 흔들림 양을 줄이기 위해 height = 0.1 인수를 사용합니다.

ggplot(hof, aes(hr_rate, Era)) + geom_jitter(height = 0.1)

그림 3.13은 여러 시대에 걸쳐 홈런 타격 비율이 어떻게 변화했는지 보여줍니다. 홈런은 19세기 및 데드볼 시대에 드물었습니다. 라이블리볼 시대에는 홈런 타격이 여전히 비교적 낮았지만,

베이브 루스와 같이 비정상적으로 훌륭한 홈런 타자들이 있었습니다. 통합, 확장 및 자유계약 시대의 홈런 비율은 꽤 비슷했습니다.

- 3.6.2 평행 상자 그림


분포를 비교하기 위한 대체 디스플레이는 geom_boxplot() 함수를 사용합니다. 여기서 x 및 y 미적 요소는 각각 Era 및 hr_rate에 매핑됩니다. coord_flip() 함수는 축을 뒤집고 상자 그림을 가로로 표시합니다. color 및 fill 인수를 사용하여 갈색 테두리가 있는 주황색 상자 그림을 표시합니다.

ggplot(hof, aes(Era, hr_rate)) + geom_boxplot(color = "brown", fill = "orange") + coord_flip()

평행 상자 그림 디스플레이는 그림 3.14에 나와 있습니다. 디스플레이의 각 사각형은 하위 4분위수, 중앙값 및 상위 4분위수의 위치를 보여주고, 극단값에 선이 그려집니다. 나머지 분포에서 멀리 떨어진 비정상적인 점(이상치)은 상자 밖의 점으로 표시됩니다. 이 그래프는 우리가 스트립차트 디스플레이를 볼 때 한 관찰을 확증해 줍니다. 홈런 타격은 첫 두 시대에 낮았고 라이블리볼 시대에 증가하기 시작했습니다. 명예의 전당 선수들 중 유일한 이상치가 베이브 루스의 통산 홈런 비율 0.085라는 것은 흥미롭습니다.

###### 3.7 루스, 애런, 본즈, 에이로드 비교하기

1장에서 야구 역사상 위대한 슬러거 4명의 통산 홈런 궤적을 비교하는 그래프를 구성했습니다. 이 절에서는 R을 사용하여 이 그래프를 만든 방법을 설명합니다. 먼저 관련 데이터를 R로 로드해야 합니다. 다음으로 슬러거에 대한 홈런 및 나이 데이터가 포함된 데이터 프레임을 구성해야 합니다. 마지막으로 R 함수를 사용하여 그래프를 구성합니다.

- 3.7.1 데이터 가져오기


그래프를 얻으려면 각 슬러거 통산의 각 시즌에 대한 홈런 수, 타수 및 나이를 수집해야 합니다. Lahman 패키지에서 관련 데이터 프레임은 People 및 Batting입니다. 데이터 프레임 People에서 네 선수의 선수 아이디와 출생 연도를 얻습니다. Batting 데이터 프레임은 홈런 및 타수 정보를 추출하는 데 사용됩니다.

Lahman 패키지를 읽어들이는 것으로 시작합니다.

library(Lahman)

People 데이터 프레임에서 특정 선수의 선수 아이디와 출생 연도를 추출하고자 합니다.

- • filter() 함수는 People 데이터 프레임에서 각 선수의 아이디와 일치하는 행을 추출하는 데 사용됩니다.

- • 메이저리그 야구에서 선수의 특정 시즌 나이는 6월 30일의 나이로 정의됩니다. 그래서 선수의 생일이 첫 6개월에 해당하는지 여부에 따라 선수의 출생 연도를 약간 조정합니다. 조정된 출생 연도는 변수 mlb_birthyear에 저장됩니다. (if_else() 함수는 조건에 기반한 할당에 유용합니다. birthMonth >= 7이 TRUE이면 birthyear <- birthYear + 1이고 그렇지 않으면 birthyear <- birthYear입니다.)


PlayerInfo <- People |> filter( playerID %in% c( "ruthba01", "aaronha01", "bondsba01", "rodrial01"

) ) |> mutate(

mlb_birthyear = if_else(

birthMonth >= 7, birthYear + 1, birthYear ),

###### 3.7 루스, 애런, 본즈, 에이로드 비교하기

1장에서 야구 역사상 위대한 슬러거 4명의 통산 홈런 궤적을 비교하는 그래프를 구성했습니다. 이 절에서는 R을 사용하여 이 그래프를 만든 방법을 설명합니다. 먼저 관련 데이터를 R로 로드해야 합니다. 다음으로 슬러거에 대한 홈런 및 나이 데이터가 포함된 데이터 프레임을 구성해야 합니다. 마지막으로 R 함수를 사용하여 그래프를 구성합니다.

- 3.7.1 데이터 가져오기


그래프를 얻으려면 각 슬러거 통산의 각 시즌에 대한 홈런 수, 타수 및 나이를 수집해야 합니다. Lahman 패키지에서 관련 데이터 프레임은 People 및 Batting입니다. 데이터 프레임 People에서 네 선수의 선수 아이디와 출생 연도를 얻습니다. Batting 데이터 프레임은 홈런 및 타수 정보를 추출하는 데 사용됩니다.

Lahman 패키지를 읽어들이는 것으로 시작합니다.

library(Lahman)

People 데이터 프레임에서 특정 선수의 선수 아이디와 출생 연도를 추출하고자 합니다.

- • filter() 함수는 People 데이터 프레임에서 각 선수의 아이디와 일치하는 행을 추출하는 데 사용됩니다.
- • 메이저리그 야구에서 선수의 특정 시즌 나이는 6월 30일의 나이로 정의됩니다. 그래서 선수의 생일이 첫 6개월에 해당하는지 여부에 따라 선수의 출생 연도를 약간 조정합니다. 조정된 출생 연도는 변수 mlb_birthyear에 저장됩니다. (if_else() 함수는 조건에 기반한 할당에 유용합니다. birthMonth >= 7이 TRUE이면 birthyear <- birthYear + 1이고 그렇지 않으면 birthyear <- birthYear입니다.)


PlayerInfo <- People |> filter( playerID %in% c( "ruthba01", "aaronha01", "bondsba01", "rodrial01"

) ) |> mutate(

mlb_birthyear = if_else(

birthMonth >= 7, birthYear + 1, birthYear ),

루스, 애런, 본즈, 에이로드 비교하기 83

Player = paste(nameFirst, nameLast) ) |> select(playerID, Player, mlb_birthyear)

PlayerInfo 데이터 프레임에는 슬러거 베이브 루스, 행크 애런, 배리 본즈 및 알렉스 로드리게스에 대한 정보가 들어 있습니다.

- 3.7.2 선수 데이터 프레임 생성

이제 선수 아이디 코드와 출생 연도를 확보했으므로 이 정보를 Lahman 타격 데이터 프레임 Batting과 함께 사용하여 이 네 선수 각각에 대한 데이터 프레임을 만듭니다. 타격 데이터 프레임의 변수 중 하나는 playerID입니다. 베이브 루스의 타격 및 나이 데이터를 얻으려면 inner_join() 함수를 사용하여 타격 데이터의 행을 PlayerInfo 데이터 프레임에서 playerID가 동일한 행에 맞춥니다. 시즌 연도에서 선수의 출생 연도를 뺀 값으로 정의된 새 변수 Age를 만듭니다. (시즌에 맞는 선수의 정확한 나이를 얻기 위해 birthyear 변수를 약간 수정했던 것을 기억하시기 바랍니다.) 마지막으로 각 선수에 대해 그룹화된 데이터에 cumsum() 함수를 사용하여 각 선수의 매 시즌 홈런 누적 카운트를 포함하는 새 변수 cHR을 만듭니다.

HR_data <- Batting |> inner_join(PlayerInfo, by = "playerID") |> mutate(Age = yearID - mlb_birthyear) |> select(Player, Age, HR) |> group_by(Player) |> mutate(cHR = cumsum(HR))

- 3.7.3 그래프 구축


네 선수 각각의 나이에 대한 누적 홈런 카운트를 플롯하고자 합니다. 데이터 프레임 HR_data에서 관련 변수는 cHR, Age 및 Player입니다. geom_line() 함수를 사용하여 나이에 대한 누적 홈런 카운트 그래프를 그립니다. color 미적 요소를 Player 변수에 매핑하면 선수별로 구별되는 누적 홈런 선이 그려집니다. 네 선수에 대해 다른 색상이 사용되며 선 유형과 선수 이름을 연결하는 범례가 자동으로 생성된다는 점에 유의합니다. scale color manual 함수를 사용하면 플롯에 사용할 색상 집합을 지정할 수 있습니다. 이 경우 벡터 crc_fc는 정렬된 사전 정의 색상 집합을 포함합니다.

ggplot(HR_data, aes(x = Age, y = cHR, color = Player)) + geom_line() + scale_color_manual(values = crc_fc)

그림 3.15는 완성된 그래프를 보여줍니다.

![image 44](images/imageFile44.png)

그림 3.15 야구 선수 4명의 나이에 따른 누적 홈런 수.

###### 3.8 1998년 홈런 레이스

Retrosheet 플레이바이플레이 파일은 특정 야구 시즌 동안의 선수 성과 패턴을 배우는 데 도움이 됩니다. 우리는 R을 사용하여 1998년 시즌의 파일을 읽어오고 마크 맥과이어와 새미 소사 사이의 유명한 홈런 대결을 그래픽으로 봅니다.

3.8.1 데이터 가져오기

1998년 플레이바이플레이 데이터를 읽어 들여 데이터 프레임 retro1998에 저장하는 것으로 시작합니다. 이 파일을 만드는 방법에 대한 정보는 A.1.3절을 참조합니다.

retro1998 <- read_rds(here::here("data/retro1998.rds"))

플레이바이플레이 데이터에서 변수 bat_id는 타석에 있는 선수의 식별 코드를 나타냅니다. 맥과이어와 소사의 타격 데이터를 추출하려면 Lahman People 데이터 프레임에서 이 두 선수의 코드를 찾아야 합니다. filter() 함수를 사용하여 nameFirst = "Sammy" 및 nameLast = "Sosa"인 id 코드를 찾습니다. 마찬가지로 마크 맥과이어에 해당하는 id 코드를 찾습니다. 이 코드들은 변수 sosa_id 및 mac_id에 저장됩니다.

- 그림 3.15 야구 선수 4명의 나이에 따른 누적 홈런 수.


###### 3.8 1998년 홈런 레이스

Retrosheet 플레이바이플레이 파일은 특정 야구 시즌 동안의 선수 성과 패턴을 배우는 데 도움이 됩니다. 우리는 R을 사용하여 1998년 시즌의 파일을 읽어오고 마크 맥과이어와 새미 소사 사이의 유명한 홈런 대결을 그래픽으로 봅니다.

- 3.8.1 데이터 가져오기


1998년 플레이바이플레이 데이터를 읽어 들여 데이터 프레임 retro1998에 저장하는 것으로 시작합니다. 이 파일을 만드는 방법에 대한 정보는 A.1.3절을 참조합니다.

retro1998 <- read_rds(here::here("data/retro1998.rds"))

플레이바이플레이 데이터에서 변수 bat_id는 타석에 있는 선수의 식별 코드를 나타냅니다. 맥과이어와 소사의 타격 데이터를 추출하려면 Lahman People 데이터 프레임에서 이 두 선수의 코드를 찾아야 합니다. filter() 함수를 사용하여 nameFirst = "Sammy" 및 nameLast = "Sosa"인 id 코드를 찾습니다. 마찬가지로 마크 맥과이어에 해당하는 id 코드를 찾습니다. 이 코드들은 변수 sosa_id 및 mac_id에 저장됩니다.

1998년 홈런 레이스 85

sosa_id <- People |> filter(nameFirst == "Sammy", nameLast == "Sosa") |> pull(retroID)

mac_id <- People |> filter(nameFirst == "Mark", nameLast == "McGwire") |> pull(retroID)

이제 선수 아이디 코드를 얻었으므로 플레이바이플레이 데이터 프레임 retro1998에서 맥과이어와 소사의 타석 데이터를 추출합니다. 이 데이터는 데이터 프레임 hr_race에 저장됩니다.

hr_race <- retro1998 |> filter(bat_id %in% c(sosa_id, mac_id))

- 3.8.2 변수 추출


각 선수에 대해 각 타석의 현재까지 친 홈런 수를 수집하고 홈런 수에 대한 날짜 그래프를 작성하는 데 관심이 있습니다. 각 선수에 대해 중요한 두 변수는 날짜와 홈런 수입니다. 선수의 플레이바이플레이 타격 데이터가 주어졌을 때 이 두 변수를 추출하는 새 함수 cum_hr()을 작성합니다.

플레이바이플레이 데이터 프레임에서 변수 game_id는 경기 위치와 날짜를 식별합니다. 예를 들어 ARI199805110이라는 game_id 값은 이 특정 플레이가 1998년 5월 11일 애리조나에서 열린 경기에서 발생했음을 나타냅니다. (이 변수는 "위치, 연도, 월, 일" 형식으로 표시됩니다.) str_sub() 함수를 사용하여 이 문자열 변수의 4번째에서 11번째 문자를 선택하고 이 날짜를 변수 Date에 할당합니다. (ymd() 함수는 날짜를 더 읽기 쉬운 "연-월-일" 형식으로 변환하고 R이 이를 Date로 인식하도록 강제합니다.) arrange() 함수를 사용하여 플레이바이플레이 데이터를 시즌 시작부터 끝까지 정렬합니다. 변수 event_cd는 타격 플레이의 결과를 포함합니다. event_cd 값이 23이면 홈런을 쳤음을 나타냅니다. 홈런 발생 여부에 따라 1 또는 0으로 새 변수 HR을 정의하고, 새 변수 cumHR은 cumsum() 함수를 사용하여 시즌 중 누적 홈런 수를 기록합니다. 함수의 출력은 각 날짜와 시즌 중 모든 타석에 대한 해당 날짜까지의 누적 홈런 수를 포함하는 새로운 데이터 프레임입니다.

cum_hr <- function(data) {

data |> mutate(Date = ymd(str_sub(game_id, 4, 11))) |> arrange(Date) |> mutate(

}

HR = if_else(event_cd == 23, 1, 0), cumHR = cumsum(HR)

) |> select(Date, cumHR)

hr_race 데이터 프레임을 선수별로 그룹화하고 해당 선수 아이디를 수집한 후, group_split() 및 map() 함수를 사용하여 소사의 타격 데이터와 맥과이어의 타격 데이터에 대해 각각 한 번씩 cum_hr()을 두 번 반복 실행하여 새 데이터 프레임 hr_ytd를 얻습니다.

hr_grouped <- hr_race |> group_by(bat_id)

keys <- hr_grouped |> group_keys() |> pull(bat_id)

hr_ytd <- hr_grouped |> group_split() |> map(cum_hr) |> set_names(keys) |> bind_rows(.id = "bat_id") |> inner_join(People, by = c("bat_id" = "retroID"))

- 3.8.3 그래프 구축


이 새로운 데이터 프레임이 생성되면 관심 있는 그래프를 작성하는 것은 간단합니다. geom_line() 함수는 날짜 대비 누적 홈런 수의 그래프를 구성합니다. nameLast를 color 미적 요소에 매핑하면 두 선수에 해당하는 선이 서로 다른 색상으로 그려집니다. geom_hline() 함수를 사용하여 홈런 값 62에 수평선을 추가하고 annotate() 함수를 적용하여 이 플롯된 선 위에 "62"라는 텍스트 문자열을 배치합니다(그림 3.16 참조).

ggplot(hr_ytd, aes(Date, cumHR, color = nameLast)) + geom_line() + geom_hline(yintercept = 62, color = crcblue) + annotate(

"text", ymd("1998-04-15"), 65, label = "62", color = crcblue

) + ylab("Home Runs in the Season")

}

HR = if_else(event_cd == 23, 1, 0), cumHR = cumsum(HR)

) |> select(Date, cumHR)

hr_race 데이터 프레임을 선수별로 그룹화하고 해당 선수 아이디를 수집한 후, group_split() 및 map() 함수를 사용하여 소사의 타격 데이터와 맥과이어의 타격 데이터에 대해 각각 한 번씩 cum_hr()을 두 번 반복 실행하여 새 데이터 프레임 hr_ytd를 얻습니다.

hr_grouped <- hr_race |> group_by(bat_id)

keys <- hr_grouped |> group_keys() |> pull(bat_id)

hr_ytd <- hr_grouped |> group_split() |> map(cum_hr) |> set_names(keys) |> bind_rows(.id = "bat_id") |> inner_join(People, by = c("bat_id" = "retroID"))

3.8.3 그래프 구축

이 새로운 데이터 프레임이 생성되면 관심 있는 그래프를 작성하는 것은 간단합니다. geom_line() 함수는 날짜 대비 누적 홈런 수의 그래프를 구성합니다. nameLast를 color 미적 요소에 매핑하면 두 선수에 해당하는 선이 서로 다른 색상으로 그려집니다. geom_hline() 함수를 사용하여 홈런 값 62에 수평선을 추가하고 annotate() 함수를 적용하여 이 플롯된 선 위에 "62"라는 텍스트 문자열을 배치합니다(그림 3.16 참조).

ggplot(hr_ytd, aes(Date, cumHR, color = nameLast)) + geom_line() + geom_hline(yintercept = 62, color = crcblue) + annotate(

"text", ymd("1998-04-15"), 65, label = "62", color = crcblue

) + ylab("Home Runs in the Season")

더 읽을거리 87

![image 45](images/imageFile45.png)

그림 3.16 새미 소사와 마크 맥과이어의 1998년 홈런 레이스 그래프.

###### 3.9 더 읽을거리

R의 전통적인 그래픽 시스템에 대한 좋은 참조는 Murrell(2006)입니다. Kabacoff(2010)와 https://www.statmeth ods.net의 Quick-R 웹 사이트는 특정 그래픽 함수에 대한 유용한 참조를 제공합니다. Albert와 Rizzo(2012)의 4장에서는 플롯 유형 및 기호 변경, 색상 사용, 곡선 및 수식 중첩과 같은 R의 전통적인 그래픽을 수정하는 여러 가지 예를 제공합니다. Wickham, C¸etinkaya-Rundel, 및 Grolemund(2023), Benjamin S. Baumer, Kaplan, 및 Horton(2021b), 그리고 Ismay와 Kim(2019) 모두 데이터 그래픽 생성을 위한 ggplot2 사용에 대해 논의합니다.

###### 3.10 연습 문제

- 1. 명예의 전당 투구 데이터 세트


abdwr3edata 패키지의 hof_pitching 데이터 프레임에는 명예의 전당에 입성한 모든 투수의 통산 투구 통계가 포함되어 있습니다. 변수 BF는 투수가 선수 생활 동안 상대한 타자 수입니다. 이 변수를 사용하여 투수를 (0, 10,000), (10,000, 15,000), (15,000, 20,000), (20,000, 30,000) 구간으로 그룹화한다고 가정해 봅시다. cut() 함수를 사용하여 변수 BF를 그룹화된 변수 BF_group으로 재표현할 수 있습니다.

hofpitching <- hofpitching |> mutate(

BF_group = cut( BF, c(0, 10000, 15000, 20000, 30000), labels = c("Less than 10000", "(10000, 15000)",

"(15000, 20000)", "more than 20000") )

)

- a. summarize() 함수를 사용하여 BF.group의 빈도표를 작성합니다.
- b. summarize()의 출력을 기반으로 막대 그래프를 작성합니다. 얼마나 많은 명예의 전당 투수들이 통산 20,000명 이상의 투수를 상대했습니까?
- c. BF.group 변수의 대안적인 그래프를 작성합니다. 네 구간의 빈도를 비교하는 데 있어서 막대 그래프와 새 그래프의 효과성을 비교합니다.


- 2. 명예의 전당 투구 데이터 세트 (계속)

변수 WAR은 투수가 선수 생활 동안 기록한 대체 선수 대비 승리 기여도의 총합입니다.

- a. geom_histogram() 함수를 사용하여 명예의 전당 데이터 세트에 있는 투수들의 WAR 히스토그램을 작성합니다.
- b. 총 WAR 변수에서 모든 명예의 전당 선수들 중 돋보이는 두 명의 투수가 있습니다. 이 두 투수를 식별합니다.


- 3. 명예의 전당 투구 데이터 세트 (계속)

투수의 시즌 기여도를 이해하기 위해 다음으로 정의된 새로운 변수 WAR_Season을 정의한다고 가정해 봅시다.

hofpitching <- hofpitching |> mutate(WAR_Season = WAR / Yrs)

- a. geom_point() 함수를 사용하여 BP.group의 다양한 수준에 대한 WAR.Season의 평행 1차원 산점도를 작성합니다.
- b. geom_boxplot() 함수를 사용하여 BP.group 전체에 대한 WAR.Season의 평행 상자 그림을 작성합니다.
- c. 작성한 그래프에 따르면 시즌당 대체 선수 대비 승리 기여도는 상대한 타자 수에 어떻게 의존합니까?


- 4. 명예의 전당 투구 데이터 세트 (계속)


통산 중반기가 1960년 이후인 투수로 탐구를 제한한다고 가정해 봅시다. 먼저 MidYear 변수를 정의한 다음 filter() 함수를 사용하여 이러한 1960년 이후 투수들로만 구성된 데이터 프레임을 구축합니다.

hofpitching <- hofpitching |> mutate(

BF_group = cut( BF, c(0, 10000, 15000, 20000, 30000), labels = c("Less than 10000", "(10000, 15000)",

"(15000, 20000)", "more than 20000") )

)

- a. summarize() 함수를 사용하여 BF.group의 빈도표를 작성합니다.
- b. summarize()의 출력을 기반으로 막대 그래프를 작성합니다. 얼마나 많은 명예의 전당 투수들이 통산 20,000명 이상의 투수를 상대했습니까?
- c. BF.group 변수의 대안적인 그래프를 작성합니다. 네 구간의 빈도를 비교하는 데 있어서 막대 그래프와 새 그래프의 효과성을 비교합니다.


- 2. 명예의 전당 투구 데이터 세트 (계속)

변수 WAR은 투수가 선수 생활 동안 기록한 대체 선수 대비 승리 기여도의 총합입니다.

- a. geom_histogram() 함수를 사용하여 명예의 전당 데이터 세트에 있는 투수들의 WAR 히스토그램을 작성합니다.
- b. 총 WAR 변수에서 모든 명예의 전당 선수들 중 돋보이는 두 명의 투수가 있습니다. 이 두 투수를 식별합니다.


- 3. 명예의 전당 투구 데이터 세트 (계속)

투수의 시즌 기여도를 이해하기 위해 다음으로 정의된 새로운 변수 WAR_Season을 정의한다고 가정해 봅시다.

hofpitching <- hofpitching |> mutate(WAR_Season = WAR / Yrs)

- a. geom_point() 함수를 사용하여 BP.group의 다양한 수준에 대한 WAR.Season의 평행 1차원 산점도를 작성합니다.
- b. geom_boxplot() 함수를 사용하여 BP.group 전체에 대한 WAR.Season의 평행 상자 그림을 작성합니다.
- c. 작성한 그래프에 따르면 시즌당 대체 선수 대비 승리 기여도는 상대한 타자 수에 어떻게 의존합니까?


- 4. 명예의 전당 투구 데이터 세트 (계속)


통산 중반기가 1960년 이후인 투수로 탐구를 제한한다고 가정해 봅시다. 먼저 MidYear 변수를 정의한 다음 filter() 함수를 사용하여 이러한 1960년 이후 투수들로만 구성된 데이터 프레임을 구축합니다.

연습 문제 89

hofpitching <- hofpitching |> mutate(MidYear = (From + To) / 2) hofpitching.recent <- hofpitching |> filter(MidYear >= 1960)

- a. arrange() 함수를 사용하여 데이터 프레임의 행을 WAR_Season 값에 따라 정렬합니다.
- b. 레이블이 투수 이름인 WAR_Season 값의 점 플롯을 작성합니다.
- c. 시즌당 대체 선수 대비 승리 기여도 측면에서 어떤 1960년 이후 투수 두 명이 돋보입니까?


- 5. 명예의 전당 투구 데이터 세트 (계속) 변수 MidYear와 WAR_Season은 이전 연습 문제에 정의되어 있습니다.

- a. MidYear(가로) 대 WAR_Season(세로)의 산점도를 작성합니다.
- b. 이 산점도에 일반적인 패턴이 있습니까? 설명합니다.
- c. 통산 중반기가 1800년대였던 투수 중 상대적으로 낮은 WAR_Season 값을 기록한 선수가 두 명 있습니다. filter() 및 geom_text() 함수를 사용하여 이 두 투수의 이름을 산점도에 추가합니다.


- 6. Lahman 타격 데이터 세트로 작업하기

- a. Lahman People 및 Batting 데이터 프레임을 R로 읽어옵니다.
- b. 위대한 타자인 타이 콥, 테드 윌리엄스, 피트 로즈의 시즌 타격 통계를 단일 데이터 프레임으로 수집합니다.
- c. 세 선수의 나이에 해당하는 변수 Age를 각 데이터 프레임에 추가합니다.
- d. geom_line() 함수를 사용하여 피트 로즈의 나이에 따른 누적 안타 수의 선 그래프를 작성합니다.
- e. geom_line() 함수를 사용하여 콥과 윌리엄스의 누적 안타 수를 겹쳐 그립니다.
- f. 이 세 선수의 타격 패턴에 대해 배운 내용을 요약하는 짧은 단락을 작성합니다.


- 7. Lahman 팀 데이터 세트로 작업하기


Lahman Teams 데이터 세트에는 메이저리그 역사의 모든 팀에 대한 연도별 통계 및 순위 정보가 포함되어 있습니다.

- a. Teams 데이터 프레임을 R로 읽어옵니다.
- b. 팀 승률 W / (W + L)로 정의되는 새로운 변수 win_pct를 생성합니다.
- c. 2022년 시즌의 팀들에 대해 팀 ERA와 승률의 산점도를 작성합니다.
- d. mlbplotR 패키지의 geom_mlb_scoreboard_logos() 함수를 사용하여 산점도에 플로팅 마크로 팀 로고를 넣습니다.


이 함수를 사용하여 (c) 부분의 그래프를 팀 로고를 사용하여 다시 그립니다.

- 8. Retrosheet 플레이바이플레이 데이터 세트로 작업하기


3.8절에서는 Retrosheet 플레이바이플레이 데이터를 사용하여 1998년 시즌 동안 마크 맥과이어와 새미 소사 간의 홈런 레이스를 탐구했습니다. 두 선수의 홈런 타격 패턴을 비교하는 또 다른 방법은 홈런 사이의 타석 수인 간격을 계산하는 것입니다.

- a. 3.8절의 작업을 따라 두 선수에 대한 타격 데이터가 포함된 두 데이터 프레임 mac_data 및 sosa_data를 생성합니다.
- b. 다음 R 명령을 사용하여 타격 이벤트가 발생한 플레이로 두 데이터 프레임을 제한합니다. (관련 변수 bat_event_fl은 TRUE 또는 FALSE입니다.)

mac_data <- filter(mac_data, bat_event_fl == TRUE) sosa_data <- filter(sosa_data, bat_event_fl == TRUE)

- c. 각 데이터 프레임에 대해 타석에 1, 2, ... 번호를 매기는 새로운 변수 PA를 생성합니다. (nrow() 함수는 데이터 프레임의 행 수를 제공합니다.)

mac_data <- mutate(mac_data, PA = 1:nrow(.)) sosa_data <- mutate(sosa_data, PA = 1:nrow(.))

- d. 다음 명령은 선수들이 홈런을 쳤을 때 타석의 번호를 반환합니다.

mac_HRPA <- mac.data |> filter(event_cd == 23) |> pull(PA)

sosa_HRPA <- sosa.data |> filter(event_cd == 23) |> pull(PA)

- e. R 함수 diff()를 사용하여 다음 명령은 홈런 발생 사이의 간격을 계산합니다.


mac_spacings <- diff(c(0, mac_HRPA)) sosa_spacings <- diff(c(0, sosa_HRPA))

선수 이름인 Player와 간격 값인 Spacing이라는 두 개의 변수가 있는 새로운 데이터 프레임 HR_Spacing을 만듭니다. f. HR_Spacing 데이터 프레임에 summarize() 및 geom_histogram() 함수를 사용하여 두 선수의 홈런 간격을 비교합니다.

8. Retrosheet 플레이바이플레이 데이터 세트로 작업하기

3.8절에서는 Retrosheet 플레이바이플레이 데이터를 사용하여 1998년 시즌 동안 마크 맥과이어와 새미 소사 간의 홈런 레이스를 탐구했습니다. 두 선수의 홈런 타격 패턴을 비교하는 또 다른 방법은 홈런 사이의 타석 수인 간격을 계산하는 것입니다.

- a. 3.8절의 작업을 따라 두 선수에 대한 타격 데이터가 포함된 두 데이터 프레임 mac_data 및 sosa_data를 생성합니다.
- b. 다음 R 명령을 사용하여 타격 이벤트가 발생한 플레이로 두 데이터 프레임을 제한합니다. (관련 변수 bat_event_fl은 TRUE 또는 FALSE입니다.)

mac_data <- filter(mac_data, bat_event_fl == TRUE) sosa_data <- filter(sosa_data, bat_event_fl == TRUE)

- c. 각 데이터 프레임에 대해 타석에 1, 2, ... 번호를 매기는 새로운 변수 PA를 생성합니다. (nrow() 함수는 데이터 프레임의 행 수를 제공합니다.)

mac_data <- mutate(mac_data, PA = 1:nrow(.)) sosa_data <- mutate(sosa_data, PA = 1:nrow(.))

- d. 다음 명령은 선수들이 홈런을 쳤을 때 타석의 번호를 반환합니다.

mac_HRPA <- mac.data |> filter(event_cd == 23) |> pull(PA)

sosa_HRPA <- sosa.data |> filter(event_cd == 23) |> pull(PA)

- e. R 함수 diff()를 사용하여 다음 명령은 홈런 발생 사이의 간격을 계산합니다.


mac_spacings <- diff(c(0, mac_HRPA)) sosa_spacings <- diff(c(0, sosa_HRPA))

선수 이름인 Player와 간격 값인 Spacing이라는 두 개의 변수가 있는 새로운 데이터 프레임 HR_Spacing을 만듭니다. f. HR_Spacing 데이터 프레임에 summarize() 및 geom_histogram() 함수를 사용하여 두 선수의 홈런 간격을 비교합니다.

