# 3장. R 모델링 기초 복습 (A Review of R Modeling Fundamentals)

모델을 구축할 때 깔끔한 데이터(tidy data) 원칙을 적용하기 위해 tidymodels를 사용하는 방법을 설명하기 전에, 핵심 R 언어(종종 base R이라고 불림)에서 모델이 어떻게 생성되고 훈련되며 사용되는지 복습해 보겠습니다. 이 장은 모델에 base R을 전혀 사용하지 않는다고 하더라도 알아두는 것이 중요한 핵심 언어의 규약들을 간략하게 보여줍니다. 이 장이 철저하고 완벽한 것은 아니지만, 독자(특히 R을 처음 접하는 독자)에게 일반적으로 사용되는 기본적인 모티프(motifs)를 제공합니다.

R의 기반이 되는 S 언어는 Chambers와 Hastie (1992)(일반적으로 The White Book으로 알려져 있음)가 출판된 이래로 풍부한 데이터 분석 환경을 갖추고 있었습니다. 이 S 언어 버전은 상징적(symbolic) 모델 공식(model formulas), 모델 행렬(model matrices) 및 데이터 프레임(data frames)과 같은 오늘날 R 사용자에게 친숙한 표준 인프라 구성 요소를 도입했으며, 데이터 분석을 위한 표준 객체 지향 프로그래밍(object-oriented programming) 방법도 도입했습니다. 이러한 사용자 인터페이스는 그 이후로 실질적으로 변경되지 않았습니다.

# 예제 (An Example)

base R에서 모델링을 위한 몇 가지 기초를 시연하기 위해, Mangiafico (2015)를 거쳐 McDonald (2009)의 실험 데이터를 사용하여 주변 온도(ambient temperature)와 분당 귀뚜라미 울음소리 빈도(rate of cricket chirps) 사이의 관계를 살펴보겠습니다. 데이터는 두 종(_O. exclamationis_ 및 _O. niveus_)에 대해 수집되었습니다. 데이터는 총 31개의 데이터 포인트를 가진 `crickets`라는 데이터 프레임에 포함되어 있습니다. 이 데이터는 다음 ggplot2 코드를 사용하여 [그림 3-1](#figure-3-1.-relationship-between-chirp-rate-and-temperature-for-two-species-of-cricket.)에 나와 있습니다.

```
library(tidyverse)

data(crickets, package = "modeldata")
names(crickets)

# x축에는 온도를, y축에는 울음소리 빈도를 도식화합니다.
# 플롯 요소들은 각 종마다 다르게 색칠됩니다.
ggplot(crickets,
       aes(x = temp, y = rate, color = species, pch = species, lty = species)) +
  # 각 데이터 포인트에 대해 점을 도식화하고 종별로 색칠합니다.
  geom_point(size = 2) +
  # 각 종에 대해 별도로 생성된 단순 선형 모델 피팅을 보여줍니다.
  geom_smooth(method = lm, se = FALSE, alpha = 0.5) +
  scale_color_brewer(palette = "Paired") +
  labs(x = "Temperature (C)", y = "Chirp Rate (per minute)")
```

\#\> \[1\] "species" "temp" "rate"

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0301.png" alt="tmwr 0301" />
<h6 id="figure-3-1.-relationship-between-chirp-rate-and-temperature-for-two-species-of-cricket.">그림 3-1. 두 종의 귀뚜라미에 대한 울음소리 빈도와 온도 간의 관계.</h6>
</figure>

데이터는 각 종에 대해 상당히 선형적인(linear) 추세를 나타냅니다. 주어진 온도에서 *O. exclamationis*가 다른 종보다 분당 더 많이 우는 것으로 보입니다. 추론적 모델의 경우, 연구자들은 데이터를 보기 전에 다음과 같은 귀무가설을 지정했을 수 있습니다.

- 온도는 울음소리 빈도에 아무런 영향을 미치지 않는다.

- 두 종의 울음소리 빈도 사이에는 차이가 없다.

울음소리 빈도를 예측하는 데 과학적이거나 실용적인 가치가 있을 수 있지만, 이 예제에서는 추론(inference)에 초점을 맞출 것입니다.

R에서 일반 선형 모델(ordinary linear model)을 피팅(fit)하려면 `lm()` 함수가 일반적으로 사용됩니다. 이 함수의 중요한 인수는 모델 공식(model formula)과 데이터가 포함된 데이터 프레임입니다. 공식은 *상징적(symbolic)*입니다. 예를 들어, 단순한 공식인:

```
rate ~ temp
```

은 울음소리 빈도가 결과 변수이고(물결표 `~`의 왼쪽에 있으므로), 온도 값이 예측 변수임을 지정합니다.<sup>[1](ch03.xhtml#idm45881871527184-marker)</sup>

만약 데이터에 측정된 시간대를 포함하는 `time`이라는 열이 있다고 가정해 봅시다. 이 경우 공식:

```
rate ~ temp + time
```

은 시간과 온도 값을 서로 더하지 않습니다. 이 공식은 온도와 시간이 모델에 별도의 *주효과(main effects)*로 추가되어야 함을 상징적으로 나타냅니다. 주효과는 단일 예측 변수를 포함하는 모델의 항(term)입니다.

이 데이터에는 시간 측정값이 없지만, 종(species)도 동일한 방식으로 모델에 추가될 수 있습니다.

```
rate ~ temp + species
```

종(species)은 양적(quantitative) 변수가 아닙니다. 데이터 프레임에서 이 변수는 `"O. exclamationis"`와 `"O. niveus"`라는 두 개의 수준(levels)을 가진 팩터(factor) 열로 표시됩니다. 대다수의 모델 함수는 숫자가 아닌 데이터에 대해 작동할 수 없습니다. 종의 경우, 모델은 종 데이터를 숫자 형식으로 인코딩해야 합니다. 일반적인 접근 방식은 원래의 질적 값 대신 _지시 변수(indicator variables)_ (가변수(dummy variables)라고도 함)를 사용하는 것입니다. 이 예에서 종은 두 가지 가능한 값을 갖기 때문에, 모델 공식은 종이 `"O. exclamationis"`일 때 값이 0이고 데이터가 `"O. niveus"`에 해당할 때 값이 1인 새 열을 추가함으로써 이 열을 자동으로 숫자로 인코딩합니다. 근본적인 공식 메커니즘은 모델을 생성하는 데 사용된 데이터 세트뿐만 아니라 모든 새로운 데이터 포인트(예를 들어, 모델이 예측에 사용될 때)에 대해서도 이러한 값을 자동으로 변환합니다.

###### 참고 (Note)

두 종이 아니라 다섯 종이 있다고 가정해 보십시오. 모델 공식은 네 종에 대한 이진 지시자(binary indicators)인 네 개의 추가적인 이진 열을 자동으로 추가할 것입니다. 팩터의 _참조 수준(reference level)_ (즉, 첫 번째 수준)은 항상 예측 변수 집합에서 제외됩니다. 그 아이디어는 네 개의 지시 변수의 값을 안다면 종의 값을 결정할 수 있다는 것입니다. [8장](ch08.xhtml#recipes)에서 이진 지시 변수에 대해 더 자세히 논의합니다.

모델 공식 `rate ~ temp + species`는 각 종에 대해 서로 다른 y 절편을 갖는 모델을 생성합니다. 회귀선의 기울기 또한 각 종마다 다를 수 있습니다. 이러한 구조를 수용하기 위해 상호작용 항(interaction term)을 모델에 추가할 수 있습니다. 이는 몇 가지 다른 방식으로 지정할 수 있으며, 기본적인 방법은 콜론(`:`)을 사용하는 것입니다.

```
rate ~ temp + species + temp:species

# 두 변수와의 상호작용을 포함하는 모든 상호작용을
# 확장하기 위해 단축키를 사용할 수 있습니다.
rate ~ (temp + species)^2

# 가능한 모든 상호작용을 포함하도록 팩터를 확장하는
# 또 다른 단축키 (이 예제에서는 위와 동일):
rate ~ temp * species
```

지시 변수를 자동으로 생성하는 편리함 외에도, 공식(formula)은 몇 가지 다른 좋은 기능들을 제공합니다.

- 공식 내에서 _인라인(In-line)_ 함수를 사용할 수 있습니다. 예를 들어 온도의 자연 로그를 사용하려면 `rate ~ log(temp)` 공식을 만들 수 있습니다. 기본적으로 공식이 상징적이기 때문에, 항등 함수(identity function) `I()`를 사용하여 예측 변수에 리터럴(literal) 수학을 적용할 수도 있습니다. 화씨 단위를 사용하려는 경우 공식은 섭씨에서 변환하기 위해 `rate ~ I( (temp * 9/5) + 32 )`가 될 수 있습니다.

- R에는 공식 내부에서 유용하게 쓰이는 함수가 많습니다. 예를 들어 `poly(x, 3)`은 `x`에 대한 선형(linear), 이차(quadratic) 및 삼차(cubic) 항을 주효과로 모델에 생성합니다. splines 패키지에도 공식 내에서 비선형 스플라인 항을 생성하는 여러 함수가 있습니다.

- 예측 변수가 많은 데이터 세트의 경우 마침표(`.`) 단축키를 사용할 수 있습니다. 마침표는 물결표의 왼쪽에 있지 않은 모든 열에 대한 주효과를 나타냅니다. `~ (.)^3`을 사용하면 주효과뿐만 아니라 모델에 모든 2변수 및 3변수 상호작용이 생성됩니다.

다시 귀뚜라미 울음소리로 돌아가서 2원 상호작용 모델(two-way interaction model)을 사용해 봅시다. 이 책에서는 피팅된 모델인 R 객체에 대해 `_fit` 접미사를 사용합니다.

```
interaction_fit <-  lm(rate ~ (temp + species)^2, data = crickets)

# 모델의 간단한 요약을 인쇄하려면:
interaction_fit
#>
#> Call:
#> lm(formula = rate ~ (temp + species)^2, data = crickets)
#>
#> Coefficients:
#>           (Intercept)                   temp       speciesO. niveus
#>               -11.041                  3.751                 -4.348
#> temp:speciesO. niveus
#>                -0.234
```

이 출력은 약간 읽기 어렵습니다. 종 지시 변수의 경우, R은 구분 기호 없이 변수 이름(`species`)을 팩터 수준(`O. niveus`)과 함께 뭉쳐 놓습니다.

이 모델에 대한 어떠한 추론 결과로 들어가기 전에, 진단 도식(diagnostic plots)을 사용하여 피팅을 평가해야 합니다. `lm` 객체에 대해 `plot()` 메소드를 사용할 수 있습니다. 이 메소드는 [그림 3-2](#figure-3-2.-residual-diagnostic-plots-for-the-linear-model-with-interactions-which-appear-reasonable-enough-to-conduct-inferential-analysis.)와 같이 객체에 대해 4개의 플롯 집합을 생성하며, 각 플롯은 피팅의 서로 다른 측면을 보여줍니다.

```
# 두 개의 플롯을 나란히 배치:
par(mfrow = c(1, 2))

# 예측값 대비 잔차 표시:
plot(interaction_fit, which = 1)

# 잔차에 대한 정규 분위수 도식(normal quantile plot):
plot(interaction_fit, which = 2)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0302.png" alt="tmwr 0302" />
<h6 id="figure-3-2.-residual-diagnostic-plots-for-the-linear-model-with-interactions-which-appear-reasonable-enough-to-conduct-inferential-analysis.">그림 3-2. 상호작용이 있는 선형 모델에 대한 잔차 진단 도식. 추론적 분석을 수행하기에 충분히 합리적으로 보임.</h6>
</figure>

###### 참고 (Note)

표현식을 평가하는 기술적인 세부 사항에 관한 한 R은 (적극적이기보다는) _게으릅니다(lazy)_. 이는 모델 피팅 함수가 일반적으로 가능한 마지막 순간에 최소한의 가능한 수량을 계산함을 의미합니다. 예를 들어 각 모델 항의 계수 테이블에 관심이 있는 경우, 이는 모델과 함께 자동으로 계산되지 않고 대신 `summary()` 메소드를 통해 계산됩니다.

귀뚜라미에 대한 우리의 다음 작업 순서는 상호작용 항의 포함이 필요한지 평가하는 것입니다. 이 모델에 적절한 접근 방식은 상호작용 항 없이 모델을 다시 계산하고 `anova()` 메소드를 사용하는 것입니다.

```
# 축소된 모델(reduced model) 피팅:
main_effect_fit <-  lm(rate ~ temp + species, data = crickets)

# 두 모델 비교:
anova(main_effect_fit, interaction_fit)
#> Analysis of Variance Table
#>
#> Model 1: rate ~ temp + species
#> Model 2: rate ~ (temp + species)^2
#>   Res.Df  RSS Df Sum of Sq    F Pr(>F)
#> 1     28 89.3
#> 2     27 85.1  1      4.28 1.36   0.25
```

이 통계적 검정은 0.25의 p-값을 산출합니다. 이는 모델에 상호작용 항이 필요하지 않다는 귀무가설에 반대하는 증거가 부족함을 암시합니다. 이러한 이유로 우리는 상호작용이 없는 모델에 대해 추가적인 분석을 수행할 것입니다.

잔차 도식을 재평가하여 우리의 이론적 가정이 모델이 생성한 p-값을 신뢰할 만큼 충분히 유효한지 확인해야 합니다(플롯은 여기에 표시되지 않지만, 미리 말씀드리자면 유효합니다).

`summary()` 메소드를 사용하여 각 모델 항의 계수, 표준 오차(standard errors) 및 p-값을 검사할 수 있습니다.

```
summary(main_effect_fit)
#>
#> Call:
#> lm(formula = rate ~ temp + species, data = crickets)
#>
#> Residuals:
#>    Min     1Q Median     3Q    Max
#> -3.013 -1.130 -0.391  0.965  3.780
#>
#> Coefficients:
#>                  Estimate Std. Error t value Pr(>|t|)
#> (Intercept)       -7.2109     2.5509   -2.83   0.0086 **
#> temp               3.6028     0.0973   37.03  < 2e-16 ***
#> speciesO. niveus -10.0653     0.7353  -13.69  6.3e-14 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#>
#> Residual standard error: 1.79 on 28 degrees of freedom
#> Multiple R-squared:  0.99,   Adjusted R-squared:  0.989
#> F-statistic: 1.33e+03 on 2 and 28 DF,  p-value: <2e-16
```

온도가 1도 상승함에 따라 각 종의 울음소리 빈도는 3.6회 증가합니다. 이 항은 p-값으로 증명되듯 강한 통계적 유의성을 보여줍니다. 종(species) 항은 –10.07의 값을 갖습니다. 이는 모든 온도 값에 걸쳐 *O. niveus*가 *O. exclamationis*보다 분당 약 10회 더 적게 우는 것을 나타냅니다. 온도 항과 마찬가지로 종 효과는 매우 작은 p-값과 관련되어 있습니다.

이 분석에서 유일한 문제는 절편(intercept) 값입니다. 0°C에서 두 종 모두 분당 울음소리가 음수임을 나타냅니다. 이는 말이 되지 않지만 데이터가 최저 17.2°C까지만 존재하므로 0°C에서 모델을 해석하는 것은 외삽(extrapolation)이 될 것입니다. 이는 나쁜 아이디어일 것입니다. 즉, 온도 값의 _적용 가능한 범위(applicable range)_ 내에서 모델 피팅은 양호합니다. 결론은 관측된 온도 범위로 제한되어야 합니다.

만약 실험에서 관측되지 않은 온도의 울음소리 빈도를 추정해야 한다면, `predict()` 메소드를 사용할 수 있습니다. 이는 모델 객체와 예측을 위한 새로운 값의 데이터 프레임을 취합니다. 예를 들어 모델이 추정하는 15°C에서 20°C 사이의 온도에 대한 *O. exclamationis*의 울음소리 빈도는 다음과 같이 계산할 수 있습니다.

```
new_values <- data.frame(species = "O. exclamationis", temp = 15:20)
predict(main_effect_fit, new_values)
#>     1     2     3     4     5     6
#> 46.83 50.43 54.04 57.64 61.24 64.84
```

###### 경고 (Warning)

숫자 형태의 이진 지시 변수 대신 `species`의 숫자가 아닌 값이 예측(predict) 메소드에 전달됨에 유의하십시오.

이 분석이 명백히 R의 모델링 기능을 완벽하게 보여주는 것은 아니었지만, 이 책의 나머지 부분에서 중요한 몇 가지 주요 기능들을 강조합니다.

- 언어는 단순한 모델과 상당히 복잡한 모델 모두에 대해 모델 항을 지정하기 위한 표현력 있는 구문을 가지고 있습니다.

- R 공식(formula) 메소드는 예측이 생성될 때 새로운 데이터에도 적용되는 모델링을 위한 여러 편리함을 제공합니다.

- 피팅된 모델이 생성된 후 특정 계산을 수행하는 데 사용할 수 있는 수많은 도우미 함수(helper functions) (`anova()`, `summary()`, `predict()`)가 있습니다.

마지막으로, 앞서 언급했듯이 이 프레임워크는 1992년에 처음 게시되었습니다. 이러한 아이디어와 방법론의 대부분은 그 시기에 개발되었지만 현재까지도 현저하게 관련성을 유지하고 있습니다. 이는 S 언어, 나아가 확장된 R이 처음부터 데이터 분석을 위해 설계되었음을 강조합니다.

# R 공식은 무엇을 하는가? (What Does the R Formula Do?)

R 모델 공식은 많은 모델링 패키지에서 사용됩니다. 주로 여러 목적을 제공합니다.

- 공식은 모델이 사용하는 열을 정의합니다.

- 표준 R 메커니즘은 공식을 사용하여 열을 적절한 형식으로 인코딩합니다.

- 열의 역할은 공식에 의해 정의됩니다.

대부분의 경우 실무자들이 공식의 기능을 이해하는 방식은 마지막 목적에 의해 좌우됩니다. 공식을 입력할 때 우리의 초점은 종종 열이 사용되는 방식을 선언하는 것입니다. 예를 들어 앞서 논의한 이전 사양은 예측 변수가 특정 방식으로 사용되도록 설정합니다.

```p
(temp + species)^2
```

이것을 볼 때 우리의 초점은 두 개의 예측 변수가 있고 모델이 그들의 주효과와 2원 상호작용을 포함해야 한다는 것입니다. 그러나 이 공식은 또한 `species`가 팩터이기 때문에, 이 예측 변수에 대한 지시 변수 열을 생성하고([8장](ch08.xhtml#recipes) 참조) 해당 열에 `temp` 열을 곱하여 상호작용을 생성해야 함을 암시합니다. 이 변환은 인코딩에 관한 우리의 두 번째 항목(bullet point)을 나타냅니다. 공식은 또한 각 열이 어떻게 인코딩되는지를 정의하며 원본 데이터에 없는 추가 열을 만들 수 있습니다.

###### 경고 (Warning)

이것은 이 책에서 여러 번 등장할 중요한 포인트이며, 특히 [8장](ch08.xhtml#recipes) 이후에서 더 복잡한 특징 공학을 논의할 때 그렇습니다. R의 공식은 몇 가지 한계를 가지고 있으며, 이를 극복하기 위한 우리의 접근 방식은 이 세 가지 측면 모두와 겨루게 됩니다.

# 모델링에서 깔끔함이 왜 중요한가 (Why Tidiness Is Important for Modeling)

R의 강점 중 하나는 개발자가 자신의 필요에 맞는 사용자 인터페이스를 만들도록 장려한다는 것입니다. 예를 들어, `plot_data`라는 데이터 프레임에서 두 숫자 변수의 산점도를 만드는 세 가지 일반적인 방법은 다음과 같습니다.

```
plot(plot_data$x, plot_data$y)

library(lattice)
xyplot(y ~ x, data = plot_data)

library(ggplot2)
ggplot(plot_data, aes(x = x, y = y)) + geom_point()
```

이 세 가지 경우, 개별 개발자 그룹들이 동일한 작업에 대해 세 가지 뚜렷이 구별되는 인터페이스를 고안했습니다. 각각은 장단점이 있습니다.

이에 비해 *Python Developer’s Guide*는 문제에 접근할 때 "그것을 수행하는 명백한 방법이 하나만, 가급적이면 하나만 있어야 한다"는 개념을 지지합니다.

이 점에서 R은 Python과 상당히 다릅니다. R 인터페이스의 다양성이 갖는 장점은 시간이 지남에 따라 진화할 수 있고 서로 다른 사용자의 서로 다른 요구를 충족할 수 있다는 점입니다.
안타깝게도, 구문상의 다양성 중 일부는 코드를 _사용하는_ 사람의 요구보다 코드를 _개발하는_ 사람의 요구에 초점을 맞추었기 때문입니다. 패키지 간의 불일치는 R 사용자에게 걸림돌이 될 수 있습니다.

여러분의 모델링 프로젝트에서 두 가지 클래스를 가진 결과 변수(outcome)가 있다고 가정해 봅시다. 여러분이 선택할 수 있는 다양한 통계 및 머신러닝 모델이 있습니다. 각 샘플에 대한 클래스 확률 추정치를 생성하기 위해, 모델 함수가 해당하는 `predict()` 메소드를 갖는 것이 일반적입니다. 그러나 클래스 확률 예측을 수행하기 위해 이러한 메소드에서 사용되는 인수 값에는 상당한 이질성(heterogeneity)이 존재합니다. 이러한 이질성은 경험이 풍부한 사용자조차 탐색하기 어려울 수 있습니다. [표 3-1](#probability-args)은 서로 다른 모델에 대한 이러한 인수 값의 샘플을 보여줍니다.

| 함수 (Function) | 패키지 (Package) | 코드 (Code)                                   |
| --------------- | ---------------- | --------------------------------------------- |
| `lda()`         | MASS             | `predict(object)`                             |
| `glm()`         | stats            | `predict(object, type = "response")`          |
| `gbm()`         | gbm              | `predict(object, type = "response", n.trees)` |
| `mda()`         | mda              | `predict(object, type = "posterior")`         |
| `rpart()`       | rpart            | `predict(object, type = "prob")`              |
| various         | RWeka            | `predict(object, type = "probability")`       |
| `logitboost()`  | LogitBoost       | `predict(object, type = "raw", nIter)`        |
| `pamr.train()`  | pamr             | `pamr.predict(object, type = "posterior")`    |

<span id="probability-args">표 3-1. 서로 다른 모델링 함수에 대한 이질적인 인수 이름들.</span>

마지막 예제는 보다 일반적인 `predict()` 인터페이스(제네릭(generic) `predict()` 메소드)를 사용하는 대신 예측을 하기 위해 사용자 정의 함수를 사용한다는 점에 유의하세요. 이러한 일관성의 부족은 모델링을 위한 일상적인 R 사용에 장벽이 됩니다.

예측 불가능성의 또 다른 예로, R 언어에는 결측 데이터(missing data) 처리에 대한 일관성 없는 규칙이 있습니다. 일반적인 규칙은 결측 데이터가 더 많은 결측 데이터를 전파한다는 것입니다. 결측 데이터 포인트가 포함된 값 세트의 평균은 그 자체가 결측되는 식입니다. 모델이 예측을 할 때, 대다수는 모든 예측 변수가 완전한 값을 갖기를 요구합니다. 이 시점에서 R에는 제네릭 함수 `na.action()`에 구워진(baked into) 몇 가지 옵션이 있습니다. 이것은 결측값이 있을 경우 함수가 어떻게 작동해야 하는지에 대한 정책을 설정합니다. 일반적인 두 가지 정책은 `na.fail()`과 `na.omit()`입니다. 전자는 결측 데이터가 존재하면 오류를 생성하는 반면, 후자는 계산 전에 케이스별 삭제(case-wise deletion)를 통해 결측 데이터를 제거합니다. 이전 예제에서:

```
# 예측 세트에 결측값을 추가합니다
new_values$temp[1] <- NA

# `lm`에 대한 예측 메소드는 기본적으로 `na.pass`로 설정되어 있습니다.
predict(main_effect_fit, new_values)
#>     1     2     3     4     5     6
#>    NA 50.43 54.04 57.64 61.24 64.84

# 대안으로
predict(main_effect_fit, new_values, na.action = na.fail)
#> Error in na.fail.default(structure(list(temp = c(NA, 16L, 17L, 18L, 19L, ...

predict(main_effect_fit, new_values, na.action = na.omit)
#>     2     3     4     5     6
#> 50.43 54.04 57.64 61.24 64.84
```

사용자 관점에서 `na.omit()`는 문제가 될 수 있습니다. 이 예에서 `new_values`에는 6개의 행이 있지만 `na.omit()`를 사용하면 5개만 반환됩니다. 이를 조정하기 위해 사용자는 예측값이 `new_values`로 병합되는 경우 어떤 행에 결측값이 있는지 결정하고 적절한 위치에 결측값을 끼워 넣어야(interleave) 합니다.<sup>[2](ch03.xhtml#idm45881872537712-marker)</sup> 예측 함수가 결측 데이터 정책으로 `na.omit()`를 사용하는 경우는 드물지만 실제로 발생합니다. 코드 오류의 원인이 이것 때문이라고 알아낸 사용자들은 꽤 기억에 남는다고 생각합니다.

여기에 설명된 사용상의 문제를 해결하기 위해, tidymodels 패키지에는 일련의 설계 목표가 있습니다. 대부분의 tidymodels 설계 목표는 tidyverse의 "인간을 위한 설계"라는 기존 루브릭에 속하지만 (Wickham et al. 2019), 모델링 코드를 위한 특정한 애플리케이션을 갖추고 있습니다. tidyverse의 목표를 보완하는 몇 가지 추가적인 tidymodels 설계 목표가 있습니다. 예를 들어:

- R은 객체 지향 프로그래밍에 뛰어난 기능이 있으며, 새로운 함수 이름(가상의 새로운 `predict​_sam⁠ples()` 함수 등)을 만드는 대신 이를 사용합니다.

- *합리적인 기본값(Sensible defaults)*은 매우 중요합니다. 또한, 사용자가 선택하도록 강제하는 것이 더 적절한 경우에는 인수에 기본값이 없어야 합니다 (`read_csv()`의 파일 이름 인수).

- 마찬가지로, 기본값을 데이터에서 도출할 수 있는 인수는 그렇게 도출되어야 합니다. 예를 들어, `glm()`의 `family` 인수는 결과 변수 데이터의 유형을 확인하여 `family`가 지정되지 않은 경우 기본값을 내부적으로 결정할 수 있습니다.

- 함수는 개발자가 원하는 데이터 구조가 아니라 *사용자가 가지고 있는 데이터 구조*를 취해야 합니다. 예를 들어, 모델 함수의 유일한 인터페이스가 행렬로 제한되어서는 안 됩니다. 종종 사용자들은 팩터와 같이 숫자가 아닌 예측 변수를 가질 것입니다.

이러한 아이디어의 상당수는 [모델 구현을 위한 tidymodels 가이드라인](https://oreil.ly/qdshy)에 설명되어 있습니다. 후속 장에서는 솔루션과 함께 기존 문제들의 예시를 설명할 것입니다.

###### 참고 (Note)

caret이나 mlr과 같은 몇 가지 기존 R 패키지들은 이러한 이질적인 모델링 API들을 조화시키기 위한 통합 인터페이스를 제공합니다. tidymodels 프레임워크는 이러한 함수 인터페이스의 통합을 채택하고 함수 이름과 반환 값의 일관성을 강제한다는 점에서 이들과 유사합니다. 그러나 이 책 전체에서 자세히 논의되듯 의견이 반영된(opinionated) 설계 목표와 모델링 구현 면에서 다릅니다.

이 책 전체에서 사용하는 `broom::tidy()` 함수는 R 객체의 구조를 표준화하기 위한 또 다른 도구입니다. 이 함수는 훨씬 유용한 형식으로 여러 유형의 R 객체를 반환할 수 있습니다. 예를 들어 결과 열(outcome column)과의 상관관계에 따라 예측 변수를 필터링(screening)한다고 가정해 봅시다. `purrr::map()`을 사용하면 `cor.test()`의 결과를 각 예측 변수에 대한 리스트로 반환할 수 있습니다.

```
corr_res <- map(mtcars %>% select(-mpg), cor.test, y = mtcars$mpg)

# 벡터의 10개 결과 중 첫 번째:
corr_res[[1]]
#>
#>  Pearson's product-moment correlation
#>
#> data:  .x[[i]] and mtcars$mpg
#> t = -8.9, df = 30, p-value = 6e-10
#> alternative hypothesis: true correlation is not equal to 0
#> 95 percent confidence interval:
#>  -0.9258 -0.7163
#> sample estimates:
#>     cor
#> -0.8522
```

이 결과를 도식(plot)에서 사용하고자 할 때 표준 형식의 가설 검정 결과는 그다지 유용하지 않습니다. `tidy()` 메소드는 이를 표준화된 이름을 가진 tibble로 반환할 수 있습니다.

```
library(broom)

tidy(corr_res[[1]])
#> # A tibble: 1 × 8
#>   estimate statistic  p.value parameter conf.low conf.high method        alternative
#>      <dbl>     <dbl>    <dbl>     <int>    <dbl>     <dbl> <chr>         <chr>
#> 1   -0.852     -8.92 6.11e-10        30   -0.926    -0.716 Pearson's pr… two.sided
```

[그림 3-3](#figure-3-3.-correlations-and-95-confidence-intervals-between-predictors-and-the-outcome-in-the-mtcars-data-set.)과 같이 이러한 결과를 "쌓아서(stacked)" `ggplot()`에 추가할 수 있습니다.

```
corr_res %>%
  # 각각을 tidy 포맷으로 변환합니다; `map_dfr()`은 데이터 프레임들을 쌓습니다(stack)
  map_dfr(tidy, .id = "predictor") %>%
  ggplot(aes(x = fct_reorder(predictor, estimate))) +
  geom_point(aes(y = estimate)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = .1) +
  labs(x = NULL, y = "Correlation with mpg")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0303.png" alt="tmwr 0303" />
<h6 id="figure-3-3.-correlations-and-95-confidence-intervals-between-predictors-and-the-outcome-in-the-mtcars-data-set.">그림 3-3. <code>mtcars</code> 데이터 세트에서 예측 변수와 결과 변수 간의 상관관계(및 95% 신뢰 구간).</h6>
</figure>

이러한 플롯을 생성하는 것은 핵심 R 언어 함수를 사용하여 가능하지만, 결과를 자동으로 다시 포맷팅하면 오류 발생 가능성을 줄이고 더 간결한 코드를 작성할 수 있습니다.

# Base R 모델과 Tidyverse 결합하기 (Combining Base R Models and the Tidyverse)

핵심 언어 또는 다른 R 패키지의 R 모델링 함수는 tidyverse, 특히 dplyr, purrr 및 tidyr 패키지와 함께 사용할 수 있습니다. 예를 들어, 각 귀뚜라미 종에 대해 개별 모델을 피팅하려는 경우 먼저 `dplyr::group_nest()`를 사용하여 이 열별로 귀뚜라미 데이터를 분리할 수 있습니다.

```
split_by_species <-
  crickets %>%
  group_nest(species)
split_by_species
#> # A tibble: 2 × 2
#>   species                        data
#>   <fct>            <list<tibble[,2]>>
#> 1 O. exclamationis           [14 × 2]
#> 2 O. niveus                  [17 × 2]
```

`data` 열에는 `crickets`의 `rate` 및 `temp` 열이 *리스트 열(list column)*로 포함되어 있습니다. 이것으로부터 `purrr::map()` 함수는 각 종에 대해 개별 모델을 생성할 수 있습니다.

```
model_by_species <-
  split_by_species %>%
  mutate(model = map(data, ~ lm(rate ~ temp, data = .x)))
model_by_species
#> # A tibble: 2 × 3
#>   species                        data model
#>   <fct>            <list<tibble[,2]>> <list>
#> 1 O. exclamationis           [14 × 2] <lm>
#> 2 O. niveus                  [17 × 2] <lm>
```

각 모델에 대한 계수를 수집하려면 `broom::tidy()`를 사용하여 일관된 데이터 프레임 형식으로 변환한 다음, 언네스트(unnested)할 수 있습니다.

```
model_by_species %>%
  mutate(coef = map(model, tidy)) %>%
  select(species, coef) %>%
  unnest(cols = c(coef))
#> # A tibble: 4 × 6
#>   species          term        estimate std.error statistic  p.value
#>   <fct>            <chr>          <dbl>     <dbl>     <dbl>    <dbl>
#> 1 O. exclamationis (Intercept)   -11.0      4.77      -2.32 3.90e- 2
#> 2 O. exclamationis temp            3.75     0.184     20.4  1.10e-10
#> 3 O. niveus        (Intercept)   -15.4      2.35      -6.56 9.07e- 6
#> 4 O. niveus        temp            3.52     0.105     33.6  1.57e-15
```

###### 참고 (Note)

리스트 열(List columns)은 모델링 프로젝트에서 매우 강력할 수 있습니다. 리스트 열은 피팅된 모델 자체에서부터 중요한 데이터 프레임 구조에 이르기까지 모든 유형의 R 객체를 위한 컨테이너를 제공합니다.

# tidymodels 메타패키지 (The tidymodels Metapackage)

tidyverse([2장](ch02.xhtml#tidyverse))는 다소 좁은 범위를 가진 모듈식 R 패키지 세트로 설계되었습니다. tidymodels 프레임워크는 이와 유사한 설계를 따릅니다. 예를 들어 rsample 패키지는 데이터 분할 및 리샘플링에 초점을 맞춥니다. 리샘플링 방법은 모델링의 다른 활동(성능 측정)에 매우 중요하지만 하나의 단일 패키지에 있으며, 성능 지표는 yardstick이라는 서로 다른 별도의 패키지에 포함되어 있습니다. 모델 배포가 덜 비대해지는 것부터 패키지 유지 관리가 더 원활해지는 것에 이르기까지 모듈식 패키지라는 철학을 채택하면 얻을 수 있는 이점이 많습니다.

이 철학의 단점은 tidymodels 프레임워크에 매우 많은 패키지가 있다는 것입니다. 이를 보완하기 위해 tidymodels _패키지_ (tidyverse 패키지와 같은 메타패키지(metapackage)로 생각할 수 있음)는 tidymodels 및 tidyverse 패키지의 핵심 세트를 로드합니다. 패키지를 로드하면 어떤 패키지가 연결되어 있는지 표시됩니다.

```
library(tidymodels)
#> ── Attaching packages ─────────────────────────────────────────── tidymodels 0.2.0 ──
#> ✔ broom        0.8.0          ✔ recipes      0.2.0
#> ✔ dials        0.1.1          ✔ rsample      0.1.1
#> ✔ dplyr        1.0.8          ✔ tibble       3.1.6
#> ✔ ggplot2      3.3.5          ✔ tidyr        1.2.0
#> ✔ infer        1.0.0          ✔ tune         0.2.0
#> ✔ modeldata    0.1.1          ✔ workflows    0.2.6
#> ✔ parsnip      0.2.1          ✔ workflowsets 0.2.1
#> ✔ purrr        0.3.4          ✔ yardstick    0.0.9
#> ── Conflicts ─────────────────────────────────────────── tidymodels_conflicts() ──
#> ✖ purrr::discard() masks scales::discard()
#> ✖ dplyr::filter()  masks stats::filter()
#> ✖ dplyr::lag()     masks stats::lag()
#> ✖ recipes::step()  masks stats::step()
#> • Learn how to get started at https://www.tidymodels.org/start
```

tidyverse를 사용해 보셨다면, dplyr 및 ggplot2와 같은 일부 tidyverse 패키지가 tidymodels 패키지와 함께 로드되기 때문에 몇 가지 익숙한 이름들을 눈치채셨을 것입니다. tidymodels 프레임워크가 모델링에 tidyverse 원칙을 적용한다고 이미 말씀드렸지만, tidymodels 프레임워크는 이와 같은 근본적인 tidyverse 패키지들을 말 그대로 기반으로 하여 구축되었습니다.

메타패키지를 로드하면 이전에 로드된 패키지들과 함수 이름 충돌(naming conflicts)이 있는지도 보여줍니다. 이름 충돌의 예로, tidymodels를 로드하기 전에 `filter()` 함수를 호출하면 stats 패키지의 함수가 실행됩니다. tidymodels를 로드한 후에는 같은 이름의 dplyr 함수가 실행됩니다.

이름 충돌을 처리하는 몇 가지 방법이 있습니다. 함수를 해당 네임스페이스와 함께 호출할 수 있습니다(`stats::filter()`). 이것은 나쁜 관행은 아니지만, 코드를 덜 읽기 쉽게 만듭니다.

다른 옵션은 conflicted 패키지를 사용하는 것입니다. 코드에 네임스페이스가 지정되지 않은 경우 항상 하나의 특정 함수가 실행되도록 보장하기 위해, R 세션이 끝날 때까지 유효하게 유지되는 규칙을 설정할 수 있습니다. 예를 들어 이전 함수의 dplyr 버전을 선호한다면:

```
library(conflicted)
conflict_prefer("filter", winner = "dplyr")
```

편의를 위해 tidymodels에는 우리가 만날 수 있는 일반적인 이름 충돌의 대부분을 포착하는 함수가 포함되어 있습니다.

```
tidymodels_prefer(quiet = FALSE)
#> [conflicted] Will prefer dplyr::filter over any other package
#> [conflicted] Will prefer dplyr::select over any other package
#> [conflicted] Will prefer dplyr::slice over any other package
#> [conflicted] Will prefer dplyr::rename over any other package
#> [conflicted] Will prefer dials::neighbors over any other package
#> [conflicted] Will prefer parsnip::fit over any other package
#> [conflicted] Will prefer parsnip::bart over any other package
#> [conflicted] Will prefer parsnip::pls over any other package
#> [conflicted] Will prefer purrr::map over any other package
#> [conflicted] Will prefer recipes::step over any other package
#> [conflicted] Will prefer themis::step_downsample over any other package
#> [conflicted] Will prefer themis::step_upsample over any other package
#> [conflicted] Will prefer tune::tune over any other package
#> [conflicted] Will prefer yardstick::precision over any other package
#> [conflicted] Will prefer yardstick::recall over any other package
#> [conflicted] Will prefer yardstick::spec over any other package
#> ── Conflicts ─────────────────────────────────────────── tidymodels_prefer() ──
```

###### 경고 (Warning)

이 함수를 사용하면 모든 네임스페이스 충돌에 대해 `conflicted::conflict_prefer()`를 사용하도록 설정되며, 모든 충돌을 오류로 만들어 여러분이 사용할 함수를 선택하도록 강제한다는 점에 유의하세요. `tidymodels::tidymodels_prefer()` 함수는 tidymodels 함수들의 흔한 충돌을 처리하지만, R 세션의 다른 충돌은 스스로 처리해야 합니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 이 책의 나머지 부분을 위한 중요한 기초인, 모델을 생성하고 사용하기 위한 핵심 R 언어 규약을 검토했습니다. 공식(formula) 연산자는 R에서 모델 피팅에 있어 표현력이 풍부하고 중요한 부분이며, 비(non)-tidymodels 함수에서도 여러 목적을 제공하는 경우가 많습니다. 전통적인 R의 모델링 접근 방식에는 한계가 있으며, 특히 모델 출력을 유연하게 다루고 시각화할 때 그렇습니다. tidymodels 메타패키지는 모델링 패키지에 tidyverse 설계 철학을 적용합니다.

<sup>[1](ch03.xhtml#idm45881871527184-marker)</sup> 대부분의 모델 함수는 절편 열(intercept column)을 암시적으로 추가합니다.

<sup>[2](ch03.xhtml#idm45881872537712-marker)</sup> `na.exclude()`라는 기본 R 정책이 정확히 이 작업을 수행합니다.
