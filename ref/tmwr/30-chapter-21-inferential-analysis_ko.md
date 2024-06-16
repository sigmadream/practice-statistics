# 21장. 추론 분석 (Inferential Analysis)

###### 참고 (Note)

[1장](ch01.xhtml#software-modeling)에서는 모델의 분류(taxonomy)를 간략하게 설명(outlined)하고 대부분의 모델을 설명적(descriptive), 추론적(inferential) 및/또는 예측적(predictive) 모델로 분류할 수 있다고 말했습니다.

이 책의 대부분의 장에서는 예측값의 정확도(예측 모델에 관련성이 높지만 모든 목적의 모델에서 중요한 품질) 관점에서 모델에 중점을 두었습니다(focused on). 추론 모델은 대개(usually) 예측뿐만 아니라 계수 값이나 다른 매개변수와 같은 모델의 일부 구성 요소(component)에 대한 추론(inferences)이나 판단을 내리기 위해 생성됩니다. 이러한 결과는 종종 사전 정의된 질문이나 가설에 대한 답을 제공하는 데(to answer) 사용됩니다. 예측 모델에서는 홀드아웃 데이터(holdout data)에 대한 예측을 사용하여 모델의 품질을 검증하거나 특성화(characterize)합니다. 추론 방법은 모델을 피팅하기 전에 만들어진 확률적(probabilistic) 또는 구조적(structural) 가정을 검증하는 데 중점을 둡니다.

예를 들어 보통 선형 회귀(ordinary linear regression)에서 일반적인(common) 가정은 잔차 값이 독립적(independent)이며 분산이 일정한(constant variance) 가우시안 분포를 따른다는 것입니다. 모델 분석을 위한 이러한 가정에 신빙성(credence)을 더해줄(lend) 과학적 또는 도메인 지식이 있을 수 있지만(may have), 가정이 좋은 아이디어였는지 판단하기 위해 대개(usually) 피팅된 모델의 잔차를 검토(examined)합니다. 그 결과, 홀드아웃 예측을 보는 것이 매우 유용할 수 있지만, 모델의 가정이 충족(met)되었는지 확인(determining)하는 방법은 그렇게 간단하지 않습니다(not as simple as).

이 장에서는 p-값을 사용합니다. 그러나 tidymodels 프레임워크는 대립 가설(alternative hypothesis)에 대한 증거(evidence)를 정량화하는 방법으로 p-값보다 신뢰 구간(confidence intervals)을 선호하는 경향이 있습니다(tends to promote). [11장](ch11.xhtml#compare)에서 이전에 본 바와 같이, 베이지안 방법은 해석의 용이성(ease of interpretation) 측면에서 종종 p-값과 신뢰 구간 모두보다 우수합니다(하지만 계산 비용이 더 많이 들(computationally expensive) 수 있습니다).

###### 경고 (Warning)

최근 몇 년 동안(in recent years) p-값에서 벗어나(move away from) 다른 방법을 선호하는(in favor of) 움직임(push)이 있었습니다(Wasserstein and Lazar 2016). 자세한 정보와 논의는 [_The American Statistician_](https://oreil.ly/UeukP) 73권을 참조하세요.

이 장에서는 추론 모델을 피팅하고 평가하기 위해 tidymodels를 사용하는 방법을 설명합니다. 어떤 경우에는 tidymodels 프레임워크가 사용자가 모델에서 생성된 객체를 작업하는 데 도움을 줄 수 있습니다. 다른 경우에는 주어진 모델의 품질을 평가하는 데 도움을 줄 수 있습니다.

# 카운트 데이터에 대한 추론 (Inference for Count Data)

tidymodels 패키지가 추론 모델링에 어떻게 사용될 수 있는지 이해하기 위해 카운트 데이터가 있는 예제에 중점을 두겠습니다(let's focus on). pscl 패키지의 생화학 출판물(biochemistry publication) 데이터를 사용합니다. 이 데이터는 915명의 생화학 박사 졸업생에 대한 정보로 구성되어 있으며 학업(academic) 생산성(졸업 후 3년 내에 발표된 논문 수로 측정됨)에 영향을 미치는 요인을 설명(explain)하려고 시도합니다. 예측 변수에는 졸업생의 성별, 혼인 여부(marital status), 졸업생의 5세 이상 자녀 수, 소속 부서(department)의 명성(prestige), 그리고 멘토가 같은 기간 동안 발표한 논문 수 등이 포함됩니다. 이 데이터는 1956년과 1963년 사이에 교육을 마친(finished) 생화학 박사 학위 취득자(doctorates)를 반영(reflect)합니다. 이 데이터는 (정보의 완전성에 기반하여) 이 기간 동안 배출된 전체 생화학 박사 학위 취득자의 다소 편향된(somewhat biased) 샘플입니다.

###### 참고 (Note)

[19장](ch19.xhtml#trust)에서 우리는 "특정 데이터 포인트를 예측하는 데 우리 모델이 적용 가능한가(applicable)?"라는 질문을 던졌습니다. 추론 분석이 적용되는 모집단(populations)을 정의하는 것은 매우 중요합니다. 이 데이터의 경우, 그 결과는 데이터가 수집된 시점 전후(around the time frame)에 박사 학위를 받은 생화학자들에게 적용될 가능성이 높습니다. 다른 화학 분야 박사(의약 화학 등)에도 적용됩니까? 이는 추론 분석을 수행할 때 해결(그리고 문서화)해야 할 중요한 질문입니다.

[그림 21-1](#counts)에 표시된 데이터의 플롯(plot)은 이 기간 동안 많은 졸업생이 어떤 논문도 출판하지 않았으며 결과가 오른쪽으로 치우친(right-skewed) 분포를 따른다는 것을 나타냅니다.

```
library(tidymodels)
tidymodels_prefer()

data("bioChemists", package = "pscl")

ggplot(bioChemists, aes(x = art)) +
  geom_histogram(binwidth = 1, color = "white") +
  labs(x = "Number of articles within 3y of graduation")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2101.png" alt="tmwr 2101" />
<h6 id="figure-21-1.-distribution-of-the-number-of-articles-written-within-3-years-of-graduation.">그림 21-1. 졸업 후 3년 이내에 작성된 논문 수의 분포.</h6>
</figure>

결과 데이터가 카운트이므로 일반적인(common) 분포 가정은 결과가 푸아송(Poisson) 분포를 따른다는 것입니다. 이 장에서는 여러 유형의 분석을 위해 이 데이터를 사용할 것입니다.

# 2-표본 테스트를 이용한 비교 (Comparisons with Two-Sample Tests)

우리는 가설 검정(hypothesis testing)부터 시작할 수 있습니다. 생화학 출판물 데이터에 대한 이 데이터 세트의 원래(original) 저자의 목표는 남성과 여성 간에 출판물의 차이가 있는지 확인(determine)하는 것이었습니다(Long 1992). 이 연구의 데이터는 다음을 보여줍니다.

```
bioChemists %>%
  group_by(fem) %>%
  summarize(counts = sum(art), n = length(art))
#> # A tibble: 2 × 3
#>   fem   counts     n
#>   <fct>  <int> <int>
#> 1 Men      930   494
#> 2 Women    619   421
```

데이터에 남성이 더 많기도 했지만(although), 남성이 훨씬 더 많은 출판물을 냈습니다. 이 데이터를 분석하는 간단한(simplest) 접근 방식은 stats 패키지의 `poisson.test()` 함수를 사용하여 2-표본 비교(two-sample comparison)를 수행하는 것입니다. 이 함수는 하나 또는 두 그룹에 대한 카운트(counts)를 요구합니다.

우리의 적용 사례에서 두 성별을 비교하기 위한 가설(hypotheses)은 다음과 같습니다.

```math
\begin{aligned}
H_{0} & {:\lambda_{m} = \lambda_{f}} \\
H_{a} & {:\lambda_{m} \neq \lambda_{f}}
\end{aligned}
```

여기서 $`\lambda`$ 값은 (동일한 기간 동안의) 출판물 비율(rates)입니다.

이 테스트의 기본 적용 사례는 다음과 같습니다.<sup><a href="ch21.xhtml#idm45881844138224" id="idm45881844138224-marker" data-type="noteref">1</a></sup>

```
poisson.test(c(930, 619), T = 3)
#>
#>  Comparison of Poisson rates
#>
#> data:  c(930, 619) time base: 3
#> count1 = 930, expected count1 = 774, p-value = 3e-15
#> alternative hypothesis: true rate ratio is not equal to 1
#> 95 percent confidence interval:
#>  1.356 1.666
#> sample estimates:
#> rate ratio
#>      1.502
```

이 함수는 p-값과 함께 출판 비율의 비(ratio)에 대한 신뢰 구간을 보고합니다(reports). 결과는 관찰된 차이가 경험적 잡음(experiential noise)보다 크며(greater than) $`H_{a}`$를 지지(favors)함을 나타냅니다.

이 함수를 사용할 때의 한 가지 문제점(issue)은 결과가 `htest` 객체로 반환(come back)된다는 것입니다. 이러한 유형의 객체는 잘 정의된(well-defined) 구조를 가지고 있지만 보고 또는 시각화와 같은 후속(subsequent) 작업에 사용하기(consume) 어려울 수 있습니다. 추론 모델을 위해 tidymodels가 제공하는(offers) 영향력 있는(impactful) 도구는 broom 패키지의 `tidy()` 함수입니다. 이전에 살펴보았듯이, 이 함수는 객체로부터 형식이 잘 갖춰지고(well-formed) 예측 가능한 이름의 티블(tibble)을 만듭니다. 우리는 2-표본 비교 테스트의 결과를 `tidy()`할 수 있습니다.

```
poisson.test(c(930, 619)) %>%
  tidy()
#> # A tibble: 1 × 8
#>   estimate statistic  p.value parameter conf.low conf.high method        alternative
#>      <dbl>     <dbl>    <dbl>     <dbl>    <dbl>     <dbl> <chr>         <chr>
#> 1     1.50       930 2.73e-15      774.     1.36      1.67 Comparison o… two.sided
```

###### 참고 (Note)

[broom](https://oreil.ly/jtbP8)과 [broom.mixed](https://oreil.ly/NIHXK) 패키지 사이에는 150개 이상의 모델에 대한 `tidy()` 메서드가 있습니다.

푸아송 분포도 합리적(reasonable)이지만, 분포에 대한 가정을 덜(fewer) 사용하여 평가(assess)하고 싶을 수도 있습니다. 도움이 될 수 있는 두 가지 방법은 부트스트랩(bootstrap)과 순열 검정(permutation tests)입니다(Davison and Hinkley 1997).

tidymodels 프레임워크의 일부인 infer 패키지는 가설 검정을 위한 강력하고 직관적인 도구입니다(Ismay and Kim 2021). 이 패키지의 구문(syntax)은 간결(concise)하며 통계 전문가가 아닌 사람(nonstatisticians)을 위해 설계되었습니다.

먼저 남녀 간 평균 논문 수의 차이를 사용할 것임을 `specify()`한 다음 데이터에서 통계량(statistic)을 `calculate()`합니다. 푸아송 평균의 최대우도추정량(maximum likelihood estimator)은 표본 평균이라는 점을 상기하십시오. 여기서 테스트한 가설은 이전 테스트와 동일합니다(하지만 다른 테스트 절차를 사용하여 수행(conducted)됨).

infer를 사용하면 결과(outcome)와 공변량(covariate)을 지정(specify)한 다음 관심 통계량(statistic of interest)을 명시(state)합니다.

```
library(infer)

observed <-
  bioChemists %>%
  specify(art ~ fem) %>%
  calculate(stat = "diff in means", order = c("Men", "Women"))
observed
#> Response: art (numeric)
#> Explanatory: fem (factor)
#> # A tibble: 1 × 1
#>    stat
#>   <dbl>
#> 1 0.412
```

여기서부터(From here) 우리는 `generate()`를 통해 부트스트랩 분포를 생성하여 이 평균에 대한 신뢰 구간을 계산합니다; 데이터의 각 재표집된 버전에 대해 동일한 통계량이 계산됩니다.

```
set.seed(2101)
bootstrapped <-
  bioChemists %>%
  specify(art ~ fem)  %>%
  generate(reps = 2000, type = "bootstrap") %>%
  calculate(stat = "diff in means", order = c("Men", "Women"))
bootstrapped
#> Response: art (numeric)
#> Explanatory: fem (factor)
#> # A tibble: 2,000 × 2
#>   replicate  stat
#>       <int> <dbl>
#> 1         1 0.467
#> 2         2 0.107
#> 3         3 0.467
#> 4         4 0.308
#> 5         5 0.369
#> 6         6 0.428
#> # … with 1,994 more rows
```

백분위수 구간(percentile interval)은 다음을 사용하여 계산됩니다.

```
percentile_ci <- get_ci(bootstrapped)
percentile_ci
#> # A tibble: 1 × 2
#>   lower_ci upper_ci
#>      <dbl>    <dbl>
#> 1    0.158    0.653
```

infer 패키지에는 [그림 21-2](#bootstrapped-mean)에 표시된 것처럼 분석 결과를 표시하기 위한 고수준 API가 있습니다.

```
visualize(bootstrapped) +
    shade_confidence_interval(endpoints = percentile_ci)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2102.png" alt="tmwr 2102" />
<h6 id="figure-21-2.-the-bootstrap-distribution-of-the-difference-in-means.-the-highlighted-region-is-the-confidence-interval.">그림 21-2. 평균 차이의 부트스트랩 분포. 강조 표시된(highlighted) 영역은 신뢰 구간입니다.</h6>
</figure>

[그림 21-2](#bootstrapped-mean)에 시각화된 구간은 0을 포함하지 않으므로(does not include zero), 이 결과는 남성이 여성보다 더 많은 논문을 발표했음을 나타냅니다.

p-값이 필요한 경우 infer 패키지는 다음 코드에 표시된 것처럼 순열 검정(permutation test)을 통해 값을 계산할 수 있습니다. 구문(syntax)은 앞서 사용한 부트스트래핑 코드와 매우 유사합니다. 테스트할 가정의 유형을 명시하기 위해 `hypothesize()` 동사(verb)를 추가하고, `generate()` 호출에는 데이터를 섞는(shuffle) 옵션이 포함됩니다.

```
set.seed(2102)
permuted <-
  bioChemists %>%
  specify(art ~ fem)  %>%
  hypothesize(null = "independence") %>%
  generate(reps = 2000, type = "permute") %>%
  calculate(stat = "diff in means", order = c("Men", "Women"))
permuted
#> Response: art (numeric)
#> Explanatory: fem (factor)
#> Null Hypothesis: independence
#> # A tibble: 2,000 × 2
#>   replicate     stat
#>       <int>    <dbl>
#> 1         1  0.201
#> 2         2 -0.133
#> 3         3  0.109
#> 4         4 -0.195
#> 5         5 -0.00128
#> 6         6 -0.102
#> # … with 1,994 more rows
```

다음 시각화 코드도 부트스트랩 접근 방식과 매우 유사합니다. 이 코드는 수직선이 관측값을 나타내는(signifies) [그림 21-3](#permutation-dist)을 생성합니다.

```
visualize(permuted) +
    shade_p_value(obs_stat = observed, direction = "two-sided")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2103.png" alt="tmwr 2103" />
<h6 id="figure-21-3.-empirical-distribution-of-the-test-statistic-under-the-null-hypothesis.-the-vertical-line-indicates-the-observed-test-statistic.">그림 21-3. 귀무가설 하에서의 검정 통계량의 경험적(Empirical) 분포. 수직선은 관찰된 검정 통계량을 나타냅니다.</h6>
</figure>

실제(actual) p-값은 다음과 같습니다.

```
permuted %>%
  get_p_value(obs_stat = observed, direction = "two-sided")
#> # A tibble: 1 × 1
#>   p_value
#>     <dbl>
#> 1   0.002
```

[그림 21-3](#permutation-dist)에서 귀무가설을 나타내는 수직선은 순열 분포에서 멀리 떨어져 있습니다. 이는 귀무가설이 실제로 참이라면 현재 가지고 있는 데이터(what is at hand) 이상으로 극단적인 데이터를 관찰할 가능성이 극히 작다는(exceedingly small) 것을 의미합니다.

이 섹션에 표시된 2-표본 테스트는 출판 비율과 성별 사이의 관찰된 관계를 설명할 수 있는 다른 요인을 설명하지 않기 때문에(do not account for) 아마 차선책(suboptimal)일 것입니다. 추가 공변량을 고려할(can consider) 수 있는 더 복잡한 모델로 이동해 보겠습니다.

# 로그-선형 모델 (Log-Linear Models)

이 장의 나머지(rest) 부분은 카운트가 푸아송 분포를 따른다고 가정하는 일반화 선형 모델(Dobson 1999)에 초점을 맞출 것입니다. 이 모델의 경우 공변량/예측 변수는 로그-선형 방식(fashion)으로 모델에 입력됩니다(enter):

```math
\log(\lambda) = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{p}x_{p}
```

여기서 $`\lambda`$는 카운트의 기대값입니다.

모든 예측 변수 열을 포함하는 간단한 모델을 적합시켜 보겠습니다. tidymodels의 parsnip 확장 패키지인 poissonreg 패키지는 이 모델 사양(specification)을 생성합니다.

```
library(poissonreg)

# 기본(default) 엔진은 'glm'입니다.
```

log_lin_spec <- poisson_reg()

log_lin_fit <-
log_lin_spec %>%
fit(art ~ ., data = bioChemists)
log_lin_fit
#> parsnip model object
#>
#>
#> Call: stats::glm(formula = art ~ ., family = stats::poisson, data = data)
#>
#> Coefficients:
#> (Intercept) femWomen marMarried kid5 phd ment
#> 0.3046 -0.2246 0.1552 -0.1849 0.0128 0.0255
#>
#> Degrees of Freedom: 914 Total (i.e., Null); 909 Residual
#> Null Deviance: 1820
#> Residual Deviance: 1630 AIC: 3310

```

`tidy()` 메서드는 모델의 계수(그리고 90% 신뢰 구간)를 간결하게(succinctly) 요약합니다.

```

tidy(log_lin_fit, conf.int = TRUE, conf.level = 0.90)
#> # A tibble: 6 × 7
#> term estimate std.error statistic p.value conf.low conf.high
#> <chr> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1 (Intercept) 0.305 0.103 2.96 3.10e- 3 0.134 0.473
#> 2 femWomen -0.225 0.0546 -4.11 3.92e- 5 -0.315 -0.135
#> 3 marMarried 0.155 0.0614 2.53 1.14e- 2 0.0545 0.256
#> 4 kid5 -0.185 0.0401 -4.61 4.08e- 6 -0.251 -0.119
#> 5 phd 0.0128 0.0264 0.486 6.27e- 1 -0.0305 0.0563
#> 6 ment 0.0255 0.00201 12.7 3.89e-37 0.0222 0.0288

````

이전 출력에서 p-값은 각 모델 매개변수에 대한 개별 가설 검정에 해당합니다.

``` math
\begin{aligned}
H_{0} & {:\beta_{j} = 0} \\
H_{a} & {:\beta_{j} \neq 0}
\end{aligned}
````

이 결과를 보면, `phd`(소속 부서의 명성)는 결과와 아무런 관련이 없을 수 있습니다.

푸아송 분포가 이와 같은 데이터에 대한 일상적인(routine) 가정이지만, 신뢰 구간을 계산하는 데 푸아송 우도(likelihood)를 사용하지 않고 모델을 피팅하여 모델 가정을 대략적으로(rough) 확인하는 것이 유익할(beneficial) 수 있습니다. rsample 패키지에는 `lm()` 및 `glm()` 모델에 대한 부트스트랩 신뢰 구간을 계산하는 편리한 함수가 있습니다. 우리는 `family = poisson`을 명시적으로(explicitly) 선언(declaring)하면서 이 함수를 사용하여 수많은(large number of) 모델 적합을 계산할 수 있습니다. 기본적으로(By default) 90% 신뢰 부트스트랩-t 구간(백분위수 구간도 사용 가능)을 계산합니다.

```
set.seed(2103)
glm_boot <-
  reg_intervals(art ~ ., data = bioChemists, model_fn = "glm", family = poisson)
glm_boot
#> # A tibble: 5 × 6
#>   term          .lower .estimate  .upper .alpha .method
#>   <chr>          <dbl>     <dbl>   <dbl>  <dbl> <chr>
#> 1 femWomen   -0.358      -0.226  -0.0856   0.05 student-t
#> 2 kid5       -0.298      -0.184  -0.0789   0.05 student-t
#> 3 marMarried  0.000264    0.155   0.317    0.05 student-t
#> 4 ment        0.0182      0.0256  0.0322   0.05 student-t
#> 5 phd        -0.0707      0.0130  0.102    0.05 student-t
```

###### 경고 (Warning)

이 결과([그림 21-4](#glm-intervals))를 `glm()`의 순수 모수적(purely parametric) 결과와 비교하면 부트스트랩 구간이 다소 더 넓습니다(somewhat wider). 데이터가 진정으로(truly) 푸아송이라면 이러한 구간은 더 유사한 너비를 가질 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2104.png" alt="tmwr 2104" />
<h6 id="figure-21-4.-two-types-of-confidence-intervals-for-the-poisson-regression-model.">그림 21-4. 푸아송 회귀 모델에 대한 두 가지 유형의 신뢰 구간.</h6>
</figure>

어떤 예측 변수를 모델에 포함할지 결정하는 것은 어려운 문제입니다. 한 가지 접근 방식은 중첩된(nested) 모델 간에 우도비 검정(likelihood ratio tests, LRT)(McCullagh and Nelder 1989)을 수행하는 것입니다. 신뢰 구간을 기반으로 볼 때, `phd`가 없는 더 간단한 모델로도 충분할 수 있다는 증거가 있습니다. 더 작은 모델을 피팅한 다음 통계적 테스트를 수행해 보겠습니다.

```math
\begin{aligned}
H_{0} & {:\beta_{phd} = 0} \\
H_{a} & {:\beta_{phd} \neq 0}
\end{aligned}
```

이 가설은 이전에 `log_lin_fit`에 대한 정리된(tidied) 결과를 보여줄 때 테스트되었습니다. 해당 특정 접근 방식은 Wald 통계량(즉, 매개변수를 해당 표준 오차로 나눈 값)을 통한 단일 모델 적합의 결과를 사용했습니다. 해당 접근 방식의 경우 p-값은 0.63이었습니다. LRT에 대한 결과를 정리(tidy)하여 p-값을 얻을 수 있습니다.

```
log_lin_reduced <-
  log_lin_spec %>%
  fit(art ~ ment + kid5 + fem + mar, data = bioChemists)

anova(
  extract_fit_engine(log_lin_reduced),
  extract_fit_engine(log_lin_fit),
  test = "LRT"
) %>%
  tidy()
#> # A tibble: 2 × 5
#>   Resid..Df Resid..Dev    df Deviance p.value
#>       <dbl>      <dbl> <dbl>    <dbl>   <dbl>
#> 1       910      1635.    NA   NA      NA
#> 2       909      1634.     1    0.236   0.627
```

결과는 동일하며, 이것과 이 매개변수에 대한 신뢰 구간을 기반으로 할 때, 결과와 관련이 없는 것으로 보이므로 향후(further) 분석에서 `phd`를 제외(exclude)할 것입니다.

# 더 복잡한 모델 (A More Complex Model)

우리는 tidymodels 접근 방식 내에서 훨씬 더 복잡한 모델로 이동할 수 있습니다. 카운트 데이터의 경우 관찰된 제로(zero) 카운트의 수가 단순한 푸아송 분포가 처방하는(prescribe) 것보다 많은(larger than) 경우가 있습니다(occasions). 이 상황에 적합한 더 복잡한 모델은 영과잉 푸아송(zero-inflated Poisson, ZIP) 모델입니다; Mullahy (1986), Lambert (1992), 및 Zeileis, Kleiber, and Jackman (2008)을 참조하세요. 여기에는 두 가지 공변량 세트가 있습니다. 하나는 카운트 데이터용이고 다른 하나는 0이 될 확률($`\pi`$로 표시됨)에 영향을 미치는(affect) 공변량입니다. 평균 $`\lambda`$에 대한 방정식은 다음과 같습니다.

```math
\lambda = 0\pi + (1 - \pi)\lambda_{nz}
```

여기서

```math
\begin{aligned}
{\log\left( \lambda_{nz} \right)} & {= \beta_{0} + \beta_{1}x_{1} + ... + \beta_{p}x_{p}} \\
{\log\left( \frac{\pi}{1 - \pi} \right)} & {= \gamma_{0} + \gamma_{1}z_{1} + ... + \gamma_{q}z_{q}}
\end{aligned}
```

이고 $`x`$ 공변량은 카운트 값에 영향을 미치는 반면 $`z`$ 공변량은 0이 될 확률에 영향을 미칩니다(influence). 두 세트의 예측 변수는 상호 배타적일(mutually exclusive) 필요는 없습니다.

전체(full set of) $`z`$ 공변량을 사용하여 모델을 피팅하겠습니다.

```
zero_inflated_spec <- poisson_reg() %>% set_engine("zeroinfl")

zero_inflated_fit <-
  zero_inflated_spec %>%
  fit(art ~ fem + mar + kid5 + ment | fem + mar + kid5 + phd + ment,
      data = bioChemists)

zero_inflated_fit
#> parsnip model object
#>
#>
#> Call:
#> pscl::zeroinfl(formula = art ~ fem + mar + kid5 + ment | fem + mar + kid5 +
#>     phd + ment, data = data)
#>
#> Count model coefficients (poisson with log link):
#> (Intercept)     femWomen   marMarried         kid5         ment
#>       0.621       -0.209        0.105       -0.143        0.018
#>
#> Zero-inflation model coefficients (binomial with logit link):
#> (Intercept)     femWomen   marMarried         kid5          phd         ment
#>     -0.6086       0.1093      -0.3529       0.2195       0.0124      -0.1351
```

이 모델의 계수도 최대우도를 사용하여 추정(estimated)되므로, 다른 우도비 검정을 사용하여 새로운 모델 항이 유용한지(helpful) 파악해 보겠습니다. 우리는 다음을 _동시에(simultaneously)_ 검정할 것입니다.

```math
\begin{aligned}
H_{0} & {:\gamma_{1} = 0,\gamma_{2} = 0,\cdots,\gamma_{5} = 0} \\
H_{a} & {:\text{적어도 하나의}\gamma \neq 0}
\end{aligned}
```

ANOVA를 다시 시도해 보겠습니다.

```
anova(
  extract_fit_engine(zero_inflated_fit),
  extract_fit_engine(log_lin_reduced),
  test = "LRT"
) %>%
  tidy()
#> Error in UseMethod("anova"): no applicable method for 'anova' applied to an
   object of class "zeroinfl"
```

`anova()` 메서드는 `zeroinfl` 객체에 대해 구현되지 않았습니다!

대안은 아카이케 정보 기준(Akaike information criterion, AIC)(Claeskens 2016)과 같은 *정보 기준 통계량(information criterion statistic)*을 사용하는 것입니다. 이는 (훈련 세트에서) 로그 우도를 계산하고 훈련 세트 크기와 모델 매개변수 수를 기반으로 해당 값에 페널티를 줍니다. R의 매개변수화에서는 AIC 값이 작을수록 좋습니다. 이 경우 우리는 공식적인 통계적 검정을 수행하는 것이 아니라 모델이 데이터에 피팅되는 능력을 *추정(estimating)*하는 것입니다.

결과는 ZIP 모델이 더 선호(preferable)됨을 나타냅니다.

```
zero_inflated_fit %>% extract_fit_engine() %>% AIC()
#> [1] 3232
log_lin_reduced   %>% extract_fit_engine() %>% AIC()
#> [1] 3312
```

그러나 이 쌍으로 된 단일 값들을 상황에 맞게 해석(contextualize)하고 실제로 _얼마나_ 다른지 평가하는 것은 어렵습니다. 이 문제를 해결하기 위해 이 두 모델 각각에 대해 수많은 횟수로 재표집을 수행하겠습니다. 여기서부터(From these) 각각에 대한 AIC 값을 계산하고 결과가 얼마나 자주 ZIP 모델을 선호(favor)하는지 결정할 수 있습니다. 기본적으로, 우리는 데이터의 잡음(noise)에 비해(relative to) 차이(difference)를 측정(gauge)하기 위해 AIC 통계량의 불확실성을 특성화(characterizing)할 것입니다.

잠시 후(in a bit) 매개변수에 대해 더 많은 부트스트랩 신뢰 구간을 계산할 것이므로 부트스트랩 샘플을 생성할 때 `apparent = TRUE` 옵션을 지정합니다. 이것은 일부 유형의 구간에서 필요합니다.

먼저 4,000개의 모델 적합을 생성합니다.

```
zip_form <- art ~ fem + mar + kid5 + ment | fem + mar + kid5 + phd + ment
glm_form <- art ~ fem + mar + kid5 + ment

set.seed(2104)
bootstrap_models <-
  bootstraps(bioChemists, times = 2000, apparent = TRUE) %>%
  mutate(
    glm = map(splits, ~ fit(log_lin_spec,       glm_form, data = analysis(.x))),
    zip = map(splits, ~ fit(zero_inflated_spec, zip_form, data = analysis(.x)))
  )
bootstrap_models
#> # Bootstrap sampling with apparent sample
#> # A tibble: 2,001 × 4
#>   splits            id            glm      zip
#>   <list>            <chr>         <list>   <list>
#> 1 <split [915/355]> Bootstrap0001 <fit[+]> <fit[+]>
#> 2 <split [915/333]> Bootstrap0002 <fit[+]> <fit[+]>
#> 3 <split [915/337]> Bootstrap0003 <fit[+]> <fit[+]>
#> 4 <split [915/344]> Bootstrap0004 <fit[+]> <fit[+]>
#> 5 <split [915/351]> Bootstrap0005 <fit[+]> <fit[+]>
#> 6 <split [915/354]> Bootstrap0006 <fit[+]> <fit[+]>
#> # … with 1,995 more rows
```

이제 모델 적합과 해당하는(corresponding) AIC 값을 추출할 수 있습니다.

```
bootstrap_models <-
  bootstrap_models %>%
  mutate(
    glm_aic = map_dbl(glm, ~ extract_fit_engine(.x) %>% AIC()),
    zip_aic = map_dbl(zip, ~ extract_fit_engine(.x) %>% AIC())
  )
mean(bootstrap_models$zip_aic < bootstrap_models$glm_aic)
#> [1] 1
```

과도한 수의(excessive number of) 영(zero) 카운트를 고려(accounting for)하는 것이 좋은 아이디어라는 것이 이 결과에서 결정적인(definitive) 것 같습니다.

###### 참고 (Note)

이러한 계산을 수행(conduct)하기 위해 `fit_resamples()` 또는 워크플로우 세트를 사용할 수 있었습니다. 이 섹션에서는 parsnip 패키지 중 하나에서 지원되지 않는 모델에 대해 tidymodels 도구를 사용하는 방법을 입증(demonstrate)하기 위해 `mutate()` 및 `map()`을 사용하여 모델을 계산했습니다.

재표집된 모델 적합을 계산했으므로, 영(zero) 확률 모델 계수(즉, $`\gamma_{j}`$)에 대한 부트스트랩 구간을 만들어 보겠습니다. `tidy()` 메서드를 사용하여 추출할 수 있으며 `type = "zero"` 옵션을 사용하여 이러한 추정치(estimates)를 얻을 수 있습니다.

```
bootstrap_models <-
  bootstrap_models %>%
  mutate(zero_coefs  = map(zip, ~ tidy(.x, type = "zero")))

# 한 가지 예:
bootstrap_models$zero_coefs[[1]]
#> # A tibble: 6 × 6
#>   term        type  estimate std.error statistic   p.value
#>   <chr>       <chr>    <dbl>     <dbl>     <dbl>     <dbl>
#> 1 (Intercept) zero   -0.128     0.497     -0.257 0.797
#> 2 femWomen    zero   -0.0764    0.319     -0.240 0.811
#> 3 marMarried  zero   -0.112     0.365     -0.307 0.759
#> 4 kid5        zero    0.270     0.186      1.45  0.147
#> 5 phd         zero   -0.178     0.132     -1.35  0.177
#> 6 ment        zero   -0.123     0.0315    -3.91  0.0000935
```

[그림 21-5](#zip-bootstrap)에서와 같이 계수의 부트스트랩 분포를 시각화하는 것이 좋습니다(good idea):

```
bootstrap_models %>%
  unnest(zero_coefs) %>%
  ggplot(aes(x = estimate)) +
  geom_histogram(bins = 25, color = "white") +
  facet_wrap(~ term, scales = "free_x") +
  geom_vline(xintercept = 0, lty = 2, color = "gray70")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2105.png" alt="tmwr 2105" />
<h6 id="figure-21-5.-bootstrap-distributions-of-the-zip-model-coefficients.-the-vertical-lines-indicate-the-observed-estimates.">그림 21-5. ZIP 모델 계수의 부트스트랩 분포. 수직선은 관찰된 추정치를 나타냅니다.</h6>
</figure>

중요한 것으로 보이는 공변량(`ment`) 중 하나는 매우 치우친(skewed) 분포를 가집니다. 일부 패싯(facets)의 추가 공간은 추정치에 약간의 특이치(outliers)가 있음을 나타냅니다. 이것은 모델이 수렴(converge)하지 않았을 때 발생할 _수(might)_ 있습니다; 이러한 결과는 재표집에서 제외(excluded)되어야 합니다. [그림 21-5](#zip-bootstrap)에 시각화된 결과의 경우 특이치는 오직(only) 극단적인 매개변수 추정치 때문입니다; 모든 모델이 수렴했습니다.

rsample 패키지에는 다양한 유형의 부트스트랩 구간을 계산하는 `int_*()`로 이름 지정된 일련의 함수가 포함되어 있습니다. `tidy()` 메서드에는 표준 오차 추정치가 포함되어 있으므로 부트스트랩-t 구간을 계산할 수 있습니다. 또한 표준 백분위수 구간도 계산할 것입니다. 기본적으로 90% 신뢰 구간이 계산됩니다.

```
bootstrap_models %>% int_pctl(zero_coefs)
#> # A tibble: 6 × 6
#>   term        .lower .estimate  .upper .alpha .method
#>   <chr>        <dbl>     <dbl>   <dbl>  <dbl> <chr>
#> 1 (Intercept) -1.75    -0.621   0.423    0.05 percentile
#> 2 femWomen    -0.521    0.115   0.818    0.05 percentile
#> 3 kid5        -0.327    0.218   0.677    0.05 percentile
#> 4 marMarried  -1.20    -0.381   0.362    0.05 percentile
#> 5 ment        -0.401   -0.162  -0.0513   0.05 percentile
#> 6 phd         -0.276    0.0220  0.327    0.05 percentile
bootstrap_models %>% int_t(zero_coefs)
#> # A tibble: 6 × 6
#>   term        .lower .estimate  .upper .alpha .method
#>   <chr>        <dbl>     <dbl>   <dbl>  <dbl> <chr>
#> 1 (Intercept) -1.61    -0.621   0.321    0.05 student-t
#> 2 femWomen    -0.482    0.115   0.671    0.05 student-t
#> 3 kid5        -0.211    0.218   0.599    0.05 student-t
#> 4 marMarried  -0.988   -0.381   0.290    0.05 student-t
#> 5 ment        -0.324   -0.162  -0.0275   0.05 student-t
#> 6 phd         -0.274    0.0220  0.291    0.05 student-t
```

이 결과로부터 0 카운트 확률 모델에 어떤 예측 변수를 포함할지에 대한 좋은 아이디어를 얻을 수 있습니다. `ment`에 대한 부트스트랩 분포가 여전히 왜곡(skewed)되어 있는지 평가하기 위해 더 작은 모델을 다시 피팅하는(refit) 것이 합리적(sensible)일 수 있습니다.

# 더 많은 추론 분석 (More Inferential Analysis)

이 장에서는 tidymodels의 추론 분석에 사용할 수 있는 것의 아주 작은 부분 집합(subset)만 입증(demonstrated)했으며 재표집 및 빈도주의(frequentist) 방법에 중점을 두었습니다. 거의 틀림없이(Arguably), 베이지안 분석은 추론을 위한 매우 효과적이고 종종 우수한 접근 방식입니다. parsnip을 통해 다양한 베이지안 모델을 사용할 수 있습니다. 또한, multilevelmod 패키지를 사용하면 계층적 베이지안 및 비베이지안 모델(혼합 모델(mixed models))을 피팅할 수 있습니다. broom.mixed 및 tidybayes 패키지는 플롯 및 요약(summaries)을 위한 데이터를 추출하는 훌륭한 도구입니다. 마지막으로, 단순 종단적(simple longitudinal) 또는 반복 측정(repeated measures) 데이터와 같이 단일 계층(single hierarchy)이 있는 데이터 세트의 경우 rsample의 `group_vfold_cv()` 함수는 모델 성능의 샘플 외 특성화(out-of-sample characterizations)를 간단하게 용이하도록 돕습니다(facilitates straightforward).

# 이 장의 요약 (Chapter Summary)

tidymodels 프레임워크는 단순히 예측 모델링만을 위한 것이 아닙니다. tidymodels의 패키지와 함수는 가설 검정(hypothesis testing)은 물론 추론 모델을 피팅하고 평가하는 데 사용할 수 있습니다. tidymodels 프레임워크는 non-tidymodels R 모델 작업을 지원하며 모델의 통계적 품질을 평가하는 데 도움을 줄 수 있습니다.

<sup>[1](ch21.xhtml#idm45881844138224-marker)</sup> `T` 인자를 사용하면 이벤트(출판물)가 계산된 시간을 고려할(account for) 수 있으며, 남성과 여성 모두 3년이었습니다. 이 데이터에는 여성보다 남성이 더 많지만 `poisson.test()`는 기능(functionality)이 제한적이므로 더 정교한 분석을 사용하여 이러한 차이를 설명(account for)할 수 있습니다.
