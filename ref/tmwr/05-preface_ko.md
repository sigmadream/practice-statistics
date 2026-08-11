# 서문

*Tidy Modeling with R*에 오신 것을 환영합니다! 이 책은 모델 구축을 위해 R 프로그래밍 언어로 작성된 tidymodels라는 소프트웨어 컬렉션을 사용하는 가이드이며, 두 가지 주요 목표를 가지고 있습니다.

- 가장 먼저, 이 책은 모델을 생성하기 위해 이러한 특정 R 패키지를 *사용하는 방법*에 대한 실용적인 소개를 제공합니다. 일관되고 인간 중심적인 철학으로 설계된 [tidyverse](https://oreil.ly/xnx26)라는 R의 방언에 초점을 맞추고, tidyverse와 tidymodels 패키지를 사용하여 고품질의 통계 및 머신러닝 모델을 생성하는 방법을 보여줍니다.

- 두 번째로, 이 책은 *좋은 방법론과 통계적 관행을 개발하는 방법*을 보여줄 것입니다. 가능하면 저희의 소프트웨어, 문서 및 기타 자료는 일반적인 함정을 방지하도록 시도합니다.

[1장](ch01.xhtml#software-modeling)에서는 모델에 대한 분류 체계를 간략하게 설명하고 모델링을 위한 좋은 소프트웨어가 어떤 것인지 강조합니다. [2장](ch02.xhtml#tidyverse)에서 소개(또는 복습)하는 tidyverse의 아이디어와 구문은 방법론 및 관행의 이러한 과제에 대한 tidymodels 접근 방식의 기반이 됩니다. [3장](ch03.xhtml#base-r)에서는 기존의 기본 R 모델링 함수에 대해 빠르게 살펴보고 해당 영역에서 충족되지 않은 요구 사항을 요약합니다.

그 후, 이 책은 깔끔한 데이터(tidy data) 원칙을 사용한 모델링의 기초부터 시작하여 여러 부분으로 나뉩니다. [4장](ch04.xhtml#ames)에서 [9장](ch09.xhtml#performance)까지는 주택 가격에 대한 예제 데이터 세트를 소개하고 근본적인 tidymodels 패키지인 recipes, parsnip, workflows, yardstick 등을 사용하는 방법을 보여줍니다.

책의 다음 부분에서는 효과적인 모델을 생성하는 프로세스에 대한 자세한 내용을 다룹니다. [10장](ch10.xhtml#resampling)에서 [15장](ch15.xhtml#workflow-sets)까지는 하이퍼파라미터 튜닝(Hyperparameter Tuning)뿐만 아니라 성능에 대한 좋은 추정치를 생성하는 데 중점을 둡니다.

마지막으로 이 책의 마지막 섹션인 [16장](ch16.xhtml#dimensionality)에서 [21장](ch21.xhtml#inferential)까지는 모델 구축을 위한 기타 중요한 주제를 다룹니다. 차원 축소(Dimensionality Reduction) 및 고차원 범주형 예측 변수(High-Cardinality Predictors) 인코딩과 같은 더 고급 특징 공학(Feature Engineering) 접근 방식은 물론, 모델이 특정 예측을 내리는 이유와 모델 예측을 언제 신뢰해야 하는지에 대한 질문에 답하는 방법을 논의합니다.

저희는 독자가 모델 구축 및 통계에 대한 폭넓은 경험이 있다고 가정하지 않습니다. 무작위 샘플링(Random Sampling), 분산(Variance), 상관관계(Correlation), 기초적인 선형 회귀(Linear Regression) 및 일반적으로 기초 학부 통계 또는 데이터 분석 과정에서 다루는 기타 주제와 같은 약간의 통계 지식이 필요합니다. 저희는 독자가 R의 dplyr, ggplot2 및 `%>%` "파이프" 연산자에 대해 최소한 약간은 익숙하고 이러한 도구를 모델링에 적용하는 데 관심이 있다고 가정합니다. 아직 이러한 배경 R 지식이 없는 사용자의 경우 Wickham과 Grolemund(2016)가 저술한 [_R for Data Science_](https://r4ds.had.co.nz)와 같은 책을 권장합니다. 데이터를 조사하고 분석하는 것은 모든 모델 프로세스에서 중요한 부분입니다.

이 책은 모델링 기법에 대한 포괄적인 참고서로 의도된 것이 아닙니다. 통계적 방법론 자체에 대해 자세히 알아보려면 다른 리소스를 제안합니다. 가장 일반적인 모델 유형인 선형 모델에 대한 일반적인 배경지식은 Fox(2008)를 제안합니다. 예측 모델의 경우 Kuhn과 Johnson(2013) 및 Kuhn과 Johnson(2020)이 좋은 리소스입니다. 머신러닝 방법의 경우 Goodfellow, Bengio 및 Courville(2016)이 훌륭한(하지만 형식적인) 정보 출처입니다. 경우에 따라 저희가 사용하는 모델에 대해 다소 자세히 설명하기도 하지만, 덜 수학적이고 더 직관적인 방식으로 설명하기를 희망합니다.

# 이 책에서 사용된 표기 규칙

이 책에서는 다음과 같은 타이포그래피 규칙이 사용됩니다.

_기울임꼴 (Italic)_  
새로운 용어, URL, 이메일 주소, 파일 이름 및 파일 확장자를 나타냅니다.

`고정 폭 (Constant width)`  
프로그램 목록뿐만 아니라 단락 내에서 변수나 함수 이름, 데이터베이스, 데이터 유형, 환경 변수, 명령문 및 키워드와 같은 프로그램 요소를 나타내는 데 사용됩니다.

**`고정 폭 굵게 (Constant width bold)`**  
사용자가 문자 그대로 입력해야 하는 명령 또는 기타 텍스트를 보여줍니다.

_`고정 폭 기울임꼴 (Constant width italic)`_  
사용자가 제공한 값이나 문맥에 따라 결정된 값으로 대체되어야 하는 텍스트를 보여줍니다.

###### 팁 (Tip)

이 요소는 팁이나 제안을 나타냅니다.

###### 참고 (Note)

이 요소는 일반적인 참고 사항을 나타냅니다.

###### 경고 (Warning)

이 요소는 경고나 주의 사항을 나타냅니다.

# 코드 예제 사용

보충 자료(코드 예제, 연습 문제 등)는 [_https://github.com/tidymodels/TMwR_](https://github.com/tidymodels/TMwR)에서 다운로드할 수 있습니다. 이 책은 [bookdown](http://bookdown.org)(Xie 2016)을 사용하여 [RStudio](https://oreil.ly/bcWV6)로 작성되었습니다. 이 책의 모든 플롯은 [ggplot2](https://oreil.ly/vEJBy)와 그 흑백 테마(`theme_bw()`)를 사용하여 생성되었습니다. 이 책의 [온라인 버전](https://tmwr.org)이 제공되며 실제 책 출판 후에도 계속 발전할 것입니다.

기술적인 질문이 있거나 코드 예제 사용에 문제가 있는 경우 *bookquestions@oreilly.com*으로 이메일을 보내주세요.

이 책은 여러분의 작업을 돕기 위해 여기에 있습니다. 일반적으로 이 책과 함께 제공되는 예제 코드는 여러분의 프로그램과 문서에서 사용할 수 있습니다. 코드의 상당 부분을 재현하지 않는 한 허락을 받기 위해 저희에게 연락할 필요는 없습니다. 예를 들어, 이 책의 여러 코드 청크를 사용하는 프로그램을 작성하는 것은 허락이 필요하지 않습니다. O'Reilly 책의 예제를 판매하거나 배포하는 데는 허가가 필요합니다. 이 책을 인용하고 예제 코드를 인용하여 질문에 답하는 것은 허락이 필요하지 않습니다. 이 책의 예제 코드를 상당량 여러분 제품의 문서에 통합하는 데는 허락이 필요합니다.

저희는 감사를 표하는 것을 높이 평가하지만 일반적으로 요구하지는 않습니다. 표창에는 대개 제목, 저자, 출판사 및 ISBN이 포함됩니다. 예: "_Tidy Modeling with R_ by Max Kuhn and Julia Silge (O’Reilly). Copyright 2022 Max Kuhn and Julia Silge, 978-1-492-09648-1."

코드 예제 사용이 공정 사용이나 위에 명시된 허가 범위를 벗어난다고 느끼시면 언제든지 *permissions@oreilly.com*으로 연락해 주세요.

이 책의 현재 버전은 R 버전 4.1.3 (2022-03-10), [pandoc](https://pandoc.org) 버전 2.17.1.1 및 다음 패키지로 빌드되었습니다.

- applicable (0.0.1.2, CRAN)
- av (0.7.0, CRAN)
- baguette (0.2.0, CRAN)
- beans (0.1.0, CRAN)
- bestNormalize (1.8.2, CRAN)
- bookdown (0.25, CRAN)
- broom (0.7.12, CRAN)
- censored (0.0.0.9000, GitHub)
- corrplot (0.92, CRAN)
- corrr (0.4.3, CRAN)
- Cubist (0.4.0, CRAN)
- DALEXtra (2.1.1, CRAN)
- dials (0.1.1, CRAN)
- dimRed (0.2.5, CRAN)
- discrim (0.2.0, CRAN)
- doMC (1.3.8, CRAN)
- dplyr (1.0.8, CRAN)
- earth (5.3.1, CRAN)
- embed (0.1.5, CRAN)
- fastICA (1.2-3, CRAN)
- finetune (0.2.0, CRAN)
- forcats (0.5.1, CRAN)
- ggforce (0.3.3, CRAN)
- ggplot2 (3.3.5, CRAN)
- glmnet (4.1-3, CRAN)
- gridExtra (2.3, CRAN)
- infer (1.0.0, CRAN)
- kableExtra (1.3.4, CRAN)
- kernlab (0.9-30, CRAN)
- kknn (1.3.1, CRAN)
- klaR (1.7-0, CRAN)
- knitr (1.38, CRAN)
- learntidymodels (0.0.0.9001, GitHub)
- lime (0.5.2, CRAN)
- lme4 (1.1-29, CRAN)
- lubridate (1.8.0, CRAN)
- mda (0.5-2, CRAN)
- mixOmics (6.18.1, Bioconductor)
- modeldata (0.1.1, CRAN)
- multilevelmod (0.1.0, CRAN)
- nlme (3.1-157, CRAN)
- nnet (7.3-17, CRAN)
- parsnip (0.2.1.9001, GitHub)
- patchwork (1.1.1, CRAN)
- pillar (1.7.0, CRAN)
- poissonreg (0.2.0, CRAN)
- prettyunits (1.1.1, CRAN)
- probably (0.0.6, CRAN)
- pscl (1.5.5, CRAN)
- purrr (0.3.4, CRAN)
- ranger (0.13.1, CRAN)
- recipes (0.2.0, CRAN)
- rlang (1.0.2, CRAN)
- rmarkdown (2.13, CRAN)
- rpart (4.1.16, CRAN)
- rsample (0.1.1, CRAN)
- rstanarm (2.21.3, CRAN)
- rules (0.2.0, CRAN)
- sessioninfo (1.2.2, CRAN)
- stacks (0.2.2, CRAN)
- stringr (1.4.0, CRAN)
- svglite (2.1.0, CRAN)
- text2vec (0.6, CRAN)
- textrecipes (0.5.1.9000, GitHub)
- themis (0.2.0, CRAN)
- tibble (3.1.6, CRAN)
- tidymodels (0.2.0, CRAN)
- tidyposterior (0.1.0, CRAN)
- tidyverse (1.3.1, CRAN)
- tune (0.2.0, CRAN)
- uwot (0.1.11, CRAN)
- workflows (0.2.6, CRAN)
- workflowsets (0.2.1, CRAN)
- xgboost (1.5.2.1, CRAN)
- yardstick (0.0.9, CRAN)

# O’Reilly 온라인 러닝 (O’Reilly Online Learning)

###### 참고 (Note)

40년 이상 동안 *O’Reilly Media*는 회사가 성공할 수 있도록 돕는 기술 및 비즈니스 교육, 지식, 통찰력을 제공해 왔습니다.

우리의 독보적인 전문가 및 혁신가 네트워크는 서적, 기사 및 온라인 학습 플랫폼을 통해 그들의 지식과 전문성을 공유합니다. O'Reilly의 온라인 학습 플랫폼은 O'Reilly 및 200개 이상의 다른 출판사의 라이브 교육 과정, 심층 학습 경로, 인터랙티브 코딩 환경, 방대한 텍스트 및 비디오 컬렉션에 대한 온디맨드 액세스를 제공합니다. 자세한 내용은 *https://oreilly.com*을 방문하십시오.

# 연락처 (How to Contact Us)

이 책과 관련된 의견이나 질문은 다음 출판사로 보내주시기 바랍니다.

- O’Reilly Media, Inc.
- 1005 Gravenstein Highway North
- Sebastopol, CA 95472
- 800-998-9938 (미국 또는 캐나다)
- 707-829-0515 (해외 또는 현지)
- 707-829-0104 (팩스)

이 책의 웹페이지에서 정오표, 예제 및 추가 정보를 확인할 수 있습니다. 이 페이지는 [_https://oreil.ly/tidy-modeling-r_](https://oreil.ly/tidy-modeling-r)에서 접근 가능합니다.

이 책에 대한 의견이나 기술적인 질문은 *bookquestions@oreilly.com*으로 이메일을 보내주세요.

책과 과정에 대한 뉴스 및 정보를 얻으시려면 [_https://oreilly.com_](https://oreilly.com)을 방문해 주세요.

LinkedIn에서 저희를 찾으실 수 있습니다. [_https://linkedin.com/company/oreilly-media_](https://linkedin.com/company/oreilly-media).

Twitter에서 팔로우하세요. [_https://twitter.com/oreillymedia_](https://twitter.com/oreillymedia).

YouTube에서 확인하세요. [_https://youtube.com/oreillymedia_](https://youtube.com/oreillymedia).

# 감사의 글 (Acknowledgments)

이 프로젝트에서 저희를 지원해 주신 분들의 기여, 도움, 그리고 관점에 깊이 감사드립니다. 특히 감사의 인사를 전하고 싶은 분들이 있습니다.

tidymodels 팀의 RStudio 동료들(Davis Vaughan, Hannah Frick, Emil Hvitfeldt, Simon Couch)과 RStudio 오픈소스 팀의 다른 동료들에게 감사드립니다. 온라인 작업의 사이트 디자인을 맡아준 Desirée De Leon에게 감사드립니다. 또한 이 책을 실질적으로 개선해 준 상세하고 통찰력 있는 피드백을 제공해 주신 기술 리뷰어 Chelsea Parlett-Pelleriti와 Dan Simpson, 그리고 글쓰기 및 출판 과정에서 관점과 지침을 제공해 주신 편집자 Nicole Taché와 Rita Fernando에게도 감사를 표합니다.

이 책은 공개적으로 쓰여졌으며, 여러 사람들이 풀 리퀘스트나 이슈를 통해 기여해 주셨습니다. 특히 GitHub 풀 리퀘스트를 통해 기여해 주신 38분께 특별한 감사를 전합니다(알파벳 순): Aris Paschalidis (@arisp99), Brad Hill (@bradisbrad), Bryce Roney (@bryceroney), Cedric Batailler (@cedricbatailler), Ildikó Czeller (@czeildi), David Kane (@davidkane9), @DavZim, @DCharIAA, Emil Hvitfeldt (@EmilHvitfeldt), Emilio (@emilopezcano), Fgazzelloni (@Fgazzelloni), Hannah Frick (@hfrick), Hlynur (@hlynurhallgrims), Howard Baek (@howardbaek), Jae Yeon Kim (@jaeyk), Jonathan D. Trattner (@jdtrat), Jeffrey Girard (@jmgirard), John W. Pickering (@JohnPickering), Jon Harmon (@jonthegeek), Joseph B. Rickert (@joseph-rickert), Maximilian Rohde (@maxdrohde), Michael Grund (@michaelgrund), @MikeJohnPage, Mine Cetinkaya-Rundel (@mine-cetinkaya-rundel), Mohammed Hamdy (@mmhamdy), @nattalides, Y. Yu (@PursuitOfDataScience), Riaz Hedayati (@riazhedayati), Rob Wiederstein (@RobWiederstein), Scott (@scottyd22), Simon Schölzel (@simonschoe), Simon Sayz (@tagasimon), @thrkng, Tanner Stauss (@tmstauss), Tony ElHabr (@tonyelhabr), Dmitry Zotikov (@x1o), Xiaochi (@xiaochi-liu), 그리고 Zach Bogart (@zachbogart).
