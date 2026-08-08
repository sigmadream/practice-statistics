# Part I. 전체 게임 (Whole Game)

책의 이 부분에서 우리의 목표는 데이터 과학의 주요 도구들에 대한 빠른 개요를 제공하는 것입니다. <a href="#fig-ds-whole-game" data-type="xref">그림 I-1</a>에 나와 있듯이 *가져오기(importing)*, *깔끔하게 정리하기(tidying)*, *변환하기(transforming)*, 그리고 *데이터 시각화(visualizing data)* 가 포함됩니다. 우리는 단순할지라도 실제 데이터셋을 다룰 수 있도록 모든 주요 요소들의 핵심만을 제공하여 데이터 과학의 "전체 게임"을 보여주고자 합니다. 책의 뒷부분에서는 이러한 각 주제를 더 깊이 다루어, 여러분이 해결할 수 있는 데이터 과학 문제의 범위를 넓혀줄 것입니다.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p100.png" alt="A diagram displaying the data science cycle: Import -&gt; Tidy -&gt; Understand (which has the phases Transform -&gt; Visualize -&gt; Model in a cycle) -&gt; Communicate. Surrounding all of these is Program Import, Tidy, Transform, and Visualize is highlighted." /></p>
<h6 id="figure-i-1.-in-this-section-of-the-book-youll-learn-how-to-import-tidy-transform-and-visualize-data.">그림 I-1. 이 섹션에서는 데이터를 가져오고, 깔끔하게 정리하고, 변환하고, 시각화하는 방법을 배웁니다.</h6>
</figure>

다음 네 개의 장은 데이터 과학의 도구들에 중점을 둡니다:

- 시각화는 R 프로그래밍을 시작하기에 아주 좋은 출발점입니다. 그 보상이 매우 명확하기 때문입니다: 데이터를 이해하는 데 도움이 되는 우아하고 유익한 그래프를 만들 수 있습니다. <a href="ch01.html#chp-data-visualize" data-type="xref">1장</a>에서는 시각화에 깊이 들어가서 ggplot2 그래프의 기본 구조와 데이터를 그래프로 바꾸는 강력한 기술을 배우게 됩니다.

- 시각화만으로는 일반적으로 충분하지 않으므로, <a href="ch03.html#chp-data-transform" data-type="xref">3장</a>에서는 중요한 변수를 선택하고, 핵심 관측치를 필터링하고, 새로운 변수를 만들고, 요약값을 계산할 수 있게 해주는 핵심 동사들을 배웁니다.

- <a href="ch05.html#chp-data-tidy" data-type="xref">5장</a>에서는 데이터를 저장하는 일관된 방법인 깔끔한 데이터(tidy data)에 대해 배울 것입니다. 이는 변환, 시각화, 모델링을 더 쉽게 만들어줍니다. 깔끔한 데이터의 기본 원칙과 데이터를 깔끔한 형태로 만드는 방법을 배우게 됩니다.

- 데이터를 변환하고 시각화하기 전에 먼저 R로 데이터를 가져와야 합니다. <a href="ch07.html#chp-data-import" data-type="xref">7장</a>에서는 `.csv` 파일을 R로 가져오는 기본 사항을 배웁니다.

이 장들 사이사이에는 R 워크플로에 중점을 둔 다른 네 개의 장이 있습니다. <a href="ch02.html#chp-workflow-basics" data-type="xref">2장</a>, <a href="ch04.html#chp-workflow-style" data-type="xref">4장</a>, <a href="ch06.html#chp-workflow-scripts" data-type="xref">6장</a>에서는 R 코드를 작성하고 구성하는 데 있어 좋은 워크플로 모범 사례를 배웁니다. 실제 프로젝트를 다룰 때 구조적으로 잘 정리할 수 있는 도구를 제공함으로써 장기적인 성공의 기반을 마련해 줄 것입니다. 마지막으로, <a href="ch08.html#chp-workflow-help" data-type="xref">8장</a>에서는 도움을 얻고 계속해서 학습하는 방법을 가르쳐 줍니다.
