# 2부. 시각화 (Part II. Visualize)

이 책의 첫 번째 부분을 읽고 나면, 여러분은 데이터 과학을 수행하기 위한 가장 중요한 도구들을 (적어도 표면적으로는) 이해하게 됩니다. 이제 세부 사항으로 깊이 들어갈 시간입니다. 이 책의 이번 부분에서는 <a href="#fig-ds-visualize" data-type="xref">그림 II-1</a>에서 더 깊이 있게 데이터를 시각화하는 방법에 대해 배울 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p200.png" alt="시각화 부분이 파란색으로 강조 표시된 데이터 과학 모델." />
<h6 id="figure-ii-1.-data-visualization-is-often-the-first-step-in-data-exploration.">그림 II-1. 데이터 시각화는 종종 데이터 탐색의 첫 번째 단계입니다.</h6>
</figure>

각 장에서는 데이터 시각화 생성의 하나에서 몇 가지 측면을 다룹니다.

- <a href="ch09.html#chp-layers" data-type="xref">9장</a>에서는 그래픽의 레이어 기반 문법(layered grammar of graphics)에 대해 배울 것입니다.

- <a href="ch10.html#chp-EDA" data-type="xref">10장</a>에서는 시각화를 호기심 및 회의론과 결합하여 데이터에 관한 흥미로운 질문을 던지고 그에 답해볼 것입니다.

- 마지막으로 <a href="ch11.html#chp-communication" data-type="xref">11장</a>에서는 여러분의 탐색적(exploratory) 그래픽을 가져와 수준을 높여 설명적(expository) 그래픽으로 바꾸는 방법을 배울 것입니다. 설명적 그래픽이란 여러분의 분석을 처음 접하는 사람이 무슨 일이 일어나고 있는지 가능한 한 빠르고 쉽게 이해할 수 있도록 돕는 그래픽입니다.

이 세 장은 시각화의 세계를 시작하게 해주지만 배울 것이 훨씬 더 많습니다. 더 많은 것을 배우기에 절대적으로 가장 좋은 곳은 ggplot2 책인 [_ggplot2: Elegant Graphics for Data Analysis_](https://oreil.ly/SO1yG) (Springer)입니다. 이 책은 기본 이론에 대해 훨씬 더 깊이 다루며 개별 조각들을 결합하여 실용적인 문제를 해결하는 방법에 대한 더 많은 예를 가지고 있습니다. 또 다른 훌륭한 리소스는 [ggplot2 extensions gallery(확장판 갤러리)](https://oreil.ly/m0OW5)입니다. 이 사이트는 새로운 geom 및 scale로 ggplot2를 확장하는 많은 패키지들을 나열합니다. ggplot2로 하기 어려워 보이는 무언가를 시도하고 있다면 시작하기에 아주 좋은 곳입니다.
