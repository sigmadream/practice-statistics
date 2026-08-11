# 제5부. 프로그램

이 책의 이 파트에서는 여러분의 프로그래밍 기술을 향상시킬 것입니다. 프로그래밍은 모든 데이터 과학 작업에 필요한 교차적(cross-cutting) 기술입니다. 데이터 과학을 하려면 컴퓨터를 사용해야 합니다. 머릿속으로나 연필과 종이로는 할 수 없습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p500.png" alt="Our model of the data science process with program (import, tidy, transform, visualize, model, and communicate, i.e. everything) highlighted in blue." />
<h6 id="figure-v-1.-programming-is-the-water-in-which-all-the-other-components-swim.">그림 V-1. 프로그래밍은 다른 모든 구성 요소가 수영하는 물과 같습니다.</h6>
</figure>

프로그래밍은 코드를 생성하며, 코드는 커뮤니케이션의 도구입니다. 분명히 코드는 컴퓨터에게 여러분이 원하는 것을 하라고 지시합니다. 하지만 그것은 다른 사람들에게 의미를 전달하기도 합니다. 수행하는 모든 프로젝트는 근본적으로 협력적이기 때문에 코드를 커뮤니케이션 수단으로 생각하는 것은 중요합니다. 다른 사람과 함께 작업하지 않더라도 미래의 여러분 자신과는 분명히 작업하게 될 것입니다! 명확한 코드를 작성하는 것은 다른 사람(미래의 여러분)이 분석을 왜 그런 방식으로 다루었는지 이해할 수 있도록 하기 위해 중요합니다. 즉, 프로그래밍에 능숙해진다는 것은 커뮤니케이션에 능숙해지는 것을 의미하기도 합니다. 시간이 지남에 따라 여러분의 코드가 쓰기 쉬워질 뿐만 아니라 다른 사람이 읽기도 쉬워지기를 바랄 것입니다.

이어지는 세 장에서는 프로그래밍 기술을 향상시키기 위한 기술을 배울 것입니다.

- 복사하여 붙여넣기는 강력한 도구이지만 두 번 이상 수행하는 것은 피해야 합니다. 코드에서 자신을 반복하는 것은 오류와 불일치로 쉽게 이어질 수 있기 때문에 위험합니다. 대신, <a href="ch25.html#chp-functions" data-type="xref">25장</a>에서는 쉽게 재사용할 수 있도록 반복되는 tidyverse 코드를 추출할 수 있게 해주는 *함수(functions)*를 작성하는 방법을 배울 것입니다.

- 함수는 반복되는 코드를 추출하지만, 종종 다른 입력에 대해 같은 동작을 반복해야 할 때가 있습니다. 비슷한 일을 반복해서 할 수 있게 해주는 *반복(iteration)*을 위한 도구가 필요합니다. 이러한 도구에는 for 루프와 함수형 프로그래밍이 포함되며, <a href="ch26.html#chp-iteration" data-type="xref">26장</a>에서 이에 대해 배울 것입니다.

- 다른 사람이 작성한 코드를 더 많이 읽다 보면 tidyverse를 사용하지 않는 코드를 더 많이 보게 될 것입니다. <a href="ch27.html#chp-base-R" data-type="xref">27장</a>에서는 실제 환경에서 보게 될 가장 중요한 기본(base) R 함수 중 일부를 배울 것입니다.

이 장들의 목표는 데이터 과학에 필요한 프로그래밍의 최소한을 가르치는 것입니다. 여기 있는 내용을 마스터하고 나면 프로그래밍 기술에 계속 투자할 것을 강력히 권장합니다. 여러분에게 도움이 될 만한 두 권의 책을 작성했습니다. 개릿 그롤문드(Garrett Grolemund)의 [_Hands on Programming with R_](https://oreil.ly/LBFUN)(O’Reilly)은 프로그래밍 언어로서의 R에 대한 소개이며 R이 여러분의 첫 프로그래밍 언어라면 시작하기 좋은 곳입니다. 해들리 위컴(Hadley Wickham)의 [_Advanced R_](https://oreil.ly/I2wE0)(CRC Press)은 프로그래밍 언어인 R의 세부 사항을 파고듭니다. 기존에 프로그래밍 경험이 있다면 시작하기 좋은 곳이며, 이 장들의 아이디어를 내면화한 후의 훌륭한 다음 단계입니다.
