# 제III부. 변환 (Transform)

이 책의 두 번째 부분에서는 데이터 시각화에 대해 깊이 파고들었습니다. 이 책의 이 부분에서는 데이터 프레임 내에서 접하게 될 가장 중요한 유형의 변수들에 대해 알아보고, 그것들을 다루기 위해 사용할 수 있는 도구들을 배울 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p300.png" alt="Our data science model, with transform highlighted in blue." />
<h6 id="figure-iii-1.-the-options-for-data-transformation-depend-heavily-on-the-type-of-data-involved-the-subject-of-this-part-of-the-book.">그림 III-1. 데이터 변환 옵션은 관련 데이터의 유형에 크게 좌우되며, 이것이 이 책의 이 부분에서 다룰 주제입니다.</h6>
</figure>

이 장들은 필요에 따라 읽을 수 있습니다. 순서와 상관없이 읽을 수 있도록 대체로 독립적으로 구성되어 있습니다.

- <a href="ch12.html#chp-logicals" data-type="xref">제12장</a>에서는 논리형 벡터(logical vectors)에 대해 가르칩니다. 이것들은 가장 단순한 유형의 벡터지만 극히 강력합니다. 숫자 비교로 이들을 생성하는 방법, 부울 대수(Boolean algebra)와 결합하는 방법, 요약(summaries)에서 사용하는 방법, 그리고 조건부 변환(condition transformations)에 사용하는 방법을 배울 것입니다.
- <a href="ch13.html#chp-numbers" data-type="xref">제13장</a>은 데이터 과학의 원동력인 숫자 벡터를 위한 도구들에 대해 깊이 다룹니다. 카운팅(counting)에 대해 더 알아보고, 여러 가지 중요한 변환 및 요약 함수들을 배울 것입니다.
- <a href="ch14.html#chp-strings" data-type="xref">제14장</a>은 문자열(strings)을 다루는 도구를 제공합니다. 문자열을 자르고(slice), 쪼개고(dice), 다시 붙이는(stick back together) 방법을 배울 것입니다. 이 장은 주로 stringr 패키지에 초점을 맞추지만, 문자열에서 데이터를 추출하는 데 특화된 더 많은 tidyr 함수들도 배울 것입니다.
- <a href="ch15.html#chp-regexps" data-type="xref">제15장</a>은 문자열을 조작하기 위한 강력한 도구인 정규 표현식(regular expressions)을 소개합니다. 이 장은 고양이가 키보드 위를 걸어간 것처럼 보이던 것을 복잡한 문자열 패턴을 읽고 쓸 수 있는 수준으로 여러분을 이끌어 줄 것입니다.
- <a href="ch16.html#chp-factors" data-type="xref">제16장</a>은 R이 범주형 데이터를 저장하는 데 사용하는 데이터 유형인 요인(factors)을 소개합니다. 변수가 고정된 가능한 값 집합을 가질 때, 또는 문자열을 알파벳 순서가 아닌 다른 순서로 사용하고 싶을 때 요인을 사용합니다.
- <a href="ch17.html#chp-datetimes" data-type="xref">제17장</a>은 날짜와 날짜-시간(date-times)을 다루기 위한 핵심 도구를 제공합니다. 불행히도 날짜-시간에 대해 더 많이 알게 될수록 더 복잡해지는 것처럼 보이지만, lubridate 패키지의 도움을 받아 가장 일반적인 과제들을 어떻게 극복하는지 배울 것입니다.
- <a href="ch18.html#chp-missing-values" data-type="xref">제18장</a>은 결측값(missing values)에 대해 깊이 있게 논의합니다. 이전에도 개별적으로 몇 번 논의했지만, 이제는 암묵적 결측값과 명시적 결측값의 차이점, 그리고 그것들을 어떻게 그리고 왜 변환해야 하는지를 파악하도록 도우면서 종합적으로 논의할 때입니다.
- <a href="ch19.html#chp-joins" data-type="xref">제19장</a>은 두 개(또는 그 이상)의 데이터 프레임을 함께 결합(join)하는 도구를 제공함으로써 이 책의 이 부분을 마무리합니다. 결합(joins)에 대해 배우면서 키(keys)라는 개념과 씨름해야 하며, 데이터세트의 각 행을 어떻게 식별할지에 대해 생각하게 될 것입니다.
