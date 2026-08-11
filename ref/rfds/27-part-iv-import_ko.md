# 4부. 가져오기(Import)

이 책의 이 부분에서는 더 넓은 범위의 데이터를 R로 가져오는 방법과 분석에 유용한 형태로 만드는 방법을 배웁니다. 때로는 적절한 데이터 가져오기 패키지에서 함수를 호출하는 것만으로 충분합니다. 하지만 더 복잡한 경우에는 작업하기 선호하는 타이디한 직사각형(tidy rectangle) 형태를 얻기 위해 타이디 작업(tidying)과 변환(transformation)이 모두 필요할 수 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p400.png" alt="가져오기(import)가 파란색으로 강조 표시된 우리의 데이터 사이언스 모델." />
<h6 id="figure-iv-1.-data-import-is-the-beginning-of-the-data-science-process-without-data-you-cant-do-data-science">그림 IV-1. 데이터 가져오기는 데이터 사이언스 프로세스의 시작입니다. 데이터 없이는 데이터 사이언스를 할 수 없습니다!</h6>
</figure>

책의 이 부분에서는 다음과 같은 방식으로 저장된 데이터에 액세스하는 방법을 배웁니다.

- <a href="ch20.html#chp-spreadsheets" data-type="xref">20장</a>에서는 Excel 스프레드시트 및 Google 스프레드시트에서 데이터를 가져오는 방법을 배웁니다.

- <a href="ch21.html#chp-databases" data-type="xref">21장</a>에서는 데이터베이스에서 데이터를 가져와 R로 불러오는 방법을 배웁니다(또한 R에서 데이터를 내보내 데이터베이스에 넣는 방법도 조금 배웁니다).

- <a href="ch22.html#chp-arrow" data-type="xref">22장</a>에서는 특히 parquet 형식으로 저장된 메모리 초과 데이터(out-of-memory data)로 작업하기 위한 강력한 도구인 Arrow에 대해 배웁니다.

- <a href="ch23.html#chp-rectangling" data-type="xref">23장</a>에서는 JSON 형식으로 저장된 데이터에 의해 생성된 깊게 중첩된 목록(nested lists)을 포함하여 계층적 데이터로 작업하는 방법을 배웁니다.

- <a href="ch24.html#chp-webscraping" data-type="xref">24장</a>에서는 웹 페이지에서 데이터를 추출하는 기술이자 과학인 웹 "스크래핑(scraping)"을 배웁니다.

여기에서 논의하지 않는 두 가지 중요한 tidyverse 패키지가 있습니다. haven과 xml2입니다. SPSS, Stata 및 SAS 파일의 데이터로 작업하는 경우 [haven 패키지](https://oreil.ly/cymF4)를 확인하세요. XML 데이터로 작업하는 경우 [xml2 패키지](https://oreil.ly/lQNBa)를 확인하세요. 그 외의 경우에는 어떤 패키지를 사용해야 할지 알아내기 위해 약간의 조사를 해야 할 것입니다. 이럴 때는 Google이 당신의 친구입니다.
