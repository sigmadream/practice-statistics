# `R`을 공부하면서 알게 된 몇가지

## 리눅스 사용자

[https://github.com/eddelbuettel/r2u](https://github.com/eddelbuettel/r2u)에서 R 패키지 설치 및 설정을 진행하세요.

## 설정

```bash
renv::restore()
```

## renv

R의 `renv`는 프로젝트 수준에서 패키지 의존성을 관리하여 환경의 재현성을 보장해 주는 강력한 도구입니다. Python의 `venv`나 `requirements.txt`, Node.js의 `package.json`과 유사한 역할을 하며, 다른 환경이나 다른 사람의 컴퓨터에서도 동일한 버전의 R 패키지 구성을 쉽게 복구할 수 있게 해줍니다.

### 핵심 워크플로우

`renv`를 사용하는 기본 과정은 초기화 → 작업(패키지 설치) → 기록(스냅샷) → 복원의 순환으로 이루어집니다.

#### 프로젝트 초기화

RStudio에서 새 프로젝트를 생성하거나, 해당 프로젝트 폴더에서 R 콘솔을 열고 초기화를 진행합니다.

```r
renv::init()
```

이 명령어를 실행하면 프로젝트 디렉토리 내에 독립적인 패키지 보관소(`renv/library/`)가 생성되며, 현재 프로젝트의 의존성을 기록하는 `renv.lock` 파일과 R 시작 시 환경을 자동으로 활성화하는 `.Rprofile` 파일이 만들어집니다.

#### 패키지 설치 및 스크립트 작성

평소와 동일한 방법으로 R을 사용합니다. 패키지를 설치하면 시스템 전역 라이브러리가 아닌 해당 프로젝트의 로컬 라이브러리에 설치됩니다.

```r
install.packages("dplyr")
install.packages("ggplot2")
```

#### 의존성 기록 (스냅샷 저장)

코드를 작성하면서 새로운 패키지를 추가했거나 업데이트를 마쳤다면, 현재 환경 상태를 저장해야 합니다.

```r
renv::snapshot()
```

이 명령어는 프로젝트 내의 R 스크립트들을 분석하여 실제로 사용된 패키지들을 찾아내고, 해당 패키지들의 정확한 버전을 `renv.lock` 파일에 업데이트합니다.

#### 환경 복원

동료와 협업하거나, 다른 컴퓨터로 프로젝트를 옮겼을 때 실행하는 명령어입니다. Git 등을 통해 공유받은 프로젝트 폴더에서 R을 실행하면 `.Rprofile`에 의해 `renv`가 자동으로 활성화됩니다.

```r
renv::restore()
```

이 명령을 실행하면 `renv.lock` 파일에 명시된 버전과 정확히 일치하는 패키지들을 다운로드하고 설치하여 이전 작업 환경을 완벽하게 재구성합니다.

### 유용한 추가 명령어

작업을 진행하다 보면 현재 상태를 점검하거나 패키지를 업데이트해야 할 때가 있습니다.

#### 상태 확인

현재 설치된 로컬 패키지 상태와 `renv.lock`에 기록된 상태 간의 불일치가 있는지 확인합니다. 스냅샷을 찍어야 할지, 복원을 해야 할지 알려주는 유용한 진단 도구입니다.

```r
renv::status()
```

#### 패키지 업데이트

프로젝트 내에 설치된 패키지들을 최신 버전으로 업데이트합니다. 업데이트 후에는 코드가 잘 동작하는지 테스트하고 다시 `renv::snapshot()`을 실행해야 합니다.

```r
renv::update()
```

#### Git 등 버전 관리 시스템 사용 시 주의사항

`renv::init()`을 실행하면 자동으로 적절한 `.gitignore` 파일이 생성되지만, 수동으로 관리할 경우 다음 규칙을 따르세요.

- 포함해야 할 파일: `renv.lock`, `.Rprofile`, `renv/activate.R`, `renv/settings.json`
- 제외해야 할 파일: `renv/library/`, `renv/staging/`

## Ref

- [ ] Analyzing Baseball Data With R 3rd
- [ ] Tidy Modeling with R
- [ ] Data Wrangling with R
- [ ] Dynamic Documents with R and knitr 2nd

---

- [ ] R for Data Science 2nd
- [ ] R을 이용한 데이터 분석 실무
- [ ] R을 이용한 누구나 하는 통계분석
- [ ] R-확률통계
- [ ] R로 배우는 딥러닝
- [ ] Do it! 쉽게 배우는 R 데이터 분석
- [ ] Do it! 쉽게 배우는 R 텍스트 마이닝
- [ ] Do it! 쉽게 배우는 R 데이터 분석
- [ ] Do it! 공공데이터로 배우는 R 데이터 분석 with 샤이니
- [ ] 시계열 데이터 처리와 분석 in R
- [ ] 빅 데이터 분석을 위한 R 프로그래밍 2nd
- [ ] Must Have 나성호의 R 데이터 분석 입문
- [ ] 데이터 과학을 위한 파이썬과 R
- [ ] 모두를 위한 R 데이터 분석 입문(2판)

---

- [ ] Hands-On Programming with R
- [ ] Advanced R
- [ ] R in Action 2nd
- [ ] Using R and RStudio for Data Management, Statistical Analysis, and Graphics 2nd
- [ ] ggplot2 Elegant Graphics for Data Analysis 2nd
- [ ] Learn R for Applied With Data Visualizations, Regressions, and Statistics
- [ ] Practical data science with R 2nd
- [ ] R Visualizations Derive Meaning from Data
- [ ] Modern Data Science with R 2nd
- [ ] R for Health Data Science
- [ ] Visualizing Data in R4
- [ ] R in Action 3rd
- [ ] blogdown - Creating Websites with R Markdown
- [ ] bookdown
- [ ] Deep Learning and Scientific Computing with R torch
- [ ] Model-Based Clustering, Classification, and Density Estimation Using mclust in R
