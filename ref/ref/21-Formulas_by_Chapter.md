# Formulas by Chapter

This appendix presents a comprehensive collection of statistical formulas referenced throughout the text. The formulas are organized by chapter to correspond with the topics discussed in the main body, providing a quick reference for descriptive statistics, reliability measures, inferential tests, and correlation and regression analyses. Each equation includes the appropriate notation and numbering to facilitate cross-referencing.

The purpose of this appendix is to provide readers with a concise and accessible resource for understanding the mathematical foundations of the statistical methods employed in this work. By including both common and specialized formulas, the appendix serves as a practical guide for researchers, students, and practitioners seeking to apply these techniques accurately and consistently.

All symbols are defined where necessary, and standard statistical conventions are used. This structured reference allows the reader to verify calculations, interpret results, and replicate analyses with clarity and precision.

## Chapter [3](./12-3._Statistical_Concepts.md): Descriptive Statistics

_Mean_

x¯\=∑Xn

(3.1)

_Median Absolute Deviation_

MAD\=medianXi−M

(3.2)

_Standard Deviation_

s\=∑(X−X¯)2n−1

(3.3)

_Standard Error of the Mean_

SEM\=sn

(3.4)

_Skewness Equation_

g\=1n∑i\=1nxi−x¯s3Where:S\=Σx1−x¯2n−1

(3.5)

_Kurtosis Equations_

FCM\=∑i\=1nxi−x¯4n−1

(3.6)

Kurtosis\=FCMs4

(3.7)

_Excess Kurtosis_

K\=3n−12n−2n−3

(3.8)

EK\=nn+1n−1n−2n−3∑i\=1nxi−x¯s4−K

(3.9)

EK\=1n∑i\=1nxi−x¯s4−3

(3.10)

_Z Scores_

Z\=score−x¯s

(3.11)

## Chapter [5](./14-5._Concepts_for_Research_and_Inferential_Statistics.md): Reliability Measures

_Cronbach’s Alpha_

α\=kk−11−∑i\=1kσyi2σx2

(5.1)

kr20\=kk−11−∑jkpjqjσ2

(5.2)

_McDonald’s Omega_

ω\=∑b̂ik2∑b̂ik2+∑θ̂ik

(5.3)

_Spearman-Brown_

radj\=kr¯1+k−1r¯

(5.4)

## Chapter [6](./15-6._The_Chi-Square_Family_of_Tests.md): Chi-Square Test

_Chi-Square_

χ2\=∑fo−fe2fe

(6.1)

fe\=RowTotal×ColumnTotalGrandTotal

(6.2)

## Chapter [7](./16-7._Group_Comparisons_with_Scale_Dependent_Variables.md): T-tests, Effect Sizes and ANOVA and Non-parametrics

_t-test_

t\=X¯1−X¯2S12n1+S22n2

(7.1)

_Cohen’s d_

d\=X¯1−X¯2S12+S222

(7.2)

d\=X¯1−X¯2n1−1S12+n2−1S22n1+n2−2

(7.3)

_Mann Whitney U_

U\=n1n2n1n1+12−R1

U\=n1n2n2n2+12−R2

(7.4)

_ANOVA_

F\=MSBMSW\=SSBDFBSSWDFW\=∑X−X¯2−∑X−X¯i2k−1∑X−X¯i2n−k

(7.5)

_Tukey HSD (q)_

q\=X¯i−X¯jMSWn

(7.6)

q\=X¯i−X¯jMSW21ni+1nj

(7.7)

_Games-Howell t_

tij\=X¯i−X¯jSi2ni+Sj2nj

(7.8)

dfij\=Si2ni+Sj2nj2Si2nini−1+Sl2njnj−1

(7.9)

_Eta-squared_

η2\=SSBSSB+SSW

(7.10)

_Kruskal-Wallis_

H\=12NN+1∑i\=1kniRi2ni−3N+1

(7.11)

_Dunn’s Test_

Z\=Ri−RJNN+112−∑t12N−11ni+1nj

(7.12)

## Chapter [8](./17-8._One-Way_Dependent_Samples_and_Time_Series_Tests.md): Paired, and Repeated Measures (Dependent Samples) Tests with Non-parametric Tests

_Paired samples t-test_

t\=D¯SD/n

(8.1)

_Wilcoxon Signed Rank_

V\=∑D1\>1RI

(8.2)

_Repeated Measures ANOVA_

F\=MSCMSE\=SSBDFBSSEDFE\=SSBDFBSSW−SSSDFE\=∑j\=1knjX¯j−X¯2k−1∑i\=1n∑j\=1kXij−X¯j2−∑i\=1nX¯i−X¯2n−1k−1

(8.3)

Equation ([8.​4](./17-8._One-Way_Dependent_Samples_and_Time_Series_Tests.md#Equ4)) just continues ANOVA.

_Friedman’s Q_

Q\=12nkk+1∑j\=1kRj2−3nk+1

(8.5)

_Kendal’s W_

W\=12∑j\=1kRj2−3nk2+1nk2−kk−1

(8.6)

## Chapter [9](./18-9._Correlation_and_Regression_Modeling.md): Regression and Correlation Tests

_Pearson r_

r\=∑xi−x¯yi−y¯∑xi−x¯2∑yi−y¯2

(9.1)

_Spearman Rho_

ρ\=1−6∑di2nn2−1

(9.2)

_Adjusted R_2

Radj2\=1−1−R2xn−1n−p−1

(9.3)

_Logistic regression_

logPY\=11−PY\=1\=β0+β1X1+β2X2…βnXn

(9.4)

_Odds Ratios_

OR\=OddsatXn+1OddsatXn

(9.5)
