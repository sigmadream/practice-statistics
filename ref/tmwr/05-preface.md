# Preface

Welcome to *Tidy Modeling with R*! This book is a guide to using a collection of software in the R programming language for model building called tidymodels, and it has two main goals:

- First and foremost, this book provides a practical introduction to *how to use* these specific R packages to create models. We focus on a dialect of R called [the tidyverse](https://oreil.ly/xnx26) that is designed with a consistent, human-centered philosophy and demonstrate how the tidyverse and the tidymodels packages can be used to produce high quality statistical and machine learning models.

- Second, this book will show you how to *develop good methodology and statistical practices*. Whenever possible, our software, documentation, and other materials attempt to prevent common pitfalls.

In <a href="ch01.xhtml#software-modeling" data-type="xref">Chapter 1</a>, we outline a taxonomy for models and highlight what good software for modeling is like. The ideas and syntax of the tidyverse, which we introduce (or review) in <a href="ch02.xhtml#tidyverse" data-type="xref">Chapter 2</a>, are the basis for the tidymodels approach to these challenges of methodology and practice. <a href="ch03.xhtml#base-r" data-type="xref">Chapter 3</a> provides a quick tour of conventional base R modeling functions and summarizes the unmet needs in that area.

After that, this book is separated into parts, starting with the basics of modeling with tidy data principles. Chapters <a href="ch04.xhtml#ames" data-type="xref" data-xrefstyle="select:labelnumber">4</a>–<a href="ch09.xhtml#performance" data-type="xref" data-xrefstyle="select:labelnumber">9</a> introduce an example data set on house prices and demonstrate how to use the fundamental tidymodels packages: recipes, parsnip, workflows, yardstick, and others.

The next part of the book moves forward with more details on the process of creating an effective model. Chapters <a href="ch10.xhtml#resampling" data-type="xref" data-xrefstyle="select:labelnumber">10</a>–<a href="ch15.xhtml#workflow-sets" data-type="xref" data-xrefstyle="select:labelnumber">15</a> focus on creating good estimates of performance as well as tuning model hyperparameters.

Finally, the last section of this book, Chapters <a href="ch16.xhtml#dimensionality" data-type="xref" data-xrefstyle="select:labelnumber">16</a>–<a href="ch21.xhtml#inferential" data-type="xref" data-xrefstyle="select:labelnumber">21</a> cover other important topics for model building. We discuss more advanced feature engineering approaches like dimensionality reduction and encoding high-cardinality predictors, as well as how to answer questions about why a model makes certain predictions and when to trust your model predictions.

We do not assume that readers have extensive experience in model building and statistics. Some statistical knowledge is required, such as random sampling, variance, correlation, basic linear regression, and other topics that are usually found in a basic undergraduate statistics or data analysis course. We do assume that the reader is at least slightly familiar with dplyr, ggplot2, and the `%>%` “pipe” operator in R, and is interested in applying these tools to modeling. For users who don’t yet have this background R knowledge, we recommend books such as [*R for Data Science*](https://r4ds.had.co.nz) by Wickham and Grolemund (2016). Investigating and analyzing data is an important part of any model process.

This book is not intended to be a comprehensive reference on modeling techniques; we suggest other resources to learn more about the statistical methods themselves. For general background on the most common type of model, the linear model, we suggest Fox (2008). For predictive models, Kuhn and Johnson (2013) and Kuhn and Johnson (2020) are good resources. For machine learning methods, Goodfellow, Bengio, and Courville (2016) is an excellent (but formal) source of information. In some cases, we do describe the models we use in some detail, but in a way that is less mathematical, and hopefully more intuitive.

# Conventions Used in This Book

The following typographical conventions are used in this book:

*Italic*  
Indicates new terms, URLs, email addresses, filenames, and file extensions.

`Constant width`  
Used for program listings, as well as within paragraphs to refer to program elements such as variable or function names, databases, data types, environment variables, statements, and keywords.

**`Constant width bold`**  
Shows commands or other text that should be typed literally by the user.

*`Constant width italic`*  
Shows text that should be replaced with user-supplied values or by values determined by context.

###### Tip

This element signifies a tip or suggestion.

###### Note

This element signifies a general note.

###### Warning

This element indicates a warning or caution.

# Using Code Examples

Supplemental material (code examples, exercises, etc.) is available for download at [*https://github.com/tidymodels/TMwR*](https://github.com/tidymodels/TMwR). This book was written with [RStudio](https://oreil.ly/bcWV6) using [bookdown](http://bookdown.org) (Xie 2016). We generated all plots in this book using [ggplot2](https://oreil.ly/vEJBy) and its black and white theme (`theme_bw()`). An [online version](https://tmwr.org) of this book is available and will continue to evolve after publication of the physical book.

If you have a technical question or a problem using the code examples, please email to <a href="mailto:bookquestions@oreilly.com" class="email"><em>bookquestions@oreilly.com</em></a>.

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission. Incorporating a significant amount of example code from this book into your product’s documentation does require permission.

We appreciate, but generally do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “*Tidy Modeling with R* by Max Kuhn and Julia Silge (O’Reilly). Copyright 2022 Max Kuhn and Julia Silge, 978-1-492-09648-1.”

If you feel your use of code examples falls outside fair use or the permission given above, feel free to contact us at <a href="mailto:permissions@oreilly.com" class="email"><em>permissions@oreilly.com</em></a>.

This version of the book was built with: R version 4.1.3 (2022-03-10), [pandoc](https://pandoc.org) version 2.17.1.1, and the following packages:

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

# O’Reilly Online Learning

###### Note

For more than 40 years, <a href="https://oreilly.com" class="orm:hideurl"><em>O’Reilly Media</em></a> has provided technology and business training, knowledge, and insight to help companies succeed.

Our unique network of experts and innovators share their knowledge and expertise through books, articles, and our online learning platform. O’Reilly’s online learning platform gives you on-demand access to live training courses, in-depth learning paths, interactive coding environments, and a vast collection of text and video from O’Reilly and 200+ other publishers. For more information, visit <a href="https://oreilly.com" class="orm:hideurl"><em>https://oreilly.com</em></a>.

# How to Contact Us

Please address comments and questions concerning this book to the publisher:

- O’Reilly Media, Inc.
- 1005 Gravenstein Highway North
- Sebastopol, CA 95472
- 800-998-9938 (in the United States or Canada)
- 707-829-0515 (international or local)
- 707-829-0104 (fax)

We have a web page for this book, where we list errata, examples, and any additional information. You can access this page at [*https://oreil.ly/tidy-modeling-r*](https://oreil.ly/tidy-modeling-r).

Email <a href="mailto:bookquestions@oreilly.com" class="email"><em>bookquestions@oreilly.com</em></a> to comment or ask technical questions about this book.

For news and information about our books and courses, visit [*https://oreilly.com*](https://oreilly.com).

Find us on LinkedIn: [*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media).

Follow us on Twitter: [*https://twitter.com/oreillymedia*](https://twitter.com/oreillymedia).

Watch us on YouTube: [*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia).

# Acknowledgments

We are so thankful for the contributions, help, and perspectives of people who have supported us in this project. There are several we would like to thank in particular.

We would like to thank our RStudio colleagues on the tidymodels team (Davis Vaughan, Hannah Frick, Emil Hvitfeldt, and Simon Couch) as well as the rest of our coworkers on the RStudio open source team. Thank you to Desirée De Leon for the site design of the online work. We would also like to thank our technical reviewers, Chelsea Parlett-Pelleriti and Dan Simpson, for their detailed, insightful feedback that substantively improved this book, as well as our editors, Nicole Taché and Rita Fernando, for their perspective and guidance during the process of writing and publishing.

This book was written in the open, and multiple people contributed via pull requests or issues. Special thanks goes to the 38 people who contributed via GitHub pull requests (in alphabetical order by username): Aris Paschalidis (@arisp99), Brad Hill (@bradisbrad), Bryce Roney (@bryceroney), Cedric Batailler (@cedricbatailler), Ildikó Czeller (@czeildi), David Kane (@davidkane9), @DavZim, @DCharIAA, Emil Hvitfeldt (@EmilHvitfeldt), Emilio (@emilopezcano), Fgazzelloni (@Fgazzelloni), Hannah Frick (@hfrick), Hlynur (@hlynurhallgrims), Howard Baek (@howardbaek), Jae Yeon Kim (@jaeyk), Jonathan D. Trattner (@jdtrat), Jeffrey Girard (@jmgirard), John W. Pickering (@JohnPickering), Jon Harmon (@jonthegeek), Joseph B. Rickert (@joseph-rickert), Maximilian Rohde (@maxdrohde), Michael Grund (@michaelgrund), @MikeJohnPage, Mine Cetinkaya-Rundel (@mine-cetinkaya-rundel), Mohammed Hamdy (@mmhamdy), @nattalides, Y. Yu (@PursuitOfDataScience), Riaz Hedayati (@riazhedayati), Rob Wiederstein (@RobWiederstein), Scott (@scottyd22), Simon Schölzel (@simonschoe), Simon Sayz (@tagasimon), @thrkng, Tanner Stauss (@tmstauss), Tony ElHabr (@tonyelhabr), Dmitry Zotikov (@x1o), Xiaochi (@xiaochi-liu), and Zach Bogart (@zachbogart).
