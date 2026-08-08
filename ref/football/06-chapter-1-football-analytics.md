# Chapter 1. Football Analytics

American football (also known as *gridiron football* or *North American football* and henceforth simply called *football*) is undergoing a drastic shift toward the quantitative. Prior to the last half of a decade or so, most of football analytics was confined to a few seminal pieces of work. Arguably the earliest example of analytics being used in football occurred when former Brigham Young University, Chicago Bears, Cincinnati Bengals, and San Diego Chargers quarterback Virgil Carter created the notion of an *expected point* as coauthor of the 1971 paper [“Technical Note: Operations Research in Football”](https://oreil.ly/CU0I6) before he teamed with the legendary Bill Walsh as the first quarterback to execute what is now known as the *West Coast offense*.

The idea of an *expected point* is incredibly important in football, as the game by its very nature is discrete: a collection of a finite number of plays (also called *downs*) that require the offense to go a certain distance (in yards) before having to surrender the ball to the opposing team. If the line to gain is the opponent’s end zone, the offense scores a touchdown, which is worth, on average, seven points after a post-touchdown conversion. Hence, the *expected point* provides an estimated, or *expected* value for the number of *points* you would expect a team to score given the current game situation on that drive.

Football statistics have largely been confined to offensive players, and have been doled out in the currency of yards gained and touchdowns scored. The problem with this is obvious. If a player catches a pass to gain 7 yards, but 8 are required to get a first down or a touchdown, the player did not gain a first down. Conversely, if a player gains 5 yards, when 5 are required to get a first down or a touchdown, the player gained a first down. Hence, “enough” yards can be better than “more” yards depending on the context of the play. As a second example, if it takes a team two plays to travel 70 yards to score a touchdown, with one player gaining the first 65 yards and the second gaining the final 5, why should the second player get all the credit for the score?

In 1988, Bob Carroll, Pete Palmer, and John Thorn wrote *The Hidden Game of Football* (Grand Central Publishing), which further explored the notions of expected points. In 2007, Brian Burke, who was a US Navy pilot before creating the Advanced Football Analytics website ([*http://www.advancedfootballanalytics.com*](http://www.advancedfootballanalytics.com/)), formulated the expected-points and expected-points-added approach, along with building a win probability model responsible for some key insights, including the [4th Down Bot](https://oreil.ly/Y9lEV) at the *New York Times* website. Players may be evaluated by the number of expected points or win probability points added to their teams when those players did things like throw or catch passes, run the ball, or sack the quarterback.

The work of Burke inspired the open source work of Ron Yurko, Sam Ventura, and Max Horowitz of Carnegie Mellon University. The trio built `nflscrapR`, an R package that scraped NFL play-by-play data. The `nflscrapR` package was built to display their own versions of *expected points added* (*EPA*) and *win probability* (*WP*) models. Using this framework, they also replicated the famous *wins above replacement* (*WAR*) framework from baseball for quarterbacks, running backs, and wide receivers, which was published in 2018. This work was later extended using different data and methods by Eric and his collaborator George Chahouri in 2020. Eric’s version of WAR, and its analogous model for college football, are used throughout the industry to this day.

The `nflscrapR` package served as a catalyst for the popularization of modern tools that use data to study football, most of which use a framework that will be replicated constantly throughout this book. The process of building an *expectation* for an outcome—in the form of points, completion percentage, rushing yards, draft-pick outcome, and many more—and measuring players or teams via the *residual* (that is, the difference between the value expected by the model and the observed value) is a process that transcends football. In soccer, for example, expected goals (xG) are the cornerstone metric upon which players and clubs are measured in the sport known as [“the Beautiful Game”](https://oreil.ly/6ol1p). And shot quality—the expected rate at which a shot is made in basketball—is a ubiquitous measure for players and teams on the hardwood. The features that go into these models, and the forms that they take, are the subject of constant research whose surface we will scratch in this book.

The rise of tools like `nflscrapR` allowed more people to show their analytical skills and flourish in the public sphere. Analysts were hired based on their tweets on Twitter because of their ingenious approaches to measuring team performance. Decisions to punt or go for it on a fourth down were evaluated by how they affected the team’s win probability. Ben Baldwin and Sebastian Carl created a spin-off R package, `nflfastR`. This package updated the models of Yurko, Ventura, and Horowitz, along with adding many of their own—models that we’ll use in this book. More recently, the data contained in the `nflfastR` package has been cloned into Python via the `nfl_data_py` package by Cooper Adams.

We hope that this book will give you the basic tools to approach some of the initial problems in football analytics and will serve as a jumping-off point for future work.

###### Tip

People looking for the cutting edge of sports analytics, including football, may want to check out the MIT Sloan Sports Analytics Conference. Since its founding in 2006, Sloan has emerged as a leading venue for the presentation of new tools for football (and other) analytics. Other, more accessible conferences, like the Carnegie Mellon Sports Analytics Conference and New England Statistics Symposium, are fantastic places for students and practitioners to present their work. Most of these conferences have hackathons for people looking to make an impression on the industry.

# Baseball Has the Three True Outcomes: Does Football?

Baseball pioneered the use of quantitative metrics, and the creation of the Society for American Baseball Research (SABR) led to the term *sabermetrics* to describe baseball analysis. Because of this long history, we start by looking at the metrics commonly used in baseball—specifically, the *three true outcomes*. One of the reasons the game of baseball has trended toward the *three true outcomes* (walks, strikeouts, and home runs) is that they were the easiest to predict from one season to the next. What batted balls did when they were in play was noisier and was the source of much of the variance in perceived play from one year to the next. The three true outcomes of baseball have also been the source for more elaborate data-collection methods and subsequent analysis in an attempt to tease additional signals from batted-ball data.

Stability analysis is a cornerstone of team and player evaluation. *Stable* production is sticky or repeatable and is the kind of production decision-makers should want to buy into year in and year out. *Stability analysis* therefore examines whether something is stable—in our case, football observation and model outputs, and you will use this analysis in <a href="ch02.html#sec-eda-stable" data-type="xref">“Player-Level Stability of Passing Yards per Attempt”</a>, <a href="ch03.html#sec-RYOE-stable" data-type="xref">“Is RYOE a Better Metric?”</a>, <a href="ch04.html#sec-ryoe_mr" data-type="xref">“Analyzing RYOE”</a>, and <a href="ch05.html#sec-cpoe-stable" data-type="xref">“Is CPOE More Stable Than Completion Percentage?”</a>. On the other hand, how well a team or player does in *high-leverage situations* (plays that have greater effects on the outcome of the game, such as converting third downs) can have an outsized impact on win-loss record or eventual playoff fate, but if it doesn’t help us predict what we want year in and year out, it might be better to ignore, or sell such things to other decision-makers.

Using play-by-play data from `nflfastR`, <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a> shows you how to slice and dice football passing data into subsets that partition a player’s performance into stable and unstable production. Through exploratory data analysis techniques, you can see whether any players break the mold and what to do with them. We preview how this work can aid in the process of feature engineering for prediction in <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a>.

# Do Running Backs Matter?

For most of the history of football, the best players played running back (in fact, early football didn’t include the forward pass until President Teddy Roosevelt worked with college football to introduce passing to make the game safer in 1906). The importance of the running back used to be an accepted truism across all levels of football, until the forward pass became an integral part of the game. Following the forward pass, rule and technology changes—along with Carter (mentioned earlier in this chapter) and his quarterbacks coach, Walsh—made throwing the football more efficient relative to running the football.

Many of our childhood memories from the 1990s revolve around Emmitt Smith and Barry Sanders trading the privilege of being the NFL rushing champion every other year. College football fans from the 1980s may remember Herschel Walker giving way to Bo Jackson in the Southeastern Conference (SEC). Even many younger fans from the 2000s and 2010s can still remember Adrian Peterson earning the last nonquarterback most valuable player (MVP) award. During the 2012 season, he rushed for over 2,000 yards while carrying an otherwise-bad Minnesota Vikings team to the playoffs.

However, the current prevailing wisdom among football analytics folks is that the running back position does not matter as much as other positions. This is for a few reasons. First, running the football is not as efficient as passing. This is plain to see with simple analyses using yards per play, but also through more advanced means like EPA. Even the worst passing plays usually produce, on average, more yards or expected points per play than running.

Second, differences in the actual player running the ball does not elicit the kind of change in rushing production that similar differences do for quarterbacks, wide receivers, or offensive or defensive linemen. In other words, additional resources used to pay for the services of this running back over that running back are probably not worth it, especially if those resources can be used on other positions. The marketplace that is the NFL has provided additional evidence that this is true, as we have seen running back salaries and draft capital used on the position decline to lows not previously seen.

This didn’t keep the New York Giants from using the second-overall pick in the 2018 NFL Draft on Pennsylvania State University’s Saquon Barkley, which was met with jeers from the analytics community, and a counter from Giants General Manager Dave Gettleman. In a post-draft press conference for the ages, Gettleman, sitting next to reams of bound paper, made fun of the analytics jabs toward his pick by mimicking a person typing furiously on a typewriter.

Chapters <a href="ch03.html#sec-lm-ryoa" data-type="xref" data-xrefstyle="select:labelnumber">3</a> and <a href="ch04.html#sec-mr-ryoe2" data-type="xref" data-xrefstyle="select:labelnumber">4</a> look at techniques for controlling play-by-play rushing data for a situation to see how much of the variability in rushing success has to do with the player running the ball.

# How Data Can Help Us Contextualize Passing Statistics

As we’ve stated previously, the passing game dominates football, and in <a href="app02.html#sec-ssdw-pass" data-type="xref">Appendix B</a>, we show you how to examine the basics of passing game data. In recent years, analysts have taken a deeper look into what constitutes accuracy at the quarterback position because raw completion percentage numbers, even among quarterbacks who aren’t considered elite players, have skyrocketed. The work of Josh Hermsmeyer with the Baltimore Ravens and later [FiveThirtyEight](https://oreil.ly/IYs71) established the significance of *air yards*, which is the distance traveled by the pass from the line of scrimmage to the intended receiver.

While Hermsmeyer’s initial research was in the fantasy football space, it spawned a significant amount of basic research into the passing game, giving rise to metrics like *completion percentage over expected* (*CPOE*), which is one of the most predictive quarterback metrics about quarterback quality available today.

In <a href="ch05.html#sec-lr-pass" data-type="xref">Chapter 5</a>, we introduce generalized linear models in the form of logistic regression. You’ll use this to estimate the completion probability of a pass, given multiple situational factors that affect a throw’s expected success. You’ll then look at a player’s *residuals* (that is, how well a player actually performs compared to the model’s prediction for that performance) and see whether there is more or less stability in the residuals—the CPOE—than in actual completion percentage.

# Can You Beat the Odds?

In 2018, the Professional and Amateur Sports Protection Act of 1992 (PASPA), which had banned sports betting in the United States (outside of Nevada), was overturned by the US Supreme Court. This court decision opened the floodgates for states to make legal what many people were already doing illegally: betting on football.

The difficult thing about sports betting is the *house advantage*—referred to as the *vigorish*, or *vig*—which makes it so that a bettor has to win more than 50% of their bets to break even. Thus, a cost exists for simply playing the game that needs to be overcome in order to beat the sportsbook (or simply the *book* for short).

American football is the largest gambling market in North America. Most successful sports bettors in this market use some form of analytics to overcome this house advantage. <a href="ch06.html#sec-pos-td" data-type="xref">Chapter 6</a> examines the passing touchdowns per game prop market, which shows how a bettor would arrive at an internal price for such a market and compare it to the market price.

# Do Teams Beat the Draft?

Owners, fans, and the broader NFL community evaluate coaches and general managers based on the quality of talent that they bring to their team from one year to the next. One complaint against New England Patriots Coach Bill Belichick, maybe the best nonplayer coach in the history of the NFL, is that he has not drafted well in recent seasons. Has that been a sequence of fundamental missteps or just a run of bad luck?

One argument in support of coaches such as Belichick may be “Well, they are always drafting in the back of the draft, since they are usually a good team.” Luckily, one can use math to control for this and to see if we can reject the hypothesis that all front offices are equally good at drafting after accounting for draft capital used. *Draft capital* comprises the resources used during the NFL Draft—notably, the number of picks, pick rounds, and pick numbers.

In <a href="ch07.html#sec-webscrap-draft" data-type="xref">Chapter 7</a>, we scrape publicly available draft data and test the hypothesis that all front offices are equally good at drafting after accounting for draft capital used, with surprising results. In <a href="ch08.html#sec-pca-clus" data-type="xref">Chapter 8</a>, we scrape publicly available NFL Scouting Combine data and use dimension-reduction tools and clustering to see how groups of players emerge.

# Tools for Football Analytics

Football analytics, and more broadly, data science, require a diverse set of tools. Successful practitioners in these fields require an understanding of these tools. Statistical programming languages, like Python and R, are a backbone of our data science toolbox. These languages allow us to clean our datasets, conduct our analyses, and readily reuse our methods.

Although many people commonly use spreadsheets (such as Microsoft Excel or Google Sheets) for data cleaning and analysis, we find spreadsheets do not scale well. For example, when working with large datasets containing tracking data, which can include thousands of rows of data per play, spreadsheets simply are not up to the task. Likewise, people commonly use business intelligence (BI) tools such as Microsoft Power BI and Tableau because of their power and ability to scale. But these tools tend to focus on point-and-click methods and require licenses, especially for commercial use.

Programming languages also allow for easy reuse because copying and pasting formulas in spreadsheets can be tedious and error prone. Lastly, spreadsheets (and, more broadly, point-and-click tools) allow undocumented errors. For example, spreadsheets do not have a way to catch a copying and pasting mistake. Furthermore, modern data science tools allow code, data, and results to be blended together in easy-to-use interfaces. Common languages include Python, R, Julia, MATLAB, and SAS. Additional languages continue to appear as computer science advances.

As practitioners of data science, we use R and Python daily for our work, which has collectively spanned the space of applied mathematics, applied statistics, theoretical ecology and, of course, football analytics. Of the languages listed previously, Python and R offer the benefit of larger user bases (and hence likely contain the tools and models we need). Both R and Python (as well as Julia) are open source. As of this writing, Julia does not have the user base of R or Python, but it may either be the cutting edge of statistical computing, a dead end that fizzles out, or possibly both.

*Open source* means two types of freedom. First, anybody can access all the code in the language, like free speech. This allows volunteers to help improve the language, such as ensuring that users can debug the code and extend the language through add-on packages (like the `nflfastR` package in R or the nfl_data_py package in Python). Second, open source also offers the benefit of being free to use for users, like free drinks. Hence users do not need to pay thousands of dollars annually in licensing fees. We were initially trained in R but have learned Python over the course of our jobs. Either language is well suited for football analytics (and sports analytics in general).

###### Note

<a href="app01.html#sec-appendix-1" data-type="xref">Appendix A</a> includes instructions for obtaining R and Python for those of you who do not currently have access to these languages. This includes either downloading and installing the programs or using web-hosted resources. The appendix also describes programs to help you more easily work with these languages, such as editors and integrated development environments (IDEs).

We encourage you to pick one language for your work with this book and learn that language well. Learning a second programming language will be easier if you understand the programming concepts behind a first language. Then you can relate the concepts back to your understanding of your original computer language.

###### Tip

For readers who want to learn the basics of programming before proceeding with our book, we recommend Al Sweigart’s *Invent Your Own Computer Games with Python*, 4th edition (No Starch Press, 2016) or Garrett Grolemund’s <a href="https://learning.oreilly.com/library/view/hands-on-programming-with/9781449359089/" class="orm:hideurl"><em>Hands-On Programming with R</em></a> (O’Reilly, 2014). Either resource will hold your hand to help you learn the basics of programming.

Although many people pick favorite languages and sometimes argue about which coding language is better (similar to Coke versus Pepsi or Ford versus General Motors), we have seen both R and Python used in production and also used with large data and complex models. For example, we have used R with 100 GB files on servers with sufficient memory. Both of us began our careers coding almost exclusively in R but have learned to use Python when the situation has called for it. Furthermore, the tools often have complementary roles, especially for advanced methods, and knowing both languages lets you have options for problems you may encounter.

###### Tip

When picking a language, we suggest you *use what your friends use*. If all your friends speak Spanish, communicating with them if you learn Spanish will probably be easier as well. You can then teach them your native language too. Likewise, the same holds for programming: your friends can then help you debug and troubleshoot. If you still need help deciding, open up both languages and play around for a little bit. See which one you like better. Personally, we like R when working with data, because of R’s data manipulation tools, and Python when building and deploying new models because of Python’s cleaner syntax for writing functions.

# First Steps in Python and R

###### Tip

If you are familiar with R and Python, you’ll still benefit from skimming this section to see how we teach a tool you are familiar with.

Opening a computer terminal may be intimidating for many people. For example, many of our friends and family will walk by our computers, see code up on the screens, and immediately turn their heads in disgust (Richard’s dad) or fear (most other people). However, terminals are quite powerful and allow more to be done with less, once you learn the language. This section will help you get started using Python or R.

The first step for using R or Python is either to install it on your computer or use a web-based version of the program. Various options exist for installing or otherwise accessing Python and R and then using them on your computer. <a href="app01.html#sec-appendix-1" data-type="xref">Appendix A</a> contains steps for this as well as installation options.

###### Note

People, like Richard, who follow the Green Bay Packers are commonly called *Cheeseheads*. Likewise, people who use Python are commonly called *Pythonistas*, and people who use R are commonly called *useRs*.

Once you have you access to R or Python, you have an expensive graphing calculator (for example, your \$1,000 laptop). In fact, both Eric and Richard, in lieu of using an actual calculator, will often calculate silly things like point spreads or totals in the console if in need of a quick calculation. Let’s see some things you can do. Type **`2 + 2`** in either the Python or R console:

```
2 + 2
```

Which results in:

4

###### Note

People use comments to leave notes to themselves and others in code. Both Python and R use the \# symbol for comments (the *pound symbol* for the authors or *hashtag* for younger readers). Comments are text (within code) that the computer does not read but that help humans to understand the code. In this book, we will use two comment symbols to tell you that a code block is Python (## Python) or R (## R)

You may also save numbers as variables. In Python, you could define `z` to be `2` and then reuse `z` and divide by 3:

```
## Python
z = 2
z / 3
```

Resulting in:

0.6666666666666666

###### Tip

In R, either `<-` or `=` may be used to create variables. We use `<-` for two reasons. First, in this book this helps you see the difference between R and Python code. Second, we use this style in our day-to-day programming as well. <a href="ch09.html#sec-advanced-tool" data-type="xref">Chapter 9</a> discusses code styles more. Regardless of which operator you use, be consistent with your programming style in any language. Your future self (and others who read your code) will thank you.

In R, you can also define `z` to be `2` and then reuse `z` and divide by 3:

```
## R
z <- 2
z / 3
```

Resulting in:

\[1\] 0.6666667

###### Note

Python and R format outputs differently. Python does not round up and includes more digits. Conversely, R shows fewer digits and rounds up.

# Example Data: Who Throws Deep?

Now that you have seen some basics in R, let’s dive into an example with football data. You will use the `nflfastR` data for many of the examples in this book. This data may be installed as an R package or as the Python package `nfl_data_py`. Specifically, we will explore the broad (and overly simple) question “Who were the most aggressive quarterbacks in 2021?” We will start off introducing the package using R because the data originated with R.

###### Note

Both Python and R have flourished because they readily allow add-on packages. Conda exists as one tool for managing these add-ons. <a href="ch09.html#sec-advanced-tool" data-type="xref">Chapter 9</a> and <a href="app01.html#sec-appendix-1" data-type="xref">Appendix A</a> discuss these add-ons in greater detail. In general, you can install packages in Python by typing **`pip install `*`package name`*** or **`conda install `*`package name`*** in the terminal (such as the bash shell on Linux, Zsh shell on macOS, or command prompt on Microsoft Windows). Sometimes you will need to use `pip3`, depending on your operating system’s configuration, if you are using the `pip` package manager system. For a concrete example, to install the `seaborn` package, you could type **`pip install seaborn`** in your terminal. In general, packages in R can be installed by opening R and then typing **`install.packages("`*`package name`*`")`**. For example, to install the `tidyverse` collection of packages, open R and run **`install.packages("tidyverse")`**.

## nflfastR in R

Starting with R, install the `nflfastR` package:

```
## R
install.packages("nflfastR")
```

###### Tip

Using single quotation marks around a name, such as `'x'`, or double quotes, such as `"x"`, are both acceptable to languages such as Python or R. Make sure the opening and closing quotes match. For example, `'x"` would not be acceptable. You may use both single and double quotes to place quotes inside of quotes. For example, in a figure caption, you might write, `"Panthers' points earned"` or `'Air temperature ("true temperature")'`. Or in Python, you can use a combination of quotes later for inputs such as `"team == 'GB'"` because you’ll need to nest quotes inside of quotes.

Next, load this package as well as the `tidyverse`, which gives you tools to manipulate and plot the data:

```
## R
library("tidyverse")
library("nflfastR")
```

###### Note

Base R contains dataframes as `data.frame()`. We use tibbles from the tidyverse instead, because these print nicer to screens and include other useful features. Many users consider base R’s `data.frame()` to be a legacy object, although you will likely see these objects when looking at help files and examples on the web. Lastly, you might see the `data.table` package in R. The `data.table` extension of dataframes is similar to a tibble and works better with larger data (for example, 10 GB or 100 GB files) and has a more compact coding syntax, but it comes with the trade-off of being less user-friendly compared to tibbles. In our own work, we use a data.table rather than a tibble or data.frame when we need high performance at the trade-off of code readability.

Once you’ve loaded the packages, you need to load the data from each play, or the *play-by-play* (pbp) data, for the 2021 season. Use the `load_pbp()` function from `nflfastR` and call the data `pbp_r` (the `_r` ending helps you tell that the code is from an R example in this book):

```
## R
pbp_r <- load_pbp(2021)
```

###### Note

We generally include `_py` in the name of Python dataframes and `_r` in the names of R dataframes to help you identify the language for various code objects.

After loading the data as `pbp_r`, pass (or *pipe*) the data along to be *filtered* by using `|>`. Use the `filter()` function to select only data where passing plays occurred (`play_type == "pass"`) and where `air_yards` are not missing, or `NA` in R syntax (in plain English, the pass had a recorded depth). <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a>, <a href="app02.html#sec-ssdw-pass" data-type="xref">Appendix B</a>, and <a href="app03.html#sec-app-dw" data-type="xref">Appendix C</a> cover data manipulation more, and most examples in this book use data wrangling to format data. So right now, simply type this code. You can probably figure out what the code is doing, but don’t worry about understanding it too much:

```
## R
pbp_r_p <-
    pbp_r |>
    filter(play_type == 'pass' & !is.na(air_yards))
```

Now you’ll look at the average depth of target (aDOT), or mean air yards per pass, for every quarterback in the NFL in 2021 who threw 100 or more passes with a designated depth. To avoid multiple players who have the same name, which happens more than you’d think, you’ll summarize by both player ID and player name.

First, *group by* both the `passer_id` and `passer`. Then *summarize* to calculate the number of plays (`n()`) and mean air yards per pass (`adot`) per player. Also, *filter* to include only players with 100 or more plays and to remove any rows without a passer name (specifically, those with missing or `NA` values).

With this and the previous example commands, the function `is.na(passer)` checks whether value in the `passer` column has the value `NA` and returns `TRUE` for columns with an `NA` value. <a href="app02.html#sec-ssdw-pass" data-type="xref">Appendix B</a> covers this logic and terminology in greater detail. Next, an exclamation point (`!`) turns this expression into the opposite of *not missing value*, so that you keep cells with a value. As an aside, we, the authors, find the use of double negatives confusing as well. Lastly, arrange by the `adot` values and then print all (or infinity, `Inf`) values:

```
## R
pbp_r_p |>
    group_by(passer_id, passer) |>
    summarize(n = n(), adot = mean(air_yards)) |>
    filter(n >= 100 & !is.na(passer)) |>
    arrange(-adot) |>
    print(n = Inf)
```

Resulting in:

\[Entire table\] A tibble: 42 × 4 \# Groups: passer_id \[42\] passer_id passer n adot \<chr\> \<chr\> \<int\> \<dbl\> 1 00-0035704 D.Lock 110 10.2 2 00-0029263 R.Wilson 400 9.89 3 00-0036945 J.Fields 268 9.84 4 00-0034796 L.Jackson 378 9.34 5 00-0036389 J.Hurts 473 9.19 6 00-0034855 B.Mayfield 416 8.78 7 00-0026498 M.Stafford 740 8.51 8 00-0031503 J.Winston 161 8.32 9 00-0029604 K.Cousins 556 8.23 10 00-0034857 J.Allen 708 8.22 11 00-0031280 D.Carr 676 8.13 12 00-0031237 T.Bridgewater 426 8.04 13 00-0019596 T.Brady 808 7.94 14 00-0035228 K.Murray 515 7.94 15 00-0036971 T.Lawrence 598 7.91 16 00-0036972 M.Jones 557 7.90 17 00-0033077 D.Prescott 638 7.81 18 00-0036442 J.Burrow 659 7.75 19 00-0023459 A.Rodgers 556 7.73 20 00-0031800 T.Heinicke 491 7.69 21 00-0035993 T.Huntley 185 7.68 22 00-0032950 C.Wentz 516 7.64 23 00-0029701 R.Tannehill 554 7.61 24 00-0037013 Z.Wilson 382 7.57 25 00-0036355 J.Herbert 671 7.55 26 00-0033119 J.Brissett 224 7.55 27 00-0033357 T.Hill 132 7.44 28 00-0028118 T.Taylor 149 7.43 29 00-0030520 M.Glennon 164 7.38 30 00-0035710 D.Jones 360 7.34 31 00-0036898 D.Mills 392 7.32 32 00-0031345 J.Garoppolo 511 7.31 33 00-0034869 S.Darnold 405 7.26 34 00-0026143 M.Ryan 559 7.16 35 00-0032156 T.Siemian 187 7.13 36 00-0036212 T.Tagovailoa 387 7.10 37 00-0033873 P.Mahomes 780 7.08 38 00-0027973 A.Dalton 235 6.99 39 00-0027939 C.Newton 126 6.97 40 00-0022924 B.Roethlisberger 647 6.76 41 00-0033106 J.Goff 489 6.44 42 00-0034401 M.White 132 5.89

The `adot` value, a commonly used measure of quarterback aggressiveness, gives a quantitative approach to rank quarterbacks by their aggression, as measured by mean air yards per pass (can you think of other ways to measure aggressiveness that pass depth alone leaves out?). Look at the results and think, do they make sense to you, or are you surprised, given your personal opinions of quarterbacks?

###### Warning

If you get unexpected errors on any of the commands, double-check that you are in the correct language environment. You may be trying to use Python in the R environment or R in the Python environment.

## nfl_data_py in Python

In Python, the `nfl_data_py` package by Cooper Adams exists as a clone of the R `nflfastR` package for data. To use the data from this package, first import the `pandas` package with the alias (or short nickname) `pd` for working with data and import the `nfl_data_py` package as `nfl`:

```
## Python
import pandas as pd
import nfl_data_py as nfl
```

Next, tell Python to import the data for 2021 (<a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a> shows how to import multiple years). Note that you need to include the year in a Python list as `[2021]`:

```
## Python
pbp_py = nfl.import_pbp_data([2021])
```

As with the R code, filter the data in Python (pandas calls filtering a `query`). Python allows you to readily pass the filter criteria (`filter_crit`) into `query()` as an object, and we have you do this to save space line space. Then *group by* `passer_id` and `passer` before *aggregating* the data by using a Python dictionary (`dict()`, or `{}` for short) with the `.agg()` function:

```
## Python
filter_crit = 'play_type == "pass" & air_yards.notnull()'

pbp_py_p = (
    pbp_py.query(filter_crit)
    .groupby(["passer_id", "passer"])
    .agg({"air_yards": ["count", "mean"]})
)
```

The pandas package also requires reformatting the column heads via a `list()` function and changing the header from being two rows to a single row via map(). Next, print the outputs after sorting by the mean of the air yards via the `query()` function (`to_string()` allows all the outputs to be printed):

```
## Python
pbp_py_p.columns = list(map("_".join, pbp_py_p.columns.values))
sort_crit = "air_yards_count > 100"
print(
    pbp_py_p.query(sort_crit)\
    .sort_values(by="air_yards_mean", ascending=[False])\
    .to_string()
)
```

This results in:

air_yards_count air_yards_mean passer_id passer 00-0035704 D.Lock 110 10.154545 00-0029263 R.Wilson 400 9.887500 00-0036945 J.Fields 268 9.835821 00-0034796 L.Jackson 378 9.341270 00-0036389 J.Hurts 473 9.190275 00-0034855 B.Mayfield 416 8.776442 00-0026498 M.Stafford 740 8.508108 00-0031503 J.Winston 161 8.322981 00-0029604 K.Cousins 556 8.228417 00-0034857 J.Allen 708 8.224576 00-0031280 D.Carr 676 8.128698 00-0031237 T.Bridgewater 426 8.037559 00-0019596 T.Brady 808 7.941832 00-0035228 K.Murray 515 7.941748 00-0036971 T.Lawrence 598 7.913043 00-0036972 M.Jones 557 7.901257 00-0033077 D.Prescott 638 7.811912 00-0036442 J.Burrow 659 7.745068 00-0023459 A.Rodgers 556 7.730216 00-0031800 T.Heinicke 491 7.692464 00-0035993 T.Huntley 185 7.675676 00-0032950 C.Wentz 516 7.641473 00-0029701 R.Tannehill 554 7.606498 00-0037013 Z.Wilson 382 7.565445 00-0036355 J.Herbert 671 7.554396 00-0033119 J.Brissett 224 7.549107 00-0033357 T.Hill 132 7.439394 00-0028118 T.Taylor 149 7.429530 00-0030520 M.Glennon 164 7.378049 00-0035710 D.Jones 360 7.344444 00-0036898 D.Mills 392 7.318878 00-0031345 J.Garoppolo 511 7.305284 00-0034869 S.Darnold 405 7.259259 00-0026143 M.Ryan 559 7.159213 00-0032156 T.Siemian 187 7.133690 00-0036212 T.Tagovailoa 387 7.103359 00-0033873 P.Mahomes 780 7.075641 00-0027973 A.Dalton 235 6.987234 00-0027939 C.Newton 126 6.968254 00-0022924 B.Roethlisberger 647 6.761978 00-0033106 J.Goff 489 6.441718 00-0034401 M.White 132 5.886364

Hopefully, this chapter whet your appetite for using math to examine football data. We glossed over some of the many topics you will learn about in future chapters such as data sorting, summarizing data, and cleaning data. You have also had a chance to compare Python and R for basic tasks for working with data, including modeling. <a href="app02.html#sec-ssdw-pass" data-type="xref">Appendix B</a> also dives deeper into the air-yards data to cover basic statistics and data wrangling.

# Data Science Tools Used in This Chapter

This chapter covered the following topics:

- Obtaining data from one season by using the `nflfastR` package either directly in R or via the `nfl_data_py` package in Python

- Using `filter()` in R or `query()` in Python to select and create a subset of data for analysis

- Using `summarize()` to group data in R with the help of `group_by()`, and aggregating (`agg()`) data by groups in Python with the help of `groupby()`

- Printing dataframe outputs to your screen to help you look at data

- Removing missing data by using `is.na()` in R or `notnull()` in Python

# Suggested Readings

If you get really interested in analytics without the programming, here are some sources we read to develop our philosophy and strategies for football analytics:

- *The Hidden Game of Football: A Revolutionary Approach to the Game and Its Statistics* by Bob Carroll et al. (University of Chicago Press, 2023). Originally published in 1988, this cult classic introduces the numerous ideas that were later formulated into the cornerstone of what has become modern football analytics.

- *Moneyball: The Art of Winning an Unfair Game* by Michael Lewis (W.W. Norton & Company, 2003). Lewis describes the rise of analytics in baseball and shows how the stage was set for other sports. The book helps us think about how modeling and data can help guide sports. A movie was made of this book as well.

- *The Signal and the Noise: Why So Many Predictions Fail, but Some Don’t* by Nate Silver (Penguin, 2012). Silver describes why models work in some instances and fail in others. He draws upon his experience with poker, baseball analytics, and running the political prediction website FiveThirtyEight. The book does a good job of showing how to think quantitatively for big-picture problems without getting bogged down by the details.

Lastly, we encourage you to read the [documentation for the `nflfastR` package](https://www.nflfastr.com). Diving into this package will help you better understand much of the data used in this book.
