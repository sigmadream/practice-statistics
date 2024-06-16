# Appendix B. Summary Statistics and Data Wrangling: Passing the Ball

This appendix contains materials to help you understand some basic statistics. If the topics are new to you, we encourage you to read this material after <a href="ch01.html#sec-introduction" data-type="xref">Chapter 1</a> or <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a> and before you dive too far into the book.

In <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a>, you looked at quarterback performance at different pass depths in an effort to understand which aspect of play was fundamental to performance and which aspect was noisier, possibly leading you astray as you aimed to make predictions about future performance. You were lucky enough to have the data more or less in ready-made form for you to perform this analysis. You did have to create your own variable for analysis, but such *data wrangling* was minimal at best.

Sports analytics generally, and football analytics specifically, are still in their early stages of development. As such, datasets may not always be the cleanest, or tidy. *Tidy* datasets are usually in a table form that computers can easily read and humans can easily understand. Furthermore, data analysis in any field (and football analytics is no different) often requires datasets that were created for different purposes. This is where data wrangling can come in handy. Because so many people have had to clean up messy data, many terms exist in this field. Some synonyms for *data wrangling* include *data cleaning*, *data manipulating*, *data mutating*, *shaping*, *tidying*, and *munging*. More specifically, these terms describe the process of using a programming language such as Python or R to update datasets to meet your needs.

###### Note

*Tidy* data has two definitions. Broadly, it refers to clean data. More formally, Hadley Wickham defines the term in a paper titled [“Tidy Data.”](https://oreil.ly/tidy) Wickham defines *tidy data* as having a specific structure: “(1) Each variable forms a column, (2) each observation forms a row, and (3) each type of observational unit forms a table.”

During the course of our careers, we have found that data wrangling takes the most time for our projects. For example, one of our bosses once pinged us on Google chat because he was having trouble fitting a new model. His problem turned out to not be the model, but rather data formatting. Figuring out how to format the data to work with the model took about 30 minutes. However, running the new model took only about 30 seconds in R after we figured out the data structure issue he was having.

###### Tip

Computer tools are ever changing, and data wrangling is no exception. During the course of his career, Richard has had to learn four tools for data wrangling: base R (around 2007), `data.table` in R (around 2012), the `tidyverse` in R (around 2015), and `pandas` in Python (around 2020). Hence, you will also likely need to update your skill set for the tools taught in this book. However, the fundamentals never change as long as you understand the basics.

Programming languages like Python or R are our most-effective tools for making our data to usable. The languages allow scripting, thereby letting us track our changes and see what we did, including whether we introduced any errors into our data.

Many people like to use spreadsheet programs such as Microsoft Excel or Google Sheets for data manipulation. Unfortunately, these programs do not easily keep track of changes. Likewise, hand-editing data does not scale, so as the size of the problem becomes too large—such as when you are working with player *tracking data*, which has one record for every player, anywhere from 10 to 25 times per second per play—you will not be able to quickly and efficiently build a workflow that works in a spreadsheet environment. Thus, editing one or two files by hand is easy to do with Excel, but editing one or two thousand files by hand is not easy.

Conversely, programming languages, such as Python or R, readily scale. For example, if you have to format data after each week’s games, Python or R could easily be used as part of a data pipeline, but spreadsheet-based data wrangling would be difficult to automate into a data pipeline.

###### Note

A *data pipeline* is the flow of data as it moves from one location to another and undergoes changes such as formatting. For example, when Eric worked at PFF, a data pipeline might take weekly numbers, format the numbers, run a model, and then generate a report. In computer science, a *pipe* operator refers to passing the outputs from one function directly to another.

That being said, we understand that many people like to use tools they are familiar with. If you are switching over to Python or R from using programs like Excel, we encourage you to switch one step at a time. As an analogy, think about a cook licking the batter spoon to taste the dish. When cooking at home for your family, many people do this. But a chef at a restaurant would hopefully be fired for licking and reusing their spoon. Likewise, recreational data analysis can reasonably use a program like Excel to edit data. But professional data analysis requires the use of code to wrangle data.

###### Tip

We encourage you to start doing one step at a time in Python or R if you already use a program like Excel. For example, let’s say you currently format your football data in Excel, plot the data in Excel, and then fit a linear regression model in Excel. Start by plotting your data in Python or R the next time you work with your data. Once you get the hang of that, start fitting your model in Python or R. Finally, switch to formatting data in Python or R. For help with this transition, <a href="https://learning.oreilly.com/library/view/advancing-into-analytics/9781492094333" class="orm:hideurl"><em>Advancing into Analytics: From Excel to Python and R</em></a> by George Mount (O’Reilly, 2021) provides resources.

Besides data wrangling, you will also learn about some basic statistics in this appendix. *Statistics* means different things to different people.

During our day jobs, we see four uses of the word. Commonly, people use the word to refer to raw or objective data. For example, the (*x*, *y*) coordinates of a targeted pass might be referred to as the *stats* for a play. Sometimes a *statistic* can be something that is estimated, like expected points added (EPA) per play, or completion percentage above expected by a quarterback or offense. More formally, *statistics* can refer to the systematic collection and analysis of data. For example, somebody might run statistical analysis as part of a science experiment or as a business analyst using data science. Finally, the corresponding field of study related to the collection and analysis of data is called *statistics*. For example, you might have taken a statistics course in high school or know somebody who works as a professional statistician.

This appendix focuses on the use of *statistic* as something that can be estimated (specifically, summary statistics), and we show you how to summarize data by using statistics. For example, rather than needing to read the play-by-play report for a game, can you get an understanding of what occurred by looking at the summary statistics from the game? In Eric’s job, he generally doesn’t have the time to watch every game even once, let alone multiple times, nor does he have the chance to manually pore through each game’s play-by-play data. As such, he generally builds systems that can deliver *key performance indicators* (*KPIs*) that can help him see trends emerge in an efficient way. Summary statistics can also serve as the *features* for models. For example, if someone wants to bet on the number of touchdowns a quarterback will throw for in a certain game, his average number of touchdown passes thrown in a game is likely to be very helpful.

# Basic Statistics

Although not as glamorous as plotting, basic summary statistics are often more important because they serve as a foundation for data analysis and many plots.

## Averages

Perhaps the simplest statistic is the *average*, or *mean*, or for the mathematically minded, the *expectation* of a set of numbers. Commonly, when we talk about the *average* for a dataset, we are talking about the central tendency of the data, the value that “balances” the data. We show how to calculate these by hand in the next section.

We intentionally do not include code for this section, as one of Eric’s professors at the University of Nebraska–Lincoln, Dave Logan, would say, “The details of most calculations should be done once in everyone’s life, but not twice.” We will show you how to use Python and R to calculate these later in this appendix.

To work through this exercise, let’s explore passing plays again, as shown in <a href="ch01.html#sec-introduction" data-type="xref">Chapter 1</a>. This time, you will look at a quarterback’s `air_yards` and study its properties in an attempt to understand his and his team’s approach to the passing game. We use data from a 2020 game between Richard’s favorite team, the Green Bay Packers, and one of their division rivals, the Detroit Lions, and only then look at plays that have an `air_yards` reading by Detroit over the middle of the field. <a href="#sec-apdx-filter-select" data-type="xref">“Filtering and Selecting Columns”</a> shows how to obtain and filter this data. Looking at a small set of data will allow you to easily do hand calculations.

First, calculate a *mean* by hand. The air yards are 5, –1, 5, 8, 5, 6, 1, 0, 16, and 17. To calculate the mean, first *sum* (or add up) the numbers:

5 + –1 + 5 + 8 + 6 + 5 + 6 + 1 + 0 + 16 + 17 = 68

Next, divide by the total number of plays with air yards:

``` math
\frac{68}{11} = 6\!.\! 2
```

This allows you to estimate the mean air yards to the middle of the field to be 6.2 yards for Detroit during its first game against Green Bay in 2020. Also, we rounded the output to be 6.2. We need to round because the decimal of the resulting mean does not end and we really need to know only the first digit after the decimal, since the data is in integers. More formally, this is known as the number of *significant digits* or figures.

###### Tip

Significant digits are important when reporting results. Although formal rules exist, a rule of thumb that works most of the time is to simply report the number of digits that are useful.

Another way to estimate the “center” of a data set is to examine the median. The *median* is simply the middle number, or the value of the average individual (rather than the average value). To calculate the median, write the numbers in order from smallest to largest and then find the middle number, or the average of the two middle numbers if you have an even number of values.

The last common method to estimate an average number is to examine the mode. The *mode* is the most common value in the dataset. To calculate the mode, we need to create a table with counts and air yards, such as <a href="#tbl-mode" data-type="xref">Table B-1</a>.

| `air_yards` | `count` |     |
|-------------|---------|-----|
| –1          | 1       |     |
| 0           | 1       |     |
| 1           | 1       |     |
| 5           | 3       |     |
| 6           | 2       |     |
| 8           | 1       |     |
| 16          | 1       |     |
| 17          | 1       |     |

Table B-1. Summary table of number of passes for each air yard. {#tbl-mode}

With this example, 5 is the mode because three observations provide a reading of 5 air yards. Data can be multimodal (that is to say, have multiple modes). For example, if two outcomes have the same number of occurrences, a bimodal outcome occurred.

Each central tendency measure has its pluses and minuses. The mean of a set of numbers depends heavily on outliers (see <a href="ch02.html#sec-boxplot" data-type="xref">“Boxplots”</a> for a formal definition of *outliers*). If 2022 NFL MVP Patrick Mahomes, who at the time of this writing makes about \$45 million annually, walks over to a set of nine practice squad players (each making \$207,000 annually as of 2022), the average salary of the group is about \$4.5 million per player, which doesn’t accurately represent anything about that group. The median and mode of this dataset doesn’t change at all (\$207,000) with the inclusion of Mahomes, as the middle player scoots over half of a spot, and the group with the highest reading is the practice squad salary. That being said, much of the theoretical mathematics that has been built works a lot better with the mean, and discarding data points because they are outside the confines of the middle of the data set is not great practice, either. Thus, as with everything, the answer to which one you should use is “it depends.”

Finally, other kinds of means exist, but they do not appear in this book. For example, in sports betting (or financial investing in general), one will often care about the geometric mean of a dataset rather than the arithmetic mean (which we’ve computed previously). The *geometric mean* is simply computed by multiplying all the numbers in the dataset together and then taking the root corresponding to the number of elements in the dataset. The reason this is preferable in betting or financial markets is clear: numbers grow (or decline) exponentially in this case rather than additively.

Let’s examine these three measures of central tendency for air yards for all pass locations for both teams from the first game between Green Bay and Detroit in 2020 in <a href="#fig-averages" data-type="xref">Figure B-1</a>. This subset of the data is more interesting to examine but would have been harder to examine *by hand*.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0B01.png" />
<h6 id="figure-b-1.-comparing-different-types-of-averages-also-known-as-the-central-tendency-of-the-data">Figure B-1. Comparing different types of averages, also known as the central tendency of the data</h6>
</figure>

First, notice that the line, labeled *mean* (blue online), is to the right of the median (red online). This means the data is skewed or has outliers to the right. Second, the median is the same as the mode in this example, so their lines overlap. Most people, including us, usually refer to the mean as the *average*.

So, what does this tell us about Detroit’s passing in this game? First, the difference between mean and median shows that many pass plays are short, with the exception of a few long passes. This histogram shows us the “shape” of the data and the story it tells us about the game, which are probably better than simple summary statistics. This doesn’t necessarily scale up with the size of the data, but it shows why it’s always good to plot your data when you start and the reason we covered plotting as part of exploratory data analysis (EDA) in <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a>.

## Variability and Distribution

The previous section dealt with central tendency. Many situations, however, require you to know about the *variability* in the data. The easiest way to examine the variability in the data is the *range*, which is the distance between the smallest (minimum, or min) and biggest number (maximum, or max) in the data. For data shown in <a href="#tbl-mode" data-type="xref">Table B-1</a>, the min is –1 and the max is 17, so the range is 17 – –1 = 17 + 1 = 18.

Another method to examine the range and distribution of a dataset is to examine the *quantiles*. These focus on specific parts of the distribution, and their use can avoid the strength of severe outliers. The $`n`$th quantile is the data point where *n%* percent of the data lies underneath that data point. You’ve already seen one of these before, as the 50th quantile is the same thing as the median. <a href="ch02.html#sec-boxplot" data-type="xref">“Boxplots”</a> covered quantiles (specifically within the context of boxplots). The 25th, 50th, and 75th quantiles are commonly referred to as *quartiles*, and the difference between the first quartile and the third is the *interquartile range* (*IQR*).

Recall that boxplots show us where the middle 50% of the data occurs. Sometimes, other types of quantiles may be used as well. The benefit of quantiles are that they are estimated end points other than the central tenancy. For example, the mean or median allows you to examine how well average players do, but a quantile allows you to examine how well the best players do (for example, what does a player need in order to be better than 95% of other players?).

The most common method for examining the variability in a dataset is to look at the variance and its square root, the *standard deviation*. Broadly, the *variance* is the average squared deviation between the data points and the mean. The square is used so that the distance between each data point and the mean is counted as positive—so variability beneath and above the mean don’t “cancel out.” Using the Detroit Lions air yards example, you can do this calculation by hand with the numbers supplied in <a href="#tbl-var" data-type="xref">Table B-2</a>. We include a `mean` column in case you are doing this calculation in a spreadsheet such as Excel and to also help you see where the numbers come from.

| `air_yards` | `mean`  | `difference` | `difference squared` |     |
|-------------|---------|--------------|----------------------|-----|
| 5           | 8.03125 | –3.03125     | 9.18848              |     |
| 13          | 8.03125 | 4.96875      | 24.6885              |     |
| 3           | 8.03125 | –5.03125     | 25.3135              |     |
| 6           | 8.03125 | –2.03125     | 4.12598              |     |
| 6           | 8.03125 | –2.03125     | 4.12598              |     |
| –1          | 8.03125 | –9.03125     | 81.5635              |     |
| 5           | 8.03125 | –3.03125     | 9.18848              |     |
| 3           | 8.03125 | –5.03125     | 25.3135              |     |
| 4           | 8.03125 | –4.03125     | 16.251               |     |
| 28          | 8.03125 | 19.9688      | 398.751              |     |
| 28          | 8.03125 | 19.9688      | 398.751              |     |
| 11          | 8.03125 | 2.96875      | 8.81348              |     |
| –6          | 8.03125 | –14.0312     | 196.876              |     |
| –4          | 8.03125 | –12.0312     | 144.751              |     |
| –3          | 8.03125 | –11.0312     | 121.688              |     |
| 0           | 8.03125 | –8.03125     | 64.501               |     |
| 8           | 8.03125 | –0.03125     | 0.000976562          |     |
| 1           | 8.03125 | –7.03125     | 49.4385              |     |
| 6           | 8.03125 | –2.03125     | 4.12598              |     |
| 2           | 8.03125 | –6.03125     | 36.376               |     |
| 5           | 8.03125 | –3.03125     | 9.18848              |     |
| 6           | 8.03125 | –2.03125     | 4.12598              |     |
| 19          | 8.03125 | 10.9688      | 120.313              |     |
| 0           | 8.03125 | –8.03125     | 64.501               |     |
| 0           | 8.03125 | –8.03125     | 64.501               |     |
| 1           | 8.03125 | –7.03125     | 49.4385              |     |
| 4           | 8.03125 | –4.03125     | 16.251               |     |
| 24          | 8.03125 | 15.9688      | 255.001              |     |
| 0           | 8.03125 | –8.03125     | 64.501               |     |
| 16          | 8.03125 | 7.96875      | 63.501               |     |
| 17          | 8.03125 | 8.96875      | 80.4385              |     |
| 50          | 8.03125 | 41.9688      | 1761.38              |     |

Table B-2. Calculating the difference and difference squared by hand using a spreadsheet {#tbl-var .pagebreak-before .less_space}

After you create this table, you then take the sum of the `difference squared` column, which is 4,177. You divide that result by 76 because this is the number of observations minus 1. The reason to subtract 1 from the denominator has to do with the number of *degrees of freedom* at dataset has—the amount of data necessary to have a unique answer to a mathematical question.

Calculating $`\frac{4,177}{(77 - 1)}`$ gives the variance, 54.96. The units for variance with this example would be yards $`\times`$ yards, or yards<sup>2</sup>. This does not relate that well to the central tendency of the data, so we take the square root to get the standard deviation: 7.41 yards, which is now in the units of the original data. All statistical software can calculate this value for you, and comparing variances and standard deviations across various datasets helps you compare the variability of multiple sources of data easily and in a way that scales up. Often, you will divide the standard deviation by the mean to get the *coefficient of variation*, which is a unit-less measure of variability that takes into account the size of data points. This might be important when comparing, say, passing yards per play to kickoff returns per play. One is on the order of 10, and the other is on the order of 20, and the variability understandably scales with that.

## Uncertainty Around Estimates

When people give you predictions or summaries, a reasonable question is how much certainty exists around the prediction or summary? You can show uncertainty around the mean using the *standard error of the mean*, often abbreviated as *SEM*, or simply *SE*, for *standard error*. More informatively, you can estimate *confidence intervals* (*CIs*). The most commonly used CI is 95% because of historical convention from statistics. The CI will contain the true or correct estimate 95% of the time if we repeat our observation process many, many times. If you accept this probability view of the world, you will know your CIs will include the mean 95% of the time, but you just won’t know which 95% of the time.

The standard error is not the same thing as the standard deviation, but they are related. The former is trying to find the variability in the estimate of the population’s mean, while the latter is trying to find the variability in the population itself. To compute the standard error from the standard deviation, you simply divide the standard deviation by the square root of the sample size. Thus, as the sample size *n* grows, your estimate for the population’s mean becomes tighter, and the standard error decreases, while the variability in the population is fixed the whole time.

To calculate upper and lower bound of a confidence interval, use the *empirical rule* as a guide. The empirical says that, approximately 68% of the data lands within one standard deviation of the data, 95% within two standard deviations, and 99.7% within three. These values change with different distributions, and work best with a normal distribution (also known as a bell curve), but are pretty stable with respect to situation. The actual value for a 95% CI is 1.96.

Continuing with the previous example, calculate the standard error: $`7.41\,/\,\sqrt{77} = 0.84`$. Thus, you can write the mean as 8.03 ± 0.84(SE), or with the 95% CI as 8.03 (95% CI 6.38 to 9.68), which is calculated by 8.03 ± 0.84 × 1.96 = 1.64.

###### Warning

Always include uncertainty, such as a CI, around estimates such as mean when presenting to a technical audience. Presenting a *naked* mean is considered bad form because it does not allow the reader to see how much uncertainty exists around an estimate.

Based on statistical convention, you can compare 95% CIs to examine whether estimates differ. For example, the estimate of 8.03 (6.38 to 9.68 95% CI) differs from 0 because the 95% CI does not include 0. Thus, you can say that the air yards by the Detroit Lions in week 2 of 2020 differ from 0 when accounting for statistical uncertainty. If you were comparing two estimated means, you could compare both 95% CIs. If the CIs did not overlap, you can say the means are statistically different.

###### Note

People use 5% / 95% out of convention. Ronald L. Wasserstein et al. discuss this in a 2019 editorial in the [*The American Statistician*](https://oreil.ly/k_83m). Their editorial presents many perspectives on alternative methods for statistical inference.

<a href="ch03.html#sec-lm-ryoa" data-type="xref">Chapter 3</a> and other chapters cover more about statistical inferences. <a href="ch05.html#sec-lr-pass" data-type="xref">Chapter 5</a> also covers more about methods for estimating variances and CIs. Now, enough about theory and hand calculations; let’s see how to estimate these values in Python and R.

# Filtering and Selecting Columns

To calculate summary statistics with Python and R, first load your data and the required R packages:

```
## R
library(tidyverse)
library(nflfastR)

# Load all data
pbp_r <- load_pbp(2020)
```

You can use similar code in Python:

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# Load all data
pbp_py = nfl.import_pbp_data([2020])
```

Resulting in:

2020 done. Downcasting floats.

After loading the data, select a subset of the data you want to use. Filtering or querying data is a fundamental skill for data science. At its core, filtering data uses logic statements. These statements can be really frustrating at times; never assume that you’ve done it correctly the first time. Richard remembers spending half a day in grad school stuck in the computer lab trying to filter out example air-quality data with R. Now, this task takes him about 30 seconds.

###### Note

Logic operators simply refer to computer code that compares a statement and provides a binary response. In Python, logical results are either `True` or `False`. In R, logical results are either `TRUE` or `FALSE`.

Python and R have different methods for filtering data. We’ve focused on the tools we use, but other useful approaches exist. For example, `pandas` dataframes have a `.query()` function that we like to use because it is more compact than `.loc[]`. However, some filtering requires `.loc[]` because `.query()` does not work in all situations. Likewise, the `tidyverse` in R has a `filter()` function. You can use these functions with logical operators.

In R, this can be done using the `filter()` function. Two true statements may be combined with an *and* (`&`) symbol. For example, select Green Bay (`GB`) as the `home_team` and Detroit (`DET`) as the away team, and then use `home_team == 'GB' & away_team == 'DET'` with the `filter()` function. Likewise, the `select()` function allows you to work with only the columns you need, creating a smaller and easier-to-use dataframe:

```
## R
# Filter out game data
gb_det_2020_r <-
    pbp_r |>
    filter(home_team == 'GB' & away_team == 'DET')

# select pass data
gb_det_2020_pass_r <-
    gb_det_2020_r |>
    select(posteam, yards_after_catch, air_yards,
           pass_location, qb_scramble)
```

Python uses a `query()` function. The input requires a set of quotes around the input, unlike R (for example, `pandas` uses `"home_team == 'GB' & away_team == 'DET'"`). In addition, `pandas` uses a list of column names to select specific columns of interest:

```
## Python
# Filter out game data
gb_det_2020_py = \
    pbp_py.query("home_team == 'GB' & away_team == 'DET'")

# Select pass some pass related columns
gb_det_2020_pass_py = \
    gb_det_2020_py[
      ["posteam", "yards_after_catch",
      "air_yards", "pass_location",
      "qb_scramble"]]
```

# Calculating Summary Statistics with Python and R

With our dataset in hand, you can calculate the summary statistics introduced previously using Python and R. In Python, use `describe()` to see similar summaries that also include the median, count, minimum, and maximum values:

```
## Python
print(gb_det_2020_pass_py.describe())
```

Resulting in:

yards_after_catch air_yards qb_scramble count 38.000000 62.000000 181.000000 mean 6.263158 8.612903 0.016575 std 5.912352 10.938509 0.128025 min -2.000000 -6.000000 0.000000 25% 2.250000 1.250000 0.000000 50% 4.000000 5.000000 0.000000 75% 9.000000 12.750000 0.000000 max 20.000000 50.000000 1.000000

In R, use the `summary()` function:

```
## R
summary(gb_det_2020_pass_r)
```

Resulting in:

posteam yards_after_catch air_yards pass_location Length:181 Min. :-2.000 Min. :-6.000 Length:181 Class :character 1st Qu.: 2.250 1st Qu.: 1.250 Class :character Mode :character Median : 4.000 Median : 5.000 Mode :character Mean : 6.263 Mean : 8.613 3rd Qu.: 9.000 3rd Qu.:12.750 Max. :20.000 Max. :50.000 NA's :143 NA's :119 qb_scramble Min. :0.00000 1st Qu.:0.00000 Median :0.00000 Mean :0.01657 3rd Qu.:0.00000 Max. :1.00000

One benefit of using `summary()` is that it shows the missing or `NA` values in R. This can help you see possible problems in the data. R also includes `1st Qu.` and `3rd Qu.`, which are the first and third quartiles, which as stated previously are two of the special quantiles.

Both languages allow you to create customized summaries. For Python, use the `.agg()` function to aggregate the data frame. Use a dictionary inside Python to tell `pandas` which column to aggregate and what functions to use. Recall that Python defines dictionaries by using `{"key" : [values]}` notation for shorter notation or `dict("key" : [values])` for a longer notation. In this case, the dictionary uses the column `air_yards` as the key and the aggregating functions as the list values:

```
## Python
print(gb_det_2020_pass_py.agg(
  {
    "air_yards": ["min", "max", "mean", "median",
                  "std", "var", "count"]
    }
))
```

Resulting in:

air_yards min -6.000000 max 50.000000 mean 8.612903 median 5.000000 std 10.938509 var 119.650978 count 62.000000

You can also summarize the data in R in a customized and repeatable way as well by using `|>` to pipe the data to the `summarize()` function. Then tell R what functions to use on which columns. Use `min()` for the minimum, `max()` for the maximum, `mean()` for the mean, `median()` for the median, `sd()` for standard deviation, `var()` for the variance, and `n()` for the count. Also tell R what to call the output columns and assign new names. We chose these names because they are short as well as relatively easy to both type and understand where the new numbers come from:

```
## R
gb_det_2020_pass_r |>
summarize(min_yac = min(air_yards),
          max_yac = max(air_yards),
          mean_yac = mean(air_yards),
          median_yac = median(air_yards),
          sd_yac = sd(air_yards),
          var_yac = var(air_yards),
          n_yac = n())
```

Resulting in:

\# A tibble: 1 × 7 min_yac max_yac mean_yac median_yac sd_yac var_yac n_yac \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<int\> 1 NA NA NA NA NA NA 181

R gives us only `NA` values. What is going on? Recall that these columns have missing data, so tell R to ignore them by using the `na.rm = TRUE` option in the functions:

```
## R
gb_det_2020_pass_r |>
summarize(min_yac = min(air_yards, na.rm = TRUE),
          max_yac = max(air_yards, na.rm = TRUE),
          mean_yac = mean(air_yards, na.rm = TRUE),
          median_yac = median(air_yards, na.rm = TRUE),
          sd_yac = sd(air_yards, na.rm = TRUE),
          var_yac = var(air_yards, na.rm = TRUE),
          n_yac = n())
```

Resulting in:

\# A tibble: 1 × 7 min_yac max_yac mean_yac median_yac sd_yac var_yac n_yac \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<int\> 1 -6 50 8.61 5 10.9 120. 181

Both Python and R allow you to *group by* variables or calculate statistics by grouping variables such as the mean `air_yards` for each `posteam`. Python has a grouping function, `groupby()`, that can take `posteam` to calculate the statistics by the possession team (notice Python does not use piping). Instead, string together one function after another. This approach is based on the object-oriented nature of Python compared to the procedural nature of R, both of which have benefits and drawbacks you have to consider:

```
## Python
print(gb_det_2020_pass_py.groupby("posteam").agg(
  {
    "air_yards": ["min", "max", "mean",
                  "median", "std", "var", "count"]
    }
))
```

Resulting in:

air_yards min max mean median std var count posteam DET -6.0 50.0 8.031250 5.0 11.607796 134.740927 32 GB -4.0 34.0 9.233333 5.0 10.338023 106.874713 30

With Python, you can include a second variable by including a second entry in the dictionary. Also, `pandas`, unlike the `tidyverse`, allows you to calculate different summaries for each variable by changing the dictionary values:

```
## Python
print(gb_det_2020_pass_py.groupby("posteam").agg(
  {
    "yards_after_catch": ["min", "max", "mean",
                          "median", "std", "var", "count"],
    "air_yards": ["min", "max", "mean",
                  "median", "std", "var", "count"]
                  }
))
```

Resulting in:

yards_after_catch ... air_yards min max mean ... std var count posteam ... DET 0.0 20.0 6.900000 ... 11.607796 134.740927 32 GB -2.0 19.0 5.555556 ... 10.338023 106.874713 30 \[2 rows x 14 columns\]

R also includes a *group by* function, `group_by()`, that may be used with piping:

```
## R
gb_det_2020_pass_r |>
  group_by(posteam) |>
  summarize(min_yac = min(air_yards, na.rm = TRUE),
            max_yac = max(air_yards, na.rm = TRUE),
            mean_yac = mean(air_yards, na.rm = TRUE),
            median_yac = median(air_yards, na.rm = TRUE),
            sd_yac = sd(air_yards, na.rm = TRUE),
            var_yac = var(air_yards, na.rm = TRUE),
            n_yac = n())
```

Resulting in:

\# A tibble: 3 × 8 posteam min_yac max_yac mean_yac median_yac sd_yac var_yac n_yac \<chr\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<dbl\> \<int\> 1 DET -6 50 8.03 5 11.6 135. 78 2 GB -4 34 9.23 5 10.3 107. 89 3 \<NA\> Inf -Inf NaN NA NA NA 14

# A Note About Presenting Summary Statistics

The key for presenting summary statistics is to make sure you use the information available to you to effectively tell your story. First, knowing your target audience is extremely important. For example, if you’re talking to Cris Collinsworth about his next *Sunday Night Football* broadcast (something Eric did at his previous job) or to your buddies at the bar during a game, you’re going to present the information differently.

Furthermore, if you’re presenting your work to the director of research and strategy for an NFL team, you’re probably going to have to supply different—specifically, more—information than in the aforementioned two examples. Likewise, when talking to the director of research and strategy, you will likely need to justify both your estimates and your methodological choices. Conversely, unless you’re having beers with Eric and Richard (or other quants), you probably will not be discussing modeling choices over beers!

The *why* is key, and you’ll have to dig into data and truly understand it well, so that you can speak it in multiple languages. For example, is the dynamic you’re seeing due to coverage differences, the wide receivers, or changes in the quarterback’s fundamental ability?

Second, use statistics and modeling to support your story, but do not use them as your entire story. Say, “Drew Brees is still the most accurate passer in football, even after you adjust for situation” rather than “Drew Brees has the highest completion percentage above expected. Period.” Adding context to one’s work is something that we, as authors, have noticed helps the best quantitative people stand out compared to many quantitative people. In fact, communication skills about numbers helped both of us get our current jobs.

Third, while a picture may be worth a thousand words, walk your reader through the picture. A graph with no context is likely worse than no graph at all because all the graph without context will do is confuse your audience. For a nontechnical audience, you may include a figure and mention the “averages” in your words. Thus, the raw summary statistics may not even be shown in your writing. For more technical audiences, include the details and uncertainty either in text for one or two numbers or in a table or supplemental materials for more summary statistics.

# Improving Your Presentation

We have found there are two good ways to improve our presentation of summary statistics. First, present early and present often to people who will give you constructive feedback. Make sure they can understand your message, and if they cannot, ask them what is unclear and figure out how to more clearly make your point. For example, we like to give lectures and seminars to students because we will ask them how they might explain a figure and then they help us to more clearly think about data. Also, if you cannot explain concepts to high school and college students, you do not clearly understand the ideas well.

Second, look at other people’s work. Read blogs, read other books, read articles, and watch or listen to podcasts. Other people’s examples will help you see what is clear and what is not. Besides casual reading, read critically. What works? What does not work? Why did the authors make a choice? How would you help the author better explain their findings to you? If you have a chance, ask the authors if you see them or interact with them on social media. A diplomatic tweet will likely start a conversation. For example, you might reply to a tweet, *I liked your model output and the insight it gave me to Friday’s game. Why did you use X rather than Y?* Conversely, replying to a tweet with *Your model sucked, you should use my favorite model* will likely be ignored or possibly start a pointless flame war and diminish not only the original poster’s view of you but also that of other people who read the tweet.

# Exercises

1.  Repeat the processes within this chapter with a different game.

2.  Repeat the processes within this chapter with a different feature, like rushing yards.

# Suggested Readings

Many books describe introductory statistics. If you want to learn more about statistics, we suggest reading the first one or two chapters of several books until you find one that *speaks* to you. The following are some books you may wish to consider:

- <a href="https://learning.oreilly.com/library/view/advancing-into-analytics/9781492094333" class="orm:hideurl"><em>Advancing into Analytics: From Excel to Python and R</em></a> by George Mount (O’Reilly, 2021). This book assumes you know Excel well but then helps you transition to either R or Python. The book covers the basis of statistics.

- *Statistical Inference via Data Science: A ModernDive into R and the Tidyverse* by Chester Ismay and Albert Y. Kim (CRC Press, 2019), also updated at the book’s home page. This book provides a robust introduction to statistical inferences for people who also want to learn R.

- <a href="https://learning.oreilly.com/library/view/practical-statistics-for/9781492072935" class="orm:hideurl"><em>Practical Statistics for Data Scientists</em>, 2nd edition</a>, by Peter Bruce et al. (O’Reilly, 2020). This book provides an introduction to statistics for people who already know some R or Python.

- *Introductory Statistics with R* by Peter Dalgaard (Springer 2008) is a classic book; while somewhat dated for code, this was the book Richard used to first learn statistics (and R).

- <a href="https://learning.oreilly.com/library/view/essential-math-for/9781098102920" class="orm:hideurl"><em>Essential Math for Data Science</em></a> by Thomas Nield (O’Reilly, 2022) provides a gentle introduction to statistics as well as mathematics for applied data scientists.
