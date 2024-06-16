# Chapter 2. Exploratory Data Analysis: Stable Versus Unstable Quarterback Statistics

In any field of study, a level of intuition (commonly known as a *gut feeling*) can exist that separates the truly great subject-matter experts from the average ones, the average ones from the early-career professionals, or the early-career professionals from the novices. In football, that skill is said to manifest itself in player evaluation, as some scouts are perceived to have a knack for identifying talent through great intuition earned over years of honing their craft. Player traits that translate from one situation to the next—whether from college football to the professional ranks, or from one coach’s scheme to another’s—require recognition and further investigation, while player outcomes that cannot be measured (at least using current data and tools) are discarded. Experts in player evaluation also know how to properly communicate the fruits of their labor in order to gain maximum collective benefit from it.

While traditional scouting and football analytics are often considered at odds with each other, the statistical evaluation of players requires essentially the same process. Great football analysts are able to, when evaluating a player’s data (or multiple players’ data), find the right data specs to interrogate, production metrics to use, situational factors to control for, and information to discard. How do you acquire such an acumen? The same way a scout does. Through years of deliberate practice and refinement, an analyst gains not only a set of tools for player, team, scheme, and game evaluation but also a knack for the right question to ask at the right time.

###### Tip

Famous statistician [John Tukey noted](https://oreil.ly/RVyMp), “Far better an approximate answer to the right question, which is often vague, than an exact answer to the wrong question, which can always be made precise.” Practically, this quote illustrates that in football, or broader data science, asking the question that meets your needs is more important than using your existing data and models precisely.

One advantage that statistical approaches have over traditional methods is that they are scalable. Once an analyst develops a tried-and-true method for player evaluation, they can use the power of computing to run that analysis on many players at once—a task that is incredibly cumbersome for traditional scouting methods.

In this chapter, we give you some of the first tools necessary to develop a knack for football evaluation using statistics. The first idea to explore in this topic is *stability*. Stability is important when evaluating anything, especially in sports. This is because stability measures provide a comparative way to determine how much of a skill is *fundamental* to the player—how much of what happened in a given setting is transferable to another setting—and how much of past performance can be attributed to *variance*. In the words of FiveThirtyEight founder Nate Silver, stability analysis helps us tease apart what is *signal* and what is *noise*. If a player does very well in the stable components of football but poorly in the unstable ones, they might be a *buy-low* candidate—a player who is underrated in the marketplace. The opposite, a player who performs well in the unstable metrics but poorly in the stable ones, might be a *sell-high* player.

Exact definitions of stability vary based on specific contexts, but in this book we refer to the stability of an evaluation metric as the metric’s consistency over a predetermined time frame. For example, for fantasy football analysts, that time frame might be week to week, while for an analyst building a draft model for a team, it might be from a player’s final few seasons in college to his first few seasons as a pro. The football industry generally uses Pearson’s correlation coefficient, or its square, the coefficient of determination, to measure stability. A *Pearson’s correlation coefficient* ranges from –1 to 1. A value of 0 corresponds to no correlation, whereas a value of 1 corresponds to a perfect positive correlation (two variables increase in value together), and a value of –1 corresponds to a perfect negative correlation (one variable increases as the second variable decreases).

While the precise numerical threshold for a metric to be considered stable is usually context- or era-specific, a higher coefficient means a more stable statistic. For example, pass-rushing statistics are generally pretty stable, while coverage statistics are not; Eric talked about this more in a recent [paper from the Sloan Sports Analytics Conference](https://oreil.ly/_GOBP). If comparing two pass-rushing metrics, the less-stable one might have a lower correlation coefficient than the more-stable coverage metric.

###### Note

Fantasy fans will know *stability analysis* by another term: *sticky stats*. This term arises because some statistical estimates “stick” around and are consistent through time.

Stability analysis is part of a subset of statistical analysis called *exploratory data analysis* (*EDA*), which was coined by the American statistician John Tukey. In contrast to formal modeling and hypothesis testing, EDA is an approach of analyzing datasets to summarize their main characteristics, often using statistical graphics and other data visualization methods. EDA is an often-overlooked step in the process of using statistical analysis to understand the game of football—both by newcomers and veterans of the discipline—but for different reasons.

###### Note

John Tukey also coined other terms, some that you may know or will hopefully know by the end of this book, including *boxplot* (a type of graph), *analysis of variance* (ANOVA for short; a type of statistical test), *software* (computer programs), and *bit* (the smallest unit of computer data, usually represented as 0/1; you’re probably more familiar with larger units such as the byte, which is 8 bits, and larger units like gigabytes). Tukey and his students also helped the Princeton University football team implement data analysis using basic statistical methods by examining over 20 years of football data. However, the lack of modern computers limited his work, and many of the tools you learn in this book are more advanced than the methods he had access to. For example, one of his former students, [Gregg Lange](https://oreil.ly/L3MCF), remembered how a simple mistake required him to reload 100 pounds of data cards into a computer. To read more about Tukey’s life and contributions, check out “John W. Tukey: His Life and Professional Contributions” by David Brillinger in the [*Annals of Statistics*](https://oreil.ly/7Wb0a).

# Defining Questions

Asking the right questions is as important as solving them. In fact, as the Tukey quote highlights, the right answer to the wrong question is useless in and of itself, while the right question can lead you to prosperous outcomes even if you fall short of the correct answer. Learning to ask the right question is a process honed by learning from asking the wrong questions. Positive results are the spoils earned from fighting through countless negative results.

To be scientific, a question needs to be about a hypothesis that is both testable and falsifiable. For example, “Throwing deep passes is more valuable than short passes, but it’s difficult to say whether or not a quarterback is good at deep passes” is a reasonable hypothesis, but to make it scientific, you need to define what “valuable” means and what you mean when we say a player is “good” (or “bad”) at deep passes. To that aim, you need data.

# Obtaining and Filtering Data

To study the stability of passing data, use the `nflfastR` package in R or the `nfl_data_py` package in Python. Start by loading the data from 2016 to 2022 as play-by-play, or `pbp`, using the tools you learned in <a href="ch01.html#sec-introduction" data-type="xref">Chapter 1</a>.

Using 2016 is largely an arbitrary choice. In this case, it’s the last year with a material rule change (moving the kickoff touchback up to the 25-yard line) that affected game play. Other seasons are natural breaking points, like 2002 (the last time the league expanded), 2011 (the last influential change to the league’s collective bargaining agreement), 2015 (when the league moved the extra point back to the 15-yard line), 2020 (the COVID-19 pandemic, and also when the league expanded the playoffs), and 2021 (when the league moved from 16 to 17 regular-season games).

In Python, load the `pandas` and `numpy` packages as well as the `nfl_data_py` package:

```
## Python
import pandas as pd
import numpy as np
import nfl_data_py as nfl
```

###### Warning

Python starts numbering at 0. R starts numbering at 1. Many an aspiring data scientist has been tripped up by this if using both languages. Because of this, you need to add **`+ 1`** to the input of the last value in `range()` in this example.

Next, tell Python what years to load by using `range()`. Then import the NFL data for those seasons:

```
## Python
seasons = range(2016, 2022 + 1)
pbp_py = nfl.import_pbp_data(seasons)
```

In R, first load the required packages. The `tidyverse` collection of packages helps you wrangle and plot the data. The `nflfastR` package provides you with the data. The `ggthemes` package assists with plotting formatting:

```
## R
library("tidyverse")
library("nflfastR")
library("ggthemes")
```

In R, you may use the shortcut `2016:2022` to specify the range 2016 to 2022:

```
## R
pbp_r <- load_pbp(2016:2022)
```

###### Warning

With any dataset, understand the *metadata*, or the data about the data. For example, what do 0 and 1 mean? Which is yes and which is no? Or do the authors use 1 and 2 for levels? We have heard about scientific studies being retracted because the data analysts and scientists misunderstood the metadata and the uses of 1 and 2 versus the standard 0 and 1. Thus, scientists had to tell people their study was flawed because they did not understand their own data structure. For example, a [2021 article in *Significance*](https://oreil.ly/9kORC) describes an occurrence of this mistake.

To get the subset of data you need for this analysis, filter down to just the passing plays, which can be done with the following code:

```
## Python
pbp_py_p = \
    pbp_py\
    .query("play_type == 'pass' & air_yards.notnull()")\
    .reset_index()
```

In R, `filter()` the data by using the same criteria:

```
## R
pbp_r_p <-
    pbp_r |>
    filter(play_type == "pass" & !is.na(air_yards))
```

Here, `play_type` being equal to `pass` will eliminate both running plays and plays that are negated because of a penalty. Sometimes you want to include plays that have a penalty (for example, if you are using a grade-based system like the one at PFF). Grade-based systems attempt to measure how well the player performed on a play independent of the final statistics of the play, so keeping data where `play_type == no_play` might have value.

For the sake of this exercise, though, we have you omit such plays. You also omit plays where `air_yards` is `NA` (in R) or `NULL` (in Python). These plays occur when a pass is not aimed at an intended receiver because it’s batted down at the line of scrimmage, thrown away, or spiked. While those passes certainly count toward a passer’s final statistics, and are fundamental to who he is as a player, they are not necessarily relevant to the question being asking here.

Next, you need to do some data cleaning and wrangling.

First, define a *long* pass as a pass that has air yards greater than or equal to 20 yards, and a *short* pass as one with air yards less than 20 yards. The NFL has a categorical variable for pass length (`pass_length`) in data, but the classifications are not completely clear to the observer (see the exercises at the end of the chapter). Luckily, you can easily calculate this on your own (and use a different criterion if desired, such as 15 yards or 25 yards).

Second, the passing yards for incomplete passes are recorded as `NA` in R, or `NULL` in Python, but should be set to 0 for this analysis (as long as you’ve filtered properly previously).

In Python, the `numpy` (imported as `np`) package’s `where()` function helps with this change. First, create the filtering criteria:

```
## Python
pbp_py_p["pass_length_air_yards"] = np.where(
    pbp_py_p["air_yards"] >= 20, "long", "short"
)
```

Then use the filtering criteria to replace missing values:

```
## Python
pbp_py_p["passing_yards"] = \
    np.where(
        pbp_py_p["passing_yards"].isnull(), 0, pbp_py_p["passing_yards"]
        )
```

In R, the `ifelse()` function inside `mutate()` allows the same change:

```
## R
pbp_r_p <-
    pbp_r_p |>
    mutate(
        pass_length_air_yards = ifelse(air_yards >= 20, "long", "short"),
        passing_yards = ifelse(is.na(passing_yards), 0, passing_yards)
    )
```

<a href="app02.html#sec-ssdw-pass" data-type="xref">Appendix B</a> covers data manipulation topics such as filtering in great detail. Refer to this source if you need help better understanding our data wrangling. We are glossing over these details to help you get into the data right away with interesting questions.

###### Tip

Naming objects can be surprisingly hard when programming. Try to balance simple names that are easier to type with longer, more informative names. This can be especially important if you start writing scripts with longer names. The most important part of naming is to create understandable names for both others and your future self.

# Summarizing Data

Briefly examine some basic numbers used to describe the `passing_yards` data. In Python, select the `passing_yards` column and then use the `describe()` function:

```
## Python
pbp_py_p["passing_yards"]\
    .describe()
```

Resulting in:

count 131606.000000 mean 7.192111 std 9.667021 min -20.000000 25% 0.000000 50% 5.000000 75% 11.000000 max 98.000000 Name: passing_yards, dtype: float64

In R, take the dataframe and select (or `pull()`) the `passing_yards` column and then calculate the `summary()` statistics:

```
## R
pbp_r_p |>
    pull(passing_yards) |>
    summary()
```

This results in:

Min. 1st Qu. Median Mean 3rd Qu. Max. -20.000 0.000 5.000 7.192 11.000 98.000

In the outputs, here are what the names describe (<a href="app02.html#sec-ssdw-pass" data-type="xref">Appendix B</a> shows how to calculate these values):

- The `count` (only in Python) is the number of records in the data.

- The `mean` in Python (`Mean` in R) is the arithmetic average.

- The `std` (only in Python) is the standard deviation.

- The `min` in Python or `Min.` in R is the lowest or minimum value.

- The `25%` in Python or `1st Qu.` in R is the first-quartile value, for which one-fourth of all values are smaller.

- The `Median` (in R) or `50%` (in Python) is the middle value, for which half of the values are bigger and half are smaller.

- The `75%` in Python or `3rd Qu.` in R is the third-quartile value, for which three-quarters of all values are smaller.

- The `max` in Python or `Max.` in R is the largest or maximum value.

What you really want to see is a summary of the data under different values of `pass_length_air_yards`. For short passes, filter out the long passes and then summarize, in Python:

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "short"')["passing_yards"]\
    .describe()
```

Resulting in:

count 116087.000000 mean 6.526812 std 7.697057 min -20.000000 25% 0.000000 50% 5.000000 75% 10.000000 max 95.000000 Name: passing_yards, dtype: float64

And in R:

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "short") |>
    pull(passing_yards) |>
    summary()
```

Which results in:

Min. 1st Qu. Median Mean 3rd Qu. Max. -20.000 0.000 5.000 6.527 10.000 95.000

Likewise, you can filter to select long passes in Python:

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "long"')["passing_yards"]\
    .describe()
```

Resulting in:

count 15519.000000 mean 12.168761 std 17.923951 min 0.000000 25% 0.000000 50% 0.000000 75% 26.000000 max 98.000000 Name: passing_yards, dtype: float64

And in R:

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "long") |>
    pull(passing_yards) |>
    summary()
```

Resulting in:

Min. 1st Qu. Median Mean 3rd Qu. Max. 0.00 0.00 0.00 12.17 26.00 98.00

The point to notice here is that the *interquartile range*, the difference between the first and third quartile, is much larger for longer passes than for short passes, even though the maximum passing yards are about the same. The minimum values are going to be higher for long passes, since it’s almost impossible to gain negative yards on a pass that travels 20 or more yards in the air.

You can perform the same analysis for expected points added (EPA), which was introduced in <a href="ch01.html#sec-introduction" data-type="xref">Chapter 1</a>. Recall that EPA is a more continuous measure of play success that uses situational factors to assign a point value to each play. You can do this in Python:

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "short"')["epa"]\
    .describe()
```

Resulting in:

count 116086.000000 mean 0.119606 std 1.426238 min -13.031219 25% -0.606135 50% -0.002100 75% 0.959107 max 8.241420 Name: epa, dtype: float64

And in R:

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "short") |>
    pull(epa) |>
    summary()
```

Which results in:

Min. 1st Qu. Median Mean 3rd Qu. Max. NA's -13.0312 -0.6061 -0.0021 0.1196 0.9591 8.2414 1

Likewise, you can do this for long passes in Python:

```
## Python
pbp_py_p\
    .query('pass_length_air_yards == "long"')["epa"]\
    .describe()
```

Resulting in:

count 15519.000000 mean 0.382649 std 2.185551 min -10.477922 25% -0.827421 50% -0.465344 75% 2.136431 max 8.789743 Name: epa, dtype: float64

Or in R:

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "long") |>
    pull(epa) |>
    summary()
```

Resulting in:

Min. 1st Qu. Median Mean 3rd Qu. Max. -10.4779 -0.8274 -0.4653 0.3826 2.1364 8.7897

You get the same dynamic here: wider outcomes for longer passes than shorter ones. Longer passes are more *variable* than shorter passes.

Furthermore, if you look at the mean passing yards per attempt (YPA) and EPA per attempt for longer passes, they are both higher than those for short passes (while the relationship flips for the median, why is that?). Thus, on average, you can informally confirm the first part of our guiding hypothesis for the chapter: “Throwing deep passes is more valuable than short passes, but it’s difficult to say whether or not a quarterback is good at deep passes.”

###### Tip

Line breaks and white space are important for coding. These breaks help make your code easier to read. Python and R also handle line breaks differently, but sometimes both languages treat line breaks as special commands. In both languages, you often split function inputs to create shorter lines that are easier to read. For example, you can space a function as follows to break up line names and make your code easier to read:

```
## Python or R
my_plot(data=big_name_data_frame,
        x="long_x_name",
        y="long_y_name")
```

In R, make sure the comma stays on a previous line. In Python, you may need to use a `\` for line breaks.

```
## Python
x =\
    2 + 4
```

Or put the entire command in parentheses:

```
## Python
x = (
    2 + 4
    )
```

You can also write one Python function per line:

```
## Python
my_out = \
    my_long_long_long_data\
    .function_1()\
    .function_2()
```

# Plotting Data

While numerical summaries of data are useful, and many people are more algebraic thinkers than they are geometric ones (Eric is this way), many people need to visualize something other than numbers. Reasons we like to plot data include the following:

- Checking to make sure the data looks OK. For example, are any values too large? Too small? Do other wonky data points exist?

- Are there outliers in the data? Do they arise naturally (e.g., Patrick Mahomes in almost every passing efficiency chart) or unnaturally (e.g., a probability below 0 or greater than 1)?

- Do any broad trends emerge at first glance?

## Histograms

*Histograms*, a type of plot, allow you to see data by summing the counts of data points into bars. These bars are called *bins*.

###### Warning

If you have previous versions of the packages used in this book installed, you may need to upgrade if our code examples will not work. Conversely, future versions of packages used in this book may update how functions work. The book’s GitHub page (github.com/raerickson/football_book_code) may have updated code.

In Python, we use the `seaborn` package for most plotting in the book. First, import `seaborn` by using the alias `sns`. Then use the `displot()` function to create the plot shown in <a href="#fig-sns_hist_1" data-type="xref">Figure 2-1</a>:

```
## Python
import seaborn as sns
import matplotlib.pyplot as plt

sns.displot(data=pbp_py, x="passing_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0201.png" />
<h6 id="figure-2-1.-a-histogram-in-python-using-seaborn-for-the-passing_yards-variable">Figure 2-1. A histogram in Python using <code>seaborn</code> for the <code>passing_yards</code> variable</h6>
</figure>

###### Warning

On macOS, you also need to include `import matplotlib.pyplot as plt` when you load other packages. Likewise, macOS users also need to include `plt.show()` for their plot to appear after their plotting code. We also found we needed to use `plt.show()` with some editors on Linux (such as Microsoft Visual Studio Code) but not others (such as JupyterLab). If in doubt, include this optional code. Running `plt.show()` will do no harm but might be needed to make your figures appear. Windows may or may not require this.

Likewise, R allows for histograms to be easily created.

###### Note

Although base R comes with its own plotting tools, we use `ggplot2` for this book. The `ggplot2` tool has its own language, based on *The Grammar of Graphics* by Leland Wilkinson (Springer, 2005) and implemented in R by Hadley Wickham during his doctoral studies at Iowa State University. Pedagogically, we agree with David Robinson, who describes his reasons for teaching plotting with `ggplot2` over base R in a blog post titled [“Don’t Teach Built-in Plotting to Beginners (Teach ggplot2)”](https://oreil.ly/QDtpo).

In R, create the histogram shown in <a href="#fig-ggplot2_hist_1" data-type="xref">Figure 2-2</a> by using `ggplot2` in R with the `ggplot()` function. In the function, use the `pbp_r_p` dataset and set the aesthetic for `x` to be `passing_yards`. Then add the geometry `geom_histogram()`:

```
## R
ggplot(pbp_r, aes(x = passing_yards)) +
    geom_histogram()
```

Resulting in:

\`stat_bin()\` using \`bins = 30\`. Pick better value with \`binwidth\`.

Warning: Removed 257229 rows containing non-finite values (\`stat_bin()\`).

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0202.png" />
<h6 id="figure-2-2.-a-histogram-in-r-using-ggplot2-for-the-passing_yards-variable">Figure 2-2. A histogram in R using <code>ggplot2</code> for the <code>passing_yards</code> variable</h6>
</figure>

###### Warning

Intentionally using the wrong number of bins to hide important attributes of your data is considered fraud by the larger statistical community. Be thoughtful and intentional when you select the number of bins for a histogram. This process requires many iterations as you explore various numbers of histogram bins.

Figures <a href="#fig-sns_hist_1" data-type="xref" data-xrefstyle="select:labelnumber">2-1</a> and <a href="#fig-ggplot2_hist_1" data-type="xref" data-xrefstyle="select:labelnumber">2-2</a> let you understand the basis of our data. Passing yards gained ranges from about –10 yards to about 75 yards, with most plays gaining between 0 (often an incompletion) and 10 yards. Notice that R warns you to be careful with the `binwidth` and the number of bins and also warns you about the removal of missing values. Rather than using the default, set each bin to be 1 yard wide. You can either ignore the second warning about missing values or filter out missing values prior to plotting to avoid the warning. With such a bin width, the data no longer looks normal, because of the many, many incomplete passes.

Next, you will make a histogram for each `pass_depth_air_yards` value. We will show you how to create the short pass in Python (<a href="#fig-sns_hist_pass_short" data-type="xref">Figure 2-3</a>) and the long pass in R (<a href="#fig-ggplot2_hist_py_long" data-type="xref">Figure 2-4</a>).

In Python, change the theme to be `colorblind` for the palette option and use a `whitegrid` option to create plots similar to `ggplot2`’s black-and-white theme:

```
## Python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="colorblind")
```

Next, filter out the short passes:

```
## Python
pbp_py_p_short = \
    pbp_py_p\
    .query('pass_length_air_yards == "short"')
```

Then create a histogram and use `set_axis_labels` to change the plot’s labels, making it look better, as shown in <a href="#fig-sns_hist_pass_short" data-type="xref">Figure 2-3</a>:

```
## Python
# Plot, change labels, and then show the output
pbp_py_hist_short = \
    sns.displot(data=pbp_py_p_short,
                binwidth=1,
                x="passing_yards");
pbp_py_hist_short\
    .set_axis_labels(
        "Yards gained (or lost) during a passing play", "Count"
        );
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0203.png" />
<h6 id="figure-2-3.-refined-histogram-in-python-using-seaborn-for-the-passing_yards-variable">Figure 2-3. Refined histogram in Python using <code>seaborn</code> for the <code>passing_yards</code> variable</h6>
</figure>

In R, filter out the long passes and make the plot look better by adding labels to the x- and y-axes and using the black-and-white theme (`theme_bw()`), creating <a href="#fig-ggplot2_hist_py_long" data-type="xref">Figure 2-4</a>:

```
## R
pbp_r_p |>
    filter(pass_length_air_yards == "long") |>
    ggplot(aes(passing_yards)) +
    geom_histogram(binwidth = 1) +
    ylab("Count") +
    xlab("Yards gained (or lost) during passing plays on long passes") +
    theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0204.png" />
<h6 id="figure-2-4.-refined-histogram-in-r-using-ggplot2-for-the-passing_yards-variable">Figure 2-4. Refined histogram in R using <code>ggplot2</code> for the <code>passing_yards</code> variable</h6>
</figure>

###### Note

We will use the black-and-white theme, `theme_bw()`, for the remainder of the book as the default for R plots, and `sns.set_theme(style="whitegrid", palette="colorblind")` for Python plots. We like these themes because we think they look better on paper.

These histograms represent pictorially what you saw numerically in <a href="#sec-chp2-sub-sum-data" data-type="xref">“Summarizing Data”</a>. Specifically, shorter passes have fewer variable outcomes than longer passes. You can do the same thing with EPA and find similar results. For the rest of the chapter, we will stick with passing YPA for our examples, with EPA per pass attempt included within the exercises for you to do on your own.

###### Note

Notice that `ggplot2`, and, more broadly, R, work well with piping objects and avoids intermediate objects. In contrast, Python works well by saving intermediate objects. Both approaches have trade-offs. For example, saving intermediate objects allows you to see the output of intermediate steps of your plotting. In contrast, rewriting the same object name can be tedious. These contrasting approaches represent a philosophical difference between the two languages. Neither is inherently right or wrong, and both have trade-offs.

## Boxplots

Histograms allow people to *see* the distribution of data points. However, histograms can be cumbersome, especially when exploring many variables. Boxplots are a compromise between histograms and numerical summaries (see <a href="#tbl-boxplot-parts" data-type="xref">Table 2-1</a> for the numerical values). *Boxplots* get their name from the rectangular *box* containing the middle 50% of the sorted data; the line in the middle of the box is the median, so half of the sorted data falls above the line, and half of the data falls under the line.

Some people call boxplots *box-and-whisker* *plots* because lines extend above and under the box. These whiskers contain the remainder of the data other than outliers. Boxplots in both `seaborn` and `ggplot` use a default for *outliers* to be points that are more than 1.5 times the *interquartile range* (the range between the 25th and 75th percentiles)—either greater than the third quartile or less than the first quartile. These outliers are plotted with dots.

| Part name                     | Range of data                           |
|-------------------------------|-----------------------------------------|
| Top dots                      | Outliers above the data                 |
| Top whisker                   | 100% to 75% of data, excluding outliers |
| Top portion of the box        | 75% to 50% of data                      |
| Line in the middle of the box | 50% of data                             |
| Bottom portion of the box     | 50% to 25% of data                      |
| Bottom whisker                | 25% to 0% of data, excluding outliers   |
| Bottom dots                   | Outliers under the data                 |

Table 2-1. Parts of a boxplot {#tbl-boxplot-parts}

Various types of outliers exist. Outliers may be problem data points (for example, somebody entered –10 yards when they meant 10 yards), but they often exist as parts of the data. Understanding the reasons behind these data points often provides keen insights to the data because outliers reflect the best and worst outcomes and may have interesting stories behind the points. Unless outliers exist because of errors (such as the wrong data being entered), outliers usually should be included in the data used to train models.

###### Note

We place a semicolon (`;`) after the Python plot commands to suppress text descriptions of the plot. These semicolons are optional and simply a preference of the authors.

In Python, use the `boxplot()` function from `seaborn` and change the axes labels to create <a href="#fig-sns-boxplot" data-type="xref">Figure 2-5</a>:

```
## Python
pass_boxplot = \
    sns.boxplot(data=pbp_py_p,
                x="pass_length_air_yards",
                y="passing_yards");
pass_boxplot.set(
    xlabel="Pass length (long >= 20 yards, short < 20 yards)",
    ylabel="Yards gained (or lost) during a passing play",
);
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0205.png" />
<h6 id="figure-2-5.-boxplot-of-yards-gained-from-long-and-short-air-yard-passes-seaborn">Figure 2-5. Boxplot of yards gained from long and short air-yard passes (<code>seaborn</code>)</h6>
</figure>

In R, use `geom_boxplot()` with `ggplot2` to create <a href="#fig-ggplot2-boxplot" data-type="xref">Figure 2-6</a>:

```
## R
ggplot(pbp_r_p, aes(x = pass_length_air_yards, y = passing_yards)) +
    geom_boxplot() +
    theme_bw() +
    xlab("Pass length in yards (long >= 20 yards, short < 20 yards)") +
    ylab("Yards gained (or lost) during a passing play")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0206.png" />
<h6 id="figure-2-6.-boxplot-of-yards-gained-from-long-and-short-air-yard-passes-ggplot2">Figure 2-6. Boxplot of yards gained from long and short air-yard passes (<code>ggplot2</code>)</h6>
</figure>

# Player-Level Stability of Passing Yards per Attempt

Now that you’ve become acquainted with our data, it’s time to use it for player evaluation. The first thing you have to do is aggregate across a prespecified time frame to get a value for each player. While week-level outputs certainly matter, especially for fantasy football and betting (see <a href="ch07.html#sec-webscrap-draft" data-type="xref">Chapter 7</a>), most of the time when teams are thinking about trying to acquire a player, they use season-level data (or sometimes data over many seasons).

Thus, you aggregate at the season level here, by using the `grouby()` syntax in Python or `group_by()` syntax in R. The *group by* concept borrows from SQL-type database languages. When thinking about the process here, *group by* may be thought of as a verb. For example, you use the play-by-play data and then *group by* the *seasons* and then *aggregate* (in Python) or *summarize* (in R) to calculate the mean of the quarterback’s passing YPA.

For this problem, take the play-by-play dataframe (`pbp_py` or `pbp_r`) and then *group by* `passer_player_name`, `passer_player_id`, `season`, and `pass_length`. Group by both the player ID and the player name column because some players have the same name (or at least same first initial and last name), but the name is important for studying the results of the analysis. Start with the whole dataset first before transitioning to the subsets.

In Python, use `groupby()` with a list of the variables (`["item1", "item2"]` in Python syntax) that you want to group by. Then aggregate the data for `passing_yards` for the `mean`:

```
## Python
pbp_py_p_s = \
    pbp_py_p\
    .groupby(["passer_id", "passer", "season"])\
    .agg({"passing_yards": ["mean", "count"]})
```

With Python, also collapse the columns to make the dataframe easier to handle (`list()` creates a list, `map()` iterates over items, like a `for` loop without the loop syntax—see <a href="ch07.html#sec-webscrap-draft" data-type="xref">Chapter 7</a> details on `for` loops):

```
## Python
pbp_py_p_s.columns = list(map("_".join, pbp_py_p_s.columns.values))
```

Next, rename the columns to names that are shorter and more intuitive:

```
pbp_py_p_s \
    .rename(columns={'passing_yards_mean': 'ypa',
                     'passing_yards_count': 'n'},
            inplace=True)
```

In R, pipe `pbp_p` to the `group_by()` function and then use the `summarize()` function to calculate the `mean()` of `passing_yards`, as well as to calculate the number, `n()` of passing attempts for each player in each season. Include `.groups = "drop"` to tell R to drop the groupings from the resulting dataframe. The resulting mean of `passing_yards` is the YPA, which is a quarterback’s average passing distance per play. Use the `<-` function to save the resulting calculations as a new dataframe, `pbp_r_p_s`:

```
## R
pbp_r_p_s <-
    pbp_r_p |>
    group_by(passer_player_name, passer_player_id, season) |>
    summarize(
        ypa = mean(passing_yards, na.rm = TRUE),
        n = n(),
        .groups = "drop"
    )
```

Now look at the top of the resulting dataframe by using `head()` in Python and then `sort()` by `ypa` to help you better see the results. The `ascending=False` option tells Python to sort high to low (for example, arranging the values as 9, 8, 7) rather than low to high (for example, arranging the values as 7, 8, 9):

```
## Python
pbp_py_p_s\
    .sort_values(by=["ypa"], ascending=False)\
    .head()
```

Resulting in:

ypa n passer_id passer season 00-0035544 T.Kennedy 2021 75.0 1 00-0033132 K.Byard 2018 66.0 1 00-0031235 O.Beckham 2018 53.0 2 00-0030669 A.Wilson 2018 52.0 1 00-0029632 M.Sanu 2017 51.0 1

In R, use `arrange()` with `ypa` to sort the outputs. The negative sign (`—`) tells R to reverse the order (for example, 7, 8, 9 becomes 9, 8, 7 when sorted):

```
## R
pbp_r_p_s |>
    arrange(-ypa) |>
    print()
```

Resulting in:

\# A tibble: 746 × 5 passer_player_name passer_player_id season ypa n \<chr\> \<chr\> \<dbl\> \<dbl\> \<int\> 1 T.Kennedy 00-0035544 2021 75 1 2 K.Byard 00-0033132 2018 66 1 3 O.Beckham 00-0031235 2018 53 2 4 A.Wilson 00-0030669 2018 52 1 5 M.Sanu 00-0029632 2017 51 1 6 C.McCaffrey 00-0033280 2018 50 1 7 W.Snead 00-0030663 2016 50 1 8 T.Boyd 00-0033009 2021 46 1 9 R.Golden 00-0028954 2017 44 1 10 J.Crowder 00-0031941 2020 43 1 \# ℹ 736 more rows

Now this isn’t really informative yet, since the players with the highest YPA values are players who threw a pass or two (usually a trick play) that were completed for big yardage. Fix this by filtering for a certain number of passing attempts in a season (let’s say 100) and see what you get.

<a href="app03.html#sec-app-dw" data-type="xref">Appendix C</a> contains more tips and tricks for data wrangling if you need more help understanding what is going on with this code. In Python, reuse `pbp_py_p_s` and the previous code, but include a `query()` for players with 100 or more pass attempts by using `'n >= 100'`:

```
## Python
pbp_py_p_s_100 = \
    pbp_py_p_s\
    .query("n >= 100")\
    .sort_values(by=["ypa"], ascending=False)
```

Now, look at the head of the data:

```
## Python
pbp_py_p_s_100.head()
```

Resulting in:

ypa n passer_id passer season 00-0023682 R.Fitzpatrick 2018 9.617886 246 00-0026143 M.Ryan 2016 9.442155 631 00-0029701 R.Tannehill 2019 9.069971 343 00-0033537 D.Watson 2020 8.898524 542 00-0036212 T.Tagovailoa 2022 8.892231 399

In R, *group by* the same variables and then *summarize*—this time including the number of observations per group with `n()`. Keep piping the results and *filter* for passers with 100 or more (`n >= 100`) passes and *arrange* the output:

```
## R
pbp_r_p_100 <-
    pbp_r_p |>
    group_by(passer_id, passer, season) |>
    summarize(
        n = n(), ypa = mean(passing_yards),
        .groups = "drop"
    ) |>
    filter(n >= 100) |>
    arrange(-ypa)
```

Then print the top 20 results:

```
## R
pbp_r_p_100 |>
    print(n = 20)
```

Which results in:

\# A tibble: 300 × 5 passer_id passer season n ypa \<chr\> \<chr\> \<dbl\> \<int\> \<dbl\> 1 00-0023682 R.Fitzpatrick 2018 246 9.62 2 00-0026143 M.Ryan 2016 631 9.44 3 00-0029701 R.Tannehill 2019 343 9.07 4 00-0033537 D.Watson 2020 542 8.90 5 00-0036212 T.Tagovailoa 2022 399 8.89 6 00-0031345 J.Garoppolo 2017 176 8.86 7 00-0033873 P.Mahomes 2018 651 8.71 8 00-0036442 J.Burrow 2021 659 8.67 9 00-0026498 M.Stafford 2019 289 8.65 10 00-0031345 J.Garoppolo 2021 511 8.50 11 00-0033319 N.Mullens 2018 270 8.43 12 00-0033537 D.Watson 2017 202 8.41 13 00-0033077 D.Prescott 2020 221 8.40 14 00-0034869 S.Darnold 2022 137 8.34 15 00-0037834 B.Purdy 2022 233 8.34 16 00-0029604 K.Cousins 2020 513 8.31 17 00-0031345 J.Garoppolo 2019 532 8.28 18 00-0025708 M.Moore 2016 122 8.28 19 00-0033873 P.Mahomes 2019 596 8.28 20 00-0020531 D.Brees 2017 606 8.26 \# i 280 more rows

Even the most astute of you probably didn’t expect the Harvard-educated Ryan Fitzpatrick’s season as Jameis Winston’s backup to appear at the top of this list. You do see the MVP seasons of Matt Ryan (2016) and Patrick Mahomes (2018), and a bunch of quarterbacks (including Matt Ryan) coached by the great Kyle Shanahan.

## Deep Passes Versus Short Passes

Now, down to the business of the chapter, testing the second part of the hypothesis: “Throwing deep passes is more valuable than short passes, but it’s difficult to say whether or not a quarterback is good at deep passes.” For this stability analysis, do the following steps:

1.  Calculate the YPA for each passer for each season.

2.  Calculate the YPA for each passer for the previous season.

3.  Look at the correlation from the values calculated in steps 1 and 2 to see the stability.

Use similar code as before, but include `pass_length_air_yards` with the *group by* commands to include pass yards. With this operation, naming becomes hard.

We have you use the dataset (*play-by-play*, `pbp`), the language (either Python, `_py`, or R, `_r`), passing plays (`_p`), seasons data (`_s`), and finally, pass length (`_pl`).

For both languages, you will create a copy of the dataframe and then shift the year by adding 1. Then you’ll merge the new dataframe with the original dataframe. This will let you have the current and previous year’s values.

###### Tip

Longer names are tedious, but we have found unique names to be important so that you can quickly search through code by using tools like Find and Replace (which are found in most code editors) to see what is occurring with your code (with Find) or change names (with Replace).

In Python, create `pbp_r_p_s_pl`, using several steps. First, *group by* and *aggregate* to get the mean and count:

```
## Python
pbp_py_p_s_pl = \
    pbp_py_p\
    .groupby(["passer_id", "passer", "season", "pass_length_air_yards"])\
    .agg({"passing_yards": ["mean", "count"]})
```

Next, flatten the column names and rename `passing_yards_mean` to **`ypa`** and `passing_yards_count` to **`n`** in order to have shorter names that are easier to work with:

```
## Python
pbp_py_p_s_pl.columns =\
    list(map("_".join, pbp_py_p_s_pl.columns.values))
pbp_py_p_s_pl\
    .rename(columns={'passing_yards_mean': 'ypa',
                     'passing_yards_count': 'n'},
            inplace=True)
```

Next, reset the index:

```
## Python
pbp_py_p_s_pl.reset_index(inplace=True)
```

Select only short-passing data from passers with more than 100 such plays and long-passing data for passers with more than 30 such plays:

```
## Python
q_value = (
    '(n >= 100 & ' +
     'pass_length_air_yards == "short") | ' +
     '(n >= 30 & ' +
     'pass_length_air_yards == "long")'
)
pbp_py_p_s_pl = pbp_py_p_s_pl.query(q_value).reset_index()
```

Then create a list of columns to save (`cols_save`) and a new dataframe with only these columns (`air_yards_py`). Include a `.copy()` so edits will not be passed back to the original dataframe:

```
## Python
cols_save =\
    ["passer_id", "passer", "season",
     "pass_length_air_yards", "ypa"]
air_yards_py =\
    pbp_py_p_s_pl[cols_save].copy()
```

Next, copy `air_yards_py` to create `air_yards_lag_py`. Take the current season value and add 1 by using the shortcut command `+=` and rename the `passing_yards_mean` to include `lag` (which refers to the one-year offset or delay between the two years):

```
## Python
air_yards_lag_py =\
    air_yards_py\
    .copy()
air_yards_lag_py["season"] += 1
air_yards_lag_py\
    .rename(columns={'ypa': 'ypa_last'},
    inplace=True)
```

Finally, `merge()` the two dataframes together to create `air_yards_both_py` and use an *inner join* so only shared years will be saved and join *on* `passer_id`, `passer`, `season`, and `pass_length_air_yards`:

```
## Python
pbp_py_p_s_pl =\
    air_yards_py\
    .merge(air_yards_lag_py,
           how='inner',
           on=['passer_id', 'passer',
               'season', 'pass_length_air_yards'])
```

Check the results of your choice in Python by examining a couple of quarterbacks of your choice such as Tom Brady (`T.Brady`) and Aaron Rodgers (`A.Rodgers`) and include only the necessary columns to have an easier-to-view dataframe:

```
## Python
print(
    pbp_py_p_s_pl[["pass_length_air_yards", "passer",
                    "season", "ypa", "ypa_last"]]\
    .query('passer == "T.Brady" | passer == "A.Rodgers"')\
    .sort_values(["passer", "pass_length_air_yards", "season"])\
    .to_string()
)
```

Resulting in:

pass_length_air_yards passer season ypa ypa_last 47 long A.Rodgers 2019 12.092593 12.011628 49 long A.Rodgers 2020 16.097826 12.092593 51 long A.Rodgers 2021 14.302632 16.097826 53 long A.Rodgers 2022 10.312500 14.302632 45 short A.Rodgers 2017 6.041475 6.693523 46 short A.Rodgers 2018 6.697446 6.041475 48 short A.Rodgers 2019 6.207224 6.697446 50 short A.Rodgers 2020 6.718447 6.207224 52 short A.Rodgers 2021 6.777083 6.718447 54 short A.Rodgers 2022 6.239130 6.777083 0 long T.Brady 2017 13.264706 15.768116 2 long T.Brady 2018 10.232877 13.264706 4 long T.Brady 2019 10.828571 10.232877 6 long T.Brady 2020 12.252101 10.828571 8 long T.Brady 2021 12.242424 12.252101 10 long T.Brady 2022 10.802469 12.242424 1 short T.Brady 2017 7.071429 7.163022 3 short T.Brady 2018 7.356452 7.071429 5 short T.Brady 2019 6.048276 7.356452 7 short T.Brady 2020 6.777600 6.048276 9 short T.Brady 2021 6.634697 6.777600 11 short T.Brady 2022 5.832168 6.634697

###### Tip

We suggest using at least two players to check your code. For example, Tom Brady is the first player by passer_id, and looking at only his values might not show a mistake that does not affect the first player in the dataframe.

In R, similar steps are taken to create `pbp_r_p_s_pl`. First, create `air_yards_r` by selecting the columns needed and arrange the dataframe:

```
## R
air_yards_r <-
    pbp_r_p |>
    select(passer_id, passer, season,
           pass_length_air_yards, passing_yards) |>
    arrange(passer_id, season,
            pass_length_air_yards) |>
    group_by(passer_id, passer,
             pass_length_air_yards, season) |>
    summarize(n = n(),
              ypa = mean(passing_yards),
              .groups = "drop") |>
    filter((n >= 100 & pass_length_air_yards == "short") |
           (n >= 30 & pass_length_air_yards == "long")) |>
    select(-n)
```

Next, create the lag dataframe including a mutate to the seasons and add 1:

```
## R
air_yards_lag_r <-
    air_yards_r |>
    mutate(season = season + 1) |>
    rename(ypa_last = ypa)
```

Last, join the dataframes to create `pbp_r_p_s_pl`:

```
## R
pbp_r_p_s_pl <-
    air_yards_r |>
    inner_join(air_yards_lag_r,
              by = c("passer_id", "pass_length_air_yards",
                     "season", "passer"))
```

Check the results in R by examining passers of your choice such as Tom Brady (`T.Brady`) and Aaron Rodgers (`A.Rodgers`):

```
## R
pbp_r_p_s_pl |>
    filter(passer %in% c("T.Brady", "A.Rodgers")) |>
    print(n = Inf)
```

Which results in:

\# A tibble: 22 × 6 passer_id passer pass_length_air_yards season ypa ypa_last \<chr\> \<chr\> \<chr\> \<dbl\> \<dbl\> \<dbl\> 1 00-0019596 T.Brady long 2017 13.3 15.8 2 00-0019596 T.Brady long 2018 10.2 13.3 3 00-0019596 T.Brady long 2019 10.8 10.2 4 00-0019596 T.Brady long 2020 12.3 10.8 5 00-0019596 T.Brady long 2021 12.2 12.3 6 00-0019596 T.Brady long 2022 10.8 12.2 7 00-0019596 T.Brady short 2017 7.07 7.16 8 00-0019596 T.Brady short 2018 7.36 7.07 9 00-0019596 T.Brady short 2019 6.05 7.36 10 00-0019596 T.Brady short 2020 6.78 6.05 11 00-0019596 T.Brady short 2021 6.63 6.78 12 00-0019596 T.Brady short 2022 5.83 6.63 13 00-0023459 A.Rodgers long 2019 12.1 12.0 14 00-0023459 A.Rodgers long 2020 16.1 12.1 15 00-0023459 A.Rodgers long 2021 14.3 16.1 16 00-0023459 A.Rodgers long 2022 10.3 14.3 17 00-0023459 A.Rodgers short 2017 6.04 6.69 18 00-0023459 A.Rodgers short 2018 6.70 6.04 19 00-0023459 A.Rodgers short 2019 6.21 6.70 20 00-0023459 A.Rodgers short 2020 6.72 6.21 21 00-0023459 A.Rodgers short 2021 6.78 6.72 22 00-0023459 A.Rodgers short 2022 6.24 6.78

###### Tip

We use the philosophy “Assume your code is wrong until you have convinced yourself it is correct.” Hence, we often peek at our code to make sure we understand what the code is doing versus what we think the code is doing. Practically, this means following the advice of former US President Ronald Reagan: “Trust but verify” your code.

The dataframes you’ve created (either `pbp_py_p_s_pl` in Python or `pbp_r_p_s_pl` in R) now contain six columns. Look at the `info()` for the dataframe in Python:

```
## Python
pbp_py_p_s_pl\
    .info()
```

Resulting in:

\<class 'pandas.core.frame.DataFrame'\> RangeIndex: 317 entries, 0 to 316 Data columns (total 6 columns): \# Column Non-Null Count Dtype --- ------ -------------- ----- 0 passer_id 317 non-null object 1 passer 317 non-null object 2 season 317 non-null int64 3 pass_length_air_yards 317 non-null object 4 ypa 317 non-null float64 5 ypa_last 317 non-null float64 dtypes: float64(2), int64(1), object(3) memory usage: 15.0+ KB

Or `glimpse()` at the dataframe in R:

```
## R
pbp_r_p_s_pl |>
    glimpse()
```

Resulting in:

Rows: 317 Columns: 6 \$ passer_id \<chr\> "00-0019596", "00-0019596", "00-0019596", "00-00… \$ passer \<chr\> "T.Brady", "T.Brady", "T.Brady", "T.Brady", "T.B… \$ pass_length_air_yards \<chr\> "long", "long", "long", "long", "long", "long", … \$ season \<dbl\> 2017, 2018, 2019, 2020, 2021, 2022, 2017, 2018, … \$ ypa \<dbl\> 13.264706, 10.232877, 10.828571, 12.252101, 12.2… \$ ypa_last \<dbl\> 15.768116, 13.264706, 10.232877, 10.828571, 12.2…

The six columns contain the following data:

- `passer_id` is the unique passer identification number for the player.

- `passer` is the (potentially) nonunique first initial and last name for the passer.

- `pass_length_air_yards` is the type of pass (either long or short) you defined earlier.

- `season` is the final season in the season pair (e.g., `season` being 2017 means you’re comparing 2016 and 2017).

- `ypa` is the yards per attempt during the stated season (e.g., 2017 in the previous example).

- `ypa_last` is the yards per attempt during the season previous to the stated season (e.g., 2016 in the previous example).

Now that we’ve reminded ourselves what’s in the data, let’s dig in and see how many quarterbacks you have. With Python, use the `passer_id` column and find the `unique()` values and then find the length of this object:

```
## Python
len(pbp_py_p_s_pl.passer_id.unique())
```

Resulting in:

65

With R, use the (distinct) function with passer_id and then see how many rows exist:

```
## R
pbp_r_p_s_pl |>
    distinct(passer_id) |>
    nrow()
```

Resulting in:

\[1\] 65

You now have a decent sample size of quarterbacks. You can plot this data by using a scatterplot. *Scatterplots* plot points on a figure, which is in contrast to histograms that plot bins of data and to boxplots that plot summaries of the data such as medians. Scatterplots allow you to “see” the data directly. The horizontal axis is called the *x-axis* and typically includes the predictor, or causal, variable, if one exists. The vertical axis is called the *y-axis* and typically includes the response, or effect, variables, if one exists. With our example, you will use the YPA from the previous year as the predictor for YPA in the current year. Plot this in R by using `geom_point()` and call this plot `scatter_ypa_r` and then print `scatter_ypa_r` to create <a href="#fig-ypa-ggplot2" data-type="xref">Figure 2-7</a>:

```
## R
scatter_ypa_r <-
    ggplot(pbp_r_p_s_pl, aes(x = ypa_last, y = ypa)) +
    geom_point() +
    facet_grid(cols = vars(pass_length_air_yards)) +
    labs(
        x = "Yards per Attempt, Year n",
        y = "Yards per Attempt, Year n + 1"
    ) +
    theme_bw() +
    theme(strip.background = element_blank())

print(scatter_ypa_r)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0207.png" />
<h6 id="figure-2-7.-stability-of-ypa-plotted-with-ggplot2.-notice-that-both-sub-plots-have-the-same-x-and-y-scales">Figure 2-7. Stability of YPA plotted with <code>ggplot2</code>. Notice that both sub-plots have the same x and y scales</h6>
</figure>

<a href="#fig-ypa-ggplot2" data-type="xref">Figure 2-7</a> is encouraging for short passes. It appears that quarterbacks who are good on short passes in one year are good the following year, and vice versa. Notice that the long passes are much more unwieldy. To help you better examine these trends, include a line of best fit to the data (this is why we had you save `scatter_ypa_r`, so that you could reuse the plot here) to create <a href="#fig-ypa-ggplot2-trend" data-type="xref">Figure 2-8</a>:

```
## R
# add geom_smooth() to the previously saved plot
scatter_ypa_r +
    geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0208.png" />
<h6 id="figure-2-8.-stability-of-ypa-plotted-with-ggplot2-and-including-a-trend-line">Figure 2-8. Stability of YPA plotted with <code>ggplot2</code> and including a trend line</h6>
</figure>

For both pass types, the lines in <a href="#fig-ypa-ggplot2-trend" data-type="xref">Figure 2-8</a> have a slightly positive slope (the lines are increasing across the plot), but this is hard to see. To obtain this estimate using the correlations, look at the numerical values:

```
## R
pbp_r_p_s_pl |>
    filter(!is.na(ypa) & !is.na(ypa_last)) |>
    group_by(pass_length_air_yards) |>
    summarize(correlation = cor(ypa, ypa_last))
```

Resulting in:

\# A tibble: 2 × 2 pass_length_air_yards correlation \<chr\> \<dbl\> 1 long 0.234 2 short 0.438

These figures and analyses may be repeated in Python to create <a href="#fig-ypa-sns-trend" data-type="xref">Figure 2-9</a>:

```
## Python
sns.lmplot(data=pbp_py_p_s_pl,
           x="ypa",
           y="ypa_last",
           col="pass_length_air_yards");
plt.show();
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] Football Analytics with Python &amp; R/assets/fapr_0209.png" />
<h6 id="figure-2-9.-stability-of-ypa-plotted-with-seaborn-and-including-a-trend-line">Figure 2-9. Stability of YPA plotted with <code>seaborn</code> and including a trend line</h6>
</figure>

Likewise, the correlation can be obtained by using `pandas` as well:

```
## Python
pbp_py_p_s_pl\
    .query("ypa.notnull() & ypa_last.notnull()")\
    .groupby("pass_length_air_yards")[["ypa", "ypa_last"]]\
    .corr()
```

Resulting in:

ypa ypa_last pass_length_air_yards long ypa 1.000000 0.233890 ypa_last 0.233890 1.000000 short ypa 1.000000 0.438479 ypa_last 0.438479 1.000000

The Pearson’s correlation coefficient numerically captures what Figures <a href="#fig-ypa-ggplot2-trend" data-type="xref" data-xrefstyle="select:labelnumber">2-8</a> and <a href="#fig-ypa-sns-trend" data-type="xref" data-xrefstyle="select:labelnumber">2-9</a> show.

While both datasets include a decent amount of noise, vis-à-vis Pearson’s correlation coefficient, a quarterback’s performance on shorter passes is twice as stable as on longer passes. Thus, you can confirm the second part of the guiding hypothesis of the chapter: “Throwing deep passes is more valuable than short passes, but it’s difficult to say whether or not a quarterback is good at deep passes.”

###### Note

A Pearson’s correlation coefficient can vary from –1 to 1. In the case of stability, a number closer to +1 implies strong, positive correlations and more stability, and a number closer to 0 implies weak correlations at best (and an unstable measure). A Pearson’s correlation coefficient of –1 implies a decreasing correlation and does not exist for stability but would mean a high value this year would be correlated with a low value next year.

## So, What Should We Do with This Insight?

Generally speaking, noisy data is a place to look for players (or teams or units within teams) that have pop-up seasons that are not likely to repeat themselves. A baseball player who sees a 20-point jump in his average based on a higher *batting average on balls in play* (*BABIP*) one year might be someone you want to avoid rostering in fantasy or real baseball. Similarly, a weaker quarterback who generates a high YPA (or EPA per pass attempt) on deep passes one year—without a corresponding increase in such metrics on shorter passes, the more stable of the two—might be what analysts call a *regression candidate*.

For example, let’s look at the leaderboard for 2017 deep passing YPA in Python:

```
## Python
pbp_py_p_s_pl\
    .query(
        'pass_length_air_yards == "long" & season == 2017'
        )[["passer_id", "passer", "ypa"]]\
    .sort_values(["ypa"], ascending=False)\
    .head(10)
```

Resulting in:

passer_id passer ypa 41 00-0023436 A.Smith 19.338235 79 00-0026498 M.Stafford 17.830769 12 00-0020531 D.Brees 16.632353 191 00-0032950 C.Wentz 13.555556 33 00-0022942 P.Rivers 13.347826 0 00-0019596 T.Brady 13.264706 129 00-0029604 K.Cousins 12.847458 114 00-0029263 R.Wilson 12.738636 203 00-0033077 D.Prescott 12.585366 109 00-0028986 C.Keenum 11.904762

Some good names are on this list (Drew Brees, Tom Brady, Russell Wilson) but also some so-so names. Let’s look at the same list in 2018:

```
## Python
pbp_py_p_s_pl\
    .query(
        'pass_length_air_yards == "long" & season == 2018'
        )[["passer_id", "passer", "ypa"]]\
    .sort_values(["ypa"], ascending=False)\
    .head(10)
```

Resulting in:

passer_id passer ypa 116 00-0029263 R.Wilson 15.597403 14 00-0020531 D.Brees 14.903226 205 00-0033077 D.Prescott 14.771930 214 00-0033106 J.Goff 14.445946 35 00-0022942 P.Rivers 14.357143 157 00-0031280 D.Carr 14.339286 188 00-0032268 M.Mariota 13.941176 64 00-0026143 M.Ryan 13.465753 193 00-0032950 C.Wentz 13.222222 24 00-0022803 E.Manning 12.941176

Alex Smith, who was long thought of as a dink-and-dunk specialist, dropped off this list completely. He led the league in passer rating in 2017, before being traded by Kansas City to Washington for a third-round pick and star cornerback Kendall Fuller (there’s a team that knows how to sell high!).

While the list includes some repeats or YPA on deep passes, many new names emerge. Specifically, if you filter for Matt Ryan’s name in the dataset, you’ll find that he averaged 17.7 YPA on deep passes in 2016 (when he won NFL MVP). In 2017, that value fell to 8.5, then rose back up to 13.5 in 2018. Did Ryan’s ability drastically change during these three years, or was he subject to significant statistical variability? The math would suggest the latter. In fantasy football or betting, he would have been a *sell-high* candidate in 2017 and a *buy-low* candidate in 2018 as a result.

# Data Science Tools Used in This Chapter

This chapter covered the following topics:

- Obtaining data from multiple seasons by using the `nflfastR` package either directly in R or via the `nfl_data_py` package in Python

- Changing columns based on conditions by using `where` in Python or `ifelse()` statements in R

- Using `describe()` for data with `pandas` or `summarize()` for data in R

- Reordering values by using `sort_by()` in Python or `arrange()` in R

- Calculating the difference between years by using `merge()` in Python or `join()` in R

# Exercises

1.  Create the same histograms as in <a href="#sec-eda-hist" data-type="xref">“Histograms”</a> but for EPA per pass attempt.

2.  Create the same boxplots as in <a href="#sec-eda-hist" data-type="xref">“Histograms”</a> but for EPA per pass attempt.

3.  Perform the same stability analysis as in <a href="#sec-eda-stable" data-type="xref">“Player-Level Stability of Passing Yards per Attempt”</a>, but for EPA per pass attempt. Do you see the same qualitative results as when you use YPA? Do any players have similar YPA numbers one year to the next but have drastically different EPA per pass attempt numbers across years? Where could this come from?

4.  One of the reasons that data for long pass attempts is less stable than short pass attempts is that there are fewer of them, which is largely a product of 20 yards being an arbitrary cutoff for long passes (by companies like PFF). Find a cutoff that equally splits the data and perform the same analysis. Do the results stay the same?

# Suggested Readings

If you want to learn more about plotting, here are some resources that we found helpful:

- [*The Visual Display of Quantitative Information* by Edward Tufte](https://oreil.ly/BYBhX) (Graphics Press, 2001). This book is a classic on how to think about data. The book does not contain code but instead shows how to see information for data. The guidance in the book is priceless.

- The [`ggplot2` package documentation](https://ggplot2.tidyverse.org). For those of you using R, this is the place to start to learn more about `ggplot2`. The page includes beginner resources and links to advanced resources. The page also includes examples that are great to browse.

- The [`seaborn` package documentation](https://seaborn.pydata.org). For those of you using Python, this is the place to start for learning more about `seaborn`. The page includes beginner resources and links to advanced resources. The page also includes examples that are great to browse. The gallery on this page is especially helpful when trying to think about how to visualize data.

- [*ggplot2: Elegant Graphics for Data Analysis*](https://ggplot2-book.org), 3rd edition, by Hadley Wickham et al. (Springer). The third edition is currently under development and accessible online. This book explains how `ggplot2` works in great detail but also provides a good method for thinking about plotting data using words. You can become an expert in `ggplot2` by reading this book while analyzing and tweaking each line of code presented. But this is not necessarily an easy route.
