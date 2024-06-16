# Appendix C. Data-Wrangling Fundamentals

> Tidy datasets are all alike, but every messy dataset is messy in its own way.
>
> Hadley Wickham

This appendix focuses on some basics of *data wrangling*, or the process of formatting and cleaning data prior to using it. We include some common but sometimes confusing tools we use on a regular basis. We need a large toolbox, because, as noted by Hadley Wickham, each messy data has its own pathologies. For a more in-depth side-by comparison of Python and R, check out the appendix in <a href="https://learning.oreilly.com/library/view/python-and-r/9781492093398" class="orm:hideurl"><em>Python and R for the Modern Data Scientist</em></a> by Rick J. Scavetta and Boyan Angelov (O’Reilly, 2021).

###### Note

*Data wrangling* has many synonyms because almost everybody working with data needs to clean it. Other terms include *data cleaning*, *data formatting*, *data tidying*, *data transformation*, *data manipulation*, *data munging*, and *data mutating*. Basically, people use various terms, so don’t be surprised if you see different terms in different sources. Also, in our experience, people inconsistently use these terms. The key take-home is that you’ll need to clean, format, transform, or otherwise change your own data at some point. Hence, we included this appendix.

# Logic Operators

Logic operators are the same across most languages, including Python and R. The upcoming <a href="#tbl-logic" data-type="xref">Table C-1</a> lists some common operators. Explore these operators by creating a vector in R:

```
## R
score <- c(21, 7, 0, 14)
team <- c("GB", "DEN", "KC", "NYJ")
```

Or, create arrays with `numpy` in Python:

```
## Python
import numpy as np
score = np.array([21, 7, 0, 14])
team = np.array(['GB', 'DEN', 'KC', 'NYJ'])
```

###### Warning

Python’s `numpy`’s arrays differ from base Python’s lists and have different behaviors with mathematical functions.

Basic operators are easy to figure out, like `>` for greater than or `<` for less than. For example, you can see which elements are greater than 7 in Python:

```
## Python
score > 7
```

Resulting in:

array(\[ True, False, False, True\])

As you can see, when you use these operators with an array, the operation is performed against each element individually, and all the results are placed into a new array. This is similar to the way the operations work in R, as you will see. For example, with R you can see which elements are less than 15 in R:

```
## R
score < 15
```

Resulting in:

\[1\] FALSE TRUE TRUE TRUE

Less than or equal to, and greater than or equal to, use the equals sign plus the operator: `>=` is greater than or equal to, and `<=` is less than or equal to. For example, compare the next code example to the previous one:

```
## Python
score <= 14
```

Resulting in:

array(\[False, True, True, True\])

Other operators are less obvious. Because we already use `=` to define objects, `==` is used for *equals*. For example, you can find elements of `team` that are equal to `GB`. Make sure you put `team` in quotes (**`"GB"`**). Otherwise, the computer thinks you are trying to use an object named `GB`:

```
## Python
team == "GB"
```

Resulting in:

array(\[ True, False, False, False\])

Using an `in`-type operator is really useful when you have multiple ways to chart a player playing a similar position. For example, `DE` (defensive end), `OLB` (outside linebacker), and `ED` (edge defender) mean similar things in football, and filtering a dataset for all three terms is often something you do in analysis.

In `numpy`, you can do this with the `.isin()` function:

```
## Python
position = np.array(['QB', 'DE', 'OLB', 'ED'])
np.isin(position, ['DE', 'OLB', 'ED'])
```

Resulting in:

array(\[False, True, True, True\])

The `pandas` package has a similar function for dataframes, covered in <a href="#sec-ap-3-sort" data-type="xref">“Filtering and Sorting Data”</a>.

R has a slightly different operator, an `%in%` function:

```
## R
position <- c("QB", "DE", "OLB", "ED")
position %in% c("DE", "OLB", "ED")
```

Resulting in:

\[1\] FALSE TRUE TRUE TRUE

When using `%in%`, be careful with the order. For example, compare `position %in% c("DE", "OLB", "ED")` from the previous example with `c("DE", "OLB", "ED") %in% position`:

```
## R
c("DE", "OLB", "ED") %in% position
```

Resulting in:

\[1\] TRUE TRUE TRUE

###### Tip

Using `in` operators can be hard. We will often grab a test subset of our data to make sure our code works as expected. More broadly, do not trust your code until you have convinced yourself that your code works as expected. Casually, use `print()` statements to peek at your code and make sure it does what you think it is doing. We do this for one-off projects. Formally, unit-testing exists as a method to test code. Python comes with the `unittest` package, and R has the `testthat` package for formal testing. We use unit-testing on code we plan to reuse or import code when failure has a large cost.

You can also string together operators by using the *and* operator (`&`) or the *or* operator (`|`). Using multiple operators requires the values to be in order as pairs. Our example implies that `score` corresponds to `team`. Both vectors are of length 4 in our examples.

For example, you can see which entries are greater than or equal to `7` for `score` and have a `team` value of `DEN`. When working with the `numpy` arrays, you need to use the `where()` function, but this logic will be the same and use similar notation with `pandas` later in this chapter. The results reveal which entry meets the criteria:

```
## Python
np.where((score >= 7) & (team == "DEN"))
```

Resulting in:

(array(\[1\]),)

You can also use an or operator for a similar comparison to see which values of `score` are greater than `7` *or* which values of `team` are equal to `DEN`:

```
## R
score > 7 | team == "DEN"
```

Resulting in:

\[1\] TRUE TRUE FALSE TRUE

You can string together multiple conditions with parentheses. For example, you can see what has `score` values greater than or equal to `7` *and* `team` equal to `DEN` *or* `score` equal to `0`:

```
## Python
np.where((score >= 7) & (team == "DEN") | (score == 0))
```

Resulting in:

(array(\[1, 2\]),)

Likewise, similar notation may be used in R:

```
## R
(score >= 7 & team == "DEN") | (score == 0)
```

Resulting in:

\[1\] FALSE TRUE TRUE FALSE

<table id="tbl-logic" style="width: 100%">
<caption>Table C-1. Common logical operators. <sup><a href="app03.html#id1131" id="id1131-marker" data-type="noteref">a</a></sup></caption>
<thead>
<tr>
<th>Symbol</th>
<th>Example</th>
<th>Name</th>
<th>Question</th>
</tr>
</thead>
<tbody>
<tr>
<td><p><code>==</code></p></td>
<td><p><code>score == 2</code></p></td>
<td><p>Equals</p></td>
<td><p>Is <code>score</code> equal to <code>2</code>?</p></td>
</tr>
<tr>
<td><p><code>!=</code></p></td>
<td><p><code>score != 2</code></p></td>
<td><p>Not equals</p></td>
<td><p>Is <code>score</code> not equal to <code>2</code>?</p></td>
</tr>
<tr>
<td><p><code>&gt;</code></p></td>
<td><p><code>score &gt; 2</code></p></td>
<td><p>Greater than</p></td>
<td><p>Is <code>score</code> greater than <code>2</code>?</p></td>
</tr>
<tr>
<td><p><code>&lt;</code></p></td>
<td><p><code>score &lt; 2</code></p></td>
<td><p>Less than</p></td>
<td><p>Is <code>score</code> less than <code>2</code>?</p></td>
</tr>
<tr>
<td><p><code>&gt;=</code></p></td>
<td><p><code>score &gt;= 2</code></p></td>
<td><p>Greater than or equal to</p></td>
<td><p>Is <code>score</code> greater than or equal to <code>2</code>?</p></td>
</tr>
<tr>
<td><p><code>&lt;=</code></p></td>
<td><p><code>score &lt;= 2</code></p></td>
<td><p>Less than or equal to</p></td>
<td><p>Is <code>score</code> less than or equal to <code>2</code>?</p></td>
</tr>
<tr>
<td><p><code>|</code></p></td>
<td><p><code>(score &gt; 2) | (team =="GB")</code></p></td>
<td><p>Or</p></td>
<td><p>Is <code>score</code> less than <code>2</code>, or <code>team</code> equal to GB?</p></td>
</tr>
<tr>
<td><p><code>&amp;</code></p></td>
<td><p><code>(score &gt; 2) &amp; (team =="GB")</code></p></td>
<td><p>And</p></td>
<td><p>Is <code>score</code> less than <code>2</code>, and <code>team</code> equal to <code>GB</code>?</p></td>
</tr>
</tbody>
<tbody>
<tr class="footnotes">
<td colspan="4"><p><sup><a href="app03.html#id1131-marker">a</a></sup> <code>pandas</code> sometimes uses <code>~</code> rather than <code>!</code> for <em>not</em> in some situations.</p></td>
</tr>
</tbody>
</table>

# Filtering and Sorting Data

In the previous section, you learned about logical operators. These functions serve as the foundation of filtering data. In fact, when we get stuck with filtering, we often build small test cases like the ones in <a href="#sec-logic-ops" data-type="xref">“Logic Operators”</a> to make sure we understand our data and the way our filters work (or, as is sometimes the case, do not work).

###### Tip

Filtering can be hard. Start small and build complexity into your filtering commands. Keep adding details until you are able to solve your problem. Sometimes you might need to use two or more smaller filters rather than one grand filter operation. This is OK. Get your code working before worrying about optimization.

You will work with the Green Bay–Detroit data from the second week of the 2020 season. First, read in the data and do a simple filter to look at plays that had a yards-after-catch value greater than 15 yards to get an idea of where some big plays were generated.

In R, load the `tidyverse` and `nflfastR` packages and then load the data for 2020:

```
## R
library(tidyverse)
library(nflfastR)

# Load all data
pbp_r <- load_pbp(2020)
```

In Python, import the `pandas`, `numpy`, and `nfl_data_py` packages and then load the data for 2020:

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

In R, use the `filter()` function next. The first argument into filter is `data`. The second argument is the `filter` criteria. Filter out the Detroit at Green Bay game and select some passing columns:

```
# Filter out game data
gb_det_2020_r_pass <-
    pbp_r |>
    filter(home_team == 'GB' & away_team == 'DET') |>
    select(posteam, yards_after_catch, air_yards,
           pass_location, qb_scramble)
```

Next, `filter()` the plays with `yards_after_catch` that were greater than `15`:

```
gb_det_2020_r_pass |>
filter(yards_after_catch > 15)
```

Resulting in:

\# A tibble: 5 × 5 posteam yards_after_catch air_yards pass_location qb_scramble \<chr\> \<dbl\> \<dbl\> \<chr\> \<dbl\> 1 DET 16 13 left 0 2 GB 19 3 right 0 3 GB 19 6 right 0 4 DET 16 1 middle 0 5 DET 20 16 middle 0

###### Tip

With R and Python, you do not always need to use argument names. Instead, the languages match arguments with their predefined order. This order is listed in the help files. For example, with `gb_det_2020_r_pass |> filter(yards_after_catch > 15)`, you could have written `gb_det_2020_r_pass |> filter(filter = yards_after_catch > 15)`. We usually define argument names for more complex functions or when we want to be clear. It is better to err on the side of being explicit and use the argument names, because doing this makes your code easier to read.

Notice in this example that plays that generated a lot of yards after the catch come in many shapes and sizes, including short throws with 1 yard in the air, and longer throws with 16 yards in the air. You can also filter with multiple arguments by using the “and” operator, `&`. For example, you can filter by yards after catch being greater than 15 and Detroit on offense:

```
## R
gb_det_2020_r_pass  |>
filter(yards_after_catch > 15 & posteam == "DET")
```

Resulting in:

\# A tibble: 3 × 5 posteam yards_after_catch air_yards pass_location qb_scramble \<chr\> \<dbl\> \<dbl\> \<chr\> \<dbl\> 1 DET 16 13 left 0 2 DET 16 1 middle 0 3 DET 20 16 middle 0

However, what if you want to look at plays with yards after catch being greater than 15 yards or air yards being greater than 20 yards and Detroit the offensive team? If you try `yards_after_catch > 15 | air_yards > 20 & posteam == "DET"` in the filter, you get results with both Green Bay and Detroit rather than only Detroit. This is because the order of the operations is different than you intended:

```
## R
gb_det_2020_r_pass |>
filter(yards_after_catch > 15 | air_yards > 20 &
       posteam == "DET")
```

Resulting in:

\# A tibble: 9 × 5 posteam yards_after_catch air_yards pass_location qb_scramble \<chr\> \<dbl\> \<dbl\> \<chr\> \<dbl\> 1 DET 16 13 left 0 2 GB 19 3 right 0 3 DET NA 28 left 0 4 DET NA 28 right 0 5 GB 19 6 right 0 6 DET 16 1 middle 0 7 DET 0 24 right 0 8 DET 20 16 middle 0 9 DET NA 50 left 0

You get all plays with yards after catching being greater than 15 or all plays with yards greater than 20 and Detroit starting with possession of the ball. Instead, add a set of parentheses to the filter: `(yards_after_catch > 15 | air_yards > 20) & posteam == "DET"`.

###### Warning

The *order of operations* refers to the way people perform math functions and computers evaluate code. The key takeaway is that both order and grouping of functions changes the output. For example, `1 + 2 x 3 = 1 + 6 = 7` is different from `(1 + 2) x 3 = 3 x 3 = 9`. When you combine operators, the default order of operations sometimes leads to unexpected outcomes, as in the previous example you expected to filter out `GB` but did not. To avoid this type of confusion, parentheses help you explicitly choose the order that you intend.

The use of parentheses in both coding and mathematics align, so the order of operations starts with the innermost set of parentheses and then moves outward:

```
## R
gb_det_2020_r_pass |>
filter((yards_after_catch > 15 | air_yards > 20) &
        posteam == "DET")
```

Resulting in:

\# A tibble: 7 × 5 posteam yards_after_catch air_yards pass_location qb_scramble \<chr\> \<dbl\> \<dbl\> \<chr\> \<dbl\> 1 DET 16 13 left 0 2 DET NA 28 left 0 3 DET NA 28 right 0 4 DET 16 1 middle 0 5 DET 0 24 right 0 6 DET 20 16 middle 0 7 DET NA 50 left 0

You can also change the filter to look at only possession teams that are not Detroit by using the “not-equal-to” operator, `!=.` In this case, the “not-equal-to” operator gives you Green Bay’s admissible offensive plays, but this would not always be the case. For example, if you were working with season long data with all teams, the “not-equal-to” operator would give you data for the 31 other NFL teams:

```
## R
gb_det_2020_r_pass |>
filter((yards_after_catch > 15 | air_yards > 20) &
       posteam != "DET")
```

Resulting in:

\# A tibble: 8 × 5 posteam yards_after_catch air_yards pass_location qb_scramble \<chr\> \<dbl\> \<dbl\> \<chr\> \<dbl\> 1 GB NA 26 left 0 2 GB NA 25 left 0 3 GB 19 3 right 0 4 GB NA 24 right 0 5 GB 4 26 right 0 6 GB NA 28 left 0 7 GB 19 6 right 0 8 GB 7 34 right 0

In Python with `pandas`, filtering is done with similar logical structure to the `tidyverse` in R, but with different syntax. First, Python uses a `.query()` function. Second, the logical operator is inside quotes:

```
## Python
gb_det_2020_py_pass = \
  pbp_py\
    .query("home_team == 'GB' & away_team == 'DET'")\
      [["posteam", "yards_after_catch","air_yards",
        "pass_location", "qb_scramble"]]

print(gb_det_2020_py_pass.query("yards_after_catch > 15"))
```

Resulting in:

posteam yards_after_catch air_yards pass_location qb_scramble 4034 DET 16.0 13.0 left 0.0 4077 GB 19.0 3.0 right 0.0 4156 GB 19.0 6.0 right 0.0 4171 DET 16.0 1.0 middle 0.0 4199 DET 20.0 16.0 middle 0.0

Notice that the or operator, `|`, works the same with both languages:

```
## Python
print(gb_det_2020_py_pass.query("yards_after_catch > 15 | air_yards > 20"))
```

Resulting in:

posteam yards_after_catch air_yards pass_location qb_scramble 4034 DET 16.0 13.0 left 0.0 4051 GB NaN 26.0 left 0.0 4055 GB NaN 25.0 left 0.0 4077 GB 19.0 3.0 right 0.0 4089 DET NaN 28.0 left 0.0 4090 DET NaN 28.0 right 0.0 4104 GB NaN 24.0 right 0.0 4138 GB 4.0 26.0 right 0.0 4142 GB NaN 28.0 left 0.0 4156 GB 19.0 6.0 right 0.0 4171 DET 16.0 1.0 middle 0.0 4176 DET 0.0 24.0 right 0.0 4182 GB 7.0 34.0 right 0.0 4199 DET 20.0 16.0 middle 0.0 4203 DET NaN 50.0 left 0.0

###### Warning

In R or Python, you can use single quotes (`'`) or double quotes (`"`). When using functions such as `.query()` in Python, you see why the languages contain two approaches for quoting. You could use `"posteam == 'DET'"` or `'posteam == "DET"'`. The languages do not care if you use single or double quotes, but you need to be consistent within the same function call.

In Python, when your code gets too long to easily read on a line, you need a backslash (`\`), for Python to understand the line break. This is because Python treats whitespace as a special type of code, whereas R usually treats *whitespace*, such as spaces, indentations, or line breaks, simply as aesthetic. To a novice, this part of Python can be frustrating, but the use of whitespace is a beautiful part of the language once you gain experience to appreciate it.

Next, look at the use of parentheses with the `or` operator and the `and` operator, just as in R:

``` nb
print(gb_det_2020_py_pass.query("(yards_after_catch > 15 | \
                               air_yards > 20) & \
                               posteam == 'DET'"))
```

Resulting in:

posteam yards_after_catch air_yards pass_location qb_scramble 4034 DET 16.0 13.0 left 0.0 4089 DET NaN 28.0 left 0.0 4090 DET NaN 28.0 right 0.0 4171 DET 16.0 1.0 middle 0.0 4176 DET 0.0 24.0 right 0.0 4199 DET 20.0 16.0 middle 0.0 4203 DET NaN 50.0 left 0.0

# Cleaning

Having accurate data is important for sports analytics, as the edge in sports like football can be as little as one or two percentage points over your opponents, the sportsbook, or other players in fantasy football. Cleaning data by hand via programs such as Excel can be tedious and leaves no log indicating which values were changed. Also, fixing one or two systematic errors by hand can easily be done with Excel, but fixing or reformatting thousands of cells in Excel would be difficult and time-consuming. Luckily, you can use scripting to help you clean data.

###### Note

When estimating which team will win a game, the *edge* refers to the difference between the predictor’s estimated probability and the market’s estimated probability (plus the book’s commission, or vigorish). For example, if sportsbooks are offering the Minnesota Vikings at a price of 2–1 to win a game against the Green Bay Packers in Lambeau Field, they are saying that to bet the Vikings, you need to believe that they have more than a 1 / (2 + 1) x 100% = 33.3% chance to win the game. If you make Vikings 36% to win the game, you have a 3% edge betting the Vikings. As information and the synthesizing of information have become more prevalent, edges have become smaller (as the markets have become more efficient). Professional bettors are always in search of better data and better ways to synthesize data, to outrun the increasingly efficient markets they play in.

Consider this example dataframe in `pandas`:

```
wrong_number = \
  pd.DataFrame({"col1": ["a", "b"],
                "col2": ["1O", "12"],
                "col3": [2, 44]})
```

Notice that `col2` has a `1O` (“one-oh”) rather than a `10` (“one-zero,” or ten). These types of mistakes are fairly common in hand-entered data. This may be fixed using code.

###### Note

Both R and Python allow you to access dataframes by using a coordinate-like system with rows as the first entry and columns as the second entry. Think of this like a game of Battleship or Bingo, when people call out cells like *A4* or *B2*. The `pandas` package has `.loc[]` to access rows or columns by names. For example, to access the first value in the `posteam` column of the play-by-play data, run `pbp_py.loc[1, "posteam"]` (`1` is the row name or index, and `posteam` is the column name). To access the first row of the first column, run `print(pbp_py.iloc[1, 0])`. Compare these two methods. What column is `0`? It is better to use filters or explicit names. This way, if your data changes, you call the correct cell. Also, this way, future you and other people will know why you are trying to access specific cells.

Use the locate function, `.loc()`, to locate the wrong cell. Also, select the column, `col2`. Last, replace the wrong value with a `10` (ten):

```
## Python
wrong_number.loc[wrong_number.col2 == "1O", "col2"] = 10
```

Look at the dataframe’s information, though, and you will see that `col2` is still an object rather than a number or integer:

```
## Python
wrong_number.info()
```

Resulting in:

\<class 'pandas.core.frame.DataFrame'\> RangeIndex: 2 entries, 0 to 1 Data columns (total 3 columns): \# Column Non-Null Count Dtype --- ------ -------------- ----- 0 col1 2 non-null object 1 col2 2 non-null object 2 col3 2 non-null int64 dtypes: int64(1), object(2) memory usage: 176.0+ bytes

###### Warning

Both R and Python usually require users to save data files as outputs after editing (a counter-example being the `inplace=True` option in some `pandas` functions). Otherwise, the computer will not save your changes. Failure to update or save objects can cost you hours of debugging code, as we have learned from our own experiences.

Change this by using the `to_numeric()` function from `pandas` and then look at the information for the dataframe. Next, save the results to `col2` and rewrite the old data. If you skip this step, the computer will not save your edits:

```
## Python
wrong_number["col2"] = \
    pd.to_numeric(wrong_number["col2"])
wrong_number.info()
```

Resulting in:

\<class 'pandas.core.frame.DataFrame'\> RangeIndex: 2 entries, 0 to 1 Data columns (total 3 columns): \# Column Non-Null Count Dtype --- ------ -------------- ----- 0 col1 2 non-null object 1 col2 2 non-null int64 2 col3 2 non-null int64 dtypes: int64(2), object(1) memory usage: 176.0+ bytes

Notice that now the column has been changed to an integer.

If you want to save these changes for later, you can use the `to_csv()` function to save the outputs. Generally, you will want to use a new filename that makes sense to you now, to others, and to your future self. Because the dataframe does not have meaningful row names or an index, tell `pandas` to not save this information by using `index=False`:

```
## Python
wrong_number.to_csv("wrong_number_corrected.csv", index = False)
```

R uses slightly different syntax. First, use the `mutate()` function to change the column. Next, tell R to change `col2` by using `col2 = ...`. Then use the `ifelse()` function to tell R to change `col2` if it is equal to `1O` (“one-oh”) to be `10` (“one-zero,” or ten), or to use the current value in `col2`:

```
## R
wrong_number <-
  tibble(col1 = c("a", "b"),
         col2 = c("1O", "12"),
         col3 = c(2, 44))
wrong_number <-
  wrong_number |>
  mutate(col2 = ifelse(col2 == "1O", 10, col2))
```

Next, just as in Python, change `col2` to be numeric. In R, use the `as.numeric()` function. Then look at the dataframe structure by using `str()`:

```
## R
wrong_number <- mutate(wrong_number, col2 = as.numeric(col2))
str(wrong_number)
```

Resulting in:

tibble \[2 × 3\] (S3: tbl_df/tbl/data.frame) \$ col1: chr \[1:2\] "a" "b" \$ col2: num \[1:2\] 10 12 \$ col3: num \[1:2\] 2 44

Finally, just as in Python, save the file by using a name that makes sense to both the current you and future you. Hopefully, this name makes sense to other people. Creating names can be one of the most difficult parts of programming. With R, use the `write_csv()` function:

```
## R
write_csv(x = wrong_number,
          file = "wrong_numbers_corrected.csv")
```

###### Warning

Python uses `False` for the logical false and `True` for the logical true. R uses `FALSE` for false and `TRUE` for true. If you are switching between the languages, be careful with these terms.

# Piping in R

With programming, sometimes you want to pass outputs from one function to another without needing to save the intermediate outputs. In mathematics, this is called *composition*, and while teaching college math classes, Eric observed this to be one of the more misunderstood procedures because of the confusing notation. In computer programming, this is called *piping* because outputs are piped from one function to another.

Luckily, R has allowed composition through the piping operators with the `tidyverse` that has a pipe function, `%>%`. As of R version 4.1, released in 2021, base R also now includes a `|>` pipe operator. We use the base R pipe operator in this book, but you may see both “in the wild” when looking at other people’s code or websites.

###### Note

The `tidyverse` pipe allows piping to any function’s input option by using a period. This period is optional with the `tidyverse` pipe. And the `tidyverse` pipe will, by default, use the first function input with piping. For example, you might code `read_csv("my_file.scv") %>% func(x = col1, data = .)`, or `read_csv("my_file.scv") %>% function(col1)`. With `|>`, you can pass to only the first input; thus you would need to define all inputs prior to the one you are piping (in this case, `data`). With the `|>` pipe, your code would be written as `read_csv("my_file.scv") |> function(x = col1)`.

###### Warning

Any reference material can become dated, especially online tutorials. The piping example demonstrates how any tutorial created before R 4.1 would not include the new piping notation. Thus, when using a tutorial, examine when the material was written and ensure that you can re-create a tutorial before applying it to your problem. And, when using sites such as [StackOverflow](https://stackoverflow.com), we look at several of the top answers and questions to make sure the accepted answer has not become outdated as languages change. The best answer in 2013 may not be the best answer in 2023.

We cover piping here for two reasons. First, you will likely see it when you start to look at other people’s code as you teach yourself. Second, piping allows you to be more efficient with coding once you get the hang of it.

# Checking and Cleaning Data for Outliers

Data often contains errors. Perhaps people collecting or entering the data made a mistake. Or, maybe an instrument like a weather station malfunctioned. Sometimes, computer systems corrupt or otherwise change files. In football, quite often there will be errors in things like number of air yards generated, yards after the catch earned, or even the player targeted. Resolving these errors quickly, and often through data wrangling, is a required process of learning more about the game. <a href="ch02.html#sec-EDA-stable" data-type="xref">Chapter 2</a> presented tools to help you catch these errors.

You’ll go through and find and remove an outlier with both languages. Revisiting the `wrong_number` dataframe, perhaps `col3` should be only single digits. The `summary()` function would help you see this value is wrong in R:

```
## R
wrong_number |>
  summary()
```

Resulting in:

col1 col2 col3 Length:2 Min. :10.0 Min. : 2.0 Class :character 1st Qu.:10.5 1st Qu.:12.5 Mode :character Median :11.0 Median :23.0 Mean :11.0 Mean :23.0 3rd Qu.:11.5 3rd Qu.:33.5 Max. :12.0 Max. :44.0

Likewise, the `describe()` function in Python would help you catch an outlier:

```
## Python
wrong_number.describe()
```

Resulting in:

col2 col3 count 2.000000 2.000000 mean 11.000000 23.000000 std 1.414214 29.698485 min 10.000000 2.000000 25% 10.500000 12.500000 50% 11.000000 23.000000 75% 11.500000 33.500000 max 12.000000 44.000000

Using the tools covered in the previous section, you can remove the outlier.

# Merging Multiple Datasets

Sometimes you will need to combine datasets. For example, often you will want to adjust the results of a play—say, the number of passing yards—by the weather in which the game was played. Both `pandas` and the `tidyverse` readily allow merging datasets. For example, perhaps you have team and game data you want to merge from both datasets. Or, maybe you want to merge weather data to the play-by-play data.

For this example, create two dataframes and then merge them. One dataframe will be city information that contains the teams’ names and cities. The other will be a schedule. We have you create a small example for multiple reasons. First, a small toy dataset is easier to handle and see, compared to a large dataset. Second, we often create toy datasets to make sure our merges work.

###### Tip

When learning something new (like merging dataframes), start with a small example you understand. The small example will be easier to debug, fail faster, and understand compared to a large example or actual dataset.

You might be wondering, why merge these dataframes? We often have to do merges like this when summarizing data because we want or need a prettier name. Likewise, we often need to change names for plots. Next, you may be wondering, why not type these values into a spreadsheet? Manually typing can be tedious and error prone. Plus, doing tens, hundreds, or even thousands of games would take a long time to type.

As you create the dataframes in R, remember that each column you create is a vector:

```
## R
library(tidyverse)
city_data <- data.frame(city = c("DET", "GB", "HOU"),
                        team = c("Lions", "Packers", "Texans"))
schedule <- data.frame(home = c("GB", "DET"),
                       away = c("DET", "HOU"))
```

As you create the dataframes in Python, remember that the `DataFrame()` function uses a dictionary to create columns and elements in the columns:

```
## Python
import pandas as pd
city_data = \
    pd.DataFrame({"city" : ["DET", "GB", "HOU"],
                  "team" : ["Lions", "Packers", "Texans"]})
schedule = \
    pd.DataFrame({"home" : ["GB", "DET"],
                  "away" : ["DET", "HOU"]})
```

Now that you have the datasets, use them to explore various merges. Both `pandas` and the `tidyverse` base their merge functions on SQL. The join functions require a common, shared key or multiple keys between the two dataframes. In the `tidyverse`, this argument is called *by*—for example, joining city and schedule dataframes by *team name* and *home* team columns. In `pandas`, this argument is called *on*—for example, joining city and schedule dataframes on *team name* and *home* team columns.

We use four main joins on a regular basis, and these are included with the `tidyverse` and `pandas`. The `pandas` package has both a `merge()` and a `join()` function. The `merge()` function contains almost everything that `join()` does, plus some more, so we will include only `merge()` here. With both Python and R, there are two datasets, a left one and a right one. The *left dataset* is the one on the left (or the first dataset), and the *right dataset* is the one on the right (or the second dataset).

For the example, you want to create a new dataframe that includes both the schedule and the teams’ names. Use this to explore the various types of joins. Think of this example as the fairy tale of Goldilocks and the four joins (based on the original story of [*Goldilocks and the Three Bears*](https://oreil.ly/iHnci)). Rather than a girl trying the bears’ beds and food, you’ll be exploring data joins, listed in <a href="#tbl-apd-join" data-type="xref">Table C-2</a>. This problem has two steps. The first step is to add in the home team’s name. The second step is to add in the away team’s name. At the end, we will show you the complete workflow because it also involves renaming columns.

###### Tip

Football analytics, like the broader field of data science, usually involves breaking big jobs into smaller jobs. As you become more experienced, you will become better at seeing the small steps and knowing where and how to reuse them. When faced with intimidating problems, we break them into smaller steps that we can readily solve. Often our first step is to write or draw out our coding needs, much as you may have outlined a paper in high school or college before writing the paper.

First, examine a *full*, or *outer, join*. This merges both dataframes based on all values in both dataframes’ keys. If one or both keys contain values not found in the other dataset, these are replaced by missing values (`NA` in R, `NaN` in Python). For both languages, `schedule` will be your left dataframe, and `city_data` will be your right dataframe. Because both dataframes do not have the same key (or, specifically, the column with the same names), the computer needs to know how to pair up the keys (specifically, which columns link the two dataframes).

In R, use the `full_join()` function. Put `schedule` in first, followed by `city_data`. Tell R to *join* the dataframes by using `home` as the left key matching up with `city` as the right key:

```
## R
print(
  full_join(schedule, city_data,
  by = c("home" = "city"))
  )
```

Resulting in:

home away team 1 GB DET Packers 2 DET HOU Lions 3 HOU \<NA\> Texans

Notice that you get three entries because the `city_data` has three rows. The missing value is replaced by `NA`. Notice that R dropped the duplicate column and has only three columns.

In Python, use the `.merge()` function on the `schedule` dataframe. And notice that `schedule` is on the left. The first argument is `city_data`. Tell `pandas` how to merge—specifically, an `outer` merge. Then tell `pandas` to use `home` as the left key and `city` as the right key:

```
## Python
print(schedule.merge(city_data, how = "outer",
                     left_on = "home", right_on = "city"))
```

Resulting in:

home away city team 0 GB DET GB Packers 1 DET HOU DET Lions 2 NaN NaN HOU Texans

Notice `pandas` kept all four columns. Also note that both `home` and `away` are `NaN` for the new dataframe.

###### Note

This example demonstrates how Python tends to be an object-orientated language, and R tends to be a functional language. Python uses `.merge()` as an object contained by the dataframe `schedule`. R uses a `full_join()` as a function on two objects, `schedule` and `city_data`. Although R and Python both contain object-orientated and functional features, this example nicely demonstrates the underlying philosophies of the two languages.

Think of this distinction of language types similar to the way some football teams are built for a run offense and others for a pass offense. Under certain circumstances, one language can be better than the other, but usually both contain the tools for a given job. Advanced data scientists recognize these trade-offs between languages and will switch languages to fit their needs.

Next, do an *inner join*. This joins only the shared key values. Whereas an outer join may possibly grow dataframes, an inner join shrinks dataframes. The R syntax is very similar to the previous example; only the function name changes. However, notice that the output has only three values:

```
## R
print(inner_join(schedule, city_data, by = c("home" = "city")))
```

Resulting in:

home away team 1 GB DET Packers 2 DET HOU Lions

Like R, the Python code is similar. In Python, use the same function, but a different `how` argument:

```
## Python
print(schedule.merge(city_data, how = "inner",
                     left_on = "home", right_on = "city"))
```

Resulting in:

home away city team 0 GB DET GB Packers 1 DET HOU DET Lions

Next, do a *right join*. The right join keeps all the values from the right dataframe. For this specific case, the outputs are the same as the outer join. This is an artifact of the example and may not always be the case. With R, simply change the function name to `right_join()`:

```
## R
print(right_join(schedule, city_data, by = c("home" = "city")))
```

Resulting in:

home away team 1 GB DET Packers 2 DET HOU Lions 3 HOU \<NA\> Texans

With Python, change `how` to `right`:

```
## Python
print(schedule.merge(city_data, how = "right",
                     left_on = "home", right_on = "city"))
```

Resulting in:

home away city team 0 DET HOU DET Lions 1 GB DET GB Packers 2 NaN NaN HOU Texans

A *left join* is the opposite of a right join. This keeps all the values from the left dataframe. In fact, rather than switching the function, you could switch the order of inputs. Consider merging dataframes `A` and `B` in Python that share a common column, `key`:

```
## Python
A.merge(B, how = "left", on = "key")
```

This could also be written in reverse:

```
## Python
B.merge(A, how = "right", on = "key")
```

Here is what the R code and output look like:

```
## R
print(left_join(schedule, city_data, by = c("home" = "city")))
```

Resulting in:

home away team 1 GB DET Packers 2 DET HOU Lions

The Python code also looks similar to the right join. For both outputs, the left join was the same as the inner join. This is an artifact of the example choice and will not always be the case. Here, the left dataframe had fewer rows than the right dataframe. Hence, this occurred in the example:

```
## Python
print(schedule.merge(city_data, how = "left",
                     left_on = "home", right_on = "city"))
```

Resulting in:

home away city team 0 GB DET GB Packers 1 DET HOU DET Lions

| Name | Brief description | tidyverse function | pandas `merge()` syntax |
|----|----|----|----|
| Full/outer join | Merges based on all key values | `full_join(left_data, right_data)` | `left_data.merge(right_data, how = "outer"` |
| Inner join | Merges based only on shared key values | `inner_join(left_data, right_data)` | `left_data.merge(right_data, how = "inner"` |
| Left join | Merges based only on *left* data’s key values | `left_join(left_data, right_data)` | `left_data.merge(right_data, how = "left"` |
| Right join | Merges based only on *right* data’s key values | `right_join(left_data, right_data)` | `left_data.merge(right_data, how = "right"` |

Table C-2. Common join types in R and Python. {#tbl-apd-join style="width: 100%"}

Let’s return to the initial problem: “How do you merge the dataframe to include the team names for both the home and away teams?”

Multiple solutions exist, as is often the case with programming. We use multiple left joins because we think about adding data to a schedule and putting this dataframe on the left. However, you might think about the problem differently, which is OK. In fact, you might be able to think about and come up with a better way to do this that is either quicker, easier to read, or uses less code.

###### Note

Unlike high school math, both statistics and coding often have no single best or right way to do something. Instead, many unique solutions exist. Some people play a game called *code golf*, in which they try to solve a problem by using the fewest lines of code; see, for example, the [Stack Exchange Code Golf page](https://codegolf.stackexchange.com). But the fewest lines of code is usually not the best answer in real life. Instead, focus on writing code that you and other people can read later. Also new tools such as GitHub’s Copilot can help you see and compare methods for coding the same task.

So, we will use a series of left joins (although we could also do everything in reverse, using right joins). Here is our step-by-step solution:

1.  Merge in for the home team.

2.  Rename column in R, rename and delete column in Python.

3.  Merge in for away team. This step is needed for clarity and to avoid duplicate names.

4.  Rename columns in R; rename and delete columns in Python.

5.  Make sure the output is saved to a new dataframe, `schedule_name`.

The following are some notes about how and why we use these specific steps. Whether we merged by the away or home order is not important, and we arbitrarily selected order. We needed to rename columns to avoid duplicate names later and to keep column names clear. The importance of this will become evident when you have to clean up your own mess or somebody else’s messy code! Lastly, we encourage you to start with one line of code and keep adding more code until you understand the big picture. That’s how we constructed this example.

With the R example, use piping to avoid rewriting objects as you did for the Python example. First, take the `schedule` dataframe and then left-join to the `city_data`. Tell R to join by (or match) the `home` column to the `city` column. Then rename the `team` column to `home_team`. This helps us keep the team columns straight in the final dataframe. Then repeat these steps and join the away team data:

```
## R
schedule_name <-
    schedule |>
    left_join(city_data, by = c("home" = "city"))  |>
    rename(home_team = team)  |>
    left_join(city_data, by = c("away" = "city"))  |>
    rename(away_team = team)
print(schedule_name)
```

Resulting in:

home away home_team away_team 1 GB DET Packers Lions 2 DET HOU Lions Texans

With Python, create temporary objects rather than piping. This is because the `pandas` piping is not as intuitive to us and requires writing custom functions, something beyond the scope of this book. Furthermore, some people like writing out code to see all the steps, and we want to show you a second technique for this example.

In Python, first do a left merge. Tell Python we use `home` for the left merge on and `city` for the right merge on. Then rename the `team` column to `home_team`. The `pandas` `rename()` function requires a dictionary as an input. Then, tell `pandas` to remove, or `.drop()`, the city column to avoid confusion later. Then repeat these steps for the away team:

```
## Python
step_1 = schedule.merge(city_data, how = "left",
                        left_on = "home", right_on = "city")
step_2 = step_1.rename(columns =
                       {"team": "home_team"}).drop(columns = "city")
step_3 = step_2.merge(city_data, how = "left",
                      left_on = "away", right_on = "city")
schedule_name = step_3.rename(columns =
                              {"team": "home_team"}).drop(columns = "city")
print(schedule_name)
```

Resulting in:

home away home_team home_team 0 GB DET Packers Lions 1 DET HOU Lions Texans
