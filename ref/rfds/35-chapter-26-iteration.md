# Chapter 26. Iteration

# Introduction

In this chapter, you’ll learn tools for iteration, repeatedly performing the same action on different objects. Iteration in R generally tends to look rather different from other programming languages because so much of it is implicit and we get it for free. For example, if you want to double a numeric vector `x` in R, you can just write `2 * x`. In most other languages, you’d need to explicitly double each element of `x` using some sort of for loop.

This book has already given you a small but powerful number of tools that perform the same action for multiple “things”:

- <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/facet_grid.html" class="orm:hideurl"><code>facet_grid()</code></a> draw a plot for each subset.
- <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a> plus <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a> computes a summary statistics for each subset.
- <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a> and <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a> create new rows and columns for each element of a list column.

Now it’s time to learn some more general tools, often called *functional programming* tools because they are built around functions that take other functions as inputs. Learning functional programming can easily veer into the abstract, but in this chapter we’ll keep things concrete by focusing on three common tasks: modifying multiple columns, reading multiple files, and saving multiple objects.

## Prerequisites

In this chapter, we’ll focus on tools provided by dplyr and purrr, both core members of the tidyverse. You’ve seen dplyr before, but [purrr](https://oreil.ly/f0HWP) is new. We’re just going to use a couple of purrr functions in this chapter, but it’s a great package to explore as you improve your programming skills:

```
library(tidyverse)
```

# Modifying Multiple Columns

Imagine you have this simple tibble and you want to count the number of observations and compute the median of every column:

```
df <- tibble(
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10)
)
```

You could do it with copy and paste:

```
df |> summarize(
  n = n(),
  a = median(a),
  b = median(b),
  c = median(c),
  d = median(d),
)
#> # A tibble: 1 × 5
#>       n      a      b       c     d
#>   <int>  <dbl>  <dbl>   <dbl> <dbl>
#> 1    10 -0.246 -0.287 -0.0567 0.144
```

That breaks our rule of thumb to never copy and paste more than twice, and you can imagine that this will get tedious if you have tens or even hundreds of columns. Instead, you can use <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>:

```
df |> summarize(
  n = n(),
  across(a:d, median),
)
#> # A tibble: 1 × 5
#>       n      a      b       c     d
#>   <int>  <dbl>  <dbl>   <dbl> <dbl>
#> 1    10 -0.246 -0.287 -0.0567 0.144
```

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> has three particularly important arguments, which we’ll discuss in detail in the following sections. You’ll use the first two every time you use <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>: the first argument, `.cols`, specifies which columns you want to iterate over, and the second argument, `.fns`, specifies what to do with each column. You can use the `.names` argument when you need additional control over the names of output columns, which is particularly important when you use <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> with <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>. We’ll also discuss two important variations, <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_any()</code></a> and <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_all()</code></a>, which work with <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>.

## Selecting Columns with .cols

The first argument to <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, `.cols`, selects the columns to transform. This uses the same specifications as <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="ch03.html#sec-select" data-type="xref">“select()”</a>, so you can use functions such as <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>starts_with()</code></a> and <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>ends_with()</code></a> to select columns based on their name.

There are two additional selection techniques that are particularly useful for <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>: <a href="https://tidyselect.r-lib.org/reference/everything.html" class="orm:hideurl"><code>everything()</code></a> and <a href="https://tidyselect.r-lib.org/reference/where.html" class="orm:hideurl"><code>where()</code></a>. <a href="https://tidyselect.r-lib.org/reference/everything.html" class="orm:hideurl"><code>everything()</code></a> is straightforward: it selects every (nongrouping) column:

```
df <- tibble(
  grp = sample(2, 10, replace = TRUE),
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10)
)

df |> 
  group_by(grp) |> 
  summarize(across(everything(), median))
#> # A tibble: 2 × 5
#>     grp       a       b     c     d
#>   <int>   <dbl>   <dbl> <dbl> <dbl>
#> 1     1 -0.0935 -0.0163 0.363 0.364
#> 2     2  0.312  -0.0576 0.208 0.565
```

Note grouping columns (`grp` here) are not included in <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, because they’re automatically preserved by <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>.

<a href="https://tidyselect.r-lib.org/reference/where.html" class="orm:hideurl"><code>where()</code></a> allows you to select columns based on their type:

`where(is.numeric)`  
Selects all numeric columns.

`where(is.character)`  
Selects all string columns.

`where(is.Date)`  
Selects all date columns.

`where(is.POSIXct)`  
Selects all date-time columns.

`where(is.logical)`  
selects all logical columns.

Just like other selectors, you can combine these with Boolean algebra. For example, `!where(is.numeric)` selects all non-numeric columns, and `starts_with("a") & where(is.logical)` selects all logical columns whose name starts with “a.”

## Calling a Single Function

The second argument to <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> defines how each column will be transformed. In simple cases, as shown, this will be a single existing function. This is a pretty special feature of R: we’re passing one function (`median`, `mean`, `str_flatten`, …) to another function (`across`). This is one of the features that makes R a functional programming language.

It’s important to note that we’re passing this function to <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, so <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> can call it; we’re not calling it ourselves. That means the function name should never be followed by `()`. If you forget, you’ll get an error:

```
df |> 
  group_by(grp) |> 
  summarize(across(everything(), median()))
#> Error in `summarize()`:
#> ℹ In argument: `across(everything(), median())`.
#> Caused by error in `is.factor()`:
#> ! argument "x" is missing, with no default
```

This error arises because you’re calling the function with no input, e.g.:

```
median()
#> Error in is.factor(x): argument "x" is missing, with no default
```

## Calling Multiple Functions

In more complex cases, you might want to supply additional arguments or perform multiple transformations. Let’s motivate this problem with a simple example: what happens if we have some missing values in our data? <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a> propagates those missing values, giving us a suboptimal output:

```
rnorm_na <- function(n, n_na, mean = 0, sd = 1) {
  sample(c(rnorm(n - n_na, mean = mean, sd = sd), rep(NA, n_na)))
}

df_miss <- tibble(
  a = rnorm_na(5, 1),
  b = rnorm_na(5, 1),
  c = rnorm_na(5, 2),
  d = rnorm(5)
)
df_miss |> 
  summarize(
    across(a:d, median),
    n = n()
  )
#> # A tibble: 1 × 5
#>       a     b     c     d     n
#>   <dbl> <dbl> <dbl> <dbl> <int>
#> 1    NA    NA    NA  1.15     5
```

It would be nice if we could pass along `na.rm = TRUE` to <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a> to remove these missing values. To do so, instead of calling <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a> directly, we need to create a new function that calls <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a> with the desired arguments:

```
df_miss |> 
  summarize(
    across(a:d, function(x) median(x, na.rm = TRUE)),
    n = n()
  )
#> # A tibble: 1 × 5
#>       a     b      c     d     n
#>   <dbl> <dbl>  <dbl> <dbl> <int>
#> 1 0.139 -1.11 -0.387  1.15     5
```

This is a little verbose, so R comes with a handy shortcut: for this sort of throwaway (or *anonymous*)<sup><a href="ch26.html#idm44771267612512" id="idm44771267612512-marker" data-type="noteref">1</a></sup> function, you can replace `function` with `\`:<sup><a href="ch26.html#idm44771267610256" id="idm44771267610256-marker" data-type="noteref">2</a></sup>

```
df_miss |> 
  summarize(
    across(a:d, \(x) median(x, na.rm = TRUE)),
    n = n()
  )
```

In either case, <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> effectively expands to the following code:

```
df_miss |> 
  summarize(
    a = median(a, na.rm = TRUE),
    b = median(b, na.rm = TRUE),
    c = median(c, na.rm = TRUE),
    d = median(d, na.rm = TRUE),
    n = n()
  )
```

When we remove the missing values from the <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>, it would be nice to know just how many values were removed. We can find that out by supplying two functions to <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>: one to compute the median and the other to count the missing values. You supply multiple functions by using a named list to `.fns`:

```
df_miss |> 
  summarize(
    across(a:d, list(
      median = \(x) median(x, na.rm = TRUE),
      n_miss = \(x) sum(is.na(x))
    )),
    n = n()
  )
#> # A tibble: 1 × 9
#>   a_median a_n_miss b_median b_n_miss c_median c_n_miss d_median d_n_miss
#>      <dbl>    <int>    <dbl>    <int>    <dbl>    <int>    <dbl>    <int>
#> 1    0.139        1    -1.11        1   -0.387        2     1.15        0
#> # … with 1 more variable: n <int>
```

If you look carefully, you might intuit that the columns are named using a glue specification (<a href="ch14.html#sec-glue" data-type="xref">“str_glue()”</a>) like `{.col}_{.fn}` where `.col` is the name of the original column and `.fn` is the name of the function. That’s not a coincidence! As you’ll learn in the next section, you can use the `.names` argument to supply your own glue spec.

## Column Names

The result of <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> is named according to the specification provided in the `.names` argument. We could specify our own if we wanted the name of the function to come first:<sup><a href="ch26.html#idm44771267377264" id="idm44771267377264-marker" data-type="noteref">3</a></sup>

```
df_miss |> 
  summarize(
    across(
      a:d,
      list(
        median = \(x) median(x, na.rm = TRUE),
        n_miss = \(x) sum(is.na(x))
      ),
      .names = "{.fn}_{.col}"
    ),
    n = n(),
  )
#> # A tibble: 1 × 9
#>   median_a n_miss_a median_b n_miss_b median_c n_miss_c median_d n_miss_d
#>      <dbl>    <int>    <dbl>    <int>    <dbl>    <int>    <dbl>    <int>
#> 1    0.139        1    -1.11        1   -0.387        2     1.15        0
#> # … with 1 more variable: n <int>
```

The `.names` argument is particularly important when you use <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> with <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>. By default, the output of <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> is given the same names as the inputs. This means that <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> in <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> will replace existing columns. For example, here we use <a href="https://dplyr.tidyverse.org/reference/coalesce.html" class="orm:hideurl"><code>coalesce()</code></a> to replace `NA`s with `0`:

```
df_miss |> 
  mutate(
    across(a:d, \(x) coalesce(x, 0))
  )
#> # A tibble: 5 × 4
#>        a      b      c     d
#>    <dbl>  <dbl>  <dbl> <dbl>
#> 1  0.434 -1.25   0     1.60 
#> 2  0     -1.43  -0.297 0.776
#> 3 -0.156 -0.980  0     1.15 
#> 4 -2.61  -0.683 -0.785 2.13 
#> 5  1.11   0     -0.387 0.704
```

If you’d like to instead create new columns, you can use the `.names` argument to give the output new names:

```
df_miss |> 
  mutate(
    across(a:d, \(x) abs(x), .names = "{.col}_abs")
  )
#> # A tibble: 5 × 8
#>        a      b      c     d  a_abs  b_abs  c_abs d_abs
#>    <dbl>  <dbl>  <dbl> <dbl>  <dbl>  <dbl>  <dbl> <dbl>
#> 1  0.434 -1.25  NA     1.60   0.434  1.25  NA     1.60 
#> 2 NA     -1.43  -0.297 0.776 NA      1.43   0.297 0.776
#> 3 -0.156 -0.980 NA     1.15   0.156  0.980 NA     1.15 
#> 4 -2.61  -0.683 -0.785 2.13   2.61   0.683  0.785 2.13 
#> 5  1.11  NA     -0.387 0.704  1.11  NA      0.387 0.704
```

## Filtering

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> is a great match for <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a> and <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>, but it’s more awkward to use with <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>, because you usually combine multiple conditions with either `|` or `&`. It’s clear that <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> can help to create multiple logical columns, but then what? So dplyr provides two variants of <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> called <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_any()</code></a> and <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_all()</code></a>:

```
# same as df_miss |> filter(is.na(a) | is.na(b) | is.na(c) | is.na(d))
df_miss |> filter(if_any(a:d, is.na))
#> # A tibble: 4 × 4
#>        a      b      c     d
#>    <dbl>  <dbl>  <dbl> <dbl>
#> 1  0.434 -1.25  NA     1.60 
#> 2 NA     -1.43  -0.297 0.776
#> 3 -0.156 -0.980 NA     1.15 
#> 4  1.11  NA     -0.387 0.704

# same as df_miss |> filter(is.na(a) & is.na(b) & is.na(c) & is.na(d))
df_miss |> filter(if_all(a:d, is.na))
#> # A tibble: 0 × 4
#> # … with 4 variables: a <dbl>, b <dbl>, c <dbl>, d <dbl>
```

## across() in Functions

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> is particularly useful to program with because it allows you to operate on multiple columns. For example, [Jacob Scott](https://oreil.ly/6vVc4) uses this little helper that wraps a bunch of lubridate functions to expand all date columns into year, month, and day columns:

```
expand_dates <- function(df) {
  df |> 
    mutate(
      across(where(is.Date), list(year = year, month = month, day = mday))
    )
}

df_date <- tibble(
  name = c("Amy", "Bob"),
  date = ymd(c("2009-08-03", "2010-01-16"))
)

df_date |> 
  expand_dates()
#> # A tibble: 2 × 5
#>   name  date       date_year date_month date_day
#>   <chr> <date>         <dbl>      <dbl>    <int>
#> 1 Amy   2009-08-03      2009          8        3
#> 2 Bob   2010-01-16      2010          1       16
```

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> also makes it easy to supply multiple columns in a single argument because the first argument uses tidy-select; you just need to remember to embrace that argument, as we discussed in <a href="ch25.html#sec-embracing" data-type="xref">“When to Embrace?”</a>. For example, this function will compute the means of numeric columns by default. But by supplying the second argument you can choose to summarize just selected columns:

```
summarize_means <- function(df, summary_vars = where(is.numeric)) {
  df |> 
    summarize(
      across({{ summary_vars }}, \(x) mean(x, na.rm = TRUE)),
      n = n()
    )
}
diamonds |> 
  group_by(cut) |> 
  summarize_means()
#> # A tibble: 5 × 9
#>   cut       carat depth table price     x     y     z     n
#>   <ord>     <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <int>
#> 1 Fair      1.05   64.0  59.1 4359.  6.25  6.18  3.98  1610
#> 2 Good      0.849  62.4  58.7 3929.  5.84  5.85  3.64  4906
#> 3 Very Good 0.806  61.8  58.0 3982.  5.74  5.77  3.56 12082
#> 4 Premium   0.892  61.3  58.7 4584.  5.97  5.94  3.65 13791
#> 5 Ideal     0.703  61.7  56.0 3458.  5.51  5.52  3.40 21551

diamonds |> 
  group_by(cut) |> 
  summarize_means(c(carat, x:z))
#> # A tibble: 5 × 6
#>   cut       carat     x     y     z     n
#>   <ord>     <dbl> <dbl> <dbl> <dbl> <int>
#> 1 Fair      1.05   6.25  6.18  3.98  1610
#> 2 Good      0.849  5.84  5.85  3.64  4906
#> 3 Very Good 0.806  5.74  5.77  3.56 12082
#> 4 Premium   0.892  5.97  5.94  3.65 13791
#> 5 Ideal     0.703  5.51  5.52  3.40 21551
```

## Versus pivot_longer()

Before we go on, it’s worth pointing out an interesting connection between <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> and <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a> (<a href="ch05.html#sec-pivoting" data-type="xref">“Lengthening Data”</a>). In many cases, you perform the same calculations by first pivoting the data and then performing the operations by group rather than by column. For example, take this multifunction summary:

```
df |> 
  summarize(across(a:d, list(median = median, mean = mean)))
#> # A tibble: 1 × 8
#>   a_median a_mean b_median b_mean c_median c_mean d_median d_mean
#>      <dbl>  <dbl>    <dbl>  <dbl>    <dbl>  <dbl>    <dbl>  <dbl>
#> 1   0.0380  0.205  -0.0163 0.0910    0.260 0.0716    0.540  0.508
```

We could compute the same values by pivoting longer and then summarizing:

```
long <- df |> 
  pivot_longer(a:d) |> 
  group_by(name) |> 
  summarize(
    median = median(value),
    mean = mean(value)
  )
long
#> # A tibble: 4 × 3
#>   name   median   mean
#>   <chr>   <dbl>  <dbl>
#> 1 a      0.0380 0.205 
#> 2 b     -0.0163 0.0910
#> 3 c      0.260  0.0716
#> 4 d      0.540  0.508
```

And if you wanted the same structure as <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, you could pivot again:

```
long |> 
  pivot_wider(
    names_from = name,
    values_from = c(median, mean),
    names_vary = "slowest",
    names_glue = "{name}_{.value}"
  )
#> # A tibble: 1 × 8
#>   a_median a_mean b_median b_mean c_median c_mean d_median d_mean
#>      <dbl>  <dbl>    <dbl>  <dbl>    <dbl>  <dbl>    <dbl>  <dbl>
#> 1   0.0380  0.205  -0.0163 0.0910    0.260 0.0716    0.540  0.508
```

This is a useful technique to know about because sometimes you’ll hit a problem that’s not currently possible to solve with <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>: when you have groups of columns that you want to compute with simultaneously. For example, imagine that our data frame contains both values and weights and we want to compute a weighted mean:

```
df_paired <- tibble(
  a_val = rnorm(10),
  a_wts = runif(10),
  b_val = rnorm(10),
  b_wts = runif(10),
  c_val = rnorm(10),
  c_wts = runif(10),
  d_val = rnorm(10),
  d_wts = runif(10)
)
```

There’s currently no way to do this with <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>,<sup><a href="ch26.html#idm44771266547392" id="idm44771266547392-marker" data-type="noteref">4</a></sup> but it’s relatively straightforward with <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>:

```
df_long <- df_paired |> 
  pivot_longer(
    everything(), 
    names_to = c("group", ".value"), 
    names_sep = "_"
  )
df_long
#> # A tibble: 40 × 3
#>   group    val   wts
#>   <chr>  <dbl> <dbl>
#> 1 a      0.715 0.518
#> 2 b     -0.709 0.691
#> 3 c      0.718 0.216
#> 4 d     -0.217 0.733
#> 5 a     -1.09  0.979
#> 6 b     -0.209 0.675
#> # … with 34 more rows

df_long |> 
  group_by(group) |> 
  summarize(mean = weighted.mean(val, wts))
#> # A tibble: 4 × 2
#>   group    mean
#>   <chr>   <dbl>
#> 1 a      0.126 
#> 2 b     -0.0704
#> 3 c     -0.360 
#> 4 d     -0.248
```

If needed, you could <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a> this back to the original form.

## Exercises

1.  Practice your <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> skills by:

    1.  Computing the number of unique values in each column of <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>palmerpenguins::penguins</code></a>.

    2.  Computing the mean of every column in `mtcars`.

    3.  Grouping `diamonds` by `cut`, `clarity`, and `color` and then counting the number of observations and computing the mean of each numeric column.

2.  What happens if you use a list of functions in <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, but don’t name them? How is the output named?

3.  Adjust `expand_dates()` to automatically remove the date columns after they’ve been expanded. Do you need to embrace any arguments?

4.  Explain what each step of the pipeline in this function does. What special feature of <a href="https://tidyselect.r-lib.org/reference/where.html" class="orm:hideurl"><code>where()</code></a> are we taking advantage of?

    ```
    show_missing <- function(df, group_vars, summary_vars = everything()) {
      df |> 
        group_by(pick({{ group_vars }})) |> 
        summarize(
          across({{ summary_vars }}, \(x) sum(is.na(x))),
          .groups = "drop"
        ) |>
        select(where(\(x) any(x > 0)))
    }
    nycflights13::flights |> show_missing(c(year, month, day))
    ```

# Reading Multiple Files

In the previous section, you learned how to use <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>dplyr::across()</code></a> to repeat a transformation on multiple columns. In this section, you’ll learn how to use <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a> to do something to every file in a directory. Let’s start with a little motivation: imagine you have a directory full of Excel spreadsheets<sup><a href="ch26.html#idm44771266252816" id="idm44771266252816-marker" data-type="noteref">5</a></sup> you want to read. You could do it with copy and paste:

```
data2019 <- readxl::read_excel("data/y2019.xlsx")
data2020 <- readxl::read_excel("data/y2020.xlsx")
data2021 <- readxl::read_excel("data/y2021.xlsx")
data2022 <- readxl::read_excel("data/y2022.xlsx")
```

Then use <a href="https://dplyr.tidyverse.org/reference/bind_rows.html" class="orm:hideurl"><code>dplyr::bind_rows()</code></a> to combine them all together:

```
data <- bind_rows(data2019, data2020, data2021, data2022)
```

You can imagine that this would get tedious quickly, especially if you had hundreds of files, not just four. The following sections show you how to automate this sort of task. There are three basic steps: use <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a> to list all the files in a directory, then use <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a> to read each of them into a list, and then use <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>purrr::list_rbind()</code></a> to combine them into a single data frame. We’ll then discuss how you can handle situations of increasing heterogeneity, where you can’t do the same thing to every file.

## Listing Files in a Directory

As the name suggests, <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a> lists the files in a directory. You’ll almost always use three arguments:

- The first argument, `path`, is the directory to look in.

- `pattern` is a regular expression used to filter the filenames. The most common pattern is something like `[.]xlsx$` or `[.]csv$` to find all files with a specified extension.

- `full.names` determines whether the directory name should be included in the output. You almost always want this to be `TRUE`.

To make our motivating example concrete, this book contains a folder with 12 Excel spreadsheets containing data from the gapminder package. Each file contains one year’s worth of data for 142 countries. We can list them all with the appropriate call to <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a>:

```
paths <- list.files("data/gapminder", pattern = "[.]xlsx$", full.names = TRUE)
paths
#>  [1] "data/gapminder/1952.xlsx" "data/gapminder/1957.xlsx"
#>  [3] "data/gapminder/1962.xlsx" "data/gapminder/1967.xlsx"
#>  [5] "data/gapminder/1972.xlsx" "data/gapminder/1977.xlsx"
#>  [7] "data/gapminder/1982.xlsx" "data/gapminder/1987.xlsx"
#>  [9] "data/gapminder/1992.xlsx" "data/gapminder/1997.xlsx"
#> [11] "data/gapminder/2002.xlsx" "data/gapminder/2007.xlsx"
```

## Lists

Now that we have these 12 paths, we could call `read_excel()` 12 times to get 12 data frames:

```
gapminder_1952 <- readxl::read_excel("data/gapminder/1952.xlsx")
gapminder_1957 <- readxl::read_excel("data/gapminder/1957.xlsx")
gapminder_1962 <- readxl::read_excel("data/gapminder/1962.xlsx")
 ...,
gapminder_2007 <- readxl::read_excel("data/gapminder/2007.xlsx")
```

But putting each sheet into its own variable is going to make it hard to work with them a few steps down the road. Instead, they’ll be easier to work with if we put them into a single object. A list is the perfect tool for this job:

```
files <- list(
  readxl::read_excel("data/gapminder/1952.xlsx"),
  readxl::read_excel("data/gapminder/1957.xlsx"),
  readxl::read_excel("data/gapminder/1962.xlsx"),
  ...,
  readxl::read_excel("data/gapminder/2007.xlsx")
)
```

Now that you have these data frames in a list, how do you get one out? You can use `files[[i]]` to extract the *i*th element:

```
files[[3]]
#> # A tibble: 142 × 5
#>   country     continent lifeExp      pop gdpPercap
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 Afghanistan Asia         32.0 10267083      853.
#> 2 Albania     Europe       64.8  1728137     2313.
#> 3 Algeria     Africa       48.3 11000948     2551.
#> 4 Angola      Africa       34    4826015     4269.
#> 5 Argentina   Americas     65.1 21283783     7133.
#> 6 Australia   Oceania      70.9 10794968    12217.
#> # … with 136 more rows
```

We’ll come back to `[[` in more detail in <a href="ch27.html#sec-subset-one" data-type="xref">“Selecting a Single Element with $ and [[”</a>.

## purrr::map() and list_rbind()

The code to collect those data frames in a list “by hand” is basically just as tedious to type as code that reads the files one by one. Happily, we can use <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a> to make even better use of our `paths` vector. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> is similar to <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, but instead of doing something to each column in a data frame, it does something to each element of a vector. `map(x, f)` is shorthand for:

```
list(
  f(x[[1]]),
  f(x[[2]]),
  ...,
  f(x[[n]])
)
```

So we can use <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> to get a list of 12 data frames:

```
files <- map(paths, readxl::read_excel)
length(files)
#> [1] 12

files[[1]]
#> # A tibble: 142 × 5
#>   country     continent lifeExp      pop gdpPercap
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 Afghanistan Asia         28.8  8425333      779.
#> 2 Albania     Europe       55.2  1282697     1601.
#> 3 Algeria     Africa       43.1  9279525     2449.
#> 4 Angola      Africa       30.0  4232095     3521.
#> 5 Argentina   Americas     62.5 17876956     5911.
#> 6 Australia   Oceania      69.1  8691212    10040.
#> # … with 136 more rows
```

(This is another data structure that doesn’t display particularly compactly with <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>, so you might want to load it into RStudio and inspect it with <a href="https://rdrr.io/r/utils/View.html" class="orm:hideurl"><code>View()</code></a>).

Now we can use <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>purrr::list_rbind()</code></a> to combine that list of data frames into a single data frame:

```
list_rbind(files)
#> # A tibble: 1,704 × 5
#>   country     continent lifeExp      pop gdpPercap
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 Afghanistan Asia         28.8  8425333      779.
#> 2 Albania     Europe       55.2  1282697     1601.
#> 3 Algeria     Africa       43.1  9279525     2449.
#> 4 Angola      Africa       30.0  4232095     3521.
#> 5 Argentina   Americas     62.5 17876956     5911.
#> 6 Australia   Oceania      69.1  8691212    10040.
#> # … with 1,698 more rows
```

Or we could do both steps at once in a pipeline:

```
paths |> 
  map(readxl::read_excel) |> 
  list_rbind()
```

What if we want to pass in extra arguments to `read_excel()`? We use the same technique that we used with <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>. For example, it’s often useful to peak at the first few rows of the data with `n_max = 1`:

```
paths |> 
  map(\(path) readxl::read_excel(path, n_max = 1)) |> 
  list_rbind()
#> # A tibble: 12 × 5
#>   country     continent lifeExp      pop gdpPercap
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 Afghanistan Asia         28.8  8425333      779.
#> 2 Afghanistan Asia         30.3  9240934      821.
#> 3 Afghanistan Asia         32.0 10267083      853.
#> 4 Afghanistan Asia         34.0 11537966      836.
#> 5 Afghanistan Asia         36.1 13079460      740.
#> 6 Afghanistan Asia         38.4 14880372      786.
#> # … with 6 more rows
```

This makes it clear that something is missing: there’s no `year` column because that value is recorded in the path, not the individual files. We’ll tackle that problem next.

## Data in the Path

Sometimes the name of the file is data itself. In this example, the filename contains the year, which is not otherwise recorded in the individual files. To get that column into the final data frame, we need to do two things.

First, we name the vector of paths. The easiest way to do this is with the <a href="https://rlang.r-lib.org/reference/set_names.html" class="orm:hideurl"><code>set_names()</code></a> function, which can take a function. Here we use <a href="https://rdrr.io/r/base/basename.html" class="orm:hideurl"><code>basename()</code></a> to extract just the file name from the full path:

```
paths |> set_names(basename) 
#>                  1952.xlsx                  1957.xlsx 
#> "data/gapminder/1952.xlsx" "data/gapminder/1957.xlsx" 
#>                  1962.xlsx                  1967.xlsx 
#> "data/gapminder/1962.xlsx" "data/gapminder/1967.xlsx" 
#>                  1972.xlsx                  1977.xlsx 
#> "data/gapminder/1972.xlsx" "data/gapminder/1977.xlsx" 
#>                  1982.xlsx                  1987.xlsx 
#> "data/gapminder/1982.xlsx" "data/gapminder/1987.xlsx" 
#>                  1992.xlsx                  1997.xlsx 
#> "data/gapminder/1992.xlsx" "data/gapminder/1997.xlsx" 
#>                  2002.xlsx                  2007.xlsx 
#> "data/gapminder/2002.xlsx" "data/gapminder/2007.xlsx"
```

Those names are automatically carried along by all the map functions, so the list of data frames will have those same names:

```
files <- paths |> 
  set_names(basename) |> 
  map(readxl::read_excel)
```

That makes this call to <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> shorthand for:

```
files <- list(
  "1952.xlsx" = readxl::read_excel("data/gapminder/1952.xlsx"),
  "1957.xlsx" = readxl::read_excel("data/gapminder/1957.xlsx"),
  "1962.xlsx" = readxl::read_excel("data/gapminder/1962.xlsx"),
  ...,
  "2007.xlsx" = readxl::read_excel("data/gapminder/2007.xlsx")
)
```

You can also use `[[` to extract elements by name:

```
files[["1962.xlsx"]]
#> # A tibble: 142 × 5
#>   country     continent lifeExp      pop gdpPercap
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 Afghanistan Asia         32.0 10267083      853.
#> 2 Albania     Europe       64.8  1728137     2313.
#> 3 Algeria     Africa       48.3 11000948     2551.
#> 4 Angola      Africa       34    4826015     4269.
#> 5 Argentina   Americas     65.1 21283783     7133.
#> 6 Australia   Oceania      70.9 10794968    12217.
#> # … with 136 more rows
```

Then we use the `names_to` argument to <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a> to tell it to save the names into a new column called `year` and then use <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>readr::parse_number()</code></a> to extract the number from the string:

```
paths |> 
  set_names(basename) |> 
  map(readxl::read_excel) |> 
  list_rbind(names_to = "year") |> 
  mutate(year = parse_number(year))
#> # A tibble: 1,704 × 6
#>    year country     continent lifeExp      pop gdpPercap
#>   <dbl> <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1  1952 Afghanistan Asia         28.8  8425333      779.
#> 2  1952 Albania     Europe       55.2  1282697     1601.
#> 3  1952 Algeria     Africa       43.1  9279525     2449.
#> 4  1952 Angola      Africa       30.0  4232095     3521.
#> 5  1952 Argentina   Americas     62.5 17876956     5911.
#> 6  1952 Australia   Oceania      69.1  8691212    10040.
#> # … with 1,698 more rows
```

In more complicated cases, there might be other variables stored in the directory name, or maybe the filename contains multiple bits of data. In that case, use <a href="https://rlang.r-lib.org/reference/set_names.html" class="orm:hideurl"><code>set_names()</code></a> (without any arguments) to record the full path and then use <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>tidyr::separate_wider_delim()</code></a> and friends to turn them into useful columns:

```
paths |> 
  set_names() |> 
  map(readxl::read_excel) |> 
  list_rbind(names_to = "year") |> 
  separate_wider_delim(year, delim = "/", names = c(NA, "dir", "file")) |> 
  separate_wider_delim(file, delim = ".", names = c("file", "ext"))
#> # A tibble: 1,704 × 8
#>   dir       file  ext   country     continent lifeExp      pop gdpPercap
#>   <chr>     <chr> <chr> <chr>       <chr>       <dbl>    <dbl>     <dbl>
#> 1 gapminder 1952  xlsx  Afghanistan Asia         28.8  8425333      779.
#> 2 gapminder 1952  xlsx  Albania     Europe       55.2  1282697     1601.
#> 3 gapminder 1952  xlsx  Algeria     Africa       43.1  9279525     2449.
#> 4 gapminder 1952  xlsx  Angola      Africa       30.0  4232095     3521.
#> 5 gapminder 1952  xlsx  Argentina   Americas     62.5 17876956     5911.
#> 6 gapminder 1952  xlsx  Australia   Oceania      69.1  8691212    10040.
#> # … with 1,698 more rows
```

## Save Your Work

Now that you’ve done all this hard work to get to a nice tidy data frame, it’s a great time to save your work:

```
gapminder <- paths |> 
  set_names(basename) |> 
  map(readxl::read_excel) |> 
  list_rbind(names_to = "year") |> 
  mutate(year = parse_number(year))

write_csv(gapminder, "gapminder.csv")
```

Now when you come back to this problem in the future, you can read in a single CSV file. For large and richer datasets, using parquet might be a better choice than `.csv`, as discussed in <a href="ch22.html#sec-parquet" data-type="xref">“The Parquet Format”</a>.

If you’re working in a project, we suggest calling the file that does this sort of data prep work, something like `0-cleanup.R`. The `0` in the filename suggests that this should be run before anything else.

If your input data files change over time, you might consider learning a tool like [targets](https://oreil.ly/oJsOo) to set up your data cleaning code to automatically rerun whenever one of the input files is modified.

## Many Simple Iterations

Here we loaded the data directly from disk and were lucky enough to get a tidy dataset. In most cases, you’ll need to do some additional tidying, and you have two basic options: you can do one round of iteration with a complex function or do multiple rounds of iteration with simple functions. In our experience, most folks reach first for one complex iteration, but you’re often better off doing multiple simple iterations.

For example, imagine that you want to read in a bunch of files, filter out missing values, pivot, and then combine. One way to approach the problem is to write a function that takes a file and does all those steps and then call <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> once:

```
process_file <- function(path) {
  df <- read_csv(path)
  
  df |> 
    filter(!is.na(id)) |> 
    mutate(id = tolower(id)) |> 
    pivot_longer(jan:dec, names_to = "month")
}

paths |> 
  map(process_file) |> 
  list_rbind()
```

Alternatively, you could perform each step of `process_file()` for every file:

```
paths |> 
  map(read_csv) |> 
  map(\(df) df |> filter(!is.na(id))) |> 
  map(\(df) df |> mutate(id = tolower(id))) |> 
  map(\(df) df |> pivot_longer(jan:dec, names_to = "month")) |> 
  list_rbind()
```

We recommend this approach because it stops you from getting fixated on getting the first file right before moving on to the rest. By considering all of the data when doing tidying and cleaning, you’re more likely to think holistically and end up with a higher-quality result.

In this particular example, there’s another optimization you could make, by binding all the data frames together earlier. Then you can rely on regular dplyr behavior:

```
paths |> 
  map(read_csv) |> 
  list_rbind() |> 
  filter(!is.na(id)) |> 
  mutate(id = tolower(id)) |> 
  pivot_longer(jan:dec, names_to = "month")
```

## Heterogeneous Data

Unfortunately, sometimes it’s not possible to go from <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> straight to <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a> because the data frames are so heterogeneous that <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a> either fails or yields a data frame that’s not useful. In that case, it’s still useful to start by loading all of the files:

```
files <- paths |> 
  map(readxl::read_excel) 
```

Then a useful strategy is to capture the structure of the data frames so that you can explore it using your data science skills. One way to do so is with this handy `df_types` function<sup><a href="ch26.html#idm44771264920592" id="idm44771264920592-marker" data-type="noteref">6</a></sup> that returns a tibble with one row for each column:

```
df_types <- function(df) {
  tibble(
    col_name = names(df), 
    col_type = map_chr(df, vctrs::vec_ptype_full),
    n_miss = map_int(df, \(x) sum(is.na(x)))
  )
}

df_types(gapminder)
#> # A tibble: 6 × 3
#>   col_name  col_type  n_miss
#>   <chr>     <chr>      <int>
#> 1 year      double         0
#> 2 country   character      0
#> 3 continent character      0
#> 4 lifeExp   double         0
#> 5 pop       double         0
#> 6 gdpPercap double         0
```

You can then apply this function to all of the files and maybe do some pivoting to make it easier to see where the differences are. For example, this makes it easy to verify that the gapminder spreadsheets that we’ve been working with are all quite homogeneous:

```
files |> 
  map(df_types) |> 
  list_rbind(names_to = "file_name") |> 
  select(-n_miss) |> 
  pivot_wider(names_from = col_name, values_from = col_type)
#> # A tibble: 12 × 6
#>   file_name country   continent lifeExp pop    gdpPercap
#>   <chr>     <chr>     <chr>     <chr>   <chr>  <chr>    
#> 1 1952.xlsx character character double  double double   
#> 2 1957.xlsx character character double  double double   
#> 3 1962.xlsx character character double  double double   
#> 4 1967.xlsx character character double  double double   
#> 5 1972.xlsx character character double  double double   
#> 6 1977.xlsx character character double  double double   
#> # … with 6 more rows
```

If the files have heterogeneous formats, you might need to do more processing before you can successfully merge them. Unfortunately, we’re now going to leave you to figure that out on your own, but you might want to read about <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_if()</code></a> and <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_at()</code></a>. <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_if()</code></a> allows you to selectively modify elements of a list based on their values; <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_at()</code></a> allows you to selectively modify elements based on their names.

## Handling Failures

Sometimes the structure of your data might be sufficiently wild that you can’t even read all the files with a single command. And then you’ll encounter one of the downsides of `map()`: it succeeds or fails as a whole. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> will either successfully read all of the files in a directory or fail with an error, reading zero files. This is annoying: why does one failure prevent you from accessing all the other successes?

Luckily, purrr comes with a helper to tackle this problem: <a href="https://purrr.tidyverse.org/reference/possibly.html" class="orm:hideurl"><code>possibly()</code></a>. <a href="https://purrr.tidyverse.org/reference/possibly.html" class="orm:hideurl"><code>possibly()</code></a> is what’s known as a *function operator*: it takes a function and returns a function with modified behavior. In particular, <a href="https://purrr.tidyverse.org/reference/possibly.html" class="orm:hideurl"><code>possibly()</code></a> changes a function from erroring to returning a value that you specify:

```
files <- paths |> 
  map(possibly(\(path) readxl::read_excel(path), NULL))

data <- files |> list_rbind()
```

This works particularly well here because <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a>, like many tidyverse functions, automatically ignores `NULL`s.

Now you have all the data that can be read easily, and it’s time to tackle the hard part of figuring out why some files failed to load and what to do about it. Start by getting the paths that failed:

```
failed <- map_vec(files, is.null)
paths[failed]
#> character(0)
```

Then call the import function again for each failure and figure out what went wrong.

# Saving Multiple Outputs

In the previous section, you learned about <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>, which is useful for reading multiple files into a single object. In this section, we’ll now explore sort of the opposite problem: how can you take one or more R objects and save it to one or more files? We’ll explore this challenge using three examples:

- Saving multiple data frames into one database
- Saving multiple data frames into multiple `.csv` files
- Saving multiple plots to multiple `.png` files

## Writing to a Database

Sometimes when working with many files at once, it’s not possible to fit all your data into memory at once, and you can’t do `map(files, read_csv)`. One approach to deal with this problem is to load your data into a database so you can access just the bits you need with dbplyr.

If you’re lucky, the database package you’re using will provide a handy function that takes a vector of paths and loads them all into the database. This is the case with duckdb’s `duckdb_read_csv()`:

```
con <- DBI::dbConnect(duckdb::duckdb())
duckdb::duckdb_read_csv(con, "gapminder", paths)
```

This would work well here, but we don’t have CSV files; instead, we have Excel spreadsheets. So we’re going to have to do it “by hand.” Learning to do it by hand will also help you when you have a bunch of CSV files and the database that you’re working with doesn’t have one function that will load them all in.

We need to start by creating a table that will fill in with data. The easiest way to do this is by creating a template, a dummy data frame that contains all the columns we want, but only a sampling of the data. For the gapminder data, we can make that template by reading a single file and adding the year to it:

```
template <- readxl::read_excel(paths[[1]])
template$year <- 1952
template
#> # A tibble: 142 × 6
#>   country     continent lifeExp      pop gdpPercap  year
#>   <chr>       <chr>       <dbl>    <dbl>     <dbl> <dbl>
#> 1 Afghanistan Asia         28.8  8425333      779.  1952
#> 2 Albania     Europe       55.2  1282697     1601.  1952
#> 3 Algeria     Africa       43.1  9279525     2449.  1952
#> 4 Angola      Africa       30.0  4232095     3521.  1952
#> 5 Argentina   Americas     62.5 17876956     5911.  1952
#> 6 Australia   Oceania      69.1  8691212    10040.  1952
#> # … with 136 more rows
```

Now we can connect to the database and use <a href="https://dbi.r-dbi.org/reference/dbCreateTable.html" class="orm:hideurl"><code>DBI::dbCreateTable()</code></a> to turn our template into a database table:

```
con <- DBI::dbConnect(duckdb::duckdb())
DBI::dbCreateTable(con, "gapminder", template)
```

`dbCreateTable()` doesn’t use the data in `template`, just the variable names and types. So if we inspect the `gapminder` table now, you’ll see that it’s empty, but it has the variables we need with the types we expect:

```
con |> tbl("gapminder")
#> # Source:   table<gapminder> [0 x 6]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#> # … with 6 variables: country <chr>, continent <chr>, lifeExp <dbl>,
#> #   pop <dbl>, gdpPercap <dbl>, year <dbl>
```

Next, we need a function that takes a single file path, reads it into R, and adds the result to the `gapminder` table. We can do that by combining `read_excel()` with <a href="https://dbi.r-dbi.org/reference/dbAppendTable.html" class="orm:hideurl"><code>DBI::dbAppendTable()</code></a>:

```
append_file <- function(path) {
  df <- readxl::read_excel(path)
  df$year <- parse_number(basename(path))
  
  DBI::dbAppendTable(con, "gapminder", df)
}
```

Now we need to call `append_file()` once for each element of `paths`. That’s certainly possible with <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>:

```
paths |> map(append_file)
```

But we don’t care about the output of `append_file()`, so instead of <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>, it’s slightly nicer to use <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>walk()</code></a>. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>walk()</code></a> does exactly the same thing as <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> but throws the output away:

```
paths |> walk(append_file)
```

Now we can see if we have all the data in our table:

```
con |> 
  tbl("gapminder") |> 
  count(year)
#> # Source:   SQL [?? x 2]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#>    year     n
#>   <dbl> <dbl>
#> 1  1952   142
#> 2  1957   142
#> 3  1962   142
#> 4  1967   142
#> 5  1972   142
#> 6  1977   142
#> # … with more rows
```

## Writing CSV Files

The same basic principle applies if we want to write multiple CSV files, one for each group. Let’s imagine that we want to take the <a href="https://ggplot2.tidyverse.org/reference/diamonds.html" class="orm:hideurl"><code>ggplot2::diamonds</code></a> data and save one CSV file for each `clarity`. First we need to make those individual datasets. There are many ways you could do that, but there’s one way we particularly like: <a href="https://dplyr.tidyverse.org/reference/group_nest.html" class="orm:hideurl"><code>group_nest()</code></a>.

```
by_clarity <- diamonds |> 
  group_nest(clarity)

by_clarity
#> # A tibble: 8 × 2
#>   clarity               data
#>   <ord>   <list<tibble[,9]>>
#> 1 I1               [741 × 9]
#> 2 SI2            [9,194 × 9]
#> 3 SI1           [13,065 × 9]
#> 4 VS2           [12,258 × 9]
#> 5 VS1            [8,171 × 9]
#> 6 VVS2           [5,066 × 9]
#> # … with 2 more rows
```

This gives us a new tibble with eight rows and two columns. `clarity` is our grouping variable, and `data` is a list column containing one tibble for each unique value of `clarity`:

```
by_clarity$data[[1]]
#> # A tibble: 741 × 9
#>   carat cut       color depth table price     x     y     z
#>   <dbl> <ord>     <ord> <dbl> <dbl> <int> <dbl> <dbl> <dbl>
#> 1  0.32 Premium   E      60.9    58   345  4.38  4.42  2.68
#> 2  1.17 Very Good J      60.2    61  2774  6.83  6.9   4.13
#> 3  1.01 Premium   F      61.8    60  2781  6.39  6.36  3.94
#> 4  1.01 Fair      E      64.5    58  2788  6.29  6.21  4.03
#> 5  0.96 Ideal     F      60.7    55  2801  6.37  6.41  3.88
#> 6  1.04 Premium   G      62.2    58  2801  6.46  6.41  4   
#> # … with 735 more rows
```

While we’re here, let’s create a column that gives the name of the output file, using <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> and <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>:

```
by_clarity <- by_clarity |> 
  mutate(path = str_glue("diamonds-{clarity}.csv"))

by_clarity
#> # A tibble: 8 × 3
#>   clarity               data path             
#>   <ord>   <list<tibble[,9]>> <glue>           
#> 1 I1               [741 × 9] diamonds-I1.csv  
#> 2 SI2            [9,194 × 9] diamonds-SI2.csv 
#> 3 SI1           [13,065 × 9] diamonds-SI1.csv 
#> 4 VS2           [12,258 × 9] diamonds-VS2.csv 
#> 5 VS1            [8,171 × 9] diamonds-VS1.csv 
#> 6 VVS2           [5,066 × 9] diamonds-VVS2.csv
#> # … with 2 more rows
```

So if we were going to save these data frames by hand, we might write something like:

```
write_csv(by_clarity$data[[1]], by_clarity$path[[1]])
write_csv(by_clarity$data[[2]], by_clarity$path[[2]])
write_csv(by_clarity$data[[3]], by_clarity$path[[3]])
...
write_csv(by_clarity$by_clarity[[8]], by_clarity$path[[8]])
```

This is a little different from our previous uses of <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> because there are two arguments that are changing, not just one. That means we need a new function: <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>map2()</code></a>, which varies both the first and second arguments. And because we again don’t care about the output, we want <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>walk2()</code></a> rather than <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>map2()</code></a>. That gives us:

```
walk2(by_clarity$data, by_clarity$path, write_csv)
```

## Saving Plots

We can take the same basic approach to create many plots. Let’s first make a function that draws the plot we want:

```
carat_histogram <- function(df) {
  ggplot(df, aes(x = carat)) + geom_histogram(binwidth = 0.1)  
}

carat_histogram(by_clarity$data[[1]])
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_26in01.png" alt="Histogram of carats of diamonds from the by_clarity dataset, ranging from 0 to 5 carats. The distribution is unimodal and right skewed with a peak around 1 carat." />
</figure>

Now we can use <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> to create a list of many plots<sup><a href="ch26.html#idm44771263913536" id="idm44771263913536-marker" data-type="noteref">7</a></sup> and their eventual file paths:

```
by_clarity <- by_clarity |> 
  mutate(
    plot = map(data, carat_histogram),
    path = str_glue("clarity-{clarity}.png")
  )
```

Then use <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>walk2()</code></a> with <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a> to save each plot:

```
walk2(
  by_clarity$path,
  by_clarity$plot,
  \(path, plot) ggsave(path, plot, width = 6, height = 6)
)
```

This is shorthand for:

```
ggsave(by_clarity$path[[1]], by_clarity$plot[[1]], width = 6, height = 6)
ggsave(by_clarity$path[[2]], by_clarity$plot[[2]], width = 6, height = 6)
ggsave(by_clarity$path[[3]], by_clarity$plot[[3]], width = 6, height = 6)
...
ggsave(by_clarity$path[[8]], by_clarity$plot[[8]], width = 6, height = 6)
```

# Summary

In this chapter, you saw how to use explicit iteration to solve three problems that come up frequently when doing data science: manipulating multiple columns, reading multiple files, and saving multiple outputs. But in general, iteration is a superpower: if you know the right iteration technique, you can easily go from fixing one problem to fixing all the problems. Once you’ve mastered the techniques in this chapter, we highly recommend learning more by reading the [“Functionals” chapter](https://oreil.ly/VmXg4) of *Advanced R* and consulting the [purrr website](https://oreil.ly/f0HWP).

If you know much about iteration in other languages, you might be surprised that we didn’t discuss the `for` loop. That’s because R’s orientation toward data analysis changes how we iterate: in most cases you can rely on an existing idiom to do something to each column or each group. And when you can’t, you can often use a functional programming tool like <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> that does something to each element of a list. However, you will see `for` loops in wild-caught code, so you’ll learn about them in the next chapter where we’ll discuss some important base R tools.

<sup>[1](ch26.html#idm44771267612512-marker)</sup> Anonymous, because we never explicitly gave it a name with `<-`. Another term programmers use for this is *lambda function*.

<sup>[2](ch26.html#idm44771267610256-marker)</sup> In older code you might see syntax that looks like `~ .x + 1`. This is another way to write anonymous functions, but it works only inside tidyverse functions and always uses the variable name `.x`. We now recommend the base syntax, `\(x) x + 1`.

<sup>[3](ch26.html#idm44771267377264-marker)</sup> You can’t currently change the order of the columns, but you could reorder them after the fact using <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a> or similar.

<sup>[4](ch26.html#idm44771266547392-marker)</sup> Maybe there will be one day, but currently we don’t see how.

<sup>[5](ch26.html#idm44771266252816-marker)</sup> If you instead had a directory of CSV files with the same format, you can use the technique from <a href="ch07.html#sec-readr-directory" data-type="xref">“Reading Data from Multiple Files”</a>.

<sup>[6](ch26.html#idm44771264920592-marker)</sup> We’re not going to explain how it works, but if you look at the docs for the functions used, you should be able to puzzle it out.

<sup>[7](ch26.html#idm44771263913536-marker)</sup> You can print `by_clarity$plot` to get a crude animation—you’ll get one plot for each element of `plots`.
