# Chapter 27. A Field Guide to Base R

# Introduction

To finish off the programming section, we’re going to give you a quick tour of the most important base R functions that we don’t otherwise discuss in the book. These tools are particularly useful as you do more programming and will help you read code you encounter in the wild.

This is a good place to remind you that the tidyverse is not the only way to solve data science problems. We teach the tidyverse in this book because tidyverse packages share a common design philosophy, increasing the consistency across functions, and making each new function or package a little easier to learn and use. It’s not possible to use the tidyverse without using base R, so we’ve actually already taught you a *lot* of base R functions, including <a href="https://rdrr.io/r/base/library.html" class="orm:hideurl"><code>library()</code></a> to load packages; <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a> and <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a> for numeric summaries; the factor, date, and POSIXct data types; and of course all the basic operators such as `+`, `-`, `/`, `*`, `|`, `&`, and `!`. What we haven’t focused on so far is base R workflows, so we will highlight a few of those in this chapter.

After you read this book, you’ll learn other approaches to the same problems using base R, data.table, and other packages. You’ll undoubtedly encounter these other approaches when you start reading R code written by others, particularly if you’re using StackOverflow. It’s 100% OK to write code that uses a mix of approaches, and don’t let anyone tell you otherwise!

In this chapter, we’ll focus on four big topics: subsetting with `[`, subsetting with `[[` and `$`, using the apply family of functions, and using `for` loops. To finish off, we’ll briefly discuss two essential plotting functions.

## Prerequisites

This package focuses on base R so it doesn’t have any real prerequisites, but we’ll load the tidyverse to explain some of the differences:

```
library(tidyverse)
```

# Selecting Multiple Elements with \[

`[` is used to extract subcomponents from vectors and data frames and is called like `x[i]` or `x[i, j]`. In this section, we’ll introduce you to the power of `[`, first showing you how you can use it with vectors, and then showing how the same principles extend in a straightforward way to 2D structures like data frames. We’ll then help you cement that knowledge by showing how various dplyr verbs are special cases of `[`.

## Subsetting Vectors

There are five main types of things that you can subset a vector with, i.e., that can be the `i` in `x[i]`:

- *A vector of positive integers*. Subsetting with positive integers keeps the elements at those positions:

  ```
  x <- c("one", "two", "three", "four", "five")
  x[c(3, 2, 5)]
  #> [1] "three" "two"   "five"
  ```

  By repeating a position, you can actually make a longer output than input, making the term “subsetting” a bit of a misnomer:

  ```
  x[c(1, 1, 5, 5, 5, 2)]
  #> [1] "one"  "one"  "five" "five" "five" "two"
  ```

- *A vector of negative integers*. Negative values drop the elements at the specified positions:

  ```
  x[c(-1, -3, -5)]
  #> [1] "two"  "four"
  ```

- *A logical vector*. Subsetting with a logical vector keeps all values corresponding to a `TRUE` value. This is most often useful in conjunction with the comparison functions:

  ```
  x <- c(10, 3, NA, 5, 8, 1, NA)

  # All non-missing values of x
  x[!is.na(x)]
  #> [1] 10  3  5  8  1

  # All even (or missing!) values of x
  x[x %% 2 == 0]
  #> [1] 10 NA  8 NA
  ```

  Unlike <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>, `NA` indices will be included in the output as `NA`s.

- *A character vector*. If you have a named vector, you can subset it with a character vector:

  ```
  x <- c(abc = 1, def = 2, xyz = 5)
  x[c("xyz", "def")]
  #> xyz def 
  #>   5   2
  ```

  As with subsetting with positive integers, you can use a character vector to duplicate individual entries.

- *Nothing*. The final type of subsetting is nothing, `x[]`, which returns the complete `x`. This is not useful for subsetting vectors, but as we’ll see shortly, it is useful when subsetting 2D structures like tibbles.

## Subsetting Data Frames

There are quite a few different ways<sup><a href="ch27.html#idm44771263328096" id="idm44771263328096-marker" data-type="noteref">1</a></sup> that you can use `[` with a data frame, but the most important way is to select rows and columns independently with `df[rows, cols]`. Here `rows` and `cols` are vectors as described earlier. For example, `df[rows, ]` and `df[, cols]` select just rows or just columns, using the empty subset to preserve the other dimension.

Here are a couple of examples:

```
df <- tibble(
  x = 1:3, 
  y = c("a", "e", "f"), 
  z = runif(3)
)

# Select first row and second column
df[1, 2]
#> # A tibble: 1 × 1
#>   y    
#>   <chr>
#> 1 a

# Select all rows and columns x and y
df[, c("x" , "y")]
#> # A tibble: 3 × 2
#>       x y    
#>   <int> <chr>
#> 1     1 a    
#> 2     2 e    
#> 3     3 f

# Select rows where `x` is greater than 1 and all columns
df[df$x > 1, ]
#> # A tibble: 2 × 3
#>       x y         z
#>   <int> <chr> <dbl>
#> 1     2 e     0.834
#> 2     3 f     0.601
```

We’ll come back to `$` shortly, but you should be able to guess what `df$x` does from the context: it extracts the `x` variable from `df`. We need to use it here because `[` doesn’t use tidy evaluation, so you need to be explicit about the source of the `x` variable.

There’s an important difference between tibbles and data frames when it comes to `[`. In this book, we’ve mainly used tibbles, which *are* data frames, but they tweak some behaviors to make your life a little easier. In most places, you can use “tibble” and “data frame” interchangeably, so when we want to draw particular attention to R’s built-in data frame, we’ll write `data.frame`. If `df` is a `data.frame`, then `df[, cols]` will return a vector if `col` selects a single column and will return a data frame if it selects more than one column. If `df` is a tibble, then `[` will always return a tibble.

```
df1 <- data.frame(x = 1:3)
df1[, "x"]
#> [1] 1 2 3

df2 <- tibble(x = 1:3)
df2[, "x"]
#> # A tibble: 3 × 1
#>       x
#>   <int>
#> 1     1
#> 2     2
#> 3     3
```

One way to avoid this ambiguity with `data.frame`s is to explicitly specify `drop = FALSE`:

```
df1[, "x" , drop = FALSE]
#>   x
#> 1 1
#> 2 2
#> 3 3
```

## dplyr Equivalents

Several dplyr verbs are special cases of `[`:

- <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> is equivalent to subsetting the rows with a logical vector, taking care to exclude missing values:

  ```
  df <- tibble(
    x = c(2, 3, 1, 1, NA), 
    y = letters[1:5], 
    z = runif(5)
  )
  df |> filter(x > 1)

  # same as
  df[!is.na(df$x) & df$x > 1, ]
  ```

  Another common technique in the wild is to use <a href="https://rdrr.io/r/base/which.html" class="orm:hideurl"><code>which()</code></a> for its side effect of dropping missing values: `df[which(df$x > 1), ]`.

- <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a> is equivalent to subsetting the rows with an integer vector, usually created with <a href="https://rdrr.io/r/base/order.html" class="orm:hideurl"><code>order()</code></a>:

  ```
  df |> arrange(x, y)

  # same as
  df[order(df$x, df$y), ]
  ```

  You can use `order(decreasing = TRUE)` to sort all columns in descending order or `-rank(col)` to sort columns in decreasing order individually.

- Both <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a> and <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a> are similar to subsetting the columns with a character vector:

  ```
  df |> select(x, z)

  # same as
  df[, c("x", "z")]
  ```

Base R also provides a function that combines the features of <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> and <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a><sup><a href="ch27.html#idm44771262898928" id="idm44771262898928-marker" data-type="noteref">2</a></sup> called <a href="https://rdrr.io/r/base/subset.html" class="orm:hideurl"><code>subset()</code></a>:

```
df |> 
  filter(x > 1) |> 
  select(y, z)
#> # A tibble: 2 × 2
#>   y           z
#>   <chr>   <dbl>
#> 1 a     0.157  
#> 2 b     0.00740
```

```
# same as
df |> subset(x > 1, c(y, z))
```

This function was the inspiration for much of dplyr’s syntax.

## Exercises

1.  Create functions that take a vector as input and return:

    1.  The elements at even-numbered positions
    2.  Every element except the last value
    3.  Only even values (and no missing values)

2.  Why is `x[-which(x > 0)]` not the same as `x[x <= 0]`? Read the documentation for <a href="https://rdrr.io/r/base/which.html" class="orm:hideurl"><code>which()</code></a> and do some experiments to figure it out.

# Selecting a Single Element with \$ and \[\[

`[`, which selects many elements, is paired with `[[` and `$`, which extract a single element. In this section, we’ll show you how to use `[[` and `$` to pull columns out of data frames, discuss a couple more differences between `data.frames` and tibbles, and emphasize some important differences between `[` and `[[` when used with lists.

## Data Frames

`[[` and `$` can be used to extract columns out of a data frame. `[[` can access by position or by name, and `$` is specialized for access by name:

```
tb <- tibble(
  x = 1:4,
  y = c(10, 4, 1, 21)
)

# by position
tb[[1]]
#> [1] 1 2 3 4

# by name
tb[["x"]]
#> [1] 1 2 3 4
tb$x
#> [1] 1 2 3 4
```

They can also be used to create new columns, the base R equivalent of <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>:

```
tb$z <- tb$x + tb$y
tb
#> # A tibble: 4 × 3
#>       x     y     z
#>   <int> <dbl> <dbl>
#> 1     1    10    11
#> 2     2     4     6
#> 3     3     1     4
#> 4     4    21    25
```

There are several other base R approaches to creating new columns including with <a href="https://rdrr.io/r/base/transform.html" class="orm:hideurl"><code>transform()</code></a>, <a href="https://rdrr.io/r/base/with.html" class="orm:hideurl"><code>with()</code></a>, and <a href="https://rdrr.io/r/base/with.html" class="orm:hideurl"><code>within()</code></a>. Hadley collected a few [examples](https://oreil.ly/z6vyT).

Using `$` directly is convenient when performing quick summaries. For example, if you just want to find the size of the biggest diamond or the possible values of `cut`, there’s no need to use <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>:

```
max(diamonds$carat)
#> [1] 5.01

levels(diamonds$cut)
#> [1] "Fair"      "Good"      "Very Good" "Premium"   "Ideal"
```

dplyr also provides an equivalent to `[[`/`$` that we didn’t mention in <a href="ch03.html#chp-data-transform" data-type="xref">Chapter 3</a>: <a href="https://dplyr.tidyverse.org/reference/pull.html" class="orm:hideurl"><code>pull()</code></a>. <a href="https://dplyr.tidyverse.org/reference/pull.html" class="orm:hideurl"><code>pull()</code></a> takes either a variable name or a variable position and returns just that column. That means we could rewrite the previous code to use the pipe:

```
diamonds |> pull(carat) |> mean()
#> [1] 0.7979397

diamonds |> pull(cut) |> levels()
#> [1] "Fair"      "Good"      "Very Good" "Premium"   "Ideal"
```

## Tibbles

There are a couple of important differences between tibbles and base `data.frame`s when it comes to `$`. Data frames match the prefix of any variable names (so-called *partial matching*) and don’t complain if a column doesn’t exist:

```
df <- data.frame(x1 = 1)
df$x
#> Warning in df$x: partial match of 'x' to 'x1'
#> [1] 1
df$z
#> NULL
```

Tibbles are more strict: they only ever match variable names exactly and they will generate a warning if the column you are trying to access doesn’t exist:

```
tb <- tibble(x1 = 1)

tb$x
#> Warning: Unknown or uninitialised column: `x`.
#> NULL
tb$z
#> Warning: Unknown or uninitialised column: `z`.
#> NULL
```

For this reason we sometimes joke that tibbles are lazy and surly: they do less and complain more.

## Lists

`[[` and `$` are also really important for working with lists, and it’s important to understand how they differ from `[`. Let’s illustrate the differences with a list named `l`:

```
l <- list(
  a = 1:3, 
  b = "a string", 
  c = pi, 
  d = list(-1, -5)
)
```

- `[` extracts a sublist. It doesn’t matter how many elements you extract, the result will always be a list.

  ```
  str(l[1:2])
  #> List of 2
  #>  $ a: int [1:3] 1 2 3
  #>  $ b: chr "a string"

  str(l[1])
  #> List of 1
  #>  $ a: int [1:3] 1 2 3

  str(l[4])
  #> List of 1
  #>  $ d:List of 2
  #>   ..$ : num -1
  #>   ..$ : num -5
  ```

  Like with vectors, you can subset with a logical, integer, or character vector.

- `[[` and `$` extract a single component from a list. They remove a level of hierarchy from the list.

  ```
  str(l[[1]])
  #>  int [1:3] 1 2 3

  str(l[[4]])
  #> List of 2
  #>  $ : num -1
  #>  $ : num -5

  str(l$a)
  #>  int [1:3] 1 2 3
  ```

The difference between `[` and `[[` is particularly important for lists because `[[` drills down into the list, while `[` returns a new, smaller list. To help you remember the difference, take a look at the unusual pepper shaker shown in <a href="#fig-pepper" data-type="xref">Figure 27-1</a>. If this pepper shaker is your list `pepper`, then `pepper[1]` is a pepper shaker containing a single pepper packet. `pepper[2]` would look the same but would contain the second packet. `pepper[1:2]` would be a pepper shaker containing two pepper packets. `pepper[[1]]` would extract the pepper packet itself.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_2701.png" alt="Three photos. On the left is a photo of a glass pepper shaker. Instead of the pepper shaker containing pepper, it contains a single packet of pepper. In the middle is a photo of a single packet of pepper. On the right is a photo of the contents of a packet of pepper." /></p>
<h6 id="figure-27-1.-left-a-pepper-shaker-that-hadley-once-found-in-his-hotel-room.-middle-pepper1.-right-pepper1.">Figure 27-1. (Left) A pepper shaker that Hadley once found in his hotel room. (Middle) <code>pepper[1]</code>. (Right) <code>pepper[[1]]</code>.</h6>
</figure>

This same principle applies when you use 1D `[` with a data frame: `df["x"]` returns a one-column data frame, and `df[["x"]]` returns a vector.

## Exercises

1.  What happens when you use `[[` with a positive integer that’s bigger than the length of the vector? What happens when you subset with a name that doesn’t exist?

2.  What would `pepper[[1]][1]` be? What about `pepper[[1]][[1]]`?

# Apply Family

In <a href="ch26.html#chp-iteration" data-type="xref">Chapter 26</a>, you learned tidyverse techniques for iteration like <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>dplyr::across()</code></a> and the map family of functions. In this section, you’ll learn about their base equivalents, the *apply family*. In this context, apply and map are synonyms because another way of saying “map a function over each element of a vector” is “apply a function over each element of a vector.” Here we’ll give you a quick overview of this family so you can recognize them in the wild.

The most important member of this family is <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>, which is similar to <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a>.<sup><a href="ch27.html#idm44771262249040" id="idm44771262249040-marker" data-type="noteref">3</a></sup> In fact, because we haven’t used any of <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code>’s</a> more advanced features, you can replace every <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> call in <a href="ch26.html#chp-iteration" data-type="xref">Chapter 26</a> with <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>.

There’s no exact base R equivalent to <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>, but you can get close by using `[` with <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>. This works because under the hood, data frames are lists of columns, so calling <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a> on a data frame applies the function to each column.

```
df <- tibble(a = 1, b = 2, c = "a", d = "b", e = 4)

# First find numeric columns
num_cols <- sapply(df, is.numeric)
num_cols
#>     a     b     c     d     e 
#>  TRUE  TRUE FALSE FALSE  TRUE

# Then transform each column with lapply() then replace the original values
df[, num_cols] <- lapply(df[, num_cols, drop = FALSE], \(x) x * 2)
df
#> # A tibble: 1 × 5
#>       a     b c     d         e
#>   <dbl> <dbl> <chr> <chr> <dbl>
#> 1     2     4 a     b         8
```

The previous code uses a new function, <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a>. It’s similar to <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>lapply()</code></a>, but it always tries to simplify the result, which is the reason for the `s` in its name, here producing a logical vector instead of a list. We don’t recommend using it for programming, because the simplification can fail and give you an unexpected type, but it’s usually fine for interactive use. purrr has a similar function called <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map_vec()</code></a> that we didn’t mention in <a href="ch26.html#chp-iteration" data-type="xref">Chapter 26</a>.

Base R provides a stricter version of <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a> called <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>vapply()</code></a>, short for *v*ector apply. It takes an additional argument that specifies the expected type, ensuring that simplification occurs the same way regardless of the input. For example, we could replace the previous <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a> call with this <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>vapply()</code></a> where we specify that we expect <a href="https://rdrr.io/r/base/numeric.html" class="orm:hideurl"><code>is.numeric()</code></a> to return a logical vector of length 1:

```
vapply(df, is.numeric, logical(1))
#>     a     b     c     d     e 
#>  TRUE  TRUE FALSE FALSE  TRUE
```

The distinction between <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>sapply()</code></a> and <a href="https://rdrr.io/r/base/lapply.html" class="orm:hideurl"><code>vapply()</code></a> is really important when they’re inside a function (because it makes a big difference to the function’s robustness to unusual inputs), but it doesn’t usually matter in data analysis.

Another important member of the apply family is <a href="https://rdrr.io/r/base/tapply.html" class="orm:hideurl"><code>tapply()</code></a>, which computes a single grouped summary:

```
diamonds |> 
  group_by(cut) |> 
  summarize(price = mean(price))
#> # A tibble: 5 × 2
#>   cut       price
#>   <ord>     <dbl>
#> 1 Fair      4359.
#> 2 Good      3929.
#> 3 Very Good 3982.
#> 4 Premium   4584.
#> 5 Ideal     3458.

tapply(diamonds$price, diamonds$cut, mean)
#>      Fair      Good Very Good   Premium     Ideal 
#>  4358.758  3928.864  3981.760  4584.258  3457.542
```

Unfortunately, <a href="https://rdrr.io/r/base/tapply.html" class="orm:hideurl"><code>tapply()</code></a> returns its results in a named vector, which requires some gymnastics if you want to collect multiple summaries and grouping variables into a data frame (it’s certainly possible to not do this and just work with free-floating vectors, but in our experience that just delays the work). If you want to see how you might use <a href="https://rdrr.io/r/base/tapply.html" class="orm:hideurl"><code>tapply()</code></a> or other base techniques to perform other grouped summaries, Hadley has collected a few techniques [in a gist](https://oreil.ly/evpcw).

The final member of the apply family is the titular <a href="https://rdrr.io/r/base/apply.html" class="orm:hideurl"><code>apply()</code></a>, which works with matrices and arrays. In particular, watch out for `apply(df, 2, something)`, which is a slow and potentially dangerous way of doing `lapply(df, something)`. This rarely comes up in data science because we usually work with data frames and not matrices.

# for Loops

`for` loops are the fundamental building block of iteration that both the apply and map families use under the hood. `for` loops are powerful and general tools that are important to learn as you become a more experienced R programmer. The basic structure of a `for` loop looks like this:

```
for (element in vector) {
  # do something with element
}
```

The most straightforward use of `for` loops is to achieve the same effect as <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>walk()</code></a>: call some function with a side effect on each element of a list. For example, in <a href="ch26.html#sec-save-database" data-type="xref">“Writing to a Database”</a>, instead of using `walk()`:

```
paths |> walk(append_file)
```

we could have used a `for` loop:

```
for (path in paths) {
  append_file(path)
}
```

Things get a little trickier if you want to save the output of the `for` loop, for example reading all of the Excel files in a directory like we did in <a href="ch26.html#chp-iteration" data-type="xref">Chapter 26</a>:

```
paths <- dir("data/gapminder", pattern = "\\.xlsx$", full.names = TRUE)
files <- map(paths, readxl::read_excel)
```

There are a few different techniques that you can use, but we recommend being explicit about what the output is going to look like up front. In this case, we’re going to want a list the same length as `paths`, which we can create with <a href="https://rdrr.io/r/base/vector.html" class="orm:hideurl"><code>vector()</code></a>:

```
files <- vector("list", length(paths))
```

Then instead of iterating over the elements of `paths`, we’ll iterate over their indices, using <a href="https://rdrr.io/r/base/seq.html" class="orm:hideurl"><code>seq_along()</code></a> to generate one index for each element of `paths`:

```
seq_along(paths)
#>  [1]  1  2  3  4  5  6  7  8  9 10 11 12
```

Using the indices is important because it allows us to link to each position in the input with the corresponding position in the output:

```
for (i in seq_along(paths)) {
  files[[i]] <- readxl::read_excel(paths[[i]])
}
```

To combine the list of tibbles into a single tibble, you can use <a href="https://rdrr.io/r/base/do.call.html" class="orm:hideurl"><code>do.call()</code></a> + <a href="https://rdrr.io/r/base/cbind.html" class="orm:hideurl"><code>rbind()</code></a>:

```
do.call(rbind, files)
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

Rather than making a list and saving the results as we go, a simpler approach is to build up the data frame piece by piece:

```
out <- NULL
for (path in paths) {
  out <- rbind(out, readxl::read_excel(path))
}
```

We recommend avoiding this pattern because it can become slow when the vector is long. This is the source of the persistent canard that `for` loops are slow: they’re not, but iteratively growing a vector is.

# Plots

Many R users who don’t otherwise use the tidyverse prefer ggplot2 for plotting due to helpful features such as sensible defaults, automatic legends, and a modern look. However, base R plotting functions can still be useful because they’re so concise—it takes very little typing to do a basic exploratory plot.

There are two main types of base plot you’ll see in the wild: scatterplots and histograms, produced with <a href="https://rdrr.io/r/graphics/plot.default.html" class="orm:hideurl"><code>plot()</code></a> and <a href="https://rdrr.io/r/graphics/hist.html" class="orm:hideurl"><code>hist()</code></a>, respectively. Here’s a quick example from the `diamonds` dataset:

```
# Left
hist(diamonds$carat)

# Right
plot(diamonds$carat, diamonds$price)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_27in01.png" alt="On the left, histogram of carats of diamonds, ranging from 0 to 5 carats. The distribution is unimodal and right-skewed. On the right, scatterplot of price versus carat of diamonds, showing a positive relationship that fans out as both price and carat increases. The scatterplot shows very few diamonds bigger than 3 carats compared to diamonds between 0 to 3 carats." />
</figure>

Note that base plotting functions work with vectors, so you need to pull columns out of the data frame using `$` or some other technique.

# Summary

In this chapter, we showed you a selection of base R functions useful for subsetting and iteration. Compared to approaches discussed elsewhere in the book, these functions tend to have more of a “vector” flavor than a “data frame” flavor because base R functions tend to take individual vectors, rather than a data frame and some column specification. This often makes life easier for programming and so becomes more important as you write more functions and begin to write your own packages.

This chapter concludes the programming section of the book. You made a solid start on your journey to becoming not just a data scientist who uses R, but a data scientist who can *program* in R. We hope these chapters have sparked your interest in programming and that you’re looking forward to learning more outside of this book.

<sup>[1](ch27.html#idm44771263328096-marker)</sup> Read the [Selecting multiple elements section](https://oreil.ly/VF0sY) in *Advanced R* to see how you can also subset a data frame like it is a 1D object and how you can subset it with a matrix.

<sup>[2](ch27.html#idm44771262898928-marker)</sup> But it doesn’t handle grouped data frames differently, and it doesn’t support selection helper functions like <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>starts_with()</code></a>.

<sup>[3](ch27.html#idm44771262249040-marker)</sup> It just lacks convenient features such as progress bars and reporting which element caused the problem if there’s an error.
