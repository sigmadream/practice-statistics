# Chapter 10. Exploratory Data Analysis

# Introduction

This chapter will show you how to use visualization and transformation to explore your data in a systematic way, a task that statisticians call *exploratory data analysis*, or EDA for short. EDA is an iterative cycle. You:

1.  Generate questions about your data.

2.  Search for answers by visualizing, transforming, and modeling your data.

3.  Use what you learn to refine your questions and/or generate new questions.

EDA is not a formal process with a strict set of rules. More than anything, EDA is a state of mind. During the initial phases of EDA you should feel free to investigate every idea that occurs to you. Some of these ideas will pan out, and some will be dead ends. As your exploration continues, you will home in on a few particularly productive insights that you’ll eventually write up and communicate to others.

EDA is an important part of any data analysis, even if the primary research questions are handed to you on a platter, because you always need to investigate the quality of your data. Data cleaning is just one application of EDA: you ask questions about whether your data meets your expectations. To do data cleaning, you’ll need to deploy all the tools of EDA: visualization, transformation, and modeling.

## Prerequisites

In this chapter we’ll combine what you’ve learned about dplyr and ggplot2 to interactively ask questions, answer them with data, and then ask new questions.

```
library(tidyverse)
```

# Questions

> “There are no routine statistical questions, only questionable statistical routines.” —Sir David Cox

> “Far better an approximate answer to the right question, which is often vague, than an exact answer to the wrong question, which can always be made precise.” —John Tukey

Your goal during EDA is to develop an understanding of your data. The easiest way to do this is to use questions as tools to guide your investigation. When you ask a question, the question focuses your attention on a specific part of your dataset and helps you decide which graphs, models, or transformations to make.

EDA is fundamentally a creative process. And like most creative processes, the key to asking *quality* questions is to generate a large *quantity* of questions. It is difficult to ask revealing questions at the start of your analysis because you do not know what insights can be gleaned from your dataset. On the other hand, each new question that you ask will expose you to a new aspect of your data and increase your chance of making a discovery. You can quickly drill down into the most interesting parts of your data—and develop a set of thought-provoking questions—if you follow up each question with a new question based on what you find.

There is no rule about which questions you should ask to guide your research. However, two types of questions will always be useful for making discoveries within your data. You can loosely word these questions as:

1.  What type of variation occurs within my variables?

2.  What type of covariation occurs between my variables?

The rest of this chapter will look at these two questions. We’ll explain what variation and covariation are, and we’ll show you several ways to answer each question.

# Variation

*Variation* is the tendency of the values of a variable to change from measurement to measurement. You can see variation easily in real life; if you measure any continuous variable twice, you will get two different results. This is true even if you measure quantities that are constant, like the speed of light. Each of your measurements will include a small amount of error that varies from measurement to measurement. Variables can also vary if you measure across different subjects (e.g., the eye colors of different people) or at different times (e.g., the energy levels of an electron at different moments). Every variable has its own pattern of variation, which can reveal interesting information about how it varies between measurements on the same observation as well as across observations. The best way to understand that pattern is to visualize the distribution of the variable’s values, which you’ve learned about in <a href="ch01.html#chp-data-visualize" data-type="xref">Chapter 1</a>.

We’ll start our exploration by visualizing the distribution of weights (`carat`) of about 54,000 diamonds from the `diamonds` dataset. Since `carat` is a numerical variable, we can use a histogram:

```
ggplot(diamonds, aes(x = carat)) +
  geom_histogram(binwidth = 0.5)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in01.png" alt="A histogram of carats of diamonds, with the x-axis ranging from 0 to 4.5 and the y-axis ranging from 0 to 30000. The distribution is right skewed with very few diamonds in the bin centered at 0, almost 30000 diamonds in the bin centered at 0.5, approximately 15000 diamonds in the bin centered at 1, and much fewer, approximately 5000 diamonds in the bin centered at 1.5. Beyond this, there&#39;s a trailing tail." />
</figure>

Now that you can visualize variation, what should you look for in your plots? And what type of follow-up questions should you ask? We’ve put together a list in the next section of the most useful types of information that you will find in your graphs, along with some follow-up questions for each type of information. The key to asking good follow-up questions will be to rely on your curiosity (what do you want to learn more about?) as well as your skepticism (how could this be misleading?).

## Typical Values

In both bar charts and histograms, tall bars show the common values of a variable, and shorter bars show less-common values. Places that do not have bars reveal values that were not seen in your data. To turn this information into useful questions, look for anything unexpected:

- Which values are the most common? Why?

- Which values are rare? Why? Does that match your expectations?

- Can you see any unusual patterns? What might explain them?

Let’s take a look at the distribution of `carat` for smaller diamonds:

```
smaller <- diamonds |> 
  filter(carat < 3)

ggplot(smaller, aes(x = carat)) +
  geom_histogram(binwidth = 0.01)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in02.png" alt="A histogram of carats of diamonds, with the x-axis ranging from 0 to 3 and the y-axis ranging from 0 to roughly 2500. The binwidth is quite narrow (0.01), resulting in a very large number of skinny bars. The distribution is right skewed, with many peaks followed by bars in decreasing heights, until a sharp increase at the next peak." />
</figure>

This histogram suggests several interesting questions:

- Why are there more diamonds at whole carats and common fractions of carats?

- Why are there more diamonds slightly to the right of each peak than there are slightly to the left of each peak?

Visualizations can also reveal clusters, which suggest that subgroups exist in your data. To understand the subgroups, ask:

- How are the observations within each subgroup similar to each other?

- How are the observations in separate clusters different from each other?

- How can you explain or describe the clusters?

- Why might the appearance of clusters be misleading?

Some of these questions can be answered with the data, while some will require domain expertise about the data. Many of them will prompt you to explore a relationship *between* variables, for example, to see if the values of one variable can explain the behavior of another variable. We’ll get to that shortly.

## Unusual Values

Outliers are observations that are unusual, in other words, data points that don’t seem to fit the pattern. Sometimes outliers are data entry errors, sometimes they are simply values at the extremes that happened to be observed in this data collection, and other times they suggest important new discoveries. When you have a lot of data, outliers are sometimes difficult to see in a histogram. For example, take the distribution of the `y` variable from the `diamonds` dataset. The only evidence of outliers is the unusually wide limits on the x-axis.

```
ggplot(diamonds, aes(x = y)) + 
  geom_histogram(binwidth = 0.5)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in03.png" alt="A histogram of lengths of diamonds. The x-axis ranges from 0 to 60 and the y-axis ranges from 0 to 12000. There is a peak around 5, and the data appear to be completely clustered around the peak." />
</figure>

There are so many observations in the common bins that the rare bins are very short, making it difficult to see them (although maybe if you stare intently at 0, you’ll spot something). To make it easy to see the unusual values, we need to zoom to small values of the y-axis with <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>:

```
ggplot(diamonds, aes(x = y)) + 
  geom_histogram(binwidth = 0.5) +
  coord_cartesian(ylim = c(0, 50))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in04.png" alt="A histogram of lengths of diamonds. The x-axis ranges from 0 to 60 and the y-axis ranges from 0 to 50. There is a peak around 5, and the data appear to be completely clustered around the peak. Other than those data, there is one bin at 0 with a height of about 8, one a little over 30 with a height of 1 and another one a little below 60 with a height of 1." />
</figure>

<a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a> also has an <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>xlim()</code></a> argument for when you need to zoom into the x-axis. ggplot2 also has <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>xlim()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>ylim()</code></a> functions that work slightly differently: they throw away the data outside the limits.

This allows us to see that there are three unusual values: 0, ~30, and ~60. We pluck them out with dplyr:

```
unusual <- diamonds |> 
  filter(y < 3 | y > 20) |> 
  select(price, x, y, z) |>
  arrange(y)
unusual
#> # A tibble: 9 × 4
#>   price     x     y     z
#>   <int> <dbl> <dbl> <dbl>
#> 1  5139  0      0    0   
#> 2  6381  0      0    0   
#> 3 12800  0      0    0   
#> 4 15686  0      0    0   
#> 5 18034  0      0    0   
#> 6  2130  0      0    0   
#> 7  2130  0      0    0   
#> 8  2075  5.15  31.8  5.12
#> 9 12210  8.09  58.9  8.06
```

The `y` variable measures one of the three dimensions of these diamonds, in mm. We know that diamonds can’t have a width of 0mm, so these values must be incorrect. By doing EDA, we have discovered missing data that were coded as 0, which we never would have found by simply searching for `NA`s. Going forward we might choose to re-code these values as `NA`s to prevent misleading calculations. We might also suspect that measurements of 32mm and 59mm are implausible: those diamonds are more than an inch long but don’t cost hundreds of thousands of dollars!

It’s good practice to repeat your analysis with and without the outliers. If they have minimal effect on the results and you can’t figure out why they’re there, it’s reasonable to omit them and move on. However, if they have a substantial effect on your results, you shouldn’t drop them without justification. You’ll need to figure out what caused them (e.g., a data entry error) and disclose that you removed them in your write-up.

## Exercises

1.  Explore the distribution of each of the `x`, `y`, and `z` variables in `diamonds`. What do you learn? Think about a diamond and how you might decide which dimension is the length, width, and depth.

2.  Explore the distribution of `price`. Do you discover anything unusual or surprising? (Hint: Carefully think about the `binwidth` and make sure you try a wide range of values.)

3.  How many diamonds are 0.99 carat? How many are 1 carat? What do you think is the cause of the difference?

4.  Compare and contrast <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>xlim()</code></a> or <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>ylim()</code></a> when zooming in on a histogram. What happens if you leave `binwidth` unset? What happens if you try to zoom so only half a bar shows?

# Unusual Values

If you’ve encountered unusual values in your dataset and simply want to move on to the rest of your analysis, you have two options:

1.  Drop the entire row with the strange values:

    ```
    diamonds2 <- diamonds |> 
      filter(between(y, 3, 20))
    ```

    We don’t recommend this option because one invalid value doesn’t imply that all the other values for that observation are also invalid. Additionally, if you have low-quality data, by the time that you’ve applied this approach to every variable you might find that you don’t have any data left!

2.  Instead, we recommend replacing the unusual values with missing values. The easiest way to do this is to use <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> to replace the variable with a modified copy. You can use the <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a> function to replace unusual values with `NA`:

    ```
    diamonds2 <- diamonds |> 
      mutate(y = if_else(y < 3 | y > 20, NA, y))
    ```

It’s not obvious where you should plot missing values, so ggplot2 doesn’t include them in the plot, but it does warn that they’ve been removed:

```
ggplot(diamonds2, aes(x = x, y = y)) + 
  geom_point()
#> Warning: Removed 9 rows containing missing values (`geom_point()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in05.png" alt="A scatterplot of widths vs. lengths of diamonds. There is a strong, linear association between the two variables. All but one of the diamonds has length greater than 3. The one outlier has a length of 0 and a width of about 6.5." />
</figure>

To suppress that warning, set `na.rm = TRUE`:

```
ggplot(diamonds2, aes(x = x, y = y)) + 
  geom_point(na.rm = TRUE)
```

Other times you want to understand what makes observations with missing values different to observations with recorded values. For example, in <a href="https://rdrr.io/pkg/nycflights13/man/flights.html" class="orm:hideurl"><code>nycflights13::flights,</code></a><sup><a href="ch10.html#idm44771307799104" id="idm44771307799104-marker" data-type="noteref">1</a></sup> missing values in the `dep_time` variable indicate that the flight was cancelled. So you might want to compare the scheduled departure times for cancelled and noncancelled times. You can do this by making a new variable, using <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a> to check whether `dep_time` is missing.

```
nycflights13::flights |> 
  mutate(
    cancelled = is.na(dep_time),
    sched_hour = sched_dep_time %/% 100,
    sched_min = sched_dep_time %% 100,
    sched_dep_time = sched_hour + (sched_min / 60)
  ) |> 
  ggplot(aes(x = sched_dep_time)) + 
  geom_freqpoly(aes(color = cancelled), binwidth = 1/4)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in06.png" alt="A frequency polygon of scheduled departure times of flights. Two lines represent flights that are cancelled and not cancelled. The x-axis ranges from 0 to 25 minutes and the y-axis ranges from 0 to 10000. The number of flights not cancelled are much higher than those cancelled." />
</figure>

However, this plot isn’t great because there are many more noncancelled flights than cancelled flights. In the next section, we’ll explore some techniques for improving this comparison.

## Exercises

1.  What happens to missing values in a histogram? What happens to missing values in a bar chart? Why is there a difference in how missing values are handled in histograms and bar charts?

2.  What does `na.rm = TRUE` do in <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a> and <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>?

3.  Re-create the frequency plot of `scheduled_dep_time` colored by whether the flight was cancelled or not. Also facet by the `cancelled` variable. Experiment with different values of the `scales` variable in the faceting function to mitigate the effect of more noncancelled flights than cancelled flights.

# Covariation

If variation describes the behavior *within* a variable, covariation describes the behavior *between* variables. *Covariation* is the tendency for the values of two or more variables to vary together in a related way. The best way to spot covariation is to visualize the relationship between two or more variables.

## A Categorical and a Numerical Variable

For example, let’s explore how the price of a diamond varies with its quality (measured by `cut`) using <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a>:

```
ggplot(diamonds, aes(x = price)) + 
  geom_freqpoly(aes(color = cut), binwidth = 500, linewidth = 0.75)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in07.png" alt="A frequency polygon of prices of diamonds where each cut of carat (Fair, Good, Very Good, Premium, and Ideal) is represented with a different color line. The x-axis ranges from 0 to 30000 and the y-axis ranges from 0 to 5000. The lines overlap a great deal, suggesting similar frequency distributions of prices of diamonds. One notable feature is that Ideal diamonds have the highest peak around 1500." />
</figure>

Note that ggplot2 uses an ordered color scale for `cut` because it’s defined as an ordered factor variable in the data. You’ll learn more about these in <a href="ch16.html#sec-ordered-factors" data-type="xref">“Ordered Factors”</a>.

The default appearance of <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a> is not that useful here because the height, determined by the overall count, differs so much across `cut`s, making it hard to see the differences in the shapes of their distributions.

To make the comparison easier, we need to swap what is displayed on the y-axis. Instead of displaying count, we’ll display the *density*, which is the count standardized so that the area under each frequency polygon is 1:

```
ggplot(diamonds, aes(x = price, y = after_stat(density))) + 
  geom_freqpoly(aes(color = cut), binwidth = 500, linewidth = 0.75)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in08.png" alt="A frequency polygon of densities of prices of diamonds where each cut of carat (Fair, Good, Very Good, Premium, and Ideal) is represented with a different color line. The x-axis ranges from 0 to 20000. The lines overlap a great deal, suggesting similar density distributions of prices of diamonds. One notable feature is that all but Fair diamonds have high peaks around a price of 1500 and Fair diamonds have a higher mean than others." />
</figure>

Note that we’re mapping the density the `y`, but since `density` is not a variable in the `diamonds` dataset, we need to first calculate it. We use the <a href="https://ggplot2.tidyverse.org/reference/aes_eval.html" class="orm:hideurl"><code>after_stat()</code></a> function to do so.

There’s something rather surprising about this plot: it appears that fair diamonds (the lowest quality) have the highest average price! But maybe that’s because frequency polygons are a little hard to interpret; there’s a lot going on in this plot.

A visually simpler plot for exploring this relationship is using side-by-side boxplots:

```
ggplot(diamonds, aes(x = cut, y = price)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in09.png" alt="Side-by-side boxplots of prices of diamonds by cut. The distribution of prices is right skewed for each cut (Fair, Good, Very Good, Premium, and Ideal). The medians are close to each other, with the median for Ideal diamonds lowest and that for Fair highest." />
</figure>

We see much less information about the distribution, but the boxplots are much more compact so we can more easily compare them (and fit more on one plot). It supports the counterintuitive finding that better-quality diamonds are typically cheaper! In the exercises, you’ll be challenged to figure out why.

`cut` is an ordered factor: fair is worse than good, which is worse than very good and so on. Many categorical variables don’t have such an intrinsic order, so you might want to reorder them to make a more informative display. One way to do that is with <a href="https://forcats.tidyverse.org/reference/fct_reorder.html" class="orm:hideurl"><code>fct_reorder()</code></a>. You’ll learn more about that function in <a href="ch16.html#sec-modifying-factor-order" data-type="xref">“Modifying Factor Order”</a>, but we wanted to give you a quick preview here because it’s so useful. For example, take the `class` variable in the `mpg` dataset. You might be interested to know how highway mileage varies across classes:

```
ggplot(mpg, aes(x = class, y = hwy)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in10.png" alt="Side-by-side boxplots of highway mileages of cars by class. Classes are on the x-axis (2seaters, compact, midsize, minivan, pickup, subcompact, and suv)." />
</figure>

To make the trend easier to see, we can reorder `class` based on the median value of `hwy`:

```
ggplot(mpg, aes(x = fct_reorder(class, hwy, median), y = hwy)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in11.png" alt="Side-by-side boxplots of highway mileages of cars by class. Classes are on the x-axis and ordered by increasing median highway mileage (pickup, suv, minivan, 2seater, subcompact, compact, and midsize)." />
</figure>

If you have long variable names, <a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a> will work better if you flip it 90°. You can do that by exchanging the x and y aesthetic mappings:

```
ggplot(mpg, aes(x = hwy, y = fct_reorder(class, hwy, median))) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in12.png" alt="Side-by-side boxplots of highway mileages of cars by class. Classes are on the y-axis and ordered by increasing median highway mileage." />
</figure>

### Exercises

1.  Use what you’ve learned to improve the visualization of the departure times of cancelled versus noncancelled flights.

2.  Based on EDA, what variable in the diamonds dataset appears to be most important for predicting the price of a diamond? How is that variable correlated with cut? Why does the combination of those two relationships lead to lower-quality diamonds being more expensive?

3.  Instead of exchanging the x and y variables, add <a href="https://ggplot2.tidyverse.org/reference/coord_flip.html" class="orm:hideurl"><code>coord_flip()</code></a> as a new layer to the vertical boxplot to create a horizontal one. How does this compare to exchanging the variables?

4.  One problem with boxplots is that they were developed in an era of much smaller datasets and tend to display a prohibitively large number of “outlying values.” One approach to remedy this problem is the letter value plot. Install the lvplot package, and try using `geom_lv()` to display the distribution of price versus cut. What do you learn? How do you interpret the plots?

5.  Create a visualization of diamond prices versus a categorical variable from the `diamonds` dataset using <a href="https://ggplot2.tidyverse.org/reference/geom_violin.html" class="orm:hideurl"><code>geom_violin()</code></a>, then a faceted <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_histogram()</code></a>, then a colored <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a>, and then a colored <a href="https://ggplot2.tidyverse.org/reference/geom_density.html" class="orm:hideurl"><code>geom_density()</code></a>. Compare and contrast the four plots. What are the pros and cons of each method of visualizing the distribution of a numerical variable based on the levels of a categorical variable?

6.  If you have a small dataset, it’s sometimes useful to use <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a> to avoid overplotting to more easily see the relationship between a continuous and categorical variable. The ggbeeswarm package provides a number of methods similar to <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>. List them and briefly describe what each one does.

## Two Categorical Variables

To visualize the covariation between categorical variables, you’ll need to count the number of observations for each combination of levels of these categorical variables. One way to do that is to rely on the built-in <a href="https://ggplot2.tidyverse.org/reference/geom_count.html" class="orm:hideurl"><code>geom_count()</code></a>:

```
ggplot(diamonds, aes(x = cut, y = color)) +
  geom_count()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in13.png" alt="A scatterplot of color vs. cut of diamonds. There is one point for each combination of levels of cut (Fair, Good, Very Good, Premium, and Ideal) and color (D, E, F, G, G, I, and J). The sizes of the points represent the number of observations for that combination. The legend indicates that these sizes range between 1000 and 4000." />
</figure>

The size of each circle in the plot displays how many observations occurred at each combination of values. Covariation will appear as a strong correlation between specific x values and specific y values.

Another approach for exploring the relationship between these variables is computing the counts with dplyr:

```
diamonds |> 
  count(color, cut)
#> # A tibble: 35 × 3
#>   color cut           n
#>   <ord> <ord>     <int>
#> 1 D     Fair        163
#> 2 D     Good        662
#> 3 D     Very Good  1513
#> 4 D     Premium    1603
#> 5 D     Ideal      2834
#> 6 E     Fair        224
#> # … with 29 more rows
```

Then visualize with <a href="https://ggplot2.tidyverse.org/reference/geom_tile.html" class="orm:hideurl"><code>geom_tile()</code></a> and the fill aesthetic:

```
diamonds |> 
  count(color, cut) |>  
  ggplot(aes(x = color, y = cut)) +
  geom_tile(aes(fill = n))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in14.png" alt="A tile plot of cut vs. color of diamonds. Each tile represents a cut/color combination and tiles are colored according to the number of observations in each tile. There are more Ideal diamonds than other cuts, with the highest number being Ideal diamonds with color G. Fair diamonds and diamonds with color I are the lowest in frequency." />
</figure>

If the categorical variables are unordered, you might want to use the seriation package to simultaneously reorder the rows and columns to more clearly reveal interesting patterns. For larger plots, you might want to try the heatmaply package, which creates interactive plots.

### Exercises

1.  How could you rescale the previous count dataset to more clearly show the distribution of cut within color, or color within cut?

2.  What different data insights do you get with a segmented bar chart if color is mapped to the `x` aesthetic and `cut` is mapped to the `fill` aesthetic? Calculate the counts that fall into each of the segments.

3.  Use <a href="https://ggplot2.tidyverse.org/reference/geom_tile.html" class="orm:hideurl"><code>geom_tile()</code></a> together with dplyr to explore how average flight departure delays vary by destination and month of year. What makes the plot difficult to read? How could you improve it?

## Two Numerical Variables

You’ve already seen one great way to visualize the covariation between two numerical variables: draw a scatterplot with <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>. You can see covariation as a pattern in the points. For example, you can see a positive relationship between the carat size and price of a diamond: diamonds with more carats have a higher price. The relationship is exponential.

```
ggplot(smaller, aes(x = carat, y = price)) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in15.png" alt="A scatterplot of price vs. carat. The relationship is positive, somewhat strong, and exponential." />
</figure>

(In this section we’ll use the `smaller` dataset to stay focused on the bulk of the diamonds that are smaller than 3 carats.)

Scatterplots become less useful as the size of your dataset grows, because points begin to overplot and pile up into areas of uniform black, making it hard to judge differences in the density of the data across the two-dimensional space as well as making it hard to spot the trend. You’ve already seen one way to fix the problem: using the `alpha` aesthetic to add transparency.

```
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_point(alpha = 1 / 100)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in16.png" alt="A scatterplot of price vs. carat. The relationship is positive, somewhat strong, and exponential. The points are transparent, showing clusters where the number of points is higher than other areas, The most obvious clusters are for diamonds with 1, 1.5, and 2 carats." />
</figure>

But using transparency can be challenging for very large datasets. Another solution is to use bins. Previously you used <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_histogram()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a> to bin in one dimension. Now you’ll learn how to use <a href="https://ggplot2.tidyverse.org/reference/geom_bin_2d.html" class="orm:hideurl"><code>geom_bin2d()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a> to bin in two dimensions.

<a href="https://ggplot2.tidyverse.org/reference/geom_bin_2d.html" class="orm:hideurl"><code>geom_bin2d()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a> divide the coordinate plane into 2D bins and then use a fill color to display how many points fall into each bin. <a href="https://ggplot2.tidyverse.org/reference/geom_bin_2d.html" class="orm:hideurl"><code>geom_bin2d()</code></a> creates rectangular bins. <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a> creates hexagonal bins. You will need to install the hexbin package to use <a href="https://ggplot2.tidyverse.org/reference/geom_hex.html" class="orm:hideurl"><code>geom_hex()</code></a>.

```
ggplot(smaller, aes(x = carat, y = price)) +
  geom_bin2d()

# install.packages("hexbin")
ggplot(smaller, aes(x = carat, y = price)) +
  geom_hex()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in17.png" alt="Plot 1: A binned density plot of price vs. carat. Plot 2: A hexagonal bin plot of price vs. carat. Both plots show that the highest density of diamonds have low carats and low prices." />
</figure>

Another option is to bin one continuous variable so it acts like a categorical variable. Then you can use one of the techniques for visualizing the combination of a categorical and a continuous variable that you learned about. For example, you could bin `carat` and then for each group display a boxplot:

```
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_boxplot(aes(group = cut_width(carat, 0.1)))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in18.png" alt="Side-by-side box plots of price by carat. Each box plot represents diamonds that are 0.1 carats apart in weight. The box plots show that as carat increases the median price increases as well. Additionally, diamonds with 1.5 carats or lower have right skewed price distributions, 1.5 to 2 have roughly symmetric price distributions, and diamonds that weigh more have left skewed distributions. Cheaper, smaller diamonds have outliers on the higher end, more expensive, bigger diamonds have outliers on the lower end." />
</figure>

`cut_width(x, width)`, as used here, divides `x` into bins of width `width`. By default, boxplots look roughly the same (apart from the number of outliers) regardless of how many observations there are, so it’s difficult to tell that each boxplot summarizes a different number of points. One way to show that is to make the width of the boxplot proportional to the number of points with `varwidth = TRUE`.

### Exercises

1.  Instead of summarizing the conditional distribution with a boxplot, you could use a frequency polygon. What do you need to consider when using <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_width()</code></a> versus <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_number()</code></a>? How does that impact a visualization of the 2D distribution of `carat` and `price`?

2.  Visualize the distribution of `carat`, partitioned by `price`.

3.  How does the price distribution of very large diamonds compare to small diamonds? Is it as you expect, or does it surprise you?

4.  Combine two of the techniques you’ve learned to visualize the combined distribution of cut, carat, and price.

5.  Two-dimensional plots reveal outliers that are not visible in one-dimensional plots. For example, some points in the following plot have an unusual combination of `x` and `y` values, which makes the points outliers even though their `x` and `y` values appear normal when examined separately. Why is a scatterplot a better display than a binned plot for this case?

    ```
    diamonds |> 
      filter(x >= 4) |> 
      ggplot(aes(x = x, y = y)) +
      geom_point() +
      coord_cartesian(xlim = c(4, 11), ylim = c(4, 11))
    ```

6.  Instead of creating boxes of equal width with <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_width()</code></a>, we could create boxes that contain roughly equal number of points with <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_number()</code></a>. What are the advantages and disadvantages of this approach?

    ```
    ggplot(smaller, aes(x = carat, y = price)) + 
      geom_boxplot(aes(group = cut_number(carat, 20)))
    ```

# Patterns and Models

If a systematic relationship exists between two variables, it will appear as a pattern in the data. If you spot a pattern, ask yourself:

- Could this pattern be due to coincidence (i.e., random chance)?

- How can you describe the relationship implied by the pattern?

- How strong is the relationship implied by the pattern?

- What other variables might affect the relationship?

- Does the relationship change if you look at individual subgroups of the data?

Patterns in your data provide clues about relationships; i.e., they reveal covariation. If you think of variation as a phenomenon that creates uncertainty, covariation is a phenomenon that reduces it. If two variables covary, you can use the values of one variable to make better predictions about the values of the second. If the covariation is due to a causal relationship (a special case), then you can use the value of one variable to control the value of the second.

Models are a tool for extracting patterns out of data. For example, consider the diamonds data. It’s hard to understand the relationship between cut and price, because cut and carat, and carat and price, are tightly related. It’s possible to use a model to remove the very strong relationship between price and carat to explore the subtleties that remain. The following code fits a model that predicts `price` from `carat` and then computes the residuals (the difference between the predicted value and the actual value). The residuals give us a view of the price of the diamond, once the effect of carat has been removed. Note that instead of using the raw values of `price` and `carat`, we log transform them first and fit a model to the log-transformed values. Then, we exponentiate the residuals to put them back in the scale of raw prices.

```
library(tidymodels)

diamonds <- diamonds |>
  mutate(
    log_price = log(price),
    log_carat = log(carat)
  )

diamonds_fit <- linear_reg() |>
  fit(log_price ~ log_carat, data = diamonds)

diamonds_aug <- augment(diamonds_fit, new_data = diamonds) |>
  mutate(.resid = exp(.resid))

ggplot(diamonds_aug, aes(x = carat, y = .resid)) + 
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in19.png" alt="A scatter plot of residuals vs. carat of diamonds. The x-axis ranges from 0 to 5, the y-axis ranges from 0 to almost 4. Much of the data are clustered around low values of carat and residuals. There is a clear, curved pattern showing decrease in residuals as carat increases." />
</figure>

Once you’ve removed the strong relationship between carat and price, you can see what you expect in the relationship between cut and price: relative to their size, better-quality diamonds are more expensive.

```
ggplot(diamonds_aug, aes(x = cut, y = .resid)) + 
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in20.png" alt="Side-by-side box plots of residuals by cut. The x-axis displays the various cuts (Fair to Ideal), the y-axis ranges from 0 to almost 5. The medians are quite similar, between roughly 0.75 to 1.25. Each of the distributions of residuals is right skewed, with many outliers on the higher end." />
</figure>

We’re not discussing modeling in this book because understanding what models are and how they work is easiest once you have tools for data wrangling and programming in hand.

# Summary

In this chapter you learned a variety of tools to help you understand the variation within your data. You saw a technique that works with a single variable at a time and with a pair of variables. This might seem painfully restrictive if you have tens or hundreds of variables in your data, but they’re the foundation upon which all other techniques are built.

In the next chapter, we’ll focus on the tools we can use to communicate our results.

<sup>[1](ch10.html#idm44771307799104-marker)</sup> Remember that when we need to be explicit about where a function (or dataset) comes from, we’ll use the special form `package::function()` or `package::dataset`.
