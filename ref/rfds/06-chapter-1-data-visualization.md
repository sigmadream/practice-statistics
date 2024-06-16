# Chapter 1. Data Visualization

# Introduction

> “The simple graph has brought more information to the data analyst’s mind than any other device.” —John Tukey

R has several systems for making graphs, but ggplot2 is one of the most elegant and most versatile. ggplot2 implements the *grammar of graphics*, a coherent system for describing and building graphs. With ggplot2, you can do more faster by learning one system and applying it in many places.

This chapter will teach you how to visualize your data using ggplot2. We will start by creating a simple scatterplot and use it to introduce aesthetic mappings and geometric objects—the fundamental building blocks of ggplot2. We will then walk you through visualizing distributions of single variables as well as visualizing relationships between two or more variables. We’ll finish off with saving your plots and troubleshooting tips.

## Prerequisites

This chapter focuses on ggplot2, one of the core packages in the tidyverse. To access the datasets, help pages, and functions used in this chapter, load the tidyverse by running:

```
library(tidyverse)
#> ── Attaching core tidyverse packages ───────────────────── tidyverse 2.0.0 ──
#> ✔ dplyr     1.1.0.9000     ✔ readr     2.1.4     
#> ✔ forcats   1.0.0          ✔ stringr   1.5.0     
#> ✔ ggplot2   3.4.1          ✔ tibble    3.1.8     
#> ✔ lubridate 1.9.2          ✔ tidyr     1.3.0     
#> ✔ purrr     1.0.1          
#> ── Conflicts ─────────────────────────────────────── tidyverse_conflicts() ──
#> ✖ dplyr::filter() masks stats::filter()
#> ✖ dplyr::lag()    masks stats::lag()
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all 
#>   conflicts to become errors
```

That one line of code loads the core tidyverse, the packages that you will use in almost every data analysis. It also tells you which functions from the tidyverse conflict with functions in base R (or from other packages you might have loaded).<sup><a href="ch01.html#idm44771333724368" id="idm44771333724368-marker" data-type="noteref">1</a></sup>

If you run this code and get the error message `there is no package called 'tidyverse'`, you’ll need to first install it, and then run <a href="https://rdrr.io/r/base/library.html" class="orm:hideurl"><code>library()</code></a> once again:

```
install.packages("tidyverse")
library(tidyverse)
```

You need to install a package only once, but you need to load it every time you start a new session.

In addition to tidyverse, we will use the palmerpenguins package, which includes the `penguins` dataset containing body measurements for penguins on three islands in the Palmer Archipelago, and the ggthemes package, which offers a colorblind safe color palette.

```
library(palmerpenguins)
library(ggthemes)
```

# First Steps

Do penguins with longer flippers weigh more or less than penguins with shorter flippers? You probably already have an answer, but try to make your answer precise. What does the relationship between flipper length and body mass look like? Is it positive? Negative? Linear? Nonlinear? Does the relationship vary by the species of the penguin? How about by the island where the penguin lives? Let’s create visualizations that we can use to answer these questions.

## The penguins Data Frame

You can test your answers to these questions with the `penguins` data frame found in palmerpenguins (aka <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>palmerpenguins::penguins</code></a>). A data frame is a rectangular collection of variables (in the columns) and observations (in the rows). `penguins` contains 344 observations collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER.<sup><a href="ch01.html#idm44771333851472" id="idm44771333851472-marker" data-type="noteref">2</a></sup>

To make the discussion easier, let’s define some terms:

Variable  
A quantity, quality, or property that you can measure.

Value  
The state of a variable when you measure it. The value of a variable may change from measurement to measurement.

Observation  
A set of measurements made under similar conditions (you usually make all of the measurements in an observation at the same time and on the same object). An observation will contain several values, each associated with a different variable. We’ll sometimes refer to an observation as a *data point*.

Tabular data  
A set of values, each associated with a variable and an observation. Tabular data is *tidy* if each value is placed in its own “cell,” each variable in its own column, and each observation in its own row.

In this context, a variable refers to an attribute of all the penguins, and an observation refers to all the attributes of a single penguin.

Type the name of the data frame in the console, and R will print a preview of its contents. Note that it says `tibble` on top of this preview. In the tidyverse, we use special data frames called *tibbles* that you will learn about soon.

```
penguins
#> # A tibble: 344 × 8
#>   species island    bill_length_mm bill_depth_mm flipper_length_mm
#>   <fct>   <fct>              <dbl>         <dbl>             <int>
#> 1 Adelie  Torgersen           39.1          18.7               181
#> 2 Adelie  Torgersen           39.5          17.4               186
#> 3 Adelie  Torgersen           40.3          18                 195
#> 4 Adelie  Torgersen           NA            NA                  NA
#> 5 Adelie  Torgersen           36.7          19.3               193
#> 6 Adelie  Torgersen           39.3          20.6               190
#> # … with 338 more rows, and 3 more variables: body_mass_g <int>, sex <fct>,
#> #   year <int>
```

This data frame contains eight columns. For an alternative view, where you can see all variables and the first few observations of each variable, use <a href="https://pillar.r-lib.org/reference/glimpse.html" class="orm:hideurl"><code>glimpse()</code></a>. Or, if you’re in RStudio, run `View(penguins)` to open an interactive data viewer.

```
glimpse(penguins)
#> Rows: 344
#> Columns: 8
#> $ species           <fct> Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, A…
#> $ island            <fct> Torgersen, Torgersen, Torgersen, Torgersen, Torge…
#> $ bill_length_mm    <dbl> 39.1, 39.5, 40.3, NA, 36.7, 39.3, 38.9, 39.2, 34.…
#> $ bill_depth_mm     <dbl> 18.7, 17.4, 18.0, NA, 19.3, 20.6, 17.8, 19.6, 18.…
#> $ flipper_length_mm <int> 181, 186, 195, NA, 193, 190, 181, 195, 193, 190, …
#> $ body_mass_g       <int> 3750, 3800, 3250, NA, 3450, 3650, 3625, 4675, 347…
#> $ sex               <fct> male, female, female, NA, female, male, female, m…
#> $ year              <int> 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2…
```

Among the variables in `penguins` are:

`species`  
A penguin’s species (Adelie, Chinstrap, or Gentoo)

`flipper_length_mm`  
The length of a penguin’s flipper, in millimeters

`body_mass_g`  
The body mass of a penguin, in grams

To learn more about `penguins`, open its help page by running <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>?penguins</code></a>.

## Ultimate Goal

Our ultimate goal in this chapter is to re-create the following visualization displaying the relationship between flipper lengths and body masses of these penguins, taking into consideration the species of the penguin.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in01.png" alt="A scatterplot of body mass vs. flipper length of penguins, with a best fit line of the relationship between these two variables overlaid. The plot displays a positive, fairly linear, and relatively strong relationship between these two variables. Species (Adelie, Chinstrap, and Gentoo) are represented with different colors and shapes. The relationship between body mass and flipper length is roughly the same for these three species, and Gentoo penguins are larger than penguins from the other two species." />
</figure>

## Creating a ggplot

Let’s re-create this plot step by step.

With ggplot2, you begin a plot with the function <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>, defining a plot object that you then add *layers* to. The first argument of <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> is the dataset to use in the graph, so `ggplot(data = penguins)` creates an empty graph that is primed to display the `penguins` data, but since we haven’t told it how to visualize it yet, for now it’s empty. This is not a very exciting plot, but you can think of it like an empty canvas where you’ll paint the remaining layers of your plot.

```
ggplot(data = penguins)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in02.png" alt="A blank, gray plot area." />
</figure>

Next, we need to tell <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> how the information from our data will be visually represented. The `mapping` argument of the <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> function defines how variables in your dataset are mapped to visual properties (*aesthetics*) of your plot. The `mapping` argument is always defined in the <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a> function, and the `x` and `y` arguments of <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a> specify which variables to map to the x- and y-axes. For now, we will map only the flipper length to the `x` aesthetic and body mass to the `y` aesthetic. ggplot2 looks for the mapped variables in the `data` argument, in this case, `penguins`.

The following plot shows the result of adding these mappings.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in03.png" alt="The plot shows flipper length on the x-axis, with values that range from 170 to 230, and body mass on the y-axis, with values that range from 3000 to 6000." />
</figure>

Our empty canvas now has more structure—it’s clear where flipper lengths will be displayed (on the x-axis) and where body masses will be displayed (on the y-axis). But the penguins themselves are not yet on the plot. This is because we have not yet articulated, in our code, how to represent the observations from our data frame on our plot.

To do so, we need to define a *geom*: the geometrical object that a plot uses to represent data. These geometric objects are made available in ggplot2 with functions that start with `geom_`. People often describe plots by the type of geom that the plot uses. For example, bar charts use bar geoms (<a href="https://ggplot2.tidyverse.org/reference/geom_bar.html" class="orm:hideurl"><code>geom_bar()</code></a>), line charts use line geoms (<a href="https://ggplot2.tidyverse.org/reference/geom_path.html" class="orm:hideurl"><code>geom_line()</code></a>), boxplots use boxplot geoms (<a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>), scatterplots use point geoms (<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>), and so on.

The function <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a> adds a layer of points to your plot, which creates a scatterplot. ggplot2 comes with many geom functions, and each adds a different type of layer to a plot. You’ll learn a whole bunch of geoms throughout the book, particularly in <a href="ch09.html#chp-layers" data-type="xref">Chapter 9</a>.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point()
#> Warning: Removed 2 rows containing missing values (`geom_point()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in04.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, linear, and relatively strong relationship between these two variables." />
</figure>

Now we have something that looks like what we might think of as a “scatterplot.” It doesn’t yet match our “ultimate goal” plot, but using this plot we can start answering the question that motivated our exploration: “What does the relationship between flipper length and body mass look like?” The relationship appears to be positive (as flipper length increases, so does body mass), fairly linear (the points are clustered around a line instead of a curve), and moderately strong (there isn’t too much scatter around such a line). Penguins with longer flippers are generally larger in terms of their body mass.

Before we add more layers to this plot, let’s pause for a moment and review the warning message we got:

> Removed 2 rows containing missing values (<a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>).

We’re seeing this message because there are two penguins in our dataset with missing body mass and/or flipper length values and ggplot2 has no way of representing them on the plot without both of these values. Like R, ggplot2 subscribes to the philosophy that missing values should never silently go missing. This type of warning is probably one of the most common types of warnings you will see when working with real data—missing values are a common issue, and you’ll learn more about them throughout the book, particularly in <a href="ch18.html#chp-missing-values" data-type="xref">Chapter 18</a>. For the remaining plots in this chapter we will suppress this warning so it’s not printed alongside every single plot we make.

## Adding Aesthetics and Layers

Scatterplots are useful for displaying the relationship between two numerical variables, but it’s always a good idea to be skeptical of any apparent relationship between two variables and ask if there may be other variables that explain or change the nature of this apparent relationship. For example, does the relationship between flipper length and body mass differ by species? Let’s incorporate species into our plot and see if this reveals any additional insights into the apparent relationship between these variables. We will do this by representing species with different colored points.

To achieve this, will we need to modify the aesthetic or the geom? If you guessed “in the aesthetic mapping, inside of <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a>,” you’re already getting the hang of creating data visualizations with ggplot2! And if not, don’t worry. Throughout the book you will make many more ggplots and have many more opportunities to check your intuition as you make them.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g, color = species)
) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in05.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, fairly linear, and relatively strong relationship between these two variables. Species (Adelie, Chinstrap, and Gentoo) are represented with different colors." />
</figure>

When a categorical variable is mapped to an aesthetic, ggplot2 will automatically assign a unique value of the aesthetic (here a unique color) to each unique level of the variable (each of the three species), a process known as *scaling*. ggplot2 will also add a legend that explains which values correspond to which levels.

Now let’s add one more layer: a smooth curve displaying the relationship between body mass and flipper length. Before you proceed, refer to the previous code, and think about how we can add this to our existing plot.

Since this is a new geometric object representing our data, we will add a new geom as a layer on top of our point geom: <a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>. And we will specify that we want to draw the line of best fit based on a `l`inear `m`odel with `method = "lm"`.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g, color = species)
) +
  geom_point() +
  geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in06.png" alt="A scatterplot of body mass vs. flipper length of penguins. Overlaid on the scatterplot are three smooth curves displaying the relationship between these variables for each species (Adelie, Chinstrap, and Gentoo). Different penguin species are plotted in different colors for the points and the smooth curves." />
</figure>

We have successfully added lines, but this plot doesn’t look like the plot from <a href="#sec-ultimate-goal" data-type="xref">“Ultimate Goal”</a>, which has only one line for the entire dataset as opposed to separate lines for each of the penguin species.

When aesthetic mappings are defined in <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a>, at the *global* level, they’re passed down to each of the subsequent geom layers of the plot. However, each geom function in ggplot2 can also take a `mapping` argument, which allows for aesthetic mappings at the *local* level that are added to those inherited from the global level. Since we want points to be colored based on species but don’t want the lines to be separated out for them, we should specify `color = species` for <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a> only.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point(mapping = aes(color = species)) +
  geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in07.png" alt="A scatterplot of body mass vs. flipper length of penguins. Overlaid on the scatterplot is a single line of best fit displaying the relationship between these variables for each species (Adelie, Chinstrap, and Gentoo). Different penguin species are plotted in different colors for the points only." />
</figure>

Voilà! We have something that looks very much like our ultimate goal, though it’s not yet perfect. We still need to use different shapes for each species of penguins and improve labels.

It’s generally not a good idea to represent information using only colors on a plot, as people perceive colors differently due to color blindness or other color vision differences. Therefore, in addition to color, we can map `species` to the `shape` aesthetic.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point(mapping = aes(color = species, shape = species)) +
  geom_smooth(method = "lm")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in08.png" alt="A scatterplot of body mass vs. flipper length of penguins. Overlaid on the scatterplot is a single line of best fit displaying the relationship between these variables for each species (Adelie, Chinstrap, and Gentoo). Different penguin species are plotted in different colors and shapes for the points only." />
</figure>

Note that the legend is automatically updated to reflect the different shapes of the points as well.

Finally, we can improve the labels of our plot using the <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a> function in a new layer. Some of the arguments to <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a> might be self-explanatory: `title` adds a title, and `subtitle` adds a subtitle to the plot. Other arguments match the aesthetic mappings: `x` is the x-axis label, `y` is the y-axis label, and `color` and `shape` define the label for the legend. In addition, we can improve the color palette to be color-blind safe with the <a href="https://rdrr.io/pkg/ggthemes/man/colorblind.html" class="orm:hideurl"><code>scale_color_colorblind()</code></a> function from the ggthemes package.

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point(aes(color = species, shape = species)) +
  geom_smooth(method = "lm") +
  labs(
    title = "Body mass and flipper length",
    subtitle = "Dimensions for Adelie, Chinstrap, and Gentoo Penguins",
    x = "Flipper length (mm)", y = "Body mass (g)",
    color = "Species", shape = "Species"
  ) +
  scale_color_colorblind()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in09.png" alt="A scatterplot of body mass vs. flipper length of penguins, with a line of best fit displaying the relationship between these two variables overlaid. The plot displays a positive, fairly linear, and relatively strong relationship between these two variables. Species (Adelie, Chinstrap, and Gentoo) are represented with different colors and shapes. The relationship between body mass and flipper length is roughly the same for these three species, and Gentoo penguins are larger than penguins from the other two species." />
</figure>

We finally have a plot that perfectly matches our “ultimate goal”!

## Exercises

1.  How many rows are in `penguins`? How many columns?

2.  What does the `bill_depth_mm` variable in the `penguins` data frame describe? Read the help for <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>?penguins</code></a> to find out.

3.  Make a scatterplot of `bill_depth_mm` versus `bill_length_mm`. That is, make a scatterplot with `bill_depth_mm` on the y-axis and `bill_length_mm` on the x-axis. Describe the relationship between these two variables.

4.  What happens if you make a scatterplot of `species` versus `bill_depth_mm`? What might be a better choice of geom?

5.  Why does the following give an error, and how would you fix it?

    ```
    ggplot(data = penguins) + 
      geom_point()
    ```

6.  What does the `na.rm` argument do in <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>? What is the default value of the argument? Create a scatterplot where you successfully use this argument set to `TRUE`.

7.  Add the following caption to the plot you made in the previous exercise: “Data come from the palmerpenguins package.” Hint: Take a look at the documentation for <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>.

8.  Re-create the following visualization. What aesthetic should `bill_depth_mm` be mapped to? And should it be mapped at the global level or at the geom level?

    <figure>
    <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in10.png" alt="A scatterplot of body mass vs. flipper length of penguins, colored by bill depth. A smooth curve of the relationship between body mass and flipper length is overlaid. The relationship is positive, fairly linear, and moderately strong." />
    </figure>

9.  Run this code in your head and predict what the output will look like. Then, run the code in R and check your predictions.

    ```
    ggplot(
      data = penguins,
      mapping = aes(x = flipper_length_mm, y = body_mass_g, color = island)
    ) +
      geom_point() +
      geom_smooth(se = FALSE)
    ```

10. Will these two graphs look different? Why/why not?

    ```
    ggplot(
      data = penguins,
      mapping = aes(x = flipper_length_mm, y = body_mass_g)
    ) +
      geom_point() +
      geom_smooth()

    ggplot() +
      geom_point(
        data = penguins,
        mapping = aes(x = flipper_length_mm, y = body_mass_g)
      ) +
      geom_smooth(
        data = penguins,
        mapping = aes(x = flipper_length_mm, y = body_mass_g)
      )
    ```

# ggplot2 Calls

As we move on from these introductory sections, we’ll transition to a more concise expression of ggplot2 code. So far we’ve been very explicit, which is helpful when you are learning:

```
ggplot(
  data = penguins,
  mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
  geom_point()
```

Typically, the first one or two arguments to a function are so important that you should know them by heart. The first two arguments to <a href="https://ggplot2.tidyverse.org/reference/ggplot.html" class="orm:hideurl"><code>ggplot()</code></a> are `data` and `mapping`; in the remainder of the book, we won’t supply those names. That saves typing and, by reducing the amount of extra text, makes it easier to see what’s different between plots. That’s a really important programming concern that we’ll come back to in <a href="ch25.html#chp-functions" data-type="xref">Chapter 25</a>.

Rewriting the previous plot more concisely yields:

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) + 
  geom_point()
```

In the future, you’ll also learn about the pipe, `|>`, which will allow you to create that plot with:

```
penguins |> 
  ggplot(aes(x = flipper_length_mm, y = body_mass_g)) + 
  geom_point()
```

# Visualizing Distributions

How you visualize the distribution of a variable depends on the type of variable: categorical or numerical.

## A Categorical Variable

A variable is *categorical* if it can take only one of a small set of values. To examine the distribution of a categorical variable, you can use a bar chart. The height of the bars displays how many observations occurred with each `x` value.

```
ggplot(penguins, aes(x = species)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in11.png" alt="A bar chart of frequencies of species of penguins: Adelie (approximately 150), Chinstrap (approximately 90), Gentoo (approximately 125)." />
</figure>

In bar plots of categorical variables with nonordered levels, like the previous penguin `species`, it’s often preferable to reorder the bars based on their frequencies. Doing so requires transforming the variable to a factor (how R handles categorical data) and then reordering the levels of that factor.

```
ggplot(penguins, aes(x = fct_infreq(species))) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in12.png" alt="A bar chart of frequencies of species of penguins, where the bars are ordered in decreasing order of their heights (frequencies): Adelie (approximately 150), Gentoo (approximately 125), Chinstrap (approximately 90)." />
</figure>

You will learn more about factors and functions for dealing with factors (such as <a href="https://forcats.tidyverse.org/reference/fct_inorder.html" class="orm:hideurl"><code>fct_infreq()</code></a>) in <a href="ch16.html#chp-factors" data-type="xref">Chapter 16</a>.

## A Numerical Variable

A variable is *numerical* (or quantitative) if it can take on a wide range of numerical values and it is sensible to add, subtract, or take averages with those values. Numerical variables can be continuous or discrete.

One commonly used visualization for distributions of continuous variables is a histogram.

```
ggplot(penguins, aes(x = body_mass_g)) +
  geom_histogram(binwidth = 200)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in13.png" alt="A histogram of body masses of penguins. The distribution is unimodal and right skewed, ranging between approximately 2500 to 6500 grams." />
</figure>

A histogram divides the x-axis into equally spaced bins and then uses the height of a bar to display the number of observations that fall in each bin. In the previous graph, the tallest bar shows that 39 observations have a `body_mass_g` value between 3,500 and 3,700 grams, which are the left and right edges of the bar.

You can set the width of the intervals in a histogram with the `binwidth` argument, which is measured in the units of the `x` variable. You should always explore a variety of `binwidth` values when working with histograms, as different `binwidth` values can reveal different patterns. In the following plots, a `binwidth` of 20 is too narrow, resulting in too many bars, making it difficult to determine the shape of the distribution. Similarly, a `binwidth` of 2,000 is too high, resulting in all data being binned into only three bars and also making it difficult to determine the shape of the distribution. A `binwidth` of 200 provides a sensible balance.

```
ggplot(penguins, aes(x = body_mass_g)) +
  geom_histogram(binwidth = 20)
ggplot(penguins, aes(x = body_mass_g)) +
  geom_histogram(binwidth = 2000)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in14.png" alt="Two histograms of body masses of penguins, one with binwidth of 20 (left) and one with binwidth of 2000 (right). The histogram with binwidth of 20 shows lots of ups and downs in the heights of the bins, creating a jagged outline. The histogram with binwidth of 2000 shows only three bins." />
</figure>

An alternative visualization for distributions of numerical variables is a density plot. A density plot is a smoothed-out version of a histogram and a practical alternative, particularly for continuous data that comes from an underlying smooth distribution. We won’t go into how <a href="https://ggplot2.tidyverse.org/reference/geom_density.html" class="orm:hideurl"><code>geom_density()</code></a> estimates the density (you can read more about that in the function documentation), but let’s explain how the density curve is drawn with an analogy. Imagine a histogram made out of wooden blocks. Then, imagine that you drop a cooked spaghetti string over it. The shape the spaghetti will take draped over blocks can be thought of as the shape of the density curve. It shows fewer details than a histogram but can make it easier to quickly glean the shape of the distribution, particularly with respect to modes and skewness.

```
ggplot(penguins, aes(x = body_mass_g)) +
  geom_density()
#> Warning: Removed 2 rows containing non-finite values (`stat_density()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in15.png" alt="A density plot of body masses of penguins. The distribution is unimodal and right skewed, ranging between approximately 2500 to 6500 grams." />
</figure>

## Exercises

1.  Make a bar plot of `species` of `penguins`, where you assign `species` to the `y` aesthetic. How is this plot different?

2.  How are the following two plots different? Which aesthetic, `color` or `fill`, is more useful for changing the color of bars?

    ```
    ggplot(penguins, aes(x = species)) +
      geom_bar(color = "red")

    ggplot(penguins, aes(x = species)) +
      geom_bar(fill = "red")
    ```

3.  What does the `bins` argument in <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_histogram()</code></a> do?

4.  Make a histogram of the `carat` variable in the `diamonds` dataset that is available when you load the tidyverse package. Experiment with different `binwidth` values. What value reveals the most interesting patterns?

# Visualizing Relationships

To visualize a relationship we need to have at least two variables mapped to aesthetics of a plot. In the following sections you will learn about commonly used plots for visualizing relationships between two or more variables and the geoms used for creating them.

## A Numerical and a Categorical Variable

To visualize the relationship between a numerical and a categorical variable we can use side-by-side box plots. A *boxplot* is a type of visual shorthand for measures of position (percentiles) that describe a distribution. It is also useful for identifying potential outliers. As shown in <a href="#fig-eda-boxplot" data-type="xref">Figure 1-1</a>, each boxplot consists of:

- A box that indicates the range of the middle half of the data, a distance known as the *interquartile range* (IQR), stretching from the 25th percentile of the distribution to the 75th percentile. In the middle of the box is a line that displays the median, i.e., 50th percentile, of the distribution. These three lines give you a sense of the spread of the distribution and whether the distribution is symmetric about the median or skewed to one side.

- Visual points that display observations that fall more than 1.5 times the IQR from either edge of the box. These outlying points are unusual so they are plotted individually.

- A line (or whisker) that extends from each end of the box and goes to the farthest nonoutlier point in the distribution.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0101.png" alt="A diagram depicting how a boxplot is created following the steps outlined above." /></p>
<h6 id="figure-1-1.-diagram-depicting-how-a-boxplot-is-created.">Figure 1-1. Diagram depicting how a boxplot is created.</h6>
</figure>

Let’s take a look at the distribution of body mass by species using <a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>:

```
ggplot(penguins, aes(x = species, y = body_mass_g)) +
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in16.png" alt="Side-by-side box plots of distributions of body masses of Adelie, Chinstrap, and Gentoo penguins. The distribution of Adelie and Chinstrap penguins&#39; body masses appear to be symmetric with medians around 3750 grams. The median body mass of Gentoo penguins is much higher, around 5000 grams, and the distribution of the body masses of these penguins appears to be somewhat right skewed." />
</figure>

Alternatively, we can make density plots with <a href="https://ggplot2.tidyverse.org/reference/geom_density.html" class="orm:hideurl"><code>geom_density()</code></a>:

```
ggplot(penguins, aes(x = body_mass_g, color = species)) +
  geom_density(linewidth = 0.75)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in17.png" alt="A density plot of body masses of penguins by species of penguins. Each species (Adelie, Chinstrap, and Gentoo) is represented with different colored outlines for the density curves." />
</figure>

We’ve also customized the thickness of the lines using the `linewidth` argument to make them stand out a bit more against the background.

Additionally, we can map `species` to both `color` and `fill` aesthetics and use the `alpha` aesthetic to add transparency to the filled density curves. This aesthetic takes values between 0 (completely transparent) and 1 (completely opaque). In the following plot it’s set to 0.5:

```
ggplot(penguins, aes(x = body_mass_g, color = species, fill = species)) +
  geom_density(alpha = 0.5)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in18.png" alt="A density plot of body masses of penguins by species of penguins. Each species (Adelie, Chinstrap, and Gentoo) is represented in different colored outlines for the density curves. The density curves are also filled with the same colors, with some transparency added." />
</figure>

Note the terminology we have used here:

- We *map* variables to aesthetics if we want the visual attribute represented by that aesthetic to vary based on the values of that variable.
- Otherwise, we *set* the value of an aesthetic.

## Two Categorical Variables

We can use stacked bar plots to visualize the relationship between two categorical variables. For example, the following two stacked bar plots both display the relationship between `island` and `species`, or, specifically, visualize the distribution of `species` within each island.

The first plot shows the frequencies of each species of penguins on each island. The plot of frequencies shows that there are equal numbers of Adelies on each island, but we don’t have a good sense of the percentage balance within each island.

```
ggplot(penguins, aes(x = island, fill = species)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in19.png" alt="Bar plots of penguin species by island (Biscoe, Dream, and Torgersen)" />
</figure>

The second plot is a relative frequency plot, created by setting `position = "fill"` in the geom, and is more useful for comparing species distributions across islands since it’s not affected by the unequal numbers of penguins across the islands. Using this plot we can see that Gentoo penguins all live on Biscoe island and make up roughly 75% of the penguins on that island, Chinstrap all live on Dream island and make up roughly 50% of the penguins on that island, and Adelie live on all three islands and make up all of the penguins on Torgersen.

```
ggplot(penguins, aes(x = island, fill = species)) +
  geom_bar(position = "fill")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in20.png" alt="Bar plots of penguin species by island (Biscoe, Dream, and Torgersen) the bars are scaled to the same height, making it a relative frequencies plot" />
</figure>

In creating these bar charts, we map the variable that will be separated into bars to the `x` aesthetic, and the variable that will change the colors inside the bars to the `fill` aesthetic.

## Two Numerical Variables

So far you’ve learned about scatterplots (created with <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>) and smooth curves (created with <a href="https://ggplot2.tidyverse.org/reference/geom_smooth.html" class="orm:hideurl"><code>geom_smooth()</code></a>) for visualizing the relationship between two numerical variables. A scatterplot is probably the most commonly used plot for visualizing the relationship between two numerical variables.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in21.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, linear, relatively strong relationship between these two variables." />
</figure>

## Three or More Variables

As we saw in <a href="#sec-adding-aesthetics-layers" data-type="xref">“Adding Aesthetics and Layers”</a>, we can incorporate more variables into a plot by mapping them to additional aesthetics. For example, in the following scatterplot the colors of points represent species, and the shapes of points represent islands:

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point(aes(color = species, shape = island))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in22.png" alt="A scatterplot of body mass vs. flipper length of penguins. The plot displays a positive, linear, relatively strong relationship between these two variables. The points are colored based on the species of the penguins and the shapes of the points represent islands (round points are Biscoe island, triangles are Dream island, and squared are Torgersen island). The plot is very busy and it&#39;s difficult to distinguish the shapes of the points." />
</figure>

However, adding too many aesthetic mappings to a plot makes it cluttered and difficult to make sense of. Another option, which is particularly useful for categorical variables, is to split your plot into *facets*, subplots that each display one subset of the data.

To facet your plot by a single variable, use <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a>. The first argument of <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a> is a formula,<sup><a href="ch01.html#idm44771330671200" id="idm44771330671200-marker" data-type="noteref">3</a></sup> which you create with `~` followed by a variable name. The variable that you pass to <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a> should be categorical.

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point(aes(color = species, shape = species)) +
  facet_wrap(~island)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_01in23.png" alt="A scatterplot of body mass vs. flipper length of penguins. The shapes and colors of points represent species. Penguins from each island are on a separate facet. Within each facet, the relationship between body mass and flipper length is positive, linear, relatively strong." />
</figure>

You will learn about many other geoms for visualizing distributions of variables and relationships between them in <a href="ch09.html#chp-layers" data-type="xref">Chapter 9</a>.

## Exercises

1.  The `mpg` data frame that is bundled with the ggplot2 package contains 234 observations collected by the US Environmental Protection Agency on 38 car models. Which variables in `mpg` are categorical? Which variables are numerical? (Hint: Type <a href="https://ggplot2.tidyverse.org/reference/mpg.html" class="orm:hideurl"><code>?mpg</code></a> to read the documentation for the dataset.) How can you see this information when you run `mpg`?

2.  Make a scatterplot of `hwy` versus `displ` using the `mpg` data frame. Next, map a third, numerical variable to `color`, then `size`, then both `color` and `size`, and then `shape`. How do these aesthetics behave differently for categorical versus numerical variables?

3.  In the scatterplot of `hwy` versus `displ`, what happens if you map a third variable to `linewidth`?

4.  What happens if you map the same variable to multiple aesthetics?

5.  Make a scatterplot of `bill_depth_mm` versus `bill_length_mm` and color the points by `species`. What does adding coloring by species reveal about the relationship between these two variables? What about faceting by species?

6.  Why does the following yield two separate legends? How would you fix it to combine the two legends?

    ```
    ggplot(
      data = penguins,
      mapping = aes(
        x = bill_length_mm, y = bill_depth_mm, 
        color = species, shape = species
      )
    ) +
      geom_point() +
      labs(color = "Species")
    ```

7.  Create the two following stacked bar plots. Which question can you answer with the first one? Which question can you answer with the second one?

    ```
    ggplot(penguins, aes(x = island, fill = species)) +
      geom_bar(position = "fill")
    ggplot(penguins, aes(x = species, fill = island)) +
      geom_bar(position = "fill")
    ```

# Saving Your Plots

Once you’ve made a plot, you might want to get it out of R by saving it as an image that you can use elsewhere. That’s the job of <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a>, which will save the plot most recently created to disk:

```
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point()
ggsave(filename = "penguin-plot.png")
```

This will save your plot to your working directory, a concept you’ll learn more about in <a href="ch06.html#chp-workflow-scripts" data-type="xref">Chapter 6</a>.

If you don’t specify the `width` and `height`, they will be taken from the dimensions of the current plotting device. For reproducible code, you’ll want to specify them. You can learn more about <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a> in the documentation.

Generally, however, we recommend that you assemble your final reports using Quarto, a reproducible authoring system that allows you to interleave your code and your prose and automatically include your plots in your write-ups. You will learn more about Quarto in <a href="ch28.html#chp-quarto" data-type="xref">Chapter 28</a>.

## Exercises

1.  Run the following lines of code. Which of the two plots is saved as `mpg-plot.png`? Why?

    ```
    ggplot(mpg, aes(x = class)) +
      geom_bar()
    ggplot(mpg, aes(x = cty, y = hwy)) +
      geom_point()
    ggsave("mpg-plot.png")
    ```

2.  What do you need to change in the previous code to save the plot as a PDF instead of a PNG? How could you find out what types of image files would work in <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a>?

# Common Problems

As you start to run R code, you’re likely to run into problems. Don’t worry—it happens to everyone. We have all been writing R code for years, but every day we still write code that doesn’t work on the first try!

Start by carefully comparing the code that you’re running to the code in the book. R is extremely picky, and a misplaced character can make all the difference. Make sure that every `(` is matched with a `)` and every `"` is paired with another `"`. Sometimes you’ll run the code and nothing happens. Check the left side of your console: if it’s a `+`, it means that R doesn’t think you’ve typed a complete expression and it’s waiting for you to finish it. In this case, it’s usually easy to start from scratch again by pressing Escape to abort processing the current command.

One common problem when creating ggplot2 graphics is to put the `+` in the wrong place: it has to come at the end of the line, not the start. In other words, make sure you haven’t accidentally written code like this:

```
ggplot(data = mpg) 
+ geom_point(mapping = aes(x = displ, y = hwy))
```

If you’re still stuck, try the help. You can get help about any R function by running `?function_name` in the console or highlighting the function name and pressing F1 in RStudio. Don’t worry if the help doesn’t seem that helpful; instead, skip down to the examples and look for code that matches what you’re trying to do.

If that doesn’t help, carefully read the error message. Sometimes the answer will be buried there! But when you’re new to R, even if the answer is in the error message, you might not yet know how to understand it. Another great tool is Google: try googling the error message, as it’s likely someone else has had the same problem and has gotten help online.

# Summary

In this chapter, you’ve learned the basics of data visualization with ggplot2. We started with the basic idea that underpins ggplot2: a visualization is a mapping from variables in your data to aesthetic properties such as position, color, size, and shape. You then learned about increasing the complexity and improving the presentation of your plots layer by layer. You also learned about commonly used plots for visualizing the distribution of a single variable, as well as for visualizing relationships between two or more variables, by levering additional aesthetic mappings and/or splitting your plot into small multiples using faceting.

We’ll use visualizations again and again throughout this book, introducing new techniques as we need them, as well as do a deeper dive into creating visualizations with ggplot2 in <a href="ch09.html#chp-layers" data-type="xref">Chapter 9</a> through <a href="ch11.html#chp-communication" data-type="xref">Chapter 11</a>.

Now that you understand the basics of visualization, in the next chapter we’re going to switch gears a little and give you some practical workflow advice. We intersperse workflow advice with data science tools throughout this part of the book because it’ll help you stay organized as you write increasing amounts of R code.

<sup>[1](ch01.html#idm44771333724368-marker)</sup> You can eliminate that message and force conflict resolution to happen on demand by using the conflicted package, which becomes more important as you load more packages. You can learn more about conflicted on the [package website](https://oreil.ly/01bKz).

<sup>[2](ch01.html#idm44771333851472-marker)</sup> Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package version 0.1.0. [*https://oreil.ly/ncwc5*](https://oreil.ly/ncwc5). doi: 10.5281/zenodo.3960218.

<sup>[3](ch01.html#idm44771330671200-marker)</sup> Here “formula” is the name of the thing created by `~`, not a synonym for “equation.”
