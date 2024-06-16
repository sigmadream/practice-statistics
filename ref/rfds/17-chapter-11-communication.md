# Chapter 11. Communication

# Introduction

In <a href="ch10.html#chp-EDA" data-type="xref">Chapter 10</a>, you learned how to use plots as tools for *exploration*. When you make exploratory plots, you know—even before looking—which variables the plot will display. You made each plot for a purpose, could quickly look at it, and could then move on to the next plot. In the course of most analyses, you’ll produce tens or hundreds of plots, most of which are immediately thrown away.

Now that you understand your data, you need to *communicate* your understanding to others. Your audience will likely not share your background knowledge and will not be deeply invested in the data. To help others quickly build up a good mental model of the data, you will need to invest considerable effort in making your plots as self-explanatory as possible. In this chapter, you’ll learn some of the tools that ggplot2 provides to do so.

This chapter focuses on the tools you need to create good graphics. We assume that you know what you want and just need to know how to do it. For that reason, we highly recommend pairing this chapter with a good general visualization book. We particularly like [*The Truthful Art*](https://oreil.ly/QIr_w) by Albert Cairo (New Riders). It doesn’t teach the mechanics of creating visualizations but instead focuses on what you need to think about to create effective graphics.

## Prerequisites

In this chapter, we’ll focus once again on ggplot2. We’ll also use a little dplyr for data manipulation; *scales* to override the default breaks, labels, transformations and palettes; and a few ggplot2 extension packages, including [ggrepel](https://oreil.ly/IVSL4) by Kamil Slowikowski and [patchwork](https://oreil.ly/xWxVV) by Thomas Lin Pedersen. Don’t forget that you’ll need to install those packages with <a href="https://rdrr.io/r/utils/install.packages.html" class="orm:hideurl"><code>install.packages()</code></a> if you don’t already have them.

```
library(tidyverse)
library(scales)
library(ggrepel)
library(patchwork)
```

# Labels

The easiest place to start when turning an exploratory graphic into an expository graphic is with good labels. You add labels with the <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a> function:

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  labs(
    x = "Engine displacement (L)",
    y = "Highway fuel economy (mpg)",
    color = "Car type",
    title = "Fuel efficiency generally decreases with engine size",
    subtitle = "Two seaters (sports cars) are an exception because of their light weight",
    caption = "Data from fueleconomy.gov"
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in01.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, where points are colored according to the car class. A smooth curve following the trajectory of the relationship between highway fuel efficiency versus engine size of cars is overlaid. The x-axis is labelled &quot;Engine displacement (L)&quot; and the y-axis is labelled &quot;Highway fuel economy (mpg)&quot;. The legend is labelled &quot;Car type&quot;. The plot is titled &quot;Fuel efficiency generally decreases with engine size&quot;. The subtitle is &quot;Two seaters (sports cars) are an exception because of their light weight&quot; and the caption is &quot;Data from fueleconomy.gov&quot;." />
</figure>

The purpose of a plot title is to summarize the main finding. Avoid titles that just describe what the plot is, e.g., “A scatterplot of engine displacement vs. fuel economy.”

If you need to add more text, there are two other useful labels: `subtitle` adds additional detail in a smaller font beneath the title, and `caption` adds text at the bottom right of the plot, often used to describe the source of the data. You can also use <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a> to replace the axis and legend titles. It’s usually a good idea to replace short variable names with more detailed descriptions and to include the units.

It’s possible to use mathematical equations instead of text strings. Just switch `""` out for <a href="https://rdrr.io/r/base/substitute.html" class="orm:hideurl"><code>quote()</code></a> and read about the available options in <a href="https://rdrr.io/r/grDevices/plotmath.html" class="orm:hideurl"><code>?plotmath</code></a>:

```
df <- tibble(
  x = 1:10,
  y = cumsum(x^2)
)

ggplot(df, aes(x, y)) +
  geom_point() +
  labs(
    x = quote(x[i]),
    y = quote(sum(x[i] ^ 2, i == 1, n))
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in02.png" alt="Scatterplot with math text on the x and y axis labels. X-axis label says x_i, y-axis label says sum of x_i squared, for i from 1 to n." />
</figure>

## Exercises

1.  Create one plot on the fuel economy data with customized `title`, `subtitle`, `caption`, `x`, `y`, and `color` labels.

2.  Re-create the following plot using the fuel economy data. Note that both the colors and shapes of points vary by type of drivetrain.

    <figure>
    <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in03.png" alt="Scatterplot of highway versus city fuel efficiency. Shapes and colors of points are determined by type of drivetrain." />
    </figure>

3.  Take an exploratory graphic that you’ve created in the last month, and add informative titles to make it easier for others to understand.

# Annotations

In addition to labeling major components of your plot, it’s often useful to label individual observations or groups of observations. The first tool you have at your disposal is <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>. <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a> is similar to <a href="https://ggplot2.tidyverse.org/reference/geom_point.html" class="orm:hideurl"><code>geom_point()</code></a>, but it has an additional aesthetic: `label`. This makes it possible to add textual labels to your plots.

There are two possible sources of labels. First, you might have a tibble that provides labels. In the following plot we pull out the cars with the highest engine size in each drive type and save their information as a new data frame called `label_info`:

```
label_info <- mpg |>
  group_by(drv) |>
  arrange(desc(displ)) |>
  slice_head(n = 1) |>
  mutate(
    drive_type = case_when(
      drv == "f" ~ "front-wheel drive",
      drv == "r" ~ "rear-wheel drive",
      drv == "4" ~ "4-wheel drive"
    )
  ) |>
  select(displ, hwy, drv, drive_type)

label_info
#> # A tibble: 3 × 4
#> # Groups:   drv [3]
#>   displ   hwy drv   drive_type       
#>   <dbl> <int> <chr> <chr>            
#> 1   6.5    17 4     4-wheel drive    
#> 2   5.3    25 f     front-wheel drive
#> 3   7      24 r     rear-wheel drive
```

Then, we use this new data frame to directly label the three groups to replace the legend with labels placed directly on the plot. Using the `fontface` and `size` arguments we can customize the look of the text labels. They’re larger than the rest of the text on the plot and bolded. (`theme(legend.position = "none"`) turns all the legends off—we’ll talk about it more shortly.)

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  geom_text(
    data = label_info, 
    aes(x = displ, y = hwy, label = drive_type),
    fontface = "bold", size = 5, hjust = "right", vjust = "bottom"
  ) +
  theme(legend.position = "none")
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in04.png" alt="Scatterplot of highway mileage versus engine size where points are colored by drive type. Smooth curves for each drive type are overlaid. Text labels identify the curves as front-wheel, rear-wheel, and 4-wheel." />
</figure>

Note the use of `hjust` (horizontal justification) and `vjust` (vertical justification) to control the alignment of the label.

However, the annotated plot we just made is hard to read because the labels overlap with each other and with the points. We can use the <a href="https://rdrr.io/pkg/ggrepel/man/geom_text_repel.html" class="orm:hideurl"><code>geom_label_repel()</code></a> function from the ggrepel package to address both of these issues. This useful package will automatically adjust labels so that they don’t overlap:

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  geom_label_repel(
    data = label_info, 
    aes(x = displ, y = hwy, label = drive_type),
    fontface = "bold", size = 5, nudge_y = 2
  ) +
  theme(legend.position = "none")
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in05.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, where points are colored according to the car class. Some points are labelled with the car&#39;s name. The labels are box with white, transparent background and positioned to not overlap." />
</figure>

You can also use the same idea to highlight certain points on a plot with <a href="https://rdrr.io/pkg/ggrepel/man/geom_text_repel.html" class="orm:hideurl"><code>geom_text_repel()</code></a> from the ggrepel package. Note another handy technique used here: we added a second layer of large, hollow points to further highlight the labeled points.

```
potential_outliers <- mpg |>
  filter(hwy > 40 | (hwy > 20 & displ > 5))
  
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  geom_text_repel(data = potential_outliers, aes(label = model)) +
  geom_point(data = potential_outliers, color = "red") +
  geom_point(
    data = potential_outliers, 
    color = "red", size = 3, shape = "circle open"
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in06.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars. Points where highway mileage is above 40 as well as above 20 with engine size above 5 are red, with a hollow red circle, and labelled with model name of the car." />
</figure>

Remember, in addition to <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_label()</code></a>, you have many other geoms in ggplot2 available to help annotate your plot. A couple ideas:

- Use <a href="https://ggplot2.tidyverse.org/reference/geom_abline.html" class="orm:hideurl"><code>geom_hline()</code></a> and <a href="https://ggplot2.tidyverse.org/reference/geom_abline.html" class="orm:hideurl"><code>geom_vline()</code></a> to add reference lines. We often make them thick (`linewidth = 2`) and white (`color = white`) and draw them underneath the primary data layer. That makes them easy to see, without drawing attention away from the data.

- Use <a href="https://ggplot2.tidyverse.org/reference/geom_tile.html" class="orm:hideurl"><code>geom_rect()</code></a> to draw a rectangle around points of interest. The boundaries of the rectangle are defined by aesthetics `xmin`, `xmax`, `ymin`, and `ymax`. Alternatively, look into the [ggforce package](https://oreil.ly/DZtL1), specifically <a href="https://ggforce.data-imaginist.com/reference/geom_mark_hull.html" class="orm:hideurl"><code>geom_mark_hull()</code></a>, which allows you to annotate subsets of points with hulls.

- Use <a href="https://ggplot2.tidyverse.org/reference/geom_segment.html" class="orm:hideurl"><code>geom_segment()</code></a> with the `arrow` argument to draw attention to a point with an arrow. Use aesthetics `x` and `y` to define the starting location, and use `xend` and `yend` to define the end location.

Another handy function for adding annotations to plots is <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a>. As a rule of thumb, geoms are generally useful for highlighting a subset of the data, while <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a> is useful for adding one or a few annotation elements to a plot.

To demonstrate using <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a>, let’s create some text to add to our plot. The text is a bit long, so we’ll use <a href="https://stringr.tidyverse.org/reference/str_wrap.html" class="orm:hideurl"><code>stringr::str_wrap()</code></a> to automatically add line breaks to it given the number of characters you want per line:

```
trend_text <- "Larger engine sizes tend to\nhave lower fuel economy." |>
  str_wrap(width = 30)
trend_text
#> [1] "Larger engine sizes tend to\nhave lower fuel economy."
```

Then, we add two layers of annotation: one with a label geom and the other with a segment geom. The `x` and `y` aesthetics in both define where the annotation should start, and the `xend` and `yend` aesthetics in the segment annotation define the starting location of the end location of the segment. Note also that the segment is styled as an arrow.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point() +
  annotate(
    geom = "label", x = 3.5, y = 38,
    label = trend_text,
    hjust = "left", color = "red"
  ) +
  annotate(
    geom = "segment",
    x = 3, y = 35, xend = 5, yend = 25, color = "red",
    arrow = arrow(type = "closed")
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in07.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars. A red arrow pointing down follows the trend of the points and the annotation placed next to the arrow reads &quot;Larger engine sizes tend to have lower fuel economy&quot;. The arrow and the annotation text is red." />
</figure>

Annotation is a powerful tool for communicating main takeaways and interesting features of your visualizations. The only limit is your imagination (and your patience with positioning annotations to be aesthetically pleasing)!

## Exercises

1.  Use <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a> with infinite positions to place text at the four corners of the plot.

2.  Use <a href="https://ggplot2.tidyverse.org/reference/annotate.html" class="orm:hideurl"><code>annotate()</code></a> to add a point geom in the middle of your last plot without having to create a tibble. Customize the shape, size, or color of the point.

3.  How do labels with <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a> interact with faceting? How can you add a label to a single facet? How can you put a different label in each facet? (Hint: Think about the dataset that is being passed to <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_text()</code></a>.)

4.  What arguments to <a href="https://ggplot2.tidyverse.org/reference/geom_text.html" class="orm:hideurl"><code>geom_label()</code></a> control the appearance of the background box?

5.  What are the four arguments to <a href="https://rdrr.io/r/grid/arrow.html" class="orm:hideurl"><code>arrow()</code></a>? How do they work? Create a series of plots that demonstrate the most important options.

# Scales

The third way you can make your plot better for communication is to adjust the scales. Scales control how the aesthetic mappings manifest visually.

## Default Scales

Normally, ggplot2 automatically adds scales for you. For example, when you type:

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class))
```

ggplot2 automatically adds default scales behind the scenes:

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  scale_x_continuous() +
  scale_y_continuous() +
  scale_color_discrete()
```

Note the naming scheme for scales: `scale_` followed by the name of the aesthetic, then `_`, and then the name of the scale. The default scales are named according to the type of variable they align with: continuous, discrete, date-time, or date. <a href="https://ggplot2.tidyverse.org/reference/scale_continuous.html" class="orm:hideurl"><code>scale_x_continuous()</code></a> puts the numeric values from `displ` on a continuous number line on the x-axis, <a href="https://ggplot2.tidyverse.org/reference/scale_colour_discrete.html" class="orm:hideurl"><code>scale_color_discrete()</code></a> chooses colors for each `class` of car, etc. There are lots of nondefault scales, which you’ll learn about next.

The default scales have been carefully chosen to do a good job for a wide range of inputs. Nevertheless, you might want to override the defaults for two reasons:

- You might want to tweak some of the parameters of the default scale. This allows you to do things like change the breaks on the axes, or the key labels on the legend.

- You might want to replace the scale altogether and use a completely different algorithm. Often you can do better than the default because you know more about the data.

## Axis Ticks and Legend Keys

Collectively axes and legends are called *guides*. Axes are used for `x` and `y` aesthetics; legends are used for everything else.

There are two primary arguments that affect the appearance of the ticks on the axes and the keys on the legend: `breaks` and `labels`. The `breaks` argument controls the position of the ticks or the values associated with the keys. The `labels` argument controls the text label associated with each tick/key. The most common use of `breaks` is to override the default choice:

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  scale_y_continuous(breaks = seq(15, 40, by = 5)) 
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in08.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, colored by drive. The y-axis has breaks starting at 15 and ending at 40, increasing by 5." />
</figure>

You can use `labels` in the same way (a character vector the same length as `breaks`), but you can also set it to `NULL` to suppress the labels altogether. This can be useful for maps or for publishing plots where you can’t share the absolute numbers. You can also use `breaks` and `labels` to control the appearance of legends. For discrete scales for categorical variables, `labels` can be a named list of the existing levels names and the desired labels for them.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  scale_x_continuous(labels = NULL) +
  scale_y_continuous(labels = NULL) +
  scale_color_discrete(labels = c("4" = "4-wheel", "f" = "front", "r" = "rear"))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in09.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, colored by drive. The x and y-axes do not have any labels at the axis ticks. The legend has custom labels: 4-wheel, front, rear." />
</figure>

The `labels` argument coupled with labeling functions from the scales package is also useful for formatting numbers as currency, percent, etc. The plot on the left shows default labeling with <a href="https://scales.r-lib.org/reference/label_dollar.html" class="orm:hideurl"><code>label_dollar()</code></a>, which adds a dollar sign as well as a thousand separator comma. The plot on the right adds further customization by dividing dollar values by 1,000 and adding a suffix “K” (for “thousands”) as well as adding custom breaks. Note that `breaks` is in the original scale of the data.

```
# Left
ggplot(diamonds, aes(x = price, y = cut)) +
  geom_boxplot(alpha = 0.05) +
  scale_x_continuous(labels = label_dollar())

# Right
ggplot(diamonds, aes(x = price, y = cut)) +
  geom_boxplot(alpha = 0.05) +
  scale_x_continuous(
    labels = label_dollar(scale = 1/1000, suffix = "K"), 
    breaks = seq(1000, 19000, by = 6000)
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in10.png" alt="Two side-by-side box plots of price versus cut of diamonds. The outliers are transparent. On both plots the x-axis labels are formatted as dollars. The x-axis labels on the plot start at $0 and go to $15,000, increasing by $5,000. The x-axis labels on the right plot start at $1K and go to $19K, increasing by $6K." />
</figure>

Another handy label function is <a href="https://scales.r-lib.org/reference/label_percent.html" class="orm:hideurl"><code>label_percent()</code></a>:

```
ggplot(diamonds, aes(x = cut, fill = clarity)) +
  geom_bar(position = "fill") +
  scale_y_continuous(name = "Percentage", labels = label_percent())
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in11.png" alt="Segmented bar plots of cut, filled with levels of clarity. The y-axis labels start at 0% and go to 100%, increasing by 25%. The y-axis label name is &quot;Percentage&quot;." />
</figure>

Another use of `breaks` is when you have relatively few data points and want to highlight exactly where the observations occur. For example, take this plot that shows when each US president started and ended their term:

```
presidential |>
  mutate(id = 33 + row_number()) |>
  ggplot(aes(x = start, y = id)) +
  geom_point() +
  geom_segment(aes(xend = end, yend = id)) +
  scale_x_date(name = NULL, breaks = presidential$start, date_labels = "'%y")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in12.png" alt="Line plot of id number of presidents versus the year they started their presidency. Start year is marked with a point and a segment that starts there and ends at the end of the presidency. The x-axis labels are formatted as two digit years starting with an apostrophe, e.g., &#39;53." />
</figure>

Note that for the `breaks` argument we pulled out the `start` variable as a vector with `presidential$start` because we can’t do an aesthetic mapping for this argument. Also note that the specification of breaks and labels for date and date-time scales is a little different:

- `date_labels` takes a format specification, in the same form as <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>parse_datetime()</code></a>.

- `date_breaks` (not shown here) takes a string like “2 days” or “1 month.”

## Legend Layout

You will most often use `breaks` and `labels` to tweak the axes. While they both also work for legends, there are a few other techniques you are more likely to use.

To control the overall position of the legend, you need to use a <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a> setting. We’ll come back to themes at the end of the chapter, but in brief, they control the nondata parts of the plot. The theme setting `legend.position` controls where the legend is drawn:

```
base <- ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class))

base + theme(legend.position = "right") # the default
base + theme(legend.position = "left")
base + 
  theme(legend.position = "top") +
  guides(col = guide_legend(nrow = 3))
base + 
  theme(legend.position = "bottom") +
  guides(col = guide_legend(nrow = 3))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in13.png" alt="Four scatterplots of highway fuel efficiency versus engine size of cars where points are colored based on class of car. Clockwise, the legend is placed on the right, left, top, and bottom of the plot." />
</figure>

If your plot is short and wide, place the legend at the top or bottom, and if it’s tall and narrow, place the legend at the left or right. You can also use `legend.position = "none"` to suppress the display of the legend altogether.

To control the display of individual legends, use <a href="https://ggplot2.tidyverse.org/reference/guides.html" class="orm:hideurl"><code>guides()</code></a> along with <a href="https://ggplot2.tidyverse.org/reference/guide_legend.html" class="orm:hideurl"><code>guide_legend()</code></a> or <a href="https://ggplot2.tidyverse.org/reference/guide_colourbar.html" class="orm:hideurl"><code>guide_colorbar()</code></a>. The following example shows two important settings: controlling the number of rows the legend uses with `nrow`, and overriding one of the aesthetics to make the points bigger. This is particularly useful if you have used a low `alpha` to display many points on a plot.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2, override.aes = list(size = 4)))
#> `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in14.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars where points are colored based on class of car. Overlaid on the plot is a smooth curve. The legend is in the bottom and classes are listed horizontally in two rows. The points in the legend are larger than the points in the plot." />
</figure>

Note that the name of the argument in <a href="https://ggplot2.tidyverse.org/reference/guides.html" class="orm:hideurl"><code>guides()</code></a> matches the name of the aesthetic, just like in <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>.

## Replacing a Scale

Instead of just tweaking the details a little, you can instead replace the scale altogether. There are two types of scales you’re most likely to want to switch out: continuous position scales and color scales. Fortunately, the same principles apply to all the other aesthetics, so once you’ve mastered position and color, you’ll be able to quickly pick up other scale replacements.

It’s useful to plot transformations of your variable. For example, it’s easier to see the precise relationship between `carat` and `price` if we log transform them:

```
# Left
ggplot(diamonds, aes(x = carat, y = price)) +
  geom_bin2d()

# Right
ggplot(diamonds, aes(x = log10(carat), y = log10(price))) +
  geom_bin2d()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in15.png" alt="Two plots of price versus carat of diamonds. Data binned and the color of the rectangles representing each bin based on the number of points that fall into that bin. In the plot on the right, price and carat values are logged and the axis labels shows the logged values." />
</figure>

However, the disadvantage of this transformation is that the axes are now labeled with the transformed values, making it hard to interpret the plot. Instead of doing the transformation in the aesthetic mapping, we can instead do it with the scale. This is visually identical, except the axes are labeled on the original data scale.

```
ggplot(diamonds, aes(x = carat, y = price)) +
  geom_bin2d() + 
  scale_x_log10() + 
  scale_y_log10()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in16.png" alt="Plot of price versus carat of diamonds. Data binned and the color of the rectangles representing each bin based on the number of points that fall into that bin. The axis labels are on the original data scale." />
</figure>

Another scale that is frequently customized is color. The default categorical scale picks colors that are evenly spaced around the color wheel. Useful alternatives are the ColorBrewer scales, which have been hand tuned to work better for people with common types of color blindness. The following two plots look similar, but there is enough difference in the shades of red and green that the dots on the right can be distinguished even by people with red-green color blindness.<sup><a href="ch11.html#idm44771304642976" id="idm44771304642976-marker" data-type="noteref">1</a></sup>

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv))

ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  scale_color_brewer(palette = "Set1")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in17.png" alt="Two scatterplots of highway mileage versus engine size where points are colored by drive type. The plot on the left uses the default ggplot2 color palette and the plot on the right uses a different color palette." />
</figure>

Don’t forget simpler techniques for improving accessibility. If there are just a few colors, you can add a redundant shape mapping. This will also help ensure your plot is interpretable in black and white.

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv, shape = drv)) +
  scale_color_brewer(palette = "Set1")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in18.png" alt="Two scatterplots of highway mileage versus engine size where both color and shape of points are based on drive type. The color palette is not the default ggplot2 palette." />
</figure>

The ColorBrewer scales are [documented online](https://oreil.ly/LNHAy) and made available in R via the RColorBrewer package, by Erich Neuwirth. <a href="#fig-brewer" data-type="xref">Figure 11-1</a> shows the complete list of all palettes. The sequential (top) and diverging (bottom) palettes are particularly useful if your categorical values are ordered or have a “middle.” This often arises if you’ve used <a href="https://rdrr.io/r/base/cut.html" class="orm:hideurl"><code>cut()</code></a> to make a continuous variable into a categorical variable.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1101.png" alt="All ColorBrewer scales. One group goes from light to dark colors. Another group is a set of non ordinal colors. And the last group has diverging scales (from dark to light to dark again). Within each set there are a number of palettes." />
<h6 id="figure-11-1.-all-colorbrewer-scales.">Figure 11-1. All ColorBrewer scales.</h6>
</figure>

When you have a predefined mapping between values and colors, use <a href="https://ggplot2.tidyverse.org/reference/scale_manual.html" class="orm:hideurl"><code>scale_color_manual()</code></a>. For example, if we map presidential party to color, we want to use the standard mapping of red for Republicans and blue for Democrats. One approach for assigning these colors is using hex color codes:

```
presidential |>
  mutate(id = 33 + row_number()) |>
  ggplot(aes(x = start, y = id, color = party)) +
  geom_point() +
  geom_segment(aes(xend = end, yend = id)) +
  scale_color_manual(values = c(Republican = "#E81B23", Democratic = "#00AEF3"))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in19.png" alt="Line plot of id number of presidents versus the year they started their presidency. Start year is marked with a point and a segment that starts there and ends at the end of the presidency. Democratic presidents are represented in blue and Republicans in red." />
</figure>

For continuous color, you can use the built-in <a href="https://ggplot2.tidyverse.org/reference/scale_gradient.html" class="orm:hideurl"><code>scale_color_gradient()</code></a> or <a href="https://ggplot2.tidyverse.org/reference/scale_gradient.html" class="orm:hideurl"><code>scale_fill_gradient()</code></a>. If you have a diverging scale, you can use <a href="https://ggplot2.tidyverse.org/reference/scale_gradient.html" class="orm:hideurl"><code>scale_color_gradient2()</code></a>. That allows you to give, for example, positive and negative values different colors. That’s sometimes also useful if you want to distinguish points above or below the mean.

Another option is to use the viridis color scales. The designers, Nathaniel Smith and Stéfan van der Walt, carefully tailored continuous color schemes that are perceptible to people with various forms of color blindness as well as perceptually uniform in both color and black and white. These scales are available as continuous (`c`), discrete (`d`), and binned (`b`) palettes in ggplot2.

```
df <- tibble(
  x = rnorm(10000),
  y = rnorm(10000)
)

ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed() +
  labs(title = "Default, continuous", x = NULL, y = NULL)

ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed() +
  scale_fill_viridis_c() +
  labs(title = "Viridis, continuous", x = NULL, y = NULL)

ggplot(df, aes(x, y)) +
  geom_hex() +
  coord_fixed() +
  scale_fill_viridis_b() +
  labs(title = "Viridis, binned", x = NULL, y = NULL)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in20.png" alt="Three hex plots where the color of the hexes show the number of observations that fall into that hex bin. The first plot uses the default, continuous ggplot2 scale. The second plot uses the viridis, continuous scale, and the third plot uses the viridis, binned scale." />
</figure>

Note that all color scales come in two varieties: `scale_color_*()` and `scale_fill_*()` for the `color` and `fill` aesthetics, respectively (the color scales are available in both UK and US spellings).

## Zooming

There are three ways to control the plot limits:

- Adjusting what data are plotted
- Setting the limits in each scale
- Setting `xlim` and `ylim` in <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>

We’ll demonstrate these options in a series of plots. The plot on the left shows the relationship between engine size and fuel efficiency, colored by type of drivetrain. The plot on the right shows the same variables but subsets the data plotted. Subsetting the data has affected the x and y scales as well as the smooth curve.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth()

# Right
mpg |>
  filter(displ >= 5 & displ <= 6 & hwy >= 10 & hwy <= 25) |>
  ggplot(aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in21.png" alt="On the left, scatterplot of highway mileage vs. displacement, with displacement. The smooth curve overlaid shows a decreasing, and then increasing, trend like a hockey stick. On the right, same variables are plotted with displacement ranging only from 5 to 6 and highway mileage ranging only from 10 to 25. The smooth curve overlaid shows a trend that&#39;s slightly increasing first and then decreasing." />
</figure>

Let’s compare these to the two following plots where the plot on the left sets the `limits` on individual scales and the plot on the right sets them in <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>. We can see that reducing the limits is equivalent to subsetting the data. Therefore, to zoom in on a region of the plot, it’s generally best to use <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>.

```
# Left
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth() +
  scale_x_continuous(limits = c(5, 6)) +
  scale_y_continuous(limits = c(10, 25))

# Right
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = drv)) +
  geom_smooth() +
  coord_cartesian(xlim = c(5, 6), ylim = c(10, 25))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in22.png" alt="On the left, scatterplot of highway mileage vs. displacement, with displacement ranging from 5 to 6 and highway mileage ranging from 10 to 25. The smooth curve overlaid shows a trend that&#39;s slightly increasing first and then decreasing. On the right, same variables are plotted with the same limits; however, the smooth curve overlaid shows a relatively flat trend with a slight increase at the end." />
</figure>

On the other hand, setting the `limits` on individual scales is generally more useful if you want to *expand* the limits, e.g., to match scales across different plots. For example, if we extract two classes of cars and plot them separately, it’s difficult to compare the plots because all three scales (the x-axis, the y-axis, and the color aesthetic) have different ranges.

```
suv <- mpg |> filter(class == "suv")
compact <- mpg |> filter(class == "compact")

# Left
ggplot(suv, aes(x = displ, y = hwy, color = drv)) +
  geom_point()

# Right
ggplot(compact, aes(x = displ, y = hwy, color = drv)) +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in23.png" alt="On the left, a scatterplot of highway mileage vs. displacement of SUVs. On the right, a scatterplot of the same variables for compact cars. Points are colored by drive type for both plots. Among SUVs, more of the cars are 4-wheel drive and the others are rear-wheel drive, while among compact cars more of the cars are front-wheel drive and the others are 4-wheel drive. SUV plot shows a clear negative relationship between highway mileage and displacement, while in the compact cars plot, the relationship is much flatter." />
</figure>

One way to overcome this problem is to share scales across multiple plots, training the scales with the `limits` of the full data.

```
x_scale <- scale_x_continuous(limits = range(mpg$displ))
y_scale <- scale_y_continuous(limits = range(mpg$hwy))
col_scale <- scale_color_discrete(limits = unique(mpg$drv))

# Left
ggplot(suv, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  x_scale +
  y_scale +
  col_scale

# Right
ggplot(compact, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  x_scale +
  y_scale +
  col_scale
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in24.png" alt="On the left, a scatterplot of highway mileage vs. displacement of SUVs. On the right, a scatterplot of the same variables for compact cars. Points are colored by drive type for both plots. Both plots are plotted on the same scale for highway mileage, displacement, and drive type, resulting in the legend showing all three types (front, rear, and 4-wheel drive) for both plots even though there are no front-wheel drive SUVs and no rear-wheel drive compact cars. Since the x and y scales are the same, and go well beyond minimum or maximum highway mileage and displacement, the points do not take up the entire plotting area." />
</figure>

In this particular case, you could have simply used faceting, but this technique is useful more generally, if, for instance, you want to spread plots over multiple pages of a report.

## Exercises

1.  Why doesn’t the following code override the default scale?

    ```
    df <- tibble(
      x = rnorm(10000),
      y = rnorm(10000)
    )

    ggplot(df, aes(x, y)) +
      geom_hex() +
      scale_color_gradient(low = "white", high = "red") +
      coord_fixed()
    ```

2.  What is the first argument to every scale? How does it compare to <a href="https://ggplot2.tidyverse.org/reference/labs.html" class="orm:hideurl"><code>labs()</code></a>?

3.  Change the display of the presidential terms by:

    1.  Combining the two variants that customize colors and x-axis breaks
    2.  Improving the display of the y-axis
    3.  Labeling each term with the name of the president
    4.  Adding informative plot labels
    5.  Placing breaks every four years (this is trickier than it seems!)

4.  First, create the following plot. Then, modify the code using `override.aes` to make the legend easier to see.

    ```
    ggplot(diamonds, aes(x = carat, y = price)) +
      geom_point(aes(color = cut), alpha = 1/20)
    ```

# Themes

Finally, you can customize the nondata elements of your plot with a theme:

```
ggplot(mpg, aes(x = displ, y = hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  theme_bw()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in25.png" alt="Scatterplot of highway mileage vs. displacement of cars, colored by class of car. The plot background is white, with gray grid lines." />
</figure>

ggplot2 includes the eight themes shown in <a href="#fig-themes" data-type="xref">Figure 11-2</a>, with <a href="https://ggplot2.tidyverse.org/reference/ggtheme.html" class="orm:hideurl"><code>theme_gray()</code></a> as the default.<sup><a href="ch11.html#idm44771303447760" id="idm44771303447760-marker" data-type="noteref">2</a></sup> Many more are included in add-on packages like [ggthemes](https://oreil.ly/F1nga), by Jeffrey Arnold. You can also create your own themes, if you are trying to match a particular corporate or journal style.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1102.png" alt="Eight barplots created with ggplot2, each with one of the eight built-in themes: theme_bw() - White background with grid lines, theme_light() - Light axes and grid lines, theme_classic() - Classic theme, axes but no grid lines, theme_linedraw() - Only black lines, theme_dark() - Dark background for contrast, theme_minimal() - Minimal theme, no background, theme_gray() - Gray background (default theme), theme_void() - Empty theme, only geoms are visible." />
<h6 id="figure-11-2.-the-eight-themes-built-in-to-ggplot2.">Figure 11-2. The eight themes built in to ggplot2.</h6>
</figure>

It’s also possible to control individual components of each theme, such as the size and color of the font used for the y-axis. We’ve already seen that `legend.position` controls where the legend is drawn. There are many other aspects of the legend that can be customized with <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a>. For example, in the following plot we change the direction of the legend as well as put a black border around it. Note that customization of the legend box and plot title elements of the theme are done with `element_*()` functions. These functions specify the styling of nondata components; e.g., the title text is bolded in the `face` argument of <a href="https://ggplot2.tidyverse.org/reference/element.html" class="orm:hideurl"><code>element_text()</code></a>, and the legend border color is defined in the `color` argument of <a href="https://ggplot2.tidyverse.org/reference/element.html" class="orm:hideurl"><code>element_rect()</code></a>. The theme elements that control the position of the title and the caption are `plot.title.position` and `plot.caption.position`, respectively. In the following plot these are set to `"plot"` to indicate these elements are aligned to the entire plot area, instead of the plot panel (the default). A few other helpful <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a> components are used to change the placement for formatting the title and caption text.

```
ggplot(mpg, aes(x = displ, y = hwy, color = drv)) +
  geom_point() +
  labs(
    title = "Larger engine sizes tend to have lower fuel economy",
    caption = "Source: https://fueleconomy.gov."
  ) +
  theme(
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal",
    legend.box.background = element_rect(color = "black"),
    plot.title = element_text(face = "bold"),
    plot.title.position = "plot",
    plot.caption.position = "plot",
    plot.caption = element_text(hjust = 0)
  )
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in26.png" alt="Scatterplot of highway fuel efficiency versus engine size of cars, colored by drive. The plot is titled &#39;Larger engine sizes tend to have lower fuel economy&#39; with the caption pointing to the source of the data, fueleconomy.gov. The caption and title are left justified, the legend is inside of the plot with a black border." />
</figure>

For an overview of all <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>theme()</code></a> components, see the help with <a href="https://ggplot2.tidyverse.org/reference/theme.html" class="orm:hideurl"><code>?theme</code></a>. The [ggplot2 book](https://oreil.ly/T4Jxn) is also a great place to go for the full details on theming.

## Exercises

1.  Pick a theme offered by the ggthemes package and apply it to the last plot you made.
2.  Make the axis labels of your plot blue and bold.

# Layout

So far we talked about how to create and modify a single plot. What if you have multiple plots you want to lay out in a certain way? The patchwork package allows you to combine separate plots into the same graphic. We loaded this package earlier in the chapter.

To place two plots next to each other, you can simply add them to each other. Note that you first need to create the plots and save them as objects (in the following example they’re called `p1` and `p2`). Then, you place them next to each other with `+`.

```
p1 <- ggplot(mpg, aes(x = displ, y = hwy)) + 
  geom_point() + 
  labs(title = "Plot 1")
p2 <- ggplot(mpg, aes(x = drv, y = hwy)) + 
  geom_boxplot() + 
  labs(title = "Plot 2")
p1 + p2
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in27.png" alt="Two plots (a scatterplot of highway mileage versus engine size and a side-by-side boxplots of highway mileage versus drivetrain) placed next to each other." />
</figure>

It’s important to note that in the previous code chunk we did not use a new function from the patchwork package. Instead, the package added a new functionality to the `+` operator.

You can also create complex plot layouts with patchwork. In the following, `|` places the `p1` and `p3` next to each other, and `/` moves `p2` to the next line:

```
p3 <- ggplot(mpg, aes(x = cty, y = hwy)) + 
  geom_point() + 
  labs(title = "Plot 3")
(p1 | p3) / p2
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in28.png" alt="Three plots laid out such that first and third plot are next to each other and the second plot stretched beneath them. The first plot is a scatterplot of highway mileage versus engine size, third plot is a scatterplot of highway mileage versus city mileage, and the third plot is side-by-side boxplots of highway mileage versus drivetrain) placed next to each other." />
</figure>

Additionally, patchwork allows you to collect legends from multiple plots into one common legend, customize the placement of the legend as well as dimensions of the plots, and add a common title, subtitle, caption, etc., to your plots. Here we created five plots. We turned off the legends on the box plots and the scatterplot and collected the legends for the density plots at the top of the plot with `& theme(legend.position = "top")`. Note the use of the `&` operator here instead of the usual `+`. This is because we’re modifying the theme for the patchwork plot as opposed to the individual ggplots. The legend is placed on top, inside the <a href="https://patchwork.data-imaginist.com/reference/guide_area.html" class="orm:hideurl"><code>guide_area()</code></a>. Finally, we have also customized the heights of the various components of our patchwork—the guide has a height of 1, the box plots 3, the density plots 2, and the faceted scatterplot 4. Patchwork divides up the area you have allotted for your plot using this scale and places the components accordingly.

```
p1 <- ggplot(mpg, aes(x = drv, y = cty, color = drv)) + 
  geom_boxplot(show.legend = FALSE) + 
  labs(title = "Plot 1")

p2 <- ggplot(mpg, aes(x = drv, y = hwy, color = drv)) + 
  geom_boxplot(show.legend = FALSE) + 
  labs(title = "Plot 2")

p3 <- ggplot(mpg, aes(x = cty, color = drv, fill = drv)) + 
  geom_density(alpha = 0.5) + 
  labs(title = "Plot 3")

p4 <- ggplot(mpg, aes(x = hwy, color = drv, fill = drv)) + 
  geom_density(alpha = 0.5) + 
  labs(title = "Plot 4")

p5 <- ggplot(mpg, aes(x = cty, y = hwy, color = drv)) + 
  geom_point(show.legend = FALSE) + 
  facet_wrap(~drv) +
  labs(title = "Plot 5")

(guide_area() / (p1 + p2) / (p3 + p4) / p5) +
  plot_annotation(
    title = "City and highway mileage for cars with different drivetrains",
    caption = "Source: https://fueleconomy.gov."
  ) +
  plot_layout(
    guides = "collect",
    heights = c(1, 3, 2, 4)
    ) &
  theme(legend.position = "top")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in29.png" alt="Five plots laid out such that first two plots are next to each other. Plots three and four are underneath them. And the fifth plot stretches under them. The patchworked plot is titled &quot;City and highway mileage for cars with different drivetrains&quot; and captioned &quot;Source: https://fueleconomy.gov&quot;. The first two plots are side-by-side box plots. Plots 3 and 4 are density plots. And the fifth plot is a faceted scatterplot. Each of these plots show geoms colored by drivetrain, but the patchworked plot has only one legend that applies to all of them, above the plots and beneath the title." />
</figure>

If you’d like to learn more about combining and laying out multiple plots with patchwork, we recommend looking through the guides on the [package website](https://oreil.ly/xWxVV).

## Exercises

1.  What happens if you omit the parentheses in the following plot layout. Can you explain why this happens?

    ```
    p1 <- ggplot(mpg, aes(x = displ, y = hwy)) + 
      geom_point() + 
      labs(title = "Plot 1")
    p2 <- ggplot(mpg, aes(x = drv, y = hwy)) + 
      geom_boxplot() + 
      labs(title = "Plot 2")
    p3 <- ggplot(mpg, aes(x = cty, y = hwy)) + 
      geom_point() + 
      labs(title = "Plot 3")

    (p1 | p2) / p3
    ```

Using the three plots from the previous exercise, re-create the following patchwork:

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_11in30.png" alt="Three plots: Plot 1 is a scatterplot of highway mileage versus engine size. Plot 2 is side-by-side box plots of highway mileage versus drivetrain. Plot 3 is side-by-side box plots of city mileage versus drivetrain. Plots 1 is on the first row. Plots 2 and 3 are on the next row, each span half the width of Plot 1. Plot 1 is labelled &quot;Fig. A&quot;, Plot 2 is labelled &quot;Fig. B&quot;, and Plot 3 is labelled &quot;Fig. C&quot;." />
</figure>

# Summary

In this chapter you learned about adding plot labels such as title, subtitle, and caption as well as modifying default axis labels, using annotation to add informational text to your plot or to highlight specific data points, customizing the axis scales, and changing the theme of your plot. You also learned about combining multiple plots in a single graph using both simple and complex plot layouts.

While you’ve so far learned about how to make many different types of plots and how to customize them using a variety of techniques, we’ve barely scratched the surface of what you can create with ggplot2. If you want to get a comprehensive understanding of ggplot2, we recommend reading the book [*ggplot2: Elegant Graphics for Data Analysis*](https://oreil.ly/T4Jxn) (Springer). Other useful resources are the [*R Graphics Cookbook*](https://oreil.ly/CK_sd) by Winston Chang (O’Reilly) and [*Fundamentals of Data Visualization*](https://oreil.ly/uJRYK) by Claus Wilke (O’Reilly).

<sup>[1](ch11.html#idm44771304642976-marker)</sup> You can use a tool like [SimDaltonism](https://oreil.ly/i11yd) to simulate color blindness to test these images.

<sup>[2](ch11.html#idm44771303447760-marker)</sup> Many people wonder why the default theme has a gray background. This was a deliberate choice because it puts the data forward while still making the grid lines visible. The white grid lines are visible (which is important because they significantly aid position judgments), but they have little visual impact, and we can easily tune them out. The gray background gives the plot a similar typographic color to the text, ensuring that the graphics fit in with the flow of a document without jumping out with a bright white background. Finally, the gray background creates a continuous field of color, which ensures that the plot is perceived as a single visual entity.
