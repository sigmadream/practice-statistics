# Chapter 4. The Ames Housing Data

In this chapter, we’ll introduce the Ames housing data set (De Cock 2011), which we will use in modeling examples throughout this book. Exploratory data analysis, like what we walk through in this chapter, is an important first step in building a reliable model. The data set contains information on 2,930 properties in Ames, Iowa, including columns related to:

- House characteristics (bedrooms, garage, fireplace, pool, porch, etc.)

- Location (neighborhood)

- Lot information (zoning, shape, size, etc.)

- Ratings of condition and quality

- Sale price

The raw housing data are provided in De Cock (2011), but in our analyses in this book, we use a transformed version available in the modeldata package. This version has several [changes and improvements to the data](https://oreil.ly/OSIQ0). For example, the longitude and latitude values have been determined for each property. Also, some columns were modified to be more analysis ready. For example:

- In the raw data, if a house did not have a particular feature, it was implicitly encoded as missing. For example, 2,732 properties did not have an alleyway. Instead of leaving these as missing, they were relabeled in the transformed version to indicate that no alley was available.

- The categorical predictors were converted to R’s factor data type. While both the tidyverse and base R have moved away from importing data as factors by default, this data type is a better approach for storing qualitative data for modeling than simple strings.\

- We removed a set of quality descriptors for each house since they are more like outcomes than predictors.

To load the data:

```
library(modeldata) # This is also loaded by the tidymodels package
data(ames)

# or, in one line:
data(ames, package = "modeldata")

dim(ames)
#> [1] 2930   74
```

<a href="#ames-map" data-type="xref">Figure 4-1</a> shows the locations of the properties in Ames. The locations will be revisited in the next section.

<figure class="width-80">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0401.png" alt="tmwr 0401" />
<h6 id="figure-4-1.-property-locations-in-ames-iowa.">Figure 4-1. Property locations in Ames, Iowa.</h6>
</figure>

The void of data points in the center of Ames corresponds to Iowa State University.

###### Note

Our modeling goal is to predict the sale price of a house based on other information we have, such as its characteristics and location.

# Exploring Features of Homes in Ames

Let’s start our exploratory data analysis by focusing on the outcome we want to predict: the last sale price of the house (in USD). We can create a histogram to see the distribution of sale prices in <a href="#ames-sale-price-hist" data-type="xref">Figure 4-2</a>:

```
library(tidymodels)
tidymodels_prefer()

ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram(bins = 50, col= "white")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0402.png" alt="tmwr 0402" />
<h6 id="figure-4-2.-sale-prices-of-houses-in-ames-iowa.">Figure 4-2. Sale prices of houses in Ames, Iowa.</h6>
</figure>

This plot shows us that the data are right-skewed; there are more inexpensive houses than expensive ones. The median sale price was \$160,000, and the most expensive house was \$755,000. When modeling this outcome, a strong argument can be made that the price should be log-transformed. The advantages of this type of transformation are that no houses would be predicted with negative sale prices and that errors in predicting expensive houses will not have an undue influence on the model. Also, from a statistical perspective, a logarithmic transform may also stabilize the variance in a way that makes inference more legitimate. We now can use similar steps to visualize the transformed data, shown in <a href="#ames-log-sale-price-hist" data-type="xref">Figure 4-3</a>:

```
ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram(bins = 50, col= "white") +
  scale_x_log10()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0403.png" alt="tmwr 0403" />
<h6 id="figure-4-3.-sale-prices-of-houses-in-ames-iowa-after-a-log-base-10-transformation.">Figure 4-3. Sale prices of houses in Ames, Iowa after a log (base 10) transformation.</h6>
</figure>

While not perfect, this will likely result in better models than using the untransformed data, for the reasons just outlined.

###### Warning

The disadvantages of transforming the outcome mostly relate to interpretation of model results.

The units of the model coefficients might be more difficult to interpret, as will measures of performance. For example, the *root mean squared error* (RMSE) is a common performance metric used in regression models. It uses the difference between the observed and predicted values in its calculations. If the sale price is on the log scale, these differences (i.e., the residuals) are also on the log scale. It can be difficult to understand the quality of a model whose RMSE is 0.15 on such a log scale.

Despite these drawbacks, the models used in this book use the log transformation for this outcome. *From this point on*, the outcome column is prelogged in the `ames` data frame:

```
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
```

Another important aspect of these data for our modeling are their geographic locations. This spatial information is contained in the data in two ways: a qualitative `Neighborhood` label as well as quantitative longitude and latitude data. To visualize the spatial information, <a href="#ames-chull" data-type="xref">Figure 4-4</a> duplicates the data from <a href="#ames-map" data-type="xref">Figure 4-1</a> with convex hulls around the data from each neighborhood.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0404.png" alt="tmwr 0404" />
<h6 id="figure-4-4.-neighborhoods-in-ames-represented-using-a-convex-hull.">Figure 4-4. Neighborhoods in Ames represented using a convex hull.</h6>
</figure>

We can see a few noticeable patterns. First, there is a void of data points in the center of Ames. This corresponds to the campus of Iowa State University where there are no residential houses. Second, while there are a number of adjacent neighborhoods, others are geographically isolated. For example, as <a href="#ames-timberland" data-type="xref">Figure 4-5</a> shows, Timberland is located apart from almost all other neighborhoods.

<a href="#ames-mitchell" data-type="xref">Figure 4-6</a> visualizes how the Meadow Village neighborhood in southwest Ames is like an island of properties inside the sea of properties that make up the Mitchell neighborhood.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0405.png" alt="tmwr 0405" />
<h6 id="figure-4-5.-locations-of-homes-in-timberland.">Figure 4-5. Locations of homes in Timberland.</h6>
</figure>

<figure class="width-80">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0406.png" alt="tmwr 0406" />
<h6 id="figure-4-6.-locations-of-homes-in-meadow-village-and-mitchell.">Figure 4-6. Locations of homes in Meadow Village and Mitchell.</h6>
</figure>

A detailed inspection of the map also shows that the neighborhood labels are not completely reliable. For example, <a href="#ames-northridge" data-type="xref">Figure 4-7</a> shows some properties labeled as being in Northridge that are surrounded by homes in the adjacent Somerset neighborhood.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0407.png" alt="tmwr 0407" />
<h6 id="figure-4-7.-locations-of-homes-in-somerset-and-northridge.">Figure 4-7. Locations of homes in Somerset and Northridge.</h6>
</figure>

Also, there are 10 isolated homes labeled as being in Crawford that, as you can see in <a href="#ames-crawford" data-type="xref">Figure 4-8</a>, are not close to the majority of the other homes in that neighborhood.

<figure class="width-80">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0408.png" alt="tmwr 0408" />
<h6 id="figure-4-8.-locations-of-homes-in-crawford.">Figure 4-8. Locations of homes in Crawford.</h6>
</figure>

Also notable is the “Iowa Department of Transportation (DOT) and Rail Road” neighborhood adjacent to the main road on the east side of Ames, shown in <a href="#ames-dot-rr" data-type="xref">Figure 4-9</a>. There are several clusters of homes within this neighborhood as well as some longitudinal outliers; the two homes farthest east are isolated from the other locations.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0409.png" alt="tmwr 0409" />
<h6 id="figure-4-9.-homes-labeled-as-iowa-department-of-transportation-dot-and-rail-road.">Figure 4-9. Homes labeled as Iowa Department of Transportation (DOT) and Rail Road.</h6>
</figure>

As described in <a href="ch01.xhtml#software-modeling" data-type="xref">Chapter 1</a>, it is critical to conduct exploratory data analysis prior to beginning any modeling. These housing data have characteristics that present interesting challenges about how the data should be processed and modeled. We describe many of these in later chapters, such as <a href="ch17.xhtml#categorical" data-type="xref">Chapter 17</a>. Some basic questions that could be examined during this exploratory stage include:

- Is there anything odd or noticeable about the distributions of the individual predictors? Is there much skewness or any pathological distributions?

- Are there high correlations between predictors? For example, there are multiple predictors related to house size. Are some redundant?

- Are there associations between predictors and the outcomes?

Many of these questions will be revisited as these data are used in upcoming examples.

# Chapter Summary

This chapter introduced the Ames housing data set and investigated some of its characteristics. This data set will be used in later chapters to demonstrate tidymodels syntax. Exploratory data analysis like this is an essential component of any modeling project; EDA uncovers information that contributes to better modeling practice.

The important code for preparing the Ames data set that we will carry forward into subsequent chapters is:

```
library(tidymodels)
data(ames)
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
```
