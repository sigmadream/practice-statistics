### 6

###### Text Output

From this chapter forward, we will start touching on the chunk options in knitr. First, in this chapter, we explain how to tune text output, including output from inline R code as well as text output from code chunks.

###### 6.1 Inline Output

If the inline R code produces character results, they will be directly written into the output. When the result is numeric, scientiﬁc notation will be considered to denote the numbers that are too big or too small.

The threshold between scientiﬁc notation and ﬁxed notation is the R option scipen (see ?options for details). By default (scipen = 0), if

- a positive number is bigger than 104 or smaller than 10−4 (this applies to the absolute values of negative numbers too), it will be denoted in


scientiﬁc notation. Depending on the output format (LATEX or HTML), knitr will use the appropriate code, such as $3.14 \times 10^5$ or 3.14 &times; 10<sup>5</sup>. The reason for scientiﬁc notation is to make it easier to read numbers such as small P-values, e.g., compare 0.000143 with 1.43 × 10−4.

Another R option digits controls how many digits a number should be rounded to. R’s default is 7, which often makes a number unnecessarily “precise.” We can change the defaults in the ﬁrst chunk of a document, like:

# numbers >= 10^5 will be denoted in scientific # notation, and rounded to 2 digits options(scipen = 1, digits = 2)

For example, this book uses digits = 4, and a number 123456789 will become 1.2346 × 108 after the book source is compiled to PDF. Note that these two options are not speciﬁc to knitr; they are global

45

options in R. If we are not satisﬁed with the default inline output, we can rewrite the inline hook as introduced in Section 5.3. Next we are going to introduce chunk options that affect the text output from code chunks.

For character results, we may have to take care of some special char-

acters especially for LATEX and HTML, e.g., % means comments in LATEX, and a literal ampersand (&) has to be written as &amp; in HTML. See Section 12.3.6 for how to escape these characters if needed.

In most cases, characters are written as is in the output. For example, \Sexpr{letters[1]} produces “a” in the output of an Rnw document, and  r month.name[2]  in an Rmd document produces “February”. A special case is the R HTML document: inline character results are written in the <code></code> tag by default, e.g., <!--rinline  ABC --> produces <code class=’knitr inline’>ABC</code>. To get rid of the code tag, we can wrap the results in the function I(), which means to print the characters as is, e.g., <!--rinline I( ABC )-->.

###### 6.2 Chunk Output

The “text output” in this section refers to any output from R that is not graphics, so even messages and warnings are classiﬁed as text output.

###### 6.2.1 Chunk Evaluation

The chunk option eval (TRUE or FALSE) decides whether a code chunk should be evaluated. When a chunk is not evaluated, there will be no results returned except the original source code. This option can also take a numeric vector to specify which expressions are to be evaluated; in this case, the code that is set not to be evaluated will be commented out. For the chunk below, we set eval = -2, which means the second expression will not be evaluated:

1 + 1 ## [1] 2

## if (TRUE) { ## print("hi") ## } dnorm(0)

## [1] 0.3989

###### 6.2.2 Code Formatting

The function tidy_source() in the formatR package (Xie, 2015a) will be used to reformat R code (when the chunk option tidy = TRUE), e.g., it can add spaces and indentation, break long lines into shorter ones, and automatically replace the assignment operator = to <-; see the manual of formatR for details. The chunk option tidy.opts (a list) is passed to tidy_source() to control the formatting of R code. The example below shows the effect of tidy = TRUE/FALSE:

# option tidy=FALSE for(k in 1:10){j=cos(sin(k)*k^2)+3;print(j-5)}

# option tidy=TRUE for (k in 1:10) {

j <- cos(sin(k) * k^2) + 3 print(j - 5)

}

We can pass an argument width.cutoff to tidy_source() through the chunk option tidy.opts = list(width.cutoff = 40) so that the width of source code is roughly 40, e.g.,

- 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9


## [1] 180 # all arguments of tidy_source() names(formals(formatR::tidy_source))

## [1] "source" "comment" "blank" ## [4] "arrow" "brace.newline" "indent"

- ## [7] "output" "text" "width.cutoff" ## [10] "..."


###### 6.2.3 Code Decoration

Syntax highlighting comes by default in knitr (chunk option highlight = TRUE), since it enhances the readability of the source code — character strings, comments, and function names, etc., are in different colors.

This is achieved by the highr package (Qiu and Xie, 2015). This option only works for LATEX and HTML output, and it is not necessary for Markdown because there are other libraries that can highlight code in Web pages; e.g., the JavaScript library highlight.js is widely used to do syntax highlighting for HTML pages.

For LATEX output, the LATEX package framed is used to decorate code chunks with a light gray background (as we can see in this book). If this package is not found in the system, a version will be copied directly from knitr. The output for HTML documents is styled with CSS, which looks similar to LATEX (with gray shadings and syntax highlighting). The background color is controlled by the chunk option background, which takes a color value such as ’#FF0000’, ’red’, or rgb(1, 0, 0) (as long as it is a valid color in R).

The prompt characters are removed by default because they mangle the R source code in the output and make it difﬁcult to copy R code. The R output is masked in comments by default based on the same rationale (option comment = ’##’). In fact, this was largely motivated from the author’s experience of grading homework when he was a teaching assistant; with the default prompts, it is difﬁcult to verify the results in the homework, because it is so inconvenient to copy the source code. Anyway, it is easy to revert to the output with prompts (set option prompt = TRUE), and we will quickly realize the inconvenience to the readers if they want to run the code in the output document, e.g., the chunk below uses prompt = TRUE and comment = NA:

> x <- rnorm(5) > x [1] -0.01156 -0.90915 0.37367 1.90694 0.16459 > var(x) [1] 1.041

While this may seem to be irrelevant to reproducible research, we would argue that it is of great importance to design styles that look appealing and helpful at ﬁrst glance, which can encourage users to write reports in this way.

For LATEX output, we can also specify the font size of the chunk output via the size option, which takes the value of LATEX font sizes such

asfootnotesize,small,large, andLarge, etc., (the default size is normal-

size). It is helpful to set a smaller font size when the output is long and the space is limited, e.g., in beamer slides. The chunk below uses size = ’footnotesize’:

<<font-size, size= footnotesize >>= x <- rnorm(20, mean = 5, sd = 3) x^2

## [1] 5.039 8.314 10.604 5.749 28.855 38.501 14.089

- ## [8] 10.535 16.023 94.736 32.549 33.854 37.890 54.440


- ## [15] 41.333 31.910 8.445 2.227 46.454 25.077 @


###### 6.2.4 Show/Hide Output

We can show or hide different parts of the text output including the source code, normal text output, warnings, messages, errors, and the whole chunk. Below are the corresponding chunk options with default values in the braces:

echo (TRUE) whether to show the source code; it can also take a numeric vector like the eval option to select which expressions to show in the output, e.g., echo = 1:3 selects the ﬁrst 3 expressions, and echo = -5 means do not show the 5th expression.

results (’markup’) how to wrap up the normal text output that would have been printed in the R console if we had run the code in R; the default value means to mark up the results in special environments such as LATEX environments or HTML div tags; other possible values are:

’asis’ write the raw output from R to the output document without any markups, e.g., the source code cat(’<em>emphasize</em>’) can produce an italic text in HTML when results = ’asis’; this is very useful when we use R to produce raw elements for the output, e.g., tables using the LATEX markup (Section 6.3);

’hold’ hold the text output and write to the end of the chunk; ’hide’ this option value hides the normal text output.

warning/error/message (TRUE) whether to show warnings, errors, and messages in the output; usually these three types of messages are produced by warning(), stop(), and message() in R.

split (FALSE) whether to redirect the chunk output to a separate ﬁle (the ﬁlename is determined by the chunk label); for LATEX, \input{} will be used if split = TRUE to input the chunk output from the ﬁle; for HTML, the <iframe> tag will be used; other output formats will ignore this option.

include (TRUE) whether to include the chunk output in the document; when it is FALSE, the whole chunk will be absent in the output, but the code chunk will still be evaluated unless eval = FALSE.

Below is an example that shows results = ’asis’ and three types of messages:

- b <- coef(lm(dist ~ speed, data = cars)) # write out the regression equation cat(sprintf("$dist = %.02f + %.02f speed$", b[1], b[2]))


dist = −17.58 + 3.93speed

- x <- dnorm(0, sd = -1) # will produce a warning ## Warning in dnorm(0, sd = -1): NaNs produced
- y <- 1 + "a" # not possible; error ## Error: non-numeric argument to binary operator message("hello world!") ## hello world!


If we did not use the results option, we will see the raw LATEX code

instead of an equation in the output: cat(sprintf("$dist = %.02f + %.02f speed$", b[1], b[2])) ## $dist = -17.58 + 3.93 speed$

As we have introduced in Section 5.1, we can use opts_chunk to set global chunk options. For instance, if we want to suppress all warnings and messages in the whole document, then we can do this in the ﬁrst chunk of the document:

knitr::opts_chunk$set(warning = FALSE, message = FALSE)

When warning = FALSE (or message = FALSE), warnings (or messages) will be printed in the R console instead of the report output. If you really want to suppress them, you have to call the function suppressWarnings() (or suppressMessages()) on the R expression, e.g.,

suppressWarnings(1:2 + 1:3) # no more warnings ## [1] 2 4 4 suppressMessages(message("foo"))

It may be very surprising to knitr users that knitr does not stop on errors! As we can see from the previous example, 1 + ’a’ should have stopped R because that is not a valid addition operation in R (a number + a string). The default behavior of knitr is to act as if the code were pasted into an R console: if you paste 1 + ’a’ to the R console, you will see an error message, but that does not halt R — you can continue to type or paste more code. To completely stop knitr when errors occur, we have to set the chunk option error = FALSE:

knitr::opts_chunk$set(error = FALSE)

###### 6.2.5 Collapse Output

Currently this feature applies to R Markdown only. If a code chunk has many short R expressions, and each expression prints some output, it will be disturbing to read the output because R each expression and output fragment occupies a separate visual block. In this case, you can collapse all code and output fragments into one block using the chunk option collapse = TRUE. Here is an example:

- 1 + 1 ## [1] 2
- 2 + 3 ## [1] 5 if (TRUE) 1:10


- ## [1] 1 2 3 4 5 6 7 8 9 10


This is what the default output looks like (i.e., when collapse = FALSE):

- 1 + 1

## [1] 2

- 2 + 3


## [1] 5

if (TRUE) 1:10

## [1] 1 2 3 4 5 6 7 8 9 10

###### 6.2.6 Trim Blank Lines

The chunk option strip.white (TRUE by default) can be used to strip blank lines at the beginning and end of a source code chunk. For example, the blank line at the end of this chunk will be removed by default:

1 + 1 # a blank line below

###### 6.3 Tables

Tables are essentially text output, but the ﬁrst edition of this book did not cover table generation for a number of reasons:

- 1. this functionality is orthogonal to knitr — as long as we can ﬁnd another package to create the table, knitr can easily show it in the output with the chunk option results = ’asis’; a few good examples include xtable (Dahl, 2014), Hmisc (Harrell, 2015), and tables (Murdoch, 2012);
- 2. it can be very challenging and complicated to generate tables for different document formats and different types of R objects, and the author has not found a perfect solution yet;


- 3. sometimes graphics can present the information better than tables, and it is much easier to make plots.


However, it seems there is still high demand on this particular feature, so we will expand this topic a little bit. For LATEX tables, the packages mentioned above should work well. For HTML tables, xtable and R2HTML (Lecoutre, 2014) can be used. Additionally, Table 1.1 is an example of kable(), a simple function provided in knitr for LATEX, HTML, and Markdown tables. More importantly, the kable() function is aware of the output format, and can automatically generate a table of the appropriate format, e.g., for the same data object, it generates a LATEX table in an Rnw document, a Markdown table in an Rmd document, and an HTML table in an R HTML document. Therefore, you do not need to remember which type of document you are in, and just call kable() in your code chunk. The code chunk below shows the source code of tables of different formats:

# define a function to print the table source kable_source <- function(...) cat(kable(...), sep = "\n") # an example data frame d <- data.frame(a = 1:3, b = pi * (1:3), c = c("ab", "cd",

"efg")) # the second argument of kable() is the output format kable_source(d, "latex")

\begin{tabular}{rrl} a & b & c\\ \hline 1 & 3.142 & ab\\ \hline 2 & 6.283 & cd\\ \hline 3 & 9.425 & efg\\ \end{tabular}

kable_source(d, "markdown")

| a| b|c | |--:|-----:|:---|

- | 1| 3.142|ab |
- | 2| 6.283|cd |
- | 3| 9.425|efg |


# center the first and third columns, and right align # the second kable_source(d, "markdown", align = c("c", "r", "c"))

| a | b| c | |:-:|-----:|:---:| | 1 | 3.142| ab | | 2 | 6.283| cd | | 3 | 9.425| efg |

###### kable_source(d, "pandoc")

a b c

--- ------ ----

- 1 3.142 ab
- 2 6.283 cd
- 3 9.425 efg


# use two digits kable_source(d, "pandoc", digits = 2)

a b c

--- ----- ---1 3.14 ab 2 6.28 cd 3 9.42 efg

# use different column names use two digits kable_source(d, "pandoc", col.names = c("AAA", "BBB", "CCC"))

AAA BBB CCC ---- ------ ----

- 1 3.142 ab
- 2 6.283 cd
- 3 9.425 efg


kable_source(d, "html") <table>

<thead> <tr>

- <th style="text-align:right;"> a </th>
- <th style="text-align:right;"> b </th> <th style="text-align:left;"> c </th>


</tr>

</thead> <tbody>

<tr> <td style="text-align:right;"> 1 </td> <td style="text-align:right;"> 3.142 </td> <td style="text-align:left;"> ab </td>

</tr> <tr>

<td style="text-align:right;"> 2 </td> <td style="text-align:right;"> 6.283 </td> <td style="text-align:left;"> cd </td>

</tr> <tr>

<td style="text-align:right;"> 3 </td> <td style="text-align:right;"> 9.425 </td> <td style="text-align:left;"> efg </td>

</tr>

</tbody> </table>

If you simply want to display rectangular data as plain tables, kable() can be a good choice. If you want more advanced and complicated features such as conditional formatting (e.g., color certain rows/cells), you are advised to use other packages.

###### 6.4 Automatic Printing

Under the hood, knitr uses the S3 generic function knit_print() to print objects in R code chunks by default. All visible objects are passed to knit_print() to render text output. Basically, knit_print() is the same as print(), but you can extend this S3 generic function by writing S3 methods for it without changing R’s print() function. To know more details about this, please see the package vignette:

vignette("knit_print", package = "knitr")

The printr package (Xie, 2014) has provided several S3 methods for the knit_print() function. Once this package is loaded, you can just type the object names in a code chunk, and knitr will know how to print them automatically according to the output format. For example, when

you type ??sunflower in the R console (?? means help.search() in R), you will see a help window pop up showing the search results using the keyword “sunﬂower.” However, if you type this in an R code chunk, and compile it using knitr, normally you will see nothing because we cannot embed a transient help window in the output. Since ?? is essentially an R function that returns a special object of the class hsearch, the printr package has deﬁned an S3 method knit_print.hsearch() to process the object of search results, so you can use the ?? command after loading the printr package:

library(printr) ??sunflower

Package Topic Title graphics sunﬂowerplot Produce a Sunﬂower Scatter Plot grDevices xyTable Multiplicities of (x,y) Points, e.g., ...

head(iris[, 1:4])

Sepal.Length Sepal.Width Petal.Length Petal.Width 5.1 3.5 1.4 0.2 4.9 3.0 1.4 0.2

- 4.7 3.2 1.3 0.2

- 4.6 3.1 1.5 0.2

- 5.0 3.6 1.4 0.2


- 5.4 3.9 1.7 0.4


From the reader’s perspective, this is cleaner than an explicit call to table-generating functions such as kable() in code chunks: the reader does not need to know what the table function was behind the scenes, and perhaps does not care either.

In fact, you do not have to use the knit_print() function. It is just the default value for the chunk option render, which takes a printing function. You are free to deﬁne another printing function and assign it to the render option. As a trivial example, you can use render = print to restore to the default printing behavior in the R console (print() is a function in base R).

###### 6.5 Themes

The syntax highlighting theme can be adjusted or completely customized. If the default theme is not satisfactory, we can use the object knit_theme

to change it. There are about 80 themes shipped with knitr, and we can view their names by knit_theme$get(). Here are the ﬁrst 20:

head(knit_theme$get(), 20)

## [1] "acid" "aiseered" "andes" ## [4] "anotherdark" "autumn" "baycomb" ## [7] "bclear" "biogoo" "bipolar" ## [10] "blacknblue" "bluegreen" "breeze" ## [13] "bright" "camo" "candy"

- ## [16] "clarity" "dante" "darkblue" ## [19] "darkbone" "darkness"


We can use knit_theme$set() to set the theme, e.g., knit_theme$set("autumn")

Each theme contains a set of color and font deﬁnitions, which will be translated to LATEX commands or CSS deﬁnitions (for HTML) in the end. Note that syntax highlighting themes only work for LATEX and HTML output. For Markdown, the highlight.js library also allows customization but that is beyond the scope of R and knitr. See http: //bit.ly/knitr-themes for a preview of all these themes.

In the next chapter, we show how to control the graphics output.

![image 6](Dynamic Documents with R and knitr 2nd_images/imageFile6.png)

