### 12

###### Tricks and Solutions

In this chapter we show some tricks that can be useful for writing and compiling reports more easily and quickly, and also solutions to frequently asked questions.

###### 12.1 Chunk Options

There are a number of built-in chunk options in knitr, and we usually assign values to them in chunk headers, but it is still possible to customize these ﬁxed options, e.g., rename the options.

###### 12.1.1 Option Aliases

We may feel some options are very frequently used but the names are too long to type. In this case we can set up aliases for chunk options using the function set_alias() in the beginning of a document, e.g.,

set_alias(w = "fig.width", h = "fig.height")

Then we will be able use w and h for the ﬁgure width and height, respectively, e.g., <<fig-size, w=5, h=3>>= plot(1:10) @

The chunk above is equivalent to:

<<fig-size, fig.width=5, fig.height=3>>= plot(1:10) @

127

###### 12.1.2 Option Templates

Besides option names, we can also bundle frequently used option values together as option templates. The object opts_template in knitr can be used to build such templates. A template is a named collection of option sets. For example, if there are a large number of plots for which we want to set the graphical device size to be 7 × 5 inches, and for other plots, we want the size to be 3.5 × 3 inches. We can certainly type fig.width = 7, fig.height = 5 for the ﬁrst group of plots, and fig.width = 3.5, fig.height = 3 for the second group, but this is apparently tedious (even with option aliases). In this case we can just put the two sets of options in templates:

opts_template$set( fig.large = list(fig.width = 7, fig.height = 5), fig.small = list(fig.width = 3.5, fig.height = 3)

)

After the templates have been set up, we can simply use the chunk option opts.label in future chunk headers to reference to them. For instance, we want the options for large plots in the chunk below:

<<fig-ex, opts.label= fig.large >>= plot(1:10) @

This is equivalent to:

<<fig-ex, fig.width=7, fig.height=7>>= plot(1:10) @

###### 12.1.3 Program Chunk Options

Since chunk options can take arbitrary R expressions, we can program chunk options besides setting ﬁxed values like numbers or logical values. We show below an example of drawing a table with the gridExtra package. First we use the tableGrob() function to create a table Grob (graphical object):

library(gridExtra)

###### g <- tableGrob(head(iris))

| |Sepal.Length|Sepal.Width|Petal.Length|Petal.Width|Species|
|---|---|---|---|---|---|
|1|5.1|3.5|1.4|0.2|setosa|
|2|4.9|3.0|1.4|0.2|setosa|
|3|4.7|3.2|1.3|0.2|setosa|
|4|4.6|3.1|1.5|0.2|setosa|
|5|5.0|3.6|1.4|0.2|setosa|
|6|5.4|3.9|1.7|0.4|setosa|


- FIGURE 12.1: A table created by the gridExtra package: we create a table Grob and draw it in a proper graphical device.


Next, we use grid.draw() in the grid package to draw the object to a plot. Prior to that, we need to determine an appropriate size for the graphical device; otherwise we might get extra white margins in the plot. In fact, the convertWidth() and convertHeight() functions in the grid package can convert the pre-calculated width and height of the Grob to inches. Therefore, we pass two function calls to the chunk options fig.width and fig.height instead of using ﬁxed numbers as we usually do. Figure 12.1 is a table of the ﬁrst four lines of the iris data drawn by grid.draw().

<<table, fig.width=convertWidth(grobWidth(g),  in , TRUE)>>= ## width and height in inches convertWidth(grobWidth(g), "in", value = TRUE)

## [1] 5.55 convertHeight(grobHeight(g), "in", value = TRUE) ## [1] 1.94 grid.draw(g) @

The programmable chunk options enable us to program our reports in many aspects. As one potential application, we may build a linear regression report including common diagnostic procedures, with each procedure in a child document (Section 9.3). Then we can decide whether to include certain procedures based on certain conditions, e.g., if we have detected outliers in the regression model, we include an outlier module to deal with outliers. The chunk below shows a sketch of this idea:

<<cooks-distance>>= cookd <- cooks.distance(fit) # include an outlier procedure if any distance is # greater than 1 <<outlier, child=if (any(cookd > 1))  outlier.Rnw >>= @

###### 12.1.4 Code in Appendix

Sometimes we do not want to show the code chunks in the body of the report, but we do not want to completely hide the code, either. In this case we can move all code chunks to the appendix, and the chunk option ref.label can be useful here (Section 9.1.2).

If there are only a small number of code chunks in the document, we can manually type their labels, e.g.,

- <<A, echo=FALSE>>=

- 1+1

<<B, echo=FALSE>>=

- 2+2


- <<C, echo=FALSE>>= rnorm(10) <<show-code, ref.label=c( A ,  B ,  C ), eval=FALSE>>= @


Here we hide the code in the previous chunks by echo = FALSE, and gather them into the last chunk by ref.label. Note the last chunk used the chunk option eval = FALSE so that the code is not evaluated again.

If there are a lot of code chunks in a document, we can use the function all_labels() in knitr to obtain all chunk labels in a document, and pass them to ref.label, e.g.,

<<show-code, ref.label=all_labels()>>= @

We can set echo = FALSE globally by opts_chunk$set(), and use echo = TRUE for the last chunk to show the code there. Of course we can also select chunk labels to include there, e.g., remove the ﬁrst chunk by all_labels()[-1].

###### 12.1.5 Local R Options

The chunk option R.options can take a list of R options to be passed to options() for a code chunk. These options will be applied to the code chunk, and restored after the chunk, so it can be useful if you want to temporarily change R options for a particular code chunk.

For example, we use local options width = 30 (the approximate width for printing) and digits = 2 (the number of digits for printing) for the following code chunk:

<<R.options = list(width=30, digits=2)>>= seq(0, 10, length = 20)

## [1] 0.00 0.53 1.05 1.58 ## [5] 2.11 2.63 3.16 3.68 ## [9] 4.21 4.74 5.26 5.79 ## [13] 6.32 6.84 7.37 7.89

- ## [17] 8.42 8.95 9.47 10.00 @


###### 12.1.6 Dynamic Code

Usually we just type the code in a chunk, or include code from other chunks by references (Chapter 9). There is yet another way to assign code to a chunk, using the chunk option named code. This makes it possible to construct a code chunk dynamically. For example, you can read the code from an external script:

<<code = readLines( foo.R )>>= @

###### 12.2 Package Options

Although we did not speciﬁcally mention it before, there is an object named opts_knit in knitr that controls some package-level options, and its usage is the same as chunk options (opts_chunk).

By default we see a progress bar when we call knitr, and we can suppress it by setting opts_knit$set(progress = FALSE). The progress

bar shows the progress of knit() so we know which chunk is currently being compiled if it takes a relatively long time. To see more information about chunks such as the source code, we can turn on the verbose mode by opts_knit$set(verbose = TRUE).

The package option root.dir can be used to set the root working directory when evaluating code chunks. The default working directory is the directory of the input document, but we can change it with this option, e.g., after we set

opts_knit$set(root.dir = "/home/foo/bar/")

Then we can read a data ﬁle under that directory without using the full path, but in general, we recommend putting datasets and source documents in the same directory, and use this directory as the working directory.

For the chunks that are not labeled, automatic labels of the form unnamed-chunk-i will be used. This can be customized via the package option unnamed.chunk.label, e.g.,

opts_knit$set(unnamed.chunk.label = "fig") Then the automatic chunk labels will be fig-1, fig-2, and so on.

###### 12.3 Typesetting

In this section we show some solutions to tweaking the typesetting of a report.

###### 12.3.1 Output Width

A common problem of using knitr in LATEX is that the output width may exceed the page margin. There are three types of widths: the width of the source code, the text output, and the graphics output. In Section 7.4 we mentioned \maxwidth, which guarantees the graphics output will not be wider than the page width.

For the width of source code and text output, it is controlled by the global option width in options() (Section 6.2.2). The default value for this option is 75, which may be too large for LATEX documents unless we have reset the page margins (e.g., using the geometry package).

When we see the source code or the text output is too wide, we can use a smaller width option, e.g.,

options(width = 55)

However, this may not work all the time: for the source code, R may not be able to ﬁnd an appropriate place to break the source lines; for text output, the original lines may not contain line breaks (because they are in the verbatim environments, LATEX will not break the lines automatically). For the example below, the text lines will not be wrapped no matter how small the width option is: # unable to wrap the source code x <- "thisistoolongandRisunabletofindaplacetoinsertthelinebreak" # unable to wrap the output line cat(x, "---") ## thisistoolongandRisunabletofindaplacetoinsertthelinebreak ---

This is an extreme example. Normally our source code can be formatted into several lines. If we have a character string that is too long in the source code, we can consider breaking it into smaller pieces manually and pasting them together with paste(), e.g.,

x <- paste("this", "is", "too", "long", "and", "R", "is", "unable", "to", "find", "a", "place", "to", "insert", "the", "line", "break", sep = "")

An alternative approach is to use the listings style (recall Figure 5.2 and the function render_listings()). We can set the breaklines option to true for the listings package in the LATEX preamble: \lstset{breaklines=true}

See Figure 12.2 for an example of this option in LATEX.

###### 12.3.2 Message Colors

For LATEX output, there are three colors deﬁned, corresponding to messages, warnings, and errors, respectively:

\definecolor{messagecolor}{rgb}{0, 0, 0} \definecolor{warningcolor}{rgb}{1, 0, 1} \definecolor{errorcolor}{rgb}{1, 0, 0}

By default messages are black, warnings are magenta, and errors are red. We can redeﬁne them using the command \definecolor{} in the LATEX preamble.

|We can set the breaklines option to true to wrap long lines.<br><br>print (” a s d l f j k s a d f l k j kljsd klwjr klwjre klwjer kljwre kljwer lkjrwee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf ”)<br><br>|[1] "asdlfjk sadflkj kljsd klwjr klwjre klwjer kljwre kljwer lkjrwee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf"<br><br>|
|---|
<br><br>By comparison, this shows breaklines=false:<br><br>print (” a s d l f j k s a d f l k j kljsd klwjr klwjre klwjer kljwre kljwer lkjrw<br><br>|[1] "asdlfjk sadflkj kljsd klwjr klwjre klwjer kljwre kljwer lkjrwee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf"<br><br>|
|---|
|
|---|


ee lkwjre lkwjere lkwjer lkwjre lkasdfa afsd afdafsd afddadf adfsadf afdasdf ”)

- FIGURE 12.2: Break long lines with listings: we can use the function render_listings() in R and \lstset{breaklines=true} in LATEX.


###### 12.3.3 Box Padding

As we introduced in Section 6.2.3, the default LATEX style of knitr is based on the framed package, and that is why we see shaded boxes underneath all code chunks. If we feel the default padding of the box is too tight, we can reset the length of \fboxsep{} by \setlength, e.g.,

\setlength\fboxsep{5mm}

## an intentional comment to to to to to to to to to to to to ## reach the page margin rpois(40, 5)

## [1] 6 4 6 4 9 5 2 4 2 4 4 10 6 3 1 8 8

- ## [18] 2 7 4 10 6 5 2 7 4 6 4 2 5 8 7 2 3 ## [35] 2 7 7 3 3 3


Now we see the gray box is larger, with a padding space of 5 mm. For HTML output, it is much easier to design the style, e.g., we can deﬁne the class chunk in CSS as this to make the padding 5 mm:

div.chunk {

padding: 5mm; }

|\documentclass{beamer} \begin{document} \title{Using knitr in Beamer} \author{Yihui Xie}<br><br>\maketitle \begin{frame} \frametitle{Introduction} This is a normal slide. \end{frame} % need the option [fragile] for code output! \begin{frame}[fragile] \frametitle{Code chunks} <<test, out.width= .6\\linewidth , fig.align= center >>= par(mar = c(4, 4, .1, .1)) x = rnorm(100) hist(x, main=  , col= lightblue , border= white ) rug(x) @ \end{frame} \end{document}|
|---|


- FIGURE 12.3: A simple example of using knitr in beamer slides: note that we need the option [fragile] after \begin{frame}.


###### 12.3.4 Beamer

Beamer (Tantau et al., 2012) is a popular document class to create slides with LATEX. Using knitr in beamer slides is not very different from other LATEX documents; the only thing to keep in mind is that we need to specify the fragile option on beamer frames when we have verbatim output. See Figure 12.3 for the Rnw source of a simple beamer example, with one page of the output in Figure 12.4.

Due to the limited space in beamer slides, it may be desirable to use smaller font sizes for the code. In this case we can set a global chunk option size, e.g.,

|Code chunks<br><br>par(mar = c(4, 4, 0.1, 0.1)) x = rnorm(100) hist(x, main = "", col = "lightblue", border = "white") rug(x)<br><br>Frequency<br><br>| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
<br><br>−2 −1 0 1 2<br><br>02015105<br><br>|
|---|


02015105

Frequency

- FIGURE 12.4: A sample page of beamer slides: a code chunk with a plot.


<<setup, include=FALSE>>= opts_chunk$set(size = "footnotesize") @

Next we show an example of programming the content of output, which makes it possible to use the beamer command \only{} to show plots one by one in the same place on the screen (for more information, see the beamer manual). The basic idea is to replace the graphics command \includegraphics{} by \only<n>{\includegraphics{}}, with n being the n-th plot in the current chunk. Below is a modiﬁed plot hook that does this job:

<<setup, include=FALSE>>= hook_plot <- knit_hooks$get("plot") # the default hook # tweak and reset the default hook knit_hooks$set(plot = function(x, options) {

txt <- hook_plot(x, options) if (options$fig.cur <= 0)

###### return(txt)

#  add \only<n> before \includegraphics gsub("(\\\\includegraphics[^}]+})",

sprintf("\\\\only<%d>{\\1}", options$fig.cur), txt)

}) @

One key here is the option fig.cur, which is an internal chunk option (not speciﬁed by users) providing the current ﬁgure number. The substitution of \includegraphics{} was done through regular expressions. After we have modiﬁed the plot hook, the plot commands in LATEX output will be changed accordingly.

###### 12.3.5 Suppress Long Output

For those who have read the book “Modern Applied Statistics with S” (MASS) by Venables and Ripley (2002), you may have noticed that the authors omitted parts of the output in the book in several places, because the output will otherwise be too long. For example, the data frame painters on page 17 has 54 rows, but only the ﬁrst 5 rows were shown on that page, and the rest of the rows were omitted (the omission was denoted by ....). We can automate this job by redeﬁning the output hook in knitr (Section 5.3), e.g.,

# the default output hook hook_output <- knit_hooks$get("output") knit_hooks$set(output = function(x, options) {

# print the first 5 lines by default if (is.null(n <- options$out.lines))

n <- 5 x <- unlist(stringr::str_split(x, "\n")) if (length(x) > n) {

# truncate the output x <- c(head(x, n), "....\n")

}

# paste first n lines together x <- paste(x, collapse = "\n") hook_output(x, options)

})

Then we can achieve a similar effect of the example in the MASS book:

library(MASS) painters

## Composition Drawing Colour Expression ## Da Udine 10 8 16 3 ## Da Vinci 15 16 4 14 ## Del Piombo 8 13 16 7 ## Del Sarto 12 16 9 8 ....

The basic idea of the hook deﬁned above is, if the number of lines of the output is greater than 5, we extract the ﬁrst 5 lines by head(x, 5), and append .... to the output vector, then pass the modiﬁed output to the default output hook function hook_output(), which was obtained before we reset the output hook. We do not have to hard-code the number of lines to be 5, so we also check if the chunk option out.lines is NULL; if it is not, it is supposed to be a number to specify the number of lines to keep in the output. For example, we print the ﬁrst 10 lines instead:

<<print-painters, out.lines=8>>= library(MASS) painters @

Note this hook applies to all document formats (Rnw and Rmd, etc.), because we do not have any document-speciﬁc code in the new deﬁnition; for different document formats, knit_hooks$get(’output’) will be different as well, hence the new hook is portable.

###### 12.3.6 Escape Special Characters

As introduced in Section 5.3, the inline hook function is used to write inline results into the output. By default, it writes characters as is, and sometimes we may want to escape special characters in LATEX or HTML, e.g., an inline R code fragment produces a percentage 30%, and we have to write % as \% in LATEX, otherwise it means LATEX comments.

It is unclear whether we should escape special characters or not, e.g., we may generate a LATEX equation from inline R code, in which case we must not escape special characters such as backslashes. Anyway, if we do want to escape them, we can create a new inline hook function, e.g.,

# get the default inline hook hook_inline <- knit_hooks$get("inline") # build a new inline hook knit_hooks$set(inline = function(x) {

if (is.character(x))

x <- knitr:::escape_latex(x) hook_inline(x)

})

An internal function escape_latex() was used to escape special LATEX characters, and the escaped text strings will be passed to the default inline hook. We only added one step before the default hook function, and all features of the default hook will be preserved, such as automatic scientiﬁc notation (Section 6.1).

Similarly, if we are writing an R HTML document instead, we can call the escape_html() function.

###### 12.3.7 The Example Environment

When writting textbooks or tutorials, it can be useful if we number the R code chunks like theorems and equations. It is easy to deﬁne an “Example” environment in the LATEX preamble, e.g., using the amsthm package:

\usepackage{amsthm} \newtheorem{rexample}{R Example}[section]

Then we can use this new environment rexample in our document:

\begin{rexample} <<test, eval=TRUE>>= 1 + 1 rnorm(10) @ \end{rexample}

In fact, we can automate this job with a chunk hook function, so that we do not have to type the environment again and again. The rexample hook below writes the environment automatically for a chunk with a non-NULL chunk option rexample:

knit_hooks$set(rexample = function(before, options, envir) {

if (before) {

sprintf("\\begin{rexample}\\label{%s}\\hfill{}",

options$label)

} else "\\end{rexample}" })

Basically this hook writes \begin{rexample} before a chunk, and \end{rexample} after it. Additionally, it writes a label for the environment so that we can reference it later, and the label is the chunk label. Now we can apply it to a chunk, e.g.,

<<test, rexample=TRUE>>= 1 + 1 @

Figure 12.5 shows a sample page that used this hook function. We can see the R code chunks are numbered after the section numbers, which is due to the [section] option in the deﬁnition of the rexample environment. Because the rexample environments also come with labels, we can use \ref{} for cross references.

It is also possible to create a similar hook for R HTML documents, but since HTML is not primarily for typesetting purposes, it is not easy to get the automatic numbering as in LATEX. Anyway, we can use our own counter in R, e.g.,

## an example counter for HTML example_count <- 0 knit_hooks$set(rexample = function(before, options, envir) {

if (before) { # increment by 1 example_count <<- example_count + 1 sprintf("<div>Example %d</div>", example_count)

} else "" })

###### 12.3.8 The Docco Style

Besides LATEX documents, you can also use typeset HTML documents. There is a function rocco() in knitr that provids a two-column layout for HTML documents. This style was borrowed from a literate programming package named Docco (https://github.com/jashkenas/ docco). The narratives and code are arranged in separate columns, so that you can keep on reading either the narratives or the code in one

###### Using the Example Environment with knitr

###### Yihui Xie January 2, 2013

Tricks and Solutions 141

|1 Introduction<br><br>This is a test of the R Example environment.<br><br>1.1 Go!<br><br>R Example 1.1. 1 + 1 ## [1] 2<br><br>Look at Example 1.1!<br><br>1.2 Ha!<br><br>R Example 1.2. x = rnorm(10)<br><br>Move on!<br><br>R Example 1.3. sd(x) # standard deviation ## [1] 1.124<br>|
|---|


How about 1.2 and 1.3? If you want to use this R Example environment for all code chunks, make

- FIGURE 12.5: R code chunks in the R Example environments: the examples are numbered following the section numbers.


rexample a global chunk option in the setup chunk.

column. You can hide either column with a keyboard shortcut. Figure 12.6 is a screenshot of a package vignette in knitr that uses this style:

1

vignette("docco-classic", package = "knitr")

###### 12.4 Utilities

There are a few utility functions in knitr to complete miscellaneous tasks such as writing BibTEX databases for R packages, base64 encoding

|![image 16](Dynamic Documents with R and knitr 2nd_images/imageFile16.png)|
|---|


###### FIGURE 12.6: The Docco style for HTML output: the narratives are inthe left column, and the R code is in the right column. You can rendersuch a page from R Markdown using the function rocco() in knitr.

images for HTML output, and compiling source documents to the ﬁnal output.

###### 12.4.1 R Package Citation

The function write_bib() is a wrapper to the functions citation() and toBibtex() in base R. By default it collects the packages loaded into the current R session and extracts their citation information. It also has an argument named tweak, which determines whether to tweak the default citation information, e.g., the author name “Duncan Temple Lang” should be “Duncan {Temple Lang}” in the bibliography database. Instead of manually modifying information like this, write_bib() can automatically deal with it.

write_bib(c("filehash", "RGtk2", "rms")) @Manual{R-filehash,

title = {filehash: Simple key-value database}, author = {Roger D. Peng}, year = {2014}, note = {R package version 2.2-2}, url = {http://CRAN.R-project.org/package=filehash},

} @Manual{R-RGtk2,

title = {RGtk2: R bindings for Gtk 2.8.0 and above}, author = {Michael Lawrence and Duncan {Temple Lang}}, year = {2014}, note = {R package version 2.20.31}, url = {http://CRAN.R-project.org/package=RGtk2},

} @Manual{R-rms,

title = {rms: Regression Modeling Strategies}, author = {Frank E. {Harrell, Jr.}}, year = {2015}, note = {R package version 4.3-0}, url = {http://CRAN.R-project.org/package=rms},

}

The second argument of write_bib() is file, and we can pass a ﬁlename to it to save the bibliography items into a ﬁle. By default, it writes to the standard output.

The advantage of generating the bibliography database using this function is that we can guarantee we always cite the package versions

that we really use in a document. If we hard-code the bibliography, the citations may be out-of-date after we update R packages.

If we do not want to write the ﬁle each time we compile the document, we can cache the chunk. Then a natural question is, when should we, or how can we update the cache? Recall Chapter 8 and one solution is to put the package version(s) in a chunk option, e.g., if the main package that we use for a document is called foo, we can write a chunk like this:

<<write-bib, cache=TRUE, version=packageVersion( foo )>>= write_bib(c("foo", "other", "packages"), file = "paper.bib") @

Then whenever the foo package is updated, the cached chunk will be updated accordingly.

###### 12.4.2 Image URI

It is convenient to publish a PDF report because a PDF document contains everything in one ﬁle, including plots in particular, but that is not true for HTML reports. If an HTML page contains images that are external ﬁles, we have to publish these images along with the HTML ﬁle, otherwise the Web browser will not be able to ﬁnd them. There is a technology called “Data URI” in Web pages that solves this problem. In short, we can encode a ﬁle into a character (base64) string and include it in HTML, so that we do not need the original ﬁle any more when publishing the HTML page. In other words, the HTML page is self-contained just like PDF.

The function image_uri() in knitr was designed to encode images as base64 strings. Obviously it only applies to HTML output (including Markdown). We can enable this function in opts_knit:

opts_knit$set(upload.fun = image_uri)

Then if we have plots in HTML output, the image ﬁle paths will be replaced by base64 character strings. Below is an example of encoding the R logo (a JPEG image):

# encode the R logo logo <- file.path(R.home("doc"), "html", "logo.jpg") uri <- image_uri(logo) # the first 250 characters uri.sub <- substring(uri, seq(1, 201, 50), seq(50, 250,

50))

cat(uri.sub, sep = "\n")

data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEBKwErAAD /4QAWRXhpZgAATU0AKgAAAAgAAAAAAAD/2wBDAAUDBAQEAwUEB AQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCE YGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUH h4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4

###### 12.4.3 Upload Images

Based on the same reason, we designed another function imgur_upload() to upload images to the website Imgur.com, and this function returns the URL of the uploaded image. Then, instead of using the image ﬁle path to reference the image (which has the problem mentioned before), we use a URL that is accessible anywhere as long as we have Internet connection. To continue the previous example, we can upload the R logo to Imgur website by:

imgur_upload(logo)

This returns a URL of the form http://i.imgur.com/xxxxx.jpg. To make things even easier, we can set the package option upload.fun like we did in the last section:

opts_knit$set(upload.fun = imgur_upload)

Then images will be automatically uploaded to Imgur when we knit a document. To avoid repeated uploading of the same image, we can turn on cache.

###### 12.4.4 Compile Documents

For some document formats, there are two steps in compilation. For example, Rnw documents are compiled through knitr to LATEX documents, which need to be compiled to PDF via LATEX. For Rmd documents, the ﬁnal product is often HTML instead of Markdown, which is the direct output of knitr.

To turn the two steps into one, the functions knit2pdf() and knit2html()

can be used. The former will ﬁrst knit() an Rnw document to a TEX document, and then call texi2pdf() in base R to compile it to PDF; the latter will knit() an Rmd document to a Markdown document, and call

markdownToHTML() in the markdown package to compile Markdown to HTML.

For users under Unix-like systems, there is a Bash script named knit under the directory bin of knitr’s installation path; we can ﬁnd it via: system.file("bin", "knit", package = "knitr") ## [1] "/home/yihui/R/knitr/bin/knit"

It is an executable script that calls R to load knitr and automatically uses knit2pdf() or knit2html() based on the ﬁlename extension; if we put this script in the PATH variable, we can call it in command line directly. For example, I have made a symbolic link under ~/bin/ to this script, and added this to ~/.bashrc:

PATH=$PATH:$HOME/bin export PATH

Then we can run knit like other programs in the terminal without having to start R and type all the commands there.

###### 12.4.5 Construct Code Chunks

So far we have been using ﬁles as the input for the knit() function in knitr. As a matter of fact, there is an alternative argument to receive the source document, which is named text.

# arguments of knit() formatR::usage(knit, width = 40)

## knit(input, output = NULL, tangle = FALSE, ## text = NULL, quiet = FALSE, envir = parent.frame(), ## encoding = getOption("encoding"))

If we provide an input ﬁle to knit(), it will be read into knitr and assigned to the text argument eventually. The content of ﬁles is usually ﬁxed, but for the text argument, we can dynamically construct it using R since it is nothing but a character variable.

Now we show a comprehensive example, which builds a PDF document for all the geom examples in the ggplot2 package; see the source code in Figure 12.7 and a sample page of the output in Figure 12.8. It may look a little bit complicated at ﬁrst glance, but the basic idea is simple:

- 1. in the setup chunk, we set two global chunk options: tidy = FALSE (optional) and cache = TRUE (because there are a large number of example code chunks to run later);
- 2. in the write-examples chunk, we use apropos() to ﬁnd all function names that start with geom_; then we ﬁnd their help ﬁles and from there extract the examples code with Rd2ex() in the tools package; ﬁnally we construct Rnw chunks using the function names as section titles and chunk labels, and assign the source text to a variable ex;
- 3. in the last step, we knit the source passed from the text argu-


ment and knit() returns the LATEX code, which we insert into the document as a text string by \Sexpr{};

This source document will produce a PDF document of more than 200 pages, taking a few minutes on the ﬁrst run. Note that it uses the document class tufte-handout, which is a LATEX class you may have to install (it is not a standard class that comes by default).

###### 12.4.6 Extract Source Code

We mentioned the function purl() brieﬂy in Section 3.4. Actually it has an additional argument named documentation, which controls the level of details of documentation chunks.

args(purl)

## function (..., documentation = 1L) ## NULL

The documentation argument takes three possible values:

- 0L discard all text chunks, including chunk headers, so the output is pure program code
- 1L discard text chunks but preserve chunk headers in the exported code ﬁle
- 2L keep everything in the source document but put text chunks in roxygen comments (i.e., after #’)


The following chunk shows examples corresponding to three values of the documentation argument. Note that the chunk headers are written after ## ----, and text chunks are after #’. When documentation = 2, the generated R script can be passed to the function spin() to restore the original document (Section 5.4).

|\documentclass[a4paper,titlepage]{tufte-handout} \title{ggplot2 Gallery} \begin{document} \maketitle \tableofcontents<br><br><<setup, include=FALSE>>= # cache chunks and do not tidy ggplot2 examples code opts_chunk$set(tidy = FALSE, cache = TRUE) @<br><br>% all geoms in ggplot2 <<write-examples, include=FALSE>>= library(ggplot2) ex = lapply(apropos("^geom_"), function(g) {<br><br>p = utils:::index.search(g, find.package(), TRUE) tools::Rd2ex(utils:::.getHelpFile(p), f <- tempfile()) c(sprintf("\\section{%s}\n\n<<%s>>=",<br><br>knitr:::escape_latex(g), g), readLines(f), "@\n\n")<br><br>}) @<br><br>\Sexpr{knit(text = unlist(ex))} \end{document}|
|---|


- FIGURE 12.7: The source document of the ggplot2 geom examples: the Rd2ex() function was used to extract all examples code for the geom functions, and we construct code chunks using the Rnw syntax for knitr to compile.


![image 17](Dynamic Documents with R and knitr 2nd_images/imageFile17.png)

35

30

25

factor(cyl)

4 6

mpg

20

8

15

10

2 3 4 wt

5

- FIGURE 12.8: A sample page of the ggplot2 documentation: the section titles, code, and plots are all dynamically generated.


src <- c("this is the source document", "<<A, tidy=FALSE>>=", "1+1", "@", "the end")

- cat(purl(text = src, documentation = 0L)) 1+1
- cat(purl(text = src, documentation = 1L))

## ----A, tidy=FALSE----------------------------------1+1

- cat(purl(text = src, documentation = 2L))


#  this is the source document ## ----A, tidy=FALSE----------------------------------1+1

#  the end

For code chunks that have the chunk option purl = FALSE, their code will be ignored. For those chunks that have eval = FALSE, their code will be commented out.

###### 12.4.7 Reproducible Simulation

As we discussed in Chapter 8, it is not trivial to write a report that can be easily and completely reproducible for others. One challenge is to make random simulations reproducible. Of course we can use set.seed() to ﬁx the random seed, but what if we have enabled cache?

The problem is, when should we update a cached chunk that involves random numbers? One sufﬁcient condition is the change of the random seed, i.e., if the random seed has changed before a chunk, this chunk should be re-evaluated.

The object rand_seed in knitr was designed for this purpose. This object is essentially an unevaluated expression: rand_seed

## { ## if (exists(".Random.seed", envir = globalenv())) ## get(".Random.seed", envir = globalenv()) ## }

###### is.language(rand_seed)

## [1] TRUE

Basically it returns the random seed if it exists. We can assign this object to a chunk option; because it is an unevaluated expression, each time a chunk is compiled, this object will be evaluated again (knitr will always evaluate unevaluated chunk options). Then if the random seed has changed, knitr will be able to detect the change and update the cached chunk accordingly. Below is an example:

<<random-cache, cache=TRUE, cache.extra=rand_seed>>= x <- rnorm(100) @

Even if we only switched the positions of two cached chunks (with the code and options untouched), the cache will be invalidated because the evaluated results of rand_seed will be different for these two chunks compared to the last run.

###### 12.4.8 R Documentation

R has a standard documentation system, and one thing that can be improved is the examples in the help pages — we can actually run these examples and put the results in the pages, so that it is easier for the reader to know the results without having to copy and paste code from the documentation.

The function knit_rd() was designed for this task: it takes a package name and extracts all its HTML help pages, then compiles all the examples. This can be handy for package authors, because it generates HTML ﬁles that can be published on the Web, and they are richer than the default R documentation. For example, we recompile all the help pages of the rpart package:

knit_rd("rpart")

We will see a few HTML ﬁles under the current working directory. If there are plots in the examples, they will be base64 encoded and embedded in the pages, so we do not need to take care of additional ﬁles — just upload all these HTML ﬁles to a website.

###### 12.4.9 Rst2pdf

Rst2pdf (http://rst2pdf.ralsina.com.ar) is a free software package to create PDF from reStructuredText. If we write the source document

|\documentclass{article} \begin{document} <<read-demo>>= library(diagram) read_demo( flowchart , package =  diagram ,<br><br>labels =  demo-flowchart ) <<demo-flowchart, dev= tikz , cache=TRUE>>= @ \end{document}|
|---|


- FIGURE 12.9: The flowchart demo in the diagram package: we read the demo into knitr, assign a label demo-flowchart to it, and insert it into the document using this label.


in the R reST format (Section 5.2.4), the output from knitr is a *.rst document, and we can call Rst2pdf (if installed) to convert it to PDF via the wrapper function rst2pdf() in knitr, or just call knit2pdf(’foo.Rrst’) in one step.

###### 12.4.10 Package Demos

Some R packages contain demos, which can be run by the demo() function, e.g.,

demo("plotmath") demo("notebook", package = "knitr")

We can insert demos into a source document using the read_demo() function in knitr, which is simply a wrapper of read_chunk() as introduced in Section 9.2.2.

Figure 12.9 shows a complete example of including the flowchart demo of the diagram package into an Rnw document; see Figure 12.10 for a sample page of the output. We can certainly use a simple chunk of one line of code demo(’flowchart’, echo = TRUE) instead, but we will lose syntax highlighting.

###### 12.4.11 Pretty Printing

When we want to see the source code of an R function, we can simply type its name and R will print its source code, e.g.,

![image 18](Dynamic Documents with R and knitr 2nd_images/imageFile18.png)

###### FIGURE 12.10: A sample page of the ﬂowchart demo: we can see thesyntax highlighting as well as the diagram.

fivenum

## function (x, na.rm = TRUE) ## { ## xna <- is.na(x) ## if (any(xna)) { ## if (na.rm) ## x <- x[!xna] ## else return(rep.int(NA, 5)) ## } ## x <- sort(x) ## n <- length(x) ## if (n == 0) ## rep.int(NA, 5) ## else { ## n4 <- floor((n + 3)/2)/2 ## d <- c(1, n4, (n + 1)/2, n + 1 - n4, n) ## 0.5 * (x[floor(d)] + x[ceiling(d)]) ## } ## } ## <environment: namespace:stats>

But since knitr supports syntax highlighting and code reformatting (Sections 6.2.2 and 6.2.3), we may also want to use these features on the function source. The only question is how to get the source code into knitr, and one answer could be read_chunk() again. We deﬁne a function insert_fun() below to assign the (dumped) source code of an R object to a chunk:

insert_fun <- function(name) {

read_chunk(lines = capture.output(dump(name, "")), labels = paste(name, "source", sep = "-")) }

For an object name, its dumped representation will be captured in a code chunk of the label name-source (see ?dump and ?capture.output for details). Now we can use this function to insert the source code of any functions into the source document, e.g., the ﬁvenum() function:

insert_fun("fivenum")

Then we only need to use the chunk label fivenum-source to show the (highlighted and reformatted) source code:

fivenum <- function(x, na.rm = TRUE) {

xna <- is.na(x) if (any(xna)) {

if (na.rm)

x <- x[!xna] else return(rep.int(NA, 5))

} x <- sort(x) n <- length(x) if (n == 0)

rep.int(NA, 5) else { n4 <- floor((n + 3)/2)/2 d <- c(1, n4, (n + 1)/2, n + 1 - n4, n) 0.5 * (x[floor(d)] + x[ceiling(d)])

} }

The source code of the above chunk is:

<<fivenum-source>>= @

###### 12.4.12 A Macro Preprocessor

The function knit_expand() was designed to pre-process a source document, which is often a template ﬁle for creating repeated text with some changing parameters. For example, we may want to build regression models for the same response variable against different independent variables, and all the models are more or less the same form; all we need to change is the variable names in the models. For example, linear regressions of mpg against two variables in the mtcars data:

- fit1 <- lm(mpg ~ cyl + disp, data = mtcars)
- fit2 <- lm(mpg ~ hp + drat, data = mtcars)


The basic idea of knit_expand() is to insert some tags in a template, and dynamically evaluate them in the current environment. Below are a few simple examples:

knit_expand(text = "The value of pi is {{ round(pi,4) }}.") ## [1] "The value of pi is 3.1416."

knit_expand(text = "The value of pi is {{ round(pi,4) }}.",

pi = 1.234567) ## [1] "The value of pi is 1.2346." knit_expand(text = "radius = {{r}} and area = {{pi*r^2}}",

r = 5) ## [1] "radius = 5 and area = 78.5398163397448" knit_expand(text = "$a = {{a}}$ and $b = {{b}}$", a = 1,

b = 2) ## [1] "$a = 1$ and $b = 2$"

As we can see above, the R expressions in {{}} are evaluated and their values are written in the output.

We can dynamically create the source document for knit() based on knit_expand() like the example in Section 12.4.5. As an example, we build the linear regression models of mpg against all combinations of two variables in the mtcars data, with each model in one section. We write a template ﬁle as shown in Figure 12.11 and name it mtcarstemplate.Rnw. Then we can build our models based on this template:

## we can build one model of mpg vs cyl+disp by knit_expand("mtcars-template.Rnw", x1 = "cyl", x2 = "disp",

i = 1) ## and we can vectorize the whole job with mapply() vars <- combn(names(mtcars)[-1], 2) src <- mapply(knit_expand, file = "mtcars-template.Rnw",

x1 = vars[1, ], x2 = vars[2, ], i = seq_len(ncol(vars)))

We used the function combn() to get all combinations of two variables, and passed them to knit_expand() via mapply(). The next step is straightforward: pass the pre-processed source text src to knit(), e.g., knit(text = src, output = ’lm-mtcars.tex’), and we will get the LATEX output with the regression results.

###### 12.4.13 Exit Knitting Early

Sometimes you may not want to knit the whole document, and the function knit_exit() allows you to quit early. Once you put it in a code chunk, the rest of the document will be ignored, and the results from all previous text/code chunks will be returned immediately.

\section{Regression against {{x1}} and {{x2}}} <<lm-{{x1}}-{{x2}}>>= fit{{i}} = lm(mpg ~ {{x1}} + {{x2}}, data = mtcars) summary(fit{{i}}) @

- FIGURE 12.11: A template of regression models: the variables x1 and x2 will be substituted by two variable names in mtcars, the chunk labels are also created from variable names (so they are unique).


###### 12.4.14 Literal knitr Source Code

You may ﬁnd it a difﬁcult task when you want to write literal knitr source code, such as the source code of an inline R expression, e.g., \Sexpr{x}. This is a common task especially when you write knitr tutorials. You certainly cannot write the source code as-is, because knitr will evaluate it. You cannot even write \verb|\Sexpr{x}|, since knitr does not understand the special meaning of the LATEX command \verb||. Similarly, it may be difﬁcult to write a literal inline expression  r x  in R Markdown.

The function inline_expr() in knitr provides one solution to this problem. It takes a character string, and wraps it using the appropriate syntax of inline expressions.

inline_expr("1 + 1") ## [1] "\\Sexpr{1 + 1}" inline_expr("paste( a ,  b )") ## [1] "\\Sexpr{paste( a ,  b )}"

Then you can call this function in an inline expression. For exam-

ple, \verb|\Sexpr{inline_expr(’1 + 1’)}| in Rnw documents, or  r inline_expr{ 1 + 1 }  in Rmd documents. Another solution is to mutate certain characters in the inline expres-

sion, e.g., instead of \Sexpr{}, you can write \textbackslash{}Sexpr{} in LATEX, since the latter will not be recognized as an inline expression.

There is a similar challenge for writing literal code chunks. Again, you just need to change the source code of the code chunk so that it is

no longer recognizable by knitr. For example, you can add an inline expression with an empty character string before the chunk header, such

- as \Sexpr{”}< <> >=, or  r       {r}. Such lines will not be treated as valid chunk headers, because knitr’s syntax only allows white spaces before the chunk header.


###### 12.4.15 Spell Checking

Base R has a spell check function aspell() in the utils package, which can perform spell check via Aspell, Hunspell, or Ispell. To check the spelling of knitr documents, you may want to skip code chunks, because program code often contains words that are considered as misspelled.

The aspell() function can take a ﬁlter function to skip certain lines in the ﬁles. The function knit_ﬁlter() was designed to skip code chunks in a ﬁle. Here are two examples of checking an Rnw and Rmd ﬁle, respectively:

library(knitr) knitr_example <- function(...) system.file("examples", ...,

package = "knitr")

# -t means the TeX mode aspell(knitr_example("knitr-minimal.Rnw"), knit_filter,

control = "-t")

## backref ## /home/yihui/R/knitr/examples/knitr-minimal.Rnw:13:37 ## ## boxplots ## /home/yihui/R/knitr/examples/knitr-minimal.Rnw:41:45 ## ## colorlinks ## /home/yihui/R/knitr/examples/knitr-minimal.Rnw:13:51 ## ## knitr ## /home/yihui/R/knitr/examples/knitr-minimal.Rnw:26:26

.... # -H is the HTML mode aspell(knitr_example("knitr-minimal.Rmd"), knit_filter,

control = "-H -t") ## knitr

## /home/yihui/R/knitr/examples/knitr-minimal.Rmd:3:38 ## /home/yihui/R/knitr/examples/knitr-minimal.Rmd:59:42 ## ## LaTeX ## /home/yihui/R/knitr/examples/knitr-minimal.Rmd:38:1

You can add words that you know are correctly spelled to a dictionary, so the spell checker does not report them the next time. R has a built-in dictionary, which contains the word “LATEX”. Once we apply this dictionary, you will see the word “LATEX” is no longer reported (but “knitr” still is):

# use a dictionary: LaTeX is a known word dict <- Sys.glob(file.path(R.home("share"), "dictionaries",

"*.rds")) # what s in the dictionary? if (length(dict) >= 1) head(readRDS(dict[1]), 20)

## [1] "Accessor" "accessor" ## [3] "accessors" "ACF" ## [5] "Affymetrix" "AIC" ## [7] "Akaike" "Akaike s" ## [9] "alikes" "ANOVA" ## [11] "API" "approximative" ## [13] "ARIMA" "ARMA" ## [15] "ascii" "AUC" ## [17] "autocorrelation" "autocorrelations" ## [19] "autocovariance" "autocovariances"

aspell(knitr_example("knitr-minimal.Rmd"), knit_filter,

control = "-H -t", dictionaries = dict)

## knitr ## /home/yihui/R/knitr/examples/knitr-minimal.Rmd:3:38 ## /home/yihui/R/knitr/examples/knitr-minimal.Rmd:59:42

###### 12.5 Debugging

Although there is no hard requirement on whether to run knitr in an interactive or non-interactive R session, it is recommended to use a new

non-interactive R session because it is less likely to be “polluted” by existing objects in the R workspace. Based on this consideration, some editors such as RStudio open a new R session to compile reports by default.

The problem with non-interactive R sessions is that debugging may be inconvenient. If an error occurs, knitr will quit from R with a message printed on screen showing the problematic chunk, including its label and line numbers.

If the information mentioned above is not enough, we can also open an interactive R session and run knit() there. When an error occurs in this case, we can use common debugging tools such as traceback() (to see the call stacks that led to the error), or debug(), or browser().

###### 12.6 Multilingual Support

If the source document was not encoded with the native encoding of the current system, we will have to manually specify its encoding via the encoding argument in knit(). For example, if the source document was written in Simpliﬁed Chinese and encoded in GB2312, we need to compile it by:

knit("yourfile.Rnw", encoding = "GB2312")

Note that knitr does not try to automatically detect the encoding of the input document, but the editors usually know the encoding information about the documents. For example, both RStudio and LYX will pass the encoding string to knitr before a document is compiled.

### 13

###### Publishing Reports

After compiling a report through knitr, the output document may not be the end product directly. In particular, output from Rnw documents and Rmd documents often needs further compilation. The direct output from Rnw is LATEX, which can be compiled to PDF. The output from Rmd is Markdown, and what we really read is a Web page converted from Markdown.

There is not much left to do with LATEX — the tool chain is fairly standard and mature (LATEX, PDFTEX, XeTEX, and LuaTEX, etc). When we publish reports based on Rnw source documents, we only need to publish a single PDF ﬁle. One thing that we may need to do is to hide the source code, since the reader may not be interested in reading it. In that case, we can set the chunk option echo to be FALSE globally, and sometimes we may also want to hide the messages and warnings from R:

<<setup, include=FALSE>>= knitr::opts_chunk$set(

echo = FALSE, message = FALSE, warning = FALSE

) @

Then only the results will be shown in the ﬁnal report. In this chapter, we introduce some tools that can help us convert the results from knitr to end products, as well as some presentation tools.

###### 13.1 RStudio

As we have introduced in Section 4.1, RStudio has comprehensive support for knitr. One thing that RStudio has made really easy is the publishing of HTML reports produced from R Markdown. After we click the Knit HTML button, we can see a button named Publish in the toolbar of the preview page. This button enables us to publish the report to the

161

website http://rpubs.com with one click. You need to register on the website in advance so that the report can be published to your account.

What happens behind the scenes when we click the Knit HTML button is that RStudio calls knitr to compile Rmd to Markdown, then RStudio calls Pandoc to convert Markdown to HTML. In the second step, Pandoc tries to ﬁnd out all possible images in the document and encodes them as base64 strings (Section 12.4.2) so that the HTML ﬁle becomes self-contained. When we publish them to the website, we do not need to upload image ﬁles separately. Alternatively, we can use imgur_upload() introduced in Section 12.4.3 to upload images to Imgur.

Besides encoding images, Pandoc also detects LATEX math expressions in the document; if there are any, the JavaScript library MathJax will be used in the HTML header, so that math expressions are rendered correctly on the Web page.

###### 13.2 Pandoc

Pandoc (http://johnmacfarlane.net/pandoc) is a universal document converter. In particular, Pandoc can convert Markdown to many other document formats, including LATEX, HTML, Rich Text Format (*.rtf), EBook (*.epub), Microsoft Word (*.docx), and OpenDocument Text (*.odt), etc. This section tells you how Pandoc works under the hood, and you should see Chapter 14 for R Markdown v2, which is much more convenient to work with than what we introduce in this section.

Pandoc is a command line tool. Linux and Mac users should be ﬁne with it; for Windows users, the command window can be accessed via the Start menu, then Run cmd. Once we have opened a command window (or terminal), we can type commands like this to convert a Markdown ﬁle, say, test.md, to other formats:

pandoc test.md -o test.html pandoc test.md -s --mathjax -o test.html pandoc test.md -o test.odt pandoc test.md -o test.rtf pandoc test.md -o test.docx pandoc test.md -o test.pdf pandoc test.md --latex-engine=xelatex -o test.html pandoc test.md -o test.epub

The option -o speciﬁes the output ﬁlename. Figure 13.1 shows a

screenshot of an OpenDocument Text document, which looks very much like Microsoft Word in terms of the appearance.

There is a function pandoc() in knitr that calls Pandoc from R. It also enables us to embed Pandoc arguments in Rmd documents; see its documentation for details.

It is always a big challenge to ﬁnd a document format that works universally. Some users are not satisﬁed with Word, and other users ﬁnd LATEX difﬁcult to learn. Markdown can be one possible solution due to Pandoc’s support for a large variety of document formats. However, the details in typesetting may not be satisfactory in all document formats, and we are very likely to have to manually tweak the converted documents later.

###### 13.3 HTML5 Slides

To make presentations, we can use the Beamer class mentioned in Section 12.3.4. With the development of Web technologies, we can also make HTML slides on the Web, which we can view in Web browsers, instead of having to download the slides as (PDF or PPT) ﬁles as usual. HTML5 slides also enable us to embed rich media in slides such as video clips and interactive content (e.g., JavaScript applications).

There are a number of ways to make HTML5 slides. One way is to go from Markdown with Pandoc. Figure 13.2 shows an Rmd document, which can be compiled to Markdown through knitr; then we can call Pandoc to convert it to HTML5 slides in the command line (suppose the ﬁlename is test.md):

pandoc -s -t dzslides test.md -o test.html

The option -s tells Pandoc to generate a standalone document (with all CSS deﬁnitions written into this document); the option -t means the format to generate to; note that dzslides is only one possible value for HTML5 slides; see the online documentation of Pandoc for other formats.

Now we can open the HTML ﬁle in a Web browser and use the left/right arrows to navigate through slides.

If we are uncomfortable with command line tools, there are a few R packages such as slidify (Vaidyanathan, 2012) and rmarkdown (Al-

- laire et al., 2015a) that can make life easier. We can create HTML slides directly from Rmd ﬁles, and there are also some nice templates and themes shipped with these packages.


FIGURE13.1:OpenDocumentTextconvertedfromMarkdown:weusedthesameMarkdowndocumentinSection

|![image 19](Dynamic Documents with R and knitr 2nd_images/imageFile19.png)|
|---|


fig.align=’center’3.2.2 but removed the chunk option.

% Writing beautiful and reproducible slides quickly % Yihui Xie % 2012/12/05

# Introduction

- - knitr
- - pandoc # A code chunk


   {r computing} head(cars) cor(cars)

FIGURE 13.2: The source of an example of HTML5 slides: we can compile this document through knitr, then convert the Markdown output to DZSlides via Pandoc.

###### 13.4 Jekyll

Jekyll (http://jekyllrb.com) is a blog engine based on plain text ﬁles. The blog posts can be written in Markdown, therefore it is possible to publish results from knitr to websites. One thing that we need to pay attention to is that the syntax of code blocks is different with traditional Markdown (three backticks): for Jekyll, we need to put code blocks in the Liquid tag:

{% highlight lang %} # code here {% endhighlight %}

We do not need to worry about this technical detail because knitr has a renderer for Jekyll: render_jekyll(). After we call this function, the R code and its output will be written into the correct tags. Actually the syntax for code blocks also depends on which markdown renderer you use for Jekyll. The default renderer is kramdown (http: //kramdown.gettalong.org), which does not support three backticks, but some other renderers may support this syntax, such as redcarpet

(https://github.com/vmg/redcarpet). Again, the big trouble of Markdown is that the syntax is different in different renderers, as we have mentioned in Section 5.2.1.

In fact, the website of knitr (http://yihui.name/knitr) was built with Jekyll and hosted on Github.

###### 13.5 WordPress

WordPress is a free, open-source, and popular blogging system based on PHP and MySQL. It has an API that allows one to publish blog posts from a third-party client. The RWordPress package provides R functions to communicate with a WordPress site. There is a wrapper function knit2wp() in knitr that makes it possible to compile an Rmd document and send it to WordPress directly. See http://yihui.name/ knitr/demo/wordpress/ for details of conﬁgurations such as the login name and password.

### 14

###### R Markdown

There has been a lot of progress on the R Markdown development since the ﬁrst edition of this book. To make it clear, there are two versions of R Markdown: we call the implementation in the markdown package (Al-

- laire et al., 2015b) “R Markdown v1” (https://github.com/rstudio/ markdown), and we call the implementation rmarkdown (Allaire et al., 2015a) “R Markdown v2” (http://rmarkdown.rstudio.com). Unless otherwise noted, use of the term “R Markdown” in this chapter refers to R Markdown v2.


R Markdown v1 is based on the C library sundown, and the major focus is HTML output. Its functionality is very limited, e.g., there is no support for citations or footnotes. R Markdown v2 is based on Pandoc, which has boosted Markdown to a whole new level. There are two aspects of the improvements: the Pandoc Markdown syntax is richer, so we can write more types of elements, and the output format is no longer limited to HTML — we can also export Markdown to LATEX/PDF, Word, and HTML5 slides, etc. In this chapter, we will introduce the design philosophy of rmarkdown, what it can do, and how to customize or extend it.

###### 14.1 Overview

Although knitr supports a variety of document formats (Chapter 5), R Markdown is probably the most popular one. Markdown, limited as it is in terms of functionality, is a nice document language for beginners. On the other hand, authors may not even want a lot of features

- at all. Markdown may be restrictive in the eyes of LATEX users, but not everyone needs to care that much about typesetting details.


The limitation of Markdown can be largely removed by Pandoc, but the problem is that Pandoc is a command-line tool. Power users may not ﬁnd this to be a real problem, but the large number of commandline arguments can be overwhelming to beginners.

167

The goal of rmarkdown and R Markdown v2 is to provide quick conversion of R Markdown ﬁles into other document formats, using reasonably beautiful templates. The way that we achieve the goal is to wrap commonly used command-line arguments into R functions in rmarkdown. The main function in rmarkdown to render R Markdown documents to other document formats is render(). The ﬁrst argument is the Rmd ﬁlename, and the second argument is the output format, which we will introduce in detail later in this chapter. For example, if you want to convert an R Markdown document foo.Rmd to Word, you only need to execute one line of code:

rmarkdown::render("foo.Rmd", "word_document")

You can certainly do it the hard way: ﬁrst, call knit() in knitr to compile foo.Rmd to foo.md; then open a terminal or use the R function system() to execute a command like this, as we introduced in Section

- 13.2:


pandoc foo.md --output foo.docx \

--from markdown+tex_math_single_backslash \

--highlight-style tango

There are seven output format functions in rmarkdown at the moment: PDF, HTML, Word, Markdown, ioslides, Slidy, and Beamer. The ﬁrst four are document formats, and the latter three are presentation formats. They are wrapper functions for both knitr and Pandoc, so you do not need to remember a lot of knitr options and Pandoc arguments — knitr chunk options and Pandoc command-line arguments are converted to rmarkdown function arguments. For example, the Pandoc argument --toc or --table-of-contents corresponds to the function argument toc = TRUE in rmarkdown.

In addition, rmarkdown has provided its own templates that aim to be visually pleasing by default. For example, for HTML output, it uses the Twitter Bootstrap styles and themes. Syntax highlighting for program code is also enabled by default.

The rmarkdown package is well supported in the RStudio IDE: you do not need to manually call the render() function, and you only need to click the Knit button on the toolbar. You can also set the output format and its options from a little GUI popped up through the gear button on the toolbar. If you wish to run rmarkdown outside of RStudio, you will want to learn more details about how rmarkdown works later.

Note RStudio has embedded Pandoc in it, so you do not need to install Pandoc separately if you use RStudio, otherwise you need to

install Pandoc by yourself. If you have a separate installation of Pandoc, RStudio will use it only if your version is higher than RStudio’s Pandoc version.

###### 14.2 Pandoc’s Markdown Extensions

First we introduce the syntax of Pandoc’s Markdown. If you are familiar with R Markdown v1, you can still use its syntax with Pandoc, and the only signiﬁcant change is how to write superscripts that are not math elements. In v1, you use a single caret, e.g., x^2. In Pandoc’s Markdown, you need to surround the superscript with ^, e.g. x^2^. For math expressions, you still use one caret, e.g., $x^2$.

###### 14.2.1 Basic Syntax

The syntax for other elements remains more or less the same in Pandoc’s Markdown. For example, you use one # sign to write the ﬁrst level section header, and two # signs for the second level header. Please review Section 5.2.1 for the syntax of basic elements in Markdown. Below are some new elements that may be useful (see http://johnmacfarlane.

net/pandoc/ for the full documentation), and we show short examples of these elements under the bullets:

- • Deﬁnition lists and example lists A Special Term : Describe/explain the term here.

(@) This is a numbered example. (@) Another numbered example.

(@cool-example) This example is labeled. This is a normal paragraph, and we can reference the example (@cool-example) here.

- • Footnotes using ^[...] and citations using [@id]


We write a nice description of X here^[Not to be confused with Y], and X is useful.

Actually you should read the reference [@joe2014] to know more about X. Here  joe2014  is a key in the bibliography database.

- • Figure/table captions

Pandoc has a Markdown extension named implicit_figures, which is enabled by default. An image

![A figure caption.](path/to/image.png) will be rendered to something like this in LaTeX: \begin{figure}

\includegraphics{path/to/image.png} \caption{A figure caption.}

\end{figure} Similarly, you can add a table caption, e.g. Table: This is a table caption.

--- ---- ---A B C

--- ---- ---a 10 bc d 25 ef --- ---- ----

- • Raw TEX/HTML content


Sometimes you still feel Markdown is limited, and you are so tempted to use LaTeX. That s fine: you can write raw \TeX{} code in Markdown.

Markdown version: ![A long caption.](foo.png)

LaTeX version: \begin{figure}

\includegraphics[width=.8\textwidth]{foo.png} \caption[A short caption]{A long caption.}

\end{figure} Pandoc can preserve the raw TeX content when converting this document to LaTeX/PDF.

When using citations, you need to specify a bibliography database. If you are familiar with LATEX, you are likely to know BibTEX as well. The bibliography database can be a .bib ﬁle speciﬁed in the bibliography ﬁeld in the YAML metadata (see next section). If you do not know BibTEX, you can embed the bibliography items in the YAML metadata using the references ﬁeld (instead of bibliography), e.g.,

--references:

- - id: joe2014 title: A Nice Paper author:

- family: Smith

given: Joe issued:

year: 2014 container-title: The Journal of Awesome Research type: article-journal

- - id: john1980 title: A Great Book author:

- family: Brown given: John issued:

year: 1980 publisher: An Excellent Publisher type: book

- ---


Except for raw TEX/HTML code, all other elements are portable across all document formats. For example, a footnote ^[foo bar] will be converted to \footnote{foo} when the output format is LATEX, and something like <a href=”#footnote-1”><sup>1</sup></a> with the

link target footnote-1 being a footnote item at the bottom of the page when the output format is HTML. You should not expect raw TEX in Markdown to be converted perfectly to Word, or raw HTML to be converted to Beamer, since raw TEX and HTML content can be fairly complicated, and perfect conversion is nearly impossible.

###### 14.2.2 YAML Metadata

Another important extension in Pandoc’s Markdown is the YAML metadata. YAML stands for “YAML Ain’t Markup Language” or “Yet Another Markup Language,” and it is basically a nested list structure. Pandoc uses YAML to write metadata of a document, such as the title, author, and date information. The metadata usually appears in the beginning of a document, and is enclosed between two lines of three dashes ---. Typical YAML metadata looks like this:

--title: "A Nice Report" author: "John Smith" date: 2014/12/31 output:

html_document: toc: yes number_sections: yes

word_document: default

--The body of the R Markdown document.

The most important ﬁeld in the YAML metadata for rmarkdown is the output ﬁeld. This is where we specify the desired output format. If it is missing, rmarkdown will assume the output format to be an HTML document. If multiple formats are speciﬁed, the render() function will use the ﬁrst format by default, unless you have speciﬁed the second argument of render() explicitly. You can also use render(’foo.Rmd’, ’all’) to render all formats deﬁned in the output ﬁeld.

###### 14.3 Output Formats

There is a series of format functions in rmarkdown with the sufﬁxes _document and _presentation, e.g., html_document(), pdf_document(),

and beamer_presentation(), etc. These functions can be used as the second argument of render(), e.g.,

library(rmarkdown) render("foo.Rmd") render("foo.Rmd", pdf_document()) render("foo.Rmd", word_document()) render("foo.Rmd", beamer_presentation()) render("foo.Rmd", ioslides_presentation())

Each output format function has its own arguments. For example, if you want to enable the table of contents for an HTML document, you can call:

library(rmarkdown) render("foo.Rmd", html_document(toc = TRUE))

This is equivalent to providing the YAML metadata as:

--output:

html_document: toc: yes

---

In YAML, both yes and true mean the logical value TRUE. You can either use the YAML metadata and call render() without the second argument, or omit/ignore the YAML metadata and provide the second argument explicitly to render(). The YAML approach is more convenient and common; the output information is contained in the source document. The second approach can be useful when you want to override the output formats deﬁned in YAML. See the help page of each output format function for what the possible options are, e.g., type ?rmarkdown::pdf_document in the R console to see the options for PDF output.

An output format function returns a list of options, including knitr package/chunk options, Pandoc arguments, and other auxiliary options for rmarkdown. We will explain them using html_document() as the example.

###### 14.3.1 HTML Document

To see what html_document() really returns, you can run it and print the structure of the object returned:

library(rmarkdown) str(html_document(), width = 55, strict.width = "wrap")

## List of 6 ## $ knitr :List of 3 ## ..$ opts_knit : NULL ## ..$ opts_chunk:List of 5 ## .. ..$ dev : chr "png" ## .. ..$ dpi : num 96 ## .. ..$ fig.width : num 7 ## .. ..$ fig.height: num 5 ## .. ..$ fig.retina: num 2 ## ..$ knit_hooks: NULL ## $ pandoc :List of 5 ## ..$ to : chr "html" ## ..$ from : chr ## "markdown+autolink_bare_uris+ascii_identifiers+te".. ## ..$ args : chr [1:8] "--smart" "--email-obfuscation" ## "none" "--self-contained" ... ## ..$ keep_tex: logi FALSE ## ..$ ext : NULL ## $ keep_md : logi FALSE ## $ clean_supporting: logi TRUE ## $ pre_processor :function (...) ## $ post_processor :function (metadata, input_file, ## output_file, clean, ## verbose) ## - attr(*, "class")= chr "rmarkdown_output_format"

As you can see, html_document() has modiﬁed some of the knitr default chunk options, such as fig.height (knitr’s default is 7), and fig.retina (the original default is 1). These changes are for aesthetic reasons, although it is somewhat subjective to decide what kind of option values give better-looking results.

The list also contains Pandoc options: the output format is html, as you can see in the element pandoc$to; a few Pandoc arguments such as

--smart and --self-contained are also included in the list.

There are some auxiliary options for rmarkdown, too. For example, clean_supporting means whether to clean up the intermediate output ﬁles after the HTML ﬁle has been rendered. Intermediate ﬁles may include ﬁgure ﬁles: if you want the HTML ﬁle to be self-contained, Pandoc will embed all external resources in it (such as images), so you no

longer need these external ﬁles. In that case, render() will delete them after rendering the HTML ﬁle.

After we know the internals of an output format function, we can write our own format functions using different knitr/Pandoc options. We will introduce how to implement custom formats later in this chapter.

Now we show a full example of an R Markdown v2 document named Rmd-v2.Rmd. It is a little bit long, but it shows most of the features of Pandoc and rmarkdown.

--title: "R Markdown v2 Demo" author:

- Li Lei

- Han Meimei date: "2015/01/01" output:

html_document:

fig_caption: yes pdf_document:

template: null

word_document: default bibliography: Rmd-v2.bib

--# Start with a cool section A bit _introduction_ here. You can use traditional **Markdown** syntax, such as [links](http://yihui.name/knitr) and  code . # Followed by another section Of course you can write lists:

- - apple
- - pear
- - banana Or ordered lists:


1. items

1. will

1. be

1. ordered

- - nested
- - items # More sections ## Hi hi hi ## Hello hello hello ## Howdy howdy howdy # Okay, some R code    {r linear-model} fit = lm(dist ~ speed, data = cars) b = coef(fit) # coefficients summary(fit)


The code will be highlighted in all output formats. # And some pictures    {r lm-vis, fig.cap= Regression diagnostics } par(mfrow = c(2, 2), pch = 20, mar = c(4, 4, 2, .1),

bg =  white ) plot(fit)

# A little bit math Our regression equation is $Y= r b[1] + r b[2] x$, and the model is:

$$ Y = \beta_0 + \beta_1 x + \epsilon$$ # Pandoc extension: definition lists Programmer : A programmer is the one who turns coffee into code. LaTeX : A simple language with a couple of backslashes. # Pandoc extension: examples We have some examples. (@) Think what is  0.3 + 0.4 - 0.7 . Zero. Easy. (@weird) Now think what is  0.3 - 0.7 + 0.4 . Still zero? People are often surprised by (@weird). # Pandoc extension: tables A table here. Table: Demonstration of simple table syntax.    {r echo=FALSE} knitr::kable(head(iris))

# Pandoc extension: footnotes We can also write footnotes[^1]. [^1]: hi, I m a footnote Or write some inline footnotes^[as you can see here]. # Pandoc extension: citations We compile the R Markdown file to Markdown through **knitr** [@R-knitr] in R [@R-base]. For more about @R-knitr, see <http://yihui.name/knitr>.

![image 20](Dynamic Documents with R and knitr 2nd_images/imageFile20.png)

- FIGURE 14.1: A preview of the HTML output document from R Markdown v2 in an RStudio window.


# References    {r include=FALSE} knitr::write_bib(c( base ,  knitr ),  Rmd-v2.bib )

You may need to review the sections 6.3 and 12.4.1 if you are not sure about how kable() or write_bib() works.

Figure 14.1 is a preview of the HTML output document after we render this example in RStudio. It shows the title, author, date, and the ﬁrst few sections of the document. That is the default Twitter Bootstrap style in rmarkdown. Figure 14.2 is a preview of the last few sections. Even though footnotes and citations are not native elements of HTML (they may be natural to LATEX users), Pandoc managed to generate them in HTML anyway.

There is a large number of options that you can tweak for the HTML output. See the help page ?rmarkdown::html_document for a full list.

|![image 21](Dynamic Documents with R and knitr 2nd_images/imageFile21.png)|
|---|


###### FIGURE 14.2: A preview of the table, footnotes, and citations: the tablewas generated by kable(), and the bibliography database was createdfrom write_bib() in knitr.

For example, we change the CSS theme using the theme ﬁeld, add a table of contents using the toc ﬁeld, and number the section titles using the number_sections ﬁeld in YAML (Figure 14.3):

--output:

html_document: fig_caption: yes number_sections: yes theme: readable toc: yes

---

Currently these CSS themes are available in rmarkdown (you can see a preview at http://bootswatch.com):

## [1] "default" "cerulean" "journal" "flatly" ## [5] "readable" "spacelab" "united" "cosmo"

If you need to further tweak the appearance of the output, you can apply your own CSS ﬁles using the css ﬁeld, e.g.,

--output:

html_document: css: my_own.css

---

If you just want to use your own CSS and do not want any themes (including syntax highlighting themes) from rmarkdown, you can remove them completely by specifying theme and highlight to be null:

--output:

html_document: css: my_own.css theme: null highlight: null

---

Because an HTML page often has external dependencies, such as CSS, JavaScript, and image ﬁles, it may be inconvenient when you share the HTML ﬁle with other people, because you have to make sure these dependencies are also included when you send the HTML ﬁle to them.

|![image 22](Dynamic Documents with R and knitr 2nd_images/imageFile22.png)|
|---|


###### FIGURE 14.3: A preview of the “readable” theme (you can see the fontsare different with Figure 14.1), with a table of contents and numberedsections.

Pandoc has an option to make the HTML ﬁle self-contained by embedding all external dependencies into the HTML ﬁle. For example, JavaScript ﬁles are read into the HTML ﬁle, and images are base64 encoded. You can share a self-contained HTML ﬁle just like a PDF ﬁle; everything you need has been embedded into a single ﬁle. In rmarkdown, this is controlled by the option self_contained. When you have multiple Rmd ﬁles to be rendered by rmarkdown, it may be a good idea to turn off the self-contained mode, otherwise there will be a lot of redundancy since some external dependencies may be embedded into every single HTML output ﬁle. When the self-contained mode is off, you can put the shared dependencies into a common directory, speciﬁed via the lib_dir option, e.g.,

--output:

html_document: self_contained: no lib_dir: assets

---

Sometimes you may want to include additional content in the HTML header, before the body, or after the body of the document. In these cases, rmarkdown has an option includes in which you can specify the ﬁlenames of the additional content. Suppose you want to use the JavaScript library D3 (http://d3js.org) in the HTML output, then you can write this in a ﬁle doc_header.html:

<script src="http://d3js.org/d3.v3.min.js" charset="utf-8"> </script>

You also have two ﬁles doc_before.html and doc_after.html, which are the content to be inserted before and after the body, respectively. For example, you may want to write a navigation menu in doc_before.html, and some copyright information in doc_after.html. These three ﬁles can be included in the HTML output ﬁle by:

--output:

html_document:

includes: in_header: doc_header.html before_body: doc_before.html after_body: doc_after.html

For any output format, Pandoc needs a template to create the output ﬁle. There are several Pandoc variables available in the template, and you can use these variables to deﬁne your own template. For example, this can be a minimal HTML template:

<html> <head>

<title>$title$</title> </head> <body> $body$ </body>

</html>

We only used two variables $title$ and $body$ in this template. The ﬁrst variable contains the document title speciﬁed in the title ﬁeld in the YAML metadata. The second variable is the body of the Markdown document after it is converted to HTML. You can learn more possible variables from either the rmarkdown source package (https://github.com/rstudio/rmarkdown) or Pandoc’s default templates (https://github.com/jgm/pandoc-templates).

To use a custom template, you can use the template ﬁeld in YAML, e.g.,

--output:

html_document: template: my_template.html

---

Finally, you can customize command-line arguments to be passed to Pandoc in the pandoc_args ﬁeld. As a matter of fact, the R arguments in html_document() are eventually converted to Pandoc arguments. For example, the R argument self_contained = TRUE (or self_contained: yes in YAML) is equivalent to the Pandoc argument --self-contained, and also equivalent to this in YAML:

--output:

html_document: pandoc_args: "--self-contained"

So far we have covered most of the possibilities to customize the output on the Pandoc’s Markdown side. It is also possible to customize knitr chunk options in YAML. Currently there are four chunk options that you can set in YAML:

ﬁg_width, ﬁg_height the default size of the ﬁgures

ﬁg_retina a scaling ratio for Retina displays; the default is 2 in rmarkdown, which means a ﬁgure of the size m × n has an actual size of 2m × 2n, but is scaled to half of its actual size in the output (this can improve the image qualities on Retina displays)

ﬁg_caption whether to render and show ﬁgure captions (this basically means the figure environment with \caption{} when the output format is LATEX); if FALSE, you will not see the ﬁgure caption in HTML output, since the caption will be put in the alt attribute of the <img> tag, which is invisible

Apparently, the fig_retina option will make the ﬁle size of images larger in return for the image quality. You can try fig_retina = TRUE and FALSE separately, and see if you can notice any differences on your device.

###### 14.3.2 LATEX/PDF Document

Once you are familiar with the HTML document format, it will be easy for you to master other output formats, because many options are common in these formats. For example, you can also use the options such as fig_width, fig_height, toc, number_sections, and highlight in pdf_document(). In this section, we only focus on the options that are speciﬁc to PDF document output.

Figure 14.4 is a preview of a page in the PDF output from the same example we used in the previous section. It does not look too much different from Figure 14.2. For the same R Markdown document, everything that worked in the HTML output still works in LATEX/PDF, including section headings, tables, footnotes, and citations, etc.

Similarly, we can add a table of contents, and number the sections as we did for the HTML output (Figure 14.5):

--output:

pdf_document: number_sections: yes toc: yes

|Pandoc extension: tables<br><br>A table here.<br><br>Table 1: Demonstration of simple table syntax.<br><br>Sepal.Length Sepal.Width Petal.Length Petal.Width Species<br><br>5.1 3.5 1.4 0.2 setosa 4.9 3.0 1.4 0.2 setosa<br><br>4.7 3.2 1.3 0.2 setosa<br><br>4.6 3.1 1.5 0.2 setosa<br>5.0 3.6 1.4 0.2 setosa<br><br><br>5.4 3.9 1.7 0.4 setosa<br><br><br>Pandoc extension: footnotes<br><br>We can also write footnotes1. Or write some inline footnotes2.<br><br>Pandoc extension: citations<br><br>We compile the R Markdown ﬁle to Markdown through knitr (Xie 2014) in R (R Core Team 2014). For more about Xie (2014), see http://yihui.name/knitr.<br><br>References<br><br>R Core Team. 2014. R: A Language and Environment for Statistical Computing. Vienna, Austria: R Foundation for Statistical Computing. http://www.R-project. org/.<br><br>Xie, Yihui. 2014. Knitr: A General-Purpose Package for Dynamic Report Generation in R. http://yihui.name/knitr/.<br><br>1hi, I’m a footnote 2as you can see here|
|---|


###### FIGURE 14.4: A preview of the 4th4 page of the PDF output documentfrom the R Markdown v2 example.

|R Markdown v2 Demo<br><br>Li Lei Han Meimei<br><br>2015/01/01<br><br>Contents<br><br>1 Start with a cool section 2<br>2 Followed by another section 2<br>3 More sections 2<br><br>3.1 Hi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2<br>3.2 Hello . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2<br>3.3 Howdy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2<br><br><br>4 Okay, some R code 3<br>5 And some pictures 3<br>6 A little bit math 4<br>7 Pandoc extension: deﬁnition lists 4<br>8 Pandoc extension: examples 4<br>9 Pandoc extension: tables 5<br>10 Pandoc extension: footnotes 5<br>11 Pandoc extension: citations 5<br><br><br>References 5|
|---|


1

- FIGURE 14.5: A preview of the PDF output document, with a table of contents and numbered sections.


Pandoc has a few LATEX-speciﬁc options that you can use in the YAML metadata, and you can ﬁnd the full documentation on the Pandoc website. We only list a few of them here:

fontsize the font size of the document, e.g., 10pt, 11pt, 12pt documentclass the document class, e.g., article, book, report classoption options for the document class, e.g., a4paper, twocolumn geometry options for the geometry package, e.g., tmargin=2cm, bmar-

gin=2cm, lmargin=3cm, rmargin=3cm

Note these are top-level options in YAML, and you should not put them under the pdf_document ﬁeld.

The default LATEX engine is pdflatex, and you can change it via the latex_engine option in pdf_document(). Currently possible engines are pdflatex, xelatex, and lualatex. You may also preserve the intermediate LATEX output ﬁle via the keep_tex option, which can be useful for debugging and other purposes.

Below is an example of the YAML metadata for a document that uses the book class, a font size of 11pt, a two-column layout, custom margin settings, the XeLATEX engine, and also preserves the LATEX ﬁle:

--documentclass: book classoption: twocolumn fontsize: 11pt geometry:

- - tmargin=2cm
- - bmargin=2cm
- - lmargin=3cm
- - rmargin=3cm

output:

pdf_document: latex_engine: xelatex keep_tex: yes

- ---


We have introduced the includes and template options in the previous section, and they may be more useful for LATEX output, because it is very common for LATEX users to customize the output using certain LATEX packages in the preamble. You can put such content in an external ﬁle, and include it in the preamble via the in_header option under the includes option. If you are not satisﬁed with the default

LATEX template, you can just write your own. Before you really do it, please check the Pandoc documentation carefully to see if you can get what you want by YAML options. It is relatively easy to write a new LATEX template, but it may not be trivial to maintain it in the future, since you need to be aware of possible future changes in Pandoc.

###### 14.3.3 Word Document

There are not many options to customize for Word documents. You can still set the ﬁgure size, and syntax highlighting themes, etc. Figure 14.6 shows the Word output from the example in Microsoft Word 2013.

The most important and useful feature for Word documents is perhaps the template. For other document formats, you can provide a plain text template, but you cannot easily do so for Word, because a Word document is a relatively complicated binary ﬁle. However, Pandoc allows you to provide a Word document as its “reference document,” which is essentially a style template. This reference document must be based on one of Pandoc’s Word output documents, in which you update its styles for different elements. Note only the styles deﬁned in the document will be used, and the content will be largely ignored.

We have prepared a short video at https://vimeo.com/110804387 to show you how to deﬁne styles in Word documents. You can also see Figure 14.7 and 14.8. The basic steps are:

- 1. Create an arbitrary Word document using Pandoc, e.g., use word_document as the output option in the YAML metadata;
- 2. Open the Word document, and ﬁnd the “Styles” panel indicated in Figure 14.7;
- 3. Put the cursor on the element of which you want to modify the style, and there should be an item in the Styles panel highlighted;
- 4. Open the item by clicking the ¶ symbol on the right, and you will see a window like Figure 14.8. That is where you can modify the styles. For example, you can change the font family of the title element to be Bookman Old Style.


After you update the styles of this Word document, you can save it (say, as template.docx under the same directory as the Rmd ﬁle) and use it as the reference document:

--output:

word_document:

![image 23](Dynamic Documents with R and knitr 2nd_images/imageFile23.png)

###### FIGURE 14.6: A preview of the Microsoft Word (2013) document fromR Markdown v2.

![image 24](Dynamic Documents with R and knitr 2nd_images/imageFile24.png)

- FIGURE 14.7: Open the styles panel in Word: ﬁnd a pane named “Styles” on the toolbar, and expand it to a ﬂoating panel.


renference_docx: template.docx

---

Besides the styles of the elements, the styles of the layout can also be respected if you use Pandoc >= 1.13. For example, the margins, page size, page orientation, header, and footer in the reference document will be carried over to the new Word document.

###### 14.3.4 Markdown Documents

An R Markdown document can be converted to different ﬂavors of Markdown documents, such as Pandoc’s Markdown, the original (strict) Markdown, Github Flavored Markdown, MultiMarkdown, and PHP Markdown Extra. You can use the function md_document() for render() or output: md_document in YAML. The main option for md_document is variant, which speciﬁed which ﬂavor of Markdown you want.

![image 25](Dynamic Documents with R and knitr 2nd_images/imageFile25.png)

- FIGURE 14.8: Modify styles of elements in Word: you can change the font family, font size, font style, and color, etc.


###### 14.3.5 ioslides Presentation

R Markdown can be used to create slides for presentation purposes. With the process of Web technologies, HTML5 slides seem to be popular nowadays. You can present slides in a Web browser. This is convenient since you do not need special software packages to display the slides, and you can ﬁnd a Web browser almost everywhere. This is not true for proprietary software such as Microsoft PowerPoint or Keynote for Mac.

There are two types of built-in HTML5 presentation formats in rmarkdown: ioslides and Slidy. You can extend rmarkdown to use your own favorite HTML5 presentation library.

For ioslides, each ﬁrst-level section heading will create a separate

![image 26](Dynamic Documents with R and knitr 2nd_images/imageFile26.png)

- FIGURE 14.9: The title slide of an ioslides presentation: you can also use the table of contents in RStudio to navigate through the slides.


slide with a dark background by default; each second-level heading creates a new slide with the content of this section on it. If you do not want a section heading, you can create a new slide with three dashes ---.

- Figure 14.9 is a screenshot of ioslides in the RStudio preview window, created using the same example as previous sections and the YAML metadata (if you really try this example, you may want to remove the content between the ﬁrst-level heading and second-level heading):


--output:

ioslides_presentation: default

---

When you do the presentation, you may want to use the fullscreen mode, which can be turned on by the keyboard shortcut f (just press

the F key). The key W toggles the widescreen mode. If the slide size is too big or too small, you can zoom in/out the page. Normally you can do it by holding the Ctrl (or Command) key, then press Plus (+) or Minus (-).

There are a few options for the ioslides_presentation format you can use to tweak the appearance of the slides: incremental (yes/no) whether to show bullets incrementally logo an image that you want to use as the logo in the slides (it will be

displayed in the footer of each slide) css a custom CSS ﬁle

You can also customize each slide individually. For example, if you put a token {.build} after a second-level section heading, the elements on this page will be displayed incrementally as you proceed in the presentation, e.g.,

## A new slide {.build} First show this. Then show that. Finally show a funny GIF animation. ![](foo.png)

HTML5 slides are usually for presentation instead of printing purposes. However, you may also print the slides as PDFs from your Web browser. At the moment, we recommend you to use Google Chrome if you want to print the slides. You should expect the appearance of printed slides to differ from that of the displayed slides.

###### 14.3.6 Slidy Presentation

The rules of writing slides for Slidy are the same as ioslides. The function for Slidy presentation output in rmarkdown is slidy_presentation().

- Figure 14.10 shows one slide of the Slidy presentation created from the R Markdown example.


A few keyboard shortcuts are available, e.g., press C to see the table of contents, S to make the font smaller, and B to make the font bigger, etc.

![image 27](Dynamic Documents with R and knitr 2nd_images/imageFile27.png)

- FIGURE 14.10: One slide from the Slidy presentation generated from the R Markdown example: you can also click “Contents” at the bottom to show the table of contents.


Besides the incremental and css options we mentioned before, Slidy

has some additional features that may be useful, including the options: duration sets a countdown timer in the footer to remind you of the

time, e.g., if you have a 50-minute talk, you can set duration: 50 in YAML

footer a custom message in the footer, e.g., you can display the name

of your institute or copyright information To print Slidy slides, you can also use Google Chrome.

###### 14.3.7 Beamer Presentation

Beamer, introduced in Section 12.3.4 is a LATEX application, so you can build an Rnw ﬁle as a LATEX document with code chunks shown in Sec-

tion 12.3.4 and compile directly into the PDF format. Markdown is simpler and faster for all but veteran LATEX users, so we recommend trying it with the beamer_presentation format. If you need some of the more advanced Beamer or LATEX features, they can be added within Markdown as Pandoc supports LATEX code within Markdown.

Figure 14.11 shows two slides of the Beamer presentation created from the previous R Markdown example. All we did was change the YAML metadata to:

--title: "R Markdown v2 Demo" author:

- Li Lei

- Han Meimei date: "2015/01/01" output:

beamer_presentation: theme: AnnArbor bibliography: Rmd-v2.bib

---

If we were to write the slides in raw LATEX, the source document would be like this:

###### \documentclass{beamer} \usetheme{AnnArbor}

\title{R Markdown v2 Demo} \author{Li Lei \and Han Meimei} \date{2015/01/01}

###### \begin{document} \frame{\titlepage}

\begin{frame}{Start with a cool section} A bit \emph{introduction} here. You can use traditional \textbf{Markdown} syntax, such as \href{http://yihui.name/knitr}{links} and \texttt{code}. \end{frame} \begin{frame}{Followed by another section}

|R Markdown v2 Demo<br><br>Li Lei Han Meimei<br><br>2015/01/01<br><br>Li Lei, Han Meimei R Markdown v2 Demo 2015/01/01 1 / 13<br><br>|
|---|


|Pandoc extension: examples<br><br>We have some examples.<br><br>1 Think what is 0.3 + 0.4 - 0.7. Zero. Easy.<br>2 Now think what is 0.3 - 0.7 + 0.4. Still zero?<br><br><br>People are often surprised by (2).<br><br>Li Lei, Han Meimei R Markdown v2 Demo 2015/01/01 9 / 13<br><br>|
|---|


###### FIGURE 14.11: Two slides from the Beamer presentation created by RMarkdown: the title slide, and the slide that shows the Pandoc exten-sion of the example environment.

Of course you can write lists:

\begin{itemize} \item

apple \item

###### pear \item

banana

\end{itemize}

.... \end{document}

Compare that with the R Markdown source code in Section 14.3.1, and hopefully you see how much more code you would have to type when writing in raw LATEX than writing in Markdown.

Each new slide is a new section in Markdown, and the level of the section is determined by the highest level in the document hierarchy that is followed immediately by the slide content. In the following example, each ﬁrst-level section (#) is a new slide:

--output: beamer_presentation

--# One Section

- - content
- - content # Another Section ![](foo.png)

And in this example, each sub-section (##) is a new slide:

- --output: beamer_presentation


--# One Section

## One Sub-section

- content - content

# Another Section ## Another Sub-section ![](foo.png)

To display list items incrementally, you can use the incremental option just like what we can do for ioslides and Slidy presentations. Other options such as toc, highlight, fig_width, fig_height, fig_caption, includes, and template have been explained in previous sections.

There are many themes (including font themes and color themes) in Beamer. You can use them via the theme, fonttheme, and colortheme options. Figure 14.11 used the AnnArbor theme, and default font/color themes. If you use RStudio, you can choose these themes from the GUI, so you do not need to remember the many theme names.

###### 14.3.8 Other Formats

Besides the document and presentation formats, rmarkdown also has two special output formats: html_vignette() for HTML package vignettes (Section 15.4) and tufte_handout() for the Tufte handout (here Tufte refers to Edward R. Tufte).

The html_vignette() format is a wrapper of html_document(), with a special CSS theme; the ﬁle size of the HTML vignette produced by html_document() is too big because it contains the Twitter Bootstrap assets, the jQuery library, and highlight.js by default. The html_vignette() format has removed all these components, and uses a single lightweight CSS ﬁle. The option fig_retina has been set to 1 to further reduce the image ﬁle sizes. This format function is a good example of how to build your own format based on existing format functions, and its source code is very simple:

html_vignette <- function(fig_width = 3, fig_height = 3, dev = "png", css = NULL,

...) { if (is.null(css)) {

css <- system.file("rmarkdown", "templates",

"html_vignette", "resources", "vignette.css", package = "rmarkdown")

} html_document(fig_width = fig_width,

fig_height = fig_height, dev = dev, fig_retina = FALSE, css = css, theme = NULL, highlight = "pygments", ...)

}

The tufte_handout() format is a wrapper for the LATEX document class tufte-handout.cls. The most notable characteristics of the Tufte handout style are perhaps the use of sidenotes, and the well-designed typography. See Figure 14.12 for an example page. Its YAML metadata is this:

--title: "Tufte Handout" author: "John Smith" date: "August 13th, 2014" output: rmarkdown::tufte_handout

---

###### 14.4 Interactive Documents with Shiny

Shiny (Chang et al., 2015) is a Web application framework that makes it easy to create interactive apps using R. You can create a Web user interface (UI) using Shiny UI functions, e.g., text input boxes, drop-down lists, radio buttons, and sliders, etc. These UI elements can interact with R after you specify the server logic in R, e.g., after you click a button, what you expect R to do. If you are not familiar with Shiny, please check out the website http://shiny.rstudio.com to learn the basics about Shiny.

Because a Shiny app is basically an HTML page, and it happens that R Markdown can be rendered to HTML, too, it is possible to combine R Markdown and Shiny in one document. We call such documents “interactive documents,” since they contain interactive components from Shiny. Figure 14.13 shows a minimal example of an interactive document. Its source document is as follows:

|![image 28](Dynamic Documents with R and knitr 2nd_images/imageFile28.png)|
|---|


###### FIGURE 14.12: An example page using the Tufte handout style: youcan arrange elements into the side margin, such as footnotes, ﬁgures,equations, and so on.

![image 29](Dynamic Documents with R and knitr 2nd_images/imageFile29.png)

###### FIGURE 14.13: A simple interactive document using R Markdown andShiny: you can change the value of the slider, and the number of binsin the histogram will be automatically changed.

--title: "R Markdown v2 Demo" runtime: shiny output: html_document

--   {r} library(shiny) sliderInput("bins", "Number of bins:", min = 1, max = 50,

value = 30)

renderPlot({ x <- faithful[, 2] # Old Faithful Geyser data bins <- seq(min(x), max(x), length.out = input$bins + 1) # draw the histogram with the specified number of bins hist(x, breaks = bins, col =  darkgray , border =  white )

})

To turn a normal R Markdown document into an interactive document, you only need to add the option runtime: shiny in the YAML metadata. Then you can use functions in the shiny package. In the above example, we created a slider on the HTML page using sliderInput(), which is a UI function in shiny. The id of the slider is bins. Then we rendered a histogram using the renderPlot() function. The most important bit in this code chunk is input$bins, which is a variable value associated with the slider with the id bins. When we update the value of the slider, its value will be passed to the expression in renderPlot(), and the plot will be redrawn accordingly.

Instead of render(), interactive documents should be compiled by the run() function in rmarkdown. If you use RStudio, you will see that the label of Knit button on the toolbar becomes Run Document after you add runtime: shiny to an R Markdown document, and you can click the button to run the document.

Not all Shiny apps can be so simple as the one in Figure 14.13. When you have several UI elements, you may want to arrange them in a separate app instead of writing them out in code chunks linearly. The function shinyApp() in shiny allows you to build a full app by specifying all UI elements and the server logic in one function. Then you can either embed full apps using shinyApp() explicitly in R Markdown, or write your own function that returns a shinyApp() object, so that other people can easily use your app as well.

Static HTML documents can be uploaded to any website or emailed

when you want to share them. For interactive documents, there must be an active R session running behind them. One possible way to share interactive documents is to publish them to http://shinyapps.io, which is hosted by RStudio. If you do not want to publish to this website, you can set up your own Shiny Server: http://www.rstudio.com/products/

shiny/shiny-server/.

###### 14.5 Extending R Markdown v2

If none of the output format functions meet your need, you can extend them or write a completely new format. Before you do it, please make sure you have looked at all the possibilities in the existing output formats. Sometimes there is no need to invent anything new. For example, if all you want is to use a different LATEX document class, you may as well set the documentclass option in the YAML metadata, although you can certainly also write a new template with the desired document class. Take the Tufte handout as an example:

--title: "R Markdown v2 Demo" author: John Smith date: "2015/01/01" output: pdf_document documentclass: tufte-handout classoption: nohyper geometry: no

---

The above YAML metadata makes use of the existing pdf_document() format. Alternatively, you can prepare a template like: \documentclass{tufte-handout} $if(title)$ \title{$title$} $endif$ $if(author)$ \author{$for(author)$$author$$sep$ \and $endfor$} $endif$ $if(date)$ \date{$date$}

$endif$ \begin{document} $if(title)$ \maketitle $endif$ $body$ \end{document}

Then use the template option in pdf_document. There are a number of disadvantages of writing a custom template like that:

- • Pandoc’s default LATEX is much more ﬂexible (https://github.com/ jgm/pandoc-templates), which can also deal with the table of contents, the list of ﬁgures, and the abstract, etc.;
- • It requires more work to write a new template than to use existing options in YAML;
- • After you write a template, you will have to watch out for future changes in Pandoc, which may break your template, or you may miss some useful new features. By comparison, if you use Pandoc’s templates, you do not need to maintain them.


Then you may ask why we have the tufte_handout() format in rmarkdown after all. Actually what this new format does is more than just a LATEX template: it also deﬁnes a few knitr chunk options to produce fullwidth ﬁgures (fig.fullwidth = TRUE) and margin ﬁgures (fig.margin

= TRUE). Existing output formats do not provide these two different ﬁgure types.

###### 14.5.1 Templates

The ﬁrst type of rmarkdown extension is to deﬁne a new template. We have shown an example above for the Tufte handout, and also an example earlier in Section 14.3.1 for HTML document output.

The repository https://github.com/jgm/pandoc-templates contains all templates used by Pandoc, and you can also take a look at the custom templates in the rmarkdown source package at https:// github.com/rstudio/rmarkdown. If there are any template variables that you do not understand, you can check out the documentation at http://johnmacfarlane.net/pandoc/.

To share a template with other users, the easiest way is to put it in an R package under the inst/rmarkdown/templates/ directory. You can create a new directory, say, my_template, and put the template ﬁle under it. Your template may require certain dependencies, such as CSS/JavaScript ﬁles, or LATEX packages. They can be collected under a sub-directory skeleton/ under my_template. In the skeleton/ directory, you can also provide a sample Rmd ﬁle skeleton.Rmd. Finally, you can describe the template in a YAML ﬁle template.yaml under my_template with three YAML ﬁelds:

name the name of the template, e.g., “Journal of Statistical Software”; description a short description of the template, e.g., “This is a template

for JSS articles”;

create_dir yes or no, or true or false (to be explained soon);

Suppose you installed such an R package named myPackage, then you can create a new draft from the template using the draft() function:

rmarkdown::draft("my_article.Rmd", template = "my_template", package = "myPackage")

This function looks for the template my_template in myPackage, copies skeleton.Rmd as my_article.Rmd to the current working directory, and also copies the dependencies. The YAML option create_dir mentioned above determines whether to create a new directory for the draft my_article.Rmd.

RStudio has made this process even easier. From the menu File New File R Markdown, you can see all templates in all locally installed packages (Figure 14.14).

The rticles package (https://github.com/rstudio/rticles) is a

collection of templates for several LATEX document classes. You can use its templates to write papers in R Markdown for the Journal of Statistical Software, and The R Journal, etc.

###### 14.5.2 New Formats

The second type of rmarkdown extension is new output formats. The new format can be based on an existing output format, or a completely new format. The former is easy: you just deﬁne an R function that returns an output format object, with certain options modiﬁed from an existing output format function. As a minimal example, we create a function html_toc below, turning the default value of the toc argument from FALSE to TRUE:

![image 30](Dynamic Documents with R and knitr 2nd_images/imageFile30.png)

- FIGURE 14.14: Create a new R Markdown document from templates: you can select a template from the list.


html_toc <- function(toc = TRUE, ...) {

rmarkdown::html_document(toc = toc, ...) }

A new format function should be put in an R package (we still assume its name is myPackage), and then you can use it in YAML. Here are two examples:

--output: myPackage::html_toc

---

--output:

myPackage::html_toc: toc: no self_contained: no

---

![image 31](Dynamic Documents with R and knitr 2nd_images/imageFile31.png)

- FIGURE 14.15: Create an E-book from R Markdown: this ﬁgure shows the title page of the EPUB book in FBReader (a free E-book reader).


For the second example, what will be called when we render this Rmd ﬁle is: rmarkdown::render("foo.Rmd", myPackage::html_doc(toc = FALSE,

self_contained = FALSE)) # which is essentially render( foo.Rmd , # html_document(toc = FALSE, self_contained = FALSE))

As we explained in Section 14.3.1, the output format is a list of three types of options: knitr options, Pandoc options, and rmarkdown options. We customized the Pandoc toc in the above minimal example, and you can certainly customize more options in the output format function. There are a few helper functions output_format(), knitr_options(), and pandoc_options() in rmarkdown that you can use to compose the output format. See the repository https://github.com/jjallaire/ revealjs for an example of how to create a new format for reveal.js (an HTML5 presentation format). Below we show a minimal example of how to create an output for EPUB (an E-book format):

#  @importFrom rmarkdown output_format #  @importFrom rmarkdown knitr_options #  @importFrom rmarkdown pandoc_options

epub_book <- function(to = c("epub", "epub3")) { to <- match.arg(to) optk <- knitr_options() optp <- pandoc_options(to, ext = ".epub") output_format(knitr = optk, pandoc = optp)

}

Put this function in the package myPackage, and you will be able to create E-books from R Markdown. Here is a minimal R Markdown example (Figure 14.15):

--title: "R Markdown v2 Demo" author:

- - Li Lei
- - Han Meimei

date: "2015/01/01" output: myPackage::epub_book

- --# Start with a cool section


   {r} 1 + 1

The key in the format function epub_book() was to specify the argument to of pandoc_options() to be either epub or epub3. Pandoc supports a large number of document formats, and rmarkdown only included a small subset of them. You can build your own format function using the approach introduced above.

- 14.5.3 HTML Widgets We explained the includes option in the YAML metadata in Section


- 14.3.1. When you want to include JavaScript libraries in the HTML document output, you can use the includes option. There are two disadvantages of this approach:


- 1. It is not portable, in the sense that when you share the R Markdown document with other people, you should remember to copy the dependencies speciﬁed in the includes option; it is not convenient for other people to reuse your dependencies, either;
- 2. You have to write (sometimes a lot of) JavaScript code in R Markdown to call the JavaScript libraries, but not all R users are familiar with JavaScript, so they may not be able to work on the R Markdown document.


The idea of HTML widgets is to provide native R interfaces to JavaScript libraries, so that even those who do not understand JavaScript can still

use the libraries without worrying about the underlying dependencies or JavaScript syntax. When you draw a plot using a JavaScript library, all you need to do is call an R function in a code chunk.

The htmlwidgets package (Vaidyanathan et al., 2014) was designed for package developers to port JavaScript libraries into R easily. It is well-documented at http://www.htmlwidgets.org, and you can see several example packages on the website, too. We will not describe the technical details here, and we just show a quick example of what an HTML widget looks like. Here is a minimal R Markdown example (you need to install the DT package from https://github.com/rstudio/DT before trying this example):

--title: "R Markdown v2 Demo" author:

- - Li Lei
- - Han Meimei

date: "2015/01/01" output: html_document

- --Here is a table generated by the DataTables library.


   {r} DT::datatable(iris)

Figure 14.16 shows the output. The DT package is an interface to the JavaScript library DataTables (http://datatables.net). As you can see, the R Markdown source document is really simple, and you do not see the JavaScript ﬁles or any JavaScript code at all. You simply call the function datatable(), and your data frame will be displayed via DataTables. The hard work of passing data to the HTML page, parsing and rendering it has been done by the package authors, and users do not have to understand all the underlying technical details.

###### 14.6 Changes in R Markdown from v1 to v2

If you happen to have started using R Markdown when it was v1, here is a list of changes that you should be aware of when you transition from v1 to v2:

![image 32](Dynamic Documents with R and knitr 2nd_images/imageFile32.png)

###### FIGURE 14.16: A table created by the DataTables library in R Mark-down: you can order the columns, search in the table, and the full tablecan be displayed on multiple pages.

- • The knitr package is no longer loaded (strictly speaking, attached) by default in v2, which means the functions and objects in the knitr package are not available unless you explicitly load the package, e.g., via the command library(knitr); otherwise, you may get errors like “object ’opts_chunk’ not found”;
- • The chunk options fig.path (ﬁgure path) and cache.path (cache path) are modiﬁed in rmarkdown when rendering an Rmd ﬁle. In knitr, they are figure/ and cache/, respectively. Now in rmarkdown, they are foo_files/figure-format/ and foo_files/cache-format/, respectively, where foo is the base ﬁlename of the input Rmd ﬁle without the ﬁle extension, and format is the output format, e.g., tex or html;
- • The chunk option error was changed from TRUE to FALSE, and the implication is that R will stop by default, instead of showing the error messages in the R Markdown output document (see Section 6.2.4);
- • The chunk options fig.width, fig.height, and fig.retina may take different values, depending on the output format. You can either check the rmarkdown documentation of output format functions, or print str(knitr::opts_chunk$get()) in your R Markdown document to see the values of chunk options.


### 15

###### Applications

So far we have been introducing the usage of knitr with short examples for the sake of simplicity. In this chapter we use some concrete and complete examples to show how knitr works with real applications; we do not explain every single detail of these applications, and we only point out the critical parts in them.

###### 15.1 Homework

For homework applications, R Markdown might be the preferred document format to work with due to its simplicity, and homework is usually not targeted at publication. As mentioned before, RPubs (http: //rpubs.com) is a platform for sharing (HTML) reports generated from RStudio by knitr. There are many homework submissions, too.

Since a homework report is relatively simple, we may not need too many knitr features; some common features used in homework are: set the size of plots (fig.width and fig.height), hide the source code because the grader may not wish to read it (echo = FALSE), and enable cache for time-consuming computing jobs (cache = TRUE), etc. Other features that come by default such as tidy = TRUE and highlight = TRUE can help users who do not care about coding styles produce more readable code in the output document.

Now we show an example of Gibbs sampling. For the bivariate Normal distribution

σX2 ρσXσY ρσXσY σY2

- X
- Y ∼ N


- µX
- µY


,

(15.1)

we know the conditional distributions

σY σX

ρ(x − µX), (1 − ρ2)σY2

Y|X = x ∼ N µY +

- σX

- σY


ρ(y − µY), (1 − ρ2)σX2 (15.2)

X|Y = y ∼ N µX +

213

so we can use the Gibbs sampling to generate random numbers from the joint Normal distribution. First we initialize x(0) and y(0), then repeatedly generate x(k) ∼ f(x|y(k−1)) and y(k) ∼ f(y|x(k)). The R code below is a translation of 15.2:

rbinormal <- function(n, mu1, mu2, sigma1, sigma2, rho) { # initialize

- x <- rnorm(1, mu1, sigma1)
- y <- rnorm(1, mu2, sigma2) xy <- matrix(nrow = n, ncol = 2, dimnames = list(NULL,


c("X", "Y"))) # sample from conditional distributions for (i in 1:n) {

- x <- rnorm(1, mu1 + sigma1/sigma2 * rho * (y - mu2),

- sqrt(1 - rho^2) * sigma1)

y <- rnorm(1, mu2 + sigma2/sigma1 * rho * (x - mu1),

- sqrt(1 - rho^2) * sigma2)




xy[i, ] <- c(x, y)

} xy

}

Figure 15.1 shows the ﬁrst 20 steps of Gibbs sampling for the bivariate Normal distribution with µX = 0, σX = 2, µY = 1, σY = 3, ρ = 0.7. set.seed(123) n <- 20 z <- rbinormal(n, mu1 = 0, mu2 = 1, sigma1 = 2, sigma2 = 3,

rho = 0.7) plot(z, pch = 19) arrows(z[-n, 1], z[-n, 2], z[-1, 1], z[-1, 2], length = 0.15,

col = "gray40")

And we can draw some samples as well:

z <- rbinormal(5000, 0, 1, 2, 3, 0.7) smoothScatter(z, nbin = 64) points(0, 1, col = "white", pch = 19) # theoretical mean

Figure 15.2 shows 5,000 samples from this distribution, and we can calculate the sample means, standard deviations, and the correlation, which should be close to the corresponding theoretical values:

6

4

2

Y

0

- -6
- -4
- -2


-3 -2 -1 0 1 2 3

X

- FIGURE 15.1: Trace of Gibbs sampling for a bivariate Normal distribution: the arrows show the ﬁrst 20 steps of Gibbs sampling.

-6 -4 -2 0 2 4 6

-5

0

5

10

X

Y

- FIGURE 15.2: 5000 points from Gibbs sampling: the smoothed scatterplot shows the density of the 2D distribution.


apply(z, 2, mean) # sample mean

## X Y ## 0.001287 0.971010

apply(z, 2, sd) # sample sd

## X Y ## 1.973 2.971

cor(z) # sample correlation

## X Y ## X 1.0000 0.6948 ## Y 0.6948 1.0000

In this small application, we used cache (although this particular example is not too slow) and TikZ graphics. We adjusted the plot sizes (5 × 3 for Figure 15.1 and 5 × 4 for Figure 15.2). Note the narratives and code chunks are interwoven, and the reader can learn the theory, see the computing, and verify the results in the same report. Everything is transparent, and it will be easy to ﬁnd out errors. Sometimes the computer code we write may not really reﬂect what we said in theory, and it will be hard to ﬁnd out such errors if we separate computing from reporting.

In terms of data, code and software sharing, we cannot yet rely on goodwill and self discipline when it comes to sharing publication material and making studies fully reproducible.

Huang and Gottardo (2013) Comparability and reproducibility of biomedical data

People have been proposing sharing data, code, and software in data analysis for the sake of reproducible research, e.g., Huang and Gottardo (2013). We believe that more efforts in education should be an important step, and we can start with reproducible homework.

###### 15.2 Serve Dynamic Documents

The servr package (Xie, 2015c) provides some simple HTTP server functions to serve ﬁles under a given directory based on the httpuv package. To some degree, this package is like python -m SimpleHTTPServer or python -m http.server if you are familiar with Python. Originally it was designed to serve static ﬁles under a directory, and the main function was httd():

servr::httd("./")

If you run the above function in the R console, R will launch your Web browser to show a list of ﬁles under the current working directory (./), or show index.html if this ﬁle exists. You can click the links on the ﬁles to view their content.

Later servr was extended based on knitr and rmarkdown, so it can also serve dynamic R Markdown documents. There are functions jekyll(), rmdv1(), and rmdv2() in this package to serve HTML ﬁles generated from R Markdown documents (via knitr or rmarkdown). R Markdown documents can be automatically recompiled when their HTML output ﬁles are older than the corresponding source ﬁles, and HTML pages in the Web browser can be automatically refreshed accordingly, so you can focus on writing R Markdown documents, and results will be updated on the ﬂy in the Web browser. This saves you two steps: click the Knit HTML button, and refresh the Web browser. Both steps can be distracting when you write a report. With servr, all you need to do is write the R Markdown document after you launch a server.

This is even more useful when you write R Markdown documents in the RStudio IDE, because servr has set the Web browser to be the RStudio Viewer by default when it detects the RStudio IDE, and you can put the source document and its output side by side like the layout in Figure 15.3. It is completely ﬁne if you do not use RStudio — the automatic compilation and refreshing also work if you use other editors and Web browsers.

The functions rmdv1() and rmdv2() correspond to R Markdown v1 and v2, respectively. After you call servr::rmdv1() or servr::rmdv2() in the R console, you can click the HTML ﬁle foo.html if it has its source document foo.Rmd, and view the HTML output. Then whenever you edit foo.Rmd and save it, servr will automatically recompile it and refresh the HTML output page.

The function jekyll() is like rmdv1() and rmdv2(), but is tailored for Jekyll websites. We have brieﬂy introduced Jekyll in Section 13.4. It

![image 33](Dynamic Documents with R and knitr 2nd_images/imageFile33.png)

- FIGURE 15.3: The layout of an R Markdown document (top-left panel) and its output in the RStudio Viewer (right panel): we typed a servr function in the R console (bottom-left), and the output of the R Markdown is showed in the RStudio Viewer. This ﬁgure is only for illustration purposes; see https://github.com/yihui/servr for the original image if you want to read the text in it.


is tedious to compile R Markdown posts or pages to Markdown again and again, and that is why jekyll() can be useful. Once you call the function servr::jekyll() in the root directory of a Jekyll website, you will get a preview of the website in your Web browser. Besides, as you edit and save your blog post, the Web browser will refresh the page to show the updated output. The knitr-jekyll repository (https: //github.com/yihui/knitr-jekyll) is an example of serving Jekyll websites using servr.

Later we will introduce package vignettes in Section 15.4, and the function vign() in servr can be used to serve HTML vignettes while we develop an R package. Its advantage is that it does not preserve the HTML output ﬁle in the source package when serving the vignette, which makes the source package clean.

For those who are curious about the technical details, the implementation is based on WebSockets. When servr shows an HTML page, it also injects a piece of JavaScript code in it to set up a WebSocket connection to talk to R periodically (e.g., on one-second basis). Every time R receives a request from the WebSocket, it will compare the timestamps of Rmd ﬁles with their output HTML ﬁles. If an Rmd ﬁle is newer than its HTML output, servr will call knitr or rmarkdown to recompile the Rmd ﬁle to HTML, then send a message back to the WebSocket.

all: example.html %.html: %.Rmd

Rscript -e "rmarkdown::render( $^ )"

- FIGURE 15.4: A Makeﬁle example for the function make() in servr: the HTML ﬁle to be generated is speciﬁed in the target all, and a rule is speciﬁed on how to generate an HTML ﬁle from an Rmd ﬁle via rmarkdown.


When the WebSocket receives this message, it calls location.reload() in JavaScript to refresh the page.

A critical step in this process is to check if we need to recompile any Rmd ﬁles. This is a task that GNU Make (http://www.gnu.org/ software/make/) is good at, so servr also provided a function make() so that you can provide your own Makeﬁle to rebuild Rmd ﬁles when necessary. Figure 15.4 is an example Makeﬁle for the make() function.

By default, a server function will block the current R session, which can be a problem if you want to continue working in the same R session. To solve this problem, you can use the argument daemon = TRUE for the server function, e.g., httd(daemon = TRUE), or rmdv2(daemon = TRUE). This tells servr to launch a daemonized server that will not block the current R session.

###### 15.3 Website and Blogging

We introduce a few websites and blogs built upon knitr in this section, and the Web pages are created from either R Markdown or R HTML.

###### 15.3.1 Vistat and Rcpp Gallery

Vistat (http://vis.supstat.com) is a website based on R Markdown and Jekyll (Section 13.4). It aims to provide a gallery of reproducible statistical graphics. The repository for the website is publicly available on Github: https://github.com/supstat/vistat.

The core of this repository is the R script ./_bin/knit, which sets some global chunk options and compiles Rmd documents to Markdown output. Math equations are rendered by MathJax, animations

are supported through the SciAnimator library (Section 7.3.1), and we can also create Web graphics via the D3 library.

After knitr has compiled Rmd source ﬁles to Markdown ﬁles, Jekyll can compile Markdown to HTML, which gives us a website.

The Rcpp Gallery (http://gallery.rcpp.org) is a website for Rcpp (Eddelbuettel et al., 2015) articles and examples, and it is also built on R Markdown; in particular, it uses knitr’s Rcpp engine (Section 11.2.1).

###### 15.3.2 UCLA R Tutorial

The UCLA Statistical Consulting Group has maintained software tutorials for several statistical packages for many years, and one of them is dedicated to R: http://www.ats.ucla.edu/stat/r/. Before 2012, this website was built by cut-and-paste. The results were generated in R and copied into the HTML pages. After knitr was released in 2012, one of the Web administrators, Joshua Wiley, decided to rewrite the R tutorial pages with knitr instead of using the R HTML format. Now it is much easier to maintain the Web pages, and the R output also has much better reproducibility. After R is updated or any dataset is changed, the whole website can be rebuilt automatically by compiling all source documents again.

###### 15.3.3 The cda and RHadoop Wiki

Github has an integrated Wiki system for each repository. We can write wiki pages in a variety of formats, such as Markdown and reStructuredText, etc. Each page is essentially a ﬁle, and the wiki is essentially a Git repository; therefore we can write Rmd ﬁles and compile them to Markdown ﬁles, and push to Github through Git.

The cda package (Auguie, 2013) used the above approach to build its wiki site on Github: https://github.com/baptiste/cda/wiki. We can ﬁnd the Rmd source ﬁles under the wiki directory of the package.

The RHadoop project has a similar wiki at https://github.com/ RevolutionAnalytics/RHadoop/wiki.

###### 15.3.4 The ggbio Package

The ggbio package (Yin et al., 2012) is an R implementation for extending the Grammar of Graphics for genomic data based on the ggplot2 package. It has a website, http://tengfei.github.com/ggbio/, on which we can ﬁnd its documentation. The function knit_rd() (Section 12.4.8) was used to compile its R documentation pages to HTML, so we

can directly see the output of the examples. Once this package has been installed, it only needs one line of code to get the HTML pages:

knitr::knit_rd("ggbio")

Then we can publish the HTML ﬁles to Github, and we do not need to do anything with the images because they are base64 encoded in the ﬁles.

By the way, the ggbio package also has a PDF vignette written with knitr, which can be found on the website or with the command: vignette("ggbio", package = "ggbio")

###### 15.3.5 Geospatial Data in R and Beyond

Barry Rowlingson gave a tutorial workshop on geospatial data analysis in R at the useR! 2012 conference, and here is the corresponding website: http://www.maths.lancs.ac.uk/~rowlings/Teaching/ UseR2012/. The website was created from R HTML ﬁles and has a nice style from Twitter Bootstrap (a popular CSS framework). The advantage of using R HTML over R Markdown is that we have full control of the style; this website is a good example of arranging R code chunks and output in div elements with custom CSS styles.

###### 15.4 Package Vignettes

As discussed by Gentleman and Temple Lang (2004), R packages have the great potential of building and disseminating reproducible reports, besides their obvious functionality of providing computing routines. Speciﬁcally, R package vignettes can be an ideal format for writing reproducible reports, with other components of the package providing the infrastructure such as functions, unit tests, and datasets. An R package vignette is just like a paper, and the output is dynamically compiled from its source document during the package building process, i.e., R CMD build.

For R under the version 3.0.0, it uses Sweave to build package vignettes. Due to the limitations of Sweave (Section 16.1) and the barrier of LATEX, R package vignettes were not widely used before R 3.0.0. BioConductor is an exception, though, because vignettes are mandatory for packages on BioConductor.

It has become much more natural and easy to compile package vignettes since R 3.0.0, thanks to Henrik Bengtsson, Duncan Murdoch, and R core. Now there are more than 500 package vignettes compiled from knitr in about 300 packages on CRAN (https://gist.github. com/yihui/7698648). In the next section, we introduce knitr vignette engines, and then we show a few examples. Sections 15.4.3 and 15.4.4 are only for those who are interested in older versions of R, and we no longer recommend that you use the tricks mentioned in these two sections.

###### 15.4.1 Vignette Metadata and Engines

To use knitr to build vignettes, we only need to follow these simple steps:

- • specify a vignette engine, such as %\VignetteEngine{knitr::knitr}, in the vignette source document (e.g., an Rnw or Rmd ﬁle)
- • add a ﬁeld VignetteBuilder: knitr in the package DESCRIPTION ﬁle
- • add knitr to the Suggests ﬁeld in DESCRIPTION


Then we can write vignettes using the knitr syntax (e.g., < <> >= or    {r} for code chunks). Remember vignettes are put under the vignettes/ directory of the package root directory.

According to the R manual “Writing R Extensions,” we also have to write the title of the vignette in \VignetteIndexEntry{}. There are a few other optional metadata speciﬁcations such as \VignetteKeyword{}. See Figure 15.5 for an example of the vignette metadata (title and vignette engine) for an R Markdown v2 vignette in knitr. After we build the package, the vignettes will be listed in an HTML index page.

The knitr package has several PDF and HTML vignettes compiled in this way, and we can view them by running: browseVignettes(package = "knitr") # or view specific vignettes if you know their filenames vignette("knitr-intro", package = "knitr") vignette("knitr-refcard", package = "knitr")

The vignette engine knitr::knitr is only one of the possible engines in knitr. To see all of them, you can use the function vignetteEngine() in the tools package:

--title: "Not An Introduction to knitr" author: "Yihui Xie" date: " r Sys.Date() " bibliography:

- ../inst/examples/knitr-packages.bib

- ../inst/examples/knitr-manual.bib

vignette: > %\VignetteEngine{knitr::rmarkdown} %\VignetteIndexEntry{Not an Introduction to knitr}

output: knitr:::html_vignette

---

- FIGURE 15.5: The metadata of a knitr vignette: this is extracted from the knitr vignette, and you can ﬁnd it from system.file(’doc’, ’knitr-intro.Rmd’, package=’knitr’).


library(knitr) sort(names(tools::vignetteEngine(package = "knitr")))

## [1] "knitr::docco_classic" ## [2] "knitr::docco_classic_notangle" ## [3] "knitr::docco_linear" ## [4] "knitr::docco_linear_notangle" ## [5] "knitr::knitr" ## [6] "knitr::knitr_notangle" ## [7] "knitr::rmarkdown" ## [8] "knitr::rmarkdown_notangle"

The engines with the sufﬁx _notangle have the same weave functions as those without the sufﬁx, but have disabled the tangle function, meaning that there will not be R scripts generated from vignettes during R CMD build or R CMD check. Sometimes we may not want to tangle R scripts from vignettes, because it is redundant for R CMD check to run the same code again after the code has been executed in weave, and currently the inline R code expressions are not included in the tangle output, which can also cause problems.

Please note the :: operator has no special meaning in a vignette engine. It can be misleading because :: is an operator in base R that fetches an exported object from a package, e.g., stats::lm. However, in the vignette engine notation, :: is nothing but a delimiter that separates the package name from the engine name, so knitr::rmarkdown does

not mean rmarkdown is a function in knitr, but only one of the vignette engines in knitr.

When you use the rmarkdown vignette engine, you are free to choose the output format, as long as the ﬁlename extension is .html or .pdf, because R only recognizes these two types of vignette output at the moment. When the output format is HTML, it can be an HTML document, or any of the HTML5 presentations (e.g., ioslides or Slidy). When it is PDF, it can be either a PDF document or Beamer slides.

###### 15.4.2 Vignette Examples

We have put together a list of vignettes from current CRAN packages using the knitr vignette engines at https://gist.github.com/yihui/ 7698648, and you can learn from these examples.

The ggplot2 transition guide by Murphy (2012) is a great example of an R package vignette, although it is not shipped with the ggplot2 package. This guide was intended to announce new features and explain changes in ggplot2 0.9.0, which may affect users of older versions.

One nice feature of this guide is that we can compile the Rnw document to either a color or a black/white version, which is controlled by a global variable bw_version; if it is TRUE, a black and white version will be produced. This is achieved by setting the chunk options eval = bw_version and echo = bw_version for the chunks that produce black/white plots, and in ggplot2 this means theme_bw() and gray scales such as scale_ﬁll_gray(). When bw_version is FALSE, these chunks will be hidden from the output (the source code is neither evaluated nor echoed). Similarly, there are some other chunks that have the options eval = !bw_version and echo = !bw_version, and these chunks produce color plots. In all, we can control if the PDF output is color or black/white by a single variable, which is very convenient (recall Section 5.1.1). Figure 15.6 is a sample page of the transition guide from the color version.

The corrplot package (Wei, 2013) has an example of HTML vignettes. You can ﬁnd the source document of its vignette on Github at https:// github.com/taiyun/corrplot/tree/master/vignettes. Obviously, it is an Rmd document (Section 5.2.1). Note it uses R Markdown v1. Open it with a text editor (e.g., RStudio) and we will see R code chunks in it. We can view the HTML vignette compiled from it in the Web browser by running:

help(package = "corrplot", help_type = "html")

This shows the HTML index page of the corrplot documentation,

cyl 4 6 8

cyl 4 6 8

35

vs: 0 vs: 1

35

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |


30

30

25

mpg

25

mpg

20

20

15

15

10

10

0 1

4 6 8 4 6 8

vs

No. cylinders

###### 3.4 geom_violin()

This function generates violin plots in ggplot2, a way to plot one or more continuous density estimates that is particularly useful when comparing multiple groups. A violin plot is a combination of a box plot and a kernel density estimate, the latter of which is rotated to run alongside the box plot symmetrically on each side. The examples below come from the function’s help page.

In geom_violin(), violins are automatically dodged when any aesthetic is a factor. By default, the maximum width is scaled to be proportional to the sample size. In the plot on the far right below, the bandwidth of the kernel density estimator is reduced from the default 1, which makes for a less smooth density estimate and hence a less smooth violin plot.

###### p <- ggplot(mtcars, aes(factor(cyl), mpg)) p + geom_violin() # default scale is "count" p + geom_violin(aes(fill = factor(cyl), colour = factor(cyl))) p + geom_violin(adjust = 0.5)

|10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>4 6 8<br><br>factor(cyl)<br><br>mpg|
|---|


|10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>4 6 8<br><br>factor(cyl)<br><br>mpg<br><br>factor(cyl) 4 6 8<br><br>| |
|---|
| |
| |
|
|---|


|10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>35<br><br>4 6 8<br><br>factor(cyl)<br><br>mpg|
|---|


mpg

mpg

The next set of plots simply play around with a few extra features. The plot on the left adds a strip plot to the violin for each group. The central plot adds ﬁll color and alpha transparency to the violins and is augmented with boxplots. The plot on the far right adds a dot plot around

19

- FIGURE 15.6: A sample page of the ggplot2 transition guide: introducing the new geom added to ggplot2 0.9.0 — geom_violin().


PDFS= foo.pdf bar.pdf all: $(PDFS) clean:

rm -f *.tex *.bbl *.blg *.aux *.out *.log %.pdf: %.Rnw

$(R_HOME)/bin/Rscript -e "knitr::knit2pdf( $*.Rnw )"

- FIGURE 15.7: The Makeﬁle to compile PDF vignettes using knitr: use knit2pdf() to compile Rnw documents to PDF.


and we can see the link to the vignette “Overview of user guides and package vignettes.” Since corrplot is a package for visualizing correlation matrices, it has many graphical examples, which are shown in its HTML vignette.

The source package of knitr contains a mixture of PDF and HTML vignettes, all of which are listed in the HTML help page of this package.

The sampSurf package (Gove, 2013) also has a nice HTML vignette at http://sampsurf.r-forge.r-project.org, which was created from an R HTML source document and even contains some 3D plots produced by the rgl package.

###### 15.4.3 PDF Vignette

If we want to build vignettes with knitr for R <= 3.0.0, we have to use some tricks. One way to do this is through a Makeﬁle (http://www. gnu.org/software/make/), which will be used by R CMD build when building vignettes. In this Makeﬁle, we can set our rules to create the PDF ﬁle using a custom tool like knitr.

The Makeﬁle is under the vignettes/ directory in the source package. When R compiles vignettes, it calls Sweave() ﬁrst; if there is a Makeﬁle, the make command will be run on it. In the Makeﬁle, we also have access to R, so it is possible to call knitr via command line to compile vignettes. Figure 15.7 shows a sample of the Makeﬁle to be used to compile vignettes with knitr. The key is to run knitr::knit2pdf() on the Rnw ﬁles; we put all PDF ﬁles to be generated in the variable PDFS.

Obviously, the disadvantage of this approach is that all Rnw documents have to be compiled by Sweave before any further processing.

HTMLS= foo.html bar.html all: $(HTMLS) clean:

rm -rf figure/ *.md %.html: %.Rmd

$(R_HOME)/bin/Rscript -e "knitr::knit2html( $*.Rmd )"

- FIGURE 15.8: The Makeﬁle to compile HTML vignettes: use knit2html() to compile Rmd documents to HTML.


Besides, the new approach in R >= 3.0.0 does not require the make utility to be installed.

###### 15.4.4 HTML Vignette

Similarly, we can create package vignettes in the HTML format from R Markdown documents. Again, the HTML vignettes had to be compiled by a Makeﬁle before R 3.0.0. Figure 15.8 shows the source of a sample Makeﬁle for building HTML vignettes, where the function knit2html() was called. Note make clean will remove the ﬁgure/ directory, which is due to the fact that images generated by knitr will be base64 encoded in the HTML output, so the image ﬁles are no longer needed.

###### 15.5 Books

We can also write books with knitr. At the time of writing this book, at least one book has been published (Lebanon, 2012), and the book Regression Modeling Strategies (Harrell, 2001) is under revision for a new edition, which is based on knitr.

###### 15.5.1 This Book

In the spirit of “eating one’s own dog food” (see Wikipedia if this is unclear), this book was written with knitr in LYX (see Section 4.2). The

whole book is in one LYX ﬁle, although it is entirely possible to split chapters into individual ﬁles.

A few chunk options were set globally in the very beginning of the document, such as cache = TRUE (for speed), dev = ’tikz’ (for style of graphics), and fig.align = ’center’ (for alignment of plots). We also set options(formatR.arrow = TRUE) (see the formatR package), because the author’s preference of the assignment operator is = instead of <-, but <- is more commonly used by R users; this option allows the equal signs to be replaced by the left arrows automatically wherever applicable, although all I typed are actually equal signs.

We have a few chunk hooks (Chapter 10) in this book for various purposes. For example, there is a par hook that sets the graphical parameters to this:

par(mar = c(4, 4, 0.1, 0.1), cex.lab = 0.95, cex.axis = 0.9,

mgp = c(2, 0.7, 0), tcl = -0.3, las = 1)

So when we want to use this set of parameters, we just add a chunk option par = TRUE instead of having to type it again and again.

Although we see the code chunks and the plots are separate in this book, that is not true in the source document: the code chunks are actually inside the figure environments, but we used the document hook hook_movecode() to move code chunks out of the ﬁgure environments eventually.

Because we have to show chunk headers occasionally for pedagogical purposes, we have a chunk hook named append to add < <> >= and @ to the chunk output:

knit_hooks$get("append")

## function(before, options, envir) { ## txt = options$append[[ifelse(before, 1, 2)]] ## txt = c("\\begin{alltt}", txt, "\\end{alltt}") ## paste(txt, collapse = "") ## }

Basically this hook enables us to write additional character strings before and/or after a chunk; e.g., we can use the chunk option append = list(’< <A> >=’, ’@’) to add the syntax information to the chunk output. We need to use this hook because we cannot write the chunk headers directly in the source document, otherwise they will be parsed and disappear in the ﬁnal output.

There is an output hook that modiﬁes the default plot hook function

by adding a frame box to a plot, and it was used in Figure 10.3 and Figure 10.4.

The bibliography database of all R packages is dynamically written by the write_bib() function as introduced in Section 12.4.1, so it is guaranteed that the version information is up to date (at least before the manuscript was submitted to the publisher).

###### 15.5.2 The Analysis of Data

Another notable example is the book The Analysis of Data by Lebanon (2012); the most notable feature of this book is that it has the double PDF/HTML versions. The HTML version is freely available at http:// theanalysisofdata.com. Both versions are produced from essentially the same set of source documents. For the HTML version, there are additional settings, for example, the typesetting of math equations is done by the MathJax library, so it has to be included in the head section of the HTML source.

###### 15.5.3 The Statistical Sleuth in R

The Statistical Sleuth (Ramsey and Schafer, 2002) is an excellent text in statistics, and one feature of this book is that it has a large number of datasets. The book itself was not written with knitr, but some other authors (Horton et al., 2012) have created a website (http://www.math. smith.edu/~nhorton/sleuth/) in which they re-did a lot of the data analysis examples in the book in R. You can check out both the PDF documents and the Rnw source ﬁles on the website.

###### 15.5.4 Text Analysis with R for Students of Literature

The book Text Analysis with R for Students of Literature by Jockers (2014) was written using LATEX and knitr. The most amazing fact about this book is perhaps that its author taught himself LATEX before he started putting together this book in LATEX, and ﬁnished the book draft in just a couple of months. The book is an introduction to computational text analysis, and has a lot of short examples. It would be extremely tedious if the author had to run each example and copy the output to the LATEX manuscript by hand.

###### 15.6 Literate Programming for R Packages

Although we have introduced Literate Programming (LP) in the beginning of this book, we do not actually use the knitr package for programming purposes. Most of the time we use knitr for data analysis and reporting purposes instead. The original LP paradigm is about both weaving and tangling: we may weave a source document to software documentation, or tangle the program code to execute it. Apparently, we do not really have to tangle the program code for execution purposes when using knitr, because code execution occurs right in the process of weaving.

Interestingly, the most common application of Knuth’s original LP paradigm seems to be documenting software (using a special form of comments) for users instead of “programming” for package authors. In other words, we use LP to document the usage of software, instead of documenting the source code. See Doxygen (van Heesch, 2008), Javadoc (http://en.wikipedia.org/wiki/Javadoc), and roxygen2 (Wickham et al., 2015) for examples. There exists one exception, though, in the LATEX world. Some LATEX package authors write both LATEX code and documentation in a single document, and weave it into a PDF document that contains both the source code and documentation. This is not entirely surprising, considering Knuth’s original implementation of LP using TEX and Pascal. There is a small number of R packages using LP as well, such as Terry Therneau’s survival and coxme packages.

LP does not seem to be a popular approach to programming, but it is still an interesting idea, and can be useful especially when it is applied to your own favorite language. It may be boring for some people to read LATEX source code, but reading R source code can be more pleasant. Objective opinions aside, we believe LP has at least two advantages:

- 1. You can write much more extensive and richer documentation than you normally could do with comments. In general, comments in code are (or should be) brief and limited to plain text. Normally you will not write ﬁve paragraphs of comments to explain a few lines of code, and you cannot write readable math expressions or embed a video in comments.
- 2. You can label code chunks and reference/reuse them using the labels, which allows you to compose your program ﬂexibly using different pieces of code chunks. For example, you can deﬁne and explain a code chunk later in the document, but insert it in a previous code chunk using its label. This feature has been emphasized by Knuth, but it is not widely


adopted for some reason. Perhaps most people are more comfortable with designing a big program by smaller units like functions instead of code chunks, which is actually a good idea.

In fact, we can apply LP to developing R packages. There are multiple ways to achieve the goal, and we only introduce one here, using the following tools:

- 1. The purl() function in knitr, which makes it possible to extract program code from a source document;
- 2. Package vignettes, which can contain both program code and documentation;
- 3. GNU Make, which allows us to deﬁne when and how to generate an output ﬁle from a source ﬁle.


The rlp package (https://github.com/yihui/rlp) is an example of writing an R package using LP techniques. You can ﬁnd details in this repository, and the basic idea of the implementation is:

- 1. Instead of writing R source code under the R/ directory of the package, we can write the code in package vignettes (R Markdown) under the vignettes/ directory;
- 2. Use a Makeﬁle to deﬁne how to generate R scripts R/*.R from vignettes vignettes/*.Rmd;
- 3. Run make to generate R scripts to R/ and R CMD build to build the package.


These steps can be made easy by using the RStudio IDE, and we can actually just click a button to do the these steps. The implementation details are too technical and speciﬁc for this book, and we will leave it to the readers to go through the documentation of this package.

### 16

###### Other Tools

Besides knitr, there is a large number of other tools for dynamic documents. Some are R packages, and others are tools in other languages such as Python and awk. We give a brief overview of these tools with comparisons to knitr in this chapter, and we especially explain the differences between Sweave and knitr for Sweave users.

###### 16.1 Sweave

The knitr package was largely motivated by Sweave (Leisch, 2002), which has been a longstanding prominent tool for dynamic documents in R, and is a part of base R (in the utils package as the Sweave() function). Sweave primarily deals with Rnw documents, although it also has a modular design that allows it to be extended to other document formats. A number of extensions based on Sweave exist on CRAN, and we will introduce them in the next section.

There are two ways to run Sweave. We can call it in an interactive R session (you do not need to load the utils package):

Sweave("your_file.Rnw") # gives you your_file.tex

In addition, we can use the command line, too: R CMD Sweave your_file.Rnw

Since Sweave is part of base R, its development has almost plateaued in recent years. Another major problem is that its modular design is not modular enough, so its extensions may become incompatible as Sweave gets updated in base R. As far as we know, a few R packages based on Sweave copied a large amount of core code from Sweave, and are no longer synchronized with the development of Sweave.

A lot of knitr’s chunk options were borrowed from Sweave, such 233

as eval, echo, results and so on, but the design is different, so there are several differences between them. Before version 1.0, knitr tried to be compatible with Sweave — knitr was able to compile Sweave documents because of some internal functions to ﬁx the differences automatically. The compatibility has been dropped since v1.0, with a conversion function Sweave2knitr() provided to convert Sweave documents to knitr manually. Below is an example of converting the Rnw document in the utils package and showing the differences after conversion (< shows the original document, and > shows the converted ﬁle):

testfile <- system.file("Sweave", "Sweave-test-1.Rnw",

package = "utils") outfile <- tempfile(fileext = ".Rnw") Sweave2knitr(testfile, output = outfile) # capitalizing true/false to TRUE/FALSE: # * fig=true # removing the unnecessary option fig=TRUE: # * fig=TRUE # * fig=TRUE # quoting the results option: # * results=hide # removing options ’print’, ’term’, ’prefix’: # * print=TRUE # * echo=TRUE,print=TRUE # capitalizing true/false to TRUE/FALSE: # * echo=true # changing \SweaveOpts{} to opts_chunk$set(): # * \SweaveOpts{echo=FALSE} # * \SweaveOpts{echo=true} # removing extra lines (#n shows line numbers): # * (#69) @ cat(system(sprintf("diff %s %s", shQuote(testfile),

shQuote(outfile)), intern = TRUE), sep = "\n")

# 7c7,14 # < \SweaveOpts{echo=FALSE} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=FALSE

# > ) # > @ # > # 15c22 # < <<print=TRUE>>= # --# > <<>>= # 17c24 # < <<results=hide>>= # --# > <<results= hide >>= # 22c29 # < <<echo=TRUE,print=TRUE>>= # --# > <<echo=TRUE>>= # 43c50,57 # < \SweaveOpts{echo=true} # --# > # > <<include=FALSE>>= # > library(knitr) # > opts_chunk$set( # > echo=TRUE # > ) # > @ # > # 53c67 # < <<fig=TRUE>>= # --# > <<>>= # 63c77 # < <<fig=true>>= # --# > <<>>= # 69d82 # < @

###### 16.1.1 Syntax

By default, knitr uses a new type of syntax to parse chunk options, which is similar to R function arguments. This gives us much more

power than the traditional Sweave syntax. We can use arbitrary objects in chunk options and make use of the full power of R.

Sweave treats chunk options as character strings and parses them by splitting the options by commas, whereas knitr uses the R syntax: if the option takes a character value, we have to quote it just like we do in R, e.g., results = ’hide’ (in Sweave we write results = hide). See Section 12.1.3 for an example of doing computing directly in chunk options. Below is another example, which shows how ﬂexible the new syntax is (we can dynamically create a ﬁgure caption):

<<cap, fig.cap=paste( The P-value is , t.test(x)$p.value)>>= x <- rnorm(100) boxplot(x) @

The other minor difference in syntax is that knitr does not recognize @ as the beginning of text chunks unless there is a chunk header before it. For example, knitr will keep the ﬁrst @ in the example below but Sweave will remove it:

text @ <<A>>= 1 + 1 @

Sweave2knitr() can ﬁx this problem automatically.

###### 16.1.2 Options

Some options of Sweave were dropped in knitr and some were changed, including:

concordance was changed mainly to support RStudio; if the package option opts_knit$get(’concordance’) is TRUE, a ﬁle named inputconcordance.tex will be written with output line numbers mapped to input line numbers; note the implementation is less accurate than Sweave

keep.source was merged into a more ﬂexible option tidy

print was dropped: whether an R expression is going to be printed is consistent with your experience of using R (e.g., x <- 1 will not be printed, while 1:10 will; just imagine you are typing the commands in an R console); if you really want the output of an expression to be invisible, you may use the function invisible()

term was dropped (think term = TRUE) preﬁx was dropped (think prefix = TRUE) preﬁx.string was renamed fig.path and it is always used for ﬁgure

ﬁlenames

eps, pdf and all logical options for graphics devices were dropped: use the new option dev instead, which is similar to grdevice in Sweave but has more than 20 predeﬁned graphical devices; see Chapter 7

ﬁg was dropped; now use fig.keep: fig.keep = ’high’ in knitr is equivalent to fig = TRUE and fig.keep = ’none’ is the same as fig

= FALSE in Sweave

width, height were renamed fig.width and fig.height, respectively Meanwhile, \SweaveOpts{} and \SweaveInput{} are deprecated; use opts_chunk$set() and the chunk option child to set global chunk options and include child documents, respectively.

For logical options, only TRUE/FALSE/T/F are supported (the ﬁrst two are recommended), and true/false will not work; e.g., eval = FALSE is OK, and eval = false is not (unless there is an R object named false that happens to take a logical value FALSE). Chunk reference using the < <label> > syntax is still available, and there are other approaches for reusing chunks, e.g., use the new option ref.label; chunk references can be recursive, as introduced in Chapter 9.

###### 16.1.3 Problems

Some known problems and frequently asked questions in Sweave have been solved in knitr:

- • empty ﬁgure chunks give LATEX errors in Sweave but not in knitr be-

cause ﬁgures will not be generated at all; knitr writes ﬁgures to LATEX only when there are plots in a chunk

- • lattice (and ggplot2) graphics do not work in Sweave if you do not explicitly print() them, and they work in knitr just like in R console (if these plot objects appear in the top environment, you do not need to print them)
- • the width of ﬁgures in the output is set to .8\textwidth in Sweave by default via \setkeys{Gin}{width=.8\textwidth} deﬁned in the


LATEX style Sweave.sty; this affects all ﬁgures in the document regardless of whether they are generated by Sweave, and there is no straightforward way to set individual widths for ﬁgures; this problem has been solved by the out.width option in knitr

- • multiple ﬁgures from one ﬁgure chunk do not work by default in

Sweave and you have to write LATEX code by yourself in this case; for knitr, it does not make any difference no matter how many plots there are in one chunk

- • it is possible to use output hooks to change the formatting of output in

knitr, and we do not have to use hard-coded LATEX environments such as Sinput/Soutput in Sweave; in fact, we can call render_sweave() to render the Sweave style from knitr

- • it is easy to produce HTML output with knitr (with either R HTML or R Markdown), and Sweave needs extensions such as R2HTML, which only deals with HTML


Sometimes we see a stray Rplots.pdf ﬁle after we run Sweave, and that is because R’s default graphical device is pdf() for non-interactive R sessions, which creates Rplots.pdf. In knitr, the default device is set to a null device (pdf(file = NULL)) so that no stray PDF ﬁles will be generated.

###### 16.2 Other R Packages

Most features in Sweave and the R packages introduced below (except R2HTML) are covered by knitr, so this section is mainly for historical interest.

The highlight package (Francois, 2013) provides syntax highlighting for R code in Rnw documents. Like pgfSweave, cacheSweave, and R2HTML below, highlight was extended based on Sweave. In early versions (before v0.6), knitr depended on highlight to do syntax highlighting, but this dependency was removed later due to maintenance problems and the fact that it has additional dependencies (the Rcpp and the parser package). Now knitr uses its own syntax highlighting functions, which were based on regular expressions before R 3.0.0 and rely on the function getParseData() in the utils package in base R after R 3.0.0. To achieve similar functionality as highlight, we just need to use the chunk option highlight = TRUE in knitr.

The cacheSweave package (Peng, 2012) added an important feature to Sweave: the cache system; the weaver package (Falcon, 2013) did a similar thing with a different implementation. Chunk options cache and dependson were added, having the same meaning as in knitr (see Chapter 8).

The pgfSweave package (Bracken and Sharpsteen, 2012) combined the features of highlight and cacheSweave, and added further support for graphics. Speciﬁcally, plots can be cached as well, and TikZ graphics via the tikzDevice package are also supported for the sake of font style consistency. The author of this book switched to pgfSweave from Sweave when it came out, and contributed the formatR support to it (the tidy option), but as time went by, it became more and more difﬁcult to keep up with changes in Sweave. This package has been removed from the CRAN repository. At any rate, the design of knitr beneﬁted a lot from the author’s experience with pgfSweave.

The brew package (Horner, 2011) is a light-weight templating framework, and its syntax is similar to PHP (<?php ?>). Basically it parses and executes R code inside the templating tag <% %>. You can think of this as the inline R code in Sweave and knitr. It has a cache system but does not have direct graphics support. The knitr package also has partial support for the brew syntax, which we did not mention in Chapter 5; below is an example that can be compiled through knitr:

The value of pi is <% pi %>, and 2 times pi is <% 2*pi %>.

If an input ﬁle has an extension *.brew, knitr will use the brew syntax automatically. Note brew actually supports incomplete code fragments in several inline expressions, which makes it really similar to PHP. Here is an example taken from brew but knitr will not be able to compile it:

<% for (i in c( 1+1 , 1+pi , 1+pi , sin(pi/2) )) { -%> > <%=i%> <% print(eval(parse(text=i))) %> <% } -%>

The R2HTML package (Lecoutre, 2014) contains a large number of functions to export R objects to HTML. The main function is an S3 generic function HTML(), which can be applied to a variety of R objects such as data frames, tables, lm objects (returned by lm()) and so on. Below is a subset of the iris data converted to an HTML table:

library(R2HTML) HTML(head(iris[, -5], 1), "", caption = NULL)

<p align= center > <table cellspacing=0 border=1><tr><td>

<table border=0 class=dataframe> <tbody> <tr class= firstline >

<th>&nbsp; </th> <th>Sepal.Length </th> <th>Sepal.Width </th> <th>Petal.Length </th> <th>Petal.Width</th>

</tr>

<tr> <td class=firstcolumn>1 </td> <td class=cellinside>5.1 </td> <td class=cellinside>3.5 </td> <td class=cellinside>1.4 </td> <td class=cellinside>0.2 </td></tr>

</tbody> </table>

</td></table>

We can make use of R2HTML inside knitr for R HTML documents, with the chunk option results = ’asis’ to write raw HTML code into the output.

The other major contribution of R2HTML is the Sweave extension, which allows one to write an HTML report based on Sweave.

There is a task view on CRAN about reproducible research: http:// cran.r-project.org/web/views/ReproducibleResearch.html, where we can ﬁnd more packages on this topic.

###### 16.3 Python Packages

In this section we introduce three packages based on Python for dynamic documents: Dexy, PythonTEX, and IPython.

###### 16.3.1 Dexy

Dexy (http://www.dexy.it) is a free Python package that features a very general design. According to its website:

Dexy is a free-form literate documentation tool for writing any kind of technical document incorporating code. Dexy helps you write correct documents, and to easily maintain them over time as your code changes.

The four major features are:

- 1. any language (source code)
- 2. any markup (output)
- 3. any template
- 4. any API (programming)


There are apparently some similarities between Dexy and knitr, such as the multi-language support. An important concept of Dexy is the “ﬁlter”: the ﬁlter takes an input ﬁle and converts it to an output ﬁle, which is similar to the pipe | in shell scripts. The ﬁlters in Dexy are actually a combination of concepts in knitr: a ﬁlter may render output (e.g., from Markdown to HTML), or run a programming language (like language engines in knitr), or do additional tasks like knitr’s chunk hooks.

Normally Dexy separates computer code from templates, which can be either good or bad. The good aspect is that the source scripts can be reused, and the bad thing is we have to jump back and forth between the report environment and the source code. By default knitr directly embeds code chunks in a report, but we can also externalize code chunks as introduced in Chapter 9.

###### 16.3.2 PythonTEX

PythonTEX (https://github.com/gpoore/pythontex) is a LATEX package, which features execution of Python code within LATEX. According to its documentation:

PythonTEX provides fast, user-friendly access to Python from within LATEX. It allows Python code entered within a LATEX document to be executed, and the results to be included within the original document. It also provides syntax highlighting for code within LATEX documents via the Pygments package.

We can insert inline Python code using the \pyb{} command, or emulate a Python session in LATEX using the pyconsole environment, e.g.,

\begin{pyconsole}[][frame=single]

- x = 123
- y = 345
- z = x + y z def f(expr):


return(expr**4)

- f(x) print( Python says hi from the console! ) \end{pyconsole}


When we compile this document, the Python code will be evaluated and the results will be inserted into the output.

Due to its Python origin, it also has integration with other Python packages such as SymPy (symbolic manipulation) and matplotlib (plots).

###### 16.3.3 IPython

IPython (http://ipython.org) is an interactive shell for Python that features a Web-based notebook with support for code, text, mathematical expressions, inline plots and other rich media, high performance tools for parallel computing, and so on.

Figure 16.1 is a screenshot of IPython in a GNOME terminal under Ubuntu. We can see that it has basic functionalities of a shell such as the auto-completion of commands: we type x.spl<TAB> in the shell and will see the auto-completion below.

The most notable feature related to report generation is its Webbased notebook: we can work in the Web browser with Python commands, view the results on the ﬂy (including both numerical and graphical results), and the notebook can be continuously updated as we input more content into the notebook. It is very much like writing code chunks in knitr.

An IPython notebook can be saved as a JSON ﬁle with the extension *.ipynb, which can be shared with others. The notebook may or may not contain output; a notebook without the output is similar to the source document for knitr (e.g., Rnw and Rmd documents).

Inspired by IPython, knitr has got a similar Web notebook (but with fewer features), which we have mentioned in Section 3.2.2.

![image 34](Dynamic Documents with R and knitr 2nd_images/imageFile34.png)

FIGURE 16.1: A screenshot of IPython: input is marked as In[n ], and output is marked as Out[n ].

###### 16.4 More Tools

In addition to R and Python packages, there are tools in other programs. It is impossible to enumerate all the tools for dynamic documents in this chapter. Schulte et al. (2012) have provided a list of existing tools for literate programming and reproducible research, such as Javadoc, cweb, noweb, Sweave, SASweave, and so on.

###### 16.4.1 Org-mode

Org-mode is a plain text markup language, with an implementation in the Emacs text editor (Schulte et al., 2012). It supports both literate programming and reproducible research (in the sense of dynamic documents). It more or less follows the syntax of early implementations of literate programming such as WEB and noweb, i.e., it has the concept of code chunks and text chunks (the text chunks are sometimes called “prose”). A code chunk in Org-mode looks like this:

#+name: c-chunk #+begin_src C

int main(){

return 0; }

#+end_src

By comparison, the same chunk is written like this in knitr: <<c-chunk, engine= c >>= int main(){

return 0;

} @

The metadata is stored in the chunk headers. Org-mode supports any input languages, with either LATEX or HTML as the output format.

Schulte et al. (2012) mentioned the capability of literate programming of existing tools (e.g., Sweave does not have it), which we did not emphasize in this book because it does not sound interesting to report writers. As a matter of fact, knitr also has this capability of reorganizing code chunks (see Chapter 9). Below is a simple example of deﬁning chunk B later but embedding it in an earlier chunk A:

- <<A>>= df <- data.frame(x = 1:10, y = rnorm(10))
- <<B>> coef(fit) @


<<B>>= fit <- lm(y ~ x, data = df) @

Powerful as it is, the Emacs nature of Org-mode may be an obstacle to beginners.

###### 16.4.2 SASweave

SASweave (http://homepage.cs.uiowa.edu/~rlenth/SASweave) is an implementation of literate programming with SAS and R. It was written in gawk. The basic idea is the same as Sweave and knitr. See Lenth and Højsgaard (2007) for more information. The knitr package has more comprehensive support for R but less support for SAS compared to SASweave.

###### 16.4.3 Ofﬁce

We do not have to choose the plain text format for dynamic documents, whereas almost everything we have introduced in this book is based on plain text. There are tools based on OpenOfﬁce (or OpenDocument Text) or Microsoft Ofﬁce products (we call them Ofﬁce documents for short), and they may seem appealing at ﬁrst glance. At its core, an Ofﬁce document is usually an XML ﬁle (which may be compressed), so it is possible to embed code chunks in it. We can parse code chunks, run them, and insert the results back.

The major problem we see is that the XML format is too complicated and there are too many standards, so it is not trivial to make sure the modiﬁed document is still a valid Ofﬁce document. As one example, the StatWeave package (http://homepage.stat.uiowa.edu/~rlenth/ StatWeave/) no longer works with OpenOfﬁce (3.2 and higher) because “OpenOfﬁce ﬂags the modiﬁed document as corrupted.”

By comparison, plain text ﬁles are much easier to deal with; there are no complicated standards such as ECMA-376 to take care of. If we want Ofﬁce documents at all, there are at least possibilities of conversion from Markdown. Recall what we quoted in Chapter 1:

The source code is real.

### A

###### Internals

In this appendix we explain some internal structures of the knitr package, which may help other developers better understand this package, and contribute code when necessary. General users do not need to read this appendix. We show the internals in three aspects: documentation, the application of closures, and the implementation of some features.

###### A.1 Documentation

There are three types of documentation in knitr: the R documentation (Rd), the PDF manuals, and the website.

The R documentation is based on roxygen2 (Wickham et al., 2015), which allows one to write Rd in roxygen comments (#’) with tags, and these comments will be translated into the real Rd. Below is an example of the roxygen comment:

#  @author Yihui Xie

It will be translated into Rd as: \author{Yihui Xie}

There is a series of tags in roxygen such as @usage, @param, @return, and @examples, which correspond to \usage{}, \arguments{\item{}}, \value{}, and \examples{}, respectively, in Rd. The advantage of writing roxygen comments over the ofﬁcial Rd is that we can keep the documentation and the source code in the same ﬁle; by comparison, the ofﬁcial approach to writing R packages is to write R sources under the R/ directory, and manual pages as *.Rd ﬁles under man/. This is not convenient because we have to jump between two ﬁles, and it is likely that we update the R source but forget to update the documentation. Roxygen comments appear right above the R functions in the source, so it is much easier to maintain both the source and documentation.

247

Below is a complete example of a function documented with roxygen comments:

#  Repeat a character string #  #  Repeat a string n times and make one string. #  @param x a character string #  @param n an integer #  @return A character string. #  @examples f( hi , n = 5) f <- function(x, n = 10) {

paste(rep(x, n), collapse = "") }

We can use the roxygenize() function in roxygen2 to convert roxygen comments to the ofﬁcial Rd ﬁles. All objects in knitr are documented in this way. Besides, roxygen2 also handles NAMESPACE and the Collate ﬁeld in DESCRIPTION automatically, so we can really focus on working R source ﬁles.

The source documents of the PDF manuals are under the examples directory (see inst/examples/ in the source package), e.g., the main manual is knitr-manual.Rnw. The Rnw ﬁles are exported from LYX ﬁles (Section 4.2), so it is recommended to open the LYX ﬁles to edit or compile PDF manuals. The PDF manuals are not shipped with the source package, because (1) I do not want to put binary ﬁles under version control (especially when they are by-products of source ﬁles) and (2) they are hosted in the package website.

The package website is built on Jekyll as introduced in Section 13.4. Speciﬁcally, all pages are written in Markdown, and put under the gh-pages branch in the Git repository (the package itself is in the master branch). Github will rebuild the website automatically once changes are pushed there through Git. If you want to contribute to the website, just switch to the gh-pages branch, and update the Markdown ﬁles.

###### A.2 Closures

Closures play a central role in knitr; some common objects such as opts_chunk (Section 5.1.1) and knit_engines (Chapter 11) are built on closures.

A closure is essentially a function, and it also has access to non-local variables. Below is a simple example:

f <- function() { x <- 1 function(y) x + y

} g <- f()

- g(5) # add 5 to x ## [1] 6 ls(environment(g)) # g can see x ## [1] "x"


The function g() was created from f() (note f() returns a function), g() uses an object x that was created inside f(), and x only exists in f(). No matter where g() is called, it always has access to this x.

In fact, we can even modify non-local variables through a closure. Below is a minimal example that shows how the chunk options manager opts_chunk works:

new_list <- function(default = list()) {

list(get = function() default, set = function(...) { x <- list(...) if (length(x)) default[names(x)] <<- x

}) }

The function new_list() returns a list of functions (a setter and a getter). The object default is bound to these two functions. You can think of it as the default list of chunk options. Next we show how to get and set the chunk options.

opts <- new_list(list(eval = TRUE)) str(opts$get())

- ## List of 1 ## $ eval: logi TRUE

opts$set(eval = FALSE) # change eval to FALSE opts$set(results = "markup") # add a chunk option str(opts$get())

- ## List of 2 ## $ eval : logi FALSE ## $ results: chr "markup"


opts$set(results = "hide") # change the results option

In the $set() function, we used < <- to assign the arguments to the object default, and that is why we can modify this object in the parent environment (had we used the normal <-, default in the parent environment would not be modiﬁed; a local copy will be created instead).

By using closures, knitr can manage objects in their own environments with the same syntax. The internal function new_defaults() in knitr is used to create such a list of closures.

Besides the objects opts_chunk (for managing chunk options) and knit_engines (for managing language engines), there are a few other similar objects:

opts_knit package options (Section 12.2) opts_current chunk options for the current chunk opts_template chunk option templates (Section 12.1.2) knit_hooks hook functions (both output hooks and chunk hooks) knit_patterns syntax patterns for the parser (Section 5.1)

###### A.3 Implementation

This section explains some implementation details for this package. One minor thing to mention ﬁrst is that I use = instead of <- as the assignment operator, and you will see = all over the place in the source code. It is a matter of personal taste, and I do not see real disadvantages in it, but you are expected to follow = when contributing code to this package. In this book, you see <- because I typed equal signs but they were automatically replaced by formatR.

###### A.3.1 Parser

The document parser (Section 5.1) works like this: the child elements chunk.begin and chunk.end in the syntax pattern object are used to split the document into pieces (code chunks and text chunks), and for the code chunks, the chunk options (i.e., the text extracted from the ﬁrst line) are parsed as R code, and this is why chunk options have to follow the R syntax. Here is an example explaining how knitr gets chunk options from a text fragment:

## suppose this is the chunk options text txt <- "label, eval=TRUE, echo=1:3, foo=if(TRUE) 2 else 5" opc <- eval(parse(text = paste("alist(", txt, ")"))) names(opc) # the chunk label is not named

## [1] "" "eval" "echo" "foo" str(opc) # some are unevaluated expressions

## List of 4 ## $ : symbol label ## $ eval: logi TRUE ## $ echo: language 1:3 ## $ foo : language if (TRUE) 2 else 5

First we added the function alist() around the text, and this function will treat its arguments as if they described function arguments, therefore no “arguments” will be evaluated at this time. However, the syntax must be valid at least; one exception is the chunk label: it is automatically quoted if necessary, since it is supposed to be a character string. The internal function parse_params() is used to parse chunk options:

p <- knitr:::parse_params str(p("chunk-label, eval=TRUE, foo=5"))

- ## List of 3 ## $ label: chr "chunk-label" ## $ eval : logi TRUE ## $ foo : num 5


# 2a is not a valid symbol in R, but knitr will quote it # automatically so parsing is OK parse(text = "alist(2a)") ## Error: <text>:1:8: unexpected symbol ## 1: alist(2a ## ^ str(p("2a, eval=FALSE"))

## List of 2 ## $ label: chr "2a" ## $ eval : logi FALSE

str(p(" 2a , eval=FALSE")) # or you can quote it manually

## List of 2 ## $ label: chr "2a" ## $ eval : logi FALSE

The chunk options are not evaluated until before the chunks are executed, so the chunk options can use objects of unknown values in the document at the parsing time. For example, the options echo and foo above are unevaluated expressions, and we will evaluate them explicitly later:

eval(opc$echo) ## [1] 1 2 3 eval(opc$foo) ## [1] 2

All code chunks are stored as a named list in an internal object knit_code; the names are chunk labels, and the content is the code. This object is also created as a list of closures, so it has the get() and set() methods, but it is not recommended to modify this object due to possible unexpected consequences. If needed, we can access code chunks via knitr:::knit_code$get(’chunk-label’).

###### A.3.2 Chunk Hooks

There is a number of default hooks in knit_hooks, which are output hooks (Section 5.3):

names(knit_hooks$get(default = TRUE))

## [1] "source" "output" "warning" "message" ## [5] "error" "plot" "inline" "chunk" ## [9] "text" "document"

Any other hooks in this object are treated as chunk hooks (Chapter 10). Before and after a code chunk is executed, all extra hooks will be called. Here is the pseudo code:

hook(before = TRUE, ...) evaluate(code) hook(before = FALSE, ...)

One issue to keep in mind is the order of the hooks to run: if there are two hooks A and B deﬁned in knit_hooks, what is the order in which they are called? This order is obtained from chunk options: there must be two chunk options, A and B, corresponding to these two hooks, and the order of chunk options determines the order in which to run the hooks; e.g., if A is before B, then hook A is called before B. However, after a code chunk has been evaluated, the order is reversed, and the reason is to make sure the results returned by the hooks pair in groups. For example, suppose the hook A returns \begin{Aenvir} before a chunk, and \end{Aenvir} after a chunk; similarly B returns Benvir. Then what we want in the output is this:

\begin{Aenvir} \begin{Benvir} % results from the chunk \end{Benvir} \end{Aenvir}

Note \end{Benvir} comes before \end{Aenvir}. For this reason, the following two chunks return different results when hooks A and B are deﬁned:

- <<A=TRUE, B=TRUE>>=
- <<B=TRUE, A=TRUE>>=


###### A.3.3 Option Aliases

It takes only a few lines to implement chunk option aliases (Section 12.1.1), since it is a simple operation of substituting certain elements in a list. Below is a short function that illustrates the idea:

apply_aliases <- function(x, list) {

## names are aliases of x list[x] <- list[names(x)] list

} al <- c(w = "fig.width", h = "fig.height", a = "fig.align") op <- list(w = 7, h = 7, echo = TRUE, a = "center") str(op) # user s options

- ## List of 4 ## $ w : num 7


## $ h : num 7 ## $ echo: logi TRUE ## $ a : chr "center"

str(apply_aliases(al, op)) # corrected options

## List of 7 ## $ w : num 7 ## $ h : num 7 ## $ echo : logi TRUE ## $ a : chr "center" ## $ fig.width : num 7 ## $ fig.height: num 7 ## $ fig.align : chr "center"

Aliases are set in a named character vector, and the names are the aliases of the elements in the vector. In the above example, apply_aliases() added elements fig.width and fig.height into the list op according to the values of w and h, respectively, which were speciﬁed by the user, but internally knitr still uses fig.width and fig.height.

###### A.3.4 Cache

The cache in knitr is also managed by an object consisting of closures, but it is more complicated (see the internal function new_cache()). The closures are used to save, load, and delete cache ﬁles, and we only explain one aspect of the cache here: how the side effect of printing is cached (Section 8.4).

As we mentioned in Section 5.3, the code chunks are evaluated by the evaluate package. As a matter of fact, printed results are returned as character strings, and the output of the whole chunk is also a character string (formatted by output renderers). This character string is assigned to a variable, with the variable name constructed from the MD5 hash and the chunk label. This variable is saved in the cache database along with all other variables created in the chunk. The next time the chunk is to be evaluated, knitr will check if the chunk needs to be updated; if not, all objects will be loaded directly, including the object of the chunk output, which also contains the printed results (in fact, everything of this chunk); instead of re-evaluating the chunk, this object is written into the output directly.

###### A.3.5 Compatibility with Sweave

Since knitr uses some different chunk options with Sweave, there is a function Sweave2knitr() to correct the inappropriate options and their values. For example, results = tex is changed to results = ’markup’ automatically (because ’tex’ is not an appropriate value to reﬂect what the results option really does).

The implementation is mainly based on regular expressions, and here is a simple example: op <- "<<eval=TRUE, results=tex>>=" gsub("(results)\\s*=\\s*tex", "\\1= markup ", op) ## [1] "<<eval=TRUE, results= markup >>="

Sweave2knitr() takes care of a large number of cases of inappropriate chunk options as well as \SweaveOpts{} and \SweaveInput{}. See Section 16.1 for examples.

###### A.3.6 Concordance

The concept of concordance is speciﬁc to Rnw/LATEX. The problem to solve is the mapping of line numbers between the TEX output and the Rnw source. When an error occurs in LATEX, we know the line number of the problematic line (by parsing the error log), but we do not know the corresponding line number in the Rnw source document, because the line numbers of the two documents may not match. One chunk of

- 5 lines in the Rnw document may produce 10 or 3 lines of LATEX code in the output.


Sweave has a better implementation of concordance than knitr. The mapping is more precise in Sweave. In knitr, it is only an approximation achieved in this way: when parsing the source document, the number of lines of the code chunks and text chunks are recorded; after these chunks have been evaluated, the number of lines of the corresponding output chunks is calculated again. Suppose one source chunk has 5 lines, and if

- • the output has 5 lines too, the i-th line in the source is mapped to the i-th line in the output
- • the output has 3 lines, the ﬁrst 3 lines of the source are mapped to the 3 lines in the output
- • the output has 10 lines, the 5 lines of the source are mapped to the ﬁrst 5 lines in the output


Obviously this may not be a good approximation, but it should be helpful enough for error navigation. At least the error number in LATEX can point to a rough area of the problematic source.

The other use of concordance is the navigation between PDF and Rnw ﬁles. SyncTEX supports this kind of navigation: you can click one line in the PDF document to jump back to the source ﬁle, or click one line in the source to jump to the PDF. Without the concordance information, we cannot navigate between Rnw and PDF (only TEX↔PDF is possible).

For now, only RStudio uses the concordance information produced by knitr. To enable concordance (it is disabled by default), you can set the package option (RStudio does this automatically):

opts_knit$set(concordance = TRUE)

When concordance is enabled, a ﬁle input-concordance.tex will be generated if the Rnw ﬁle is named as input.Rnw. This ﬁle contains compressed mapping information.

###### A.4 Syntax

Users may wonder why knitr uses different input syntax for different document formats (Section 5.1), e.g., Rnw uses < <> >=, and Rmd uses    {r}. In fact, the syntax is not tied to document formats; we can certainly use the Rnw syntax for Rmd documents.

# This is a markdown document Here is a **code chunk**: <<test>>= 1 + 1 rnorm(5) @ And an inline value \Sexpr{pi}.

For the example document above (suppose it is named test.Rmd), we can compile it by:

library(knitr) pat_rnw() # input is Rnw syntax render_markdown() # output is markdown knit("test.Rmd")

The function pat_rnw() sets the syntax to be Rnw, and the function render_markdown() sets the output renders to be Markdown hooks.

But why not use the Rnw syntax for all documents? The decision was made because I wanted more natural syntax according to the authoring format, and < <> >= is not a valid markup in any document format; e.g., it is neither a LATEX command nor an HTML tag. In fact, Sweave has another set of syntax that is LATEX-like, e.g.,

\begin{Scode}{fig = TRUE, echo = FALSE} library("graphics") boxplot(Ozone ~ Month, data = airquality) \end{Scode}

I would prefer [] to {} for chunk options, which will be a more

natural choice in LATEX. Anyway, < <> >= remained in knitr due to its popularity.

Except for Rnw documents (due to historic reasons), other formats make the knitr source documents still valid documents even before the R code is executed. For example, R code in R HTML documents is put in HTML comments (<!-- -->).

###### Bibliography

Adler, D. and Murdoch, D. (2014). rgl: 3D visualization device system (OpenGL). R package version 0.95.1201.

Allaire, J., Cheng, J., Xie, Y., McPherson, J., Chang, W., Allen, J., Wickham, H., and Hyndman, R. (2015a). rmarkdown: Dynamic Documents for R. R package version 0.5.1.

Allaire, J., Horner, J., Marti, V., and Porte, N. (2015b). markdown: Markdown Rendering for R. R package version 0.7.7.

Auguie, B. (2013). cda: Coupled dipole approximation in electromagnetic scattering. R package version 1.3.3.

Baggerly, K. A., Morris, J. S., and Coombes, K. R. (2004). Reproducibility of seldi-tof protein patterns in serum: comparing datasets from different experiments. Bioinformatics, 20(5):777–785.

Bracken, C. and Sharpsteen, C. (2012). pgfSweave: Quality speedy graphics compilation and caching with Sweave. R package version 1.3.0.

Buckheit, J. and Donoho, D. (1995). Wavelab and reproducible research. Wavelets and Statistics, 103:55.

Chang, W., Cheng, J., Allaire, J., Xie, Y., and McPherson, J. (2015). shiny: Web Application Framework for R. R package version 0.11.1.

Dahl, D. B. (2014). xtable: Export tables to LaTeX or HTML. R package version 1.7-4.

Eddelbuettel, D., Francois, R., Allaire, J., Ushey, K., Bates, D., and Chambers, J. (2015). Rcpp: Seamless R and C++ Integration. R package version 0.11.5.

Ellson, J., Gansner, E., Koutsoﬁos, L., North, S., and Woodhull, G.

(2002). Graphviz — open source graph drawing tools. In Graph Drawing, pages 483–484. Springer-Verlag.

Falcon, S. (2013). weaver: Tools and extensions for processing Sweave documents. R package version 1.26.0.

259

Fomel, S. and Claerbout, J. (2009). Guest editors’ introduction: Reproducible research. Computing in Science & Engineering, 11(1):5–7.

Francois, R. (2013). highlight: Syntax highlighter. R package version 0.4.3. Friedl, J. (2006). Mastering Regular Expressions. O’Reilly Media, Incor-

porated.

Gentleman, R. (2005). Reproducible research: A bioinformatics case study. Statistical Applications in Genetics and Molecular Biology, 4(1):1034.

Gentleman, R. and Temple Lang, D. (2004). Statistical analyses and reproducible research. Bioconductor Project Working Papers. URL: http://biostats.bepress.com/bioconductor/paper2.

Gove, J. H. (2013). sampSurf: Sampling Surface Simulation for Areal Sampling Methods. R package version 0.6-8.

Gruber, J. (2004). The Markdown Project. URL: http://daringfireball. net/projects/markdown/.

Guo, J., Betancourt, M., Brubaker, M., Carpenter, B., Gao, Y., Goodrich, B., Hoffman, M., Lee, D., Li, P., Malecki, M., and Gelman, A. (2014). rstan: RStan: R interface to Stan. R package version 2.5.0.

Harrell, Jr., F. E. (2001). Regression Modeling Strategies: With Applications to Linear Models, Logistic Regression, and Survival Analysis. Springer New York.

Harrell, Jr., F. E. (2015). Hmisc: Harrell Miscellaneous. R package version 3.15-0.

Horner, J. (2011). brew: Templating Framework for Report Generation. R package version 1.0-6.

Horton, N., Aloisio, K., Zhang, R., and Loi, L. (2012). The statistical sleuth (2nd edition) in R. URL: http://www.math.smith.edu/ ~nhorton/sleuth/.

Huang, Y. and Gottardo, R. (2013). Comparability and reproducibility of biomedical data. Brieﬁngs in Bioinformatics, 14(4):391–401.

Ihaka, R. and Gentleman, R. (1996). R: A language for data analysis and graphics. Journal of Computational and Graphical Statistics, 5(3):299– 314.

Jockers, M. L. (2014). Text Analysis with R for Students of Literature. Springer.

- Knuth, D. E. (1983). The WEB system of structured documentation. Technical report, Department of Computer Science, Stanford University.
- Knuth, D. E. (1984). Literate programming. The Computer Journal, 27(2):97–111.


Lebanon, G. (2012). Probability: The Analysis of Data, volume 1. CreateSpace Independent Publishing Platform.

Lecoutre, E. (2014). R2HTML: HTML exportation for R objects. R package version 2.3.1.

Leisch, F. (2002). Sweave: Dynamic generation of statistical reports using literate data analysis. In COMPSTAT 2002 Proceedings in Computational Statistics, number 69, pages 575–580. Heidelberg: Physica Verlag.

Lenth, R. V. and Højsgaard, S. (2007). Sasweave: Literate programming using sas. Journal of Statistical Software, 19(8):1–20.

Murdoch, D. (2012). tables: Formula-driven table generation. R package version 0.7.

Murphy, D. (2012). Changes and additions to ggplot2 0.9.0. URL: https://github.com/djmurphy420/ggplot2-transition-guide. Murrell, P. (2011). R Graphics, Second Edition. Chapman & Hall/CRC. Murrell, P. and Ripley, B. (2006). Non-standard fonts in PostScript and

PDF graphics. R News, 6(2):41–47.

Oetiker, T., Partl, H., Hyna, I., and Schlegl, E. (1995). The not so short introduction to LATEX2ε. URL: http://www.ctan.org/tex-archive/ info/lshort/.

Peng, R. (2009). Reproducible research and biostatistics. Biostatistics, 10(3):405–408.

Peng, R. D. (2012). cacheSweave: Tools for caching Sweave computations. R package version 0.6-1.

Qiu, Y. and Xie, Y. (2015). highr: Syntax Highlighting for R Source Code. R package version 0.5.

Qiu, Y., Xie, Y., and Bracken, C. (2015). R2SWF: Convert R Graphics to Flash Animations. R package version 0.9.

- R Core Team (2014). R Language Deﬁnition. R Foundation for Statistical Computing, Vienna, Austria.
- R Core Team (2015). R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria.


Ramsey, F. and Schafer, D. (2002). The Statistical Sleuth: A Course in Methods of Data Analysis, Second Edition. Duxbury Press.

Ramsey, N. (1994). Literate programming simpliﬁed. Software, IEEE, 11(5):97–105.

Rossini, A. (2002). Literate statistical analysis. In Proceedings of the 2nd International Workshop on Distributed Statistical Computing, pages 15– 17, Vienna, Austria.

Rossini, A., Heiberger, R., Sparapani, R., Maechler, M., and Hornik, K. (2004). Emacs speaks statistics: A multiplatform, multipackage development environment for statistical analysis. Journal of Computational and Graphical Statistics, 13(1):247–261.

Schulte, E., Davison, D., Dye, T., and Dominik, C. (2012). A multilanguage computing environment for literate programming and reproducible research. Journal of Statistical Software, 46(3):1–24.

Sharpsteen, C. and Bracken, C. (2015). tikzDevice: R Graphics Output in LaTeX Format. R package version 0.8.1.

Tantau, T. (2008). The TikZ and PGF Packages. URL: http:// sourceforge.net/projects/pgf/.

Tantau, T., Wright, J., and Miletic, V. (2012). User’s Guide to the Beamer Class. URL: http://bitbucket.org/rivanvx/beamer.

Temple Lang, D., Swayne, D., Wickham, H., and Lawrence, M. (2014). rggobi: Interface between R and GGobi. R package version 2.1.20.

Vaidyanathan, R. (2012). slidify: Generate reproducible html5 slides from R markdown. R package version 0.4.5.

Vaidyanathan, R., Cheng, J., Allaire, J., Xie, Y., and Russell, K. (2014). htmlwidgets: HTML Widgets for R. R package version 0.3.2.

van Heesch, D. (2008). Doxygen: Source code documentation generator tool. URL: http://www.doxygen.org/.

Venables, W. N. and Ripley, B. D. (2002). Modern Applied Statistics with S. Springer-Verlag, 4th edition.

Wei, T. (2013). corrplot: Visualization of a correlation matrix. R package version 0.73.

Wickham, H. (2015). evaluate: Parsing and Evaluation Tools that Provide More Details than the Default. R package version 0.7.

Wickham, H., Danenberg, P., and Eugster, M. (2015). roxygen2: In-Source Documentation for R. R package version 4.1.1.

- Xie, Y. (2013). runr: Run External Programs from R. R package version 0.0.6.
- Xie, Y. (2014). printr: Automatically Print R Objects According to knitr Output Format. R package version 0.0.3.
- Xie, Y. (2015a). formatR: Format R Code Automatically. R package version 1.2.


- Xie, Y. (2015b). knitr: A General-Purpose Package for Dynamic Report Generation in R. R package version 1.10.
- Xie, Y. (2015c). servr: A Simple HTTP Server to Serve Static Files or Dynamic Documents. R package version 0.2.


Yin, T., Cook, D., and Lawrence, M. (2012). ggbio: an R package for extending the grammar of graphics for genomic data. Genome Biology, 13(8):R77.

![image 35](Dynamic Documents with R and knitr 2nd_images/imageFile35.png)

Statistics

Suitable for both beginners and advanced users, Dynamic Documents with R and knitr, Second Edition makes writing statistical reports easier by integrating computing directly with reporting. Reports range from homework, projects, exams, books, blogs, and Web pages to virtually any documents related to statistical graphics, computing, and data analysis. The book covers basic applications for beginners while guiding power users in understanding the extensibility of the knitr package.

###### New to the Second Edition

- • A new chapter that introduces R Markdown v2
- • Changes that reflect improvements in the knitr package
- • New sections on generating tables, defining custom printing methods for objects in code chunks, the C/Fortran engines, the Stan engine, running engines in a persistent session, and starting a local server to serve dynamic documents


Like its highly praised predecessor, this edition shows you how to improve your efficiency in writing reports. The book takes you from program output to publication-quality reports, helping you fine-tune every aspect of your report. Demos and other information about the package are available on the author’s website.

Yihui Xie is a software engineer at RStudio. He earned a PhD from the Department of Statistics at Iowa State University. His research focuses on interactive statistical graphics and statistical computing. He is an active R user and the author of several award-winning R packages. He is also the founder of “Capital of Statistics,” a large online statistics community in China.

K25425

w w w . c r c p r e s s . c o m

Second Edition

DynamicDocumentswithRandknitr

Xie

## TheRSeries

# Dynamic Documents with R and knitr

Second Edition

Yihui Xie

K25425_cover.indd 1 4/17/15 11:01 AM

