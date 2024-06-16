### 3

###### A First Look

The knitr package is a general-purpose literate programming engine it supports document formats including LATEX, HTML, and Markdown (see Chapter 5), and programming languages such as R, Python, awk, C++, and shell scripts (Chapter 11). Before we get started, we need to install knitr in R. Then we will introduce the basic concepts with minimal examples. Finally, we will show how to generate reports quickly from pure R scripts, which can be useful for beginners who do not know anything about dynamic documents.

###### 3.1 Setup

Since knitr is an R package, it can be installed from CRAN in the usual way in R:

install.packages("knitr", dependencies = TRUE)

Note here that dependencies = TRUE is optional, and will install all packages that are not absolutely necessary but can enhance this package with some useful features. The development version is hosted on Github: https://github.com/yihui/knitr, and you can always check out the latest development version, which may not be stable but contains the latest bug ﬁxes and new features. If you have any problems with knitr, the ﬁrst thing to check is its version:

packageVersion("knitr") # if not the latest version, run update.packages()

If you choose LATEX as the typesetting tool, you may need to install MiKTEX (Windows, http://miktex.org/), MacTEX (Mac OS, http:// tug.org/mactex/), or TEXLive (Linux, http://tug.org/texlive/). If

11

you are going to work with HTML or Markdown, nothing else needs to be installed, since the output will be Web pages, which you can view with a Web browser.

Once we have knitr installed, we can compile source documents using the function knit(), e.g.,

library(knitr) knit("your-file.Rnw")

A *.Rnw ﬁle is usually a LATEX document with R code embedded in it, as we will see in the following section and Chapter 5, in which more types of documents will be introduced.

###### 3.2 Minimal Examples

We use two minimal examples written in LATEX and Markdown, respectively, to illustrate the structure of dynamic documents. We do not dis-

cuss the syntax of LATEX or Markdown for the time being (see Chapter 5 instead). For the sake of simplicity, the cars dataset in base R is used to build a simple linear regression model. Type ?cars in R to see detailed documentation. Basically it has two variables, speed and distance:

str(cars)

##  data.frame : 50 obs. of 2 variables: ## $ speed: num 4 4 7 7 8 9 10 10 10 11 ... ## $ dist : num 2 10 4 22 16 10 18 26 34 17 ...

###### 3.2.1 An Example in LATEX

Figure 3.1 is a full example of R code embedded in LATEX; we call this kind of documents Rnw documents hereafter because their ﬁlename extension is Rnw by convention. If we save it as a ﬁle minimal.Rnw and run knit(’minimal.Rnw’) in R as described before, knitr will generate a LATEX output document named minimal.tex. For those who are familiar with LATEX, you can compile this document to PDF via pdflatex. Figure

- 3.2 is the PDF document compiled from the Rnw document.


What is essential here is how we embedded R code in LATEX. In an Rnw document, < <> >= marks the beginning of code chunks, and @ terminates a code chunk (this description is not rigorous but is often easier

|\documentclass{article} \begin{document} \title{A Minimal Example} \author{Yihui Xie} \maketitle<br><br>We examine the relationship between speed and stopping distance using a linear regression model: $Y = \beta_0 + \beta_1 x + \epsilon$.<br><br><<model, fig.width=4, fig.height=3, fig.align= center >>= par(mar = c(4, 4, 1, 1), mgp = c(2, 1, 0), cex = 0.8) plot(cars, pch = 20, col =  darkgray ) fit <- lm(dist ~ speed, data = cars) abline(fit, lwd = 2) @<br><br>The slope of a simple linear regression is \Sexpr{coef(fit)[2]}. \end{document}|
|---|


- FIGURE 3.1: The source of a minimal Rnw document: see output in Figure 3.2.


to understand); we have four lines of R code between the two markers in this example to draw a scatterplot, ﬁt a linear model, and add a regression line to the scatterplot. The command \Sexpr{} is used to embed inline R code, e.g., coef(fit)[2] in this example. We can write chunk options for a code chunk between < < and > >=; the chunk options in this example speciﬁed the plot size to be 4 by 3 inches (fig.width and fig.height), and plots should be aligned in the center (fig.align).

In this minimal example, we have most basic elements of a report:

- 1. title, author, and date
- 2. model description
- 3. data and computation
- 4. graphics
- 5. numeric results


All the output is generated dynamically from R. Even if the data has

|A Minimal Example<br><br>Yihui Xie April 11, 2015<br><br>We examine the relationship between speed and stopping distance using a linear regression model: Y = β0 + β1x + .<br><br>par(mar = c(4, 4, 1, 1), mgp = c(2, 1, 0), cex = 0.8) plot(cars, pch = 20, col = "darkgray") fit <- lm(dist ~ speed, data = cars) abline(fit, lwd = 2)<br><br>5 10 15 20 25<br><br>020406080100<br><br>speed<br><br>dist<br><br>The slope of a simple linear regression is 3.9324088.|
|---|


- FIGURE 3.2: A minimal example in LATEX with an R code chunk, a plot, and numeric output (regression coefﬁcient).


1

|--title: A Minimal Example<br><br>--We examine the relationship between speed and stopping distance using a linear regression model: $Y = \beta_0 + \beta_1 x + \epsilon$.    {r fig.width=4, fig.height=3, fig.align= center } par(mar = c(4, 4, 1, 1), mgp = c(2, 1, 0), cex = 0.8) plot(cars, pch = 20, col =  darkgray ) fit <- lm(dist ~ speed, data = cars) abline(fit, lwd = 2)<br><br>The slope of a simple linear regression is  r coef(fit)[2] .|
|---|


- FIGURE 3.3: The source of a minimal Rmd document: see output in Figure 3.4.


changed, we do not need to redo the report from the ground up, and the output will be updated accordingly if we update the data and recompile the report.

###### 3.2.2 An Example in Markdown

LATEX may look overwhelming to beginners due to the large number of commands. By comparison, Markdown (Gruber, 2004) is a much simpler format. Figure 3.3 is a Markdown example doing the same analysis with the previous example:

The ideal output from Markdown is an HTML Web page, as shown in Figure 3.4 (in Mozilla Firefox). Similarly, we can see the syntax for R code in a Markdown document:    {r} opens a code chunk, terminates a chunk, and inline R code can be put inside  r , where is a backtick.

A slightly longer example in knitr is a demo named notebook, which is based on Markdown. It shows not only the potential power of Markdown, but also the possibility of building Web applications with knitr. To watch the demo, run the code below:

![image 2](Dynamic Documents with R and knitr 2nd_images/imageFile2.png)

- FIGURE 3.4: A minimal example in Markdown with the same analysis


- as in Figure 3.2, but the output is HTML instead of PDF now.


if (!require("shiny")) install.packages("shiny") demo("notebook", package = "knitr")

Your default Web browser will be launched to show a Web notebook. The source code is in the left panel, and the live results are in the right panel. You are free to experiment with the source code and recompile the notebook.

###### 3.3 Quick Reporting

If a user only has basic knowledge of R but knows nothing about knitr, or one does not want to write anything other than an R script, it is also possible to generate a quick report from this R script using the stitch() function.

The basic idea of stitch() is that knitr provides a template of the source document with some default settings, so that the user only needs to feed this template with an R script (as one code chunk); then knitr will compile the template to a report. Currently it has built-in templates for LATEX, HTML, and Markdown. The usage is like this:

library(knitr) stitch("your-script.R")

###### 3.4 Extracting R Code

For a literate programming document, we can either compile it to a report (run the code), or extract the program code in it. They are called “weaving” and “tangling,” respectively. Apparently the function knit() is for weaving, and the corresponding tangling function is purl() in knitr. For example,

library(knitr) purl("your-file.Rnw") purl("your-file.Rmd")

The result of tangling is an R script; in the above examples, the default output will be your-ﬁle.R, which consists of all code chunks in the source document.

So far we have been introducing the command line usage of knitr, and it is often tedious to type the commands repeatedly. In the next chapter, we show how a decent editor can help edit and compile the source document with one single mouse click or a keyboard shortcut.

