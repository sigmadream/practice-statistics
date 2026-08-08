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

