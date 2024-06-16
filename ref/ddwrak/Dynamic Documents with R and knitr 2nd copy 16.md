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
