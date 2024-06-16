### 5

###### Document Formats

The design of the knitr package is ﬂexible enough to process any plain text documents in theory. Below are the three key components of the design:

- 1. a source parser
- 2. a code evaluator
- 3. an output renderer


The parser parses the source document and identiﬁes computer code chunks as well as inline code from the document; the evaluator executes the code and returns results; the renderer formats the results from computing in an appropriate format, which will ﬁnally be combined with the original documentation.

The code evaluator is independent of the document format, whereas the parser and the renderer have to take the document format into consideration. The former corresponds to the input syntax, and the latter is related to the output syntax.

###### 5.1 Input Syntax

Regular expressions (Friedl, 2006, or see Wikipedia) are used to identify code blocks (chunks) and other elements such as inline code in a document. These regular expression patterns are stored in the all_patterns object in knitr. For example, the pattern for the beginning of a code chunk in an Rnw document is:

all_patterns$rnw$chunk.begin ## [1] "^\\s*<<(.*)>>=.*$"

In a regular expression, ^ means the beginning of a character string; \s* matches any number (including zero) of white spaces; .* matches

27

any number of any characters. This regular expression means “any white spaces in the beginning of the line + << + any characters + >>=,” therefore the lines below are possible chunk headers:

<<>>= <<foo>>= <<bar, echo=TRUE>>=

<<a=1, b=2>>=

And these are not valid chunk headers (< < does not appear in the beginning of the line in the ﬁrst one; there is only one > in the second one; = is missing in the third one):

hi<<>>= <<foo>= <<bar>>

Two more technical notes about the regular expression above:

- 1. \s denotes a white space in regular expressions, but in R we have to write double backslashes because \\ in an R string really means one backslash (the ﬁrst backslash acts as escaping the second character, which is also a backslash); the backslash as the escape character can be rather confusing to beginners, and the rule of thumb is, when you want a real backslash, you may need two backslashes;
- 2. the braces () in the regular expression group a series of characters so that we can extract them with back references, e.g., we extract the second group of characters from abbbc:


# [b]+ means to match  b  for one or more times gsub("(a)([b]+)(c)", "\\2", "abbbc")

## [1] "bbb"

We need to extract the chunk options in the chunk headers, and that is why we wrapped .* in () in the regular expression as < <(.*)> >=.

###### 5.1.1 Chunk Options

As mentioned in Chapter 3, we can write chunk options in the chunk header. The syntax for chunk options is almost exactly the same as the

syntax for function arguments in R. They are of the form

option = value

There is nothing to remember about this syntax due to the consistency with the syntax of R: as long as the option values are valid R code, they are valid to knitr. Besides constant values like echo = TRUE (a logical value) or out.width = ’\\linewidth’ (character string) or fig.height = 5 (a number), we can write arbitrary valid R code for chunk options, which makes a source document programmable. Here is a trivial example:

<<foo, eval=if (bar < 5) TRUE else FALSE>>=

Suppose bar is a numeric variable created in the source document before this chunk. We can pass an expression if (bar < 5) TRUE else FALSE to the option eval, which makes the option eval depend on the value of bar, and the consequence is we evaluate this chunk based on the value of bar (if it is greater than 5, the chunk will not be evaluated), i.e., we are able to selectively evaluate certain chunks. This example is supposed to show that we can write arbitrarily complicated R expressions in chunk options. In fact, it can be simpliﬁed to eval = bar <

- 5 since the expression bar < 5 normally returns TRUE or FALSE (unless bar is NA).


###### 5.1.2 Chunk Label

The only possible exception is the chunk label, which does not have to follow the syntax rule. In other words, it can be invalid R code. This is due to both historical reasons (Sweave convention) and laziness (avoid typing quotes). Strictly speaking, the chunk label, as a part of chunk options, should take a character value, hence it should be quoted, but in most cases, knitr can take care of the unquoted labels and quote them internally, even if the “objects” used in the label expression do not exist. Below are all valid ways to write chunk labels:

<<foo>>= <<foo-bar>>= <<foo_bar>>= <<"foo">>= << foo-bar >>= <<label="foo">>= <<echo=FALSE, label="foo-bar">>=

Chunk labels are supposed to be unique id’s in a document, and they are mainly used to generate external ﬁles such as images (Chapter 7) and cache ﬁles (Chapter 8). If two non-empty chunks have the same label, knitr will stop and emit an error message, because there is potential danger that the ﬁles generated from one chunk may override the other chunk. If we leave a chunk label empty, knitr will automatically generate a label of the form unnamed-chunk-i, where i is an incremental chunk number from 1, 2, 3, · · · .

###### 5.1.3 Global Options

Chunk options control every aspect of a code chunk, as we will see in more detail in Chapters 6 through 11. If there are certain options that are used commonly for most chunks, we can set them as global chunk options using the object opts_chunk. Global options are shared across all the following chunks after the location in which the options are set, and local options in the chunk header can override global options. For example, we set the option echo to FALSE globally:

opts_chunk$set(echo = FALSE)

Then for the two chunks below, echo will be FALSE and TRUE, respectively:

<<foo>>= 1+1 @ <<bar, echo=TRUE>>= rnorm(10) @

###### 5.1.4 Chunk Syntax

The original syntax of literate programming is actually this: use one marker to denote the beginning of computer code (< <> >=), and one marker to denote the beginning of the documentation (@). This has a subtle difference from what we introduced in Chapter 3. In the literate programming paradigm, this is what a source document may look like:

@ This is documentation. @

Another line of documentation. <<>>= 1 + 1 # some code <<>>= rnorm(10) # another code chunk @ More documentation.

In knitr syntax, we open and close code chunks instead of opening code chunks and opening documentation chunks. The reason why we dropped the traditional syntax is that in a report, the code chunks often appear less frequently than normal text, so we only focus on the syntax for code chunks. It also looks more intuitive that we are “embedding” code into a report. Based on the new syntax, this is also a legitimate fragment of a source document for knitr:

Documentation here. <<>>= 1+1 <<>>= rnorm(10) @ More documentation.

###### 5.2 Document Formats

We have been using the syntax of Rnw documents as examples. Next we are going to introduce how to write R code in other document formats; Table 5.1 is a summary of the syntax. Note that code chunks can be indented by any number of spaces in all document formats.

###### 5.2.1 Markdown

For an R Markdown (Rmd) document, we write code chunks between    {r} and , and inline R code is written in  r . Chunk options are written before the closing brace in the chunk header. Note that the inline R code is not allowed to contain backticks, e.g.,  r pi*2  is ﬁne, but  r  pi *2  is not; although  pi *2 is valid R code, the parser is unable to know the ﬁrst backtick is not for terminating the inline R code expression.

TABLE5.1:Asyntaxsummaryofalldocumentformats:RLTX,RMarkdown,RHTML,RreStructuredText,RAE

|inline| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
|end| | | | | | | | |
|start| | | | | |//<br><br>###|.| |
|format|R<br><br>R|md<br><br>htm|l| | | | | |


in.rcode*end.rcode--><!--rinlinex-->

.rcode*%end.rcode\rinline{x}

@\Sexpr{x}

<%x%>

}....:r:`x`

}````rx`

n.rcode*//end.rcode`rx`

gin.rcode*###.end.rcode@rx@

endinline

AsciiDoc, R Textile, and brew.

Markdown allows us to write using an easy-to-read, easy-to-write plain text format, then convert it to structurally valid XHTML or HTML. As long as one knows how to write emails, one can learn it in a few minutes: http://en.wikipedia.org/wiki/Markdown. Below is a short example:

# First level header ## Second level This is a paragraph. This is **bold**, and _italic_.

- - list item
- - list item


Backticks produce the  <code>  tag. This is [a link](url), and this is an ![image](url). A block of code ( <pre>  tag):

1 + 1 rnorm(10)

### Third level section title You can write an ordered list:

- 1. item 1
- 2. item 2


The original Markdown syntax was designed to be simple, so it is inevitable to have some restrictions in terms of an authoring environment, such as the ability to write tables, LATEX math expressions, or, bibliography. In some cases, such as writing a short homework assignment, we do not need complicated features, so Markdown should work reasonably well.

One problem of Markdown is its derivatives: there are a number of variants such as Pandoc’s Markdown (http://johnmacfarlane.net/ pandoc), Github Flavored Markdown (http://github.com), kramdown

(http://kramdown.rubyforge.org) and so on. These ﬂavors may have their own deﬁnitions of how to write certain elements (such as tables). CommonMark (http://commonmark.org) is an effort at deﬁning the Markdown syntax unambiguously, and Pandoc’s Markdown is compatible with the CommonMark standards. Besides, Pandoc is probably the most comprehensive tool for Markdown at the moment. It added many useful extensions to the original Markdown such as:

- 1. Fenced code blocks within a pair of three backticks;
- 2. LATEX math via either plain LATEX (for PDF output) or MathJax (http://mathjax.org, for HTML output), which allows us

to write math equations in Web pages using the LATEX syntax, i.e., $math$ or $$math$$;

- 3. Metadata for the document, e.g., the title, author, and date information;
- 4. Tables, with columns separated by white spaces or pipes;
- 5. Deﬁnition lists, footnotes, and citations, etc.


Below is how some of the extensions look:

--title: The Title of My Report author: Yihui Xie

--Write code under or indent by 4 spaces as usual.    r 1 + 1 rnorm(10)

Inline math: $\alpha + \beta$. Display style: $$f(x) = x^{2} + 1$$ A simple table from the citation [@joe2014]:

| id | age | sex | |:----|----:|:---:|

- | a | 49 | M |
- | b | 32 | F |


More importantly, Pandoc can convert Markdown to several other document formats, including PDF/LATEX, HTML, Word (Microsoft Word or OpenOfﬁce), and presentation slides (either LATEX beamer or HTML5 slides). The R package rmarkdown (Allaire et al., 2015a) is based on knitr and Pandoc, and contains a few commonly used output formats so users can quickly create reasonably beautiful output by default.

The rmarkdown package was introduced by the RStudio developers, so it is not surprising that the R Markdown document format is

best supported by RStudio. When we open or create an Rmd document in RStudio (File New R Markdown), we can see a wizard asking you which output format you want. We will cover R Markdown in detail in Chapter 14.

- 5.2.2 LATEX Markdown was primarily designed for the Web, and for more complicated typesetting purposes, LATEX may be preferred. For example, this book was written in LATEX. Oetiker et al. (1995) is a classic tutorial for

beginners to learn LATEX. The learning curve can be steep but it is rewarding if you care a lot about typesetting by yourself.

For LATEX documents, R code chunks are embedded between < <> >= and @, and inline R code is written in \Sexpr{}, as we have seen many times before.

- 5.2.3 HTML


HTML (Hyper-Text Markup Language) is the language behind Web pages; normally we do not see HTML code directly because the Web browser has parsed it and rendered the elements. For example, when we see bold texts, the source code might be <strong>bold</strong>. Most Web browsers can show the HTML source code; e.g., for Firefox and Google Chrome, we can press Ctrl + U to view the page source.

There is a large (but limited) number of tags in HTML to represent different elements in a page. HTML is like LATEX in the sense that we can have precise control over the typesetting by carefully organizing the tags/commands. The price to pay is that it may take a long time to write a document since there are many tags to type. That is why Markdown can be better for small-scale documents. Anyway, due to the fact that HTML has the full power, sometimes we have to use it. Below is an example of an HTML document:

<html> <head>

<title>This is an HTML page</title> </head> <body>

<p>This is a <em>paragraph</em>.</p> <div>A <code>div</code> layer.</div> <!-- I m a comment; you cannot see me. -->

</body> </html>

To write R code in an HTML document, we use the comment syntax of HTML, e.g.,

<!--begin.rcode test-html, eval=TRUE 1 + 1 rnorm(10) end.rcode-->

<p>And here is the value of pi: <!--rinline pi -->.</p>

###### 5.2.4 reStructuredText

We can also embed R code in a reStructuredText (reST) document (http: //docutils.sourceforge.net/rst.html), which is like Markdown but more powerful (and complicated accordingly). Below is an example of R code embedded in an R reST document:

A reST document for knitr =========================

This is a reStructuredText document (*.Rrst). Here is how we write R code for **knitr**:

.. {r test-rst, eval=TRUE} 1 + 1 rnorm(10)

.. .. The value of pi is :r: pi .

The Docutils system (written in Python) is often used to convert reST documents to HTML.

###### 5.2.5 AsciiDoc

AsciiDoc (http://en.wikipedia.org/wiki/AsciiDoc) is a plain-text document format that can be converted to multiple types of output, such as software documentation, articles, books, and HTML pages. Below is a minimal R AsciiDoc example for writing a book:

= The Book Title :author: A Knitter

== The first chapter Hello world! // begin.rcode test, eval=TRUE 1 + 1 rnorm(10) // end.rcode The value of pi is  r pi .

###### 5.2.6 Textile

Textile is yet another lightweight markup language, and it is usually converted to HTML. You can ﬁnd more information on the Wikipedia page http://en.wikipedia.org/wiki/Textile_(markup_language).

Here is an R Textile example demonstrating the syntax: h1. Knitting Textile Files Hello world! ###. begin.rcode test, tidy=FALSE if (1 + 1 == 2) {

 of course! 

} ###. end.rcode

And an inline expression @r 2*pi@.

###### 5.2.7 Customization

It is possible to deﬁne one’s own syntax to parse a source document. As we have seen before, the parsing is done through regular expressions. Internally, knitr uses the object knit_patterns to manage the regular expressions. For example, the three major patterns for this book are:

knit_patterns$get(

c("chunk.begin", "chunk.end", "inline.code") )

## $chunk.begin ## [1] "^\\s*<<(.*)>>=.*$" ## ## $chunk.end ## [1] "^\\s*@\\s*(%+.*|)$" ## ## $inline.code ## [1] "\\\\Sexpr\\{([^}]+)\\}"

To specify our own syntax, we can use knit_patterns$set(), which will override the default syntax, e.g., knit_patterns$set(

chunk.begin = "^<<r(.*)", chunk.end = "^r>>$", inline.code = "\\{\\{([^}]+)\\}\\}"

)

Then we will be able to parse a document like this with the custom syntax:

<<r test-syntax, eval=TRUE 1 + 1 x <- rnorm(10) r>>

The mean of x is {{mean(x)}}.

In practice, however, this kind of customization is often unnecessary. It is better to follow the default syntax, otherwise additional instructions will be required in order to compile a source document.

There is a series of functions with the preﬁx pat_ in knitr, which are convenience functions to set up the syntax patterns, e.g., pat_rnw() calls knit_hooks$set() to set patterns for Rnw documents. All pattern functions include:

grep("^pat_", ls("package:knitr"), value = TRUE) ## [1] "pat_asciidoc" "pat_brew" "pat_html"

- ## [4] "pat_md" "pat_rnw" "pat_rst" ## [7] "pat_tex" "pat_textile"


When parsing a source document, knitr will ﬁrst decide which pattern list to use according to the ﬁlename extension; e.g., *.Rmd documents use the R Markdown syntax. If the ﬁle extension is unknown,

knitr will further detect the code chunks in the document and see if the syntax matches with any existing pattern list; if it does, that pattern list will be used; e.g., for a ﬁle foo.txt, the extension txt is unknown to knitr, but if this ﬁle contains a code chunk that begins with    {r}, knitr will use the R Markdown syntax automatically.

###### 5.3 Output Renderers

The evaluate package (Wickham, 2015) is used to execute code chunks, and the eval() function in base R is used to execute inline R code. The latter is easy to understand and made possible by the power of “computing on the language” (R Core Team, 2014) of R. Suppose we have a code fragment 1+1 as a character string; we can parse and evaluate it as R code:

eval(parse(text = "1+1")) ## [1] 2

For code chunks, it is more complicated. The evaluate package takes a piece of R source code, evaluates it, and returns a list containing results of six possible classes: character (normal text output), source (source code), warning, message, error, and recordedplot (plots).

In order to write these results into the output, we have to take the output format into consideration. For example, if the source code is 1+1 and the output format is TEX, we may use the verbatim environment, whereas if the output is supposed to be HTML, we may write <pre>1+1</pre> into the output instead. The key question is, how should we wrap up the raw results from R? This is answered by the knit_hooks object, which contains a list of output hook functions to construct the ﬁnal output. A hook function is often deﬁned in this form:

hook_fun <- function(x, options) {

# returns a character string with markup }

In an output hook, x is usually the raw output from R, and options is a list of current chunk options. The hook names in knit_hooks corresponding to the output classes are listed in Table 5.2.

If we want to put the message output (emitted from message() func-

TABLE 5.2: Output hook functions and the object classes of results from the evaluate package.

|Class<br><br>|Output hook|Arguments<br><br>|
|---|---|---|
|source|source<br><br>|x, options|
|character|output|x, options<br><br>|
|recordedplot<br><br>|plot|x, options<br><br>|
|message<br><br>|message<br><br>|x, options|
|warning<br><br>|warning|x, options|
|error|error<br><br>|x, options|
| |chunk<br><br>|x, options|
| |inline<br><br>|x|
| |text<br><br>|x|
| |document<br><br>|x|


tion) into a custom LATEX environment, say, Rmessage, we can set the message hook as:

knit_hooks$set(message = function(x, options) {

paste0("\\begin{Rmessage}\n", x, "\\end{Rmessage}") })

Of course, we have to deﬁne the Rmessage environment in advance in the LATEX preamble, e.g., \newenvironment{Rmessage}{

\rule[0.5ex]{1\columnwidth}{1pt} % a horizontal line }{

\rule[0.5ex]{1\columnwidth}{1pt} }

Then, whenever we have a message in the output, we will see a horizontal line above and below it.

By default, knitr will set up a series of default output hooks for each output format, so normally we do not have to set up all the hooks by ourselves. A series of functions with the preﬁx render_ in knitr can be used to set up default output hooks for various output formats:

grep("^render_", ls("package:knitr"), value = TRUE)

## [1] "render_asciidoc" "render_html" ## [3] "render_jekyll" "render_latex"

- ## [5] "render_listings" "render_markdown" ## [7] "render_rst" "render_sweave" ## [9] "render_textile"


|This is all you need to do if you want to go back to the Sweave style: The quick brown fox jumps over the lazy dog the quick brown fox jumps<br><br>over the lazy dog the quick brown fox jumps over the lazy dog. > 1 + 1 [1] 2 > rnorm(30)<br><br>[1] -0.56048 -0.23018 1.55871 0.07051 0.12929 1.71506 0.46092 [8] -1.26506 -0.68685 -0.44566 1.22408 0.35981 0.40077 0.11068<br><br>[15] -0.55584 1.78691 0.49785 -1.96662 0.70136 -0.47279 -1.06782 [22] -0.21797 -1.02600 -0.72889 -0.62504 -1.68669 0.83779 0.15337 [29] -1.13814 1.25381<br><br>The quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog.|
|---|


- FIGURE 5.1: The Sweave style in knitr: if we run render_sweave() in the beginning of an Rnw document, we will see the Sweave style.


The functions render_latex(), render_html(), and render_markdown() are called when the output formats are LATEX, HTML, and Markdown, respectively; render_sweave() and render_listings() are two variants of LATEX output — the former uses the traditional Sweave environments deﬁned in Sweave.sty (e.g., Sinput and Soutput, etc.), and the latter uses the listings package in LATEX to decorate the output. See Figure 5.1 and Figure

- 5.2 for how the two styles look. Note that if we want to set up the output hooks, it is better to do


it in the very beginning of a source document so that the rest of the output can be affected. For example, the chunk below can be the ﬁrst chunk of an Rnw document (the chunk option include = FALSE means do not show anything from this chunk in the output because it is not interesting to the readers):

<<setup, include=FALSE>>= render_sweave() @

Then the output will be rendered in the Sweave style. This book

used the default LATEX style, which supports syntax highlighting, and code chunks are put in gray shaded boxes.

1

Among all output hooks in Table 5.2, there are ﬁve special hooks that need further explanation:

|This is all you need to do if you want to use the listings package: The quick brown fox jumps over the lazy dog the quick brown fox jumps<br><br>over the lazy dog the quick brown fox jumps over the lazy dog.<br><br>1 + 1<br><br>|[1] 2|
|---|
<br><br>rnorm (30)<br><br>|[1] -0.56048 -0.23018 1.55871 0.07051 0.12929 1.71506 0.46092 [8] -1.26506 -0.68685 -0.44566 1.22408 0.35981 0.40077 0.11068<br><br>[15] -0.55584 1.78691 0.49785 -1.96662 0.70136 -0.47279 -1.06782 [22] -0.21797 -1.02600 -0.72889 -0.62504 -1.68669 0.83779 0.15337 [29] -1.13814 1.25381<br><br>|
|---|
<br><br>The quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog.|
|---|


- FIGURE 5.2: The listings style in knitr: render_listings() produces a style like this (colored text and gray shading).


- • the plot hook takes the ﬁlename as input x which is a character string of the ﬁlename (e.g., foo.pdf); below is a simpliﬁed version of the plot

hook for LATEX output (the actual hook is much more complicated than this, because there are many chunk options to take into account, such as out.width and out.height, etc.)

knit_hooks$set(plot = function(x, options) {

paste("\\includegraphics{", x, "}", sep = "") })

- • the chunk hook takes the output of the whole chunk as input, which is generated from other hooks such as source, output, and message, etc.; for example, if we want to put the chunk output in a div tag with the class Rchunk in HTML, we can deﬁne the chunk hook as:

knit_hooks$set(chunk = function(x, options) {

paste("<div class= Rchunk >", x, "</div>") })

then we need to deﬁne the style of Rchunk in the CSS stylesheet for this HTML document;

- • the inline hook is not associated with a code chunk; it deﬁnes how to format the output from inline R code. For example, we may want


1

to round all the numbers from inline output to 2 digits and we can deﬁne the inline hook as:

knit_hooks$set(inline = function(x) {

###### if (is.numeric(x))

x <- round(x, 2)

as.character(x) # convert x to character and return })

knitr takes care of rounding in the default inline hook (Section 6.1), so we do not really have to reset this hook;

- • the text hook processes text chunks, i.e., narratives; for example, we set up a hook to trim the white spaces around the text chunks:

knit_hooks$set(text = function(x) {

gsub("^\\s*|\\s*$", "", x) })

- • the document hook is similar to the chunk hook, and it takes the output of the whole document as input x; this hook can be useful for postprocessing the document; in fact, this book used this hook to add a vertical space \medskip{} under all table captions (before the tabular environment):


knit_hooks$set(document = function(x) {

gsub("\\begin{tabular}", "\\medskip{}\\begin{tabular}",

x, fixed = TRUE) })

###### 5.4 R Scripts

There is a special source document format in knitr, which is essentially an R script with roxygen comments (for more on roxygen, see Wickham et al. (2015) and Appendix A.1). We know a normal R comment starts with #, and a roxygen comment has an apostrophe after #, e.g.,

#  this is a roxygen comment ##  me too

Sometimes we do not want to mix R code with normal text, but write text in comments, so that the whole document is a valid R script. The function spin() in knitr can deal with such R scripts if the comments are written using the roxygen syntax. The basic idea of spin() is also inspired by literate programming: when we compile this R script, #  will be removed so that normal text is “restored,” and R code will be evaluated. Anything that is not behind a roxygen comment is treated as a code chunk. To write chunk options, we can use another type of special comment #+ or #- followed by chunk options. Below is a simple example:

#  Introduce the method here; then write R code: 1 + 1 x <- rnorm(10)

#  It is also possible to write chunk options, e.g., #  #+ test-label, fig.height=4 plot(x) #  The document is done now.

We can save this script to a ﬁle called test.R, and compile it to a report:

library(knitr) spin("test.R")

The spin() function has a format argument that speciﬁes the output document format (default to R Markdown). For example, if format = ’Rnw’, the R code will ﬁrst be inserted between < <> >= and @, and then compiled to generate LATEX output.

This looks similar to the stitch() function in Section 3.3, which also creates a report based on an R script, but spin() makes it possible to write text chunks and stitch() can only use a predeﬁned template, so there is less freedom.

