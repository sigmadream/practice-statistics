### 11

###### Language Engines

We can work with a lot of languages and tools in knitr, including but not limited to R, although knitr is an R package and has to be run within the R environment in the ﬁrst place. Currently knitr supports Python, Ruby, Haskell, awk/gawk, sed, shell scripts, Perl, SAS, TikZ, Graphviz, and C++, etc. We have to install the corresponding software package in advance to use an engine.

###### 11.1 Design

Like chunk hooks, all language engines are essentially R functions in knitr. These functions pass the code chunk to external programs, run the code there, get the results back, and write to the output. In most cases, the code is passed to external programs via the system() function. For example, we can pass code to bash via the -c option.

system("bash -c  ls ~ | grep ^D ", intern = TRUE) ## [1] "Desktop" "Downloads" "Dropbox"

For those who are not familiar with bash scripts, the code ls ~ | grep ^D means to list ﬁles under the home directory (~) and pass the ﬁlenames to grep through the pipe (|) to match those starting with the letter D; ls and grep are standard Linux commands.

The chunk option engine can be used to specify the language engine for a chunk, e.g., the chunk below uses engine = ’bash’: ls ∼ | grep ^D

## Desktop ## Downloads ## Dropbox

111

Then the code in the chunk will be treated as a bash script instead of an R script. The output rendering is similar to R output: the source code is passed to the source hook (i.e., knit_hooks$get(’source’)), and the output is passed to the output hook (knit_hooks$get(’output’)). The built-in output hooks are fairly general in terms of document formats; we do not need to think about whether the output is to be LATEX or HTML or Markdown; everything will be automatically and properly marked up according to the output document format.

###### 11.1.1 The Engine Function

All language engines are stored in the object knit_engines, which has the $get() and $set() methods like knit_hooks (chunk hooks) and opts_chunk (chunk options); e.g., we can get the Python engine by knit_engines$get(’python’), or override the built-in Python engine by knit_engines$set(python = function(options) {...}).

An engine has one argument: options, which is a list of current chunk options. Among all options there is one special option named code, which is the code (as a character string) of the current chunk and plays the central role in the language engine.

To continue the bash example, we can deﬁne a preliminary engine like this:

knit_engines$set(bash = function(options) { code <- paste(options$code, collapse = "\n") out <- system(paste("bash -c", shQuote(code), sep = " "),

intern = TRUE)

paste(c(code, out), collapse = "\n") })

What this engine does is to concatenate the command bash -c with the source code, execute the whole command via system(), and return both the source code and output as one character string separated by line breaks. The returned character string will be written into the output document.

The real bash engine is more complicated than this: it has to take care of some chunk options such as echo, results, include, cache, and so on. For example, when echo = FALSE, the source code should be hidden, and when cache = TRUE, the code chunk should be cached. In all, the behavior of these language engines is very similar to the R engine, although the support is not as comprehensive as R.

Note in particular the cache of language engines other than R: in most cases, only the side effects such as printing are cached, due to the

fact that it is difﬁcult for R to know which objects are created in a code chunk if the code is not written in R. In other words, objects are lost when we exit from a chunk (unless they are exported to ﬁles). Normally we will not be able to reuse an object created from previous chunks. The reason that we can use R objects across different chunks is that all R chunks are evaluated in the same R session, but other languages are evaluated in separate sessions per chunk basis.

- 11.1.2 Engine Options For language engines, there are two common chunk options:


engine.path speciﬁes the full path to the engine program as a character string; this may be useful to Windows users when the program to be called is not in the environmental variable PATH (i.e., the program cannot be run without full path in the command line), or to Linux users when there are multiple versions of one program installed and we do not want to use the default version; in both cases, we can set the chunk option engine.path = ’full/path/to/program’, e.g., engine.path = ’/usr/bin/ruby1.9.1’ (if there are multiple versions of Ruby) or engine.path = ’C:/Program Files/SASHome/x86/9.3/sas.exe’ (to

specify the full path of SAS);

engine.opts additional options to be passed to an engine; its value depends on the speciﬁc engine; for most engines, it contains additional command line arguments, e.g., for engine = ’ruby’, we can set engine.opts = ’-v’ for Ruby to print its version number, then turn on the verbose mode.

###### 11.2 Languages and Tools

Most languages and tools are supported through the system() interface, as mentioned in the last section. There are a few exceptions, however, such as C++ and TikZ.

###### 11.2.1 C++

C++ is supported in knitr through the Rcpp package (Eddelbuettel et al., 2015). When we set engine = ’Rcpp’, the function sourceCpp() in Rcpp is used to compile C++ code chunks, which in fact calls R CMD

SHLIB internally to build a shared library and load it into R for future use.

Below is an example for the Fibonacci series (xi = xi−1 + xi−2, x0 = 0 and x1 = 1) in C++ with Rcpp: #include <Rcpp.h> // [[Rcpp::export]] int fibCpp(const int x) {

if (x == 0 || x == 1) return(x); return (fibCpp(x - 1)) + fibCpp(x - 2);

}

After it is compiled, we can call the function ﬁbCpp() in R directly because we have marked it with the Rcpp::export attribute. fibCpp(10L) ## [1] 55 system.time(fibCpp(27L))

## user system elapsed ## 0.001 0.000 0.001

Below is the version implemented in pure R: fibR <- function(x) {

if (x == 0L || x == 1L) return(x)

return(fibR(x - 1L) + fibR(x - 2L)) }

Unsurprisingly, the R version is much slower, although the numeric results are the same:

fibR(10L) ## [1] 55 system.time(fibR(27L))

## user system elapsed ## 0.708 0.000 0.708

Finally, we can pass additional arguments to sourceCpp() via the chunk option engine.opts. For example, we can specify engine.opts = list(showOutput = TRUE) to show the output of R CMD SHLIB (note showOutput is an argument of sourceCpp()).

###### 11.2.2 C/Fortran

There are two simple language engines c and fortran for the C language and Fortran, respectively. These engines are nothing but wrappers for the command R CMD SHLIB and the R function dyn.load(). What they do is to write the code chunk to a temporary ﬁle, run R CMD SHLIB to compile it, and use dyn.load() to load the compiled library (a .dll or .so ﬁle). To use these engines, you have to make sure you have the C/Fortran compilers in your system, such as GCC.

# the compilers in the environment in which this book # was written Sys.which("gcc")

## gcc ## "/usr/bin/gcc"

Sys.which("gfortran")

## gfortran ## "/usr/bin/gfortran"

Below are two examples demonstrating the usage of these two engines. First, we set the chunk option engine = ’c’ for this example: /* calculate the square of a number */ void my_square(double *x) {

*x = *x * *x; }

After compiling the above code chunk, we can call the C function my_square() via the .C() interface:

.C("my_square", 9)

## [[1]] ## [1] 81

.C("my_square", 123)

## [[1]] ## [1] 15129

Next, we show a Fortran example by setting the chunk option engine

= ’fortran’ for the chunk below: C Fortran test

subroutine fexp(n, x) double precision x

C output

integer n, i C input value

do 10 i = 1, n

x = dexp(dcos(dsin(dble(float(i)))))

###### 10 continue return end

And we can call the Fortran sub-routine via the .Fortran() interface:

res <- .Fortran("fexp", n = 100000L, x = 0) str(res)

## List of 2 ## $ n: int 100000 ## $ x: num 2.72

###### 11.2.3 Interpreted Languages

C++, C, and Fortran belong to compiled languages, and there are other languages that are interpreted languages. For these languages, we can execute the code without compiling it. Examples include awk and shell scripts. There are also some languages that belong to both categories, such as Python. Table 11.1 lists some interpreted languages supported by knitr via the system() interface.

For example, a Perl chunk is executed with perl -e code where code is the character string of the code chunk. For awk and sed, the argument after the program is treated as the source code, so they do not need an argument name for the code, e.g., awk ’END{print NR;}’ README counts the number of lines in the ﬁle README. For SAS, the code chunk is written into a ﬁle tempﬁle.sas, and executed as sas -SYSIN tempfile.sas. There are three shell variants: sh, bash, and zsh.

TABLE 11.1: Interpreted languages supported by knitr: the language name, engine name, and the command line argument to execute code.

|Language|Engine|Code argument<br><br>|
|---|---|---|
|Python|python<br><br>|-c|
|Ruby|ruby|-e<br><br>|
|(g)awk|(g)awk| |
|sed<br><br>|sed| |
|shell|sh/bash/zsh<br><br>|-c|
|Perl<br><br>|perl<br><br>|-e|
|Haskell|haskell|-e<br><br>|
|CoffeeScript|coffee<br><br>|-e|
|Groovy<br><br>|groovy<br><br>|-e|
|Node.js<br><br>|node|-e<br><br>|
|Scala|scala<br><br>|-e|
|SAS|sas<br><br>|-SYSIN|


As we mentioned before, the engine name itself may not be the executable, so we may need to specify the path to the real path of the program. For Haskell, haskell is not the program to run Haskell, whereas ghc is, so we need to specify both engine = ’haskell’ and engine.path = ’ghc’.

We give a few examples of the above languages. Here is a Python chunk (chunk option engine = ’python’): x = ’hello, python world!’ print x print x.split(’ ’)

## hello, python world! ## [ hello, ,  python ,  world! ]

Here is a Ruby chunk:

x = ’hello, ruby world!’ p x.split(’ ’)

## ["hello,", "ruby", "world!"]

Below is an awk script to count the number of non-empty lines in the NEWS.Rd ﬁle of the knitr package: in awk, NF denotes the number of ﬁelds on a line; when it is not 0, the variable i increases by 1, and that is why the script counts the non-empty lines in the ﬁle. Note that

we used engine.opts = shQuote(system.file(’NEWS.Rd’, package = ’knitr’)) for this chunk; i.e., we get the path to the NEWS.Rd ﬁle from R, quote it by shQuote(), and pass it to awk as the second argument (remember the ﬁrst argument is the code chunk), which means the ﬁle to be read into awk.

# how many non-empty lines in the NEWS file? NF {

i = i + 1

###### } END { print i }

## 8 Finally we have a Perl code chunk:

$test = "jello world"; $test =∼ s/j/h/; print $test

## hello world

###### 11.2.4 Stan

We can use the rstan package (Guo et al., 2014) to compile models of Stan, a relatively new programming language featuring Bayesian statistical inference. There is a language engine called stan in knitr that allows us to write Stan models in code chunks. We can certainly compile a Stan model in a normal R code chunk without using a special language engine, by saving the model as a ﬁle, or writing the model as a long character string in R code. Both ways have their disadvantages: it is not convenient for the reader to see the real model in the report if it is in an external ﬁle, and it is cumbersome to write a model as a long character string of multiple lines in R. The stan engine makes it possible to write the model as a code chunk, which solves both problems mentioned before. Here is a simple example of sampling from the posterior distribution of the parameter p (probability of X = 1) of a Bernoulli distribution:

<<engine= stan , engine.opts = list(x =  ex1 )>>= data {

int<lower=0,upper=1> X[20];

} parameters {

real<lower=0,upper=1> p;

} model {

###### X ∼ bernoulli(p);

} @

Besides the chunk option engine = ’stan’, we also speciﬁed the option engine.opts = list(x = ’ex1’). Here x means the name of the Stan model to be saved in the R session. This code chunk will pass the model to the function stan_model() in rstan, and save the model to the object ex1. That is why we can use the object ex1 in the next chunk:

library(rstan) fit <- sampling(ex1, data = list(X = rbinom(20, 1, 0.3)))

SAMPLING FOR MODEL  anon_model  NOW (CHAIN 1).

Iteration: 1 / 2000 [ 0%] (Warmup) Iteration: 200 / 2000 [ 10%] (Warmup) Iteration: 400 / 2000 [ 20%] (Warmup) Iteration: 600 / 2000 [ 30%] (Warmup) Iteration: 800 / 2000 [ 40%] (Warmup) Iteration: 1000 / 2000 [ 50%] (Warmup) Iteration: 1001 / 2000 [ 50%] (Sampling)

.... print(fit)

Inference for Stan model: anon_model. 4 chains, each with iter=2000; warmup=1000; thin=1; post-warmup draws per chain=1000, total post-warmup draws=4000.

mean se_mean sd 2.5% 25% 50% 75% p 0.36 0.00 0.10 0.18 0.29 0.36 0.43 lp__ -14.93 0.02 0.73 -16.99 -15.12 -14.65 -14.47

97.5% n_eff Rhat p 0.57 1498 1 lp__ -14.42 1703 1 ....

We generated 20 random data points from the Bernoulli distribution with p = 0.3, and used them as the sample data Y for the Bayesian inference. You can see from the sampling output that the posterior mean of p is near 0.3.

###### 11.2.5 TikZ

We introduced the tikzDevice package in Section 7.6, which enables us to convert R graphics to TikZ (Tantau, 2008). In fact, we can write raw TikZ code directly in knitr with the engine tikz.

What the tikz engine does internally is: use a LATEX template to insert the code chunk and compile the tex document to PDF. By default it uses the template in knitr (named tikz2pdf.tex under the misc directory in knitr’s installation directory):

- f <- system.file("misc", "tikz2pdf.tex", package = "knitr") cat(readLines(f), sep = "\n")


\documentclass{article} \include{preview} \usepackage[pdftex,active,tightpage]{preview} \usepackage{amsmath} \usepackage{tikz} \usetikzlibrary{matrix} \begin{document} \begin{preview} %% TIKZ_CODE %% \end{preview} \end{document}

The line %% TIKZ_CODE %% will be replaced by the TikZ code chunk. If the default template is not satisfactory, we can provide a template via the chunk option engine.opts, e.g., engine.opts = list(template = ’path/to/tikz/template.tex’). Then this TEX ﬁle is compiled to PDF via the R function tools::texi2pdf(). If the speciﬁed ﬁgure ﬁle extension (chunk option fig.ext) is not pdf, ImageMagick (via its convert utility) will be called to convert the PDF ﬁle to other ﬁle formats such as PNG, e.g., when the document format is HTML.

Figure 11.1 is a diagram drawn from raw TikZ code below:

\usetikzlibrary{arrows} \begin{tikzpicture}[node distance=2cm, auto,>=latex’, thick] \node (P) {$P$};

fˆ k

- Pˆ

f

g

f

- gˆ g


P B

A C

- FIGURE 11.1: A diagram drawn with TikZ: the source code is written into a *.tex ﬁle and compiled to PDF by LATEX.


- \node (B) [right of=P] {$B$}; \node (A) [below of=P] {$A$};
- \node (C) [below of=B] {$C$}; \node (P1) [node distance=1.4cm, left of=P, above of=P]


{$\hat{P}$}; \draw[->] (P) to node {$f$} (B); \draw[->] (P) to node [swap] {$g$} (A);

- \draw[->] (A) to node [swap] {$f$} (C);
- \draw[->] (B) to node {$g$} (C); \draw[->, bend right] (P1) to node [swap] {$\hat{g}$} (A); \draw[->, bend left] (P1) to node {$\hat{f}$} (B); \draw[->, dashed] (P1) to node {$k$} (P); \end{tikzpicture}


To develop tikz graphics, the programs qtikz or ktikz can be helpful, since they provide a graphical user interface (an editor), which allows one to preview the results.

###### 11.2.6 Graphviz

Graphviz (Ellson et al., 2002) is an open source and popular graph visualization software package (http://www.graphviz.org); it is powerful for drawing diagrams of abstract graphs and networks. Graphviz contains a few “ﬁlters,” such as dot, to draw directed graphs, and neato to draw undirected graphs. When engine = ’dot’, dot is used by default; to use other ﬁlters, we can set, e.g., engine.path = ’neato’.

Figure 11.2 is an example taken from the documentation of Graphviz.

a

|b|
|---|


x y

hi

multi-line label

hello world

z

- FIGURE 11.2: A diagram drawn with dot in Graphviz (taken from the dot manual).


We used fig.ext = ’pdf’ here to produce a PDF graph ﬁle, and we can change it to other ﬁle formats like PNG as well.

digraph test123 { a -> b -> c;

- a -> {x y};
- b [shape=box];
- c [label="hello\nworld",color=blue,fontsize=24, fontname="Palatino-Italic",fontcolor=red,style=filled];


- a -> z [label="hi", weight=100]; x -> z [label="multi-line\nlabel"]; edge [style=dashed,color=red];
- b -> x; {rank=same; b x}


}

If you want to draw diagrams in HTML documents generated from R Markdown, you may consider the DiagrammeR package (https: //github.com/rich-iannone/DiagrammeR), which is an HTML widget package that wraps a few JavaScript libraries (see Section 14.5.3 for more information about HTML widgets).

###### 11.2.7 Highlight

Highlight is a free and open source software package by Andre Simon (http://www.andre-simon.de) to do syntax highlighting for a large va-

riety of languages, including C, PHP, and R, etc. It can write the output in either LATEX or HTML.

When the chunk option engine = ’highlight’, the highlight program is called to generate the highlighted code chunk. The chunk option engine.opts is a character string to pass additional arguments to Highlight, e.g., we can specify the input syntax via -S, and the type of output via -O.

The chunk below was taken from the previous awk example; it uses the chunk option engine.opts = ’-S awk -O latex’ to tell Highlight that the input syntax is awk, and the output type is LATEX, so that Highlight can produce appropriate LATEX commands on keywords. It may be difﬁcult to see the colors in the printed version of this book, but at least we can see the ﬁrst line is italic (comments).

# how many non-empty lines in the NEWS file? NF {

i = i + 1

###### } END { print i }

Note that Highlight generates commands like \hlnum{} (for numbers) and \hlstr{} (for strings) to mark up different tokens in the code. These commands are mostly consistent with knitr’s syntax highlighting commands, but there are a few exceptions, e.g., \hlslc{} (for comments) produced by Highlight is not a part of knitr’s commands, so we need to deﬁne it in the LATEX preamble. Similarly, if the Highlight output is HTML, we need to deﬁne CSS styles for the class hl slc.

###### 11.2.8 Other Engines

There are two more engines that are essentially for any language: cat and asis. The cat engine calls the function cat() to write the code chunk to a ﬁle, and the ﬁlename can be provided in the chunk option engine.opts = list(file = ?). The asis engine does nothing but just write the code chunk as-is in the output. However, it respects the chunk options eval and echo: if either of these options is FALSE, the code chunk will be hidden from the output, which can be useful when you want to dynamically control whether to show some content in the output.

For example, we can write the code chunk below to a ﬁle named styles.css through the cat engine:

<<engine= cat , engine.opts = list(file =  styles.css )>>= p {

margin: 5px 2px 5px 2px;

} @

The following code chunk will be included in the ﬁnal output if the variable internal.only is TRUE (imagine you have a portion of the report content that you only want to show internally in your group):

<<engine= asis , echo = internal.only>>= Here are some top secrets about our analysis that are hidden in the public version of this report by setting  internal.only  to TRUE. Secret number one: ... @

###### 11.3 Persistent Sessions

In fact, there is a major ﬂaw in the engines for interpreted languages introduced before: a new engine session is established for every single code chunk of this engine. This means all code chunks are independent in memory, and the variables created in previous chunks will not be available in latter chunks. The only exception is R code chunks: all of them are evaluated in the same R session. To address this issue, we need to open a persistent session for an engine, and keep on running code chunks in this session. For example, we can create a variable in a Python code chunk, and continue using it in the next Python chunk.

The runr package (Xie, 2013) is an attempt to solve this problem. Currently it has experimental support for Bash and Julia code, based on socket connections. The basic idea is like this (take the Julia engine as example):

- 1. Open a background Julia process that starts a socket server and keeps listening (the background process is detached from


the current R session by system(’julia script.jl’, wait

= FALSE));

- 2. R connects to the Julia socket server via socketConnection(open

= ’w’), and writes the Julia code chunk to the server;

- 3. Julia receives the code, evaluates it, and writes the standard output (as plain text) to the socket;
- 4. R reads from the socket via socketConnection(open = ’r’), and writes the Julia output to the report just like R code chunk output;
- 5. Repeat steps 2–4 if the next Julia code chunk comes in, and Julia will quit if we send the code quit() to it.


In this way, the Julia session will be live until we explicitly shut it down from R, and all Julia code chunks will be evaluated in the same Julia session. The runr package is still at its early stage, and community contribution is welcome.

