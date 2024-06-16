# Appendix A. Python and R Basics

This appendix provides details on the basics of Python and R. For readers unfamiliar with either language, we recommend you spend an hour or two working through basic tutorials such as those listed on the project’s home pages listed in <a href="#sec-local" data-type="xref">“Local Installation”</a>.

###### Tip

In both Python and R, you may access built-in help files for a function by typing **`help("function")`**. For example, typing `help("+")` loads a help file for the addition operator. In R, you can often also use the `?` shortcut (for example `?"+"`), but you need to use quotes for some operators such as `+` or `-`.

# Obtaining Python and R

Two options exist for running Python or R. You may install the program locally on a computer or may use a cloud-hosted version. From the arrival of the *personal computer* in the late-1980s up until the 2010s, almost all computing was local. People downloaded and installed software to their local computer unless their employer hosted (and paid for) cloud access or local servers. Most people still use this approach and use local computing rather than the cloud. More recently, consumer-scale cloud tools have become readily available. Trade-offs exist for using both approaches:

- Benefits of using a personal computer:

  - You have known, up-front, one-time costs for hardware.

  - You control all hardware and software (including storage devices).

  - You can install anything you want.

  - Your program will run without internet connections, such as during internet outages, in remote locations, or while flying.

- Benefits of using consumer-scale cloud-based computing:

  - You pay for only the computing resources you need.

  - Many free consumer services exist.

  - You do not need expensive hardware; for example, you could use a Chromebook or even tablets such as an iPad.

  - Cloud-based computing is cheaper to upscale when you need more resources.

  - Depending on platform and support levels, you’re not responsible for installing programs.

# Local Installation

Directions for installing Python and R appear on each project’s respective pages:

- [Python](https://www.python.org)

- [R Project](https://www.r-project.org)

We do not include directions here because the programs may change and depend on your specific operating systems.

## Cloud-Based Options

Many vendors offer cloud-based Python and R. Vendors in 2023 include the following:

- [Google Colab](https://colab.research.google.com)

- [Posit Cloud](https://posit.cloud)

- [DataCamp Workspaces](https://www.datacamp.com/workspace)

- [O’Reilly learning platform](https://learning.oreilly.com/home)

# Scripts

*Script files* (or *scripts*, for short) allow you to save your work for future applications of your code. Although our examples show typing directly in the terminal, typing in the terminal is not effective or easy. These files typically have an ending specific for the language. Python files end in *.py*, and R files end in *.R*.

During the course of this book, you will be using Python and R interactively. However, some people also run these languages as batch files. A *batch file* simply tells the computer to run an entire file and then spits out the outputs from the file. An example batch file might calculate summary statistics that are run weekly during the NFL season by a company like PFF and then placed into client reports.

In addition to batch files, some people have large amounts of text needed to describe their code or code outputs. Other formats often work better than script files for these applications. For example, Jupyter Notebook allows you to create, embed, and use runnable code with easy-to-read documents. The *Notebooks* in the name “Jupyter Notebook” suggests a similarity to a scientist’s laboratory or field notebook, where text and observations become intermingled.

Likewise, Markdown-based files (such as Quarto or the older R Markdown files) allow you to embed static code and code outputs into documents. We use both for our work. When we want other people to be readily able to interactively run our code, we use Jupyter Notebook. When we want to create static reports such as project files for NFL teams or scientific articles, we use R Markdown-based workflows.

Early drafts of this book were written in Jupyter Notebook because we were not aware of O’Reilly’s support for Quarto-based workflows. Later drafts were written with Quarto because we (as authors) know this language better. Additionally, Quarto worked better with both Python and R in the same document. Hence we, as authors, adapted our tool choice based on our publisher’s systems, which illustrates the importance of flexibility in your toolbox.

###### Warning

Many programs, such as Microsoft Word, use “smart quotes” and other text formatting. Because of this, be wary of any code copied over from files that are not plain-text files such as PDFs, Word documents, or web pages.

# Packages in Python and R

Both Python and R succeed and thrive as languages because they can be extended via packages. Python has multiple package-management systems, which can be confusing for novice and advanced users. Novice users often don’t know which system to use, whereas advanced users run into conflicting versions of packages across systems.

The simplest method to install Python packages is probably from the terminal outside of Python, using the `pip` command. For example, you may install `seaborn` by using `pip install seaborn`. R currently has only one major repository of interest to readers of this book, the Comprehensive R Archive Network (CRAN); a second repository, Bioconductor, mainly supports bioinformatics projects. An R package may be installed inside R by using the `install.packages()` function. For example, you may install `ggplot2` with `install.packages("ggplot2")`.

You may also, at some point, find yourself needing to install packages from sites like GitHub because the package is either not on CRAN or you need the development version to fix a bug that is blocking you from using your code. Although these installation methods are beyond the scope of this book, you can easily find them by using a search engine. Furthermore, environment-management software exists that allows you to lock down package versions. We discussed these in <a href="ch09.html#sec-advanced-tool" data-type="xref">Chapter 9</a>.

# nflfastR and nfl_data_py Tips

This section provides our specific observations about the `nflfastR` and `nfl_data_py` packages. First, we (the authors) love these packages. They are great free resources of data.

###### Tip

Update the `nflfastR` and `nfl_data_py` packages on a regular basis throughout the season as more data is added from new games. That will give you access to the newest data.

You may have noticed that you have to do a fair amount of wrangling with data. That’s not a downfall of these packages. Instead, this shows the flexibility and depth of these data sources. We also have you clean up the datasets, rather than provide you with clean datasets, because we want you to learn how to clean up your data. Also, these datasets are updated nightly during the football season, so, by showing you how to use these packages, we give you the tools to obtain your own data.

To learn more about both packages, we encourage you to dive into their details. Once you’ve looked over the basic functions and structure of the packages, you may even want to look at the code and “peek under the hood.” Both [`nflfastR`](https://oreil.ly/B6qiV) and [`nfl_data_py`](https://oreil.ly/ss6d_) are on GitHub. This site allows you to report issues and bugs, and to suggest changes—ideally by providing your own code!

Lastly, we encourage you to give back to the open source community that supplies these tools. Although not everyone can contribute code, helping with other tasks like documentation may be more accessible.

# Integrated Development Environments

You can use powerful code-editing tools called *integrated development environments* (*IDEs*). Much like football fans fight over who is the best quarterback of all time, programmers often argue over which IDEs are best. Although powerful (for example, IDEs often include syntax checkers similar to a spellchecker and autocompletion tools), IDEs can have downsides.

Some IDEs are complex, which can be great for expert users, but overwhelming for beginners and casual users. For example, the Emacs text editor has been jokingly described as an operating system with a good text editor or two built into it. Others, such as the web comic [“xkcd,” poke fun at IDEs such as emacs as well](https://xkcd.com/378). Likewise, some professional programmers feel that the shortcuts built into some IDEs limit or constrain understanding of languages because they do not require the programmer to have as deep of an understanding of the language they are working in.

However, for most users, especially casual users, the benefits of IDEs far outweigh the downsides. If you already use another IDE for a different language at work or elsewhere, that IDE likely works with Python and possibly R as well.

When writing this book, we used different editors at different times and for different file types. Editors we used included RStudio Desktop, JupyterLab, Visual Studio Code, and Emacs. Many good IDEs exist, including many we do not list. Some options include the following:

- [Microsoft Visual Studio Code](https://code.visualstudio.com)

- [Posit RStudio Desktop](https://posit.co)

- [JetBrain PyCharm](https://www.jetbrains.com/pycharm)

- [Project Jupyter JupyterLab](https://jupyter.org)

- [GNU Project Emacs](https://www.gnu.org/software/emacs)

# Basic Python Data Types

As a crash course in Python, various types of objects exist within the language. The most basic types, at least for this book, include *integers*, *floating-point numbers*, *logical*, *strings*, *lists*, and *dictionaries*. In general, Python takes care of thinking about these data types for you and will usually change the type of number for you.

*Integers* (or *ints* for short) are whole numbers like 1, 2, and 3. Sometimes, you need integers to index objects (for example, taking the second element, for a vector `x` by using `x[1]`; recall that Python starts counting with `0`). For example, you can create an integer `x` in Python:

```
## Python
x = 1
x.__class__
```

Resulting in:

\<class 'int'\>

*Floating-point numbers* (or *floats*, for short) are decimal numbers that computers keep track of to a finite number of digits, such as `float16`. Python will turn ints into floats when needed, as shown here:

```
## Python
y = x/2
y.__class__
```

Resulting in:

\<class 'float'\>

Computers cannot remember all digits for a number (for example, `float16` keeps track of only 16 digits), and computer numbers are not mathematical numbers. Consider the parlor trick Richard’s math co-advisor taught him in grad school:

```
## Python
1 + 1 + 1 == 3
```

Resulting in:

True

But, this does not always work:

```
## Python
0.1 + 0.1 + 0.1 == 0.3
```

Resulting in:

False

Why does this not work? The computer chip has a rounding error that occurs when calculating numbers. Hence, the computer “fails” at mathematical addition.

Python also has `True` and `False` logical operators that are called *Boolean objects*, or *bools*. The previous examples showed how bools result from logical operators.

Letters are stored as *strings* (or *str*, for short). Numbers stored as strings do not behave like numbers. For example, look at this:

```
## Python
a = "my two"
c = 2
a * c
```

Resulting in:

'my twomy two'

Python has groups of values. Lists may be created using `list()` or `[]`:

```
## Python
my_list = [1, 2, 3]
my_list_2 = list(("a", "b", "c"))
```

Dictionaries may also store values with a *key* and are created with `dict()` or `{}`. We use the `{}` almost exclusively but show the other method for completeness. For example:

```
## Python
my_dict = dict([('fred', 2), ('sally', 5), ('joe', 3)])
my_dict_2 = {"a": 0, "b": 1}
```

Additionally, the `numpy` package allows data to be stored in arrays, and the `pandas` package allows data to be stored in dataframes. Arrays must be the same data type, but dataframes may have mixed columns (such as one column of numbers and a second column of names).

# Basic R Data Types

As a crash course in R, multiple types of objects exist within the language, as in Python. The most basic types, as least for this book, include *integers*, *numeric floating-point numbers*, *logical*, *characters*, and *lists*. In general, R takes care of thinking about these structures for you and will usually change the type for you. R uses slightly different names than Python; hence this section is similar to <a href="#sec-py-datatypes" data-type="xref">“Basic Python Data Types”</a>, but slightly different.

*Integers* are whole numbers like 1, 2, and 3. Sometimes, you need integers to index objects—for example, taking the second element for a vector `x` by using `x[2]`; recall that R starts counting with `1`. For example, you can create an integer `x` in R by using `L`:

```
## R
x <- 1L
class(x)
```

Resulting in:

\[1\] "integer"

*Numerical floating-point numbers* (or *numeric*, for short) are decimal numbers that computers keep track of to a finite number of digits. R will turn integers into numerics when needed, as shown here:

```
## R
y <- x / 2
class(y)
```

Resulting in:

\[1\] "numeric"

Computers cannot remember all digits for a number (for example, `float16` keeps track of only 16 digits), and computer numbers are not mathematical numbers. Consider the parlor trick Richard’s math co-advisor taught him in grad school:

```
## R
1 + 1 + 1 == 3
```

Resulting in:

\[1\] TRUE

But this does not always work:

```
## R
0.1 + 0.1 + 0.1 == 0.3
```

Resulting in:

\[1\] FALSE

This is due to a rounding error that occurs on the computer chip calculating the answer.

R also has `TRUE` and `FALSE` logical operators that are called *logical* for short. The previous examples showed how logical outputs result from logical operators.

Letters are stored as *characters* in R. These do not work with numeric operators. For example, look at the following:

```
## R
a <- "my two"
c <- 2
a * c
```

R has groups of values called *lists* or *vectors*. Lists may be created by using a *combine* or *concatenate* function. For example:

```
## R
my_list <- c(1, 2, 3)
```

Additionally, base R contains matrices to store numbers and dataframes to store mixed columns (such as one column of numbers and a second column of names). We use tibbles from the `tidyverse` in this book as an upgraded version of dataframes.
