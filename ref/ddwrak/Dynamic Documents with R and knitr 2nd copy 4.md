### 4

###### Editors

We can write documents for knitr with any text editor, because these documents are plain text ﬁles. For example, lightweight editors like Notepad under Windows or Gedit under Linux will work. The main reasons that we need special text editors are

- 1. we want to input R code chunks more easily, e.g., input < <> >= and @ with a keyboard shortcut instead of typing these characters every time;
- 2. we wish to call R and knitr to compile source documents to PDF/HTML within an editor instead of opening R and typing the command knitr::knit(), and even better, to send R code chunks to R from within the editor directly.


There are many mature and nice editors for LATEX, HTML, and Markdown documents, and some have integrated knitr within them, as we will explain in the following sections.

###### 4.1 RStudio

RStudio is a relatively new editor specially targeted at R. It may be the best editor to start with for a beginner, since it has the most comprehensive support to Sweave and knitr. RStudio is cross-platform, free and open-source software; it is available at http://www.rstudio.com. Besides its excellent support for programming with R, it has a most notable feature that is missing in many other editors: it has a server version that looks identical to the desktop version, and we can use R in a Web browser after we have installed the server version on a Linux server.

The complete documentation can be found on the website. Here we only brieﬂy introduce the features related to dynamic documents. If you are going to write Rnw documents (LATEX), the ﬁrst thing to do to use knitr in RStudio is to change the option from the menu Tools

19

fig.FIGURE4.1:EditanRnwdocumentinRStudio:thereisauto-completioninsidethechunkheader(wetype“”

andwillseeallcandidates);thecodechunkcanbeeitherinsertedfromthemenuorakeyboardshortcut;thebutton

![image 3](Dynamic Documents with R and knitr 2nd_images/imageFile3.png)

CompilePDFsupports one-click generation of PDF from Rnw.

Options Sweave; the default option for weaving (i.e., compiling) Rnw documents is Sweave, and we can switch it to knitr, as long as we have installed knitr in R. For more discussion about knitr vs. Sweave, see Section 16.1. If you plan to work with other types of documents such as R Markdown, you do not need to conﬁgure any options, and RStudio will give you tips to install the required packages if they are missing.

All document formats supported by RStudio can be found under the menu File New. Currently they include R Sweave, R Markdown, and R HTML. For all document formats, there is one-click compilation support, i.e., we can click a button to compile a source document to the corresponding output format (LATEX to PDF, Markdown to HTML, and so on). We can input R code chunks with Ctrl + Alt + I; there is auto-completion of chunk options in the chunk header; e.g., if we type “fig.” between < < and > >= in an Rnw document, we will see possible candidates like fig.width, fig.height, and so on. The R code in chunks can be sent to the R console with Ctrl + Enter, just like what we do in a normal R script. In this way, we can run certain R code chunks interactively before we compile a whole document. Figure 4.1 is a screenshot of how an Rnw document looks in RStudio.

For an Rnw document, its ﬁnal output format is usually PDF (via

LATEX). RStudio provides synchronization between the PDF document and the source document, which implies these features:

- 1. forward search: we can navigate from one line in the source document to an appropriate location in the PDF document that corresponds to the source line;
- 2. inverse search: we can also click in the PDF document and RStudio can bring us back to the corresponding lines in the Rnw source;
- 3. error navigation: when an error occurs in R or LATEX, RStudio can bring us to a place in the source document that is the


source of the error; this can help us ﬁx problems in R or LATEX code more quickly.

For R Markdown documents, RStudio provides one-click compilation to a variety of formats, including HTML. Besides, it can also base64 encode images and render LATEX math expressions (through the MathJax library) in the HTML output. The former feature is to guarantee that the HTML page generated is self-contained, i.e., it does not depend on external images since they have been embedded in the page; the latter feature is especially useful for statisticians when they want to write math in a Web page.

The R Markdown (Rmd) format is fairly simple, and can be easily

KnitFIGURE 4.2:Edit an Rmd document in RStudio:there is also auto-completion for chunk option values; the button

![image 4](Dynamic Documents with R and knitr 2nd_images/imageFile4.png)

HTMLsupports one-click generation of an HTML page from Rmd.

mastered in ﬁve minutes. Due to its simplicity, there has been a huge number of reports written in this format and published on RPubs, a free platform provided by RStudio to host knitr reports from users. See http://rpubs.com for more examples. Figure 4.2 shows a sample Rmd document in RStudio.

We mentioned quick reporting in Section 3.3, and this is also supported in RStudio. For an R script in RStudio, we can create an “R Notebook” (a report purely based on an R script) from it by clicking the button on the toolbar.

###### 4.2 LYX

LYX is essentially a front-end for LATEX, which has a nice GUI to assist document writing. On screen, it looks like many word processors, but

- at its core, it is LATEX. One major difference between raw LATEX editors and LYX is that we only see \alpha + \beta in raw LATEX, whereas we see α + β in LYX, which is essentially \alpha + \beta behind the


screen. Everything is LATEX in LYX but our vision is not distorted by a full screen of backslashes.

Since version 2.0.3, LYX has started to support knitr as an ofﬁcial module. Details can be found at http://yihui.name/knitr/demo/lyx/. This module works in this way:

∗.tex LaTeX−→ ∗.pdf (weave) ∗.R (tangle)

∗.lyx −→LyX ∗.Rnw R+−→knitr

Note that currently Rnw is the only possible format to use in LYX. It seems we are mixing R code with LYX, but LYX is really only a wrapper so we are actually embedding R code in Rnw documents.

For Linux and Mac OS users, the usage of the module is:

- 1. create a new LYX document;
- 2. go to Document Settings Modules and insert the module named Rnw (knitr);
- 3. insert R code chunks into the document with Insert TEX Code, then start typing <<>>= and @ as usual.


Click the View button on the toolbar or press Ctrl + R to compile the document to PDF and view the results. We can also extract R code from a LYX document from the menu File Export R/S code. A screenshot of LYX with R code is shown in Figure 4.3.

ViewFIGURE 4.3:Usingin LX:R code is inserted in a red box using the Rnw syntax; when we click thebutton,knitrY

![image 5](Dynamic Documents with R and knitr 2nd_images/imageFile5.png)

we will see a PDF document compiled throughand LX.knitrY

There is one more step before we can use the knitr module under Windows: go to Tools Preferences Paths PATH preﬁx and add the bin path of R there, which is often like C:\Program Files\R\R-x.x.x\bin and you can ﬁnd it in R:

R.home("bin")

After you have made this change, you need to reconﬁgure LYX by Tools Reconﬁgure. This is to make sure LYX knows where R is installed so that it can call R and knitr to compile the Rnw document. Speciﬁcally, it needs to know where Rscript.exe is. If it is not present in PATH, the knitr module will be unavailable. This step is often not needed for Linux and Mac OS because these systems will put the R executable on PATH by default.

Although the graphical interface looks easy enough to use, we still strongly recommend users to master LATEX before trying LYX; otherwise it can be difﬁcult to diagnose LATEX problems when errors occur. LYX is not Word, after all.

###### 4.3 Emacs/ESS

ESS (Emacs Speaks Statistics) is an add-on package for the text editor Emacs (Rossini et al., 2004). It supports statistical software packages like R, S-Plus, SAS, JAGS, and so on. The support for knitr was added after version 12.09; before that, only Sweave was supported.

ESS is also free and open-source software; it is available at http: //ess.r-project.org. After it has been installed along with Emacs, it is fairly easy to call knitr in Emacs. The default option for Rnw documents is Sweave, and we can change it to knitr with the following commands (in Emacs key notation, M stands for the Meta key, which is the Alt key on most keyboards, and M-x means to hold Meta and press x):

M-x customize-group ess-R

Find the ess-swv-processor option and change it to knitr. Then we can create a new Rnw document, press M-n s to compile Rnw to TEX, and M-n P to compile TEX to PDF.

The support of Rmd documents and other document formats in ESS is still under development. According to the developers, this feature

may come in ESS 13.03, and readers can pay attention to their ofﬁcial announcement in the future.

###### 4.4 Other Editors

It is not hard to add support in other editors as long as they allow deﬁning custom commands to compile documents. Generally speaking, the custom command looks like:

Rscript -e "library(knitr); knit( input.ext )"

This command calls R to load the knitr package and compile the input document named input.ext using the function knit().

WinEdt (proprietary software) has a mode named R-Sweave to support knitr; and Tinn-R (free) has built-in support. It is also possible to conﬁgure other text editors such as Texmaker, Eclipse, TextMate, TEXShop, and Vim so that we can conveniently compile reports inside them. The conﬁguration instructions are collected at http://yihui. name/knitr/demo/editors/.

