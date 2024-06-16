### 9

###### Cross Reference

We can cross reference both code chunks and child documents in knitr. This enables us to better organize our source documents. Below is a practical example: we have a custom ggplot2 theme and we want to apply it to a few plots in the document.

<<my-theme, eval=FALSE>>= theme(legend.text = element_text(size = 12, angle = 45)) +

theme(legend.position = "bottom") @

If we were to use this piece of code only once, we can just copy and paste it to the code chunk, but it is certainly not a good idea to paste it to multiple chunks, since it will be a disaster to maintain. We can simply use a reference to it using its chunk label, e.g.,

qplot(carat, price, data = diamonds, color = cut) +

<<my-theme>>

Then knitr will expand < <my-theme> > to the real source code before evaluating this chunk. We can use this reference in multiple places but only maintain one copy of the source.

###### 9.1 Chunk Reference

With chunk references, we can easily reuse code chunks without typing them again. We can embed a deﬁned chunk into another chunk, or just reuse a whole chunk as a new chunk.

###### 9.1.1 Embed Code Chunks

One chunk can be used as a part of another chunk, and the syntax is < <label> > (white spaces are allowed before it; label means the chunk

91

label); note there is no = after > > like chunk headers. For example, we embed chunk A in B:

- <<A>>= x <- rnorm(1) @
- <<B>>= x <<A>> x @


In this case, chunk B is essentially this (< <A> > is replaced by the code in chunk A but note all chunk options in A are ignored, including eval):

x x <- rnorm(1) x

Chunks can be nested recursively within each other as long as the recursion is ﬁnite, e.g., we embed A into B, and B into C, but we must not embed C into A again, otherwise there will be inﬁnite recursion.

###### 9.1.2 Reuse Whole Chunks

There are two ways to reuse a whole chunk. The ﬁrst one is to use the same label but leave the chunk empty. One problem with this approach is that we cannot cache both chunks if their chunk options are different because their MD5 hashes will be different, and knitr only allows one set of cache ﬁles per label. Here is one example:

<<chunkA, eval=FALSE>>= x <- 1 + 1 @ <<chunkA, eval=TRUE>>= @

The second approach is to use the ref.label option, which takes a vector of the chunk labels of source chunks. We can use a new label for the target chunk. In the following example, chunk C uses code from both A and B:

- <<A>>=

- x <- rnorm(1) @

<<B>>=

- y <- x + 2 @




- <<C, ref.label=c( A ,  B )>>= @


The code for chunk C is essentially this:

- x <- rnorm(1)
- y <- x + 2


###### 9.2 Code Externalization

It can be more convenient to write R code chunks in a separate R script, rather than mixing them into a source document; for example, we can run R code successively in a pure R script from one chunk to the other without jumping through other text.

The other reason is that some editors such as LYX do not have support to run R code interactively, and we have to recompile the whole document each time, even if we only want to know the results of a single chunk.

Therefore knitr introduced the feature of code externalization: code chunks can be read from an external R script via read_chunk(). The R script can be written in two forms: we either use labels in the script to separate code chunks, or specify chunks based on line numbers.

###### 9.2.1 Labeled Chunks

The setting is like this: the R script also uses chunk labels (marked in the form ## ---- chunk-label); if the code chunk in the source document is empty, knitr will match its label with the label in the R script to input external R code.

For example, suppose this is a code chunk labelled as Q1 in an R script named shared.R, which is under the same directory as the source document:

## ---- Q1 ---gcd <- function(m, n) {

while ((r <- m%%n) != 0) {

- m <- n
- n <- r


} n

}

In the source document, we can ﬁrst read the script using the function read_chunk():

read_chunk("shared.R")

This is usually done in an early chunk such as the ﬁrst chunk of a document, and we can use the chunk Q1 later in the source document:

<<Q1>>= @

###### 9.2.2 Line-Based Chunks

By default, read_chunk() assumes that the R script is labeled (## ---is the delimiter), and there is an alternative approach to specify code chunks via the three arguments labels, from, and to, which are vectors of the same length. The starting and ending line numbers of code chunks can be set through from and to, respectively, and labels is a vector of chunk labels.

For example, if we want the lines 1-5, 7-9, and 15-21 in the R script foo.R to form three chunks with labels A, B, and C, we can call the function read_chunk() like this:

read_chunk("foo.R", labels = c("A", "B", "C"), from = c(1,

7, 15), to = c(5, 9, 21))

Then we can write three empty chunks in the source document, with labels A, B, and C. Alternatively, from and to can be regular expressions for the starting and ending lines.

Different documents can read the same R script, so the R code can be reusable across different input documents.

###### 9.3 Child Documents

The concept of child documents should be familiar to LATEX users when the main document is large, we can split it into smaller parts and input them into the main document using \input{foo.tex}. For example, a book can be split into chapters, with each chapter in one ﬁle.

###### 9.3.1 Input Child Documents

Similarly, we can manage a knitr source document as a collection of child documents. The chunk option child provides a reference to child documents. Suppose we have a main document named book.Rnw, and a child document named chap1.Rnw under the same directory. In the main document, we have:

Here is one chunk in the main document.

- <<A, eval=TRUE>>=

- x <- rnorm(12) @ We include a child document which uses the variable x.

<<B, child= chapt1.Rnw >>= @

One realization of a Chi-square random variable with df 12 is \Sexpr{y}.

We referenced the child document in chunk B. When the main document is compiled, knitr will look for the child document and compile it accordingly; everything in the environment of the main document up to this point will be available to the child document, e.g., the variable x. The child document is:

This is a child document. <<B1>>=

- y <- sum(x^2) @




We created a new object y in the child document; after the child document has been compiled, it will be available to the later chunks in

the main document as well. That is why \Sexpr{y} will work. As a side note, the sum of n i.i.d standard Normal random variables follows the χ2n distribution (with n degrees of freedom), so y is one random number generated from χ212.

Like chunk references, child documents have no limits on the levels of nesting. One child document can have further children documents, and one chunk can include more than one child document.

###### 9.3.2 Child Documents as Templates

It is common to do the same analysis using a template with different data input, and child documents can be helpful for such tasks as well. As a trivial example, we continue to generate another random number from the Chi-square distribution in the main document:

% second part of book.Rnw Continue the above example. Now we change the degrees of freedom to 8.

- <<C, eval=TRUE>>= x <- rnorm(8) @ And include the child document again.
- <<D, child= chapt1.Rnw >>= @


One realization of a Chi-square random variable with df 8 is \Sexpr{y}.

What the child document does here is only to calculate the sum of squares for x and assign the result to y. It is very similar to a subroutine, even though it is not “pure source code” as we usually see.

With chunk references and child documents, we can modularize an analysis in the same manner of programming.

###### 9.3.3 Standalone Mode

This section is speciﬁc to LATEX. Rnw child documents are often incomplete in the sense that they do not have the LATEX preamble (lines from \documentclass to \begin{document}), so if we compile them directly, we will end up with LATEX errors.

Although child documents are supposed to be related to the parent document, it is not necessarily true in some cases. Sometimes a child document is there only for the purpose of organizing a huge document, and the computation in the child document may be completely irrelevant to the parent. In this case, all we need is to borrow the preamble of the parent document and append it to the child document when compiling the results.

The function set_parent() notiﬁes knitr of the parent document of a child; once this function is called, knitr will read the preamble of the parent document and write it to the child document when an Rnw document is compiled to TEX. For example, we can do this in chapt1.Rnw:

<<parent, include=FALSE>>= set_parent("book.Rnw") @

Then, whatever LATEX styles are deﬁned in the preamble of book.Rnw will be available to chapt1.tex as if the content of chapt1.Rnw were in book.Rnw.

