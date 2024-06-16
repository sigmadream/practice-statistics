# Part IV. Import

In this part of the book, you’ll learn how to import a wider range of data into R, as well as how to get it into a form useful form for analysis. Sometimes this is just a matter of calling a function from the appropriate data import package. But in more complex cases it might require both tidying and transformation to get to the tidy rectangle that you’d prefer to work with.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p400.png" alt="Our data science model with import highlighted in blue." />
<h6 id="figure-iv-1.-data-import-is-the-beginning-of-the-data-science-process-without-data-you-cant-do-data-science">Figure IV-1. Data import is the beginning of the data science process; without data you can’t do data science!</h6>
</figure>

In this part of the book you’ll learn how to access data stored in the following ways:

- In <a href="ch20.html#chp-spreadsheets" data-type="xref">Chapter 20</a>, you’ll learn how to import data from Excel spreadsheets and Google Sheets.

- In <a href="ch21.html#chp-databases" data-type="xref">Chapter 21</a>, you’ll learn about getting data out of a database and into R (and you’ll also learn a little about how to get data out of R and into a database).

- In <a href="ch22.html#chp-arrow" data-type="xref">Chapter 22</a>, you’ll learn about Arrow, a powerful tool for working with out-of-memory data, particularly when it’s stored in the parquet format.

- In <a href="ch23.html#chp-rectangling" data-type="xref">Chapter 23</a>, you’ll learn how to work with hierarchical data, including the deeply nested lists produced by data stored in the JSON format.

- In <a href="ch24.html#chp-webscraping" data-type="xref">Chapter 24</a>, you’ll learn web “scraping,” the art and science of extracting data from web pages.

There are two important tidyverse packages that we don’t discuss here: haven and xml2. If you are working with data from SPSS, Stata, and SAS files, check out the [haven package](https://oreil.ly/cymF4). If you’re working with XML data, check out the [xml2 package](https://oreil.ly/lQNBa). Otherwise, you’ll need to do some research to figure out which package you’ll need to use; Google is your friend here.
