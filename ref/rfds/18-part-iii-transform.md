# Part III. Transform

The second part of the book was a deep dive into data visualization. In this part of the book, you’ll learn about the most important types of variables that you’ll encounter inside a data frame and learn the tools you can use to work with them.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_p300.png" alt="Our data science model, with transform highlighted in blue." />
<h6 id="figure-iii-1.-the-options-for-data-transformation-depend-heavily-on-the-type-of-data-involved-the-subject-of-this-part-of-the-book.">Figure III-1. The options for data transformation depend heavily on the type of data involved, the subject of this part of the book.</h6>
</figure>

You can read these chapters as you need them; they’re designed to be largely standalone so that they can be read out of order.

- <a href="ch12.html#chp-logicals" data-type="xref">Chapter 12</a> teaches you about logical vectors. These are the simplest types of vectors, but they are extremely powerful. You’ll learn how to create them with numeric comparisons, how to combine them with Boolean algebra, how to use them in summaries, and how to use them for condition transformations.

- <a href="ch13.html#chp-numbers" data-type="xref">Chapter 13</a> dives into tools for vectors of numbers, the powerhouse of data science. You’ll learn more about counting and a bunch of important transformation and summary functions.

- <a href="ch14.html#chp-strings" data-type="xref">Chapter 14</a> gives you the tools to work with strings: you’ll slice them, you’ll dice them, and you’ll stick them back together again. This chapter mostly focuses on the stringr package, but you’ll also learn some more tidyr functions devoted to extracting data from character strings.

- <a href="ch15.html#chp-regexps" data-type="xref">Chapter 15</a> introduces you to regular expressions, a powerful tool for manipulating strings. This chapter will take you from thinking that a cat walked over your keyboard to reading and writing complex string patterns.

- <a href="ch16.html#chp-factors" data-type="xref">Chapter 16</a> introduces factors: the data type that R uses to store categorical data. You use a factor when a variable has a fixed set of possible values, or when you want to use a nonalphabetical ordering of a string.

- <a href="ch17.html#chp-datetimes" data-type="xref">Chapter 17</a> gives you the key tools for working with dates and date-times. Unfortunately, the more you learn about date-times, the more complicated they seem to get, but with the help of the lubridate package, you’ll learn to how to overcome the most common challenges.

- <a href="ch18.html#chp-missing-values" data-type="xref">Chapter 18</a> discusses missing values in depth. We’ve discussed them a couple of times in isolation, but now it’s time to discuss them holistically, helping you come to grips with the difference between implicit and explicit missing values and how and why you might convert between them.

- <a href="ch19.html#chp-joins" data-type="xref">Chapter 19</a> finishes up this part of the book by giving you the tools to join two (or more) data frames together. Learning about joins will force you to grapple with the idea of keys and think about how you identify each row in a dataset.
