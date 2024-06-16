# Preface

Data Science is a vast field of study. There is so much to learn about and, every day, more and more is added to this pile. It is fascinating, for sure, the way we can analyze data and extract insights that will serve as a base for better decisions. The big companies have learned that data is what can take them to the next level of business achievement and are leading the way by building strong data science teams.

However, just data by itself is not the answer. It is like crude oil: out of it, we can make plenty of things, but just that black liquid from the ground won’t serve us very well. So, raw data is something, but when we clean, transform, and analyze it, we are transforming data into information, and that brings us the power to make better decisions.

In this book, we will go over many aspects of data wrangling, where we will learn how to transform data into knowledge for our business. Our chosen programming language is R, an amazing piece of software that was initially created as a statistical program but became much more than that. If we know what we need to achieve, getting there is just a matter of finding the right tools. Many of those tools are in this book.

# Who this book is for

This book is written for professionals from academia and industry looking to acquire or enhance their capabilities of wrangling data using the R language with RStudio. You are expected to be familiar with basic programming concepts in R, such as types of variables and how to create them, loops, and functions. This book provides a complete flow for data wrangling, starting with loading the dataset in the IDE until it is ready for visualization and modeling. A background in **science**, **technology**, **engineering**, and **math** (**STEM**) areas or in the data science field are not required but will help you to internalize knowledge and get the most from the content.

# What this book covers

[*Chapter 1*](B18355_01.xhtml#_idTextAnchor029), *Fundamentals of Data Wrangling*, will introduce this book’s main theme, explaining what data wrangling is and why and when to use it. In addition, it also shows the main steps of a data science project and covers three well-known frameworks for data science projects.

[*Chapter 2*](B18355_02.xhtml#_idTextAnchor039), *Load and Explore Datasets*, provides different ways to load datasets to RStudio. Every project begins with data, so it is important to know how to load it into your session. It also begins exploring that data to familiarize you with exploratory data analysis.

[*Chapter 3*](B18355_03.xhtml#_idTextAnchor057), *Basic Data Visualization*, is the first touch point with data visualization, which is an important component of any data science project. In this chapter, we will learn about the first steps to creating compelling and meaningful graphics using only the built-in library from R.

[*Chapter 4*](B18355_04.xhtml#_idTextAnchor075), *Working with Strings*, starts our journey of learning about the wrangling functions for each major variable type. In this chapter, we study many possible transformations with text, from detecting words in a phrase or in a dataset to some highly customized functions that involve regular expressions and text mining concepts.

[*Chapter 5*](B18355_05.xhtml#_idTextAnchor099), *Working with Numbers*, comprises the transformations for numerical variables. The chapter covers operations with vectors, matrices, and data frames and also covers the apply functions and how to make a good read of the descriptive statistics of a dataset.

[*Chapter 6*](B18355_06.xhtml#_idTextAnchor114), *Working with Date and Time Objects*, is where we will learn more about this fascinating object type, date and time. It introduces concepts from the basics of creating a date and time object to a practical project that shows how it can be used in an analysis.

[*Chapter 7*](B18355_07.xhtml#_idTextAnchor126), *Transformations with Base R*, is the core of the book, exploring the most important transformations to be performed in a dataset. This chapter covers tasks such as slicing, grouping, replacing, arranging, binding data, and more. The most used transformations are covered here and mostly use the built-in functions without the need to load extra libraries.

[*Chapter 8*](B18355_08.xhtml#_idTextAnchor142), *Transformations with tidyverse Libraries*, follows the same idea as [*Chapter 7*](B18355_07.xhtml#_idTextAnchor126), but this time, the transformations are performed with **tidyverse**, which is a highly used R package for data science.

[*Chapter 9*](B18355_09.xhtml#_idTextAnchor167), *Exploratory Data Analysis*, is all about practice. After going over many transformation functions for different types of variables, it’s time to put the acquired knowledge into practice and work on a complete exploratory data analysis project.

[*Chapter 10*](B18355_10.xhtml#_idTextAnchor184), *Introduction to ggplot2*, introduces the visualization library, **ggplot2**, which is the most used library for data visualization in the R language, given its flexibility and robustness. In this chapter, we will learn more about the grammar of graphics and how ggplot2 is created based on this concept. We will also cover many kinds of plots and how to create them.

[*Chapter 11*](B18355_11.xhtml#_idTextAnchor207), *Enhanced Visualizations with ggplot2*, covers more advanced types of graphics that can be created with ggplot2, such as facet grids, maps, and 3D graphics.

[*Chapter 12*](B18355_12.xhtml#_idTextAnchor218), *Other Data Visualization Options*, is where we will see yet more options to visualize data, such as creating a basic plot in **Microsoft Power BI** but using the R language. We will also cover how to create word clouds and when that kind of visualization can be useful.

[*Chapter 13*](B18355_13.xhtml#_idTextAnchor228), *Build a Model with R*, is all about an end-to-end data science project. We will get a dataset and start exploring it, then we will clean the data and create some visualizations that help us to explain the steps taken, and that will lead us to the best model to be created.

[*Chapter 14*](B18355_14.xhtml#_idTextAnchor254), *Build an Application with Shiny in R*, is the final chapter, where we will take the model created in [*Chapter 13*](B18355_13.xhtml#_idTextAnchor228) and put it in production using a web application created with Shiny for R.

# To get the most out of this book

The get the most out of the content presented in this book, it is expected that you have a minimum knowledge of object-oriented programming (creating variables, loops, and functions) and have already worked with R. A basic knowledge of data science concepts is also welcome and can help you understand the tutorials and projects.

All the software and code are created using RStudio for Windows 10, and if you want to code along with the examples, you will need to install R and RStudio on your local machine. To do that, you should go to <https://cran.r-project.org/>, click on **Download R for Windows** (or for your operating system), then click on **base**, and finally, click on **Download R-X.X.X for Windows**. This will download the R language executable file to your machine. Then, you can double-click on the file to install, accepting the default selections.

Next, you need to install RStudio, renamed to Posit in 2022. The URL to download the software is found here: <https://posit.co/download/rstudio-desktop/>. Click on **Download** and look for the version of your operating system. The software has a free of charge version and you can install it, accepting the default options once again.

The main libraries used in the tutorials from this book are indicated as follows:

|                      |                           |
|----------------------|---------------------------|
| **Software/Library** | **Version**               |
| R                    | 4.1.0                     |
| RStudio              | 2022.02.3+492 for Windows |
| Tidyverse            | 1.3.1                     |
| Tidytext             | 0.3.2                     |
| Gutenbergr           | 0.2.1                     |
| Patchwork            | 1.1.1                     |
| wordcloud2           | 0.2.1                     |
| ROCR                 | 1.0-11                    |
| Shinythemes          | 1.2.0                     |
| Plotly               | 4.10.0                    |
| Caret                | 6.0-90                    |
| Shiny                | 1.7.1                     |
| Skimr                | 2.1.4                     |
| Lubridate            | 1.8.0                     |
| randomForest         | 4.7-1                     |
| data.table           | 1.14.2                    |

To install any library in RStudio, just use the following code snippet:

\# Installing libraries to RStudio install.packages(“package_name”) \# Loading a library to a session library(package_name)

In R, it can be useful to remind yourself of, or have in mind, these two code snippets. The first one is how to write **for** loops. We can write it as, for a given condition, execute a piece of code until the condition is not met anymore:

for (num in 1:5) {     print(num) }

The other one is the skeleton of a function written in R language, where we provide variables and the code of what should be done with those variables, returning the resulting calculation:

custom_sum_function \<- function(var1, var2) {     # Function code     my_sum = sum(var1 + var2)     return(my_sum) }

If you are using a digital version of this book, we advise you to type the code yourself or access the code from the book’s GitHub repository, preventing any potential errors with code broken due to copy and paste.

# Download the example code files

You can download the example code files for the tutorials contained in this book from GitHub at <https://github.com/PacktPublishing/Data-Wrangling-with-R>. If there are any changes to the code, it will be updated in the GitHub repository.

# Conventions used

There are a number of text conventions used throughout this book.

**Code in text**: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: “Mount the downloaded **WebStorm-10\*.dmg** disk image file as another disk in your system.”

A block of code is set as follows:

html, body, \#map { height: 100%; margin: 0; padding: 0 }

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

\[default\] exten =\> s,1,Dial(Zap/1\|30) exten =\> s,2,Voicemail(u100) **exten =\> s,102,Voicemail(b100)** exten =\> i,1,Voicemail(s0)

Any command-line input or output is written as follows:

\$ mkdir css \$ cd css

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For instance, words in menus or dialog boxes appear in **bold**. Here is an example: “Select **System info** from the **Administration** panel.”

Tips or important notes

Appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, email us at <customercare@packtpub.com> and mention the book title in the subject of your message.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) and fill in the form.

**Piracy**: If you come across any illegal copies of our works in any form on the internet, we would be grateful if you would provide us with the location address or website name. Please contact us at <copyright@packt.com> with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com).

# Share Your Thoughts

Once you’ve read *Data Wrangling with R*, we’d love to hear your thoughts! Please [click here to go straight to the Amazon review page](https://packt.link/r/1-803-23540-3) for this book and share your feedback.

Your review is important to us and the tech community and will help us make sure we’re delivering excellent quality content..

# Download a free PDF copy of this book

Thanks for purchasing this book!

Do you like to read on the go but are unable to carry your print books everywhere?

Is your eBook purchase not compatible with the device of your choice?

Don’t worry, now with every Packt book you get a DRM-free PDF version of that book at no cost.

Read anywhere, any place, on any device. Search, copy, and paste code from your favorite technical books directly into your application.

The perks don’t stop there, you can get exclusive access to discounts, newsletters, and great free content in your inbox daily

Follow these simple steps to get the benefits:

1.  Scan the QR code or visit the link below

![](D:\sd\Practices\any2md\output\[2023] Data Wrangling with R/image/B18355_QR_Free_PDF.jpg)

<https://packt.link/free-ebook/9781803235400>

1.  Submit your proof of purchase
2.  That’s it! We’ll send your free PDF and other benefits to your email directly
