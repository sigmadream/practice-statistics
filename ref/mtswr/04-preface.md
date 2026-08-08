#  Preface 

Welcome to the book *Modern Time Series Analysis with R* .

In today’s world, we are overwhelmed by choice. An abundance of data and rapid technological advancements driven by both academic research and industry innovation are available at our fingertips. While collecting data has become remarkably inexpensive, performing the precise analysis needed to solve complex business problems has never been more challenging. With a wide array of methodologies and limited time to master them, practitioners often lack the critical skills to select the right approach. This book bridges the gap in a time series context by showing you how to assess business research problems, evaluate data characteristics, and determine the analytical roadmap required to extract value from time series data.

Time series analysis is a mature field of scientific literature, with the availability of resources serving the needs of practitioners in both academia and industry. Methods and models for time series analysis have been implemented in both open source and proprietary software for decades, and some of the best-known libraries are in the open-source space. Despite these facts, the adaptation of time series analysis is lagging in the industry.

The rise of **machine learning** ( **ML** ) in the early 2000s gave businesses an opportunity to harness the power of computing in the business decision-making process. Major enterprise platforms, such as Google BigQuery (BigQuery ML to ARIMA_PLUS), and Databricks (MLflow to AutoML for Time Series), had a couple of years' gap in adding dedicated time series forecasting to their respective automated ML suites. This again highlights the need for specialized treatment for time series using ML workflows and perhaps also sheds light on the benefits and value of using time-series-based techniques in the business domain.

From our experience across retail, finance and banking, telco, technology, and consulting, we realized that the challenge is not the availability of resources or tools. To us, the challenges are two-fold: (i) a gap in translating business problems into a time-series context, (ii) awareness of time-series-based techniques that could solve business problems with less complexity and cost.

To navigate this complexity, we leverage R, a language built by and for data professionals, which offers an unparalleled ecosystem for temporal data. The journey begins with the foundational building blocks of R, ensuring you have a stable environment to execute analysis and run examples presented throughout the book. We cover essential setup – from installation to mastering basic R functions and the mechanics of data input/output.

From this solid base, we move on to time series. We describe their characteristics and explain why they deserve special analytical treatment by highlighting their differences from other data types. We follow a “tidy” workflow of time-series data wrangling, transforming raw timestamps into structured formats. Through a visual-first approach, you will learn about specialized plotting techniques for time series to reveal hidden patterns. You will also learn about the wide array of time-series-based techniques to solve practical business problems.

The next step is the essential, often under-appreciated, stage of data preparation and pre-processing. We discuss techniques to adjust, transform, and decompose time-series data. You will learn the art of distilling raw data into meaningful descriptors by performing time-series feature extraction, alongside advanced filtering and smoothing methods to isolate the true underlying signal from the noise.

Building on these skills, the book explores the full evolution of modern forecasting, from classical statistical methods, such as ETS and ARIMA, to ML and deep learning. We tackle high-dimensional challenges such as hierarchical forecasting and multiple time series methods to ensure consistency across complex business structures.

Beyond forecasting, we investigate the “why” and “when” of data shifts by estimating causal impact. We discuss time-series-based methodologies to analyze business intervention data where traditional A/B testing is not applicable. You will learn how to identify critical turning points through changepoint analysis and anomaly detection. Ultimately, this book will equip you with the rigorous methodology and technical command necessary to transform volatile temporal data into a cornerstone of evidence-based decision-making.

#  Who this book is for 

This book is uniquely written by developers for developers. It moves beyond abstract theory to provide a practical manual for those tasked with building, deploying, and interpreting time-series models in high-stakes environments. We focus on the critical “missing link” in data science: the ability to connect complex business problems with the correct analytical solutions.

We cater to three distinct levels of professional expertise:

-  **Early career** **practitioners** : For those just beginning their journey, this book provides a rigorous introduction to time-series analysis. It serves as a foundational roadmap, transforming you from someone who writes code into someone who designs robust, reproducible analytical workflows.
-  **Mid-career** **professionals** : For analysts and engineers, we focus on methodological precision. You will learn how to navigate the nuances of time series data pre-processing, the selection of appropriate forecasting models, and conducting proper analysis of business interventions – ensuring your outputs are not just technically sound, but strategically relevant to your organization’s objectives.
-  **Expert** **leaders and researchers** : For senior leads, architects, and other data professionals, this book serves as a high-level strategic guide for building time-series-based solutions. A defining feature of this book is that for every methodology discussed, we explicitly highlight the rationale (why a specific model is the superior choice for given data characteristics) and the application (the precise business problems the model is designed to solve). It will empower business leaders by providing the methodological rigor required to lead teams toward evidence-based decision-making.

#  What this book covers 

*<a href="Chapter_1.xhtml#h1_14" class="chapref"> Chapter 1 </a>* , *R* *, R* *s* *tudio* *,* *and R* *P* *ackages* , details the ways of installing R, RStudio IDE, and R packages on different operating platforms. It also includes guidelines for setting up the R ecosystem for seamless development.

*<a href="Chapter_2.xhtml#h1_47" class="chapref"> Chapter 2 </a>* , *Objects and Functions in R* , provides a foundational guide to R programming for beginners, detailing the properties of base and S3 objects and core data types, and demonstrating how to build custom functions using control flow and iterative techniques.

*<a href="Chapter_3.xhtml#h1_84" class="chapref"> Chapter 3 </a>* , *Data Input/Output in R* *,* provides a comprehensive guide to data input and output in R, detailing methods for importing and exporting files from different sources while demonstrating how to securely connect to relational databases, execute SQL queries, and maintain rigorous data governance standards.

*<a href="Chapter_4.xhtml#h1_113" class="chapref"> Chapter 4 </a>* , *Time Series Characteristics* , provides a formal introduction to the unique characteristics of time series data, distinguishing it from other time-indexed formats using the N-T decision space while detailing its fundamental components—trend, seasonality, and cycles—and explaining the diagnostic importance of stationarity and autocorrelation.

*<a href="Chapter_5.xhtml#h1_129" class="chapref"> Chapter 5 </a>* , *Time Series Data Wrangling and Visualization* , demonstrates how to parse date-time variables, transform data into tsibble objects, and apply specialized plotting techniques—such as seasonal, subseries, and lag plots—to identify underlying patterns.

*<a href="Chapter_6.xhtml#h1_146" class="chapref"> Chapter 6 </a>* , *Business Applications of Time Series Analysi* *s* , presents an overview of business applications for time series analysis, categorizing problem domains into the study of inherent characteristics (including trend, seasonality, and anomaly detection), inference and attribution modeling, causal impact estimation for evaluating interventions, and predictive forecasting to support strategic decision-making.

*<a href="Chapter_7.xhtml#h1_161" class="chapref"> Chapter 7 </a>* , *Time Series Adjustments, Transformations* *,* *and Decomposition* , presents a comprehensive guide to time series pre-processing, detailing how to apply adjustments to correct for external biases, mathematical transformations such as Box-Cox to stabilize variance, and decomposition frameworks such as STL and X-13ARIMA to isolate inherent trend and seasonal components from random noise.

*<a href="Chapter_8.xhtml#h1_184" class="chapref"> Chapter 8 </a>* , *Time* *S* *eries Features* , explores the extraction of diverse time series features—including descriptive statistics, autocorrelation profiles, STL-based strength measures, and windowed dynamics—to quantify temporal characteristics and facilitate advanced analytical tasks such as dimensionality reduction and the unsupervised clustering of large datasets.

*<a href="Chapter_9.xhtml#h1_201" class="chapref"> Chapter 9 </a>* , *Time* *S* *eries Smoothing and Filtering* , explores techniques to reduce random noise and isolate meaningful signals through time series smoothing and filtering, detailing the application of **Exponentially Weighted Moving Averages** ( **EWMA** ) for quality control, linear filtering to manage error-term dependencies, the Hodrick-Prescott filter for separating long-term trends from business cycles, and the Kalman filter for recursively estimating latent states within complex dynamic systems

*<a href="Chapter_10.xhtml#h1_216" class="chapref"> Chapter 10 </a>* , *Basics of Forecasting* , establishes the foundational tidy forecasting workflow with the fable and fabletools R packages, detailing the implementation of baseline methods such as naïve, seasonal naïve, and average forecasting alongside **Time Series Linear Models** ( **TSLM** ) for scenario analysis, while explaining rigorous point forecast accuracy metrics and time series cross-validation techniques to ensure model generalizability.

*<a href="Chapter_11.xhtml#h1_232" class="chapref"> Chapter 11 </a>* , *Exponential Smoothing* , provides a comprehensive guide to the exponential smoothing family of forecasting, meticulously distinguishing between procedural methods and state-space models while detailing specific applications for simple exponential smoothing, Holt’s linear trend, and Holt-Winters patterns alongside automated model selection leveraging information criteria.

*<a href="Chapter_12.xhtml#h1_254" class="chapref"> Chapter 12 </a>* , *ARIMA* *Forecasting Models* , explores the ARIMA forecasting framework, detailing its core components of **autoregressive** ( **AR** ) dependencies, **integration** ( **I** ) through differencing to achieve stationarity, and **moving average** ( **MA** ) error modeling while demonstrating the implementation of **seasonal variations** ( **SARIMA** ) and the automated Hyndman-Khandakar algorithm for efficient model selection.

*<a href="Chapter_13.xhtml#h1_275" class="chapref"> Chapter 13 </a>* , *Advanced Computational Methods for Forecasting* , explores advanced computational methods for forecasting by reframing time series as supervised learning problems, detailing the implementation of **Neural Network Autoregressive** ( **NNETAR** ) models, ML workflows using the forecastML package with regularized regression, the Prophet curve-fitting method, and the strategic use of forecast ensembles to enhance predictive accuracy.

*<a href="Chapter_14.xhtml#h1_291" class="chapref"> Chapter 14 </a>* , *Forecasting Models for Multiple Time Series* , explores methodologies for forecasting multiple time series simultaneously, specifically detailing the application of **Vector Autoregression** ( **VAR** ) to capture bidirectional feedback loops and various reconciliation frameworks—such as top-down, bottom-up, and optimal trace minimization—to ensure consistency across hierarchical and grouped data structures.

*<a href="Chapter_15.xhtml#h1_318" class="chapref"> Chapter 15 </a>* , *Causal Impact Estimation* , provides a comprehensive framework for causal impact estimation, detailing the equivalence between hypothesis testing and regression while demonstrating the application of **Interrupted Time Series** ( **ITS** ) analysis to isolate the effects of business interventions within stationary, trended, and autocorrelated data using specialized methods such as ARIMAX and robust statistical adjustments.

*<a href="Chapter_16.xhtml#h1_346" class="chapref"> Chapter 16 </a>* , *Changepoint Detection* , provides a guide to changepoint detection for identifying sudden, persistent shifts in time series behavior, detailing the recursive BFAST framework for trend and seasonal breaks, the optimization-based PELT algorithm for exact segmentation, and the BEAST probabilistic approach for quantifying uncertainty in the location and magnitude of structural breaks.

*<a href="Chapter_1.xhtml#h1_14" class="chapref"> Chapter 1 </a>* *7* , *Anomaly Detection and Imputation* , discusses a framework for identifying point and collective anomalies using STL-based and Isolation Forest methodologies while detailing the application of linear interpolation for imputing missing values to preserve the integrity and temporal structure of time series data.

#  To get the most out of this book 

This book is structured to serve as both a step-by-step tutorial and a long-term technical reference. Each chapter follows a logical progression, starting with theoretical underpinnings before moving on to hands-on implementation using R. For the most effective learning experience, we recommend the following:

-  **A** **sequential path** : Beginners should sequentially progress through the chapters to establish a solid foundation in computing and knowledge.
-  **An** **applied path** : Experienced practitioners may choose to dive directly into specific chapters, such as the ones about forecasting or causal impact estimation.
-  **Interactive learning** : All code snippets are designed for reproducibility and follow a standardized structure. We encourage you to execute the code as you read and benefit from the detailed explanation of the results.
-  **Leveraging the modern** **framework** : The code throughout the book adheres to the tidyverse style guide, ensuring that your scripts are readable, maintainable, and scalable. You will work extensively with the R packages, such as tidyverse, fable, and so on, which provide a unified and consistent interface for time-series analysis, making it easier to transition between modeling techniques.

##  Download the example code files 

The code bundle for the book is hosted on GitHub at <a href="https://github.com/PacktPublishing/Modern-Time-Series-Analysis-with-R.git" style="text-decoration: none;">  https://github.com/PacktPublishing/Modern-Time-Series-Analysis-with-R.git  </a> . We also have other code bundles from our rich catalog of books and videos available at https://github.com/PacktPublishing. Check them out!

##  Download the color images 

Your purchase includes a color, DRM-free PDF copy of this book, ideal for viewing color images, screenshots, and diagrams. Refer to the *Free* *benefits* *with* *your* *book* section at the end of the *Preface* to unlock your PDF copy.

##  Conventions used 

There are a number of text conventions used throughout this book.

` `` CodeInText `` ` : Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. For example: “You have already used functions from this package, such as ` `` ACF() `` ` , ` `` gg_season() `` ` , ` `` gg_subseries() `` ` , and so on, in previous chapters.”

A block of code is set as follows:

```
dimnames(mat_b) <- list(
paste0("row_",1:3), paste0("col_",1:3))
mat_b
```

Any command-line input or output is written as follows:

```
      col_1 col_2 col_3
row_1  TRUE  TRUE FALSE
row_2  TRUE  TRUE FALSE
row_3  TRUE  TRUE FALSE
```

**Bold** : Indicates a new term, an important word, or words that you see on the screen. For instance, words in menus or dialog boxes appear in the text like this. For example: “The **exponentially weighted moving average** ( **EWMA** ) method is applicable in such a scenario.”

Warnings or important notes appear like this.

Tips and tricks appear like this.

#  Get in touch 

Feedback from our readers is always welcome.

**General feedback** : If you have questions about any aspect of this book or have any general feedback, please email us at ` `` customercare `` ` ` `` @packt.com `` ` ` ` and mention the book's title in the subject of your message.

**Errata** : Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you reported this to us. Please visit , click **Submit Errata** , and fill in the form.

**Piracy** : If you come across any illegal copies of our works in any form on the internet, we would be grateful if you would provide us with the location address or website name. Please contact us at ` `` copyright@packt.com `` ` with a link to the material.

**If** **you** **are** **interested in becoming an author** : If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit <a href="https://http://authors.packt.com/" style="text-decoration: none;">  http://authors.packt.com/  </a> .

#  Free benefits with your book 

This book comes with free benefits to support your learning. Activate them now for instant access (see the “ *How to Unlock* ” section for instructions).

Here’s a quick overview of what you can instantly unlock with your purchase:

<figure class="mediaobject">
 <span class="image placeholder" data-original-image-src="../Images/B21040_Preface_1.png" data-original-image-title="">Image 1</span> 
</figure>

##  How to Unlock 

Scan the QR code (or go to <a href="https://packtpub.com/unlock" style="text-decoration: none;">  packtpub.com/unlock  </a> ). Search for this book by name, confirm the edition, and then follow the steps on the page.

<span class="image placeholder" data-original-image-src="../Images/B21040_Preface_2.png" data-original-image-title="">Image</span>

<span class="image placeholder" data-original-image-src="../Images/B21040_Preface_3.png" data-original-image-title="">Image</span>

*Note: Keep your invoice handy. Purchases made directly from Packt don’t require one*

#  Share your thoughts 

Once you’ve read *Modern Time Series Analysis with R* , we’d love to hear your thoughts! Scan the QR code below to go straight to the Amazon review page for this book and share your feedback.

<span class="image placeholder" data-original-image-src="../Images/B21040_Preface_4.png" data-original-image-title="">Image</span>

<a href="https://packt.link/r/1805124013" style="text-decoration: none;">  https://packt.link/r/1805124013  </a>

Your review is important to us and the tech community and will help us make sure we’re delivering excellent quality content.
