### 8

###### Cache

One challenge of dynamic documents is that some code chunks may take a long time to run, and these chunks may not be modiﬁed or updated frequently. In this case, caching can be very helpful. The basic idea is, a chunk will not be re-executed as long as it has not been modiﬁed since the last run, and old results will be directly loaded instead.

###### 8.1 Implementation

Cache is not a new idea — both the packages cacheSweave and weaver have implemented it based on Sweave, with the former using ﬁlehash and the latter using *.RData images; cacheSweave also supports lazyloading of objects based on ﬁlehash. The knitr package directly uses internal base R functions to save (tools:::makeLazyLoadDB()) and lazyload objects (lazyLoad()).

The cacheSweave vignette has clearly explained the concept of lazyloading. Roughly speaking, lazy-loading means an object will not be loaded into memory until it is really used anywhere — only a “promise” is created instead, which is usually fast and cheap in terms of memory consumption; when this promise is to be used for computation, the real object will be loaded from a hard disk. This is very useful for cache; sometimes we read a large object and cache it, then take a subset for analysis and this subset is also cached; in the future, the initial large object will not be loaded into R if our computation is only based on the object of its subset. For more details about promises in R, see ?promise.

To turn on caching, we can set the chunk option cache to TRUE (default is FALSE). Below is a code chunk that quickly shows the effect of cache:

- x <- 1 Sys.sleep(10)
- x <- 2


81

We used Sys.sleep() to let R sleep for 10 seconds. We can see the pause the ﬁrst time this chunk is compiled, but when we compile it again, there will be no pause, because the code evaluation is actually completely skipped. There is an object x created in this chunk, and it will be lazy-loaded next time; knitr will ﬁgure out all newly created objects in a chunk and save them to lazy-load databases (*.rdb and *.rdx ﬁles). Now we can check the value of x:

x # value from previous chunk ## [1] 2

###### 8.2 Write Cache

The path of cache ﬁles is determined by the chunk option cache.path; by default all cache ﬁles are created under a directory cache/ relative to the current working directory. If the option value contains a directory (e.g., cache.path = ’cache/abc-’), cache ﬁles will be stored under that directory. Similar to ﬁgure paths, the cache directory will be automatically created if it does not exist, and cache.path can also be a preﬁx for cache ﬁles instead of a physical path.

The cache is invalidated and purged on any changes to the code chunk, including both the R code and chunk options; this means old cache ﬁles of this chunk are removed and replaced by new cache ﬁles. Cache ﬁlenames are identiﬁed by the chunk label as the preﬁx (recall that chunk labels have to be unique in a document), and the sufﬁx of cache ﬁlenames is an MD5 hash string of an R object, which is a list including the R code, chunk options, and the value getOption(’width’). The MD5 hash is calculated by the digest package, and it will be clear how it works by the example below, which emulates the cache ﬁlename generation in knitr:

d <- digest::digest ## imagine x$code is the code chunk; x$options are chunk ## options

- x <- list(code = "1+1", options = list(results = "asis", fig.height = 3), width = getOption("width"))


d(x) ## [1] "667308d70fc72f26eb7454dde04af9a0"

x$code <- "1 + 1" # add spaces to code d(x)

## [1] "e903b616477cfa3e2314a3da65062dfb" x$options$eval <- FALSE # add option eval as FALSE d(x) ## [1] "8decb2a180f7f49b47de54bd5ec8fb34" x$width <- 40 d(x) ## [1] "7e1d77987b195b14d9b563b9a8f0ca6c"

The character strings of width 32 above are MD5 hashes. We can see that an MD5 hash is sensitive to changes in content. Any change will lead to a new hash string, even if the change is simply a white space. The cache ﬁlenames are of the form label_hash.rdb. Each time, knitr will compare the hash of the current chunk to the cache ﬁlenames; if they do not match, it means there has been a change in the chunk, and the old cache should be purged.

One exception is the include option, which is not cached because include = TRUE / FALSE does not affect code evaluation, so we can change this chunk option without affecting cache.

The reason that getOption(’width’) affects cache is that it may affect the width of printed text output.

###### 8.3 When to Update Cache

It may not be clear when to update cache in certain circumstances, although the three components described above seem to be reasonable to take into consideration. Let’s consider two cases as follows:

- 1. R is still being updated every few months, with each new version ﬁxing bugs and introducing new features; should we update cache after we upgrade R to a newer version? (similar concern applies to R packages)
- 2. If we read an external data ﬁle in a source document, and that ﬁle has been modiﬁed; how can we tell knitr that all the


cached results need to be updated (even if the source document is not changed)?

In these cases, we need to put more components into the object to calculate the hash. Since a code chunk can accept arbitrary options (not only the options introduced in this book), and all chunk options are reﬂected in the hash, we can use additional chunk options to affect the cache.

To answer the ﬁrst question, we can add a chunk option, say, version to the document, which takes the version of R as its value, e.g., <<cache-rversion, cache=TRUE, version=R.version.string>>= # code which may be affected by R version R.version.string ## [1] "R version 3.2.0 (2015-04-16)" @

Then if R has been upgraded, this chunk will be re-executed. To solve the second problem, we need to let knitr know changes in external ﬁles. One natural indicator is the modiﬁcation time of ﬁles, which can be obtained by the function ﬁle.info(). Suppose the data ﬁle is named iris.csv, and we can put its modiﬁcation time in a chunk option iris_time, e.g.,

<<itime, cache=TRUE, iris_time=file.info( iris.csv )$mtime>>= # data will be re-read if iris.csv becomes newer iris <- read.csv("iris.csv") @

There are no ﬁxed rules about when or whether to update cache; it is up to the speciﬁc applications; e.g., we do not have to purge cache after R has been upgraded. Anyway, we need to set up chunk options carefully to guarantee the results are always up-to-date.

###### 8.4 Side Effects

In computer science, a side effect refers to a state change that occurs outside of a function that is not the returned value. Common side effects include creating a plot (window or ﬁle), writing a ﬁle, and printing results to the console, etc. Side effects are not straightforward to be cached — we can easily save an R object into the cache database, but it

is unclear how to save a plot window because it is not a value returned by a function. For this reason, packages like weaver and cacheSweave do not cache side effects, but knitr will try to preserve some side effects, such as:

- 1. printed results: meaning that any output of a code chunk will be loaded into the output document for a cached chunk, even if it is not really evaluated. The reason is knitr also caches the output of a chunk as a character string. Note this means graphics output is also cached since it is part of the output;
- 2. loaded packages: after the evaluation of each cached chunk, the list of packages used in the current R session is written to a ﬁle under the cache path with a sufﬁx __packages; next time, if a cached chunk needs to be rebuilt, these packages will be loaded ﬁrst. The reasons for caching package names are: it can be slow to load some packages, and a package might be loaded in a previous cached chunk that is not available to the next cached chunk when only the latter needs to be rebuilt. Note that this only applies to cached chunks, and for uncached chunks, you must always use library() to load packages explicitly;
- 3. the random seed: if a chunk created a random seed (an integer vector), the seed will be saved and loaded next time to improve reproducibility of random simulations (also see Section 12.4.7).


Although knitr tries to keep some side effects, there are still other types of side effects like setting par() or options() that are not cached. Users should be aware of these special cases, and make sure to clearly separate the code that is not supposed to be cached into uncached chunks, e.g., set all global options in the ﬁrst chunk of a document and do not cache that chunk. Normally we have this chunk as the ﬁrst chunk of a document:

<<setup, cache=FALSE, include=FALSE>>= # set up some global options for the document options(width = 60, show.signif.stars = FALSE) # also set up global chunk options library(knitr) opts_chunk$set(fig.width = 5, fig.height = 4, tidy = FALSE) @

In the above chunk, cache = FALSE is often unnecessary because it is the default; we can put it there if we are conservative and want to make sure this chunk is indeed not cached.

###### 8.5 Chunk Dependencies

Sometimes a cached chunk may need to use objects from other cached chunks, which can bring about a serious problem — if objects in previ-

- ous chunks have changed, this chunk will not be aware of the changes and will still use old cached results, unless there is a way to detect such changes from other chunks. Therefore we have to introduce dependencies into cached chunks.


###### 8.5.1 Manual Dependency

There is a chunk option called dependson in knitr (idea taken from cacheSweave), which speciﬁes which other chunks this chunk depends on by setting a vector of chunk labels like dependson = c(’chunkA’, ’chunkB’). Then each time either of the cached chunks chunkA or chunkB is rebuilt, this chunk will lose its cache and be rebuilt as well.

Chunk dependencies can form a chain; in the following example, chunkC depends on chunkB, which in turn depends on chunkA: <<chunkA>>=

- x <- 1

- <<chunkB, dependson= chunkA >>=

y <- x + 2

- <<chunkC, dependson= chunkB >>=




- y + 5 @


The dependency is necessary because chunkC uses the object y that was created in chunkB, and chunkB needs the value of x created in chunkA. When x in the ﬁrst chunk is changed, the latter two chunks have to be updated accordingly.

The option dependson can also take an integer vector of chunk indices, e.g., dependson = 1 means this chunk depends on the ﬁrst chunk in the document, and dependson = c(3, 5) indicates dependency on the third and ﬁfth chunks. If the indices are negative, it means counting backwards from this chunk. For example, dependson = -1 means

this chunk depends on the previous chunk, and -c(1, 2, 3) means the previous three chunks. Note that when dependson takes integer values, it cannot make a chunk depend on later chunks (only previous chunks are possible candidates); character values of dependson do not have this restriction.

###### 8.5.2 Automatic Dependency

Another way to specify the dependencies among chunks is to use the chunk option autodep and the function dep_auto(). This is an experimental feature borrowed from weaver, which frees us from setting chunk dependencies manually. The basic idea is, if a latter chunk uses any objects created from a previous chunk, the latter chunk is said to depend on the previous one.

The function ﬁndGlobals() in the codetools package is used to ﬁnd

- out all global objects in a chunk, and according to its documentation, the result is an approximation. Global objects roughly mean the ones that are not created locally, e.g., in the expression function() {y <x}, x must be an existing global object outside (no matter what object it really is) because we do not see its creation in the body of this function, whereas y is local. Meanwhile, we also need to save the list of objects created in each cached chunk, so that we can compare them to the global objects in latter chunks. For example, if chunk A created an object x and chunk B uses this object, chunk B must depend on A, i.e., whenever A changes, B must also be updated.


When autodep = TRUE, knitr will write out the names of objects created in a cached chunk as well as those global objects in two ﬁles named __objects and __globals, respectively; later we can use the function dep_auto() to analyze the object names to ﬁgure out the dependencies automatically. A typical use is:

<<setup, cache=FALSE, include=FALSE>>= opts_chunk$set(autodep = TRUE) # set autodep globally dep_auto() # figure out dependencies @

Yet another way to specify dependencies is dep_prev(): this is a conservative approach that sets the dependencies so that a cached chunk will depend on all its previous chunks, i.e., whenever a previous chunk is updated, all later chunks will be updated accordingly.

In any case, dependency on uncached chunks is meaningless to knitr, because knitr only checks changes for cached chunks; knitr will give a warning when it sees dependency on uncached chunks. If we have

to depend on uncached chunks at all, we can use the trick introduced in Section 8.3, i.e., to put the uncached objects in the chunk options of cached chunks. Below is an example:

- <<A, cache=FALSE>>=

- x <- 1 @

<<B, cache=TRUE, foo=x>>=

- y <- x + 2 @




We created an object x in an uncached chunk A, and used it in a cached chunk B. If there is no dependency between the two chunks, B will not update when A is updated, but if we have set an option foo = x in chunk B, B will automatically be updated if the value of x has changed, which leads to changes in B’s chunk options.

###### 8.6 Load Cache Manually

Usually the cache database is automatically loaded for a cached chunk, and we can actually load it manually. This has a useful application: imagine you calculated a value x in a later chunk, but you want to use it earlier in the document. That is not possible because knitr compiles the document in a linear fashion, and you cannot use an object created in the future. However, if you have turned on the cache for that chunk, you may just load its cache database early.

The function load_cache() in knitr was designed for this purpose. It takes a chunk label to ﬁnd the cache database, and optionally you can specify the object that you want this function to return from the cache.

load_cache(label, object, notfound = "NOT AVAILABLE", path = opts_chunk$get("cache.path"), lazy = TRUE)

Now suppose you have a cached chunk named foo later in the document, which creates an object x, you can load_cache(’foo’, ’x’) to fetch the value of x in that chunk. Of course, the ﬁrst time you compile the document, x will not be available, and that is what the argument notfound is for. If you use x in an inline R expression, you will see NOT AVAILABLE in the output, and it will be replaced by the value of x after you compile the document again, since the chunk foo has been cached.

###### 8.7 Other Options

Although lazy-loading is useful, it may not work in certain cases for reasons that are still not clear to us. Anyway, you can turn off lazyloading using the chunk option cache.lazy = FALSE. In this case, knitr will just save the objects with save(), and load them with load(), which should always work.

Sometimes you may be tweaking comments in code without really changing other parts of the code, and you certainly do not want to update the cache database just because you updated the code comments. In this case, you can use the chunk option cache.comments = FALSE. Then comments will be excluded when calculating the MD5 hash, and therefore changes in comments will not affect the cache.

