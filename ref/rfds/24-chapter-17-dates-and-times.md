# Chapter 17. Dates and Times

# Introduction

This chapter will show you how to work with dates and times in R. At first glance, dates and times seem simple. You use them all the time in your regular life, and they don’t seem to cause much confusion. However, the more you learn about dates and times, the more complicated they seem to get!

To warm up, think about how many days there are in a year and how many hours there are in a day. You probably remembered that most years have 365 days, but leap years have 366. Do you know the full rule for determining if a year is a leap year?<sup><a href="ch17.html#idm44771289225952" id="idm44771289225952-marker" data-type="noteref">1</a></sup> The number of hours in a day is a little less obvious: most days have 24 hours, but in places that use daylight saving time (DST), one day each year has 23 hours and another has 25.

Dates and times are hard because they have to reconcile two physical phenomena (the rotation of Earth and its orbit around the sun) with a whole raft of geopolitical phenomena including months, time zones, and DST. This chapter won’t teach you every last detail about dates and times, but it will give you a solid grounding of practical skills that will help you with common data analysis challenges.

We’ll begin by showing you how to create date-times from various inputs, and then once you’ve got a date-time, you’ll learn how you can extract components such as year, month, and day. We’ll then dive into the tricky topic of working with time spans, which come in a variety of flavors depending on what you’re trying to do. We’ll conclude with a brief discussion of the additional challenges posed by time zones.

## Prerequisites

This chapter will focus on the lubridate package, which makes it easier to work with dates and times in R. As of the latest tidyverse release, lubridate is part of core tidyverse. We will also need nycflights13 for practice data.

```
library(tidyverse)
library(nycflights13)
```

# Creating Date/Times

There are three types of date/time data that refer to an instant in time:

- A *date*. Tibbles print this as `<date>`.

- A *time* within a day. Tibbles print this as `<time>`.

- A *date-time* is a date plus a time: it uniquely identifies an instant in time (typically to the nearest second). Tibbles print this as `<dttm>`. Base R calls these POSIXct, but that doesn’t exactly trip off the tongue.

In this chapter we are going to focus on dates and date-times as R doesn’t have a native class for storing times. If you need one, you can use the hms package.

You should always use the simplest possible data type that works for your needs. That means if you can use a date instead of a date-time, you should. Date-times are substantially more complicated because of the need to handle time zones, which we’ll come back to at the end of the chapter.

To get the current date or date-time, you can use <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>today()</code></a> or <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>now()</code></a>:

```
today()
#> [1] "2023-03-12"
now()
#> [1] "2023-03-12 13:07:31 CDT"
```

Otherwise, the following sections describe the four ways you’re likely to create a date/time:

- While reading a file with readr
- From a string
- From individual date-time components
- From an existing date/time object

## During Import

If your CSV contains an ISO8601 date or date-time, you don’t need to do anything; readr will automatically recognize it:

```
csv <- "
  date,datetime
  2022-01-02,2022-01-02 05:12
"
read_csv(csv)
#> # A tibble: 1 × 2
#>   date       datetime           
#>   <date>     <dttm>             
#> 1 2022-01-02 2022-01-02 05:12:00
```

If you haven’t heard of *ISO8601* before, it’s an [international standard](https://oreil.ly/19K7t) for writing dates where the components of a date are organized from biggest to smallest separated by `-`. For example, in ISO8601 May 3, 2022, is `2022-05-03`. ISO8601 dates can also include times, where hour, minute, and second are separated by `:`, and the date and time components are separated by either a `T` or a space. For example, you could write 4:26 p.m. on May 3, 2022, as either `2022-05-03 16:26` or `2022-05-03T16:26`.

For other date-time formats, you’ll need to use `col_types` plus <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>col_date()</code></a> or <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>col_datetime()</code></a> along with a date-time format. The date-time format used by readr is a standard used across many programming languages, describing a date component with a `%` followed by a single character. For example, `%Y-%m-%d` specifies a date that’s a year, `-`, month (as number) `-`, day. <a href="#tbl-date-formats" data-type="xref">Table 17-1</a> lists all the options.

| Type  | Code  | Meaning                        | Example         |
|-------|-------|--------------------------------|-----------------|
| Year  | `%Y`  | 4-digit year                   | 2021            |
|       | `%y`  | 2-digit year                   | 21              |
| Month | `%m`  | Number                         | 2               |
|       | `%b`  | Abbreviated name               | Feb             |
|       | `%B`  | Full name                      | February        |
| Day   | `%d`  | Two digits                     | 02              |
|       | `%e`  | One or two digits              | 2               |
| Time  | `%H`  | 24-hour hour                   | 13              |
|       | `%I`  | 12-hour hour                   | 1               |
|       | `%p`  | a.m./p.m.                      | pm              |
|       | `%M`  | Minutes                        | 35              |
|       | `%S`  | Seconds                        | 45              |
|       | `%OS` | Seconds with decimal component | 45.35           |
|       | `%Z`  | Time zone name                 | America/Chicago |
|       | `%z`  | Offset from UTC                | +0800           |
| Other | `%.`  | Skip one nondigit              | :               |
|       | `%*`  | Skip any number of nondigits   |                 |

Table 17-1. All date formats understood by readr {#tbl-date-formats .table}

This code shows a few options applied to a very ambiguous date:

```
csv <- "
  date
  01/02/15
"

read_csv(csv, col_types = cols(date = col_date("%m/%d/%y")))
#> # A tibble: 1 × 1
#>   date      
#>   <date>    
#> 1 2015-01-02

read_csv(csv, col_types = cols(date = col_date("%d/%m/%y")))
#> # A tibble: 1 × 1
#>   date      
#>   <date>    
#> 1 2015-02-01

read_csv(csv, col_types = cols(date = col_date("%y/%m/%d")))
#> # A tibble: 1 × 1
#>   date      
#>   <date>    
#> 1 2001-02-15
```

Note that no matter how you specify the date format, it’s always displayed the same way once you get it into R.

If you’re using `%b` or `%B` and working with non-English dates, you’ll also need to provide a <a href="https://readr.tidyverse.org/reference/locale.html" class="orm:hideurl"><code>locale()</code></a>. See the list of built-in languages in <a href="https://readr.tidyverse.org/reference/date_names.html" class="orm:hideurl"><code>date_names_langs()</code></a>, or create your own with <a href="https://readr.tidyverse.org/reference/date_names.html" class="orm:hideurl"><code>date_names()</code></a>.

## From Strings

The date-time specification language is powerful but requires careful analysis of the date format. An alternative approach is to use lubridate’s helpers, which attempt to automatically determine the format once you specify the order of the component. To use them, identify the order in which year, month, and day appear in your dates; then arrange “y,” “m,” and “d” in the same order. That gives you the name of the lubridate function that will parse your date. For example:

```
ymd("2017-01-31")
#> [1] "2017-01-31"
mdy("January 31st, 2017")
#> [1] "2017-01-31"
dmy("31-Jan-2017")
#> [1] "2017-01-31"
```

<a href="https://lubridate.tidyverse.org/reference/ymd.html" class="orm:hideurl"><code>ymd()</code></a> and friends create dates. To create a date-time, add an underscore and one or more of “h”, “m”, and “s” to the name of the parsing function:

```
ymd_hms("2017-01-31 20:11:59")
#> [1] "2017-01-31 20:11:59 UTC"
mdy_hm("01/31/2017 08:01")
#> [1] "2017-01-31 08:01:00 UTC"
```

You can also force the creation of a date-time from a date by supplying a time zone:

```
ymd("2017-01-31", tz = "UTC")
#> [1] "2017-01-31 UTC"
```

Here I use the UTC<sup><a href="ch17.html#idm44771288853952" id="idm44771288853952-marker" data-type="noteref">2</a></sup> timezone, which you might also know as GMT, or Greenwich Mean Time, the time at 0° longitude.<sup><a href="ch17.html#idm44771288853216" id="idm44771288853216-marker" data-type="noteref">3</a></sup> It doesn’t use daylight saving time, making it a bit easier to compute with.

## From Individual Components

Instead of a single string, sometimes you’ll have the individual components of the date-time spread across multiple columns. This is what we have in the `flights` data:

```
flights |> 
  select(year, month, day, hour, minute)
#> # A tibble: 336,776 × 5
#>    year month   day  hour minute
#>   <int> <int> <int> <dbl>  <dbl>
#> 1  2013     1     1     5     15
#> 2  2013     1     1     5     29
#> 3  2013     1     1     5     40
#> 4  2013     1     1     5     45
#> 5  2013     1     1     6      0
#> 6  2013     1     1     5     58
#> # … with 336,770 more rows
```

To create a date/time from this sort of input, use <a href="https://lubridate.tidyverse.org/reference/make_datetime.html" class="orm:hideurl"><code>make_date()</code></a> for dates, or use <a href="https://lubridate.tidyverse.org/reference/make_datetime.html" class="orm:hideurl"><code>make_datetime()</code></a> for date-times:

```
flights |> 
  select(year, month, day, hour, minute) |> 
  mutate(departure = make_datetime(year, month, day, hour, minute))
#> # A tibble: 336,776 × 6
#>    year month   day  hour minute departure          
#>   <int> <int> <int> <dbl>  <dbl> <dttm>             
#> 1  2013     1     1     5     15 2013-01-01 05:15:00
#> 2  2013     1     1     5     29 2013-01-01 05:29:00
#> 3  2013     1     1     5     40 2013-01-01 05:40:00
#> 4  2013     1     1     5     45 2013-01-01 05:45:00
#> 5  2013     1     1     6      0 2013-01-01 06:00:00
#> 6  2013     1     1     5     58 2013-01-01 05:58:00
#> # … with 336,770 more rows
```

Let’s do the same thing for each of the four time columns in `flights`. The times are represented in a slightly odd format, so we use modulus arithmetic to pull out the hour and minute components. Once we’ve created the date-time variables, we focus in on the variables we’ll explore in the rest of the chapter.

```
make_datetime_100 <- function(year, month, day, time) {
  make_datetime(year, month, day, time %/% 100, time %% 100)
}

flights_dt <- flights |> 
  filter(!is.na(dep_time), !is.na(arr_time)) |> 
  mutate(
    dep_time = make_datetime_100(year, month, day, dep_time),
    arr_time = make_datetime_100(year, month, day, arr_time),
    sched_dep_time = make_datetime_100(year, month, day, sched_dep_time),
    sched_arr_time = make_datetime_100(year, month, day, sched_arr_time)
  ) |> 
  select(origin, dest, ends_with("delay"), ends_with("time"))

flights_dt
#> # A tibble: 328,063 × 9
#>   origin dest  dep_delay arr_delay dep_time            sched_dep_time     
#>   <chr>  <chr>     <dbl>     <dbl> <dttm>              <dttm>             
#> 1 EWR    IAH           2        11 2013-01-01 05:17:00 2013-01-01 05:15:00
#> 2 LGA    IAH           4        20 2013-01-01 05:33:00 2013-01-01 05:29:00
#> 3 JFK    MIA           2        33 2013-01-01 05:42:00 2013-01-01 05:40:00
#> 4 JFK    BQN          -1       -18 2013-01-01 05:44:00 2013-01-01 05:45:00
#> 5 LGA    ATL          -6       -25 2013-01-01 05:54:00 2013-01-01 06:00:00
#> 6 EWR    ORD          -4        12 2013-01-01 05:54:00 2013-01-01 05:58:00
#> # … with 328,057 more rows, and 3 more variables: arr_time <dttm>,
#> #   sched_arr_time <dttm>, air_time <dbl>
```

With this data, we can visualize the distribution of departure times across the year:

```
flights_dt |> 
  ggplot(aes(x = dep_time)) + 
  geom_freqpoly(binwidth = 86400) # 86400 seconds = 1 day
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in01.png" alt="A frequency polyon with departure time (Jan-Dec 2013) on the x-axis and number of flights on the y-axis (0-1000). The frequency polygon is binned by day so you see a time series of flights by day. The pattern is dominated by a weekly pattern; there are fewer flights on weekends. The are few days that stand out as having a surprisingly few flights in early February, early July, late November, and late December." />
</figure>

Or within a single day:

```
flights_dt |> 
  filter(dep_time < ymd(20130102)) |> 
  ggplot(aes(x = dep_time)) + 
  geom_freqpoly(binwidth = 600) # 600 s = 10 minutes
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in02.png" alt="A frequency polygon with departure time (6am - midnight Jan 1) on the x-axis, number of flights on the y-axis (0-17), binned into 10 minute increments. It&#39;s hard to see much pattern because of high variability, but most bins have 8-12 flights, and there are markedly fewer flights before 6am and after 8pm." />
</figure>

Note that when you use date-times in a numeric context (like in a histogram), 1 means 1 second, so a binwidth of 86400 means one day. For dates, 1 means 1 day.

## From Other Types

You may want to switch between a date-time and a date. That’s the job of <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_datetime()</code></a> and <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_date()</code></a>:

```
as_datetime(today())
#> [1] "2023-03-12 UTC"
as_date(now())
#> [1] "2023-03-12"
```

Sometimes you’ll get date/times as numeric offsets from the “Unix epoch,” 1970-01-01. If the offset is in seconds, use <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_datetime()</code></a>; if it’s in days, use <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_date()</code></a>.

```
as_datetime(60 * 60 * 10)
#> [1] "1970-01-01 10:00:00 UTC"
as_date(365 * 10 + 2)
#> [1] "1980-01-01"
```

## Exercises

1.  What happens if you parse a string that contains invalid dates?

    ```
    ymd(c("2010-10-10", "bananas"))
    ```

2.  What does the `tzone` argument to <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>today()</code></a> do? Why is it important?

3.  For each of the following date-times, show how you’d parse it using a readr column specification and a lubridate function.

    ```
    d1 <- "January 1, 2010"
    d2 <- "2015-Mar-07"
    d3 <- "06-Jun-2017"
    d4 <- c("August 19 (2015)", "July 1 (2015)")
    d5 <- "12/30/14" # Dec 30, 2014
    t1 <- "1705"
    t2 <- "11:15:10.12 PM"
    ```

# Date-Time Components

Now that you know how to get date-time data into R’s date-time data structures, let’s explore what you can do with them. This section will focus on the accessor functions that let you get and set individual components. The next section will look at how arithmetic works with date-times.

## Getting Components

You can pull out individual parts of the date with the accessor functions <a href="https://lubridate.tidyverse.org/reference/year.html" class="orm:hideurl"><code>year()</code></a>, <a href="https://lubridate.tidyverse.org/reference/month.html" class="orm:hideurl"><code>month()</code></a>, <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>mday()</code></a> (day of the month), <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>yday()</code></a> (day of the year), <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>wday()</code></a> (day of the week), <a href="https://lubridate.tidyverse.org/reference/hour.html" class="orm:hideurl"><code>hour()</code></a>, <a href="https://lubridate.tidyverse.org/reference/minute.html" class="orm:hideurl"><code>minute()</code></a>, and <a href="https://lubridate.tidyverse.org/reference/second.html" class="orm:hideurl"><code>second()</code></a>. These are effectively the opposites of <a href="https://lubridate.tidyverse.org/reference/make_datetime.html" class="orm:hideurl"><code>make_datetime()</code></a>.

```
datetime <- ymd_hms("2026-07-08 12:34:56")

year(datetime)
#> [1] 2026
month(datetime)
#> [1] 7
mday(datetime)
#> [1] 8

yday(datetime)
#> [1] 189
wday(datetime)
#> [1] 4
```

For <a href="https://lubridate.tidyverse.org/reference/month.html" class="orm:hideurl"><code>month()</code></a> and <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>wday()</code></a> you can set `label = TRUE` to return the abbreviated name of the month or day of the week. Set `abbr = FALSE` to return the full name.

```
month(datetime, label = TRUE)
#> [1] Jul
#> 12 Levels: Jan < Feb < Mar < Apr < May < Jun < Jul < Aug < Sep < ... < Dec
wday(datetime, label = TRUE, abbr = FALSE)
#> [1] Wednesday
#> 7 Levels: Sunday < Monday < Tuesday < Wednesday < Thursday < ... < Saturday
```

We can use <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>wday()</code></a> to see that more flights depart during the week than on the weekend:

```
flights_dt |> 
  mutate(wday = wday(dep_time, label = TRUE)) |> 
  ggplot(aes(x = wday)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in03.png" alt="A bar chart with days of the week on the x-axis and number of flights on the y-axis. Monday-Friday have roughly the same number of flights, ~48,0000, decreasingly slightly over the course of the week. Sunday is a little lower (~45,000), and Saturday is much lower (~38,000)." />
</figure>

We can also look at the average departure delay by minute within the hour. There’s an interesting pattern: flights leaving in minutes 20–30 and 50–60 have much lower delays than the rest of the hour!

```
flights_dt |> 
  mutate(minute = minute(dep_time)) |> 
  group_by(minute) |> 
  summarize(
    avg_delay = mean(dep_delay, na.rm = TRUE),
    n = n()
  ) |> 
  ggplot(aes(x = minute, y = avg_delay)) +
  geom_line()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in04.png" alt="A line chart with minute of actual departure (0-60) on the x-axis and average delay (4-20) on the y-axis. Average delay starts at (0, 12), steadily increases to (18, 20), then sharply drops, hitting at minimum at ~23 minute past the hour and 9 minutes of delay. It then increases again to (17, 35), and sharply decreases to (55, 4). It finishes off with an increase to (60, 9)." />
</figure>

Interestingly, if we look at the *scheduled* departure time, we don’t see such a strong pattern:

```
sched_dep <- flights_dt |> 
  mutate(minute = minute(sched_dep_time)) |> 
  group_by(minute) |> 
  summarize(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()
  )

ggplot(sched_dep, aes(x = minute, y = avg_delay)) +
  geom_line()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in05.png" alt="A line chart with minute of scheduled departure (0-60) on the x-axis and average delay (4-16). There is relatively little pattern, just a small suggestion that the average delay decreases from maybe 10 minutes to 8 minutes over the course of the hour." />
</figure>

So why do we see that pattern with the actual departure times? Well, like much data collected by humans, there’s a strong bias toward flights leaving at “nice” departure times, as <a href="#fig-human-rounding" data-type="xref">Figure 17-1</a> shows. Always be alert for this sort of pattern whenever you work with data that involves human judgment!

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1701.png" alt="A line plot with departure minute (0-60) on the x-axis and number of flights (0-60000) on the y-axis. Most flights are scheduled to depart on either the hour (~60,000) or the half hour (~35,000). Otherwise, all most all flights are scheduled to depart on multiples of five, with a few extra at 15, 45, and 55 minutes. " />
<h6 id="figure-17-1.-a-frequency-polygon-showing-the-number-of-flights-scheduled-to-depart-each-hour.-you-can-see-a-strong-preference-for-round-numbers-like-0-and-30-and-generally-for-numbers-that-are-a-multiple-of-five.">Figure 17-1. A frequency polygon showing the number of flights scheduled to depart each hour. You can see a strong preference for round numbers like 0 and 30 and generally for numbers that are a multiple of five.</h6>
</figure>

## Rounding

An alternative approach to plotting individual components is to round the date to a nearby unit of time, with <a href="https://lubridate.tidyverse.org/reference/round_date.html" class="orm:hideurl"><code>floor_date()</code></a>, <a href="https://lubridate.tidyverse.org/reference/round_date.html" class="orm:hideurl"><code>round_date()</code></a>, and <a href="https://lubridate.tidyverse.org/reference/round_date.html" class="orm:hideurl"><code>ceiling_date()</code></a>. Each function takes a vector of dates to adjust and then the name of the unit to round down (floor), round up (ceiling), or round to. This, for example, allows us to plot the number of flights per week:

```
flights_dt |> 
  count(week = floor_date(dep_time, "week")) |> 
  ggplot(aes(x = week, y = n)) +
  geom_line() + 
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in06.png" alt="A line plot with week (Jan-Dec 2013) on the x-axis and number of flights (2,000-7,000) on the y-axis. The pattern is fairly flat from February to November with around 7,000 flights per week. There are far fewer flights on the first (approximately 4,500 flights) and last weeks of the year (approximately 2,500 flights)." />
</figure>

You can use rounding to show the distribution of flights across the course of a day by computing the difference between `dep_time` and the earliest instant of that day:

```
flights_dt |> 
  mutate(dep_hour = dep_time - floor_date(dep_time, "day")) |> 
  ggplot(aes(x = dep_hour)) +
  geom_freqpoly(binwidth = 60 * 30)
#> Don't know how to automatically pick scale for object of type <difftime>.
#> Defaulting to continuous.
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in07.png" alt="A line plot with depature time on the x-axis. This is units of seconds since midnight so it&#39;s hard to interpret." />
</figure>

Computing the difference between a pair of date-times yields a difftime (more on that in <a href="#sec-intervals" data-type="xref">“Intervals”</a>). We can convert that to an `hms` object to get a more useful x-axis:

```
flights_dt |> 
  mutate(dep_hour = hms::as_hms(dep_time - floor_date(dep_time, "day"))) |> 
  ggplot(aes(x = dep_hour)) +
  geom_freqpoly(binwidth = 60 * 30)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in08.png" alt="A line plot with departure time (midnight to midnight) on the x-axis and number of flights on the y-axis (0 to 15,000). There are very few (&lt;100) flights before 5am. The number of flights then rises rapidly to 12,000 / hour, peaking at 15,000 at 9am, before falling to around 8,000 / hour for 10am to 2pm. Number of flights then increases to around 12,000 per hour until 8pm, when they rapidly drop again." />
</figure>

## Modifying Components

You can also use each accessor function to modify the components of a date/time. This doesn’t come up much in data analysis but can be useful when cleaning data that has clearly incorrect dates.

``` p
(datetime <- ymd_hms("2026-07-08 12:34:56"))
#> [1] "2026-07-08 12:34:56 UTC"

year(datetime) <- 2030
datetime
#> [1] "2030-07-08 12:34:56 UTC"
month(datetime) <- 01
datetime
#> [1] "2030-01-08 12:34:56 UTC"
hour(datetime) <- hour(datetime) + 1
datetime
#> [1] "2030-01-08 13:34:56 UTC"
```

Alternatively, rather than modifying an existing variable, you can create a new date-time with <a href="https://rdrr.io/r/stats/update.html" class="orm:hideurl"><code>update()</code></a>. This also allows you to set multiple values in one step:

```
update(datetime, year = 2030, month = 2, mday = 2, hour = 2)
#> [1] "2030-02-02 02:34:56 UTC"
```

If values are too big, they will roll over:

```
update(ymd("2023-02-01"), mday = 30)
#> [1] "2023-03-02"
update(ymd("2023-02-01"), hour = 400)
#> [1] "2023-02-17 16:00:00 UTC"
```

## Exercises

1.  How does the distribution of flight times within a day change over the course of the year?

2.  Compare `dep_time`, `sched_dep_time`, and `dep_delay`. Are they consistent? Explain your findings.

3.  Compare `air_time` with the duration between the departure and arrival. Explain your findings. (Hint: Consider the location of the airport.)

4.  How does the average delay time change over the course of a day? Should you use `dep_time` or `sched_dep_time`? Why?

5.  On what day of the week should you leave if you want to minimize the chance of a delay?

6.  What makes the distribution of `diamonds$carat` and `flights$sched_dep_time` similar?

7.  Confirm our hypothesis that the early departures of flights in minutes 20–30 and 50–60 are caused by scheduled flights that leave early. Hint: Create a binary variable that tells you whether a flight was delayed.

# Time Spans

Next you’ll learn about how arithmetic with dates works, including subtraction, addition, and division. Along the way, you’ll learn about three important classes that represent time spans:

Durations  
Represent an exact number of seconds

Periods  
Represent human units like weeks and months

Intervals  
Represent a starting and ending point

How do you pick between duration, periods, and intervals? As always, pick the simplest data structure that solves your problem. If you care only about physical time, use a duration; if you need to add human times, use a period; and if you need to figure out how long a span is in human units, use an interval.

## Durations

In R, when you subtract two dates, you get a `difftime` object:

```
# How old is Hadley?
h_age <- today() - ymd("1979-10-14")
h_age
#> Time difference of 15855 days
```

A `difftime` class object records a time span of seconds, minutes, hours, days, or weeks. This ambiguity can make difftimes a little painful to work with, so lubridate provides an alternative that always uses seconds: the *duration*.

```
as.duration(h_age)
#> [1] "1369872000s (~43.41 years)"
```

Durations come with a bunch of convenient constructors:

```
dseconds(15)
#> [1] "15s"
dminutes(10)
#> [1] "600s (~10 minutes)"
dhours(c(12, 24))
#> [1] "43200s (~12 hours)" "86400s (~1 days)"
ddays(0:5)
#> [1] "0s"                "86400s (~1 days)"  "172800s (~2 days)"
#> [4] "259200s (~3 days)" "345600s (~4 days)" "432000s (~5 days)"
dweeks(3)
#> [1] "1814400s (~3 weeks)"
dyears(1)
#> [1] "31557600s (~1 years)"
```

Durations always record the time span in seconds. Larger units are created by converting minutes, hours, days, weeks, and years to seconds: 60 seconds in a minute, 60 minutes in an hour, 24 hours in a day, and 7 days in a week. Larger time units are more problematic. A year uses the “average” number of days in a year, i.e., 365.25. There’s no way to convert a month to a duration, because there’s just too much variation.

You can add and multiply durations:

``` m
2 * dyears(1)
#> [1] "63115200s (~2 years)"
dyears(1) + dweeks(12) + dhours(15)
#> [1] "38869200s (~1.23 years)"
```

You can add and subtract durations to and from days:

```
tomorrow <- today() + ddays(1)
last_year <- today() - dyears(1)
```

However, because durations represent an exact number of seconds, sometimes you might get an unexpected result:

```
one_am <- ymd_hms("2026-03-08 01:00:00", tz = "America/New_York")

one_am
#> [1] "2026-03-08 01:00:00 EST"
one_am + ddays(1)
#> [1] "2026-03-09 02:00:00 EDT"
```

Why is one day after 1 a.m. March 8, returning as 2 a.m. on March 9? If you look carefully at the date, you might also notice that the time zones have changed. March 8 has only 23 hours because it’s when DST starts, so if we add a full day’s worth of seconds, we end up with a different time.

## Periods

To solve this problem, lubridate provides *periods*. Periods are time spans but don’t have a fixed length in seconds; instead, they work with “human” times, like days and months. That allows them to work in a more intuitive way:

```
one_am
#> [1] "2026-03-08 01:00:00 EST"
one_am + days(1)
#> [1] "2026-03-09 01:00:00 EDT"
```

Like durations, periods can be created with a number of friendly constructor functions:

```
hours(c(12, 24))
#> [1] "12H 0M 0S" "24H 0M 0S"
days(7)
#> [1] "7d 0H 0M 0S"
months(1:6)
#> [1] "1m 0d 0H 0M 0S" "2m 0d 0H 0M 0S" "3m 0d 0H 0M 0S" "4m 0d 0H 0M 0S"
#> [5] "5m 0d 0H 0M 0S" "6m 0d 0H 0M 0S"
```

You can add and multiply periods:

``` m
10 * (months(6) + days(1))
#> [1] "60m 10d 0H 0M 0S"
days(50) + hours(25) + minutes(2)
#> [1] "50d 25H 2M 0S"
```

And of course, add them to dates. Compared to durations, periods are more likely to do what you expect:

```
# A leap year
ymd("2024-01-01") + dyears(1)
#> [1] "2024-12-31 06:00:00 UTC"
ymd("2024-01-01") + years(1)
#> [1] "2025-01-01"

# Daylight savings time
one_am + ddays(1)
#> [1] "2026-03-09 02:00:00 EDT"
one_am + days(1)
#> [1] "2026-03-09 01:00:00 EDT"
```

Let’s use periods to fix an oddity related to our flight dates. Some planes appear to have arrived at their destination *before* they departed from New York City:

```
flights_dt |> 
  filter(arr_time < dep_time) 
#> # A tibble: 10,633 × 9
#>   origin dest  dep_delay arr_delay dep_time            sched_dep_time     
#>   <chr>  <chr>     <dbl>     <dbl> <dttm>              <dttm>             
#> 1 EWR    BQN           9        -4 2013-01-01 19:29:00 2013-01-01 19:20:00
#> 2 JFK    DFW          59        NA 2013-01-01 19:39:00 2013-01-01 18:40:00
#> 3 EWR    TPA          -2         9 2013-01-01 20:58:00 2013-01-01 21:00:00
#> 4 EWR    SJU          -6       -12 2013-01-01 21:02:00 2013-01-01 21:08:00
#> 5 EWR    SFO          11       -14 2013-01-01 21:08:00 2013-01-01 20:57:00
#> 6 LGA    FLL         -10        -2 2013-01-01 21:20:00 2013-01-01 21:30:00
#> # … with 10,627 more rows, and 3 more variables: arr_time <dttm>,
#> #   sched_arr_time <dttm>, air_time <dbl>
```

These are overnight flights. We used the same date information for both the departure and the arrival times, but these flights arrived on the following day. We can fix this by adding `days(1)` to the arrival time of each overnight flight:

```
flights_dt <- flights_dt |> 
  mutate(
    overnight = arr_time < dep_time,
    arr_time = arr_time + days(overnight),
    sched_arr_time = sched_arr_time + days(overnight)
  )
```

Now all of our flights obey the laws of physics:

```
flights_dt |> 
  filter(arr_time < dep_time) 
#> # A tibble: 0 × 10
# … with 10 variables: origin <chr>, dest <chr>, dep_delay <dbl>,
#   arr_delay <dbl>, dep_time <dttm>, sched_dep_time <dttm>, …
# ℹ Use `colnames()` to see all variable names
#> # … with 10,627 more rows, and 4 more variables: 
```

## Intervals

What does `dyears(1) / ddays(365)` return? It’s not quite 1, because `dyears()` is defined as the number of seconds per average year, which is 365.25 days.

What does `years(1) / days(1)` return? Well, if the year is 2015, it should return 365, but if it is 2016, it should return 366! There’s not quite enough information for lubridate to give a single clear answer. What it does instead is give an estimate:

```
years(1) / days(1)
#> [1] 365.25
```

If you want a more accurate measurement, you’ll have to use an *interval*. An interval is a pair of starting and ending date times, or you can think of it as a duration with a starting point.

You can create an interval by writing `start %--% end`:

```
y2023 <- ymd("2023-01-01") %--% ymd("2024-01-01")
y2024 <- ymd("2024-01-01") %--% ymd("2025-01-01")

y2023
#> [1] 2023-01-01 UTC--2024-01-01 UTC
y2024
#> [1] 2024-01-01 UTC--2025-01-01 UTC
```

You could then divide it by <a href="https://lubridate.tidyverse.org/reference/period.html" class="orm:hideurl"><code>days()</code></a> to find out how many days fit in the year:

```
y2023 / days(1)
#> [1] 365
y2024 / days(1)
#> [1] 366
```

## Exercises

1.  Explain `days(!overnight)` and `days(overnight)` to someone who has just started learning R. What is the key fact you need to know?

2.  Create a vector of dates giving the first day of every month in 2015. Create a vector of dates giving the first day of every month in the *current* year.

3.  Write a function that, given your birthday (as a date), returns how old you are in years.

4.  Why can’t `(today() %--% (today() + years(1))) / months(1)` work?

# Time Zones

Time zones are an enormously complicated topic because of their interaction with geopolitical entities. Fortunately we don’t need to dig into all the details as they’re not all important for data analysis, but there are a few challenges we’ll need to tackle head on.

The first challenge is that everyday names of time zones tend to be ambiguous. For example, if you’re American, you’re probably familiar with Eastern Standard Time (EST). However, both Australia and Canada also have EST! To avoid confusion, R uses the international standard IANA time zones. These use a consistent naming scheme `{area}/{location}`, typically in the form `{continent}/{city}` or `{ocean}/{city}`. Examples include “America/New_York,” “Europe/Paris,” and “Pacific/Auckland.”

You might wonder why the time zone uses a city when typically you think of time zones as associated with a country or region within a country. This is because the IANA database has to record decades worth of time zone rules. Over the course of decades, countries change names (or break apart) fairly frequently, but city names tend to stay the same. Another problem is that the name needs to reflect not only the current behavior but also the complete history. For example, there are time zones for both “America/New_York” and “America/Detroit.” These cities both currently use Eastern Standard Time, but in 1969–1972 Michigan (the state in which Detroit is located) did not follow DST, so it needs a different name. It’s worth reading the [raw time zone database](https://oreil.ly/NwvsT) just to read some of these stories!

You can find out what R thinks your current time zone is with <a href="https://rdrr.io/r/base/timezones.html" class="orm:hideurl"><code>Sys.timezone()</code></a>:

```
Sys.timezone()
#> [1] "America/Chicago"
```

(If R doesn’t know, you’ll get an `NA`.)

And see the complete list of all time zone names with <a href="https://rdrr.io/r/base/timezones.html" class="orm:hideurl"><code>OlsonNames()</code></a>:

```
length(OlsonNames())
#> [1] 597
head(OlsonNames())
#> [1] "Africa/Abidjan"     "Africa/Accra"       "Africa/Addis_Ababa"
#> [4] "Africa/Algiers"     "Africa/Asmara"      "Africa/Asmera"
```

In R, the time zone is an attribute of the date-time that only controls printing. For example, these three objects represent the same instant in time:

```
x1 <- ymd_hms("2024-06-01 12:00:00", tz = "America/New_York")
x1
#> [1] "2024-06-01 12:00:00 EDT"

x2 <- ymd_hms("2024-06-01 18:00:00", tz = "Europe/Copenhagen")
x2
#> [1] "2024-06-01 18:00:00 CEST"

x3 <- ymd_hms("2024-06-02 04:00:00", tz = "Pacific/Auckland")
x3
#> [1] "2024-06-02 04:00:00 NZST"
```

You can verify that they’re the same time using subtraction:

```
x1 - x2
#> Time difference of 0 secs
x1 - x3
#> Time difference of 0 secs
```

Unless otherwise specified, lubridate always uses UTC. UTC is the standard time zone used by the scientific community and is roughly equivalent to GMT. It does not have DST, which makes a convenient representation for computation. Operations that combine date-times, like <a href="https://rdrr.io/r/base/c.html" class="orm:hideurl"><code>c()</code></a>, will often drop the time zone. In that case, the date-times will display in the time zone of the first element:

```
x4 <- c(x1, x2, x3)
x4
#> [1] "2024-06-01 12:00:00 EDT" "2024-06-01 12:00:00 EDT"
#> [3] "2024-06-01 12:00:00 EDT"
```

You can change the time zone in two ways:

- Keep the instant in time the same, and change how it’s displayed. Use this when the instant is correct but you want a more natural display.

  ```
  x4a <- with_tz(x4, tzone = "Australia/Lord_Howe")
  x4a
  #> [1] "2024-06-02 02:30:00 +1030" "2024-06-02 02:30:00 +1030"
  #> [3] "2024-06-02 02:30:00 +1030"
  x4a - x4
  #> Time differences in secs
  #> [1] 0 0 0
  ```

  (This also illustrates another challenge of time zones: they’re not all integer hour offsets!)

- Change the underlying instant in time. Use this when you have an instant that has been labeled with the incorrect time zone and you need to fix it.

  ```
  x4b <- force_tz(x4, tzone = "Australia/Lord_Howe")
  x4b
  #> [1] "2024-06-01 12:00:00 +1030" "2024-06-01 12:00:00 +1030"
  #> [3] "2024-06-01 12:00:00 +1030"
  x4b - x4
  #> Time differences in hours
  #> [1] -14.5 -14.5 -14.5
  ```

# Summary

This chapter introduced you to the tools that lubridate provides to help you work with date-time data. Working with dates and times can seem harder than necessary, but we hope this chapter has helped you see why—date-times are more complex than they seem at first glance, and handling every possible situation adds complexity. Even if your data never crosses a DST boundary or involves a leap year, the functions need to be able to handle it.

The next chapter gives a roundup of missing values. You’ve seen them in a few places and have no doubt encountered them in your own analysis, and it’s now time to provide a grab bag of useful techniques for dealing with them.

<sup>[1](ch17.html#idm44771289225952-marker)</sup> A year is a leap year if it’s divisible by 4, unless it’s also divisible by 100, except if it’s also divisible by 400. In other words, in every set of 400 years, there’s 97 leap years.

<sup>[2](ch17.html#idm44771288853952-marker)</sup> You might wonder what UTC stands for. It’s a compromise between the English “Coordinated Universal Time” and French “Temps Universel Coordonné.”

<sup>[3](ch17.html#idm44771288853216-marker)</sup> No prizes for guessing which country came up with the longitude system.
