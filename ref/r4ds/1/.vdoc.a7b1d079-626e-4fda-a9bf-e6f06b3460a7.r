sched_dep <- flights_dt |>
    mutate(minute = minute(sched_dep_time)) |>
    group_by(minute) |>
    summarize(
        avg_delay = mean(arr_delay, na.rm = TRUE),
        n = n()
    )

ggplot(sched_dep, aes(x = minute, y = avg_delay)) +
    geom_line()
