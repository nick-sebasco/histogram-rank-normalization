# histogram-rank-normalization

This code is from the kidney image classifier.  The purpose of this project is to find a way to rank normalize that scales well for large datasets.  I wrote a detailed description of the new algorithm in the file `combined_age_score_spline.py`.


## Test data:
https://thejacksonlaboratory.box.com/s/5wjknw1ow4uw3w1zql3yw0uxcqlti6de


## Tests
```
py -3.8 -m pytest -s --log-cli-level=DEBUG -s .\tests\test_combined_age_scores.py
```

## Performance

Based on the profiling data:

### CMF Method:

    ~77,180 function calls

    Total time: ~1.015 seconds

### Spline Method (old method):

    ~6,676,786 function calls

    Total time: ~5.751 seconds

### Comparison:

Speed:
The CMF method is roughly 5–6 times faster than the spline method. Lower number of function calls (77K vs. over 6.6M).

Efficiency:
The CMF approach uses vectorized operations for computing the cumulative mass function and applies the inverse CDF directly to create the lookup table. In contrast, the spline method relies on spline interpolation and has more overhead with many more function calls.

Scalability:
The vectorized, streamlined CMF method is more scalable to larger datasets. With fewer function calls and a simpler workflow, it can process larger image datasets more efficiently.

## TODO:

### Vectorize inverse cdf somehow

From cprofile:

        ncalls  tottime  percall  cumtime  percall filename:lineno(function)

        1    0.009    0.009    5.119    5.119 combined_age_score_spline.py:266(combined_age_score_spline)

        3    0.074    0.025    4.655    1.552 combined_age_score_spline.py:124(create_spline_lut_from_histogram)

    30000    0.683    0.000    3.637    0.000 _distn_infrastructure.py:2311(ppf)


A significant amount of time is spent in the look up table creation process. Since ppf (inverse cdf) is being called once per bin inside a Python loop, if you use a very large number of bins (like 10,000 or more), this loop becomes a major bottleneck.

### Is a spline needed?

YOu could directly use the discrete cdf (computed from histogram) as map between bin and rank.  SHould be valid if histogram is goof approximation of underlying continuous distribution.  Problem is that to satisfy this assumption I think we need to increase number of bins.

### Assets

#### Histogram SPline Algorithm (1000 bins)
<img src="assets\distribution_comparison.png">

#### Old Algorithm
<img src="assets\old_distribution_comparison.png">

The normalized score distribution looks prettier from the old algorithm. 