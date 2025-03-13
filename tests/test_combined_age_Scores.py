import os
import logging
import zarr
import cProfile
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from combined_age_score_stitched import combined_age_score
from combined_age_score_spline import (
    build_histograms_and_threshold_spline,
    create_spline_lut_from_histogram,
    apply_spline_histogram_lut,
    combined_age_score_spline
)


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------
# Original Algorithm Tests
# ---------------------------------------------------------------------------------------
def test_combined_age_score(output_score_path, image_dir, combined_score_path):
    bounds = combined_age_score(
        output_score_path,
        str(combined_score_path),
        image_dir
    )
    logging.info("Min: %s, Max: %s", bounds[0], bounds[1])


def test_visualization_old(tmp_path, combined_score_path, output_score_path, image_dir):
    """
    Visualize normalized output and then compares the raw vs. old-normalized
    distributions with a histogram plot.
    """
    # Run the old function
    combined_age_score(
        output_score_path,
        str(combined_score_path) + os.sep,
        image_dir=image_dir,
        thresh_method="GaussianBlur",
        thresh_param=200,
    )

    # Gather raw files and old-normalized files
    raw_files = sorted([f for f in os.listdir(output_score_path) if f.endswith(".zarr")])
    norm_files = sorted([f for f in os.listdir(combined_score_path) if f.endswith("_age_scores.zarr")])

    assert len(raw_files) > 0, "No raw zarr files found."
    assert len(norm_files) > 0, "No old-normalized files found."

    # Load the first raw zarr and the corresponding old-normalized zarr
    raw_zarr_path = os.path.join(output_score_path, raw_files[0])
    norm_zarr_path = os.path.join(combined_score_path, raw_files[0].replace(".zarr", "_age_scores.zarr"))

    raw_arr = zarr.open(raw_zarr_path)[:]
    norm_arr = zarr.open(norm_zarr_path)[:]

    # Flatten for histogram
    raw_flat = raw_arr.flatten()
    norm_flat = norm_arr.flatten()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(raw_flat, bins=50)
    axes[0].set_title("Raw Score Distribution (Old)")

    axes[1].hist(norm_flat, bins=50)
    axes[1].set_title("Old Normalized Score Distribution")

    # Save figure to tmp_path
    fig_file = tmp_path / "old_distribution_comparison.png"
    plt.savefig(str(fig_file))
    plt.close(fig)

    # Optional: Check mean and std
    mean_norm = np.mean(norm_flat)
    std_norm = np.std(norm_flat)
    logging.info(f"Old method normalized mean={mean_norm:.3f}, std={std_norm:.3f}")
    # If you want to assert they're near 0, 1:
    assert abs(mean_norm) < 1.0, f"Mean {mean_norm} is too far from 0"
    assert abs(std_norm - 1.0) < 1.0, f"Std {std_norm} is too far from 1"


# Spline Algorithm Tests
# ---------------------------------------------------------------------------------------
def test_sanity_spline_histograms(output_score_path, image_dir):
    hist_ch0, hist_ch1, hist_ch2 = build_histograms_and_threshold_spline(
        score_dir=output_score_path,
        image_dir=image_dir,
        thresh_method="GaussianBlur",
        thresh_param=200,
        n_bins=1000,
        multiply_factor=10000
    )

    # Basic checks:
    assert hist_ch0.shape == (1000,)
    assert hist_ch1.shape == (1000,)
    assert hist_ch2.shape == (1000,)
    logging.info(
        "Histogram shapes 0: %s, 1: %s, 2: %s",
        hist_ch0.shape,
        hist_ch2.shape,
        hist_ch2.shape
    )

    # Hist counts shouldn't all be zero if there's any foreground data
    total_ch0 = hist_ch0.sum()
    total_ch1 = hist_ch1.sum()
    total_ch2 = hist_ch2.sum()
    logging.info(
        "Histogram counts 0: %s, 1: %s, 2: %s",
        total_ch0,
        total_ch1,
        total_ch2
    )
    assert total_ch0 > 0, "No data in channel 0 histogram!"
    assert total_ch1 > 0, "No data in channel 1 histogram!"
    assert total_ch2 > 0, "No data in channel 2 histogram!"


def test_create_spline_lut_from_histogram():
    """
    Test that a synthetic histogram produces a monotonic increasing LUT.
    """
    # Create a simple synthetic histogram: a ramp from 1 to 100 over 1000 bins.
    hist = np.linspace(1, 100, 1000, dtype=np.uint64)
    lut = create_spline_lut_from_histogram(hist, dist=scipy.stats.norm)
    assert lut.shape == (1000,)
    # Check that the LUT is (mostly) monotonic increasing.
    assert np.all(np.diff(lut) >= 0), "The LUT is not monotonic increasing."


def test_apply_spline_histogram_lut(output_score_path, image_dir, combined_score_path):
    """
    Test that the second pass runs without error and produces output files of expected shape.
    """
    
    # Build histograms and LUTs.
    hist0, hist1, hist2 = build_histograms_and_threshold_spline(
        score_dir=output_score_path,
        image_dir=image_dir,
        thresh_method="GaussianBlur",
        thresh_param=200,
        n_bins=1000,
        multiply_factor=10000
    )
    lut0 = create_spline_lut_from_histogram(hist0, dist=scipy.stats.norm)
    lut1 = create_spline_lut_from_histogram(hist1, dist=scipy.stats.norm)
    lut2 = create_spline_lut_from_histogram(hist2, dist=scipy.stats.norm)
    
    apply_spline_histogram_lut(
        score_dir=output_score_path,
        output_dir=str(combined_score_path),
        image_dir=image_dir,
        lut0=lut0,
        lut1=lut1,
        lut2=lut2,
        multiply_factor=10000,
        thresh_method="GaussianBlur",
        thresh_param=200
    )
    output_files = list(combined_score_path.glob("*_age_scores.zarr"))
    assert len(output_files) > 0, "No output normalized score files were generated."
    
    # Check that the output has the expected shape.
    sample_arr = zarr.open(str(output_files[0]))[:]
    # If input was 4D, our updated function outputs (H, W, 1, patches).
    # If input was 3D, output should be (H, W, 1).
    if len(sample_arr.shape) == 4:
        assert sample_arr.shape[2] == 1, "Expected a single channel in the output (4D case)."
    elif len(sample_arr.shape) == 3:
        assert sample_arr.shape[2] == 1, "Expected a single channel in the output (3D case)."


def test_combined_age_score_spline(
    output_score_path,
    image_dir,
    combined_score_path
):
    profiler = cProfile.Profile()
    profiler.enable()
    combined_age_score_spline(
        output_score_path,
        str(combined_score_path),
        image_dir,
        multiply_factor=10_000,
        n_bins=10_000
    )
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    logging.info("Finished.")


def test_visualization(tmp_path, output_score_path, image_dir, combined_score_path):
    """
    Generates before and after histograms for a sample zarr file.
    The generated plot is saved to a temporary file for inspection.
    This helps visualize that raw scores are transformed into a normalized distribution.
    """
    # Run the entire pipeline.
    combined_age_score_spline(
        score_dir=output_score_path,
        combined_score_dir=str(combined_score_path),
        image_dir=image_dir,
        thresh_method="GaussianBlur",
        thresh_param=200,
        multiply_factor=1000,
        n_bins=1000
    )
    
    # Choose the first raw zarr file and its corresponding normalized file.
    raw_files = sorted([f for f in os.listdir(output_score_path) if f.endswith(".zarr")])
    norm_files = sorted(list(combined_score_path.glob("*_age_scores.zarr")))
    assert len(raw_files) > 0 and len(norm_files) > 0, "Missing raw or normalized files."
    
    raw_arr = zarr.open(os.path.join(output_score_path, raw_files[0]))[:]
    norm_arr = zarr.open(str(norm_files[0]))[:]
    
    # Flatten arrays to compute histograms.
    raw_flat = raw_arr.flatten()
    norm_flat = norm_arr.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(raw_flat, bins=50)
    axes[0].set_title("Raw Score Distribution")
    axes[1].hist(norm_flat, bins=50)
    axes[1].set_title("Normalized Score Distribution")
    
    # Save the figure to a temporary file.
    fig_file = tmp_path / "distribution_comparison.png"
    plt.savefig(str(fig_file))
    plt.close(fig)

    # Assert that the normalized distribution has mean ~0 and std ~1.
    mean_norm = np.mean(norm_flat)
    std_norm = np.std(norm_flat)
    assert abs(mean_norm) < 0.5, f"Normalized mean {mean_norm} is off."
    assert abs(std_norm - 1) < 0.5, f"Normalized std {std_norm} is off."


# Two Histogram Algorithm Tests
# ---------------------------------------------------------------------------------------
# TODO: I think there is a way to do this with just two histograms and no spline.