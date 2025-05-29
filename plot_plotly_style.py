import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def plot_plotly_style_24hour(meter_id="AES2020896472402", target_date="2023-05-10"):
    """Plot 24-hour consumption with Plotly-style theme"""

    print(f"Loading 24-hour data for meter {meter_id} on {target_date}...")

    # Load the combined data
    df = pd.read_csv("combined_load_profile_electrical.csv")
    df["Meter Datetime"] = pd.to_datetime(df["Meter Datetime"])
    df["Hour"] = df["Meter Datetime"].dt.hour
    df["Date"] = df["Meter Datetime"].dt.date

    # Convert target date to date object
    target_date_obj = pd.to_datetime(target_date).date()

    # Filter for specific meter and date
    meter_data = df[df["HES Meter Id"] == meter_id]
    day_data = meter_data[meter_data["Date"] == target_date_obj].copy()

    if len(day_data) == 0:
        print(f"No data found for meter {meter_id} on {target_date}")
        return None

    # Sort by time
    day_data = day_data.sort_values("Meter Datetime")

    print(f"Found {len(day_data)} records for {target_date}")

    # Create figure with Plotly-style settings
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set Plotly-like colors and styling
    plotly_bg = "#e5ecf6"
    plotly_grid = "#E5E5E5"
    plotly_text = "#2A3F5F"
    plotly_line = "#636EFA"  # Plotly's default blue

    # Set figure and axes background
    fig.patch.set_facecolor(plotly_bg)
    ax.set_facecolor(plotly_bg)

    # Create time axis starting from midnight
    start_time = pd.to_datetime(f"{target_date} 00:00:00")
    end_time = start_time + timedelta(days=1)

    # Plot the main consumption line with Plotly-style
    ax.plot(
        day_data["Meter Datetime"],
        day_data["Import active power (QI+QIV)[W]"],
        linewidth=2.5,
        alpha=0.95,
        color=plotly_line,
        marker="o",
        markersize=3,
        markerfacecolor=plotly_line,
        markeredgewidth=0,
        markevery=2,
    )  # Add subtle markers

    # Set title and labels with Plotly-style typography
    ax.set_title(
        f"24-Hour Power Consumption - Meter {meter_id}\\n{target_date}",
        fontsize=18,
        fontweight="600",
        pad=25,
        color=plotly_text,
        fontfamily="sans-serif",
    )
    ax.set_xlabel("Time of Day", fontsize=14, color=plotly_text, fontweight="500")
    ax.set_ylabel(
        "Power Consumption (W)", fontsize=14, color=plotly_text, fontweight="500"
    )

    # Format x-axis to show hours from 0 to 24
    ax.set_xlim(start_time, end_time)

    # Set major ticks every 2 hours
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Set minor ticks every hour for finer grid
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    # Plotly-style grid
    ax.grid(
        True, which="major", alpha=0.6, linestyle="-", linewidth=0.8, color=plotly_grid
    )
    ax.grid(
        True, which="minor", alpha=0.3, linestyle="-", linewidth=0.4, color=plotly_grid
    )

    # Remove all spines (Plotly doesn't show axis lines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Style the ticks (Plotly-style)
    ax.tick_params(colors=plotly_text, which="both", labelsize=12)
    ax.tick_params(axis="both", which="major", length=0)  # Remove tick marks
    ax.tick_params(axis="both", which="minor", length=0)

    # Add subtle border around the plot area
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=plotly_grid,
            linewidth=1,
        )
    )

    # Set y-axis to start from 0 for better visual representation
    y_min = 0
    y_max = day_data["Import active power (QI+QIV)[W]"].max() * 1.1
    ax.set_ylim(y_min, y_max)

    # Format y-axis with comma separators for large numbers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Add hover-like effect with better spacing
    ax.margins(x=0.01, y=0.05)

    # Plotly-style font settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

    # Adjust layout
    plt.tight_layout()

    # Save the plot with high quality
    filename = f"plotly_style_24hour_{meter_id}_{target_date.replace('-', '_')}.png"
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=plotly_bg,
        edgecolor="none",
        pad_inches=0.2,
    )
    plt.close()

    print(f"Plotly-style 24-hour consumption plot saved as: {filename}")

    # Print basic summary
    consumption_values = day_data["Import active power (QI+QIV)[W]"]
    avg_consumption = consumption_values.mean()
    max_consumption = consumption_values.max()
    min_consumption = consumption_values.min()

    print(f"\\nConsumption Summary for {target_date}:")
    print(f"Average: {avg_consumption:,.0f} W")
    print(f"Maximum: {max_consumption:,.0f} W")
    print(f"Minimum: {min_consumption:,.0f} W")

    return day_data


if __name__ == "__main__":
    # Generate the Plotly-style 24-hour consumption plot
    plot_plotly_style_24hour("AES2020896472402", "2023-05-10")
