from manim import *
import pandas as pd
import numpy as np
from datetime import datetime


class MeterConsumptionAnimation(Scene):
    def construct(self):
        # Load and process the data
        df = pd.read_csv("cleaned_meter_KFM2020660190982.csv")
        df["Meter Datetime"] = pd.to_datetime(df["Meter Datetime"])
        df = df.sort_values("Meter Datetime")

        # Sample every 2nd point to get hourly data (from 30min intervals)
        df_hourly = df.iloc[::2].reset_index(drop=True)
        
        # Reduced to 4 weeks for faster rendering
        week_starts = [
            0,      # Week 1: Spring
            800,    # Week 2: Summer  
            1600,   # Week 3: Fall
            2400,   # Week 4: Winter
        ]
        
        # Season names for display
        season_names = [
            "Spring",
            "Summer", 
            "Fall",
            "Winter"
        ]
        
        # Use same color for all seasons
        season_colors = [BLUE] * 4
        
        # Get weeks data for each season
        all_weeks_data = []
        valid_weeks = []
        valid_seasons = []
        valid_colors = []
        
        for i, start_idx in enumerate(week_starts):
            week_end = start_idx + 168  # 7 days * 24 hours
            if week_end <= len(df_hourly):
                week_data = df_hourly.iloc[start_idx:week_end]
                if len(week_data) == 168:  # Ensure complete week
                    all_weeks_data.append(week_data)
                    valid_weeks.append(i + 1)
                    valid_seasons.append(season_names[i])
                    valid_colors.append(season_colors[i])

        # Get overall max for consistent y-axis across all weeks
        all_power_values = []
        for week_data in all_weeks_data:
            all_power_values.extend(week_data["Import active power (QI+QIV)[W]"].values)
        max_power = np.max(all_power_values)

        # Add subtle background for professional look
        background = Rectangle(
            width=config.frame_width, 
            height=config.frame_height,
            fill_color=BLACK,
            fill_opacity=0.9,
            stroke_width=0
        )
        self.add(background)

        # Create axes for 24-hour display (0-23 hours)
        axes = Axes(
            x_range=[0, 23, 1],  # 0 to 23 hours
            y_range=[0, max_power * 1.1, max(1, max_power // 5)],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
        )

        # Add axis labels
        x_label = Text("Hour of Day", font_size=24, color=WHITE)
        y_label = Text("Power", font_size=24, color=WHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.5)
        y_label.next_to(axes.y_axis, LEFT, buff=0.5).rotate(PI/2)

        # Position axes
        axes.shift(DOWN * 0.5)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Add hour labels for key times
        hour_labels = VGroup()
        
        # Label for hour 0 (midnight)
        hour_0_label = Text("0", font_size=16, color=WHITE)
        hour_0_label.next_to(axes.c2p(0, 0), DOWN, buff=0.2)
        hour_labels.add(hour_0_label)
        
        # Label for hour 12 (noon)
        hour_12_label = Text("12", font_size=16, color=WHITE)
        hour_12_label.next_to(axes.c2p(12, 0), DOWN, buff=0.2)
        hour_labels.add(hour_12_label)
        
        # Label for hour 23 (11 PM, close to midnight)
        hour_24_label = Text("24", font_size=16, color=WHITE)
        hour_24_label.next_to(axes.c2p(23, 0), DOWN, buff=0.2)
        hour_labels.add(hour_24_label)

        self.play(Write(hour_labels))

        # Store all daily graphs across all weeks
        all_daily_graphs = []
        
        # Pre-create all daily graphs for all weeks
        for week_idx, week_data in enumerate(all_weeks_data):
            # Group week data by day (24 hours per day)
            for day in range(7):
                start_idx = day * 24
                end_idx = (day + 1) * 24
                day_data = week_data.iloc[start_idx:end_idx]
                if len(day_data) == 24:
                    power_values = day_data["Import active power (QI+QIV)[W]"].values
                    
                    # Create line segments for this day (to be animated hour by hour)
                    day_segments = []
                    for hour in range(23):  # 0 to 22 (connecting to hour 23)
                        line_segment = Line(
                            axes.c2p(hour, power_values[hour]),
                            axes.c2p(hour + 1, power_values[hour + 1]),
                            color=BLUE,
                            stroke_width=3
                        )
                        line_segment.set_opacity(0)  # Start invisible
                        day_segments.append(line_segment)
                    
                    all_daily_graphs.append(day_segments)

        # Add all segments to scene
        for day_segments in all_daily_graphs:
            for segment in day_segments:
                self.add(segment)

        # Much faster animation - animate entire days at once with fewer calls
        base_speed = 0.05  # Faster base speed
        
        for day_idx, day_segments in enumerate(all_daily_graphs):
            # Calculate exponentially faster speed
            speed_multiplier = 2.0 ** day_idx  # Faster exponential growth
            current_speed = max(0.005, base_speed / speed_multiplier)  # Minimum speed limit
            
            # Dim all previous days first (only once per day)
            if day_idx > 0:
                prev_animations = []
                for prev_day_idx in range(day_idx):
                    for prev_segment in all_daily_graphs[prev_day_idx]:
                        prev_segment.set_opacity(0.2)  # Direct set instead of animate
            
            # Animate current day segments in batches for speed
            batch_size = max(1, 23 // max(1, int(day_idx / 3)))  # Larger batches for later days
            
            for batch_start in range(0, 23, batch_size):
                batch_end = min(batch_start + batch_size, 23)
                batch_animations = []
                
                for hour in range(batch_start, batch_end):
                    batch_animations.append(day_segments[hour].animate.set_opacity(1))
                
                if batch_animations:
                    self.play(*batch_animations, run_time=current_speed)

        # Final pause
        self.wait(0.3)


if __name__ == "__main__":
    scene = MeterConsumptionAnimation()
    scene.render()
