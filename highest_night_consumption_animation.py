from manim import *
import pandas as pd
import numpy as np

class HighestNightConsumptionAnimation(Scene):
    def construct(self):
        # Load and process the data
        df = pd.read_csv("cleaned_meter_KFM2020660190982.csv")
        df["Meter Datetime"] = pd.to_datetime(df["Meter Datetime"])
        df = df.sort_values("Meter Datetime")
        
        # Add hour and date columns
        df['hour'] = df['Meter Datetime'].dt.hour
        df['date'] = df['Meter Datetime'].dt.date
        
        # Get data for June 20, 2022 (highest night consumption day)
        target_date = pd.to_datetime('2022-06-20').date()
        day_data = df[df['date'] == target_date].copy()
        
        # Convert to hourly data (taking one reading per hour)
        hourly_data = day_data.groupby('hour')['Import active power (QI+QIV)[W]'].first().reset_index()
        
        # Define night hours (9 PM to 4 AM)
        night_hours = [21, 22, 23, 0, 1, 2, 3]
        
        # Create title
        title = Text("Highest Night Consumption Day", font_size=36, color=BLUE)
        subtitle = Text("June 20, 2022 - Total Night: 2976W", font_size=24, color=GRAY)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.add(title, subtitle)
        
        # Create axes
        max_power = hourly_data['Import active power (QI+QIV)[W]'].max()
        axes = Axes(
            x_range=[0, 23, 1],
            y_range=[0, max_power * 1.1, max(1, max_power // 5)],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
        )
        
        # Add axis labels
        x_label = Text("Hour of Day", font_size=24, color=WHITE)
        y_label = Text("Power (W)", font_size=24, color=WHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.5)
        y_label.next_to(axes.y_axis, LEFT, buff=0.5).rotate(PI/2)
        
        # Position axes
        axes.shift(DOWN * 0.8)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Add hour labels
        hour_labels = VGroup()
        for hour in [0, 6, 12, 18, 23]:
            label = Text(str(hour), font_size=16, color=WHITE)
            label.next_to(axes.c2p(hour, 0), DOWN, buff=0.2)
            hour_labels.add(label)
        self.play(Write(hour_labels))
        
        # Create legend
        legend = VGroup()
        normal_line = Line(ORIGIN, RIGHT * 0.5, color=BLUE, stroke_width=4)
        normal_text = Text("Normal Hours", font_size=18, color=WHITE)
        night_line = Line(ORIGIN, RIGHT * 0.5, color=RED, stroke_width=4)
        night_text = Text("Night Hours (9PM-4AM)", font_size=18, color=WHITE)
        
        normal_legend = VGroup(normal_line, normal_text.next_to(normal_line, RIGHT, buff=0.2))
        night_legend = VGroup(night_line, night_text.next_to(night_line, RIGHT, buff=0.2))
        legend.add(normal_legend, night_legend)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend.to_corner(UR, buff=0.5)
        self.play(Write(legend))
        
        # Current hour display
        current_hour_text = Text("Hour: 0", font_size=24, color=YELLOW)
        current_power_text = Text("Power: 0W", font_size=24, color=YELLOW)
        
        current_hour_text.to_corner(UL, buff=0.5)
        current_power_text.next_to(current_hour_text, DOWN, buff=0.2)
        self.add(current_hour_text, current_power_text)
        
        # Create all line segments
        all_segments = []
        all_dots = []
        
        for i in range(len(hourly_data) - 1):
            hour = hourly_data.iloc[i]['hour']
            next_hour = hourly_data.iloc[i + 1]['hour']
            power = hourly_data.iloc[i]['Import active power (QI+QIV)[W]']
            next_power = hourly_data.iloc[i + 1]['Import active power (QI+QIV)[W]']
            
            # Determine color based on night hours
            color = RED if hour in night_hours else BLUE
            
            # Create line segment
            segment = Line(
                axes.c2p(hour, power),
                axes.c2p(next_hour, next_power),
                color=color,
                stroke_width=4
            )
            all_segments.append(segment)
            
            # Create dot
            dot = Dot(axes.c2p(hour, power), color=color, radius=0.06)
            all_dots.append(dot)
        
        # Add final dot
        final_hour = hourly_data.iloc[-1]['hour']
        final_power = hourly_data.iloc[-1]['Import active power (QI+QIV)[W]']
        final_color = RED if final_hour in night_hours else BLUE
        final_dot = Dot(axes.c2p(final_hour, final_power), color=final_color, radius=0.06)
        all_dots.append(final_dot)
        
        # Animate through each hour
        for i, (segment, dot) in enumerate(zip(all_segments, all_dots)):
            hour = hourly_data.iloc[i]['hour']
            power = hourly_data.iloc[i]['Import active power (QI+QIV)[W]']
            
            # Update text displays
            new_hour_text = Text(f"Hour: {hour}", font_size=24, color=YELLOW)
            new_power_text = Text(f"Power: {power:.0f}W", font_size=24, color=YELLOW)
            new_hour_text.move_to(current_hour_text.get_center())
            new_power_text.move_to(current_power_text.get_center())
            
            # Show if it's a night hour
            night_indicator = Text("NIGHT HOUR", font_size=20, color=RED) if hour in night_hours else Text("", font_size=20)
            night_indicator.next_to(new_power_text, DOWN, buff=0.2)
            
            self.play(
                Create(dot),
                Create(segment),
                Transform(current_hour_text, new_hour_text),
                Transform(current_power_text, new_power_text),
                Write(night_indicator) if hour in night_hours else Wait(0),
                run_time=0.3
            )
            
            if hour in night_hours:
                self.wait(0.2)  # Pause longer on night hours
                self.play(FadeOut(night_indicator), run_time=0.2)
        
        # Add final dot
        final_hour = hourly_data.iloc[-1]['hour']
        final_power = hourly_data.iloc[-1]['Import active power (QI+QIV)[W]']
        new_hour_text = Text(f"Hour: {final_hour}", font_size=24, color=YELLOW)
        new_power_text = Text(f"Power: {final_power:.0f}W", font_size=24, color=YELLOW)
        new_hour_text.move_to(current_hour_text.get_center())
        new_power_text.move_to(current_power_text.get_center())
        
        self.play(
            Create(all_dots[-1]),
            Transform(current_hour_text, new_hour_text),
            Transform(current_power_text, new_power_text),
            run_time=0.3
        )
        
        # Highlight night hours with emphasis
        night_emphasis = []
        for i, dot in enumerate(all_dots):
            if i < len(hourly_data):
                hour = hourly_data.iloc[i]['hour']
                if hour in night_hours:
                    emphasis = Circle(radius=0.15, color=RED, stroke_width=3)
                    emphasis.move_to(dot.get_center())
                    night_emphasis.append(emphasis)
        
        self.play(*[Create(emphasis) for emphasis in night_emphasis], run_time=1)
        
        # Show statistics
        night_data = hourly_data[hourly_data['hour'].isin(night_hours)]
        stats = VGroup(
            Text(f"Night Hours Total: {night_data['Import active power (QI+QIV)[W]'].sum():.0f}W", font_size=20, color=RED),
            Text(f"Night Hours Avg: {night_data['Import active power (QI+QIV)[W]'].mean():.0f}W", font_size=20, color=RED),
            Text(f"Peak Night: {night_data['Import active power (QI+QIV)[W]'].max():.0f}W", font_size=20, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT)
        stats.to_corner(DL, buff=0.5)
        
        self.play(Write(stats))
        
        # Final pause
        self.wait(3)
        
        # Fade out emphasis
        self.play(*[FadeOut(emphasis) for emphasis in night_emphasis])
        self.wait(2)

if __name__ == "__main__":
    scene = HighestNightConsumptionAnimation()
    scene.render()