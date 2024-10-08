import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from datetime import datetime, timedelta

# Create a dictionary with your schedule
schedule = {
    'Task': [
        'Auditing', 'Data Analytics: Statistical Programming', 'Corporate Finance', 
        'Strategic Management', 'Auditing Exercise', 'Corporate Finance Exercise',
        'Strategic Management Exercise', 'CS lectures 1', 'Data Handling', 'Data Handling Exercise',
        'Methods in Social Studies', 'Machine Learning in Finance', 'CS lectures 2',
        'CS Exercise', 'Capstone'
    ],
    'Start': [
        datetime(2024, 9, 16, 10, 0), datetime(2024, 11, 11, 8, 0), 
        datetime(2024, 11, 11, 8, 0), datetime(2024, 11, 11, 8, 0), 
        datetime(2024, 9, 17, 10, 0), datetime(2024, 11, 19, 12, 0), 
        datetime(2024, 11, 12, 20, 0), datetime(2024, 9, 18, 12, 0),
        datetime(2024, 9, 19, 10, 0), datetime(2024, 9, 26, 16, 0),
        datetime(2024, 11, 15, 8, 0), datetime(2024, 9, 20, 8, 0), 
        datetime(2024, 9, 20, 8, 0), datetime(2024, 9, 20, 10, 0),
        datetime(2024, 9, 20, 17, 0)
    ],
    'End': [
        datetime(2024, 10, 21, 12, 0), datetime(2024, 12, 16, 12, 0), 
        datetime(2024, 12, 16, 12, 0), datetime(2024, 12, 16, 12, 0),
        datetime(2024, 10, 22, 14, 0), datetime(2024, 12, 17, 14, 0),
        datetime(2024, 12, 17, 22, 0), datetime(2024, 12, 18, 16, 0),
        datetime(2024, 12, 19, 12, 0), datetime(2024, 12, 19, 18, 0),
        datetime(2024, 12, 20, 12, 0), datetime(2024, 10, 25, 14, 0),
        datetime(2024, 12, 20, 10, 0), datetime(2024, 12, 20, 12, 0),
        datetime(2024, 12, 20, 21, 0)
    ],
    'Day': [
        'Monday', 'Monday', 'Monday', 'Monday', 'Tuesday', 'Tuesday',
        'Tuesday', 'Wednesday', 'Thursday', 'Thursday',
        'Friday', 'Friday', 'Friday', 'Friday', 'Friday'
    ]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(schedule)

# Add a column for the duration of the task
df['Duration'] = df['End'] - df['Start']

# Convert the start date to numeric format for plotting
df['Start_num'] = df['Start'].apply(date2num)
df['End_num'] = df['End'].apply(date2num)

# Define colors for each day of the week
colors = {
    'Monday': 'lightblue',
    'Tuesday': 'lightgreen',
    'Wednesday': 'lightcoral',
    'Thursday': 'lightgoldenrodyellow',
    'Friday': 'lightsalmon'
}

# Plot the calendar
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each task
for i, row in df.iterrows():
    bar = ax.barh(row['Task'], row['End_num'] - row['Start_num'], left=row['Start_num'], height=0.5, color=colors[row['Day']])
    
    # Add start and end times as annotations
    start_time_str = row['Start'].strftime('%d-%m-%Y %H:%M')
    end_time_str = row['End'].strftime('%d-%m-%Y %H:%M')
    label = f"{start_time_str} - {end_time_str}"
    ax.text(row['Start_num'] + (row['End_num'] - row['Start_num']) / 2, i, label, 
            ha='center', va='center', color='black', fontsize=8)

# Format the plot
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.xlabel('Date')
plt.ylabel('Task')
plt.title('Weekly Schedule')

# Remove duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
