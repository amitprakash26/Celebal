import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file
data = pd.read_csv('D:\celebal\company_sales_data.csv')

# Calculate total profit for each month
total_profit = data.groupby('month_number')['total_profit'].sum()

# Create the figure and the axes
fig, ax = plt.subplots()

# Plot the data with the desired properties
ax.plot(total_profit.index, total_profit.values, color='red', linestyle=':', linewidth=3, marker='o', markerfacecolor='red')

# Set the title and labels
ax.set_title('Total Profit by Month')
ax.set_xlabel('Month Number')
ax.set_ylabel('Sold Units Number')

# Show legend at the lower right location
ax.legend(['Total Profit'], loc='lower right')

# Display the plot
plt.show()
