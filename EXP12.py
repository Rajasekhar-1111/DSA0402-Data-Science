import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

temperature = [24, 26, 29, 32, 35, 34,
               31, 30, 29, 28, 26, 24]

rainfall = [20, 15, 10, 25, 40, 80,
            120, 140, 100, 160, 180, 90]

# Line Plot for Temperature
plt.plot(months, temperature, marker="o")
plt.title("Monthly Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.show()

# Scatter Plot for Rainfall
plt.scatter(months, rainfall)
plt.title("Monthly Rainfall")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.show()
