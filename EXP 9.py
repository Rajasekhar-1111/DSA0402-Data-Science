import pandas as pd

data = {
    "Property_ID": [101, 102, 103, 104, 105],
    "Location": ["Chennai", "Bangalore", "Chennai",
                 "Bangalore", "Chennai"],
    "Bedrooms": [3, 5, 4, 6, 5],
    "Area": [1200, 1800, 1500, 2200, 2000],
    "Price": [5000000, 8000000, 6500000,
              9500000, 8500000]
}

property_data = pd.DataFrame(data)

# 1. Average price by location
avg_price = property_data.groupby("Location")["Price"].mean()
print("Average Price by Location:")
print(avg_price)

# 2. Properties with more than 4 bedrooms
count = len(property_data[property_data["Bedrooms"] > 4])
print("\nProperties with more than 4 bedrooms:", count)

# 3. Property with largest area
largest = property_data.loc[property_data["Area"].idxmax()]
print("\nProperty with Largest Area:")
print(largest)
