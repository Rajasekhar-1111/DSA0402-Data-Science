import pandas as pd

data = {
    "Product": ["Laptop", "Mouse", "Keyboard", "Phone",
                "Tablet", "Laptop", "Mouse", "Phone"],
    "Quantity": [5, 10, 7, 15, 8, 6, 12, 10]
}

sales = pd.DataFrame(data)

total_sales = sales.groupby("Product")["Quantity"].sum()

top5 = total_sales.nlargest(5)

print("Top 5 Products:")
print(top5)
