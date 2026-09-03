prices = [100, 200, 150]
quantity = [2, 1, 3]

discount = 10      # 10%
tax = 5            # 5%

total = 0

for i in range(len(prices)):
    total += prices[i] * quantity[i]

discounted = total - (total * discount / 100)
final_cost = discounted + (discounted * tax / 100)

print("Total Cost:", final_cost)
