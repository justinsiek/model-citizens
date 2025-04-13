from model import predict_preference

result = predict_preference("Do you like pinapple on pizza?", "DO you like pinapple on pizza? DO you like pinapple on pizza?", "yes I love pizza with pinapple yes I love pizza with pinapple")
print(result)
sum = 0
for x in result:
  if x['prefer_a_probability'] > 0.5:
    sum += 1

print(sum / len(result))
