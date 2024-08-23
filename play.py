import json

with open("Spaghetti_0.json") as jf:
    data = json.load(jf)
    
print(len(data["value"]))