import json

import numpy as np
from sklearn.linear_model import Perceptron

with open("./dataset.json") as file:
    dataset = json.load(file)

clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(
    np.array([row["data"] for row in dataset]),
    np.array([row["result"] for row in dataset])
)

weights = clf.coef_[0]
bias = clf.intercept_[0]

SCALE_FACTOR = 10 
w_int = np.round(weights*SCALE_FACTOR).astype(int)
b_int = int(np.round(bias*SCALE_FACTOR))

def to_hex(val, bits=16):
    val = int(val)
    if val < 0: val = (1 << bits) + val
    return f"{val:04x}"

print("Model created")
print(f"Weights: {w_int}")
print(f"Bias: {b_int}")
print(f"Bias (hex): {to_hex(b_int)}")

with open("./weights.txt", "w") as file:
    file.write(" ".join([to_hex(w) for w in w_int]))
