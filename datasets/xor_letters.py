import pandas as pd
import itertools

n = 8
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
options = ["True", "False"]


def get_all_subsets(letters):
    subsets = []
    for i in range(len(letters) + 1):
        subsets_len_i = itertools.combinations(letters, i)
        for subset in subsets_len_i:
            if len(subset) > 0:
                subsets.append("".join(subset))
    return subsets


def get_label_value(label, combo):
    combo_TF = {l: c == "True" for l, c in zip(letters, combo)}
    return sum(combo_TF[l] for l in list(label)) % 2 == 1


# Generate all possible combinations
combinations = list(itertools.product(options, repeat=n))
all_labels = get_all_subsets(letters)

# Create rows for the DataFrame
rows = []
label_values = {k: [] for k in all_labels}

for combo in combinations:
    row = ", ".join([f"{letter} is {value}" for letter, value in zip(letters, combo)])
    for label in all_labels:
        label_values[label].append(get_label_value(label, combo))
    rows.append(row + ".")

for combo in combinations:
    row = ", ".join([f"{letter}: {value}" for letter, value in zip(letters, combo)])
    for label in all_labels:
        label_values[label].append(get_label_value(label, combo))
    rows.append(row + ".")

for combo in combinations:
    row = ". ".join(
        [f"The class of {letter} is {value}" for letter, value in zip(letters, combo)]
    )
    for label in all_labels:
        label_values[label].append(get_label_value(label, combo))
    rows.append(row + ".")

for combo in combinations:
    row = ", ".join([f"{letter} - {value}" for letter, value in zip(letters, combo)])
    for label in all_labels:
        label_values[label].append(get_label_value(label, combo))
    rows.append(row + ".")

for combo in combinations:
    row = "| ".join([f"{letter} , {value}" for letter, value in zip(letters, combo)])
    for label in all_labels:
        label_values[label].append(get_label_value(label, combo))
    rows.append(row + ".")

for combo in combinations:
    row = ", ".join([f"{letter} -> {value}" for letter, value in zip(letters, combo)])
    for label in all_labels:
        label_values[label].append(get_label_value(label, combo))
    rows.append(row + ".")


# Create the DataFrame
d = {"statement": rows}
d.update(label_values)
df = pd.DataFrame(d)
df.to_csv("datasets/xor_letters.csv", index=False)
