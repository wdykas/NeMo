import random

train_set = open("train_data.jsonl").readlines()
train_set_size = len(train_set)
val_set_size = int(train_set_size * 0.10)
val_set = random.sample(train_set, val_set_size)
new_train_set = list(set(train_set) - set(val_set))

new_train_file = open("data/train_data.jsonl", "w")
new_val_file = open("data/val_data.jsonl", "w")

for line in new_train_set:
    line = line.strip()
    new_train_file.write(line + "\n")

for line in val_set:
    line = line.strip()
    new_val_file.write(line + "\n")

print(f"Originial Train Set Size: {train_set_size}")
print(f"New Train Set Size: {len(new_train_set)}")
print(f"Val Set Size: {len(val_set)}")






