import os
import re

outstring = "| Architecture | Versions |\n|:---|:---|\n"

model_re = re.compile("def (.*)\(.*")

file_list = sorted(list(os.listdir("architectures")))

for file in file_list:
    if not file.endswith(".py"):
        continue
    print(file)

    outstring += f"| {file[:-3]}| "

    is_model = False
    n_matches = 0
    with open(f"architectures/{file}", "r") as f:
        for line in f:
            line = line.strip()
            if is_model:
                match = model_re.match(line)
                if match is not None:
                    outstring += f"`{match.group(1)}`, "
                    n_matches += 1
                else:
                    print(f"\t\033[93mCould not match line:\033[0m\n\t\t{line}")
            is_model = False
            if "@register_model" in line:
                is_model = True
    if n_matches >= 1:
        outstring = outstring[:-2]
    outstring += "|\n"
    print(f"\tfound {n_matches} models")

with open("model_list.md", "w+") as f:
    f.write(outstring + "\n")
