import os
import sys

assert len(sys.argv) >= 2, f"Add a sbatch script as the first argument"
assert os.path.isfile(
    sys.argv[1]
), f"First argument has to be an executable script (a file that exists), but got '{sys.argv[1]}'"

with open(sys.argv[1], "r") as f:
    script = f.readlines()

script = [l.strip() for l in script if len(l.strip()) > 0]

# join lines ending with \
joined_script = []
current_line = ""
for line in script:
    current_line += line
    if current_line.endswith("\\"):
        current_line = current_line[:-1]
    else:
        joined_script.append(current_line)
        current_line = ""
script = joined_script

additional_srun_params = []
srun_lines = []
for line in script:
    if line.upper().startswith("#SBATCH "):
        param = line[len("#SBATCH ") :]
        if param.startswith("--output="):
            continue
        if param.startswith("--export="):
            param = param.replace("TQDM_DISABLE=1", "").replace(",,", ",")
        additional_srun_params.append(param)
    elif line.startswith("srun "):
        srun_lines.append(line)

further_args = " ".join(sys.argv[2:])

sruns = [
    line.replace("srun ", "srun " + " ".join(additional_srun_params) + " ").replace('"$@"', further_args)
    for line in srun_lines
]

for srun_line in sruns:
    print(f"I will run:\n{srun_line}", flush=True)
    os.system(srun_line)
