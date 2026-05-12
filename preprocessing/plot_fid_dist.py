import argparse
from pathlib import Path
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
args = parser.parse_args()

args.file = Path(args.file)
with open(args.file, "r") as f:
    lines = f.readlines()

seed_fid = []
for line in lines:
    try:
        seed, fid = line.split("\t")
        seed = int(seed.split(": ")[1])
        fid = float(fid.split(": ")[1])
        seed_fid.append((seed, fid))

    except:
        pass

print(seed_fid)
seed_fid = sorted(seed_fid, key=lambda x: x[1])
print(f"best: {seed_fid[:5]}")
print(f"worst: {seed_fid[-5:]}")
plt.hist([fid for _, fid in seed_fid])
plt.show()
