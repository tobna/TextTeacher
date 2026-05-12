import argparse
import os
import pathlib

import zipfile
import numpy as np
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", type=str, required=True, help="Folder of encodings to get the normalizer")
args = parser.parse_args()

args.folder = pathlib.Path(args.folder)

_MODE = "folder"
if os.path.isfile(os.path.join(args.folder, "all_encodings.zip")):
    _MODE = "zip"
    zf = zipfile.ZipFile(os.path.join(args.folder, "all_encodings.zip"), "r")
    encoding_files = zf.namelist()
else:
    encoding_files = [f for f in tqdm(args.folder.glob("*.emb.npy"), desc="getting encoding files")]


def load_encoding(filename):
    if _MODE == "zip":
        return np.load(zf.open(filename))
    return np.load(os.path.join(args.folder, filename))


print(len(encoding_files))
enc_0 = load_encoding(encoding_files[0])
print(enc_0.shape)

encoding_matrix = np.zeros((len(encoding_files), enc_0.shape[0]), dtype=enc_0.dtype)
for i, enc in enumerate(tqdm(encoding_files, desc="gathering encodings")):
    encoding_matrix[i] = load_encoding(enc)

mean = np.mean(encoding_matrix, axis=0)
std = np.std(encoding_matrix, axis=0)
print(f"mean={mean}, std={std}")
stats = np.stack((mean, std))
np.save(os.path.join(args.folder, "stats.npy"), stats)
