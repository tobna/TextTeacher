from matplotlib import pyplot as plt

from load_dataset import prepare_dataset
from utils import prep_kwargs

ds_args = prep_kwargs(
    {
        "imsize": 512,
        "num_workers": 4,
        "shuffle": False,
        "augment_strategy": "minimal",
        "batch_size": 1,
        "aug_normalize": False,
        "augment_engine": "torchvision",
    }
)

_TRAIN = True
base_ds = prepare_dataset("tinyimagenet", ds_args, train=_TRAIN)[0].dataset

recomb_ds = prepare_dataset("tinyrecombnet-7-1/same/0.8/range", ds_args, train=_TRAIN)[0].dataset
print(f"Recombnet keys: {recomb_ds.foregrounds[:5]}")
name_header = "/".join(recomb_ds.foregrounds[0].split("/")[:-2])

# with open("data/misc_dataset_files/imagenet_labels.txt", "r") as f:
# labels = f.read().split("\n")
# labels = {lbl.split(" ")[0]: lbl.split(" ")[-1] for lbl in labels if len(lbl) > 2}
with open("data/misc_dataset_files/tinyimagenet_synset_names.txt", "r") as f:
    labels = f.read().split("\n")
labels = {lbl.split(": ")[0]: lbl.split(": ")[-1] for lbl in labels if len(lbl) > 2}
print("labels:", list(labels.keys())[:5])
lbl_to_cls_offset = sorted(list(labels.keys()), key=lambda x: int(x[1:]))

# search_class = "n07579787"

# save variants of image with index
save_idx = 1
save_data = base_ds[save_idx]
save_key = save_data["key"]
print(f"Saving image with key {save_key}")


for data in base_ds:
    key = data["key"]
    cls_offset = lbl_to_cls_offset[data["label"]]
    # if cls_offset != search_class:
    #     continue
    fg_file_name = f"{name_header}/{cls_offset}/{key[:-5]}.WEBP"
    # if fg_file_name not in recomb_ds.foregrounds:
    #     print(f"Skipping {key}, {fg_file_name}")
    #     continue
    recomb_ds_index = recomb_ds.foregrounds.index(fg_file_name)
    print(f"Base image: {key}, FG name: {fg_file_name}, Recomb index: {recomb_ds_index}")
    class_name = labels[cls_offset]

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0, 0].imshow(data["image"].permute(1, 2, 0))
    axs[0, 1].imshow(recomb_ds[recomb_ds_index][0].permute(1, 2, 0))
    axs[1, 0].imshow(recomb_ds[recomb_ds_index][0].permute(1, 2, 0))
    axs[1, 1].imshow(recomb_ds[recomb_ds_index][0].permute(1, 2, 0))
    axs[0, 0].axis("off")
    axs[0, 1].axis("off")
    axs[1, 0].axis("off")
    axs[1, 1].axis("off")
    fig.suptitle(f"Class: {class_name}")
    plt.tight_layout()

    plt.show()
