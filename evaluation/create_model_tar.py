import os
import sys

sys.path.append(".")
from load_dataset import prepare_dataset
from main import parse_args
from models import prepare_model
from utils import prep_kwargs, save_model_state

if __name__ == "__main__":
    args = prep_kwargs(parse_args())
    if args.run_name is None:
        raise ValueError("Please specify 'run_name'.")
    if args.experiment_name is None:
        raise ValueError("Please specify 'experiment_name'.")
    if args.dataset is not None:
        _, num_classes, __, ___, ____ = prepare_dataset(args.dataset, args, train=False)
    else:
        num_classes = 1000
    args.n_classes = num_classes
    model = prepare_model(args.model, args)
    print("model:")
    print(model)

    model_folder = (
        f"{args.results_folder}/models/for_eval_{args.run_name.replace('/', '_')}_{args.model.replace('/', '_')}/"
    )

    os.makedirs(model_folder, exist_ok=True)

    print(f"saving 'initial.tar' to '{model_folder}'")
    save_model_state(
        model_folder,
        0,
        args,
        model.state_dict(),
        regular_save=False,
        additional_reason="initial",
    )
    print("done.")
