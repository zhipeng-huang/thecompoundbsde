# The Compound BSDE method
This library provides an implementation for the numerical methods considered in the paper:
>Z.Huang, C.W.Oosterlee (2026): The Compound BSDE method: A fully-forward method for option pricing and optimal stopping problems in finance


## Usage
Set up a .json configuration file in /configs/, setting the parameters of the equation/bsde/COS configuration/etc. `example.json`

Suppose you want to run the method on ex1.json file, then run the following command:

>`python main.py --config_file=configs/ex1.json --exp_name=ex1`

The results (the model and training history) will be saved under `/logs/ex1`

Then you can run the following to get the result of the model and the errors, but you need to specifiy the folder_path in get_results.py.

>`python get_results.py`


Analogously for all other main scripts.