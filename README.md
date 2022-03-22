# Package Dependencies

We used FrEIA v0.2 to build our project (https://github.com/VLL-HD/FrEIA.git)

Install via pip:

> pip install git+https://github.com/VLL-HD/FrEIA.git

Other package dependencies:

+ pytorch (torch, torchvision)
+ numpy
+ matplotlib

# ./Toy_project
Contains the code used for section 2 (Toy Example) in the project report.

Execute via

> python toy.py

# ./FashionMNIST
Contains the code used for section 3 (FashionMNIST) in the project report.

Install custom package:
> pip install FashionMNIST

The experiments described in the paper can be found under:
+ (3.1.1) Baseline `./FashionMNIST/FashionMNIST/experiments/fcn-only/` 
+ (3.1.2) Convolutional network with FCN conditioning `./FashionMNIST/FashionMNIST/experiments/only-fcn-conditioning/` 
+ (3.1.3) Conditioning on all coupling blocks `./FashionMNIST/FashionMNIST/`
+ (3.1.4) Removing skip connections `./FashionMNIST/FashionMNIST/experiments/no-skip-connections/` 
+ (3.1.5) SoftFlow `./FashionMNIST/FashionMNIST/experiments/softflow/` 

You can train each experimental model by moving to the corresponding directory and running `train.py`. To adjust the training parameters, you may also change the `config.json` file. During training you can observe intermediate generated samples each 10 epochs in the `{experiment}/train_output/` folder. The final model will be saved in `{experiment}/output/` folder. To perform evaluation, execute `eval.py` file after training the model and the generated samples will be stored to disk at `{experiment}/eval_output/`. 