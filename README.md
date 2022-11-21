# Optional Project: Distillation and Compilation

----
## Table Of Contents
- [Description](#description)
- [Commands](#commands)
- [Team](#team)
----
## Description
Main scripts lies under src directory.
File  | Usage
------------- | -------------
MNIST.py | The original model
prune_MNIST2.py | The file prunes the MNIST model and performs distillation
preprocess.py | Changes the image to MNIST compatible input
postprocess.py | Interprets the prediction output 
----
## Commands
1. Run the prune_MNIST2.py file. This generates the pruned and distilled model as model_o.onnx
2. Compile the model using tvmc compile --target "llvm"  --output mnist.tar model_o.onnx 
3. Run the preprocess file
4. Run the model to get the output. tvmc run --inputs imagenet_num1.npz --output predictions.npz mnist.tar
5. Run the preprocess file to get the output

## Team
Name  | Unity id
------------- | -------------
Saksham Thakur  | sthakur5
Xinning Hui | xhui
---

