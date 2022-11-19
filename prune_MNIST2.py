import torch
import torch.nn.functional as F
from torch.optim import SGD
import torch.nn as nn
from MNIST import TorchModel, trainer, evaluator, device, train_loader
from torch.optim.lr_scheduler import MultiStepLR
import os

# define the model
model = TorchModel().to(device)
model_list = nn.ModuleList([])

# show the model structure, note that pruner will wrap the model layer.
print(model)


# %%

# define the optimizer and criterion for pre-training

optimizer = SGD(model.parameters(), 1e-2)
criterion = F.nll_loss

# pre-train and evaluate the model on MNIST dataset
for epoch in range(10):
    trainer(model, optimizer, criterion)
    evaluator(model)
torch.save(model.state_dict(), "model.pt")
input_names = ["input"]
dummy_input = torch.randn(1,1,28,28).to(device)
torch.onnx.export(model, dummy_input,  'model_o_ori.onnx', input_names=input_names)

# %%
# Pruning Model
# -------------
#
# Using L1NormPruner to prune the model and generate the masks.
# Usually, a pruner requires original model and ``config_list`` as its inputs.
# Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.
#
# The following `config_list` means all layers whose type is `Linear` or `Conv2d` will be pruned,
# except the layer named `fc3`, because `fc3` is `exclude`.
# The final sparsity ratio for each layer is 50%. The layer named `fc3` will not be pruned.

config_list = [{
    'sparsity_per_layer': 0.95,
    'op_types': ['Linear', 'Conv2d']
}, {
    'exclude': True,
    'op_names': ['fc3']
}]

# %%
# Pruners usually require `model` and `config_list` as input arguments.

from nni.compression.pytorch.pruning import L1NormPruner
pruner = L1NormPruner(model, config_list)

# show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
print(model)

# %%

# compress the model and generate the masks
_, masks = pruner.compress()
# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

# %%
# Speedup the original model with masks, note that `ModelSpeedup` requires an unwrapped model.
# The model becomes smaller after speedup,
# and reaches a higher sparsity ratio because `ModelSpeedup` will propagate the masks across layers.

# need to unwrap the model, if the model is wrapped before speedup
pruner._unwrap_model()

# speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
from nni.compression.pytorch.speedup import ModelSpeedup

ModelSpeedup(model, torch.rand(3, 1, 28, 28).to(device), masks).speedup_model()

# %%
# the model will become real smaller after speedup
print(model)

# %%
# Fine-tuning Compacted Model
# ---------------------------
# Note that if the model has been sped up, you need to re-initialize a new optimizer for fine-tuning.
# Because speedup will replace the masked big layers with dense small ones.

optimizer = SGD(model.parameters(), 1e-2)
for epoch in range(10):
    trainer(model, optimizer, criterion)
    evaluator(model)


input_names = ["input"]
dummy_input = torch.randn(1,1,28,28).to(device)
torch.onnx.export(model, dummy_input,  'model_o_pru.onnx', input_names=input_names)

model_list.append(model)
model_1 = TorchModel().to(device)
# In this example, we set the architecture of teacher and student to be the same. It is feasible to set a different teacher architecture.
model_1.load_state_dict(torch.load('model.pt'))
model_list.append(model_1)

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def train_stu( models):
    global optimizer
    global criterion
    model_s = models[0].train()
    model_t = models[-1].eval()
    cri_cls = criterion
    cri_kd = DistillKL(4.0)


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_s = model_s(data)
        output_t = model_t(data)

        loss_cls = cri_cls(output_s, target)
        loss_kd = cri_kd(output_s, output_t)
        loss = loss_cls + loss_kd
        loss.backward()

        optimizer.step()

os.makedirs('./experiment_data', exist_ok=True)
best_top1 = -1
for epoch in range(160):
    print('# Epoch {} #'.format(epoch), flush=True)
    train_stu(model_list)
    scheduler = MultiStepLR(
                optimizer, milestones=[int(160*0.5), int(160*0.75)], gamma=0.1)
    scheduler.step()

    # test student only
    top1 = evaluator( model_list[0] )
    print(top1,'is error', flush=True)
    if top1 > best_top1:
        best_top1 = top1
        torch.save(model_list[0].state_dict(), 'model_trained.pth')
        print('Model trained saved to current dir with loss %f',top1 , flush=True)
test final model
model.load_state_dict(torch.load('model_trained.pth'))
print(model)
evaluator(model)

input_names = ["input"]
dummy_input = torch.randn(1,1,28,28).to(device)
torch.onnx.export(model, dummy_input,  'model_o.onnx', input_names=input_names)
