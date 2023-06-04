# import mindspore
# import mindspore.nn as nn
# import numpy as np
# import mindspore.dataset.transforms as transforms
# from mindspore import ops
#
# # ================================================================== #
# #                         Table of Contents                          #
# # ================================================================== #
#
# # 1. Basic autograd example 1               (Line 25 to 39)
# # 2. Basic autograd example 2               (Line 46 to 83)
# # 3. Loading data from numpy                (Line 90 to 97)
# # 4. Input pipline                          (Line 104 to 129)
# # 5. Input pipline for custom dataset       (Line 136 to 156)
# # 6. Pretrained model                       (Line 163 to 176)
# # 7. Save and load model                    (Line 183 to 189)
#
#
# # ================================================================== #
# #                     1. Basic autograd example 1                    #
# # ================================================================== #
#
# # Create tensors.
# x = mindspore.Tensor(1.)
# w = mindspore.Tensor(2.)
# b = mindspore.Tensor(3.)
#
# # Build a computational graph.
# y = w * x + b  # y = 2 * x + 3
#
# # Compute gradients.
# grad_fn = ops.grad(y)
#
# # Print out the gradients.
# print(grad_fn(x))  # x.grad = 2
# print(grad_fn(w))  # w.grad = 1
# print(grad_fn(b))  # b.grad = 1
#
# # ================================================================== #
# #                    2. Basic autograd example 2                     #
# # ================================================================== #
#
# # Create tensors of shape (10, 3) and (10, 2).
# x = ops.randn(10, 3)
# y = ops.randn(10, 2)
#
# # Build a fully connected layer.
# linear = nn.Linear(3, 2)
# print('w: ', linear.weight)
# print('b: ', linear.bias)
#
# # Build loss function and optimizer.
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
#
# # Forward pass.
# pred = linear(x)
#
# # Compute loss.
# loss = criterion(pred, y)
# print('loss: ', loss.item())
#
# # Backward pass.
# loss.backward()
#
# # Print out the gradients.
# print('dL/dw: ', linear.weight.grad)
# print('dL/db: ', linear.bias.grad)
#
# # 1-step gradient descent.
# optimizer.step()
#
# # You can also perform gradient descent at the low level.
# # linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# # linear.bias.data.sub_(0.01 * linear.bias.grad.data)
#
# # Print out the loss after 1-step gradient descent.
# pred = linear(x)
# loss = criterion(pred, y)
# print('loss after 1 step optimization: ', loss.item())
#
# # ================================================================== #
# #                     3. Loading data from numpy                     #
# # ================================================================== #
#
# # Create a numpy array.
# x = np.array([[1, 2], [3, 4]])
#
# # Convert the numpy array to a torch tensor.
# y = torch.from_numpy(x)
#
# # Convert the torch tensor to a numpy array.
# z = y.numpy()
#
# # ================================================================== #
# #                         4. Input pipeline                           #
# # ================================================================== #
#
# # Download and construct CIFAR-10 dataset.
# train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                              train=True,
#                                              transform=transforms.ToTensor(),
#                                              download=True)
#
# # Fetch one data pair (read data from disk).
# image, label = train_dataset[0]
# print(image.size())
# print(label)
#
# # Data loader (this provides queues and threads in a very simple way).
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=64,
#                                            shuffle=True)
#
# # When iteration starts, queue and thread start to load data from files.
# data_iter = iter(train_loader)
#
# # Mini-batch images and labels.
# images, labels = data_iter.next()
#
# # Actual usage of the data loader is as below.
# for images, labels in train_loader:
#     # Training code should be written here.
#     pass
#
#
# # ================================================================== #
# #                5. Input pipeline for custom dataset                 #
# # ================================================================== #
#
# # You should build your custom dataset as below.
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # TODO
#         # 1. Initialize file paths or a list of file names.
#         pass
#
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         pass
#
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return 0
#
#
# # You can then use the prebuilt data loader.
# custom_dataset = CustomDataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
#                                            batch_size=64,
#                                            shuffle=True)
#
# # ================================================================== #
# #                        6. Pretrained model                         #
# # ================================================================== #
#
# # Download and load the pretrained ResNet-18.
# resnet = torchvision.models.resnet18(pretrained=True)
#
# # If you want to finetune only the top layer of the model, set as below.
# for param in resnet.parameters():
#     param.requires_grad = False
#
# # Replace the top layer for finetuning.
# resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.
#
# # Forward pass.
# images = torch.randn(64, 3, 224, 224)
# outputs = resnet(images)
# print(outputs.size())  # (64, 100)
#
# # ================================================================== #
# #                      7. Save and load the model                    #
# # ================================================================== #
#
# # Save and load the entire model.
# torch.save(resnet, 'model.ckpt')
# model = torch.load('model.ckpt')
#
# # Save and load only the model parameters (recommended).
# torch.save(resnet.state_dict(), 'params.ckpt')
# resnet.load_state_dict(torch.load('params.ckpt'))
