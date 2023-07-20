"""线性回归"""
import math

import mindspore
import numpy as np
from matplotlib import pyplot as plt
from mindspore import nn
from mindspore.common.initializer import HeUniform

# 超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# 简单的数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 线性回归模型
model = nn.Dense(input_size, output_size, weight_init=HeUniform(math.sqrt(5)))

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = nn.optim.SGD(model.trainable_params(), learning_rate=learning_rate)

# 绑定网络和损失函数
model_with_loss = nn.WithLossCell(model, criterion)
# 封装训练网络
train_model = nn.TrainOneStepCell(model_with_loss, optimizer)

# 训练模型
for epoch in range(num_epochs):
    train_model.set_train()
    inputs = mindspore.Tensor.from_numpy(x_train)
    targets = mindspore.Tensor.from_numpy(y_train)

    loss = train_model(inputs, targets)

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.asnumpy().item():.4f}')

# Plot the graph
predicted = model(mindspore.Tensor.from_numpy(x_train)).asnumpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
mindspore.save_checkpoint(model, 'model.ckpt')
