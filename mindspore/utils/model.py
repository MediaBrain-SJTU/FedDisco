import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class SimpleCNN_header(nn.Cell):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(input_dim, hidden_dims[0])
        self.fc2 = nn.Dense(hidden_dims[0], hidden_dims[1])
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SimpleCNNMNIST_header(nn.Cell):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(input_dim, hidden_dims[0])
        self.fc2 = nn.Dense(hidden_dims[0], hidden_dims[1])
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class ModelFedCon_noheader(nn.Cell):
    def __init__(self, base_model, out_dim, n_classes, dataset=None):
        super(ModelFedCon_noheader, self).__init__()
        
        # Note: ResNet implementation in MindSpore might differ slightly or need import from mindspore.model_zoo
        # For now, we will focus on the SimpleCNN parts as they are explicitly defined here.
        # If ResNet is needed, we would typically use `mindspore.models.resnet18` if available or define it.
        # Given the context, I will implement the logic for SimpleCNNs which seems to be the default.
        
        if base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        else:
             # Fallback or placeholder for ResNet if strictly required, but for now raising error or simplifying
             # Assuming simple-cnn for this conversion pass as per common FL benchmarks
             raise NotImplementedError("ResNet conversion requires importing MindSpore model zoo or custom implementation.")

        # Last layer
        self.l3 = nn.Dense(num_ftrs, n_classes)

    def construct(self, x):
        h = self.features(x)
        h = h.squeeze() # MindSpore Dense expects (N, in_channels), Flatten already did this
        y = self.l3(h)
        return y
