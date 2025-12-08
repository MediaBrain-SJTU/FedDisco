import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications


class SimpleCNN_header(keras.Model):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = layers.Conv2D(6, 5, activation='relu')
        self.pool = layers.MaxPool2D(2, 2)
        self.conv2 = layers.Conv2D(16, 5, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(hidden_dims[0], activation='relu')
        self.fc2 = layers.Dense(hidden_dims[1], activation='relu')

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SimpleCNNMNIST_header(keras.Model):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST_header, self).__init__()
        self.conv1 = layers.Conv2D(6, 5, activation='relu')
        self.pool = layers.MaxPool2D(2, 2)
        self.conv2 = layers.Conv2D(16, 5, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(hidden_dims[0], activation='relu')
        self.fc2 = layers.Dense(hidden_dims[1], activation='relu')

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ModelFedCon_noheader(keras.Model):
    def __init__(self, base_model, out_dim, n_classes, dataset=None):
        super(ModelFedCon_noheader, self).__init__()
        if base_model == 'resnet18_gn':
            basemodel = applications.ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
            # Note: TensorFlow ResNet doesn't have exact same structure, using ResNet50 as alternative
            # For resnet18_gn with GroupNorm modifications, would need custom implementation
            self.features = basemodel
            num_ftrs = 2048  # ResNet50 feature size
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        # Last layer
        self.l3 = layers.Dense(n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    # def call(self, x):
    #     h = self.features(x)
    #     if len(h.shape) > 2:
    #         h = tf.reduce_mean(h, axis=[1, 2])  # global average pooling
    #     # 保证 h 至少 2D
    #     if len(h.shape) == 1:
    #         h = tf.expand_dims(h, axis=0)
    #     y = self.l3(h)
    #     return h, h, y


    # def call(self, x):
    #     h = self.features(x)
    #     if len(h.shape) > 2:
    #         h = tf.reduce_mean(h, axis=[1, 2])  # Global average pooling instead of squeeze
    #     else:
    #         h = tf.squeeze(h)
    #     y = self.l3(h)
    #     return h, h, y

    def call(self, x):
        h = self.features(x)  # [B, C, H, W]

        # 手动 flatten，等价 PyTorch Linear 的行为
        h = tf.reshape(h, [h.shape[0], -1])  # [B, C*H*W]

        y = self.l3(h)  # Dense 正常输入
        return h, h, y
