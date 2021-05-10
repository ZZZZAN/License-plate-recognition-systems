from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from conv.data import read_data
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt

# x_train, y_train = read_data(r'D:\Graduation_project\train', r'D:\Graduation_project\train1.txt')
# x_test, y_test = read_data(r'D:\Graduation_project\test', r'D:\Graduation_project\test1.txt')
# x_train_save = np.reshape(x_train, (len(x_train), -1))
# x_test_save = np.reshape(x_test, (len(x_test), -1))
# np.save(r'D:\Graduation_project\x_train.npy', x_train_save)
# np.save(r'D:\Graduation_project\y_train.npy', y_train)
# np.save(r'D:\Graduation_project\x_test.npy', x_test_save)
# np.save(r'D:\Graduation_project\y_test.npy', y_test)
x_train = []
y_train = []
x_test = []
y_test = []
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\0', '0', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\1', '1', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\2', '2', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\3', '3', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\4', '4', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\5', '5', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\6', '6', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\7', '7', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\8', '8', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\9', '9', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\A', '10', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\B', '11', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\C', '12', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\D', '14', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\E', '15', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\F', '17', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\G', '18', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\H', '23', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\J', '26', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\K', '31', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\L', '32', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\M', '35', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\N', '38', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\P', '40', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\Q', '41', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\R', '44', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\S', '45', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\T', '49', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\U', '50', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\V', '51', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\W', '52', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\X', '54', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\Y', '57', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\chars2\Z', '62', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_cuan', '13', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_e', '16', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_gan', '19', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_gan1', '20', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_gui', '21', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_gui1', '22', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_hei', '24', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_hu', '25', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_ji', '27', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_jin', '28', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_jing', '29', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_jl', '30', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_liao', '33', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_lu', '34', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_meng', '36', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_min', '37', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_ning', '39', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_qing', '42', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_qiong', '43', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_shan', '46', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_su', '47', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_sx', '48', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_wan', '53', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_xiang', '55', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_xin', '56', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_yu', '58', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_yu1', '59', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_yue', '60', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_yun', '61', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_zang', '63', x_train, y_train, x_test, y_test)
x_train, y_train, x_test, y_test = read_data(r'D:\new_folder\train\charsChinese\zh_zhe', '64', x_train, y_train, x_test, y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_train = y_train.astype(np.int64)
y_test = np.array(y_test)
y_test = y_test.astype(np.int64)
x_train_save = np.reshape(x_train, (len(x_train), -1))
x_test_save = np.reshape(x_test, (len(x_test), -1))
np.save(r'D:\Graduation_project\x_train.npy', x_train_save)
np.save(r'D:\Graduation_project\y_train.npy', y_train)
np.save(r'D:\Graduation_project\x_test.npy', x_test_save)
np.save(r'D:\Graduation_project\y_test.npy', y_test)


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        # self.c1 = Conv2D(filters=6, kernel_size=(5, 5),
        #                  activation='sigmoid')
        self.c1 = Conv2D(filters=10, kernel_size=(3, 3),
                         activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        # self.c2 = Conv2D(filters=16, kernel_size=(5, 5),
        #                  activation='sigmoid')
        self.c2 = Conv2D(filters=20, kernel_size=(3, 3),
                         activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.flatten = Flatten()
        self.f1 = Dense(1000, activation='sigmoid')
        self.f2 = Dense(1000, activation='sigmoid')
        self.f3 = Dense(65, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


model = LeNet5()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/LeNet5.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=256, epochs=50,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()