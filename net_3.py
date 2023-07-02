from keras.datasets import mnist
import keras
from keras.layers import Dense, Dropout, Flatten
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

# Модель нейросети (784 входа, 128 нейронов скрытого слоя, 10 нейронов выходного слоя)
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
 
print(model.summary())
# Загрузка обучающих данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Вывод первых 25 образцов обучающих данных
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

#Настройка данных
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Выбор способа оптимизации 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Обучение нейросети
model.fit(x_train, y_train_cat, batch_size = 32, epochs = 5, validation_split = 0.2)
print("Обучение завершено")


model.evaluate(x_test, y_test_cat)

input("Нажмите любую клавишу чтобы приступить к распознованию цифр")
while(True):
    file = input("Введите имя файла с изображением цифры: ")
    if(file == "\n"):
        break
    try:
        image_in = "Test/" + file
        img = im.open(image_in) 
        img.show()
        img = img.resize((28, 28))
        img = img.convert('L')
        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        # Вызов предсказания
        result = model.predict([img])[0]
    except FileNotFoundError:
        print("Файл не найден")
        continue
    # Вывод индекса с максимальным значением в выходном векторе
    print("Значение ", np.argmax(result), " Точность: ", max(result))