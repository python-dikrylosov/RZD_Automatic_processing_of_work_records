import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2

# Загрузка модели
model = load_model('model.h5')

# Загрузка нового изображения
new_image_path = image
new_image = load_img(new_image_path, target_size=(278, 278))
new_image = img_to_array(new_image)
new_image = new_image / 255.0

# Предсказание
prediction = model.predict(np.expand_dims(new_image, axis=0))

# Показать результат
print("Predicted probability :", prediction)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('File Loader')
        self.geometry('600x480')

        # Создаем кнопку для выбора файла изображения
        self.load_image_button = tk.Button(self, text='Load Image', command=self.open_image)
        self.load_image_button.pack(pady=10)

        # Создаем фрейм для отображения содержимого файла
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(pady=10)

        # Функция для открытия файла изображения
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "    *    .jpg     *    .png     *    .gif")])
        if file_path:
            with open(file_path, 'rb') as file:
                image_data = file.read()  # Получаем бинарный код изображения
                self.show_image(image_data, file_path)

    # Функция для показа содержимого файла
    def show_image(self, image_data, file_path):
        self.content_frame.destroy()  # Удаляем старый фрейм, если он есть
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(pady=10)

        # Загружаем изображение с помощью PIL
        image = Image.open(file_path)
        image = image.resize((1920, 1080), resample=Image.NEAREST)  # Используйте Image.NEAREST вместо Image.ANTIALIAS
        image.show()  # Показываем изображение

        # Создаем Label для отображения изображения
        self.image_label = tk.Label(self.content_frame, image=image)
        self.image_label.pack()

        # Добавляем кнопку для сохранения файла в формате XLS
        save_xls_button = tk.Button(self, text='Save as XLS', command=lambda: self.save_as_xls(image_data, file_path))
        save_xls_button.pack(pady=10)

        # Загрузка модели
        model = load_model('model.h5')

        # Загрузка нового изображения
        new_image_path = image
        new_image = load_img(new_image_path, target_size=(278, 278))
        new_image = img_to_array(new_image)
        new_image = new_image / 255.0

        # Предсказание
        prediction = model.predict(np.expand_dims(new_image, axis=0))

        # Показать результат
        print("Predicted probability :", prediction)

        # Создайте маску для добавления рамок
        mask = np.zeros(shape=new_image.shape, dtype=np.uint8)
        mask[0:100, 0:100] = [255, 0, 0]  # Это красная рамка сверху слева
        mask[0:100, -100:] = [0, 255, 255]  # Это зеленая рамка снизу
        mask[-100:, 0:100] = [255, 0, 255]  # Это синяя рамка справа

        # Примените маску к изображению
        result = cv2.addWeighted(new_image, 0.7, mask, 0.3, 0)

        # Покажите результат
        plt.imshow(result)
        plt.show()

        # Добавляем этот код в метод show_image после создания Label для отображения изображения
        self.image_label = tk.Label(self.content_frame, image=result)
        self.image_label.pack()

        # Если вы хотите отобразить предсказание модели, вы можете добавить это ниже:
        self.prediction_label = tk.Label(self.content_frame, text="Predicted probability: " + str(prediction))
        self.prediction_label.pack()










    def save_as_xls(self, image_data, file_path):
        # Преобразуем бинарный код изображения в Series и добавляем в DataFrame
        images_dicts = [{'image': pd.Series(image_data), 'path': file_path}]
        df = pd.DataFrame(images_dicts)
        df.to_excel(str(file_path) + 'output.xlsx', index=False)
        print('save to')

app = App()
app.mainloop()
