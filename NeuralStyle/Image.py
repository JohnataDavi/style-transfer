from IPython.display import Image, display
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

class ImageClass:
    def __init__(self, dir_content_img, dir_style_img):
        self.dir_content_image = dir_content_img
        self.dir_style_image = dir_style_img
        self.content_image = self.load_image(self.dir_content_image, max_size=1500)
        self.style_image = self.load_image(self.dir_style_image, max_size=1500)

    def load_image(self, filename, max_size=None):
        image = PIL.Image.open(filename)

        if max_size is not None:
            factor = max_size / np.max(image.size)
            size = np.array(image.size) * factor
            size = size.astype(int)
            image = image.resize(size, PIL.Image.LANCZOS)

        return np.float32(image)

    def array_np_to_img(self, image):
        new = (image * 255 / np.max(image)).astype('uint8')
        return PIL.Image.fromarray(new)

    def save_any_image(self, image, case):
        date = datetime.now()
        if case == 0:
            image.save('Img/Results/Plot - {}-{}-{} {}-{}-{}.jpg'.format(date.year, date.month, date.day, date.hour, date.minute, date.second))
        elif case == 1:
            image.save('Img/Test/Plot - {}-{}-{} {}-{}-{}.jpg'.format(date.year, date.month, date.day, date.hour, date.minute, date.second))
        
    def plot_images(self, mixed_image):
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        smooth = True

        if smooth:
            interpolation = 'sinc'
        else:
            interpolation = 'nearest'

        ax = axes.flat[0]
        ax.imshow(self.style_image / 255.0, interpolation=interpolation)
        ax.set_xlabel("Estilo")
        
        ax = axes.flat[1]
        ax.imshow(self.content_image / 255.0, interpolation=interpolation)
        ax.set_xlabel("Original")
        
        ax = axes.flat[2]
        ax.imshow(mixed_image / 255.0, interpolation=interpolation)
        ax.set_xlabel("Resultado")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def plot_any_image(self, image):
        image.show()

