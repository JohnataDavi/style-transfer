from Image import ImageClass
from Style_Transfer import Style_Transfer
import tensorflow as tf
import vgg16

if __name__ == "__main__":
    print("Versão tensorflow: ", tf.__version__)
    print("Obs: Ultilizar imagens .jpg!!!")
    print("Para o funcionamento do programa, é necessario baixar o modelo VGG16, e colocar na pasta /vgg16 !!!")
    print("Para isso basta descomentar a linha a baixo ou acessar o link (https://s3.amazonaws.com/cadl/models/vgg16.tfmodel)")
    #vgg16.download_model()
    
    dir_content_img = 'Img/Content/felipe.jpg'
    dir_style_img = 'Img/Styles/flor.jpg'
    img = ImageClass(dir_content_img, dir_style_img)
    
   
    sty_tranfer = Style_Transfer(img)
    img_merge = sty_tranfer.start_style_transfer(weight_content=1.5, weight_style=10.0, weight_denoise=0.3, num_iterations=120, step_size=10.0)
    
    print("\nImage (PIL): ", img_merge)
    img.save_any_image(img_merge, 0)
    img.plot_any_image(img_merge)
    
