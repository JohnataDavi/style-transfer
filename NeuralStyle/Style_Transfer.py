from IPython.display import Image, display
from Image import ImageClass

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16

class Style_Transfer:
    def __init__(self, ImgClass):
        self.__content_layer_ids = [4]
        self.__style_layer_ids = list(range(13))
        self.ImageClass = ImgClass

    def __enter__(self):
        print("__enter__")

    def __exit__(self, exception_type, exception_value, traceback):
        print("__exit__")

    def __mean_squared_error(self, a, b):
        """
        Diferença quadrática média entre os estimados valores e o que é estimado
        """
        return tf.reduce_mean(tf.square(a - b))

    def __gram_matrix(self, tensor):
        
        # gram matrix é vetor de produtos de ponto para vetores
        # das ativações de recurso de uma camada de estilo
        # 4d tensor da camada convolucional
        shape = tensor.get_shape()
        
        # Obtem o número de canais de recursos para o tensor de entrada
        # que é assumido como sendo de uma camada convolucional com 4-dim.
        num_channels = int(shape[3])

        # Remodela o tensor para que seja uma matriz 2-dim. 
        # Isso essencialmente achata o conteúdo de cada canal de recursos.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])
        
        # Calcule a matriz de Gram como o produto matricial de a matriz 2-dim com ela mesma. 
        # Isso calcula o produtos de ponto de todas as combinações dos canais de recursos
        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram

    def __create_style_loss(self, model, session):
        """
        Cria a função de perda para a imagem de estilo.
        
        Parâmetros:
        session: Uma sessão aberta do TensorFlow para executar o gráfico do modelo.
        modelo: o modelo, é uma instância da classe VGG16.
        """
        feed_dict = model.create_feed_dict(self.ImageClass.style_image)
        
        # Obtenha referências aos tensores para as camadas dadas.
        layers = model.get_layer_tensors(self.__style_layer_ids)

        # Defina o gráfico do modelo como padrão para que possamos adicionar
        # nós computacionais para ele. Nem sempre é claro
        # quando isso é necessário no TensorFlow, mas se você
        # deseja reutilizar este código, então pode ser necessário. '__enter__', '__exit__'
        with model.graph.as_default():
            # Construct the TensorFlow-operations for calculating
            # the Gram-matrices for each of the layers.
            gram_layers = [self.__gram_matrix(layer) for layer in layers]

            # Calcular os valores dessas matrizes de Gram quando
            # alimentando a imagem de estilo para o modelo.
            values = session.run(gram_layers, feed_dict)

            layer_losses = []
            for value, gram_layer in zip(values, gram_layers):
                
                # Estes são os valores da matriz grama que são calculados
                # para esta camada no modelo ao inserir o
                # style-image. Envolva-o para garantir que é um const,
                # embora isso possa ser feito automaticamente pelo TensorFlow.
                value_const = tf.constant(value)
                
                # A função de perda para esta camada é a
                # Mean Squared Error entre os valores da matriz gramatical
                # para o conteúdo e imagens mistas.
                # Observe que a imagem misturada não é calculada
                # no entanto, estamos apenas criando as operações
                # para calcular o MSE entre esses dois.
                loss = self.__mean_squared_error(gram_layer, value_const)

                layer_losses.append(loss)
            
            # A perda combinada para todas as camadas é apenas a média.
            # As funções de perda podem ser ponderadas diferentemente para
            # cada camada. Você pode tentar e ver o que acontece.
            total_loss = tf.reduce_mean(layer_losses)
        return total_loss

    def __create_content_loss(self, model, session):
        """
        Cria a função de perda para a imagem do conteúdo.
        Parâmetros:
        session: Uma sessão aberta do TensorFlow para executar o gráfico do modelo.
        modelo: O modelo, é uma instância da classe VGG16.
        """
        # Cria um feed-dict com uma imagem de conteúdo.
        feed_dict = model.create_feed_dict(self.ImageClass.content_image)


        # Obtenha referências aos tensores para as camadas dadas.
        layers = model.get_layer_tensors(self.__content_layer_ids)

        # Calcula os valores de saída dessas camadas quand alimentando a imagem de conteúdo para o modelo.
        values = session.run(layers, feed_dict)

        # Define o gráfico do modelo como padrão para que possamos adicionar nós computacionais para ele. 
        # Nem sempre é claro quando isso é necessário no TensorFlow, 
        # mas se você deseja reutilizar este código,então pode ser necessário.
        with model.graph.as_default():
            
            # Inicializa uma lista vazia de funções de perda.
            layer_losses = []

            print("\nValues: ", values)
            print("\n Layesrs: ", layers)
            # Para cada camada e seus valores correspondentes
            # para a imagem de conteúdo.
            for value, layer in zip(values, layers):
                
                # Estes são os valores calculados para esta camada no modelo ao inserir a imagem de conteúdo.
                # Envolva-o para garantir é uma const
                # (embora isso possa ser feito automaticamente pelo TensorFlow.)
                value_const = tf.constant(value)

                # A função de perda para esta camada é a Mean Squared Error 
                # entre os valores da camada ao inserir o conteúdo e imagens mistas.
                # Observe que a imagem misturada não é calculada no entanto,
                # estamos apenas criando as operações para calcular o MSE entre esses dois.
                loss = self.__mean_squared_error(layer, value_const)
                
                # Adicione a função de perda para esta camada na lista de funções de perda.
                layer_losses.append(loss)
            
            # A perda combinada para todas as camadas é apenas a média.
            # As funções de perda podem ser ponderadas diferentemente para
            # cada camada. Você pode tentar e ver o que acontece.
            total_loss = tf.reduce_mean(layer_losses)
        return total_loss

    def __create_denoise_loss(self, model):
        """
        Isso cria a função de perda para minimizar a imagem mista.
        O algoritmo é chamado de [Total Variation Denoising]
        (https://en.wikipedia.org/wiki/Total_variation_denoising)
        """
        loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
        return loss

    def start_style_transfer(self, weight_content=1.5, weight_style=10.0, weight_denoise=0.3, num_iterations=120, step_size=10.0):
        """
        Ultiliza do gradiente descendente para encontrar uma imagem que minimize a
        funções de perda das camadas de conteúdo e camadas de estilo. Este
        deve resultar em uma imagem mista que se assemelha aos contornos
        da imagem de conteúdo e se assemelha às cores e texturas
        da imagem de estilo.
        
        Parâmetros:
        weight_content: peso para a função de perda de conteúdo.
        weight_style: Peso para a função de perda de estilo.
        weight_denoise: Peso para a função de perda de denoização.
        num_iterations: Número de iterações de otimização a serem executadas.
        step_size: tamanho do passo para o gradiente em cada iteração.
        """

        # Cria uma instância do modelo VGG16. Isso está feito em cada chamada desta função, 
        # porque vamos adicionar operações para o gráfico para que ele possa crescer muito
        # e fique sem RAM se continuarmos usando a mesma instância.
        model = vgg16.VGG16() 
        # - foi pro construtor

        # Crie uma sessão no TensorFlow.    
        session = tf.InteractiveSession(graph=model.graph)
        #self.__enter__
    
        print("\nCamadas de conteúdo: ", model.get_layer_names(self.__content_layer_ids))

        print("\nCamadas de estilo: ", model.get_layer_names(self.__style_layer_ids))
    
        # Crie a função Perda para as camadas de conteúdo e imagem. 
        loss_content = self.__create_content_loss(model, session)

        # Cria a função Perda para as camadas de estilo e imagem.
        loss_style = self.__create_style_loss(model, session)    

        # Create the loss-function for the denoising of the mixed-image.
        loss_denoise = self.__create_denoise_loss(model)
    
        # Crie variáveis ​​TensorFlow para ajustar os valores de as funções de perda.
        adj_content = tf.Variable(1e-10, name='adj_content')
        adj_style = tf.Variable(1e-10, name='adj_style')
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')

        # Inicialize os valores de ajuste para as funções de perda. 
        session.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])

        # Crie operações TensorFlow para atualizar os valores de ajuste.
        # Estes são basicamente apenas os valores recíprocos do
        # loss-functions, com um pequeno valor 1e-10 adicionado para evitar
        # possibilidade de divisão por zero.
        update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
        update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
        update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

        # Esta é a função de perda ponderada que iremos minimizar abaixo para gerar a imagem mista.
        # Porque multiplicamos os valores de perda com o seu recíproco valores de ajuste, 
        # podemos usar pesos relativos para o loss-functions que são mais fáceis de selecionar, 
        # como são independente da escolha exata das camadas de estilo e conteúdo
        loss_combined = weight_content * adj_content * loss_content + \
                        weight_style * adj_style * loss_style + \
                        weight_denoise * adj_denoise * loss_denoise

        # Usa o TensorFlow para obter a função matemática para o gradiente da função de perda 
        # combinada em relação a imagem de entrada.
        gradient = tf.gradients(loss_combined, model.input)

        # Lista de tensores que serão executados em cada iteração de otimização.
        run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

        # A imagem mista é inicializada com ruído aleatório. É do mesmo tamanho que a imagem de conteúdo.
        mixed_image = np.random.rand(*self.ImageClass.content_image.shape) + 128

        for i in range(num_iterations):
            # Cria um feed-dict com a imagem mista.
            feed_dict = model.create_feed_dict(image=mixed_image)

            # Usa o TensorFlow para calcular o valor do gradiente, bem como atualizar os valores de ajuste.
            grad, adj_content_val, adj_style_val, adj_denoise_val \
            = session.run(run_list, feed_dict=feed_dict)
    
            # Reduz a dimensionalidade do gradiente.
            grad = np.squeeze(grad)
            
            # Escale o tamanho do passo de acordo com os valores de gradiente.
            step_size_scaled = step_size / (np.std(grad) + 1e-8)

            # Atualize a imagem seguindo o gradiente.
            mixed_image -= grad * step_size_scaled

            # Assegure-se de que a imagem tenha valores de pixel válidos entre 0 e 255.
            mixed_image = np.clip(mixed_image, 0.0, 255.0)

            print("#", end=' ')
            if (i % 60 == 0) or (i == num_iterations - 1):
                print("\nIteração: ", i)
                # Imprime pesos de ajuste para funções de perda.
                print("\nPeso ajustados para conteúdo: {0:.3e}, Estilo: {1:.3e}, Denoise: {2:.3e}".format(adj_content_val, adj_style_val, adj_denoise_val))
                self.ImageClass.plot_images(mixed_image)

        print("\nImagem Final Array: ", mixed_image)
        return self.ImageClass.array_np_to_img(mixed_image)

