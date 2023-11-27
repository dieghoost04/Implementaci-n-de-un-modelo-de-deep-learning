import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import models
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import DenseNet169

from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import os
from PIL import Image
import numpy as np

directory = './images'
directorio_train = './images/entrenamiento'
directorio_test = './images/prueba'
directorio_val = './images/validacion'


train_dir = os.path.join(directorio_train)
test_dir = os.path.join(directorio_test)
val_dir = os.path.join(directorio_val)


class cnn:
    def __init__(self, train_dir, test_dir, val_dir):
        '''
        Constructor de la clase cnn.
        '''
        print('\n INIT \n')


        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def data_augmentation(self, brightness_range = [0.5, 1.5], zoom_range = [0.7, 1.0], batch_size = 16):
        '''
        Crea imágenes con atributos nuevos, utilizando la función de 
        ImageDataGenerator, y flow_from_directory
        '''
        print('\n DATA AUGMENTATION \n')

        # Utilizamos el ImageDataGenerator para ajustar el brillo de las imágenes y un zoom aleatorio que se les haga
        train_datagen = ImageDataGenerator(
            rescale=1./255,  
            brightness_range=(brightness_range[0], brightness_range[1]),  
            zoom_range=[zoom_range[0],zoom_range[1]]
        )

        val_datagen = ImageDataGenerator(rescale = 1./255)

        self.train_generator = train_datagen.flow_from_directory(
                                    self.train_dir,
                                    target_size = (224, 224),
                                    batch_size = batch_size,
                                    class_mode ='binary')

        self.val_generator = val_datagen.flow_from_directory(
                                    self.val_dir,
                                    target_size = (224, 224),
                                    batch_size = batch_size,
                                    class_mode ='binary')

        self.class_names = list(self.train_generator.class_indices.keys())
    
    def plot_images(self):
        '''
        Muestra algunas de las imágenes con sus nuevas características
        '''

        print('\n PLOT IMAGES \n')

        # Cargamos las imágenes con sus labels dentro de dos listas para poderlas visualizar
        batch_images, batch_labels = next(self.train_generator)

        num_images_to_plot = 5

        for i in range(num_images_to_plot):
            image = batch_images[i]  
            label = batch_labels[i]

            plt.subplot(1, num_images_to_plot, i + 1)
            plt.imshow(image)
            
            plt.title('NORMAL') if label == 0 else plt.title('PNEUMONIA') 
            plt.axis('off')

        plt.show()
    
    def model_compile(self,  learning_rate = 0.0001, loss='binary_crossentropy', metrics='acc'):
        '''
        Se compila el modelo con la arquitectura de DenseNet169 y las últimas 4 capas que se 
        agregan.
        '''

        print('\n MODEL COMPILE \n')

        # Se utiliza DenseNet169 ya que es mas ligera y permite un entrenamiento mas rápido.
        base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        # Hacemos las capas de DenseNet no entrenables excepto la última capa
        for layer in base_model.layers:
            layer.trainable = False

        base_model.layers[-2].trainable = True


        self.model = models.Sequential()
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dropout(0.5))
        self.model.add(layers.Dense(512, activation='selu'))  # Reducción del número de neuronas
        self.model.add(layers.Dense(1, activation='sigmoid'))  # Activación 'sigmoid' para clasificación binaria
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def model_fit(self, patience = 3, epochs = 25):
        '''
        Entrena el modelo previamente compilado
        '''
    
        self.data_augmentation()
        self.model_compile()
        
        print('\n MODEL FIT \n')

        # Se utiliza para detener el entrenamiento cuando 'val_acc' no mejora despues n epochs
        early_stopping = EarlyStopping(
            monitor='val_acc',  
            patience=patience,  
            restore_best_weights=True  
        )


        model_checkpoint = ModelCheckpoint(
            filepath='best_model.h5',  
            monitor='val_acc',  
            save_best_only=True,  
        )

        self.history = self.model.fit(self.train_generator,
                            validation_data = self.val_generator,
                            epochs=epochs, 
                            callbacks=[early_stopping, model_checkpoint])
        
				
    def plot_train_parameters(self):
        '''
        Grafica el historial de las métricas
        durante el entrenamiento del modelo
        '''
        
        print('\n PLOT TRAIN PARAMETERS \n')

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc)+1)

        plt.plot(epochs,acc,'bo',label='train accuracy')
        plt.plot(epochs,val_acc, 'b', label='validation accuracy')
        plt.title('train acc vs val acc')
        plt.legend()

        plt.figure()

        plt.plot(epochs,loss, 'bo', label ='training loss')
        plt.plot(epochs,val_loss, 'b', label = 'validation loss')
        plt.title('train loss vs val loss')
        plt.legend()

        plt.show()

    def evaluation(self, batch_size = 16):
        '''Este método se utiliza para evaluar los mejores pesos que se encontraron
           durante el entrenamiento con la arquitectura que definimos.'''
        
        self.model_compile()

        print('\n EVALUATION \n')

        # Se cargan los mejores pesos que se guardan del entrenamiento
        #self.model.load_weights('best_model.h5')
        test_datagen = ImageDataGenerator(1./255)

        self.test_generator = test_datagen.flow_from_directory(
                            self.test_dir,
                            target_size = (224, 224),
                            batch_size = batch_size,
                            class_mode= 'binary')

        test_loss, test_acc = self.model.evaluate(self.test_generator)
        self.class_names = list(self.test_generator.class_indices.keys())

        print('\nTest accuracy :\n', test_acc)
        print('\nTest loss :\n', test_loss)

        self.y_pred = self.model.predict(self.test_generator)

        self.y_pred_labels = (self.y_pred > 0.5).astype(int)

        self.test_labels = self.test_generator.labels

        confusion = confusion_matrix(self.test_labels, self.y_pred_labels)

        # Muestra la matriz de confusión
        disp = ConfusionMatrixDisplay(confusion, display_labels = self.class_names)  # Reemplaza "classes" con tus nombres de clases
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Matriz de Confusión')
        plt.show()

    def prediction(self):
        '''
        Este método hace predicciones de imágenes aleatorias, y muestra la predicción y la clase real.
        '''
        while True:
            # Selecciona una imagen del conjunto de prueba
            image_index = np.random.randint(0, len(self.test_labels))  # Escoge una imagen aleatoria

            # Obtiene la ruta de la imagen
            image_path = self.test_generator.filepaths[image_index]

            # Carga la imagen
            image = plt.imread(image_path)

            # Realiza la predicción en la imagen seleccionada
            predicted_class = int(self.y_pred_labels[image_index])
            real_class = self.test_labels[image_index]

            # Muestra la imagen
            plt.imshow(image, cmap='gray')
            plt.title(f'Predicción: {self.class_names[predicted_class]}, Clase Real: {self.class_names[real_class]}')
            plt.show()

    def prediction_internet(self):
        '''
        Este método hace predicciones de imágenes aleatorias, y muestra la predicción y la clase real.
        '''
        image_path = './imagen.jpg'  # Reemplaza con la ruta correcta a tu imagen
        
        image = Image.open(image_path)

        # Asegúrate de que la imagen tenga las dimensiones correctas y se preprocese según sea necesario
        # Aquí, se asume que el modelo originalmente fue entrenado con imágenes de tamaño (224, 224, 3)
        # Si tu modelo espera un tamaño diferente, ajusta el tamaño y el preprocesamiento en consecuencia

        # Ajusta el tamaño a (224, 224)
        image = image.resize((224, 224))
        
        # Convierte la imagen a un array NumPy
        image_array = np.array(image)

        # Normaliza los valores de píxeles
        image_array = image_array / 255.0

        # Agrega una dimensión adicional para la dimensión del lote (batch)
        image_array = np.expand_dims(image_array, axis=0)

        # Realiza la predicción en la imagen seleccionada
        predicted_prob = self.model.predict(image_array)[0][0]
        predicted_class = int(predicted_prob > 0.5)  # Usando un umbral de 0.5 para clasificación binaria

        # Muestra la imagen
        plt.imshow(image_array[0], cmap='gray')
        plt.title(f'Predicción: {self.class_names[predicted_class]}, Probabilidad: {predicted_prob:.4f}')
        plt.show()

    # Ejemplo de uso
    
    



def evaluate():
    '''Para evaluar el modelo con los mejores pesos del entrenamiento
       solo llamar al metodo de evaluación'''
    
    xray = cnn(train_dir, test_dir, val_dir)
    xray.evaluation()
    xray.prediction_internet()

def train():
    '''Para entrenar el modelo, poder visualizar los cambios que se hicieron
       en las imágenes y graficar los paramétros del entrenamiento'''
    
    xray = cnn(train_dir, test_dir, val_dir)
    xray.plot_images()
    xray.model_fit()
    xray.plot_train_parameters()
    


evaluate()
