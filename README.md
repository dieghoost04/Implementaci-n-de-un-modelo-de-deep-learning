# Implementación de un modelo de deep learning
Se decidió abordar el problema de clasificación de radiografías en el pecho de pacientes con neumonía y pacientes sanos.

Este dataset se recupero de la plataforma de kaggle en la siguiente liga: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

En este repositorio se encuentran los siguientes elementos:
- test: En este carpetas se encuentran las imágenes que se pueden utilizar para la evaluación del modelo, dentro de este se puede encontrar dos carpetas en las que se encuentran las imágenes correspondientes a cada clase
- modelo_deep_learning.py: En este código se encuentra la clase 'cnn', la cual contiene todos los métodos que se utilizaron para entrenar este modelo, visualizar los cambios que se hicieron en las imágenes y las métricas utilizadas.
- reporte.pdf: En este reporte se puede encontrar información sobre las decisiones que se tomaron en la preparación de los datos y en el entrenamiento del modelo.

Muchas de las decisiones que se tomaron para entrenar este modelo se basaron en el artículo de (Lee & Lim, 2022), en el cual se abordo un problema similar con radiografías de pecho para pacientes con COVID-19.

### Referencias
- Lee, C. P., & Lim, K. M. (2022). COVID-19 diagnosis on chest radiographs with enhanced deep neural networks. Diagnostics, 12(8), 1828. https://doi.org/10.3390/diagnostics12081828

## Evaluación

Subcompetencia
	
Indicador

SMA0401C            							Aprendizaje e IA

Se utilizó una arquitectura de aprendizaje profundo, en este caso fue una red convolutiva la cual se utilizó resolver un problema de clasificación con imágenes.
Se utilizaron técnicas de regularización como lo es el data augmentation, para que nuestro dataset fuera más variado.
Se utilizaron técnicas de transfer learning como adoptar los pesos y arquitectura de una red en específico que se mostró útil para nuestro problema en específico.
