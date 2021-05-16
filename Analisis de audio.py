#!/usr/bin/env python
# coding: utf-8

# # Clasificador de audios de emociones

# Hemos seleccionado este dataset porque las emociones quizá sean más relevantes que los ruidos de fondo o los sonidos de animales ya que se tratan de conversaciones entre un agente y el cliente, y hay una variedad de fuentes. 

# Fuentes de datos:
#  - Surrey Audio-Visual Expressed Emotion (SAVEE)
#  - Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
#  - Toronto emotional speech set (TESS)
#  - Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)

# ## **Exploración de datos:**

# In[1]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob 
from sklearn.metrics import confusion_matrix
import IPython.display as ipd
import os
import sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[2]:


TESS =  "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/TESS/TESS Toronto emotional speech set data/"
RAV =   "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/"
SAVEE = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/SAVEE/ALL/"
CREMA = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/CREMA-D/AudioWAV/"


dir_list = os.listdir(SAVEE)
dir_list[0:5]


# **1. SAVEE dataset**

# Las letras del prefijo describen las clases de emoción de la siguiente manera:
# 
# 'a' = 'ira'
# 'd' = 'disgusto'
# 'f' = 'miedo'
# 'h' = 'felicidad'
# 'n' = 'neutral'
# 'sa' = 'tristeza'
# 'su' = 'sorpresa'

# Cada carpeta es un orador; para distinguirlos los audios se hn codificado poniendo primero el hablante Dc y después el número de oración 03. En este conjunto de datos todos los halantesson hombres.
# 
# Exploramos dos emociones diferentes para ver si la calidad de los datos (audios) es buena. Y así saber si el clasificador de audio será bueno o no:

# In[3]:


dir_list = os.listdir(SAVEE)              # Directorio de SAVEE

# Parseamos las carpetas para obtener las emociones
emotion=[]
path = []
for i in dir_list:
    if i[-8:-6]=='_a':
        emotion.append('male_angry')
    elif i[-8:-6]=='_d':
        emotion.append('male_disgust')
    elif i[-8:-6]=='_f':
        emotion.append('male_fear')
    elif i[-8:-6]=='_h':
        emotion.append('male_happy')
    elif i[-8:-6]=='_n':
        emotion.append('male_neutral')
    elif i[-8:-6]=='sa':
        emotion.append('male_sad')
    elif i[-8:-6]=='su':
        emotion.append('male_surprise')
    else:
        emotion.append('male_error') 
    path.append(SAVEE + i)
    
# Hacemos un conteo de las emociones
SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
SAVEE_df.labels.value_counts()


# Necesitamos encontrar el patrón clave que nos ayudará a distinguir las diferentes emociones:

# In[4]:


fname = SAVEE + 'DC_f11.wav'  
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

ipd.Audio(fname)


# Los datos tienen calidad ya que no hay mucho ruido de fondo y el habla es muy clara.  El gráfico de ondas no dice mucho más que una variación en la onda, lo cual es bueno para identificar claramente qué emoción transmite: miedo.

# **2. RAVDESS dataset**

# Identificadores de nombre de archivo según el sitio web:
#  - Modalidad (01 = AV completo, 02 = solo video, 03 = solo audio).
#  - Canal vocal (01 = habla, 02 = canción).
#  - Emoción (01 = neutral, 02 = calma, 03 = feliz, 04 = triste, 05 = enojado, 06 = temeroso, 07 = disgusto, 08 = sorprendido).
#  - Intensidad emocional (01 = normal, 02 = fuerte) NOTA: No hay una intensidad fuerte para la emoción 'neutral'.
#  - Declaración (01 = "Los niños están hablando junto a la puerta", 02 = "Los perros están sentados junto a la puerta").
#  - Repetición (01 = 1ª repetición, 02 = 2ª repetición).
#  - Actor (01 a 24. Los actores impares son hombres, los actores pares son mujeres).
# 

# Metadatos del archivo de audio (02-01-06-01-02-01-12.mp4):
# 
#  - Solo video (02)
#  - Discurso (01)
#  - Temeroso (06)
#  - Intensidad normal (01)
#  - Declaración "perros" (02)
#  - 1a repetición (01)
#  - 12 ° actor (12) - Mujer (ya que el número de identificación del actor es par)

# Los hablantes masculinos y femeninos deben ser entrenados por separado para obtener una buena precisión, ya que las mujeres tienen un tono más alto que los hombres. Entonces, si no etiquetamos la etiqueta de género de un audio, no se podrá detectar el enfado o miedo el hablante fuese un hombre. Simplemente se colocará en neutral.
# 
# Modelemos específicamente los 2 altavoces por separado siendo "tranquila" y "neutral" la misma categoría.

# In[5]:


dir_list = os.listdir(RAV)
dir_list.sort()

emotion = []
gender = []
path = []
for i in dir_list:
    fname = os.listdir(RAV + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append(RAV + i + '/' + f)

        
RAV_df = pd.DataFrame(emotion)
RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns = ['gender','emotion']
RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion
RAV_df['source'] = 'RAVDESS'  
RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
RAV_df.labels.value_counts()


# In[6]:


fname = RAV + 'Actor_14/03-01-06-02-02-02-14.wav'  
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

ipd.Audio(fname)


# Comorobamos que la calidad de audio es muy buena. S epuede sentir el miedo del hablante.

# **3. TESS dataset**

# Si no incluimos a las mujeres, terminaremos con una IA que tiene un sesgo contra un género y no es ética a menos que haya una buena razón. Por lo que en este conjunto de datos todos los hablantes son mujeres. Los oradores y las emociones están organizados en carpetas separadas.

# Tiene las mismas 7 emociones clave que nos interesan. Pero en vez de tener la emoción "sorpresa" tenemos "sorpresa agradable". Habría que comprobar si en los conjuntos de datos RADVESS y SAVEE, las sorpresas son desagradables.

# In[7]:


dir_list = os.listdir(TESS)
dir_list.sort()
dir_list


# In[8]:


path = []
emotion = []

for i in dir_list:
    fname = os.listdir(TESS + i)
    for f in fname:
        if i == 'OAF_angry' or i == 'YAF_angry':
            emotion.append('female_angry')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotion.append('female_disgust')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotion.append('female_fear')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotion.append('female_happy')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotion.append('female_neutral')                                
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotion.append('female_surprise')               
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotion.append('female_sad')
        else:
            emotion.append('Unknown')
        path.append(TESS + i + "/" + f)

TESS_df = pd.DataFrame(emotion, columns = ['labels'])
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)
TESS_df.labels.value_counts()


# In[9]:


fname = TESS + 'YAF_fear/YAF_dog_fear.wav' 
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


ipd.Audio(fname)


# La expresión de las emociones es muy similar a RAVDESS y, por lo tanto, servirá como un buen conjunto de datos de entrenamiento. La duración del audio también es aproximadamente la misma.

# **4. CREMA-D dataset**

# Es un conjunto de datos muy grande y tiene una buena variedad de altavoces diferentes, aparentemente sacados de películas. Los hablantes son de diferentes etnias. Esto es bueno porque implica una mejor generalización cuando transferimos el aprendizaje. Este conjunto de datos no tiene la emoción de "sorpresa", pero podemos usar el resto.
# 
# 
# Los hablantes y las emociones, están etiquetados en el nombre del archivo de audio. Nos falta es el género, que se guarda como un archivo csv separado que mapea a los actores.

# In[10]:


dir_list = os.listdir(CREMA)
dir_list.sort()
print(dir_list[0:10])


# In[11]:


gender = []
emotion = []
path = []
female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

for i in dir_list: 
    part = i.split('_')
    if int(part[0]) in female:
        temp = 'female'
    else:
        temp = 'male'
    gender.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotion.append('male_sad')
    elif part[2] == 'ANG' and temp == 'male':
        emotion.append('male_angry')
    elif part[2] == 'DIS' and temp == 'male':
        emotion.append('male_disgust')
    elif part[2] == 'FEA' and temp == 'male':
        emotion.append('male_fear')
    elif part[2] == 'HAP' and temp == 'male':
        emotion.append('male_happy')
    elif part[2] == 'NEU' and temp == 'male':
        emotion.append('male_neutral')
    elif part[2] == 'SAD' and temp == 'female':
        emotion.append('female_sad')
    elif part[2] == 'ANG' and temp == 'female':
        emotion.append('female_angry')
    elif part[2] == 'DIS' and temp == 'female':
        emotion.append('female_disgust')
    elif part[2] == 'FEA' and temp == 'female':
        emotion.append('female_fear')
    elif part[2] == 'HAP' and temp == 'female':
        emotion.append('female_happy')
    elif part[2] == 'NEU' and temp == 'female':
        emotion.append('female_neutral')
    else:
        emotion.append('Unknown')
    path.append(CREMA + i)
    
CREMA_df = pd.DataFrame(emotion, columns = ['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
CREMA_df.labels.value_counts()


# In[12]:


fname = CREMA + '1012_IEO_HAP_HI.wav'  
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
 
ipd.Audio(fname)


# El audio tiene un poco de eco. No es tan claro como lo que hemos visto en los otros conjuntos de datos. Suena muy neutral en vez de feliz pero podría deberse a la calidad del audio. Comprobamos otro:

# In[13]:


fname = CREMA + '1012_IEO_FEA_HI.wav'  
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

ipd.Audio(fname)


# Aquí si que se trasmite miedo. Este conjunto de datos CREMA-D es que es muy variado en su calidad. Algunos audios son claros y nítidos y otras son  apagados o con eco. También hay mucho silencio. Por lo que es una versión que quizá introdue un poco de ruido pero como los datos siguen siendo de buena calidad, los usaremos. 

# En resumidas cuentas los 4 conjuntos de datos son buenos. Necesitamos combinarlos todos para evitar el sobreajuste y que funcione bien en un nuevo conjunto de datos no visto.

# Combinamos todos los metadatos en uno solo:

# In[14]:


df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)
print(df.labels.value_counts())
df.head()
df.to_csv("Data_path.csv",index=False)


#  

# ## **Extracción de caracterísicas:**

# En términos generales, hay dos categorías de características:
# 
# - **Características del dominio del tiempo**: son más simples de extraer y comprender (energía de la señal, la tasa de cruce, amplitud, energía, etc).
# 
# 
# - **Funciones basadas en frecuencia**: se obtienen convirtiendo la señal basada en el tiempo en el dominio de la frecuencia. Son más difíciles de comprender, proporciona información adicional que puede ser realmente útil (tono, ritmos, melodía, etc).

# Usaremos MFCC porque es la mejor característica para este problema. Más adelante, durante la fase de mejora de la precisión, podemos ampliar nuestro conjunto de funciones para incluir Mel-Spectogram, Chroma, HPSS, etc., y no solo un medio simple.

# **1. MFCC**

# El coeficiente cepstral de frecuencia Mel es una buena "representación" del tracto vocal que produce el sonido. La aplicación de aprendizaje automático más común trata al MFCC en sí mismo como una 'imagen' y se convierte en una función ofreciendo una buena predicción.
# 
# El beneficio de tratarlo como una imagen es que proporciona más información y le da a uno la capacidad de aprovechar el aprendizaje por transferencia.

# In[15]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import os
import IPython.display as ipd


# **2. Deepdive**

# Podemos seleccionar algunos ejemplos y visualizar el MFCC. Seleccionamos 2 emociones diferentes y 2 géneros diferentes para ver si la calidad de los datos (audios) es buena:

# In[19]:


# Fuente RAVDESS
# Género Femenino
# Emoción Enfado

path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# Onda de sonido
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# In[20]:


# Fuente RAVDESS
# Género Masculino
# Emoción Enfado

path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# Onda de sonido
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# In[21]:


# Fuente RAVDESS
# Género Femenino
# Emoción Felicidad

path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_12/03-01-03-01-02-01-12.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# Onda de sonido
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# In[22]:


# Fuente RAVDESS
# Género Masculino
# Emoción Felicidad

path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_11/03-01-03-01-02-02-11.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# Onda de sonido
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# **3. Características estadísticas**

# Hemos visto la  salida MFCC para cada archivo, y es un formato de matriz 2D. En este apartado se va a calcular la media de cada banda a lo largo del tiempo para ver si la primera banda en la parte inferior es la banda más distintiva sobre las otras bandas. 
# 
# Dado que la ventana de tiempo es corta, los cambios observados con el tiempo no varían mucho. Vamos a comparar a la mujer y el hombre enfadados con la misma oración pronunciada.

# In[23]:


# Fuente RAVDESS
# Género Femenino
# Emoción Enfado
path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(female))

# Fuente RAVDESS
# Género Masculino
# Emoción Enfado
path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
male = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
male = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(male))

# Ondas de sonido
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
plt.plot(female, label='female')
plt.plot(male, label='male')
plt.legend()


# Para la misma oración que se pronuncia, hay una clara diferencia distintiva entre hombres y mujeres ya que las mujeres tienden a tener un tono más alto.

# In[24]:


# Fuente RAVDESS
# Género Femenino
# Emoción Felicidad
path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_12/03-01-03-01-02-01-12.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(female))

# Fuente RAVDESS
# Género Masculino
# Emoción Felicidad
path = "C:/Users/María/MASTER/Datos No Estruct/Sonido/datos/RAVDESS/audio_speech_actors_01-24/Actor_11/03-01-03-01-02-02-11.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
male = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
male = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(male))

# Ondas de sonido
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
plt.plot(female, label='female')
plt.plot(male, label='male')
plt.legend()


# Concluímos que el uso de MFCC es una buena característica para diferenciar el género y las emociones.

#  

# ## **Modelado de datos:**

# En este apartado crearemos el clasificador de emociones.

# In[2]:


import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import json
import seaborn as sns
import pickle


# **1. Preparación y procesamiento de datos**

# Ejecutar un bucle sobre el archivo de los metadatos para leer todos los archivos de audio distribuidos en los 4 directorios. Hay que poner la ruta a los archivos de audio sin procesar para el entrenamiento.

# In[3]:


ref = pd.read_csv("./datos/Data_path.csv")
ref.head()


# Para optimizar  la memoria, vamos a leer cada archivo de audio, extraer su media en todas las bandas de MFCC por tiempo y  mantener las características extraídas, eliminando todos los datos del archivo de audio.

# In[6]:


df = pd.DataFrame(columns=['feature'])

counter=0
for index,path in enumerate(ref.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast' ,duration=2.5 ,sr=44100,offset=0.5)
    sample_rate = np.array(sample_rate)
    
    # Media
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    df.loc[counter] = [mfccs]
    counter=counter+1   


# In[7]:


print(len(df))
df.head()


# **2. Procesamiento de datos**

# In[8]:


# Extraemos las medias de las bandas
df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)
df[:5]


# In[9]:


# Reemplazamos los NA con 0
df=df.fillna(0)
print(df.shape)
df[:5]


# Dividimos el conjunto de datos en entrenamiento y test

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1)
                                                    , df.labels, test_size=0.25, shuffle=True, random_state=42 )

X_train[150:160]


# Como estamos mezclando algunas fuentes de datos diferentes, hay que normalizar los datos para mejorar la precisión y el proceso de entrenamiento. 

# In[11]:


mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

X_train[150:160]


# Necesitaremos convertir el formato de datos a una matriz numérica, porque estamos usando keras CNN 2D.

# In[12]:


# Formateamos los conjutnos de datos
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# One hot Encoding para dicotomicar variables
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

print(X_train.shape)
print(lb.classes_)

filename = 'labels'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()


# Ahora que estamos usando una CNN, necesitamos especificar la tercera dimensión, que para nosotros es 1 porque estamos haciendo una CNN 1D y no una CNN 2D. Si usamos los datos de MFCC en su totalidad lo podemos convertir en una CNN 2D.

# In[13]:


X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_train.shape


# **2. Modelado**

# In[28]:


model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(14))
model.add(Activation('softmax'))


# In[35]:


from tensorflow import keras
from tensorflow.keras import layers

opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
model.summary()


# In[37]:


model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model_history=model.fit(X_train, y_train, batch_size=16, epochs=60, validation_data=(X_test, y_test))


# In[40]:


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Función de pérdida')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# La función de pérdida comienza a estabilizarse en 50 épocas pero lo mantendremos en 60.

# **3. Serialización del modelo**

# Guardamos el modelo para su reutilización:

# In[41]:


#  Gaurdamos la arquitectura del modelo y los pesos
model_name = 'Emotion_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Guardamos el modelo y los pesos en %s ' % model_path)

# Guardamos en disco
model_json = model.to_json()
with open("model_json.json", "w") as json_file:
    json_file.write(model_json)


# **4. Validación del modelo**

# Para validar el modelo predecimos las emociones con los datos de la prueba. Cargamos el modelo serializado anterior sin tener que volverlo a entrenar. En la última época 60, tenemos una precisión del 44.03%

# In[44]:


# Cargamos el JSON
json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargamos los pesos
loaded_model.load_weights("saved_models/Emotion_Model.h5")
print("Cargamos el modelo guardado con una precisión del")
 
# optimizador Keras
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[45]:


preds= loaded_model.predict(X_test,batch_size=16, verbose=1)
preds= preds.argmax(axis=1)
preds


# In[46]:


# Predicciones
preds = preds.astype(int).flatten()
preds = (lb.inverse_transform((preds)))
preds = pd.DataFrame({'predictedvalues': preds})

# Etiquetas
actual=y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'actualvalues': actual})

# Salida
finaldf = actual.join(preds)
finaldf[170:180]


# In[47]:


# Guardamos las predicciones en disco
finaldf.to_csv('Predictions.csv', index=False)
finaldf.groupby('predictedvalues').count()


# Para saber como de lo hemos hecho estudiamos el prcentaje de registros donde Real = Predicho.

# In[48]:


# Representaciónde la matriz de cnfusión mediante un mapa de calor
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names, )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Recodificación de género
def gender(row):
    if row == 'female_disgust' or 'female_fear' or 'female_happy' or 'female_sad' or 'female_surprise' or 'female_neutral':
        return 'female'
    elif row == 'male_angry' or 'male_fear' or 'male_happy' or 'male_sad' or 'male_surprise' or 'male_neutral' or 'male_disgust':
        return 'male'


# Exactitud de la emoción por género:

# In[49]:


# Predicciones
finaldf = pd.read_csv("Predictions.csv")
classes = finaldf.actualvalues.unique()
classes.sort()    

# Matriz de confusión
c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
print_confusion_matrix(c, class_names = classes)


# In[50]:


# Fichero de clasificación
classes = finaldf.actualvalues.unique()
classes.sort()    
print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))


# La precisión absoluta para el género por emociones es del 44% lo cual está basante bien. Como la clasificación de género es más precisa, agrupamos y medimos la precisión de nuevo:

# In[51]:


modidf = finaldf
modidf['actualvalues'] = finaldf.actualvalues.replace({'female_angry':'female'
                                       , 'female_disgust':'female'
                                       , 'female_fear':'female'
                                       , 'female_happy':'female'
                                       , 'female_sad':'female'
                                       , 'female_surprise':'female'
                                       , 'female_neutral':'female'
                                       , 'male_angry':'male'
                                       , 'male_fear':'male'
                                       , 'male_happy':'male'
                                       , 'male_sad':'male'
                                       , 'male_surprise':'male'
                                       , 'male_neutral':'male'
                                       , 'male_disgust':'male'
                                      })

modidf['predictedvalues'] = finaldf.predictedvalues.replace({'female_angry':'female'
                                       , 'female_disgust':'female'
                                       , 'female_fear':'female'
                                       , 'female_happy':'female'
                                       , 'female_sad':'female'
                                       , 'female_surprise':'female'
                                       , 'female_neutral':'female'
                                       , 'male_angry':'male'
                                       , 'male_fear':'male'
                                       , 'male_happy':'male'
                                       , 'male_sad':'male'
                                       , 'male_surprise':'male'
                                       , 'male_neutral':'male'
                                       , 'male_disgust':'male'
                                      })

classes = modidf.actualvalues.unique()  
classes.sort() 

# Matriz de confusión
c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
print_confusion_matrix(c, class_names = classes)


# In[52]:


# Informe de clasificación
classes = modidf.actualvalues.unique()
classes.sort()    
print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))


# Solo con el género obtenemos una precisión del 78%. El modelo es muy preciso para detectar voces femeninas. Sin embargo, las voces masculinas tienden a ser más duras y comete más errores al pensar que es femenina.

# Ahora ignoraremos el género y simplemente agruparemos por las 7 emociones centrales para estudiar la exactitud de la emoción:

# In[53]:


modidf = pd.read_csv("Predictions.csv")
modidf['actualvalues'] = modidf.actualvalues.replace({'female_angry':'angry'
                                       , 'female_disgust':'disgust'
                                       , 'female_fear':'fear'
                                       , 'female_happy':'happy'
                                       , 'female_sad':'sad'
                                       , 'female_surprise':'surprise'
                                       , 'female_neutral':'neutral'
                                       , 'male_angry':'angry'
                                       , 'male_fear':'fear'
                                       , 'male_happy':'happy'
                                       , 'male_sad':'sad'
                                       , 'male_surprise':'surprise'
                                       , 'male_neutral':'neutral'
                                       , 'male_disgust':'disgust'
                                      })

modidf['predictedvalues'] = modidf.predictedvalues.replace({'female_angry':'angry'
                                       , 'female_disgust':'disgust'
                                       , 'female_fear':'fear'
                                       , 'female_happy':'happy'
                                       , 'female_sad':'sad'
                                       , 'female_surprise':'surprise'
                                       , 'female_neutral':'neutral'
                                       , 'male_angry':'angry'
                                       , 'male_fear':'fear'
                                       , 'male_happy':'happy'
                                       , 'male_sad':'sad'
                                       , 'male_surprise':'surprise'
                                       , 'male_neutral':'neutral'
                                       , 'male_disgust':'disgust'
                                      })

classes = modidf.actualvalues.unique() 
classes.sort() 

# Matriz de confusión
c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
print_confusion_matrix(c, class_names = classes)


# In[54]:


# Informe de clasificación
classes = modidf.actualvalues.unique()
classes.sort()    
print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))


# Se consigue un 50% lo cual no está mal. La precisión de 'Sorpresa' y 'Enfadado' es bastante buena.

# La separación de género es curcial para clasificar con precisión las emociones. Las mujeres tienden a expresar emociones de una manera más obvia mientras que los hombres tienden a ser más neutrales (un hombre feliz y enfadado se confunde mucho). Probablemente esta sea la razón por la que vemos que la tasa de error entre los hombres es realmente alta.

#  

# ## **Aplicar a nuevos datos de audio:**

# Para llevar más lejos este estudio se podría aplicar un conjunto de datos de audio nuevo para ver si el clasificador de audio es capaz de generalizar las emociones realmente y no aprenderse las circustancias. Si fuese necesario aplicaríamos Data Augmentation.
