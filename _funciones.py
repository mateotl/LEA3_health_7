
import numpy as np
import pandas as pd
from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
from os.path import join ### para unir ruta con archivo 
import cv2 ### para leer imagenes jpg
from glob import glob
import zipfile
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

def crear_diccionario(extract_to):
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(extract_to, '*.jpg'))
    }
    return imageid_path_dict

def cargar_zip(Zip1,Zip2,ExtraerDonde):
    zip_path1 = Zip1 #'/content/drive/MyDrive/cod/LEA3_health_7/data/HAM10000_images_part_1.zip'
    zip_path2 = Zip2#'/content/drive/MyDrive/cod/LEA3_health_7/data/HAM10000_images_part_2.zip'
    extract_to = ExtraerDonde #'/content/ham10000_images'

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path1, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


def cargar_csv(CSV):
    csv_path = CSV #'/content/drive/MyDrive/cod/LEA3_health_7/data/HAM10000_metadata.csv'
    skin_df = pd.read_csv(csv_path)
    return skin_df

# Para preprocesar
def crear_ruta(df, imageid_path_dict, lesion_type_dict):
    df = df.copy()
    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    return df

def crear_binaria(df):
    df = df.copy()
    malignas = ['Actinic keratoses', 'Basal cell carcinoma', 'Melanoma']
    df['cell_type_idx'] = df['cell_type'].apply(lambda x: 1 if x in malignas else 0)
    return df

def reemplazar_nulos(df, columna):
    df = df.copy()
    df[columna] = df[columna].fillna(df[columna].mean())
    return df

def redimensionar_imagen(df, size):
    df = df.copy()
    df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize(size)))
    return df

# Para preprocesar datos y entrenar modelos no CNN
def procesar_tabulares(df):
    df_tab = df[['age', 'sex', 'localization']].copy()
    df_tab['age'] = MinMaxScaler().fit_transform(df_tab[['age']])
    df_tab = pd.get_dummies(df_tab, columns=['sex', 'localization'], drop_first=True)
    return df_tab

def aplanar_imagenes(df):
    X_img = np.stack(df['image'].values)                  # (n, 100, 75, 3)
    X_img_flat = X_img.reshape(X_img.shape[0], -1)        # (n, 22500)
    return X_img_flat

def combinar_datos(df):
    X_tab = procesar_tabulares(df)
    X_img_flat = aplanar_imagenes(df)
    X_final = np.concatenate([X_img_flat, X_tab.values], axis=1)
    return X_final