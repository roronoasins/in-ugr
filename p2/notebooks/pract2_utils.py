# -*- coding: utf-8 -*-
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import pandas as pd
import seaborn
from math import floor
import matplotlib.pyplot as plt

def to_matrix(df, columns=[]):
    """Devuelve los atributos seleccionados como valores"""
    return df[columns].dropna().values

def norm(data):
    """Normaliza una serie de datos"""
    return (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

def measures_silhoutte_calinski(data, labels):
    """
    Devuelve el resultado de evaluar los clusters de data asociados con labels.
    
    Parámetros:
    
    - data vector de datos ya normalizados.
    - labels: etiquetas.
    """
    # Hacemos una muestra de sólo el 20% porque son muchos elementos
    muestra_silhoutte = 0.2 if (len(data) > 10000) else 1.0
    silhouette = silhouette_score(data, labels, metric='euclidean', sample_size=int(floor(data.shape[0]*muestra_silhoutte)))
    calinski = calinski_harabasz_score(data, labels)
    return silhouette, calinski

def print_measure(measure, value):
    """
    Muestra el valor con un número fijo de decimales
    """
    print("{}: {:.3f}".format(measure, value))

def pairplot(df, columns, labels, case, algorithm, subcase, k):
    """
    Devuelve una imagen pairplot.

    Parámetros:

    - df: dataframe
    - columns: atributos a considerar
    - labels: etiquetas
    """
    df_plot = df.loc[:,columns].dropna()
    df_plot['classif'] = labels
    pp =seaborn.pairplot(df_plot, hue='classif', palette='Paired', diag_kws={'bw': 0.3})
    file = '../results/case'+str(case)+'/'+algorithm+'/pairplots/pairplot_'+algorithm+'_case'+str(case)+'_'+subcase+'_k'+str(k)
    #pp.savefig(file+'.png')

def denorm(data, df):
    """
    Permite desnormalizar
    """
    return data*(df.max(axis=0)-df.min(axis=0))+df.min(axis=0)

def visualize_centroids(centers, data, columns, case, algorithm, subcase, k):
    """
    Visualiza los centroides.

    Parametros:

    - centers: centroides.
    - data: listado de atributos.
    - columns: nombres de los atributos.
    """
    df_centers = pd.DataFrame(centers,columns=columns)
    centers_desnormal=denorm(centers, data)
    fig, ax = plt.subplots(figsize=(10,10))
    hm = seaborn.heatmap(df_centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f', ax=ax)
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    # estas tres lineas las he añadido para evitar que se corten la linea superior e inferior del heatmap
    file = '../results/case'+str(case)+'/'+algorithm+'/heatmaps/hm_'+algorithm+'_case'+str(case)+'_'+subcase+'_k'+str(k)
    #fig.savefig(file+'.png')
    
    return hm
