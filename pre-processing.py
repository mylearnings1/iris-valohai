import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import valohai

def main():
    
    valohai.prepare(
        step='Pre-process Dataset',
        image='python:3.9',
        default_inputs={
            'dataset': 'https://depprocureformstorage.blob.core.windows.net/iris-dataset?sp=r&st=2022-04-28T05:15:15Z&se=2022-04-28T13:15:15Z&spr=https&sv=2020-08-04&sr=c&sig=aKggMxPZ4cXq%2FU1cf9tJ6hI1gCRM8DgKiB%2F1uCSAwUE%3D'
            },
    )
    
    print('Loading data')
    iris=pd.read_csv(valohai.inputs('dataset').path())
    iris=iris.drop('Id',axis=1,inplace=True)
    
    #X=iris.drop(columns=['Species'],axis=1)
    #y=iris['Species']
   
    print('Saving preprocessed data')
    path_iris = valohai.outputs().path('iris_preprocessed.csv')
    iris.to_csv(path_iris)
    
   
if __name__=='__main__':
    main()
