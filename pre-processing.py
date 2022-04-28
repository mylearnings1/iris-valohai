import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
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
    
    X=iris.drop(columns=['Species'],axis=1)
    y=iris['Species']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    
    X_train = pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test=pd.DataFrame(y_test)
    
    print('Saving train test split data')
    path_X_train = valohai.outputs().path('X_train.csv')
    X_train.to_csv(path_X_train)
    path_X_test = valohai.outputs().path('X_test.csv')
    X_test.to_csv(path_X_test)
    path_y_train = valohai.outputs().path('y_train.csv')
    y_train.to_csv(path_y_train)
    path_y_test = valohai.outputs().path('y_test.csv')
    y_test.to_csv(path_y_test)
    
    
if __name__=='__main__':
    main()
