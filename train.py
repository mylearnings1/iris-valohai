import uuid
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier
import joblib
import valohai
    
def main():
    
    valohai.prepare(
        step = 'Train-Model',
        image = 'valohai/notebook:sklearn-0.24.2',
        default_inputs={
            'dataset': 'datum://01806ecf-1182-ee2a-1c45-4484c447375d'
        },
    )
    
    iris=valohai.inputs('dataset').path()
    iris=pd.read_csv(iris,index_col=0)
    X=iris.drop(columns=['Species'],axis=1)
    y=iris['Species']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    
    #X_train = pd.DataFrame(X_train)
    #X_test=pd.DataFrame(X_test)
    #y_train = pd.DataFrame(y_train)
    #y_test=pd.DataFrame(y_test)
    
    model=DecisionTreeClassifier()
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    model_accuracy=metrics.accuracy_score(prediction,y_test)
    print('The accuracy of the Decision Tree is',model_accuracy)
    
    cm= confusion_matrix(y_test,prediction)
    plt.rcParams['figure.figsize']=(5,5)
    sns.set(style='dark',font_scale=1.4)
    sns.heatmap(cm,annot=True,annot_kws={"size":15})
    
    suffix=uuid.uuid4()
    save_path=valohai.outputs().path('confusion_matrix.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    model_path=valohai.outputs().path('model_dt.jbl')
    joblib.dump(moodel,open(model_path,'wb'))
    
if __name__=='__main__':
    main()
    
