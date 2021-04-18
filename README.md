# Gradio-web-app
What is Gradio?
Gradio is an open-source Python library that allows you to build a user interface for machine learning models and deploy it in a few lines of code. If you worked with Dash or Streamlit in python before, it’s similar; however, it integrates directly with notebooks and doesn’t require a separate python script.
Gradio can be installed directly through pip. Creating a Gradio interface only requires adding a couple lines of code to your project. You can choose from a variety of interface types to interface your function.

### Installation
```bash
# gradio is a Python package, so it can be installed with pip.
!pip install --quiet gradio
# or
!pip install gradio
```
### Documentation
[ click here ](https://gradio.app/docs#i_slider)

### Code
#### Run the code below as a Python script or in a Python notebook (or in a colab notebook).
```python

import gradio as gr
import joblib
from PIL import Image
#Loading Our final trained Knn model 
model= open("Knn_Classifier.pkl", "rb")
knn_clf=joblib.load(model)
class_name=['setosa','versicolor','virginica']

setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')


def model_predict(Sepal_length_in_cm,Sepal_width_in_cm,Petal_length_in_cm,Petal_Width_in_cm):
   prediction=knn_clf.predict([[Sepal_length_in_cm,Sepal_width_in_cm,Petal_length_in_cm,Petal_Width_in_cm]])[0]
   #pred_class=class_name[prediction]
   img=[setosa,versicolor,virginica]
   return img[prediction]
	
   
app = gr.Interface(
    model_predict,
    [
        gr.inputs.Slider(0,8,default=0.8,label='Sepal length (cm) '),
        gr.inputs.Slider(0, 8,default=4,label='Sepal Width (cm)'),
        gr.inputs.Slider(0, 8,default=5,label='Petal length (cm)'),
        gr.inputs.Slider(0, 8,default=3.5,label='Petal Width (cm)'),
    ],
    "image",
  examples=[[4.5,2.5,5,4.2],[0.5,0.8,5,3],[0.4,3.2,2.5,4]])

app.test_launch()

if __name__ == "__main__":
    app.launch()
    
```
### App view
![image](https://github.com/vishalbpatil1/Gradio-web-app/blob/main/app_view.png)
