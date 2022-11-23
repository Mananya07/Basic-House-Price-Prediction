#!/usr/bin/env python
# coding: utf-8

# In[1]:


from distutils.log import debug
import numpy as np
from flask import Flask, render_template, request
import pickle


# In[2]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = np.round(prediction[0],2)
    return render_template('index.html', prediction_text='Price is :{}'.format(output))


# In[ ]:
if __name__ == "main":
    app.run(debug=True)



# %%
