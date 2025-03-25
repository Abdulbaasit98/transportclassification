# app.py

from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px 
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.title('Transportni klasifikatsiya qilish')
#rasm yuklash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg', 'svg', 'jfif', 'gif'])
if file: 
    st.image(file)
#PIL image
    img = PILImage.create(file)
#model
    model = load_learner('transport_model.pkl')

#prediction
    pred, pred_id, probs = model.predict(img)
    st.success(pred)
    st.info(f'Ehtimollik:{probs[pred_id]*100:.1f}')

    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)