import streamlit as st
import tensorflow as tf
import numpy as np

#tensorflow model prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

#sidebar
st.sidebar.title('DashBoard')
app_mode=st.sidebar.selectbox('Select Page', ['Home','About','Disease Recognition'])

#home page
if(app_mode=='Home'):
    st.header('Rice Leaf Disease Detection system')
    image_path='home_page.jpg'
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Rice Leaf Disease Detection System! 🌿🔍
    
    Our mission is to help in identifying Rice Leaf diseases efficiently. Upload an image of a Leaf and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a Leaf with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Rice Leaf Disease Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)



#about
elif(app_mode=='About'):
    st.header('About')
    st.markdown("""
           #### Content
            1. train (14023 images)
            2. test (3101 images)
            3. validation (14023 images)


   """ )

#prediction page
elif(app_mode=='Disease Recognition'):
    st.header('Disease Recognition')
    test_image=st.file_uploader('choose an image')
    if(st.button('Show Image')):
        st.image(test_image,use_column_width=True)

#predict button
    if(st.button('Detect')):
     st.write('prediction')
     result_index=model_prediction(test_image)

    #define class
    class_name =  ['bacterial_leaf_blight',
    'brown_spot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot',
    'Neck_Blast',
    'Rice_Hispa',
    'Sheath_Blight']
    st.success('Model is Predicting  It is a {}'.format (class_name [result_index]) )
        


