import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    return df

df = load_data()

# Sidebar - Collects user input features into dataframe
st.sidebar.header('User Input Parameters')

def user_input_features():
    height = st.sidebar.slider('Height', int(df['Height'].min()), int(df['Height'].max()), int(df['Height'].mean()))
    weight = st.sidebar.slider('Weight', int(df['Weight'].min()), int(df['Weight'].max()), int(df['Weight'].mean()))
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    data = {'Height': height,
            'Weight': weight,
            'Gender': gender}
    features = pd.DataFrame(data, index=[0])
    return features

df_user = user_input_features()

# Main panel
st.subheader('User Input parameters')
st.write(df_user)

# Prepare the dataset
X = df[['Height', 'Weight', 'Gender']]
y = df['Age']

# Encode categorical variables
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KNN model
k = 3  # Number of neighbors
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_scaled, y)

# Encode user input features
df_user['Gender'] = df_user['Gender'].map({'Male': 1, 'Female': 0})

# Make predictions
prediction = knn_model.predict(scaler.transform(df_user))

# Round down the prediction to the nearest integer
prediction = int(round(prediction[0]))

st.subheader('Predicted Age')
st.write(prediction)
