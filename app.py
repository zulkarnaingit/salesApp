import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="Sales Prediction App", layout="wide")

# Title and description
st.title("📈 Sales Prediction App")
st.write("""
This app predicts sales based on advertising spending across different channels.
Upload your data or use our sample dataset to get started.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Options")

# Function to load sample data
def load_sample_data():
    # Generate sample data
    np.random.seed(42)
    tv = np.random.rand(100) * 200
    radio = np.random.rand(100) * 50
    newspaper = np.random.rand(100) * 30
    sales = 3 + 0.3*tv + 0.5*radio + 0.2*newspaper + np.random.randn(100)*2
    
    data = pd.DataFrame({
        'TV': tv,
        'Radio': radio,
        'Newspaper': newspaper,
        'Sales': sales
    })
    return data

# Upload or use sample data
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    data = load_sample_data()
    st.sidebar.info("Using sample dataset. Upload a file to use your own data.")

# Show raw data
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)

# Select features and target
st.sidebar.subheader("Model Configuration")
features = st.sidebar.multiselect(
    "Select features (independent variables)",
    data.columns[:-1],  # Exclude the last column (assumed to be target)
    default=list(data.columns[:-1]))
target = st.sidebar.selectbox(
    "Select target variable (dependent variable)",
    data.columns,
    index=len(data.columns)-1)

# Train-test split ratio
test_size = st.sidebar.slider(
    "Select test set size ratio",
    0.1, 0.5, 0.2, 0.05)

# Main content
st.subheader("Data Exploration")

# Show basic statistics
if st.checkbox("Show statistics"):
    st.write(data.describe())

# Show correlation matrix
if st.checkbox("Show correlation matrix"):
    corr = data.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.pyplot(fig)

# Train the model
st.subheader("Model Training")

# Prepare data
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Model Coefficients: {model.coef_}")
st.write(f"Model Intercept: {model.intercept_:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs. Predicted Sales')
st.pyplot(fig)

# Prediction interface
st.subheader("Make Predictions")
st.write("Enter values for the selected features to predict sales:")

input_data = {}
col1, col2 = st.columns(2)
for i, feature in enumerate(features):
    if i % 2 == 0:
        with col1:
            input_data[feature] = st.number_input(f"{feature} spending", min_value=0.0, value=data[feature].mean())
    else:
        with col2:
            input_data[feature] = st.number_input(f"{feature} spending", min_value=0.0, value=data[feature].mean())

if st.button("Predict Sales"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"Predicted Sales: ${prediction[0]:.2f}")

# Save model option
if st.sidebar.checkbox("Save model"):
    import joblib
    joblib.dump(model, 'sales_prediction_model.joblib')
    st.sidebar.success("Model saved as 'sales_prediction_model.joblib'")