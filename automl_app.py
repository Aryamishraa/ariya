import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import hashlib
import time

# Configure page
st.set_page_config(
    page_title="AutoML Streamlit Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'users' not in st.session_state:
    # Simple user database (in production, use proper database)
    st.session_state.users = {
        "admin": hashlib.sha256("password123".encode()).hexdigest(),
        "user1": hashlib.sha256("demo123".encode()).hexdigest(),
        "test": hashlib.sha256("test123".encode()).hexdigest()
    }

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    if username in st.session_state.users:
        return st.session_state.users[username] == hash_password(password)
    return False

def register_user(username, password):
    if username not in st.session_state.users:
        st.session_state.users[username] = hash_password(password)
        return True
    return False

def login_page():
    st.title("🤖 AutoML Streamlit Interface")
    st.markdown("### Sign in to access AutoML workflows")
    
    tab1, tab2 = st.tabs(["Sign In", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Sign In", use_container_width=True)
            
            if login_btn:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.info("Demo accounts: admin/password123, user1/demo123, test/test123")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_btn = st.form_submit_button("Register", use_container_width=True)
            
            if register_btn:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif register_user(new_username, new_password):
                    st.success("Registration successful! Please sign in.")
                else:
                    st.error("Username already exists")

def preprocess_data(df, target_column, problem_type):
    """Preprocess data for AutoML"""
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill missing values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0] if not df[categorical_columns].mode().empty else 'Unknown')
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Encode target if it's categorical (for classification)
    target_encoder = None
    if problem_type == 'Classification' and df[target_column].dtype == 'object':
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column])
    
    return df, label_encoders, target_encoder

def train_models(X_train, X_test, y_train, y_test, problem_type):
    """Train multiple models and return results"""
    results = {}
    
    if problem_type == 'Classification':
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42)
        }
        
        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'report': report
                }
    
    else:  # Regression
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
        
        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred
                }
    
    return results

def main_app():
    # Header with logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 AutoML Streamlit Interface")
        st.markdown(f"Welcome, **{st.session_state.username}**!")
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ AutoML Workflow")
        workflow_step = st.selectbox(
            "Select Step:",
            ["📊 Data Upload", "🔧 Data Preprocessing", "🤖 Model Training", "📈 Results & Visualization"]
        )
    
    # Main content based on workflow step
    if workflow_step == "📊 Data Upload":
        st.header("📊 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                st.success(f"File uploaded successfully! Shape: {df.shape}")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Data types
                st.subheader("Data Types")
                st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column'}))
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        else:
            # Sample data option
            st.info("No file uploaded. You can use sample data to explore the interface.")
            if st.button("Load Sample Data (Iris Dataset)"):
                # Create sample iris-like data
                np.random.seed(42)
                n_samples = 150
                
                data = {
                    'sepal_length': np.random.normal(5.8, 0.8, n_samples),
                    'sepal_width': np.random.normal(3.0, 0.4, n_samples),
                    'petal_length': np.random.normal(3.8, 1.8, n_samples),
                    'petal_width': np.random.normal(1.2, 0.8, n_samples),
                    'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
                }
                
                df = pd.DataFrame(data)
                st.session_state.df = df
                st.success("Sample data loaded!")
                st.rerun()
    
    elif workflow_step == "🔧 Data Preprocessing":
        st.header("🔧 Data Preprocessing")
        
        if 'df' not in st.session_state:
            st.warning("Please upload data first!")
            return
        
        df = st.session_state.df
        
        # Target selection
        st.subheader("Target Variable Selection")
        target_column = st.selectbox("Select target column:", df.columns)
        
        # Problem type
        problem_type = st.selectbox(
            "Select problem type:",
            ["Classification", "Regression"]
        )
        
        st.session_state.target_column = target_column
        st.session_state.problem_type = problem_type
        
        # Feature selection
        st.subheader("Feature Selection")
        available_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select features:",
            available_features,
            default=available_features
        )
        
        st.session_state.selected_features = selected_features
        
        if selected_features:
            # Data preprocessing preview
            st.subheader("Preprocessing Preview")
            
            try:
                df_processed, _, _ = preprocess_data(df.copy(), target_column, problem_type)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data (first 5 rows):**")
                    st.dataframe(df[selected_features + [target_column]].head())
                
                with col2:
                    st.write("**Processed Data (first 5 rows):**")
                    st.dataframe(df_processed[selected_features + [target_column]].head())
                
                st.session_state.df_processed = df_processed
                st.success("Data preprocessing completed!")
                
            except Exception as e:
                st.error(f"Preprocessing error: {str(e)}")
    
    elif workflow_step == "🤖 Model Training":
        st.header("🤖 Model Training")
        
        if 'df_processed' not in st.session_state:
            st.warning("Please complete data preprocessing first!")
            return
        
        df = st.session_state.df_processed
        target_column = st.session_state.target_column
        selected_features = st.session_state.selected_features
        problem_type = st.session_state.problem_type
        
        # Train-test split configuration
        st.subheader("Train-Test Split Configuration")
        test_size = st.slider("Test size (proportion):", 0.1, 0.5, 0.2, 0.05)
        
        # Prepare data
        X = df[selected_features]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        st.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Feature scaling option
        scale_features = st.checkbox("Apply feature scaling (StandardScaler)")
        
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.subheader("Training Progress")
            
            # Train models
            results = train_models(X_train, X_test, y_train, y_test, problem_type)
            st.session_state.training_results = results
            st.session_state.test_data = (X_test, y_test)
            
            st.success("All models trained successfully!")
            
            # Quick results summary
            st.subheader("Quick Results Summary")
            
            if problem_type == 'Classification':
                summary_data = []
                for name, result in results.items():
                    summary_data.append({
                        'Model': name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'Precision': f"{result['report']['macro avg']['precision']:.4f}",
                        'Recall': f"{result['report']['macro avg']['recall']:.4f}",
                        'F1-Score': f"{result['report']['macro avg']['f1-score']:.4f}"
                    })
            else:
                summary_data = []
                for name, result in results.items():
                    summary_data.append({
                        'Model': name,
                        'MSE': f"{result['mse']:.4f}",
                        'R² Score': f"{result['r2']:.4f}",
                        'RMSE': f"{np.sqrt(result['mse']):.4f}"
                    })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    elif workflow_step == "📈 Results & Visualization":
        st.header("📈 Results & Visualization")
        
        if 'training_results' not in st.session_state:
            st.warning("Please train models first!")
            return
        
        results = st.session_state.training_results
        problem_type = st.session_state.problem_type
        X_test, y_test = st.session_state.test_data
        
        # Model selection for detailed analysis
        selected_model = st.selectbox("Select model for detailed analysis:", list(results.keys()))
        
        model_result = results[selected_model]
        
        # Create visualizations
        if problem_type == 'Classification':
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_model} - Performance Metrics")
                
                # Metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Value': [
                        model_result['accuracy'],
                        model_result['report']['macro avg']['precision'],
                        model_result['report']['macro avg']['recall'],
                        model_result['report']['macro avg']['f1-score']
                    ]
                })
                
                fig = px.bar(metrics_df, x='Metric', y='Value', 
                           title=f'{selected_model} Performance Metrics')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Model Comparison")
                
                # Compare all models
                comparison_data = []
                for name, result in results.items():
                    comparison_data.append({
                        'Model': name,
                        'Accuracy': result['accuracy']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                fig = px.bar(comparison_df, x='Model', y='Accuracy',
                           title='Model Accuracy Comparison')
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix visualization would go here
            st.subheader("Classification Report")
            report_df = pd.DataFrame(model_result['report']).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        else:  # Regression
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_model} - Predictions vs Actual")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=model_result['predictions'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, opacity=0.7)
                ))
                
                # Perfect prediction line
                min_val = min(min(y_test), min(model_result['predictions']))
                max_val = max(max(y_test), max(model_result['predictions']))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                fig.update_layout(
                    title=f'{selected_model}: Predictions vs Actual',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Model Comparison")
                
                comparison_data = []
                for name, result in results.items():
                    comparison_data.append({
                        'Model': name,
                        'R² Score': result['r2'],
                        'RMSE': np.sqrt(result['mse'])
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig = px.bar(comparison_df, x='Model', y='R² Score',
                           title='Model R² Score Comparison')
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics table
            st.subheader("Detailed Metrics")
            metrics_data = []
            for name, result in results.items():
                metrics_data.append({
                    'Model': name,
                    'MSE': f"{result['mse']:.6f}",
                    'RMSE': f"{np.sqrt(result['mse']):.6f}",
                    'R² Score': f"{result['r2']:.6f}"
                })
            
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Download results
        st.subheader("📥 Download Results")
        
        # Create results summary
        results_summary = {
            'problem_type': problem_type,
            'target_column': st.session_state.target_column,
            'selected_features': st.session_state.selected_features,
            'model_results': {}
        }
        
        for name, result in results.items():
            if problem_type == 'Classification':
                results_summary['model_results'][name] = {
                    'accuracy': result['accuracy'],
                    'classification_report': result['report']
                }
            else:
                results_summary['model_results'][name] = {
                    'mse': result['mse'],
                    'r2': result['r2'],
                    'rmse': np.sqrt(result['mse'])
                }
        
        # Convert to JSON string for download
        import json
        results_json = json.dumps(results_summary, indent=2, default=str)
        
        st.download_button(
            label="📄 Download Results (JSON)",
            data=results_json,
            file_name=f"automl_results_{int(time.time())}.json",
            mime="application/json"
        )

# Main app logic
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
 main()