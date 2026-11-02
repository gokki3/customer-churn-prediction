
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. FILE PATHS AND MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "cleaned_data.csv"
MODEL_PATH = PROJECT_DIR / "model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Customer Churn Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Business Problem",
        "Data Insights",
        "Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning for Business Applications")

# --------------------------------------------------
# 4. BUSINESS PROBLEM PAGE
# --------------------------------------------------

if page == "Business Problem":

    st.title("Customer Churn Prediction System")

    st.subheader("Project Overview")

    st.write(
        "Customer churn refers to customers discontinuing "
        "their services with a telecommunications company. "
        "Predicting churn helps businesses identify customers "
        "who may leave and take preventive action."
    )

    st.subheader("Business Challenge")

    st.write(
        "Telecommunication companies operate in a competitive "
        "market where retaining existing customers is important. "
        "Losing customers can affect revenue and increase the "
        "cost of acquiring new customers."
    )

    st.subheader("Project Objective")

    st.write(
        "The objective of this project is to develop a Machine "
        "Learning model that predicts whether a telecom customer "
        "is likely to churn based on customer information, "
        "service usage and contract details."
    )

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customer Records", f"{len(df):,}")
    col2.metric("Dataset Columns", len(df.columns))
    col3.metric("Target Variable", "Churn")

    st.write("**Dataset:** IBM Telco Customer Churn")
    st.write("**Target Variable:** Churn")
    st.write("**Prediction Categories:** Yes / No")

    st.subheader("Business Significance")

    st.write(
        "The prediction results can help telecom companies "
        "identify customers who may leave, develop targeted "
        "retention campaigns, improve customer relationships "
        "and support data-driven business decisions."
    )

# --------------------------------------------------
# 5. DATA INSIGHTS PAGE
# --------------------------------------------------

elif page == "Data Insights":

    st.title("Data Insights and Exploratory Analysis")

    st.write(
        "This section explores customer characteristics "
        "and patterns associated with telecom customer churn."
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Total Features", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    # Churn Distribution

    st.subheader("1. Customer Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    plt.close(fig)

    # Contract vs Churn

    st.subheader("2. Contract Type vs Customer Churn")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn",
        ax=ax
    )
    ax.set_title("Contract Type vs Churn")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    # Tenure vs Churn

    st.subheader("3. Customer Tenure vs Churn")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure",
        ax=ax
    )
    ax.set_title("Customer Tenure vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Tenure (Months)")

    st.pyplot(fig)
    plt.close(fig)

    # Monthly Charges vs Churn

    st.subheader("4. Monthly Charges vs Churn")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        ax=ax
    )
    ax.set_title("Monthly Charges vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")

    st.pyplot(fig)
    plt.close(fig)

    # Internet Service vs Churn

    st.subheader("5. Internet Service vs Churn")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=df,
        x="InternetService",
        hue="Churn",
        ax=ax
    )
    ax.set_title("Internet Service vs Churn")
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Key Observations")

    st.write(
        "- Month-to-month contracts show a higher concentration "
        "of customers who churn."
    )

    st.write(
        "- Customers with shorter tenure appear more likely "
        "to leave the company."
    )

    st.write(
        "- Customer charges and internet service categories "
        "provide additional information for analysing churn."
    )

# --------------------------------------------------
# 6. PREDICTION PAGE
# --------------------------------------------------

elif page == "Prediction":

    st.title("Customer Churn Prediction")

    st.write(
        "Enter the customer's information below to estimate "
        "the likelihood of customer churn."
    )

    st.info(
        "Complete the customer details and click Predict Churn."
    )

    with st.form("customer_prediction_form"):

        st.subheader("Customer Information")

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)",
                0, 72, 12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=18.25,
                max_value=118.75,
                value=70.00
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=9000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        try:

            prediction = model.predict(input_data)[0]

            st.subheader("Prediction Result")

            if prediction == "Yes":

                st.error(
                    "The customer is likely to churn."
                )

                st.warning(
                    "Consider targeted customer retention "
                    "strategies for this customer."
                )

            else:

                st.success(
                    "The customer is unlikely to churn."
                )

                st.info(
                    "Continue monitoring customer satisfaction "
                    "and service engagement."
                )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                class_names = model.classes_
                churn_index = list(class_names).index("Yes")

                churn_probability = probabilities[churn_index]

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2%}"
                )

                st.progress(float(churn_probability))

        except Exception as e:

            st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("Customer Churn Prediction | MBA Business Analytics")
