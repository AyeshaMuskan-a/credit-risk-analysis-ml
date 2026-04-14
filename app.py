# ==========================================
# IMPORTS
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Risk System", layout="wide")

# ==========================================
# CSS (FULL BLACK + WHITE TEXT)
# ==========================================
st.markdown("""
<style>
.stApp { background-color: #000000; color: white; }

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* HEADER CENTER */
.title { text-align: center; }

/* BUTTON */
.stButton>button {
    background-color: #6C5CE7;
    color: white;
    border-radius: 10px;
}

/* RESULT BOX */
.result-low {
    background-color: #1abc9c;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}
.result-high {
    background-color: #e74c3c;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}

/* DROPDOWN */
div[data-baseweb="select"] > div {
    background-color: #111111 !important;
    color: white !important;
}
div[data-baseweb="select"] span {
    color: white !important;
}

/* TOP BAR */
header[data-testid="stHeader"] {
    background-color: #000000 !important;
}
header[data-testid="stHeader"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")

    if 'customer_id' in df.columns:
        df.drop('customer_id', axis=1, inplace=True)

    # CLEAN TEXT (VERY IMPORTANT)
    df['gender'] = df['gender'].str.strip().str.title()
    df['employment_status'] = df['employment_status'].str.strip().str.title()

    # SAME MAP AS TRAINING
    gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
    employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}

    df['gender'] = df['gender'].map(gender_map)
    df['employment_status'] = df['employment_status'].map(employment_map)

    df.dropna(inplace=True)

    return df

df = load_data()
# ==========================================
# MODEL
# ==========================================
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
model.fit(X_scaled, y)

# ==========================================
# HEADER
# ==========================================
st.markdown("<h1 class='title'>💳 Credit Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='title'>Analyze customer financial data and predict credit risk using Machine Learning</p>", unsafe_allow_html=True)


# ==========================================
# INPUT
# ==========================================
st.markdown("## 🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox("Age", sorted(df['age'].unique()))
    income = st.selectbox("Annual Income", sorted(df['annual_income'].unique()))

with col2:
    credit_score = st.selectbox("Credit Score", sorted(df['credit_score'].unique()))
    loan_amount = st.selectbox("Loan Amount", sorted(df['loan_application_amount'].unique()))

with col3:
    late_payment = st.selectbox("Late Payments", sorted(df['late_payment_count'].unique()))
    debt = st.selectbox("Outstanding Debt", sorted(df['total_outstanding_debt'].unique()))

gender = st.selectbox("Gender", ["Male","Female", "Other"])
employment = st.selectbox("Employment Status", ["Employed","Self-Employed","Unemployed"])

# ==========================================
# FILTER DATA (DYNAMIC)
# ==========================================
filtered_df = df.copy()

# Use wider, more flexible filtering
filtered_df = filtered_df[
    (filtered_df['age'] >= age - 10) &
    (filtered_df['age'] <= age + 10)
]

filtered_df = filtered_df[
    (filtered_df['annual_income'] >= income * 0.5) &
    (filtered_df['annual_income'] <= income * 1.5)
]

filtered_df = filtered_df[
    (filtered_df['credit_score'] >= credit_score - 100) &
    (filtered_df['credit_score'] <= credit_score + 100)
]

# DO NOT RESET to full dataset immediately
if len(filtered_df) < 10:
    st.info("Showing closest available data (limited matches)")
# ==========================================
# KPI CARDS
# ==========================================
st.markdown("## 📊 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Customers", len(filtered_df))
c2.metric("💰 Avg Income", int(filtered_df['annual_income'].mean()))
c3.metric("📊 Avg Score", int(filtered_df['credit_score'].mean()))
c4.metric("⚠️ Risk %", int(filtered_df['credit_risk'].mean()*100))

high_risk = len(filtered_df[filtered_df['credit_risk']==1])
low_risk = len(filtered_df[filtered_df['credit_risk']==0])

c8, c9 = st.columns(2)
c8.metric("🟥 High Risk Count", high_risk)
c9.metric("🟩 Low Risk Count", low_risk)


# ==========================================
# PREP INPUT (FIXED)
# ==========================================
gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}

input_dict = {
    'age': age,
    'gender': gender_map[gender],
    'employment_status': employment_map[employment],
    'annual_income': income,
    'credit_score': credit_score,
    'total_outstanding_debt': debt,
    'late_payment_count': late_payment,
    'loan_application_amount': loan_amount
}

# Create input with all columns using mean values
input_df = pd.DataFrame([X.mean()])

# Overwrite with user inputs
for key in input_dict:
    if key in input_df.columns:
        input_df[key] = input_dict[key]
input_scaled = scaler.transform(input_df)

# ==========================================
# PREDICTION
# ==========================================
st.markdown("## 🤖 Prediction")

if st.button("🔍 Predict Credit Risk", key="predict_btn"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write("### 📌 Prediction Result:")

    if pred == 1:
        st.markdown(
            f"<div class='result-high'>⚠️ High Risk<br>Customer likely to default<br>Probability: {prob:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-low'>✅ Low Risk<br>Customer is safe<br>Probability: {prob:.2f}</div>",
            unsafe_allow_html=True
        )

# ==========================================
# GRAPHS (BLACK BG)
# ==========================================
st.markdown("## 📈 Insights")

g1, g2 = st.columns(2)

#graph 1-age distribution
with g1:
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sns.histplot(filtered_df['age'], color='deepskyblue', ax=ax)
    ax.axvline(age, color='red',linewidth=2)

    ax.set_title("Age Distribution", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)
    st.write("Shows distribution of customer age across dataset. Red line highlights selected user's age.")

#graph 2- risk distribution 
with g2:
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sns.countplot(x='credit_risk', data=filtered_df, palette=['lime','red'], ax=ax)
    
    ax.set_title("Risk Distribution", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)
    st.write("Displays count of low-risk and high-risk customers based on selected filters.")

g3, g4 = st.columns(2)

#graph 3- income vs score 
with g3:
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sns.scatterplot(x='annual_income', y='credit_score', data=filtered_df, ax=ax)
    ax.scatter(income, credit_score, color='yellow', s=150)

    ax.set_title("Income vs Score", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)
    st.write("Shows relationship between income and credit score for filtered customers. Yellow point represents selected user.")

#graph 4-income vs risk 
with g4:
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sns.boxplot(x='credit_risk', y='annual_income', data=filtered_df, palette=['lime','red'], ax=ax)
    ax.axhline(income, color='yellow', linewidth=2)

    ax.set_title("Income vs Risk", color='white')
    ax.tick_params(colors='white')

    st.pyplot(fig)
    st.write("Compares income distribution across risk levels. Yellow line shows selected user's income.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style='text-align:center; line-height:1.6;'>

📌 <b>Project:</b> Credit Risk Analysis using Machine Learning <br>

🎯 <b>Objective:</b> Predict customer loan default risk <br>

💡 <b>Business Impact:</b> Helps banks reduce financial losses <br>

🛠 <b>Tools Used:</b> Python | Scikit-learn | Streamlit | Power BI | SQL  

</div>
""", unsafe_allow_html=True)

