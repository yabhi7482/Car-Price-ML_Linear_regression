
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# SIMPLE STREAMLIT STYLE
# NO HTML REQUIRED
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0f172a;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: white !important;
    }

    p, label {
        color: #cbd5e1 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #334155;
    }

    div[data-testid="stMetricValue"] {
        color: #22c55e !important;
    }

    .stButton button {
        width: 100%;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model = joblib.load("Car_price_model.pkl")

except FileNotFoundError:

    st.error(
        "❌ Car_price_model.pkl nahi mila."
    )

    st.info(
        "Car_price_model.pkl ko main.py ke same folder me rakho."
    )

    st.stop()

except Exception as e:

    st.error("❌ Model load nahi ho paya.")

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Used Car Price Prediction")

st.write(
    "### Machine Learning Based Car Price Estimation"
)

st.caption(
    "Enter the details of your car and get an estimated market price."
)

st.divider()


# ============================================================
# CAR DETAILS
# ============================================================

st.subheader("🚘 Enter Car Details")


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    brand = st.selectbox(
        "🏷️ Brand",
        [
            "Audi",
            "BMW",
            "Mercedes-Benz",
            "Mitsubishi",
            "Renault",
            "Toyota",
            "Volkswagen"
        ]
    )


with col2:

    body = st.selectbox(
        "🚙 Body Type",
        [
            "sedan",
            "hatch",
            "vagon",
            "van",
            "other"
        ]
    )


with col3:

    engine_type = st.selectbox(
        "⛽ Engine Type",
        [
            "Gas",
            "Diesel",
            "Other",
            "Petrol"
        ]
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    mileage = st.number_input(
        "🛣️ Mileage",
        min_value=0.1,
        max_value=1000.0,
        value=150.0,
        step=1.0
    )


with col2:

    engine_v = st.number_input(
        "⚙️ Engine Volume",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1
    )


with col3:

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2018,
        step=1
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    registration = st.selectbox(
        "📋 Registration",
        [
            "yes",
            "no"
        ]
    )


with col2:

    model_name = st.text_input(
        "🚘 Car Model",
        placeholder="Example: Camry, Golf, A6"
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Car Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # VALIDATE MODEL NAME
        # ====================================================

        if model_name.strip() == "":

            st.warning(
                "⚠️ Please enter the car model."
            )

            st.stop()


        # ====================================================
        # MODEL FREQUENCY
        # ====================================================
        #
        # IMPORTANT:
        # Agar training ke time Model ko frequency encoding
        # kiya tha, exact frequency dictionary ideally save
        # honi chahiye.
        #
        # Abhi unknown model ke liye 0.001 use kar rahe hain.
        #
        # ====================================================

        model_frequency = 0.001


        # ====================================================
        # MILEAGE TRANSFORMATION
        # ====================================================

        mileage_value = float(mileage)

        if mileage_value <= 0:

            mileage_value = 0.01

        mileage_value = np.log(mileage_value)


        # ====================================================
        # YEAR TRANSFORMATION
        # ====================================================

        car_age = 2026 - int(year)

        if car_age < 0:

            car_age = 0

        year_value = np.log1p(car_age)


        # ====================================================
        # NUMERICAL DATAFRAME
        # ====================================================

        numerical_df = pd.DataFrame(
            {
                "Mileage": [mileage_value],
                "EngineV": [float(engine_v)],
                "Year": [year_value],
                "Model": [model_frequency]
            }
        )


        # ====================================================
        # CATEGORICAL DATAFRAME
        # ====================================================

        categorical_df = pd.DataFrame(
            {
                "Brand": [brand],
                "Body": [body],
                "Engine Type": [engine_type],
                "Registration": [registration]
            }
        )


        # ====================================================
        # ONE HOT ENCODING
        # ====================================================

        categorical_encoded = pd.get_dummies(
            categorical_df,
            columns=[
                "Brand",
                "Body",
                "Engine Type",
                "Registration"
            ],
            drop_first=True,
            dtype=int
        )


        # ====================================================
        # COMBINE ALL FEATURES
        # ====================================================

        input_df = pd.concat(
            [
                numerical_df,
                categorical_encoded
            ],
            axis=1
        )


        # ====================================================
        # MATCH MODEL TRAINING COLUMNS
        # ====================================================

        if hasattr(model, "feature_names_in_"):

            trained_columns = list(
                model.feature_names_in_
            )

            input_df = input_df.reindex(
                columns=trained_columns,
                fill_value=0
            )

        else:

            st.error(
                "❌ Model ke training columns available nahi hain."
            )

            st.stop()


        # ====================================================
        # FINAL DATA CHECK
        # ====================================================

        input_df = input_df.astype(float)


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction_log = model.predict(
            input_df
        )[0]


        # ====================================================
        # REVERSE LOG TRANSFORMATION
        # ====================================================
        #
        # Training me:
        #
        # df['Price'] = np.log(df['Price'])
        #
        # Isliye prediction ko original price me lane ke liye
        # np.exp() use karna hai.
        #
        # ====================================================

        predicted_price = np.exp(
            prediction_log
        )


        # ====================================================
        # CHECK PREDICTION
        # ====================================================

        if not np.isfinite(predicted_price):

            st.error(
                "❌ Model ne invalid price generate ki."
            )

            st.stop()


        if predicted_price < 0:

            st.error(
                "❌ Invalid negative price generated."
            )

            st.stop()


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ AI Prediction Generated Successfully!"
        )


        # ====================================================
        # PRICE RESULT
        # ====================================================

        st.subheader(
            "💰 Estimated Market Price"
        )


        st.metric(
            label="🚗 Estimated Car Price",
            value=f"₹ {predicted_price:,.0f}"
        )


        st.info(
            "The price is estimated using your trained "
            "Machine Learning model."
        )


        # ====================================================
        # CAR SUMMARY
        # ====================================================

        st.subheader(
            "📊 Car Details Summary"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🏷️ Brand",
                brand
            )


        with col2:

            st.metric(
                "🚘 Model",
                model_name
            )


        with col3:

            st.metric(
                "📅 Year",
                str(year)
            )


        with col4:

            st.metric(
                "⚙️ Engine",
                f"{engine_v} L"
            )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🛣️ Mileage",
                str(mileage)
            )


        with col2:

            st.metric(
                "🚙 Body",
                body
            )


        with col3:

            st.metric(
                "⛽ Engine Type",
                engine_type
            )


        with col4:

            st.metric(
                "📋 Registration",
                registration
            )


        # ====================================================
        # MODEL INPUT PREVIEW
        # ====================================================

        with st.expander(
            "🔍 View Processed Data"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction ke time error aaya."
        )

        st.write(
            "Error details:"
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚗 Used Car Price Prediction | "
    "Python • Pandas • NumPy • Scikit-Learn • Streamlit"
)

