import pandas as pd 
import streamlit as st 
import joblib 

st.set_page_config(
    page_title="ChurnWise",
    page_icon="📊",
    layout="wide"
)


st.title('🎯 CUSTOMER CHURN PREDICTION ')
st.markdown('### TELECOM CUSTOMER INTELLIGENCE')
st.write('🧠 PREDICTING CUSTOMER CHURN USING MACHINE LEARNING')
col1 ,col2,col3 =st.columns(3)
with col1:
    with st.container(border=True):
        st.subheader('👤 PERSONAL Customer Info')
        gender = st.selectbox('GENDER', ['Male','Female'])
        seniorcitizen = st.selectbox('SENIOR CITIZEN',[1,0])
        partner = st.selectbox('PARTNER',['Yes','No'])
        dependents =st.selectbox('DEPENDENT',['No','Yes'])


with col2:
    with st.container(border=True):
        st.subheader('📱 PLAN SERVICES')
        phoneservice=st.selectbox('PHONE SERVICE',['Yes','No'])
        multiplelines =st.selectbox('LINES',['No phone service', 'No', 'Yes'])
        internetservice = st.selectbox('INTERNET SERVICE',['DSL', 'Fiber optic', 'No'])
        onlinesecurity = st.selectbox('SECURITY',['No', 'Yes', 'No internet service'])
        onlinebackup =st.selectbox('BACKUP',['Yes', 'No', 'No internet service'])
        deviceprotection =st.selectbox('Protection',['No', 'Yes', 'No internet service'])
        techsupport =st.selectbox('TECH SUPPORT',['No', 'Yes', 'No internet service'])
        streamingtv = st.selectbox('STREAMING TV',['No', 'Yes', 'No internet service'])
        streamingmovies =st.selectbox('STREAMING MOVIES',['No', 'Yes', 'No internet service'])


with col3:
    with st.container(border = True):
        st.subheader('PLAN INFORMATION')
        tenure = st.slider('TENURE',min_value=0,max_value=72)
        contract = st.selectbox('CONTRACT TYPE',['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox('BILLING TYPE',['Yes', 'No'])
        paymentmethod = st.selectbox('PAYMENT TYPE',[         'Electronic check',              'Mailed check',
             'Bank transfer (automatic)',   'Credit card (automatic)'])
        monthlycharges = st.number_input('MONTHLY BILL',min_value=0)
        totalcharges =st.number_input('Total Charges',min_value=0)


model = joblib.load("telecom_churn_model.pkl")
if "probability" not in st.session_state:
    st.session_state.probability = None


input_data= pd.DataFrame({
'gender':[gender],
'seniorcitizen':[seniorcitizen],
'partner':[partner],
'dependents':[dependents],
'tenure':[tenure],
'phoneservice':[phoneservice],
'multiplelines':[multiplelines],
'internetservice':[internetservice],
'onlinesecurity':[onlinesecurity],
'onlinebackup':[onlinebackup],
'deviceprotection':[deviceprotection],
'techsupport':[techsupport],
'streamingtv':[streamingtv],
'streamingmovies':[streamingmovies],
'contract':[contract],
'paperlessbilling':[paperlessbilling],
'paymentmethod':[paymentmethod],
'monthlycharges':[monthlycharges],
'totalcharges':[totalcharges]
})

st.divider()

left ,right =st.columns([2,1],gap='large')
with left:
    with st.container(border=True):
        st.subheader('🚀 MODEL IS READY TO GIVE PREDICTION')

        if st.button('🤖PREDICT'):
            prediction = model.predict(input_data)[0]
            st.session_state.probability = model.predict_proba(input_data)[0][1]
            
            if prediction == 1:
                    st.error('🚨 CUSTOMER WILL LIKELY TO CHURN')
                    cola ,colb = st.columns(2)
                    with cola:
                        with st.container(border=True):
                            st.subheader('RISK CUSTOMER')
                            st.markdown('## HIGH RISK')
                            st.write('BASED OF PREDICTED CHURN PROBABILITY')


                    with colb:
                         with st.container(border=True):
                               st.subheader("🤝 Recommended Action")
                               st.markdown("### Retention Offer")
                               st.write("Review contract and service concerns")


                    with st.container(border=True):
                        st.subheader("💡 Customer Retention Insights")
                        st.write("✅ Review contract type and tenure")
                        st.write("✅ Check billing and payment experience")
                        st.write("✅ Offer relevant support or retention options")




            else:
                    st.success('🛡️ CUSTOMER WILL LIKELY TO STAY')
            
    
        
                
               
        


        
        
        




with right:
    with st.container(border=True):

        st.subheader("📊 GENERAL INFORMATION")
        probability = st.session_state.probability


        if probability is not None:
         st.metric("🎯 CHURN PROBABILITY", f"{probability:.2%}")
         st.progress(float(probability))
        else:
          st.metric("🎯 CHURN PROBABILITY", "—")
        
        
        st.metric("👥 Total Customers", "7,043")
        st.metric("📉 Overall Churn Rate", "26.54%")
        st.metric("🚨 Churned Customers", "1,869")
        
        
        

   







