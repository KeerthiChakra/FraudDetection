import os
from LLM import askLLM, askLLM2
import streamlit as st
from joblib import dump, load
from catboost import CatBoostClassifier
from dotenv import find_dotenv, load_dotenv



st.title('Financial Transaction Fraud Detection')
flag=0

# selection=st.selectbox('Select your desired model for fraud detection out of the given options',('None','CatBoost','LogisticRegression'),placeholder='Select Model')
selection='CatBoost'
if selection=='CatBoost':
    from catboost import CatBoostClassifier
    cat=CatBoostClassifier().load_model("CatModel")
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    with col1:
        amount=st.number_input("Amount: ", placeholder="Transaction amount...")
    with col3:
        sndr_ini_blc=st.number_input("Sender Initial Bank Balance: ", placeholder="Sender bank balance before transaction")
    with col4:
        sndr_fin_blc=st.number_input("Sender Final Bank Balance: ", placeholder="Sender bank balance after transaction")
    with col5:
        rec_ini_blc=st.number_input("Reciever Initial Bank Balance: ", placeholder="Reciever bank balance before transaction")
    with col6:
        rec_fin_blc=st.number_input("Reciever Final Bank Balance: ", placeholder="Reciever bank balance after transaction")
    with col2:
        type=st.selectbox("Type of transaction",('CASHIN','CASHOUT','TRANSFER','PAYMENT','DEBIT'))
    
    if st.button("Predict"):
        result=cat.predict([[amount,sndr_ini_blc,sndr_fin_blc,rec_ini_blc,rec_fin_blc,type]])
        if result==0:
            ans="Not Fraud"
            st.header("Not Fraud")
            flag=1
        else:
            ans="Fraud"
            st.header("Fraud")
            flag=1
        
        st.markdown(askLLM(amount,sndr_ini_blc,sndr_fin_blc,rec_ini_blc,rec_fin_blc,type,ans))
        
    # st.write(f'{amount}, {sndr_ini_blc}, {sndr_fin_blc}, {rec_ini_blc}, {rec_fin_blc}, {type}')
elif selection=='LogisticRegression':
    # st.warning("Page Under Construction",icon="🤦‍♂️")
    logReg = load('logReg.joblib')  
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        amount=st.number_input("Amount: ", placeholder="Transaction amount...")
    with col2:
        type=st.selectbox("Type of transaction",('CASHIN','CASHOUT','TRANSFER','PAYMENT','DEBIT'))
    with col3:
        sndr_ini_blc=st.number_input("Sender Initial Bank Balance: ", placeholder="Sender bank balance before transaction")
    with col4:
        sndr_fin_blc=st.number_input("Sender Final Bank Balance: ", placeholder="Sender bank balance after transaction")
    
    if type==[i for i in ['CASHIN','DEBIT','PAYMENT']]:
        isPayment=1
        isMovement=0
    else:
        isPayment=0
        isMovement=1
    difference=abs(sndr_fin_blc-sndr_ini_blc)
    if st.button("Predict"):
        x=logReg.predict([[amount,isPayment,isMovement,difference]])
        if x==1:
            ans="FRAUD"
            st.header("FRAUD")
        else:
            ans="NOT FRAUD"
            st.header("NOT FRAUD")
        with st.spinner("Loading..."):
            st.markdown(askLLM2(amount,sndr_ini_blc,sndr_fin_blc,type,ans))
        