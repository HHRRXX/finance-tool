import streamlit as st
import pandas as pd
from config_tickers import DOW_30_TICKER, NAS_100_TICKER
from login import make_hashes, check_hashes, create_usertable,add_userdata,login_user,view_all_users
import sqlite3 
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

# dataframe
df_30=pd.DataFrame(DOW_30_TICKER,columns=["DOW_30_TICKER Name"])
website=[]
for i in range(len(DOW_30_TICKER)):
    website.append(f"https://finance.yahoo.com/quote/{DOW_30_TICKER[i]}?p={DOW_30_TICKER[i]}&.tsrc=fin-srch")
df_30["website"] = pd.Series(website)

df_100 = pd.DataFrame(NAS_100_TICKER,columns=["NAS_100_TICKER Name"])
website = []
for i in range(len(NAS_100_TICKER)):
    website.append(f"https://finance.yahoo.com/quote/{NAS_100_TICKER[i]}?p={NAS_100_TICKER[i]}&.tsrc=fin-srch")
df_100["website"] = pd.Series(website)


st.title('Finance Tool on Deep Reinforcement Learning Multiple Stock Trading')
st.markdown('Raw data source is **https://finance.yahoo.com/quotes/API,Documentation/view/v1/**.')
st.caption('User Instruction: You can choose two type in DOW_30_TICKER or NAS_100_TICKER to start your trade on the left navigation bar. And just Input your Trade Start Date and Trade End Date, at the same time you could modify initial asset. You get the result of your everyday action, value and finance report.')

menu = ["Home","Login","Register"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Home":
    create_usertable()
    st.subheader("Home")

elif choice == "Register":
    st.sidebar.subheader("Input your name and password")
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button('Submit'):
        add_userdata(username=username,password=password)
        st.sidebar.write("You have registered successfully!")

elif choice == "Login":
    st.sidebar.subheader("Login Section: Please Input your username and password")
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.checkbox("Login"):
        # if password == '12345':
        #create_usertable()
        hashed_pswd = make_hashes(password)
        result = login_user(username,check_hashes(password,hashed_pswd))
        if result:
            st.sidebar.success("Logged In as {}".format(username))
            col1, col2 = st.columns(2)
            with col1:
                st.header("Dow 30 stocks")
                st.write(df_30)
            with col2:
                st.header("Nasdaq 100 stocks")
                st.write(df_100)
            task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
            if task == "Add Post":
                st.subheader("Add Your Post")
                st.text_input("Your question:")
            elif task == "Analytics":
                st.subheader("Analytics")
            elif task == "Profiles":
                st.subheader("User Profiles")
                user_result = view_all_users()
                clean_db = pd.DataFrame(user_result, columns=["Username","Password"])
                st.dataframe(clean_db)
        else:
            st.sidebar.warning("Incorrect Username/Password")

st.caption("Created by Yang Wenkai, Huang Runxing and Chen Haoyang")