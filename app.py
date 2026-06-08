import streamlit as st

st.title("歷史迷因濾鏡專題 📸")
st.write("這是我們的專題網頁，目前正在開發中！")

role = st.selectbox("請選擇角色", ["愛因斯坦", "孔子", "秦始皇", "釋迦牟尼佛", "路易十六"])
st.write(f"你選擇了：{role}")
