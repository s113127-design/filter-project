import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2

st.title("歷史迷因濾鏡專題 📸")

# 使用 Session State 記錄鏡頭目前的開關狀態
if "camera_on" not in st.session_state:
    st.session_state.camera_on = True

role = st.selectbox("請選擇角色", ["愛因斯坦", "孔子", "秦始皇", "釋迦牟尼佛", "路易十六"])
st.write(f"當前模式：{role}")

# 製作控制按鈕
col1, col2 = st.columns(2)
with col1:
    if st.button("開啟/重設鏡頭 🟢"):
        st.session_state.camera_on = True
        st.rerun()

with col2:
    if st.button("完全關閉鏡頭 🔴"):
        st.session_state.camera_on = False
        st.rerun()

# 影像處理器 (新版寫法)
class VideoProcessor:
    def __init__(self):
        self.role_mode = "愛因斯坦"

    def recv(self, frame):
        # 接收新版傳進來的影格
        img = frame.to_ndarray(format="bgr24")
        
        # 測試：愛因斯坦變黑白
        if self.role_mode == "愛因斯坦":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        return frame.from_ndarray(img, format="bgr24")

# 根據開關狀態決定要不要顯示鏡頭
if st.session_state.camera_on:
    ctx = webrtc_streamer(
        key="meme-filter", 
        video_processor_factory=VideoProcessor,
        rtc_configuration={ # 加入這段可以大幅提升手機與各瀏覽器的連線成功率
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    
    if ctx.video_processor:
        ctx.video_processor.role_mode = role
else:
    st.warning("⚠️ 鏡頭已關閉。如需重新使用，請點選上方的「開啟/重設鏡頭」按鈕。")
