import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp

st.title("歷史迷因濾鏡專題 📸")

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

# --- MediaPipe 輕量化初始化 ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self):
        self.role_mode = "愛因斯坦"
        # 關鍵優化 1：refine_landmarks 改為 False（不偵測眼球微細孔，省下大量運算）
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 關鍵優化 2：限制只偵測一隻手，降低負擔
        self.hands = mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 轉成 RGB 供 MediaPipe 處理
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_img)
        
        # 只有在特定模式才啟動手部偵測，平常關閉省效能
        hand_results = None
        if self.role_mode in ["孔子", "秦始皇"]:
            hand_results = self.hands.process(rgb_img)
        
        # --- 輕量化愛因斯坦邏輯 ---
        if self.role_mode == "愛因斯坦" and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 抓取上下唇內側關鍵點（用輕量化的索引點）
            # 在 refine_landmarks=False 時，上唇 13, 下唇 14 依然適用
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            face_height = abs(forehead.y - chin.y) * h
            
            # 只要偵測到張嘴，畫面就變黑白
            if lip_dist > (face_height * 0.15):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # 在畫面上打個字提示吐舌頭成功，先代替貼頭髮測試
                cv2.putText(img, "Einstein Mode ACTIVE!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # --- 輕量化秦始皇邏輯 ---
        elif self.role_mode == "秦始皇" and hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # 比讚判斷：大拇指尖端高於食指尖端
            if thumb_tip.y < index_tip.y:
                cv2.putText(img, "Qin Shihuang Mode ACTIVE!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

if st.session_state.camera_on:
    ctx = webrtc_streamer(
        key="meme-filter", 
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    if ctx.video_processor:
        ctx.video_processor.role_mode = role
else:
    st.warning("⚠️ 鏡頭已關閉。")
