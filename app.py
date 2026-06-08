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

# --- MediaPipe 初始化設定 ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # 用來畫出關節點線條的工具

class VideoProcessor:
    def __init__(self):
        self.role_mode = "愛因斯坦"
        # 在這裡初始化偵測器，這樣網頁開起來時只會載入一次
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # MediaPipe 需要 RGB 格式
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 開始偵測臉和手
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        # 【測試用】如果偵測到臉，就把藍色網格畫在臉上
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELLATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )
                
        # 【測試用】如果偵測到手，就把紅色骨架畫在手上
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

        # 這裡保留原本的愛因斯坦變黑白測試
        if self.role_mode == "愛因斯坦":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
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
