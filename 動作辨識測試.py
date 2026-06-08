import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("歷史迷因濾鏡 - 動作觸發測試功能 🧪")
st.caption("本程式不需圖片，單純測試 AI 是否能辨識你的表情與手勢。")

if "camera_on" not in st.session_state:
    st.session_state.camera_on = True

role = st.selectbox("請選擇要測試的角色", ["愛因斯坦", "孔子", "秦始皇", "釋迦牟尼佛", "路易十六"])
st.write(f"目前測試模式：**{role}**")

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

# --- MediaPipe 初始化 ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class TestVideoProcessor:
    def __init__(self):
        self.role_mode = "愛因斯坦"
        # 載入 AI 模型
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 進行偵測
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        triggered = False # 是否達成動作條件
        debug_info = ""   # 顯示目前的數據數值

        # ---- 1. 臉部特徵偵測 ----
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 【視覺回饋】畫出臉部基礎網格（幫助確認有沒有抓到臉）
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1)
            )

            # 測試：愛因斯坦 (嘴張大/吐舌)
            if self.role_mode == "愛因斯坦":
                top_lip = face_landmarks.landmark[13].y
                bot_lip = face_landmarks.landmark[14].y
                lip_distance = bot_lip - top_lip
                debug_info = f"Mouth Open: {lip_distance:.3f} (Target > 0.04)"
                if lip_distance > 0.04:  
                    triggered = True

            # 測試：釋迦牟尼佛 (閉眼)
            elif self.role_mode == "釋迦牟尼佛":
                # 左眼上下眼瞼距離
                p_top = face_landmarks.landmark[159].y
                p_bot = face_landmarks.landmark[145].y
                eye_distance = p_bot - p_top
                debug_info = f"Eye Open: {eye_distance:.3f} (Target < 0.015)"
                if eye_distance < 0.015:  
                    triggered = True

        # ---- 2. 手勢特徵偵測 ----
        if hand_results.multi_hand_landmarks:
            # 【視覺回饋】畫出手的骨架
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

            # 測試：孔子 (兩手合十 / 畫面上出現兩隻手)
            if self.role_mode == "孔子":
                hand_count = len(hand_results.multi_hand_landmarks)
                debug_info = f"Hands Count: {hand_count} (Target == 2)"
                if hand_count == 2:
                    triggered = True
                    
            # 測試：秦始皇 (比讚)
            elif self.role_mode == "秦始皇":
                hand_lms = hand_results.multi_hand_landmarks[0]
                # 簡單判定：大拇指尖(4)高於大拇指根部(2)，且食指尖(8)低於食指第二關節(6)
                thumb_up = hand_lms.landmark[4].y < hand_lms.landmark[2].y
                index_down = hand_lms.landmark[8].y > hand_lms.landmark[6].y
                debug_info = f"Thumb Up: {thumb_up}, Index Down: {index_down}"
                if thumb_up and index_down:
                    triggered = True

        # ---- 3. 渲染測試文字與特效 ----
        # 顯示即時偵測數值（左下角）
        if debug_info:
            cv2.putText(img, debug_info, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 判斷是否顯示 "達成條件"
        if self.role_mode == "路易十六":
            # 路易十六不需要動作，直接觸發
            triggered = True
            # 模擬一下黑框馬賽克的感覺
            cv2.rectangle(img, (int(w/4), int(h/4)), (int(3*w/4), int(3*h/4)), (0, 0, 255), 3)
            cv2.putText(img, "Louis XVI Mode (No Action Required)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if triggered:
            # 畫出綠色半透明遮罩層
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            
            # 畫出「達成條件」的大字（畫面正中央）
            # 因為 OpenCV 預設不支援中文，這裡先用英文代替以免變亂碼
            cv2.putText(img, "SUCCESS! CONDITION MET", (int(w/2) - 180, int(h/2)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Active: [{self.role_mode}]", (int(w/2) - 100, int(h/2) + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if self.role_mode != "路易十六":
                cv2.putText(img, "Waiting for action...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# --- 啟動 WebRTC 串流 ---
if st.session_state.camera_on:
    ctx = webrtc_streamer(
        key="test-meme-filter", 
        video_processor_factory=TestVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    if ctx.video_processor:
        ctx.video_processor.role_mode = role
else:
    st.warning("⚠️ 鏡頭已關閉。")
