import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸")
st.write("👉 做出指定表情與手勢來變身！按下 Capture 拍照！")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_resources():
    # 讀取你上傳的所有圖片檔案
    buddha_light = cv2.imread("assets/buddha holy light.png", cv2.IMREAD_UNCHANGED)
    einstein_hair = cv2.imread("assets/Einstein hair.png", cv2.IMREAD_UNCHANGED)
    einstein_tongue = cv2.imread("assets/Einstein tongue.png", cv2.IMREAD_UNCHANGED)
    kongzi_beard = cv2.imread("assets/kongzi beard.png", cv2.IMREAD_UNCHANGED)
    kongzi_cap = cv2.imread("assets/kongzi cap.png", cv2.IMREAD_UNCHANGED)
    kongzi_sleeves = cv2.imread("assets/kongzi sleeves.png", cv2.IMREAD_UNCHANGED)
    polar_bear = cv2.imread("assets/polar_bear.png", cv2.IMREAD_UNCHANGED)
    qin_cap = cv2.imread("assets/qinshihuang cap.png", cv2.IMREAD_UNCHANGED)
    tomato = cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED)
    
    return buddha_light, einstein_hair, einstein_tongue, kongzi_beard, kongzi_cap, kongzi_sleeves, polar_bear, qin_cap, tomato

# 載入貼圖
buddha_light, einstein_hair, einstein_tongue, kongzi_beard, kongzi_cap, kongzi_sleeves, polar_bear, qin_cap, tomato = load_resources()

def overlay_image(background, overlay, x, y, size=None):
    if overlay is None: return background
    bg_h, bg_w = background.shape[:2]
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0: return background
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    crop_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    crop_bg = background[y1:y2, x1:x2]
    if crop_overlay.shape[2] == 4:
        alpha = crop_overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        composite = crop_overlay[:, :, :3] * alpha + crop_bg * (1 - alpha)
        background[y1:y2, x1:x2] = composite
    else:
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
    return background

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        status_text = "Scanning... Make a gesture!"
        mode_triggered = False  # 用來記錄是否有觸發前四個偉人
        
        # 1. 臉部特徵偵測
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 取得點位
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            # 眼睛點位 (用來算閉眼)
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bot = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bot = face_landmarks.landmark[374]
            
            # 計算基準距離
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            face_height = abs(forehead.y - chin.y) * h
            left_eye_dist = abs(left_eye_top.y - left_eye_bot.y) * h
            right_eye_dist = abs(right_eye_top.y - right_eye_bot.y) * h
            
            # 💡 【愛因斯坦模式】：張開嘴巴
            if lip_dist > (face_height * 0.15):
                status_text = "ACTIVE: Einstein Mode"
                mode_triggered = True
                # 畫面轉黑白
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # P 上頭髮
                if einstein_hair is not None:
                    hair_w = int(face_height * 1.8)
                    hair_h = int(hair_w * (einstein_hair.shape[0] / einstein_hair.shape[1]))
                    img = overlay_image(img, einstein_hair, int(forehead.x * w - hair_w / 2), int(forehead.y * h - hair_h * 0.75), size=(hair_w, hair_h))
                # P 上舌頭
                if einstein_tongue is not None:
                    tongue_w = int(face_height * 0.5)
                    tongue_h = int(tongue_w * (einstein_tongue.shape[0] / einstein_tongue.shape[1]))
                    img = overlay_image(img, einstein_tongue, int(lower_lip.x * w - tongue_w / 2), int(lower_lip.y * h), size=(tongue_w, tongue_h))

            # 💡 【釋迦牟尼佛模式】：閉上眼睛 (兩眼都很瞇)
            elif left_eye_dist < (face_height * 0.03) and right_eye_dist < (face_height * 0.03):
                status_text = "ACTIVE: Buddha Mode"
                mode_triggered = True
                # P 上聖光在頭頂後面
                if buddha_light is not None:
                    light_w = int(face_height * 2.5)
                    light_h = int(light_w * (buddha_light.shape[0] / buddha_light.shape[1]))
                    img = overlay_image(img, buddha_light, int(forehead.x * w - light_w / 2), int(forehead.y * h - light_h * 0.6), size=(light_w, light_h))

        # 2. 手勢偵測 (如果臉部還沒觸發愛因斯坦或佛祖)
        if not mode_triggered and hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            index_tip = hand_landmarks.landmark[8]
            index_mcp = hand_landmarks.landmark[5]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            # 判斷是左手還是右手 (用來確認手心方向)
            # MediaPipe 預設有些鏡像，我們用最簡單的相對位置判斷
            
            # 💡 【秦始皇模式】：比讚 (大拇指朝上，食指低於指節)
            if thumb_tip.y < thumb_ip.y and index_tip.y > index_mcp.y:
                status_text = "ACTIVE: Qin Shihuang Mode"
                mode_triggered = True
                if face_results.multi_face_landmarks:
                    # P 上秦始皇帽子
                    if qin_cap is not None:
                        hat_w = int(face_height * 1.8)
                        hat_h = int(hat_w * (qin_cap.shape[0] / qin_cap.shape[1]))
                        img = overlay_image(img, qin_cap, int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.85), size=(hat_w, hat_h))
                # P 上北極熊在讚的上面
                if polar_bear is not None:
                    bear_w = int(w * 0.25)
                    bear_h = int(bear_w * (polar_bear.shape[0] / polar_bear.shape[1]))
                    img = overlay_image(img, polar_bear, int(thumb_tip.x * w - bear_w / 2), int(thumb_tip.y * h - bear_h - 10), size=(bear_w, bear_h))

            # 💡 【孔子模式】：手心朝自己、手攤平不做動作
            # 攤平判斷：食、中、無名、小指的指尖都高於大拇指，且手指伸直
            elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y < index_mcp.y:
                status_text = "ACTIVE: Confucius Mode"
                mode_triggered = True
                if face_results.multi_face_landmarks:
                    # P 上孔子帽子
                    if kongzi_cap is not None:
                        hat_w = int(face_height * 1.6)
                        hat_h = int(hat_w * (kongzi_cap.shape[0] / kongzi_cap.shape[1]))
                        img = overlay_image(img, kongzi_cap, int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.85), size=(hat_w, hat_h))
                    # P 上鬍鬚
                    if kongzi_beard is not None:
                        beard_w = int(face_height * 1.0)
                        beard_h = int(beard_w * (kongzi_beard.shape[0] / kongzi_beard.shape[1]))
                        img = overlay_image(img, kongzi_beard, int(chin.x * w - beard_w / 2), int(chin.y * h - beard_h * 0.2), size=(beard_w, beard_h))
                # P 上袖子在手掌位置
                if kongzi_sleeves is not None:
                    s_w = int(w * 0.4)
                    s_h = int(s_w * (kongzi_sleeves.shape[0] / kongzi_sleeves.shape[1]))
                    img = overlay_image(img, kongzi_sleeves, int(index_mcp.x * w - s_w / 2), int(index_mcp.y * h), size=(s_w, s_h))

        # 💡 【路易十六模式】：如果前面四個通通都沒被觸發（不做動作）
        if not mode_triggered:
            status_text = "ACTIVE: Louis XVI Mode (No Gesture)"
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                forehead = face_landmarks.landmark[10]
                chin = face_landmarks.landmark[152]
                face_height = abs(forehead.y - chin.y) * h
                
                # 把整顆頭 P 成大番茄
                if tomato is not None:
                    tomato_w = int(face_height * 2.2)
                    tomato_h = int(tomato_w * (tomato.shape[0] / tomato.shape[1]))
                    # 計算中心點，蓋住整張臉
                    center_x = int(forehead.x * w)
                    center_y = int((forehead.y + chin.y) / 2 * h)
                    img = overlay_image(img, tomato, center_x - tomato_w // 2, center_y - tomato_h // 2, size=(tomato_w, tomato_h))

        # 顯示當前狀態文字
        cv2.putText(img, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 網頁畫面佈局 ---
ctx = webrtc_streamer(
    key="auto-meme-filter", 
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(current_orig, caption="拍到的原影像", use_container_width=True)
    with col_filt:
        st.image(current_filter, caption="加上濾鏡後的影相", use_container_width=True)

    st.markdown("---")

    st.subheader("📜 歷史拍照紀錄")
    cols = st.columns(max(5, len(st.session_state.history)))
    for idx, (orig, filt) in enumerate(st.session_state.history):
        if idx < 5:
            with cols[idx]:
                st.image(filt, caption= f"紀錄 #{len(st.session_state.history)-idx}", use_container_width=True)

