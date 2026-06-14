import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸")
st.write("👉 做出表情或手勢來變身！\n"
         "- 閉上雙眼 👁️❌：變身【釋迦牟尼佛】（即時頭頂發光）\n"
         "- 對鏡頭比讚 👍：變身【秦始皇】（即時戴帽子＋下方北極熊）\n"
         "- 張開嘴巴 😮：變身【愛因斯坦】（即時畫面轉黑白＋爆炸頭吐舌）\n"
         "- 直接拍照（不動作）：拍下後變身【路易十六】（頭變番茄）")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_resources():
    e_hair = cv2.imread("assets/Einstein_hair.png", cv2.IMREAD_UNCHANGED)
    e_tongue = cv2.imread("assets/Einstein_tongue.png", cv2.IMREAD_UNCHANGED)
    tomato = cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED)
    q_cap = cv2.imread("assets/qinshihuang_cap.png", cv2.IMREAD_UNCHANGED)
    p_bear = cv2.imread("assets/polar_bear.png", cv2.IMREAD_UNCHANGED)
    h_light = cv2.imread("assets/holy_light.png", cv2.IMREAD_UNCHANGED)
    return e_hair, e_tongue, tomato, q_cap, p_bear, h_light

einstein_hair, einstein_tongue, louis_tomato, qin_cap, polar_bear, holy_light = load_resources()

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
        background[y1:y2, x1:x2] = composite.astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
    return background

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands  # ─── 新增：手部偵測模組 ───

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.latest_orig = None
        self.latest_filter = None
        self.is_einstein_active = False
        self.is_qin_active = False # 紀錄拍照瞬間是不是秦始皇狀態
        self.is_buddha_active = False # ─── 新增：紀錄拍照瞬間是不是佛祖狀態 ───


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 進行 AI 辨識
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        status_text = "Scanning... Make a gesture!"
        self.is_einstein_active = False
        self.is_qin_active = False
        
        # ─── 步驟一：先判定手勢是否「比讚」───
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
            
            # 比讚演算法：大拇指尖(4) 高於 大拇指根部(2)，且其餘四指(8, 12, 16, 20)皆收起（其指尖低於關節）
            # 由於網頁視訊 Y 軸朝下，指尖 Y 小於關節 Y 代表指尖在上。
            thumb_is_up = hand_landmarks[4].y < hand_landmarks[2].y
            index_is_closed = hand_landmarks[8].y > hand_landmarks[6].y
            middle_is_closed = hand_landmarks[12].y > hand_landmarks[10].y
            ring_is_closed = hand_landmarks[16].y > hand_landmarks[14].y
            pinky_is_closed = hand_landmarks[20].y > hand_landmarks[18].y
            
            if thumb_is_up and index_is_closed and middle_is_closed and ring_is_closed and pinky_is_closed:
                self.is_qin_active = True

        # ─── 步驟二：處理人臉與濾鏡疊加 ───
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            forehead = face_landmarks[10]
            chin = face_landmarks[152]
            
            face_height = abs(forehead.y - chin.y) * h
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            
              # ─── 新增：計算閉眼縱橫比點位 ───
            left_eye_dist = abs(face_landmarks[159].y - face_landmarks[145].y) * h
            right_eye_dist = abs(face_landmarks[386].y - face_landmarks[374].y) * h
            
            # 優先級 1：如果兩眼垂直距離小於人臉高度的 1.5% -> 觸發【釋迦牟尼佛模式】
            if left_eye_dist < (face_height * 0.015) and right_eye_dist < (face_height * 0.015):
                self.is_buddha_active = True
                status_text = "ACTIVE: Shakyamuni Buddha Mode 🪷"
                
                if holy_light is not None:
                    left_face = face_landmarks[234]
                    right_face = face_landmarks[454]
                    face_width = abs(right_face.x - left_face.x) * w
                    
                    light_w = int(face_width * 2.2) # 聖光寬度放大
                    light_scale = holy_light.shape[0] / holy_light.shape[1]
                    light_h = int(light_w * light_scale)
                    
                    light_x = int(forehead.x * w - light_w / 2)
                    light_y = int(forehead.y * h - light_h * 0.65) # 置中偏頭頂上方
                    img = overlay_image(img, holy_light, light_x, light_y, size=(light_w, light_h))

            # 優先級 2：如果比讚 -> 觸發【秦始皇模式】
            if self.is_qin_active:
                status_text = "ACTIVE: Qin Shi Huang Mode 👍"
                
                # A. P 上秦始皇帽子 (qinshihuang_cap.png)
                if qin_cap is not None:
                    left_face = face_landmarks[234]
                    right_face = face_landmarks[454]
                    face_width = abs(right_face.x - left_face.x) * w
                    
                    cap_w = int(face_width * 4.0)  # 冕冠通常比較寬大
                    cap_scale = qin_cap.shape[0] / qin_cap.shape[1]
                    cap_h = int(cap_w * cap_scale)
                    
                    cap_x = int(forehead.x * w - cap_w / 2.05)
                    cap_y = int(forehead.y * h - cap_h * 0.3) # 帽簷蓋到額頭上方
                    img = overlay_image(img, qin_cap, cap_x, cap_y, size=(cap_w, cap_h))
                
                # B. P 上北極熊在畫面最下方中間 (polar_bear.png)
                if polar_bear is not None:
                    bear_w = int(w * 0.35)  # 佔據畫面底部約 35% 寬度
                    bear_scale = polar_bear.shape[0] / polar_bear.shape[1]
                    bear_h = int(bear_w * bear_scale)
                    
                    bear_x = int(w / 2 - bear_w / 2)  # 置中
                    bear_y = h - bear_h               # 貼齊最下方
                    img = overlay_image(img, polar_bear, bear_x, bear_y, size=(bear_w, bear_h))

            # 優先級 2：如果張嘴 -> 觸發【愛因斯坦模式】
            elif lip_dist > (face_height * 0.15):
                self.is_einstein_active = True
                status_text = "ACTIVE: Einstein Mode"
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
                if einstein_hair is not None:
                    left_face = face_landmarks[234]
                    right_face = face_landmarks[454]
                    face_width = abs(right_face.x - left_face.x) * w
                    hair_w = int(face_width * 1.75)
                    scale = einstein_hair.shape[0] / einstein_hair.shape[1]
                    hair_h = int(hair_w * scale)
                    hair_x = int(forehead.x * w - hair_w / 2.5)
                    hair_y = int(forehead.y * h - hair_h * 0.35)
                    img = overlay_image(img, einstein_hair, hair_x, hair_y, size=(hair_w, hair_h))
            
                if einstein_tongue is not None:
                    left_mouth = face_landmarks[61]
                    right_mouth = face_landmarks[291]
                    mouth_width = abs(right_mouth.x - left_mouth.x) * w
                    tongue_w = int(mouth_width * 1.5)
                    scale = einstein_tongue.shape[0] / einstein_tongue.shape[1]
                    tongue_h = int(tongue_w * scale)
                    tongue_x = int(lower_lip.x * w - tongue_w / 2)
                    tongue_y = int(lower_lip.y * h - tongue_h * 0.4)
                    img = overlay_image(img, einstein_tongue, tongue_x, tongue_y, size=(tongue_w, tongue_h))
            
            # 優先級 3：無動作狀態 -> 提示拍下照片會變成【路易十六】
            else:
                status_text = "ACTIVE: Louis XVI Mode (Ready to Tomato)"

        cv2.putText(img, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 網頁畫面佈局 ---
ctx = webrtc_streamer(
    key="camera-step-by-step",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_img = ctx.video_processor.latest_orig.copy()
        filter_img = ctx.video_processor.latest_filter.copy()
        
        # ─── 拍照邏輯：只有在「非秦始皇」且「非愛因斯坦」時，才觸發路易十六（番茄頭） ───
        if not ctx.video_processor.is_qin_active and not ctx.video_processor.is_einstein_active:
            h, w, _ = orig_img.shape
            rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            face_results = ctx.video_processor.face_mesh.process(rgb_img)
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                forehead = face_landmarks[10]
                chin = face_landmarks[152]
                face_height = abs(forehead.y - chin.y) * h
                
                face_size = int(face_height * 1.8)
                center_x = int(((face_landmarks[234].x + face_landmarks[454].x) / 2) * w)
                center_y = int(((forehead.y + chin.y) / 2) * h)
                
                if louis_tomato is not None:
                    tomato_scale = louis_tomato.shape[0] / louis_tomato.shape[1]
                    tomato_w = face_size
                    tomato_h = int(tomato_w * tomato_scale)
                    tomato_x = center_x - int(tomato_w / 2)
                    tomato_y = center_y - int(tomato_h / 2)
                    filter_img = overlay_image(filter_img, louis_tomato, tomato_x, tomato_y, size=(tomato_w, tomato_h))
        
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)
        
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        if len(st.session_state.history) > 8:
            st.session_state.history.pop()
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
        st.image(current_filter, caption="濾鏡影像（路易十六模式會在這加上番茄）", use_container_width=True)

    st.markdown("---")
    st.subheader("📜 歷史拍照紀錄 (最多儲存 8 張)")
    
    cols = st.columns(4)
    for idx, (orig, filt) in enumerate(st.session_state.history):
        col_idx = idx % 4
        with cols[col_idx]:
            st.image(filt, use_container_width=True)
