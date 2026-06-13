import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(layout="wide")
st.title("像個偉(偽)人一樣 📸")
st.write("👉 **做出指定動作來變身：**")

# 介面說明表格
st.markdown("""
| 偉人 | 觸發動作 | 濾鏡效果與貼圖 |
| :--- | :--- | :--- |
| **愛因斯坦** | 張開嘴巴 | 畫面轉黑白，P 上頭髮、舌頭 |
| **孔子** | 手心朝內且手攤平 | P 上帽子、鬍鬚 |
| **秦始皇** | 比個大大的「讚」👍 | P 上皇冠帽子、旁邊出現北極熊 |
| **釋迦牟尼佛** | 閉上雙眼 | 頭頂後方出現聖光 |
| **路易十六** | 不做動作（或偵測不到上述動作） | 臉部直接被 P 成大番茄 🍅 |
""")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_resources():
    res = {
        "e_hair": cv2.imread("assets/Einstein_hair.png", cv2.IMREAD_UNCHANGED),
        "e_tongue": cv2.imread("assets/Einstein_tongue.png", cv2.IMREAD_UNCHANGED),
        "c_hat": cv2.imread("assets/confucius_hat.png", cv2.IMREAD_UNCHANGED),
        "c_beard": cv2.imread("assets/confucius_beard.png", cv2.IMREAD_UNCHANGED),
        "q_hat": cv2.imread("assets/qin_hat.png", cv2.IMREAD_UNCHANGED),
        "bear": cv2.imread("assets/polar_bear.png", cv2.IMREAD_UNCHANGED),
        "b_light": cv2.imread("assets/buddha_light.png", cv2.IMREAD_UNCHANGED),
        "tomato": cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED)
    }
    return res

assets = load_resources()

def overlay_image(background, overlay, x, y, size=None):
    if overlay is None:
        return background
    bg_h, bg_w = background.shape[:2]
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background
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

# 初始化 MediaPipe 模組
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.latest_orig = None
        self.latest_filter = None

    def get_ear(self, landmarks, eye_indices):
        # 計算眼睛縱橫比 (Eye Aspect Ratio) 來判斷閉眼
        p2_p6 = np.linalg.norm(np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) - 
                               np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y]))
        p3_p5 = np.linalg.norm(np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y]) - 
                               np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]))
        p1_p4 = np.linalg.norm(np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) - 
                               np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y]))
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        # 狀態旗標
        is_mouth_open = False
        is_eyes_closed = False
        is_thumbs_up = False
        is_hand_flat = False
        
        face_landmarks = None
        
        # 1. 解析人臉特徵
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            
            # --- 判斷張嘴 (愛因斯坦) ---
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            forehead = face_landmarks[10]
            chin = face_landmarks[152]
            face_height = abs(forehead.y - chin.y) * h
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            if lip_dist > (face_height * 0.15):
                is_mouth_open = True
            
            # --- 判斷閉眼 (釋迦牟尼) ---
            # MediaPipe 眼睛特徵點索引
            left_eye_idx = [362, 385, 387, 263, 373, 380]
            right_eye_idx = [33, 160, 158, 133, 153, 144]
            left_ear = self.get_ear(face_landmarks, left_eye_idx)
            right_ear = self.get_ear(face_landmarks, right_eye_idx)
            # 當雙眼 EAR 都小於 0.2 判定為閉眼
            if left_ear < 0.21 and right_ear < 0.21:
                is_eyes_closed = True

        # 2. 解析手部特徵
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
            
            # 取得手指尖端與關節點 (Y軸越小代表在畫面上方)
            thumb_tip = hand_landmarks[4]
            thumb_ip = hand_landmarks[3]
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]
            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]
            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]
            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]
            
            # 判斷個別手指是否伸直 (Tip 比 Pip 更靠近上方)
            index_open = index_tip.y < index_pip.y
            middle_open = middle_tip.y < middle_pip.y
            ring_open = ring_tip.y < ring_pip.y
            pinky_open = pinky_tip.y < pinky_pip.y
            
            # --- 判斷比讚 (秦始皇) ---
            # 大拇指高於倒數第二個關節，且其他四指收起
            if thumb_tip.y < thumb_ip.y and not (index_open or middle_open or ring_open or pinky_open):
                is_thumbs_up = True
                
            # --- 判斷手攤平 (孔子) ---
            # 四隻手指皆伸展
            elif index_open and middle_open and ring_open and pinky_open:
                is_hand_flat = True

        # 3. 依條件優先權套用濾鏡與貼圖
        status_text = "Mode: Louis XVI (Tomato)"
        
        if is_mouth_open:
            status_text = "Mode: Einstein"
            # 畫面轉黑白
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if face_landmarks:
                # 算人臉寬度放頭髮
                face_width = abs(face_landmarks[454].x - face_landmarks[234].x) * w
                hair_w = int(face_width * 1.8)
                hair_h = int(hair_w * (assets["e_hair"].shape[0] / assets["e_hair"].shape[1])) if assets["e_hair"] is not None else 10
                img = overlay_image(img, assets["e_hair"], int(face_landmarks[10].x * w - hair_w/2), int(face_landmarks[10].y * h - hair_h * 0.45), size=(hair_w, hair_h))
                # 放舌頭
                mouth_w = abs(face_landmarks[291].x - face_landmarks[61].x) * w
                tongue_w = int(mouth_w * 0.6)
                tongue_h = int(tongue_w * (assets["e_tongue"].shape[0] / assets["e_tongue"].shape[1])) if assets["e_tongue"] is not None else 10
                img = overlay_image(img, assets["e_tongue"], int(face_landmarks[14].x * w - tongue_w/2), int(face_landmarks[14].y * h + tongue_h * 0.2), size=(tongue_w, tongue_h))

        elif is_hand_flat and face_landmarks:
            status_text = "Mode: Confucius"
            face_width = abs(face_landmarks[454].x - face_landmarks[234].x) * w
            # 孔子帽子
            hat_w = int(face_width * 1.5)
            hat_h = int(hat_w * (assets["c_hat"].shape[0] / assets["c_hat"].shape[1])) if assets["c_hat"] is not None else 10
            img = overlay_image(img, assets["c_hat"], int(face_landmarks[10].x * w - hat_w/2), int(face_landmarks[10].y * h - hat_h * 0.7), size=(hat_w, hat_h))
            # 孔子鬍鬚
            beard_w = int(face_width * 0.8)
            beard_h = int(beard_w * (assets["c_beard"].shape[0] / assets["c_beard"].shape[1])) if assets["c_beard"] is not None else 10
            img = overlay_image(img, assets["c_beard"], int(face_landmarks[152].x * w - beard_w/2), int(face_landmarks[152].y * h - beard_h * 0.1), size=(beard_w, beard_h))

        elif is_thumbs_up and face_landmarks:
            status_text = "Mode: Qin Shi Huang"
            face_width = abs(face_landmarks[454].x - face_landmarks[234].x) * w
            # 秦始皇帽子
            hat_w = int(face_width * 1.6)
            hat_h = int(hat_w * (assets["q_hat"].shape[0] / assets["q_hat"].shape[1])) if assets["q_hat"] is not None else 10
            img = overlay_image(img, assets["q_hat"], int(face_landmarks[10].x * w - hat_w/2), int(face_landmarks[10].y * h - hat_h * 0.65), size=(hat_w, hat_h))
            # 旁邊出現北極熊 (固定右下角)
            bear_w = int(w * 0.3)
            bear_h = int(bear_w * (assets["bear"].shape[0] / assets["bear"].shape[1])) if assets["bear"] is not None else 10
            img = overlay_image(img, assets["bear"], w - bear_w - 20, h - bear_h - 20, size=(bear_w, bear_h))

        elif is_eyes_closed and face_landmarks:
            status_text = "Mode: Buddha"
            face_width = abs(face_landmarks[454].x - face_landmarks[234].x) * w
            # 聖光 (放在頭部正後方偏上)
            light_w = int(face_width * 3.0)
            light_h = int(light_w * (assets["b_light"].shape[0] / assets["b_light"].shape[1])) if assets["b_light"] is not None else 10
            img = overlay_image(img, assets["b_light"], int(face_landmarks[10].x * w - light_w/2), int(face_landmarks[10].y * h - light_h * 0.6), size=(light_w, light_h))

        else:
            # 預設：路易十六模式（把整張臉 P 成番茄）
            if face_landmarks:
                face_width = abs(face_landmarks[454].x - face_landmarks[234].x) * w
                tomato_w = int(face_width * 2.0)
                tomato_h = int(tomato_w * (assets["tomato"].shape[0] / assets["tomato"].shape[1])) if assets["tomato"] is not None else 10
                # 以人臉中心點定位
                center_x = int(face_landmarks[1].x * w)
                center_y = int(face_landmarks[1].y * h)
                img = overlay_image(img, assets["tomato"], center_x - tomato_w // 2, center_y - tomato_h // 2, size=(tomato_w, tomato_h))

        # 顯示當前模式文字
        cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 網頁畫面佈局與拍照快照功能 ---
ctx = webrtc_streamer(
    key="historical-filter",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        if len(st.session_state.history) > 8:
            st.session_state.history.pop()
        st.success("拍照成功！已新增至下方紀錄。")
    else:
        st.warning("請先點擊 START 開啟相機再進行拍照！")

st.markdown("---")

if st.session_state.history:
    st.subheader("🖼️ 最新拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(current_orig, caption="原始影像", use_container_width=True)
    with col_filt:
        st.image(current_filter, caption="套用濾鏡影像", use_container_width=True)

    st.markdown("---")
    st.subheader("📜 歷史拍照紀錄 (最多儲存 8 張)")
    cols = st.columns(4)
    for idx, (orig, filt) in enumerate(st.session_state.history):
        col_idx = idx % 4
        with cols[col_idx]:
            st.image(filt, use_container_width=True)
