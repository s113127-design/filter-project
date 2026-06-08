import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import os

st.title("歷史迷因濾鏡專題 📸")

if "camera_on" not in st.session_state:
    st.session_state.camera_on = True

role = st.selectbox("請選擇角色", ["愛因斯坦", "孔子", "秦始皇", "釋迦牟尼佛", "路易十六"])
st.write(f"當前選定角色：{role}")

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

# --- MediaPipe & 圖片載入設定 ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# 輔助函式：疊加具備透明通道(Alpha)的 PNG 圖片
def overlay_transparent(background, overlay, x, y, size=None):
    if overlay is None:
        return background
    bg_h, bg_w, _ = background.shape
    
    # 縮放貼圖大小
    if size:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
        
    o_h, o_w, o_c = overlay.shape
    if o_c != 4:
        # 如果不是 4 通道，強制轉換或直接返回（防止報錯）
        return background

    # 計算貼圖範圍，防止超出邊界
    x1, x2 = max(x, 0), min(x + o_w, bg_w)
    y1, y2 = max(y, 0), min(y + o_h, bg_h)
    
    o_x1, o_x2 = max(0, -x), min(o_w, bg_w - x)
    o_y1, o_y2 = max(0, -y), min(o_h, bg_h - y)

    if x1 >= x2 or y1 >= y2:
        return background

    # 分離顏色與透明通道
    overlay_crop = overlay[o_y1:o_y2, o_x1:o_x2]
    overlay_img = overlay_crop[:, :, :3]
    mask = overlay_crop[:, :, 3] / 255.0
    mask_inv = 1.0 - mask

    # 像素權重融合
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (mask * overlay_img[:, :, c] +
                                      mask_inv * background[y1:y2, x1:x2, c])
    return background

# 影片處理器類別
class VideoProcessor:
    def __init__(self):
        self.role_mode = "愛因斯坦"
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        
        # 預先載入所有貼圖 (自備 images 資料夾)
        self.images = {}
        img_names = {
            "hair": "images/hair.png",
            "hat": "images/hat.png",
            "beard": "images/beard.png",
            "crown": "images/crown.png",
            "bear": "images/bear.png",
            "halo": "images/halo.png",
            "tomato": "images/tomato.png"
        }
        for key, path in img_names.items():
            if os.path.exists(path):
                self.images[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                self.images[key] = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # AI 偵測
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        triggered = False  # 標記是否成功觸發目前角色的特殊動作
        face_box = None    # 儲存人臉的大致外框，供路易十六或P圖使用

        # 獲取人臉資訊
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 計算人臉方框範圍 (ROI)
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            face_box = (int(min(xs)*w), int(min(ys)*h), int((max(xs)-min(xs))*w), int((max(ys)-min(ys))*h))
            
            # --- 角色動作判定 A：愛因斯坦 (嘴張大/吐舌) ---
            if self.role_mode == "愛因斯坦":
                top_lip = face_landmarks.landmark[13].y
                bot_lip = face_landmarks.landmark[14].y
                if (bot_lip - top_lip) > 0.04:  # 嘴巴張開閾值
                    triggered = True

            # --- 角色動作判定 B：釋迦牟尼佛 (閉眼) ---
            elif self.role_mode == "釋迦牟尼佛":
                # 計算左眼 EAR (上下點距離)
                p_top = face_landmarks.landmark[159].y
                p_bot = face_landmarks.landmark[145].y
                if (p_bot - p_top) < 0.015:  # 閉眼閾值
                    triggered = True

        # 獲取雙手資訊
        if hand_results.multi_hand_landmarks:
            # --- 角色動作判定 C：孔子 (兩手合十 / 偵測到兩隻手) ---
            if self.role_mode == "孔子":
                if len(hand_results.multi_hand_landmarks) == 2:
                    triggered = True
                    
            # --- 角色動作判定 D：秦始皇 (比讚) ---
            elif self.role_mode == "秦始皇":
                hand_lms = hand_results.multi_hand_landmarks[0]
                # 比讚演算法：大拇指尖(4)高於大拇指根部(2)，且其餘四指尖低於其對應的第二關節
                thumb_up = hand_lms.landmark[4].y < hand_lms.landmark[2].y
                index_down = hand_lms.landmark[8].y > hand_lms.landmark[6].y
                if thumb_up and index_down:
                    triggered = True

        # ==========================================
        # 渲染特效與貼圖
        # ==========================================
        if self.role_mode == "愛因斯坦" and triggered:
            # 1. 畫面轉黑白
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # 2. P 上爆炸頭
            if face_box and self.images["hair"] is not None:
                fx, fy, fw, fh = face_box
                p_w = int(fw * 1.8)
                p_h = int(fh * 1.5)
                p_x = fx - int((p_w - fw) / 2)
                p_y = fy - int(p_h * 0.7)
                img = overlay_transparent(img, self.images["hair"], p_x, p_y, (p_w, p_h))

        elif self.role_mode == "釋迦牟尼佛" and triggered:
            # P 上聖光 (置於頭部中央後方)
            if face_box and self.images["halo"] is not None:
                fx, fy, fw, fh = face_box
                p_w = int(fw * 2.5)
                p_x = fx - int((p_w - fw) / 2)
                p_y = fy - int(p_w * 0.5)
                img = overlay_transparent(img, self.images["halo"], p_x, p_y, (p_w, p_w))

        elif self.role_mode == "孔子" and triggered:
            # P 上孔子帽子與鬍鬚
            if face_box:
                fx, fy, fw, fh = face_box
                if self.images["hat"] is not None:
                    img = overlay_transparent(img, self.images["hat"], fx - int(fw*0.1), fy - int(fh*0.7), (int(fw*1.2), int(fh*0.8)))
                if self.images["beard"] is not None:
                    img = overlay_transparent(img, self.images["beard"], fx + int(fw*0.1), fy + int(fh*0.7), (int(fw*0.8), int(fh*0.8)))
            cv2.putText(img, "Confucius Sleeves!", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif self.role_mode == "秦始皇" and triggered:
            # P 上皇冠與左下角的北極熊
            if face_box and self.images["crown"] is not None:
                fx, fy, fw, fh = face_box
                img = overlay_transparent(img, self.images["crown"], fx - int(fw*0.1), fy - int(fh*0.8), (int(fw*1.2), int(fh*0.9)))
            if self.images["bear"] is not None:
                img = overlay_transparent(img, self.images["bear"], 20, h - 170, (150, 150))

        elif self.role_mode == "路易十六":
            # 不需要特殊動作：頭變紅 + 馬賽克 + 旁邊有番茄
            if face_box:
                fx, fy, fw, fh = face_box
                # 防止邊界溢出
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(fw, w - fx), min(fh, h - fy)
                
                if fw > 0 and fh > 0:
                    roi = img[fy:fy+fh, fx:fx+fw]
                    # 做馬賽克：縮小再放大
                    roi_small = cv2.resize(roi, (12, 12), interpolation=cv2.INTER_LINEAR)
                    roi_mosaic = cv2.resize(roi_small, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    # 混合紅色調
                    red_layer = np.zeros_like(roi_mosaic)
                    red_layer[:, :, 2] = 180  # BGR 格式，Index 2 是紅色
                    roi_final = cv2.addWeighted(roi_mosaic, 0.5, red_layer, 0.5, 0)
                    img[fy:fy+fh, fx:fx+fw] = roi_final
                    
            if self.images["tomato"] is not None:
                img = overlay_transparent(img, self.images["tomato"], w - 140, h - 140, (120, 120))

        # 若選擇了某角色但未觸發動作，在畫面上給予友情提示
        if not triggered and self.role_mode != "路易十六":
            cv2.putText(img, f"Action Required! ({self.role_mode})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# --- 啟動串流服務 ---
if st.session_state.camera_on:
    ctx = webrtc_streamer(
        key="meme-filter", 
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    # 將網頁前端選擇的下拉選單角色，即時同步傳入給 WebRTC 影像監聽器
    if ctx.video_processor:
        ctx.video_processor.role_mode = role
else:
    st.warning("⚠️ 鏡頭已關閉。")
