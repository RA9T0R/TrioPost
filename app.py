import streamlit as st
import os
from PIL import Image

from core.workflow import build_workflow

st.set_page_config(page_title="TrioPost: AI Content Creator", page_icon="🤖", layout="centered")

st.title("🤖 TrioPost: ผู้ช่วยเขียนโพสต์ขายของ")
st.markdown(
    "อัปโหลดรูปภาพสินค้า และพิมพ์คำสั่งของคุณ เพื่อให้ AI Agents (Vision + Search + RAG + Copywriter) ช่วยแต่งแคปชั่นให้!")

os.makedirs("assets", exist_ok=True)

# Input
uploaded_file = st.file_uploader("📸 1. อัปโหลดรูปภาพสินค้า (JPG/PNG)", type=["jpg", "jpeg", "png"])

user_prompt = st.text_area(
    "✍️ 2. คำสั่งเพิ่มเติม (Prompt)",
    value="ขอแบบทางการหน่อย และขายราคา 990 บาทเท่านั้นนะ ห้ามตั้งราคาอื่น",
    height=100
)

# ส่วนประมวลผล
if st.button("🚀 สร้างโพสต์เลย!", use_container_width=True):
    if uploaded_file is not None:
        # 1. แสดงรูปภาพที่อัปโหลดให้ผู้ใช้ดู
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปภาพสินค้าของคุณ", use_column_width=True)

        # 2. บันทึกรูปภาพลงเครื่องชั่วคราว เพื่อให้ Vision Agent เข้าไปอ่านได้
        temp_image_path = os.path.join("assets", "temp_uploaded_image.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 3. เริ่มรันระบบ AI
        with st.spinner("⏳ ระบบกำลังประมวลผล (วิเคราะห์ภาพ ➔ หาราคา ➔ ดึงสไตล์ ➔ แต่งแคปชั่น)..."):
            try:
                # เรียกสายพาน LangGraph
                app = build_workflow()

                # เตรียม State เริ่มต้น
                initial_state = {
                    "image_path": temp_image_path,
                    "user_prompt": user_prompt
                }

                # สั่งรัน!
                final_result = app.invoke(initial_state)

                # 4. แสดงผลลัพธ์
                st.success("✨ สร้างแคปชั่นเสร็จสมบูรณ์!")

                st.subheader("📝 แคปชั่นพร้อมโพสต์:")
                # ใช้ตู้ข้อความเพื่อให้กดก๊อปปี้ไปใช้งานได้ง่ายๆ
                st.info(final_result["final_post"])

                # (แถม) ทำปุ่มเปิด/ปิด เพื่อโชว์ความฉลาดเบื้องหลังให้อาจารย์ดูตอน Demo
                with st.expander("🔍 ดูข้อมูลเบื้องหลังการคิดของ AI Agents"):
                    st.markdown("**👁️ สิ่งที่ Vision Agent เห็น:**")
                    st.write(final_result.get("vision_detail", "ไม่มีข้อมูล"))

                    st.markdown("**🔍 ข้อมูลที่ Researcher Agent หามาได้:**")
                    st.write(final_result.get("market_price", "ไม่มีข้อมูล"))

                    st.markdown("**🧠 สไตล์ที่ RAG Node ดึงมาใช้:**")
                    st.write(final_result.get("rag_context", "ไม่มีข้อมูล"))

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในระบบ AI: {e}")

    else:
        st.warning("⚠️ กรุณาอัปโหลดรูปภาพสินค้าก่อนคลิกสร้างโพสต์ครับ!")