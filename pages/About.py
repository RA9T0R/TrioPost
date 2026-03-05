import streamlit as st

st.set_page_config(page_title="About TrioPost", layout="wide")

st.title("💡 เกี่ยวกับระบบ TrioPost")
st.divider()

st.markdown("""
**TrioPost** คือผู้ช่วยปัญญาประดิษฐ์ (AI Assistant) ที่ออกแบบมาเพื่อพ่อค้าแม่ค้าออนไลน์โดยเฉพาะ 
แก้ปัญหาการคิดแคปชั่นไม่ออก หาราคาตลาดไม่เจอ หรือคุมโทนร้านไม่ได้ โดยประยุกต์ใช้เทคโนโลยี **Generative AI** และการทำงานร่วมกันของ AI หลายตัว (Multi-Agent System) แบบอัตโนมัติ
""")

st.header("🧠 สถาปัตยกรรมระบบ (System Architecture)")
st.markdown("ระบบถูกพัฒนาบนแนวคิด **State Graph Workflow** ควบคุมการไหลของข้อมูลด้วย **LangGraph**")

st.subheader("🛡️ กระบวนการทำงานของ 4 Agents")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.info("""
    **👁️ 1. Vision Agent (Typhoon OCR)**\n
    ทำหน้าที่เปรียบเสมือน 'ดวงตา' สแกนภาพสินค้าเพื่อสกัดจุดเด่น สี สภาพ และรายละเอียดต่างๆ ออกมาเป็นข้อความ โดยมี Guardrails ป้องกันการมโนข้อมูล (Hallucination)
    """)

    # 💡 อัปเดตคำอธิบาย RAG ให้ดูเทพขึ้น อวดเรื่อง Few-Shot
    st.success("""
    **🧠 3. RAG Node (ChromaDB)**\n
    ทำหน้าที่เป็น 'สมองส่วนความจำ' ดึงคู่มือแบรนด์ (Brand Guidelines) และประยุกต์ใช้เทคนิค **Few-Shot Prompting** (แนบตัวอย่างโพสต์เก่า) เพื่อคุมสไตล์การเขียนให้เป๊ะและเนียนที่สุด
    """)

with col2:
    st.warning("""
    **🔍 2. Researcher Agent (Tavily Search)**\n
    ทำหน้าที่เป็น 'นักสืบ' วิ่งออกไปค้นหาราคาตลาดและเทรนด์ปัจจุบันจากแพลตฟอร์ม E-commerce (Shopee/Lazada) แบบ Real-time
    """)

    st.error("""
    **✍️ 4. Copywriter Agent (Typhoon LLM)**\n
    ทำหน้าที่เป็น 'นักเขียนมือฉมัง' รวบรวมข้อมูลจาก 3 Agent แรก มาแต่งเป็นแคปชั่นโฆษณาที่สละสลวยและปิดการขายได้จริง ภายใต้คำสั่งที่เคร่งครัด
    """)

st.divider()

st.header("🛠️ เทคโนโลยีที่ใช้ (Tech Stack)")

t1, t2, t3 = st.columns(3)

with t1:
    st.markdown("""
    **🤖 AI & Models**
    - **Typhoon v2.5** (LLM สำหรับแต่งประโยค)
    - **Typhoon OCR** (Vision สำหรับอ่านภาพ)
    - **Sentence-Transformers** (Embedding)
    """)

with t2:
    st.markdown("""
    **⚙️ Frameworks**
    - **LangChain** (เชื่อมต่อ AI)
    - **LangGraph** (จัดการ Workflow)
    """)

with t3:
    st.markdown("""
    **💻 Frontend & Database**
    - **Streamlit** (Web Application)
    - **ChromaDB** (Vector Database)
    - **Tavily Search API** (Web Search)
    """)

st.divider()

st.header("👨‍💻 ผู้จัดทำโครงงาน (Developer)")
st.markdown("โครงงานนี้เป็นส่วนหนึ่งของการพัฒนาทักษะด้าน Generative AI Engineering")

# 💡 อย่าลืมใส่ชื่อตัวเองตรงนี้นะครับ!
st.markdown("""
- **ชื่อ-นามสกุล:** พงษ์พัฒน์ บางข่า
- **รหัสนักศึกษา:** 6604062630358
- **บทบาท:** AI Engineer & Developer
""")