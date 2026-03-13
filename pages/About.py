import streamlit as st

st.set_page_config(page_title="About TrioPost", layout="wide")

with st.sidebar:
    st.markdown("### 📃 TrioPost")
    st.caption("AI Social Commerce Assistant")

st.title("💡 เกี่ยวกับ TrioPost")
st.divider()

st.markdown("""
**TrioPost** คือ AI Assistant ที่ออกแบบมาเพื่อผู้ที่ต้องการลดเวลาในการเขียน Post ของขาย
โดยจะแก้ปัญหาการคิดแคปชั่นไม่ออก หรือคุมโทนร้านไม่ได้ โดยประยุกต์ใช้เทคโนโลยี **Generative AI** และการทำงานร่วมกันของ AI หลายตัว (Multi-Agent System) แบบอัตโนมัติ
""")

st.header("🧠 สถาปัตยกรรมระบบ (System Architecture)")
st.markdown("ระบบถูกพัฒนาบนแนวคิด **State Graph Workflow** ควบคุม flow การทำงานผ     **LangGraph**")

st.subheader("🛡️ กระบวนการทำงานของ 4 Agents")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.info("""
    **👁️ 1. Vision Agent (Typhoon OCR)**\n
    ทำหน้าที่เปรียบเสมือน 'ดวงตา' สแกนภาพสินค้าเพื่อสกัดจุดเด่น สี สภาพ และรายละเอียดต่างๆ ออกมาเป็นข้อความ โดยมี Guardrails ป้องกันการมั่วข้อมูล
    """)

    st.success("""
    **🧠 3. RAG Node (ChromaDB)**\n
    ทำหน้าที่เป็น 'สมองส่วนความจำ' ดึงคู่มือแบรนด์ (Brand Guidelines) และประยุกต์ใช้เทคนิค **Few-Shot Prompting** (แนบตัวอย่างโพสต์เก่า) เพื่อคุมสไตล์การเขียนให้เป๊ะและเนียนที่สุด
    """)

with col2:
    st.warning("""
    **🔍 2. Researcher Agent (Tavily Search)**\n
    ทำหน้าที่เป็น 'นักสืบ' วิ่งออกไปค้นหาข้อมูลของสินค้าจากแหล่งต่างๆ มาประกอบการเขียนโพสน์
    """)

    st.error("""
    **✍️ 4. Copywriter Agent (Typhoon LLM)**\n
    ทำหน้าที่เป็น 'นักเขียนมือฉมัง' รวบรวมข้อมูลจาก 3 Agent แรก มาแต่งเป็นแคปชั่นโฆษณาที่สละสลวยและปิดการขายได้จริง ภายใต้คำสั่งที่เคร่งครัด
    """)

st.divider()

st.header("🛠️ เทคโนโลยีที่ใช้ (Tech Stack)")

t1, t2, t3 = st.columns(3)

with t1:
    with st.container(border=True):
        st.markdown("""
        **🤖 AI & Models**
        - **Typhoon v2.5** (LLM)
        - **Typhoon OCR** (Vision)
        - **Tavily API** (Web Search)
        - **Intfloat**
        """)

with t2:
    with st.container(border=True):
        st.markdown("""
        **⚙️ Frameworks**
        - **LangChain** (AI Orchestration)
        - **LangGraph** (Workflow)
        """)

with t3:
    with st.container(border=True):
        st.markdown("""
        **💻 Frontend & Database**
        - **Streamlit** (Web App)
        - **ChromaDB** (Vector DB)
        """)

st.divider()