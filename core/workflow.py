import os
import base64
from functools import lru_cache # 💡 1. เพิ่มตัวนี้เข้ามา

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from core.state import TrioPostState

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

@lru_cache(maxsize=1)
def get_cached_embeddings():
    print("🧠 [System] โหลด Embedding Model เข้าสู่ RAG Node (โหลดครั้งเดียว)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def vision_node(state: TrioPostState):
    image_path = state.get("image_path")
    print(f"👁️ [Vision Agent] กำลังวิเคราะห์รูปภาพจาก: {image_path}")

    try:
        base64_image = encode_image(image_path)

        vision_llm = ChatOpenAI(
            api_key=os.getenv("TYPHOON_API_KEY"),
            base_url="https://api.opentyphoon.ai/v1",
            model="typhoon-ocr",
            max_tokens=4096
        )

        message = HumanMessage(
            content=[
                {"type": "text",
                 "text": "ตอบคำถามต่อไปนี้สั้นๆ ตรงไปตรงมา ห้ามแต่งเรื่อง ห้ามวิเคราะห์ประวัติศาสตร์:\n"
                         "1. วัตถุชิ้นหลักในภาพคืออะไร? (เช่น เสื้อยืด, นาฬิกาข้อมือแบบเข็ม)\n"
                         "2. วัสดุที่เห็นคืออะไร? (เช่น ผ้าฝ้าย, สายโลหะสแตนเลส)\n"
                         "3. สีหลักของวัตถุคือสีอะไร?\n"
                         "4. มีตัวอักษร ตัวเลข หรือโลโก้อะไรปรากฏอยู่บ้าง?"
                        "🚨กฎเหล็ก: ห้ามจินตนาการหรือเดาฟังก์ชันการใช้งานที่มองไม่เห็น (เช่น ระบบสมาร์ทวอทช์, บลูทูธ, วัดชีพจร) ให้บรรยายเฉพาะรูปธรรมที่ปากฏในภาพเท่านั้น"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = vision_llm.invoke([message])
        vision_detail = response.content

        print("✅ วิเคราะห์ภาพสำเร็จ! สกัดจุดเด่นได้เรียบร้อย")
        print(f"[สิ่งที่ AI เห็น]: {vision_detail[:100]}...")  # ปริ้นท์ให้ดูแค่ 100 ตัวอักษรแรก

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {e}")
        vision_detail = "สินค้าแฟชั่น (ข้อมูลภาพขัดข้อง)"

    return {"vision_detail": vision_detail}


def researcher_node(state: TrioPostState):
    item_to_search = state.get("vision_detail", "สินค้าทั่วไป")

    print("🔍 [Researcher Agent] กำลังสกัดคีย์เวิร์ดเพื่อไปค้นหาข้อมูลเชิงลึก...")

    llm_for_search = ChatOpenAI(
        api_key=os.getenv("TYPHOON_API_KEY"),
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        temperature=0.2,
        max_tokens=2048
    )

    keyword_prompt = f"จากรายละเอียดสินค้านี้: '{item_to_search}' จงสกัดชื่อสินค้าหลักเพื่อนำไปค้นหา 'จุดเด่น รีวิว และข้อมูลสินค้า' ในอินเทอร์เน็ต ให้ตอบกลับมาเป็นคำสั้นๆ ไม่เกิน 3-5 คำ (เช่น 'เสื้อยืดลายพราง' หรือ 'เดรสลูกคุณหนู') ห้ามมีน้ำเด็ดขาด"

    short_keyword = llm_for_search.invoke(keyword_prompt).content.strip()

    search_query = f"ข้อมูล รีวิว จุดเด่น ราคา {short_keyword}"
    print(f"🎯 คีย์เวิร์ดที่สกัดได้คือ: '{short_keyword}' -> นำไปค้นหา: '{search_query}'")

    try:
        tavily_tool = TavilySearchResults(max_results=3)
        search_results = tavily_tool.invoke(search_query)

        market_data = ""
        for i, res in enumerate(search_results):
            # เอาข้อความมาเยอะขึ้นนิดนึง (600 ตัวอักษร) เพื่อให้ได้เนื้อหารีวิว
            content_snippet = res['content'][:600].replace('\n', ' ')
            market_data += f"[ข้อมูลอ้างอิง {i + 1}]: {content_snippet}...\n"

        if not market_data:
            market_data = "ไม่พบข้อมูลเชิงลึก แนะนำให้เน้นบรรยายจากรูปลักษณ์ที่เห็นในภาพ"

        print("✅ ค้นหาข้อมูลตลาดและจุดเด่นสำเร็จ!")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล Tavily: {e}")
        market_data = "ไม่สามารถเข้าถึงข้อมูลอินเทอร์เน็ตได้ ให้แต่งแคปชั่นโดยอิงจากภาพเป็นหลัก"

    return {"market_price": market_data}


def rag_node(state: TrioPostState):
    store_name = state.get("store_name", "ไม่ระบุ")
    print(f"🧠 [RAG Node] ดึงคู่มือแบรนด์เจาะจงเฉพาะร้าน: '{store_name}'")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "database", "chroma_db")

    try:
        embeddings = get_cached_embeddings()
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        results = vectorstore.similarity_search(
            query=store_name,
            k=1,
            filter={"store_name": store_name}
        )

        if results:
            retrieved_style = results[0].page_content
            print(f"   ✅ ดึงข้อมูลสำเร็จ! ได้คู่มือของร้าน: {store_name}")
        else:
            retrieved_style = "ใช้สไตล์การเขียนขายของออนไลน์แบบมาตรฐาน เป็นกันเองและสุภาพ"
            print(f"   ⚠️ ไม่พบข้อมูลของร้าน {store_name} ใช้สไตล์มาตรฐานแทน")

    except Exception as e:
        print(f"   ❌ เกิดข้อผิดพลาดในการโหลด RAG: {e}")
        retrieved_style = "ใช้สไตล์การเขียนแบบมาตรฐาน เนื่องจากระบบขัดข้อง"

    return {"rag_context": retrieved_style}


def copywriter_node(state: TrioPostState):
    print(f"✍️ [Copywriter Agent] กำลังแต่งโพสต์ขายของด้วย Typhoon LLM...")

    detail = state.get("vision_detail", "ไม่มีข้อมูลสินค้า")
    price = state.get("market_price", "ไม่มีข้อมูลราคา")
    style = state.get("rag_context", "เขียนสไตล์มาตรฐาน")
    user_prompt = state.get("user_prompt", "")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "คุณคือสุดยอดนักเขียน Copywriter โฆษณาสินค้าออนไลน์มืออาชีพของประเทศไทย\n"
                   "หน้าที่ของคุณคือเขียนแคปชั่นขายของให้น่าสนใจและดูแพง\n"
                   "🚨 กฎเหล็ก (Guardrails) ที่คุณต้องทำตามอย่างเคร่งครัด:\n"
                   "1. สไตล์: จงเขียนตาม 'สไตล์และกฎของร้าน' ที่กำหนดให้อย่างเคร่งครัด\n"
                   "2. ภาษา: ‼️ ต้องเขียนเนื้อหาทั้งหมดเป็น 'ภาษาไทย' เท่านั้น ‼️\n"
                   "3. ความเป็นจริง: บรรยายรูปลักษณ์ตาม [รายละเอียดสินค้า] เท่านั้น ห้ามมโนฟีเจอร์เว่อร์วัง (เช่น ห้ามบอกว่าเป็นสมาร์ทวอทช์ถ้ารูปคือนาฬิกาเข็ม)\n"
                   "4. การกรองข้อมูลขยะ: หาก [ข้อมูลราคาตลาด] มีข้อมูลที่ 'ขัดแย้ง' กับภาพสินค้า ให้เพิกเฉยต่อข้อมูลตลาดนั้นทันที ห้ามนำมาเขียนเด็ดขาด!\n"
                   "5. กฎสูงสุด (Highest Priority): หาก [คำสั่งเพิ่มเติมจากลูกค้า] มีการระบุ 'ราคา' ให้พิมพ์ราคาตามนั้นเป๊ะๆ ห้ามคิดไปเองว่าลูกค้าพิมพ์ตกหล่น และห้ามใช้ราคาจากตลาด"),
        ("user", "ข้อมูลสำหรับเขียนโพสต์มีดังนี้:\n"
                 "📌 รายละเอียดสินค้า: {detail}\n"
                 "💰 ข้อมูลราคาตลาด: {price}\n"
                 "🧠 สไตล์และกฎของร้าน: {style}\n"
                 "🗣️ คำสั่งเพิ่มเติมจากลูกค้า: {user_prompt}\n\n"
                 "ช่วยเขียนแคปชั่นขายของ พร้อมใส่ Hashtag ที่เหมาะสมให้หน่อย พร้อมโพสต์เลย!")
    ])

    llm = ChatOpenAI(
        api_key=os.getenv("TYPHOON_API_KEY"),
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        temperature=0.7,
        max_tokens = 4096
    )

    chain = prompt_template | llm

    response = chain.invoke({
        "detail": detail,
        "price": price,
        "style": style,
        "user_prompt": user_prompt
    })

    print("✅ แต่งแคปชั่นเสร็จสมบูรณ์!")

    return {"final_post": response.content}

def build_workflow():
    builder = StateGraph(TrioPostState)

    builder.add_node("vision", vision_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("rag", rag_node)
    builder.add_node("copywriter", copywriter_node)

    builder.add_edge(START, "vision")
    builder.add_edge("vision", "researcher")
    builder.add_edge("researcher", "rag")
    builder.add_edge("rag", "copywriter")
    builder.add_edge("copywriter", END)

    return builder.compile()

if __name__ == "__main__":
    print("🚀 เริ่มต้นเดินสายพาน TrioPost Workflow (Mock Mode)...\n")

    app = build_workflow()

    initial_state = {
        "image_path": "../assets/test_image.jpg",
        "user_prompt": "ขอแบบทางการหน่อย และขายราคา 990 บาทเท่านั้นนะ ห้ามตั้งราคาอื่น",
        "store_name": "LuxeAura"
    }

    final_result = app.invoke(initial_state)

    print("\n" + "=" * 40)
    print("🎉 ผลลัพธ์สุดท้ายที่ได้จากสายพาน (Final Post):")
    print("=" * 40)
    print(final_result["final_post"])