import os
import base64
from functools import lru_cache

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
    product_name = state.get("product_name", "").strip()

    print(f"👁️ [Vision Agent] กำลังวิเคราะห์รูปภาพจาก: {image_path}")

    if product_name:
        instruction = f"รูปนี้คือ '{product_name}' จงอธิบายรูปลักษณ์ สี วัสดุ และความสวยงามของมันมาสั้นๆ ห้ามเดาชื่อแบรนด์อื่น ห้ามอ่านตัวเลขหน้าปัด"
    else:
        instruction = (
            "วิเคราะห์ภาพสินค้าอย่างตรงไปตรงมาและตอบเป็นข้อๆ:\n"
            "1. สินค้าในภาพคืออะไร? (ตอบสั้นๆ เช่น นาฬิกาข้อมือ, เสื้อยืด)\n"
            "2. ชื่อแบรนด์/ยี่ห้อ: อ่านเฉพาะ 'ตัวอักษรที่เป็นชื่อยี่ห้อหรือโลโก้' ที่เด่นชัดที่สุด (🚨 ห้ามอ่านตัวเลขบอกเวลาบนหน้าปัดนาฬิกาเด็ดขาด ถัาไม่มีโลโก้ให้ตอบว่า ไม่ระบุ)\n"
            "3. สีและวัสดุ: (เช่น โลหะสีเงิน หน้าปัดดำ)\n"
            "🚨 กฎเหล็ก: ห้ามจินตนาการฟังก์ชัน ห้ามแต่งเรื่อง ห้ามพ่นตัวเลขเรียงกัน ตอบแค่เนื้อๆ"
        )

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
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = vision_llm.invoke([message])
        vision_detail = response.content

        print("✅ วิเคราะห์ภาพสำเร็จ! สกัดจุดเด่นได้เรียบร้อย")
        print(f"[สิ่งที่ AI เห็น]: {vision_detail[:100]}...")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {e}")
        vision_detail = "สินค้าแฟชั่น (ข้อมูลภาพขัดข้อง)"

    return {"vision_detail": vision_detail}

def researcher_node(state: TrioPostState):
    item_to_search = state.get("vision_detail", "สินค้าทั่วไป")
    product_name = state.get("product_name", "").strip()

    print("🔍 [Researcher Agent] กำลังสกัดคีย์เวิร์ดเพื่อไปค้นหาข้อมูลเชิงลึก...")

    if product_name:
        short_keyword = product_name
        print(f"🎯 ใช้คีย์เวิร์ดจากผู้ใช้โดยตรง: '{short_keyword}'")
    else:
        llm_for_search = ChatOpenAI(
            api_key=os.getenv("TYPHOON_API_KEY"),
            base_url="https://api.opentyphoon.ai/v1",
            model="typhoon-v2.5-30b-a3b-instruct",
            temperature=0.1,
            max_tokens=2048
        )

        keyword_prompt = f"""
        จากข้อมูลที่ Vision Agent มองเห็น: '{item_to_search}'

        หน้าที่ของคุณคือสกัด 'ชื่อแบรนด์ และ ประเภทสินค้า' ออกมาเป็นคีย์เวิร์ดสำหรับค้นหา
        - เช่น หากข้อมูลคือ "สินค้า: นาฬิกาข้อมือ, แบรนด์: ROLEX Air-King" ให้ตอบว่า "นาฬิกา ROLEX Air-King"
        - 🚨 ตัดข้อมูลเรื่องสี วัสดุ หรือตัวเลขทิ้งไปให้หมด

         จงตอบแค่ "คีย์เวิร์ดสั้นๆ 1-4 คำ" เท่านั้น
        """

        short_keyword = llm_for_search.invoke(keyword_prompt).content.strip()
        print(f"🎯 คีย์เวิร์ดที่สกัดด้วย AI คือ: '{short_keyword}'")

    search_query = f"ข้อมูล รีวิว สเปค {short_keyword}"
    print(f"-> นำไปค้นหา: '{search_query}'")

    try:
        tavily_tool = TavilySearchResults(max_results=3)
        search_results = tavily_tool.invoke(search_query)

        market_data = ""
        for i, res in enumerate(search_results):
            content_snippet = res['content'][:600].replace('\n', ' ')
            market_data += f"[ข้อมูลอ้างอิง {i + 1}]: {content_snippet}...\n"

        if not market_data:
            market_data = "ไม่พบข้อมูลเชิงลึก แนะนำให้เน้นบรรยายจากรูปลักษณ์ที่เห็นในภาพ"

        print("✅ ค้นหาข้อมูลตลาดและจุดเด่นสำเร็จ!")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล Tavily: {e}")
        market_data = "ไม่สามารถเข้าถึงข้อมูลอินเทอร์เน็ตได้ ให้แต่งแคปชั่นโดยอิงจากภาพเป็นหลัก"

    return {"research_data": market_data}

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
    research_data = state.get("research_data", "ไม่มีข้อมูลอ้างอิงหรือสเปค")
    style = state.get("rag_context", "เขียนสไตล์มาตรฐาน")
    user_prompt = state.get("user_prompt", "")
    platform = state.get("platform", "Facebook (จัดเต็ม)")
    product_name = state.get("product_name", "").strip()

    display_product_name = product_name if product_name else "สินค้าในภาพ"

    platform_instructions = {
        "Facebook (จัดเต็ม)": (
            "เขียนในรูปแบบ Facebook Post: เน้นการเล่าเรื่อง (Storytelling) บรรยายรายละเอียด สเปค และจุดเด่นให้ครบถ้วน "
            "แบ่งย่อหน้าให้อ่านง่าย สบายตา มี Call-to-Action (CTA) ปิดการขายที่ชัดเจน และใส่แฮชแท็กที่เกี่ยวข้องประมาณ 3-5 คำตอนท้าย"
        ),
        "Instagram (เน้นแฮชแท็ก)": (
            "เขียนในรูปแบบ Instagram Caption: เน้นความสวยงามทางภาษา เปิดด้วยประโยคฮุก (Hook) ที่ดึงดูดสายตาตั้งแต่บรรทัดแรก "
            "เนื้อหาสั้นกระชับแต่ดูแพง ใช้อีโมจิช่วยคุมโทนภาพรวมให้ดูดี 🚨 และที่สำคัญที่สุด: ต้องมีแฮชแท็กจำนวนมาก (10-15 คำ) กองรวมกันไว้ที่ย่อหน้าสุดท้ายเพื่อเน้นการค้นหา"
        ),
        "X / Twitter (สั้นกระชับ)": (
            "เขียนในรูปแบบ Twitter (X): 🚨 ข้อบังคับสูงสุด: ต้องสั้น กระชับ กระแทกใจ และความยาวรวมต้องไม่เกิน 280 ตัวอักษรเด็ดขาด! "
            "ตัดน้ำออกให้หมด เน้นเฉพาะจุดเด่นที่ว้าวที่สุด 1-2 อย่าง ใช้อีโมจิน้อยๆ และใส่แฮชแท็กที่คิดว่ากำลังเป็นกระแส (Trending) เพียง 1-3 คำเท่านั้น"
        )
    }

    specific_platform_rule = platform_instructions.get(platform, platform_instructions["Facebook (จัดเต็ม)"])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "คุณคือสุดยอดนักเขียน Copywriter โฆษณาสินค้าออนไลน์มืออาชีพของประเทศไทย\n"
                   "หน้าที่ของคุณคือเขียนแคปชั่นขายของให้น่าสนใจและดูแพง\n"
                   "🚨 กฎเหล็ก (Guardrails) ที่คุณต้องทำตามอย่างเคร่งครัด:\n"
                   "1. สไตล์: จงเขียนตาม 'สไตล์และกฎของร้าน' ที่กำหนดให้อย่างเคร่งครัด\n"
                   "2. ภาษา: ‼️ ต้องเขียนเนื้อหาทั้งหมดเป็น 'ภาษาไทย' เท่านั้น ‼️\n"
                   "3. ชื่อสินค้า: ‼️ ต้องระบุชื่อสินค้า '{display_product_name}' ลงไปในเนื้อหาแคปชั่นอย่างเป็นธรรมชาติ ‼️\n"
                   "4. ความเป็นจริง: บรรยายรูปลักษณ์ตาม [รายละเอียดจากภาพ] เท่านั้น ห้ามมโนฟีเจอร์เว่อร์วัง\n"
                   "5. การใช้ข้อมูลสเปค: นำ [ข้อมูลเชิงลึกและสเปค] มาผสมผสานในการบรรยายเพื่อเพิ่มความน่าเชื่อถือ แต่ถ้าข้อมูลไหนขัดแย้งกับภาพให้ตัดทิ้งทันที\n"
                   "6. กฎสูงสุด (Highest Priority): หาก [คำสั่งพิเศษ] มีการระบุ 'ราคา' ให้พิมพ์ราคาตามนั้นเป๊ะๆ 🚨 ห้ามใช้ราคาจาก [ข้อมูลเชิงลึกและสเปค] มาปะปนเด็ดขาด\n"
                   "7. รูปแบบการจัดหน้าและแพลตฟอร์ม: {specific_platform_rule}"),
        ("user", "ข้อมูลสำหรับเขียนโพสต์มีดังนี้:\n"
                 "🏷️ ชื่อแบรนด์/รุ่นสินค้า: {display_product_name}\n"
                 "📌 รายละเอียดจากภาพ: {detail}\n"
                 "🌐 ข้อมูลเชิงลึกและสเปค: {research_data}\n"
                 "🧠 สไตล์และกฎของร้าน: {style}\n"
                 "🗣️ คำสั่งพิเศษจากลูกค้า: {user_prompt}\n"
                 "📱 แพลตฟอร์มเป้าหมาย: {platform}\n\n"
                 "ช่วยเขียนแคปชั่นขายของให้เป๊ะตามกฎ พร้อมโพสต์เลย!")
    ])

    llm = ChatOpenAI(
        api_key=os.getenv("TYPHOON_API_KEY"),
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        temperature=0.7,
        max_tokens=4096
    )

    chain = prompt_template | llm

    response = chain.invoke({
        "display_product_name": display_product_name,
        "detail": detail,
        "research_data": research_data,
        "style": style,
        "user_prompt": user_prompt,
        "platform": platform,
        "specific_platform_rule": specific_platform_rule
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
        "store_name": "LuxeAura",
        "platform": "Facebook (จัดเต็ม)"  # 💡 ส่งลองแพลตฟอร์มด้วย
    }

    final_result = app.invoke(initial_state)

    print("\n" + "=" * 40)
    print("🎉 ผลลัพธ์สุดท้ายที่ได้จากสายพาน (Final Post):")
    print("=" * 40)
    print(final_result["final_post"])