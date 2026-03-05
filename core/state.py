from typing import TypedDict, Optional

class TrioPostState(TypedDict):
    # 1. Input
    image_path: str
    user_prompt: str
    store_name: str

    # 2. Vision Agent
    vision_detail: Optional[str]

    # 3. Researcher Agent
    market_price: Optional[str]

    # 4. RAG
    rag_context: Optional[str]

    # 5. Copywriter Agent
    final_post: Optional[str]