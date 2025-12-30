# app.py
import os
import json
import torch
import torch.nn.functional as F
import streamlit as st

from model import AdvancedTransformerClassifier

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI


def tokenize(text: str):
    return text.lower().split()

def numericalize(text: str, vocab):
    toks = tokenize(text)
    unk = vocab.get("<unk>", 1)
    return [vocab.get(t, unk) for t in toks]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_id2label(label2id: dict):
    max_id = max(int(v) for v in label2id.values())
    id2label = [""] * (max_id + 1)
    for lab, i in label2id.items():
        id2label[int(i)] = lab
    return id2label


# --------------------------
# 本地模型加载
# --------------------------
@st.cache_resource
def load_local_model(ckpt_dir: str, device: str):
    vocab = load_json(os.path.join(ckpt_dir, "vocab.json"))
    label2id = load_json(os.path.join(ckpt_dir, "label2id.json"))
    id2label = build_id2label(label2id)

    # config
    config_path = os.path.join(ckpt_dir, "config.json")
    config = load_json(config_path) if os.path.exists(config_path) else {}
    max_len = int(config.get("max_len", 256))

    d_model = int(config.get("d_model", 384))
    nhead = int(config.get("nhead", 8))
    num_layers = int(config.get("num_layers", 6))
    d_ff = int(config.get("d_ff", 1536))
    dropout = float(config.get("dropout", 0.1))
    attn_dropout = float(config.get("attn_dropout", 0.1))
    use_attention_pooling = bool(config.get("use_attention_pooling", True))

    model = AdvancedTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(id2label),
        pad_id=vocab["<pad>"],
        max_len=max_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        attn_dropout=attn_dropout,
        use_attention_pooling=use_attention_pooling,
    ).to(device)

    model_path1 = os.path.join(ckpt_dir, "model.pt")
    model_path2 = os.path.join(ckpt_dir, "best.pt")

    if os.path.exists(model_path1):
        state = torch.load(model_path1, map_location=device)
    elif os.path.exists(model_path2):
        obj = torch.load(model_path2, map_location=device)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    else:
        raise FileNotFoundError("找不到 model.pt 或 best.pt")

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, vocab, id2label, max_len


@torch.no_grad()
def local_predict_one(model, vocab, id2label, text: str, max_len: int, device: str):
    ids = numericalize(text, vocab)[:max_len]
    if len(ids) == 0:
        return "EMPTY", 0.0

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
    logits = model(x)[0]  # [C]
    prob = F.softmax(logits, dim=-1)
    pred_id = int(prob.argmax().item())
    return id2label[pred_id], float(prob[pred_id].item()), prob.detach().cpu().numpy().tolist()


# --------------------------
# LLM 分类（LangChain）
# --------------------------
def build_llm(api_key: str, base_url: str, model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )

def _escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")

def llm_classify_one(text: str, labels: list[str], llm):
    # 动态 schema
    class ClsOut(BaseModel):
        label: str = Field(..., description=f"One of: {labels}")

    parser = PydanticOutputParser(pydantic_object=ClsOut)

    fmt = _escape_braces(parser.get_format_instructions())

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict text classifier. "
         "Return ONLY a valid JSON that matches the schema.\n"
         f"{fmt}\n"
         f"Allowed labels: {labels}"
        ),
        ("human",
         "Classify the following text into exactly one label.\n\n"
         "Text:\n{txt}\n"
        )
    ])

    chain = prompt | llm | parser
    out = chain.invoke({"txt": text})
    lab = str(out.label).strip()

    lab_norm = lab.strip().lower()
    label_map = {str(x).lower(): x for x in labels}  

    return lab



# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Text Classification Demo", layout="wide")
st.title("文本分类系统")

DEFAULT_CKPT_DIR = "./runs/agnews_adv_tf1/artifacts"  
device = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.header("设置")

    ckpt_dir = st.text_input("本地模型目录", value=DEFAULT_CKPT_DIR)

    st.divider()
    st.subheader("大模型（LangChain）")

    env_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    api_key = st.text_input("API Key", value=env_key, type="password")

    base_url = st.text_input("Base URL", value=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    llm_model_name = st.text_input("模型名", value=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct"))

    st.caption("提示：Streamlit 可能拿不到你终端里的 export。读不到就会报 `OPENAI_API_KEY` 未设置。")


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("输入文本")
    text = st.text_area("Text", height=220, placeholder="输入一段新闻/文本…")

    run_local = st.button("用本地模型分类", type="primary")
    run_llm = st.button("用大模型（LangChain）分类")

with col2:
    st.subheader("输出")
    if not text.strip():
        st.info("先在左边输入文本，再点击按钮。")
    else:

        try:
            model, vocab, id2label, max_len = load_local_model(ckpt_dir, device)
            labels = id2label 
        except Exception as e:
            st.error(f"本地模型加载失败：{e}")
            st.stop()

        if run_local:
            lab, score, probs = local_predict_one(model, vocab, id2label, text, max_len, device)
            st.success(f"本地模型预测：**{lab}**  | 置信度：**{score:.4f}**")
            st.write("各类别概率：")
            st.json({id2label[i]: float(probs[i]) for i in range(len(id2label))})

        if run_llm:
            if not api_key:
                st.error("没有拿到 API Key：请在侧边栏填写，或设置环境变量 SILICONFLOW_API_KEY / OPENAI_API_KEY")
            else:
                try:
                    llm = build_llm(api_key=api_key, base_url=base_url, model_name=llm_model_name)
                    lab = llm_classify_one(text, labels=labels, llm=llm)
                    st.success(f"大模型预测：**{lab}**")
                except Exception as e:
                    st.error(f"大模型调用失败：{e}")
