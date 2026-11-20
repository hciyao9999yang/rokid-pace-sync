import io
import os
import re
import json
import time
import random
import base64
import asyncio
from PIL import Image
from openai import AsyncOpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI(title="Rokid 眼镜配速与补给同步服务")

# 默认兼容 MiniMax M3，部署到 Zeabur 时请在 Variables 中设置 VLM_API_KEY
client = AsyncOpenAI(
    api_key=os.getenv("VLM_API_KEY", ""),
    base_url=os.getenv("VLM_BASE_URL", "https://api.minimaxi.com/v1")
)
MODEL_NAME = os.getenv("VLM_MODEL", "MiniMax-M3")

# 内存缓存：{ "839201": {"expire_at": 1730000000, "data": {...}} }
PLAN_STORE: dict[str, dict] = {}

SYSTEM_PROMPT = """
你是马拉松配速与补给数据提取助手。请从微信小程序配速长图（或局部切片）中提取结构化数据：
1. 配速统一换算为【每公里整数秒 (target_pace_sec)】，如 5'05" 或 5:05 换算为 305。
2. 公里段 (start_km, end_km) 为浮点数。如果是逐公里表格（如第1公里、第2公里），则 start_km=0, end_km=1；如果是分段表格（如 0-5km），则 start_km=0, end_km=5。
3. 补给信息提取到 fuel_events 中（trigger_km, item, note），明确触发的公里点、补给物品和服用说明；如果是赛前补给没有具体公里数，trigger_km 填 0。
4. 若切片顶部或底部有被截断的残缺行，直接忽略该残缺行。
5. 严格输出纯 JSON，不要包含任何解释文字或 Markdown 标记。格式如下：
{
  "race_name": "赛事名称",
  "total_distance_km": 21.0975,
  "target_time_sec": 6300,
  "segments": [{"start_km": 0, "end_km": 5, "target_pace_sec": 300, "hr_zone": "Z2", "elevation_note": ""}],
  "fuel_events": [{"trigger_km": 7.5, "item": "1支能量胶", "note": "配合清水"}]
}
"""


def safe_float(val, default: float = 0.0) -> float:
    """安全转换数字，兼容模型返回的 null 或非数字字符串"""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def slice_long_image(image_bytes: bytes, slice_height: int = 1400, overlap: int = 200) -> list[str]:
    """将长图按纵向滑动窗口切分为多张 Base64 图片，防止缩放模糊和行截断"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    if height <= slice_height * 1.3:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return [base64.b64encode(buf.getvalue()).decode("utf-8")]

    slices = []
    step = slice_height - overlap
    for top in range(0, height, step):
        bottom = min(top + slice_height, height)
        cropped = img.crop((0, top, width, bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        slices.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        if bottom == height:
            break
    return slices


async def parse_single_slice(b64_img: str) -> dict:
    """调用多模态大模型解析单张切片，兼容推理思考标签清洗"""
    if not os.getenv("VLM_API_KEY"):
        raise HTTPException(status_code=500, detail="服务端未配置 VLM_API_KEY 环境变量")

    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请提取这张配速策略长图中的结构化数据，直接输出纯 JSON。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        temperature=0.1
    )
    raw_text = resp.choices[0].message.content or ""

    # 1. 过滤掉推理模型可能输出的 <think>...</think> 思考过程
    clean_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()

    # 2. 提取最外层的 JSON 对象（兼容带 ```json 标记的情况）
    match = re.search(r"\{[\s\S]*\}", clean_text)
    if not match:
        raise ValueError(f"模型未返回有效 JSON: {clean_text[:200]}")

    return json.loads(match.group(0))


async def extract_full_plan(image_bytes: bytes) -> dict:
    """并发识别所有切片并去重合并（带空值容错）"""
    slices = slice_long_image(image_bytes)
    results = await asyncio.gather(*(parse_single_slice(s) for s in slices))

    merged = {
        "race_name": "",
        "total_distance_km": 0.0,
        "target_time_sec": 0,
        "segments": [],
        "fuel_events": []
    }
    seen_segs, seen_fuels = set(), set()

    for res in results:
        if not merged["race_name"] and res.get("race_name"):
            merged["race_name"] = str(res["race_name"])

        dist = safe_float(res.get("total_distance_km"), 0.0)
        if dist > merged["total_distance_km"]:
            merged["total_distance_km"] = dist

        t_sec = int(safe_float(res.get("target_time_sec"), 0))
        if t_sec > merged["target_time_sec"]:
            merged["target_time_sec"] = t_sec

        for seg in res.get("segments") or []:
            if seg.get("start_km") is None and seg.get("end_km") is None:
                continue
            s_km = round(safe_float(seg.get("start_km"), 0.0), 2)
            e_km = round(safe_float(seg.get("end_km"), s_km), 2)
            pace = int(safe_float(seg.get("target_pace_sec"), 0))
            if pace <= 0:
                continue

            key = (s_km, e_km)
            if key not in seen_segs:
                seen_segs.add(key)
                merged["segments"].append({
                    "start_km": s_km,
                    "end_km": e_km,
                    "target_pace_sec": pace,
                    "hr_zone": seg.get("hr_zone") or "",
                    "elevation_note": seg.get("elevation_note") or ""
                })

        for fuel in res.get("fuel_events") or []:
            item_name = str(fuel.get("item") or "").strip()
            if not item_name:
                continue
            t_km = round(safe_float(fuel.get("trigger_km"), 0.0), 2)
            key = (t_km, item_name)
            if key not in seen_fuels:
                seen_fuels.add(key)
                merged["fuel_events"].append({
                    "trigger_km": t_km,
                    "item": item_name,
                    "note": fuel.get("note") or ""
                })

    merged["segments"].sort(key=lambda x: x["start_km"])
    merged["fuel_events"].sort(key=lambda x: x["trigger_km"])
    return merged


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/upload-plan")
async def upload_plan(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片请控制在 10MB 以内")

    try:
        plan = await extract_full_plan(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")

    # 清理过期数据并生成 6 位不重复提取码（有效期 24 小时）
    now = time.time()
    expired_keys = [k for k, v in PLAN_STORE.items() if v["expire_at"] < now]
    for k in expired_keys:
        del PLAN_STORE[k]

    code = f"{random.randint(100000, 999999)}"
    while code in PLAN_STORE:
        code = f"{random.randint(100000, 999999)}"

    PLAN_STORE[code] = {"expire_at": now + 86400, "data": plan}
    return {"code": code, "plan": plan}


@app.get("/api/plan/{code}")
async def get_plan(code: str):
    item = PLAN_STORE.get(code)
    if not item or item["expire_at"] < time.time():
        raise HTTPException(status_code=404, detail="提取码不存在或已过期")
    return item["data"]
