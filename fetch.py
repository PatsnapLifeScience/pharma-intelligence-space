"""
PatSnap MCP 数据采集脚本 — 在你本地 VS Code 里跑
用法:
  1. 把你的 API Key 设成环境变量: export PATSNAP_KEY="sk-xxx"
  2. python fetch.py
  3. 把生成的 real_data.json 内容发给我

需要先装依赖: pip install mcp
"""
import json
import asyncio
import os
import sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# 从命令行参数或环境变量获取 key
if len(sys.argv) > 1:
    API_KEY = sys.argv[1]
elif os.getenv("PATSNAP_KEY", ""):
    API_KEY = os.getenv("PATSNAP_KEY", "")
else:
    print("❌ 用法: python fetch.py YOUR_API_KEY")
    print("   或: set PATSNAP_KEY=你的key && python fetch.py")
    sys.exit(1)

SERVER_URL = f"https://connect.patsnap.com/096456/Logic-mcp?apikey={API_KEY}"

# ============ 要查的内容 ============
# 每个查询定义: (标签, 工具名, 参数)
QUERIES = [
    # --- Pharma Intelligence: 药物搜索 ---
    ("EGFR_drugs", "ls_drug_search", {"target": ["EGFR"], "limit": 15}),
    ("PD1_drugs", "ls_drug_search", {"target": ["PD-1"], "limit": 15}),
    ("NSCLC_drugs", "ls_drug_search", {"disease": ["non-small cell lung cancer"], "limit": 15}),
    ("bispecific_drugs", "ls_drug_search", {"drug_type": ["Bispecific antibody"], "limit": 10}),

    # --- 如果还有可用工具，试试这些 (按需取消注释) ---
    # ("EGFR_target_info", "ls_target_search", {"target": ["EGFR"], "limit": 5}),
    # ("PDL1_clinical_trials", "ls_clinical_trial_search", {"target": ["PD-L1"], "limit": 10}),
]

# ============ 执行 ============
async def fetch_one(label, tool, args):
    print(f"\n{'='*60}")
    print(f"🔍 {label}")
    print(f"   Tool: {tool}")
    print(f"   Args: {json.dumps(args)}")
    
    async with streamablehttp_client(SERVER_URL, timeout=60, sse_read_timeout=60) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 先看看有哪些工具可用
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"   Available tools: {tool_names}")

            if tool not in tool_names:
                print(f"   ⚠️  Tool '{tool}' not found, skipping")
                return None
            
            result = await session.call_tool(tool, arguments=args)
            if result.content:
                text = result.content[0].text
                data = json.loads(text) if isinstance(text, str) else text
                total = data.get("total", "?")
                items = len(data.get("items", []))
                print(f"   ✅ {items} items / {total} total")
                return data
    return None


async def main():
    results = {}
    for label, tool, args in QUERIES:
        try:
            data = await fetch_one(label, tool, args)
            if data:
                results[label] = data
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # 保存
    if results:
        out_path = "real_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"✅ 成功获取 {len(results)}/{len(QUERIES)} 个查询结果")
        print(f"📁 已保存到: {out_path}")
        print(f"📏 文件大小: {os.path.getsize(out_path)} bytes")
        print(f"\n把 real_data.json 文件内容发给青崖即可！")
    else:
        print("\n❌ 没有任何结果，检查 API Key 或网络")

if __name__ == "__main__":
    asyncio.run(main())
