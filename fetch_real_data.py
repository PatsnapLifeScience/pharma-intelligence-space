"""抓取 PatSnap MCP 真实数据，替换 HF Space demo 的 mock"""
import json
import asyncio
import sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

API_KEY = sys.argv[1] if len(sys.argv) > 1 else ""
SERVER_URL = f"https://connect.patsnap.com/096456/Logic-mcp?apikey={API_KEY}"

QUERIES = [
    ("EGFR", {"target": ["EGFR"], "limit": 10}),
    ("PD-1", {"target": ["PD-1"], "limit": 10}),
    ("NSCLC", {"disease": ["non-small cell lung cancer"], "limit": 10}),
]

async def fetch(label, args):
    print(f"\n🔍 Fetching: {label} ...")
    async with streamablehttp_client(SERVER_URL, timeout=30, sse_read_timeout=30) as (r, w, _):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            res = await sess.call_tool("ls_drug_search", arguments=args)
            if res.content:
                txt = res.content[0].text
                data = json.loads(txt) if isinstance(txt, str) else txt
                print(f"  ✅ Got {data.get('total', 0)} records, {len(data.get('items', []))} items")
                return data
    return None

async def main():
    if not API_KEY:
        print("❌ No API key. Usage: python fetch_real_data.py YOUR_KEY")
        return

    results = {}
    for label, args in QUERIES:
        try:
            data = await fetch(label, args)
            if data:
                results[label] = data
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    if results:
        with open("real_data.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(results)} result sets to real_data.json")
    else:
        print("\n❌ No data fetched.")

asyncio.run(main())
