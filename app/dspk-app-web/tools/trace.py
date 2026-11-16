"""Where a page's threads spend native time (Chrome tracing over the DevTools protocol).
usage: trace.py <url> <seconds idle> <seconds interacting> [port]"""
import asyncio, json, sys, urllib.request, collections
import websockets

url, idle, busy = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
port = int(sys.argv[4]) if len(sys.argv) > 4 else 9222

async def main():
    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
    page = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        n = 0
        events = []
        done = asyncio.Event()
        async def call(method, **params):
            nonlocal n
            n += 1
            mine = n
            await ws.send(json.dumps({"id": mine, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("method") == "Tracing.dataCollected":
                    events.extend(msg["params"]["value"])
                if msg.get("method") == "Tracing.tracingComplete":
                    done.set()
                if msg.get("id") == mine:
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg.get("result", {})
        await call("Page.enable")
        await call("Page.navigate", url=url)
        async def pump(seconds):
            end = asyncio.get_event_loop().time() + seconds
            while asyncio.get_event_loop().time() < end:
                await call("Runtime.evaluate", expression="1")
                await asyncio.sleep(0.25)
        await pump(idle)
        await call("Tracing.start", traceConfig={"includedCategories": ["toplevel", "devtools.timeline", "disabled-by-default-devtools.timeline", "gpu", "blink", "v8", "cc", "viz"], "recordMode": "recordAsMuchAsPossible"}, transferMode="ReportEvents")
        await call("Runtime.evaluate", expression="window.__f=0; (function t(){window.__f++; requestAnimationFrame(t)})(); window.__t0=performance.now()")
        end = asyncio.get_event_loop().time() + busy
        k = 0
        while asyncio.get_event_loop().time() < end:
            k += 1
            await call("Input.dispatchMouseEvent", type="mouseWheel", x=640, y=330, deltaX=0, deltaY=(120 if k % 40 < 20 else -120))
            await call("Input.dispatchMouseEvent", type="mouseMoved", x=500 + (k * 7) % 300, y=300 + (k * 3) % 100)
            await asyncio.sleep(0.03)
        frames = await call("Runtime.evaluate", expression="JSON.stringify({frames: window.__f, ms: performance.now()-window.__t0})")
        print("frames:", frames["result"].get("value"))
        await call("Tracing.end")
        while not done.is_set():
            msg = json.loads(await ws.recv())
            if msg.get("method") == "Tracing.dataCollected":
                events.extend(msg["params"]["value"])
            if msg.get("method") == "Tracing.tracingComplete":
                done.set()
    names = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            names[(e["pid"], e["tid"])] = e["args"]["name"]
    per = collections.defaultdict(collections.Counter)
    for e in events:
        if e.get("ph") == "X" and "dur" in e:
            per[names.get((e["pid"], e["tid"]), "?")][e["name"]] += e["dur"]
    for thread, counter in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:7]:
        print(f"=== {thread}")
        for name, us in counter.most_common(16):
            print(f"   {us/1e3:9.0f} ms  {name}")

asyncio.run(main())
