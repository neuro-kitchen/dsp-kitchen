"""Profiles a page's main thread in headless Chrome over the DevTools protocol.
usage: prof.py <url> <seconds idle> <seconds interacting> <screenshot.png> [port]"""
import asyncio, base64, json, sys, urllib.request, collections
import websockets

url, idle, busy, shot = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
port = int(sys.argv[5]) if len(sys.argv) > 5 else 9222

async def main():
    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
    page = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        n = 0
        logs = []
        workers = []
        async def call(method, **params):
            nonlocal n
            n += 1
            mine = n
            await ws.send(json.dumps({"id": mine, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mine:
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg.get("result", {})
                if msg.get("method") == "Target.attachedToTarget":
                    info = msg["params"]["targetInfo"]
                    workers.append(info.get("type", "?") + " " + info.get("url", "")[-60:])
                    n += 1
                    await ws.send(json.dumps({"id": n, "method": "Runtime.enable", "sessionId": msg["params"]["sessionId"]}))
                if msg.get("method") == "Runtime.consoleAPICalled":
                    logs.append(" ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:300])
                if msg.get("method") == "Runtime.exceptionThrown":
                    logs.append("EXC " + json.dumps(msg["params"]["exceptionDetails"].get("exception", {}))[:3000]); print(logs[-1])
        await call("Runtime.enable")
        await call("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=False, flatten=True)
        await call("Page.enable")
        await call("Profiler.enable")
        await call("Profiler.setSamplingInterval", interval=500)
        await call("Page.navigate", url=url)
        async def pump(seconds):
            end = asyncio.get_event_loop().time() + seconds
            while asyncio.get_event_loop().time() < end:
                await call("Runtime.evaluate", expression="1")
                await asyncio.sleep(0.25)
        await pump(idle)
        info = await call("Runtime.evaluate", expression="JSON.stringify({iso: self.crossOriginIsolated, gpu: !!navigator.gpu, cores: navigator.hardwareConcurrency, w: innerWidth, h: innerHeight})")
        print("page:", info["result"].get("value"))
        await call("Profiler.start")
        # Frames the page gets while it is interacted with
        await call("Runtime.evaluate", expression="window.__f=0; window.__long=0; window.__last=performance.now(); (function t(){const n=performance.now(); if(n-window.__last>50) window.__long++; window.__last=n; window.__f++; requestAnimationFrame(t)})(); window.__t0=performance.now()")
        end = asyncio.get_event_loop().time() + busy
        k = 0
        while asyncio.get_event_loop().time() < end:
            k += 1
            # Wheel over the traces (pan in time), then drag
            await call("Input.dispatchMouseEvent", type="mouseWheel", x=640, y=330, deltaX=0, deltaY=(120 if k % 40 < 20 else -120), modifiers=(8 if mode_shift else 0))
            await call("Input.dispatchMouseEvent", type="mouseMoved", x=500 + (k * 7) % 300, y=300 + (k * 3) % 100)
            await asyncio.sleep(0.03)
        frames = await call("Runtime.evaluate", expression="JSON.stringify({frames: window.__f, long: window.__long, ms: performance.now()-window.__t0})")
        print("frames:", frames["result"].get("value"))
        prof = (await call("Profiler.stop"))["profile"]
        png = (await call("Page.captureScreenshot"))["data"]
        open(shot, "wb").write(base64.b64decode(png))
        nodes = {nd["id"]: nd for nd in prof["nodes"]}
        self_us = collections.Counter()
        deltas = prof.get("timeDeltas", [])
        for node_id, dt in zip(prof.get("samples", []), deltas):
            f = nodes[node_id]["callFrame"]
            self_us[(f["functionName"] or "(anonymous)")[:150]] += dt
        total = sum(self_us.values()) or 1
        print(f"profile: {total/1e6:.1f} s sampled")
        for name, us in self_us.most_common(25):
            print(f"{us/total*100:5.1f}%  {us/1e3:8.0f} ms  {name}")
        # Inclusive time of a few frames of interest
        children = {i: nd.get("children", []) for i, nd in nodes.items()}
        incl = collections.Counter()
        own = collections.Counter()
        for node_id, dt in zip(prof.get("samples", []), deltas):
            own[node_id] += dt
        def total_of(i):
            return own[i] + sum(total_of(c) for c in children[i])
        sys.setrecursionlimit(100000)
        byname = collections.Counter()
        def walk(i, seen):
            name = nodes[i]["callFrame"]["functionName"][:150]
            t = None
            if name and name not in seen:
                t = total_of(i)
                byname[name] += t
                seen = seen | {name}
            for c in children[i]:
                walk(c, seen)
        walk(prof["nodes"][0]["id"], frozenset())
        print("--- inclusive")
        for name, us in byname.most_common(int(sys.argv[6]) if len(sys.argv) > 6 else 45):
            print(f"{us/total*100:5.1f}%  {us/1e3:8.0f} ms  {name}")
        print("--- workers", len(workers), workers[:4])
        print("--- console", len(logs))
        import collections as c2
        for l, k in c2.Counter(x[:160] for x in logs).most_common(30):
            print(k, l)

mode_shift = False
asyncio.run(main())
