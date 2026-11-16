"""Opens a page of `dspk serve --web`, answers its pairing prompt, waits, saves a screenshot.
usage: pair.py <url> <code> <seconds> <screenshot.png> [port]"""
import asyncio, base64, json, sys, urllib.request, collections
import websockets

url, code, wait, shot = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
port = int(sys.argv[5]) if len(sys.argv) > 5 else 9225

async def main():
    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
    page = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        n = 0
        logs = []
        dialogs = []
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
                m = msg.get("method")
                if m == "Page.javascriptDialogOpening":
                    dialogs.append(msg["params"]["message"])
                    n += 1
                    await ws.send(json.dumps({"id": n, "method": "Page.handleJavaScriptDialog", "params": {"accept": True, "promptText": code}}))
                if m == "Target.attachedToTarget":
                    n += 1
                    await ws.send(json.dumps({"id": n, "method": "Runtime.enable", "sessionId": msg["params"]["sessionId"]}))
                if m == "Runtime.consoleAPICalled":
                    logs.append(" ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:400])
                if m == "Runtime.exceptionThrown":
                    logs.append("EXC " + json.dumps(msg["params"]["exceptionDetails"].get("exception", {}).get("description", ""))[:2600])
        await call("Runtime.enable")
        await call("Page.enable")
        await call("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=False, flatten=True)
        await call("Page.navigate", url=url)
        end = asyncio.get_event_loop().time() + wait
        while asyncio.get_event_loop().time() < end:
            await call("Runtime.evaluate", expression="1")
            await asyncio.sleep(0.25)
        token = await call("Runtime.evaluate", expression="(localStorage.getItem('dspk-token')||'').length")
        print("dialogs:", dialogs, "| token length:", token["result"].get("value"))
        png = (await call("Page.captureScreenshot"))["data"]
        open(shot, "wb").write(base64.b64decode(png))
        for l, k in collections.Counter(x[:2600] for x in logs).most_common(25):
            if "deprecated parameters" not in l:
                print(k, l)

asyncio.run(main())
