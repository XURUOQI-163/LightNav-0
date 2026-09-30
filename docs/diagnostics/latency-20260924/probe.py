"""Read one demo frame, then benchmark an isolated model session; no robot commands."""
import asyncio,base64,json,time
from pathlib import Path
import aiohttp

async def main():
    rows=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http:
        async with http.get('http://127.0.0.1:8088/api/camera.jpg') as r:
            r.raise_for_status();jpeg=await r.read()
        image=base64.b64encode(jpeg).decode('ascii')
        async with http.ws_connect('http://127.0.0.1:8050',heartbeat=10) as ws:
            await ws.send_json({'action':'login','data':{'clientId':'isolated-latency-diagnostic'}})
            print('login',await ws.receive_json(),flush=True)
            for phase,count in [('growing_history',80),('after_reset',5)]:
                if phase=='after_reset':
                    await ws.send_json({'action':'reset','data':{}})
                    print('reset',await ws.receive_json(),flush=True)
                for step in range(1,count+1):
                    start=time.perf_counter()
                    await ws.send_json({'action':'next','data':{'seq':step,'image':image,'instruction':'move forward, then go to the green trashcan on the right'}})
                    response=await ws.receive_json();data=response['data']
                    if data.get('rc') != 0:raise RuntimeError(response)
                    row={'phase':phase,'step':step,'roundtrip_ms':1000*(time.perf_counter()-start),'server_latency_ms':data.get('latency_ms'),'timings':data.get('timings_ms')}
                    rows.append(row)
                    if step in [1,5,10,20,40,60,80]:print(json.dumps(row),flush=True)
            await ws.send_json({'action':'reset','data':{}});await ws.receive_json()
    Path(__file__).with_name('results.json').write_text(json.dumps(rows,indent=2))
asyncio.run(main())
