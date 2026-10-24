"""Connect the installed NVIDIA viewer to the local simulator and maximize it."""
import json
import time
import urllib.request
import urllib.parse
import websocket

def main():
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        try:
            pages=json.load(urllib.request.urlopen('http://127.0.0.1:9231/json/list',timeout=2))
            page=next(p for p in pages if p['type']=='page')
            break
        except (OSError,StopIteration):time.sleep(.25)
    else:raise TimeoutError('NVIDIA desktop viewer did not open')
    socket=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=10,suppress_origin=True)
    index=0
    def call(method,params):
        nonlocal index
        index+=1;socket.send(json.dumps({'id':index,'method':method,'params':params}))
        while True:
            reply=json.loads(socket.recv())
            if reply.get('id')==index:
                if reply.get('error'):raise RuntimeError(reply['error'])
                return reply.get('result',{})
    try:
        url=page['url'].split('?')[0]
        if not url.startswith('file:') or '/renderer/index.html' not in url:
            raise ValueError('Unexpected NVIDIA viewer page')
        call('Page.navigate',{'url':url+'?server=127.0.0.1&autoConnect=1'})
        # Some Electron viewer builds expose only the page CDP domain.
        # Window sizing is optional; connecting and receiving frames is required.
        try:
            window=call('Browser.getWindowForTarget',{})
            call('Browser.setWindowBounds',{'windowId':window['windowId'],'bounds':{'windowState':'maximized'}})
        except RuntimeError:
            pass
        for _ in range(100):
            result=call('Runtime.evaluate',{'expression':"[...document.querySelectorAll('video')].some(v=>v.readyState>=2 && v.videoWidth>0)",'returnByValue':True})
            if result.get('result',{}).get('value'):
                print('HOUSEHOLD_VIEWER_CONNECTED',flush=True);return
            time.sleep(.5)
        raise TimeoutError('NVIDIA viewer received no simulation frames')
    finally:socket.close()
if __name__=='__main__':main()
