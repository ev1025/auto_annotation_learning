# -*- coding: utf-8 -*-
"""공유기에 이 장비의 포트(7862 대시보드 · 9412 추론API)를 열어준다.

Thor 는 공유기 뒤(192.168.0.x)에 있고 보통 22번만 포워딩돼 있어서, 사무실 망의 다른 PC 는
대시보드에 못 들어온다. 이 스크립트는 관리자 페이지 로그인 없이 UPnP(공유기의 자동 포트개방
기능)로 규칙을 만든다. 공유기·내부IP·제어주소를 모두 자동으로 찾으므로 어느 Thor 에서든 같다.

  반드시 그 Thor 안에서 실행(같은 내부망에 있어야 공유기가 요청을 받는다):
    python3 deploy/router_portmap.py add     # 규칙 추가
    python3 deploy/router_portmap.py show    # 현재 상태
    python3 deploy/router_portmap.py del     # 규칙 제거

주의: UPnP 로 만든 규칙은 공유기를 재부팅하면 사라진다. 영구히 두려면 공유기 관리자 페이지
      (http://192.168.0.1 -> 고급 설정 -> NAT/라우터 관리 -> 포트포워드 설정)에 같은 내용을
      수동 등록할 것. 그때는 이 스크립트로 UPnP 규칙을 먼저 지워 외부 포트 충돌을 피한다.
"""
import re
import socket
import sys
import urllib.request as u
from urllib.parse import urljoin

PORTS = [7862, 9412]          # 대시보드 · 추론 API
SSDP_ADDR, SSDP_PORT = "239.255.255.250", 1900


def my_ip():
    """공유기로 나가는 인터페이스의 주소(= 포워딩 목적지)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]
    finally:
        s.close()


def discover():
    """SSDP 로 공유기(IGD)의 rootDesc 주소를 찾는다. 공유기마다 포트가 달라 하드코딩하면 안 된다."""
    msg = ("M-SEARCH * HTTP/1.1\r\n"
           "HOST:%s:%d\r\n"
           'MAN:"ssdp:discover"\r\n'
           "MX:2\r\n"
           "ST:urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n" % (SSDP_ADDR, SSDP_PORT))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(4)
    s.sendto(msg.encode(), (SSDP_ADDR, SSDP_PORT))
    try:
        while True:
            data, _ = s.recvfrom(2048)
            for line in data.decode("utf-8", "ignore").splitlines():
                if line.lower().startswith("location:"):
                    return line.split(":", 1)[1].strip()
    except socket.timeout:
        pass
    finally:
        s.close()
    raise SystemExit("공유기 UPnP 응답 없음 - 관리자 페이지에서 수동 등록해야 한다")


def service(root):
    """rootDesc 에서 포트포워딩을 담당하는 서비스(WANIPConnection)의 제어 주소를 뽑는다."""
    xml = u.urlopen(root, timeout=6).read().decode("utf-8", "ignore")
    for m in re.finditer(r"<service>(.*?)</service>", xml, re.S):
        b = m.group(1)
        if "WANIPConnection" in b or "WANPPPConnection" in b:
            st = re.search(r"<serviceType>(.*?)</serviceType>", b).group(1)
            cu = re.search(r"<controlURL>(.*?)</controlURL>", b).group(1)
            # controlURL 은 절대·상대(/ctl/IPConn) 둘 다 온다. 손으로 이어붙이면 틀린다.
            return st, urljoin(root, cu)
    raise SystemExit("WANIPConnection 서비스를 찾지 못했다")


ROOT = discover()
ST, URL = service(ROOT)
ME = my_ip()


def soap(action, body=""):
    env = ('<?xml version="1.0"?>'
           '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
           's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
           '<s:Body><u:%s xmlns:u="%s">%s</u:%s></s:Body></s:Envelope>' % (action, ST, body, action))
    req = u.Request(URL, data=env.encode(),
                    headers={"Content-Type": 'text/xml; charset="utf-8"',
                             "SOAPAction": '"%s#%s"' % (ST, action)})
    return u.urlopen(req, timeout=8).read().decode("utf-8", "ignore")


def _one(port):
    return ("<NewRemoteHost></NewRemoteHost><NewExternalPort>%d</NewExternalPort>"
            "<NewProtocol>TCP</NewProtocol>" % port)


def add(port):
    soap("AddPortMapping",
         _one(port) +
         "<NewInternalPort>%d</NewInternalPort>"
         "<NewInternalClient>%s</NewInternalClient>"
         "<NewEnabled>1</NewEnabled>"
         "<NewPortMappingDescription>xr-%d</NewPortMappingDescription>"
         "<NewLeaseDuration>0</NewLeaseDuration>" % (port, ME, port))


def show(port):
    r = soap("GetSpecificPortMappingEntry", _one(port))
    g = lambda t: (re.search("<%s>(.*?)<" % t, r) or [None, "?"])[1]
    return "%s:%s (%s)" % (g("NewInternalClient"), g("NewInternalPort"), g("NewPortMappingDescription"))


def delete(port):
    soap("DeletePortMapping", _one(port))


def wan_ip():
    try:
        return re.search(r"<NewExternalIPAddress>(.*?)<", soap("GetExternalIPAddress")).group(1)
    except Exception:
        return "?"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "show"
    print("공유기 %s · 외부IP %s · 이 장비 %s" % (ROOT.split("/")[2], wan_ip(), ME))
    for p in PORTS:
        try:
            if mode == "add":
                add(p)
                print("  추가  외부 %d -> %s" % (p, show(p)))
            elif mode == "del":
                delete(p)
                print("  삭제  %d" % p)
            else:
                print("  조회  외부 %d -> %s" % (p, show(p)))
        except Exception as e:
            msg = ""
            if hasattr(e, "read"):
                body = e.read().decode("utf-8", "ignore")
                m = re.search(r"<errorDescription>(.*?)<", body)
                msg = m.group(1) if m else body[:120]
            print("  실패  %d: %s %s" % (p, type(e).__name__, msg))
