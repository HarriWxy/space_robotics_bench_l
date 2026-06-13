import socket
host = "192.168.1.116"
port = 8899
s = socket.socket()
s.settimeout(3)
try:
    s.connect((host, port))
    print("tcp_connect_ok")
except Exception as e:
    print(type(e).__name__, e)
finally:
    s.close()
    
from openpi_client import websocket_client_policy as w
client = w.WebsocketClientPolicy("192.168.1.116", 8899)
print(client.get_server_metadata())