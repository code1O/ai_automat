# ASYNC COMMUNICATION BETWEEN PORTS

import serial
import sqlite3
import socket

conn = sqlite3.connect("")
cursor = conn.cursor()

sock_hostName = socket.gethostname()
IP_ADDR = socket.gethostbyname(sock_hostName)
STREAMER_IPADDR, RECEIVER_IPADDR = IP_ADDR, ...
