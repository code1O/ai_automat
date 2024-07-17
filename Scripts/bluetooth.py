import asyncio
from bleak import BleakScanner, BleakClient

async def get_devices():
    addresses = []
    names = []
    devices = await BleakScanner.discover()
    for d in devices:
        addresses.append(d.address)
        names.append(d.name)
    dictionary = dict(address=addresses, name=names)
    return dictionary

async def connect(address, device_name):
    Client = BleakClient(address)
    try:
        print("Trying connecting")
        await Client.connect()
        is_connected = Client.is_connected
        if is_connected:
            print("Connected to {}".format(device_name))
        else:
            print("Failed to connect to {}".format(device_name))
    except Exception as e:
        print(e)

async def close_connection(address):
    Client = BleakClient(address)
    await Client.disconnect()