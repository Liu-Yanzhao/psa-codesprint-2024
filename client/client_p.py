import logging
import asyncio

from amqtt.client import MQTTClient, ClientException
from amqtt.mqtt.constants import QOS_1, QOS_2
from amqtt.codecs import int_to_bytes_str

logger = logging.getLogger(__name__)

async def uptime_coro():
    c = MQTTClient()
    await c.connect("mqtt://auth_handler:auth_handler@127.0.0.1:1883")  # connect broker client to broker
    while True: 
        await c.publish(f"DATA_SEN/0", int_to_bytes_str("123"), qos=0x00)  # publishing back any errors to client
        await asyncio.sleep(1)

if __name__ == "__main__":
    formatter = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)
    asyncio.get_event_loop().run_until_complete(uptime_coro())