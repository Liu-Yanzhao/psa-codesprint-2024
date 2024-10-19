import logging
import asyncio
import random

from amqtt.client import MQTTClient, ClientException
from amqtt.mqtt.constants import QOS_1, QOS_2
from amqtt.codecs import int_to_bytes_str

logger = logging.getLogger(__name__)
AGV_INSTANCE = 0

async def uptime_coro():
    C = MQTTClient()
    await C.connect("mqtt://auth_handler:auth_handler@127.0.0.1:1883")
    await C.subscribe(
        [
            (f"PATH_SEN/{AGV_INSTANCE}", QOS_1)
        ]
    )
    logger.info("Subscribed")
    try:
        while True:
            battery = random.randint(0, 100)

            message = await C.deliver_message()
            packet = message.publish_packet
            print(
                "%s => %s"
                % (packet.variable_header.topic_name, str(packet.payload.data))
            )

            await C.publish(f"DATA_SEN/{AGV_INSTANCE}", int_to_bytes_str(battery), qos=0x00)  # publishing back any errors to client
        await C.unsubscribe(["$SYS/broker/path"])
        logger.info("UnSubscribed")
        await C.disconnect()
    except ClientException as ce:
        logger.error("Client exception: %s" % ce)


if __name__ == "__main__":
    formatter = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)
    asyncio.get_event_loop().run_until_complete(uptime_coro())