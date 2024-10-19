import os
import json
import logging
import asyncio
from amqtt.broker import Broker
from amqtt.client import MQTTClient

# Importing important server functions
from amqtt.codecs import int_to_bytes_str
from amqtt.mqtt.constants import QOS_1, QOS_2

# Configuration for AMQTT server
config = {
    "listeners": {
        "default": {
            "bind": "127.0.0.1:1883",
            "type": "tcp",
            "max_connections": 10,
        },
    },
    "auth": {
        "allow-anonymous": False,
        "plugins": ["auth_file", "auth_anonymous"],
        "password-file": os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "password.txt"
        ),
    },
    "topic-check": {
        "enabled": True,
        "plugins": ["topic_acl", "topic_taboo"],
        "acl": {
            "auth_handler": ["PATH_SEN/#", "DATA_SEN/#"],
        },
    },
}

# creating amqtt broker class
broker = Broker(config)

async def broker_coro():
    """
    asynchronous function to subscribed to and publish AMQTT messages
    """
    await broker.start()  # wait for broker to start
    
    # try:
    #     c = MQTTClient()
    #     await c.connect("mqtt://auth_handler:auth_handler@127.0.0.1:1883")  # connect broker client to broker
    #     await c.subscribe([  # subscribe to 5 topics
    #         ("DATA_REC/#", QOS_1),

    #         ])
    #     while True:  # loop to continuously receive process and send messages
    #         message = await c.deliver_message() # wait until message is received
    #         packet = message.publish_packet  # get packet payload
    #         topic_name = packet.variable_header.topic_name  # get MQTT topic
    #         payload = str(packet.payload.data)[12:-2]  # get payload

    #         if topic_name[0:9] == "DATA_REC/":  # if topic is requesting to change payload
    #             new_data = json.loads(payload)  # turning payload into a json
    #             print(new_data)
                
    #             # await c.publish(f"DATA_ERROR/{topic_name[9:14]}", int_to_bytes_str(error), qos=0x00)  # publishing back any errors to client


    # except ConnectionError:  # if there is any connection error
    #     asyncio.get_event_loop().stop()  # stop server

# Run code
if __name__ == '__main__':
    formatter = "[%(asctime)s] :: %(levelname)s :: %(name)s :: %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)  # for logging
    asyncio.get_event_loop().run_until_complete(broker_coro())  # to run asyncronous function forever
    asyncio.get_event_loop().run_forever()  # to run asyncronous function forever