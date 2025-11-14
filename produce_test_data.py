from confluent_kafka import Producer
import socket
import json 

conf = {'bootstrap.servers': 'localhost:29092',
        'client.id': socket.gethostname()}

producer = Producer(conf)

producer.produce("analytics", key="key", value=json.dumps({
    "device_id": "id",
    "service_id": "analytics",
    "Humidity": 10,
    "Humidity_Time": "2024-04-22T14:00:25.737774",
    "time": "2024-04-23T14:00:25.737774"
}))
producer.flush()
