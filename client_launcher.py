import os

from aicrowd_gym.clients.zmq_oracle_client import ZmqOracleClient
from agents.orderenforcingwrapper import OrderEnforcingAgent
from aicrowd_gym.serializers import MessagePackSerializer


def start_test_client():
    HOST = os.getenv("AICROWD_REMOTE_SERVER_HOST", "0.0.0.0")
    PORT = int(os.getenv("AICROWD_REMOTE_SERVER_PORT", "5000"))

    client = ZmqOracleClient(host=HOST,
                             port=PORT,
                             serializer=MessagePackSerializer())
    agent = OrderEnforcingAgent()
    client.register_agent(agent)
    client.run_agent()

if __name__ == '__main__':
    start_test_client()

