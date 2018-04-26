#!/usr/bin/env python

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import tensorflow as tf
from pygen.example import Example


class ExampleHandler:
    def __init__(self):
        self.log = {}

    def ping(self):
        return "pong"

    def derek(self):
        return "derek was here"

    def tensorflow_hello(self):
        hello = tf.constant('Hello, TensorFlow!')
        sess = tf.Session()
        return "tensorflow works?: " + str(sess.run(hello))

    def say(self, msg):
        print(msg)


handler = ExampleHandler()
processor = Example.Processor(handler)
transport = TSocket.TServerSocket(port=30303)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

print("Starting python server...")
server.serve()
print("done!")
