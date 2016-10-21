__author__ = 'MA573RWARR10R'
import socket
import os
from rec_settings import *
# Lets imitate server code to send image as binary data to client

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
client_socket.connect((host, port))

while True:
    filename = raw_input('Input filename: ')

    if os.path.isfile(filename):
        filesize = os.path.getsize(filename)
        client_socket.send(str(filesize))

        with open(filename, 'rb') as fp:
            print 'sending...'
            client_socket.send(fp.read(filesize))
        fp.close()

        print 'send!'

    else:
        print 'wrong path!'

    predicted_data = client_socket.recv(buffer_size)

    if predicted_data is not None:
        print predicted_data

