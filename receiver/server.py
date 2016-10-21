__author__ = 'MA573RWARR10R'
import socket
from nn import capdecoder as net
from rec_settings import *
import os
from settings import *

ccracker = net.CaptchaCracker(model_params_file_path='../train_data_is_in/m/18_10_2016.npz')

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
server_socket.bind((host, port))
server_socket.listen(5)


while True:
    print "Wait..."
    client_socket, addr = server_socket.accept()
    if client_socket:
        print "Connected to - ", addr, "\n"
        filesize = int(client_socket.recv(1024))

        with open(temp_dir + temp_name, 'wb+') as fp:
            print 'Receiving...'
            l = client_socket.recv(filesize)
            fp.write(l)
        fp.close()

        print "Got!"

        predicted_chars = net.predict(ccracker, temp_dir + temp_name)

        result = ""
        for char in predicted_chars:
            result += char

        client_socket.send(result)
        os.remove(temp_dir + temp_name)
        os.remove(edged_captcha_path + temp_name)

        client_socket.close()
