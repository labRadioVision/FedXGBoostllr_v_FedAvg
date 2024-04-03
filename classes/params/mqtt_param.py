# MAIN CONFIGURATION PARAMETERS ##############################
import os
import platform
import sys
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from pathlib import Path
import json


OS = platform.system()

print('\n' + OS)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if OS != 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1 to not use GPU

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if OS == 'Windows':
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096*1)])
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


REMOTE = 0  # Test local or remote
TLS_USAGE = 0
LAUNCH_MQTT_BROKER = 0

cwd = os.path.split(os.path.abspath(__file__))[0]  # os.getcwd()           
cwd = str(Path(Path(cwd).parent.absolute()).parent.absolute())
save_weights = True                                               

# Directories for WINDOWS
python_conda_dir = sys.executable 
MQTT_broker_dir = '"C:\\Program Files\mosquitto\mosquitto.exe"'
MQTT_broker_config_file = '"C:\\Program Files\mosquitto\mosquitto.conf"'


# Environment
try:
    conda_env = os.environ['CONDA_DEFAULT_ENV'] 
except:
    conda_env = sys.executable.split(os.sep)[3]

# MQTT Topics and Broker address
# localhost ='127.0.0.1'  # '128.141.183.190'
if  REMOTE:
    with open(os.path.join(cwd, 'MQTT_broker_config.json')) as f:
        config = json.load(f)
else:
    with open(os.path.join(cwd, 'MQTT_broker_config_local.json')) as f:
        config = json.load(f)    

ADDRESS = config['broker_address']
# broker_address = localhost
PORT = config['MQTT_port']   # 11883(linux cern)

#### QOS MQTT
QOS = 2

### Compression
COMPRESSION_ENABLED = True

if not TLS_USAGE:
    TLS_ = None       
    AUTH_ = None   
else:
    TLS_ = config['TLS_']       
    TLS_['tls_version'] = mqtt.ssl.PROTOCOL_TLS
    AUTH_ = config['AUTH_']

################ MQTT TOPICS ###############
server_weights_topic = config["Name_of_federation"] + 'server_weights' # POSTING GLOBAL MODELS TO INDIVIDUAL CLIENTS
client_weights_topic = config["Name_of_federation"] + 'client_weights'# POSTING LOCAL MODELS (SHARED TOPIC AMONG ALL CLIENTS)