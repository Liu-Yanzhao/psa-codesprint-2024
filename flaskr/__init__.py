from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
import multiprocessing as mp
from queue import Empty
import logging
import asyncio
import json
import random

import tensorflow as tf
from tensorflow import keras
import numpy as np

import requests
import schedule
import time
from datetime import datetime

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

from amqtt.client import MQTTClient, ClientException
from amqtt.mqtt.constants import QOS_1, QOS_2
from amqtt.codecs import int_to_bytes_str

from .pathfinding import Spot, astar_3d
from .pathfinding import create_grid_and_nodes, compute_neighbors, update_neighbors
from .sprite import Sprite
from .var import SIZE, ROWS, COLS, SCROLL_SPEED, WIDTH, HEIGHT, TOTAL_HEIGHT, TOTAL_WIDTH, SCROLL_X, SCROLL_Y, TIME
from datetime import datetime

logger = logging.getLogger(__name__)
nodes = None
all_sprites = []
sprite_coord = []
run = True
cycles_elapsed = 0

NUM_AGVS = 10
MAX_TASKS = 20
MAX_BATTERY = 100

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
RAINFALL_THRESHOLD = 2.0  # Moderate rain threshold (in mm)

def get_rainfall_data():
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    params = {"date": current_time}
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad responses
        data = response.json()
 
        for i in data['data']['stations']:
            if i['name'] == 'Tuas South Avenue 3':
                station_id = i['id']
        latest_readings = data['data']['readings'][0]['data']
        for latest_reading in latest_readings:
            if latest_reading['stationId'] == station_id:
                rainfall_value = latest_reading['value']
        return rainfall_value
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def check_rainfall():
    rainfall = get_rainfall_data()
    if rainfall is not None:
        output = 1 if rainfall >= RAINFALL_THRESHOLD else 0
        return output
    else:
        print("Failed to retrieve rainfall data")
        return None

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['month'] = pd.to_datetime(df['month'])
    df['months_since_start'] = (df['month'] - df['month'].min()).dt.days / 30.44
    return df

def create_features_and_target(df):
    X = df[['months_since_start']]
    y = df['number_of_vessels']
    return X, y

def create_model(X, y, poly_degree=6, alpha=30):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X_scaled)
    
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    
    return model, scaler, poly

def predict_vessel_arrival(input_date, df, model, scaler, poly, min_vessels, max_vessels):
    months_since_start = (input_date - df['month'].min()).days / 30.44
    months_since_start_scaled = scaler.transform([[months_since_start]])
    user_poly = poly.transform(months_since_start_scaled)
    
    prediction = model.predict(user_poly)[0]
    normalized_prediction = (prediction - min_vessels) / (max_vessels - min_vessels)
    
    return normalized_prediction, prediction, months_since_start

def evaluate_model(model, X_poly, y):
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Model Performance:\nMean Squared Error (MSE): {mse:.2f}\nR-squared (R²): {r2:.4f}")

def get_demand():
    file_path = "2"
    if file_path == '1':
        file_path = 'AIS_VesselArrivalsTotalMonthly_SAMPLE.csv'
    else:
        file_path = 'VesselArrivals75GTTotalMonthly.csv'
    df = load_and_preprocess_data(file_path)
    X, y = create_features_and_target(df)
    min_vessels, max_vessels = y.min(), y.max()
    
    model, scaler, poly = create_model(X, y)

    current_datetime = datetime.now()
    user_input = str(current_datetime.year) + "-" + str(current_datetime.month) 
    input_date = pd.to_datetime(user_input, format='%Y-%m')
    normalized_prediction, predicted_vessels, _ = predict_vessel_arrival(
        input_date, df, model, scaler, poly, min_vessels, max_vessels
    )

    return normalized_prediction


def load_model_with_fallback(model_path):
    try:
        # Attempt to load the model normally
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except ValueError as ve:
        print(f"ValueError encountered: {ve}")
        print("Attempting to load with custom objects...")
        
        # Custom objects to handle potential 'batch_shape' issue
        custom_objects = {'Input': lambda shape, batch_shape=None, **kwargs: 
                          keras.layers.Input(batch_shape=batch_shape, shape=shape, **kwargs)}
        
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully with custom objects.")
            return model
        except Exception as e:
            print(f"Error loading model with custom objects: {e}")
            
            # If all else fails, try to load just the weights
            print("Attempting to load model weights...")
            try:
                # Create a new model with the same architecture
                inputs = keras.layers.Input(shape=(NUM_AGVS * 4 + MAX_TASKS * 3 + 1,))
                x = keras.layers.Dense(256, activation="relu")(inputs)
                x = keras.layers.Dense(128, activation="relu")(x)
                outputs = [keras.layers.Dense(MAX_TASKS + 2, activation="linear", name=f"agv_{i}_output")(x) for i in range(NUM_AGVS)]
                model = keras.Model(inputs=inputs, outputs=outputs)
                
                # Load weights
                model.load_weights(model_path)
                print("Model weights loaded successfully.")
                return model
            except Exception as e:
                print(f"Error loading model weights: {e}")
                return None

model_path = 'agv_task_assignment_model_episode_900.h5'
model = load_model_with_fallback(model_path)


def interpret_action(action):
    if action == 0:
        return "Parking"
    elif action == 1:
        return "Charging"
    else:
        return action - 2  # Adjusted to match 0-indexing of tasks

def use_model(model, state):
    state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    actions = [tf.argmax(a[0]).numpy() for a in action_probs]
    return actions

def create_sample_state(agvs, tasks):
    state = []
    # Add AGV information
    for agv in agvs:
        state.extend([agv['x'] / 100, agv['y'] / 100, agv['battery'] / MAX_BATTERY, agv['status'] / 2])
    
    # Add task information
    for task in tasks:
        state.extend([tasks[task]['x'] / 100, tasks[task]['y'] / 100, tasks[task]['priority'] / 5])
    
    # Pad with zeros if there are fewer than MAX_TASKS tasks
    state.extend([0, 0, 0] * (MAX_TASKS - len(tasks)))
    
    # Add demand (number of tasks)
    state.append(len(tasks) / MAX_TASKS)
    
    return np.array(state, dtype=np.float32)

def gen_wall(nodes):
    print("Generating Wall ...") 
    for wallx in range(12, 140, 12):
        for wallxx in range(8):
            for wally in range(5, ROWS):
                for t in range(TIME):
                    nodes[(wallx+wallxx, wally, t)].wall = True
    for chargex in range(2,8):
        for chargey in range(5, ROWS):
            for t in range(TIME):
                nodes[(chargex, chargey, t)].charge = True
    for parkx in range(1, COLS, 2):
        for t in range(TIME):
            nodes[(parkx, 0, t)].park = True
    print("Done!")
    return nodes

def update_wall(nodes, cycles_elapsed):
    recompute = []
    for x in range(COLS):
        for y in range(ROWS):
            nodes[(x, y, TIME + cycles_elapsed)] = Spot(x, y, TIME + cycles_elapsed)
            recompute.append((x, y, TIME + cycles_elapsed))
            recompute.append((x, y, TIME + cycles_elapsed - 1))
            nodes.pop((x, y, cycles_elapsed))
    
    for wallx in range(12, 140, 12):
        for wallxx in range(8):
            for wally in range(5, ROWS):
                nodes[(wallx+wallxx, wally, TIME+cycles_elapsed)].wall = True
    for chargex in range(2,8):
        for chargey in range(5, ROWS):
            nodes[(chargex, chargey, TIME+cycles_elapsed)].charge = True
    for parkx in range(1, COLS, 2):
        nodes[(parkx, 0, TIME+cycles_elapsed)].park = True

    update_neighbors(nodes, recompute, (COLS, ROWS, TIME+cycles_elapsed+1))
    print(TIME+cycles_elapsed-10, nodes[(13, 13, TIME+cycles_elapsed-10)].wall)
    return nodes

def astar_process(start, goal, result_queue, sprite_index):
    path = astar_3d(start, goal)
    result_queue.put((sprite_index, path))

def background_task(socketio):
    global nodes, all_sprites, run, sprite_coord, cycles_elapsed
    
    print("Starting ...")
    shape = (COLS, ROWS, TIME)
    nodes = create_grid_and_nodes(shape)
    compute_neighbors(nodes, shape)
    
    print("Generating Walls ...")
    nodes = gen_wall(nodes)

    print("Getting Environment Variables")
    demand = get_demand()
    print(f"demand is {demand:.4f}")
    print(f"increasing speed by {demand*0.5:.4f}")
    
    print("Generating AGVs ...")
    agvs = []
    for i in range(NUM_AGVS):
        while True:
            x = random.randint(0, COLS-1)
            y = random.randint(0, ROWS-1)
            if nodes[(x,y,1)].wall == False:
                break
        # angle = random.randint(0, 360) 
        angle = 0
        status = random.randint(0, 3)
        battery = random.randint(0, 100)

        sprite_ = Sprite(x, y, angle, status, battery)
        sprite_.speed += demand*0.5
        all_sprites.append(sprite_)
        sprite_coord.append([x,y,angle])

        agvs.append({
            'x': x/100,
            'y': y/100,
            'battery': battery,
            'status': status
                     })
    
    print("Starting Background Task...")
    
    # Create a queue for inter-process communication
    result_queue = mp.Queue()
    
    # # Start A* processes
    processes = []
    # start = nodes[(0,0,9)]
    # goal = nodes[(COLS-1, 1, 99)]
    # p = mp.Process(target=astar_process, args=(start, goal, result_queue, 0))
    # p.start()
    # processes.append(p)
    
    x = 0

    tasks = {}
    done = []
    dropoff = []
    for i in range(11, 140, 12):
        dropoff.append(i+1)
        dropoff.append(i+8+1)
    print(dropoff)

    for i in range(20):
        priority = random.randint(0, 5)
        while True:
            x = random.randint(0, COLS-1)
            y = 0
            if nodes[(x, y, 1)].park == True:
                break
        tasks[i] = {
            'x': x,
            'y': y,
            'priority': priority,
            'xd': random.choice(dropoff),
            'yd': random.randint(5, ROWS-1),
            }


    charging = {
        (6, 9): None,
        (6, 13): None,
        (6, 17): None,
        (6, 21): None,
        (6, 25): None,
        (6, 29): None,
        (6, 31): None,
        (6, 35): None,
        (6, 39): None,
        (6, 41): None,
    } 

    extra = 15
    stop = False
    
    container = 0

    while run:
        if cycles_elapsed % 600 == 0:
            weather = check_rainfall()
            print(f"weather is {weather}")
            print(f"decreasing speed by {weather*0.5}")
            for s in all_sprites:
                s.speed -= weather * 0.5

        total_speed = 0
        sample_state = create_sample_state(agvs, tasks)
        actions = use_model(model, sample_state)

        if not stop:
            for i, action in enumerate(actions):
                if all_sprites[i].path == []:
                    final_act = interpret_action(action)
                    if final_act == "Charging" or final_act == "Parking":
                        if all_sprites[i].status != 0:
                            for spot in charging:
                                if charging[spot] == None:
                                    charging[spot] = i
                                    start = nodes[(all_sprites[i].x, all_sprites[i].y, cycles_elapsed+extra)]
                                    goal = nodes[(spot[0], spot[1], TIME+cycles_elapsed-1)]
                                    p = mp.Process(target=astar_process, args=(start, goal, result_queue, i))
                                    p.start()
                                    processes.append(p)
                                    all_sprites[i].status = 0
                                    stop = True
                                    break
                        break
                    else:
                        if all_sprites[i].path == [] and all_sprites[i].ontask == False and final_act in tasks.keys():
                            if not final_act in done:
                                start = nodes[(all_sprites[i].x, all_sprites[i].y, cycles_elapsed+extra)]
                                goal = nodes[(tasks[final_act]["x"], tasks[final_act]["y"], TIME+cycles_elapsed-1)]
                                p = mp.Process(target=astar_process, args=(start, goal, result_queue, i))
                                p.start()
                                processes.append(p)
                                all_sprites[i].status = action
                                all_sprites[i].ontask = final_act 
                                stop = True
                                done.append(final_act)
                                break
        if not stop:
            for s in range(len(all_sprites)):
                if (all_sprites[s].recieved == True) and (all_sprites[s].ontask != False) and (all_sprites[s].ontask in done):
                    print("finding")
                    task = all_sprites[s].ontask
                    start = nodes[(all_sprites[s].x, all_sprites[s].y, cycles_elapsed+extra)]
                    goal = nodes[(tasks[task]["xd"], tasks[task]["yd"], TIME+cycles_elapsed-1)]
                    print(cycles_elapsed+extra)
                    print(TIME+cycles_elapsed-1)
                    p = mp.Process(target=astar_process, args=(start, goal, result_queue, s))
                    p.start()
                    processes.append(p)
                    stop = True
                    all_sprites[s].recieved = False
                    done.pop(done.index(task))
                    tasks.pop(task)
                    container += 2
                    break

        try:
            while True:
                sprite_index, path = result_queue.get_nowait()
                all_sprites[sprite_index].set_path(path)
                print(path)
                for coord in path:
                    nodes[(coord[0], coord[1], coord[2])].wall = True
                stop = False
        except Empty:
            pass

        for s in range(len(all_sprites)):
            speed, next_move = all_sprites[s].sprite_move(cycles_elapsed)
            total_speed += speed 
            sprite_coord[s] = next_move

        socketio.emit('grid_update', {
            "next_move": sprite_coord,
            "tasks":tasks,
            "container": container,
            "speed": (total_speed*9/10),
                                      })

        nodes = update_wall(nodes, cycles_elapsed)

        print(cycles_elapsed)
        cycles_elapsed += 1
        socketio.sleep(0.5)
    
    # Clean up processes
    for p in processes:
        p.join()

async def mqtt_client_coro():
    try:
        c = MQTTClient()
        await c.connect("mqtt://auth_handler:auth_handler@127.0.0.1:1883")  # connect broker client to broker
        await c.subscribe(
            [
                ("DATA_SEN/#", QOS_1)
            ]
        )
        while True:
            try:
                message = await asyncio.wait_for(c.deliver_message(), timeout=0.2)
                if message:
                    packet = message.publish_packet  # get packet payload
                    topic_name = packet.variable_header.topic_name  # get MQTT topic
                    payload = str(packet.payload.data)[12:-2]  # get payload
                    new_data = json.loads(payload)
                    agv_instance = topic_name[-1]
                    # remove this comment to update the battery. but for the simulation we will not be doing so. 
                    # all_sprites[agv_instance].battery = int(new_data)
            except asyncio.TimeoutError:
                pass # No message received in the given time; continue the loop

            for sprite in range(len(sprite_coord)):
                await c.publish(f"PATH_SEN/{sprite}", int_to_bytes_str(sprite_coord[sprite]), qos=0x00)  # publishing test message
                await asyncio.sleep(1)
    except ConnectionError as ce:
        print(f"Connection failed: {ce}")
        mqtt_thread.join()

def start_mqtt_client():
    asyncio.run(mqtt_client_coro())

def start_mqtt_thread():
    global mqtt_thread
    mqtt_thread = Thread(target=start_mqtt_client)
    mqtt_thread.daemon = True
    mqtt_thread.start()

def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode='threading')

    @app.route('/')
    def index():
        return render_template('index.html')

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")

    start_mqtt_thread()

    # Start the background task
    socketio.start_background_task(background_task, socketio)

    return app, socketio

app, socketio = create_app()

# This line is important for Gunicorn to find your app
__all__ = ['app']