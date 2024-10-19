from flaskr import socketio, app

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)