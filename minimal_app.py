from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World! This is a test of the Jarvis web interface."

@app.route('/api/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'API test successful'
    })

if __name__ == '__main__':
    print("Starting minimal Flask server...")
    app.run(debug=True, port=5000)
