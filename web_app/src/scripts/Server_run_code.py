from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

ThisLocation = __file__

@app.route("/run-python", methods=['POST'])
def run_script():
    code = request.json.get('code')
    with open(f'{ThisLocation}\\temp_file.py', 'w') as temporal_file_code:
        temporal_file_code.write(code)
    try:
        result = subprocess.run(
            ['python3', '-m', f'{ThisLocation}\\temp_file.py'], 
            capture_output=True, text=True, check=True
        )
        output, error = (result.stdout, result.stderr)
        return jsonify({'output': output, 'error': error})
    except Exception as e:
        return jsonify({'output': e.stdout, 'error': e.stderr})

if __name__ == '__main__':
    app.run(debug=True)