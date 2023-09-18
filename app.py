from flask import Flask, render_template, request
import joblib
import os
import numpy as np

# Cargar el modelo entrenado
model_path = os.path.join(os.path.dirname(__file__), 'models', 'ProyectoFinal_arbol.pkl')
model = joblib.load(model_path)

# Crear una aplicación Flask
app = Flask(__name__)

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'

# Definir la ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    Fiebre = int(request.form['Fiebre'])
    Tos = int(request.form['Tos'])
    DolorGarganta = int(request.form['DolorGarganta'])
    CongestionNasal = int(request.form['CongestionNasal'])
    DificultadRespiratoria = int(request.form['DificultadRespiratoria'])
    
    # Realizar una predicción de probabilidades utilizando el modelo cargado
    pred_probabilities = np.array([[Fiebre, Tos, DolorGarganta, CongestionNasal, DificultadRespiratoria]])
    
    # Obtener los nombres de las clases (Deserción, Alerta, Buen estudiante)
    prediccion = model.predict(pred_probabilities)

    # Renderizar la plantilla 'result.html' y pasar el mensaje a la plantilla
    return render_template('result.html', pred=prediccion[0])

# Iniciar la aplicación si este script es el punto de entrada
if __name__ == '__main__':
    app.run(debug=True)
