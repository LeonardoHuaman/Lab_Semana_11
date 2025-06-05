from flask import Flask, request, render_template
import numpy as np
import pickle

# ————————————————
# Carga del modelo y del MinMaxScaler (ms)
model = pickle.load(open('model.pkl', 'rb'))
ms    = pickle.load(open('minmaxscaler.pkl', 'rb'))
# ————————————————

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index_hiring.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender         = request.form['gender']          
        ssc_p          = float(request.form['ssc_p'])      
        hsc_p          = float(request.form['hsc_p'])      
        degree_p       = float(request.form['degree_p'])   
        workex         = request.form['workex']            
        etest_p        = float(request.form['etest_p'])    
        specialisation = request.form['specialisation']   
        mba_p          = float(request.form['mba_p'])   

        gender_num = 0 if gender.upper() == 'M' else 1
        workex_num = 1 if workex.lower() == 'yes' else 0
        specialisation_num = 1 if specialisation == 'Mkt&Fin' else 0

        X_raw = np.array(
            [[ssc_p, hsc_p, degree_p, etest_p, mba_p,
              gender_num, workex_num, specialisation_num]],
            dtype=float
        )

        X_scaled = ms.transform(X_raw)
        pred_code = model.predict(X_scaled)[0]

        if pred_code == 1:
            mensaje = "<strong>¡Contratado!</strong>"
        else:
            mensaje = "<strong>No Contratado.</strong>"

    except KeyError:
        mensaje = "Error: faltan campos en el formulario."
    except ValueError:
        mensaje = "Error: asegúrate de que los porcentajes sean números válidos."
    except Exception as e:
        mensaje = f"Ocurrió un error inesperado: {str(e)}"

    return render_template('index_hiring.html', result=mensaje)


if __name__ == "__main__":
    app.run(debug=True)
