from flask import Flask, render_template, session
from tablut import Tafl
app = Flask(__name__)
app.secret_key = "KEYULTRASECRETADELAMUERTEqueAUNQUELOINTENTENniPAATRASVANaPODERACABARDEScubr13nd0"

@app.route("/")
def tablut():
    if 'jogando' in session:
        gamestring = session['gamestring']
        game = Tafl(fromjson=gamestring)        

    else:
        session['jogando'] = True
        game = Tafl()
        game.in_step(game.mask[-1])
        session['gamestring'] = game.json()

    return render_template('tablut.html', tablero=game.state, moves=game.mask)

@app.route("/<move>", methods=['POST'])
def tablut2():
    pass

if __name__ == "__main__":
    app.run()
