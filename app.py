from flask import Flask, render_template, session, url_for, redirect
from tablut import Tafl
from random import choice
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

    moves = list(map(game.action_dec, game.mask))
    print(moves)
    return render_template('tablut.html', tablero=game.state, moves=moves)

@app.route("/<move>", methods=['POST'])
def tablut2(move):
    if 'jogando' in session:
        gamestring = session['gamestring']
        game = Tafl(fromjson=gamestring)
        if move in game.mask:
            game.in_step(move)
            # El agente hace un mov aleatorio
            game.in_step(choice(game.mask))
            moves = list(map(game.action_dec, game.mask))
            return render_template('tablut.html', tablero=game.state, moves=moves, jugador=game.currentPlayer)
        else:
            pass
            #Movimiento inválido
        

    else:
        return redirect(url_for(tablut))

if __name__ == "__main__":
    app.run(debug=True)
