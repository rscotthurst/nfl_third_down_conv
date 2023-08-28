from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model/log_reg.pkl', 'rb'))


# localhost:8080/third_down?yds_to_go=8&is_home=1
@app.route('/third_down')
def third_down():
    yds_to_go = request.args.get('yds_to_go')
    is_home = request.args.get('is_home')
    home_text = 'home' if is_home == 1 else 'away'

    input_val = np.array([[int(yds_to_go), int(is_home)]])
    print(input_val)

    success_proba = model.predict_proba(input_val)[0][1]

    return (
        f'There are {yds_to_go} yards to go, the team with the ball is the {home_text} team.\n'
        f'Odds of success: {round(success_proba * 100)}%'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
