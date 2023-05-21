from flask import Flask, render_template, request
from pickle import load
import numpy as np
import pandas as pd
import cvxopt as opt
from cvxopt import blas, solvers

app = Flask(__name__, template_folder='templateFiles',
            static_folder='staticFiles')


# creating assets list
assets = pd.read_csv('SP500Data.csv', index_col=0)
assets = assets.drop(assets.columns[[0]], axis=1)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
# Fill the missing values with the last value available in the dataset.
assets = assets.fillna(method='ffill')

options = []
for tic in assets.columns:
    #{'label': 'user sees', 'value': 'script sees'}
    mydict = {}
    mydict['label'] = tic  # Apple Co. AAPL
    mydict['value'] = tic
    options.append(mydict)


@app.route('/')
def start():
    return render_template('dashboard.html')


riskTolerance = 0


@app.route('/predict', methods=['GET', 'POST'])
def predict_riskTolerance():
    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    # estimate accuracy on validation set
    # Age,Edu,Married,Kids,Occ,Inccl, Risk,Nwcat
    Age = request.form.get("age")
    Edu = request.form.get("edu")
    Married = request.form.get("marriagestatus")
    Kids = request.form.get("kids")
    Occ = request.form.get("occu")
    Inccl = request.form.get("inccl")
    Risk = request.form.get("risk")
    Nwcat = request.form.get("nwcat")
    # Nwcat = Nwcat*1000000
    X_input = [[Age, Edu, Married, Kids, Occ, Inccl, Risk, Nwcat]]
    predictions = loaded_model.predict(X_input)
    global riskTolerance
    riskTolerance = round(float(predictions*100), 2)
    return render_template('dashboard.html', riskTolerance=riskTolerance, options=options)
    # return options


@app.route('/asset_allocation', methods=['GET', 'POST'])
def get_asset_allocation():
    stock_dict = request.form.to_dict(flat=False)
    stock_tick = stock_dict["stock_tick"]

    assets_selected = assets.loc[:, stock_tick]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1-riskTolerance

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    print(w)
    Alloc = pd.DataFrame(data=np.array(
        portfolios['x']), index=assets_selected.columns)
    # Calculate efficient frontier weights using quadratic programming
    returns_final = (np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final, axis=1)
    returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0, :] + 100

    labels = Alloc.index.values.tolist()
    values = Alloc.iloc[:, 0].tolist()
    app.logger.info(labels)
    app.logger.info(values)
    labels2 = returns_sum_pd.index.values.tolist()
    values2 = returns_sum_pd.iloc[:, 0].tolist()
    return render_template('dashboardwithgraph.html', labels=labels, values=values, labels2=labels2, values2=values2, riskTolerance=riskTolerance, options=options)
    # return values


if __name__ == '__main__':
    app.run(debug=True)
