from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from models import load_model

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / 'templates'))
FIELDS = [('sepal_length', 'Длина чашелистика'), ('sepal_width', 'Ширина чашелистика'),
          ('petal_length', 'Длина лепестка'), ('petal_width', 'Ширина лепестка')]

@asynccontextmanager
async def lifespan(app):
    # Загрузка один раз при старте. Обучения в приложении нет.
    app.state.classifier = load_model(BASE / 'iris_weights.json')
    app.state.regressor = load_model(BASE / 'regression_weights.json', regression=True)
    yield

app = FastAPI(title='Ирисы: классификация и регрессия', lifespan=lifespan)
Measurement = Annotated[float, Form(gt=0, le=30, allow_inf_nan=False)]

def render(request, regression=False, values=None, prediction=None, error=None, status=200):
    return templates.TemplateResponse(request=request, name='form.html', status_code=status, context={
        'regression': regression,
        'fields': [f for f in FIELDS if not regression or f[0] != 'petal_length'],
        'values': values or {}, 'prediction': prediction, 'error': error})

@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return render(request, regression=request.url.path == '/regression',
                  error='Введите все размеры числами больше 0 и не больше 30 см.', status=422)

@app.get('/')
def index():
    return RedirectResponse('/classification')

@app.get('/classification', response_class=HTMLResponse)
def classification_page(request: Request):
    return render(request)

@app.post('/classification', response_class=HTMLResponse)
def classification(request: Request, sepal_length: Measurement, sepal_width: Measurement,
                   petal_length: Measurement, petal_width: Measurement):
    values = dict(zip([f[0] for f in FIELDS], [sepal_length, sepal_width, petal_length, petal_width]))
    net, mn, mx, metadata = request.app.state.classifier
    x = np.array(list(values.values()))
    prediction = metadata['names'][net.predict((x - mn) / (mx - mn))]
    return render(request, values=values, prediction=prediction)

@app.get('/regression', response_class=HTMLResponse)
def regression_page(request: Request):
    return render(request, regression=True)

@app.post('/regression', response_class=HTMLResponse)
def regression(request: Request, sepal_length: Measurement, sepal_width: Measurement, petal_width: Measurement):
    values = dict(sepal_length=sepal_length, sepal_width=sepal_width, petal_width=petal_width)
    net, mn, mx, _ = request.app.state.regressor
    value = net.predict((np.array(list(values.values())) - mn) / (mx - mn))
    return render(request, regression=True, values=values, prediction=f'{value:.3f} см')
