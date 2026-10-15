"""Запуск: python -m pip install httpx; python test_app.py"""
import re
import numpy as np
from fastapi.testclient import TestClient
from main import app

examples = [[5.1,3.5,1.4,0.2], [6.0,2.9,4.5,1.5], [6.7,3.3,5.7,2.1]]
with TestClient(app) as client:
    for path in ['/classification','/regression']:
        assert client.get(path).status_code == 200
    for x, expected in zip(examples, ['setosa','versicolor','virginica']):
        payload = dict(zip(['sepal_length','sepal_width','petal_length','petal_width'],x))
        response = client.post('/classification', data=payload)
        assert response.status_code == 200
        result = re.search(r'id="prediction">(.*?)</strong>',response.text).group(1)
        assert result == expected, result
        print('Классификация:',x,'->',result)
        del payload['petal_length']
        response = client.post('/regression',data=payload)
        assert response.status_code == 200
        result = re.search(r'id="prediction">(.*?)</strong>',response.text).group(1)
        net,mn,mx,_ = app.state.regressor
        expected_value=net.predict((np.array(x)[[0,1,3]]-mn)/(mx-mn))
        assert result == f'{expected_value:.3f} см'
        print('Регрессия:',list(payload.values()),'->',result)
    for bad in ['abc', '-1','nan','inf','0','31']:
        response = client.post('/regression',data={'sepal_length':bad,'sepal_width':3,'petal_width':1})
        assert response.status_code == 422
        assert 'Введите все размеры' in response.text
    assert client.post('/classification',data={}).status_code == 422
print('Все проверки форм пройдены.')
