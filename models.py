import json
import numpy as np

# Функция активации (сигмоида)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоиды.
# Принимает УЖЕ посчитанный ответ нейрона s = sigmoid(x): производная равна s * (1 - s)
def sigmoid_derivative(s):
    return s * (1 - s)


# Класс нейрона — такой же, как в лабе 1
class Neuron:
    def __init__(self, input_size):
        # Случайные веса от -1 до 1: и положительные, и отрицательные,
        # чтобы нейроны с самого начала «смотрели» на признаки по-разному
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias   # взвешенная сумма
        return sigmoid(total)                              # ответ от 0 до 1

# Класс простой нейронной сети с одним скрытым слоем
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Скрытый слой: hidden_size нейронов, каждый смотрит на все входы
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        # Выходной слой: output_size нейронов, каждый смотрит на все скрытые нейроны
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def feedforward(self, inputs):
        # Прямой проход через скрытый слой
        hidden_outputs = np.array([neuron.feedforward(inputs) for neuron in self.hidden_layer])
        # Прямой проход через выходной слой
        outputs = np.array([neuron.feedforward(hidden_outputs) for neuron in self.output_layer])
        return outputs

    def predict(self, inputs):
        # Номер вида, чей выходной нейрон выдал самое большое число
        return int(np.argmax(self.feedforward(inputs)))

    def accuracy(self, X, labels):
        # Доля цветков, для которых сеть назвала правильный вид
        predictions = np.array([self.predict(x) for x in X])
        return np.mean(predictions == labels)

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(inputs)):
                # 1. Прямой проход (запоминаем ответы скрытого слоя — они нужны для обучения)
                hidden_outputs = np.array([neuron.feedforward(inputs[i]) for neuron in self.hidden_layer])
                outputs = np.array([neuron.feedforward(hidden_outputs) for neuron in self.output_layer])

                # 2. Ошибка на выходном слое: правильный ответ минус ответ сети
                errors = targets[i] - outputs
                total_error += np.mean(errors ** 2)

                # 3. Поправка для выходных нейронов: ошибка × производная сигмоиды
                d_outputs = errors * sigmoid_derivative(outputs)

                # 4. Ошибка течёт назад: каждый скрытый нейрон k собирает вину
                #    всех выходных нейронов j пропорционально весу связи между ними.
                #    Считаем ДО изменения весов выходного слоя!
                d_hidden = np.array([
                    sum(d_outputs[j] * self.output_layer[j].weights[k] for j in range(len(outputs)))
                    * sigmoid_derivative(hidden_outputs[k])
                    for k in range(len(self.hidden_layer))
                ])

                # 5. Подправляем веса выходного слоя (вход для них — ответы скрытого слоя)
                for j, neuron in enumerate(self.output_layer):
                    neuron.weights += learning_rate * d_outputs[j] * hidden_outputs
                    neuron.bias += learning_rate * d_outputs[j]

                #    ...и веса скрытого слоя (вход для них — признаки цветка)
                for k, neuron in enumerate(self.hidden_layer):
                    neuron.weights += learning_rate * d_hidden[k] * inputs[i]
                    neuron.bias += learning_rate * d_hidden[k]

            # Логирование процесса обучения
            if epoch % 100 == 0 or epoch == epochs - 1:
                mse = total_error / len(inputs)
                acc = self.accuracy(inputs, targets.argmax(axis=1))
                print(f"эпоха {epoch:4d} | ошибка {mse:.4f} | точность на обучении {acc:.0%}")


class RegressionNetwork(SimpleNeuralNetwork):
    """Сигмоидный скрытый слой, ровно один линейный выход."""
    def __init__(self, input_size=3, hidden_size=8):
        super().__init__(input_size, hidden_size, 1)

    def feedforward(self, inputs):
        hidden = np.array([n.feedforward(inputs) for n in self.hidden_layer])
        neuron = self.output_layer[0]
        return np.array([np.dot(hidden, neuron.weights) + neuron.bias])

    def predict(self, inputs):
        return float(self.feedforward(inputs)[0])

    def train(self, inputs, targets, learning_rate=0.02, epochs=1000):
        for epoch in range(epochs):
            for x, target in zip(inputs, targets):
                hidden = np.array([n.feedforward(x) for n in self.hidden_layer])
                out = self.output_layer[0]
                prediction = np.dot(hidden, out.weights) + out.bias
                # Производная линейной активации равна 1.
                delta = float(target) - prediction
                delta_hidden = delta * out.weights.copy() * sigmoid_derivative(hidden)
                out.weights += learning_rate * delta * hidden
                out.bias += learning_rate * delta
                for k, neuron in enumerate(self.hidden_layer):
                    neuron.weights += learning_rate * delta_hidden[k] * x
                    neuron.bias += learning_rate * delta_hidden[k]
            if epoch % 200 == 0 or epoch == epochs - 1:
                print(f"эпоха {epoch:4d} | MAE на обучении {self.mae(inputs, targets):.4f} см")

    def mae(self, inputs, targets):
        return float(np.mean(np.abs(np.array([self.predict(x) for x in inputs]) - targets)))


def save_regression(net, mn, mx, path):
    payload = {
        "hidden": [{"weights": n.weights.tolist(), "bias": float(n.bias)} for n in net.hidden_layer],
        "output": [{"weights": n.weights.tolist(), "bias": float(n.bias)} for n in net.output_layer],
        "mn": mn.tolist(), "mx": mx.tolist(),
        "features": ["sepal_length", "sepal_width", "petal_width"],
        "target": "petal_length", "units": "cm", "output_activation": "linear"
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(path, regression=False):
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    net = RegressionNetwork() if regression else SimpleNeuralNetwork(4, 8, 3)
    for layer, key in [(net.hidden_layer, "hidden"), (net.output_layer, "output")]:
        if len(layer) != len(saved[key]):
            raise ValueError("Неверная архитектура модели")
        for neuron, params in zip(layer, saved[key]):
            weights = np.asarray(params["weights"], dtype=float)
            if weights.shape != neuron.weights.shape or not np.isfinite(weights).all():
                raise ValueError("Неверные веса модели")
            neuron.weights = weights
            neuron.bias = float(params["bias"])
    mn, mx = np.array(saved["mn"]), np.array(saved["mx"])
    if mn.shape != (3 if regression else 4,) or mx.shape != mn.shape or not np.all(mx > mn):
        raise ValueError("Неверные параметры нормализации")
    return net, mn, mx, saved
