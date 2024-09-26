import numpy as np
import random

input_x = [
    [-0.2, 0.5],
    [0.2, -0.7],
    [0.8, -0.8],
    [0.8, 1]
]
output_class = [0, 0, 1, 1]

class ArtificialNeuron:
    # Klasės konstruktorius priskiriantis svorius ir aktyvacijos funkciją
    def __init__(self, w0, w1, w2, activation_fn):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.activation_fn = activation_fn

    # Apskaičiuojame prognozę naudojant x1 ir x2 įvestis
    def predict(self, x1, x2):
        z = self.w1 * x1 + self.w2 * x2 + self.w0
        prediction = self.activation_fn(z)
        if self.activation_fn == self.sigmoid_activation:
            return round(prediction)
        return prediction

    # Aktyvacijos funkcijos, kurios pažymėtos kaip statinės (t. y. galima pasiekti nesukuriant objekto)
    @staticmethod
    def threshold_activation(a):
        return 1 if a >= 0 else 0

    @staticmethod
    def sigmoid_activation(a):
        return 1 / (1 + np.exp(-a))

# Įvertinama, ar prognozuotos išvestys atitinka tikėtinas išvestis
def evaluate_model(weights, inputs, expected_outputs, activation_fn):
    neuron = ArtificialNeuron(weights[0], weights[1], weights[2], activation_fn)
    predictions = [neuron.predict(x1, x2) for [x1, x2] in inputs]
    return np.array_equal(predictions, expected_outputs)

# Egzistuojanti funkcija rasti geriausius svorius iteruojant per kiekvieną kombinaciją
def find_weights(input_x, output_class, activation_fn):
    best_weights = []

    # Nustatomi svorių ir poslinkio intervalai, per kuriuos bus ieškoma geriausių kombinacijų
    w0_range = np.arange(-0.6, 1, 0.1)
    w1_range = np.arange(-1, 1, 0.1)
    w2_range = np.arange(-1, 1, 0.1)

    # Ieškoma geriausių svorių kombinacijų, kurios leidžia modeliui teisingai atpažinti duomenis
    for w0 in w0_range:
        for w1 in w1_range:
            for w2 in w2_range:
                if evaluate_model([w0, w1, w2], input_x, output_class, activation_fn):
                    best_weights.append((w0, w1, w2))
                if len(best_weights) == 5:
                    return best_weights
    return best_weights

# Nauja funkcija rasti geriausius svorius atsitiktine paieška
def find_weights_randomly(input_x, output_class, activation_fn, max_attempts=10000, desired_count=5):
    best_weights = set()

    # Nustatomi svorių ir poslinkio intervalai
    w0_values = np.arange(-0.6, 1, 0.1)
    w1_values = np.arange(-1, 1, 0.1)
    w2_values = np.arange(-1, 1, 0.1)

    for _ in range(max_attempts):
        # Atsitiktinis svorių pasirinkimas
        w0 = random.choice(w0_values)
        w1 = random.choice(w1_values)
        w2 = random.choice(w2_values)

        weights = (w0, w1, w2)
        
        # Tikriname, ar svoriai jau yra surasti
        if weights in best_weights:
            continue

        if evaluate_model(weights, input_x, output_class, activation_fn):
            best_weights.add(weights)
            if len(best_weights) >= desired_count:
                break

    return list(best_weights)

for activation_function in [ArtificialNeuron.threshold_activation, ArtificialNeuron.sigmoid_activation]:
    print(f"\nFinding best weights with {activation_function.__name__} using exhaustive search:")
    best_weights_exhaustive = find_weights(input_x, output_class, activation_function)

    print("w0    w1    w2")
    for (w0, w1, w2) in best_weights_exhaustive:
        print(f"{w0:.2f} {w1:.2f} {w2:.2f}")

    print(f"\nFinding best weights with {activation_function.__name__} using random search:")
    best_weights_random = find_weights_randomly(input_x, output_class, activation_function)

    print("w0    w1    w2")
    for (w0, w1, w2) in best_weights_random:
        print(f"{w0:.2f} {w1:.2f} {w2:.2f}")