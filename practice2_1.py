import numpy as np

# Встановлюється початкове значення для генератора випадкових чисел.
# Це забезпечує відтворюваність результатів при кожному запуску коду.
np.random.seed(0)

# Визначає вхідні дані мережі у формі масив масивів.
# Кожен внутрішній масив є одним вхідним вектором.
X = [[4, 6, 3, 6], [2, 5, 2, 5], [6, 6, 1, 3]]

# Представляє щільний (повнозв'язний) шар нейронної мережі
class Layer_Dense:
    # Ініціалізує ваги і зміщення для кожного нейрону в шарі.
    # Ваги ініціалізуються випадковими значеннями з нормального розподілу, модифіковані додаванням 0.10,
    # щоб забезпечити не нульові початкові значення. Зміщення ініціалізуються нулями.
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 + np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Реалізує пряме поширення інформації через шар: множення вхідних даних на матрицю ваг і додавання зміщення.
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Rectified Linear Unit - це популярна активаційна функція.
# Це дозволяє моделі вводити нелінійність і ефективно навчатися на складних даних.
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs);

# Створюються три об'єкти LayerDense з різними розмірами.
# Перший шар приймає 4 вхідні сигнали і виводить 8,
# другий бере 8 вхідних сигналів з першого шару і теж виводить 8,
# третій шар бере 8 вхідних сигналів з другого шару і виводить 2.
layer1 = Layer_Dense(4, 8)
layer2 = Layer_Dense(8, 8)
layer3 = Layer_Dense(8, 2)

# Створюються 2 об'єкти Activation_ReLU для подальшої активації після 2 та 3 шару
activation2 = Activation_ReLU()
activation3 = Activation_ReLU()

# За допомогою методу forward кожен шар послідовно обробляє дані,
# передані від попереднього шару та активує його за допомогою "Rectified Linear Unit" функції,
# починаючи з вхідних даних X.
layer1.forward(X)
layer2.forward(layer1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation3.forward(layer3.output)

# Виводяться ваги третього шару. Це демонструє,
# як можна доступитися до параметрів мережі після її ініціалізації та прямого поширення.
print("Layer 3:\n", layer3.output)
print("Activated layer 3:\n", activation3.output)