import numpy
import numpy as np
# Создайте двумерный массив размером (3, 4):
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
#
# С помощью среза получите массив, состоящий из первых двух строк и
# столбцов 1 и 2, т.е. на выходе должен получится следующий массив размером (2, 2):
# [[2 3]
#  [6 7]]

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
sliced_arr = [[*arr[0][1:3]],
              [*arr[1][1:3]]]
print(sliced_arr[0])
print(sliced_arr[1])

# Отобразить два графика функций sin(x) и cos(x) в одном окне (используйте функцию
# subplot из пакета matplotlib.pyplot.) В простейшем варианте вот так должен выглядеть
# ваш результат: ( Картинку вставить не могу, но тут 2 графика синусоида и косинусоида )

from matplotlib import pyplot as plt

arrange = np.arange(-10, 10, 0.1)
sin = np.sin(arrange)
cos = np.cos(arrange)

plt.subplot(1, 2, 1)

plt.plot(arrange, sin, label='Sin Function')
plt.title('Sin')

plt.show()

plt.plot(arrange, cos, label='Cos Function', c='red')
plt.title('Cos')

plt.show()

