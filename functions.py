import numpy as np
from numpy import pi, abs
import matplotlib.pyplot as plt
import ruptures as rpt
import os
from numpy.polynomial.legendre import Legendre  

# константа в плотности
a = 6 / pi

# количество сигм
m = 2

# папка для сохранения графиков
output_folder = f'pictures_{m}m/Andrews'
output_folder_sep = f'pictures_{m}m/separation'
output_folder_trend = f'pictures_{m}m/trend'

def density(s, lam = 1/3):
    return  (1 - (1 + lam*np.abs(s)) * np.exp(-lam*np.abs(s))) / lam**2


def method1(X):      # без проецирования
    mean = np.mean(X)
    sigma_i_square = (X - mean) ** 2
    u_0 = (np.sum(X / sigma_i_square)) / np.sum(1 / sigma_i_square)
    W_i_0 = 1 / sigma_i_square
    iterations = 100   # количество итераций
    for _ in range(iterations):
        u_1 = np.sum(X * W_i_0) / np.sum(W_i_0)
        W_i_0 = (1 / (X - u_1) ** 2) * density((X - u_1) / (sigma_i_square ** (0.5)))
        u_0 = u_1
    return u_0

def method2(X, m=m):         # с проецированием
    X = np.array(X)
    mean = np.mean(X)
    sigma_i_square = (X - mean) ** 2
    u_0 = (np.sum(X / sigma_i_square)) / np.sum(1 / sigma_i_square)
    W_i_0 = 1 / sigma_i_square
    iterations = 100   # количество итераций
    for i in range(iterations):
        std = np.std(X)
        X[(X <= u_0 - m * std)] = u_0 - m * std
        X[(X >= u_0 + m * std)] = u_0 + m * std
        u_1 = np.sum(X * W_i_0) / np.sum(W_i_0)
        W_i_0 = (1 / (X - u_1) ** 2) * density((X - u_1) / (sigma_i_square ** (0.5)))
        u_0 = u_1
    return u_0, std

def method2_sep(X_, m=m):         # с проецированием (для отделения данных)
    X_ = np.array(X_)
    X = np.copy(X_)
    X = X[~np.isnan(X)]
    mean = np.mean(X)
    sigma_i_square = (X - mean) ** 2
    u_0 = (np.sum(X / sigma_i_square)) / np.sum(1 / sigma_i_square)
    W_i_0 = 1 / sigma_i_square
    iterations = 100   # количество итераций
    for i in range(iterations):
        std = np.std(X)
        X[(X <= u_0 - m * std)] = u_0 - m * std
        X[(X >= u_0 + m * std)] = u_0 + m * std
        u_1 = np.sum(X * W_i_0) / np.sum(W_i_0)
        W_i_0 = (1 / (X - u_1) ** 2) * density((X - u_1) / (sigma_i_square ** (0.5)))
        u_0 = u_1
    return u_0, std

def data_cleaning(data, m=m):
    # u, std = method2(data, m)  # с проецированием
    u = method1(data)  # без проецирования
    std = np.std(data)

    # Находим индексы значений, выходящих за границы
    mask = (data < u - m * std) | (data > u + m * std)
    
    # Создаём копию массива для модификации
    data_cleaned = np.array(data, dtype=np.float64)

    # Обрабатываем выбросы
    for i in range(len(data)):
        if mask[i]:  # Если значение не удовлетворяет условию
            if i == 0:  
                data_cleaned[i] = data[i + 1]  # Если это первый элемент, заменяем на следующий
            elif i == len(data) - 1:  
                data_cleaned[i] = data[i - 1]  # Если последний — заменяем на предыдущий
            else:  
                data_cleaned[i] = (data[i - 1] + data[i + 1]) / 2  # В остальных случаях — на среднее соседей
    
    return data_cleaned
def result_plot(lst_of_data, column_names_data, n2, output_folder=output_folder, m=m):
    lst_percent = []
    for i, data_new in enumerate(lst_of_data):
        fig, axs = plt.subplots(figsize = (20, 5), nrows= 1, ncols= 2)
        data = data_new[0]
        max_y, min_y = np.max(data), np.min(data)
        data_without_outliers, original_indices = data_new[1][0], data_new[1][1]
        outliers, outliers_indices = data_new[2][0], data_new[2][1]

        # u, std = method2(data, m)    # с проецированием

        u = method1(data)            # без проецирования
        std = np.std(data)

        data_after_method = data[(data >= u - m * std) & (data <= u + m * std)]
        mask = np.isin(data_after_method, outliers)
        lst_percent.append(np.sum(mask) * 100 / len(mask))
        # mask_unique = np.unique(mask)
        # indices_outliers_after = np.arange(len(data_after_method))[mask]
        # original_indices = np.arange(len(data_after_method))[~mask]
        axs[1].set_ylim(min_y - 0.1, max_y + 0.1)
        axs[1].axhline(y=u, color='b', label='оценка')
        axs[1].axhline(y=u + m * std, color='r', label=f'оценка + {m} σ')
        axs[1].axhline(y=u - m * std, color='r', label=f'оценка - {m} σ')
        # axs[1].axhline(y=np.mean(data), color='b', label='среднее')
        #res_indices = np.concatenate([indices_outliers_after, original_indices])
        # axs[i, 1].plot(indices_outliers_after, data_after_method[mask], color='b')
        # axs[i, 1].plot(original_indices, data_after_method[~mask], color='b')
        #data_concatenate = np.concatenate([data_after_method[mask], data_after_method[~mask]])
        axs[1].plot(np.arange(len(data_after_method)), data_after_method)
        axs[1].grid(True)
        axs[1].set_title(f'{column_names_data[i]}')
        axs[1].legend()
        axs[0].set_ylim(min_y - 0.1, max_y + 0.1)
        # axs[i, 0].plot(original_indices, data_without_outliers, color='b')
        # axs[i, 0].plot(outliers_indices, outliers, color='b')
        axs[0].plot(data)
        axs[0].grid(True)
        plot_filename = os.path.join(output_folder, f"{column_names_data[i]}.png")
        fig.savefig(plot_filename)
        plt.close(fig)
    fig.tight_layout()
    plt.show()
    percent_outliers = [f"{lst_percent[i]}  {column_names_data[i]}\n" for i in range(n2)]
    for i in percent_outliers:
        print(i)

def result_plot_ver2(lst_of_data, column_names_data, output_folder=output_folder, m=m):
    for i, data in enumerate(lst_of_data):
        fig, axs = plt.subplots(figsize = (20, 5), nrows= 1, ncols= 2)
        max_y, min_y = np.max(data), np.min(data)

        # u, std = method2(data, m)    # с проецированием

        u = method1(data)            # без проецирования
        std = np.std(data)

        data_after_method = data[(data >= u - m * std) & (data <= u + m * std)]
        axs[1].set_ylim(min_y - 0.1, max_y + 0.1)
        axs[1].axhline(y=u, color='b', label='оценка')
        axs[1].axhline(y=u + m * std, color='r', label=f'оценка + {m} σ')
        axs[1].axhline(y=u - m * std, color='r', label=f'оценка - {m} σ')
        axs[1].plot(np.arange(len(data_after_method)), data_after_method)
        axs[1].grid(True)
        axs[1].set_title(f'{column_names_data[i]}')
        axs[1].legend()
        axs[0].set_ylim(min_y - 0.1, max_y + 0.1)
        axs[0].plot(data)
        axs[0].grid(True)
        plot_filename = os.path.join(output_folder, f"{column_names_data[i]}.png")
        fig.savefig(plot_filename)
        plt.close(fig)
    fig.tight_layout()
    plt.show()

def result_plot_medicine_data(lst_of_data, column_names_data, m=m):
    for i, data_new in enumerate(lst_of_data):
        fig, axs = plt.subplots(figsize = (20, 5), nrows= 1, ncols= 2)
        try:
            data = data_new
            max_y, min_y = np.max(data), np.min(data)

            # u, std = method2(data, m)    # с проецированием

            u = method1(data)            # без проецирования
            std = np.std(data)

            data_after_method = data[(data >= u - m * std) & (data <= u + m * std)]
            # mask_unique = np.unique(mask)
            # indices_outliers_after = np.arange(len(data_after_method))[mask]
            # original_indices = np.arange(len(data_after_method))[~mask]
            axs[1].set_ylim(min_y - 0.1, max_y + 0.1)
            axs[1].axhline(y=u, color='b', label='оценка')
            axs[1].axhline(y=u + m * std, color='r', label=f'оценка + {m} сигма')
            axs[1].axhline(y=u - m * std, color='r', label=f'оценка - {m} сигма')
            # axs[1].axhline(y=np.mean(data), color='b', label='среднее')
            #res_indices = np.concatenate([indices_outliers_after, original_indices])
            # axs[i, 1].plot(indices_outliers_after, data_after_method[mask], color='b')
            # axs[i, 1].plot(original_indices, data_after_method[~mask], color='b')
            #data_concatenate = np.concatenate([data_after_method[mask], data_after_method[~mask]])
            axs[1].plot(np.arange(len(data_after_method)), data_after_method)
            axs[1].grid(True)
            axs[1].set_title(f'{column_names_data[i]}')
            axs[1].legend()
            axs[0].set_ylim(min_y - 0.1, max_y + 0.1)
            # axs[i, 0].plot(original_indices, data_without_outliers, color='b')
            # axs[i, 0].plot(outliers_indices, outliers, color='b')
            axs[0].plot(data)
            axs[0].grid(True)
            safe_filename = column_names_data[i].replace("/", "_")
            plot_filename = os.path.join(output_folder, f"{safe_filename}.png")
            fig.savefig(plot_filename)
            plt.close(fig)
        except TypeError:
            print(f"Ошибка для колонки {column_names_data[i]}")
            


def plotting_pieces(x_axis, data, y_min, y_max, x_min, x_max, n, name_graph,
                     output_folder=output_folder_sep, m=m):
    fig, axs = plt.subplots(figsize = (20, 5), nrows= 1, ncols= 2)
    data_inner = np.full_like(data, np.nan)
    x_axis_all = np.arange(len(data))
    data_inner[x_axis] = data[x_axis]
    axs[0].set_ylim(y_min - 0.1, y_max + 0.1)
    axs[0].set_xlim(x_min - n // 100, x_max + n // 100) 
    axs[0].plot(x_axis_all, data_inner)
    axs[0].grid(True)
    axs[0].set_title(name_graph)
    axs[1].set_title(f'{name_graph} после удаления выбросов')
    axs[1].set_ylim(y_min - 0.1, y_max + 0.1)
    axs[1].set_xlim(x_min - n // 100, x_max + n // 100)
    axs[1].grid(True)

    u, std = method2_sep(data_inner)
    cond = (data_inner >= u - m * std) & (data_inner <= u + m * std)
    ind_delete = np.where(~cond)[0]
    data_inner[ind_delete] = np.nan
    axs[1].plot(x_axis_all, data_inner)
    axs[1].axhline(y=u, color='b', label='оценка')
    axs[1].axhline(y=u + m * std, color='r', label=f'оценка + {m} σ')
    axs[1].axhline(y=u - m * std, color='r', label=f'оценка - {m} σ')
    axs[1].legend()
    safe_filename = name_graph.replace("/", "_")
    plot_filename = os.path.join(output_folder, f"{safe_filename}.png")
    fig.savefig(plot_filename)  # Сохраняем фигуру
    plt.close(fig)
    return data_inner
    
    
    
def save_total(data, result, y_min, y_max, x_min, x_max, n, name_graph, output_folder=output_folder_sep, m=m):
    # Создаем график с тремя подграфиками
    fig, axs = plt.subplots(figsize=(20, 5), nrows=1, ncols=3)  # Три графика: два для участков и один для общего

    # Переменная для объединённых данных
    combined_data = np.full_like(data, np.nan)
    
    # Обрабатываем каждый участок данных
    data_inner = np.full_like(data, np.nan)
    x_axis_all = np.arange(len(data))
    for i in range(len(result) - 1):
        x_axis_segment = np.arange(result[i], result[i + 1])
        data_inner[x_axis_segment] = data[x_axis_segment]
        
        # Добавляем участок данных в общий массив
        combined_data[x_axis_segment] = data[x_axis_segment]
        
        # График для каждого участка
        axs[0].set_ylim(y_min - 0.1, y_max + 0.1)
        axs[0].set_xlim(x_min - n // 100, x_max + n // 100)
        axs[0].plot(x_axis_all, data_inner)
        axs[0].grid(True)
        axs[0].set_title(name_graph)

    # График после фильтрации выбросов
    axs[1].set_title(f'{name_graph} после удаления выбросов')
    axs[1].set_ylim(y_min - 0.1, y_max + 0.1)
    axs[1].set_xlim(x_min - n // 100, x_max + n // 100)
    axs[1].grid(True)

    u, std = method2_sep(data_inner)
    cond = (data_inner >= u - m * std) & (data_inner <= u + m * std)
    ind_delete = np.where(~cond)[0]
    data_inner[ind_delete] = np.nan
    axs[1].plot(x_axis_all, data_inner)
    axs[1].axhline(y=u, color='b', label='оценка')
    axs[1].axhline(y=u + m * std, color='r', label=f'оценка + {m} σ')
    axs[1].axhline(y=u - m * std, color='r', label=f'оценка - {m} σ')
    axs[1].legend()

    # График для объединённых данных
    axs[2].set_title(f'Общий график {name_graph}')
    axs[2].set_ylim(y_min - 0.1, y_max + 0.1)
    axs[2].set_xlim(x_min - n // 100, x_max + n // 100)
    axs[2].grid(True)

    axs[2].plot(x_axis_all, combined_data, label='Общий график', color='g')
    axs[2].legend()

    # Сохраняем фигуру
    
    
    safe_filename = name_graph.replace("/", "_")
    plot_filename = os.path.join(output_folder, f"{safe_filename}.png")
    fig.savefig(plot_filename)
    plt.close(fig)
    
    
    

def data_separation(data, num_points, model='l2'):
    # Сигнал — скользящее среднее
    algo = rpt.Binseg(model=model).fit(data)
    result = algo.predict(n_bkps=num_points)  # Ожидаем num_points точек разрыва
    return [0] + result

import ruptures as rpt

# def data_separation(data, model='l2', penalty=None): Пока функция не проверялась
#     """
#     Автоматическое определение точек разрыва без указания их количества.

#     Параметры:
#         data (list или np.array): Входные данные.
#         model (str): Модель для поиска разрывов ('l2', 'l1', 'rbf' и т.д.).
#         penalty (float или None): Штрафной коэффициент для автоматического выбора точек разрыва.
#                                   Если None, используется кросс-валидация.

#     Возвращает:
#         list: Список индексов точек разрыва, включая 0.
#     """
#     # Инициализация алгоритма
#     algo = rpt.Binseg(model=model).fit(data)

#     # Автоматический выбор точек разрыва
#     if penalty is not None:
#         # Используем штрафной коэффициент (например, BIC)
#         result = algo.predict(pen=penalty)
#     else:
#         # Используем кросс-валидацию для выбора оптимального количества точек
#         result = algo.predict(n_bkps=None)  # Автоматический выбор

#     return [0] + result

def trend_orthogonal(x, y, deg):    
    coeffs = np.polynomial.legendre.legfit(x, y, deg=deg)
    poly = Legendre(coeffs)
    return poly






def plot_trend(lst_of_data, column_names_data, m=m, output_folder=output_folder_trend, COLOR = 'green', DEGREE = 3):
    for i, data_new in enumerate(lst_of_data):
        fig, axs = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
        try:
            data = data_new
            poly = trend_orthogonal(np.arange(len(data)), data, deg=DEGREE)
            x_axis = np.linspace(0, len(data), 1000)
            axs[0].plot(data)
            axs[0].plot(x_axis, poly(x_axis), color= COLOR)
            data_after_method = data_cleaning(data)

            poly = trend_orthogonal(np.arange(len(data_after_method)), data_after_method, deg= DEGREE)
            axs[1].plot(data_after_method)
            axs[1].plot(x_axis, poly(x_axis), color=COLOR)
            axs[1].set_xlim(axs[0].get_xlim())  
            axs[1].set_ylim(axs[0].get_ylim()) 
            

            safe_filename = column_names_data[i].replace("/", "_")
            plot_filename = os.path.join(output_folder, f"{safe_filename}.png")
            
            axs[0].set_title("До чистки")
            axs[1].set_title("После чистки")
            fig.suptitle(f"{safe_filename}", fontsize=16) 
            fig.savefig(plot_filename)
        except TypeError:
            print(f"Ошибка для колонки {column_names_data[i]}")
        
        
        plt.close(fig)
        plt.clf()
    




def plot_filtered_and_original(result, data, filter_data, name):
    
    fig, axs = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)

    x_axis_all = np.arange(len(data))  # Общая ось X для всех данных
    
    
    combined_data = np.full_like(data, np.nan)
    for i, temp in enumerate(filter_data):
        x_axis = np.arange(result[i], result[i + 1])
        combined_data[x_axis] = temp[x_axis]  # Сохраняем отфильтрованные данные для каждого участка в общий массив
    
    
    axs[0].plot(x_axis_all, data)
    axs[0].set_title(f'Оригинальыне данные')
    axs[0].grid(True)
    
    
    
    axs[1].plot(x_axis_all, combined_data)
    axs[1].set_title(f'Данные после чистки')
    axs[1].grid(True)
    
    axs[1].set_xlim(axs[0].get_xlim())  
    axs[1].set_ylim(axs[0].get_ylim()) 
    
    
    safe_filename = name.replace("/", "_")
    fig.suptitle(safe_filename, fontsize=16) 
    plot_filename = os.path.join(output_folder_sep, f"{safe_filename}.png")
    fig.savefig(plot_filename)
    plt.close(fig)
    


if __name__ == '__main__':
    print(density(5))