import cv2
import os

def extract_frames(video_path, N):
    print(os.getcwd())
    # Проверяем существует ли папка 'frames', если нет - создаем
    if not os.path.exists('data/input/'):
        os.makedirs('data/input/')

    # Загружаем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Не удалось открыть видео.")
        return

    # Определяем общее количество кадров в видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Вычисляем интервал кадров для сохранения
    frame_interval = total_frames // N if N < total_frames else 1

    current_frame = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()

        # Если кадр успешно прочитан
        if ret:
            if current_frame % frame_interval == 0 and saved_frames < N:
                # Сохраняем кадр в папке 'data/input' с именем 'frame(номер).png'
                cv2.imwrite(f'data/input/frame{saved_frames}.png', frame)
                saved_frames += 1
            current_frame += 1
        else:
            break

        # Прекращаем обработку, если достигнут лимит N кадров
        if saved_frames >= N:
            break

    # Освобождаем ресурсы
    cap.release()
    print(f"Сохранено {saved_frames} кадров.")

def read_config(file_path, parameter):
    """
    Функция для чтения конфигурационного файла и извлечения значения заданного параметра.

    :param file_path: Путь к конфигурационному файлу.
    :param parameter: Название параметра, значение которого нужно извлечь.
    :return: Значение параметра, если найдено, иначе None.
    """
    with open(file_path, 'r') as file:
        for line in file:
            # Удаление пробельных символов в начале и конце строки
            line = line.strip()
            # Пропуск комментариев и пустых строк
            if line.startswith('#') or not line:
                continue
            # Разделение строки на параметр и значение
            if '=' in line:
                param, value = map(str.strip, line.split('=', 1))
                if param == parameter:
                    # Удаление кавычек, если значение заключено в кавычки
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    return value
    return None

# Путь к конфигурационному файлу
config_file_path = '../config.txt'
param_to_extract = 'video_name'
frames_parametrs = 'frames'
video_name = read_config(config_file_path, param_to_extract)
frames = read_config(config_file_path, frames_parametrs)
print(f"The value of '{param_to_extract}' is: {video_name}, '{frames_parametrs}' is {frames}")
# Нарезка видео на заданное колличество кадров
extract_frames("input_data/"+video_name, int(frames))
