import cv2
import os

def extract_frames(video_path, N):
    print(os.getcwd())
    # Проверяем существует ли папка 'frames', если нет - создаем
    if not os.path.exists('frames'):
        os.makedirs('frames')

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
                # Сохраняем кадр в папке 'frames' с именем 'frame(номер).png'
                cv2.imwrite(f'frames/frame{saved_frames}.png', frame)
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


extract_frames('playground.mp4', 200)
