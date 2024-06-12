# GaussianSplatting_3DR_SLAM
This repository contains implementations of Gaussian Splatting for 3D reconstruction and SLAM. It provides the necessary code to deploy these techniques for generating detailed 3D maps and performing efficient and accurate localization and mapping in dynamic environments. 
# Использование
Использование данного окружения .yml использует следующие зависимости:
* pytorch-cuda=12.1.  
=======
Что подходит для CUDA 12.0 и выше.
Для обучения модели можно использовать приложенный colab. Обучение на GPU Tesla T4 занимает 58 минут. После заврешения обучения папку с сохраненными чекпоинтами надо скачивать, на данный момент визуализаци и адаптивный просмотр сцены находятся в разработке. Также в работе находится реализация автоматического сравнения с аналогами для упрощения оценочной части.
3d_reconstraction
Исходные датасеты и результат обучения будут размещаться в следующей папке https://drive.google.com/drive/folders/1d-tfXUtZhvQ9Aye_yC4MYuc-IwaClLve?usp=sharing   
# Демонстрация результатов
Реконструкция помещений  
![Демонстрация работы](images/lamp.gif)  
Реконструкция открытых пространств  
![Демонстрация работы](images/garden.gif)  
Интерактивный рендеринг  
![Демонстрация работы](images/gaussians.gif)  
Работа SLAM  
![Демонстрация работы](images/slam.gif)  
# Текущая структура репозитория
Все, что связанно с 3D реконструкцией размещается в ветке 3D_reconstraction, а с SLAM в ветке slam
