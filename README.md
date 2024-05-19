# RZD_Automatic_processing_of_work_records
Цифровой прорыв 2024 Сезон ИИ УрФо Челябинск команда python

Автоматическая обработка трудовых книжек

В компаниях существует необходимость быстрого и точного переноса данных из рукописных трудовых книжек в корпоративную систему управления трудовыми ресурсами (ЕК АСУТР). Традиционный ручной ввод данных занимает значительное время и подвержен ошибкам, что замедляет процесс найма и увеличивает затраты на обработку информации.
На хакатоне предлагается разработать программное обеспечение, использующее технологии искусственного интеллекта для автоматического распознавания текста в рукописных трудовых книжках. Это решение должно ускорять процесс ввода данных, повышать его точность и сокращать время, необходимое для обработки информации о новых сотрудниках, потенциально интегрируясь с существующей корпоративной системой управления.

Без метрик и лидерборда.

Создаем папки для классификации документов
>> train    >> images       >> img_1.jpg  (shows Object_1)
            >> annotations  >> img_1.txt  (describes Object_1)

>> test  >> images       >> img_1.jpg (shows Object_1 and Object_n)
         >> annotations  >> img_1.txt (describes Object_1,and Object_n)

Для запуска обучения модели открываем файл train_model.py

После обучения и сохрания модели model.h5 запускаем файл interface.py для запуска интерфейса
