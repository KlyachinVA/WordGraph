# WordGraph
Графовая модель текста. Вложение слов. 

Для загрузки исходного кода в терминале необходимо выполнить команду

git clone git@github.com:KlyachinVA/WordGraph.git

Для использования скачанных текстов программ дополнительно понадобиться скачать с ресурса https://rusvectores.org/ru/models/ языковую модель tayga_1_2 и разместить соответствующий файл в каталоге data. Для индексирования  слов из этой модели с целью ускорения вычисления вложений слов необходимо установить значение build_index=True  в вызов метода


PreTrainedEmbeddings.from_embeddings_file("./data/tayga_1_2.csv",
                                            build\_index=False,
                                            index_file="./indexes/index-tayga-csv.bin").
