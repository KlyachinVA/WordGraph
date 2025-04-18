from embedding import PreTrainedEmbeddings


if __name__ == "__main__":
    embeddings = PreTrainedEmbeddings.from_embeddings_file('../../data/models/tayga_1_2.csv',
                                                           build_index=False,
                                                           index_file="index_tayga.bin")
                                                                  # index_file="index1.bin")
    # embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/ruscorpora_upos_skipgram_300_5_2018.vec',
    #                                                        build_index=True,
    #                                                        index_file="index.bin")
    # embeddings = PreTrainedEmbeddings.from_embeddings_file('../../data/models/model.csv',
    #                                                        build_index=True,
    #                                                        index_file="index-model.bin")
    # embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/my-lenta-ru-texts-model-16000epochs-1818words-100features.txt',
    #                                                        build_index=True,
    #                                                        index_file="my-index.bin")

    # words = embeddings.closest('лодка')
    #
    # print(words)


    #embeddings.compute_and_print_analogy('мужчина', 'он', 'женщина')


    #embeddings.compute_and_print_analogy('король_NOUN', 'королева_NOUN', 'мужчина_NOUN')
    # embeddings.compute_and_print_analogy('cat', 'kitten', 'dog')
    #embeddings.compute_and_print_analogy('синий_ADJ', 'цвет_NOUN', 'кошка_NOUN')
    #embeddings.compute_and_print_analogy('нога_NOUN', 'ноги_NOUN', 'рука_NOUN')
    # embeddings.compute_and_print_analogy('toe', 'foot', 'finger')
    # embeddings.compute_and_print_analogy('talk', 'communicate', 'read')
    # embeddings.compute_and_print_analogy('blue', 'democrat', 'red')
    # embeddings.compute_and_print_analogy('кошка', 'животное', 'ель')
    # embeddings.compute_and_print_analogy('сталь', 'гайка', 'глина')
    # embeddings.compute_and_print_analogy('мужчина_NOUN', 'учитель_NOUN', 'женщина_NOUN')
    # embeddings.compute_and_print_analogy('учитель_NOUN', 'тетрадь_NOUN', 'слесарь_NOUN')
    # embeddings.compute_and_print_analogy('тетрадь', 'ученик', 'молоток')
    # embeddings.compute_and_print_analogy('молот', 'кузнец', 'топор')
    # embeddings.compute_and_print_analogy('карандаш', 'художник', 'скрипка')
    # embeddings.compute_and_print_analogy('мяч', 'футболист', 'клюшка')
    # embeddings.compute_and_print_analogy('мяч', 'футболист', 'клюшка')
    # embeddings.compute_and_print_analogy('париж', 'франция', 'берлин')
    # embeddings.compute_and_print_analogy('цена', 'зафиксировать', 'допинг')
    # embeddings.compute_and_print_analogy('глубоко', 'глубже', 'светло')
    # embeddings.compute_and_print_analogy('высоко', 'выше', 'далеко')
    # embeddings.compute_and_print_analogy('далеко', 'близко', 'высоко')
    # embeddings.compute_and_print_analogy('колесо', 'машина', 'ручка')

    # embeddings.compute_and_print_analogy('колесо', 'катить', 'ручка')
    # embeddings.compute_and_print_analogy('чашка', 'стоять', 'нож')

    # embeddings.compute_place("машина","ехать", "дорога",
    #                          "художник", "рисовать")

    # embeddings.compute_place("певец", "петь", "песня",
    #                          "ученик", "учить")

    # embeddings.compute_place("птица", "махать", "крыло",
    #                          "медведь", "ловить")

    # embeddings.compute_place("цветы", "стоять", "ваза",
    #                          "ложка", "лежать")

    embeddings.compute_place("ученик", "читать", "книга",
                             "печник", "класть")
    # dim = 300
    # alpha = embeddings.train_pairs("./data/datasets/words-pairs.txt",dim)
    # print(alpha)
    #

