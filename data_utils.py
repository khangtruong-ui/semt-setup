import tensorflow as tf


def get_data_json() -> dict[str, list[dict[str, str | int]]]:
    fname = 'dataset_nwpu.json'
    with open(fname) as f:
        data = json.load(f)
    return data

def extractor():
    def getSentence(label: str):
        rg = re.split('[ ,]', re.sub(r'[^\x00-\x7F]+', ' ', label))
        return [x for x in rg if x]

    raw_data = get_data_json()
    processed_data = {
        f'02_NWPU_RESISC45/{category}/{dictionary["filename"]}': [
          dictionary['raw'],
          dictionary['raw_1'],
          dictionary['raw_2'],
          dictionary['raw_3'],
          dictionary['raw_4']
        ] for category in raw_data for dictionary in raw_data[category]
    }
    very_processed_data = {key: [getSentence(x) for x in processed_data[key]] for key in processed_data}
    return very_processed_data


def store_tokenizer_and_data():
    processed_data = extractor()
    all_words = {word.lower() for name in processed_data for sentence in processed_data[name] for word in sentence}
    all_words = ['', '[begin]', '[end]'] + list(sorted(all_words))
    tokenizer = {word: i for i, word in enumerate(all_words)}
    tokenized_data = {key: [[tokenizer[word.lower()] for word in ['[begin]'] + sentence + ['[end]']] for sentence in processed_data[key]] for key in processed_data}
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer, f)
    with open('data.json', 'w') as f:
        json.dump(tokenized_data, f)


def load_tokenizer():
    with open('tokenizer.json') as f:
        return json.load(f)


def split_data():
    with open('data.json') as f:
        data = json.load(f)
    test_lst = []
    train_lst = []
    for i, k in enumerate(data):
        if i % 5 > 3:
            tgt_lst = test_lst
        else:
            tgt_lst = train_lst
        tgt_lst.append((k, data[k]))
    with open('train.json', 'w') as f:
        json.dump(train_lst, f)
    with open('test.json', 'w') as f:
        json.dump(test_lst, f)


class ImageTextLoader:
    def __init__(self):
        with open(self.getDirectory()) as f:
            data = json.load(f)
        self.data: list[tuple] = data
        self.batch = BATCH_SIZE

    @abstractmethod
    def getDirectory(self) -> str:
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, inp, label = self.getData(item)
        return (img, inp), label

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def getData(self, item):
        directory, vectors = self.data[item]
        vectors: list
        good_vectors = [vector[:-1] for vector in vectors]
        [good_vector.extend([0] * (MAXIMUM_LENGTH - len(good_vector))) for good_vector in good_vectors]
        [vector.extend([0] * (MAXIMUM_LENGTH - len(vector))) for vector in vectors]

        textTensor = tf.constant(good_vectors, dtype=tf.int32)
        shiftedVectors = [vector[1:] + [0] for vector in vectors]
        shiftedTensor = tf.constant(shiftedVectors, dtype=tf.int32)

        img = tf.constant(tf.image.resize(tf.image.decode_image(tf.io.read_file(directory)), IDEAL_SHAPE[:-1]), dtype=tf.float32)
        return img, textTensor, shiftedTensor

    def get_dataset(self):
        directories, inputs, labels = [], [], []
        for directory, vectors in self.data:
            good_vectors = [vector[:-1] for vector in vectors]
            [good_vector.extend([0] * (MAXIMUM_LENGTH - len(good_vector))) for good_vector in good_vectors]
            [vector.extend([0] * (MAXIMUM_LENGTH - len(vector))) for vector in vectors]
            textTensor = tf.constant(good_vectors, dtype=tf.int32)
            shiftedVectors = [vector[1:] + [0] for vector in vectors]
            shiftedTensor = tf.constant(shiftedVectors, dtype=tf.int32)
            directories.append(directory)
            inputs.append(textTensor)
            labels.append(shiftedTensor)
        return tf.data.Dataset.from_tensor_slices(
            (directories, inputs, labels)
        ).map(
            lambda x, y, z: ((read_image(x), y), z),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)


class TrainDataset(ImageTextLoader):
    def __init__(self):
        super().__init__()
        random.Random(42).shuffle(self.data)

    def getDirectory(self) -> str:
        return 'train.json'


class TestDataset(ImageTextLoader):
    def __init__(self):
        super().__init__()
        random.shuffle(self.data)
        self.batch *= 3

    def getDirectory(self) -> str:
        return 'test.json'


def read_image(directory):
    img = tf.io.read_file(directory)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IDEAL_SHAPE[:-1])
    img = tf.cast(img, tf.float32)
    return img
