


def load_checkpoint():
    with open('weights.msgpack', 'rb') as f:
        return serialization.from_bytes(f.read())
