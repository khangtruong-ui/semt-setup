BATCH_SIZE = 512
MAX_LENGTH = 53
INPUT_SEQ_LENGTH = MAXIMUM_LENGTH = MAX_LENGTH 
TEXT_EMBEDDING_DIM = 768
VOCAB_SIZE = 4000
MESHED_BUFFER_SIZE = 100
MESHED_DEPTH = 4
ATTENTION_HEAD = 16
ATTENTION_CHOICE = 3
DECODER_ATTENTION_CHOICE = 2
BACKBONE_CHOICE = 7
NUM_CAPTIONS = 1
EXPANSION_LENGTH = 100

model_name = {
    0: 'RVSA',
    1: 'Meshed_memory',
    2: 'Zero_mesh',
    3: 'Static_attention',
    4: 'Captured_static_attention',
}[ATTENTION_CHOICE]

decoder_model = {
    0: 'Mesh_decoder',
    1: 'T5',
    2: 'No_mesh',
    3: 'Dynamic_attention',
}[DECODER_ATTENTION_CHOICE]

vision_name = {
    0: 'Resnet152',
    1: 'Resnet50',
    2: 'VGG16',
    3: 'EfficientNetB2',
    4: 'Short',
    5: 'Swin',
    6: 'Inception',
    7: 'MobileNet'
}[BACKBONE_CHOICE]
