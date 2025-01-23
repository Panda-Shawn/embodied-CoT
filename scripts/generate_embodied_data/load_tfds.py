import tensorflow as tf

def parse_step_features(step_proto):
    step_description = {
        'action': tf.io.FixedLenFeature([7], tf.float32),
        'is_terminal': tf.io.FixedLenFeature([], tf.bool),
        'is_last': tf.io.FixedLenFeature([], tf.bool),
        'language_instruction': tf.io.FixedLenFeature([], tf.string),
        'observation': {
            'wrist_image': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'state': tf.io.FixedLenFeature([8], tf.float32),
            'joint_state': tf.io.FixedLenFeature([7], tf.float32)
        },
        'is_first': tf.io.FixedLenFeature([], tf.bool),
        'discount': tf.io.FixedLenFeature([], tf.float32),
        'reward': tf.io.FixedLenFeature([], tf.float32)
    }
    return tf.io.parse_single_example(step_proto, step_description)

def parse_tfrecord(example_proto):
    feature_description = {
        'steps': tf.io.VarLenFeature(tf.string),
        'episode_metadata': {
            'file_path': tf.io.FixedLenFeature([], tf.string)
        }
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Parse each step in the sequence
    dense_steps = tf.sparse.to_dense(parsed_features['steps'])
    parsed_steps = tf.map_fn(
        parse_step_features,
        dense_steps,
        dtype={
            'action': tf.float32,
            'is_terminal': tf.bool,
            'is_last': tf.bool,
            'language_instruction': tf.string,
            'observation': {
                'wrist_image': tf.string,
                'image': tf.string,
                'state': tf.float32,
                'joint_state': tf.float32
            },
            'is_first': tf.bool,
            'discount': tf.float32,
            'reward': tf.float32
        }
    )
    
    # Decode images
    parsed_steps['observation']['wrist_image'] = tf.map_fn(
        tf.io.decode_jpeg,
        parsed_steps['observation']['wrist_image'],
        dtype=tf.uint8
    )
    parsed_steps['observation']['image'] = tf.map_fn(
        tf.io.decode_jpeg,
        parsed_steps['observation']['image'],
        dtype=tf.uint8
    )
    
    return parsed_steps, parsed_features['episode_metadata']

# Load and parse the dataset
def load_dataset(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    return dataset.map(parse_tfrecord)

# Usage example
file_path = "/data/lzx/libero/libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-00000-of-00032"
dataset = load_dataset(file_path)

