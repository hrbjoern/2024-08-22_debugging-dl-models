import tensorflow as tf

from toy_modules.utils import generate_datasets
from toy_modules.models import RegressorNet
from toy_modules.train import train

tf.get_logger().setLevel('ERROR')



if __name__ == '__main__':

    tf.get_logger().setLevel('ERROR')


    # Generate datasets
    train_dataset, test_dataset = generate_datasets()

    # Create the model. Please, use SGD optimizer for this exercise,
    regressor = RegressorNet(
        input_shape=20,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    )

    # Train (leave the number of epochs unchanged and equal to 200)
    train(
        model=regressor,
        epochs=200,
        # For testing: 
        #epochs=10,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        save_dir='run2'
    )
