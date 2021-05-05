import numpy as np
import tensorflow as tf

def anomaly_detect_split(images, labels, normals=[1, 3, 5, 7]):
    '''
    normal     = a: 50%, b: 50% (label = 0)
    abnormal   = c: 50%, d: 50% (label = 1)
    training   = a
    validation = b + c
    testing    = b + d
    '''
    
    normal_mask     = np.isin(labels, normals)
    normal_labels   = labels[normal_mask]
    normal_images   = images[normal_mask]
    abnormal_labels = labels[np.invert(normal_mask)]
    abnormal_images = images[np.invert(normal_mask)]
    
    normal_shuffler = np.random.permutation(len(normal_labels))
    train_splitter  = normal_shuffler[:int(len(normal_labels)/2)]
    train_labels    = normal_labels[train_splitter]
    train_labels    = (train_labels - 1) / 2 # 1, 3, 5, 7 => 0, 1, 2, 3 | For sparse categorical cross entropy
    train_images    = normal_images[train_splitter]
    
    test_normal_splitter = normal_shuffler[int(len(normal_labels)/2):]
    val_labels           = np.zeros(len(test_normal_splitter))
    val_images           = normal_images[test_normal_splitter]
    test_labels          = np.zeros(len(test_normal_splitter))
    test_images          = normal_images[test_normal_splitter]
    
    abnormal_shuffler      = np.random.permutation(len(abnormal_labels))
    abnormal_val_splitter  = abnormal_shuffler[:int(len(abnormal_labels)/2)]
    val_labels             = np.concatenate((val_labels, np.ones(len(abnormal_val_splitter))))
    val_images             = np.concatenate((val_images, abnormal_images[abnormal_val_splitter]))
    abnormal_test_splitter = abnormal_shuffler[int(len(abnormal_labels)/2):]
    test_labels            = np.concatenate((test_labels, np.ones(len(abnormal_test_splitter))))
    test_images            = np.concatenate((test_images, abnormal_images[abnormal_test_splitter]))
    
    return (train_images, train_labels, val_images, val_labels, test_images, test_labels)

class AnomalyValidation(tf.keras.callbacks.Callback):
    def __init__(self, ATH, ds_val, log_step=1):
        super(AnomalyValidation, self).__init__()
        self.best_weights = None
        self.best_acc = 0
        self.ATH = ATH # anomaly confidence threshold
        self.ds_val = ds_val
        self.log_step = log_step
    def on_epoch_end(self, epoch, logs=None):
        global best_classifier, best_acc
        num_correct = 0
        num_total = 0
        for image, label in self.ds_val:
            if 'sparse_categorical_accuracy' in logs.keys():
                confidence = tf.math.reduce_max(tf.nn.softmax(self.model(image)), 1).numpy()
                num_correct += ((confidence < self.ATH) == label.numpy()).sum()
            else:
                num_correct += (tf.keras.losses.MSE(self.model(image), image).numpy().mean(axis=1) < self.ATH).sum()
            num_total += label.shape[0]
        
        acc = num_correct/num_total
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_weights = self.model.get_weights()
        
        if epoch % self.log_step != 0:
            return
        
        print(f"Epoch {epoch+1: >2d}", end='')
        if 'sparse_categorical_accuracy' in logs.keys():
            print(
                f"\x1b[32m Train \x1b[0m "
                f"Loss: {logs['loss']: .3f}, "
                f"Acc: {logs['sparse_categorical_accuracy']: .3f}",
                end = '\t'
            )
        else:
            print(
                f"\x1b[32m Train \x1b[0m "
                f"MSE: {logs['loss']: .3f}, ",
                end = '\t'
            )
        print(f'Anomaly detection accuracy:\x1b[31m {acc: .5f}\x1b[0m')
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        