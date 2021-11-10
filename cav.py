import os
import logging
import os.path
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from .utils import *

class CAV(object):
    """
        CAV class contains methods for concept activation vector (CAV).

        CAV represents semenatically meaningful vector directions in
        network's embeddings (bottlenecks).
        """

    @staticmethod
    def default_hparams():
        """HParams used to train the CAV.

        you can use logistic regression or linear regression, or different
        regularization of the CAV parameters.

        Returns:
          parameter dict for training.
        """
        return {'model_type': 'logistic', 'C': 1.0}
        # return {'model_type': 'linear', 'alpha': 0.1}

    @staticmethod
    def load_cav(cav_path):
        """Make a CAV instance from a saved CAV (pickle file).

        Args:
          cav_path: the location of the saved CAV

        Returns:
          CAV instance.
        """
        with open(cav_path, 'rb') as pkl_file:
            save_dict = pickle.load(pkl_file)

        cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
                  save_dict['hparams'], save_dict['saved_path'])
        cav.accuracies = save_dict['accuracies']
        cav.cavs = save_dict['cavs']
        return cav

    @staticmethod
    def check_cav_exists(cav_dir, concepts, model_name, bottleneck, cav_hparams):
        """Check if a CAV is saved in cav_dir.

        Args:
          cav_dir: where cav pickles might be saved
          concepts: set of concepts used for CAV
          bottleneck: the bottleneck used for CAV
          cav_hparams: a parameter used to learn CAV

        Returns:
          True if exists, False otherwise.
        """
        cav_path = os.path.join(
            cav_dir,
            get_cav_key(concepts, model_name, bottleneck, cav_hparams) + '.pkl')
        return os.path.exists(cav_path)

    @staticmethod
    def _create_cav_training_set(concepts, bottleneck, acts):
        """Flattens acts, make mock-labels and returns the info.

        Labels are assigned in the order that concepts exists.

        Args:
            concepts: names of concepts
            bottleneck: the name of bottleneck where acts come from
            acts: a dictionary that contains activations
        Returns:
            x -  flattened acts
            labels - corresponding labels (integer)
            labels2text -  map between labels and text.
        """

        x = []
        labels = []
        labels2text = {}
        # to make sure postiive and negative examples are balanced,
        # truncate all examples to the size of the smallest concept.
        min_data_points = np.min(
            [acts[concept][bottleneck].shape[0] for concept in acts.keys()])

        for i, concept in enumerate(concepts):
            x.extend(acts[concept][bottleneck][:min_data_points].reshape(
                min_data_points, -1))
            labels.extend([i] * min_data_points)
            labels2text[i] = concept
        x = np.array(x)
        labels = np.array(labels)

        return x, labels, labels2text

    def __init__(self, concepts, bottleneck, hparams, save_path=None):
        """Initialize CAV class.

        Args:
          concepts: set of concepts used for CAV
          bottleneck: the bottleneck used for CAV
          hparams: a parameter used to learn CAV
          save_path: where to save this CAV
        """
        self.concepts = concepts
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path

    def train(self, acts):
        """Train the CAVs from the activations.

        Args:
          acts: is a dictionary of activations. In particular, acts takes for of
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
        Raises:
          ValueError: if the model_type in hparam is not compatible.
        """

        logging.info('training with {}'.format(','.join([str(k)+'='+str(v) for k, v in self.hparams.items()])))
        x, labels, labels2text = CAV._create_cav_training_set(
            self.concepts, self.bottleneck, acts)

        if self.hparams['model_type'] == 'linear':
            lm = linear_model.SGDClassifier(alpha=self.hparams['alpha'], tol=1e-3, max_iter=1000)
        elif self.hparams['model_type'] == 'logistic':
            lm = linear_model.LogisticRegression(C=self.hparams['C'])
        else:
            raise ValueError('Invalid hparams.model_type: {}'.format(
                self.hparams['model_type']))

        self.accuracies = self._train_lm(lm, x, labels, labels2text)
        if len(lm.coef_) == 1:
            # if there were only two labels, the concept is assigned to label 0 by
            # default. So we flip the coef_ to reflect this.
            self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cavs = [c for c in lm.coef_]
        self._save_cavs()

    def get_direction(self, concept):
        """Get CAV direction.

        Args:
          concept: the conept of interest

        Returns:
          CAV vector.
        """
        return self.cavs[self.concepts.index(concept)]

    def _save_cavs(self):
        """Save a dictionary of this CAV to a pickle."""
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            logging.info('save_path is None. Not saving anything')

    def _train_lm(self, lm, x, y, labels2text):
        """Train a model to get CAVs.

        Modifies lm by calling the lm.fit functions. The cav coefficients are then
        in lm._coefs.

        Args:
          lm: An sklearn linear_model object. Can be linear regression or
            logistic regression. Must support .fit and ._coef.
          x: An array of training data of shape [num_data, data_dim]
          y: An array of integer labels of shape [num_data]
          labels2text: Dictionary of text for each label.

        Returns:
          Dictionary of accuracies of the CAVs.

        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, stratify=y)
        # if you get setting an array element with a sequence, chances are that your
        # each of your activation had different shape - make sure they are all from
        # the same layer, and input image size was the same
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        # get acc for each class.
        num_classes = max(y) + 1
        acc = {}
        num_correct = 0
        for class_id in range(num_classes):
            # get indices of all test data that has this class.
            idx = (y_test == class_id)
            acc[labels2text[class_id]] = metrics.accuracy_score(
                y_pred[idx], y_test[idx])
            # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[labels2text[class_id]])
        acc['overall'] = float(num_correct) / float(len(y_test))
        logging.info('acc per class %s' % (str(acc)))
        return acc


def get_or_train_cav(concepts,
                     model_name, 
                     bottleneck,
                     acts,
                     cav_dir=None,
                     cav_hparams=None,
                     overwrite=False):
    """Gets, creating and training if necessary, the specified CAV.

    Assumes the activations already exists.

    Args:
      concepts: set of concepts used for CAV
              Note: if there are two concepts, provide the positive concept
                    first, then negative concept (e.g., ['striped', 'random500_1']
      bottleneck: the bottleneck used for CAV
      acts: dictionary contains activations of concepts in each bottlenecks
            e.g., acts[concept][bottleneck]
      cav_dir: a directory to store the results.
      cav_hparams: a parameter used to learn CAV
      overwrite: if set to True overwrite any saved CAV files.

    Returns:
      returns a CAV instance
    """

    if cav_hparams is None:
        cav_hparams = CAV.default_hparams()

    cav_path = None
    if cav_dir is not None:
        make_dir_if_not_exists(cav_dir)
        cav_path = os.path.join(
            cav_dir,
            get_cav_key(concepts, model_name, bottleneck, cav_hparams).replace('/', '.') + '.pkl')

        if not overwrite and os.path.exists(cav_path):
            logging.info('CAV already exists: {}'.format(cav_path))
            cav_instance = CAV.load_cav(cav_path)
            return cav_instance

    logging.info('Training CAV {} - {} - {}'.format(
        concepts, 
        bottleneck, 
        ','.join([str(k)+'='+str(v) for k, v in cav_hparams.items()])))
    cav_instance = CAV(concepts, bottleneck, cav_hparams, cav_path)
    cav_instance.train({c: acts[c] for c in concepts})
    return cav_instance
