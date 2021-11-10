import os
import logging
from multiprocessing import dummy as multiprocessing
import time
import pickle
import datetime
import numpy as np

from .cav import CAV
from .cav import get_or_train_cav
from .model import get_or_load_gradients
from .run_params import RunParams
from .utils import *

class TCAV(object):
    """TCAV object: runs TCAV for one target and a set of concepts.
    The static methods (get_direction_dir_sign, compute_tcav_score,
    get_directional_dir) invole getting directional derivatives and calculating
    TCAV scores. These are static because they might be useful independently,
    for instance, if you are developing a new interpretability method using
    CAVs.
    """
    @staticmethod
    def compute_tcav_score(mymodel,
                           target_class,
                           concept,
                           cav,
                           class_acts,
                           grad_dir,
                           bottleneck):
        """Compute TCAV score.

        Args:
          mymodel: a model class instance
          target_class: one target class
          concept: one concept
          cav: an instance of cav
          grad_dir: target class gradients file directory
          class_acts: activations of the images in the target class.

        Returns:
            TCAV score (i.e., ratio of pictures that returns negative dot product
            wrt loss),
            directional dirivative values
        """
        directional_dir_vals = TCAV.get_directional_dir(
            mymodel, target_class, concept, cav, class_acts, grad_dir, bottleneck)
        return sum(directional_dir_vals < 0) / len(directional_dir_vals), directional_dir_vals

    @staticmethod
    def get_cosine_similarity(concept, cav, class_acts):
        cav_vector = cav.get_direction(concept)
        below = (np.linalg.norm(cav_vector) * np.linalg.norm(class_acts, axis=1))
        return np.dot(class_acts, cav_vector) / below

    @staticmethod
    def get_directional_dir(mymodel, 
                            target_class, 
                            concept, 
                            cav, 
                            class_acts, 
                            grad_dir, 
                            bottleneck):
        """Return the list of values of directional derivatives.

           (Only called when the values are needed as a referece)

        Args:
          mymodel: a model class instance
          target_class: one target class
          concept: one concept
          cav: an instance of cav
          class_acts: activations of the images in the target class
          grad_dir: target class gradients file directory
          bottleneck: activation layer name of the model.

        Returns:
          list of values of directional derivatives.
        """
        grads = get_or_load_gradients(mymodel, class_acts, grad_dir, target_class, bottleneck)
        directional_dir_vals = np.dot(grads, cav.get_direction(concept))
        return directional_dir_vals

    def __init__(self,
                 target,
                 concepts,
                 bottlenecks,
                 activation_generator,
                 cav_hparams=None,
                 random_counterpart=None,
                 cav_dir=None,
                 grads_dir=None,
                 num_random_exp=5,
                 random_concepts=None):
        """Initialze tcav class.

        Args:
          target: one target class
          concepts: one concept
          bottlenecks: the name of a bottleneck of interest.
          activation_generator: an ActivationGeneratorInterface instance to return
                                activations.
          cav_hparams: the hyper parameters of the cav in dictionary
          cav_dir: the path to store CAVs
          random_counterpart: the random concept to run against the concepts for
                      statistical testing. If supplied, only this set will be
                      used as a positive set for calculating random TCAVs
          num_random_exp: number of random experiments to compare against.
          random_concepts: A list of names of random concepts for the random
                           experiments to draw from. Optional, if not provided, the
                           names will be random500_{i} for i in num_random_exp.
          grads_dir: the gradients directory
        """
        self.target = target
        self.concepts = concepts
        self.bottlenecks = bottlenecks
        self.activation_generator = activation_generator
        self.cav_dir = cav_dir
        self.grads_dir = grads_dir
        self.cav_hparams = cav_hparams
        self.mymodel = activation_generator.get_model()
        self.model_to_run = self.mymodel.model_name
        self.random_counterpart = random_counterpart
        
        if self.cav_hparams is None:
            self.cav_hparams = CAV.default_hparams()

        if random_concepts:
            num_random_exp = len(random_concepts)

        # make pairs to test.
        self._process_what_to_run_expand(num_random_exp=num_random_exp,
                                         random_concepts=random_concepts)
        # parameters
        self.params = self.get_params()
        logging.info('TCAV will {} params'.format(len(self.params)))

    def run(self, num_workers=10, run_parallel=False):
        """Run TCAV for all parameters (concept and random), write results to html.

        Args:
          num_workers: number of workers to parallelize
          run_parallel: run this parallel.

        Returns:
          results: result dictionary.
        """
        # for random exp,  a machine with cpu = 30, ram = 300G, disk = 10G and
        # pool worker 50 seems to work.
        logging.info('running %s params' % len(self.params))
        now = time.time()
        if run_parallel:
            pool = multiprocessing.Pool(num_workers)
            results = pool.map(lambda param: self._run_single_set(param), self.params)
        else:
            results = []
            for i, param in enumerate(self.params):
                logging.info('Running param {} of {}'.format(i, len(self.params)))
                results.append(self._run_single_set(param))
        logging.info('Done running %s params. Took %s seconds...' % (len(
            self.params), time.time() - now))
        return results

    def _run_single_set(self, param):
        """Run TCAV with provided for one set of (target, concepts).

        Args:
          param: parameters to run

        Returns:
          a dictionary of results (panda frame)
        """
        bottleneck = param.bottleneck
        concepts = param.concepts
        target_class = param.target_class
        activation_generator = param.activation_generator
        cav_hparams = param.cav_hparams
        mymodel = param.model
        cav_dir = param.cav_dir
        # first check if target class is in model.

        logging.info('running %s %s' % (target_class, concepts))

        # Get acts
        acts = activation_generator.process_and_load_activations(
            [bottleneck], concepts + [target_class])
        # Get CAVs
        cav_instance = get_or_train_cav(
            concepts, 
            param.activation_generator.model.model_name, 
            bottleneck, 
            acts, 
            cav_dir=cav_dir, 
            cav_hparams=self.cav_hparams)

        # clean up
        for c in concepts:
            del acts[c]

        # Hypo testing
        a_cav_key = get_cav_key(concepts, param.activation_generator.model.model_name, bottleneck, self.cav_hparams)
        target_class_for_compute_tcav_score = target_class

        cav_concept = concepts[0]

        i_up, val_directional_dirs = self.compute_tcav_score(
            mymodel, target_class_for_compute_tcav_score, cav_concept,
            cav_instance, acts[target_class][cav_instance.bottleneck], self.grads_dir, bottleneck)
        logging.info('Get TCAV score {}'.format(str(i_up)))
        result = {
            'cav_key':
                a_cav_key,
            'cav_concept':
                cav_concept,
            'target_class':
                target_class,
            'i_up':
                i_up,
            'val_directional_dirs_abs_mean':
                np.mean(np.abs(val_directional_dirs)),
            'val_directional_dirs_mean':
                np.mean(val_directional_dirs),
            'val_directional_dirs_std':
                np.std(val_directional_dirs),
            'bottleneck':
                bottleneck
        }
        result.update(cav_hparams)
        del acts
        return result

    def _process_what_to_run_expand(self, num_random_exp=100, random_concepts=None):
        """Get tuples of parameters to run TCAV with.

        TCAV builds random concept to conduct statistical significance testing
        againts the concept. To do this, we build many concept vectors, and many
        random vectors. This function prepares runs by expanding parameters.

        Args:
          num_random_exp: number of random experiments to run to compare.
          random_concepts: A list of names of random concepts for the random experiments
                       to draw from. Optional, if not provided, the names will be
                       random500_{i} for i in num_random_exp.
        """

        target_concept_pairs = [(self.target, self.concepts)]

        # take away 1 random experiment if the random counterpart already in random concepts
        all_concepts_concepts, pairs_to_run_concepts = process_what_to_run_expand(
            process_what_to_run_concepts(target_concept_pairs),
            self.random_counterpart,
            num_random_exp=num_random_exp - (
                1 if random_concepts and self.random_counterpart in random_concepts else 0),
            random_concepts=random_concepts)

        pairs_to_run_randoms = []
        all_concepts_randoms = []

        # ith random concept
        def get_random_concept(i):
            return (random_concepts[i] if random_concepts
                    else 'random500_{}'.format(i))

        if self.random_counterpart is None:
            # TODO random500_1 vs random500_0 is the same as 1 - (random500_0 vs random500_1)
            for i in range(num_random_exp):
                all_concepts_randoms_tmp, pairs_to_run_randoms_tmp = (
                    process_what_to_run_expand(
                        process_what_to_run_randoms(target_concept_pairs,
                                                          get_random_concept(i)),
                        num_random_exp=num_random_exp - 1,
                        random_concepts=random_concepts))

                pairs_to_run_randoms.extend(pairs_to_run_randoms_tmp)
                all_concepts_randoms.extend(all_concepts_randoms_tmp)

        else:
            # run only random_counterpart as the positve set for random experiments
            all_concepts_randoms_tmp, pairs_to_run_randoms_tmp = (
                process_what_to_run_expand(
                    process_what_to_run_randoms(target_concept_pairs,
                                                      self.random_counterpart),
                    self.random_counterpart,
                    num_random_exp=num_random_exp - (1 if random_concepts and
                                                          self.random_counterpart in random_concepts else 0),
                    random_concepts=random_concepts))

            pairs_to_run_randoms.extend(pairs_to_run_randoms_tmp)
            all_concepts_randoms.extend(all_concepts_randoms_tmp)

        self.all_concepts = list(set(all_concepts_concepts + all_concepts_randoms))
        self.pairs_to_test = pairs_to_run_concepts + pairs_to_run_randoms

    def get_params(self):
        """Enumerate parameters for the run function.

        Returns:
          parameters
        """
        params = []
        for bottleneck in self.bottlenecks:
            for target_in_test, concepts_in_test in self.pairs_to_test:
                param = RunParams(bottleneck, concepts_in_test, target_in_test,
                                            self.activation_generator, self.cav_dir,
                                            self.cav_hparams, self.mymodel)
                params.append(param)
                logging.info(param.get_key())
        return params

def run_tcav(cfg):
    from .activation_generator import ImageActivationGenerator
    from .model import get_model_wrapper

    get_timestamp = lambda : datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Make directories
    make_dir_if_not_exists(cfg.activation_dir)
    make_dir_if_not_exists(cfg.working_dir)
    make_dir_if_not_exists(cfg.cav_dir)
    make_dir_if_not_exists(cfg.grads_dir)
    make_dir_if_not_exists(cfg.results_dir)
    make_dir_if_not_exists(cfg.logs_dir)

    # Get model and activation generator
    mymodel = get_model_wrapper(cfg.model_wrapper)()
    act_generator = ImageActivationGenerator(mymodel, cfg.source_dir, cfg.activation_dir, max_examples=cfg.max_examples)

    # Logging
    logging.basicConfig(
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a %d %b %Y %H:%M:%S',
        filename=os.path.join(cfg.logs_dir, get_timestamp() + '.log'),
        level=logging.INFO)

    # Run TCAV
    mytcav = TCAV(
                cfg.target,
                cfg.concepts,
                cfg.bottlenecks,
                act_generator,
                cav_hparams=cfg.cav_hparams,
                random_counterpart=cfg.random_counterpart,
                cav_dir=cfg.cav_dir,
                grads_dir=cfg.grads_dir,
                num_random_exp=cfg.num_random_exp)
    results = mytcav.run()

    # Save results and return
    with open(os.path.join(cfg.results_dir, get_timestamp() + '.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return results

