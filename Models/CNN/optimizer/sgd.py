"""
SGD Optimizer.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
from ._base_optimizer import _BaseOptimizer

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from sgd.py!")

class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum
        self.velocity = {}

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                
                if idx not in self.velocity:
                    self.velocity[idx] = {'weight': np.zeros_like(m.weight)}

                if 'weight' not in self.velocity[idx]:
                    self.velocity[idx]['weight'] = np.zeros_like(m.weight)
                
                self.velocity[idx]['weight'] = (self.momentum * self.velocity[idx]['weight']) - (self.learning_rate * m.dw)
                m.weight += self.velocity[idx]['weight']

                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
               
                if idx not in self.velocity:
                    self.velocity[idx] = {'bias': np.zeros_like(m.bias)}

                if 'bias' not in self.velocity[idx]:
                    self.velocity[idx]['bias'] = np.zeros_like(m.bias)
                
                self.velocity[idx]['bias'] = (self.momentum * self.velocity[idx]['bias']) - (self.learning_rate * m.db)
                m.bias += self.velocity[idx]['bias']

                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
